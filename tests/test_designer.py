"""Vue designer contracts, scoped capabilities and offline simulation regressions."""
import base64
import copy
import io
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.application.contracts import ApplicationError, connector
from agent.workflows.contracts import definition, execution_hash, DefinitionError
from agent.workflows.data import ingest, inspect_values, settings, table_values
from agent.workflows.designer import Designer
from agent.workflows.sources import query, search, preview_row, documents
from agent.workflows.templates import templates
from agent.workflows.tool_contracts import tool, tool_key, tool_wire


def binding(**changes):
    return {'tool_id': 'lookup', 'version': 1, 'conversation': True, 'workflow': True,
            'scope': {}, 'instructions': '', 'origins': ['internal', 'external']} | changes


def graph():
    value = copy.deepcopy(next(t for t in templates() if t['agent_id'] == 'blank-voice'))
    value.update(schema_version=2, tool_bindings=[binding()], goal='', instructions='', capture_tool_content=False)
    from agent.workflows.templates import node, conversation
    phase = node('conversation', 'conversation.collect', conversation('Help', {}))
    phase['config'].update(tool_mode='inherit', tools=[])
    value['nodes'].insert(1, phase)
    value['edges'] = [{'source': 'call', 'outcome': 'success', 'target': 'conversation'}, {'source': 'conversation', 'outcome': 'success', 'target': 'done'}]
    return value


def table_settings():
    return settings({'name': 'Orders', 'kind': 'table', 'format': 'csv', 'mapping': {},
                     'country_code': '39', 'decimal_separator': '.', 'header_row': 1,
                     'columns': [{'name': 'customer', 'source': 'Customer', 'type': 'string', 'required': True, 'sensitive': False},
                                 {'name': 'amount', 'source': 'Amount', 'type': 'number', 'required': True, 'sensitive': False},
                                 {'name': 'secret', 'source': 'Secret', 'type': 'string', 'required': False, 'sensitive': True}]})


def query_config():
    return {'resource': {'resource_id': 'orders', 'version': 'latest'}, 'fields': ['customer', 'amount'], 'filters': [], 'limit': 10}


def test_schema_two_preserves_layout_independent_hash_and_phase_restrictions():
    value = graph(); definition(value)
    changed = copy.deepcopy(value); changed['layout'] = {}; assert execution_hash(value) == execution_hash(changed)
    changed['tool_bindings'][0]['conversation'] = False; assert execution_hash(value) != execution_hash(changed)
    phase = next(n for n in value['nodes'] if n['type'] == 'conversation.collect')
    assert Designer.enabled(value, phase, {'caller': {'origin': 'external'}}) == value['tool_bindings']
    phase['config'].update(tool_mode='selected', tools=[])
    assert Designer.enabled(value, phase, {'caller': {'origin': 'external'}}) == []
    phase['config']['tools'] = [tool_key(binding())]
    assert Designer.enabled(value, phase, {'caller': {'origin': 'external'}})
    value['tool_bindings'][0]['origins'] = ['internal']
    assert Designer.enabled(value, phase, {'caller': {'origin': 'external'}}) == []


def test_wire_names_do_not_collide_and_legacy_graph_does_not_acquire_registry_authority():
    assert tool_wire({'tool_id': 'foo-bar', 'version': 1}) != tool_wire({'tool_id': 'foo_bar', 'version': 1})
    value = graph(); value['schema_version'] = 1
    with pytest.raises(DefinitionError): definition(value)
    assert Designer.enabled(value, {}, {}) == []


def test_table_typed_mapping_identifier_and_sensitive_projection():
    cfg = table_settings(); rows, errors = ingest(b'Customer,Amount,Secret\n001,12.50,hidden\n002,8.25,private\n', cfg)
    assert not errors and rows[0]['customer'] == '001' and rows[0]['amount'] == 12.5
    assert 'secret' not in preview_row(rows[0], cfg)
    result = query(rows, query_config(), {}, {'verified': True, 'customer': '001'}, {'identity_field': 'customer'}, cfg['columns'])
    assert result['count'] == 1 and result['source_rows'] == [2]
    result = query(rows, query_config() | {'aggregate': {'field': 'amount', 'operation': 'sum'}}, {}, {}, {}, cfg['columns'])
    assert result['value'] == 20.75
    with pytest.raises(ApplicationError, match='identity_denied'):
        query(rows, query_config(), {}, {}, {'identity_field': 'customer'}, cfg['columns'])
    with pytest.raises(ApplicationError, match='unknown_or_sensitive_field'):
        query(rows, query_config() | {'fields': ['secret']}, {}, {}, {}, cfg['columns'])


def test_inspection_has_no_cell_content_and_keeps_text_codes():
    inspected = inspect_values([['Resident code', 'Amount'], ['001', 1.5]], {'header_row': 1})
    assert inspected['headers'] == ['Resident code', 'Amount']
    assert inspected['columns'][0]['type'] == 'string' and inspected['columns'][1]['type'] == 'number'
    assert '001' not in json.dumps(inspected)
    with pytest.raises(ApplicationError): inspect_values([['Code', 'Code']], {'header_row': 1})


def test_formula_workbook_is_rejected_before_inspection():
    from openpyxl import Workbook
    book = Workbook(); book.active.append(['Code']); book.active.append(['=1+1'])
    stream = io.BytesIO(); book.save(stream)
    with pytest.raises(ApplicationError, match='formula_values_require_export'):
        table_values(stream.getvalue(), {'format': 'xlsx'})


def test_local_documents_rank_and_cite_and_scanned_pdf_is_not_searchable():
    raw = json.dumps({'files': [{'document_id': 'policy', 'name': 'Policy.txt', 'format': 'txt',
                               'content': base64.b64encode(b'Office opening hours are nine to five.').decode()}]}).encode()
    rows, errors = documents(raw, {}); assert not errors
    result = search(rows, 'opening hours', 5)
    assert result['passages'][0]['document_id'] == 'policy' and result['passages'][0]['page'] is None
    from pypdf import PdfWriter
    pdf = PdfWriter(); pdf.add_blank_page(width=300, height=400); stream = io.BytesIO(); pdf.write(stream)
    scanned = json.dumps({'files': [{'document_id': 'scan', 'name': 'Scan.pdf', 'format': 'pdf',
                                   'content': base64.b64encode(stream.getvalue()).decode()}]}).encode()
    rows, _ = documents(scanned, {}); assert rows[0]['state'] == 'needs_verification'
    assert search(rows, 'opening hours', 5)['passages'] == []


@pytest.mark.asyncio
async def test_registry_direct_write_validates_scope_and_records_once_without_confirmation():
    value = graph(); phase = next(n for n in value['nodes'] if n['type'] == 'conversation.collect')
    value['tool_bindings'][0]['scope'] = {'filters': {'customer': '001'}}
    definition_ = {'name': 'Write', 'description': 'Write', 'kind': 'api', 'execution_policy': 'direct', 'timeout_seconds': 10,
                   'input_schema': {'type': 'object'}, 'output_schema': {'type': 'object'},
                   'config': {'operation': {'connector_id': 'crm', 'version': 1, 'operation_id': 'write'}}}
    workflows = SimpleNamespace(db=AsyncMock(return_value=True), check=AsyncMock(), application=SimpleNamespace(key=object()))
    designer = Designer(workflows); designer.execute = AsyncMock(return_value={'saved': True})
    row = {'binding': value['tool_bindings'][0], 'definition': definition_}
    context = {'run_id': 'run', 'workflow_graph': value, 'workflow_node': phase, 'registry_tools': [row],
               '_tool_invocation_id': 'call', 'caller': {'origin': 'external'}}
    assert await designer.invoke(tool_key(binding()), {'customer': '001'}, context) == {'saved': True}
    assert designer.execute.await_count == 1
    assert workflows.db.await_args_list[-1].args[0] == 'invocation_finish'
    assert workflows.db.await_args_list[-1].args[-2] is None  # capture defaults off
    with pytest.raises(ApplicationError, match='caller_scope_mismatch'):
        await designer.invoke(tool_key(binding()), {'customer': '002'}, context)
    workflows.db = AsyncMock(side_effect=[definition_, False])
    with pytest.raises(ApplicationError, match='invocation_already_recorded'):
        await designer.invoke(tool_key(binding()), {'customer': '001'}, context)
    assert designer.execute.await_count == 1


@pytest.mark.asyncio
async def test_simulation_checks_phase_permissions_without_dispatch():
    value = graph(); phase = next(n for n in value['nodes'] if n['type'] == 'conversation.collect')
    published = {'name': 'Lookup', 'description': 'Lookup', 'kind': 'api', 'execution_policy': 'read_only',
                 'input_schema': {'type': 'object'}, 'output_schema': {'type': 'object'}, 'config': {}}
    async def db(method, *args):
        if method == 'tool_version': return published
        if method == 'simulation_save': return args[2] | {'run_id': args[0], 'test_mode': 'mock'}
        raise AssertionError(method)
    workflows = SimpleNamespace(db=db, application=SimpleNamespace(key=object()),
                                test=AsyncMock(return_value={'status': 'completed', 'trace': [{'node_id': phase['id']}]}))
    designer = Designer(workflows); designer.execute = AsyncMock(side_effect=AssertionError('must stay offline'))
    fixtures = {phase['id']: {'outcome': 'success', 'output': {}, 'tool_calls': [{'tool': {'tool_id': 'lookup', 'version': 1}, 'arguments': {}, 'result': {}}]}}
    result = await designer.simulation(value, fixtures, {}, {'origin': 'external'}, [], 'admin')
    assert result['test_mode'] == 'mock' and len(result['tool_calls']) == 1
    designer.execute.assert_not_awaited()
    phase['config'].update(tool_mode='selected', tools=[])
    with pytest.raises(ApplicationError, match='operation_denied'):
        await designer.simulation(value, fixtures, {}, {'origin': 'external'}, [], 'admin')


def test_native_tools_accept_their_exported_dynamic_result_schemas():
    from agent.tools.registry import MANIFESTS
    for manifest in MANIFESTS:
        value = {'name': manifest.id, 'description': manifest.description, 'kind': 'telephony',
                 'input_schema': manifest.input_schema, 'output_schema': manifest.output_schema,
                 'config': {'tool_id': manifest.id}, 'timeout_seconds': 15,
                 'execution_policy': 'read_only' if manifest.read_only else 'direct'}
        assert tool(value) == value


@pytest.mark.asyncio
async def test_private_designer_routes_require_token_actor_and_mock_never_dispatches(monkeypatch):
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    from agent.workflows.api import create_workflow_router
    monkeypatch.setenv('API_TOKEN', 'synthetic-private-token')
    service = SimpleNamespace(require_available=lambda: None, db=AsyncMock(return_value={'tools': [], 'tool_versions': []}))
    app = FastAPI(); app.include_router(create_workflow_router(service))
    prefix = '/api/agent/v1/application/workflows'
    async with AsyncClient(transport=ASGITransport(app), base_url='http://test') as client:
        assert (await client.get(prefix + '/tools')).status_code == 401
        assert (await client.get(prefix + '/tools', headers={'Authorization': 'Bearer synthetic-private-token'})).status_code == 401
        headers = {'Authorization': 'Bearer synthetic-private-token', 'X-Agents-Actor': 'test-admin'}
        assert (await client.get(prefix + '/tools', headers=headers)).json() == {'tools': [], 'tool_versions': []}
        definition_ = {'name': 'Write', 'description': '', 'kind': 'api', 'execution_policy': 'direct', 'timeout_seconds': 10,
                       'input_schema': {'type': 'object', 'additionalProperties': False, 'properties': {}},
                       'output_schema': {'type': 'object', 'additionalProperties': False, 'properties': {}},
                       'config': {'operation': {'connector_id': 'crm', 'operation_id': 'write', 'version': 1}}}
        result = await client.post(prefix + '/tools/test', headers=headers, json={'definition': definition_, 'arguments': {}, 'fixture': {}})
        assert result.status_code == 200 and result.json()['test_mode'] == 'mock'
        # The only database call was the authenticated inventory read. There is
        # deliberately no application transport or dispatcher on this service.
        assert service.db.await_count == 1


@pytest.mark.asyncio
async def test_generic_sheet_publication_keeps_typed_columns_and_kind(monkeypatch):
    import asyncio
    from agent.workflows.service import Workflows
    import agent.workflows.service as module
    cfg = table_settings() | {'format': 'google_sheets', 'secret_ref': 'sheet'}
    monkeypatch.setattr(module, 'google_values', AsyncMock(return_value=[['Customer', 'Amount', 'Secret'], ['001', 12.5, 'private']]))
    instance = Workflows.__new__(Workflows)
    instance._refresh = set(); instance._parse_slots = asyncio.Semaphore(2)
    instance.application = SimpleNamespace(db=AsyncMock(return_value='synthetic'), key=object(), transport=object())
    async def db(method, *args):
        if method == 'data_settings': return {'settings': cfg}
        if method == 'data_publish':
            assert args[4]['kind'] == 'table' and args[4]['columns'] == cfg['columns']
            return {'version': 1}
        raise AssertionError(method)
    instance.db = db
    assert (await instance.refresh_sheet('orders', 1, 'test-admin'))['version'] == 1


@pytest.mark.asyncio
async def test_schema_two_reusable_write_retains_scope_and_schema_one_confirmation():
    from agent.workflows.service import Workflows
    instance = Workflows.__new__(Workflows)
    instance.check = AsyncMock(); instance.application = SimpleNamespace(db=AsyncMock(return_value=True))
    ref = {'connector_id': 'crm', 'operation_id': 'write', 'version': 1}
    item = {'reference': ref, 'operation': {'identity_field': 'customer_id', 'public_voice': False, 'read_only': False}}
    reusable = {'definition': {'execution_policy': 'direct'}, 'binding': {'scope': {'filters': {'customer_id': '001'}}}}
    context = {'workflow_graph': {'schema_version': 2, 'tool_grants': ['connector.crm.write.v1']}, 'workflow_node': {'config': {}},
               '_registry_subflow_binding': reusable, 'execution_kind': 'voice', 'identity': {'customer_id': '001', 'verified': True}}
    await instance.authorize_connector(item, {'customer_id': '001'}, context)
    with pytest.raises(ApplicationError, match='caller_scope_mismatch'):
        await instance.authorize_connector(item, {'customer_id': '002'}, context)
    context['workflow_graph']['schema_version'] = 1
    with pytest.raises(ApplicationError, match='write_confirmation_required'):
        await instance.authorize_connector(item, {'customer_id': '001'}, context)


def test_removing_the_last_document_can_publish_an_empty_collection():
    rows, errors = documents(b'{"files":[]}', {})
    assert rows == [] and errors == []
    assert search(rows, 'opening hours', 5)['passages'] == []


def test_query_numbers_compare_as_json_numbers_without_boolean_coercion():
    from agent.workflows.sources import matches
    assert matches(12.5, 'gt', 12)
    assert matches(1.0, 'eq', 1)
    assert not matches(True, 'eq', 1)
    assert not matches('1', 'eq', 1)
    assert not matches(True, 'in', [1])
