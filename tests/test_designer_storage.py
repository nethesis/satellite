"""Explicitly isolated PostgreSQL acceptance for the designer migration."""
import base64
import os
import time

import pytest

from agent.application.contracts import ApplicationError
from agent.application.crypto import ContentKey
from agent.monitoring.repository import HistoryRepository
from agent.workflows.repository import WorkflowRepository
from agent.workflows.templates import templates
from tests.test_designer import table_settings, query_config, binding

pytestmark = pytest.mark.skipif(
    os.getenv('SATELLITE_DESIGNER_ACCEPTANCE') != 'isolated' or
    os.getenv('PGVECTOR_HOST') != '127.0.0.1' or os.getenv('PGVECTOR_PORT') != '54329',
    reason='requires the explicitly isolated designer database on 127.0.0.1:54329')


@pytest.fixture
def storage():
    repo = WorkflowRepository()
    with repo.connect() as db:
        for schema in ('agent_workflows', 'agent_application', 'agent_monitoring'):
            db.execute('DROP SCHEMA IF EXISTS ' + schema + ' CASCADE')
    repo.initialize('designer-test'); HistoryRepository().initialize('designer-test')
    repo.initialize_workflows('designer-test')
    key = ContentKey(base64.b64encode(b'd' * 32).decode())
    return repo, key


def seed(storage):
    repo, key = storage
    cfg = table_settings(); repo.data_save('orders', cfg, 0, 'test-admin')
    columns = cfg['columns']
    repo.data_publish('orders', 1, [{'customer': '001', 'amount': 12.5, 'secret': 'private', 'source_row': 2}],
                      None, {'kind': 'table', 'columns': columns}, key, 'test-admin')
    draft = {'name': 'Order totals', 'description': 'Synthetic orders', 'kind': 'query', 'config': query_config(),
             'input_schema': {'type': 'object', 'properties': {}, 'additionalProperties': False}, 'output_schema': {'type': 'object', 'properties': {}, 'additionalProperties': False},
             'execution_policy': 'read_only', 'timeout_seconds': 10}
    repo.tool_save('lookup', draft, 0, 'test-admin'); repo.tool_publish('lookup', 1, 'test-admin')
    graph = next(g for g in templates() if g['agent_id'] == 'blank-api')
    graph.update(schema_version=2, tool_bindings=[binding()], goal='', instructions='', capture_tool_content=False)
    return repo, key, graph


def test_publication_cas_dependencies_and_versioned_source_metadata(storage):
    repo, key, graph = seed(storage)
    repo.workflow_save('agent', graph['agent_id'], graph, 0, 'test-admin')
    result = repo.workflow_publish('agent', graph['agent_id'], 1, 'test-admin', 'Reviewed')
    assert result['version'] == 1
    with pytest.raises(ApplicationError, match='revision_conflict'):
        repo.workflow_publish('agent', graph['agent_id'], 1, 'test-admin')
    assert any(d['resource_id'] == 'lookup' for d in repo.designer_dependencies())
    # Editing the source draft does not invalidate a tool pinned to its published columns.
    cfg = table_settings(); cfg['columns'][1]['name'] = 'changed'
    repo.data_save('orders', cfg, 2, 'test-admin')
    with repo.connect() as db: repo.validate_references(db, graph)
    with repo.connect() as db:
        assert db.execute('SELECT note FROM agent_workflows.publication_notes').fetchone()['note'] == 'Reviewed'


def test_encrypted_call_content_dedup_restart_and_expiry(storage):
    repo, key, graph = seed(storage)
    repo.execution_admit('run', graph, 1, 'api', 'test-admin', 'designer-test')
    context = {'run_id': 'run', 'workflow_node': {'id': 'done'}, 'subflow_path': []}
    assert repo.invocation_start(context, 'call', {'tool_id': 'lookup', 'version': 1}, {'secret': 'private'}, 'conversation')
    assert not repo.invocation_start(context, 'call', {'tool_id': 'lookup', 'version': 1}, {'secret': 'private'}, 'conversation')
    with pytest.raises(ApplicationError, match='invocation_conflict'):
        repo.invocation_start(context, 'call', {'tool_id': 'lookup', 'version': 1}, {'secret': 'other'}, 'conversation')
    repo.invocation_finish('run', 'call', 5, None, {'result': 'private'}, key)
    metadata = repo.invocation_records('run')
    assert 'private' not in str(metadata) and metadata[0]['content_available']
    assert repo.invocation_content('run', 'call', key, 'test-admin') == {'result': 'private'}
    repo.invocation_start(context, 'pending', {'tool_id': 'lookup', 'version': 1}, {}, 'workflow')
    repo.initialize_workflows('new-epoch')
    assert repo.invocation_records('run')[1]['state'] == 'interrupted'
    with repo.connect() as db: db.execute('UPDATE agent_workflows.tool_calls SET content_expires=0')
    repo.designer_purge()
    with pytest.raises(ApplicationError, match='content_unavailable'):
        repo.invocation_content('run', 'call', key, 'test-admin')


def test_saved_simulation_and_activity_counts_do_not_duplicate_workflows(storage):
    repo, key, graph = seed(storage)
    repo.execution_admit('run', graph, 1, 'api', 'test-admin', 'designer-test')
    result = {'status': 'completed', 'result': {}, 'trace': [{'node_id': 'done', 'block_type': 'end', 'outcome': 'success', 'error_code': None, 'output': 'private'}]}
    repo.simulation_save('mock', graph, result, [{'role': 'caller', 'text': 'private'}], key, 'test-admin')
    detail = repo.simulation_detail('mock', key, 'test-admin')
    assert 'private' not in str(detail['trace']) and detail['content']['messages'][0]['text'] == 'private'
    activity = repo.designer_activity(0, time.time() + 1, '', '', '', '', 1)
    assert len(activity['runs']) == 1 and activity['next_cursor']
    assert sum(r['count'] for r in activity['counts']) == 2
    next_page = repo.designer_activity(0, time.time() + 1, '', '', '', activity['next_cursor'], 1)
    assert next_page['runs'][0]['run_id'] != activity['runs'][0]['run_id']
    assert repo.designer_activity(0, time.time() + 1, '', '', '', '', 10, False)['runs']


def test_legacy_api_monitoring_record_does_not_duplicate_activity(storage):
    repo, key, graph = seed(storage)
    now = time.time()
    with repo.connect() as db:
        db.execute("INSERT INTO agent_application.runs(run_id,client_id,idempotency_hash,request_digest,preset_version,revision,started,result_hours,metadata,expires) VALUES('legacy','fixture','once','digest',1,1,%s,24,'{}',%s)", (now, now + 86400))
        db.execute("INSERT INTO agent_monitoring.runs(run_id,agent_id,execution_kind,started,capture_version,transcript_state,expires) VALUES('legacy','support-request','api',%s,1,'disabled',%s)", (now, now + 86400))
    result = repo.designer_activity(0, now + 1, '', '', '', '', 10)
    assert len(result['runs']) == 1 and result['runs'][0]['run_id'] == 'legacy'
    assert sum(row['count'] for row in result['counts']) == 1
