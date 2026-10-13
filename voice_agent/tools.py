"""Allowlisted v0.1 tools. Prompts and SIP headers never grant permissions."""
import asyncio
import copy
import json
from pathlib import Path

DEFINITIONS = json.loads(Path(__file__).with_name('tools-v1.json').read_text())
MANIFESTS = {item['id']:{'type':'function','name':item['wire_name'],'description':item['description'],'parameters':item['input_schema']} for item in DEFINITIONS}
WIRE_IDS = {value['name']: key for key, value in MANIFESTS.items()}


class ToolRegistry:
    def __init__(self, snapshot, targets, controller, definitions=None):
        if definitions is not None and sorted(definitions,key=lambda item:item['id']) != sorted(DEFINITIONS,key=lambda item:item['id']):
            raise ValueError('unsupported_tool_definition')
        self.snapshot = copy.deepcopy(snapshot)
        self.targets = targets
        self.controller = controller
        self.invocations = {}
        self.lock = asyncio.Lock()

    def advertised(self):
        return [copy.deepcopy(MANIFESTS[key]) for key in self.snapshot['enabled_tools'] if key in MANIFESTS]

    async def invoke(self, invocation_id, name, arguments):
        if not isinstance(name,str) or len(name)>128:
            return {'error': 'tool_denied'}
        try:
            if len(json.dumps(arguments).encode())>8192:
                return {'error': 'invalid_arguments'}
        except (TypeError,ValueError,RecursionError):
            return {'error': 'invalid_arguments'}
        if not isinstance(invocation_id, str) or not 1 <= len(invocation_id) <= 128:
            return {'error': 'invalid_invocation'}
        async with self.lock:
            if invocation_id not in self.invocations:
                if len(self.invocations) >= 256:
                    return {'error': 'tool_capacity_exceeded'}
                self.invocations[invocation_id] = asyncio.create_task(self._execute(name, copy.deepcopy(arguments)))
            task = self.invocations[invocation_id]
        return copy.deepcopy(await asyncio.shield(task))

    async def _execute(self, name, arguments):
        tool_id = WIRE_IDS.get(name, name)
        if tool_id not in MANIFESTS or tool_id not in self.snapshot['enabled_tools']:
            return {'error': 'tool_denied'}
        if not isinstance(arguments, dict):
            return {'error': 'invalid_arguments'}
        allowed = set(self.snapshot['approved_destination_ids'])
        try:
            rows = await asyncio.wait_for(asyncio.to_thread(self.targets), timeout=5)
            rows = [row for row in rows if row['id'] in allowed]
            if tool_id == 'directory.find_destinations':
                query = arguments.get('query', '')
                if set(arguments) - {'query'} or not isinstance(query, str) or len(query) > 100:
                    return {'error': 'invalid_arguments'}
                query = query.casefold()
                return {'destinations': [{'id': row['id'], 'name': row['name']} for row in rows if query in row['id'].casefold() or query in row['name'].casefold()][:100]}
            if set(arguments) != {'destination_id'} or not isinstance(arguments['destination_id'], str):
                return {'error': 'invalid_arguments'}
            target = arguments['destination_id']
            if target not in {row['id'] for row in rows}:
                return {'error': 'destination_denied'}
            return await asyncio.wait_for(self.controller.transfer(target), timeout=15)
        except (Exception, asyncio.TimeoutError):
            return {'error': 'tool_unavailable'}

    async def close(self, exclude=None):
        tasks = [task for task in self.invocations.values() if task is not exclude]
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
