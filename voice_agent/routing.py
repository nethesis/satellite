"""Typed PBX routing policy; imported names and menu text are always untrusted."""
import copy
import re

ID = re.compile(r'(extension|queue|ivr):[0-9]{1,20}\Z')


class RoutingPolicy:
    def __init__(self, snapshot):
        self.binding = next((b for b in snapshot['tool_bindings'] if b['tool_id'] == 'pbx.transfer'), None)
        self.lookup = next((b for b in snapshot['tool_bindings'] if b['tool_id'] == 'pbx.find_recent_callers'), None)

    def recent_enabled(self):
        return bool(self.binding and self.binding['enabled']
                    and self.binding['config'].get('recent_callers_enabled')
                    and self.lookup and self.lookup['enabled'])

    def permits_recent(self, destination_id):
        return bool(self.recent_enabled() and isinstance(destination_id, str)
                    and re.fullmatch(r'extension:[0-9]{1,20}', destination_id)
                    and destination_id not in self.binding['config']['recent_caller_denylist'])

    def permits(self, destination_id):
        if not self.binding or not self.binding['enabled'] or not isinstance(destination_id, str) or not ID.fullmatch(destination_id):
            return False
        config = self.binding['config']
        kind = destination_id.split(':')[0]
        if kind == 'extension':
            return config['extensions_enabled'] and destination_id in config['extension_allowlist']
        return config[kind + 's_enabled'] and destination_id not in config[kind + '_denylist']

    def directory(self, rows):
        inventory = {row['id']: row for row in rows if isinstance(row.get('id'), str) and ID.fullmatch(row['id']) and row.get('available', True)}
        descriptions = {d['destination_id']: d['description'] for d in self.binding['config']['descriptions']} if self.binding else {}

        def describe(row, path, depth, budget):
            item = {'id': row['id'], 'name': str(row['name'])[:100], 'description': str(descriptions.get(row['id'], row.get('description', '')))[:500]}
            if row['id'].startswith('ivr:'):
                if row['id'] in path:
                    item['context_status'] = 'cycle'
                elif depth >= 4:
                    item['context_status'] = 'depth_limit'
                else:
                    item['branches'] = []
                    for branch in row.get('branches', []):
                        if budget[0] <= 0:
                            item['context_status'] = 'branch_limit'
                            break
                        budget[0] -= 1
                        detail = {'selection': str(branch.get('selection', ''))[:30]}
                        target = inventory.get(branch.get('destination_id'))
                        if target and self.permits(target['id']):
                            detail['destination'] = describe(target, path | {row['id']}, depth + 1, budget)
                        else:
                            # Native IVR behavior may reach a destination without a
                            # direct grant. Do not leak denied directory names/IDs.
                            detail['context_status'] = 'native_behavior_only' if target else 'unknown'
                        item['branches'].append(detail)
                    if row.get('context_status'):
                        item['context_status'] = row['context_status']
            return item
        return [describe(inventory[key], set(), 0, [50]) for key in sorted(inventory) if self.permits(key)]
