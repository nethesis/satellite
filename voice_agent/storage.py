"""Least-privilege MariaDB connections and durable pending-call correlation."""
import base64
import hashlib
import json
import os
import re
from datetime import datetime
from contextlib import contextmanager

from mysql.connector import connect
from nacl.secret import SecretBox


@contextmanager
def database(role):
    connection = connect(host='127.0.0.1', port=int(os.environ['VOICE_AGENT_DB_PORT']),
                         user='voice_agent_config' if role == 'config' else 'voice_agent_monitor',
                         password=os.environ['VOICE_AGENT_CONFIG_PASSWORD' if role == 'config' else 'VOICE_AGENT_MONITOR_PASSWORD'],
                         connection_timeout=3, read_timeout=3, write_timeout=3, time_zone='+00:00')
    try:
        yield connection
        if role == 'monitor':
            connection.commit()
    finally:
        connection.close()


class Storage:
    def snapshot(self, agent_id):
        from .repository import PublishedRepository
        return PublishedRepository(lambda: connect(host='127.0.0.1', port=int(os.environ['VOICE_AGENT_DB_PORT']),
            user='voice_agent_config', password=os.environ['VOICE_AGENT_CONFIG_PASSWORD'],
            connection_timeout=3, read_timeout=3, write_timeout=3)).load(agent_id)

    def binding(self, binding_id):
        with database('config') as db:
            with db.cursor(dictionary=True) as cursor:
                cursor.execute('SELECT binding_id,provider,settings_json,secrets_encrypted,freepbx_trunk_id,trunk_name FROM asterisk.voice_agent_runtime_bindings WHERE binding_id=%s', (binding_id,))
                row = cursor.fetchone()
        if not row:
            raise ValueError('binding_unavailable')
        master = os.environ['SATELLITE_AGENT_CONFIG_KEY'].encode()
        secret = SecretBox(hashlib.sha256(master).digest()).decrypt(base64.b64decode(row.pop('secrets_encrypted'), validate=True))
        row.update(json.loads(secret))
        if not row.get('api_key'):
            names=('OPENAI_API_KEY',) if row['provider']=='openai' else ('XAI_API_KEY','GROK_API_KEY')
            row['api_key']=next((os.environ[name] for name in names if os.environ.get(name)),None)
        if not isinstance(row.get('api_key'),str) or not row['api_key'] or any(c in row['api_key'] for c in '\r\n\x00'):
            raise ValueError('provider_credentials_unavailable')
        row.update(json.loads(row.pop('settings_json')))
        if not re.fullmatch(r'VoiceAgentTrunk_[1-9][0-9]*', row['trunk_name']):
            raise ValueError('invalid_native_binding')
        return row

    def targets(self):
        with database('config') as db:
            with db.cursor(dictionary=True) as cursor:
                cursor.execute('SELECT id,name FROM asterisk.voice_agent_extension_directory LIMIT 1000')
                return cursor.fetchall()

    def tools(self):
        with database('config') as db:
            with db.cursor(dictionary=True) as cursor:
                cursor.execute('SELECT tool_id,tool_version,manifest_json FROM asterisk.voice_agent_runtime_tools')
                return [json.loads(row['manifest_json']) for row in cursor.fetchall()]

    def admit(self, call):
        with database('monitor') as db:
            with db.cursor() as cursor:
                cursor.execute('INSERT INTO asteriskcdrdb.voice_agent_runs (run_id,agent_id,workflow_version,linkedid,retention_until) VALUES (%s,%s,%s,%s,DATE_ADD(NOW(),INTERVAL 7 DAY))', (call.run_id, call.snapshot['agent_id'], call.snapshot['workflow_version'], call.linkedid))
                cursor.execute('INSERT INTO asteriskcdrdb.voice_agent_pending_calls (session_id,run_id,provider,binding_id,agent_id,provider_leg_id,expires_at) VALUES (%s,%s,%s,%s,%s,%s,DATE_ADD(NOW(),INTERVAL 30 SECOND))',
                    (call.session_id, call.run_id, call.binding['provider'], call.binding['binding_id'], call.snapshot['agent_id'], call.leg_id))

    def accept(self, call, provider_call_id):
        with database('monitor') as db:
            with db.cursor() as cursor:
                cursor.execute("UPDATE asteriskcdrdb.voice_agent_pending_calls SET state='accepted',provider_call_id=%s WHERE session_id=%s AND state='pending' AND expires_at>NOW()", (provider_call_id, call.session_id))
                if cursor.rowcount != 1:
                    raise ValueError('pending_call_expired')
                cursor.execute('UPDATE asteriskcdrdb.voice_agent_runs SET provider_call_id=%s WHERE run_id=%s', (provider_call_id, call.run_id))

    def event(self, event):
        with database('monitor') as db:
            with db.cursor() as cursor:
                occurred=datetime.fromisoformat(event['occurred_at']).replace(tzinfo=None)
                cursor.execute('INSERT IGNORE INTO asteriskcdrdb.voice_agent_run_events (event_id,run_id,event_type,error_code,tool_id,destination_id,occurred_at) VALUES (%s,%s,%s,%s,%s,%s,%s)', (event['event_id'], event['run_id'], event['event_type'], event.get('error_code'),event.get('tool_id'),event.get('destination_id'),occurred))
                if event.get('outcome'):
                    cursor.execute("UPDATE asteriskcdrdb.voice_agent_runs SET ended_at=%s,outcome=%s,error_code=%s WHERE run_id=%s AND (outcome='running' OR (outcome='interrupted' AND error_code='runtime_restarted'))", (occurred,event['outcome'],event.get('error_code'),event['run_id']))
                    cursor.execute("UPDATE asteriskcdrdb.voice_agent_pending_calls SET state='finished' WHERE run_id=%s", (event['run_id'],))

    def recover(self):
        with database('monitor') as db:
            with db.cursor() as cursor:
                cursor.execute("UPDATE asteriskcdrdb.voice_agent_runs SET ended_at=NOW(6),outcome='interrupted',error_code='runtime_restarted' WHERE outcome='running'")
                cursor.execute("UPDATE asteriskcdrdb.voice_agent_pending_calls SET state='finished' WHERE state<>'finished'")

    def expire_pending(self):
        with database('monitor') as db:
            with db.cursor() as cursor:
                cursor.execute("UPDATE asteriskcdrdb.voice_agent_runs r JOIN asteriskcdrdb.voice_agent_pending_calls p ON p.run_id=r.run_id SET r.ended_at=NOW(6),r.outcome='fallback',r.error_code='setup_expired' WHERE p.state='pending' AND p.expires_at<NOW() AND r.outcome='running'")
                cursor.execute("UPDATE asteriskcdrdb.voice_agent_pending_calls SET state='finished' WHERE state='pending' AND expires_at<NOW()")


def trusted_target(destination_id):
    if not isinstance(destination_id, str) or not re.fullmatch(r'extension:[0-9]{1,20}', destination_id):
        raise ValueError('invalid_destination')
    return {'context': 'voice-agent-transfer', 'extension': destination_id.split(':', 1)[1], 'priority': 1}
