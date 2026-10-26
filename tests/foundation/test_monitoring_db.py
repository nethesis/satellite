"""Projection checks against the PHP-migrated disposable MariaDB, never a PBX."""
import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import pytest

pytestmark = pytest.mark.skipif(not os.getenv('M0_TEST_PORT'), reason='disposable MariaDB not selected')


def test_idempotent_projection_and_hourly_histograms(monkeypatch):
    import mysql.connector
    import voice_agent.storage as module
    from voice_agent.telemetry import CallTrace
    from voice_agent.monitoring import Monitoring
    from voice_agent.runtime import Call
    @contextmanager
    def database(role):
        assert role == 'monitor'
        db = mysql.connector.connect(host='127.0.0.1', port=int(os.environ['M0_TEST_PORT']), user='root', password='', time_zone='+00:00')
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()
    monkeypatch.setattr(module, 'database', database)
    call = Call(str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4()), 'synthetic',
        {'agent_id': 'receptionist', 'workflow_version': 11, 'max_call_duration_seconds': 60},
        {'binding_id': 'synthetic', 'provider': 'openai', 'adapter': 'realtime', 'model': 'synthetic-model'}, None)
    storage = module.Storage()
    storage.admit(call)
    monitor = Monitoring(storage)
    trace = CallTrace(monitor, call.run_id)
    span = trace.start('tool', tool_id='pbx.transfer')
    trace.end(span)
    trace.usage({'scope':'latest_provider_response','input_tokens':12,'output_tokens':4},'synthetic-response')
    details = trace.finish()
    monitor.emit(call.run_id,'completed',outcome='completed',**details)
    for event in monitor.pending.values():
        storage.event(event)
        storage.event(event)
    with database('monitor') as db:
        cursor=db.cursor(dictionary=True)
        cursor.execute('SELECT duration_ms,usage_json,trace_status,provider,adapter FROM asteriskcdrdb.voice_agent_runs WHERE run_id=%s',(call.run_id,))
        row=cursor.fetchone()
        assert row['trace_status']=='complete'
        assert json.loads(row['usage_json'])['response_input_tokens']==12
        cursor.execute("SELECT SUM(samples) samples FROM asteriskcdrdb.voice_agent_metric_hours WHERE agent_id='receptionist' AND metric='tool_duration' AND dimension='pbx.transfer'")
        assert cursor.fetchone()['samples']==1
        cursor.execute('SELECT COUNT(*) count FROM asteriskcdrdb.voice_agent_run_events WHERE run_id=%s',(call.run_id,))
        assert cursor.fetchone()['count']==len(monitor.pending)
        # A DB failure after event insertion must roll back the event and rollup together.
        cursor.execute("CREATE TRIGGER asteriskcdrdb.synthetic_metric_failure BEFORE INSERT ON asteriskcdrdb.voice_agent_metric_hours FOR EACH ROW SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT='synthetic failure'")
    event={'event_id':str(uuid.uuid4()),'run_id':call.run_id,'event_type':'trace_finished','outcome':None,
        'error_code':None,'tool_id':None,'destination_id':None,'monitoring_version':2,'occurred_at':datetime.now(timezone.utc).isoformat(),
        'span_id':str(uuid.uuid4()),'elapsed_ms':300,'phase':'provider_sideband','status':'completed','duration_ms':200}
    try:
        with pytest.raises(mysql.connector.Error): storage.event(event)
        with database('monitor') as db:
            cursor=db.cursor()
            cursor.execute('SELECT COUNT(*) FROM asteriskcdrdb.voice_agent_run_events WHERE event_id=%s',(event['event_id'],))
            assert cursor.fetchone()[0]==0
    finally:
        with database('monitor') as db:
            db.cursor().execute('DROP TRIGGER asteriskcdrdb.synthetic_metric_failure')
    storage.health({**monitor.health(),'ari_connected':True,'active_calls':0})
    # Native handoff can commit before cleanup/final-summary spooling.
    committed=Call(str(uuid.uuid4()),str(uuid.uuid4()),str(uuid.uuid4()),'synthetic',call.snapshot,call.binding,None)
    storage.admit(committed)
    storage.event({'event_id':str(uuid.uuid4()),'run_id':committed.run_id,'event_type':'transfer_committed',
        'outcome':'transferred','error_code':None,'tool_id':'pbx.transfer','destination_id':'extension:203',
        'monitoring_version':2,'occurred_at':datetime.now(timezone.utc).isoformat()})
    newer=Call(str(uuid.uuid4()),str(uuid.uuid4()),str(uuid.uuid4()),'synthetic',call.snapshot,call.binding,None)
    newer.started_at=datetime.now(timezone.utc)+timedelta(seconds=10)
    storage.admit(newer)
    storage.recover()
    with database('monitor') as db:
        cursor=db.cursor(dictionary=True)
        cursor.execute('SELECT outcome,trace_status FROM asteriskcdrdb.voice_agent_runs WHERE run_id=%s',(committed.run_id,))
        assert cursor.fetchone()=={'outcome':'transferred','trace_status':'partial'}
        cursor.execute('SELECT outcome,trace_status FROM asteriskcdrdb.voice_agent_runs WHERE run_id=%s',(newer.run_id,))
        assert cursor.fetchone()=={'outcome':'running','trace_status':'recording'}
        cursor.execute('SELECT state FROM asteriskcdrdb.voice_agent_pending_calls WHERE run_id=%s',(newer.run_id,))
        assert cursor.fetchone()['state']=='pending'
