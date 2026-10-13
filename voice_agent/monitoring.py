"""Bounded metadata-only outbox. Monitoring outages do not block active audio."""
import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path


class Monitoring:
    def __init__(self, storage, directory=None):
        self.storage = storage
        self.queue = asyncio.Queue(maxsize=256)
        self.degraded = False
        self.worker = None
        self.directory = Path(directory or os.environ['VOICE_AGENT_EVENT_DIR']) if directory or os.getenv('VOICE_AGENT_EVENT_DIR') else None
        if self.directory:
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            self.directory.chmod(0o700)
            self.degraded=(self.directory/'monitoring-degraded').exists()
        self.pending = {}

    async def start(self):
        if self.directory:
            for path in sorted(self.directory.glob('*.json'))[:256]:
                try:
                    if path.stat().st_size > 4096:
                        raise ValueError('invalid_outbox_event')
                    event = json.loads(path.read_text())
                    if set(event)!={'event_id','run_id','event_type','outcome','error_code','tool_id','destination_id','occurred_at'} or str(uuid.UUID(event['event_id'])) != path.stem:
                        raise ValueError('invalid_outbox_event')
                    uuid.UUID(event['run_id'])
                    datetime.fromisoformat(event['occurred_at'])
                    self.pending[event['event_id']] = event
                except Exception:
                    self.degraded = True
            for event in sorted(self.pending.values(),key=lambda item:item['occurred_at']):
                self.queue.put_nowait(event)
        self.worker = asyncio.create_task(self._write())

    def mark_degraded(self):
        self.degraded=True
        if self.directory:
            try:
                marker=self.directory/'monitoring-degraded'
                descriptor=os.open(marker,os.O_WRONLY|os.O_CREAT,0o600)
                os.close(descriptor)
            except OSError:
                pass

    def emit(self, run_id, kind, *, outcome=None, error=None, key=None, tool_id=None, destination_id=None):
        event_id = str(uuid.uuid5(uuid.UUID(run_id), key or kind))
        if event_id in self.pending:
            return
        event = {'event_id':event_id,'run_id':run_id,'event_type':kind,'outcome':outcome,
                 'error_code':error,'tool_id':tool_id,'destination_id':destination_id,
                 'occurred_at':datetime.now(timezone.utc).isoformat()}
        # Reserve room for all 32 admitted calls' terminal outcomes during outages.
        terminal=outcome is not None
        if len(self.pending) >= (256 if terminal else 192) or self.queue.full():
            self.mark_degraded()
            return
        if self.directory:
            path = self.directory / (event_id + '.json')
            temporary = path.with_suffix('.tmp')
            try:
                with temporary.open('w',encoding='utf-8') as stream:
                    os.chmod(temporary,0o600)
                    json.dump(event,stream,separators=(',',':'))
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(path)
            except OSError:
                self.mark_degraded()
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
        self.pending[event_id] = event
        self.queue.put_nowait(event)

    async def _write(self):
        while True:
            event = await self.queue.get()
            try:
                while True:
                    try:
                        await asyncio.wait_for(asyncio.to_thread(self.storage.event,event),timeout=5)
                        self.pending.pop(event['event_id'],None)
                        if self.directory:
                            (self.directory / (event['event_id']+'.json')).unlink(missing_ok=True)
                        break
                    except Exception:
                        self.mark_degraded()
                        await asyncio.sleep(1)
            finally:
                self.queue.task_done()

    async def stop(self):
        try:
            await asyncio.wait_for(self.queue.join(),timeout=5)
        except asyncio.TimeoutError:
            self.degraded = True
        if self.worker:
            self.worker.cancel()
            await asyncio.gather(self.worker,return_exceptions=True)
