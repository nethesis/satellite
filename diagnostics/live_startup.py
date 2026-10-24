"""Synthetic-only adapter variants, independent of the production factory.

An external harness must select this adapter for an exactly owned call. These
helpers have no environment switches, credentials, logging, or runtime imports.
"""
import asyncio
import time

from voice_agent.providers.live import OpenAILiveAdapter
from .live_audio import SignalStats


MEDIA_READY = ('Caller media is now connected. Say the prepared opening greeting '
               'now in the configured language, without delegation, then listen.')


class DiagnosticLiveAdapter(OpenAILiveAdapter):
    def __init__(self, binding, *, variant='current', session=None, clock=time.monotonic):
        if variant not in ('current', 'concise'):
            raise ValueError('invalid_diagnostic_variant')
        super().__init__(binding, session=session)
        self.variant, self.clock = variant, clock
        self.started = clock()
        self.input_stats, self.output_stats = SignalStats(clock=clock), SignalStats(clock=clock)
        self.commands = []
        self.command_overflow = 0
        self.errors = 0

    async def accept(self, provider_call_id, profile, tools):
        # The investigation never exports or persists content, regardless of the
        # operator's unchanged published transcript-retention setting.
        await super().accept(provider_call_id, {**profile, '_capture_transcripts': False}, tools)

    async def _command(self, kind, expected, **fields):
        record = None
        if len(self.commands) < 64:
            record = {'kind': kind if kind in ('session.instructions.append', 'session.commentary.append', 'session.update') else 'other',
                      'sent_ms': round((self.clock() - self.started) * 1000, 3), 'outcome': 'pending'}
            self.commands.append(record)
        else:
            self.command_overflow += 1
        try:
            await super()._command(kind, expected, **fields)
            outcome = 'acknowledged'
        except asyncio.CancelledError:
            outcome = 'cancelled'
            raise
        except asyncio.TimeoutError:
            outcome = 'timeout'
            raise
        except Exception:
            outcome = 'rejected_or_closed'
            raise
        finally:
            if record is not None:
                record.update(outcome=outcome, finished_ms=round((self.clock() - self.started) * 1000, 3))

    async def respond(self, instructions=None, *, response_id=None):
        if self.variant == 'concise' and response_id is None:
            # Full opening remains seeded at acceptance. One short notification
            # avoids serially injecting that same text a second time.
            await self._append('instructions', MEDIA_READY)
        else:
            await super().respond(instructions, response_id=response_id)

    async def _handle(self, raw):
        kind = raw.get('type')
        if kind == 'session.input_audio.append':
            self.input_stats.encoded(raw.get('audio'))
            return
        if kind == 'session.output_audio.delta':
            self.output_stats.encoded(raw.get('delta'))
            return
        if kind == 'error':
            self.errors += 1
        # Transcript deltas are deliberately discarded before the normal handler.
        if kind not in ('session.input_transcript.delta', 'session.output_transcript.delta'):
            await super()._handle(raw)

    def measurements(self):
        return {'clock_origin_ms': round(self.started * 1000, 3), 'variant': self.variant, 'commands': [dict(c) for c in self.commands],
                'command_overflow': self.command_overflow, 'provider_errors': self.errors,
                'provider_input': self.input_stats.snapshot(), 'provider_output': self.output_stats.snapshot()}
