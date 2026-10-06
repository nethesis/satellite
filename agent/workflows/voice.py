"""Caller DTMF confirmations and private, extension-first consultations."""

import asyncio
import secrets
import time
from dataclasses import dataclass, field

from agent.application.contracts import ApplicationError
from agent.context import allowed, visible


@dataclass
class Consultation:
    call: object
    attempt_id: str
    channel_id: str
    answered: asyncio.Future
    decision: asyncio.Future
    bridge_id: str
    released: bool = False
    destroyed: bool = False
    accepting: bool = False
    operator_connected: bool = False
    end_outcome: str = "unavailable"


class VoiceWorkflows:
    def __init__(self, runtime):
        self.runtime = runtime
        self.consultations = {}
        self.digits = {}

    async def event(self, event):
        channel_id = (event.get("channel") or {}).get("id")
        kind = event.get("type")
        if kind == "ChannelDtmfReceived" and channel_id in self.digits:
            future = self.digits[channel_id]
            digit = event.get("digit")
            if not future.done() and digit in ("1", "2"):
                future.set_result(digit == "1")
            return True
        attempt = self.consultations.get(channel_id)
        if not attempt:
            return False
        if kind == "StasisStart":
            expected = ["consult", attempt.call.session_id, attempt.attempt_id]
            if event.get("args") != expected:
                await self.runtime.controller.hangup(channel_id)
            elif not attempt.answered.done():
                attempt.operator_connected = True
                attempt.answered.set_result(True)
        elif kind == "ChannelDtmfReceived" and attempt.accepting and not attempt.decision.done():
            if event.get("digit") in ("1", "2"):
                attempt.decision.set_result("accepted" if event["digit"] == "1" else "declined")
        elif kind in ("ChannelDestroyed", "StasisEnd") and not attempt.released:
            attempt.destroyed = True
            attempt.end_outcome = "busy" if event.get("cause") == 17 else "no_answer" if event.get("cause") in (18, 19) else "unavailable"
            if not attempt.answered.done():
                attempt.answered.set_result(False)
            if not attempt.decision.done():
                attempt.decision.set_result("unavailable")
        return True

    async def confirm(self, call, prompt):
        if call.caller_id in self.digits:
            raise ApplicationError("confirmation_in_progress", 409)
        # Only digits received after read-back has finished can authorize this action.
        await self.runtime.workflow_speak(call, prompt + " Press 1 to confirm or 2 to decline.")
        future = asyncio.get_running_loop().create_future()
        self.digits[call.caller_id] = future
        try:
            return await asyncio.wait_for(future, min(30, max(0, call.deadline - time.monotonic())))
        except asyncio.TimeoutError:
            return False
        finally:
            self.digits.pop(call.caller_id, None)

    async def consult(self, call, destination_id, summary, cfg):
        from agent.runtime import CallState
        target = next((d for d in call.directory if d["id"] == destination_id), None)
        context = self.runtime._context(call)
        if not target or target["type"] != "extension" or not visible(context, target) or not allowed(context, "telephony.consultative_transfer") or not allowed(context, "telephony.transfer.extension"):
            return {"status": "unavailable"}
        extension = destination_id.split(":", 1)[-1]
        if not extension.isdigit():
            return {"status": "unavailable"}
        async with call.lock:
            if call.terminal or call.state != CallState.CONVERSING:
                raise ApplicationError("call_not_conversing", 409)
            call.state = CallState.CONSULTING
            call.state_revision += 1
        loop = asyncio.get_running_loop()
        attempt = Consultation(call, secrets.token_hex(16), "consult-" + secrets.token_hex(10),
            loop.create_future(), loop.create_future(), "agent-consult-" + secrets.token_hex(16))
        self.consultations[attempt.channel_id] = attempt
        outcome = "failed"; moved = False; bridge_created = False; muted = False
        try:
            await call.adapter.workflow_update("Wait silently; a private consultation is being arranged.", [], auto_response=False)
            await self.runtime.controller.remove_from_bridge(call.bridge_id, [call.local_id])
            moved = True
            await self.runtime.controller.moh(call.caller_id, True)
            await self.runtime.controller.originate_consult(call.session_id, attempt.attempt_id, attempt.channel_id, extension, cfg["ring_seconds"])
            answered = await asyncio.wait_for(attempt.answered, cfg["ring_seconds"] + 2)
            if not answered:
                outcome = attempt.end_outcome
            else:
                await self.runtime.controller.create_bridge(attempt.bridge_id); bridge_created = True
                # The summary reaches the operator, while audio sent from the
                # PBX to the provider is muted. Operator speech never becomes
                # caller-session context; acceptance uses the operator's DTMF.
                muted = True
                await self.runtime.controller.mute(call.local_id, True, 'out')
                call.private_consultation = True
                await self.runtime.controller.add_to_bridge(attempt.bridge_id, [call.local_id, attempt.channel_id])
                await call.adapter.workflow_update("You are speaking privately to the operator. Summarize only the supplied issue and ask them to press 1 to accept or 2 to decline. Do not repeat any private operator speech to the caller.", [], auto_response=False)
                await call.adapter.workflow_input({"summary": summary, "decision_prompt": "Press 1 to accept this caller or 2 to decline."})
                await call.adapter.workflow_respond(wait=True)
                attempt.accepting = True
                outcome = await asyncio.wait_for(attempt.decision, cfg["consult_seconds"])
                if outcome == "accepted" and (call.terminal or attempt.destroyed):
                    outcome = "unavailable"
                if outcome == "accepted" and not call.terminal and not attempt.destroyed:
                    # Release the answered operator into a native holding bridge,
                    # then Bridge the original caller onto that existing leg.
                    async with call.lock:
                        if call.terminal:
                            raise ApplicationError("caller_disconnected")
                        await self.runtime.controller.remove_from_bridge(attempt.bridge_id, [call.local_id, attempt.channel_id])
                        await self.runtime.controller.set_variable(call.caller_id, "AGENT_CONSULT_CHANNEL", attempt.channel_id)
                        await self.runtime.controller.set_variable(call.caller_id, "AGENT_CONSULT_RETURN_CONTEXT", "satellite-agent-destination-" + call.destination_id)
                        await self.runtime.controller.set_variable(call.caller_id, "TIMEOUT(absolute)", 30)
                        await self.runtime.controller.set_variable(call.caller_id, "AGENT_HANDOFF_ATTEMPT_ID", attempt.attempt_id)
                        call.handoff_attempt_id = attempt.attempt_id
                        call.handoff_target_id = destination_id
                        call.state = CallState.HANDING_OFF; call.state_revision += 1
                        attempt.released = True
                        await self.runtime.controller.continue_channel(attempt.channel_id, {"context": "satellite-agent-consult-wait", "exten": "s", "priority": 1})
                        await self.runtime.controller.moh(call.caller_id, False)
                        try:
                            await asyncio.wait_for(self.runtime.controller.continue_channel(call.caller_id,
                                {"context": "satellite-agent-consult-connect", "exten": "s", "priority": 1}), 10)
                        except BaseException:
                            call.handoff_ambiguous = True
                            self.runtime._spawn(self.runtime._reconcile_handoff(call))
                            outcome = "unknown"
                        else:
                            call.handed_off = True; call.terminal = True; call.cancellation.set()
                            call.state = CallState.TERMINATING; call.state_revision += 1
                    if call.handed_off:
                        await self.runtime._cleanup(call, "consultative_transfer", caller_action=None)
        except asyncio.TimeoutError:
            outcome = "no_answer" if not attempt.operator_connected else "unavailable"
        except asyncio.CancelledError:
            raise
        except Exception:
            outcome = "unknown" if call.handoff_attempt_id else "failed"
        finally:
            self.consultations.pop(attempt.channel_id, None)
            call.private_consultation = False
            if not attempt.released:
                try:
                    await self.runtime.controller.hangup(attempt.channel_id)
                except Exception:
                    pass
            if bridge_created:
                try:
                    await self.runtime.controller.destroy_bridge(attempt.bridge_id)
                except Exception:
                    pass
            if moved and not call.terminal and not call.handoff_attempt_id:
                try:
                    await self.runtime.controller.add_to_bridge(call.bridge_id, [call.local_id])
                    if muted:
                        await self.runtime.controller.mute(call.local_id, False, 'out')
                    await self.runtime.controller.moh(call.caller_id, False)
                    call.state = CallState.CONVERSING; call.state_revision += 1
                    await call.adapter.workflow_update("Resume speaking to the original caller. The consultation did not connect. Do not disclose private operator speech.", [], auto_response=False)
                except Exception:
                    await self.runtime._finish(call, "consultation_resume_failed", fallback=True)
        self.runtime.events.emit("workflow.consultation", run_id=call.run_id, attempt_id=attempt.attempt_id,
                                 destination_id=destination_id, outcome=outcome)
        return {"status": outcome, "destination_id": destination_id}
