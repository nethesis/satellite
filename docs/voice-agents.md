# Voice agents runtime alpha

The `voice_agent` package is independent of Satellite transcription and its
PostgreSQL store. PHP owns all agent configuration in MariaDB. This runtime reads
only **v8** immutable publications; older development databases and contracts
require a fresh installation of the matching NethVoice alpha. There is no config
writer, migration service or configuration-push endpoint.

The mirrored `published-v8.schema.json`, `tools-v5.json` and
`provider-models-v2.json` must exactly match PHP. A detached publication is pinned
at admission. Each enabled tool uses its own agent binding and effective grants;
no catalog-wide permission exists. The current catalog contains typed PBX search,
transfer, recent-caller lookup and optional owned-channel hangup.

The runner validates contract, provider capability and trusted tool definitions
before paid provider usage. The conversation's model and voice belong to the
agent; stale trunk model settings cannot override them. Direct extensions are
initially disabled, queues enabled, callbacks enabled for seven
days, and call ending disabled. Callback evidence is short-lived and bound to
the inbound caller, live session, agent and publication. A model cannot supply
an authoritative number or turn a direct destination into callback evidence.
IVR routing is removed: directories, model advertisements and transfer execution
support extensions and queues only. Existing v8 IVR fields are inert and do not
authorize transfers. Native administrator-selected fallback routing is unchanged.

Realtime and Grok use separate SIP/sideband adapters. GPT-Live uses primary
WebSocket audio through private Asterisk `chan_websocket`; see
[live transport](live-websocket.md). No automatic SIP failover exists.
All transfers use a common executor: check current inventory and grants, subscribe
to the owned caller, flush provider media and wait for both native token/destination
acknowledgement and Stasis release. A tool acknowledgement alone is never success.
Compact media/provider channel identities fit native CDR/CEL limits; session/run
UUIDs remain independent. Hangup is opt-in, bounded and rechecks caller ownership.

The opening receives the PBX-resolved caller name from the admitted ARI channel,
bounded to 256 UTF-8 bytes with control characters removed. The model is asked to
use a confidently identified first name and otherwise greet neutrally, including
for anonymous, numeric, company-only or ambiguous names. This display data grants
no permissions and does not change the configured welcome or language policy.
It is call-local; it is not added to publications or monitoring metadata. Actual
pronunciation and name selection still require human listening acceptance.

Attended transfer is not enabled. Its service-authenticated presence/chat and
native handoff integration gates must pass before introducing a new publication
contract or consultation flow. Existing v8 publications and tools-v5 are unchanged.

Configuration and monitoring use separate least-privilege MariaDB accounts. Runs,
format-2 metadata/events and optional encrypted debug transcripts are stored in
`asteriskcdrdb`. Durable bounded outboxes preserve idempotency across retries and
restart. Incomplete traces/usage remain unavailable or partial; they are never
invented as zero. Metadata retention defaults to 90 days, controlled by
`VOICE_AGENT_MONITORING_RETENTION_DAYS` (1–3650). Debug capture defaults off and
has a separate 1–24 hour lifetime. Audio is never persisted by the media bridge.

`python main.py` starts the shared application and binds the voice-agent HTTP
listener to loopback. The standalone `python -m voice_agent.runner` remains for
agent-only diagnostics and the standalone image. Set
`VOICE_AGENT_RUNTIME_TOKEN` for authenticated `/internal/ready`; documentation and
configuration routes are absent. Traefik forwards only the provider webhook prefix;
exact provider routes verify raw bytes, timestamps and pending-call correlation.
`VOICE_AGENT_ENABLED=true` permits call admission. NS8 generates private credentials
and starts voice agents even when transcription is disabled. The shared process
starts transcription and its separate private API only when either existing
transcription flag is true. Each component owns a different ARI application;
transcription connection failures retry without restarting agent calls. Shutdown
closes both listeners, RTP/ARI/MQTT resources and per-request workers. Keep Python
on 3.12.13.

NethVoice currently wraps the published Satellite PR #56 image, pinned by digest
in `build-images.sh`, and checks out its matching source revision in CI.
**Before merging, replace that PR image with a versioned Satellite release and
update the digest and CI revision together.**

## Checks and packaging

```sh
python -m voice_agent.runner --check-snapshot voice_agent/published-v8.example.json
python -m pytest --confcutdir=tests/foundation -o addopts='' tests/foundation -q
python -m pytest -o addopts='' tests -q --ignore=tests/foundation
docker build -f Containerfile.agent -t nethvoice-voice-agent:alpha .
docker build -t nethvoice-satellite:alpha .
```

Install `requirements-agent.lock` and `requirements-dev.txt` in an isolated Python
3.12 environment; install `requirements.txt` and native PortAudio dependencies for
transcription tests. Foundation checks isolate their fixtures from transcription
metrics. The sibling NethVoice `freepbx/tests/agents/run-foundation.sh`
prepares disposable MariaDB, applies the fresh baseline repeatedly, tests grants and
restore, and runs paired contracts. Without this harness DB-specific cases skip.
The standalone image starts only the runner as UID/GID 65532. The normal image
uses `python main.py` for the shared lifecycle. Set
`SATELLITE_CALL_TRANSCRIPTION_ENABLED=true` or
`SATELLITE_VOICEMAIL_TRANSCRIPTION_ENABLED=true` to start transcription.
The NethVoice image pin must be refreshed after these changes are published; its
old PR image is rejected by the shared-lifecycle import check.
Diagnostics remain in the source tree and are excluded from production images.

Use exact image/source provenance, a protected backup and a zero-call gate for
staging. Rollback across the alpha reset needs matching pre-reset databases,
secrets and images. Deployment-specific receipts stay outside published docs.
See [validation](voice-agents-validation.md) for evidence and remaining gaps.
Duplication, admission policy and graph workflows remain future work.

## CleverAI is executed outside this runtime

NethVoice can configure CleverAI trunks and agents. CleverAI uses OpenAI SIP, with
the project's incoming-call webhook configured externally to point to CleverAI.
No CleverAI webhook relay, automatic provisioning, API key or webhook secret is
configured locally. CleverAI owns session acceptance, conversation settings,
tools and monitoring.

These agents use **Save** to request the shared `retrieveHelper.sh` PBX reload;
there is no local Publish step. Native FreePBX destinations and VisualPlan handle
routing and fallback. The native dialplan dials the managed OpenAI trunk directly,
with a 20-second connection timeout, and does not enter `Stasis(voice-agent)`.
Connection failures use the configured fallback; normal remote hangup ends the
call. Transfers back into NethVoice are deferred.

CleverAI configuration is outside the v8 publication contract. No dummy snapshot,
local provider session, run, transcript or usage record is created. CleverAI
trunks are excluded from `voice_agent_runtime_bindings`; local OpenAI/Grok
publications, adapters and monitoring remain unchanged. Native CDR and unrelated
Satellite transcription remain under their existing owners.

The four outbound metadata headers use inherited PBX caller number/name and
inbound `FROM_DID`, with the native `agent:<agent_id>` identity proposed for
`X-OS-Extension` and for a missing DID. **CleverAI's acceptance/registration of that
identity and empty caller fields is not verified:** the experimental dialplan was
not available in the workspace. Real CleverAI calling, audio, webhook setup and
remote identity mapping require external acceptance testing before rollout.
