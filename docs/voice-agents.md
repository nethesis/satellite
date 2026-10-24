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
initially disabled, queues enabled, IVRs disabled, callbacks enabled for seven
days, and call ending disabled. Callback evidence is short-lived and bound to
the inbound caller, live session, agent and publication. A model cannot supply
an authoritative number or turn a direct destination into callback evidence.

Realtime and Grok use separate SIP/sideband adapters. GPT-Live uses primary
WebSocket audio through private Asterisk `chan_websocket`; see
[live transport](live-websocket.md). No automatic SIP failover exists.
All transfers use a common executor: check current inventory and grants, subscribe
to the owned caller, flush provider media and wait for both native token/destination
acknowledgement and Stasis release. A tool acknowledgement alone is never success.
Compact media/provider channel identities fit native CDR/CEL limits; session/run
UUIDs remain independent. Hangup is opt-in, bounded and rechecks caller ownership.

Configuration and monitoring use separate least-privilege MariaDB accounts. Runs,
format-2 metadata/events and optional encrypted debug transcripts are stored in
`asteriskcdrdb`. Durable bounded outboxes preserve idempotency across retries and
restart. Incomplete traces/usage remain unavailable or partial; they are never
invented as zero. Metadata retention defaults to 90 days, controlled by
`VOICE_AGENT_MONITORING_RETENTION_DAYS` (1–3650). Debug capture defaults off and
has a separate 1–24 hour lifetime. Audio is never persisted by the media bridge.

`python -m voice_agent.runner` binds its private HTTP listener to loopback. Set
`VOICE_AGENT_RUNTIME_TOKEN` for authenticated `/internal/ready`; documentation and
configuration routes are absent. Traefik forwards only the provider webhook prefix;
exact provider routes verify raw bytes, timestamps and pending-call correlation.
`VOICE_AGENT_ENABLED=true` permits call admission. NS8 generates private credentials
and starts the runner independently of transcription. Keep Python on 3.12.13.

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
preserves `python main.py` transcription startup and includes the optional package.
Diagnostics remain in the source tree and are excluded from production images.

Use exact image/source provenance, a protected backup and a zero-call gate for
staging. Rollback across the alpha reset needs matching pre-reset databases,
secrets and images. Deployment-specific receipts stay outside published docs.
See [validation](voice-agents-validation.md) for evidence and remaining gaps.
Duplication, admission policy and graph workflows remain future work.
