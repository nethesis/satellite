# Independent voice-agent runner

`voice_agent` is separate from the existing transcription modules and their
PostgreSQL transcript store. PHP owns all configuration in MariaDB. The runner
reads immutable published snapshots and owns provider sideband, ARI call control
and the two allowlisted v0.1 tools. It has no configuration writer or push API.

The PHP-owned v1, v2 and v3 schemas are mirrored for runtime validation. `PublishedRepository`
loads only the enabled published MariaDB view through an injected connection,
verifies row/JSON identity and returns a detached snapshot pinned at admission. See the sibling NethVoice repository's
`docs/voice-agents-contract-v1.md` for IDs, bounds, ownership and grants.

The runner can validate a synthetic fixture offline:

```sh
python -m voice_agent.runner --check-snapshot /path/to/published-v1.example.json
```

For private readiness, set a separately provisioned `VOICE_AGENT_RUNTIME_TOKEN`
and run `python -m voice_agent.runner`. It binds to `VOICE_AGENT_HTTP_HOST`
(default `127.0.0.1`) and `VOICE_AGENT_HTTP_PORT` (default 8001). M0 permits only
loopback addresses. `/internal/ready` requires a Bearer token; docs, OpenAPI and
configuration routes are absent. Without `VOICE_AGENT_ENABLED=true`, calls remain
disabled. NS8 provisions the enabled runner through `voice-agent.service`, a private
credential file and separate read/write DB identities. Traefik forwards only the
public `/satellite-webhooks` prefix; FastAPI accepts exactly `/openai` and `/grok`
under it with raw-body signature and pending-call correlation checks. See
`ns8-nethvoice/docs/voice-agents-v01.md` for configuration, lifecycle, retention and
remaining real-call gates. No actual provider/telephony success follows from mocks.

`Containerfile.agent` builds a standalone image with runtime dependencies pinned
in `requirements-agent.lock`, running as UID/GID 65532. The normal `Containerfile`
also includes the optional package and retains `python main.py` startup, running
as the unprivileged `satellite` user (UID/GID 65532). Installed packages are in
`/usr/local`, so startup does not require access to root’s home. Its
Python base is pinned to the tested 3.12.13 version. `main.py` now honors the NS8
unit's `HTTP_HOST`; its default API bind is loopback. Transcription startup,
MQTT subscription and shutdown are regression-tested with synthetic dependencies.

## Checks

Install `requirements-agent.txt` and `requirements-dev.txt` in an isolated Python
3.12 environment. The NethVoice `freepbx/tests/agents/run-foundation.sh` starts and
removes a disposable MariaDB container, applies PHP migrations twice, checks grants
and runs cross-repository tests. Set `M0_PYTHON` to that Python interpreter.
For the complete transcription suite, also install `requirements.txt` and its
native PortAudio dependencies. No provider key or real telephone call is needed.

```sh
PYTHONPATH=. python -m pytest -o addopts='' tests/foundation -q
PYTHONPATH=. python -m pytest -o addopts='' tests -q --ignore=tests/foundation
docker build -f Containerfile.agent -t nethvoice-voice-agent:v01-configuration .
docker build -t nethvoice-satellite:v01-configuration .
```

The MariaDB integration case requires the disposable test database prepared by
the PHP harness; otherwise it reports a skip. Credentials must never be supplied
through CLI arguments or printed in diagnostic output on a real installation.

## Provider credentials and fallback version two

An encrypted binding API key takes precedence. If it is absent, admission uses
`OPENAI_API_KEY`, or `XAI_API_KEY` followed by `GROK_API_KEY`. Missing or malformed
keys fail admission safely. NS8 copies only the required secrets into the private
runner env file and refreshes it before startup.

Version two adds Hangup and validated native FreePBX fallback destinations while
retaining version-one compatibility. PHP emits a native fallback wrapper keyed by
agent ID and fallback target, retaining wrappers for historical publications.
The runner computes only that trusted wrapper name from the pinned snapshot; it
does not accept model-supplied dialplan strings. Provider SIP hosts are fixed in
admission, and the native dial string preserves the existing TLS proxy path.

Default response language can change on a caller's explicit request or substantive
utterance in another language. Provider connection, media bridge and completed
responses produce bounded, deduplicated metadata events without transcripts or
audio. Hangup before media connection is recorded as interrupted. Failures expose
safe stage/status codes; provider responses, credentials and call content are not
logged. The signed webhook permits repeated unrelated SIP headers such as Via but
continues to reject duplicate correlation headers.

## Version three tools and native transfer confirmation

Version three pins immutable tool manifest version two and each agent's typed
`tool_bindings`. The trusted registry validates tool configuration at admission,
advertises only enabled operations and rejects disabled or wrong-version wire
names. Directory search intersects current extension inventory with the enabled
transfer tool's allowlist. Search result limits are strictly integer-valued. Older
v1/v2 publications retain their original manifests and semantics.

Before transferring, the runner subscribes to its own caller channel, assigns a
fresh transfer token and removes the caller from provider media. A transfer is
committed only after both Stasis release and the native wrapper's
`VoiceAgentTransfer` event match that channel, token and approved destination.
Either event order is supported. An acknowledgement of the ARI continuation alone
does not establish success. Uncertain release and cancellation take safe cleanup
paths; provider-leg deletion cannot prevent independent media-bridge cleanup.

Four authorized synthetic OpenAI calls on nethvoice57 verified conversation,
backend denial, approved native extension transfer and provider-disconnection
fallback. One initial transfer failed and exposed the confirmation bug corrected
above; the subsequent transfer passed. Grok has local adapter/emulator coverage
but no live credential-backed result. See the sibling repository's
`docs/voice-agents-receptionist-verification.md` for source/image provenance,
cleanup, rollback and outstanding release gates.
