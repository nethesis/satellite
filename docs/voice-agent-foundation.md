# Independent voice-agent runner

`voice_agent` is separate from the existing transcription modules and their
PostgreSQL transcript store. PHP owns all configuration in MariaDB. The runner
reads immutable published snapshots and owns provider sideband, ARI call control
and the two allowlisted v0.1 tools. It has no configuration writer or push API.

The PHP-owned v1 schema is mirrored for runtime validation. `PublishedRepository`
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
