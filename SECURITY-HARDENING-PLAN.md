# HTTP API hardening

Private HTTP binds to `HTTP_HOST`, default `127.0.0.1`. NS8 sets the loopback
address explicitly and uses host networking for its local consumers.

All legacy `/api/*` routes require `API_TOKEN`. A missing server token returns
503; missing or invalid request credentials return 401. Private agent routes also
fail closed. Non-ASCII credentials are compared as bytes and rejected normally.

NS8 exposes only `/agents-api/v1` through Traefik. This surface uses expiring,
revocable machine-client bearer tokens and operation/customer grants. Private
configuration, provider events, monitoring and administration routes stay under
`/api/agent/v1`. Do not proxy these private routes. Tokens are not query parameters.

The PBX data endpoint accepts only direct local requests and the separate
`SATELLITE_PBX_DATA_TOKEN`. Forwarded requests are rejected. The Satellite
container receives only explicit credentials and runs as UID/GID 1001.
Transcript reads and deletion require the gateway's verified actor and record
an audit event. Test requests that approve workflow writes require explicit
administrator write confirmation.

Before release, run the database, PHP gateway, HTTP and provider acceptance
suites with synthetic credentials on an isolated network. Verify loopback binding,
mandatory credentials, actor auditing, graceful SIGTERM and volume ownership.
Publish the tested image and set its immutable digest for the NethVoice build.
