"""M0 standalone runner: validates pinned snapshots; no provider or ARI traffic."""
import argparse
import hmac
import ipaddress
import json
import os
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
import uvicorn

from .contract import validate_snapshot


def create_app(token: str) -> FastAPI:
    if not token:
        raise ValueError('runtime_token_required')
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get('/internal/ready')
    def ready(authorization: str = Header(default='')):
        if not hmac.compare_digest(authorization, 'Bearer ' + token):
            raise HTTPException(status_code=401, detail='unauthorized')
        return {'status': 'ready', 'milestone': 'M0', 'contract_version': 1,
                'voice_calls_enabled': False}

    return app


def settings():
    host = os.getenv('VOICE_AGENT_HTTP_HOST', '127.0.0.1')
    # M0 has no public webhooks. Keep this listener private until routing is added.
    if not ipaddress.ip_address(host).is_loopback:
        raise ValueError('loopback_bind_required')
    port = int(os.getenv('VOICE_AGENT_HTTP_PORT', '8001'))
    if not 1 <= port <= 65535:
        raise ValueError('invalid_port')
    return host, port


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-snapshot', type=Path, help='Validate a synthetic snapshot offline')
    args = parser.parse_args()
    if args.check_snapshot:
        validate_snapshot(json.loads(args.check_snapshot.read_text()))
        print(json.dumps({'valid': True, 'contract_version': 1}))
        return
    host, port = settings()
    app = create_app(os.getenv('VOICE_AGENT_RUNTIME_TOKEN', ''))
    uvicorn.run(app, host=host, port=port, access_log=False)


if __name__ == '__main__':
    main()
