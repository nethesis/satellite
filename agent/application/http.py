"""HTTPS JSON transport with connection-time network policy and bounded bodies."""

import asyncio
import ipaddress
import json
import socket
from urllib.parse import quote, urlsplit

import aiohttp

from .contracts import ApplicationError, canonical, validate

PRIVATE = tuple(ipaddress.ip_network(v) for v in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7"))
MAX_RESPONSE = 256 * 1024


def permitted_address(value, networks):
    ip = ipaddress.ip_address(value.split("%")[0])
    if "%" in value or (ip.version == 6 and ip.ipv4_mapped):
        return False
    if ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_unspecified or ip.is_reserved:
        return False
    if ip.is_global:
        return True
    return any(ip in n for n in PRIVATE if n.version == ip.version) and any(
        ip in ipaddress.ip_network(n) for n in networks if ipaddress.ip_network(n).version == ip.version)


class PolicyResolver(aiohttp.abc.AbstractResolver):
    def __init__(self, hostname, networks):
        self.hostname = hostname
        self.networks = networks

    async def resolve(self, host, port=0, family=socket.AF_UNSPEC):
        if host != self.hostname:
            raise ApplicationError("network_denied", 403)
        answers = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM, family=family)
        result = []
        for af, _, proto, _, address in answers:
            ip = address[0]
            if not permitted_address(ip, self.networks):
                raise ApplicationError("network_denied", 403)
            result.append({"hostname": host, "host": ip, "port": port, "family": af,
                           "proto": proto, "flags": socket.AI_NUMERICHOST})
        if not result:
            raise ApplicationError("network_unavailable", 503)
        return result

    async def close(self):
        pass


class RemoteError(ApplicationError):
    def __init__(self, code, http_status=None):
        super().__init__(code, 503)
        self.http_status = http_status


def bounded_json(raw, limit=MAX_RESPONSE):
    if len(raw) > limit:
        raise RemoteError("response_too_large")
    # Bound nesting before decoding, including escaped quotes in string values.
    depth = 0; quoted = False; escaped = False
    for byte in raw:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            quoted = True
        elif byte in (91, 123):
            depth += 1
            if depth > 24:
                raise RemoteError("invalid_remote_json")
        elif byte in (93, 125):
            depth -= 1
    try:
        return json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, UnicodeDecodeError, RecursionError):
        raise RemoteError("invalid_remote_json") from None


class HttpTransport:
    async def json_request(self, origin, path, method, headers, query, body, networks, timeout, max_bytes=MAX_RESPONSE):
        url = urlsplit(origin)
        if url.scheme != "https" or path.startswith("//") or not path.startswith("/"):
            raise ApplicationError("network_denied", 403)
        # aiohttp bypasses its resolver for IP literals, so validate these here.
        try:
            address = ipaddress.ip_address(url.hostname)
        except ValueError:
            address = None
        if address is not None and not permitted_address(str(address), networks):
            raise ApplicationError("network_denied", 403)
        connector = aiohttp.TCPConnector(resolver=PolicyResolver(url.hostname, networks),
            use_dns_cache=False, force_close=True, limit=1, ssl=True)
        timeout_config = aiohttp.ClientTimeout(total=timeout, connect=min(3, timeout), sock_read=min(10, timeout))
        try:
            async with aiohttp.ClientSession(connector=connector, trust_env=False, auto_decompress=False, timeout=timeout_config) as session:
                async with session.request(method, origin + path, headers={"Accept": "application/json", "Accept-Encoding": "identity", **headers},
                    params=query, data=canonical(body).encode() if body is not None else None, allow_redirects=False) as response:
                    if not 200 <= response.status < 300:
                        raise RemoteError("remote_http_error", response.status)
                    media = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
                    if media != "application/json" and not media.endswith("+json"):
                        raise RemoteError("invalid_remote_media", response.status)
                    if response.headers.get("Content-Encoding", "identity").lower() not in ("identity", ""):
                        raise RemoteError("compressed_response_denied", response.status)
                    if response.content_length is not None and response.content_length > max_bytes:
                        raise RemoteError("response_too_large", response.status)
                    chunks = bytearray()
                    async for chunk in response.content.iter_chunked(8192):
                        chunks.extend(chunk)
                        if len(chunks) > max_bytes:
                            raise RemoteError("response_too_large", response.status)
                    return bounded_json(chunks, max_bytes)
        except (ApplicationError, asyncio.CancelledError):
            raise
        except (aiohttp.ClientError, TimeoutError, OSError):
            raise RemoteError("remote_unavailable") from None


def mapped_request(operation, args):
    path = operation["path"]
    for key in operation["input_schema"].get("properties", {}):
        marker = "{" + key + "}"
        if marker in path:
            if key not in args or not isinstance(args[key], (str, int)) or isinstance(args[key], bool):
                raise ApplicationError("invalid_input")
            value = str(args[key])
            if value in (".", ".."):
                raise ApplicationError("invalid_input")
            path = path.replace(marker, quote(value, safe=""))
    query = {target: args[source] for target, source in operation["query"].items() if source in args}
    if any(not isinstance(v, (str, int, float, bool)) or isinstance(v, (list, dict)) for v in query.values()):
        raise ApplicationError("invalid_input")
    query = {key: ("true" if value else "false") if isinstance(value, bool) else value for key, value in query.items()}
    body = {target: args[source] for target, source in operation["body"].items() if source in args} if operation["method"] in ("POST", "PUT", "PATCH") else None
    return path, query, body


def project(response, operation):
    result = {}
    def field(item, path):
        if path == "$":
            return item
        if path == "$count" and isinstance(item, list):
            return len(item)
        for key in path.split("."):
            if isinstance(item, dict) and key in item:
                item = item[key]
            elif isinstance(item, list) and key.isdigit() and int(key) < len(item):
                item = item[int(key)]
            else:
                raise ApplicationError("invalid_remote_result", 503)
        return item
    for target, path in operation["projection"].items():
        item = field(response, path)
        if target in operation.get("array_projection", {}):
            if not isinstance(item, list) or len(item) > 100:
                raise ApplicationError("invalid_remote_result", 503)
            item = [{k: field(row, v) for k, v in operation["array_projection"][target].items()} for row in item]
        if operation.get("coercions", {}).get(target) == "string":
            if type(item) not in (str, int):
                raise ApplicationError("invalid_remote_result", 503)
            item = str(item)
        result[target] = item
    if len(canonical(result).encode()) > 16384:
        raise ApplicationError("projected_result_too_large", 503)
    try:
        validate(result, operation["output_schema"])
    except ApplicationError:
        raise ApplicationError("invalid_remote_result", 503) from None
    return result
