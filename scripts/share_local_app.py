"""Temporary authenticated gateway to the already-running local app.

This process never imports the reconstruction app or opens its database.
Only expose this gateway (default 8002), not the unprotected local port 8001.
The state file contains a secret: keep it outside Git (*.local is ignored).
"""
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import hmac
from html import escape
import json
import os
from pathlib import Path
import secrets
import time
from urllib.parse import urlencode, parse_qs

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, StreamingResponse
from starlette.background import BackgroundTask
import uvicorn

COOKIE = "spatial_share_access"
HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
               "te", "trailer", "transfer-encoding", "upgrade"}
SAFE_HEADERS = {"cache-control": "private, no-store", "referrer-policy": "same-origin",
                "x-robots-tag": "noindex, nofollow, noarchive",
                "x-content-type-options": "nosniff"}


def create_app(token: str, expires_at: float, upstream_port: int = 8001, transport=None, public=False):
    if len(token) < 32:
        raise ValueError("Use a randomly generated access token of at least 32 characters")
    upstream = httpx.URL(f"http://127.0.0.1:{upstream_port}")

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(transport=transport, trust_env=False,
                                    timeout=httpx.Timeout(3600, connect=10),
                                    follow_redirects=False) as client:
            app.state.client = client
            yield

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    def valid(value):
        return bool(value) and hmac.compare_digest(value.encode("utf-8"), token.encode("utf-8"))

    def denied(message, code):
        return PlainTextResponse(message, status_code=code, headers=SAFE_HEADERS)

    def destination(value):
        if not value.startswith('/') or value.startswith('//') or '\\' in value or any(ord(c) < 32 for c in value):
            return '/'
        return value

    def login_page(target='/', error=''):
        return HTMLResponse('''<!doctype html><html lang="ja"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>スパメモを開く</title>
<style>body{margin:0;background:#f4f7fb;font-family:system-ui,sans-serif;color:#233044;display:grid;min-height:100vh;place-items:center}main{box-sizing:border-box;background:white;border-radius:20px;padding:32px;width:min(92vw,460px);box-shadow:0 8px 35px #24334a12}h1{font-size:25px}p{line-height:1.8}input,button{box-sizing:border-box;width:100%;padding:14px;border-radius:10px;font-size:16px;margin-top:10px}input{border:1px solid #b8c3d0}button{border:0;background:#2474df;color:white;font-weight:bold;cursor:pointer}.error{color:#b42318}.note{font-size:13px;color:#576579}</style>
<main><h1>スパメモを開く</h1><p>この部屋は共有メンバー向けです。<br>送られてきた共有コードを入力してください。</p>'''
            + ('<p role="alert" class="error">' + escape(error) + '</p>' if error else '')
            + '<form method="post" action="/__share/login"><label for="code">共有コード</label>'
            + '<input id="code" name="code" type="password" autocomplete="off" required maxlength="256" placeholder="共有コードを貼り付け">'
            + '<input type="hidden" name="next" value="' + escape(destination(target), quote=True) + '">'
            + '<button type="submit">スパメモを開く</button></form>'
            + '<p class="note">コードが分からない場合は、リンクを送った人に確認してください。鍵付きの共有リンクからも開けます。</p></main></html>',
            # A no-referrer document makes browser form POSTs send Origin: null.
            # Preserve the origin for our same-site form without exposing it elsewhere.
            status_code=401, headers={**SAFE_HEADERS, 'referrer-policy': 'same-origin'})

    def signed_in(target):
        response = RedirectResponse(destination(target), status_code=303, headers={**SAFE_HEADERS, 'referrer-policy': 'no-referrer'})
        response.set_cookie(COOKIE, token, httponly=True, secure=True, samesite="lax",
                            max_age=max(1, int(expires_at - time.time())))
        return response

    @app.api_route("/{path:path}", methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    async def proxy(request: Request, path: str):
        if time.time() >= expires_at:
            return denied("共有の有効期限が切れました。リンクを送った人に確認してください。", 410)
        if request.url.path == '/__share/login':
            if public:
                return RedirectResponse('/', status_code=303, headers=SAFE_HEADERS)
            if request.method == 'GET':
                return login_page()
            if request.method != 'POST':
                return denied('Method not allowed', 405)
            if request.headers.get('origin') != 'https://' + request.headers.get('host', ''):
                return denied('Cross-site login is not allowed.', 403)
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > 4096:
                    return denied('Request too large', 413)
            form = parse_qs(body.decode('utf-8', errors='replace'))
            target = form.get('next', ['/'])[0]
            if not valid(form.get('code', [''])[0].strip()):
                return login_page(target, '共有コードが違います。前後の空白やコピー漏れをご確認ください。')
            return signed_in(target)
        key = request.query_params.get("access_token")
        if key is not None and not public:
            if request.method not in {"GET", "HEAD"} or not valid(key):
                return login_page('/', '共有リンクの鍵が正しくありません。共有コードを入力してください。')
            # Remove the secret from the address bar before fetching any assets.
            query = urlencode([(k, v) for k, v in request.query_params.multi_items() if k != "access_token"])
            target = "/" + request.url.path.lstrip("/") + ("?" + query if query else "")
            return signed_in(target)
        auth = request.headers.get("authorization", "")
        bearer = auth[7:] if auth.lower().startswith("bearer ") else ""
        if not public and not (valid(bearer) or valid(request.cookies.get(COOKIE))):
            if request.method == 'GET' and ('text/html' in request.headers.get('accept', '') or request.url.path in {'/', '/rooms', '/room/new'}):
                target = request.url.path + ('?' + request.url.query if request.url.query else '')
                return login_page(target)
            return denied("共有コードでログインしてください。", 401)
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            expected_origin = "https://" + request.headers.get("host", "")
            if (origin and origin != expected_origin) or request.headers.get("sec-fetch-site") == "cross-site":
                return denied("Cross-site changes are not allowed.", 403)
            if not origin and not valid(bearer):
                return denied("An origin header is required for browser changes.", 403)

        connection_headers = set(request.headers.get("connection", "").lower().replace(" ", "").split(","))
        excluded = HOP_HEADERS | connection_headers | {"host", "cookie", "authorization", "forwarded"}
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in excluded and not k.lower().startswith("x-forwarded-")}
        headers["host"] = request.headers.get("host", "127.0.0.1")
        headers["x-forwarded-proto"] = "https"
        raw_path = request.scope.get("raw_path", request.url.path.encode("utf-8"))
        if request.scope["query_string"]:
            raw_path += b"?" + request.scope["query_string"]
        url = upstream.copy_with(raw_path=raw_path)
        client = request.app.state.client
        try:
            outgoing = client.build_request(request.method, url, headers=headers, content=request.stream())
            result = await client.send(outgoing, stream=True)
        except httpx.HTTPError:
            return denied("The local Spatial Memory app is not responding.", 502)
        response_excluded = HOP_HEADERS | {"set-cookie", "access-control-allow-origin", "access-control-allow-credentials"}
        response_excluded |= set(result.headers.get("connection", "").lower().replace(" ", "").split(","))
        response_headers = {k: v for k, v in result.headers.items() if k.lower() not in response_excluded}
        response_headers.update(SAFE_HEADERS)
        return StreamingResponse(result.aiter_raw(), status_code=result.status_code,
                                 headers=response_headers, background=BackgroundTask(result.aclose))

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--upstream-port", type=int, default=8001)
    parser.add_argument("--hours", type=float, default=48)
    parser.add_argument("--state", type=Path, default=Path("test-assets/local-share.local"))
    parser.add_argument("--resume", action="store_true", help="Keep the existing invitation and expiry when restarting only the gateway")
    parser.add_argument("--public", action="store_true", help="Explicitly allow anonymous access to the full app")
    args = parser.parse_args()
    if not 0 < args.hours <= 168:
        parser.error("hours must be between 0 and 168")
    args.state.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        state = json.loads(args.state.read_text(encoding='utf-8-sig'))
        if state['expires_at'] <= time.time() or state['port'] != args.port or state['upstream_port'] != args.upstream_port:
            parser.error('Cannot resume an expired share or a different port configuration')
        state['gateway_pid'] = os.getpid()
    else:
        state = {"token": secrets.token_urlsafe(32), "expires_at": time.time() + args.hours * 3600,
                 "port": args.port, "upstream_port": args.upstream_port, "gateway_pid": os.getpid()}
    if args.public:
        state['public'] = True
    # Refuse to silently overwrite another live share's credentials.
    with args.state.open("w" if args.resume else "x", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    uvicorn.run(create_app(state["token"], state["expires_at"], args.upstream_port, public=state.get('public', False)),
                host="127.0.0.1", port=args.port, access_log=False, proxy_headers=False)


if __name__ == "__main__":
    main()
