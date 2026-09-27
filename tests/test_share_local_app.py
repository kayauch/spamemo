import importlib.util
from pathlib import Path
import time

import httpx
from fastapi.testclient import TestClient

spec = importlib.util.spec_from_file_location("share_local_app", Path(__file__).parents[1] / "scripts/share_local_app.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
TOKEN = "test-only-access-token-" + "x" * 32


def gateway(handler, expires=None):
    app = module.create_app(TOKEN, expires if expires is not None else time.time() + 60,
                            transport=httpx.MockTransport(handler))
    return TestClient(app, base_url="https://share.example")


def test_unauthenticated_requests_never_reach_app():
    calls = []
    with gateway(lambda r: calls.append(r)) as client:
        for url in ["/", "/health", "/scenes", "/static/solid-room.js", "/?access_token=wrong"]:
            assert client.get(url).status_code == 401
        assert client.post("/scenes").status_code == 401
    assert calls == []


def test_link_removes_secret_and_cookie_covers_assets():
    def upstream(request):
        assert request.url.host == "127.0.0.1"
        assert request.url.port == 8001
        assert TOKEN not in str(request.url) + str(request.headers)
        return httpx.Response(200, stream=httpx.ByteStream(b"room"), headers={"content-type": "text/html"})
    with gateway(upstream) as client:
        response = client.get("/?scene=bedroom&access_token=" + TOKEN, follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/?scene=bedroom"
        cookie = response.headers["set-cookie"]
        assert "Secure" in cookie and "HttpOnly" in cookie and "SameSite=lax" in cookie
        assert client.get("/static/solid-room.js").content == b"room"


def test_post_upload_stream_preserved_but_cross_site_rejected():
    bodies = []
    def upstream(request):
        bodies.append(request.read())
        assert request.headers["content-type"] == "application/octet-stream"
        return httpx.Response(201, stream=httpx.ByteStream(b"saved"))
    with gateway(upstream) as client:
        client.get("/?access_token=" + TOKEN, follow_redirects=False)
        assert client.post("/scenes", headers={"origin": "https://evil.example"}).status_code == 403
        assert client.post("/scenes").status_code == 403
        payload = b"sample-video-bytes\x00" * 1000
        result = client.post("/scenes", content=payload,
                             headers={"origin": "https://share.example", "content-type": "application/octet-stream"})
        assert result.status_code == 201
        assert bodies == [payload]


def test_expiration_blocks_even_valid_credentials():
    with gateway(lambda r: None, expires=time.time() - 1) as client:
        assert client.get("/?access_token=" + TOKEN).status_code == 410
        assert client.get("/scenes", headers={"authorization": "Bearer " + TOKEN}).status_code == 410


def test_gateway_does_not_redirect_requests_to_an_external_host():
    seen = []
    def upstream(request):
        seen.append(str(request.url))
        return httpx.Response(206, stream=httpx.ByteStream(b"part"), headers={"content-range": "bytes 0-3/10"})
    with gateway(upstream) as client:
        result = client.get("/https://external.example/asset", headers={"authorization": "Bearer " + TOKEN})
        assert result.status_code == 206
        assert result.headers["content-range"] == "bytes 0-3/10"
        assert result.headers["cache-control"] == "private, no-store"
    assert seen == ["http://127.0.0.1:8001/https://external.example/asset"]


def test_code_login_recovers_bare_link_and_preserves_target():
    with gateway(lambda r: httpx.Response(200, stream=httpx.ByteStream(b"rooms"))) as client:
        entry = client.get('/rooms')
        assert entry.status_code == 401 and '共有コード' in entry.text
        assert entry.headers['referrer-policy'] == 'same-origin'
        assert TOKEN not in entry.text
        result = client.post('/__share/login', data={'code': TOKEN, 'next': '/rooms'},
                             headers={'origin': 'https://share.example'}, follow_redirects=False)
        assert result.status_code == 303 and result.headers['location'] == '/rooms'
        assert client.get('/rooms').content == b'rooms'


def test_code_login_rejects_wrong_code_cross_site_and_external_redirect():
    calls = []
    with gateway(lambda r: calls.append(r)) as client:
        bad = client.post('/__share/login', data={'code': 'wrong'}, headers={'origin': 'https://share.example'})
        assert bad.status_code == 401 and '共有コードが違います' in bad.text
        assert not client.cookies.get(module.COOKIE)
        assert client.post('/__share/login', data={'code': TOKEN}, headers={'origin': 'https://evil.example'}).status_code == 403
        good = client.post('/__share/login', data={'code': TOKEN, 'next': '//evil.example'},
                           headers={'origin': 'https://share.example'}, follow_redirects=False)
        assert good.status_code == 303 and good.headers['location'] == '/'
    assert calls == []


def test_explicit_public_mode_without_cookie_preserves_write_origin_check():
    calls = []
    def upstream(request):
        calls.append(request.method)
        return httpx.Response(200, stream=httpx.ByteStream(b'ok'))
    app = module.create_app(TOKEN, time.time()+60, transport=httpx.MockTransport(upstream), public=True)
    with TestClient(app, base_url='https://share.example') as client:
        assert client.get('/scenes').status_code == 200
        assert not client.cookies
        assert client.post('/entities', headers={'origin':'https://share.example'}).status_code == 200
        assert client.post('/entities', headers={'origin':'https://evil.example'}).status_code == 403
        assert client.post('/entities').status_code == 403
        assert client.get('/__share/login', follow_redirects=False).headers['location'] == '/'
    assert calls == ['GET', 'POST']
