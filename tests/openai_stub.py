from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send(self, code: int, body: dict) -> None:
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        except OSError:
            pass

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "нет такого пути"})

    def do_POST(self):
        stub = self.server.stub
        size = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(size) or b"{}")
        stub.requests.append({"path": self.path, "body": body})
        if self.path != "/v1/chat/completions":
            self._send(404, {"error": "нет такого пути"})
            return
        time.sleep(stub.delay_s)
        if stub.fail:
            self._send(500, {"error": "сбой заглушки"})
            return
        self._send(200, {
            "choices": [{"index": 0, "message": {"role": "assistant", "content": stub.reply}}],
            "timings": {"predicted_per_second": stub.tps},
        })


class Stub:
    def __init__(self, delay_s: float = 0.0, fail: bool = False, reply: str = "ответ заглушки", tps: float = 42.5):
        self.delay_s = delay_s
        self.fail = fail
        self.reply = reply
        self.tps = tps
        self.requests: list[dict] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.daemon_threads = True
        self._server.stub = self
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)

    def __enter__(self) -> Stub:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
