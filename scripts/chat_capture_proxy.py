"""Transparent HTTP proxy that records every chat completion (request + full response).

    python scripts/chat_capture_proxy.py --listen 127.0.0.1:8008 --upstream http://127.0.0.1:8009 --out captures.jsonl

lm-evaluation-harness keeps only ``message.content`` of a chat completion; the Muse Spark
server also returns ``reasoning_content`` (the ` to=self` channel), ``finish_reason`` and
``usage.completion_tokens``. Put this proxy between the harness and the server to keep all of
it: one JSON line per request ``{"ts", "path", "seconds", "status", "request", "response"}``
(``response`` is the parsed JSON when the upstream returned JSON, else the raw text).
"""

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def make_handler(upstream, out_path, lock):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quiet
            pass

        def _forward(self, method):
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else None
            headers = {}
            for key in ("Content-Type", "Authorization", "Accept"):
                if self.headers.get(key):
                    headers[key] = self.headers[key]
            req = urllib.request.Request(upstream + self.path, data=body, method=method, headers=headers)
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=7200) as resp:
                    status, payload, ctype = resp.status, resp.read(), resp.headers.get("Content-Type", "application/json")
            except urllib.error.HTTPError as error:
                status, payload, ctype = error.code, error.read(), error.headers.get("Content-Type", "application/json")
            except Exception as error:  # upstream down
                status, payload, ctype = 502, json.dumps({"error": f"{type(error).__name__}: {error}"}).encode(), "application/json"
            seconds = time.perf_counter() - started
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            if method == "POST" and out_path:
                record = {"ts": time.time(), "path": self.path, "seconds": seconds, "status": status}
                try:
                    record["request"] = json.loads(body) if body else None
                except ValueError:
                    record["request"] = body.decode("utf-8", "replace") if body else None
                try:
                    record["response"] = json.loads(payload)
                except ValueError:
                    record["response"] = payload.decode("utf-8", "replace")
                with lock, open(out_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

        def do_GET(self):
            self._forward("GET")

        def do_POST(self):
            self._forward("POST")

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--listen", default="127.0.0.1:8008")
    parser.add_argument("--upstream", required=True, help="base URL of the real server")
    parser.add_argument("--out", default=None, help="jsonl of captured POST requests/responses")
    args = parser.parse_args(argv)
    host, _, port = args.listen.rpartition(":")
    server = ThreadingHTTPServer((host or "127.0.0.1", int(port)), make_handler(args.upstream.rstrip("/"), args.out, threading.Lock()))
    print(f"capture proxy on {args.listen} -> {args.upstream} (log: {args.out})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
