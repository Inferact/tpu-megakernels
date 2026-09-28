"""OpenAI-compatible `/v1/chat/completions`, `/v1/completions`, `/v1/models`, and `/health` HTTP server.

Designed to run on coordinator rank 0 (`jax.process_index() == 0`) while worker processes
(`1 .. jax.process_count() - 1`) participate in lockstep SPMD execution via `engine.worker_serve_loop()`.
Compatible with `lm_eval --model local-chat-completions` and standard OpenAI clients.
"""

from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from typing import Any, Iterator
import uuid

import jax

from deepseek_v41.engine import DSV41Engine


class OpenAIServerState:
    """Thread-safe state shared across HTTP handler requests on process 0."""

    def __init__(
        self,
        engine: DSV41Engine,
        *,
        model_name: str = "deepseek-v4.1-flash",
        default_use_dspark: bool = True,
        default_backend: str = "megakernel",
    ) -> None:
        self.engine = engine
        self.model_name = model_name
        self.default_use_dspark = bool(default_use_dspark)
        self.default_backend = default_backend
        self.lock = threading.Lock()
        self.total_requests = 0
        self.total_generated_tokens = 0
        self.total_decode_s = 0.0
        self.cumulative_histogram = [0 for _ in range(engine.cfg.dspark_block_size + 1)]

    def handle_completion(self, payload: dict[str, Any], *, is_chat: bool) -> dict[str, Any]:
        if is_chat:
            messages = payload.get("messages", [])
            prompt_text = self.engine.tokenizer.apply_chat_template(messages)
            prompt_ids = self.engine.tokenizer.encode(prompt_text, add_bos=True)
        else:
            prompt_raw = payload.get("prompt", "")
            if isinstance(prompt_raw, list):
                if prompt_raw and isinstance(prompt_raw[0], int):
                    prompt_ids = [int(t) for t in prompt_raw]
                else:
                    prompt_ids = self.engine.tokenizer.encode(str(prompt_raw[0]), add_bos=True)
            else:
                prompt_ids = self.engine.tokenizer.encode(str(prompt_raw), add_bos=True)

        max_tokens = int(
            payload.get("max_completion_tokens")
            or payload.get("max_tokens")
            or payload.get("max_new_tokens")
            or 256
        )
        # Clamp max_tokens to remaining context window
        avail = max(1, self.engine.max_seq_len - len(prompt_ids) - 8)
        max_tokens = min(max_tokens, avail)

        use_dspark = bool(payload.get("use_dspark", self.default_use_dspark))
        backend = str(payload.get("backend", self.default_backend))
        stop_seqs = payload.get("stop") or []
        if isinstance(stop_seqs, str):
            stop_seqs = [stop_seqs]

        with self.lock:
            res = self.engine.spmd_generate_greedy(
                prompt_ids,
                max_new_tokens=max_tokens,
                use_dspark=use_dspark,
                backend=backend,
                eos_token_id=self.engine.tokenizer.eos_token_id,
            )
            self.total_requests += 1
            self.total_generated_tokens += len(res["generated_ids"])
            self.total_decode_s += float(res["decode_s"])
            hist = res["dspark_stats"]["histogram"]
            for i in range(min(len(self.cumulative_histogram), len(hist))):
                self.cumulative_histogram[i] += int(hist[i])

        text = self.engine.tokenizer.decode(res["generated_ids"])
        finish_reason = "length"
        if res["generated_ids"] and res["generated_ids"][-1] == self.engine.tokenizer.eos_token_id:
            finish_reason = "stop"
        for s in stop_seqs:
            if s and s in text:
                text = text.split(s, 1)[0]
                finish_reason = "stop"

        created = int(time.time())
        req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        usage = {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(res["generated_ids"]),
            "total_tokens": len(prompt_ids) + len(res["generated_ids"]),
        }
        metrics = {
            "prefill_s": res["prefill_s"],
            "decode_s": res["decode_s"],
            "tok_per_s": res["tok_per_s"],
            "median_step_ms": res["median_step_ms"],
            "p99_step_ms": res["p99_step_ms"],
            "use_dspark": use_dspark,
            "backend": backend,
            "dspark_stats": res["dspark_stats"],
            "device_kind": jax.devices()[0].device_kind,
            "world_size": jax.device_count(),
        }

        if is_chat:
            return {
                "id": req_id,
                "object": "chat.completion",
                "created": created,
                "model": self.model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": usage,
                "x_dsv41_metrics": metrics,
            }
        return {
            "id": req_id,
            "object": "text_completion",
            "created": created,
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "text": text,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
            "x_dsv41_metrics": metrics,
        }


def _make_handler(state: OpenAIServerState):
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, code: int, obj: dict[str, Any]) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.startswith("/health"):
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "model": state.model_name,
                        "device_kind": jax.devices()[0].device_kind,
                        "world_size": jax.device_count(),
                        "process_count": jax.process_count(),
                        "default_use_dspark": state.default_use_dspark,
                        "default_backend": state.default_backend,
                        "total_requests": state.total_requests,
                        "total_generated_tokens": state.total_generated_tokens,
                        "cumulative_dspark_histogram": state.cumulative_histogram,
                    },
                )
                return
            if self.path.startswith("/v1/models"):
                self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": state.model_name,
                                "object": "model",
                                "created": 1750000000,
                                "owned_by": "deepseek",
                            }
                        ],
                    },
                )
                return
            self._send_json(404, {"error": f"Unknown path {self.path}"})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception as exc:
                self._send_json(400, {"error": f"Invalid JSON: {exc}"})
                return

            try:
                if self.path.startswith("/v1/chat/completions"):
                    resp = state.handle_completion(payload, is_chat=True)
                    self._send_json(200, resp)
                    return
                if self.path.startswith("/v1/completions"):
                    resp = state.handle_completion(payload, is_chat=False)
                    self._send_json(200, resp)
                    return
                self._send_json(404, {"error": f"Unknown POST endpoint {self.path}"})
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})

    return _Handler


@contextmanager
def run_server_in_thread(
    engine: DSV41Engine,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    model_name: str = "deepseek-v4.1-flash",
    default_use_dspark: bool = True,
    default_backend: str = "megakernel",
) -> Iterator[tuple[str, OpenAIServerState]]:
    """Start the OpenAI HTTP server in a background daemon thread on process 0."""
    state = OpenAIServerState(
        engine,
        model_name=model_name,
        default_use_dspark=default_use_dspark,
        default_backend=default_backend,
    )
    httpd = ThreadingHTTPServer((host, port), _make_handler(state))
    actual_port = httpd.server_address[1]
    base_url = f"http://{host}:{actual_port}"
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield base_url, state
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5.0)
