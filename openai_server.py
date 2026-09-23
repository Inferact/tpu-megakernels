"""A minimal OpenAI-compatible HTTP front end for the standalone K3 demo.

The demo's generation loop runs on all four hosts in lockstep; only JAX
process 0 talks HTTP. :class:`OpenAIServer` accepts requests on a background
thread, turns each prompt into a :class:`Job` (token ids plus generation
parameters), and hands the jobs one at a time to the generation loop through
:meth:`next_job`. The loop pushes generated tokens into the job as they are
produced (``Job.emit``) and finally fills in a :class:`Result`; the handler
that is waiting on the job writes the OpenAI-shaped response, streamed as
server-sent events when the request asked for ``stream``.

Endpoints: ``GET /v1/models``, ``GET /health``, ``POST /tokenize`` and
``POST /detokenize`` (vLLM style), ``POST /v1/completions`` (``prompt`` as
text, a list of texts, token ids, or a list of token-id lists;
``max_tokens``, ``stop``, ``ignore_eos``, ``echo``, ``logprobs``, ``stream``,
``stream_options.include_usage``), and ``POST /v1/chat/completions``
(``messages`` with system, user and assistant turns, rendered with the K3
message template of ``render_chat_tokens``; ``stream``). Decoding is greedy
unless the request sets ``temperature > 0`` (the server default is greedy,
``--temperature`` changes it): then the generation loop samples from the
top-p nucleus (``top_p``, default 1.0) of the temperature-scaled target
distribution, exactly, with speculative rejection sampling against the
greedy drafts; ``seed`` makes a request reproducible (a random seed is drawn
otherwise). ``n`` must be 1; ``top_k`` and the penalties are ignored.
With ``echo`` and ``logprobs`` the (non-streamed) completion carries the
log-probability of every prompt token (``token_logprobs``, first entry
``null``) and, as ``top_logprobs``, the greedy token at each position, which
is what the evaluation harnesses use for loglikelihood tasks. Generated
tokens get ``null`` log-probabilities.
"""

from __future__ import annotations

import codecs
import dataclasses
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import queue
import threading
import time
import uuid

import numpy as np

CHANNEL_SWITCH = "<|close|>think<|sep|><|open|>response<|sep|>"
RESPONSE_TAIL = "<|close|>response<|sep|><|close|>message<|sep|>"
CONTROL = ("<|open|>", "<|close|>", "<|sep|>")
END_TOKENS = ("[EOS]", "<|end_of_msg|>")
_DONE = object()


@dataclasses.dataclass
class Job:
    tokens: np.ndarray  # int32 prompt token ids
    max_tokens: int
    stop: list[str]
    score: bool  # return prompt-token log-probabilities (echo + logprobs)
    kind: str  # "completion" or "chat"
    chat_mode: str  # "response" or "think" (chat only)
    ignore_eos: bool = False
    stream: bool = False
    echo: bool = False
    logprobs: int | None = None
    temperature: float = 0.0  # 0: greedy
    top_p: float = 1.0
    seed: int | None = None  # None: the generation loop draws one
    result: "Result | None" = None
    error: str | None = None
    done: threading.Event = dataclasses.field(default_factory=threading.Event)
    chunks: queue.Queue = dataclasses.field(default_factory=queue.Queue)

    def emit(self, tokens) -> None:
        """Generation loop: new token ids (the streaming handler forwards them)."""
        if self.stream and tokens:
            self.chunks.put([int(t) for t in tokens])

    def complete(self, result: "Result") -> None:
        self.result = result
        self.done.set()
        self.chunks.put(_DONE)

    def fail(self, message: str) -> None:
        self.error = message
        self.done.set()
        self.chunks.put(_DONE)


@dataclasses.dataclass
class Params:
    """Generation parameters of one request (``OpenAIServer._params``)."""
    max_tokens: int
    stop: list[str]
    logprobs: int | None
    ignore_eos: bool
    include_usage: bool
    temperature: float  # 0: greedy
    top_p: float
    seed: int | None


def pack_sampling(temperature: float, top_p: float, seed: int) -> tuple[int, int, int]:
    """Sampling parameters as three int32 values for the demo's job broadcast (float bits, float bits, seed)."""
    bits = np.array([temperature, top_p], np.float32).view(np.int32)
    return int(bits[0]), int(bits[1]), int(seed)


def unpack_sampling(temperature_bits: int, top_p_bits: int, seed: int) -> tuple[float, float, int]:
    values = np.array([temperature_bits, top_p_bits], np.int32).view(np.float32)
    return float(values[0]), float(values[1]), int(seed)


@dataclasses.dataclass
class Result:
    generated: list[int]  # generated token ids, end token included when one stopped the generation
    finish_reason: str  # "stop" or "length"
    prompt_logprobs: list[float] | None = None  # per prompt position >= 1 (position 0 has none)
    prompt_greedy: list[int] | None = None  # greedy token at each prompt position >= 1
    prompt_greedy_logprobs: list[float] | None = None  # its log-probability
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    steps: int = 0


def render_chat_tokens(tokenizer, messages, mode):
    """Token ids of a conversation and the assistant generation prompt.

    Mirrors ``demo_kimi_dspark.render_chat_tokens`` (one user turn) for a list
    of OpenAI-style messages: system and user turns become ``message``
    segments with their role; earlier assistant turns are wrapped in a
    ``response`` channel; the generation prompt opens the ``response``
    channel (``mode == "response"``) or the ``think`` channel (``mode ==
    "think"``, with the tokenizer's default ``thinking_effort=max`` system
    message when the conversation has no system message).
    """
    control = lambda text: tokenizer.encode(text, allowed_special="all")
    text = lambda text: tokenizer.encode(text, disallowed_special=())
    open_tag = lambda tag, attrs="": control("<|open|>") + text(tag + attrs) + control("<|sep|>")
    close_tag = lambda tag: control("<|close|>") + text(tag) + control("<|sep|>")
    end = control("<|end_of_msg|>")

    def content_text(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):  # OpenAI content parts; text parts only
            return "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type", "text") == "text")
        return "" if content is None else str(content)

    tokens = []
    if mode == "think" and not any(m.get("role") == "system" for m in messages):
        body = (
            "`thinking_effort` guides on how much to think in your thinking channel (not including the "
            "response channel), supported values include `low`, `medium`, `high`, and `max`.\n"
            "Now the system is invoked with `thinking_effort=max`."
        )
        tokens += open_tag("message", ' role="system" type="thinking-effort"') + text(body) + close_tag("message") + end
    for message in messages:
        role = message.get("role", "user")
        body = content_text(message.get("content"))
        if role == "assistant":
            tokens += open_tag("message", ' role="assistant"') + open_tag("response") + text(body) + close_tag("response")
            tokens += close_tag("message") + end
        elif role in ("system", "user", "developer"):
            role = "system" if role == "developer" else role
            tokens += open_tag("message", f' role="{role}"') + text(body) + close_tag("message") + end
        else:
            raise ValueError(f"unsupported message role {role!r}")
    tokens += open_tag("message", ' role="assistant"') + open_tag("think" if mode == "think" else "response")
    return np.asarray(tokens, np.int32)


def split_channels(text):
    """(reasoning, content) of a decoded assistant turn: the think channel, if
    any, and the response with the closing tags removed."""
    reasoning = None
    if CHANNEL_SWITCH in text:
        reasoning, text = text.split(CHANNEL_SWITCH, 1)
    text = text.replace(RESPONSE_TAIL, "")
    return reasoning, text


def apply_stop_strings(text, stop):
    """Truncate ``text`` at the earliest stop string; returns (text, stopped)."""
    cut = None
    for s in stop:
        index = text.find(s)
        if index >= 0 and (cut is None or index < cut):
            cut = index
    return (text[:cut], True) if cut is not None else (text, False)


class TokenStreamer:
    """Turns generated token ids into text deltas per channel.

    Bytes are decoded incrementally (a multi-byte character split over tokens
    is emitted once complete). In chat mode the K3 control tokens are tracked:
    text between ``<|open|>``/``<|close|>`` and ``<|sep|>`` is a tag name, an
    open tag selects the channel (``think`` or ``response``) and its close tag
    leaves it; only the text of the two channels is emitted, as
    ``("reasoning", text)`` or ``("content", text)``. Raw completions emit
    everything as ``("content", text)``. End tokens are never emitted.
    """

    def __init__(self, tokenizer, *, chat: bool, initial_channel: str):
        self.tokenizer = tokenizer
        self.chat = chat
        self.channel = initial_channel  # "think", "response" or None (outside a channel)
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.control = {name: tokenizer.encode(name, allowed_special="all")[0] for name in CONTROL}
        self.end = {tokenizer.encode(name, allowed_special="all")[0] for name in END_TOKENS}
        self.tag = None  # (kind, name bytes) while inside a tag

    def feed(self, tokens):
        """List of (channel, text) deltas for the new tokens."""
        deltas = []
        for token in tokens:
            token = int(token)
            if token in self.end:
                continue
            if self.chat:
                if token in (self.control["<|open|>"], self.control["<|close|>"]):
                    self.decoder.reset()
                    self.tag = ("open" if token == self.control["<|open|>"] else "close", b"")
                    continue
                if self.tag is not None:
                    if token == self.control["<|sep|>"]:
                        kind, name = self.tag
                        name = name.decode("utf-8", errors="replace").split(" ")[0]
                        if kind == "open" and name in ("think", "response"):
                            self.channel = name
                        elif kind == "close" and name == self.channel:
                            self.channel = None
                        self.tag = None
                    else:
                        self.tag = (self.tag[0], self.tag[1] + self.tokenizer.decode_single_token_bytes(token))
                    continue
                if self.channel is None:
                    continue
            text = self.decoder.decode(self.tokenizer.decode_single_token_bytes(token))
            if text:
                deltas.append(("reasoning" if self.channel == "think" else "content", text))
        return deltas


class OpenAIServer:
    """HTTP front end; ``next_job`` hands requests to the generation loop."""

    def __init__(self, host, port, tokenizer, *, model_name, chat_mode, default_max_tokens, max_prompt_tokens,
                 bos_token_id, max_model_len=None, default_temperature=0.0, default_top_p=1.0,
                 log=print, chat_render=None, streamer_factory=None, channel_splitter=None):
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.chat_mode = chat_mode
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature
        self.default_top_p = default_top_p
        self.max_prompt_tokens = max_prompt_tokens
        self.max_model_len = max_model_len or max_prompt_tokens
        self.bos_token_id = bos_token_id
        # model-family hooks: chat message rendering and the streaming text splitter
        self.chat_render = chat_render or (lambda tok, messages, mode: render_chat_tokens(tok, messages, mode))
        self.streamer_factory = streamer_factory or (
            lambda tok, *, chat, initial_channel: TokenStreamer(tok, chat=chat, initial_channel=initial_channel)
        )
        self.channel_splitter = channel_splitter or split_channels
        self.log = log
        self.jobs: queue.Queue[Job | None] = queue.Queue()
        self.stopping = threading.Event()
        self.served = 0
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):  # quiet; the demo logs one line per job
                pass

            def _send(self, status, payload):
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _error(self, status, message, kind="invalid_request_error"):
                self._send(status, {"error": {"message": message, "type": kind, "param": None, "code": None}})

            def do_GET(self):
                if self.path in ("/v1/models", "/models"):
                    self._send(200, {"object": "list", "data": [
                        {"id": server.model_name, "object": "model", "created": 0, "owned_by": "tpu-megakernel",
                         "max_model_len": server.max_model_len}]})
                elif self.path in ("/health", "/v1/health", "/", "/ping"):
                    self._send(200, {"status": "ok", "served": server.served})
                elif self.path == "/version":
                    self._send(200, {"version": "tpu-megakernel-dspark"})
                else:
                    self._error(404, f"no route for GET {self.path}")

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                except json.JSONDecodeError as error:
                    return self._error(400, f"invalid JSON: {error}")
                try:
                    if self.path in ("/v1/completions", "/completions"):
                        if body.get("stream"):
                            return server.stream_completion(body, self)
                        payload = server.completions(body)
                    elif self.path in ("/v1/chat/completions", "/chat/completions"):
                        if body.get("stream"):
                            return server.stream_chat(body, self)
                        payload = server.chat_completions(body)
                    elif self.path in ("/tokenize", "/v1/tokenize"):
                        payload = server.tokenize(body)
                    elif self.path in ("/detokenize", "/v1/detokenize"):
                        payload = server.detokenize(body)
                    else:
                        return self._error(404, f"no route for POST {self.path}")
                except ValueError as error:
                    return self._error(400, str(error))
                except RuntimeError as error:
                    return self._error(500, str(error), "server_error")
                self._send(200, payload)

        self.http = ThreadingHTTPServer((host, port), Handler)
        self.thread = threading.Thread(target=self.http.serve_forever, name="openai-http", daemon=True)

    # --- lifecycle -------------------------------------------------------------
    def start(self):
        self.thread.start()
        host, port = self.http.server_address[:2]
        self.log(f"OpenAI-compatible server listening on http://{host}:{port}/v1 (model '{self.model_name}')")

    def shutdown(self):
        self.stopping.set()
        self.http.shutdown()
        while True:  # fail whatever is still queued
            try:
                job = self.jobs.get_nowait()
            except queue.Empty:
                break
            if job is not None:
                job.fail("server shutting down")

    def next_job(self, timeout=None) -> Job | None:
        """Block until a request arrives (``None`` on timeout)."""
        try:
            return self.jobs.get(timeout=timeout)
        except queue.Empty:
            return None

    # --- request parsing -------------------------------------------------------
    def _prompt_tokens(self, prompt):
        if isinstance(prompt, str):
            ids = self.tokenizer.encode(prompt, disallowed_special=())
            ids = ([self.bos_token_id] if self.bos_token_id is not None else []) + ids
        elif isinstance(prompt, list) and all(isinstance(t, int) for t in prompt):
            ids = list(prompt)
        else:
            raise ValueError("prompt must be a string or a list of token ids")
        if not ids:
            raise ValueError("empty prompt")
        if len(ids) > self.max_prompt_tokens:
            raise ValueError(f"prompt has {len(ids)} tokens, the server accepts at most {self.max_prompt_tokens}")
        return np.asarray(ids, np.int32)

    def _params(self, body) -> Params:
        max_tokens = body.get("max_tokens", body.get("max_completion_tokens"))
        max_tokens = self.default_max_tokens if max_tokens is None else int(max_tokens)
        if max_tokens < 0:
            raise ValueError("max_tokens must be >= 0")
        stop = body.get("stop") or []
        if isinstance(stop, str):
            stop = [stop]
        if not isinstance(stop, list) or not all(isinstance(s, str) for s in stop):
            raise ValueError("stop must be a string or a list of strings")
        if body.get("n", 1) not in (1, None):
            raise ValueError("n > 1 is not supported (one sequence per request)")
        logprobs = body.get("logprobs")
        if logprobs is True:
            logprobs = 1
        ignore_eos = bool(body.get("ignore_eos", False))
        include_usage = bool((body.get("stream_options") or {}).get("include_usage", False))
        temperature = body.get("temperature")
        temperature = self.default_temperature if temperature is None else float(temperature)
        if not temperature >= 0.0:
            raise ValueError("temperature must be >= 0")
        top_p = body.get("top_p")
        top_p = self.default_top_p if top_p is None else float(top_p)
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        seed = body.get("seed")
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError("seed must be an integer")
            seed = int(seed) & 0x7FFFFFFF
        return Params(max_tokens, [s for s in stop if s], (int(logprobs) if logprobs else None), ignore_eos,
                      include_usage, temperature, top_p, seed)

    @staticmethod
    def _job(tokens, params: Params, **fields) -> Job:
        return Job(tokens=tokens, max_tokens=params.max_tokens, stop=params.stop, ignore_eos=params.ignore_eos,
                   temperature=params.temperature, top_p=params.top_p, seed=params.seed, **fields)

    def _submit(self, job: Job):
        if self.stopping.is_set():
            raise RuntimeError("server shutting down")
        self.jobs.put(job)

    def _run(self, job: Job, timeout=None) -> Result:
        self._submit(job)
        if not job.done.wait(timeout):
            raise RuntimeError("generation timed out")
        if job.error:
            raise RuntimeError(job.error)
        self.served += 1
        return job.result

    # --- tokenizer endpoints -----------------------------------------------------
    def tokenize(self, body):
        if "messages" in body:
            ids = self.chat_render(self.tokenizer, body["messages"], self.chat_mode).tolist()
        else:
            prompt = body.get("prompt", "")
            ids = self.tokenizer.encode(prompt, disallowed_special=())
            if body.get("add_special_tokens", True) and self.bos_token_id is not None:
                ids = [self.bos_token_id] + ids
        return {"count": len(ids), "max_model_len": self.max_model_len, "tokens": ids}

    def detokenize(self, body):
        tokens = body.get("tokens")
        if not isinstance(tokens, list):
            raise ValueError("tokens must be a list of ids")
        return {"prompt": self._decode(tokens)}

    # --- non-streaming endpoints -------------------------------------------------
    def completions(self, body):
        prompts = body.get("prompt")
        if prompts is None:
            raise ValueError("prompt is required")
        batched = isinstance(prompts, list) and prompts and (isinstance(prompts[0], (str, list)))
        prompt_list = prompts if batched else [prompts]
        params = self._params(body)
        echo = bool(body.get("echo", False))
        choices = []
        prompt_tokens_total = completion_tokens_total = 0
        for index, prompt in enumerate(prompt_list):
            tokens = self._prompt_tokens(prompt)
            job = self._job(tokens, params, score=bool(echo and params.logprobs), kind="completion",
                            chat_mode=self.chat_mode, echo=echo, logprobs=params.logprobs)
            result = self._run(job)
            choices.append(self._completion_choice(index, tokens, job, result))
            prompt_tokens_total += len(tokens)
            completion_tokens_total += len(result.generated)
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:24]}", "object": "text_completion", "created": int(time.time()),
            "model": self.model_name, "choices": choices,
            "usage": self._usage(prompt_tokens_total, completion_tokens_total),
        }

    @staticmethod
    def _usage(prompt_tokens, completion_tokens):
        return {"prompt_tokens": int(prompt_tokens), "completion_tokens": int(completion_tokens),
                "total_tokens": int(prompt_tokens + completion_tokens)}

    def _completion_choice(self, index, prompt_tokens, job: Job, result: Result):
        generated = list(result.generated)
        end_stripped = generated[:-1] if result.finish_reason == "stop" and generated else generated
        text = self._decode(end_stripped)
        text, stopped = apply_stop_strings(text, job.stop)
        finish = "stop" if (stopped or result.finish_reason == "stop") else "length"
        prompt_text = self._decode([int(t) for t in prompt_tokens]) if job.echo else ""
        choice = {"index": index, "text": prompt_text + text, "finish_reason": finish, "logprobs": None}
        if job.logprobs:
            token_ids = ([int(t) for t in prompt_tokens] if job.echo else []) + end_stripped
            token_logprobs = []
            top_logprobs = []
            if job.echo:
                # Position 0 has no prediction. For every later prompt position:
                # the log-probability of the prompt token, and as "top" entry the
                # greedy token at that position with its own log-probability (the
                # harnesses compare its identity with the prompt token).
                token_logprobs.append(None)
                top_logprobs.append(None)
                greedy_logprobs = result.prompt_greedy_logprobs or result.prompt_logprobs or []
                for lp, greedy, greedy_lp in zip(result.prompt_logprobs or [], result.prompt_greedy or [], greedy_logprobs):
                    token_logprobs.append(float(lp))
                    top_logprobs.append({self._decode([int(greedy)]): float(greedy_lp)})
            token_logprobs += [None] * len(end_stripped)
            top_logprobs += [None] * len(end_stripped)
            offsets, position = [], 0
            for token in token_ids:
                offsets.append(position)
                position += len(self._decode([token]))
            choice["logprobs"] = {
                "tokens": [self._decode([t]) for t in token_ids],
                "token_logprobs": token_logprobs,
                "top_logprobs": top_logprobs,
                "text_offset": offsets,
            }
        return choice

    def chat_completions(self, body):
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        params = self._params(body)
        tokens = self.chat_render(self.tokenizer, messages, self.chat_mode)
        if len(tokens) > self.max_prompt_tokens:
            raise ValueError(f"conversation has {len(tokens)} tokens, the server accepts at most {self.max_prompt_tokens}")
        job = self._job(tokens, params, score=False, kind="chat", chat_mode=self.chat_mode)
        result = self._run(job)
        generated = list(result.generated)
        end_stripped = generated[:-1] if result.finish_reason == "stop" and generated else generated
        reasoning, content = self.channel_splitter(self._decode(end_stripped))
        content, stopped = apply_stop_strings(content, job.stop)
        message = {"role": "assistant", "content": content}
        if reasoning is not None:
            message["reasoning_content"] = reasoning
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion", "created": int(time.time()),
            "model": self.model_name,
            "choices": [{"index": 0, "message": message, "logprobs": None,
                         "finish_reason": "stop" if (stopped or result.finish_reason == "stop") else "length"}],
            "usage": self._usage(len(tokens), len(generated)),
        }

    # --- streaming endpoints -----------------------------------------------------
    def stream_completion(self, body, handler):
        prompts = body.get("prompt")
        if prompts is None:
            raise ValueError("prompt is required")
        if isinstance(prompts, list) and prompts and isinstance(prompts[0], (str, list)):
            if len(prompts) != 1:
                raise ValueError("streaming serves one prompt per request")
            prompts = prompts[0]
        params = self._params(body)
        tokens = self._prompt_tokens(prompts)
        job = self._job(tokens, params, score=False, kind="completion", chat_mode=self.chat_mode, stream=True)
        head = {"id": f"cmpl-{uuid.uuid4().hex[:24]}", "object": "text_completion", "created": int(time.time()),
                "model": self.model_name}

        def chunk(text, finish=None):
            return {**head, "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish}]}

        self._stream(job, handler, chunk, params.include_usage, chat=False)

    def stream_chat(self, body, handler):
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        params = self._params(body)
        tokens = self.chat_render(self.tokenizer, messages, self.chat_mode)
        if len(tokens) > self.max_prompt_tokens:
            raise ValueError(f"conversation has {len(tokens)} tokens, the server accepts at most {self.max_prompt_tokens}")
        job = self._job(tokens, params, score=False, kind="chat", chat_mode=self.chat_mode, stream=True)
        head = {"id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion.chunk", "created": int(time.time()),
                "model": self.model_name}

        def chunk(text, finish=None, channel="content", first=False):
            delta = {}
            if first:
                delta["role"] = "assistant"
            if text:
                delta["reasoning_content" if channel == "reasoning" else "content"] = text
            return {**head, "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}]}

        self._stream(job, handler, chunk, params.include_usage, chat=True)

    def _stream(self, job, handler, chunk, include_usage, *, chat):
        """Run ``job`` and write its tokens as server-sent events (chunked transfer)."""
        self._submit(job)
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Transfer-Encoding", "chunked")
        handler.end_headers()

        def write(payload):
            data = f"data: {payload if isinstance(payload, str) else json.dumps(payload)}\n\n".encode("utf-8")
            handler.wfile.write(f"{len(data):x}\r\n".encode("ascii") + data + b"\r\n")
            handler.wfile.flush()

        streamer = self.streamer_factory(self.tokenizer, chat=chat, initial_channel=("think" if job.chat_mode == "think" else "response"))
        holdback = max((len(s) for s in job.stop), default=1) - 1  # text that could still start a stop string
        pending = {"content": "", "reasoning": ""}
        stopped = False
        if chat:
            write(chunk("", first=True))

        def flush(final):
            nonlocal stopped
            for channel in ("reasoning", "content"):
                text = pending[channel]
                if channel == "content" and job.stop:
                    text, hit = apply_stop_strings(text, job.stop)
                    if hit:
                        stopped = True
                        pending[channel] = text
                if final or stopped:
                    visible = text
                else:
                    visible = text[: max(len(text) - holdback, 0)]
                if visible:
                    write(chunk(visible, **({"channel": channel} if chat else {})))
                    pending[channel] = text[len(visible):]
                if stopped:
                    break

        try:
            while True:
                item = job.chunks.get()
                if item is _DONE:
                    break
                if stopped:
                    continue  # the loop still runs to its own end; drop the rest
                for channel, text in streamer.feed(item):
                    pending[channel] += text
                flush(final=False)
            if job.error:
                write({"error": {"message": job.error, "type": "server_error"}})
            else:
                flush(final=True)
                result = job.result
                finish = "stop" if (stopped or result.finish_reason == "stop") else "length"
                write(chunk("", finish=finish))
                if include_usage:
                    usage_chunk = chunk("")
                    usage_chunk["choices"] = []
                    usage_chunk["usage"] = self._usage(len(job.tokens), len(result.generated))
                    write(usage_chunk)
                self.served += 1
            write("[DONE]")
        finally:
            handler.wfile.write(b"0\r\n\r\n")
            handler.wfile.flush()

    def _decode(self, tokens):
        return self.tokenizer.decode([int(t) for t in tokens])
