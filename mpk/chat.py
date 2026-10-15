"""Muse Spark chat formatting and token streaming."""

import math
import queue
import sys
import threading
import time

import numpy as np

CHAT_MODES = ("none", "response")
STOP_IDS = frozenset((200001, 200008))  # <|end_of_text|>, <|eot|>


def render_chat_tokens(tokenizer, messages, mode):
    """Muse-Spark chat: system + user turns, assistant generation prefix.

    Mirrors chat_template.jinja for the plain text path (no tools): BOS, then
    ``<|start|>{role}<|message|>...<|eot|>`` turns; the system turn carries the
    reasoning-strength hint (xhigh=512); the assistant prefix is left open.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    parts = []
    has_system = any(m.get("role") == "system" for m in messages)
    for message in messages:
        role = message.get("role", "user")
        body = message.get("content") or ""
        if isinstance(body, list):
            body = "".join(
                part.get("text", "")
                for part in body
                if isinstance(part, dict) and part.get("type") == "text"
            )
        parts.append((role, body))
    if not has_system:
        parts.insert(0, ("system", "You are a helpful assistant.\nReasoning strength: 512."))
    text = ["<|begin_of_text|>"]
    for role, body in parts:
        text.append(f"<|start|>{role}<|message|>{body}<|eot|>")
    text.append("<|start|>assistant")
    return np.asarray(tokenizer.encode("".join(text)), np.int32)


class WholeTextStreamer:
    """Emits text deltas by re-decoding the full token list on each feed (raw path)."""

    def __init__(self, tokenizer, **_unused):
        self.tokenizer = tokenizer
        self.tokens = []
        self.text = ""

    def feed(self, tokens):
        self.tokens.extend(int(t) for t in tokens if int(t) not in STOP_IDS)
        new = self.tokenizer.decode(self.tokens)
        delta = new[len(self.text) :]
        self.text = new
        return [("content", delta)] if delta else []


# Muse recipient-segment structure tokens (from the checkpoint's tokenizer_config).
MUSE_START = 200022  # <|start|>
MUSE_MSG = 200023    # <|message|>
MUSE_EOM = 200007    # <|eom|>  (segment ends, turn continues)


class MuseTokenStreamer:
    """feed(token_ids) -> [(channel, text_delta)] over Muse recipient segments.

    An assistant turn is a run of <|start|>assistant to=<recipient><|message|>
    body <|eom|> segments; the to=self body is reasoning, to=user is the answer.
    Control tokens never reach the decoded text (the tokenizer drops specials,
    so the structure is parsed at the token-id level here).
    """

    def __init__(self, tokenizer, **_unused):
        self.tokenizer = tokenizer
        self.in_header = True  # generation starts right after <|start|>assistant
        self.header = []
        self.body = []
        self.text = ""
        self.channel = "content"

    def feed(self, tokens):
        deltas = []
        for token in tokens:
            t = int(token)
            if t in STOP_IDS:
                continue
            if self.in_header:
                if t == MUSE_MSG:
                    header = self.tokenizer.decode(self.header)
                    self.channel = "reasoning" if "to=self" in header else "content"
                    self.in_header, self.header, self.body, self.text = False, [], [], ""
                    continue
                if t == MUSE_START:
                    self.header = []
                    continue
                self.header.append(t)
                continue
            if t in (MUSE_EOM, MUSE_START):
                self.in_header, self.header = True, []
                continue
            if t == MUSE_MSG:
                continue
            self.body.append(t)
            new = self.tokenizer.decode(self.body)
            delta = new[len(self.text) :]
            if delta:
                deltas.append((self.channel, delta))
                self.text = new
        return deltas


def muse_split_channels(tokenizer, tokens):
    """Split a full Muse assistant turn before decoding removes its markers."""
    deltas = MuseTokenStreamer(tokenizer).feed(tokens)
    reasoning = "".join(text for channel, text in deltas if channel == "reasoning")
    content = "".join(text for channel, text in deltas if channel == "content")
    return reasoning or None, content.strip()


class _InteractiveStream:
    """Print generated text as it is produced, paced to roughly device speed.

    Token chunks arrive from the device in bursts; this replays them on a
    background thread instead of dumping a burst at a time.
    """

    TERMINAL_CHUNK_TOKENS = 8

    def __init__(self, tokenizer, *, chat=True):
        self.items = queue.Queue()
        self.streamer = MuseTokenStreamer(tokenizer) if chat else WholeTextStreamer(tokenizer)
        self.channel = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, tokens, device_seconds=0.0):
        token_ids = [int(t) for t in tokens]
        if token_ids:
            self.items.put((token_ids, max(float(device_seconds), 0.0)))

    def close(self):
        self.items.put(None)
        self.thread.join()

    def _run(self):
        while True:
            item = self.items.get()
            if item is None:
                return
            token_ids, device_seconds = item
            step = self.TERMINAL_CHUNK_TOKENS
            chunks = math.ceil(len(token_ids) / step)
            delay = device_seconds / max(chunks, 1)
            for start in range(0, len(token_ids), step):
                for channel, text in self.streamer.feed(token_ids[start : start + step]):
                    if channel == "content" and self.channel == "reasoning":
                        sys.stdout.write("\n\n[response]\n")
                    elif channel == "reasoning" and self.channel is None:
                        sys.stdout.write("[reasoning] ")
                    if text:
                        sys.stdout.write(text)
                        self.channel = channel
                sys.stdout.flush()
                if start + step < len(token_ids) and delay:
                    time.sleep(delay)
