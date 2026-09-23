"""Streaming TPOT bench client for an OpenAI-compatible completions server.

Measures what a streaming client sees: time to the first content chunk (TTFT),
steady-state time per output token (TPOT), and end-to-end tok/s. Standard
library only; SSE over HTTP chunked transfer.

Run: python bench_openai_stream.py --base-url http://HOST:PORT
Env: ISL=1 OSL=1000 N_WARMUP=2 N_BENCH=4.
"""

import argparse
import http.client
import json
import os
import time


def stream_completion(base, prompt, max_tokens):
    """POST /v1/completions with stream=True; returns (ttft, tpot, n_tokens, total_s)."""
    host, port = base.rsplit(":", 1)
    conn = http.client.HTTPConnection(host, int(port), timeout=3600)
    body = json.dumps({
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "ignore_eos": True,
        "stream_options": {"include_usage": True},
    })
    t0 = time.perf_counter()
    conn.request("POST", "/v1/completions", body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:400]}")

    # http.client decodes the transfer chunking; SSE events are "\n\n"-separated
    buf = b""
    t_first = None  # first content chunk
    t_last = None
    n_tokens = 0

    def events():
        nonlocal buf
        while True:
            data = resp.read1(65536)
            if not data:
                return
            buf += data
            while b"\n\n" in buf:
                frame, _, buf = buf.partition(b"\n\n")
                yield frame

    for frame in events():
        if not frame.startswith(b"data:"):
            continue
        data = frame[5:].strip()
        if data == b"[DONE]":
            continue
        chunk = json.loads(data)
        if chunk.get("usage"):
            n_tokens = chunk["usage"]["completion_tokens"]
            continue
        choices = chunk.get("choices") or []
        text = choices[0].get("text", "") if choices else ""
        if not text:
            continue
        now = time.perf_counter()
        if t_first is None:
            t_first = now
        t_last = now
    conn.close()
    total = t_last - t0 if t_last else time.perf_counter() - t0
    ttft = (t_first - t0) if t_first else total
    tpot = (t_last - t_first) / max(n_tokens - 1, 1) if t_first and n_tokens > 1 else None
    return ttft, tpot, n_tokens, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://localhost:8000"))
    args = ap.parse_args()
    base = args.base_url.removeprefix("http://").removesuffix("/v1").removesuffix("/")
    isl = int(os.environ.get("ISL", "1"))
    osl = int(os.environ.get("OSL", "1000"))
    n_warmup = int(os.environ.get("N_WARMUP", "2"))
    n_bench = int(os.environ.get("N_BENCH", "4"))
    prompt = "hi" if isl <= 1 else " ".join(["word"] * isl)

    print(f"{n_warmup} warmup requests (absorb server compiles)...", flush=True)
    for _ in range(n_warmup):
        stream_completion(base, prompt, osl)
    rows = []
    for _ in range(n_bench):
        ttft, tpot, n_tokens, total = stream_completion(base, prompt, osl)
        rows.append((ttft, tpot, n_tokens, total))
        print(f"  ttft {ttft*1e3:.1f}ms; tpot {tpot*1e3:.4f}ms; {n_tokens} tokens in {total:.3f}s "
              f"({n_tokens/total:.2f} tok/s e2e)", flush=True)
    ttfts = [r[0] for r in rows]
    tpots = [r[1] for r in rows if r[1] is not None]
    e2e = [r[2] / r[3] for r in rows]
    n = len(rows)
    print(f"TTFT: {sum(ttfts)/n*1e3:.1f} ms (mean of {n}; ISL={isl})")
    mean_tpot = sum(tpots)/len(tpots)
    print(f"TPOT: {mean_tpot*1e3:.4f} ms/token -> {1/mean_tpot:.1f} tok/s (mean of {n})")
    print(f"e2e: {sum(e2e)/n:.2f} tok/s (mean of {n}; ISL={isl} OSL={osl}; incl. prefill)")


if __name__ == "__main__":
    main()
