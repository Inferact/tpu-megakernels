"""Summarise a Muse Spark GSM8K run made by scripts/eval_musespark_gsm8k.sh.

    python scripts/eval_musespark_accuracy.py <output_path> [--base-url http://host:port] [--examples 3]

Reads the lm-eval results / samples under ``<output_path>`` and the demo's ``server.log``
(one ``chat: ... N generated in S s (K steps), finish R`` line per request) and prints a
Markdown summary: accuracy (strict / flexible extraction), generated tokens per problem
(mean / median / max), how many requests hit the token budget (``finish length``: no final
answer, counted as wrong), decode throughput, and a few example transcripts. With
``--base-url`` and the server still running, the example problems are re-sent (greedy, so the
output is identical to the evaluated one) to fetch the ``reasoning_content`` channel, which
lm-eval does not record.
"""

import argparse
import glob
import json
import os
import re
import statistics
import sys
import urllib.request

LINE = re.compile(
    r"\[\s*(?P<t>[\d.]+)s\]\s+(?P<kind>chat|completion): (?P<prompt>\d+) prompt tokens "
    r"\(prefill (?P<prefill>[\d.]+) ms\), (?P<gen>\d+) generated in (?P<sec>[\d.]+) s "
    r"\((?P<steps>\d+) steps\), finish (?P<finish>\w+)"
)


def parse_server_log(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            m = LINE.search(line)
            if m:
                rows.append(
                    {
                        "t": float(m["t"]),
                        "prompt": int(m["prompt"]),
                        "prefill_ms": float(m["prefill"]),
                        "generated": int(m["gen"]),
                        "seconds": float(m["sec"]),
                        "steps": int(m["steps"]),
                        "finish": m["finish"],
                    }
                )
    return rows


def find_lm_eval_outputs(output_path):
    results = sorted(glob.glob(os.path.join(output_path, "**", "results_*.json"), recursive=True))
    samples = sorted(
        glob.glob(os.path.join(output_path, "**", "samples_gsm8k*.jsonl"), recursive=True)
    )
    return (results[-1] if results else None), (samples[-1] if samples else None)


def load_samples(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def gold_answer(doc):
    return doc["answer"].split("####")[-1].strip()


def sample_correct(sample):
    """(strict, flexible) exact_match of one lm-eval sample."""
    strict = flexible = None
    for key, value in sample.items():
        if key.startswith("exact_match,strict"):
            strict = float(value)
        elif key.startswith("exact_match,flexible"):
            flexible = float(value)
    return strict, flexible


def request_messages(sample):
    """The chat messages lm-eval sent for this sample (from ``arguments``), or None."""
    args = sample.get("arguments") or {}
    for value in args.values():
        arg0 = value.get("arg_0") if isinstance(value, dict) else None
        if isinstance(arg0, str):
            try:
                arg0 = json.loads(arg0)
            except ValueError:
                return [{"role": "user", "content": arg0}]
        if isinstance(arg0, list):
            return arg0
        if isinstance(arg0, dict) and "messages" in arg0:
            return arg0["messages"]
    return None


def fetch_transcript(base_url, model, messages, max_tokens):
    body = json.dumps(
        {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0}
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=3600) as resp:
        payload = json.load(resp)
    choice = payload["choices"][0]
    return (
        choice["message"].get("reasoning_content") or "",
        choice["message"].get("content") or "",
        choice.get("finish_reason"),
        payload.get("usage", {}).get("completion_tokens"),
    )


def excerpt(text, head=400, tail=300):
    text = (text or "").strip()
    if len(text) <= head + tail + 20:
        return text
    return text[:head] + f"\n[... {len(text) - head - tail} chars ...]\n" + text[-tail:]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("output_path")
    ap.add_argument("--base-url", default=None, help="running server for reasoning transcripts")
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--model", default="muse-spark-1.2-816b-a42b")
    args = ap.parse_args()

    results_path, samples_path = find_lm_eval_outputs(args.output_path)
    rows = parse_server_log(os.path.join(args.output_path, "server.log"))
    print(f"# Muse Spark GSM8K summary: {args.output_path}\n")

    # --- accuracy -------------------------------------------------------------------------------
    samples = load_samples(samples_path) if samples_path else []
    if results_path:
        with open(results_path) as f:
            results = json.load(f)
        task_name = next(iter(results["results"]), "gsm8k_cot")
        task = results["results"].get(task_name, {})
        n = results.get("n-samples", {}).get(task_name, {}).get("effective", len(samples))
        cfg = results.get("configs", {}).get(task_name, {})
        print(f"lm-eval results: `{results_path}`  ")
        print(
            f"task {task_name}, {cfg.get('num_fewshot', '?')}-shot, n = {n}, "
            f"gen_kwargs = {cfg.get('generation_kwargs', {})}\n"
        )
        print("| metric | accuracy | stderr |")
        print("|---|---|---|")
        for key in sorted(task):
            if key.startswith("exact_match,") and not key.endswith("_stderr"):
                stderr = task.get(key.replace("exact_match,", "exact_match_stderr,"), float("nan"))
                print(f"| {key} | {task[key] * 100:.1f}% | {stderr * 100:.1f}% |")
        print()
    else:
        print("lm-eval results: none found (run incomplete?)\n")
    if samples:
        strict = [sample_correct(s)[0] for s in samples]
        flexible = [sample_correct(s)[1] for s in samples]
        empty = sum(1 for s in samples if not (s.get("resps") or [[""]])[0][0].strip())
        print(
            f"samples: {len(samples)}; strict correct {int(sum(x or 0 for x in strict))}, "
            f"flexible correct {int(sum(x or 0 for x in flexible))}, empty responses "
            f"(no final answer) {empty}\n"
        )

    # --- generation length and throughput (server side) -----------------------------------------
    if rows:
        gen = [r["generated"] for r in rows]
        total_gen = sum(gen)
        total_dec = sum(r["seconds"] for r in rows)
        total_pre = sum(r["prefill_ms"] for r in rows) / 1e3
        truncated = [r for r in rows if r["finish"] != "stop"]
        wall = rows[-1]["t"] - rows[0]["t"] + rows[0]["seconds"] + rows[0]["prefill_ms"] / 1e3
        print(f"server log: {len(rows)} requests ({rows[0]['kind']})  ")
        print(
            f"generated tokens per request: mean {statistics.mean(gen):.0f}, "
            f"median {statistics.median(gen):.0f}, max {max(gen)}, min {min(gen)}  "
        )
        print(f"prompt tokens per request: mean {statistics.mean(r['prompt'] for r in rows):.0f}  ")
        print(
            f"hit max_tokens (finish length, no final answer -> wrong): {len(truncated)} / {len(rows)}  "
        )
        print(
            f"decode throughput: {total_gen} tokens in {total_dec:.1f} s = "
            f"{total_gen / max(total_dec, 1e-9):.1f} tok/s (single row, B=1); "
            f"prefill total {total_pre:.1f} s (mean {1e3 * total_pre / len(rows):.0f} ms); "
            f"end-to-end {total_gen / max(wall, 1e-9):.1f} tok/s over {wall:.0f} s wall\n"
        )
    else:
        print("server log: no completion lines found\n")

    # --- examples -------------------------------------------------------------------------------
    if samples and args.examples:
        print(f"## Examples (first {args.examples} problems)\n")
        for i, s in enumerate(samples[: args.examples]):
            doc = s["doc"]
            content = (s.get("resps") or [[""]])[0][0]
            filtered = s.get("filtered_resps", [None])[0]
            strict_ok, flex_ok = sample_correct(s)
            reasoning, finish, ntok = None, None, None
            if args.base_url:
                try:
                    msgs = request_messages(s) or [{"role": "user", "content": doc["question"]}]
                    reasoning, content2, finish, ntok = fetch_transcript(
                        args.base_url, args.model, msgs, args.max_tokens
                    )
                    if content2.strip() != content.strip():
                        content = content2 + "\n[replayed answer differs from the evaluated one]"
                except Exception as error:  # noqa: BLE001
                    reasoning = f"[reasoning replay failed: {error}]"
            print(f"### {i + 1}. doc {s.get('doc_id')}\n")
            print(f"**Question:** {doc['question']}\n")
            if reasoning is not None:
                print(f"**Reasoning (excerpt):**\n\n```\n{excerpt(reasoning)}\n```\n")
            print(f"**Final answer:**\n\n```\n{excerpt(content, 600, 200)}\n```\n")
            print(
                f"**Extracted:** {filtered!r}; **gold:** {gold_answer(doc)}; "
                f"strict {'correct' if strict_ok else 'wrong'}, "
                f"flexible {'correct' if flex_ok else 'wrong'}"
                + (f"; finish {finish}, {ntok} tokens" if finish else "")
                + "\n"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
