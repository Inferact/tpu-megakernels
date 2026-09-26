"""Classify the failures of a Muse Spark GSM8K run (scripts/eval_musespark_gsm8k.sh).

    python scripts/analyze_gsm8k_failures.py <run_dir> [--examples 3] [--json out.json]
    python scripts/analyze_gsm8k_failures.py --compare <run_dir> [<run_dir> ...]

Reads the lm-eval samples (``<run_dir>/*/samples_gsm8k_cot_*.jsonl``, both filters), the
capture proxy's ``captures.jsonl`` (reasoning_content / finish_reason / completion tokens per
request; optional) and the demo's ``server.log`` (fallback for the generated-token counts:
the harness sends the problems in doc order with num_concurrent=1, so the ``chat:`` lines
after the smoke test map onto the docs one to one).

Every problem the flexible-extract filter scores wrong is put into one class:

  truncated     no final answer within the token budget (``finish length`` / empty content);
                the repeated-5-gram rate of the transcript says whether the reasoning loops
  extraction    the gold number is in the answer text but the harness extracted another
                number (flexible-extract takes the LAST number: trailing remarks, hedged
                alternative readings, units such as "in 1 week", or a \\boxed{} / "####"
                form the strict regex does not know)
  arithmetic    the final answer is wrong and some "a op b = c" in the answer is false
  reading       the final answer is wrong, the arithmetic shown is right (misread problem)

and the script prints per-class counts with verbatim examples, generated-token statistics,
the repetition rate of correct vs truncated transcripts and the accuracy the run would have
had without truncations / with the extraction fixed. ``--compare`` prints one side-by-side
table for several runs.
"""

import argparse
import collections
import glob
import json
import os
import re
import statistics
import sys

NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")
ARITH = re.compile(
    r"(-?\$?\d[\d,]*(?:\.\d+)?)\s*([-+*/x×÷])\s*(-?\$?\d[\d,]*(?:\.\d+)?)\s*=\s*(-?\$?\d[\d,]*(?:\.\d+)?)"
)
SERVER_LINE = re.compile(
    r"\[\s*(?P<t>[\d.]+)s\]\s+(?P<kind>chat|completion): (?P<prompt>\d+) prompt tokens "
    r"\(prefill (?P<prefill>[\d.]+) ms\), (?P<gen>\d+) generated in (?P<sec>[\d.]+) s "
    r"\((?P<steps>\d+) steps\), finish (?P<finish>\w+)"
)


def norm_number(s):
    """'$1,430.' -> '1430' (the harness's exact_match normalisation: drop ',', '$', final '.')."""
    s = s.replace(",", "").replace("$", "").strip()
    s = re.sub(r"\.$", "", s)
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else str(v)
    except ValueError:
        return s


def numbers_in(text):
    return [norm_number(m.group(0)) for m in NUMBER.finditer(text)]


def arithmetic_errors(text):
    """[(expression, claimed, actual)] for every 'a op b = c' in `text` that is false."""
    errors = []
    for m in ARITH.finditer(text):
        a, op, b, c = m.groups()
        try:
            a, b, c = (float(norm_number(x)) for x in (a, b, c))
        except ValueError:
            continue
        if op == "+":
            v = a + b
        elif op == "-":
            v = a - b
        elif op in "*x×":
            v = a * b
        else:
            if b == 0:
                continue
            v = a / b
        if abs(v - c) > 0.011 * max(1.0, abs(v)):
            errors.append((m.group(0), c, v))
    return errors


def ngram_repeat_rate(text, n=5):
    """1 - unique n-grams / n-grams over whitespace tokens (0 = no repetition)."""
    words = text.split()
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def load_samples(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "*", "samples_gsm8k_cot_*.jsonl")))
    if not files:
        raise SystemExit(f"no lm-eval samples under {run_dir}")
    docs = {}
    with open(files[-1]) as f:
        for line in f:
            r = json.loads(line)
            d = docs.setdefault(
                r["doc_id"],
                {
                    "doc_id": r["doc_id"],
                    "question": r["doc"]["question"],
                    "gold": norm_number(str(r["target"])),
                    "content": r["resps"][0][0],
                },
            )
            d[r["filter"]] = {
                "extracted": r["filtered_resps"][0],
                "correct": bool(r["exact_match"]),
            }
    return [docs[k] for k in sorted(docs)]


def load_captures(run_dir):
    """question -> first captured chat completion for it (replays after the eval are ignored)."""
    path = os.path.join(run_dir, "captures.jsonl")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            req, resp = rec.get("request") or {}, rec.get("response") or {}
            msgs = req.get("messages") or []
            if not msgs or not isinstance(resp, dict) or "choices" not in resp:
                continue
            q = msgs[-1].get("content", "")
            q = q[2:] if q.startswith("Q:") else q
            q = q.rsplit("\nA:", 1)[0].strip()
            if q in out:
                continue
            choice = resp["choices"][0]
            msg = choice.get("message", {})
            out[q] = {
                "reasoning": msg.get("reasoning_content") or "",
                "content": msg.get("content") or "",
                "finish": choice.get("finish_reason"),
                "gen_tokens": (resp.get("usage") or {}).get("completion_tokens"),
                "prompt_tokens": (resp.get("usage") or {}).get("prompt_tokens"),
                "seconds": rec.get("seconds"),
            }
    return out


def load_server_rows(run_dir):
    path = os.path.join(run_dir, "server.log")
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                m = SERVER_LINE.search(line)
                if m:
                    rows.append({k: (float(v) if k in ("t", "prefill", "sec") else v) for k, v in m.groupdict().items()})
                    rows[-1]["gen"], rows[-1]["prompt"], rows[-1]["steps"] = int(rows[-1]["gen"]), int(rows[-1]["prompt"]), int(rows[-1]["steps"])
    return rows


def attach_generation(docs, captures, server_rows, budget):
    """Fill gen_tokens / finish / reasoning per doc from the captures, else from server.log."""
    matched = 0
    for d in docs:
        c = captures.get(d["question"].strip())
        if c is not None:
            d.update(gen_tokens=c["gen_tokens"], finish=c["finish"], reasoning=c["reasoning"], seconds=c["seconds"])
            matched += 1
    if matched == len(docs):
        return "captures"
    # fallback: server.log rows in request order; the first row(s) are the smoke test(s)
    # (short prompt), the harness sends the docs in doc order
    rows = [r for r in server_rows if r["kind"] == "chat"]
    if len(rows) >= len(docs):
        start = len(rows) - len(docs) if len(rows) - len(docs) <= 3 else None
        if start is not None:
            for d, r in zip(docs, rows[start : start + len(docs)]):
                if "gen_tokens" not in d:
                    d.update(gen_tokens=r["gen"], finish=r["finish"], reasoning=None, seconds=r["sec"])
            # sanity: an empty content must come from a budget hit and the content cannot
            # be longer than the generation
            bad = [d["doc_id"] for d in docs if not d["content"].strip() and d["finish"] != "length"]
            return "server.log" + (f" (WARNING: empty content without finish length for docs {bad})" if bad else "")
    for d in docs:
        d.setdefault("gen_tokens", None)
        d.setdefault("finish", "length" if not d["content"].strip() else "stop")
        d.setdefault("reasoning", None)
    return "none"


def classify(d, budget):
    """(class, note) of a doc the flexible filter scores wrong."""
    content, gold = d["content"], d["gold"]
    if d.get("finish") == "length" or not content.strip():
        note = f"finish={d.get('finish')} gen_tokens={d.get('gen_tokens')}"
        if d.get("reasoning"):
            note += "; gold answer " + ("PRESENT" if gold in numbers_in(d["reasoning"]) else "absent") + " in the reasoning channel"
        return "truncated", note
    nums = numbers_in(content)
    extracted = norm_number(str(d["flexible-extract"]["extracted"]))
    if gold in nums:
        hints = []
        if "\\boxed" in content:
            hints.append("\\boxed{}")
        if "####" in content:
            hints.append("'####'")
        if re.search(r"\b(if|alternatively|literal|colloquial|remark|or |assuming)\b", content, re.I):
            hints.append("hedged alternative")
        return "extraction", f"gold {gold} present, extracted {extracted}; last number is not the answer" + (f" ({', '.join(hints)})" if hints else "")
    errors = arithmetic_errors(content)
    if errors:
        e = errors[0]
        return "arithmetic", f"extracted {extracted} != gold {gold}; false step: '{e[0]}' (actually {e[2]:g})"
    return "reading", f"extracted {extracted} != gold {gold}; shown arithmetic is consistent"


def r_has_reasoning(docs):
    return all(d.get("reasoning") is not None for d in docs)


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0, "mean": None, "median": None, "max": None, "min": None}
    return {"n": len(values), "mean": statistics.fmean(values), "median": statistics.median(values), "max": max(values), "min": min(values)}


def transcript(d):
    return (d.get("reasoning") or "") + "\n" + d["content"]


def analyze(run_dir, budget=None):
    docs = load_samples(run_dir)
    captures = load_captures(run_dir)
    rows = load_server_rows(run_dir)
    config = {}
    cfg_path = os.path.join(run_dir, "run-config.txt")
    if os.path.exists(cfg_path):
        for line in open(cfg_path):
            for m in re.finditer(r"(\w+): (\S+)", line):
                config[m.group(1)] = m.group(2)
    config.setdefault("task", "gsm8k_cot")
    config.setdefault("sampling", "greedy")
    server_log = os.path.join(run_dir, "server.log")
    if "weights" not in config and os.path.exists(server_log):
        m = re.search(r"loading weights from (\S+)", open(server_log).read())
        if m:
            config["weights"] = m.group(1)
    lm_log = os.path.join(run_dir, "lm_eval.log")
    if os.path.exists(lm_log):
        m = re.search(r"'max_gen_toks': (\d+)", open(lm_log).read())
        if m:
            config.setdefault("max_gen_toks", m.group(1))
    if budget is None:
        budget = int(config.get("max_gen_toks", 0) or 0) or None
    source = attach_generation(docs, captures, rows, budget)

    n = len(docs)
    flex_correct = [d for d in docs if d["flexible-extract"]["correct"]]
    strict_correct = [d for d in docs if d["strict-match"]["correct"]]
    failures = [d for d in docs if not d["flexible-extract"]["correct"]]
    classes = collections.OrderedDict((k, []) for k in ("truncated", "extraction", "arithmetic", "reading"))
    for d in failures:
        cls, note = classify(d, budget)
        d["class"], d["note"] = cls, note
        classes[cls].append(d)
    for d in docs:
        d["repeat5"] = ngram_repeat_rate(transcript(d))
        d["repeat5_content"] = ngram_repeat_rate(d["content"])
    truncated = classes["truncated"]
    extraction = classes["extraction"]
    result = {
        "run_dir": run_dir,
        "config": config,
        "n": n,
        "generation_source": source,
        "has_reasoning": all(d.get("reasoning") is not None for d in docs),
        "flexible_acc": len(flex_correct) / n,
        "strict_acc": len(strict_correct) / n,
        "empty": sum(1 for d in docs if not d["content"].strip()),
        "finish_length": sum(1 for d in docs if d.get("finish") == "length"),
        "classes": {k: len(v) for k, v in classes.items()},
        "gen_tokens": stats([d.get("gen_tokens") for d in docs]),
        "gen_tokens_correct": stats([d.get("gen_tokens") for d in flex_correct]),
        "gen_tokens_wrong_answered": stats([d.get("gen_tokens") for d in failures if d["class"] != "truncated"]),
        "repeat5_correct": stats([d["repeat5"] for d in flex_correct]),
        "repeat5_truncated": stats([d["repeat5"] for d in truncated]),
        "repeat5_answered_wrong": stats([d["repeat5"] for d in failures if d["class"] != "truncated"]),
        "loops": sum(1 for d in truncated if d["repeat5"] > 0.3),
        "truncated_gold_in_reasoning": (sum(1 for d in truncated if d.get("reasoning") and d["gold"] in numbers_in(d["reasoning"])) if r_has_reasoning(docs) else None),
        "acc_without_truncations": (len(flex_correct) / (n - len(truncated))) if n > len(truncated) else None,
        "acc_extraction_fixed": (len(flex_correct) + len(extraction)) / n,
        "acc_both": ((len(flex_correct) + len(extraction)) / (n - len(truncated))) if n > len(truncated) else None,
        "seconds_total": sum(d.get("seconds") or 0 for d in docs),
        "docs": docs,
    }
    return result


def fmt(v, digits=1, pct=False):
    if v is None:
        return "-"
    if pct:
        return f"{100 * v:.{digits}f}%"
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def render(result, examples=3, snippet=900):
    r = result
    out = []
    cfg = r["config"]
    out.append(f"# GSM8K failure analysis: `{r['run_dir']}`\n")
    out.append(
        f"weights `{cfg.get('weights', '?')}`, max_gen_toks {cfg.get('max_gen_toks', '?')}, "
        f"reasoning_effort {cfg.get('reasoning_effort', 'medium')}, n = {r['n']}; token counts / finish "
        f"reasons from {r['generation_source']}; reasoning transcripts "
        f"{'available' if r['has_reasoning'] else 'NOT available (content channel only)'}.\n"
    )
    out.append("| metric | value |\n|---|---|")
    out.append(f"| flexible-extract accuracy | {fmt(r['flexible_acc'], pct=True)} ({round(r['flexible_acc'] * r['n'])}/{r['n']}) |")
    out.append(f"| strict-match accuracy | {fmt(r['strict_acc'], pct=True)} |")
    out.append(f"| empty answers (no content) | {r['empty']} |")
    out.append(f"| finish = length (budget hit) | {r['finish_length']} |")
    g = r["gen_tokens"]
    out.append(f"| generated tokens: mean / median / max | {fmt(g['mean'])} / {fmt(g['median'])} / {fmt(g['max'])} |")
    g = r["gen_tokens_correct"]
    out.append(f"| generated tokens, correct problems: mean / median / max | {fmt(g['mean'])} / {fmt(g['median'])} / {fmt(g['max'])} |")
    g = r["gen_tokens_wrong_answered"]
    out.append(f"| generated tokens, wrong-but-answered: mean / median / max | {fmt(g['mean'])} / {fmt(g['median'])} / {fmt(g['max'])} |")
    out.append(f"| repeated-5-gram rate, correct (mean / max) | {fmt(r['repeat5_correct']['mean'], 3)} / {fmt(r['repeat5_correct']['max'], 3)} |")
    out.append(f"| repeated-5-gram rate, truncated (mean / max) | {fmt(r['repeat5_truncated']['mean'], 3)} / {fmt(r['repeat5_truncated']['max'], 3)} |")
    out.append(f"| truncated transcripts that loop (rate > 0.3) | {r['loops']} / {r['classes']['truncated']} |")
    if r["truncated_gold_in_reasoning"] is not None:
        out.append(f"| truncated transcripts whose reasoning contains the gold answer | {r['truncated_gold_in_reasoning']} / {r['classes']['truncated']} |")
    out.append(f"| accuracy excluding truncations | {fmt(r['acc_without_truncations'], pct=True)} |")
    out.append(f"| accuracy with extraction fixed | {fmt(r['acc_extraction_fixed'], pct=True)} |")
    out.append(f"| accuracy excluding truncations and with extraction fixed | {fmt(r['acc_both'], pct=True)} |")
    out.append("")
    out.append("## Failure classes\n")
    out.append("| class | count | doc ids |\n|---|---:|---|")
    by_class = collections.defaultdict(list)
    for d in r["docs"]:
        if "class" in d:
            by_class[d["class"]].append(d)
    for k, c in r["classes"].items():
        out.append(f"| {k} | {c} | {', '.join(str(d['doc_id']) for d in by_class[k])} |")
    out.append("")
    for k in r["classes"]:
        ds = by_class[k]
        if not ds:
            continue
        out.append(f"### {k} ({len(ds)})\n")
        for d in ds:
            out.append(f"- doc {d['doc_id']}: gold {d['gold']}, extracted {d['flexible-extract']['extracted']}, gen tokens {d.get('gen_tokens')}, finish {d.get('finish')}, repeat5 {d['repeat5']:.2f}: {d['note']}")
        out.append("")
        for d in ds[:examples]:
            out.append(f"**doc {d['doc_id']}** (gold {d['gold']}): {d['question']}\n")
            if d.get("reasoning"):
                rs = d["reasoning"]
                shown = rs if len(rs) <= snippet else rs[: snippet // 2] + f"\n[... {len(rs) - snippet} chars ...]\n" + rs[-snippet // 2 :]
                out.append("reasoning:\n```\n" + shown + "\n```")
            c = d["content"]
            shown = c if len(c) <= snippet else c[: snippet // 2] + f"\n[... {len(c) - snippet} chars ...]\n" + c[-snippet // 2 :]
            out.append("content:\n```\n" + (shown if shown.strip() else "(empty)") + "\n```\n")
    return "\n".join(out)


def compare(results, labels=None):
    labels = labels or [os.path.basename(r["run_dir"].rstrip("/")) for r in results]
    rows = [
        ("weights", lambda r: os.path.basename(r["config"].get("weights", "?"))),
        ("max_gen_toks", lambda r: r["config"].get("max_gen_toks", "?")),
        ("reasoning effort", lambda r: r["config"].get("reasoning_effort", "medium")),
        ("task / decoding", lambda r: f"{r['config']['task']} / {r['config']['sampling']}"),
        ("problems", lambda r: r["n"]),
        ("flexible-extract accuracy", lambda r: fmt(r["flexible_acc"], pct=True)),
        ("strict-match accuracy", lambda r: fmt(r["strict_acc"], pct=True)),
        ("empty answers", lambda r: r["empty"]),
        ("budget hits (finish length)", lambda r: r["finish_length"]),
        ("truncated / extraction / arithmetic / reading", lambda r: " / ".join(str(r["classes"][k]) for k in ("truncated", "extraction", "arithmetic", "reading"))),
        ("gen tokens mean / median / max", lambda r: f"{fmt(r['gen_tokens']['mean'])} / {fmt(r['gen_tokens']['median'])} / {fmt(r['gen_tokens']['max'])}"),
        ("gen tokens (correct) mean / median", lambda r: f"{fmt(r['gen_tokens_correct']['mean'])} / {fmt(r['gen_tokens_correct']['median'])}"),
        ("repeat-5-gram rate correct / truncated", lambda r: f"{fmt(r['repeat5_correct']['mean'], 3)} / {fmt(r['repeat5_truncated']['mean'], 3)}"),
        ("truncated with gold answer in reasoning", lambda r: "-" if r["truncated_gold_in_reasoning"] is None else f"{r['truncated_gold_in_reasoning']} / {r['classes']['truncated']}"),
        ("acc. excl. truncations", lambda r: fmt(r["acc_without_truncations"], pct=True)),
        ("acc. extraction fixed", lambda r: fmt(r["acc_extraction_fixed"], pct=True)),
        ("acc. excl. trunc. + extraction fixed", lambda r: fmt(r["acc_both"], pct=True)),
    ]
    out = ["| metric | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for name, fn in rows:
        out.append(f"| {name} | " + " | ".join(str(fn(r)) for r in results) + " |")
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--snippet", type=int, default=900, help="max chars of a transcript shown")
    parser.add_argument("--json", default=None, help="write the per-doc classification here")
    parser.add_argument("--compare", action="store_true", help="side-by-side table of several runs")
    parser.add_argument("--labels", default=None, help="comma-separated column labels for --compare")
    args = parser.parse_args(argv)
    results = [analyze(d) for d in args.run_dirs]
    if args.compare:
        print(compare(results, args.labels.split(",") if args.labels else None))
    else:
        for r in results:
            print(render(r, args.examples, args.snippet))
    if args.json:
        with open(args.json, "w") as f:
            json.dump([{k: v for k, v in r.items()} for r in results], f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
