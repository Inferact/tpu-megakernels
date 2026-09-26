# Muse Spark 1.2 on TPU: GSM8K accuracy diagnosis

Runs: `gsm8k_cot`, 5-shot, chat template, greedy, lm-eval `local-chat-completions` (`scripts/eval_musespark_gsm8k.sh`), seed 0, the first 100 test problems (the 50-problem low-effort run is the first 50 of the same list). Token counts, finish reasons and the reasoning channel come from the capture proxy (`captures.jsonl`); the 4096 run predates the proxy and uses `server.log` (no reasoning transcripts). Failure classes: `scripts/analyze_gsm8k_failures.py` (per-run `failures.md`).

## Side by side

| metric | int4 4096 | int4 8192 | nvfp4 8192 | int4 low (50) | nvfp4 llama greedy | nvfp4 llama sampled |
|---|---|---|---|---|---|---|
| weights | muse-spark-tp8-int4 | muse-spark-tp8-int4 | muse-spark-tp8-nvfp4 | muse-spark-tp8-int4 | muse-spark-tp8-nvfp4 | muse-spark-tp8-nvfp4 |
| max_gen_toks | 4096 | 8192 | 8192 | 8192 | 8192 | 8192 |
| reasoning effort | medium | medium | medium | low | medium | medium |
| task / decoding | gsm8k_cot / greedy | gsm8k_cot / greedy | gsm8k_cot / greedy | gsm8k_cot / greedy | gsm8k_cot_llama / greedy | gsm8k_cot_llama / t1.0k64p1.0 |
| problems | 100 | 100 | 100 | 50 | 100 | 100 |
| flexible-extract accuracy | 87.0% | 87.0% | 92.0% | 88.0% | 97.0% | 98.0% |
| strict-match accuracy | 53.0% | 53.0% | 57.0% | 46.0% | 97.0% | 98.0% |
| empty answers | 5 | 5 | 1 | 2 | 0 | 0 |
| budget hits (finish length) | 5 | 5 | 1 | 2 | 0 | 0 |
| truncated / extraction / arithmetic / reading | 5 / 7 / 0 / 1 | 5 / 7 / 0 / 1 | 1 / 6 / 0 / 1 | 2 / 3 / 0 / 1 | 0 / 2 / 1 / 0 | 0 / 2 / 0 / 0 |
| gen tokens mean / median / max | 711.9 / 491.5 / 4096 | 916.7 / 491.5 / 8192 | 591.1 / 464.0 / 8192 | 745.4 / 413.0 / 8192 | 665.8 / 639.5 / 1355 | 561.6 / 521.5 / 2411 |
| gen tokens (correct) mean / median | 501.6 / 473 | 501.6 / 473 | 494.1 / 454.5 | 414.2 / 400.0 | 654.6 / 634 | 535.0 / 521.0 |
| repeat-5-gram rate correct / truncated | 0.001 / 0.000 | 0.038 / 0.950 | 0.037 / 0.943 | 0.033 / 0.956 | 0.001 / - | 0.000 / - |
| truncated with gold answer in reasoning | - | 5 / 5 | 1 / 1 | 2 / 2 | - | - |
| acc. excl. truncations | 91.6% | 91.6% | 92.9% | 91.7% | 97.0% | 98.0% |
| acc. extraction fixed | 94.0% | 94.0% | 98.0% | 94.0% | 99.0% | 100.0% |
| acc. excl. trunc. + extraction fixed | 98.9% | 98.9% | 99.0% | 97.9% | 99.0% | 100.0% |

Decode throughput (sum of generated tokens / sum of decode seconds, B=1 server):

| run | generated tokens | decode s | tok/s |
|---|---:|---:|---:|
| int4 4096 | 71189 | 301.9 | 235.8 |
| int4 8192 | 91669 | 343.7 | 266.7 |
| nvfp4 8192 | 59110 | 237.1 | 249.3 |
| int4 low (50) | 37549 | 140.9 | 266.5 |
| nvfp4 llama greedy | 66584 | 272.0 | 244.8 |
| nvfp4 llama sampled | 56156 | 238.4 | 235.6 |

Run directories:

- int4 4096: `gsm8k-musespark-100-20260926T030101Z` (head cd0d3c3)
- int4 8192: `gsm8k-musespark-100-int4-8192-medium-20260926T045954Z` (head 37bd619)
- nvfp4 8192: `gsm8k-musespark-100-nvfp4-8192-medium-20260926T051225Z` (head 37bd619)
- int4 low (50): `gsm8k-musespark-50-int4-8192-low-20260926T053256Z` (head c11b8b2)
- nvfp4 llama greedy: `gsm8k-musespark-100-nvfp4-8192-medium-llama-20260926T053831Z` (head c11b8b2)
- nvfp4 llama sampled: `gsm8k-musespark-100-nvfp4-8192-medium-llama-t1.0k64p1.0-20260926T054705Z` (head c11b8b2)

## Failure classification (int4, 100 problems; identical at 4096 and 8192)

| class | count | doc ids | what happened |
|---|---:|---|---|
| truncated | 5 | 29, 50, 51, 81, 95 | reasoning channel reaches the right answer within the first ~300 tokens, then degenerates into a 2-4 sentence loop ("We should answer. We should provide final answer. We should do that. We should ensure we are done.") until the budget; the answer channel is never opened |
| extraction | 7 | 7, 20, 45, 61, 70, 85, 93 | the answer text contains the gold number but flexible-extract takes the LAST number in the response: a trailing remark, alternative reading, unit ("in 1 week"), or sub-total |
| arithmetic | 0 |  | a shown a op b = c step is false |
| reading | 1 | 37 | consistent arithmetic, misread problem (doc 37: answers 0 lego sets left instead of 2) |

- Repeated-5-gram rate: correct transcripts 0.038 (max 0.134); the 5 truncated transcripts 0.950 (all 5 > 0.9, loops). All 5 contain the gold answer in the reasoning channel.
- Generated tokens: all problems mean/median/max 917/492/8192; correct problems 502/473/1508; wrong-but-answered 883/891/1071.
- Counterfactual accuracy: excluding truncations 91.6%; with extraction fixed (gold number present in the answer) 94.0%; both 98.9%.
- No `\boxed{}` or `####` answers in any run; the strict regex misses (53-57%) are the model not using the exact "The answer is N." phrase, not a formatting bug.

### Three verbatim failures (int4 8192)

**truncated loop, doc 50 (gold 294)**: Lloyd has an egg farm. His chickens produce 252 eggs per day and he sells them for $2 per dozen. How much does Lloyd make on eggs per week?

reasoning channel (40683 chars, 8192 tokens, finish length), head and tail:
```
We need to answer. The user is giving simple math problems. This one: 252 eggs per day, $2 per dozen. How much per week?

252 eggs per day = 252/12 = 21 dozen per day. 21 * $2 = $42 per day. Per week = $42 * 7 = $294.

We should answer straightforward. Probably continue pattern: Show reasoning.

We should answer: $294.

We should be consistent with previous style: "Originally ... So ... The answer is ..."

We should provide answer.

Potentially user expects simple answer.

We should not overcomplicate.

We should answer: $294 per week.

Make sure formatting simple.

We should continue to be helpful.

We should not add unnecessary fluff.

We should answer directly.

Probably add steps: 252/12
[... 39583 chars of the same loop ...]


We should ensure we don't add extra.

We should answer.

We should provide final answer.

We should ensure we don't add extra.

We should answer.

We should provide final answer.

We should ensure we don't add extra.

We should answer.

We should provide final answer.

We should ensure we don't add extra.

We should answer.

We should provide final answer.

We should ensure we don't add extra.


```
content channel: (empty)

**extraction, doc 61 (gold 1430)**: Janet buys a brooch for her daughter.  She pays $500 for the material to make it and then another $800 for the jeweler to construct it.  After that, she pays 10% of that to get it insured.  How much did she pay?

content channel (extracted `$130`):
```
Janet paid $500 + $800 = $1300 to make the brooch. 10% of $1300 is $130 for insurance, for a total of $1300 + $130 = $1430. The answer is $1430 total, $130 for insurance alone.
```

**extraction, doc 85 (gold 44)**: Gene is sewing a quilt out of old souvenir t-shirts. He has one shirt from each vacation he has been on. Every shirt is its own quilt block. Each row is made of blocks from a different year of vacations. He goes on four vacations a year and has been vacationing since he was 23 years old. He is now 34. How many quilt blocks does he have in total?

content channel (extracted `48`):
```
Gene has been vacationing for 34 - 23 = 11 years. At 4 vacations per year that is 11 * 4 = 44 vacations, so 44 shirts/blocks. 

*If counting both the year he was 23 and the year he is now 34 as full years, it is 12 years = 48 blocks.*
```

## int4 vs NVFP4 per problem

Only 10/100 answer texts are byte-identical between the two containers (expected: different quantization changes the greedy path), but the failure profile is the same: no arithmetic errors in either; the same doc 37 misreading; the same style of extraction misses; and the same loop pattern (nvfp4 loops on doc 80, where int4 answered in 443 tokens, while int4 loops on 5 problems nvfp4 answers in 364-552 tokens). Problems that changed outcome:

| doc | gold | int4 8192 (tokens) | nvfp4 8192 (tokens) |
|---|---|---|---|
| 15 | 125 | correct (383) | extraction (608) |
| 20 | 15 | extraction (918) | correct (781) |
| 29 | 104 | truncated (8192) | correct (364) |
| 45 | 104 | extraction (1071) | correct (843) |
| 50 | 294 | truncated (8192) | correct (468) |
| 51 | 5 | truncated (8192) | correct (418) |
| 57 | 83 | correct (556) | extraction (607) |
| 61 | 1430 | extraction (631) | correct (446) |
| 70 | 7425 | extraction (817) | correct (769) |
| 80 | 10 | correct (443) | truncated (8192) |
| 81 | 17 | truncated (8192) | correct (497) |
| 87 | 9360 | correct (1508) | extraction (1010) |
| 95 | 40 | truncated (8192) | correct (552) |

## Reasoning-effort sensitivity (int4, first 50 problems)

| | medium (strength 64) | low (strength 16) |
|---|---|---|
| flexible-extract | 45/50 = 90% | 44/50 = 88% |
| budget hits (loops) | 1 | 2 |
| classes trunc/extr/arith/read | 1/3/0/1 | 2/3/0/1 |
| gen tokens mean / median / max | 681 / 492 / 8192 | 745 / 413 / 8192 |
| repeat-5-gram, correct / truncated | 0.034 / 0.951 | 0.033 / 0.956 |

Low effort does not remove the loops: 2/50 transcripts (docs 14 and 23, both answered in 371-644 tokens at medium) degenerate into the same "We should answer. We should provide final answer." cycle after reaching the right answer, while the medium-effort looper (doc 29) answers at low effort in 310 tokens. Reasoning gets ~20% shorter (median 1341 vs 1690 chars) and the correct answers come in fewer tokens (mean 414 vs 502), but accuracy is unchanged (44 vs 45 / 50) and the loop rate (4% vs 2-5%) is within noise. The loop is therefore not a budget or effort artefact: it is a greedy-decoding attractor of this checkpoint's self-talk, reached after the arithmetic is done.


## Answer-format sensitivity (nvfp4, gsm8k_cot_llama, 100 problems)

With the 8-shot llama-format prompt ("Your response should end with 'The final answer is [answer]'") greedy decoding scores 97% flexible / 97% strict with 0 budget hits (classes trunc/extr/arith/read = 0/2/1/0), tokens mean/median/max 666/640/1355.

With the checkpoint's generation_config sampling (temperature 1.0, top_k 64, top_p 1, seed 0) on the same prompt: 98% flexible / 98% strict, 0 budget hits (0 looping), classes 0/2/0/0, tokens mean/median/max 562/522/2411.

## Conclusion

**The failures are not a numerical (translation) error and only marginally a quantization effect; they are a decoding/format artefact of a long-CoT chat model driven greedily through a completion-style prompt.**

1. **No arithmetic errors.** Across 400 greedy int4/nvfp4 transcripts at medium effort (and 50 at low effort) not a single shown `a op b = c` step is false, and every truncated transcript reaches the gold answer within its first ~300 tokens before it starts looping. A kernel or reference maths bug (wrong RoPE, scale, softcap, expert dequant ...) would surface as wrong numbers or garbled prose, not as correct arithmetic followed by a grammatical self-talk loop. The reference-vs-sglang CPU check (commit c11b8b2) is being run separately and is expected to confirm this.

2. **The residual failures are answer-format and extraction problems of the `gsm8k_cot` prompt.** With "Q: ... A:" 5-shot prompts the model treats the format as free-form and ends with hedged remarks ("... or 2 hours 40 minutes", "12 years = 48 blocks", "$130 for insurance alone"), so flexible-extract (last number) picks the wrong number in 6-7/100 cases even though the gold number is present. With the llama-format prompt (8-shot, "Your response should end with 'The final answer is [answer]'") the same NVFP4 kernel scores **97% flexible / 97% strict, 0 empties, 0 loops**, tokens mean/median/max 666/640/1355; its 3 misses are one off-by-one reading (doc 12), the doc-37 misreading shared by every run, and one "36.36 seconds, 36 seconds" answer. That is above the > 95% requirement.

3. **The loops are a greedy-argmax attractor, not a budget or effort effect.** Doubling the budget from 4096 to 8192 changes nothing (the same 5 problems loop for 8192 tokens with repeated-5-gram rate 0.95; the answered problems are byte-identical). Low effort shortens the reasoning by ~20% but still loops on 2/50 (different problems). The loop starts after the answer is found, in the "We should answer / We should provide final answer / We should do that / We should ensure ..." self-talk that appears in 94/95 correct transcripts too; under argmax this 3-4 sentence cycle becomes a fixed point. The llama-format prompt, which gives the model an explicit terminal phrase, removes it entirely (0/100 loops, no answer over 1355 tokens). With the checkpoint's own sampling defaults (temperature 1.0, top_k 64, top_p 1, seed 0) on the same llama-format prompt the kernel scores **98% / 98%**, again 0 loops, tokens mean/median/max 562/522/2411, and the correct transcripts have a repeated-5-gram rate of 0.000; the two misses are doc 37 ("The final answer is 2, literally 0 if he sold all 13": gold present, last number taken) and doc 87 (adds a compounded-interest alternative). Sampling is the vendor's shipped decoding, and it is at least as accurate as argmax here.

4. **Quantization: int4 RTN is measurably worse than NVFP4 but it is not the cause.** Same problems, same prompt, same kernel: int4 87% / 5 loops, nvfp4 92% / 1 loop; only 10/100 answer texts are identical, but the per-problem flips go both ways (int4 loops on 29/50/51/81/95 which nvfp4 answers in < 560 tokens; nvfp4 loops on 80 which int4 answers in 443 tokens) and both containers have the same class profile (0 arithmetic, 1 reading, 6-7 extraction). The loop attractor exists in both; int4 falls into it more often because its greedy path is perturbed. Since the vendor ships the NVFP4 experts (the int4 container is our own RTN of the bf16 experts), the NVFP4 container should be the deployment default; the int4 path stays for comparison only.

**Recommendation for the > 95% target:** deploy the NVFP4 container through the decode kernel, evaluate with `gsm8k_cot_llama` (or any prompt that names the terminal answer phrase), and use the checkpoint's sampling defaults rather than argmax for open-ended generation. If argmax must be kept, a repetition guard (stop when a 4-paragraph cycle of the reasoning channel repeats > 3 times and force ` to=user`) would recover the 1-5 looping problems, whose reasoning already contains the right answer.

**Side finding (bug, not in my files):** `musespark.sampling.sample` is wrong for `temperature == 0` requests on a server started in sampling mode (`--temperature 1.0`): `scaled = logits / jnp.maximum(temperature, float32.tiny)` overflows every softcapped logit above ~4 to `+inf`, `lax.top_k` then ties at `inf` and the "greedy" branch (`choice = 0`) picks the lowest token id among them, so the request returns `! " # $ %` garbage (ids 0-5) until the budget. Reproduced on CPU: `sample(logits, key, 0.0, 64, 1.0)` = 0 while `argmax` = 22988 and `sample(..., 1e-3, ...)` = 22988. The sampled-run smoke test (temperature 0.0 in the request) shows it (`smoke.json`: 1024 tokens of `%!#$`); the lm-eval requests (temperature 1.0) are unaffected. Fix: `scaled = jnp.where(temperature > 0, logits / jnp.maximum(temperature, tiny), logits)` (or clamp at, say, 1e-4).

