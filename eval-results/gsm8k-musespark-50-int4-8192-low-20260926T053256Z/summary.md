# Muse Spark GSM8K summary: /filestore/srcs/tpu-megakernels/eval-results/gsm8k-musespark-50-int4-8192-low-20260926T053256Z

lm-eval results: `/filestore/srcs/tpu-megakernels/eval-results/gsm8k-musespark-50-int4-8192-low-20260926T053256Z/muse-spark-1.2-816b-a42b/results_2026-09-26T05-38-22.590854.json`  
task gsm8k_cot, 5-shot, n = 50, gen_kwargs = {'do_sample': False, 'until': ['Q:', '</s>', '<|im_end|>'], 'temperature': 0.0, 'max_gen_toks': 8192}

| metric | accuracy | stderr |
|---|---|---|
| exact_match,flexible-extract | 88.0% | 4.6% |
| exact_match,strict-match | 46.0% | 7.1% |

samples: 100; strict correct 0, flexible correct 0, empty responses (no final answer) 4

