# Muse Spark GSM8K summary: /filestore/srcs/tpu-megakernels/eval-results/gsm8k-musespark-100-int4-8192-medium-20260926T045954Z

lm-eval results: `/filestore/srcs/tpu-megakernels/eval-results/gsm8k-musespark-100-int4-8192-medium-20260926T045954Z/muse-spark-1.2-816b-a42b/results_2026-09-26T05-14-46.108400.json`  
task gsm8k_cot, 5-shot, n = 100, gen_kwargs = {'do_sample': False, 'until': ['Q:', '</s>', '<|im_end|>'], 'temperature': 0.0, 'max_gen_toks': 8192}

| metric | accuracy | stderr |
|---|---|---|
| exact_match,flexible-extract | 87.0% | 3.4% |
| exact_match,strict-match | 53.0% | 5.0% |

samples: 200; strict correct 0, flexible correct 0, empty responses (no final answer) 10

