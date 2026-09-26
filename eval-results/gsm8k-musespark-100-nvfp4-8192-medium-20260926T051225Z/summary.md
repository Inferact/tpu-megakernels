# Muse Spark GSM8K summary: /filestore/srcs/tpu-megakernels/eval-results/gsm8k-musespark-100-nvfp4-8192-medium-20260926T051225Z

lm-eval results: `/filestore/srcs/tpu-megakernels/eval-results/gsm8k-musespark-100-nvfp4-8192-medium-20260926T051225Z/muse-spark-1.2-816b-a42b/results_2026-09-26T05-32-56.223141.json`  
task gsm8k_cot, 5-shot, n = 100, gen_kwargs = {'do_sample': False, 'until': ['Q:', '</s>', '<|im_end|>'], 'temperature': 0.0, 'max_gen_toks': 8192}

| metric | accuracy | stderr |
|---|---|---|
| exact_match,flexible-extract | 92.0% | 2.7% |
| exact_match,strict-match | 57.0% | 5.0% |

samples: 200; strict correct 0, flexible correct 0, empty responses (no final answer) 2

