# Muse Spark GSM8K summary: /filestore/srcs/tpu-megakernels/eval-results/gsm8k-musespark-100-nvfp4-8192-medium-llama-20260926T053831Z

lm-eval results: `/filestore/srcs/tpu-megakernels/eval-results/gsm8k-musespark-100-nvfp4-8192-medium-llama-20260926T053831Z/muse-spark-1.2-816b-a42b/results_2026-09-26T05-47-04.719154.json`  
task gsm8k_cot_llama, 8-shot, n = 100, gen_kwargs = {'do_sample': False, 'until': ['<|eot_id|>', '<|start_header_id|>user<|end_header_id|>', 'Q:', '</s>', '<|im_end|>'], 'temperature': 0.0, 'max_gen_toks': 8192}

| metric | accuracy | stderr |
|---|---|---|
| exact_match,flexible-extract | 97.0% | 1.7% |
| exact_match,strict-match | 97.0% | 1.7% |

samples: 200; strict correct 0, flexible correct 0, empty responses (no final answer) 0

