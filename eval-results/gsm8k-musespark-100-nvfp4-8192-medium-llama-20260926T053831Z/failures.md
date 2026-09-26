# GSM8K failure analysis: `/filestore/srcs/tpu-megakernels/eval-results/gsm8k-musespark-100-nvfp4-8192-medium-llama-20260926T053831Z`

weights `/filestore/weights/muse-spark-tp8-nvfp4`, max_gen_toks 8192, reasoning_effort medium, n = 100; token counts / finish reasons from server.log; reasoning transcripts NOT available (content channel only).

| metric | value |
|---|---|
| flexible-extract accuracy | 97.0% (97/100) |
| strict-match accuracy | 97.0% |
| empty answers (no content) | 0 |
| finish = length (budget hit) | 0 |
| generated tokens: mean / median / max | 665.8 / 639.5 / 1355 |
| generated tokens, correct problems: mean / median / max | 654.6 / 634 / 1355 |
| generated tokens, wrong-but-answered: mean / median / max | 1028.3 / 972 / 1276 |
| repeated-5-gram rate, correct (mean / max) | 0.001 / 0.062 |
| repeated-5-gram rate, truncated (mean / max) | - / - |
| truncated transcripts that loop (rate > 0.3) | 0 / 0 |
| accuracy excluding truncations | 97.0% |
| accuracy with extraction fixed | 99.0% |
| accuracy excluding truncations and with extraction fixed | 99.0% |

## Failure classes

| class | count | doc ids |
|---|---:|---|
| truncated | 0 |  |
| extraction | 2 | 37, 93 |
| arithmetic | 1 | 12 |
| reading | 0 |  |

### extraction (2)

- doc 37: gold 2, extracted 0, gen tokens 1276, finish stop, repeat5 0.00: gold 2 present, extracted 0; last number is not the answer (hedged alternative)
- doc 93: gold 36, extracted 10, gen tokens 972, finish stop, repeat5 0.00: gold 36 present, extracted 10; last number is not the answer

**doc 37** (gold 2): John plans to sell all his toys and use the money to buy video games. He has 13 lego sets and he sells them for $15 each. He ends up buying 8 video games for $20 each and has $5 left. How many lego sets does he still have?

content:
```
He started with 13 sets and sold them all so regardless of the purchase price no sets remain. At $15 each 13 sets is $195 and 8 games at $20 is $160 leaving $35 not $5. If he only sold enough to cover $160 + $5 = $165 he sold 11 sets and would have 2 left. On the literal reading he sold all. The final answer is 0
```

**doc 93** (gold 36): Lee used to be able to run the 400-meter hurdles two seconds faster than Gerald would run the 400-meter hurdles.  But Gerald changed his diet, which improved his speed by 10%.  If Lee runs the 400-meter hurdles in 38 seconds, how fast can Gerald, with his improved diet, run the 400-meter hurdles, in seconds?

content:
```
Lee at 38 sec is 2 sec faster than Gerald's old time so Gerald's
old time was $38+2=40$ sec. At $10\%$ greater speed
$T_{new}=T_{old}/1.1 = 40/1.1 =400/11\approx 36.4$ sec, $0.9T_{old}=36$ sec
for a $10\%$ reduction in time. The final answer is 36.36 seconds, 36 seconds with a 10% time reduction
```

### arithmetic (1)

- doc 12: gold 13, extracted 12, gen tokens 837, finish stop, repeat5 0.00: extracted 12 != gold 13; false step: '1.5 - 3 = 7.5' (actually -1.5)

**doc 12** (gold 13): Carlos is planting a lemon tree. The tree will cost $90 to plant. Each year it will grow 7 lemons, which he can sell for $1.5 each. It costs $3 a year to water and feed the tree. How many years will it take before he starts earning money on the lemon tree?

content:
```
Annual net is $7 \times 1.5 - 3 = 7.5$. To recover the $\$90$ planting cost requires $90 / 7.5 = 12$ years to break even with the first net profit in the next year. The final answer is 12
```

