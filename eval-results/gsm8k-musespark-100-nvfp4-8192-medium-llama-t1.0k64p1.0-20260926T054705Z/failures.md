# GSM8K failure analysis: `/filestore/srcs/tpu-megakernels/eval-results/gsm8k-musespark-100-nvfp4-8192-medium-llama-t1.0k64p1.0-20260926T054705Z`

weights `/filestore/weights/muse-spark-tp8-nvfp4`, max_gen_toks 8192, reasoning_effort medium, n = 100; token counts / finish reasons from server.log; reasoning transcripts NOT available (content channel only).

| metric | value |
|---|---|
| flexible-extract accuracy | 98.0% (98/100) |
| strict-match accuracy | 98.0% |
| empty answers (no content) | 0 |
| finish = length (budget hit) | 0 |
| generated tokens: mean / median / max | 561.6 / 521.5 / 2411 |
| generated tokens, correct problems: mean / median / max | 535.0 / 521.0 / 1199 |
| generated tokens, wrong-but-answered: mean / median / max | 1861.5 / 1861.5 / 2411 |
| repeated-5-gram rate, correct (mean / max) | 0.000 / 0.023 |
| repeated-5-gram rate, truncated (mean / max) | - / - |
| truncated transcripts that loop (rate > 0.3) | 0 / 0 |
| accuracy excluding truncations | 98.0% |
| accuracy with extraction fixed | 100.0% |
| accuracy excluding truncations and with extraction fixed | 100.0% |

## Failure classes

| class | count | doc ids |
|---|---:|---|
| truncated | 0 |  |
| extraction | 2 | 37, 87 |
| arithmetic | 0 |  |
| reading | 0 |  |

### extraction (2)

- doc 37: gold 2, extracted 13, gen tokens 2411, finish stop, repeat5 0.00: gold 2 present, extracted 13; last number is not the answer (hedged alternative)
- doc 87: gold 9360, extracted $9,583.20, gen tokens 1312, finish stop, repeat5 0.00: gold 9360 present, extracted 9583.2; last number is not the answer

**doc 37** (gold 2): John plans to sell all his toys and use the money to buy video games. He has 13 lego sets and he sells them for $15 each. He ends up buying 8 video games for $20 each and has $5 left. How many lego sets does he still have?

content:
```
If he actually liquidates the whole stock the stock on hand is
$13-13=0$.
At $15$ a set the $13$ sets would be $13\times 15 =195$.
$8$ games at $20$ is $160$ leaving $195-160 =35$, not $5$.
To have $160+5 =165$ to spend he needs to sell $165/15 =11$ sets,
so $13-11 =2$ sets must remain unsold to make the $5$ figure consistent,
literally $0$ sold out.

The final answer is 2, literally 0 if he sold all 13
```

**doc 87** (gold 9360): A company pays each of its employees $600 in a month. The company has a policy of increasing the salaries of each of its employees by 10% of the initial salary every year for those who've stayed in the company for five years. If Sylvie just clocked 5 years in the company last December, what's her annual salary after three more years of service?

content:
```
Monthly salary is $600$ so annual is $7200$.
10% of initial is $60$ per month $720$ per year.
After $3$ yearly increases $600+3*60 = 780$ per month $7200+3*720 = 9360$ per year compounded would be $600*1.1^3 = 798.60$ per month $9583.20$ per year. The final answer is $9,360 per year $780 per month, $9,583.20 compounded$
```

