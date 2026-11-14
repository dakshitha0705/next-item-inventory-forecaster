# Evaluation: next-item top-N accuracy (200 held-out cases)

Each case is a held-out basket with one item hidden; a method gets a hit at N if the hidden item is in its first N suggestions. Model: `models/forecaster`, 20 samples per prediction, 3.84 s per case on a laptop CPU (12.8 min in all).

**Note:** 200 cases were used to save time (a 400-case run was stopped at ~2.8 s per case); with 200 cases one case moves a hit rate by 0.005.

| method | hits@1 | hits@5 | hits@10 |
|---|---|---|---|
| model | 0.025 | 0.060 | 0.075 |
| model only | 0.025 | 0.060 | 0.075 |
| frequency baseline | 0.005 | 0.080 | 0.120 |
| item-to-item counting | 0.135 | 0.170 | 0.200 |

On average 100% of the model's 10 suggestions come from the model itself; the rest are filled from the baseline.

Paired differences with a bootstrap 95% range (resampling cases):

| comparison | difference | 95% range |
|---|---|---|
| model - frequency baseline, hits@1 | +0.020 | +0.000 to +0.045 |
| model - frequency baseline, hits@5 | -0.020 | -0.060 to +0.020 |
| model - frequency baseline, hits@10 | -0.045 | -0.090 to +0.005 |
| model - item-to-item counting, hits@1 | -0.110 | -0.155 to -0.065 |
| model - item-to-item counting, hits@5 | -0.110 | -0.160 to -0.060 |
| model - item-to-item counting, hits@10 | -0.125 | -0.180 to -0.070 |
