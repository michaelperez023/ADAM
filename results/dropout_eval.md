Final test accuracy (%) of the dropout runs, tested with dropout off and with dropout on (mean of 20 passes over the test set), mean ± std over seeds.

| Model | Dropout | Optimizer | Dropout off | Dropout on | Off - on |
|---|---|---|---|---|---|
| MLP | 0.2 | Adam | 98.08 ± 0.20 | 98.04 ± 0.20 | +0.04 ± 0.01 |
| MLP | 0.2 | AdaMax | 98.35 ± 0.05 | 98.32 ± 0.05 | +0.03 ± 0.03 |
| CNN | 0.2 | Adam | 99.23 ± 0.10 | 99.11 ± 0.12 | +0.12 ± 0.02 |
| CNN | 0.2 | AdaMax | 99.23 ± 0.11 | 99.13 ± 0.10 | +0.10 ± 0.03 |
