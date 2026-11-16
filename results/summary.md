Final test accuracy (%) after 20 epochs, mean ± std over seeds.

| Model | Dropout | Adam | AdaMax | Adam - AdaMax |
|---|---|---|---|---|
| Logistic regression | none | 92.73 ± 0.04 | 92.44 ± 0.04 | +0.29 ± 0.06 |
| MLP | none | 98.22 ± 0.22 | 98.31 ± 0.03 | -0.08 ± 0.23 |
| MLP | 0.2 | 98.08 ± 0.20 | 98.35 ± 0.05 | -0.27 ± 0.19 |
| CNN | none | 99.09 ± 0.15 | 99.17 ± 0.08 | -0.08 ± 0.18 |
| CNN | 0.2 | 99.23 ± 0.10 | 99.23 ± 0.11 | 0.00 ± 0.09 |
