| Method | Total gradients | Sampling efficiency | Total efficiency | Divergences |
|---|---:|---:|---:|---:|
| RBesT 1.11 NCP (its control) · Native Stan | 328,781 | 1× | 1× | 0 |
| RBesT 1.11 CP (its control) · Native Stan | 323,012 | 0.0336× | 0.0319× | 533 |
| RBesT 1.11 NCP (Stan defaults) · Native Stan | 145,801 | 2.44× | 2.69× | 4 |
| RBesT 1.11 CP (Stan defaults) · Native Stan | 138,693 | 0.0627× | 0.0702× | 670 |
| RBesT PR64 S2Z NCP · Native Stan | 262,110 | 1.63× | 1.68× | 0 |
| RBesT PR64 S2Z CP · Native Stan | 182,173 | 1.75× | 1.73× | 9 |
| BRM ordinary NCP · WHMC | 128,794 | 2.16× | 2.59× | 1 |
| BRM ordinary CP · WHMC | 94,686 | 1.08× | 1.29× | 4 |
| BRM ordinary post-hoc position · WHMC | 219,838 | 2.88× | 1.39× | 0 |
| BRM ordinary post-hoc gradient · WHMC | 209,470 | 3.56× | 1.64× | 1 |
| BRM ordinary online position · WHMC | 83,667 | 3.14× | 3.74× | 14 |
| BRM ordinary online gradient · WHMC | 121,115 | 2.45× | 2.93× | 5 |
| BRM total NCP · WHMC | 133,444 | 0.324× | 0.383× | 6 |
| BRM total CP · WHMC | 87,205 | 2.49× | 3× | 61 |
| BRM total post-hoc position · WHMC | 204,019 | 2.3× | 0.919× | 40 |
| BRM total post-hoc gradient · WHMC | 238,614 | 2.34× | 1.23× | 7 |
| BRM total online position · WHMC | 69,353 | 0.226× | 0.269× | 79 |
| BRM total online gradient · WHMC | 89,144 | 1.3× | 1.55× | 19 |
