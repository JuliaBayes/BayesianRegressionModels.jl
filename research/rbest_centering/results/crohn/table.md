| Method | Total gradients | Sampling efficiency | Total efficiency | Divergences |
|---|---:|---:|---:|---:|
| RBesT 1.11 NCP (its control) · Native Stan | 466,164 | 1× | 1× | 0 |
| RBesT 1.11 CP (its control) · Native Stan | 446,598 | 0.377× | 0.376× | 53 |
| RBesT 1.11 NCP (Stan defaults) · Native Stan | 224,025 | 1.79× | 2× | 8 |
| RBesT 1.11 CP (Stan defaults) · Native Stan | 163,427 | 1.73× | 1.92× | 124 |
| RBesT PR64 S2Z NCP · Native Stan | 246,616 | 2.33× | 2.29× | 0 |
| RBesT PR64 S2Z CP · Native Stan | 167,317 | 2.49× | 2.41× | 42 |
| BRM ordinary NCP · WHMC | 194,050 | 1.18× | 1.38× | 7 |
| BRM ordinary CP · WHMC | 97,820 | 1.46× | 1.63× | 44 |
| BRM ordinary post-hoc position · WHMC | 315,028 | 3.55× | 1.63× | 9 |
| BRM ordinary post-hoc gradient · WHMC | 292,381 | 3.48× | 1.4× | 12 |
| BRM ordinary online position · WHMC | 138,160 | 2.82× | 3.38× | 0 |
| BRM ordinary online gradient · WHMC | 99,443 | 3.55× | 4.25× | 47 |
| BRM total NCP · WHMC | 387,777 | 0.0844× | 0.0349× | 510 |
| BRM total CP · WHMC | 70,628 | 2.9× | 3.13× | 134 |
| BRM total post-hoc position · WHMC | 460,830 | 3.88× | 0.731× | 42 |
| BRM total post-hoc gradient · WHMC | 464,218 | 4.36× | 0.854× | 59 |
| BRM total online position · WHMC | 69,343 | 3.59× | 4.28× | 38 |
| BRM total online gradient · WHMC | 71,425 | 6.42× | 7.65× | 62 |
