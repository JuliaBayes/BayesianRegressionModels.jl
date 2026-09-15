| Method | Total gradients | Sampling efficiency | Total efficiency |
|---|---:|---:|---:|
| brms NCP · Native Stan | 2,122,482 | 1× | 1× |
| brms NCP · WHMC | 511,283 | 1.9× | 2.46× |
| brms CP · WHMC | 182,890 | 1× | 1.14× |
| brms S2Z CP · Native Stan | 245,020 | 160× | 59.4× |
| brms S2Z CP · WHMC | 104,514 | 152× | 148× |
| brms S2Z NCP · Native Stan | 1,442,653 | 1.19× | 1.19× |
| brms S2Z NCP · WHMC | 187,780 | 3.44× | 4.35× |
| brms S2Z auto · Native Stan | 171,127 | 218× | 105× |
| brms S2Z auto · WHMC | 97,516 | 209× | 188× |
| BRM total NCP · WHMC | 363,592 | 0.924× | 1.23× |
| BRM total CP · WHMC | 291,873 | 229× | 37× |
| BRM total post-hoc position · WHMC | 413,396 | 182× | 25× |
| BRM total post-hoc gradient · WHMC | 410,621 | 408× | 42.3× |
| BRM total online position · WHMC | 38,606 | 400× | 442× |
| BRM total online gradient · WHMC | 39,224 | 338× | 376× |
