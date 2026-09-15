| Model | Sampler | Total gradients | Relative sampling efficiency | Relative total efficiency |
|---|---|---:|---:|---:|
| brms NCP | Native Stan | 1063829 | 1× | 1× |
| brms NCP | WHMC | 274051 | 1.43× | 1.91× |
| brms CP | WHMC | 80373 | 2.54× | 3.68× |
| brms S2Z CP | Native Stan | 319693 | 135× | 40× |
| brms S2Z CP | WHMC | 188667 | 137× | 66× |
| brms S2Z NCP | Native Stan | 771035 | 1.17× | 1.22× |
| brms S2Z NCP | WHMC | 123267 | 3.19× | 4.48× |
| brms S2Z auto | Native Stan | 150012 | 187× | 62.6× |
| brms S2Z auto | WHMC | 64875 | 173× | 194× |
| Our totals CP | WHMC | 54587 | 131× | 195× |
| Our totals NCP | WHMC | 134652 | 3.03× | 4.46× |
| Our totals ACP | WHMC | 168932 | 258× | 71.2× |
