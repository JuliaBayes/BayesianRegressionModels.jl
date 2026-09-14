# BRM future-capability deck

This directory contains the collaborator-facing Quarto RevealJS deck comparing
the future capability enabled by brms and BayesianRegressionModels.jl. The deck
uses three explicit claim tiers:

- **Demonstrated** — executable in the cited repository revision or reported by
  a named primary source.
- **Architecture-enabled** — a concrete extension seam exists, but design,
  implementation, validation, and maintenance remain.
- **Research question** — a direction to investigate, not a promise.

The visible backend legend is independent of the claim tier: `Both`,
`StanBlocks only`, `Turing only`, and `Planned / research` describe current
execution coverage.

The visual system follows the StanCon 2026 StanBlocks.jl presentation at
StanBlocks.jl revision `c0b5b9197e2d06cf284f1990db024c37ed2b9d47`:
Reveal `simple`, a 1600×900 canvas, warm paper, Avenir/Inter typography, Stan
maroon rules, muted blue/gold accents, and flat cards. This deck retains its
own evidence tiers and backend-status legend.

## Render

From the repository root:

```sh
docs/presentations/brm-futures/render.sh
```

The script renders one self-contained RevealJS file, verifies a pinned MathJax
3 runtime, freezes its SVG equations into the document, removes that build-time
runtime, and prints the result to a landscape PDF with Chrome's RevealJS print
stylesheet. It rejects external runtime resources and writes the public
artifacts to `docs/src/public/decks/` so the normal VitePress build publishes
them unchanged.

The HTML includes speaker notes (`S` in a non-embedded presentation; support
can be limited by RevealJS self-contained mode). The PDF intentionally omits
notes. Source, notes, and references remain in `brm-futures.qmd` and this file.

## BRM provenance

The independent deck was drafted against BRM revision
`bca7093487eff9d7c55ea200b256bd7655e15f52`. Adaptive-centering claims were
integrated from corrected reviewed revision
`ea27c18ed2517e3061f441971f561235f48cceb8`, canonically integrated in
`2db645e5e9bdc59465ecdaa18d039460f682cda2`. The live executable case study is
<https://nsiccha.github.io/BayesianRegressionModels.jl/dev/adaptive-centering.html>.

The adaptive case is derived from:

- Generable, “HSGP Reparameterization”:
  <https://www.generable.com/post/hsgp-reparam>
- Companion materials, immutable revision
  `0d00b8535e2c20c49017d03c7b060940eb8e7041`:
  <https://github.com/generable/public-materials/tree/0d00b8535e2c20c49017d03c7b060940eb8e7041/blog/hsgp-reparam>
- `MASS::mcycle` data revision
  `1dcc2bf5f955cc1224a3e1307256e1fe86b68dae`, raw CSV SHA-256
  `b89a1e4eb0391a982b32be3e378df00e8593ff9971e9425e9c5d7929b74f9801`.

The adaptive panels use the committed source-faithful artifacts under
`research/adaptive_centering/results/source-faithful/`: all 133 observations,
two 20-frequency HSGPs, source-equivalent log-hyperpriors, `Xoshiro(1)`,
10,000 retained draws per fit, and unchanged WarmupHMC defaults. The source
reproduction has a noncentered pilot and a fresh selected-partial refit; the
online StanBlocks fit is a separate extension. Their divergence counts are
34, 16, and 0. Each result is one chain, so split R-hat is a within-chain
diagnostic rather than evidence that independent chains agree, and the counts
are not a general efficiency guarantee. No corrected Turing samples exist:
sampling remains disabled until matched target-value and warmed
gradient-runtime gates pass.

## Primary comparison sources

- Bürkner, P.-C. (2018). “Advanced Bayesian Multilevel Modeling with the R
  Package brms.” *The R Journal* 10(1), 395–411.
  <https://paulbuerkner.com/publications/pdf/2018__Buerkner__R_Journal.pdf>
- brms 2.23.2 overview and model-fitting reference:
  <https://paulbuerkner.com/brms/> and
  <https://paulbuerkner.com/brms/reference/brm.html>
- brms formula, custom-family, Stan-variable, and GP references:
  <https://paulbuerkner.com/brms/reference/brmsformula.html>,
  <https://paulbuerkner.com/brms/reference/custom_family.html>,
  <https://paulbuerkner.com/brms/reference/stanvar.html>, and
  <https://paulbuerkner.com/brms/reference/gp.html>
- Riutort-Mayol, G., Bürkner, P.-C., Andersen, M. R., Solin, A., & Vehtari,
  A. (2023). “Practical Hilbert space approximate Bayesian Gaussian processes
  for probabilistic programming.” *Statistics and Computing* 33, 17.
  <https://doi.org/10.1007/s11222-022-10167-2>
- Papaspiliopoulos, O., Roberts, G. O., & Sköld, M. (2007). “A General
  Framework for the Parametrization of Hierarchical Models.” *Statistical
  Science* 22(1), 59–73. <https://doi.org/10.1214/088342307000000014>
- Gorinova, M. I., Moore, D., & Hoffman, M. D. (2020). “Automatic
  Reparameterisation of Probabilistic Programs.” *ICML 2020*.
  <https://proceedings.mlr.press/v119/gorinova20a.html>
- Stan User’s Guide, “Reparameterization and Change of Variables”:
  <https://mc-stan.org/docs/stan-users-guide/reparameterization.html>
- Quarto RevealJS and print-to-PDF documentation:
  <https://quarto.org/docs/presentations/revealjs/> and
  <https://quarto.org/docs/presentations/revealjs/presenting.html#print-to-pdf>

All comparison language is the deck authors' synthesis. Cited authors and
prospective guests have not reviewed or endorsed the deck.
