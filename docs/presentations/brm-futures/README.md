# Simple formulas, room for custom models

This editable Quarto RevealJS deck explains how BayesianRegressionModels.jl
combines concise regression formulas with custom scientific calculations.
The main example uses formulas for subject effects and a custom block for a
drug-concentration curve. It is followed by larger structural examples, an
authoring comparison with brms and hand-written models, the corrected HSGP
centering study, prediction, and a possible future Reactant/XLA GPU path.

Execution coverage and verification limits are stated next to the examples.
The six-slide appendix supplies the complete PK example, current execution
support, generated-code references, scientific evidence, and sources. The
deck does not claim universal backend parity or propose an inference-method
research agenda.

`source-examples.lua` reads the Gaussian and population-PK declarations from
`docs/src/feature-atlas.md` at render time. Two PK slides extract its model body
and data; appendix B shows the full declaration. These are source excerpts,
not manually invented generated Stan/Turing programs. The normal documentation
build executes the feature atlas and generates its backend views.

The visual system follows the StanCon 2026 StanBlocks.jl presentation at
StanBlocks.jl revision `c0b5b9197e2d06cf284f1990db024c37ed2b9d47`:
Reveal `simple`, a 1600×900 canvas, warm paper, Avenir/Inter typography, Stan
maroon rules, muted blue/gold accents, and flat cards. This deck retains its
example-specific support labels.

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

The formulas/custom-code rewrite was checked against BRM revision
`8dfe41253af3043482cb3270cf513b50a1de5437`. Adaptive-centering claims were
integrated from corrected reviewed revision
`ea27c18ed2517e3061f441971f561235f48cceb8`, canonically integrated in
`2db645e5e9bdc59465ecdaa18d039460f682cda2`. The executable case-study source is
<https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/docs/src/adaptive-centering.md>.

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

## Possible Julia-to-XLA GPU path

Primary documentation checked on 2026-09-14:

- Reactant's Julia-to-MLIR/XLA compilation and device support:
  <https://enzymead.github.io/Reactant.jl/stable/>
- Probabilistic programming, including a custom-log-density interface:
  <https://enzymead.github.io/Reactant.jl/stable/tutorials/probprog/>
- Tracing and control-flow requirements:
  <https://enzymead.github.io/Reactant.jl/stable/tutorials/control-flow>
- brms already exposes Stan/OpenCL for eligible operations:
  <https://paulbuerkner.com/brms/reference/opencl.html>

This establishes a compiler route to investigate. It does not establish that
BRM's current Julia/Turing models already compile through Reactant or run faster
on a GPU. The user's suggested Wren connection could not be identified in the
public primary sources reviewed; no specific Wren-to-Reactant relationship is
asserted in the presentation.

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
