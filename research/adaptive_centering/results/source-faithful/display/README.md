# Scatter display receipts

These tables describe the static figures in `docs/src/assets/adaptive-hsgp/`.
They record the exact axis limits, input-point counts and jointly visible
counts for each facet. `panel=1` is the mean GP and `panel=2` is the log-SD GP
in the pair figures. Row and column indices follow the rendered facets.

Every axis uses its own empirical 1.25th–98.75th percentile range. This is a
central 97.5% **marginal** display window, not a joint 97.5% probability region.
Points outside the window are not removed or clamped; the underlying AoV
specification, posterior summaries, gradient calculations and loss curves
remain unchanged. Pair facets use all 10,000 draws; gradient facets use the
1,000 evenly selected display draws described on the page.

The renderer is `research/adaptive_centering/plot_results.jl` plus
`plots/gradient_preview.jl`; both use `plots/scatter_display.jl` only for
axis limits. The gradient input table's SHA-256 is
`21e01d829d11407bf653517f1da8a386a59ce923e7f13940d6e7017d36d0dbb4`.
All fits and fit-cost tables are the source-configuration results in the
parent directory; no sampler is invoked by these renderers.

Pair markers have size 8 and gradient markers size 12. Pair and gradient
facets identify bases by row, without a redundant legend. The centeredness
and loss figures have one shared categorical legend; the basis/spectrum
illustration keeps its two distinct legends.

Verification: `test/scatter_display.jl` passes 22 checks, including exact
unchanged specifications and point counts after the axis zoom;
`test/plotting_diagnostics.jl` passes 38 native-AoV specification checks.
All 15 regenerated figures were inspected, with representative pair and
gradient figures additionally checked at a 688-pixel browser content width.

`saved_fit_bindings.toml` records an independent full-table check with
`validate_plot_bindings.jl`: all 240,000 pair-input rows and all 240,000
gradient-coordinate rows match their named saved fit. Physical basis weights
are checked against the source spectral formula, and online display coordinates
against the saved learned centeredness. The validator does not call BRM's
extraction/transport helpers. It checks coordinate provenance, not gradient
accuracy; the finite-difference receipt in the parent directory covers the
gradient values separately.
