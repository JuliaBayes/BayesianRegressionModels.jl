# Adaptive HSGP centering reproduction

This directory translates Generable's heteroscedastic motorcycle HSGP case
study into one BRM formula that lowers to both StanBlocks and Turing. The two
predictors are

```text
mu ~ HSGP(time)
log(sigma) ~ HSGP(time)
y ~ Normal(mu, sigma)
```

and each HSGP basis weight can independently interpolate between the
noncentered (`c=0`) and centered (`c=1`) coordinates. The pilot is sampled in
the noncentered coordinates, `select_hsgp_centeredness` searches the original
`0:0.01:1` grid, and the selected vector is fixed as data in the refit. The
posterior target is unchanged; only its coordinates change.

## Provenance

- Article: <https://www.generable.com/post/hsgp-reparam>
- Companion repository: <https://github.com/generable/public-materials/tree/0d00b8535e2c20c49017d03c7b060940eb8e7041/blog/hsgp-reparam>
- Exact companion revision: `0d00b8535e2c20c49017d03c7b060940eb8e7041`
- Data repository revision: `1dcc2bf5f955cc1224a3e1307256e1fe86b68dae`
- Raw `MASS::mcycle` CSV SHA-256: `b89a1e4eb0391a982b32be3e378df00e8593ff9971e9425e9c5d7929b74f9801`

The committed CSV is the 133-row `MASS::mcycle` table used by the source via
RDatasets. `reproduce.jl` follows its executable preprocessing: acceleration is
divided by its sample standard deviation, while time is min–max mapped to
`[-1,1]` inside the HSGP and evaluated in the `[-1.5,1.5]` boundary. It retains
the source's 20 basis functions and four `Normal(0,4)` log-scale priors by
default.

## Run

After resolving the repository's `test` environment:

```sh
# Compile both lowerings and evaluate their Enzyme gradients.
julia --startup-file=no --project=test research/adaptive_centering/reproduce.jl

# Bounded multi-chain comparison used by the documentation artifact.
BRM_ADAPTIVE_RUNTIME=1 BRM_ADAPTIVE_K=8 \
  BRM_ADAPTIVE_CHAINS=4 BRM_ADAPTIVE_DRAWS=75 BRM_ADAPTIVE_EVALS=350 \
  BRM_ADAPTIVE_OUTPUT="$PWD/research/adaptive_centering/results" \
  julia --startup-file=no --project=test research/adaptive_centering/reproduce.jl
```

Omit `BRM_ADAPTIVE_K` for the source-faithful 20-function basis. The bounded
artifact reduces only the truncation count and Monte Carlo budget; model,
data, priors, pilot rule, and fixed seeds are unchanged. Every run uses four
independent `Xoshiro` chains. The output records maximum classical R-hat,
minimum initial-positive-sequence ESS, divergences, gradient evaluations, and
wall time for all three geometries on each backend.

This is **offline pilot/refit adaptation**. It chooses a fixed formula/data
geometry before the second fit. WarmupHMC's online nonlinear reparameterizer is
a different mechanism: it learns inside warmup and does not rewrite the BRM
formula's HSGP coordinates.
