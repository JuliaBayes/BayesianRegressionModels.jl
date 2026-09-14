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

# Run the compiled-StanBlocks online HSGP adaptation showcase.
BRM_ADAPTIVE_ONLINE=1 BRM_ADAPTIVE_K=8 \
  BRM_ADAPTIVE_DRAWS=20 BRM_ADAPTIVE_EVALS=120 \
  BRM_ADAPTIVE_OUTPUT="$PWD/research/adaptive_centering/results" \
  julia --startup-file=no --project=test research/adaptive_centering/reproduce.jl

# Run the matching native-Turing online HSGP adaptation showcase.
BRM_ADAPTIVE_TURING_ONLINE=1 BRM_ADAPTIVE_K=8 \
  BRM_ADAPTIVE_DRAWS=20 BRM_ADAPTIVE_EVALS=120 \
  BRM_ADAPTIVE_OUTPUT="$PWD/research/adaptive_centering/results" \
  julia --startup-file=no --project=test research/adaptive_centering/reproduce.jl

# Bounded multi-chain comparison used by the documentation artifact.
BRM_ADAPTIVE_RUNTIME=1 BRM_ADAPTIVE_K=8 \
  BRM_ADAPTIVE_CHAINS=4 BRM_ADAPTIVE_DRAWS=75 BRM_ADAPTIVE_EVALS=350 \
  BRM_ADAPTIVE_OUTPUT="$PWD/research/adaptive_centering/results" \
  julia --startup-file=no --project=test research/adaptive_centering/reproduce.jl
```

Omit `BRM_ADAPTIVE_K` for the source-faithful 20-function basis. The bounded
artifact reduces the truncation count and Monte Carlo budget, uses one fixed
warmup window, and starts from explicit finite physical values instead of the
source run's Pathfinder initialization. Model, data, priors, pilot rule, and
fixed seeds are unchanged. Every run uses four independent `Xoshiro` chains.
The output records maximum classical R-hat,
minimum initial-positive-sequence ESS, divergences, gradient evaluations, and
wall time for all three geometries on each backend.

The checked bounded result in `results/` does **not** converge. Centered and
adaptive coordinates remove the noncentered run's divergences, but all maximum
R-hat values exceed 2 and all minimum ESS values are about 2–3. Its curve and
timing outputs are smoke-test evidence only, not a performance ranking or a
scientific posterior. A substantive analysis should restore `k=20`, the
ordinary adaptive warmup/Pathfinder path, many more draws, and convergence
criteria chosen before looking at results.

The six-fit comparison is **offline pilot/refit adaptation**. It chooses a
fixed formula/data geometry before the second fit. The separate
`BRM_ADAPTIVE_ONLINE=1` and `BRM_ADAPTIVE_TURING_ONLINE=1` runs exercise
WarmupHMC's online nonlinear reparameterizer on the same two-HSGP model. Each
learns one coordinate per basis weight inside warmup and preserves its native
target via the same exact transform and Jacobian. StanBlocks writes
`results/online_centeredness.tsv`; Turing writes
`results/online_turing_centeredness.tsv`. Neither reuses pilot draws nor
rewrites the BRM formula.

The committed bounded online receipt retained 20 draws with a 120-evaluation
warmup budget, reported zero divergences, and moved all 16 same-axis HSGP
cells away from their initial `c=0`. Those settings verify the executable
transform; they are not a convergence or efficiency study.

The corresponding native-Turing receipt also retained 20 draws with the same
120-evaluation budget and reported zero divergences. It learned mean
centeredness `[1.0,1.0,1.0,0.5,1.0,0.8,1.0,0.4]` and log-scale centeredness
`[0.9,1.0,0.6,0.5,0.2,0.0,0.0,0.0]`. The deterministic finite initialization
is part of the reproduction and is not a pilot or posterior draw.
