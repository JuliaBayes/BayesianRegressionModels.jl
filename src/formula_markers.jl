# Backend-independent formula marker bindings. Backends dispatch on these
# callable identities without owning the public formula vocabulary.

"""
    me(x_obs, sd_x)

brms-style measurement-error predictor marker. `me(x_obs, sd_x)` declares
that the observed `x_obs` has Gaussian measurement error with SD `sd_x`;
the backend allocates a latent `x_true` and emits the observation
likelihood `x_obs ~ Normal(x_true, sd_x)`. Dispatch tag — see `_sb_me`.
"""
function me end

"""
    s(x)

Add a penalized one-dimensional thin-plate regression spline to an `SBBRMI`
linear predictor. The fixed rank-10 basis contains the
unpenalized null space `{1, x}` and eight penalty-whitened range columns whose
shared smoothing standard deviation has a standard half-normal prior.

The public syntax is exactly one finite numeric predictor, `s(x)`. At least ten
unique training values are required; keyword arguments such as `k=` and
`knots=` are not supported. The term owns its complete smooth contribution and
does not receive an additional population coefficient.

Prediction and replay through [`reprocess`](@ref) or [`restan_data`](@ref) use
the frozen training basis by default. This marker is implemented only by the
StanBlocks backend; it is not a deterministic B-spline expansion and is not
available to `VBRMI`.

See [Formula terms](@ref) for an example and a comparison with
`bs(...)` formulas. Dispatch tag — see `_sb_s`.
"""
function s end

"""
    t2(x, z; k=(5, 5), basis=(:cr, :cr), full=false)

Tensor-product smooth marker. The StanBlocks backend builds cubic-regression-
spline margins, separates their null and penalized spaces, and gives the RR,
RN, and NR tensor blocks independent smoothing scales. The current surface is
two-dimensional, supports only `basis=(:cr, :cr)`, and requires `full=false`.
Prediction/replay freezes the training knots and penalty decomposition by
default. See [Formula terms](@ref) for the full contract. Dispatch tag — see
`_sb_t2`.
"""
function t2 end

"""
    ar(time; p=1)

Autoregressive-noise predictor marker. Adds an AR(p) noise process
ordered by `time`. Only `p=1` is supported in the current sbimpl
emitter. Dispatch tag — see `_sb_ar1`.
"""
function ar end

"""
    dar(time; p=1)

Differenced-autoregressive trajectory marker. The StanBlocks backend emits the
weekly path

`x[1] = 0`, `d[t] = beta * d[t-1] + sigma * z[t]`,
`x[t+1] = x[t] + d[t]`,

with `0 <= beta <= 1`, `sigma > 0`, and standardized innovations `z`. It is a
direct predictor summand, so a formula intercept supplies the initial level and
no additional population coefficient multiplies the path. The time values must
be finite and strictly increasing. Only `p=1` is supported. Dispatch tag — see
`_sb_dar1`.
"""
function dar end

"""
    rw(time)

Random-walk trajectory marker. The StanBlocks backend emits the path

`x[1] = 0`, `x[t+1] = x[t] + sigma * z[t]`,

over the sorted distinct values of `time` — i.e. `dar(time)` with the
differences' persistence fixed at zero — with `sigma > 0` and standardized
innovations `z`; each row reads the point of its own time value, so rows may
share a time (several groups per day get one shared walk). It is a direct
predictor summand: a formula intercept supplies the initial level and no
additional population coefficient multiplies the path. `sd(lp, rw(time))`
addresses `sigma`; the term has no persistence, so `ar(lp, rw(time))` is
refused. On replay the grid may gain new times (a forecast). Dispatch tag —
see `_sb_rw1`.
"""
function rw end

"""
    cdar(step; by=group, cor=C)

Grouped, correlated, damped random-walk deviation marker. For every level `g`
of the `by` group column and every distinct value `w` of the `step` column
(sorted), the StanBlocks backend emits

`delta[:, 1] = sigma * L * eta[:, 1]`,
`delta[:, w] = rho * delta[:, w-1] + sigma * sqrt(1 - rho^2) * L * eta[:, w]`,

with `L L' = C` the Cholesky factor of the group correlation (or covariance)
matrix `C` — a `P × P` matrix given as a data field or a literal, `P` the number
of group levels — `0 <= rho <= 1`, `sigma > 0`, and standardized innovations
`eta`. Each row contributes `delta[group(row), step(row)]` directly, with no
additional population coefficient. `sd(lp, cdar(step))` addresses `sigma` and
`ar(lp, cdar(step))` addresses `rho`. On replay the group levels and `C` are
frozen from the fit while the step grid may grow (a forecast). Dispatch tag —
see `_sb_cdar`.
"""
function cdar end

"""
    Horseshoe

Carvalho-Polson-Scott horseshoe shrinkage prior marker. Use as a prior
on a coefficient:

```julia
coef ~ Horseshoe()
coef ~ Horseshoe(local_scale=0.5, global_scale=0.1)
```

`local_scale` and `global_scale` are positive finite formula constants and
default to one. sbimpl emits the standard reparameterised hierarchy
`beta = raw * lambda * tau`. Each scalar call owns its own `tau`; "global"
means global only within that call, not shared across several coefficients.
Marker struct only — the `@brm` parser never constructs an instance.
"""
struct Horseshoe end

"""
    @plate for i in eachindex(per_subject_axis)
        pred[i] = cell(data[i], per_subject_lp[i])
    end

General group-local submodel surface: broadcast an inline cell over the groups
of the linear predictors it reads.

The per-subject LP formulas own the population effects, covariates, links and
random-effect buckets — they are ordinary `@brm` formulas declared in the same
block. `@plate for` derives ONE shared grouping from those LPs and evaluates the
body once per group. An indexed read is that group's slice; an indexed LHS is a
model-scope output, while a bare LHS is local to the cell:

```julia
log_CL ~ 1 + weight + (1 | p | subject)
log_Vc ~ 1 + (1 | p | subject)

@plate for i in eachindex(log_CL)
    CL = exp(log_CL[i])
    Vc = exp(log_Vc[i])
    conc[i] = pk_cell(t_obs[i], dose[i], CL, Vc)
end
```

When the cell's last statement is its final indexed assignment, as `conc[i] = ...`
is here, that right-hand side is the cell's return value and the plate stores it
once. An indexed output that later cell statements read is bound as a cell value
first and then returned.

A cell needs no indexed output. When its only indexed statements are
observations (`dv[i] ~ Normal(cp, sigma)` after a bare `cp = ...`), it has no
collected result: `cp` stays a cell local and nothing is stored at model scope.
Both backends build such a cell.

A plate over long-format rows names its grouping in the header:
`@plate for p in groups(g)` has one cell per distinct label of the data column
`g`, in sorted order, and each read `x[p]` is `x`'s values on group `p`'s rows
(a response, a data column, or a predictor with one value per row);
`groups(g; order = t)` takes each group's rows in ascending `t`. Values shared
by every row are read bare. Such a cell observes, and collects no result
(`loc[p] = …`): nothing outside it could tell which group a collected value
belongs to.

A cell may hold a `@scan begin … end` recurrence, one per cell: seeds by literal
index, then one `for` loop of steps reading earlier entries at a fixed lag. A
carried array the scan writes (`h[t] ~ Normal(phi * h[t - 1], s)`) is a latent
state per cell; observing the cell's own response entry by entry
(`y[i][1] ~ …`, `y[i][t] ~ Normal(… y[i][t - 1] …)`) gives each entry's
conditional density, and with `y` omitted the block draws whole trajectories.
[`ar_errors`](@ref) is shorthand for one such recurrence.

Indexed inputs split by kind. A raw data column on the plate's own
one-row-per-subject frame is gathered into a per-group slice. A column or
linear predictor living on a DIFFERENT frame — a dose-event table, say — is
grouped by a model-scope [`ragged`](@ref) statement and still read as `x[i]`.
A per-subject linear predictor is passed as that group's scalar value. One
with random effects over several groupings — Kreise nested in Länder, `(1 |
land | land) + (1 | kreis | kreis)` — is read on the grouping whose column lists
one unique label per row; the others are ordinary terms of its value. One
without a random effect defines no grouping of its own: it is read on the
subject frame a random-effect predictor defines — or, in a panel with no random
effects, the frame of the pre-grouped columns — so its data columns must have
one row per subject.

`@brm` lowers this annotated-loop surface to the shared kernel IR, so SBBRMI and
RKBRMI use the same grouping, ragged, observation, and omitted-outcome paths.
`kernel(...) do` is deprecated and emits a deprecation warning when a model is
constructed. It remains a temporary source-compatibility spelling while known
consumers migrate; new code should use `@plate for`. The much older `model=` /
`obs=` spelling and its anonymous `n_eta` block remain removed. A name such as
`eta_CL` is merely a user-chosen ordinary linear-predictor name—there is no
kernel-owned eta vector or positional eta-index contract in the current API.
"""
function kernel end

"""
    ragged(x, group)
    ragged(x, group => cells)
    ragged(idx, group; into=target)

Group a FLAT secondary row axis by a raw data column that names the subject of
every row. The marker has two formula positions:

- As a model-scope statement, `ragged(x, group)` declares that `@plate for`
  cells read `x[i]` as cell `i`'s ragged per-subject vector. `x` may be a flat
  data column or an event-axis linear predictor. The declaration emits nothing
  itself; it may precede or follow the cells, a name is grouped at most once,
  and a declaration no cell reads is refused. The deprecated in-body read
  `ragged(x, group)[i]` and the compatibility `kernel(...)` positional lower
  to exactly the same kernel argument.
- As an observation LHS, `ragged(y, group) ~ Family(pred, ...)` groups a flat
  response before applying the top-level likelihood. The referenced
  `kernel(...)` result supplies the authoritative subject row order, so labels
  are joined rather than sorted or inferred from first occurrence. The emitted
  observation keeps the logical name `y` and therefore uses StanBlocks' normal
  top-level ragged outputs: flat `y_gen`, group-aggregate `y_likelihood`, and
  descriptor `segments`. This formula-boundary grouping is SBBRMI/sbimpl-only.

For the kernel-positional form, `x` lives on some frame other than the kernel's
one-row-per-subject frame — a dose-event table, say — and `group` names, for
every row of that frame, which subject it belongs to. The cell receives `x` as
a RAGGED per-subject vector.

`x` may be either a linear predictor declared in the same `@brm` block, or a raw
flat data column. It is the same grouping either way:

```julia
log_F  ~ 1 + vessel + mo(diet) + hsgp(log_dose)     # rows = dose events
log_CL ~ 1 + weight + (1 | p | subject)             # rows = subjects
ragged(dose_amount, dose_subject)                   # a flat data column
ragged(log_F, dose_subject)                         # a linear predictor

@plate for i in eachindex(log_CL)
    effective_dose = dose_amount[i] .* exp.(log_F[i])
    pred[i] = pk_cell(t_obs[i], effective_dose, log_CL[i])
    dv[i] ~ normal(pred[i], sigma)
end
```

Only the REALIZATION differs, and the difference is forced: a linear predictor
is a Stan parameter, so it cannot be gathered on the Julia side — the plate
takes an index column and the cell fancy-indexes the unsliced predictor
(`log_F[rows]`). A data column is Julia data, so it is gathered directly into a
`Vector{Vector{T}}`, which StanBlocks ingests as a ragged column natively —
exactly what a hand-prepared per-subject view would have been. Wrapping a column
does not consume the flat original: a term that names it on its own axis
(`hsgp(log_dose)` above) still sees it.

When no per-subject linear predictor carries a random effect, nothing labels
the cells for `group` to join against. Name the cell label column then:
`ragged(x, group => cells)`, where `cells` lists each cell's label once, on the
cells' own frame. The plate has one cell per row of `cells`, in that row order,
and every per-cell data column belongs to that frame. A district total observes
the summed rate of the facilities in its district:

```julia
log_lambda ~ 1 + (1 | fac | facility)               # rows = facilities
ragged(log_lambda, fac_district => district)        # rows = districts

@plate for d in eachindex(Y)                        # Y: one total per district
    Y[d] ~ Poisson(sum(exp.(log_lambda[d])))
end
```

Rows join their cell by label, so the totals' frame may list its districts in
any order. Every `ragged(...)` input of one plate names the same `cells`; their
`group` columns may differ. Under a random-effect predictor the cells are its
grouping, so `cells` may be named only as that same column.

A plate with no random-effect predictor and no `cells` is refused: matching rows
to cells by position cannot be checked against a per-cell column, which would
then bind to the wrong cells silently. A plate over long-format rows with
nothing per cell names its grouping in its header instead,
`@plate for p in groups(group; order = time)` (see [`kernel`](@ref)): one cell
per distinct label of `group`, each read `x[p]` the group's rows in `time`
order.

Why the grouping is an explicit argument rather than derived: an ordinary
per-subject LP is grouped by the random-effect grouping that labels its rows
(`_sb_kernel_lp_buckets`), but a secondary-axis population LP like
the one above has no random-effect term at all, and a raw column carries no grouping
whatsoever. The axis has to be declared, and `group` is that declaration.

ONE VALUE PER ROW OF `x`'s OWN FRAME — no expansion happens anywhere. If the
event table stores a compact schedule (one row per dose OP, carrying an interval
and a count) then `x` has one value per OP. Constructing `ragged(x, group)`
requires one `group` key per row of `x` and joins those keys to the kernel's
outer subject labels. That is local validation of the grouping operation, not
an inner-shape contract between cell arguments. Kernel compares no totals or
per-subject inner lengths across positionals; once each positional supplies one
outer cell per subject, relationships among the values inside a cell belong to
the cell body.

INDEX COLUMNS INTO ANOTHER AXIS. `ragged(idx, group; into=target)` declares a
data column of 1-based row numbers on the flat axis whose subjects `target`
labels — an event schedule's global read number per observation, say. Each
value becomes its position among its own subject's rows of `target`, in flat
order (the order `ragged(x, target)` gathers), so the cell indexes its own
subject's slice of that axis: `conc[obs_read[i]]` with `conc` computed per
local read. `0` stays `0` ("no row"). An index outside `target`'s rows, or
into another subject's rows, is refused; it would otherwise read the wrong
subject silently. `into=` is model-scope only and data-only: an observation
LHS takes no keywords.

Dispatch tag only — the model-scope statement is consumed by the `@plate for`
rewrite (`_brm_plate_kernel`, macro.jl), kernel-argument lowering lives in
`_sb_kernel_doblock!` (sbimpl), and observation-LHS lowering lives in
`_sb_sampling!`.
"""
ragged

# `order=` is the kernel positional a `@plate for p in groups(g; order = t)`
# header lowers to; it is not written on `ragged` itself.
_check_term_kwargs(::typeof(ragged), kwargs) =
    all(in((:into, :order)), keys(kwargs)) || error(
    "@brm: ragged(...) takes only the `into=` keyword, got $(keys(kwargs)). The ",
    "spelling is `ragged(<linear predictor or flat column>, <grouping column>)`, ",
    "`ragged(x, group => cells)` with a cell label column `cells`, or ",
    "`ragged(<index column>, <grouping column>; into=<grouping column of the ",
    "indexed axis>)`.")

"""
    y ~ ar_errors(Normal(mu, sigma), phi; time = t, by = g)

Gaussian observations whose residuals follow an AR(1) process within each group
of `g`, in the order of the integer time column `t`:
`y[t] - mu[t] = phi^k * (y[t-k] - mu[t-k]) + e[t]`, where `k` is the number of
time steps since the group's previous row.

`ar_errors` is shorthand for the recurrence it states. `@brm` replaces the
statement by this `@plate` / `@scan` source before building the model, and
`show` and the descriptor print that expansion:

```julia
@plate for p in groups(g; order = t)    # one cell per group, rows in `t` order
    @scan begin
        y[p][1] ~ Normal(mu[p][1], sigma / sqrt(1 - phi ^ 2))
        for s in 2:length(t[p])
            k = t[p][s] - t[p][s - 1]
            y[p][s] ~ Normal(mu[p][s] + phi ^ k * (y[p][s - 1] - mu[p][s - 1]),
                             sigma * sqrt((1 - phi ^ (2 * k)) / (1 - phi ^ 2)))
        end
    end
end
```

- `sigma` is the innovation SD. The first row of each group comes from the
  stationary law `Normal(mu, sigma / sqrt(1 - phi^2))`, so `|phi| < 1` is
  required. This is the exact AR(1) likelihood (brms' `ar(cov = TRUE)`).
- A gap of `k` steps uses `phi^k` and innovation variance
  `sigma^2 (1 - phi^(2k)) / (1 - phi^2)`: the exact density of the process
  observed only on the recorded steps. Without gaps this is the plain recursion.
- Rows may come in any order; each group's rows are taken in `t` order. Within
  a group every time must be a whole number and occur once.
- `mu`, `sigma` and `phi` are names or numbers. A name a formula declares
  (`mu ~ 1 + x + (1 | g)`, `log(sigma) ~ 1 + (1 | g)`,
  `atanh(phi) ~ 1 + (1 | g)`) or a data column varies by row and is read as
  `mu[p]`, as above; a name with a distribution prior (`phi ~ Uniform(-1, 1)`)
  is shared and read bare, as `sigma` and `phi` are above.
- Only `Normal` innovations and order 1. For another family, a higher order or
  a different start or gap rule, write the `@plate for` / `@scan` recurrence
  directly; the expansion above is the template.

Each row's pointwise log likelihood is its conditional density given the
group's earlier rows. With `y` omitted, the same block draws whole AR(1)
trajectories per group.
"""
function ar_errors end

# Reject the retired keyword surface at CONSTRUCTION — i.e. at the `@brm` /
# `kernel(...)` call the consumer actually wrote — not merely when the model is
# lowered.
#
# The loud rejections further down (`_sb_kernel_doblock!`, `_sb_submodel_rhs!`)
# run inside `SBBRMI(...)`, and `@brm` is a pure parser that captured `by=` /
# `n_eta=` / `model=` / `obs=` into the term's `getkwargs()` without looking at
# them. So a model written in the retired v1 spelling BUILT cleanly and objected
# only once someone lowered it. A consumer whose compatibility gate stops at
# BRMI construction — a reasonable gate, since it needs no Stan toolchain — saw
# retired syntax keep passing, and carried `by=`/`n_eta=` across 13 executable
# kernel sites for eight days after the removal landed while `brm-use` told them
# the keywords were rejected loudly (snag `by-and-n-eta-are-3625f645`).
#
# Checked in the same order the lowering-time guards use, so the reported
# keyword does not change when several are present. Keys off the `kernel`
# function object, so `hsgp(x; by=g)` — where `by=` is live — is unaffected, and
# an aliased `kernel` is still caught. The lowering-time guards stay as the
# backstop for a BRMI assembled without the macro.
function _check_term_kwargs(::typeof(kernel), kwargs)
    for (k, replacement) in (
            (:by, "Grouping is DERIVED from the ranef bucket of the per-subject \
                   linear-predictor positional args."),
            (:n_eta, "Declare per-subject linear predictors with `(1 | ID | group)` \
                      terms and pass those LPs positionally."),
            (:model, "Write the per-subject cell INLINE as a `do`-block."),
            (:obs, "Write observation likelihoods as ordinary `~` statements inside \
                    the cell body."),
        )
        haskey(kwargs, k) && error(
            "@brm: kernel(...) no longer accepts `$k=`. ", replacement,
            "\nThe surface is formula linear predictors plus one inline do-block:\n",
            "    log_CL ~ 1 + weight + (1 | p | subject)\n",
            "    log_V  ~ 1 +          (1 | p | subject)\n",
            "    pred ~ kernel(t, dose, dv, log_CL, log_V) do ts, d, yy, lCL, lV\n",
            "        mu = <prediction from exp(lCL), exp(lV) over ts, d>\n",
            "        yy ~ normal(mu, sigma)\n",
            "        mu\n",
            "    end\n",
            "See the `brm-use` skill, `kernel(...)` section.")
    end

    # `kernel(...)` takes NO keywords: the cell is the do-block and everything it
    # consumes is positional. An unrecognised keyword is therefore a typo or
    # retired syntax, never a live option, and silently ignoring one is exactly
    # how the retired v1 spelling kept passing a consumer's construction-time
    # gate for eight days after its removal (snag `by-and-n-eta-are-3625f645`).
    # The four named guards above stay because they can say what to write
    # instead; this catches everything else.
    for k in keys(kwargs)
        error("@brm: kernel(...) does not accept `$k=` — it accepts no keywords. ",
              "The cell is the do-block and everything it consumes is positional; ",
              "named values the cell assigns are addressable from the descriptor ",
              "without any annotation (`brm_output(d, :$k)` if `$k` is one). ",
              "See the `brm-use` skill, `kernel(...)` section.")
    end
    nothing
end
# Formula markers needed by common term preparation before backend-specific
# implementations add methods.
function mo end
function mo1 end
