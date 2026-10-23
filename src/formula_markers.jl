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
    car(area; edges=(from, to), type=:icar | :bym2)

Areal spatial effect over an adjacency graph. `area` is the grouping column of
the rows; `edges` names two equal-length data vectors whose `k`-th entries are
the labels (values of `area`) of two neighbouring areas. Each undirected pair is
listed once; self-loops, repeated pairs and missing labels are refused. The
graph's nodes are every label in `edges` and in `area`, so an area without rows
keeps its place in the graph, and an area without neighbours is an island.

`type=:icar` emits `sigma * phi[area]`, where `phi` carries the intrinsic CAR
prior `-1/2 * sum((phi[i] - phi[j])^2)` over the edges plus a soft sum-to-zero
per connected component of at least two areas; islands get no spatial effect.
`type=:bym2` emits the Riebler et al. (2016) convolution
`sigma * (sqrt(rho / s) * phi + sqrt(1 - rho) * theta)[area]` with independent
standard-normal `theta`, the mixing share `rho` in `[0, 1]`, and `s` the scaling
factor of each component (the geometric mean of the diagonal of the generalized
inverse of its ICAR precision), computed from the graph at preparation; an
island's effect is `sigma * theta` (Freni-Sterrantino et al. 2018).

It is a direct predictor summand: the formula intercept carries the level and no
population coefficient multiplies it. `sd(lp, car(area))` addresses `sigma`
(default half-`Normal(0, 1)`), `mixing(lp, car(area))` the BYM2 `rho` (default
`Beta(0.5, 0.5)`). On replay the graph is frozen from the fit and every row must
name a fitted area. Dispatch tag — see `_sb_car_icar` / `_sb_car_bym2`.
"""
function car end

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

A plate over long-format rows iterates [`groups`](@ref), an ordinary function:
in `@plate for p in groups(g; order = t)`, `p` is one group's indices (one
cell per distinct label of the data vector `g`, in sorted order, its indices in
ascending `t`), so a read `x[p]` is `x` at those indices (a response, a data
vector, or a predictor indexed like `g`), as the same Julia loop reads it. Values shared by every row are read bare. Such a cell observes, and
collects no result (`loc[p] = …`): nothing outside it could tell which group a
collected value belongs to.

A cell may hold a `@scan begin … end` recurrence, one per cell: seeds by literal
index, then one `for` loop of steps reading earlier entries at a fixed lag. A
carried array the scan writes (`h[t] ~ Normal(phi * h[t - 1], s)`) is a latent
state per cell; observing the cell's own response entry by entry
(`y[i][1] ~ …`, `y[i][t] ~ Normal(… y[i][t - 1] …)`) gives each entry's
conditional density, and with `y` omitted the block draws whole trajectories.
[`AR1Normal`](@ref) observes a group's series under one such recurrence.

Indexed inputs split by kind. A data vector with one entry per cell is read
`x[i]`, and a vector of per-cell vectors gives cell `i` its own vector. A
vector or linear predictor indexed like another grouping vector — dose events,
say — is read at the cell's indices through a model-scope binding:

```julia
log_F  ~ 1 + vessel + mo(diet) + hsgp(log_dose)     # one value per dose event
log_CL ~ 1 + weight + (1 | p | subject)             # one value per subject
doses = groups(dose_subject => subject)             # each subject's dose indices
obs = groups(obs_subject => subject)

@plate for i in eachindex(log_CL)
    effective_dose = dose_amount[doses[i]] .* exp.(log_F[doses[i]])
    pred[i] = pk_cell(t_obs[obs[i]], effective_dose, log_CL[i])
    dv[obs[i]] ~ normal(pred[i], sigma)
end
```

[`groups`](@ref)`(dose_subject => subject)` is, for each entry of `subject`,
the indices of `dose_subject` that hold it, so `x[doses[i]]` is `x` at subject
`i`'s doses, as the same Julia loop reads it. Entries join their cell by label,
never by position. Every binding one plate reads names the same label vector,
and under a random-effect predictor that is the predictor's grouping vector.
When no predictor carries a random effect, the label vector lists the cells:
a district total observes the summed rate of its facilities,

```julia
log_lambda ~ 1 + (1 | fac | facility)                # one value per facility
facilities = groups(fac_district => district)       # each district's facilities

@plate for d in eachindex(Y)                        # Y: one total per district
    Y[d] ~ Poisson(sum(exp.(log_lambda[facilities[d]])))
end
```

and the plate has one cell per entry of `district`, in that order. An index
vector into another grouping's entries — an event schedule's global read number
per observation — is read as positions among the cell's indices of that
grouping, `indexin(obs_read[obs[i]], reads[i])` with
`reads = groups(read_subject => subject)`; an index that belongs to another
subject is refused, and `something.(indexin(…), 0)` keeps `0` as "no index".
A binding no cell reads is refused.

A per-subject linear predictor is passed as that group's scalar value. One
with random effects over several groupings — Kreise nested in Länder, `(1 |
land | land) + (1 | kreis | kreis)` — is read on the grouping whose column lists
one unique label per entry; the others are ordinary terms of its value. One
without a random effect defines no grouping of its own: it is read per subject
as a random-effect predictor defines the subjects — or, in a panel with no
random effects, as the pre-grouped vectors do — so its data vectors must have
one entry per subject.

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
    kernel(ragged(x, group), ...) do ...

The kernel input that groups a vector by a grouping vector naming the subject
of each entry. A `@plate for` cell does not write it: it reads `x[rows[i]]`
through a model-scope `rows = groups(group => labels)` binding (see
[`kernel`](@ref) and [`groups`](@ref)), which lowers to
`ragged(x, group, labels)`. The model-scope `ragged(x, group)` statement and the
in-cell `ragged(x, group)[i]` read are retired and refused with the binding that
replaces them (todo `1nfukuv`): they changed what `x[i]` means in a cell
without saying so there.

The formula-level observation `ragged(y, group) ~ Family(pred, ...)` is
retired the same way: observe the response in the cell that computes its
prediction, `y[rows[i]] ~ Family(pred[i], ...)`. `ragged` remains only as a
positional of the deprecated `kernel(...) do` spelling: `ragged(x, group)`,
`ragged(x, group, labels)` or `ragged(idx, group; into = target)`.

ONE VALUE PER ENTRY OF `x` — no expansion happens anywhere. If an event
schedule stores a compact schedule (one entry per dose OP, carrying an interval
and a count) then `x` has one value per OP. Grouping requires one `group` key
per entry of `x` and joins those keys to the kernel's subject labels. That is
local validation of the grouping operation, not an inner-shape contract between
cell arguments.

Dispatch tag only — kernel-argument lowering lives in `_sb_kernel_doblock!`
(sbimpl) and the RK kernel preparation.
"""
ragged

# `order=` is the kernel positional a `@plate for p in groups(g; order = t)`
# header lowers to; it is not written on `ragged` itself.
_check_term_kwargs(::typeof(ragged), kwargs) =
    all(in((:into, :order)), keys(kwargs)) || error(
    "@brm: ragged(...) takes only the `into=` keyword, got $(keys(kwargs)). In a ",
    "`@plate for` cell, read `x[rows[i]]` with `rows = groups(group => labels)`; ",
    "an index vector into another grouping is `indexin(idx[rows[i]], target[i])`.")

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
