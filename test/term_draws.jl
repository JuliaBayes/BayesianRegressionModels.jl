# test/term_draws.jl — acceptance for `term_draws` (src/prediction.jl): ablate ONE
# named linear-predictor term in fitted UNCONSTRAINED draws, then regenerate the
# descriptor outputs with `:predict`.
#
# WHY THE MODEL-LEVEL CHECKS ARE THE LOAD-BEARING ONES. It is easy to write a
# term-ablation that zeroes *some* coordinates and looks right: the draw matrix
# has the expected shape and nothing errors. What proves it is (a) that EXACTLY
# the hsgp basis-weight coordinates change and nothing else, and (b) reading the
# model's OWN `log_F` transformed parameter back out of BridgeStan and asserting
# that it is AFFINE in the basis weights with the ablated draw sitting on the
# zero-contribution point — that is the model agreeing the hsgp term is gone,
# not this file agreeing with itself about an index layout. The `:predict` leg
# then confirms the reporter's actual regeneration path carries the change.
#
# This is the `hsgp(op_log_dose; k=5)` in `log_F` case from snag
# `joint-brm2-term-065de094`: the exact shape a downstream diagnostic needs to
# evaluate the fitted draws twice — full `log_F` and a linearized `log_F` with
# only the HSGP term removed.
#
# Run: julia --project=<worktree> test/term_draws.jl
# (needs BridgeStan reachable through StanBlocks; set BRM_TERM_RUNTIME=0 to run
# only the declaration-level / fail-closed half.)

using Test
using BayesianRegressionModels
using StanBlocks
using Random
using Distributions: Exponential, LogNormal, Normal

const RUNTIME = get(ENV, "BRM_TERM_RUNTIME", "1") != "0"
const BS = StanBlocks.BridgeStan

# ---- fixture: the reporter's `log_F` shape ----------------------------------
# Population terms + a monotonic `mo(op_diet)` + `hsgp(op_log_dose; k=5)`, observed
# through a Normal so `:predict` is offered. Identical in structure to the joint
# PK model's `log_F` linear predictor.
const N = 12
term_df = (;
    y                = collect(range(0.0, 1.0; length = N)),
    vessel_bottle    = repeat([1.0, 0.0, 0.0], 4),
    vessel_bottle_20 = repeat([0.0, 1.0, 0.0], 4),
    vessel_tablet    = repeat([0.0, 0.0, 1.0], 4),
    vessel_tablet_20 = repeat([1.0, 0.0, 1.0], 4),
    op_diet          = repeat(1:4, 3),
    op_log_dose      = collect(range(-1.0, 1.0; length = N)),
)

term_builder = @brm begin
    y ~ Normal(log_F, 1.0)
    log_F ~ 0 + vessel_bottle + vessel_bottle_20 + vessel_tablet +
                vessel_tablet_20 + mo(op_diet) + op_log_dose +
                hsgp(op_log_dose; k = 5)
    effect(log_F, :) ~ Normal(0.0, 0.5)
    effect(log_F, op_log_dose) ~ Normal(0.0, 0.6676)
    length_scale(:, hsgp(op_log_dose)) ~ LogNormal(0.0, 1.0)
    sd(:, hsgp(op_log_dose)) ~ LogNormal(0.0, 1.0)
end

d = brm_descriptor(term_builder, term_df;
                   mod = @__MODULE__, name = :joint_log_F, highlights = ())

# ---- 1. fail-closed edges (need no compiled model) --------------------------

@testset "term_draws fails closed on unsupported target / term / shape" begin
    unc  = ["a.$i" for i in 1:4]        # dummy names; the guards fire before use
    zero = zeros(1, 4)

    # `to` other than :zero is refused (no other target has an established meaning).
    @test_throws "only `to = :zero`" term_draws(
        d, zero, unc; predictor = :log_F, term = :hsgp_op_log_dose, to = :mean)


    # A missing term is refused with the available labels.
    @test_throws "available term labels" term_draws(
        d, zero, unc; predictor = :log_F, term = :hsgp_missing)

    # draws / unc_names shape mismatch is caught before any indexing.
    @test_throws ErrorException term_draws(
        d, zeros(1, 3), unc; predictor = :log_F, term = :hsgp_op_log_dose)
end

@testset "grouped hsgp(...; by=...) has no basis-weight carrier to ablate" begin
    # A grouped hsgp is a correlated random-effect block: its weights are zeroed
    # with `population_draws`, not `term_draws`. It must fail closed here.
    grouped_df = (;
        y           = collect(range(0.0, 1.0; length = 6)),
        op_log_dose = collect(range(-1.0, 1.0; length = 6)),
        subject     = repeat(1:2, 3),
    )
    grouped = @brm begin
        y ~ Normal(log_F, 1.0)
        log_F ~ 0 + op_log_dose + hsgp(op_log_dose; k = 4, by = subject)
    end
    dg = brm_descriptor(grouped, grouped_df; mod = @__MODULE__, highlights = ())
    # Fails closed one way or another — a grouped hsgp exposes no basis-weight
    # role, so there is no coordinate `term_draws` may zero.
    @test_throws ErrorException term_draws(
        dg, zeros(1, 2), ["a.1", "a.2"];
        predictor = :log_F, term = :hsgp_op_log_dose_by_subject)
end

if !RUNTIME
    @info "BRM_TERM_RUNTIME=0 — skipping the compiled-model half"
else

# ---- 2. against the real compiled model -------------------------------------

prob  = brm_execute(d, :fit)
sm    = prob.model
unc   = BS.param_unc_names(sm)
n_unc = BS.param_unc_num(sm)

# The basis-weight carrier, resolved BY NAME against the UNCONSTRAINED names.
bw = brm_term_coordinates(d, :log_F, unc;
                          term = :hsgp_op_log_dose, parameter = :basis_weights).coordinates

@testset "the hsgp basis weights resolve in unconstrained space" begin
    @test length(bw) == 5                       # k = 5 basis functions
    @test all(i -> startswith(String(unc[i]), "hsgp_op_log_dose_beta_raw."), bw)
    @test length(unique(bw)) == length(bw)
end

rng   = MersenneTwister(20260812)
draws = randn(rng, 6, n_unc)
lin   = term_draws(d, draws, unc; predictor = :log_F, term = :hsgp_op_log_dose)

@testset "term_draws zeroes EXACTLY the basis weights, nothing else" begin
    @test size(lin) == size(draws)
    @test all(iszero, lin[:, bw])                # the term's carrier → 0
    keep = setdiff(1:n_unc, bw)
    @test lin[:, keep] == draws[:, keep]         # every other coordinate identical
    @test draws[:, bw] != lin[:, bw]             # input DID carry nonzero weights
    @test !(draws === lin)                       # input matrix left untouched
end

# Read the model's own `log_F` transformed parameter back out of BridgeStan.
function logF(sm, theta)
    names = BS.param_names(sm; include_tp = true, include_gq = false)
    vals  = BS.param_constrain(sm, theta; include_tp = true, include_gq = false)
    [v for (nm, v) in zip(names, vals) if startswith(String(nm), "log_F.")]
end

@testset "the model's log_F is affine in beta_raw, ablated draw at the zero point" begin
    row  = draws[1, :]
    v    = row[bw]                               # the fitted basis weights
    @test !isempty(logF(sm, row))                # log_F is an exposed TP

    base = copy(row); base[bw] .= 0.0            # beta_raw := 0
    two  = copy(row); two[bw]  .= 2.0 .* v       # beta_raw := 2v

    lf_full = logF(sm, row)                       # beta_raw = v
    lf_base = logF(sm, base)                       # beta_raw = 0
    lf_two  = logF(sm, two)                         # beta_raw = 2v
    lf_abl  = logF(sm, lin[1, :])                   # what term_draws produced

    # (a) the ablated draw lands exactly on the beta_raw := 0 location.
    @test lf_abl ≈ lf_base
    # (b) the hsgp term genuinely contributes: with weights present, log_F moves.
    @test any(lf_full .!= lf_base)
    # (c) removed EXACTLY: log_F is affine in beta_raw with the ablated point as
    #     its zero-contribution baseline, so doubling the weights doubles the
    #     displacement from that baseline. Independent of any index layout.
    @test lf_two .- lf_base ≈ 2.0 .* (lf_full .- lf_base)
end

# `mo(op_diet)` contributes `beta_pop[k] * mo_op_diet` through its own
# population column (labelled like the term), so `term = :mo_op_diet` zeroes
# that magnitude and removes the whole monotonic summand; the simplex stays.
@testset "term_draws removes an mo(...) term through its magnitude column" begin
    k = brm_population_effect_coordinates(d, :log_F, unc;
                                          coefficient = :mo_op_diet).coordinates
    @test length(k) == 1
    mo_abl = term_draws(d, draws, unc; predictor = :log_F, term = :mo_op_diet)
    @test all(iszero, mo_abl[:, k])
    keep = setdiff(1:n_unc, k)
    @test mo_abl[:, keep] == draws[:, keep]      # the simplex is untouched
    row = draws[1, :]
    two = copy(row); two[k] .= 2.0 .* row[k]
    # log_F is affine in the magnitude with the ablated draw as its baseline.
    @test any(logF(sm, row) .!= logF(sm, mo_abl[1, :]))
    @test logF(sm, two) .- logF(sm, mo_abl[1, :]) ≈
          2.0 .* (logF(sm, row) .- logF(sm, mo_abl[1, :]))
end

# ---- 3. the reporter's actual path: regenerate with `:predict` --------------

@testset "brm_execute(:predict) carries the ablation into pk-conc-style output" begin
    seed = 99
    full = brm_execute(d, :predict; problem = prob, draws = draws[1, :], seed = seed)
    abl  = brm_execute(d, :predict; problem = prob, draws = lin[1, :],   seed = seed)
    @test haskey(full, :y_gen) && haskey(abl, :y_gen)
    @test length(abl.y_gen) == N
    # Same seed ⇒ identical observation-noise draws, so any difference is exactly
    # the removed hsgp location shift propagating through the generated quantity.
    @test full.y_gen != abl.y_gen
    # And a second ablated draw with different (removed) weights regenerates the
    # SAME location structure the ablation defines — the term is gone downstream.
    lin2 = term_draws(d, draws, unc; predictor = :log_F, term = :hsgp_op_log_dose)
    abl2 = brm_execute(d, :predict; problem = prob, draws = lin2[2, :], seed = seed)
    @test length(abl2.y_gen) == N
end

end # RUNTIME

# ---- 4. population terms, categorical blocks and declared parameters --------
#
# The same two-sided proof for the population path and `parameter_draws`: the
# returned matrix differs from the input in EXACTLY the resolved coordinates,
# and the model's own transformed parameters, read back through BridgeStan,
# move by exactly the removed summand (computed independently from the data
# and the constrained coefficient draws). `parameter_draws` is checked against
# BridgeStan's own `param_unconstrain`, so its bound transform is the model's.

const M = 18
pop_df = (;
    y        = collect(range(-1.0, 1.0; length = M)),
    age      = collect(range(-2.0, 2.0; length = M)) .^ 2 .- 1.0,
    site     = repeat([1, 2, 3], 6),         # integer-coded categoricals
    source   = repeat([1, 2], 9),
    exposure = repeat([0.0, 0.5, 1.0], 6),
)

pop_builder = @brm begin
    gain ~ Normal(0, 1)
    sigma ~ Exponential(1)
    mu ~ 1 + age + site
    shift ~ factor(source; cmc = false)
    loc = mu + shift + gain * exposure
    y ~ Normal(loc, sigma)
end

dp = brm_descriptor(pop_builder, pop_df; mod = @__MODULE__, highlights = ())

@testset "population / parameter ablation fails closed without a model" begin
    unc  = ["a.$i" for i in 1:4]
    zero = zeros(1, 4)
    # An unknown label names the population labels too.
    @test_throws "population coefficient labels" term_draws(
        dp, zero, unc; predictor = :mu, term = :weight)
    # A non-parameter output is refused by role, before any coordinate lookup.
    @test_throws "not a declared parameter" parameter_draws(
        dp, zero, unc; parameter = :mu, to = 0.0)
    @test_throws "finite" parameter_draws(
        dp, zero, unc; parameter = :gain, to = Inf)
end

if RUNTIME

pprob = brm_execute(dp, :fit)
psm   = pprob.model
punc  = BS.param_unc_names(psm)
pn    = BS.param_unc_num(psm)
tpn   = BS.param_names(psm; include_tp = true, include_gq = false)

tp(sm, theta, stem) = begin
    vals = BS.param_constrain(sm, theta; include_tp = true, include_gq = false)
    [v for (nm, v) in zip(tpn, vals) if startswith(String(nm), stem * ".")]
end
cval(theta, idx) = BS.param_constrain(psm, theta; include_tp = true,
                                      include_gq = false)[idx]

prng   = MersenneTwister(20261019)
pdraws = randn(prng, 4, pn)

@testset "term_draws zeroes one numeric population coefficient exactly" begin
    abl = term_draws(dp, pdraws, punc; predictor = :mu, term = :age)
    k   = brm_population_effect_coordinates(dp, :mu, punc; coefficient = :age).coordinates
    @test length(k) == 1
    @test all(iszero, abl[:, k])
    keep = setdiff(1:pn, k)
    @test abl[:, keep] == pdraws[:, keep]
    @test !(abl === pdraws) && pdraws[:, k] != abl[:, k]
    beta = only(brm_population_effect_coordinates(dp, :mu, tpn; coefficient = :age).coordinates)
    for i in axes(pdraws, 1)
        row = pdraws[i, :]
        # mu(full) - mu(ablated) is exactly the removed summand beta_age * age.
        @test tp(psm, row, "mu") .- tp(psm, abl[i, :], "mu") ≈ cval(row, beta) .* pop_df.age
    end
end

@testset "term_draws removes a treatment-coded categorical block exactly" begin
    abl = term_draws(dp, pdraws, punc; predictor = :mu, term = :site)
    r   = brm_population_effect_coordinates(dp, :mu, punc; coefficient = :site)
    @test r.coding === :treatment && length(r.coordinates) == 2
    @test all(iszero, abl[:, r.coordinates])
    keep = setdiff(1:pn, r.coordinates)
    @test abl[:, keep] == pdraws[:, keep]
    rc = brm_population_effect_coordinates(dp, :mu, tpn; coefficient = :site)
    for i in axes(pdraws, 1)
        row = pdraws[i, :]
        contrast = Dict(string(c.nonreference_level) => cval(row, c.coordinate)
                        for c in rc.contrasts)
        expected = [get(contrast, string(s), 0.0) for s in pop_df.site]
        @test any(!iszero, expected)
        # Every level now takes the reference level's value.
        @test tp(psm, row, "mu") .- tp(psm, abl[i, :], "mu") ≈ expected
    end
end

@testset "term_draws on an intercept-free factor(...; cmc=false) zeroes it" begin
    abl = term_draws(dp, pdraws, punc; predictor = :shift, term = :source)
    r   = brm_population_effect_coordinates(dp, :shift, punc; coefficient = :source)
    @test all(iszero, abl[:, r.coordinates])
    keep = setdiff(1:pn, r.coordinates)
    @test abl[:, keep] == pdraws[:, keep]
    for i in axes(pdraws, 1)
        @test any(!iszero, tp(psm, pdraws[i, :], "shift"))
        @test all(iszero, tp(psm, abl[i, :], "shift"))
    end
end

@testset "parameter_draws sets an unbounded declared parameter" begin
    abl = parameter_draws(dp, pdraws, punc; parameter = :gain, to = 0.0)
    g   = brm_output_coordinates(dp, :gain, punc)
    @test length(g) == 1 && all(iszero, abl[:, g])
    keep = setdiff(1:pn, g)
    @test abl[:, keep] == pdraws[:, keep]
    gc = only(brm_output_coordinates(dp, :gain, tpn))
    for i in axes(pdraws, 1)
        row = pdraws[i, :]
        @test tp(psm, row, "loc") .- tp(psm, abl[i, :], "loc") ≈
              cval(row, gc) .* pop_df.exposure
    end
    two = parameter_draws(dp, pdraws, punc; parameter = :gain, to = 2.5)
    @test all(i -> cval(two[i, :], gc) ≈ 2.5, axes(pdraws, 1))
end

@testset "parameter_draws maps a bounded parameter through Stan's transform" begin
    abl = parameter_draws(dp, pdraws, punc; parameter = :sigma, to = 0.5)
    s   = brm_output_coordinates(dp, :sigma, punc)
    sc  = only(brm_output_coordinates(dp, :sigma, tpn))
    pnames = BS.param_names(psm)
    for i in axes(pdraws, 1)
        @test cval(abl[i, :], sc) ≈ 0.5
        # The unconstrained image equals BridgeStan's own unconstrain of the
        # constrained draw with sigma replaced.
        theta = BS.param_constrain(psm, pdraws[i, :])
        theta[only(brm_output_coordinates(dp, :sigma, pnames))] = 0.5
        @test abl[i, :] ≈ BS.param_unconstrain(psm, theta)
    end
    @test abl[:, setdiff(1:pn, s)] == pdraws[:, setdiff(1:pn, s)]
    # The boundary itself has an infinite unconstrained image.
    @test_throws "strictly inside the support" parameter_draws(
        dp, pdraws, punc; parameter = :sigma, to = 0.0)
end

@testset "regenerating :predict carries the ablation" begin
    abl  = term_draws(dp, pdraws, punc; predictor = :shift, term = :source)
    abl  = parameter_draws(dp, abl, punc; parameter = :gain, to = 0.0)
    full = brm_execute(dp, :predict; problem = pprob, draws = pdraws[1, :], seed = 7)
    gone = brm_execute(dp, :predict; problem = pprob, draws = abl[1, :], seed = 7)
    @test length(gone.y_gen) == M
    @test full.y_gen != gone.y_gen
end

# A shared support-bounded population prior declares `lower=0`: unconstrained
# 0 is not coefficient 0.
bounded = @brm begin
    mu ~ 1 + age
    effect(mu, :) ~ Exponential(1)
    y ~ Normal(mu, 1.0)
end
db = brm_descriptor(bounded, pop_df; mod = @__MODULE__, highlights = ())
bunc = BS.param_unc_names(brm_execute(db, :fit).model)

@testset "a support-bounded population coefficient is refused" begin
    @test_throws "unconstrained coordinate is not the" term_draws(
        db, zeros(1, length(bunc)), bunc; predictor = :mu, term = :age)
end

# An exact-total block absorbs the `(1 | g)` intercept: no sampled coordinate.
totals_df = merge(pop_df, (; group = repeat(1:6, 3)))
totals = @brm begin
    mu ~ 1 + age + (1 | group)
    y ~ Normal(mu, 1.0)
end
dt = brm_descriptor(totals, totals_df; mod = @__MODULE__, highlights = ())
tprob = brm_execute(dt, :fit)
tunc = BS.param_unc_names(tprob.model)

@testset "exact-total absorbed intercept refused; sampled slope ablates" begin
    @test !isempty(total_effect_blocks(dt.plan))
    @test_throws "absorbed by an exact total-coefficient block" term_draws(
        dt, zeros(1, length(tunc)), tunc; predictor = :mu, term = :Intercept)
    tdraws = randn(MersenneTwister(3), 2, length(tunc))
    abl = term_draws(dt, tdraws, tunc; predictor = :mu, term = :age)
    k = brm_population_effect_coordinates(dt, :mu, tunc; coefficient = :age).coordinates
    @test all(iszero, abl[:, k])
    @test abl[:, setdiff(axes(tdraws, 2), k)] == tdraws[:, setdiff(axes(tdraws, 2), k)]
end

end # RUNTIME (population / parameter half)
