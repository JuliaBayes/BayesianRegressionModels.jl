# test/ranef_draws.jl — `brm_ranef_draws`: every random-effect block's
# between-group SDs, correlation matrix and per-level effects per draw, read
# from the carriers the compiled program emits (snag
# `public-ranef-sd-cda807e5`; shape chosen on decision `0phr2dj`).
#
# The oracles never read the carriers the accessor reads. Per-level effects are
# checked against each block's per-ROW random-effect contribution (which reads
# them through the design `Z` and the group index, so margin order, level order
# and orientation are all pinned), SDs and correlations against the standardised
# draw `z` (`effects = diag(sd) chol(cor) z`), exact totals against
# `total - A * population`, and known values against the constants written in
# the formula. A stratified `gr(g, by=b)` block reports one entry per stratum
# (decision `0zvkadt`); each row's level lies in exactly one entry, and the
# stratum's own SDs and correlation reproduce its groups' effects.
#
# Run: julia --project=test test/ranef_draws.jl
# (set BRM_RANEF_DRAWS_RUNTIME=0 to run only the declaration-level half.)

using Test
using BayesianRegressionModels
using StanBlocks
using Random
using LinearAlgebra: cholesky, Symmetric, dot, I
using Distributions: Exponential, Normal, TDist, Beta, LKJCholesky

const BRM = BayesianRegressionModels
const BS = StanBlocks.BridgeStan
const RUNTIME = get(ENV, "BRM_RANEF_DRAWS_RUNTIME", "1") != "0"

rng = MersenneTwister(11)
const N = 24
data = (;
    subject = repeat(string.("s", 1:6); inner = 4),
    site = repeat(["a", "b", "c"], N ÷ 3),
    x = randn(rng, N),
    y = randn(rng, N),
    z = randn(rng, N),
    omega_sd = 0.3,
    g1 = repeat(["a", "b", "c"], N ÷ 3),
    g2 = repeat(["b", "c", "a"], N ÷ 3),
    w1 = rand(rng, N),
    w2 = rand(rng, N),
)
# Strata: site `c` alone in stratum `u`; subjects s1-s3 in arm `t`, s4-s6 in `c`.
strata = merge(data, (; group_of_site = [s == "c" ? "u" : "v" for s in data.site],
                        arm = [s in ("s1", "s2", "s3") ? "t" : "c" for s in data.subject]))

shared_builder = @brm begin
    sigma ~ Exponential(1)
    tau_z ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject) + (1 | site)
    nu ~ 1 + (1 | p | subject)
    y ~ Normal(mu, sigma)
    z ~ Normal(nu, tau_z)
end
student_builder = @brm begin
    sigma ~ Exponential(1)
    tau_z ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject) + (1 + x | site)
    nu ~ 1 + (1 | p | subject)
    ranef(:, p) ~ TDist(4)
    y ~ Normal(mu, sigma)
    z ~ Normal(nu, tau_z)
end
generic_builder = @brm begin
    sigma ~ Exponential(1)
    tau_z ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject)
    nu ~ 1 + (1 | p | subject)
    sd(mu, p, x) ~ Exponential(1 / 4)
    cor(:, p) ~ LKJCholesky(3, 2)
    y ~ Normal(mu, sigma)
    z ~ Normal(nu, tau_z)
end
nocor_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x || subject) + (1 + x | site)
    y ~ Normal(mu, sigma)
end
totals_builder = @brm begin
    sigma ~ Exponential(1)
    tau_z ~ Exponential(1)
    mu ~ 1 + x + (1 + x || subject)
    nu ~ 1 + (1 | q | subject)
    y ~ Normal(mu, sigma)
    z ~ Normal(nu, tau_z)
end
known_mixed_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject)
    sd(mu, p, Intercept) = sqrt(0.1)
    sd(mu, p, x) ~ Exponential(0.5)
    y ~ Normal(mu, sigma)
end
known_all_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject)
    sd(:, p) = omega_sd
    cor(:, p) = [1 0.3; 0.3 1]
    y ~ Normal(mu, sigma)
end
r2d2_bucket_builder = @brm begin
    sigma ~ Exponential(1)
    log_Vc ~ 1 + (1 | p | subject)
    log_k10 ~ 1 + (1 | p | subject)
    sd(:, p) ~ r2d2(mean_R2 = 0.5, prec_R2 = 2, concentration = 1,
                    reference_scale = sigma)
    cor(:, p) ~ LKJCholesky(2, 2)
    y ~ Normal(log_Vc + log_k10, sigma)
end
r2d2_flat_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | subject)
    effect(mu, :) ~ r2d2(R2 = Beta(1, 1), tau_bsv = 0.5, alpha = 1)
    y ~ Normal(mu, sigma)
end
mm_builder = @brm begin
    sigma ~ Exponential(1)
    loc ~ 1 + (1 | mm(g1, g2; weights = (w1, w2)))
    y ~ Normal(loc, sigma)
end
stratified_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | gr(site, by = group_of_site))
    y ~ Normal(mu, sigma)
end
stratified_id_builder = @brm begin
    sigma ~ Exponential(1)
    tau_z ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | gr(subject, by = arm))
    nu ~ 1 + (1 | p | gr(subject, by = arm))
    y ~ Normal(mu, sigma)
    z ~ Normal(nu, tau_z)
end

build(builder, d = data; kw...) = SBBRMI(builder(d); mod = @__MODULE__, kw...)
no_z = Base.structdiff(data, NamedTuple{(:z,)})

@testset "selection, margins and levels (declaration level)" begin
    sb = build(shared_builder; total_groups = ())
    d = brm_descriptor(sb)
    names = ["x"]
    draws = zeros(1, 1)
    # refused: a selection naming no block must not silently return nothing
    # (fail closed, BRM primer "Invariants": never infer coverage from empty).
    @test_throws "no random-effect block matches" brm_ranef_draws(d, draws, names; id = :nope)
    @test_throws "no random-effect block matches" brm_ranef_draws(d, draws, names; group = :nope)
    # refused: the carriers are absent from these names, which is drift
    # between the draws and the model, never an empty result.
    @test_throws "brm_ranef_draws" brm_ranef_draws(d, draws, names; id = :p)
end

if !RUNTIME
    @info "BRM_RANEF_DRAWS_RUNTIME=0 — skipping the compiled-model half"
else

# Constrained draws (transformed parameters and generated quantities) at a few
# random unconstrained points, with a name -> column lookup for the oracles.
function constrained_draws(sb; n_draws = 3, seed = 1)
    sm = BRM.stan_instantiate(sb).model
    names = BS.param_names(sm; include_tp = true, include_gq = true)
    r = MersenneTwister(seed)
    rows = map(1:n_draws) do i
        theta = 0.7 .* randn(r, BS.param_unc_num(sm))
        BS.param_constrain(sm, theta; include_tp = true, include_gq = true,
                           rng = BS.StanRNG(sm, seed + i))
    end
    draws = permutedims(reduce(hcat, rows))
    col = Dict(n => j for (j, n) in enumerate(names))
    (; sm, names, draws, col, d = brm_descriptor(sb))
end
value(c, i, name) = c.draws[i, c.col[name]]
level_of(levels, labels) = [findfirst(==(l), levels) for l in labels]
selected(blocks, binding) = only(b for b in blocks if b.binding === binding)

# `rowcarrier[r] == Z(r) ⋅ effects[g(r), margins]` for every draw and row, the
# level `g(r)` read from the one entry (of a block's per-stratum entries) whose
# levels hold row `r`'s label.
function check_rows(c, entries::AbstractVector, rowcarrier, margins, Z, labels)
    ok = true
    for r in eachindex(labels)
        block = only(b for b in entries if labels[r] in b.levels)
        g = findfirst(==(labels[r]), block.levels)
        for i in axes(c.draws, 1)
            expected = dot(Z(r), block.effects[i, g, margins])
            ok &= isapprox(value(c, i, "$rowcarrier.$r"), expected; atol = 1e-10)
        end
    end
    ok
end
check_rows(c, block::NamedTuple, args...) = check_rows(c, [block], args...)

# `effects[g, :] == diag(sd) chol(cor) z[:, zlevels[g]]` (times the Student-t
# weight); `zlevels` places the entry's levels on the block's `z` level axis.
function check_noncentered(c, block, zname; mixing = nothing,
                           zlevels = eachindex(block.levels))
    K = length(block.margins)
    ok = true
    for i in axes(c.draws, 1)
        L = K == 1 ? ones(1, 1) : Matrix(cholesky(Symmetric(block.cor[i, :, :])).L)
        for (g, zg_index) in enumerate(zlevels)
            zg = [value(c, i, "$zname.$k.$zg_index") for k in 1:K]
            w = isnothing(mixing) ? 1.0 : sqrt(value(c, i, "$mixing.$zg_index"))
            ok &= isapprox(block.effects[i, g, :], w .* (block.sd[i, :] .* (L * zg));
                           atol = 1e-10)
        end
    end
    ok
end

Zint(r) = [1.0]
Zslope(r) = [data.x[r]]
Zboth(r) = [1.0, data.x[r]]

@testset "shared |ID| block and a plain intercept, non-centred" begin
    c = constrained_draws(build(shared_builder; total_groups = ()))
    blocks = brm_ranef_draws(c.d, c.draws, c.names)
    p = only(brm_ranef_draws(c.d, c.draws, c.names; id = :p))
    @test p.binding === :b_p_subject && p.id === :p && p.group === :subject
    @test p.by === nothing && p.stratum === nothing
    @test p.margins == [(predictor = :mu, coefficient = :Intercept),
                        (predictor = :mu, coefficient = :x),
                        (predictor = :nu, coefficient = :Intercept)]
    @test size(p.sd) == (3, 3) && size(p.cor) == (3, 3, 3) && size(p.effects) == (3, 6, 3)
    @test check_rows(c, p, "r_mu_p_subject", 1:2, Zboth, data.subject)
    @test check_rows(c, p, "r_nu_p_subject", 3:3, Zint, data.subject)
    @test check_noncentered(c, p, "b_p_subject_z")
    @test all(isapprox(p.cor[i, k, k], 1.0; atol = 1e-12) for i in 1:3, k in 1:3)
    # The SDs agree with the per-margin public resolver.
    for (k, m) in enumerate(p.margins)
        r = brm_ranef_sd_coordinates(c.d, m.predictor, c.names; id = :p,
                                     coefficient = m.coefficient)
        @test p.sd[:, k] == c.draws[:, only(r.coordinates)]
    end
    site = only(brm_ranef_draws(c.d, c.draws, c.names; group = :site))
    @test site.family === :ranef_intercept && site.id === nothing
    @test site.margins == [(predictor = :mu, coefficient = :Intercept)]
    @test site.sd[:, 1] ≈ exp.(c.draws[:, c.col["r_mu_site_log_scale"]])
    @test check_rows(c, site, "r_mu_site", 1:1, Zint, data.site)
    @test site.cor == ones(3, 1, 1)
    @test Set(b.binding for b in blocks) == Set([:b_p_subject, :r_mu_site])
    @test length(brm_ranef_draws(c.d, c.draws, c.names; group = :subject)) == 1
end

@testset "centred emission reads the sampled effects" begin
    c = constrained_draws(build(shared_builder; total_groups = (),
                                centered_groups = [:subject, :site]))
    p = only(brm_ranef_draws(c.d, c.draws, c.names; id = :p))
    @test p.family === :ranef_correlated_draws_centered
    @test check_rows(c, p, "r_mu_p_subject", 1:2, Zboth, data.subject)
    @test check_rows(c, p, "r_nu_p_subject", 3:3, Zint, data.subject)
    site = only(brm_ranef_draws(c.d, c.draws, c.names; group = :site))
    @test site.family === :ranef_intercept_centered
    @test check_rows(c, site, "r_mu_site", 1:1, Zint, data.site)
end

@testset "sd and cor priors: the generic emissions, non-centred and centred" begin
    for (centered, family) in ((Symbol[], :ranef_correlated_draws_generic),
                               ([:subject], :ranef_correlated_draws_centered_generic))
        c = constrained_draws(build(generic_builder; total_groups = (),
                                    centered_groups = centered))
        p = only(brm_ranef_draws(c.d, c.draws, c.names; id = :p))
        @test p.family === family
        @test check_rows(c, p, "r_mu_p_subject", 1:2, Zboth, data.subject)
        @test check_rows(c, p, "r_nu_p_subject", 3:3, Zint, data.subject)
        isempty(centered) && @test check_noncentered(c, p, "b_p_subject_z")
        for (k, m) in enumerate(p.margins)
            r = brm_ranef_sd_coordinates(c.d, m.predictor, c.names; id = :p,
                                         coefficient = m.coefficient)
            @test p.sd[:, k] == c.draws[:, only(r.coordinates)]
        end
    end
end

@testset "Student-t |ID| block and a plain correlated block" begin
    c = constrained_draws(build(student_builder; total_groups = ()))
    p = only(brm_ranef_draws(c.d, c.draws, c.names; id = :p))
    @test check_rows(c, p, "r_mu_p_subject", 1:2, Zboth, data.subject)
    @test check_noncentered(c, p, "b_p_subject_z"; mixing = "b_p_subject_w")
    site = only(brm_ranef_draws(c.d, c.draws, c.names; group = :site))
    @test site.family === :ranef_correlated
    @test site.margins == [(predictor = :mu, coefficient = :Intercept),
                           (predictor = :mu, coefficient = :x)]
    @test size(site.effects) == (3, 3, 2)
    @test check_rows(c, site, "r_mu_site", 1:2, Zboth, data.site)
    @test check_noncentered(c, site, "r_mu_site_z")
end

@testset "independent || margins and a centred plain block" begin
    c = constrained_draws(build(nocor_builder; total_groups = (),
                                centered_groups = [:site]))
    blocks = brm_ranef_draws(c.d, c.draws, c.names; group = :subject)
    @test [b.family for b in blocks] == [:ranef_intercept, :ranef_slope]
    @test [only(b.margins).coefficient for b in blocks] == [:Intercept, :x]
    @test check_rows(c, blocks[1], "r_mu_subject__nocor__1", 1:1, Zint, data.subject)
    @test check_rows(c, blocks[2], "r_mu_subject__nocor__2", 1:1, Zslope, data.subject)
    @test blocks[2].sd[:, 1] == c.draws[:, c.col["r_mu_subject__nocor__2_tau.1"]]
    site = only(brm_ranef_draws(c.d, c.draws, c.names; group = :site))
    @test site.family === :ranef_correlated_centered
    @test check_rows(c, site, "r_mu_site", 1:2, Zboth, data.site)
end

@testset "exact totals: one entry per absorbed column, Stan's deviations" begin
    sb = build(totals_builder)
    c = constrained_draws(sb)
    blocks = brm_ranef_draws(c.d, c.draws, c.names)
    @test [(b.family, b.id, only(b.margins)) for b in blocks] ==
        [(:brm_total, nothing, (predictor = :mu, coefficient = :Intercept)),
         (:brm_total, nothing, (predictor = :mu, coefficient = :x)),
         (:brm_total, :q, (predictor = :nu, coefficient = :Intercept))]
    @test only(brm_ranef_draws(c.d, c.draws, c.names; id = :q)).binding === :total_nu
    for t in total_effect_blocks(sb)
        J, K = length(blocks[1].levels), length(t.columns)
        mine = [b for b in blocks if b.binding === t.binding]
        for (k, b) in enumerate(mine), i in 1:3
            beta = [value(c, i, "$(t.population).$j") for j in 1:length(t.population_columns)]
            pop = t.A * beta
            total = [value(c, i, "$(t.binding).$g.$k") for g in 1:J]
            @test b.effects[i, :, 1] ≈ total .- pop[k] atol = 1e-10
            @test b.sd[i, 1] == value(c, i, "$(t.scales).$k")
        end
    end
    # The deviations are generated quantities: names without them are refused
    # with the remedy, not resolved to an empty slice.
    sm = c.sm
    tp_names = BS.param_names(sm; include_tp = true, include_gq = false)
    tp_draws = c.draws[:, [c.col[n] for n in tp_names]]
    @test_throws "include_gq=true" brm_ranef_draws(c.d, tp_draws, tp_names; id = :q)
end

@testset "known scales and correlations are reported as their values" begin
    c = constrained_draws(build(known_mixed_builder))
    p = only(brm_ranef_draws(c.d, c.draws, c.names; id = :p))
    @test p.family === :ranef_correlated_draws_known_scale
    @test all(p.sd[:, 1] .≈ sqrt(0.1))
    @test check_rows(c, p, "r_mu_p_subject", 1:2, Zboth, data.subject)
    @test check_noncentered(c, p, "b_p_subject_z")

    c = constrained_draws(build(known_all_builder))
    p = only(brm_ranef_draws(c.d, c.draws, c.names; id = :p))
    @test p.family === :ranef_correlated_draws_known_cor
    @test all(p.sd .≈ 0.3)
    @test all(isapprox(p.cor[i, :, :], [1 0.3; 0.3 1]; atol = 1e-12) for i in 1:3)
    @test check_rows(c, p, "r_mu_p_subject", 1:2, Zboth, data.subject)
    @test check_noncentered(c, p, "b_p_subject_z")

    # The centered emissions report the same values and read the sampled effects.
    c = constrained_draws(build(known_mixed_builder; total_groups = (), centered_groups = [:subject]))
    p = only(brm_ranef_draws(c.d, c.draws, c.names; id = :p))
    @test p.family === :ranef_correlated_draws_centered_known_scale
    @test all(p.sd[:, 1] .≈ sqrt(0.1))
    @test check_rows(c, p, "r_mu_p_subject", 1:2, Zboth, data.subject)
    c = constrained_draws(build(known_all_builder; total_groups = (), centered_groups = [:subject]))
    p = only(brm_ranef_draws(c.d, c.draws, c.names; id = :p))
    @test p.family === :ranef_correlated_draws_centered_known_cor
    @test all(p.sd .≈ 0.3)
    @test all(isapprox(p.cor[i, :, :], [1 0.3; 0.3 1]; atol = 1e-12) for i in 1:3)
    @test check_rows(c, p, "r_mu_p_subject", 1:2, Zboth, data.subject)
end

@testset "R2D2-derived scales" begin
    c = constrained_draws(build(r2d2_bucket_builder; total_groups = ()))
    p = only(brm_ranef_draws(c.d, c.draws, c.names; id = :p))
    @test p.family === :ranef_correlated_draws_r2d2
    @test p.sd == c.draws[:, [c.col["b_p_subject_r2d2_tau.$k"] for k in 1:2]]
    @test check_noncentered(c, p, "b_p_subject_z")

    c = constrained_draws(build(r2d2_flat_builder; total_groups = ()))
    s = only(brm_ranef_draws(c.d, c.draws, c.names; group = :subject))
    @test s.family === :ranef_intercept_r2d2
    @test s.sd[:, 1] == c.draws[:, c.col["r_mu_subject_r2d2_scale"]]
    @test check_rows(c, s, "r_mu_subject", 1:1, Zint, data.subject)
end

@testset "multi-membership intercept" begin
    c = constrained_draws(build(mm_builder; total_groups = ()))
    b = only(brm_ranef_draws(c.d, c.draws, c.names; group = :g2))
    @test b.family === :ranef_intercept_draws && b.group == (:g1, :g2)
    binding = String(b.binding)
    for i in 1:3, g in eachindex(b.levels)
        @test b.effects[i, g, 1] ≈ b.sd[i, 1] * value(c, i, "$(binding)_xi.$g") atol = 1e-12
    end
    @test b.sd[:, 1] ≈ exp.(c.draws[:, c.col["$(binding)_log_scale"]])
end

@testset "an omitted response: the generated completion of an |ID| block" begin
    c = constrained_draws(build(shared_builder, no_z; total_groups = ()))
    p = only(brm_ranef_draws(c.d, c.draws, c.names; id = :p))
    @test p.family === :ranef_correlated_draws_completion
    @test length(p.margins) == 3
    # The sampled active margins (mu's) are carried into the completion.
    active = c.draws[:, vec([c.col["b_p_subject_active.$g.$k"] for g in 1:6, k in 1:2])]
    @test reshape(p.effects[:, :, 1:2], 3, 12) ≈ active atol = 1e-12
    @test check_rows(c, p, "r_mu_p_subject", 1:2, Zboth, data.subject)
    @test check_rows(c, p, "r_nu_p_subject", 3:3, Zint, data.subject)
    @test check_noncentered(c, p, "b_p_subject_z")
    # Generated, so names without generated quantities are refused.
    tp_names = BS.param_names(c.sm; include_tp = true, include_gq = false)
    tp_draws = c.draws[:, [c.col[n] for n in tp_names]]
    @test_throws "include_gq=true" brm_ranef_draws(c.d, tp_draws, tp_names; id = :p)
end

@testset "stratified gr(g, by=b): one entry per stratum" begin
    sb = build(stratified_builder, strata; total_groups = ())
    c = constrained_draws(sb)
    blocks = brm_ranef_draws(c.d, c.draws, c.names; group = :site)
    @test [(b.by, b.stratum) for b in blocks] == [(:group_of_site, "u"), (:group_of_site, "v")]
    @test [b.levels for b in blocks] == [["c"], ["a", "b"]]
    @test all(b.binding === blocks[1].binding && b.family === :ranef_correlated_by &&
              b.group === :site && b.id === nothing for b in blocks)
    @test all(b.margins == [(predictor = :mu, coefficient = :Intercept),
                            (predictor = :mu, coefficient = :x)] for b in blocks)
    @test [size(b.sd) for b in blocks] == [(3, 2), (3, 2)]
    @test [size(b.cor) for b in blocks] == [(3, 2, 2), (3, 2, 2)]
    @test [size(b.effects) for b in blocks] == [(3, 1, 2), (3, 2, 2)]
    @test check_rows(c, blocks, "$(blocks[1].binding)", 1:2, Zboth, data.site)
    all_levels = only(b for b in ranef_blocks(sb) if b.binding === blocks[1].binding).levels
    z = "$(blocks[1].binding)_b_T_z_g"
    for b in blocks
        @test check_noncentered(c, b, z; zlevels = level_of(all_levels, b.levels))
        @test all(isapprox(b.cor[i, k, k], 1.0; atol = 1e-12) for i in 1:3, k in 1:2)
    end
    # Each stratum reports its own scales: Stan's per-group gather of its
    # stratum's `tau` (`b_T_tau_g`, through the stratum index) equals the SDs
    # of the entry that holds the group.
    @test blocks[1].sd != blocks[2].sd
    for b in blocks, g in level_of(all_levels, b.levels), i in 1:3, k in 1:2
        @test b.sd[i, k] == value(c, i, "$(b.binding)_b_T_tau_g.$k.$g")
    end

    sb = build(stratified_id_builder, strata; total_groups = ())
    c = constrained_draws(sb)
    blocks = brm_ranef_draws(c.d, c.draws, c.names; id = :p)
    @test [(b.by, b.stratum) for b in blocks] == [(:arm, "c"), (:arm, "t")]
    @test [b.levels for b in blocks] == [["s4", "s5", "s6"], ["s1", "s2", "s3"]]
    @test all(b.family === :ranef_correlated_by_draws for b in blocks)
    @test all(b.margins == [(predictor = :mu, coefficient = :Intercept),
                            (predictor = :mu, coefficient = :x),
                            (predictor = :nu, coefficient = :Intercept)] for b in blocks)
    @test length(brm_ranef_draws(c.d, c.draws, c.names; group = :subject)) == 2
    binding = String(blocks[1].binding)
    all_levels = only(b for b in ranef_blocks(sb) if b.binding === blocks[1].binding).levels
    @test check_rows(c, blocks, replace(binding, r"^b_" => "r_mu_"), 1:2, Zboth, data.subject)
    @test check_rows(c, blocks, replace(binding, r"^b_" => "r_nu_"), 3:3, Zint, data.subject)
    for b in blocks
        @test check_noncentered(c, b, "$(binding)_b_T_z_g";
                                zlevels = level_of(all_levels, b.levels))
    end
end

@testset "transformed parameters are required" begin
    c = constrained_draws(build(shared_builder; total_groups = ()))
    par_names = BS.param_names(c.sm; include_tp = false, include_gq = false)
    par_draws = c.draws[:, [c.col[n] for n in par_names]]
    @test_throws "include_tp=true" brm_ranef_draws(c.d, par_draws, par_names; id = :p)
end

end # RUNTIME
