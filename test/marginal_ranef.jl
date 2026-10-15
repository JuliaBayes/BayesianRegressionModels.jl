# test/marginal_ranef.jl — likelihood-inactive margins of a shared |ID| block.
#
# Run: julia --project=test test/marginal_ranef.jl
#
# Omitting a response can leave some margins of a shared `(… | p | g)` block
# outside every likelihood. SBBRMI then samples only the active margins'
# marginal block, `LKJ(a, eta + (K - a) / 2)`, and completes the inactive ones
# exactly in generated quantities (src/sb_marginal_ranef.jl).
using Test
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using LinearAlgebra
using Random
using Statistics
using Distributions: Beta, Exponential, LKJCholesky, Normal
import BridgeStan

const MR = BayesianRegressionModels

const MR_DF = (;
    subject = [1, 1, 2, 2, 3, 3, 4, 4],
    x = [-1.0, 0.5, -0.5, 1.0, 0.0, 1.5, -1.5, 0.25],
    ya = [0.3, 0.9, -0.2, 0.4, 0.1, 1.2, -0.8, 0.0],
    yb = [1.1, 0.7, 0.2, 0.5, -0.3, 0.6, 0.9, 1.4],
    yc = [-0.4, -0.1, 0.8, 0.3, 0.5, 0.2, -0.6, 0.1],
)
# `yb` omitted: margin 3 (`b_loc`) of the four `|p|` margins is inactive, in the
# middle of the margin order.
const MR_OMIT_B = (; MR_DF.subject, MR_DF.x, MR_DF.ya, MR_DF.yc)

const MR_BUILDER = @brm begin
    a_loc ~ 1 + x + (1 + x | p | subject)
    b_loc ~ 1 + (1 | p | subject)
    c_loc ~ 1 + (1 | p | subject)
    ya ~ Normal(a_loc, 0.5)
    yb ~ Normal(b_loc, 0.5)
    yc ~ Normal(c_loc, 0.5)
end

const MR_CONFIGURED = @brm begin
    a_loc ~ 1 + x + (1 + x | p | subject)
    b_loc ~ 1 + (1 | p | subject)
    c_loc ~ 1 + (1 | p | subject)
    sd(:, p) ~ Exponential(0.5)
    sd(b_loc, p) ~ Exponential(0.2)
    cor(:, p) ~ LKJCholesky(4, 2.0)
    ya ~ Normal(a_loc, 0.5)
    yb ~ Normal(b_loc, 0.5)
    yc ~ Normal(c_loc, 0.5)
end

mr_build(builder, df; kwargs...) = SBBRMI(builder(df); mod=@__MODULE__, total_groups=(), kwargs...)

function mr_block(code, name)
    m = match(Regex("(?ms)^$name \\{(.*?)^\\}"), code)
    isnothing(m) && error("missing `$name` block")
    only(m.captures)
end

# The parameter value BridgeStan reports under `name.i.j...`, reshaped.
function mr_value(names, values, stem, dims...)
    out = Array{Float64}(undef, dims...)
    for I in CartesianIndices(out)
        out[I] = values[findfirst(==(join((stem, Tuple(I)...), ".")), names)]
    end
    out
end

# The onion completion of a fixed active Cholesky factor, in active-first order.
function mr_onion(LA, eta, K, rng)
    a = size(LA, 1)
    Lp = zeros(K, K)
    Lp[1:a, 1:a] = LA
    for r in 1:(K - a)
        j = a + r
        y = rand(rng, Beta((j - 1) / 2, eta + (K - j) / 2))
        u = randn(rng, j - 1)
        Lp[j, 1:(j - 1)] = sqrt(y) .* u ./ norm(u)
        Lp[j, j] = sqrt(1 - y)
    end
    Lp
end

function mr_ks(x, y)
    xs, ys = sort(x), sort(y)
    maximum(abs(searchsortedlast(xs, g) / length(xs) - searchsortedlast(ys, g) / length(ys))
            for g in vcat(xs, ys))
end
# Two-sample Kolmogorov-Smirnov critical value at level 0.001.
mr_ks_crit(n, m) = 1.95 * sqrt((n + m) / (n * m))

@testset "the onion completion of LKJ(a, eta + m/2) is LKJ(K, eta)" begin
    rng = Xoshiro(20261015)
    K, eta, N = 4, 1.7, 20_000
    active, inactive = [1, 2, 4], [3]
    order = vcat(active, inactive)
    corr(L) = L * L'
    full = [corr(Matrix(rand(rng, LKJCholesky(K, eta)).L)) for _ in 1:N]
    split = map(1:N) do _
        Lp = mr_onion(Matrix(rand(rng, LKJCholesky(length(active), eta + 0.5)).L), eta, K, rng)
        O = zeros(K, K)
        O[order, order] = corr(Lp)
        O
    end
    for (i, j) in ((1, 2), (1, 3), (2, 3), (3, 4), (2, 4))
        @test mr_ks([O[i, j] for O in full], [O[i, j] for O in split]) < mr_ks_crit(N, N)
    end
    @test mr_ks(det.(full), det.(split)) < mr_ks_crit(N, N)
end

@testset "fully observed responses keep the ordinary shared block" begin
    sb = mr_build(MR_BUILDER, MR_DF)
    code = MR.stan_code(sb)
    @test occursin("cholesky_factor_corr[n_terms_p_subject] b_p_subject_L;",
                   mr_block(code, "parameters"))
    @test !occursin("b_p_subject_active", code)
end

@testset "an omitted response integrates its margins out of the sampled block" begin
    sb = @test_logs (:warn, r"`yb` bind\(s\) no data column") mr_build(MR_BUILDER, MR_OMIT_B)
    code = MR.stan_code(sb)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    params = mr_block(code, "parameters")
    @test occursin("cholesky_factor_corr[n_terms_p_subject_active] b_p_subject_active_L;", params)
    @test occursin("b_p_subject_active_tau;", params)
    @test occursin("b_p_subject_active_z_flat;", params)
    @test !occursin(r"\bb_p_subject_(L|tau|z_flat);", params)
    # LKJ(4, 1) restricted to three margins.
    @test occursin("b_p_subject_active_L ~ lkj_corr_cholesky(1.5);", code)
    gq = mr_block(code, "generated quantities")
    for name in ("b_p_subject_L", "b_p_subject_tau", "b_p_subject_z_flat", "b_p_subject")
        @test occursin(Regex("\\b$name ="), gq)
    end
    @test sb.data[:b_p_subject_active_margins] == [1, 2, 4]
    @test sb.data[:b_p_subject_inactive_margins] == [3]
    @test sb.data[:b_p_subject_onion_alpha] == [1.5]
    @test sb.data[:b_p_subject_onion_beta] == [1.0]

    problem = StanBlocks.stan_instantiate(sb.model)
    # Active block 3 + 3 + 3*4, population a_loc 2 and c_loc 1. The full block
    # would sample 6 + 4 + 4*4.
    @test LogDensityProblems.dimension(problem) == 21

    model = problem.model
    unc = BridgeStan.param_unc_names(model)
    names = BridgeStan.param_names(model; include_tp=true, include_gq=true)
    rng = Xoshiro(7)
    q = 0.3 .* randn(rng, length(unc))
    draws = [BridgeStan.param_constrain(model, q; include_tp=true, include_gq=true,
                                        rng=BridgeStan.StanRNG(model, seed))
             for seed in 1:4000]
    LA = mr_value(names, first(draws), "b_p_subject_active_L", 3, 3)
    tauA = mr_value(names, first(draws), "b_p_subject_active_tau", 3)
    bA = mr_value(names, first(draws), "b_p_subject_active", 4, 3)
    for d in draws
        L = mr_value(names, d, "b_p_subject_L", 4, 4)
        tau = mr_value(names, d, "b_p_subject_tau", 4)
        b = mr_value(names, d, "b_p_subject", 4, 4)
        z = reshape(mr_value(names, d, "b_p_subject_z_flat", 16), 4, 4)
        @test istril(L)
        @test (L * L')[[1, 2, 4], [1, 2, 4]] ≈ LA * LA' atol=1e-10
        @test tau[[1, 2, 4]] == tauA
        @test b[:, [1, 2, 4]] == bA
        @test b ≈ (Diagonal(tau) * L * z)' atol=1e-10
    end
    # The generated inactive margin against the exact conditional law.
    reference = Xoshiro(11)
    zA = (Diagonal(tauA) * LA) \ bA'
    expected_corr = Float64[]
    expected_b = Float64[]
    for _ in 1:20_000
        Lp = mr_onion(LA, 1.0, 4, reference)
        push!(expected_corr, (Lp * Lp')[4, 1])
        tauI = abs(randn(reference))
        push!(expected_b, tauI * (dot(Lp[4, 1:3], zA[:, 2]) + Lp[4, 4] * randn(reference)))
    end
    observed_corr = [(L * L')[3, 1] for L in (mr_value(names, d, "b_p_subject_L", 4, 4) for d in draws)]
    observed_b = [mr_value(names, d, "b_p_subject", 4, 4)[2, 3] for d in draws]
    @test mr_ks(observed_corr, expected_corr) < mr_ks_crit(4000, 20_000)
    @test mr_ks(observed_b, expected_b) < mr_ks_crit(4000, 20_000)
end

@testset "configured scale and correlation priors split by margin" begin
    sb = mr_build(MR_CONFIGURED, MR_OMIT_B)
    code = MR.stan_code(sb)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test occursin("b_p_subject_active_L ~ lkj_corr_cholesky(2.5);", code)
    @test sb.data[:b_p_subject_onion_beta] == [2.0]
    # The active scales keep their block-wide Exponential(0.5) prior; only the
    # inactive `b_loc` margin reads Exponential(0.2).
    @test occursin("b_p_subject_active_tau ~ exponential(", code)
    gq = mr_block(code, "generated quantities")
    @test occursin("b_p_subject_tau_inactive = ", gq)
    @test LogDensityProblems.dimension(StanBlocks.stan_instantiate(sb.model)) == 21
end

@testset "centered, cv and held-out builds reduce like the fit" begin
    fit = mr_build(MR_BUILDER, MR_OMIT_B)
    fit_unc = BridgeStan.param_unc_names(StanBlocks.stan_instantiate(fit.model).model)

    centered = mr_build(MR_BUILDER, MR_OMIT_B; centered_groups=[:subject])
    code = MR.stan_code(centered)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test occursin("b_p_subject_active_b_cols_bc", mr_block(code, "parameters"))
    @test LogDensityProblems.dimension(StanBlocks.stan_instantiate(centered.model)) == 21

    # New-population replay re-emits with cv sizing and marks the group index;
    # the probe ignores those marks, so the replay keeps the fit's reduction.
    future = (; subject=[11, 11, 12, 12, 13, 13], x=[0.2, -0.4, 1.1, 0.3, -0.9, 0.6],
              ya=zeros(6), yc=zeros(6))
    replay = reprocess(fit, future; resample_groups=:subject)
    code = MR.stan_code(replay)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    params = mr_block(code, "parameters")
    @test occursin("b_p_subject_active_L;", params)
    @test occursin("b_p_subject_active_tau;", params)
    @test !occursin("b_p_subject_active_z_flat", params)
    replay_unc = BridgeStan.param_unc_names(StanBlocks.stan_instantiate(replay.model).model)
    @test issubset(replay_unc, fit_unc)
    @test !any(startswith("b_p_subject_active_z_flat"), replay_unc)

    held = mr_build(MR_BUILDER, MR_OMIT_B; held_out=:yc)
    held_unc = BridgeStan.param_unc_names(StanBlocks.stan_instantiate(held.model).model)
    @test held_unc == fit_unc
end

@testset "random-effect blocks, scales and population draws" begin
    sb = mr_build(MR_BUILDER, MR_OMIT_B)
    blocks = ranef_blocks(sb)
    carrier = only(b for b in blocks if b.binding === :b_p_subject_active)
    completion = only(b for b in blocks if b.binding === :b_p_subject)
    @test carrier.family === :ranef_correlated_draws_marginal
    @test isnothing(carrier.id) && carrier.group === :subject
    @test carrier.n_terms == 3 && !carrier.generated
    @test completion.family === :ranef_correlated_draws_completion
    @test completion.id === :p && completion.n_terms == 4 && completion.generated

    problem = StanBlocks.stan_instantiate(sb.model)
    unc = BridgeStan.param_unc_names(problem.model)
    draws = 0.2 .* randn(Xoshiro(3), 2, length(unc))
    population = population_draws(sb, draws, unc; groups=:subject)
    zcols = findall(startswith("b_p_subject_active_z_flat."), unc)
    @test length(zcols) == 12
    @test all(iszero, population[:, zcols])
    @test population[:, setdiff(eachindex(unc), zcols)] == draws[:, setdiff(eachindex(unc), zcols)]

    d = brm_descriptor(sb)
    names = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
    scale = brm_ranef_sd_coordinates(d, :b_loc, names; id=:p)
    @test names[only(scale.coordinates)] == "b_p_subject_tau.3"
end
