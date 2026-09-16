# test/probe_v2_lp_arg.jl — kernel(...) v2 blocker: a LATENT per-subject linear
# predictor as a positional kernel arg (todo `0h5cs3w`, GO `0dnesv9`).
#
# `log_CL` / `log_V` are NOT dataframe columns — they are formula statements with
# their own `|p|` ranef bucket. Before this, `_sb_kernel_doblock!` rejected them
# with "positional args (after the do-block) must be data columns", so a cell
# could only ever receive raw data and the typical values had to be hard-coded
# inside the cell body. Lifting that is what the whole v2 surface stands on.
#
# WHY THE `sb.data` ASSERTION IS THE LOAD-BEARING ONE: the tempting fix is to
# register the LP like any other column. That transpiles, passes stanc, and
# samples — while silently shadowing the sampled parameter with a constant. Text
# checks cannot see it. Absence from `sb.data` plus a finite BridgeStan GRADIENT
# is what actually pins the parameter as live.
#
# Grouping is derived from the LPs' shared ranef grouping (`0xuaz0k`): `by=` is
# gone, disagreeing LP groupings fail loudly, and a kernel with NO per-subject LP
# fails loudly (decision `1kg5340`).

using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Exponential

const V2_RUN_BRIDGESTAN = get(ENV, "BRM_KERNEL_RUNTIME", "1") != "0"
const V2_N = 6

# Pre-grouped per-subject frame (the kernel contract): one row per subject, so an
# ordinary LP over it is ALREADY length n_subjects — no second LP shape needed.
function v2_df(order = collect(1:V2_N))
    base = (;
        t       = [abs.(sin.(1:4)) .+ 0.5 for _ in 1:V2_N],
        dose    = fill(100.0, V2_N),
        dv      = [abs.(cos.(1:4)) .+ 0.1 for _ in 1:V2_N],
        weight  = collect(range(60.0, 90.0; length = V2_N)),
        subject = ["s$i" for i in 1:V2_N],
        site    = ["site$i" for i in V2_N:-1:1],
    )
    (; (k => v[order] for (k, v) in pairs(base))...)
end

v2_lp_model(df) = @brm df begin
    sigma  ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    log_V  ~ 1 + (1 | p | subject)
    pred   ~ kernel(t, dose, dv, log_CL, log_V) do ts, d, yy, lCL, lV
        CL = exp(lCL); Vc = exp(lV); Ka = 1.5
        ke = CL / Vc
        mu = d * Ka / (Vc * (Ka - ke)) * (exp(-ke * ts) - exp(-Ka * ts))
        yy ~ normal(mu, sigma)
        mu
    end
end

v2_no_lp_model(df) = @brm df begin
    sigma ~ Exponential(1)
    pred  ~ kernel(t, dose, dv) do ts, d, yy
        mu = d * exp(-ts)
        yy ~ normal(mu, sigma)
        mu
    end
end

v2_by_model(df) = @brm df begin
    sigma  ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    pred   ~ kernel(t, dose, dv, log_CL; by = subject) do ts, d, yy, lCL
        mu = d * exp(-exp(lCL) * ts)
        yy ~ normal(mu, sigma)
        mu
    end
end

v2_neta_model(df) = @brm df begin
    sigma  ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    pred   ~ kernel(t, dose, dv, log_CL; n_eta = 1) do ts, d, yy, lCL
        mu = d * exp(-exp(lCL) * ts)
        yy ~ normal(mu, sigma)
        mu
    end
end

v2_disagreeing_model(df) = @brm df begin
    sigma  ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    log_V  ~ 1 + (1 | q | site)
    pred   ~ kernel(t, dose, dv, log_CL, log_V) do ts, d, yy, lCL, lV
        mu = d * exp(-exp(lCL - lV) * ts)
        yy ~ normal(mu, sigma)
        mu
    end
end

v2_two_bucket_model(df) = @brm df begin
    sigma  ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    log_V  ~ 1 + (1 | q | subject)
    pred   ~ kernel(t, dose, dv, log_CL, log_V) do ts, d, yy, lCL, lV
        mu = d * exp(-exp(lCL - lV) * ts)
        yy ~ normal(mu, sigma)
        mu
    end
end

v2_link_contract_model(df) = @brm df begin
    sigma             ~ Exponential(1)
    eta_additive      ~ 0 + (1 | p | subject)
    eta_log_location  ~ 0 + (1 | q | subject)
    eta_log_slope     ~ 0 + (1 | r | subject)
    eta_raw_magnitude ~ 0 + (1 | s | subject)
    eta_shifted       ~ 1 + (1 | u | subject)
    pred ~ kernel(
        dv,
        eta_additive, eta_log_location, eta_log_slope, eta_raw_magnitude, eta_shifted,
    ) do yy, additive, log_location, log_slope, raw_magnitude, shifted
        mu = additive + exp(log_location) + exp(log_slope) + raw_magnitude + shifted
        yy ~ normal(mu, sigma)
        mu
    end
end

@testset "kernel(...) v2 — latent per-subject LP as a positional arg" begin
    df = v2_df()
    sb = SBBRMI(v2_lp_model(df); mod = @__MODULE__)

    @testset "grouping is derived from one shared LP grouping" begin
        # A kernel with NO per-subject LP is no longer an error — it is a
        # no-random-effects panel whose grouping comes from the pre-grouped
        # columns' length (snag `a-hierarchical-b-78a26fe9`, covered in its own
        # testset below). What stays rejected are the retired kwargs.
        # Retired kwargs are rejected at CONSTRUCTION, not at lowering: these
        # assert the builder call itself throws, with NO `SBBRMI` in sight.
        # Previously only `SBBRMI(...)` objected, so a consumer gate that stopped
        # at BRMI construction saw the retired v1 spelling pass silently for days
        # (snag `by-and-n-eta-are-3625f645`).
        @test_throws "no longer accepts `by=`" v2_by_model(df)
        @test_throws "no longer accepts `n_eta=`" v2_neta_model(df)
        # `by=` is retired for `kernel(...)` ONLY — it stays live elsewhere.
        @test (@brm df begin
            y ~ 1 + hsgp(weight; by = subject)
        end) isa BRMI
        @test_throws "disagree on their grouping" SBBRMI(
            v2_disagreeing_model(df); mod = @__MODULE__)
        @test SBBRMI(v2_two_bucket_model(df); mod = @__MODULE__) isa SBBRMI
    end

    @testset "LPs have no implicit link; zero-mean ranef-only LPs stay zero-mean" begin
        link_brmi = v2_link_contract_model(df)
        for nm in (:eta_additive, :eta_log_location, :eta_log_slope, :eta_raw_magnitude)
            @test BayesianRegressionModels.popcoefnames(link_brmi, nm) == Symbol[]
        end
        @test BayesianRegressionModels.popcoefnames(link_brmi, :eta_shifted) == [:Intercept]

        link_code = StanBlocks.stan_code(SBBRMI(link_brmi; mod = @__MODULE__).model)
        @test occursin(r"eta_additive\s*=\s*r_eta_additive_p_subject", link_code)
        @test occursin(r"eta_log_location\s*=\s*r_eta_log_location_q_subject", link_code)
        @test occursin(r"eta_log_slope\s*=\s*r_eta_log_slope_r_subject", link_code)
        @test occursin(r"eta_raw_magnitude\s*=\s*r_eta_raw_magnitude_s_subject", link_code)
        @test occursin(
            r"eta_shifted\s*=\s*\(?\s*pop_eta_shifted\s*\+\s*r_eta_shifted_u_subject\s*\)?",
            link_code,
        )
        @test occursin(r"exp\s*\(\s*eta_log_location(?:\[[^\]]+\])?\s*\)", link_code)
        @test occursin(r"exp\s*\(\s*eta_log_slope(?:\[[^\]]+\])?\s*\)", link_code)
        @test !occursin("exp(additive)", link_code)
        @test !occursin("exp(raw_magnitude)", link_code)
    end

    @testset "LPs and structural string labels are not Stan data" begin
        for nm in (:log_CL, :log_V)
            @test !haskey(sb.data, nm)
        end
        @test !haskey(sb.data, :subject)
        @test sb.data[:subject_idx] == collect(1:V2_N)
        @test sb.data[:n_subject] == V2_N
    end

    @testset "transpile + stanc" begin
        @test StanBlocks.stan.transpiles(sb.model)
        code = StanBlocks.stan_code(sb.model)
        for nm in ("log_CL", "log_V")
            @test occursin(nm, code)
        end
        @test !occursin("kernel_L_", code)
        @test !occursin("kernel_om_", code)
        @test !occursin("kernel_z_", code)
        @test !occursin(r"(^|[^A-Za-z0-9_])_[A-Za-z]", code)
        @test !occursin("vector[\"", code)
        @test StanBlocks.stanc_check(code; warn_pedantic = false).ok
    end

    @testset "BridgeStan runtime — the LP is a live parameter" begin
        if V2_RUN_BRIDGESTAN
            using LogDensityProblems
            cache = joinpath(tempdir(), "brm-v2-lp-arg")
            isdir(cache) || mkpath(cache)
            code = StanBlocks.stan_code(sb.model)
            prob = StanBlocks.stan_instantiate(sb.model;
                                               path = joinpath(cache, string(hash(code)) * ".stan"))
            dim = LogDensityProblems.dimension(prob)
            q = [0.1 * ((i % 5) - 2) for i in 1:dim]
            lp, g = LogDensityProblems.logdensity_and_gradient(prob, q)
            @test isfinite(lp)
            @test length(g) == dim
            @test all(isfinite, g)

            # Row-order invariant: factor levels sort independently of dataframe
            # order. Permuting every row must leave the density and gradient
            # unchanged at the SAME unconstrained point; a level-order plate would
            # cross-wire each row-ordered LP with another subject's observation.
            shuffled = SBBRMI(v2_lp_model(v2_df([3, 1, 6, 2, 5, 4]));
                               mod = @__MODULE__)
            shuffled_prob = StanBlocks.stan_instantiate(
                shuffled.model; path = joinpath(cache, string(hash(code)) * ".stan"))
            shuffled_lp, shuffled_g =
                LogDensityProblems.logdensity_and_gradient(shuffled_prob, q)
            @test isapprox(shuffled_lp, lp; atol = 1e-8, rtol = 0)
            @test isapprox(shuffled_g, g; atol = 1e-8, rtol = 0)
        else
            @info "Skipping BridgeStan runtime gate (BRM_KERNEL_RUNTIME=0)"
        end
    end
end

# snag `a-hierarchical-b-78a26fe9`: a NO-random-effects panel (Charles Driver's
# ctsem fit sets `indvarying = FALSE`) has no ranef bucket to derive grouping
# from, yet the panel is real — many subjects, ALL parameters shared, each
# subject's latent path marginalized in the cell. The data arrives PRE-GROUPED
# (one entry per subject in every positional column, exactly as the pre-ragged
# `Vector{Vector}` columns `t`/`dv` already are; `dose` is scalar-per-subject),
# so the subject COUNT — the only fact the ranef path ever contributed here — is
# the columns' common length. Before this, `kernel(...)` with no per-subject LP
# failed loudly ("needs at least one per-subject linear-predictor").
@testset "kernel(...) — no-random-effects panel (grouping from pre-grouped columns)" begin
    df = v2_df()
    sb = SBBRMI(v2_no_lp_model(df); mod = @__MODULE__)
    @test sb isa SBBRMI

    @testset "subject count is the pre-grouped columns' length; no group leakage" begin
        @test sb.data[:kernel_nsub_pred] == V2_N
        # No ranef, so no group-index / subject-count-by-label machinery, and the
        # string subject column is not consulted (there is none passed to kernel).
        @test !haskey(sb.data, :subject_idx)
        @test !haskey(sb.data, :n_subject)
    end

    @testset "transpile + stanc" begin
        @test StanBlocks.stan.transpiles(sb.model)
        code = StanBlocks.stan_code(sb.model)
        @test !occursin(r"(^|[^A-Za-z0-9_])_[A-Za-z]", code)
        @test StanBlocks.stanc_check(code; warn_pedantic = false).ok
    end

    @testset "BridgeStan runtime — a shared-parameter panel is a live density" begin
        if V2_RUN_BRIDGESTAN
            using LogDensityProblems
            cache = joinpath(tempdir(), "brm-v2-no-ranef")
            isdir(cache) || mkpath(cache)
            code = StanBlocks.stan_code(sb.model)
            prob = StanBlocks.stan_instantiate(
                sb.model; path = joinpath(cache, string(hash(code)) * ".stan"))
            dim = LogDensityProblems.dimension(prob)
            q = [0.1 * ((i % 5) - 2) for i in 1:dim]
            lp, g = LogDensityProblems.logdensity_and_gradient(prob, q)
            @test isfinite(lp)
            @test length(g) == dim
            @test all(isfinite, g)
        else
            @info "Skipping BridgeStan runtime gate (BRM_KERNEL_RUNTIME=0)"
        end
    end
end
