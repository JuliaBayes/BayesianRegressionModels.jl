# test/kernel_linked_lp.jl — linked linear predictors as `kernel(...)`
# positionals (snag `linked-lp-kernel-66e54eca`).
#
# A linked declaration (`log(Vc) ~ ...`) binds TWO names: the LINKED-scale
# emitted vector (`log_Vc`, the fitted quantity) and the RESPONSE-scale public
# name (`Vc = exp(log_Vc)`). A `kernel(...)` positional must name the BARE
# public name — `Vc`, or `ragged(F, group)` on a secondary axis; the link
# spelling (`log(Vc)`) is rejected with a redirect to it. The plate slices the
# response-scale binding, so the cell uses it directly: re-applying the inverse
# link in the cell silently double-applies it. (The `ev_lesion_model` fixture in
# `test/kernel_event_axis_lp.jl` did exactly that before this snag's fix.)
#
# WHAT THE LOAD-BEARING ASSERTIONS ARE:
#
#   1. Both classifier sites reject the link spelling with the redirect naming
#      the bare name — not a bare-type error.
#   2. The bare-name spellings lower, transpile, and pass stanc; the LP stays a
#      parameter (absent from `sb.data`); the emitted program binds
#      `Vc = exp(log_Vc)` and the cell reads `Vc[plate_i__pl_1]`.
#   3. BridgeStan runtime: finite density/gradient for both bare-name models;
#      and the linked spelling is the SAME model as the inert bare spelling
#      (identical dimension, identical density at the same point) — passing the
#      bare name gives up nothing.
#   4. The descriptor still resolves the clean linked address (`:Vc` with
#      `link === log`), which is why a consumer declares the linked form.

using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Exponential, Normal

const LINKED_RUN_BRIDGESTAN = get(ENV, "BRM_KERNEL_RUNTIME", "1") != "0"
const LINKED_CACHE = joinpath(tempdir(), "brm-kernel-linked-lp")

function linked_df()
    (;
        subject = ["s1", "s2", "s3"],
        weight  = [60.0, 75.0, 90.0],
        t_obs   = [abs.(sin.(1:4)) .+ 0.5 for _ in 1:3],
        dv      = [abs.(cos.(1:4)) .+ 0.1 for _ in 1:3],
        dose_subject = ["s1", "s1", "s2", "s2", "s2", "s3", "s3"],
        diet = [1, 2, 1, 3, 2, 3, 1],
    )
end

# The reporter's tried spelling — a link call as a plain positional.
linked_plain_guess_model(df) = @brm df begin
    sigma ~ Exponential(1)
    log(Vc) ~ 1 + weight + (1 | p | subject)
    pred ~ kernel(t_obs, dv, log(Vc)) do ts, yy, lV
        mu = exp(lV) .* exp(-ts)
        yy ~ normal(mu, sigma)
        mu
    end
end

# The reporter's tried spelling inside `ragged(...)`.
linked_ragged_guess_model(df) = @brm df begin
    sigma ~ Exponential(1)
    log(F) ~ 1 + factor(diet)
    log(CL) ~ 1 + weight + (1 | p | subject)
    pred ~ kernel(t_obs, dv, ragged(log(F), dose_subject), CL) do ts, yy, lF, cCL
        mu = sum(exp.(lF)) .* exp(-cCL .* ts)
        yy ~ normal(mu, sigma)
        mu
    end
end

# The working spelling — the bare public name, on the response scale.
linked_plain_model(df) = @brm df begin
    sigma ~ Exponential(1)
    log(Vc) ~ 1 + weight + (1 | p | subject)
    pred ~ kernel(t_obs, dv, Vc) do ts, yy, vV
        mu = vV .* exp(-ts)
        yy ~ normal(mu, sigma)
        mu
    end
end

linked_ragged_model(df) = @brm df begin
    sigma ~ Exponential(1)
    log(F) ~ 1 + factor(diet)
    log(CL) ~ 1 + weight + (1 | p | subject)
    pred ~ kernel(t_obs, dv, ragged(F, dose_subject), CL) do ts, yy, vF, cCL
        mu = sum(vF) .* exp(-cCL .* ts)
        yy ~ normal(mu, sigma)
        mu
    end
end

# The inert bare-name control — the same fitted quantity, link spelled in cell.
linked_bare_model(df) = @brm df begin
    sigma ~ Exponential(1)
    log_Vc ~ 1 + weight + (1 | p | subject)
    pred ~ kernel(t_obs, dv, log_Vc) do ts, yy, lV
        mu = exp(lV) .* exp(-ts)
        yy ~ normal(mu, sigma)
        mu
    end
end

linked_problem(sb, tag) = begin
    isdir(LINKED_CACHE) || mkpath(LINKED_CACHE)
    code = StanBlocks.stan_code(sb.model)
    StanBlocks.stan_instantiate(sb.model; path = joinpath(LINKED_CACHE, "$(tag)_$(hash(code)).stan"))
end

@testset "kernel(...) — linked linear-predictor positionals" begin
    df = linked_df()

    @testset "the link spelling is rejected with a redirect, not a bare-type error" begin
        @test_throws "pass the bare name `Vc`" SBBRMI(
            linked_plain_guess_model(df); mod = @__MODULE__)
        @test_throws "pass the bare name `F`" SBBRMI(
            linked_ragged_guess_model(df); mod = @__MODULE__)
        # The redirect names the response-scale binding, so the consumer knows
        # to drop the inverse-link call from the cell.
        @test_throws "Vc = exp(log_Vc)" SBBRMI(
            linked_plain_guess_model(df); mod = @__MODULE__)
    end

    @testset "the bare name lowers on the response scale" begin
        sb = SBBRMI(linked_plain_model(df); mod = @__MODULE__)
        # The LP stays a parameter, never registered as data.
        @test !haskey(sb.data, :Vc)
        @test !haskey(sb.data, :log_Vc)
        @test sb.data[:kernel_nsub_pred] == 3

        @test StanBlocks.stan.transpiles(sb.model)
        code = StanBlocks.stan_code(sb.model)
        @test occursin(r"Vc\s*=\s*exp\(log_Vc\)", code)
        # The cell slices the response-scale binding, not the linked vector.
        @test occursin(r"Vc\[\s*plate_i__pl_1\s*\]", code)
        @test StanBlocks.stanc_check(code; warn_pedantic = false).ok

        rsb = SBBRMI(linked_ragged_model(df); mod = @__MODULE__)
        @test !haskey(rsb.data, :F)
        @test rsb.data[:kernel_pred_F_ragged] == [[1, 2], [3, 4, 5], [6, 7]]
        @test StanBlocks.stan.transpiles(rsb.model)
        rcode = StanBlocks.stan_code(rsb.model)
        @test occursin(r"F\[\s*kernel_pred_F_ragged\.1\[", rcode)
        @test StanBlocks.stanc_check(rcode; warn_pedantic = false).ok
    end

    @testset "BridgeStan runtime — the linked spelling is the same model" begin
        if LINKED_RUN_BRIDGESTAN
            using LogDensityProblems
            sb = SBBRMI(linked_plain_model(df); mod = @__MODULE__)
            rsb = SBBRMI(linked_ragged_model(df); mod = @__MODULE__)
            bare = SBBRMI(linked_bare_model(df); mod = @__MODULE__)

            for (tag, msb) in (("linked", sb), ("ragged", rsb), ("bare", bare))
                prob = linked_problem(msb, tag)
                dim = LogDensityProblems.dimension(prob)
                q = [0.1 * ((i % 5) - 2) for i in 1:dim]
                lp, g = LogDensityProblems.logdensity_and_gradient(prob, q)
                @test isfinite(lp)
                @test length(g) == dim
                @test all(isfinite, g)
            end

            # Identical dimension and identical density at the same point: the
            # bare-name spelling gives up nothing over the inert bare form.
            prob = linked_problem(sb, "linked")
            bprob = linked_problem(bare, "bare")
            dim = LogDensityProblems.dimension(prob)
            @test LogDensityProblems.dimension(bprob) == dim
            q = [0.1 * ((i % 5) - 2) for i in 1:dim]
            lp, g = LogDensityProblems.logdensity_and_gradient(prob, q)
            blp, bg = LogDensityProblems.logdensity_and_gradient(bprob, q)
            @test isapprox(lp, blp; atol = 1e-12, rtol = 0)
            @test maximum(abs.(g .- bg)) <= 1e-12

            # The clean linked address survives: `:Vc` with the log/exp pair.
            # GQ-inclusive names, since the recovered carrier may live there.
            d = brm_descriptor(sb; name = :linked_lp_kernel, highlights = ())
            cnames = StanBlocks.BridgeStan.param_names(
                prob.model; include_tp = true, include_gq = true)
            r = brm_population_effect_coordinates(d, :Vc, cnames)
            @test r.logical === :Vc
            @test r.link === log
            @test r.inverse_link === exp
            @test !isempty(r.coordinates)
        else
            @info "Skipping BridgeStan runtime gate (BRM_KERNEL_RUNTIME=0)"
        end
    end
end
