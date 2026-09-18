# Does the totals recovery route serve KERNEL models? A kernel fit whose
# subject block integrates into totals under the default `:auto` cannot use
# `reprocess(...; resample_groups=...)` (refused for totals models) — so its
# only new-subject route is `generative_plan` + `transport_draws` recovery.
# This file proves that route end to end: old groups keep their totals, new
# groups share one recovered population draw, and the moved draws evaluate on
# the compiled target. (Question todo 0o4ipvg; the `total_groups=()`+GQ
# alternative stays documented in brm-use for conventionally-built fits.)
#
# Run: julia --project=<worktree> test/total_effects_kernel_route.jl

using Test
using BayesianRegressionModels
using StanBlocks
using Random
using Statistics: mean
using Distributions: Exponential, Normal

const BS = StanBlocks.BridgeStan
const BRM = BayesianRegressionModels

kernel_totals_builder = @brm begin
    sigma  ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    pred   ~ kernel(ragged(pk_time, pk_subject), log_CL) do ts, lCL
        exp(-exp(lCL) .* ts)
    end
    ragged(pk_y, pk_subject) ~ Normal(pred, sigma)
end

kernel_totals_df(; subs, seed = 1) = (;
    subject    = subs,
    weight     = collect(range(60.0, 90.0; length = length(subs))),
    pk_subject = vcat([fill(s, 2) for s in subs]...),
    pk_time    = repeat([0.5, 1.0], length(subs)),
    pk_y       = rand(MersenneTwister(seed), 2 * length(subs)),
)

@testset "kernel totals — :auto integrates the subject bucket" begin
    train = kernel_totals_df(; subs = ["s1", "s2", "s3"])
    sb = SBBRMI(kernel_totals_builder(train); mod = @__MODULE__)
    # The premise of the question: default `:auto` claims this shape, so the
    # GQ resample route is closed and recovery is the only way through.
    @test !isempty(total_effect_blocks(sb))
    @test_throws ArgumentError reprocess(
        sb, kernel_totals_df(; subs = ["n1", "n2"]); resample_groups = [:subject])
end

if get(ENV, "BRM_TOTALS_KERNEL_RUNTIME", "1") == "1"
@testset "kernel totals — recovery transports onto new subjects" begin
    train  = kernel_totals_df(; subs = ["s1", "s2", "s3"])
    future = kernel_totals_df(; subs = ["s2", "s3", "n1", "n2"], seed = 7)
    sb = SBBRMI(kernel_totals_builder(train); mod = @__MODULE__)
    new_plan = generative_plan(kernel_totals_builder, future; mod = @__MODULE__)

    fit_p = StanBlocks.stan_instantiate(sb.model)
    new_p = StanBlocks.stan_instantiate(new_plan.model)
    unc_train = BS.param_unc_names(fit_p.model)
    unc_new = BS.param_unc_names(new_p.model)
    # One draw repeated: the recovered conditional is then fixed, so the new
    # groups' sample mean must match it (same design as
    # test/total_effects_integration.jl).
    q = randn(MersenneTwister(11), BS.param_unc_num(fit_p.model))
    draws = repeat(q', 1000, 1)

    totals_from = total_effect_blocks(sb)
    totals_to = total_effect_blocks(new_plan)
    @test length(totals_from) == length(totals_to) == 1
    bf = only(totals_from)
    bt = only(totals_to)
    c_from = BRM._total_coordinates(sb, bf, unc_train)
    c_to = BRM._total_coordinates(new_plan, bt, unc_new)
    conditional = BRM._total_conditional(bf, c_from, q)

    moved = transport_draws(sb, new_plan, draws, unc_train, unc_new;
                            rng = Xoshiro(23))
    lf = sb.preproc[bf.group_index].const_.levels
    lt = new_plan.preproc[bt.group_index].const_.levels
    newcols = Int[]
    for (g, lvl) in enumerate(lt)
        gf = findfirst(==(lvl), lf)
        if isnothing(gf)
            push!(newcols, g)
        else
            # A retained group keeps its fitted totals verbatim.
            @test moved[:, c_to.totals[g, :]] == draws[:, c_from.totals[gf, :]]
        end
    end
    # Each new group is drawn around the ONE shared recovered population draw
    # (with independent deviation noise per group — means match, values differ).
    @test length(newcols) == 2
    for g in newcols
        @test vec(mean(moved[:, c_to.totals[g, :]]; dims = 1)) ≈
            bf.A * conditional.mean atol = 0.15
    end
    @test all(i -> isfinite(BS.log_density(new_p.model, moved[i, :])),
              1:size(moved, 1))

    # `resample=:subject` additionally re-draws the RETAINED groups around the
    # same shared recovered draw.
    loo = transport_draws(sb, new_plan, draws, unc_train, unc_new;
                          resample = :subject, rng = Xoshiro(23))
    for g in 1:length(lt)
        @test vec(mean(loo[:, c_to.totals[g, :]]; dims = 1)) ≈
            bf.A * conditional.mean atol = 0.15
    end
    # ...so the retained groups no longer carry their fitted values.
    oldcols = [g for (g, lvl) in enumerate(lt) if lvl in lf]
    @test !isempty(oldcols)
    @test loo[:, c_to.totals[oldcols[1], :]] != moved[:, c_to.totals[oldcols[1], :]]
    @test all(i -> isfinite(BS.log_density(new_p.model, loo[i, :])),
              1:size(loo, 1))
end
end
