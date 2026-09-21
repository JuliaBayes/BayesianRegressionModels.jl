# test/centered_same_subject_redraw.jl — REGRESSION for snag
# centered-populat-419018ae.
#
# The reporter's need: a CENTERED fit with plain / `|ID|` `:subject` blocks,
# replayed on a NEW schedule with the SAME subject labels and FROZEN
# preprocessing, with ALL `:subject` blocks redrawn from the fitted covariance
# — no fresh HSGP-basis refit, no reparameterization. The supported route is
# the model/draws split across the two replay halves:
#
#   sb2   = reprocess(sb, new_df)                        # frozen, same labels
#   moved = transport_draws(sb, sb2, draws, unc, unc2; resample=:subject)
#
# `resample=` on a centered conventional block is SUPPORTED (the centered
# exemption in `_ranef_fresh_draws_refused`): each level is drawn `b = C * z`
# per draw from that draw's fitted `tau` / `L`. The Stan-side twin
# (`resample_groups`) stays non-centered-only and fails loudly on a centered
# source; the Julia-side refusal stays the non-centered emission's.
#
# Run: julia --project=<worktree> test/centered_same_subject_redraw.jl
# (declaration-level only — no BridgeStan involved.)

using Test
using Random
using BayesianRegressionModels

const BRM = BayesianRegressionModels

# One shared `|p|` bucket plus one plain correlated block, both on `:subject`,
# over `zscale` population terms (whose training mean/sd make freezing
# observable: the replay frame shifts `x` by +5).
csd_builder = @brm begin
    sigma  ~ Exponential(1)
    eta_CL ~ 0 + zscale(x) + (1 | p | subject)
    eta_V  ~ 0 + zscale(x) + (1 | p | subject)
    mu     ~ 1 + zscale(t) + (1 + x | subject)
    y      ~ Normal(mu + eta_CL + eta_V, sigma)
end

function csd_df(subjects; seed, xshift = 0.0)
    rng = MersenneTwister(seed)
    n_per = 4
    n = length(subjects) * n_per
    (; subject = repeat(subjects; inner = n_per),
       x = randn(rng, n) .+ xshift,
       t = repeat(collect(1.0:n_per), length(subjects)),
       y = randn(rng, n))
end

# Synthetic unconstrained names (same construction as
# test/probe_prediction_modes.jl `centered_fake_unc`; no BridgeStan involved).
function csd_fake_unc(blocks)
    names = String["sigma", "pop_mu_beta_pop.1"]
    for b in blocks
        layout = BRM._RANEF_FAMILIES[b.family].layout
        for g in 1:b.n_groups, t in 1:b.n_terms
            push!(names, BRM._ranef_coord_name(b, layout, t, g))
        end
        K = b.n_terms
        if b.family === :ranef_intercept_centered
            push!(names, "$(b.binding)_log_scale")
        else
            append!(names, ["$(b.binding)_tau.$i" for i in 1:K])
            append!(names, ["$(b.binding)_L.$i" for i in 1:(K * (K - 1) ÷ 2)])
        end
    end
    names
end

csd_from_df = csd_df([11, 12, 13, 14]; seed = 1)
csd_to_df = csd_df([11, 12, 13, 14]; seed = 2, xshift = 5.0)
csd_from_sb = SBBRMI(csd_builder(csd_from_df); mod = @__MODULE__,
                     centered_groups = [:subject], total_groups = ())

@testset "centered same-labels frozen redraw (centered-populat-419018ae)" begin
    from_blocks = ranef_blocks(csd_from_sb)
    @test length(from_blocks) == 2
    @test all(!b.noncentered for b in from_blocks)
    @test Set(b.group for b in from_blocks) == Set([:subject])

    # Frozen-preprocessing replay on the new schedule, same labels.
    sb2 = reprocess(csd_from_sb, csd_to_df)
    @test stan_code(sb2) == stan_code(csd_from_sb)
    to_blocks = ranef_blocks(sb2)
    @test [b.family for b in to_blocks] == [b.family for b in from_blocks]
    @test [b.levels for b in to_blocks] == [b.levels for b in from_blocks]
    @test keys(sb2.preproc) == keys(csd_from_sb.preproc)
    for (k, e) in csd_from_sb.preproc
        @test isequal(sb2.preproc[k].const_, e.const_)
    end
    # Freezing is load-bearing: a fresh build on the replay frame re-derives
    # different constants (the shifted `zscale(x)` mean).
    fresh_sb = SBBRMI(csd_builder(csd_to_df); mod = @__MODULE__,
                      centered_groups = [:subject], total_groups = ())
    @test any(!isequal(fresh_sb.preproc[k].const_, sb2.preproc[k].const_)
              for (k, e) in csd_from_sb.preproc)

    # Redraw every :subject level from the fitted covariance.
    unc = csd_fake_unc(from_blocks)
    unc2 = csd_fake_unc(to_blocks)
    draws = randn(MersenneTwister(9), 4, length(unc))
    moved = transport_draws(csd_from_sb, sb2, draws, unc, unc2;
                            resample = :subject, rng = MersenneTwister(3))
    @test size(moved) == (4, length(unc2))
    @test all(isfinite, moved)
    subj_cols = Set{Int}()
    for (bf, bt) in zip(from_blocks, to_blocks)
        cf = ranef_coordinates(bf, unc)
        ct = ranef_coordinates(bt, unc2)
        union!(subj_cols, vec(ct))
        # Same labels, but `resample=` redraws ALL of them: no column copied.
        for g in 1:bt.n_groups, t in 1:bt.n_terms
            @test moved[:, ct[t, g]] != draws[:, cf[t, g]]
        end
    end
    # Everything else (`L`/`tau`, population, scales) crosses verbatim by
    # name — that is what makes `b = C * z` a draw from the FITTED covariance.
    pos_from = Dict(String(n) => i for (i, n) in enumerate(unc))
    for (j, nm) in enumerate(unc2)
        j in subj_cols && continue
        @test moved[:, j] == draws[:, pos_from[String(nm)]]
    end
end

@testset "boundary: resample_groups refuses a centered source" begin
    err = try
        reprocess(csd_from_sb, csd_to_df; resample_groups = [:subject])
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("non-centered", err.msg)
    @test occursin("centered_groups", err.msg)
end

@testset "CONTROL: non-centered plain + fresh level still refuses" begin
    nc_sb = SBBRMI(csd_builder(csd_from_df); mod = @__MODULE__, total_groups = ())
    @test all(b.noncentered for b in ranef_blocks(nc_sb))
    newlevel_df = csd_df([11, 12, 13, 99]; seed = 2, xshift = 5.0)
    nc_plan = generative_plan(csd_builder, newlevel_df; mod = @__MODULE__)
    unc_nc = csd_fake_unc(ranef_blocks(nc_sb))
    unc_np = csd_fake_unc(ranef_blocks(nc_plan))
    draws_nc = randn(MersenneTwister(9), 4, length(unc_nc))
    err = try
        transport_draws(nc_sb, nc_plan, draws_nc, unc_nc, unc_np;
                        rng = MersenneTwister(3))
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("resample_groups", err.msg)
end
