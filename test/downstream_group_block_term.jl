# test/downstream_group_block_term.jl — contract for the downstream grouped-term
# pattern (snag brm-parametric-t-ae5e02f9; user decision: option B — documented
# downstream reusable-term pattern, NOT BRM-core transient/saturating terms).
#
# A downstream package defines grouped parametric terms through BRM's public
# seams ONLY: its own markers + `_sb_term_group_block` fields-form declarations
# + `_sb_emit_group_block_term!` methods. No BRM src change, no custom
# @slic/@deffun — each emit hook reuses a StanBlocks builtin. This file ACTS as
# that downstream module (separate `module`, methods added via `import`) with
# a bordet-inspired pair (`transient`, 3 params; `saturating`, 2 params) plus
# their `base + bump * resp` composition, and pins the supported contract:
#   - SBBRMI lowers to a stanc-clean hierarchical model;
#   - `brm_descriptor` and frozen-level replay work;
#   - unseen replay levels and TuringBRMI refuse LOUDLY.
# Promotion of such terms into BRM core (option A) waits on multi-package
# adoption; see docs/src/formula-terms.md "Downstream grouped terms".

using Test
using BayesianRegressionModels
using StanBlocks
using Turing
import StanBlocks.stan: transpiles

module DownstreamTransientTerms
using StanBlocks
import BayesianRegressionModels: _sb_term_group_block, _sb_emit_group_block_term!

# Single-peak transient over `logt`, with per-`series` hierarchical
# (loc, log_slope, mag). Deliberately NOT exported from BRM core.
function transient end

_sb_term_group_block(::typeof(transient)) = (; fields=[
    (; name=:transient, n_per_group=3, group=(; kwarg=:series),
       prior=:correlated_normal),
])

function _sb_emit_group_block_term!(stmts, data, target, ::typeof(transient),
                                    rhs_e, block_info)
    (; block_name, idx_name) = block_info
    push!(stmts, :(tloc = $(block_name)[$(idx_name), 1]))
    push!(stmts, :(tlog_slope = $(block_name)[$(idx_name), 2]))
    push!(stmts, :(tmag = $(block_name)[$(idx_name), 3]))
    push!(stmts, :($target =
        biomarker_time_response(logt, tloc, tlog_slope, tmag)))
end

# Saturating 0-to-1 dose multiplier over `logd`, with per-`series`
# hierarchical (loc, log_slope). The StanBlocks builtin returns the
# LOG-sigmoid, so the emit hook exps it — the bordet composition shape.
function saturating end

_sb_term_group_block(::typeof(saturating)) = (; fields=[
    (; name=:saturating, n_per_group=2, group=(; kwarg=:series),
       prior=:correlated_normal),
])

function _sb_emit_group_block_term!(stmts, data, target, ::typeof(saturating),
                                    rhs_e, block_info)
    (; block_name, idx_name) = block_info
    push!(stmts, :(dloc = $(block_name)[$(idx_name), 1]))
    push!(stmts, :(dlog_slope = $(block_name)[$(idx_name), 2]))
    push!(stmts, :($target =
        exp(biomarker_dose_response(logd, dloc, dlog_slope))))
end
end

using .DownstreamTransientTerms: transient, saturating

# Long-format data: 2 series x 6 observations.
downstream_df() = (;
    logt=[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0],
    series=[1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
    y=[0.1, 0.4, 0.8, 0.5, 0.2, 0.05, 0.05, 0.3, 0.9, 0.6, 0.25, 0.08],
)

downstream_model(df=downstream_df()) = @brm df begin
    sigma ~ Exponential(1)
    mu ~ transient(; logt, series)
    y ~ Normal(mu, sigma)
end

@testset "downstream grouped transient lowers to a stanc-clean hierarchical bump" begin
    brmi = downstream_model()
    sb = SBBRMI(brmi; mod=@__MODULE__)
    code = StanBlocks.stan_code(sb.model)
    # Per-series hierarchical block: LKJ + scales + non-centered draws.
    @test occursin("b_transient_series_L ~ lkj_corr_cholesky", code)
    @test occursin("b_transient_series =", code)
    # Per-observation bump from the block columns via the StanBlocks builtin.
    @test occursin("tloc = b_transient_series[series_idx, 1]", code)
    @test occursin("tlog_slope = b_transient_series[series_idx, 2]", code)
    @test occursin("tmag = b_transient_series[series_idx, 3]", code)
    @test occursin("mu = biomarker_time_response(logt, tloc, tlog_slope, tmag)",
        code)
    @test sb.data[:n_series] == 2
    @test sb.data[:n_terms_transient_series] == 3
    @test sb.data[:series_idx] == [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
    @test transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
end

@testset "downstream grouped term keeps descriptor and frozen replay" begin
    sb = SBBRMI(downstream_model(); mod=@__MODULE__)
    @test !isnothing(brm_descriptor(sb))
    fitted = (; logt=[-1.5, 0.25], series=[1, 2], y=[0.2, 0.4])
    sb2 = reprocess(sb, fitted)
    @test sb2.data[:n_series] == 2
    @test sb2.data[:series_idx] == [1, 2]
    @test StanBlocks.stanc_check(StanBlocks.stan_code(sb2.model);
        warn_pedantic=false).ok
    unseen = (; logt=[-1.5, 0.25], series=[1, 3], y=[0.2, 0.4])
    @test_throws "unseen level" reprocess(sb, unseen)
end

@testset "downstream grouped term refuses Turing loudly" begin
    # The marker has no Julia call semantics, and the grouped hierarchical
    # parameters have no Turing term model: lowering must fail closed, never
    # silently drop the term. If Turing support lands, this test (and the
    # docs boundary table) must be updated to the new contract.
    @test_throws "no method matching transient" TuringBRMI(downstream_model())
end

# Bordet-inspired composition: baseline + transient time bump x saturating
# dose multiplier, each shape with its own per-series hierarchy.
composed_df() = (;
    logt=[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0],
    logd=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    series=[1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
    y=[0.05, 0.2, 0.5, 0.4, 0.2, 0.1, 0.03, 0.15, 0.6, 0.45, 0.22, 0.12],
)

composed_model(df=composed_df()) = @brm df begin
    sigma ~ Exponential(1)
    base ~ Normal(0, 1)
    bump ~ transient(; logt, series)
    resp ~ saturating(; logd, series)
    mu = base + bump * resp
    y ~ Normal(mu, sigma)
end

@testset "bordet-inspired transient x saturating composition lowers clean" begin
    sb = SBBRMI(composed_model(); mod=@__MODULE__)
    code = StanBlocks.stan_code(sb.model)
    @test occursin("biomarker_time_response(logt, tloc, tlog_slope, tmag)", code)
    @test occursin("bump .* resp", code)  # composed mean keeps both shapes
    @test occursin("biomarker_dose_response(logd, dloc, dlog_slope)", code)
    @test occursin("b_transient_series_L ~ lkj_corr_cholesky", code)
    @test occursin("b_saturating_series_L ~ lkj_corr_cholesky", code)
    @test sb.data[:n_terms_transient_series] == 3
    @test sb.data[:n_terms_saturating_series] == 2
    @test transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
end
