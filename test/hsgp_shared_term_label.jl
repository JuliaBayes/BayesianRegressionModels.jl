# test/hsgp_shared_term_label.jl — REGRESSION for snag hsgp-term-label-3222f035.
#
# Two predictors carrying `hsgp(...)` over the SAME axes emit distinct
# carriers: the first keeps the historical `hsgp_<axes>` binding and a repeat
# takes `hsgp_<target>_<axes>`, so the term outputs' `logical` names in
# `brm_outputs(d)` differ by declaration order. The PUBLIC term label does not:
# `term=:hsgp_<axes>` is derived from the formula term alone, and the
# `logical` predictor argument of `brm_term_coordinates` selects whose
# length-scale, marginal sd, basis weights and collected value (`:value`)
# resolve — the `mo(c)` rule.
# The predictor-scoped compiler name is not a term label and is refused.
#
# Run: julia --project=test test/hsgp_shared_term_label.jl
# Set BRM_HSGP_LABEL_RUNTIME=0 to skip the BridgeStan coordinates gate.
using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Exponential, Normal

const BRM = BayesianRegressionModels
const BS = StanBlocks.BridgeStan
const HSGP_LABEL_RUNTIME = get(ENV, "BRM_HSGP_LABEL_RUNTIME", "1") != "0"

hsgp_label_df = (;
    x_d=collect(range(-1.0, 1.0; length=12)),
    y_d=[0.3, -0.8, 0.1, 0.9, -0.4, 0.6, -1.0, 0.2, 0.8, -0.2, 0.5, -0.6],
    yr=zeros(12),
    yi=zeros(12),
)

hsgp_label_builder = @brm begin
    alpha_r ~ 1 + hsgp(x_d, y_d; k=3)
    alpha_i ~ 1 + hsgp(x_d, y_d; k=3)
    sd(alpha_r, hsgp(x_d, y_d)) ~ Exponential(0.5)
    sd(alpha_i, hsgp(x_d, y_d)) ~ Exponential(1)
    yr ~ Normal(alpha_r, 1.0)
    yi ~ Normal(alpha_i, 1.0)
end

@testset "repeated 2-D HSGP: one public term label per predictor" begin
    sb = SBBRMI(hsgp_label_builder(hsgp_label_df); mod=@__MODULE__)
    code = BRM.stan_code(sb)
    # Each predictor owns its own carriers and its own sd prior.
    @test occursin("hsgp_x_d_y_d_sigma ~ exponential((1.0 ./ 0.5))", code)
    @test occursin("hsgp_alpha_i_x_d_y_d_sigma ~ exponential((1.0 ./ 1.0))", code)
    d = brm_descriptor(sb)
    # The term OUTPUTS are compiler-named by declaration order ...
    logicals = [o.logical for o in d.outputs
                if o.logical isa Symbol && startswith(string(o.logical), "hsgp_")]
    @test sort(logicals) == [:hsgp_alpha_i_x_d_y_d, :hsgp_x_d_y_d]
    # ... while the term LABEL is the formula's, for both predictors.
    @test only(unique(e.term for e in BRM._brm_term_coordinate_entries(sb.parent, :alpha_r))) ===
          :hsgp_x_d_y_d
    @test only(unique(e.term for e in BRM._brm_term_coordinate_entries(sb.parent, :alpha_i))) ===
          :hsgp_x_d_y_d
end

@testset "per-predictor HSGP coordinates under the shared label" begin
    if !HSGP_LABEL_RUNTIME
        @info "Skipping shared-label HSGP BridgeStan coordinates gate (BRM_HSGP_LABEL_RUNTIME=0)"
    else
        d = brm_descriptor(hsgp_label_builder, hsgp_label_df; mod=@__MODULE__,
            name=:hsgp_shared_term_label)
        prob = brm_execute(d, :fit)
        names = BS.param_names(prob.model; include_tp=false, include_gq=false)
        expected = (;
            alpha_r=(; sd=:hsgp_x_d_y_d_sigma, length_scale=:hsgp_x_d_y_d_rho_iso,
                     basis_weights=:hsgp_x_d_y_d_beta_raw),
            alpha_i=(; sd=:hsgp_alpha_i_x_d_y_d_sigma,
                     length_scale=:hsgp_alpha_i_x_d_y_d_rho_iso,
                     basis_weights=:hsgp_alpha_i_x_d_y_d_beta_raw),
        )
        resolved = Dict{Symbol,Vector{Int}}()
        for lp in (:alpha_r, :alpha_i), parameter in (:sd, :length_scale, :basis_weights)
            c = brm_term_coordinates(d, lp, names; term=:hsgp_x_d_y_d, parameter)
            @test c.output.name === expected[lp][parameter]
            @test c.term === :hsgp_x_d_y_d
            stem = string(expected[lp][parameter])
            @test all(n -> n == stem || startswith(n, stem * "."), names[c.coordinates])
            @test length(c.coordinates) == (parameter === :basis_weights ? 9 : 1)
            append!(get!(resolved, lp, Int[]), c.coordinates)
        end
        @test isempty(intersect(resolved[:alpha_r], resolved[:alpha_i]))
        # refused: a predictor-scoped carrier spelling is compiler-owned, not a
        # public term label (brm-use descriptor.md: never hardcode emitted names).
        @test_throws "available term labels are (:hsgp_x_d_y_d,)" brm_term_coordinates(
            d, :alpha_i, names; term=:hsgp_alpha_i_x_d_y_d, parameter=:sd)

        # `:value` reaches each predictor's own collected term value under the
        # shared label (design 0mc2uvo): with `1 + hsgp(...)` the predictor is
        # its intercept plus that value at any point.
        tnames = BS.param_names(prob.model; include_tp=true, include_gq=false)
        q = collect(range(-0.5, 0.5; length=BS.param_unc_num(prob.model)))
        C = BS.param_constrain(prob.model, q; include_tp=true, include_gq=false)
        values = Dict{Symbol,Vector{Int}}()
        for lp in (:alpha_r, :alpha_i)
            v = brm_term_coordinates(d, lp, tnames; term=:hsgp_x_d_y_d, parameter=:value)
            @test v.output.name === (lp === :alpha_r ? :hsgp_x_d_y_d : :hsgp_alpha_i_x_d_y_d)
            @test length(v.coordinates) == length(hsgp_label_df.x_d)
            intercept = brm_population_effect_coordinates(
                d, lp, tnames; coefficient=:Intercept)
            lp_coords = brm_output_coordinates(d, lp, tnames)
            @test C[lp_coords] ≈ C[only(intercept.coordinates)] .+ C[v.coordinates]
            values[lp] = v.coordinates
        end
        @test isempty(intersect(values[:alpha_r], values[:alpha_i]))
        @test C[values[:alpha_r]] != C[values[:alpha_i]]
    end
end

@testset "a grouped HSGP's :value is its per-row contribution" begin
    if !HSGP_LABEL_RUNTIME
        @info "Skipping grouped HSGP :value BridgeStan gate (BRM_HSGP_LABEL_RUNTIME=0)"
    else
        df = (; x=collect(range(-1.0, 1.0; length=8)),
              g=repeat(["a", "b"]; inner=4), y=zeros(8))
        builder = @brm begin
            loc ~ 1 + hsgp(x; k=3, by=g)
            y ~ Normal(loc, 1.0)
        end
        d = brm_descriptor(builder, df; mod=@__MODULE__, name=:hsgp_grouped_value)
        prob = brm_execute(d, :fit)
        tnames = BS.param_names(prob.model; include_tp=true, include_gq=false)
        q = collect(range(-0.5, 0.5; length=BS.param_unc_num(prob.model)))
        C = BS.param_constrain(prob.model, q; include_tp=true, include_gq=false)
        v = brm_term_coordinates(d, :loc, tnames; term=:hsgp_x_by_g, parameter=:value)
        @test length(v.coordinates) == length(df.x)
        intercept = brm_population_effect_coordinates(
            d, :loc, tnames; coefficient=:Intercept)
        @test C[brm_output_coordinates(d, :loc, tnames)] ≈
              C[only(intercept.coordinates)] .+ C[v.coordinates]
    end
end
