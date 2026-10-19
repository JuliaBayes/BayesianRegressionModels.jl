# test/string_categorical_coordinates.jl — descriptor population coordinates of
# a predictor whose categorical column is a plain `Vector{String}` (todo
# `0cidxk1`). Such a column has no contrast block: the shared design codes its
# level indicators `<c>_lvl_<k>` into the predictor's own `beta_pop`. The
# resolver must address them by the formula column, exactly like a
# `CategoricalVector` column's separate contrast block.
#
# Oracle: a point of the `CategoricalVector` model, carried to the String model
# through the resolved coordinates, gives the same pointwise log likelihood; the
# same point with two levels swapped does not.
#
# Run: julia --project=test test/string_categorical_coordinates.jl

using Test, BayesianRegressionModels, BridgeStan, CategoricalArrays
using ReactiveKernels, ReactiveKernelsPPL
import StanBlocks
const BRM = BayesianRegressionModels
include(joinpath(@__DIR__, "testset_filter.jl"))

const STRING_CAT_CACHE = mktempdir()
compiled(sb, name) = StanBlocks.stan_instantiate(sb.model;
    path=joinpath(STRING_CAT_CACHE, "$name.stan")).model

const ARM = ["p", "q", "r", "q"]
string_cat_flat_data(arm) = (; subject=repeat(["a", "b", "c", "d"], 3),
    arm=repeat(arm, 3), y=sin.(1:12))
string_cat_plate_data(arm) = (; subject=["a", "b", "c", "d"], arm,
    t=[[0.1, 0.7], [0.3], [0.2, 0.8, 1.3], [0.5]],
    y=[[0.2, -0.1], [0.4], [0.1, 0.3, -0.2], [0.0]])

string_cat_flat(data) = @brm data begin
    sigma ~ Exponential(1)
    log_k ~ 1 + arm + (1 | e | subject)
    y ~ Normal(log_k, sigma)
end

string_cat_plate(data) = @brm data begin
    log_k ~ 1 + arm + (1 | e | subject)
    sigma ~ Exponential(1)
    @plate for i in eachindex(log_k)
        loc[i] = log_k[i] .+ t[i]
        y[i] ~ Normal(loc[i], sigma)
    end
end

function string_cat_fit(model, data, arm, name; total_groups=())
    sb = SBBRMI(model(data(arm)); mod=@__MODULE__, total_groups)
    m = compiled(sb, name)
    (; sb, m, d=brm_descriptor(sb), names=String.(BridgeStan.param_names(m)))
end

loglik(fit, u) = begin
    all_names = String.(BridgeStan.param_names(fit.m; include_tp=true, include_gq=true))
    θ = BridgeStan.param_constrain(fit.m, u; include_tp=true, include_gq=true,
        rng=BridgeStan.StanRNG(fit.m, 1))
    out = only(brm_outputs(fit.d; role=:pointwise_loglik))
    θ[brm_output_coordinates(out, all_names)]
end

# The String model's constrained point equal to `θc` of the CategoricalVector
# model: shared parameters by name, the population coefficients by address.
# `swap` exchanges two levels, the control the oracle must catch.
function string_cat_point(cat, str, θc; swap=false)
    θs = fill(NaN, length(str.names))
    for (i, n) in enumerate(str.names)
        j = findfirst(==(n), cat.names)
        isnothing(j) || (θs[i] = θc[j])
    end
    qc = brm_population_effect_coordinates(cat.d, :log_k, cat.names)
    qs = brm_population_effect_coordinates(str.d, :log_k, str.names)
    θs[only(qs.coordinates)] = θc[only(qc.coordinates)]
    ac = brm_population_effect_coordinates(cat.d, :log_k, cat.names; coefficient=:arm)
    as = brm_population_effect_coordinates(str.d, :log_k, str.names; coefficient=:arm)
    by_level = Dict(BRM._brm_transport_level(c.nonreference_level) => θc[c.coordinate]
                    for c in ac.contrasts)
    levels = [BRM._brm_transport_level(c.nonreference_level) for c in as.contrasts]
    swap && (levels = reverse(levels))
    for (c, level) in zip(as.contrasts, levels)
        θs[c.coordinate] = by_level[level]
    end
    θs
end

for (shape, model, data) in (("flat", string_cat_flat, string_cat_flat_data),
                             ("@plate", string_cat_plate, string_cat_plate_data))
    @stestset "a String column resolves like a CategoricalVector ($shape)" begin
        tag = shape == "flat" ? "flat" : "plate"
        str = string_cat_fit(model, data, ARM, "$(tag)_string")
        cat = string_cat_fit(model, data, categorical(ARM), "$(tag)_cat")

        q = brm_population_effect_coordinates(str.d, :log_k, str.names)
        @test length(q.coordinates) == 1
        @test q.recovered === false
        arm = brm_population_effect_coordinates(str.d, :log_k, str.names;
                                                coefficient=:arm)
        catarm = brm_population_effect_coordinates(cat.d, :log_k, cat.names;
                                                   coefficient=:arm)
        @test arm.coding === catarm.coding === :treatment
        @test arm.predictor === catarm.predictor === :arm
        @test arm.reference_level == "p"
        @test arm.nonreference_levels == ["q", "r"]
        @test [c.nonreference_level for c in arm.contrasts] == ["q", "r"]
        @test all(c -> c.reference_level == "p", arm.contrasts)
        @test isempty(arm.cells)
        @test arm.output === q.output
        @test length(unique([q.coordinates; arm.coordinates])) == 3
        @test arm.link === identity && arm.inverse_link === identity

        uc = [0.1 * sin(i) for i in 1:BridgeStan.param_unc_num(cat.m)]
        θc = BridgeStan.param_constrain(cat.m, uc)
        us = BridgeStan.param_unconstrain(str.m, string_cat_point(cat, str, θc))
        @test loglik(str, us) ≈ loglik(cat, uc) rtol=1e-12
        swapped = BridgeStan.param_unconstrain(str.m,
            string_cat_point(cat, str, θc; swap=true))
        @test !isapprox(loglik(str, swapped), loglik(cat, uc); rtol=1e-6)
    end
end

@stestset "exact totals: Intercept recovered, the String levels stay sampled" begin
    str = string_cat_fit(string_cat_flat, string_cat_flat_data, ARM, "flat_string_auto";
                         total_groups=:auto)
    gq = String.(BridgeStan.param_names(str.m; include_tp=true, include_gq=true))
    q = brm_population_effect_coordinates(str.d, :log_k, gq)
    @test q.recovered
    arm = brm_population_effect_coordinates(str.d, :log_k, str.names;
                                            coefficient=:arm)
    @test arm.output.kind === :parameter
    @test arm.nonreference_levels == ["q", "r"]
    @test length(arm.coordinates) == 2
end

@stestset "brm_start sets a String column's levels on SBBRMI" begin
    data = string_cat_flat_data(ARM)
    brmi = string_cat_flat(data)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    m = compiled(sb, "flat_string_start")
    inits = (; log_k=(; Intercept=-0.5, arm=["q" => 0.25, "r" => -0.4]),
             e=fill(0.04, 1, 1), sigma=0.2)
    s = brm_start(sb, m, inits)
    rk = RKBRMI(brmi)
    t = brm_coordinate_transport(rk, sb, String.(BridgeStan.param_unc_names(m)))
    @test brm_rk_point_to_stan(t, brm_start(rk, inits)) ≈ s rtol=1e-10 atol=1e-12
    names = String.(BridgeStan.param_names(m))
    θ = BridgeStan.param_constrain(m, s)
    d = brm_descriptor(sb)
    arm = brm_population_effect_coordinates(d, :log_k, names; coefficient=:arm)
    @test [c.nonreference_level for c in arm.contrasts] == ["q", "r"]
    @test θ[[c.coordinate for c in arm.contrasts]] ≈ [0.25, -0.4] rtol=1e-12
    @test θ[only(brm_population_effect_coordinates(d, :log_k, names).coordinates)] ≈ -0.5
    # refused: the reference level has no coefficient (brm_start contract).
    @test_throws ErrorException brm_start(sb, m, (; log_k=(; arm=["p" => 0.1])))
end

@stestset "an intercept-free String column is cell-mean coded" begin
    data = string_cat_flat_data(ARM)
    brmi = @brm data begin
        sigma ~ Exponential(1)
        log_k ~ 0 + arm + (1 | e | subject)
        y ~ Normal(log_k, sigma)
    end
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    m = compiled(sb, "flat_string_cells")
    names = String.(BridgeStan.param_names(m))
    d = brm_descriptor(sb)
    arm = brm_population_effect_coordinates(d, :log_k, names; coefficient=:arm)
    @test arm.coding === :cellmeans
    @test arm.reference_level === nothing
    @test isempty(arm.contrasts)
    @test [c.level for c in arm.cells] == ["p", "q", "r"]
    lvl2 = brm_population_effect_coordinates(d, :log_k, names; coefficient=:arm_lvl_2)
    @test only(lvl2.coordinates) == arm.cells[2].coordinate
    # refused: an unknown coefficient fails closed and names the column address
    # among the available ones (resolution never selects by descriptor order).
    err = try
        brm_population_effect_coordinates(d, :log_k, names; coefficient=:nope)
        nothing
    catch e
        sprint(showerror, e)
    end
    @test !isnothing(err) && occursin(r":arm[,)]", err)
end
