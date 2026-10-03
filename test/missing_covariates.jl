using Test
using BayesianRegressionModels
using StanBlocks
using Distributions
using Statistics
using BridgeStan
using LogDensityProblems
include(joinpath(@__DIR__, "testset_filter.jl"))

const MISSING_TOY = (;
    subject=[1, 2, 3, 4, 5],
    x=Union{Missing,Float64}[0.8, missing, 2.1, missing, 1.4],
    z=Union{Missing,Float64}[missing, 1.1, 0.9, 1.6, missing],
    y=[-0.2, 0.7, 1.2, -0.4, 0.5],
)
const MISSING_PLAIN = @brm begin
    mx ~ Normal(0, 1)
    sx ~ LogNormal(0, 0.3)
    mi(x) ~ LogNormal(mx, sx)
    mu ~ 1 + x
    y ~ Normal(mu, 1)
end

const MISSING_TRANSFORMS = @brm begin
    mi(x) ~ LogNormal(0, 0.7)
    mu ~ 1 + standardize(x) + center(x) + zscale(log(x)) + log(x)
    y ~ Normal(mu, 1)
end

const MISSING_CENTER = @brm begin
    mi(x) ~ Normal(0, 1)
    mu ~ 1 + center(x)
    y ~ Normal(mu, 1)
end

@stestset "missing covariate anchors reject unidentified scale" begin
    no_observed = merge(MISSING_TOY, (; x=Union{Missing,Float64}[missing for _ in 1:5],))
    one_observed = merge(MISSING_TOY, (; x=Union{Missing,Float64}[missing, 2.0, missing, missing, missing],))
    constant_observed = merge(MISSING_TOY, (; x=Union{Missing,Float64}[2.0, missing, 2.0, missing, 2.0],))
    nonfinite_observed = merge(MISSING_TOY, (; x=Union{Missing,Float64}[Inf, missing, 2.0, missing, 3.0],))
    # An observed-only sample SD needs two finite observations with variation;
    # replacing it with posterior-dependent or arbitrary constants changes the model.
    @test_throws r"at least two training values" SBBRMI(MISSING_TRANSFORMS(no_observed); mod=@__MODULE__)
    @test_throws r"at least two training values" SBBRMI(MISSING_TRANSFORMS(one_observed); mod=@__MODULE__)
    @test_throws r"nonzero sample variance" SBBRMI(MISSING_TRANSFORMS(constant_observed); mod=@__MODULE__)
    @test_throws r"finite real training values" SBBRMI(MISSING_TRANSFORMS(nonfinite_observed); mod=@__MODULE__)
    @test_throws r"at least one training value" SBBRMI(MISSING_CENTER(no_observed); mod=@__MODULE__)
    centered = SBBRMI(MISSING_CENTER(one_observed); mod=@__MODULE__)
    @test centered.data[:center_x_mean] == 2.0
end

@stestset "missing covariate transforms use observed fixed constants" begin
    sb = SBBRMI(MISSING_TRANSFORMS(MISSING_TOY); mod=@__MODULE__)
    observed = collect(skipmissing(MISSING_TOY.x))
    @test sb.data[:standardize_x_mean] ≈ mean(observed)
    @test sb.data[:standardize_x_scale] ≈ std(observed)
    @test sb.data[:center_x_mean] ≈ mean(observed)
    @test !haskey(sb.data, :standardize_x)
    @test popcoefnames(sb.parent, :mu)[1:3] == [:Intercept, :standardize_x, :center_x]
    code = BayesianRegressionModels.stan_code(sb)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test occursin("vector<lower=0.0>[x_n_mis] x_y_mis;", code)
    same_mask = merge(MISSING_TOY, (;
        x=Union{Missing,Float64}[1.3, missing, 2.7, missing, 1.8],))
    replay = reprocess(sb, same_mask)
    @test replay.data[:standardize_x_mean] == sb.data[:standardize_x_mean]
    @test replay.data[:standardize_x_scale] == sb.data[:standardize_x_scale]
    @test BayesianRegressionModels.stan_code(replay) == code
    fresh = reprocess(sb, same_mask; freeze_constants=false)
    @test fresh.data[:standardize_x_mean] ≈ mean(skipmissing(same_mask.x))
    @test fresh.data[:standardize_x_scale] ≈ std(collect(skipmissing(same_mask.x)))
end

@stestset "missing covariate replay preserves fitted row positions" begin
    original = deepcopy(MISSING_TOY)
    sb = SBBRMI(MISSING_PLAIN(MISSING_TOY); mod=@__MODULE__)
    same_mask = merge(MISSING_TOY, (;
        x=Union{Missing,Float64}[1.3, missing, 2.7, missing, 1.8],))
    replay = reprocess(sb, same_mask)
    @test replay.data[:x_obs] == [1.3, 2.7, 1.8]
    @test replay.data[:Jobs_x] == [1, 3, 5]
    @test replay.data[:Jmis_x] == [2, 4]
    @test BayesianRegressionModels.stan_code(replay) == BayesianRegressionModels.stan_code(sb)
    changed = merge(MISSING_TOY, (;
        x=Union{Missing,Float64}[missing, 0.6, 2.1, missing, 1.4],))
    # Fitted draws address rows, so a changed frozen mask would reassign them.
    @test_throws r"same fitted missing-row positions" reprocess(sb, changed)
    fresh = reprocess(sb, changed; freeze_constants=false)
    @test fresh.data[:Jobs_x] == [2, 3, 5]
    @test fresh.data[:Jmis_x] == [1, 4]
    @test fresh.data[:x_obs] == [0.6, 2.1, 1.4]
    @test fresh.preproc[:x_obs].const_.missing_indices == [1, 4]
    more_missing = merge(MISSING_TOY, (;
        x=Union{Missing,Float64}[missing, 0.6, missing, missing, 1.4],))
    @test_throws r"same fitted missing-row positions" reprocess(sb, more_missing)
    resized = reprocess(sb, more_missing; freeze_constants=false)
    @test resized.data[:Jobs_x] == [2, 5]
    @test resized.data[:Jmis_x] == [1, 3, 4]
    @test isequal(MISSING_TOY, original)
    @test sb.data[:x_obs] == [0.8, 2.1, 1.4]
end

const MISSING_JOINT_NORMAL = @brm begin
    mx ~ Normal(0, 1)
    sx ~ LogNormal(0, 0.3)
    mi(x) ~ Normal(mx, sx)
    mz = 0.2 + 0.3 * x
    zs = exp(0.1 + 0.2 * x)
    mi(z) ~ Normal(mz, zs)
    combined = x + z
    mu ~ 1 + standardize(x) + center(z) + z
    y ~ Normal(mu, 1)
end
const MISSING_JOINT_LOGNORMAL = @brm begin
    mx ~ Normal(0, 1)
    sx ~ LogNormal(0, 0.3)
    mi(x) ~ LogNormal(mx, sx)
    mz = 0.2 + 0.3 * x
    zs = exp(0.1 + 0.2 * x)
    mi(z) ~ LogNormal(mz, zs)
    combined = x + z
    mu ~ 1 + standardize(x) + center(z) + log(z)
    y ~ Normal(mu, 1)
end

# Independent explicit observed/missing reference. Only the model parameter
# names enter the coordinate map; row ordering and densities come from the toy
# data and Julia distributions, never from BRM's emitted completion helper.
function missing_reference(q, names, family, df)
    at = Dict(name => q[i] for (i, name) in enumerate(names))
    positive = family === LogNormal
    complete(column, stem) = [ismissing(value) ?
        (positive ? exp(at["$(stem)_y_mis.$j"]) : at["$(stem)_y_mis.$j"]) :
        Float64(value) for (i, value, j) in
        zip(eachindex(column), column, cumsum(ismissing.(column)))]
    x, z = complete(df.x, "x"), complete(df.z, "z")
    mx, sx = at["mx"], exp(at["sx"])
    beta = [at["pop_mu_beta_pop.$i"] for i in 1:4]
    mz, zs = 0.2 .+ 0.3 .* x, exp.(0.1 .+ 0.2 .* x)
    observed_x, missing_x = findall(!ismissing, df.x), findall(ismissing, df.x)
    observed_z, missing_z = findall(!ismissing, df.z), findall(ismissing, df.z)
    lp = logpdf(Normal(), mx) + logpdf(LogNormal(0, 0.3), sx) + at["sx"]
    lp += sum(logpdf.(Normal(), beta))
    lp += sum(logpdf(family(mx, sx), Float64(df.x[i])) for i in observed_x)
    lp += sum(logpdf(family(mx, sx), x[i]) for i in missing_x)
    lp += sum(logpdf(family(mz[i], zs[i]), Float64(df.z[i])) for i in observed_z)
    lp += sum(logpdf(family(mz[i], zs[i]), z[i]) for i in missing_z)
    if positive
        lp += sum(at["x_y_mis.$i"] for i in eachindex(missing_x))
        lp += sum(at["z_y_mis.$i"] for i in eachindex(missing_z))
    end
    ox, oz = collect(skipmissing(df.x)), collect(skipmissing(df.z))
    extra = positive ? log.(z) : z
    mu = beta[1] .+ beta[2] .* ((x .- mean(ox)) ./ std(ox)) .+
         beta[3] .* (z .- mean(oz)) .+ beta[4] .* extra
    lp += sum(logpdf(Normal(mu[i], 1), df.y[i]) for i in eachindex(mu))
    (; lp, x, z, combined=x .+ z, mu)
end

@stestset "joint Normal and LogNormal missing covariate densities and named outputs" begin
    for (builder, family) in ((MISSING_JOINT_NORMAL, Normal),
                              (MISSING_JOINT_LOGNORMAL, LogNormal))
        sb = SBBRMI(builder(MISSING_TOY); mod=@__MODULE__)
        code = BayesianRegressionModels.stan_code(sb)
        @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
        @test occursin("mz[Jobs_z]", code) && occursin("mz[Jmis_z]", code)
        @test occursin("zs[Jobs_z]", code) && occursin("zs[Jmis_z]", code)
        problem = StanBlocks.stan_instantiate(sb.model)
        names = BridgeStan.param_unc_names(problem.model)
        @test Set(names) == Set(vcat(["mx", "sx"],
            ["$(stem)_y_mis.$i" for stem in ("x", "z") for i in 1:2],
            ["pop_mu_beta_pop.$i" for i in 1:4]))
        descriptor = brm_descriptor(sb)
        full_names = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
        for offset in (0.0, 0.17, -0.11)
            q = [offset + 0.07 * (i - 5) for i in eachindex(names)]
            reference = missing_reference(q, names, family, MISSING_TOY)
            gradient = zeros(length(q))
            lp, _ = BridgeStan.log_density_gradient!(problem.model, q, gradient;
                propto=false, jacobian=true)
            @test lp ≈ reference.lp atol=1e-9 rtol=0
            fd = BayesianRegressionModels._sb_central_diff(
                v -> missing_reference(v, names, family, MISSING_TOY).lp, q, 1e-6)
            @test gradient ≈ fd atol=2e-7 rtol=2e-7
            full = BridgeStan.param_constrain(problem.model, q;
                include_tp=true, include_gq=true, rng=BridgeStan.StanRNG(problem.model, 42))
            for logical in (:x, :z, :combined, :mu)
                coordinates = brm_output_coordinates(descriptor, logical, full_names)
                @test length(coordinates) == length(MISSING_TOY.y)
                @test full[coordinates] ≈ getproperty(reference, logical) atol=1e-12 rtol=1e-12
            end
        end
    end
end

const MISSING_COMPLETE_CONDITIONAL = @brm begin
    mi(z) ~ LogNormal(0.1, 0.4)
    x ~ LogNormal(0.2 + 0.3 * z, 0.5)
    combined = x + z
    y ~ Normal(combined, 1)
end

@stestset "complete observed covariate conditional on an imputed predictor" begin
    complete = merge(MISSING_TOY, (; x=[0.8, 1.2, 2.1, 1.7, 1.4],))
    # Complete observations use the ordinary likelihood spelling. mi() is
    # deliberately reserved for columns containing missing entries.
    @test_throws r"drop `mi" SBBRMI(MISSING_PLAIN(complete); mod=@__MODULE__)
    sb = SBBRMI(MISSING_COMPLETE_CONDITIONAL(complete); mod=@__MODULE__)
    descriptor = brm_descriptor(sb)
    @test sb.data[:x] == complete.x
    @test any(i -> i.name === :x && i.column === :x && i.observed, descriptor.inputs)
    @test brm_output(descriptor, :x; role=:posterior_predictive).source === :x
    problem = StanBlocks.stan_instantiate(sb.model)
    names = BridgeStan.param_unc_names(problem.model)
    @test Set(names) == Set(["z_y_mis.1", "z_y_mis.2"])
    full_names = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
    for q in ([0.1, -0.2], [-0.3, 0.4])
        completed_z = Float64[ismissing(z) ? exp(q[findfirst(==(i), [1, 5])]) : z
            for (i, z) in enumerate(complete.z)]
        expected = sum(logpdf.(LogNormal(0.1, 0.4), completed_z)) + sum(q) +
            sum(logpdf.(LogNormal.(0.2 .+ 0.3 .* completed_z, 0.5), complete.x)) +
            sum(logpdf.(Normal.(complete.x .+ completed_z, 1), complete.y))
        gradient = zeros(2)
        lp, _ = BridgeStan.log_density_gradient!(problem.model, q, gradient;
            propto=false, jacobian=true)
        @test lp ≈ expected atol=1e-9 rtol=0
        @test all(isfinite, gradient)
        full = BridgeStan.param_constrain(problem.model, q;
            include_tp=true, include_gq=true, rng=BridgeStan.StanRNG(problem.model, 42))
        combined = full[brm_output_coordinates(descriptor, :combined, full_names)]
        @test combined ≈ complete.x .+ completed_z atol=1e-12 rtol=1e-12
        @test sb.data[:x] == complete.x
    end
end

const MISSING_KERNEL = @brm begin
    mx ~ Normal(0, 1)
    sx ~ LogNormal(0, 0.3)
    mi(x) ~ Normal(mx, sx)
    mu ~ 1 + standardize(x) + (1 | subject)
    out ~ kernel(y, mu) do yy, mm
        yy ~ normal(mm, 1.0)
        mm
    end
end

const MISSING_KERNEL_POSITIVE = @brm begin
    mx ~ Normal(0, 1)
    sx ~ LogNormal(0, 0.3)
    mi(x) ~ LogNormal(mx, sx)
    mu ~ 1 + standardize(x) + (1 | subject)
    out ~ kernel(y, mu) do yy, mm
        yy ~ normal(mm, 1.0)
        mm
    end
end

@stestset "missing covariates stay fitted through subject kernel CV" begin
    for builder in (MISSING_KERNEL, MISSING_KERNEL_POSITIVE)
        sb = SBBRMI(builder(MISSING_TOY); mod=@__MODULE__)
        cv = reprocess(sb, MISSING_TOY; resample_groups=[:subject])
        source = StanBlocks.stan_instantiate(sb.model)
        target = StanBlocks.stan_instantiate(cv.model)
        source_names, target_names = BridgeStan.param_unc_names(source.model), BridgeStan.param_unc_names(target.model)
        @test all("x_y_mis.$i" in source_names && "x_y_mis.$i" in target_names for i in 1:2)
        @test length(source_names) - length(target_names) == length(MISSING_TOY.subject)
        @test Set(target_names) ⊆ Set(source_names)
        random_outputs = brm_outputs(brm_descriptor(sb); role=:random_effect, kind=:parameter)
        random_coordinates = reduce(vcat,
            (brm_output_coordinates(output, source_names) for output in random_outputs))
        @test Set(setdiff(source_names, target_names)) ⊆ Set(source_names[random_coordinates])
        source_q = [0.09 * (i - 4) for i in eachindex(source_names)]
        target_q = [source_q[findfirst(==(name), source_names)] for name in target_names]
        source_full = BridgeStan.param_constrain(source.model, source_q;
            include_tp=true, include_gq=true, rng=BridgeStan.StanRNG(source.model, 41))
        target_full = BridgeStan.param_constrain(target.model, target_q;
            include_tp=true, include_gq=true, rng=BridgeStan.StanRNG(target.model, 42))
        source_full_names = BridgeStan.param_names(source.model; include_tp=true, include_gq=true)
        target_full_names = BridgeStan.param_names(target.model; include_tp=true, include_gq=true)
        source_x = source_full[brm_output_coordinates(brm_descriptor(sb), :x, source_full_names)]
        target_x = target_full[brm_output_coordinates(brm_descriptor(cv), :x, target_full_names)]
        @test brm_output(brm_descriptor(cv), :y; role=:posterior_predictive).source === :y
        @test source_x == target_x
        @test source_x[findall(!ismissing, MISSING_TOY.x)] == collect(skipmissing(MISSING_TOY.x))
        lp, gradient = LogDensityProblems.logdensity_and_gradient(source, source_q)
        @test isfinite(lp) && all(isfinite, gradient)
        for sb_ in (sb, cv)
            @test sb_.data[:standardize_x_mean] == sb.data[:standardize_x_mean]
            @test sb_.data[:standardize_x_scale] == sb.data[:standardize_x_scale]
            @test sb_.data[:Jmis_x] == [2, 4]
        end
        changed = merge(MISSING_TOY, (; x=Union{Missing,Float64}[missing, 0.6, 2.1, missing, 1.4],))
        @test_throws r"same fitted missing-row positions" reprocess(sb, changed; resample_groups=[:subject])
    end
end
