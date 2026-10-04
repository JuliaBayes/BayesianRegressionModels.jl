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

const MISSING_DERIVED_TOY = merge(MISSING_TOY, (;
    z=Union{Missing,Float64}[1.2, missing, 0.9, missing, 1.5],
    c=[1, 2, 1, 2, 2],))
const MISSING_DERIVED = @brm begin
    mx ~ Normal(0, 1)
    sx ~ LogNormal(0, 0.3)
    mi(x) ~ LogNormal(mx, sx)
    mi(z) ~ LogNormal(0.2 + 0.3 * x, 0.5)
    combined = x + z
    doubled = 2 * combined
    ratio = x / z^2
    shifted = doubled + ratio
    mu ~ 1 + standardize(x) + factor(c) + log(shifted) + combined +
         center(ratio) + standardize(log(combined))
    effect(mu, Intercept) ~ Normal(0.7, 0.4)
    y ~ Normal(mu, 1)
end

function missing_derived_reference(q, names, df; anchors=MISSING_DERIVED_TOY)
    at = Dict(name => q[i] for (i, name) in enumerate(names))
    complete(col, stem) = Float64[ismissing(v) ? exp(at["$(stem)_y_mis.$j"]) : v
        for (v, j) in zip(col, cumsum(ismissing.(col)))]
    x, z = complete(df.x, "x"), complete(df.z, "z")
    mx, sx = at["mx"], exp(at["sx"])
    beta = [at["pop_mu_beta_pop.$i"] for i in 1:6]
    cat_names = filter(n -> startswith(n, "cat_mu_c_beta."), names)
    contrast = at[only(cat_names)]
    combined = x .+ z
    doubled = 2 .* combined
    ratio = x ./ z.^2
    shifted = doubled .+ ratio
    ox = collect(skipmissing(anchors.x))
    complete_rows = findall(i -> !ismissing(anchors.x[i]) && !ismissing(anchors.z[i]),
                            eachindex(anchors.x))
    observed_log = [log(anchors.x[i] + anchors.z[i]) for i in complete_rows]
    observed_ratio = [anchors.x[i] / anchors.z[i]^2 for i in complete_rows]
    mu = beta[1] .+ beta[2] .* ((x .- mean(ox)) ./ std(ox)) .+
         contrast .* (df.c .== 2) .+ beta[3] .* log.(shifted) .+
         beta[4] .* combined .+ beta[5] .* (ratio .- mean(observed_ratio)) .+
         beta[6] .* ((log.(combined) .- mean(observed_log)) ./ std(observed_log))
    lp = logpdf(Normal(), mx) + logpdf(LogNormal(0, 0.3), sx) + at["sx"] +
         logpdf(Normal(0.7, 0.4), beta[1]) + sum(logpdf.(Normal(), beta[2:end])) +
         logpdf(Normal(), contrast) + sum(logpdf.(LogNormal(mx, sx), x)) +
         sum(logpdf.(LogNormal.(0.2 .+ 0.3 .* x, 0.5), z)) +
         sum(logpdf.(Normal.(mu, 1), df.y)) +
         sum(at["$(stem)_y_mis.$i"] for stem in ("x", "z") for i in 1:2)
    (; lp, x, z, combined, doubled, ratio, shifted, mu)
end

@stestset "derived missing covariate transforms retain effect labels and fixed anchors" begin
    bound = MISSING_DERIVED(MISSING_DERIVED_TOY)
    labels = popcoefnames(bound, :mu)
    @test length(labels) == 6
    @test labels[[1, 2, 4, 5]] == [:Intercept, :standardize_x, :combined, :center_ratio]
    @test startswith(string(labels[3]), "log_")
    @test startswith(string(labels[6]), "standardize_")
    sb = SBBRMI(bound; mod=@__MODULE__)
    code = BayesianRegressionModels.stan_code(sb)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    observed_log = log.([2.0, 3.0, 2.9])
    @test sb.data[Symbol(labels[6], :_mean)] ≈ mean(observed_log)
    @test sb.data[Symbol(labels[6], :_scale)] ≈ std(observed_log)
    @test sb.data[:center_ratio_mean] ≈ mean([0.8/1.2^2, 2.1/0.9^2, 1.4/1.5^2])
    problem = StanBlocks.stan_instantiate(sb.model)
    names = BridgeStan.param_unc_names(problem.model)
    full_names = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
    descriptor = brm_descriptor(sb)
    for shift in (0.0, 0.17, -0.11)
        q = [shift + 0.03 * (i - 7) for i in eachindex(names)]
        ref = missing_derived_reference(q, names, MISSING_DERIVED_TOY)
        gradient = zeros(length(q))
        lp, _ = BridgeStan.log_density_gradient!(problem.model, q, gradient;
            propto=false, jacobian=true)
        @test lp ≈ ref.lp atol=1e-9 rtol=0
        fd = BayesianRegressionModels._sb_central_diff(
            v -> missing_derived_reference(v, names, MISSING_DERIVED_TOY).lp, q, 1e-6)
        @test gradient ≈ fd atol=2e-7 rtol=2e-7
        full = BridgeStan.param_constrain(problem.model, q;
            include_tp=true, include_gq=true, rng=BridgeStan.StanRNG(problem.model, 42))
        for logical in (:x, :z, :combined, :doubled, :ratio, :shifted, :mu)
            coords = brm_output_coordinates(descriptor, logical, full_names)
            @test length(coords) == 5
            @test full[coords] ≈ getproperty(ref, logical) atol=1e-12 rtol=1e-12
        end
    end
    updated = merge(MISSING_DERIVED_TOY, (;
        x=Union{Missing,Float64}[1.3, missing, 2.7, missing, 1.8],
        z=Union{Missing,Float64}[1.6, missing, 1.0, missing, 1.2],))
    replay = reprocess(sb, updated)
    @test BayesianRegressionModels.stan_code(replay) == code
    for key in (:center_ratio_mean, Symbol(labels[6], :_mean), Symbol(labels[6], :_scale))
        @test replay.data[key] == sb.data[key]
    end
    replay_problem = StanBlocks.stan_instantiate(replay.model)
    q = [0.02 * i for i in eachindex(names)]
    replay_full = BridgeStan.param_constrain(replay_problem.model, q;
        include_tp=true, include_gq=true, rng=BridgeStan.StanRNG(replay_problem.model, 43))
    ref = missing_derived_reference(q, names, updated)
    @test replay_full[brm_output_coordinates(brm_descriptor(replay), :mu, full_names)] ≈ ref.mu
    fresh = reprocess(sb, updated; freeze_constants=false)
    @test fresh.data[Symbol(labels[6], :_mean)] ≈ mean(log.([2.9, 3.7, 3.0]))
    one_complete = merge(MISSING_DERIVED_TOY, (;
        z=Union{Missing,Float64}[missing, missing, 0.9, missing, missing],))
    @test_throws r"at least two training values" SBBRMI(MISSING_DERIVED(one_complete); mod=@__MODULE__)
end

@stestset "derived missing covariate anchors require observed inputs" begin
    with_parameter = @brm begin
        mi(x) ~ LogNormal(0, 0.5)
        alpha ~ Normal(0, 1)
        combined = x + alpha
        mu ~ 1 + standardize(combined)
        y ~ Normal(mu, 1)
    end
    @test_throws r"fixed observed-only transform anchors.*alpha" SBBRMI(
        with_parameter(MISSING_TOY); mod=@__MODULE__)
    with_observation = @brm begin
        mi(z) ~ LogNormal(0, 0.5)
        x ~ LogNormal(0.2 + 0.3 * z, 0.5)
        combined = x + z
        mu ~ 1 + standardize(combined)
        y ~ Normal(mu, 1)
    end
    complete = merge(MISSING_TOY, (; x=[0.8, 1.2, 2.1, 1.7, 1.4],))
    sb = SBBRMI(with_observation(complete); mod=@__MODULE__)
    observed = [complete.x[i] + complete.z[i] for i in (2, 3, 4)]
    @test sb.data[:standardize_combined_mean] ≈ mean(observed)
    @test sb.data[:standardize_combined_scale] ≈ std(observed)
    @test StanBlocks.stanc_check(BayesianRegressionModels.stan_code(sb); warn_pedantic=false).ok
    changed = merge(complete, (; x=complete.x .+ 0.2,))
    @test reprocess(sb, changed).data[:standardize_combined_mean] == sb.data[:standardize_combined_mean]
    @test reprocess(sb, changed; freeze_constants=false).data[:standardize_combined_mean] ≈ mean(observed) + 0.2
end

const MISSING_SHARED = @brm begin
    mi(x) ~ LogNormal(0, 0.5)
    mu1 ~ 1 + standardize(x) + log(x)
    mu2 ~ 1 + standardize(x) + log(x)
    y ~ Normal(mu1, 1)
    y2 ~ Normal(mu2, 1)
end

@stestset "shared missing covariate transforms across two observed likelihoods" begin
    df = merge(MISSING_TOY, (; y2=MISSING_TOY.y .+ 0.3,))
    bound = MISSING_SHARED(df)
    @test popcoefnames(bound, :mu1) == popcoefnames(bound, :mu2)
    sb = SBBRMI(bound; mod=@__MODULE__)
    @test StanBlocks.stanc_check(BayesianRegressionModels.stan_code(sb); warn_pedantic=false).ok
    problem = StanBlocks.stan_instantiate(sb.model)
    names = BridgeStan.param_unc_names(problem.model)
    full_names = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
    q = [0.04 * (i - 4) for i in eachindex(names)]
    reference(v) = begin
        at = Dict(name => v[i] for (i, name) in enumerate(names))
        x = Float64[ismissing(value) ? exp(at["x_y_mis.$j"]) : value
            for (value, j) in zip(df.x, cumsum(ismissing.(df.x)))]
        ox = collect(skipmissing(df.x))
        means = map((:mu1, :mu2)) do logical
            beta = [at["pop_$(logical)_beta_pop.$i"] for i in 1:3]
            beta[1] .+ beta[2] .* ((x .- mean(ox)) ./ std(ox)) .+ beta[3] .* log.(x)
        end
        lp = sum(logpdf.(LogNormal(0, 0.5), x)) + sum(at["x_y_mis.$i"] for i in 1:2) +
             sum(logpdf(Normal(), at["pop_$(logical)_beta_pop.$i"])
                 for logical in (:mu1, :mu2) for i in 1:3) +
             sum(logpdf.(Normal.(means[1], 1), df.y)) +
             sum(logpdf.(Normal.(means[2], 1), df.y2))
        (; lp, x, mu1=means[1], mu2=means[2])
    end
    ref = reference(q)
    gradient = zeros(length(q))
    lp, _ = BridgeStan.log_density_gradient!(problem.model, q, gradient;
        propto=false, jacobian=true)
    @test lp ≈ ref.lp atol=1e-9 rtol=0
    fd = BayesianRegressionModels._sb_central_diff(v -> reference(v).lp, q, 1e-6)
    @test gradient ≈ fd atol=2e-7 rtol=2e-7
    full = BridgeStan.param_constrain(problem.model, q;
        include_tp=true, include_gq=true, rng=BridgeStan.StanRNG(problem.model, 42))
    for logical in (:x, :mu1, :mu2)
        coords = brm_output_coordinates(brm_descriptor(sb), logical, full_names)
        @test full[coords] ≈ getproperty(ref, logical) atol=1e-12 rtol=1e-12
    end
end

const MISSING_DERIVED_SHARED_KERNEL = @brm begin
    mi(x) ~ LogNormal(0, 0.5)
    mi(z) ~ LogNormal(0.2 + 0.3 * x, 0.5)
    combined = x + z
    doubled = 2 * combined
    mu1 ~ 1 + standardize(x) + log(doubled) + standardize(log(combined)) + (1 | p | subject)
    mu2 ~ 1 + standardize(x) + log(doubled) + standardize(log(combined)) + (1 | p | subject)
    effect(mu1, Intercept) ~ Normal(0.7, 0.4)
    out ~ kernel(y, mu1, mu2) do yy, m1, m2
        yy ~ normal(m1 + m2, 1.0)
        m1 + m2
    end
end

@stestset "shared derived missing covariate transforms survive frozen subject CV" begin
    sb = SBBRMI(MISSING_DERIVED_SHARED_KERNEL(MISSING_DERIVED_TOY); mod=@__MODULE__)
    cv = reprocess(sb, MISSING_DERIVED_TOY; resample_groups=[:subject])
    @test StanBlocks.stanc_check(BayesianRegressionModels.stan_code(cv); warn_pedantic=false).ok
    source, target = StanBlocks.stan_instantiate(sb.model), StanBlocks.stan_instantiate(cv.model)
    source_names, target_names = BridgeStan.param_unc_names(source.model), BridgeStan.param_unc_names(target.model)
    @test all("$(stem)_y_mis.$i" in source_names && "$(stem)_y_mis.$i" in target_names
        for stem in ("x", "z") for i in 1:2)
    @test length(source_names) - length(target_names) == 2 * length(MISSING_DERIVED_TOY.subject)
    random_outputs = brm_outputs(brm_descriptor(sb); role=:random_effect, kind=:parameter)
    random_coords = reduce(vcat, (brm_output_coordinates(output, source_names) for output in random_outputs))
    @test Set(setdiff(source_names, target_names)) ⊆ Set(source_names[random_coords])
    # The shared block's SD vector and correlation factor also have the
    # random-effect role. Their public constraints/type distinguish the
    # covariance parameters that remain fitted from the subject latents.
    covariance_outputs = filter(o -> !isempty(o.constraints) || o.type === :cholesky_factor_corr,
                                random_outputs)
    covariance_coords = reduce(vcat,
        (brm_output_coordinates(output, source_names) for output in covariance_outputs))
    @test length(covariance_coords) == 3
    @test Set(source_names[covariance_coords]) ⊆ Set(target_names)
    @test Set(target_names) ⊆ Set(source_names)
    source_q = [0.04 * (i - 5) for i in eachindex(source_names)]
    target_q = [source_q[findfirst(==(name), source_names)] for name in target_names]
    source_full = BridgeStan.param_constrain(source.model, source_q;
        include_tp=true, include_gq=true, rng=BridgeStan.StanRNG(source.model, 41))
    target_full = BridgeStan.param_constrain(target.model, target_q;
        include_tp=true, include_gq=true, rng=BridgeStan.StanRNG(target.model, 42))
    source_full_names = BridgeStan.param_names(source.model; include_tp=true, include_gq=true)
    target_full_names = BridgeStan.param_names(target.model; include_tp=true, include_gq=true)
    for logical in (:x, :z, :combined, :doubled)
        source_values = source_full[brm_output_coordinates(brm_descriptor(sb), logical, source_full_names)]
        target_values = target_full[brm_output_coordinates(brm_descriptor(cv), logical, target_full_names)]
        @test source_values == target_values
    end
    for (key, entry) in sb.preproc
        entry.kind in (:missing_standardize, :missing_center, :missing_zscale) || continue
        @test cv.data[key] == sb.data[key]
        @test cv.data[entry.const_.scale_key] == sb.data[entry.const_.scale_key]
    end
    @test brm_output(brm_descriptor(cv), :y; role=:posterior_predictive).source === :y
end

@stestset "frozen derived transform anchors survive group replay with new observations" begin
    sb = SBBRMI(MISSING_DERIVED_SHARED_KERNEL(MISSING_DERIVED_TOY); mod=@__MODULE__)
    changed = merge(MISSING_DERIVED_TOY, (;
        x=Union{Missing,Float64}[1.3, missing, 2.7, missing, 1.8],
        z=Union{Missing,Float64}[1.6, missing, 1.0, missing, 1.2],))
    cv = reprocess(sb, changed; resample_groups=[:subject])
    @test cv.data[:x_obs] == [1.3, 2.7, 1.8]
    @test cv.data[:z_obs] == [1.6, 1.0, 1.2]
    @test popcoefnames(cv.parent, :mu1) == popcoefnames(sb.parent, :mu1)
    for (key, entry) in sb.preproc
        entry.kind in (:missing_standardize, :missing_center, :missing_zscale) || continue
        @test haskey(cv.data, key)
        @test cv.data[key] == sb.data[key]
        @test cv.data[entry.const_.scale_key] == sb.data[entry.const_.scale_key]
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
    # A complete real column retains the authored observed law and has no
    # missing coordinates; the model body need not change with its mask.
    complete_mi = SBBRMI(MISSING_PLAIN(complete); mod=@__MODULE__)
    @test complete_mi.data[:x_obs] == complete.x
    @test isempty(complete_mi.data[:Jmis_x])
    @test StanBlocks.stanc_check(BayesianRegressionModels.stan_code(complete_mi)).ok
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
