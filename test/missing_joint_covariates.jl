using Test
using BayesianRegressionModels
using StanBlocks
using Distributions
using Statistics
using LinearAlgebra
using BridgeStan
using LogDensityProblems
include(joinpath(@__DIR__, "testset_filter.jl"))

const JOINT_MISSING_TOY = (;
    subject=collect(1:4), u=[-0.6, 0.2, 0.7, 1.1],
    x=Union{Missing,Float64}[0.4, missing, 1.2, missing],
    z=Union{Missing,Float64}[0.8, 0.3, missing, missing],
    y=[0.3, 1.1, -0.2, 0.5])
const JOINT_MISSING_FORMULA = @brm begin
    L ~ LKJCovarianceFactor(2; scale_prior=Exponential(1), shape=2)
    x_loc ~ 1 + u
    z_loc ~ 1 + u
    effect(x_loc, Intercept) ~ Normal(0.2, 0.7)
    effect(z_loc, u) ~ Normal(-0.1, 0.4)
    mi([x, z]) ~ MvNormalCholesky([x_loc, z_loc], L)
    positive = exp(x)
    derived = positive + exp(z)
    log(mu) ~ 1 + center(x) + z + positive
    y ~ Normal(mu, 0.8)
end

const JOINT_POSITIVE_FORMULA = @brm begin
    L ~ LKJCovarianceFactor(2; scale_prior=Exponential(1))
    x_loc ~ 1 + u
    z_loc ~ 1 + u
    mi([x, z]) ~ MvNormalCholesky([x_loc, z_loc], L)
    physical_x = exp(x)
    physical_z = exp(z)
    area = sqrt(physical_x * physical_z)
    mu ~ 1 + standardize(physical_x) + zscale(log(physical_z)) + center(area)
    y ~ Normal(mu, 1)
end

@stestset "log-space completed assignments shadow raw columns with frozen anchors" begin
    df = merge(JOINT_MISSING_TOY, (;
        physical_x=[ismissing(v) ? missing : exp(v) for v in JOINT_MISSING_TOY.x],
        physical_z=[ismissing(v) ? missing : exp(v) for v in JOINT_MISSING_TOY.z]))
    original = deepcopy(df)
    sb = SBBRMI(JOINT_POSITIVE_FORMULA(df); mod=@__MODULE__)
    @test !haskey(sb.data, :physical_x)
    @test !haskey(sb.data, :physical_z)
    @test sb.data[:standardize_physical_x_mean] ≈ mean(skipmissing(df.physical_x))
    @test sb.data[:standardize_physical_x_scale] ≈ std(collect(skipmissing(df.physical_x)))
    problem = StanBlocks.stan_instantiate(sb.model)
    names = BridgeStan.param_unc_names(problem.model)
    full_names = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
    q = [0.03 * (i - 6) for i in eachindex(names)]
    desc = brm_descriptor(sb)
    full = BridgeStan.param_constrain(problem.model, q; include_tp=true, include_gq=true,
        rng=BridgeStan.StanRNG(problem.model, 7))
    physical = Dict(column => full[brm_output_coordinates(desc, column, full_names)]
                    for column in (:physical_x, :physical_z, :area))
    @test all(>(0), physical[:physical_x]) && all(>(0), physical[:physical_z])
    @test physical[:area] ≈ sqrt.(physical[:physical_x] .* physical[:physical_z])
    for (physical_name, log_name) in ((:physical_x, :x), (:physical_z, :z))
        observed = findall(!ismissing, getproperty(df, log_name))
        @test physical[physical_name][observed] ≈ getproperty(df, physical_name)[observed]
    end
    # Assignments take precedence even over populated same-named df columns.
    decoy = merge(df, (; physical_x=fill(99., 4), physical_z=fill(77., 4)))
    @test BayesianRegressionModels.stan_code(SBBRMI(JOINT_POSITIVE_FORMULA(decoy); mod=@__MODULE__)) ==
          BayesianRegressionModels.stan_code(sb)
    @test isequal(original, df)
    updated = merge(df, (; x=Union{Missing,Float64}[0.6, missing, 1.4, missing]))
    replay = reprocess(sb, updated)
    @test replay.data[:standardize_physical_x_mean] == sb.data[:standardize_physical_x_mean]
    @test replay.data[:standardize_physical_x_scale] == sb.data[:standardize_physical_x_scale]
    @test BayesianRegressionModels.stan_code(replay) == BayesianRegressionModels.stan_code(sb)
    fresh = reprocess(sb, updated; freeze_constants=false)
    @test fresh.data[:standardize_physical_x_mean] ≈ mean(exp.(collect(skipmissing(updated.x))))
end

@stestset "public completed joint columns and descriptor log aliases" begin
    sb = SBBRMI(JOINT_POSITIVE_FORMULA(JOINT_MISSING_TOY); mod=@__MODULE__)
    desc = brm_descriptor(sb)
    @test all(in(brm_columns(desc)), (:x, :z))
    @test all(column -> any(output -> output.logical === column, desc.outputs),
              (:x, :z, :physical_x, :physical_z, :area))
end

const JOINT_ROW_VECTOR_FORMULA = @brm begin
    L ~ LKJCovarianceFactor(2; scale_prior=Exponential(1))
    locations ~ MvNormal(loc_prior, 0.5)
    mi([x, z]) ~ MvNormalCholesky([locations, 0.2 + u], L)
    mu ~ 1 + x + z
    y ~ Normal(mu, 1)
end

@stestset "joint block complete and entirely missing columns with vector locations" begin
    for (x, z, count_missing) in (
        ([0.4, 0.7, 1.2, 0.9], [0.8, 0.3, 0.6, 0.2], 0),
        (fill(missing, 4), [0.8, 0.3, 0.6, 0.2], 4),
        (fill(missing, 4), fill(missing, 4), 8))
        df = merge(JOINT_MISSING_TOY, (; loc_prior=zeros(4), x=Union{Missing,Float64}[x...],
                                      z=Union{Missing,Float64}[z...]))
        sb = SBBRMI(JOINT_ROW_VECTOR_FORMULA(df); mod=@__MODULE__)
        problem = StanBlocks.stan_instantiate(sb.model)
        names = BridgeStan.param_unc_names(problem.model)
        @test count(name -> startswith(name, "brm_joint_x__z_y_mis."), names) == count_missing
        q = fill(0.1, length(names))
        lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
        @test isfinite(lp) && all(isfinite, gradient)
        @test reprocess(sb, df).data[:brm_joint_x__z_Jmis] == sb.data[:brm_joint_x__z_Jmis]
    end
end

const JOINT_REPLAY_FORMULA = @brm begin
    L ~ LKJCovarianceFactor(2; scale_prior=Exponential(1))
    x_loc ~ 1 + u
    z_loc ~ 1 + u
    mi([x, z]) ~ MvNormalCholesky([x_loc, z_loc], L)
    mu ~ 1 + standardize(exp(x)) + z + (1 | subject)
    y ~ Normal(mu, 1)
end

@stestset "joint missing coordinates remain fitted during same-cohort group resampling" begin
    sb = SBBRMI(JOINT_REPLAY_FORMULA(JOINT_MISSING_TOY); mod=@__MODULE__, total_groups=())
    cv = reprocess(sb, JOINT_MISSING_TOY; resample_groups=[:subject])
    source, target = StanBlocks.stan_instantiate(sb.model), StanBlocks.stan_instantiate(cv.model)
    source_names, target_names = BridgeStan.param_unc_names(source.model), BridgeStan.param_unc_names(target.model)
    missing_names = filter(name -> startswith(name, "brm_joint_x__z_y_mis."), source_names)
    @test length(missing_names) == 4
    @test all(in(target_names), missing_names)
    source_q = [0.03 * (i - 4) for i in eachindex(source_names)]
    target_q = [source_q[findfirst(==(name), source_names)] for name in target_names]
    completions = Dict{Symbol,Vector{Float64}}[]
    for (model, fitted, q) in ((source, sb, source_q), (target, cv, target_q))
        names = BridgeStan.param_names(model.model; include_tp=true, include_gq=true)
        draws = [BridgeStan.param_constrain(model.model, q; include_tp=true, include_gq=true,
            rng=BridgeStan.StanRNG(model.model, seed)) for seed in (9, 29)]
        desc = brm_descriptor(fitted)
        completion = Dict{Symbol,Vector{Float64}}()
        for column in (:x, :z)
            coordinates = brm_output_coordinates(desc, column, names)
            completion[column] = draws[1][coordinates]
            @test draws[1][coordinates] == draws[2][coordinates]
            observed = findall(!ismissing, getproperty(JOINT_MISSING_TOY, column))
            @test draws[1][coordinates][observed] == getproperty(JOINT_MISSING_TOY, column)[observed]
        end
        push!(completions, completion)
    end
    @test completions[1] == completions[2]
    updated = merge(JOINT_MISSING_TOY, (;
        x=Union{Missing,Float64}[0.6, missing, 1.4, missing], u=[-0.5, 0.3, 0.9, 1.2]))
    refreshed = reprocess(sb, updated; resample_groups=[:subject])
    @test refreshed.data[:brm_joint_x__z_obs] == [0.6, 0.8, 0.3, 1.4]
    anchor = only(key for (key, entry) in sb.preproc if entry.kind === :missing_standardize)
    @test refreshed.data[anchor] == sb.data[anchor]
    @test BayesianRegressionModels.stan_code(refreshed) == BayesianRegressionModels.stan_code(cv)
    changed = merge(JOINT_MISSING_TOY, (; z=Union{Missing,Float64}[missing, 0.3, 0.6, missing]))
    # Reassigning fitted block coordinates is invalid even in group replay.
    @test_throws r"same fitted missing-row positions" reprocess(sb, changed; resample_groups=[:subject])
end

# Independent full joint density: data and row order come from the fixture,
# while the Stan names only identify the unconstrained coordinates.
function joint_missing_reference(q, names, df; anchors=JOINT_MISSING_TOY)
    at = Dict(name => q[i] for (i, name) in enumerate(names))
    missing = [at["brm_joint_x__z_y_mis.$i"] for i in 1:count(ismissing, vcat(df.x, df.z))]
    cursor = 0
    complete = Matrix{Float64}(undef, length(df.y), 2)
    for row in eachindex(df.y)
        for (col, column) in enumerate((df.x, df.z))
            if ismissing(column[row])
                cursor += 1
                complete[row, col] = missing[cursor]
            else
                complete[row, col] = Float64(column[row])
            end
        end
    end
    x, z = complete[:, 1], complete[:, 2]
    scales = exp.([at["L_scales.$i"] for i in 1:2])
    corr_name = only(filter(name -> startswith(name, "L_L_corr."), names))
    rho = tanh(at[corr_name])
    covariance = Symmetric(Diagonal(scales) * [1.0 rho; rho 1.0] * Diagonal(scales))
    bx = [at["pop_x_loc_beta_pop.$i"] for i in 1:2]
    bz = [at["pop_z_loc_beta_pop.$i"] for i in 1:2]
    beta = [at["pop_log_mu_beta_pop.$i"] for i in 1:4]
    xloc, zloc = bx[1] .+ bx[2] .* df.u, bz[1] .+ bz[2] .* df.u
    row_lp = [logpdf(MvNormal([xloc[i], zloc[i]], covariance), complete[i, :])
              for i in eachindex(df.y)]
    positive = exp.(x)
    mu = exp.(beta[1] .+ beta[2] .* (x .- mean(skipmissing(anchors.x))) .+
              beta[3] .* z .+ beta[4] .* positive)
    # K=2, eta=2: rho has normalized density 3/4*(1-rho^2).
    # Include the tanh Jacobian and both positive scale Jacobians.
    prior = sum(logpdf.(Exponential(1), scales)) + sum(log.(scales)) +
            log(0.75) + 2 * log1p(-rho^2) +
            logpdf(Normal(0.2, 0.7), bx[1]) + logpdf(Normal(), bx[2]) +
            logpdf(Normal(), bz[1]) + logpdf(Normal(-0.1, 0.4), bz[2]) +
            sum(logpdf.(Normal(), beta))
    lp = prior + sum(row_lp) + sum(logpdf(Normal(mu[i], 0.8), df.y[i]) for i in eachindex(mu))
    (; lp, row_lp, x, z, positive, derived=positive .+ exp.(z), mu)
end

function joint_finite_difference(f, q)
    [begin
        plus, minus = copy(q), copy(q)
        plus[i] += 1e-6
        minus[i] -= 1e-6
        (f(plus) - f(minus)) / 2e-6
    end for i in eachindex(q)]
end

@stestset "joint block independent normalized density gradients and completed outputs" begin
    sb = SBBRMI(JOINT_MISSING_FORMULA(JOINT_MISSING_TOY); mod=@__MODULE__)
    problem = StanBlocks.stan_instantiate(sb.model)
    names = BridgeStan.param_unc_names(problem.model)
    @test count(name -> startswith(name, "brm_joint_x__z_y_mis."), names) == 4
    full_names = BridgeStan.param_names(problem.model; include_tp=true, include_gq=true)
    descriptor = brm_descriptor(sb)
    for offset in (0.0, 0.13, -0.19)
        q = [offset + 0.04 * (i - 5) for i in eachindex(names)]
        reference = joint_missing_reference(q, names, JOINT_MISSING_TOY)
        gradient = zeros(length(q))
        lp, _ = BridgeStan.log_density_gradient!(problem.model, q, gradient;
            propto=false, jacobian=true)
        @test lp ≈ reference.lp atol=1e-9
        @test gradient ≈ joint_finite_difference(
            v -> joint_missing_reference(v, names, JOINT_MISSING_TOY).lp, q) atol=2e-7 rtol=2e-7
        pointwise = brm_execute(descriptor, :pointwise_loglik; problem, draws=q, seed=61)
        @test pointwise.brm_joint_x__z_observed_likelihood ≈ reference.row_lp
        full = BridgeStan.param_constrain(problem.model, q; include_tp=true, include_gq=true,
            rng=BridgeStan.StanRNG(problem.model, 61))
        for output in (:x, :z, :positive, :derived, :mu)
            values = full[brm_output_coordinates(descriptor, output, full_names)]
            @test values ≈ getproperty(reference, output)
        end
        for column in (:x, :z)
            observed = findall(!ismissing, getproperty(JOINT_MISSING_TOY, column))
            @test getproperty(reference, column)[observed] == getproperty(JOINT_MISSING_TOY, column)[observed]
        end
    end
    changed = merge(JOINT_MISSING_TOY, (; x=Union{Missing,Float64}[0.6, missing, 1.4, missing]))
    replay = reprocess(sb, changed)
    replay_problem = StanBlocks.stan_instantiate(replay.model)
    q = [0.02 * (i - 3) for i in eachindex(names)]
    reference = joint_missing_reference(q, names, changed)
    gradient = zeros(length(q))
    lp, _ = BridgeStan.log_density_gradient!(replay_problem.model, q, gradient;
        propto=false, jacobian=true)
    @test lp ≈ reference.lp atol=1e-9
    @test gradient ≈ joint_finite_difference(
        v -> joint_missing_reference(v, names, changed).lp, q) atol=2e-7 rtol=2e-7
end

@stestset "partially observed joint block emission and replay" begin
    original = deepcopy(JOINT_MISSING_TOY)
    brmi = JOINT_MISSING_FORMULA(JOINT_MISSING_TOY)
    @test length(outcomes(brmi)) == 2
    sb = SBBRMI(brmi; mod=@__MODULE__)
    code = BayesianRegressionModels.stan_code(sb)
    checked = StanBlocks.stanc_check(code; warn_pedantic=false)
    checked.ok || @error "stanc rejected joint imputation" output=checked.output code
    @test checked.ok
    @test sb.data[:brm_joint_x__z_Jmis] == [3, 6, 7, 8]
    @test sb.data[:brm_joint_x__z_obs] == [0.4, 0.8, 0.3, 1.2]
    @test sb.data[:center_x_mean] == 0.8
    @test isequal(original, JOINT_MISSING_TOY)
    changed = merge(JOINT_MISSING_TOY, (;
        x=Union{Missing,Float64}[0.6, missing, 1.4, missing]))
    replay = reprocess(sb, changed)
    @test replay.data[:brm_joint_x__z_obs] == [0.6, 0.8, 0.3, 1.4]
    @test replay.data[:center_x_mean] == sb.data[:center_x_mean]
    @test BayesianRegressionModels.stan_code(replay) == code
    moved = merge(JOINT_MISSING_TOY, (;
        x=Union{Missing,Float64}[missing, 0.6, 1.4, missing]))
    # Frozen positions are fitted coordinate identities, never a new mask.
    @test_throws r"same fitted missing-row positions" reprocess(sb, moved)
    fresh = reprocess(sb, moved; freeze_constants=false)
    @test fresh.data[:brm_joint_x__z_Jmis] == [1, 6, 7, 8]
end
