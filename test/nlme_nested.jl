# Nested random effects (`nested_model` in nlme_fixtures.jl) through the NLME protocol.
# RUN: julia --project=<env with NLMEEstimation> test/nlme_nested.jl
using BridgeStan, NLMEEstimation, Random, LogDensityProblems, WarmupHMC
include(joinpath(@__DIR__, "nlme_fixtures.jl"))

function nested_oracle(m, θ, σ, H, data)
    scale = exp(only(σ))
    subject_index = Dict(level => i for (i, level) in enumerate(m.view.levels))
    unit_effect = Dict{String,Float64}()
    for (i, margins) in enumerate(m.view.subject_margins)
        for (k, margin) in enumerate(margins)
            margin.block.group === :unit || continue
            unit_effect[margin.level] = H[i][k]
        end
    end
    values = zeros(length(H))
    gradients = [zeros(length(block)) for block in H]
    for r in eachindex(data.y)
        i = subject_index[data.obs_subject[r]]
        k = findfirst(margin -> margin.block.group === :unit &&
            margin.level == data.unit[r], m.view.subject_margins[i])
        loc = only(θ) + H[i][1] + unit_effect[data.unit[r]]
        values[i] += logpdf(Normal(loc, scale), data.y[r])
        derivative = (data.y[r] - loc) / scale^2
        gradients[i][1] += derivative
        gradients[i][k] += derivative
    end
    values, gradients
end

@stestset "ragged SAEM population gradient handles mixed pointwise fields" begin
    data = merge(NESTED_DATA, (; subject_y=[0.1, 0.4, -0.3]))
    m = brm_nlme_model(RKBRMI(nested_mixed_observations(data));
        ad_backend=NLME_AD, subject_group=:subject,
        nested_groups=(unit=:obs_subject,))
    @test length(m.attribution) == 2
    θ = [0.15]
    σ = [log(0.7)]
    H = [[0.1, 0.2], [-0.2, 0.3, -0.1], [0.2, -0.1, 0.05, 0.3]]
    subjects = [3, 1]
    value, gradient = NLMEEstimation._saem_population_value_gradient(
        BatchedEvaluation(), m, subjects, θ, σ, H[subjects])

    scale = exp(only(σ))
    scale2 = scale^2
    expected = 0.0
    gθ = 0.0
    gσ = 0.0
    for i in subjects
        rows = findall(==(m.view.levels[i]), data.obs_subject)
        loc = only(θ) + H[i][1]
        residual = data.subject_y[i] - loc
        expected += logpdf(Normal(loc, scale), data.subject_y[i])
        gθ += residual / scale2
        gσ += residual^2 / scale2 - 1
        for r in rows
            k = findfirst(margin -> margin.block.group === :unit &&
                margin.level == data.unit[r], m.view.subject_margins[i])
            μ = loc + H[i][k]
            residual = data.y[r] - μ
            expected += logpdf(Normal(μ, scale), data.y[r])
            gθ += residual / scale2
            gσ += residual^2 / scale2 - 1
        end
    end
    @test value ≈ expected rtol=1e-12
    @test gradient ≈ [gθ, gσ] rtol=1e-9 atol=1e-10

    layout = nlme_layout(m)
    ω = omega_parameters(layout,
        [reshape([0.49], 1, 1), reshape([0.36], 1, 1)])
    fit = saem(m, θ, σ, ω; η0=H, rng=Random.Xoshiro(2026),
        options=SAEMOptions(iterations=2, burnin=1, mcmc_burnin=0,
            adapt_iterations=0, numeric_maxiter=2, mstep_interval=1,
            numeric_mstep=:mixture, omega_floor=0.05))
    @test fit.numeric_mstep == :mixture
    @test all(isfinite, (fit.theta..., fit.sigma..., fit.omega...))
end

@stestset "nested random effects retain unequal subject blocks" begin
    backend = RKBRMI(nested_model(NESTED_DATA))
    view = brm_nlme_view(backend; subject_group=:subject,
        nested_groups=(unit=:obs_subject,))
    @test view.levels == ["s1", "s2", "s3"]
    @test view.family_sizes == [1, 1]
    @test view.subject_families == [[1, 2], [1, 2, 2], [1, 2, 2, 2]]
    @test length.(view.subject_coordinates) == [2, 3, 4]
    @test all(view.subject[c] == i for (i, block) in
        enumerate(view.subject_coordinates) for c in block)

    m = brm_nlme_model(backend; ad_backend=NLME_AD,
        subject_group=:subject, nested_groups=(unit=:obs_subject,))
    layout = nlme_layout(m)
    @test layout isa RaggedNLMELayout
    @test [neta(layout, i) for i in 1:3] == [2, 3, 4]
    @test length.(parameter_names(m).eta) == [2, 3, 4]
    θ = [0.15]
    σ = [log(0.7)]
    H = [[0.1, 0.2], [-0.2, 0.3, -0.1], [0.2, -0.1, 0.05, 0.3]]
    expected, expected_gradient = nested_oracle(m, θ, σ, H, NESTED_DATA)
    values, gradients = brm_nlme_loglikelihoods_and_gradients(m, θ, σ, H)
    @test values ≈ expected rtol=1e-12
    for i in 1:3
        @test gradients[i] ≈ expected_gradient[i] rtol=1e-10 atol=1e-12
        @test brm_nlme_loglikelihood(m, i, θ, σ, H[i]) ≈ expected[i] rtol=1e-12
    end
    subset = [3, 1]
    subset_values, subset_gradients = conditional_loglikelihoods_and_gradients(
        m, subset, θ, σ, H[subset])
    @test subset_values ≈ expected[subset] rtol=1e-12
    @test subset_gradients[1] ≈ expected_gradient[3] rtol=1e-10
    @test subset_gradients[2] ≈ expected_gradient[1] rtol=1e-10
    for i in 1:3
        changed = deepcopy(H)
        changed[i][end] += 0.17
        moved = brm_nlme_loglikelihoods(m, θ, σ, changed)
        @test all(j == i || moved[j] ≈ values[j] for j in 1:3)
        @test moved[i] != values[i]
    end
    @test check_protocol(m, θ, σ, zeros(nomega(layout)))
end

@stestset "nested ownership is explicit and unique" begin
    backend = RKBRMI(nested_model(NESTED_DATA))
    # A second grouping without a declared parent is not an independent
    # subject partition.
    @test_throws BRMNLMEViewError brm_nlme_view(backend)
    bad_parent = collect(NESTED_DATA.obs_subject)
    bad_parent[7] = "s2" # u1 would belong to both s1 and s2.
    bad = merge(NESTED_DATA, (; obs_subject=bad_parent))
    @test_throws BRMNLMEViewError brm_nlme_view(RKBRMI(nested_model(bad));
        subject_group=:subject, nested_groups=(unit=:obs_subject,))
end

@stestset "ragged posthoc and SAEM retain every nested effect" begin
    m = brm_nlme_model(RKBRMI(nested_model(NESTED_DATA)); ad_backend=NLME_AD,
        subject_group=:subject, nested_groups=(unit=:obs_subject,))
    layout = nlme_layout(m)
    θ = [0.15]
    σ = [log(0.7)]
    ω = omega_parameters(layout,
        [reshape([0.49], 1, 1), reshape([0.36], 1, 1)])
    modes = empirical_bayes(m, θ, σ, ω; subjects=[3, 1])
    @test length.(modes.modes) == [4, 2]
    @test all(all(isfinite, block) for block in modes.modes)
    fit = saem(m, θ, σ, ω; rng=Random.Xoshiro(2026),
        options=SAEMOptions(iterations=3, burnin=1, mcmc_burnin=0,
            adapt_iterations=0, numeric_maxiter=2, mstep_interval=3,
            omega_floor=0.05))
    @test fit isa RaggedSAEMResult
    @test length.(fit.eta) == [2, 3, 4]
    @test length(fit.omega_matrix) == 2
    @test all(all(isfinite, block) for block in fit.omega_matrix)
end

@stestset "ragged NLME estimates reconstruct the full Stan point" begin
    brmi = nested_model(NESTED_DATA)
    rk = RKBRMI(brmi)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    map = brm_nlme_joint_map(rk, sb;
        path=joinpath(mktempdir(), "nested_joint.stan"),
        subject_group=:subject, nested_groups=(unit=:obs_subject,))
    θ = [0.15]
    σ = [log(0.7)]
    Ω = [reshape([0.49], 1, 1), reshape([0.36], 1, 1)]
    H = [[0.1, 0.2], [-0.2, 0.3, -0.1], [0.2, -0.1, 0.05, 0.3]]
    order = [3, 1, 2]
    start = brm_nlme_joint_start(map; theta=θ, sigma=σ, omega=Ω,
        eta=H[order], subjects=map.view.levels[order])
    inverse = brm_nlme_joint_values(map, start.stan)
    @test inverse.theta ≈ θ
    @test inverse.sigma ≈ σ
    @test length(inverse.omega) == 2
    @test all(isapprox(inverse.omega[f], Ω[f]) for f in 1:2)
    @test all(isapprox(inverse.eta[i], H[i]) for i in 1:3)
    @test length(start.fixed.indices) + length(start.free.indices) == length(start.stan)
    @test length(start.free.indices) == sum(length, H)
    @test start.checked.max_error <= 1e-9
    @test isfinite(BridgeStan.log_density(map.stan_model, start.stan;
        propto=false, jacobian=true))
    conditional = copy(start.stan)
    conditional[start.free.indices] .+= 0.05
    @test conditional[start.fixed.indices] == start.fixed.values
    @test isfinite(BridgeStan.log_density(map.stan_model, conditional;
        propto=false, jacobian=true))
end

@stestset "an NLMEEstimation fit conditions both backends for subject sampling" begin
    brmi = nested_model(NESTED_DATA)
    rk = RKBRMI(brmi)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    nesting = (; subject_group=:subject, nested_groups=(unit=:obs_subject,))
    map = brm_nlme_joint_map(rk, sb; path=joinpath(mktempdir(), "nested_fit.stan"),
        nesting...)
    m = brm_nlme_model(rk; ad_backend=NLME_AD, nesting...)
    layout = nlme_layout(m)
    ω = omega_parameters(layout,
        [reshape([0.49], 1, 1), reshape([0.36], 1, 1)])
    fit = saem(m, [0.15], [log(0.7)], ω; rng=Random.Xoshiro(2026),
        options=SAEMOptions(iterations=3, burnin=1, mcmc_burnin=0,
            adapt_iterations=0, numeric_maxiter=2, mstep_interval=3,
            omega_floor=0.05))
    start = brm_nlme_joint_start(map; theta=fit.theta, sigma=fit.sigma,
        omega=fit.omega_matrix, eta=fit.eta, subjects=map.view.levels[fit.subjects])
    for (backend, options) in ((map.sb, (;)), (map.rk, (; ad_backend=NLME_AD)))
        c = brm_nlme_conditional_problem(map, start; backend, options...)
        @test LogDensityProblems.dimension(c) == sum(length, fit.eta)
        result = adaptive_warmup_mcmc(Random.Xoshiro(7), c;
            init=(; position=copy(c.position),
                squared_scale=ones(length(c.position))),
            n_draws=100)
        draws = convert(Matrix{Float64}, result.posterior_position)
        @test size(draws, 2) >= 100
        @test all(isfinite, draws)
        values = brm_nlme_joint_values(c, draws[:, end])
        @test values.theta ≈ fit.theta atol=1e-10
        @test values.sigma ≈ fit.sigma atol=1e-10
        @test all(isapprox(values.omega[f], fit.omega_matrix[f]; atol=1e-10)
            for f in eachindex(fit.omega_matrix))
        @test length.(values.eta) == [2, 3, 4]
    end
end
