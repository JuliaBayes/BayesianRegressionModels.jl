# brm_nlme_view: the θ / Ω / σ / η_i partition of an RK-lowered population model.
#
# The partition is checked against the executable model, not only against its
# own bookkeeping: perturbing one subject's block must change the pointwise
# densities of exactly that subject's rows, and nothing else.
#
# RUN: julia --project=test test/nlme_view.jl [testset-filter...]
include(joinpath(@__DIR__, "nlme_fixtures.jl"))

@stestset "partition of a correlated two-parameter population model" begin
    backend = RKBRMI(nlme_pk(NLME_DATA))
    view = brm_nlme_view(backend)
    n = length(view.levels)
    @test view.group === :subject
    @test n == 4
    @test sort(string.(view.levels)) == ["s1", "s2", "s3", "s4"]
    @test length(view.coordinates) == length(view.role) == length(view.subject)
    @test all(!=(:unclaimed), view.role)
    @test count(==(:population), view.role) == 3
    @test count(==(:scalar), view.role) == 1
    @test count(==(:subject_scale), view.role) == 2
    @test count(==(:subject_correlation), view.role) == 1
    @test count(==(:subject_effect), view.role) == 2n
    @test all(length.(view.subject_coordinates) .== 2)
    # Each subject coordinate is claimed by exactly one subject, in its block.
    for (i, block) in enumerate(view.subject_coordinates), c in block
        @test view.subject[c] == i
    end
    @test sort(reduce(vcat, view.subject_coordinates)) ==
        findall(r -> r === :subject_effect, view.role)
    @test [(m.predictor, m.coefficient) for m in view.block_margins] ==
        [(:CL, :Intercept), (:V, :Intercept)]
    # The subject-constant weight column also shifts through CL's random
    # intercept, even without a random weight slope.
    @test Set((r.predictor, r.coefficient) for r in view.mu_references) ==
        Set([(:CL, :Intercept), (:CL, :wt), (:V, :Intercept)])
    @test only(r.design for r in view.mu_references if r.coefficient === :wt) ==
        [0.1, -0.3, 0.6, -0.8]
    for ref in view.mu_references
        @test ref.link === :log
        @test view.role[ref.population] === :population
        @test view.block_margins[ref.subject].predictor === ref.predictor
    end
    # Rows partition the observation axis by subject label.
    @test sort(reduce(vcat, view.rows)) == collect(eachindex(NLME_DATA.y))
    for (i, rows) in enumerate(view.rows)
        @test all(r -> NLME_DATA.subject[r] == string(view.levels[i]), rows)
    end
end

@stestset "a within-subject covariate cannot shift through a random intercept" begin
    data = merge(NLME_DATA, (; wt=[i == 4 ? 0.25 : w for (i, w) in enumerate(NLME_DATA.wt)]))
    view = brm_nlme_view(RKBRMI(nlme_pk(data)))
    @test Set((r.predictor, r.coefficient) for r in view.mu_references) ==
        Set([(:CL, :Intercept), (:V, :Intercept)])

    brmi = @brm data begin
        sigma ~ Exponential(1)
        log(CL) ~ 1 + wt + (1 + wt | pk | subject)
        log(V) ~ 1 + (1 | pk | subject)
        m = dose / V * exp(-CL / V * t)
        y ~ Normal(m, sigma)
    end
    slope_view = brm_nlme_view(RKBRMI(brmi))
    slope = only(r for r in slope_view.mu_references if r.coefficient === :wt)
    @test slope_view.block_margins[slope.subject].coefficient === :wt
    @test all(==(1.0), slope.design)
    model = brm_nlme_model(RKBRMI(brmi); ad_backend=NLME_AD)
    θ = fill(0.1, length(model.theta))
    σ = fill(-1.0, length(model.sigma))
    H = zeros(length(model.view.block_margins), length(model.view.levels))
    shifted_θ = copy(θ)
    shifted_θ[findfirst(==(slope.population), model.theta)] += 0.3
    shifted_H = copy(H)
    shifted_H[slope.subject, :] .-= 0.3
    @test brm_nlme_loglikelihoods(model, shifted_θ, σ, shifted_H) ≈
        brm_nlme_loglikelihoods(model, θ, σ, H) rtol=1e-12
end

@stestset "a subject block moves exactly that subject's densities" begin
    backend = RKBRMI(nlme_pk(NLME_DATA))
    view = brm_nlme_view(backend)
    query = pointwise_query(backend)
    u0 = fill(0.05, length(view.coordinates))
    base = Base.invokelatest(query, u0).y
    @test length(base) == length(NLME_DATA.y)
    for (i, block) in enumerate(view.subject_coordinates)
        u = copy(u0)
        u[block] .+= [0.4, -0.3]
        moved = Base.invokelatest(query, u).y
        changed = findall(.!isapprox.(moved, base; atol=0, rtol=1e-12))
        @test sort(changed) == sort(view.rows[i])
    end
    # A population coordinate reaches every row.
    pop = first(r.population for r in view.mu_references)
    u = copy(u0); u[pop] += 0.3
    moved = Base.invokelatest(query, u).y
    @test all(.!isapprox.(moved, base; atol=0, rtol=1e-12))
end

@stestset "an @plate population-PK cell has the same NLME reading" begin
    backend = RKBRMI(nlme_plate_pk(NLME_PLATE_DATA))
    view = brm_nlme_view(backend)
    @test length(view.levels) == 3
    @test count(==(:population), view.role) == 3
    @test count(==(:scalar), view.role) == 1
    @test count(r -> r === :subject_scale || r === :subject_correlation, view.role) == 3
    @test all(length.(view.subject_coordinates) .== 2)
    @test Set((r.predictor, r.coefficient) for r in view.mu_references) ==
        Set([(:log_CL, :Intercept), (:log_CL, :weight), (:log_V, :Intercept)])
    @test only(r.design for r in view.mu_references if r.coefficient === :weight) ==
        NLME_PLATE_DATA.weight
    # The in-cell observation keeps one density array per subject; a subject
    # block moves exactly its own array.
    query = pointwise_query(backend)
    u0 = fill(0.05, length(view.coordinates))
    base = Base.invokelatest(query, u0).dv
    @test length.(base) == length.(NLME_PLATE_DATA.dv)
    for (i, block) in enumerate(view.subject_coordinates)
        u = copy(u0)
        u[block] .+= [0.4, -0.3]
        moved = Base.invokelatest(query, u).dv
        changed = findall(k -> !isapprox(moved[k], base[k]; atol=0, rtol=1e-12),
            eachindex(base))
        @test changed == view.rows[i]
    end
end

@stestset "legacy closure kernel cells retain the NLME subject reading" begin
    legacy = RKBRMI(nlme_kernel_pk(NLME_PLATE_DATA))
    plate = RKBRMI(nlme_plate_pk(NLME_PLATE_DATA))
    @test legacy.plan isa BRM._RKValuePlan
    @test plate.plan isa BRM._RKValuePlan
    old_view = brm_nlme_view(legacy)
    new_view = brm_nlme_view(plate)
    @test old_view.coordinates == new_view.coordinates
    @test old_view.role == new_view.role
    @test old_view.subject_coordinates == new_view.subject_coordinates
    @test old_view.rows == new_view.rows

    old_model = brm_nlme_model(legacy; ad_backend=NLME_AD)
    new_model = brm_nlme_model(plate; ad_backend=NLME_AD)
    named = Dict(Symbol("pop_log_CL.beta_pop.1") => 0.2,
        Symbol("pop_log_CL.beta_pop.2") => 0.1, :log_V_Intercept => 1.6)
    θ = [named[c] for c in old_view.coordinates[old_model.theta]]
    σ = [log(0.8)]
    H = [0.15 -0.2 0.05; -0.1 0.25 0.3]
    old_values, old_gradients = brm_nlme_loglikelihoods_and_gradients(
        old_model, θ, σ, H)
    new_values, new_gradients = brm_nlme_loglikelihoods_and_gradients(
        new_model, θ, σ, H)
    @test old_values ≈ new_values rtol=1e-12
    @test old_gradients ≈ new_gradients rtol=1e-10
    for i in eachindex(old_view.levels)
        @test brm_nlme_loglikelihood(old_model, i, θ, σ, H[:, i]) ≈
            brm_nlme_loglikelihood(new_model, i, θ, σ, H[:, i]) rtol=1e-12
    end
end

@stestset "models without an NLME reading are refused by name" begin
    no_ranef = @brm NLME_DATA begin
        sigma ~ Exponential(1)
        mu ~ 1 + wt
        y ~ Normal(mu, sigma)
    end
    err = try brm_nlme_view(RKBRMI(no_ranef)); nothing catch e; e end
    @test err isa BRMNLMEViewError
    @test occursin("no subject-level random effects", err.message)

    crossed_data = (; NLME_DATA..., site=repeat(["a", "b"], 7)[1:13])
    crossed = @brm crossed_data begin
        sigma ~ Exponential(1)
        mu ~ 1 + wt + (1 | subject) + (1 | site)
        y ~ Normal(mu, sigma)
    end
    err = try brm_nlme_view(RKBRMI(crossed)); nothing catch e; e end
    @test err isa BRMNLMEViewError
    @test occursin("2 grouping columns", err.message)
end

@stestset "lockstep per-subject log-likelihoods and gradients match an oracle" begin
    backend = RKBRMI(nlme_plate_pk(NLME_PLATE_DATA))
    m = brm_nlme_model(backend; ad_backend=NLME_AD)
    n = length(m.view.levels)
    @test m.eta_blocks == [2]
    @test length(m.theta) == 3 && length(m.sigma) == 1
    θ = [0.4, -0.7, 0.2][sortperm(m.view.coordinates[m.theta])]
    σ = [log(0.8)]
    H = [0.15 -0.2 0.05; -0.1 0.25 0.3]
    # σ is the unconstrained residual scale: exp(σ) is the constrained sigma.
    u = BRM._brm_nlme_point(m, θ, σ, H)
    @test constrain(backend.model.layout, u).sigma ≈ exp(only(σ))
    values, G = brm_nlme_loglikelihoods_and_gradients(m, θ, σ, H)
    @test values ≈ plate_pk_oracle(m, θ, σ, H) rtol=1e-12
    @test brm_nlme_loglikelihoods(m, θ, σ, H) == values
    # Attribution is complete: subjects sum to the model's whole likelihood.
    @test sum(values) ≈ Base.invokelatest(likelihood_query(backend), u) rtol=1e-12
    # The η gradients against the oracle's analytic gradients.
    @test size(G) == size(H)
    @test G ≈ plate_pk_oracle_gradients(m, θ, σ, H) rtol=1e-10
    # Moving one subject's η moves only that subject's value.
    H2 = copy(H); H2[:, 2] .+= [0.3, -0.2]
    moved = brm_nlme_loglikelihoods(m, θ, σ, H2)
    @test moved[[1, 3]] == values[[1, 3]]
    @test moved[2] != values[2]
end

@stestset "a subject plate cell gives one subject value and η gradient" begin
    m = brm_nlme_model(RKBRMI(nlme_plate_pk(NLME_PLATE_DATA)); ad_backend=NLME_AD)
    named = Dict(Symbol("pop_log_CL.beta_pop.1") => 0.2,
        Symbol("pop_log_CL.beta_pop.2") => 0.1, :log_V_Intercept => 1.6)
    θ = [named[c] for c in m.view.coordinates[m.theta]]
    σ = [log(0.8)]
    H = [0.15 -0.2 0.05; -0.1 0.25 0.3]
    oracle = plate_pk_oracle(m, θ, σ, H)
    gradients = plate_pk_oracle_gradients(m, θ, σ, H)
    for i in eachindex(m.view.levels)
        η = H[:, i]
        @test brm_nlme_loglikelihood(m, i, θ, σ, η) ≈ oracle[i] rtol=1e-12
        value, gradient = brm_nlme_loglikelihood_and_gradient(m, i, θ, σ, η)
        @test value ≈ oracle[i] rtol=1e-12
        @test gradient ≈ gradients[:, i] rtol=1e-10
    end
    @test_throws BoundsError brm_nlme_loglikelihood(m, 4, θ, σ, H[:, 1])
    @test_throws DimensionMismatch brm_nlme_loglikelihood(m, 1, θ, σ, [0.1])
end

@stestset "one subject's work stays flat as other subjects are added" begin
    d = NLME_PLATE_DATA
    measured = map((3, 9, 27)) do n
        data = (; subject=["s$j" for j in 1:n],
            dose=[d.dose[mod1(j, 3)] for j in 1:n],
            weight=[d.weight[mod1(j, 3)] for j in 1:n],
            t=[d.t[mod1(j, 3)] for j in 1:n],
            dv=[d.dv[mod1(j, 3)] for j in 1:n])
        m = brm_nlme_model(RKBRMI(nlme_plate_pk(data)); ad_backend=NLME_AD)
        named = Dict(Symbol("pop_log_CL.beta_pop.1") => 0.2,
            Symbol("pop_log_CL.beta_pop.2") => 0.1, :log_V_Intercept => 1.6)
        θ = [named[c] for c in m.view.coordinates[m.theta]]
        σ = [0.0]
        η = [0.05, -0.1]
        i = findfirst(==("s2"), m.view.levels)
        value() = brm_nlme_loglikelihood(m, i, θ, σ, η)
        gradient() = brm_nlme_loglikelihood_and_gradient(m, i, θ, σ, η)
        value(); gradient()
        primal_bytes = minimum(@allocated(value()) for _ in 1:3)
        gradient_bytes = minimum(@allocated(gradient()) for _ in 1:3)
        (; result=gradient(), primal_bytes, gradient_bytes)
    end
    @test all(x -> isapprox(x.result[1], measured[1].result[1]; rtol=1e-12), measured)
    @test all(x -> isapprox(x.result[2], measured[1].result[2]; rtol=1e-10), measured)
    # The RK cell query and Enzyme η reverse must not allocate with cohort size.
    @test maximum(x.primal_bytes for x in measured) <=
        1.2minimum(x.primal_bytes for x in measured)
    @test maximum(x.gradient_bytes for x in measured) <=
        1.2minimum(x.gradient_bytes for x in measured)
end

@stestset "flat-row attribution sums to the whole likelihood" begin
    backend = RKBRMI(nlme_pk(NLME_DATA))
    m = brm_nlme_model(backend; ad_backend=NLME_AD)
    n = length(m.view.levels)
    θ = fill(0.1, length(m.theta)); σ = [log(2.0)]
    H = 0.1 .* reshape(collect(1:2n), 2, n) ./ n
    u = BRM._brm_nlme_point(m, θ, σ, H)
    values = brm_nlme_loglikelihoods(m, θ, σ, H)
    @test sum(values) ≈ Base.invokelatest(likelihood_query(backend), u) rtol=1e-12
    pointwise = Base.invokelatest(pointwise_query(backend), u).y
    for i in 1:n
        @test values[i] ≈ sum(pointwise[m.view.rows[i]]) rtol=1e-13
    end
    # Decision 110evqv requires O(n_i) single-subject calls. Flat rows have no
    # subject cell, so the lockstep path is the supported evaluation here.
    err = try brm_nlme_loglikelihood(m, 1, θ, σ, H[:, 1]); nothing catch e; e end
    @test err isa BRMNLMEViewError
    @test occursin("flat-row grouped", err.message)
end

@stestset "a ragged(y, group) join over interleaved rows is attributed by subject" begin
    backend = RKBRMI(nlme_ragged_pk(NLME_RAGGED_DATA))
    m = brm_nlme_model(backend; ad_backend=NLME_AD)
    n = length(m.view.levels)
    @test n == 3
    @test m.eta_blocks == [2]
    θ = [0.4, 2.1]
    σ = [log(0.8)]
    H = [0.15 -0.2 0.05; -0.1 0.25 0.3]
    u = BRM._brm_nlme_point(m, θ, σ, H)
    values, G = brm_nlme_loglikelihoods_and_gradients(m, θ, σ, H)
    @test values ≈ ragged_pk_oracle(m, θ, σ, H, NLME_RAGGED_DATA) rtol=1e-12
    @test brm_nlme_loglikelihoods(m, θ, σ, H) == values
    @test sum(values) ≈ Base.invokelatest(likelihood_query(backend), u) rtol=1e-12
    @test G ≈ ragged_pk_oracle_gradients(m, θ, σ, H, NLME_RAGGED_DATA) rtol=1e-10
    # Against the executable model: a subject block moves exactly the joined
    # densities attributed to that subject, and every subject's value but its own
    # stays put.
    subjects = Dict(m.attribution)[:y]
    base = Base.invokelatest(pointwise_query(backend), u).y
    @test length(base) == length(subjects) == length(NLME_RAGGED_DATA.y)
    for (i, block) in enumerate(m.view.subject_coordinates)
        moved_u = copy(u)
        moved_u[block] .+= [0.4, -0.3]
        moved = Base.invokelatest(pointwise_query(backend), moved_u).y
        @test findall(.!isapprox.(moved, base; atol=0, rtol=1e-12)) == findall(==(i), subjects)
        H2 = copy(H); H2[:, i] .+= [0.3, -0.2]
        moved_values = brm_nlme_loglikelihoods(m, θ, σ, H2)
        @test moved_values[setdiff(1:n, i)] == values[setdiff(1:n, i)]
        @test moved_values[i] != values[i]
    end
end
