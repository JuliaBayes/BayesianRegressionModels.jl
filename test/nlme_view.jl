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
    # Both margins are mu-referenced on their log-scale intercepts.
    @test length(view.mu_references) == 2
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
        Set([(:log_CL, :Intercept), (:log_V, :Intercept)])
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
    # Central-difference reference for the η gradients (test oracle only).
    step = 1e-6
    for i in 1:n, k in 1:size(H, 1)
        plus, minus = copy(H), copy(H)
        plus[k, i] += step; minus[k, i] -= step
        reference = (plate_pk_oracle(m, θ, σ, plus)[i] -
            plate_pk_oracle(m, θ, σ, minus)[i]) / 2step
        @test G[k, i] ≈ reference rtol=1e-6 atol=1e-8
    end
    # Moving one subject's η moves only that subject's value.
    H2 = copy(H); H2[:, 2] .+= [0.3, -0.2]
    moved = brm_nlme_loglikelihoods(m, θ, σ, H2)
    @test moved[[1, 3]] == values[[1, 3]]
    @test moved[2] != values[2]
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
end
