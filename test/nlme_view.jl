# brm_nlme_view: the θ / Ω / σ / η_i partition of an RK-lowered population model.
#
# The partition is checked against the executable model, not only against its
# own bookkeeping: perturbing one subject's block must change the pointwise
# densities of exactly that subject's rows, and nothing else.
#
# RUN: julia --project=test test/nlme_view.jl [testset-filter...]
using Test, BayesianRegressionModels, Distributions
using ReactiveKernels, ReactiveKernelsPPL, Enzyme
const BRM = BayesianRegressionModels
include(joinpath(@__DIR__, "testset_filter.jl"))

# Four subjects, interleaved rows, unequal row counts.
const NLME_DATA = (;
    subject=["s1", "s2", "s3", "s1", "s4", "s2", "s3", "s1", "s4", "s2", "s3", "s4", "s1"],
    t=[0.5, 0.5, 0.5, 1.0, 0.5, 1.5, 2.0, 3.0, 2.0, 4.0, 6.0, 6.0, 8.0],
    dose=fill(100.0, 13),
    wt=[0.1, -0.3, 0.6, 0.1, -0.8, -0.3, 0.6, 0.1, -0.8, -0.3, 0.6, -0.8, 0.1],
    y=[18.0, 22.0, 15.0, 16.5, 25.0, 17.0, 10.0, 12.0, 14.0, 9.0, 4.0, 5.0, 3.5],
)

function nlme_pk(data)
    @brm data begin
        sigma ~ Exponential(1)
        log(CL) ~ 1 + wt + (1 | pk | subject)
        log(V) ~ 1 + (1 | pk | subject)
        m = dose / V * exp(-CL / V * t)
        y ~ Normal(m, sigma)
    end
end

const RKEXT = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)

pointwise_query(backend) = prepare_query(backend.model,
    RKEXT._rk_translated_plan(backend.plan), :pointwise)

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
