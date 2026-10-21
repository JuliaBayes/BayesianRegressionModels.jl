# test/rk_model_offset.jl — RK `offset(x)` of model values (snag
# `rk-offset-sample-53f14e00`).
#
# `offset(theta)` adds a sampled scalar to a predictor with coefficient one, as
# SBBRMI does (`test/formula_offset.jl`). RKBRMI lowers it, and compositions
# with raw data or other model values, as a graph summand: no design column, no
# coefficient, the value's own coordinates. Each model is checked against an
# independent density oracle with finite-difference gradients and against the
# emitted Stan program.
#
# Run: julia --startup-file=no --project=test test/rk_model_offset.jl

include(joinpath(@__DIR__, "rk_consumer_support.jl"))

const OFFSET_PANEL = (;
    subject=["b", "c", "a"],
    t=[[0.1, 0.7], [0.3], [0.2, 0.8, 1.3]],
    y=[[0.2, -0.1], [0.4], [0.1, 0.3, -0.2]],
)

const OFFSET_ROWS = (;
    subject=[1, 2, 1, 2, 3],
    x=[0.1, -0.2, 0.3, 0.5, -0.4],
    exposure=[1.0, 2.0, 3.0, 4.0, 5.0],
    y=[0.1, 0.2, 0.3, 0.4, 0.0],
)

coordinates_matching(names, part) = findall(n -> occursin(part, string(n)), names)

# The reporter's shape: a declared scalar is the typical value of a
# per-subject log parameter whose between-subject variability is a random
# intercept, read in a plate cell.
plate_offset(data; theta_prior=:normal) = theta_prior === :normal ? @brm(data, begin
    sigma ~ Exponential(1)
    theta ~ Normal(-0.5, 1)
    log_k ~ 0 + offset(theta) + (1 | p | subject)
    sd(:, p) ~ Exponential(1)
    @plate for i in eachindex(log_k)
        loc[i] = exp(log_k[i]) .* t[i]
        y[i] ~ Normal(loc[i], sigma)
    end
end) : @brm(data, begin
    sigma ~ Exponential(1)
    theta ~ Flat(; upper=4.6)
    log_k ~ 0 + offset(theta) + (1 | p | subject)
    sd(:, p) ~ Exponential(1)
    @plate for i in eachindex(log_k)
        loc[i] = exp(log_k[i]) .* t[i]
        y[i] ~ Normal(loc[i], sigma)
    end
end)

@stestset "a sampled scalar offset beside a random intercept in a plate" begin
    data = OFFSET_PANEL
    saved = deepcopy(data)
    brmi = plate_offset(data)

    backend, problem = consumer_problem(brmi)
    names = coordinate_names(backend.model.layout)
    ith = coordinate(names, :theta)
    is = coordinate(names, :sigma)
    it = only(coordinates_matching(names, ".tau."))
    iz = coordinates_matching(names, ".z.")
    # The offset allocates nothing: theta keeps its own coordinate.
    @test length(names) == 6
    @test length(iz) == 3
    @test isempty(coordinates_matching(names, "pop_log_k"))
    levels = sort(unique(data.subject))
    order = [findfirst(==(label), levels) for label in data.subject]
    oracle(u) = begin
        tau, sigma = exp(u[it]), exp(u[is])
        log_k = u[ith] .+ tau .* u[iz][order]
        means = reduce(vcat, [exp(log_k[i]) .* data.t[i] for i in eachindex(data.t)])
        logpdf(Normal(-0.5, 1), u[ith]) + sum(logpdf.(Normal(), u[iz])) +
            logpdf(Exponential(), tau) + u[it] +
            logpdf(Exponential(), sigma) + u[is] +
            sum(logpdf.(Normal.(means, sigma), reduce(vcat, data.y)))
    end
    stan = consumer_stan(brmi, "rk-model-offset-plate")
    mapping = [names[ith] => "theta"; names[is] => "sigma";
        names[it] => "b_p_subject_tau.1";
        [names[iz[i]] => "b_p_subject_z_flat.$i" for i in 1:3]]
    for u in (zeros(6), collect(range(-0.2, 0.3; length=6)), fill(-0.1, 6))
        check_consumer_point(problem, u, oracle)
        check_consumer_stan(problem, stan, mapping, backend, u)
    end
    # The NLME view reads theta as an ordinary sampled scalar.
    view = brm_nlme_view(backend)
    @test view.role[ith] === :scalar
    @test all(view.role[i] === :subject_effect for i in iz)
    @test isequal(data, saved)
end

@stestset "a bounded flat scalar offset matches Stan" begin
    brmi = plate_offset(OFFSET_PANEL; theta_prior=:flat)
    backend, problem = consumer_problem(brmi)
    names = coordinate_names(backend.model.layout)
    iz = coordinates_matching(names, ".z.")
    stan = consumer_stan(brmi, "rk-model-offset-plate-flat")
    mapping = [:theta => "theta"; :sigma => "sigma";
        names[only(coordinates_matching(names, ".tau."))] => "b_p_subject_tau.1";
        [names[iz[i]] => "b_p_subject_z_flat.$i" for i in 1:3]]
    for u in (zeros(6), collect(range(-0.2, 0.3; length=6)))
        check_consumer_stan(problem, stan, mapping, backend, u)
    end
end

# Flat-row predictors: beside a random intercept, beside population terms,
# alone, composed with data, through an authored assignment and through
# another predictor.
const FLAT_CASES = [
    (:ranef, @brm(OFFSET_ROWS, begin
        s ~ Exponential(1)
        theta ~ Normal(0, 1)
        mu ~ 0 + offset(theta) + (1 | p | subject)
        sd(:, p) ~ Exponential(1)
        y ~ Normal(mu, s)
    end)),
    (:population, @brm(OFFSET_ROWS, begin
        s ~ Exponential(1)
        theta ~ Normal(0, 1)
        mu ~ 1 + x + offset(theta)
        effect(mu, :) ~ Normal(0, 1)
        y ~ Normal(mu, s)
    end)),
    (:alone, @brm(OFFSET_ROWS, begin
        s ~ Exponential(1)
        theta ~ Normal(0, 1)
        mu ~ 0 + offset(theta)
        y ~ Normal(mu, s)
    end)),
    (:composed, @brm(OFFSET_ROWS, begin
        s ~ Exponential(1)
        theta ~ Normal(0, 1)
        mu ~ 1 + offset(theta * log(exposure))
        effect(mu, :) ~ Normal(0, 1)
        y ~ Normal(mu, s)
    end)),
    (:assignment, @brm(OFFSET_ROWS, begin
        s ~ Exponential(1)
        theta ~ Normal(0, 1)
        shift = 2 * theta
        mu ~ 1 + offset(shift)
        effect(mu, :) ~ Normal(0, 1)
        y ~ Normal(mu, s)
    end)),
    (:predictor, @brm(OFFSET_ROWS, begin
        s ~ Exponential(1)
        eta ~ 1 + x
        effect(eta, :) ~ Normal(0, 1)
        mu ~ 0 + offset(eta) + (1 | p | subject)
        sd(:, p) ~ Exponential(1)
        y ~ Normal(mu, s)
    end)),
]

function flat_oracle(case, names, data)
    is = coordinate(names, :s)
    function value(u)
        s = exp(u[is])
        lp = logpdf(Exponential(), s) + u[is]
        if case === :predictor
            ia = coordinate(names, Symbol("pop_eta.beta_pop.1"))
            ix = coordinate(names, Symbol("pop_eta.beta_pop.2"))
            lp += logpdf(Normal(), u[ia]) + logpdf(Normal(), u[ix])
            shift = u[ia] .+ u[ix] .* data.x
        else
            ith = coordinate(names, :theta)
            lp += logpdf(Normal(), u[ith])
            theta = u[ith]
            shift = case === :composed ? theta .* log.(data.exposure) :
                case === :assignment ? 2 * theta : theta
        end
        mu = fill(0.0, length(data.y)) .+ shift
        if case in (:ranef, :predictor)
            it = only(coordinates_matching(names, ".tau."))
            iz = coordinates_matching(names, ".z.")
            tau = exp(u[it])
            lp += logpdf(Exponential(), tau) + u[it] + sum(logpdf.(Normal(), u[iz]))
            mu = mu .+ tau .* u[iz][data.subject]
        elseif case === :population
            ia = coordinate(names, Symbol("pop_mu.beta_pop.1"))
            ix = coordinate(names, Symbol("pop_mu.beta_pop.2"))
            lp += logpdf(Normal(), u[ia]) + logpdf(Normal(), u[ix])
            mu = mu .+ u[ia] .+ u[ix] .* data.x
        elseif case in (:composed, :assignment)
            ia = coordinate(names, :mu_Intercept)
            lp += logpdf(Normal(), u[ia])
            mu = mu .+ u[ia]
        end
        lp + sum(logpdf.(Normal.(mu, s), data.y))
    end
end

@stestset "model-value offsets on flat rows" begin
    for (case, brmi) in FLAT_CASES
        @testset "$case" begin
            backend, problem = consumer_problem(brmi)
            names = coordinate_names(backend.model.layout)
            oracle = flat_oracle(case, names, OFFSET_ROWS)
            for u in (zeros(length(names)), collect(range(-0.3, 0.2; length=length(names))))
                check_consumer_point(problem, u, oracle)
            end
        end
    end
end

@stestset "a row-aligned model-value offset on other rows is refused" begin
    data = merge(OFFSET_ROWS, (; w=[0.1, 0.2, 0.3]))
    brmi = @brm data begin
        s ~ Exponential(1)
        eta ~ 1 + w
        mu ~ 0 + offset(eta) + (1 | p | subject)
        y ~ Normal(mu, s)
    end
    # refused: adding a 3-row value to a 5-row predictor would misalign rows
    # (an offset adds a scalar or a value on the predictor's own rows).
    @test_throws "reads `w`-aligned values with 3 rows, but the predictor has 5 rows" RKBRMI(brmi)
end
