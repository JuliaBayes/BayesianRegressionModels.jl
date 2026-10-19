# test/kernel_population_lp.jl — per-subject linear predictors without a
# random effect in kernel cells (snag `plate-population-96580da1`).
#
# A population predictor (a typical value plus subject-level covariates, no
# between-subject variability: `rate ~ 1 + w`) defines no grouping. It is a
# row-ordered vector over the subject frame and is read as `rate[i]`, beside a
# random-effect predictor (`test/plate_grouping.jl` pins it against the old
# one-element gather) or in a no-random-effects panel, whose subjects come from
# the pre-grouped columns. Here both are checked against an independent density
# oracle with finite-difference gradients on RK and matched on Stan.
#
# Run: julia --startup-file=no --project=test test/kernel_population_lp.jl

include(joinpath(@__DIR__, "rk_consumer_support.jl"))

const POP_PANEL = (;
    subject=["b", "c", "a"],
    w=[0.4, -0.3, 0.1],
    t=[[0.1, 0.7], [0.3], [0.2, 0.8, 1.3]],
    y=[[0.2, -0.1], [0.4], [0.1, 0.3, -0.2]],
)

# The reporter's shape: a typical value plus a covariate, no `(1 | subject)`.
pop_grouped(data) = @brm data begin
    theta ~ 1 + (1 | p | subject)
    rate ~ 1 + w
    effect(theta, Intercept) ~ Normal(0, 1)
    effect(rate, Intercept) ~ Normal(0, 1)
    effect(rate, w) ~ Normal(0, 1)
    sd(:, p) ~ Exponential(1)
    sigma ~ Exponential(1)
    @plate for i in eachindex(theta)
        loc[i] = theta[i] .* t[i] .+ rate[i]
        y[i] ~ Normal(loc[i], sigma)
    end
end

pop_panel(data) = @brm data begin
    rate ~ 1 + w
    effect(rate, Intercept) ~ Normal(0, 1)
    effect(rate, w) ~ Normal(0, 1)
    sigma ~ Exponential(1)
    @plate for i in eachindex(rate)
        loc[i] = rate[i] .* t[i]
        y[i] ~ Normal(loc[i], sigma)
    end
end

# The cell slices only the predictor beside an omitted outcome, so the subject
# count comes from the column the predictor reads.
pop_panel_prior(data) = @brm data begin
    rate ~ 1 + w
    sigma ~ Exponential(1)
    @plate for i in eachindex(rate)
        loc[i] = exp(rate[i])
        z[i] ~ Normal(loc[i], sigma)
    end
end

# A string categorical as the only covariate (snag `axis-witness-and-6a7991c9`):
# the predictor's row axis is the categorical column's, which never reaches Stan
# data itself, so the intercept must be sized from an emitted column rather than
# from a zero `offset(...)` witness.
const POP_CATEGORICAL = merge(POP_PANEL, (; ind=["BLCL", "AITL", "BLCL"]))

# Default population priors: an `effect(rate, ...)` address on a plain String
# categorical predictor is a separate, still-open gap (`popcoefnames`; owner
# todo `0cidxk1`).
pop_categorical(data) = @brm data begin
    theta ~ 1 + (1 | p | subject)
    rate ~ 1 + ind
    sd(:, p) ~ Exponential(1)
    sigma ~ Exponential(1)
    @plate for i in eachindex(theta)
        loc[i] = theta[i] .* t[i] .+ rate[i]
        y[i] ~ Normal(loc[i], sigma)
    end
end

# The reporter's workaround: a zero `offset(...)` names the subject axis.
pop_categorical_witness(data) = @brm data begin
    theta ~ 1 + (1 | p | subject)
    rate ~ 1 + ind + offset(zero_axis)
    sd(:, p) ~ Exponential(1)
    sigma ~ Exponential(1)
    @plate for i in eachindex(theta)
        loc[i] = theta[i] .* t[i] .+ rate[i]
        y[i] ~ Normal(loc[i], sigma)
    end
end

pop_categorical_panel(data) = @brm data begin
    rate ~ 1 + ind
    sigma ~ Exponential(1)
    @plate for i in eachindex(rate)
        loc[i] = rate[i] .* t[i]
        y[i] ~ Normal(loc[i], sigma)
    end
end

# refused: a predictor without a random effect is read on the subject frame, so
# an event-frame predictor read as `x[i]` would silently misalign
# (snag plate-body-reads-c69f8a88).
pop_panel_other_axis(data) = @brm data begin
    rate ~ 1 + event_w
    sigma ~ Exponential(1)
    @plate for i in eachindex(t)
        loc[i] = rate[i] .* t[i]
        y[i] ~ Normal(loc[i], sigma)
    end
end

function coordinate(names, name)
    index = findfirst(==(name), names)
    index === nothing && error("missing coordinate $name in $names")
    index
end

@stestset "a population predictor beside a random-effect predictor" begin
    data = POP_PANEL
    saved = deepcopy(data)
    brmi = pop_grouped(data)

    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    # The predictor stays a parameter and is sliced by the plate.
    @test !haskey(sb.data, :rate)
    @test sb.data[:kernel_nsub_loc] == 3
    code = StanBlocks.stan_code(sb.model)
    @test occursin(r"rate\[\s*plate_i__pl_1\s*\]", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok

    backend, problem = consumer_problem(brmi)
    names = coordinate_names(backend.model.layout)
    ia = coordinate(names, :theta_Intercept)
    ra = coordinate(names, Symbol("pop_rate.beta_pop.1"))
    rw = coordinate(names, Symbol("pop_rate.beta_pop.2"))
    it = findfirst(n -> occursin(".tau.", string(n)), names)
    iz = findall(n -> occursin(".z.", string(n)), names)
    is = coordinate(names, :sigma)
    @test length(names) == 8
    @test length(iz) == 3
    levels = sort(unique(data.subject))
    order = [findfirst(==(label), levels) for label in data.subject]
    oracle(u) = begin
        tau, sigma = exp(u[it]), exp(u[is])
        theta = u[ia] .+ tau .* u[iz][order]
        rate = u[ra] .+ u[rw] .* data.w
        means = reduce(vcat, [theta[i] .* data.t[i] .+ rate[i] for i in eachindex(data.t)])
        logpdf(Normal(), u[ia]) + logpdf(Normal(), u[ra]) + logpdf(Normal(), u[rw]) +
            sum(logpdf.(Normal(), u[iz])) +
            logpdf(Exponential(), tau) + u[it] +
            logpdf(Exponential(), sigma) + u[is] +
            sum(logpdf.(Normal.(means, sigma), reduce(vcat, data.y)))
    end
    stan = consumer_stan(brmi, "population-lp-grouped")
    mapping = [names[ia] => "pop_theta_beta_pop.1";
        names[ra] => "pop_rate_beta_pop.1"; names[rw] => "pop_rate_beta_pop.2";
        names[it] => "b_p_subject_tau.1"; names[is] => "sigma";
        [names[iz[i]] => "b_p_subject_z_flat.$i" for i in 1:3]]
    for u in (zeros(8), collect(range(-0.2, 0.3; length=8)), fill(-0.1, 8))
        check_consumer_point(problem, u, oracle)
        check_consumer_stan(problem, stan, mapping, backend, u)
    end
    @test isequal(data, saved)
end

@stestset "a no-random-effects panel reads population predictors per subject" begin
    data = POP_PANEL
    saved = deepcopy(data)
    brmi = pop_panel(data)

    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    @test !haskey(sb.data, :rate)
    @test sb.data[:kernel_nsub_loc] == 3
    @test StanBlocks.stanc_check(StanBlocks.stan_code(sb.model); warn_pedantic=false).ok

    backend, problem = consumer_problem(brmi)
    names = coordinate_names(backend.model.layout)
    ra = coordinate(names, Symbol("pop_rate.beta_pop.1"))
    rw = coordinate(names, Symbol("pop_rate.beta_pop.2"))
    is = coordinate(names, :sigma)
    @test length(names) == 3
    oracle(u) = begin
        sigma = exp(u[is])
        rate = u[ra] .+ u[rw] .* data.w
        means = reduce(vcat, [rate[i] .* data.t[i] for i in eachindex(data.t)])
        logpdf(Normal(), u[ra]) + logpdf(Normal(), u[rw]) +
            logpdf(Exponential(), sigma) + u[is] +
            sum(logpdf.(Normal.(means, sigma), reduce(vcat, data.y)))
    end
    stan = consumer_stan(brmi, "population-lp-panel")
    mapping = [names[ra] => "pop_rate_beta_pop.1"; names[rw] => "pop_rate_beta_pop.2";
        names[is] => "sigma"]
    for u in (zeros(3), [0.2, -0.4, 0.1], [-0.3, 0.5, -0.2])
        check_consumer_point(problem, u, oracle)
        check_consumer_stan(problem, stan, mapping, backend, u)
    end
    @test isequal(data, saved)
end

@stestset "a panel slicing only population predictors counts their rows" begin
    data = (; w=POP_PANEL.w)
    sb = @test_logs (:warn, r"`z` bind\(s\) no data column") match_mode=:any SBBRMI(
        pop_panel_prior(data); mod=@__MODULE__, total_groups=())
    @test sb.data[:kernel_nsub_loc] == 3
    @test StanBlocks.stanc_check(StanBlocks.stan_code(sb.model); warn_pedantic=false).ok
end

@stestset "a panel's population predictor on another row axis is refused" begin
    data = merge(POP_PANEL, (; event_w=[0.1, 0.2, 0.3, 0.4, 0.5]))
    brmi = pop_panel_other_axis(data)
    @test_throws "data column `event_w` has 5 rows for 3 subjects" SBBRMI(
        brmi; mod=@__MODULE__, total_groups=())
    @test_throws "data column `event_w` has 5 rows for 3 subjects" RKBRMI(brmi)
end

# Both string spellings of the categorical: a plain `Vector{String}` lowers
# through the shared design's level indicators, a string-leveled
# `CategoricalVector` through its own `cat_rate_ind` block. Neither raw column
# is Stan data, and neither needs the zero-offset witness.
const POP_CATEGORICAL_SPELLINGS = (
    ("String", POP_CATEGORICAL),
    ("CategoricalVector", merge(POP_CATEGORICAL,
        (; ind=CategoricalArrays.categorical(POP_CATEGORICAL.ind)))),
)

with_witness(data) = merge(data, (; zero_axis=zeros(length(data.subject))))

@stestset "a categorical-only population predictor needs no axis witness on Stan" begin
    for (label, data) in POP_CATEGORICAL_SPELLINGS
        saved = deepcopy(data)
        brmi = pop_categorical(data)
        sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
        @test !haskey(sb.data, :rate)
        @test !haskey(sb.data, :ind)
        code = StanBlocks.stan_code(sb.model)
        @test occursin(r"rate\[\s*plate_i__pl_1\s*\]", code)
        @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
        names, values = stan_density_points(brmi, "population-lp-categorical-$label")
        witness_names, witness_values = stan_density_points(
            pop_categorical_witness(with_witness(data)), "population-lp-witness-$label")
        @test names == witness_names
        @test values ≈ witness_values atol=1e-12
        @test isequal(data, saved)

        panel = SBBRMI(pop_categorical_panel(data); mod=@__MODULE__, total_groups=())
        @test panel.data[:kernel_nsub_loc] == 3
        @test StanBlocks.stanc_check(StanBlocks.stan_code(panel.model);
            warn_pedantic=false).ok
    end
end

@stestset "a categorical-only population predictor needs no axis witness on RK" begin
    for (label, data) in POP_CATEGORICAL_SPELLINGS
        backend, problem = consumer_problem(pop_categorical(data))
        witness_backend, witness_problem = consumer_problem(
            pop_categorical_witness(with_witness(data)))
        names = coordinate_names(backend.model.layout)
        @test names == coordinate_names(witness_backend.model.layout)
        n = length(names)
        for u in (zeros(n), fill(0.1, n), collect(range(-0.4, 0.4; length=n)))
            value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
            witness_value, witness_gradient =
                LogDensityProblems.logdensity_and_gradient(witness_problem, u)
            @test value ≈ witness_value atol=1e-12
            @test gradient ≈ witness_gradient atol=1e-10
        end
        _, panel = consumer_problem(pop_categorical_panel(data))
        @test isfinite(LogDensityProblems.logdensity(panel,
            zeros(LogDensityProblems.dimension(panel))))
    end
end

# The same intercept extent outside any plate: an observed top-level predictor
# whose first covariate is a string categorical.
@stestset "a top-level predictor led by a string categorical needs no axis witness" begin
    flat = (; ind=["AITL", "BLCL", "AITL", "BLCL"], w=[0.1, 0.2, 0.3, 0.4],
        yflat=[0.2, 0.4, -0.1, 0.3])
    led(data) = @brm data begin
        mu ~ 1 + ind + w
        sigma ~ Exponential(1)
        yflat ~ Normal(mu, sigma)
    end
    witnessed(data) = @brm data begin
        mu ~ 1 + ind + w + offset(zero_axis)
        sigma ~ Exponential(1)
        yflat ~ Normal(mu, sigma)
    end
    names, values = stan_density_points(led(flat), "population-lp-top-level")
    witness_names, witness_values = stan_density_points(
        witnessed(merge(flat, (; zero_axis=zeros(4)))), "population-lp-top-level-witness")
    @test names == witness_names
    @test values ≈ witness_values atol=1e-12
end
