include(joinpath(@__DIR__, "rk_consumer_support.jl"))

module PublicGroupedBoundedResponse
using BayesianRegressionModels, Distributions

function build(data, law; nested=false)
    if nested
        return @brm data begin
            theta ~ 1 + (1 | p | subject)
            effect(theta, Intercept) ~ Normal(0, 0.7)
            sd(:, p) ~ Exponential(0.9)
            pred ~ kernel(t, theta) do ts, a
                ts * a
            end
            y ~ law(Normal(pred, 0.8); lower=lo, upper=hi)
            z ~ Normal(0, 1)
        end
    end
    @brm data begin
        theta ~ 1 + (1 | p | subject)
        effect(theta, Intercept) ~ Normal(0, 0.7)
        sd(:, p) ~ Exponential(0.9)
        pred ~ kernel(t, theta) do ts, a
            ts * a
        end
        ragged(y, event_subject) ~ law(Normal(pred, 0.8); lower=lo, upper=hi)
        z ~ Normal(lo, 1)
    end
end

function interval(data)
    @brm data begin
        theta ~ 1 + (1 | p | subject)
        effect(theta, Intercept) ~ Normal(0, 0.7)
        sd(:, p) ~ Exponential(0.9)
        pred ~ kernel(t, theta) do ts, a
            ts * a
        end
        ragged(y, event_subject) ~ interval_censored(Normal(pred, 0.8); upper=hi)
        z ~ Normal(lo, 1)
    end
end
end

@stestset "grouped bounded responses gather values and local bounds on the same axis" begin
    flat = (; subject=["b", "empty", "a"],
        t=[[0.2, 0.5], Float64[], [0.7]],
        event_subject=["a", "b", "b"], z=[0.2, -0.1, 0.3])
    permutation = [2, 3, 1]
    levels = sort(unique(flat.subject))
    subject_order = [findfirst(==(s), levels) for s in flat.subject]
    cases = (
        (:scalar, censored, merge(flat, (; y=[0.1, 0.0, 0.4], lo=0.0, hi=0.4)), false),
        (:rows, censored, merge(flat, (; y=[0.1, -0.2, 0.5],
            lo=[-0.1, -0.2, -0.3], hi=[0.6, 0.7, 0.5])), false),
        (:nested, censored, merge(flat, (; y=[[-0.2, 0.5], Float64[], [0.1]],
            lo=[[-0.2, -0.3], Float64[], [-0.1]],
            hi=[[0.7, 0.5], Float64[], [0.6]],
            z=[[0.2, -0.1], Float64[], [0.3]])), true),
        (:truncated, truncated, merge(flat, (; y=[0.1, 0.2, 0.3],
            lo=[-0.1, -0.2, -0.3], hi=[0.6, 0.7, 0.5])), false),
        (:interval, interval_censored, merge(flat, (; y=[0.1, -0.2, 0.3],
            lo=[-0.1, -0.2, -0.3], hi=[0.6, 0.7, 0.5])), false),
    )
    for (label, law, data, nested) in cases
        saved = deepcopy(data)
        brmi = law === interval_censored ? PublicGroupedBoundedResponse.interval(data) :
            PublicGroupedBoundedResponse.build(data, law; nested)
        backend, problem = consumer_problem(brmi)
        names = coordinate_names(backend.model.layout)
        ia = findfirst(==(:theta_Intercept), names)
        it = findfirst(n -> occursin("_sd.", string(n)), names)
        iz = findall(n -> occursin("_z.", string(n)), names)
        @test length(names) == 5
        @test length(iz) == 3
        @test backend.plan.columns[:y] == (nested ? reduce(vcat, data.y) : data.y[permutation])
        raw_lower = nested ? reduce(vcat, data.lo) : data.lo
        @test backend.plan.columns[:lo] == data.lo
        # An independent likelihood still reads the original lower-bound axis.
        gather(v) = v isa Real ? fill(v, 3) : nested ? reduce(vcat, v) : v[permutation]
        observed = nested ? reduce(vcat, data.y) : data.y[permutation]
        lo, hi = gather(data.lo), gather(data.hi)
        oracle(u) = begin
            tau = exp(u[it])
            theta = u[ia] .+ tau .* u[iz][subject_order]
            means = reduce(vcat, [data.t[i] .* theta[i] for i in eachindex(data.t)])
            base = logpdf(Normal(0, 0.7), u[ia]) + sum(logpdf.(Normal(), u[iz])) +
                logpdf(Exponential(0.9), tau) + u[it]
            evidence = sum(eachindex(observed)) do i
                d = Normal(means[i], 0.8)
                law === interval_censored ? log(cdf(d, hi[i]) - cdf(d, observed[i])) :
                    logpdf(law(d, lo[i], hi[i]), observed[i])
            end
            z = nested ? reduce(vcat, data.z) : data.z
            base + evidence + sum(logpdf.(Normal.(nested ? 0 : raw_lower, 1), z))
        end
        stan = consumer_stan(brmi, "grouped-bounded-$label"; mod=PublicGroupedBoundedResponse)
        mapping = [names[ia] => "pop_theta_beta_pop.1";
            names[it] => "b_p_subject_tau.1";
            [names[iz[i]] => "b_p_subject_z_flat.$i" for i in 1:3]]
        for u in (zeros(5), collect(range(-0.2, 0.3; length=5)), fill(-0.1, 5))
            check_consumer_point(problem, u, oracle)
            check_consumer_stan(problem, stan, mapping, backend, u)
        end
        @test isequal(data, saved)
    end
end

@stestset "grouped bounded responses reject invalid gathered evidence" begin
    data = (; subject=["b", "empty", "a"],
        t=[[0.2, 0.5], Float64[], [0.7]],
        y=[0.1, 0.0, 0.4], event_subject=["a", "b", "b"],
        lo=[0.0, 0.0, 0.0], hi=[1.0, 1.0, 1.0], z=zeros(3))
    @test_throws "outside its bounds" RKBRMI(PublicGroupedBoundedResponse.build(
        merge(data, (; y=[0.1, -0.1, 0.4])), censored))
    @test_throws "lower bounds must not exceed" RKBRMI(PublicGroupedBoundedResponse.build(
        merge(data, (; hi=[1.0, -0.1, 1.0])), censored))
    @test_throws "bound has 2 rows; expected 3" RKBRMI(PublicGroupedBoundedResponse.build(
        merge(data, (; lo=zeros(2))), censored))
    @test_throws "strictly below" RKBRMI(PublicGroupedBoundedResponse.interval(
        merge(data, (; hi=[1.0, 0.0, 1.0]))))
end
