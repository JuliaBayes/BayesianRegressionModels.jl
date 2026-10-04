include(joinpath(@__DIR__, "rk_consumer_support.jl"))

module PublicKernelScope
using BayesianRegressionModels, Distributions, StanBlocks
StanBlocks.@deffun public_cell(ts::vector[n], a::real) = ts * a
public_cell(ts::AbstractVector, a::Real) = ts .* a
function build(data)
    @brm data begin
        theta ~ 1 + (1 | p | subject)
        effect(theta, Intercept) ~ Normal(0, 1)
        sd(:, p) ~ Exponential(1)
        sigma ~ Exponential(1)
        pred ~ kernel(t, theta) do ts, a
            return public_cell(ts, a)
        end
        y ~ Normal(pred, sigma)
    end
end
end

@stestset "kernel response aliases withhold one likelihood on independent axes" begin
    data = (; index=[[1, 2, 1]], y=[[0.1, -0.2, 0.4]],
        z=[[0.2, -0.3]], catalog=[[0.2, 0.7]])
    saved = deepcopy(data)
    brmi = @brm data begin
        shift ~ Normal(0, 1)
        pred ~ kernel(index, y, z, catalog) do ii, yy, zz, cc
            mu_y = cc[ii] .+ shift
            mu_z = cc .+ shift
            yy ~ normal(mu_y, 1)
            zz ~ normal(mu_z, 1)
            mu_y
        end
    end
    copy_brmi = deepcopy(brmi)
    @test copy_brmi !== brmi
    for (selection, active) in ((:yy, :z), (:z, :y))
        backend = check_rk_source_roundtrip(RKBRMI(brmi; held_out=selection))
        @test coordinate_names(backend.model.layout) == [:shift]
        @test BRM._rk_observed_names(backend.plan) == (active,)
        problem = rk_logdensity_problem(backend; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=(), held_out=selection)
        stan = BRM.stan_instantiate(sb; path=joinpath(tempdir(), "brm-rk-consumer",
            "kernel-held-out-$selection.stan"))
        oracle(u) = logpdf(Normal(), u[1]) +
            sum(logpdf.(Normal.(
                (active === :y ? data.catalog[1][data.index[1]] : data.catalog[1]) .+ u[1], 1),
                getproperty(data, active)[1]))
        for u in ([0.0], [0.2], [-0.3])
            check_consumer_point(problem, u, oracle)
            check_consumer_stan(problem, stan, [:shift => "shift"], backend, u)
        end
    end
    @test_throws "covers every observation" RKBRMI(brmi; held_out=(:yy, :z))
    @test_throws "unknown response" RKBRMI(brmi; held_out=:typo)
    @test isequal(data, saved)
end

@stestset "ragged kernel readers preserve subject joins and original axes" begin
    data = (; subject=["b", "empty", "a"],
        t=[[0.1, 0.7], Float64[], [0.2, 0.8, 1.3]],
        y=[[0.2, -0.1], Float64[], [0.1, 0.3, -0.2]],
        z=[0.1, -0.2, 0.3], event_subject=["a", "b", "b", "a", "a"],
        event_time=[0.2, 0.1, 0.7, 0.8, 1.3])
    saved = deepcopy(data)
    for mode in (:grouped, :ragged, :scope)
        brmi = if mode === :grouped
            @brm data begin
                theta ~ 1 + (1 | p | subject)
                effect(theta, Intercept) ~ Normal(0, 1)
                sd(:, p) ~ Exponential(1)
                sigma ~ Exponential(1)
                pred ~ kernel(t, theta) do ts, a
                    ts * a
                end
                y ~ Normal(pred, sigma)
            end
        elseif mode === :ragged
            @brm data begin
                theta ~ 1 + (1 | p | subject)
                effect(theta, Intercept) ~ Normal(0, 1)
                sd(:, p) ~ Exponential(1)
                sigma ~ Exponential(1)
                pred ~ kernel(ragged(event_time, event_subject), theta) do ts, a
                    ts * a
                end
                y ~ Normal(pred, sigma)
            end
        else
            PublicKernelScope.build(data)
        end
        backend, problem = consumer_problem(brmi)
        names = coordinate_names(backend.model.layout)
        ia = findfirst(==(:theta_Intercept), names)
        it = findfirst(n -> occursin("_sd.", string(n)), names)
        iz = findall(n -> occursin("_z.", string(n)), names)
        is = findfirst(==(:sigma), names)
        @test length(names) == 6
        @test length(iz) == 3
        levels = sort(unique(data.subject))
        order = [findfirst(==(label), levels) for label in data.subject]
        oracle(u) = begin
            tau, sigma = exp(u[it]), exp(u[is])
            theta = u[ia] .+ tau .* u[iz][order]
            means = reduce(vcat, [data.t[i] .* theta[i] for i in eachindex(data.t)])
            logpdf(Normal(), u[ia]) + sum(logpdf.(Normal(), u[iz])) +
                logpdf(Exponential(), tau) + u[it] +
                logpdf(Exponential(), sigma) + u[is] +
                sum(logpdf.(Normal.(means, sigma), reduce(vcat, data.y)))
        end
        stan = consumer_stan(brmi, "ragged-reader-$mode";
            mod=mode === :scope ? PublicKernelScope : @__MODULE__)
        mapping = [names[ia] => "pop_theta_beta_pop.1";
            names[it] => "b_p_subject_tau.1"; names[is] => "sigma";
            [names[iz[i]] => "b_p_subject_z_flat.$i" for i in 1:3]]
        for u in (zeros(6), collect(range(-0.2, 0.3; length=6)), fill(-0.1, 6))
            check_consumer_point(problem, u, oracle)
            check_consumer_stan(problem, stan, mapping, backend, u)
        end
        artifact = BRM.emit_rk_artifact(brmi; case_id="ragged-reader-$mode")
        @test length(artifact.defs) >= 2
        @test all(d -> d.head in (:function, :(=)), artifact.defs)
        @test isequal(data, saved)
    end
end

@stestset "kernel cell inputs keep distinct vector axes" begin
    data = (; index=[[1, 2, 1]], y=[[0.1, -0.2, 0.4]], catalog=[[0.2, 0.7]])
    saved = deepcopy(data)
    brmi = @brm data begin
        shift ~ Normal(0, 1)
        pred ~ kernel(index, y, catalog) do ii, yy, cc
            mu = cc[ii] .+ shift
            yy ~ normal(mu, 1)
            mu
        end
    end
    backend, problem = consumer_problem(brmi)
    @test coordinate_names(backend.model.layout) == [:shift]
    stan = consumer_stan(brmi, "kernel-distinct-axes")
    oracle(u) = logpdf(Normal(), u[1]) +
        sum(logpdf.(Normal.(data.catalog[1][data.index[1]] .+ u[1], 1), data.y[1]))
    for u in ([0.0], [0.2], [-0.3])
        check_consumer_point(problem, u, oracle)
        check_consumer_stan(problem, stan, [:shift => "shift"], backend, u)
    end
    @test isequal(data, saved)
end
