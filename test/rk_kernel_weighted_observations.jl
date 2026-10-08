include(joinpath(@__DIR__, "rk_consumer_support.jl"))

# In-cell SLIC observation weighting `weighted(family, weight, args...)`: a
# power likelihood scaling each row's log density by its weight. The RK
# backend must observe the family's own law with that weight, never bind the
# family token as data.
module KernelWeightedObservations
using BayesianRegressionModels, StanBlocks, Distributions
import ReactiveKernels
import BayesianRegressionModels: _rk_observation_source!

# A public caller-owned scalar law with an exact native graph counterpart.
StanBlocks.@deffun begin
    @lhs @lpxf shifted_normal_lpdf(y::vector[n], location::vector[n],
            shift::real, scale::real)::real = begin
        result::real = 0.0
        for i in 1:n
            result += normal_lpdf(y[i], location[i] + shift, scale)
        end
        return result
    end
    shifted_normal_lpdfs(y::vector[n], location::vector[n],
            shift::real, scale::real)::vector[n] = begin
        result::vector[n]
        for i in 1:n
            result[i] = normal_lpdf(y[i], location[i] + shift, scale)
        end
        return result
    end
    shifted_normal_rng(vector[n], location::vector[n],
            shift::real, scale::real)::vector[n] = begin
        result::vector[n]
        for i in 1:n
            result[i] = normal_rng(location[i] + shift, scale)
        end
        return result
    end
end

function _rk_observation_source!(definitions, bindings, entry,
        ::typeof(shifted_normal))
    push!(definitions, :(ReactiveKernels.@kernel $entry(value, location, shift, scale) = begin
        shifted_residual = (value - location - shift) / scale
        shifted_logdensity = -0.5 * log(2 * pi) - log(scale) -
            0.5 * shifted_residual * shifted_residual
        return shifted_logdensity
    end))
    :done
end

function build(data, form)
    if form === :token
        @brm data begin
            s ~ Normal(0, 1)
            sigma ~ Exponential(1)
            @plate for i in eachindex(x)
                m = s .* x[i]
                y[i] ~ weighted(normal, w[i], m, sigma)
                pred[i] = x[i]
            end
        end
    elseif form === :call
        @brm data begin
            s ~ Normal(0, 1)
            sigma ~ Exponential(1)
            @plate for i in eachindex(x)
                m = s .* x[i]
                y[i] ~ weighted(normal(m, sigma), w[i])
                pred[i] = x[i]
            end
        end
    elseif form === :expression
        @brm data begin
            s ~ Normal(0, 1)
            sigma ~ Exponential(1)
            @plate for i in eachindex(x)
                m = s .* x[i]
                v = 0.5 .* w[i]
                y[i] ~ weighted(normal, v .+ 0.25, m, sigma)
                pred[i] = x[i]
            end
        end
    elseif form === :subject
        @brm data begin
            s ~ Normal(0, 1)
            sigma ~ Exponential(1)
            @plate for i in eachindex(x)
                m = s .* x[i]
                y[i] ~ weighted(normal, ws[i], m, sigma)
                pred[i] = x[i]
            end
        end
    elseif form === :caller
        @brm data begin
            s ~ Normal(0, 1)
            sigma ~ Exponential(1)
            delta ~ Normal(0, 0.5)
            @plate for i in eachindex(x)
                m = s .* x[i]
                y[i] ~ weighted(shifted_normal, w[i], m, delta, sigma)
                pred[i] = x[i]
            end
        end
    elseif form === :caller_call
        @brm data begin
            s ~ Normal(0, 1)
            sigma ~ Exponential(1)
            delta ~ Normal(0, 0.5)
            @plate for i in eachindex(x)
                m = s .* x[i]
                y[i] ~ weighted(shifted_normal(m, delta, sigma), w[i])
                pred[i] = x[i]
            end
        end
    else
        @brm data begin
            s ~ Normal(0, 1)
            sigma ~ Exponential(1)
            @plate for i in eachindex(x)
                m = s .* x[i]
                y[i] ~ interval_censored(normal, lo[i], hi[i], m, sigma)
                pred[i] = x[i]
            end
        end
    end
end
end

const WEIGHTED_DATA = (; x=[[1.0, 2.0], Float64[], [0.5, 1.5, -0.4]],
    y=[[0.1, 0.2], Float64[], [0.3, 0.5, -0.1]],
    w=[[1.0, 0.0], Float64[], [2.0, 0.5, 1.0]],
    ws=[1.5, 0.7, 0.25])

# The weight reader keeps only the data it reads, so RKPPL binds it as data.
function weight_reader_arguments(main)
    calls = [statement.args[2] for statement in main.args
        if Meta.isexpr(statement, :(=)) && occursin("_weight_", string(statement.args[1]))]
    only(calls).args[2:end]
end

# A per-subject observation is one nested plate `y[i] .~ weighted.(law, w)`.
# Its weight reads only data (a bound port or a reader over data ports), so
# RKPPL binds it as data.
function nested_weight_reads(main)
    plate = only(s for s in main.args if Meta.isexpr(s, :macrocall))
    cell = only(x for x in last(last(plate.args).args).args if !(x isa LineNumberNode))
    weight = cell.args[3].args[2].args[2]
    reads = BRM._rk_source_symbols!(Set{Symbol}(), weight)
    definitions = Dict(x.args[1] => x.args[2] for x in main.args if Meta.isexpr(x, :(=)))
    pending = collect(reads)
    while !isempty(pending)
        definition = get(definitions, pop!(pending), nothing)
        definition === nothing && continue
        for name in BRM._rk_source_symbols!(Set{Symbol}(), definition)
            name in reads || (push!(reads, name); push!(pending, name))
        end
    end
    reads
end

@stestset "in-cell weighted native family observes a data-weighted power likelihood" begin
    data = WEIGHTED_DATA
    saved = deepcopy(data)
    row_weights = Dict(:token => data.w, :call => data.w,
        :expression => [0.5 .* w .+ 0.25 for w in data.w],
        :subject => [fill(ws, length(x)) for (x, ws) in zip(data.x, data.ws)])
    for form in (:token, :call, :expression, :subject)
        brmi = KernelWeightedObservations.build(data, form)
        backend, problem = consumer_problem(brmi)
        names = coordinate_names(backend.model.layout)
        @test sort(names) == [:s, :sigma]
        is, iσ = findfirst(==(:s), names), findfirst(==(:sigma), names)
        oracle(u) = begin
            s, sigma = u[is], exp(u[iσ])
            logpdf(Normal(), s) + logpdf(Exponential(), sigma) + u[iσ] +
                sum((weight * logpdf(Normal(s * x, sigma), y)
                    for (xs, ys, weights) in zip(data.x, data.y, row_weights[form])
                    for (x, y, weight) in zip(xs, ys, weights)); init=0.0)
        end
        stan = consumer_stan(brmi, "kernel-weighted-$form";
            mod=KernelWeightedObservations)
        for u in ([0.0, 0.0], [0.3, -0.2], [-0.4, 0.25])
            check_consumer_point(problem, u, oracle)
            check_consumer_stan(problem, stan, [:s => "s", :sigma => "sigma"], backend, u)
        end
        main = BRM._rk_emit_ast(backend.plan).main
        @test occursin("y[i] .~ weighted.(Normal.(", sprint(Base.show_unquoted, main))
        @test isdisjoint(nested_weight_reads(main), (:s, :sigma))
    end
    @test isequal(data, saved)
end

@stestset "in-cell weighted caller-owned law scales its scalar graph by the row weight" begin
    data = WEIGHTED_DATA
    saved = deepcopy(data)
    for form in (:caller, :caller_call)
        brmi = KernelWeightedObservations.build(data, form)
        backend, problem = consumer_problem(brmi)
        names = coordinate_names(backend.model.layout)
        @test sort(names) == [:delta, :s, :sigma]
        is, iσ, iδ = (findfirst(==(name), names) for name in (:s, :sigma, :delta))
        oracle(u) = begin
            s, sigma, delta = u[is], exp(u[iσ]), u[iδ]
            logpdf(Normal(), s) + logpdf(Exponential(), sigma) + u[iσ] +
                logpdf(Normal(0, 0.5), delta) +
                sum((weight * logpdf(Normal(s * x + delta, sigma), y)
                    for (xs, ys, weights) in zip(data.x, data.y, data.w)
                    for (x, y, weight) in zip(xs, ys, weights)); init=0.0)
        end
        stan = consumer_stan(brmi, "kernel-weighted-$form";
            mod=KernelWeightedObservations)
        for u in ([0.0, 0.0, 0.0], [0.3, -0.2, 0.15], [-0.4, 0.25, -0.3])
            check_consumer_point(problem, u, oracle)
            check_consumer_stan(problem, stan,
                [:s => "s", :sigma => "sigma", :delta => "delta"], backend, u)
        end
        emitted = BRM._rk_emit_ast(backend.plan)
        main = sprint(Base.show_unquoted, emitted.main)
        @test occursin("y .~ LogDensity.(y_scalar_logdensity_weighted, pred_weight_y,", main)
        @test isdisjoint(weight_reader_arguments(emitted.main), (:s, :sigma, :delta))
        @test any(definition -> occursin("weighted_density = weight * density",
            sprint(Base.show_unquoted, definition)), emitted.defs)
    end
    @test isequal(data, saved)
end

@stestset "in-cell bounded combinators keep their family token out of the data" begin
    data = (; x=[[1.0, 2.0], [0.5]], y=[[0.1, 0.2], [0.3]],
        lo=[[0.0, 0.1], [0.2]], hi=[[0.4, 0.5], [0.6]])
    brmi = KernelWeightedObservations.build(data, :bounded)
    # Capability gap, not a wrong shape: in-cell censoring combinators are
    # valid SLIC observations that the RK backend does not lower yet.
    @test_broken (RKBRMI(brmi); true)
end
