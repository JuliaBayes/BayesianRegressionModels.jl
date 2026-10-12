include(joinpath(@__DIR__, "rk_consumer_support.jl"))

# One caller-owned observation family with two Stan overloads: a common lower
# limit, and per-row intervals. RK asks the scalar-law provider once per
# observation with the argument count that observation writes, so one BRMI
# observes both overloads under the original constructor name.
module OverloadedObservationFamily
using BayesianRegressionModels, StanBlocks, Distributions
import ReactiveKernels, SpecialFunctions
import BayesianRegressionModels: _rk_observation_source!

StanBlocks.@deffun begin
    limit_normal_lpdfs(y::vector[n], location::vector[n], scale::real,
            limit::real)::vector[n] = begin
        result::vector[n]
        for i in 1:n
            result[i] = y[i] <= limit ?
                normal_lcdf_stable(limit, location[i], scale) :
                normal_lpdf(y[i], location[i], scale)
        end
        return result
    end
    @lhs @lpxf limit_normal_lpdf(y::vector[n], location::vector[n], scale::real,
        limit::real)::real = sum(limit_normal_lpdfs(y, location, scale, limit))
    limit_normal_rng(vector[n], location::vector[n], scale::real,
            limit::real)::vector[n] = begin
        result::vector[n]
        for i in 1:n
            result[i] = normal_rng(location[i], scale)
        end
        return result
    end

    limit_normal_lpdfs(y::vector[n], location::vector[n], scale::real,
            hi::vector[n], lo::vector[n])::vector[n] = begin
        result::vector[n]
        for i in 1:n
            if lo[i] < hi[i]
                result[i] = log_diff_exp(normal_lcdf_stable(hi[i], location[i], scale),
                    normal_lcdf_stable(lo[i], location[i], scale))
            else
                result[i] = normal_lpdf(y[i], location[i], scale)
            end
        end
        return result
    end
    @lhs limit_normal_lpdf(y::vector[n], location::vector[n], scale::real,
            hi::vector[n], lo::vector[n])::real =
        sum(limit_normal_lpdfs(y, location, scale, hi, lo))
    limit_normal_rng(vector[n], location::vector[n], scale::real,
            hi::vector[n], lo::vector[n])::vector[n] = begin
        result::vector[n]
        for i in 1:n
            result[i] = normal_rng(location[i], scale)
        end
        return result
    end
end

# `limit_normal(location, scale, limit)`: at or below the limit, the Normal
# lower tail at the limit; above it, the Normal density.
function _rk_observation_source!(definitions, bindings, entry,
        ::typeof(limit_normal), ::Val{3})
    logerfc = Symbol(entry, :_logerfc)
    push!(bindings, logerfc => SpecialFunctions.logerfc)
    push!(definitions, :(ReactiveKernels.@kernel $entry(value, location, scale, limit) = begin
        limit_logdensity = if value <= limit
            $logerfc(-(limit - location) / (scale * sqrt(2.0))) - log(2.0)
        else
            -0.5 * log(2 * pi) - log(scale) -
                0.5 * ((value - location) / scale) * ((value - location) / scale)
        end
        return limit_logdensity
    end))
    :done
end

# `limit_normal(location, scale, hi, lo)`: a row with `lo < hi`
# is the Normal mass between them; any other row is observed exactly.
function _rk_observation_source!(definitions, bindings, entry,
        ::typeof(limit_normal), ::Val{4})
    logerfc = Symbol(entry, :_logerfc)
    push!(bindings, logerfc => SpecialFunctions.logerfc)
    push!(definitions, :(ReactiveKernels.@kernel $entry(value, location, scale, hi, lo) = begin
        interval_logdensity = if lo < hi
            $logerfc(-(hi - location) / (scale * sqrt(2.0))) - log(2.0) +
                log1p(-exp($logerfc(-(lo - location) / (scale * sqrt(2.0))) -
                    $logerfc(-(hi - location) / (scale * sqrt(2.0)))))
        else
            -0.5 * log(2 * pi) - log(scale) -
                0.5 * ((value - location) / scale) * ((value - location) / scale)
        end
        return interval_logdensity
    end))
    :done
end

function build(data, form)
    if form === :plate
        @brm data begin
            s ~ Normal(0, 1)
            sigma ~ Exponential(1)
            @plate for i in eachindex(x)
                m = s .* x[i]
                ya[i] ~ limit_normal(m, sigma, limit)
                yb[i] ~ limit_normal(m, sigma, hi[i], lo[i])
                pred[i] = x[i]
            end
        end
    else
        @brm data begin
            s ~ Normal(0, 1)
            sigma ~ Exponential(1)
            m = s * x
            ya ~ limit_normal(m, sigma, limit)
            yb ~ limit_normal(m, sigma, hi, lo)
        end
    end
end

# A provider whose entry ports disagree with the observation's arguments.
interval_only_normal(location, scale, hi, lo) =
    error("observed through its RK scalar law only")
function _rk_observation_source!(definitions, bindings, entry,
        ::typeof(interval_only_normal))
    push!(definitions, :(ReactiveKernels.@kernel $entry(value, location, scale, hi, lo) = begin
        residual = (value - location) / scale
        interval_logdensity = -0.5 * log(2 * pi) - log(scale) - 0.5 * residual * residual
        return interval_logdensity
    end))
    :done
end

function build_mismatch(data)
    @brm data begin
        s ~ Normal(0, 1)
        sigma ~ Exponential(1)
        m = s * x
        ya ~ interval_only_normal(m, sigma, limit)
    end
end
end

const OVERLOAD_GROUPED = (; x=[[0.2, 0.5], Float64[], [0.7, -0.3, 1.1]],
    ya=[[-1.0, 0.4], Float64[], [0.3, -1.0, 0.9]],
    yb=[[0.1, 0.5], Float64[], [0.2, -0.6, 1.0]],
    hi=[[0.3, 0.5], Float64[], [0.2, -0.4, 1.0]],
    lo=[[-0.5, 0.5], Float64[], [0.2, -1.5, 1.0]],
    limit=-1.0)
const OVERLOAD_FLAT = (; x=reduce(vcat, OVERLOAD_GROUPED.x),
    ya=reduce(vcat, OVERLOAD_GROUPED.ya), yb=reduce(vcat, OVERLOAD_GROUPED.yb),
    hi=reduce(vcat, OVERLOAD_GROUPED.hi),
    lo=reduce(vcat, OVERLOAD_GROUPED.lo), limit=OVERLOAD_GROUPED.limit)

@stestset "an overloaded caller-owned family observes each argument count through its own law" begin
    for (form, data) in ((:plate, OVERLOAD_GROUPED), (:flat, OVERLOAD_FLAT))
        saved = deepcopy(data)
        brmi = OverloadedObservationFamily.build(data, form)
        backend, problem = consumer_problem(brmi)
        names = coordinate_names(backend.model.layout)
        @test sort(names) == [:s, :sigma]
        is, iσ = findfirst(==(:s), names), findfirst(==(:sigma), names)
        rows = OVERLOAD_FLAT
        oracle(u) = begin
            s, sigma = u[is], exp(u[iσ])
            limited = sum(zip(rows.x, rows.ya)) do (x, y)
                d = Normal(s * x, sigma)
                y <= rows.limit ? logcdf(d, rows.limit) : logpdf(d, y)
            end
            interval = sum(zip(rows.x, rows.yb, rows.hi, rows.lo)) do (x, y, hi, lo)
                d = Normal(s * x, sigma)
                lo < hi ? logdiffcdf(d, hi, lo) : logpdf(d, y)
            end
            logpdf(Normal(), s) + logpdf(Exponential(), sigma) + u[iσ] +
                limited + interval
        end
        stan = consumer_stan(brmi, "overloaded-observation-$form";
            mod=OverloadedObservationFamily)
        for u in ([0.0, 0.0], [0.3, -0.2], [-0.4, 0.25])
            check_consumer_point(problem, u, oracle)
            check_consumer_stan(problem, stan, [:s => "s", :sigma => "sigma"], backend, u)
        end
        emitted = BRM._rk_emit_ast(backend.plan)
        main = sprint(Base.show_unquoted, emitted.main)
        @test occursin("LogDensity.(ya_scalar_logdensity,", main)
        @test occursin("LogDensity.(yb_scalar_logdensity,", main)
        ports = Dict(BRM._rk_source_definition(d).name =>
                length(BRM._rk_kernel_entry_ports(d))
            for d in emitted.defs if BRM._rk_source_definition(d).kind === :kernel)
        @test ports[:ya_scalar_logdensity] == 4
        @test ports[:yb_scalar_logdensity] == 5
        @test isequal(data, saved)
    end
end

@stestset "a scalar-law entry must take the observed argument count" begin
    data = (; x=OVERLOAD_FLAT.x, ya=OVERLOAD_FLAT.ya, limit=OVERLOAD_FLAT.limit)
    brmi = OverloadedObservationFamily.build_mismatch(data)
    # refused: an entry with other ports than the observation's value plus its
    # constructor arguments would observe a different law than the BRMI wrote.
    err = try
        RKBRMI(brmi)
        nothing
    catch caught
        caught
    end
    @test err isa ErrorException
    @test occursin("writes 3 constructor arguments", sprint(showerror, err))
    @test occursin("::Val{3}", sprint(showerror, err))
end
