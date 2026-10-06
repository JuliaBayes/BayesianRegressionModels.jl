# Exercise a batch of independently named, structurally identical public BRMs.
# Distinct operation names give each model a different `BRMI` carrier type while
# keeping the statistical work fixed, exposing compiler work that scales with a
# corpus rather than with model complexity.
#
#   julia --startup-file=no --project=. test/benchmark_emission_breadth.jl [count] [rk|sb|both]
#
# This is a measurement harness, not a wall-clock assertion. Compare cumulative
# compiler time and allocations across revisions on the same host and Julia.
using Test

loaded = @timed begin
    using BayesianRegressionModels
    using Distributions: Normal, Exponential
end
const BRM = BayesianRegressionModels
Base.cumulative_compile_timing(true)

const X = [-1.5, -0.4, 0.1, 0.8, 1.2, 1.8, 2.4, 3.1]
const Y = [0.3, 0.7, 1.3, 1.8, 2.0, 2.2, 2.9, 3.4]

function define_case(index)
    x = Symbol(:x_, index)
    y = Symbol(:y_, index)
    mu = Symbol(:mu_, index)
    builder = Symbol(:breadth_model_, index)
    Core.eval(@__MODULE__, quote
        function $builder(df)
            @brm df begin
                scale ~ Exponential(1)
                $mu ~ 1 + $x
                $y ~ Normal($mu, scale)
            end
        end
    end)
    data = NamedTuple{(x, y)}((X, Y))
    getfield(@__MODULE__, builder), data
end

# Keep the harness from moving work across the timer or specializing on the
# model whose specialization is under measurement.
Base.@nospecializeinfer function measure(@nospecialize(f), phase, index)
    GC.gc()
    before = Base.cumulative_compile_time_ns()
    stat = @timed f()
    after = Base.cumulative_compile_time_ns()
    println("PHASE ", phase, " index=", index,
        " time=", stat.time, " bytes=", stat.bytes, " gc=", stat.gctime,
        " compile=", (after[1] - before[1]) / 1e9,
        " recompile=", (after[2] - before[2]) / 1e9)
    stat.value
end

Base.@nospecializeinfer build_model(@nospecialize(builder), data) = builder(data)
Base.@nospecializeinfer emit_rk(@nospecialize(model), index) =
    BRM.emit_rk_artifact(model; case_id="public-breadth-$index")
Base.@nospecializeinfer emit_sb(@nospecialize(model)) =
    SBBRMI(model; mod=@__MODULE__, total_groups=())

count = isempty(ARGS) ? 12 : parse(Int, ARGS[1])
mode = length(ARGS) < 2 ? "both" : ARGS[2]
count >= 2 || error("count must be at least 2")
mode in ("rk", "sb", "both") || error("mode must be rk, sb, or both")

cases = [define_case(index) for index in 1:count]
models = Any[]
for (index, (builder, data)) in pairs(cases)
    push!(models, measure(() -> build_model(builder, data), "construct", index))
end

println("LOAD time=", loaded.time, " bytes=", loaded.bytes, " gc=", loaded.gctime)
if mode in ("rk", "both")
    artifacts = Any[]
    for (index, model) in pairs(models)
        push!(artifacts, measure(() -> emit_rk(model, index), "rk", index))
    end
    repeated = measure(() -> emit_rk(first(models), 1), "rk-repeat", 1)
    @test repeated.ast == first(artifacts).ast
    @test repeated.defs == first(artifacts).defs
end
if mode in ("sb", "both")
    emitted = Any[]
    for (index, model) in pairs(models)
        push!(emitted, measure(() -> emit_sb(model), "sb", index))
    end
    repeated = measure(() -> emit_sb(first(models)), "sb-repeat", 1)
    @test BRM.stan_code(repeated) == BRM.stan_code(first(emitted))
end
