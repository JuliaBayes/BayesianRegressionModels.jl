# Run each case in a fresh Julia process to measure its first emission:
#   julia --startup-file=no --project=test test/benchmark_emission_latency.jl regression [output-dir]
#   julia --startup-file=no --project=test test/benchmark_emission_latency.jl panel [output-dir]
# Package loading is reported separately. No wall-clock threshold is asserted:
# compiler time, allocation and unchanged source are the useful controls on a
# shared host. These independently authored examples contain no application data.
using Test

loaded = @timed begin
    using BayesianRegressionModels
    using Distributions: Normal, Exponential
end
const BRM = BayesianRegressionModels
Base.cumulative_compile_timing(true)

regression_data() = (;
    x=[-1.5, -0.4, 0.1, 0.8, 1.2, 1.8, 2.4, 3.1],
    y=[0.3, 0.7, 1.3, 1.8, 2.0, 2.2, 2.9, 3.4],
    site=["north", "north", "east", "east", "east", "west", "west", "west"])
regression_model(df) = @brm df begin
    scale ~ Exponential(1)
    mu ~ 1 + x + (1 + x | site)
    effect(mu, Intercept) ~ Normal(0, 2)
    effect(mu, x) ~ Normal(0, 1)
    y ~ Normal(mu, scale)
end

panel_data() = (;
    site=["north", "east", "west"], x=[-0.5, 0.25, 1.5],
    times=[[0.1, 0.7, 1.6], [0.2, 1.2], [0.0, 0.3, 0.9, 1.7]],
    readings=[[0.2, 0.9, 1.8], [0.1, 1.0], [0.0, 0.4, 1.2, 2.1]])
panel_model(df) = @brm df begin
    noise ~ Exponential(1)
    log(amplitude) ~ 1 + x + (1 + x | site)
    effect(amplitude, Intercept) ~ Normal(0, 1)
    fitted ~ kernel(times, readings, amplitude) do t, observed, a
        predicted = a .* t
        observed ~ normal(predicted, noise)
        predicted
    end
end

# Invoke dynamically so inference of the timed closure does not move compiler
# work into compilation of this measurement wrapper, before its counters start.
Base.@nospecializeinfer function measure(@nospecialize(f), label)
    GC.gc()
    before = Base.cumulative_compile_time_ns()
    stat = @timed f()
    after = Base.cumulative_compile_time_ns()
    println("PHASE ", label, " time=", stat.time, " bytes=", stat.bytes,
        " gc=", stat.gctime, " compile=", (after[1] - before[1]) / 1e9,
        " recompile=", (after[2] - before[2]) / 1e9)
    stat.value
end

function benchmark(builder, data, case_id, output)
    construct = () -> builder(data)
    model = measure(construct, "construct-cold")
    emit = () -> BRM.emit_rk_artifact(model; case_id)
    first = measure(emit, "emit-cold")
    measure(construct, "construct-warm")
    second = measure(emit, "emit-warm")
    @testset "cold and warm artifact source" begin
        @test first.ast == second.ast
        @test first.defs == second.defs
        @test first.plan.columns == second.plan.columns
    end
    if output !== nothing
        mkpath(output)
        BRM.write_rk_artifact(joinpath(output, "artifact.jls"), first)
        open(joinpath(output, "source.txt"), "w") do io
            foreach(def -> println(io, def), first.defs)
            println(io, first.ast)
        end
        # Separate checkouts record different source paths in LineNumberNodes.
        # Retain the full artifact/source above, and normalize only locations
        # on owned copies for a portable cross-revision source comparison.
        open(joinpath(output, "source-no-locations.txt"), "w") do io
            foreach(def -> println(io, Base.remove_linenums!(deepcopy(def))), first.defs)
            println(io, Base.remove_linenums!(deepcopy(first.ast)))
        end
    end
end

case = isempty(ARGS) ? "regression" : ARGS[1]
output = length(ARGS) >= 2 ? abspath(ARGS[2]) : nothing
println("LOAD time=", loaded.time, " bytes=", loaded.bytes, " gc=", loaded.gctime)
if case == "regression"
    benchmark(regression_model, regression_data(), "generic-regression", output)
elseif case == "panel"
    benchmark(panel_model, panel_data(), "generic-panel", output)
else
    error("unknown case `$case`; choose regression or panel")
end
