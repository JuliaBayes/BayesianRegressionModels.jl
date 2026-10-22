# test/rk_precompile_workload.jl — RK translation in downstream environments:
# while a downstream package image is generated (snag
# `rkbrmi-inside-a-140a7cf4`), and for an application or package that reaches
# BRM, ReactiveKernels and ReactiveKernelsPPL only through a dependency (snag
# `rkbrmi-resolves-4b269923`).
#
# While Julia generates a package image it refuses `eval` into every module
# but the package being precompiled and the modules created during its
# precompilation. RK translation evaluated a program's definitions into an
# anonymous `Module()` and resolved a cell's callee and family names with
# `Core.eval(scope, …)`, so `RKBRMI(brmi)` failed in any downstream precompile
# workload. Three throwaway packages are built in a temporary environment
# layered on this one: `RKImageModels` defines public synthetic models;
# `RKImageConsumer` depends on it and translates them while its own image is
# generated, up to `build_kernel`; `RKImageWorkload` builds a whole `RKBRMI`
# there, which still stops in ReactiveKernelsPPL's `build_kernel`
# (ReactiveKernels snag `rkppl-build-kern-4607ff90`).
#
# Run: julia --startup-file=no --project=test test/rk_precompile_workload.jl

using Test, BayesianRegressionModels, Pkg, TOML
const BRM = BayesianRegressionModels

const MODELS_SOURCE = """
module RKImageModels
using BayesianRegressionModels, Distributions, ReactiveKernels
import BayesianRegressionModels: _rk_callable_source!

const PK = (; subject=["s1", "s2", "s3", "s4"], weight=[60.0, 72.0, 81.0, 55.0],
    dose=[100.0, 120.0, 80.0, 100.0],
    t=[[0.5, 1.0, 2.0], [0.25, 1.5], [0.5, 1.0, 2.0, 4.0], [1.0]],
    y=[[1.0, 0.8, 0.5], [1.2, 0.7], [0.9, 0.7, 0.4, 0.2], [0.6]])

# A provider claims the curve as an RK graph: a `@kernel` definition.
function dosed_curve(t, dose, log_CL)
    k = exp(log_CL)
    [dose * exp(-k * ti) for ti in t]
end
function _rk_callable_source!(definitions, bindings, entry, ::typeof(dosed_curve))
    push!(definitions, :(ReactiveKernels.@kernel \$entry(t, dose, log_CL) = begin
        k = exp(log_CL)
        reads = ReactiveKernels.plate(eachindex(t)) do j
            dose * exp(-k * t[j])
        end
        return reads
    end))
    :done
end
# A provider claims the curve as an ordinary Julia function.
native_curve(t, dose, log_CL) = dose .* exp.(-exp(log_CL) .* t)
function _rk_callable_source!(definitions, bindings, entry, ::typeof(native_curve))
    push!(definitions, :(function \$entry(t, dose, log_CL)
        return dose .* exp.(-exp(log_CL) .* t)
    end))
    :done
end

# The cells call this module's functions and observe through `Normal`, a
# name this module binds: the planner reads both from here.
provided(d) = @brm d begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + zscale(weight) + (1 | p | subject)
    effect(log_CL, Intercept) ~ Normal(0, 1)
    sd(:, p) ~ Exponential(1)
    @plate for i in eachindex(log_CL)
        conc[i] = dosed_curve(t[i], dose[i], log_CL[i])
        y[i] ~ Normal(conc[i], sigma)
    end
end
native(d) = @brm d begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + zscale(weight) + (1 | p | subject)
    effect(log_CL, Intercept) ~ Normal(0, 1)
    sd(:, p) ~ Exponential(1)
    @plate for i in eachindex(log_CL)
        conc[i] = native_curve(t[i], dose[i], log_CL[i])
        y[i] ~ Normal(conc[i], sigma)
    end
end
end
"""

# Planned and translated as `RKBRMI` does, up to `build_kernel`.
const CONSUMER_SOURCE = """
module RKImageConsumer
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
import RKImageModels
const BRM = BayesianRegressionModels
const EXT = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
function translated(model)
    brmi = BRM._brm_resolve_optional_priors(model(RKImageModels.PK))
    plan = BRM._brm_rk_plan(brmi; held_out=())
    (; brmi, plan, program=EXT._RKProgram(BRM._rk_emit_ast(plan)))
end
const TRANSLATED = (provided=translated(RKImageModels.provided),
    native=translated(RKImageModels.native))
const WAS_PRECOMPILED = ccall(:jl_generating_output, Cint, ()) != 0
# A whole backend built at run time, for a caller that need not load any of
# the packages it uses.
fit(name::Symbol) = RKBRMI(getfield(RKImageModels, name)(RKImageModels.PK))
# A backend's image program, bound and built at run time.
function bound_backend(t)
    bound = EXT._rk_bind_program(t.program, t.plan)
    BRM.RKBRMI(t.brmi, t.plan, build_kernel(bound);
        program=BRM._RKOnce(t.program), bound=BRM._RKOnce(bound), source=t.brmi)
end
function concentrations(rk)
    bound = rk_bound_data(rk)
    query = ReactiveKernels.prepare(rk.model.spec;
        have=(:unconstrained, keys(bound)...), want=(:conc,), bound)
    u = collect(range(-0.6, 0.8; length=rk.model.layout.total))
    Base.invokelatest(query, u)
end
end
"""

# The emission module once required BRM, ReactiveKernels and
# ReactiveKernelsPPL by name, which resolves against the active project, or
# against the package whose image is generated, rather than against what is
# loaded. `RKImageIndirect` declares none of the three and translates while its
# image is generated; an application environment that declares only
# `RKImageConsumer` builds a whole `RKBRMI` through it at run time.
const INDIRECT_SOURCE = """
module RKImageIndirect
import RKImageConsumer, RKImageModels
const TRANSLATED = RKImageConsumer.translated(RKImageModels.native)
end
"""

const WORKLOAD_SOURCE = """
module RKImageWorkload
using BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
import RKImageModels
const RK = RKBRMI(RKImageModels.provided(RKImageModels.PK))
end
"""

function write_package(root, name, uuid, deps, source)
    dir = joinpath(root, name)
    mkpath(joinpath(dir, "src"))
    write(joinpath(dir, "Project.toml"), sprint(TOML.print, Dict(
        "name" => name, "uuid" => uuid, "version" => "0.1.0", "deps" => deps)))
    write(joinpath(dir, "src", name * ".jl"), source)
    dir
end

# An environment's project and manifest, with every developed path made
# absolute, so the copy resolves exactly what that one does.
function copy_environment(env, project=Base.active_project())
    source = dirname(project)
    cp(project, joinpath(env, "Project.toml"))
    manifest = TOML.parsefile(joinpath(source, "Manifest.toml"))
    for entries in values(manifest["deps"]), entry in entries
        haskey(entry, "path") && (entry["path"] = abspath(source, entry["path"]))
    end
    open(io -> TOML.print(io, manifest), joinpath(env, "Manifest.toml"), "w")
end

function run_julia(env, code; vars...)
    cmd = addenv(`$(Base.julia_cmd()) --startup-file=no --project=$env -e $code`,
        "JULIA_PKG_PRECOMPILE_AUTO" => "0", (string(k) => v for (k, v) in vars)...)
    out = IOBuffer()
    process = run(pipeline(ignorestatus(cmd); stdout=out, stderr=out))
    process.exitcode == 0 && process.termsignal == 0, String(take!(out))
end

const UUIDS = Dict("RKImageModels" => "5a1c7e2b-6d3f-4b8a-9c41-2e7f0d6b3a11",
    "RKImageConsumer" => "5a1c7e2b-6d3f-4b8a-9c41-2e7f0d6b3a12",
    "RKImageWorkload" => "5a1c7e2b-6d3f-4b8a-9c41-2e7f0d6b3a13",
    "RKImageIndirect" => "5a1c7e2b-6d3f-4b8a-9c41-2e7f0d6b3a14")
dep(name) = name => string(Base.identify_package(name).uuid)

@testset "a cell's names are read without eval" begin
    scope = Module(:CellScope)
    @test BRM._rk_kernel_binding(scope, :sum) === sum
    @test BRM._rk_kernel_binding(scope, GlobalRef(Base, :sum)) === sum
    @test BRM._rk_kernel_binding(scope, :(Base.Math.sin)) === sin
    # An unbound SLIC family token is StanBlocks'.
    @test BRM._rk_kernel_binding(scope, :normal) === BRM.StanBlocks.stan.normal
    @test_throws "names nothing in module" BRM._rk_kernel_binding(scope, :no_such_name)
end

mktempdir() do root
    env = mkpath(joinpath(root, "env"))
    copy_environment(env)
    common = Dict(dep.(["BayesianRegressionModels", "ReactiveKernels",
        "ReactiveKernelsPPL"]))
    models = write_package(root, "RKImageModels", UUIDS["RKImageModels"],
        merge(common, Dict(dep("Distributions"))), MODELS_SOURCE)
    imports = merge(common, Dict("RKImageModels" => UUIDS["RKImageModels"]))
    consumer = write_package(root, "RKImageConsumer", UUIDS["RKImageConsumer"],
        imports, CONSUMER_SOURCE)
    workload = write_package(root, "RKImageWorkload", UUIDS["RKImageWorkload"],
        imports, WORKLOAD_SOURCE)
    indirect = write_package(root, "RKImageIndirect", UUIDS["RKImageIndirect"],
        Dict(name => UUIDS[name] for name in ("RKImageModels", "RKImageConsumer")),
        INDIRECT_SOURCE)
    ok, log = run_julia(env, """
        using Pkg
        Pkg.develop([PackageSpec(path=p) for p in
            $(repr([models, consumer, workload, indirect]))])
        """)
    ok || println(log)
    @test ok
    # What the consumer's backend computes where the environment declares
    # every package it uses.
    declared = nothing

    @testset "an RK program translated while an image is generated" begin
        for package in ("RKImageConsumer", "RKImageIndirect")
            ok, log = run_julia(env, "using Pkg; Pkg.precompile(\"$package\")")
            ok || println(log)
            @test ok
        end
        ok, log = run_julia(env, """
            using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
            using RKImageModels, RKImageConsumer, RKImageIndirect
            const BRM = BayesianRegressionModels
            const C = RKImageConsumer
            const EXT = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
            @test C.WAS_PRECOMPILED
            for (owner, name, t) in ((C, :provided, C.TRANSLATED.provided),
                    (C, :native, C.TRANSLATED.native),
                    (RKImageIndirect, :native, RKImageIndirect.TRANSLATED))
                # The definitions live in a module of the image that
                # translated them ...
                @test parentmodule(t.program.mod) === owner
                # ... and run after loading: the image's program binds and
                # computes what a run-time translation does.
                fresh = C.fit(name)
                # At run time a translation's module stays anonymous.
                @test parentmodule(EXT._rk_program(fresh).mod) === Main
                @test C.concentrations(C.bound_backend(t)) == C.concentrations(fresh)
            end
            println("CONCENTRATIONS=", repr(C.concentrations(C.fit(:provided))))
            """)
        ok || println(log)
        @test ok
        declared = match(r"CONCENTRATIONS=(.*)", log)
    end

    @testset "a library's RKBRMI built for an application that declares only the library" begin
        # The application reaches BRM, ReactiveKernels and ReactiveKernelsPPL
        # only through the consumer, and its load path has no other
        # environment that could declare them.
        app = mkpath(joinpath(root, "app"))
        copy_environment(app, joinpath(env, "Project.toml"))
        ok, log = run_julia(app, """
            using Pkg
            Pkg.rm(["BayesianRegressionModels", "ReactiveKernels", "ReactiveKernelsPPL"])
            """)
        ok || println(log)
        @test ok
        ok, log = run_julia(app, """
            using Test, RKImageConsumer
            @test all(isnothing ∘ Base.identify_package, ("BayesianRegressionModels",
                "ReactiveKernels", "ReactiveKernelsPPL"))
            println("CONCENTRATIONS=", repr(RKImageConsumer.concentrations(
                RKImageConsumer.fit(:provided))))
            """; JULIA_LOAD_PATH="@:@stdlib")
        ok || println(log)
        @test ok
        built = match(r"CONCENTRATIONS=(.*)", log)
        @test built !== nothing && declared !== nothing &&
            only(built.captures) == only(declared.captures)
    end

    @testset "a whole RKBRMI built while an image is generated" begin
        ok, log = run_julia(env, "using Pkg; Pkg.precompile(\"RKImageWorkload\")")
        # ReactiveKernels snag `rkppl-build-kern-4607ff90`: RKPPL's
        # `build_kernel` evaluates into its own `PPLGeneratedModels`.
        @test_broken ok
        ok || @test occursin("closed module `PPLGeneratedModels`", log)
    end
end
