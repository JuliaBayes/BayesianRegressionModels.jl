# Shared acceptance for a complete printed model, its definitions, ordinary
# callable bindings and data. No translated-plan modifications are replayed.
import ReactiveKernelsPPL

function check_rk_source_roundtrip(backend)
    brm = BayesianRegressionModels
    ext = Base.get_extension(brm, :BayesianRegressionModelsReactiveKernelsExt)
    emitted = brm._rk_emit_ast(backend.plan)
    retyped = brm._RKEmittedProgram(
        [Meta.parse(sprint(Base.show_unquoted, definition)) for definition in emitted.defs],
        Meta.parse(sprint(Base.show_unquoted, emitted.main)), emitted.bindings)
    original = ext._rk_translated_plan(backend.plan)
    translated = ext._rk_translate_from_emitted(backend.plan, retyped)
    rebuilt = Base.invokelatest(ReactiveKernelsPPL.build_kernel, translated)
    @test ReactiveKernelsPPL.coordinate_names(rebuilt.layout) ==
        ReactiveKernelsPPL.coordinate_names(backend.model.layout)
    N = backend.model.layout.total
    probes = (zeros(N), fill(0.13, N), N <= 1 ? fill(-0.2, N) :
        collect(range(-0.2, 0.3; length=N)))
    for u in probes, preset in (:sampler, :prior, :likelihood)
        before = copy(u)
        a = Base.invokelatest(
            ReactiveKernelsPPL.prepare_query(backend.model, original, preset), u)
        b = Base.invokelatest(
            ReactiveKernelsPPL.prepare_query(rebuilt, translated, preset), u)
        @test isequal(a, b)
        @test isequal(u, before)
    end
    return backend
end
