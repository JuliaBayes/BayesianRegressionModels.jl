#!/usr/bin/env julia

# Offline validation for the BRMDescriptor → semantic_app mount.
# Run from the repository root:
#   julia --project=web-macro web-macro/validate_descriptor_mount.jl
#
# No listener is opened and no Stan program is compiled: the tiny example
# descriptor transpiles (pure), execution/decline paths run in-process, the
# semantic app compiles offline, and real HTTP requests drive the registered
# routes in-process through HTMXObjects' own router.
#
# Governed execution promotes slow results to a disk cache rooted at the
# working directory, so run from a scratch dir (all paths below are absolute).

cd(mktempdir())

using Test
using DynamicObjects
using HTMXObjects

include(joinpath(@__DIR__, "src", "descriptor_mount.jl"))
using .DescriptorMount
using BayesianRegressionModels: brm_descriptor, brm_operation

module SmokeModel
# Whole-module import: the `@brm` expansion emits bare `@getproperty`
# (/src/macro.jl), so a narrow `using` leaves the expansion unresolvable.
using BayesianRegressionModels
using Distributions: Normal, Exponential

builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x
    y ~ Normal(mu, sigma)
end

df = (; x=[1.0, 2.0, 3.0, 4.0], y=[0.2, -0.1, 0.3, 0.0])

end # module SmokeModel

@testset "descriptor mount" begin
    d = brm_descriptor(SmokeModel.builder, SmokeModel.df;
                       mod=SmokeModel, name=:smoke)

    # The example offers the ordinary dense-model set (builder form, so
    # :replay; no random effects, so :reprocess) and nothing else.
    @test Set(op.name for op in d.operations) ==
        Set((:transpile, :instantiate, :fit, :predict, :pointwise_loglik,
             :replay, :reprocess))

    mount = BRMDescriptorMount(; descriptor=d)

    # The picker domain is exactly the derived set, in derivation order,
    # labelled with the derived titles.
    records = option_records(property_options(mount, :operation))
    @test records !== nothing
    @test [r.value for r in records] == [op.name for op in d.operations]
    @test [r.label for r in records] == [op.title for op in d.operations]

    # Suppression removes the entry; retitling relabels it. No parallel list.
    suppressed = brm_descriptor(SmokeModel.builder, SmokeModel.df;
                                mod=SmokeModel, name=:smoke,
                                operations=Dict(:predict => nothing))
    suppressed_records = option_records(property_options(
        BRMDescriptorMount(; descriptor=suppressed), :operation))
    @test :predict ∉ [r.value for r in suppressed_records]
    @test length(suppressed_records) == length(records) - 1

    retitled = brm_descriptor(SmokeModel.builder, SmokeModel.df;
                              mod=SmokeModel, name=:smoke,
                              titles=Dict(:fit => "Run the sampler"))
    retitled_records = option_records(property_options(
        BRMDescriptorMount(; descriptor=retitled), :operation))
    fit_record = only(r for r in retitled_records if r.value === :fit)
    @test fit_record.label == "Run the sampler"

    # Override additions appear under their own name.
    extended = brm_descriptor(SmokeModel.builder, SmokeModel.df;
                              mod=SmokeModel, name=:smoke,
                              operations=Dict(:summarise => (dd; kwargs...) -> "ok"))
    extended_records = option_records(property_options(
        BRMDescriptorMount(; descriptor=extended), :operation))
    @test :summarise ∈ [r.value for r in extended_records]

    # Needs-derivation per operation, pinned for this model.
    @test !mount_needs_draws(d, brm_operation(d, :transpile))
    @test !mount_needs_draws(d, brm_operation(d, :fit))
    @test mount_needs_draws(d, brm_operation(d, :predict))
    @test mount_needs_draws(d, brm_operation(d, :pointwise_loglik))
    @test !mount_needs_dataframe(brm_operation(d, :fit))
    @test mount_needs_dataframe(brm_operation(d, :replay))
    @test mount_needs_dataframe(brm_operation(d, :reprocess))

    # Execution: :transpile runs (pure, no compile); draws/dataframe ops
    # decline explicitly; unknown ops fail closed.
    stan = mount_execute_operation(d, :transpile)
    @test stan isa SemanticCode
    @test occursin("generated quantities", repr("text/markdown", stan))

    predict = mount_execute_operation(d, :predict)
    @test predict isa SemanticUnavailable
    @test occursin("draws", repr("text/markdown", predict))

    pointwise = mount_execute_operation(d, :pointwise_loglik)
    @test pointwise isa SemanticUnavailable
    @test occursin("draws", repr("text/markdown", pointwise))

    replay = mount_execute_operation(d, :replay)
    @test replay isa SemanticUnavailable
    @test occursin("dataframe", repr("text/markdown", replay))

    overridden = mount_execute_operation(extended, :summarise)
    @test overridden isa SemanticCode
    @test occursin("ok", repr("text/markdown", overridden))

    @test_throws ErrorException mount_execute_operation(d, :simulate)
    @test_throws ArgumentError mount_execute_operation(nothing, :transpile)
    @test_throws ArgumentError DescriptorMount._require_descriptor(nothing)

    # The compiled-problem summary is pure (literals, no Stan compile).
    summary = mount_problem_summary("Fit", d.id, 3, [:sigma, :mu])
    summary_md = repr("text/markdown", summary)
    @test occursin("3", summary_md)
    @test occursin("sigma", summary_md)
    @test occursin(d.id, summary_md)

    # Reflection nodes carry the descriptor's own facts.
    @test occursin("y ~", repr("text/plain", mount_formula_node(d)))
    schema_md = repr("text/markdown", mount_schema_node(d))
    @test occursin("stan_inputs", schema_md)
    @test occursin("x", schema_md)
    @test occursin("y", schema_md)
    inputs_md = repr("text/markdown", mount_inputs_node(d))
    @test occursin("observed", inputs_md)
    @test occursin("held_out", inputs_md)
    outputs_md = repr("text/markdown", mount_outputs_node(d))
    @test occursin("posterior_predictive", outputs_md)
    @test occursin("pointwise_loglik", outputs_md)
    ops_md = repr("text/markdown", mount_operations_node(d))
    for op in d.operations
        @test occursin(String(op.name), ops_md)
    end

    # A bare mount renders an empty picker and throws loudly on use.
    bare_records = option_records(property_options(BRMDescriptorMount(), :operation))
    @test bare_records !== nothing
    @test isempty(bare_records)

    # The semantic app compiles offline: the operation control and every
    # derived operation value are in the generated form.
    app_html = repr("text/html", semantic_app(mount; title="t", submit="Run"))
    @test occursin("name=\"operation\"", app_html)
    for op in d.operations
        @test occursin(String(op.name), app_html)
    end
    @test occursin("execute", app_html)

    # Real requests, in-process through the registered router (no sockets).
    # The page carries required descriptor state, so the default
    # fresh-per-request factory cannot build it: retain one source root
    # behind the key and remount it per request (htmxo-use §10.6 adapter
    # seam; `remount` + `OperationContext` docstrings).
    page = mount_descriptor_page(d)
    slot = Ref{Any}(nothing)
    provider = RootProvider(
        (T, ctx) -> begin
            if slot[] === nothing
                slot[] = T(d)
            end
            DynamicObjects.remount(slot[]; __req__=ctx.request, __prefix__=ctx.prefix)
        end;
        scope=:job, key=req -> "smoke",
    )
    route!(page; root_provider=provider)
    drive(path; headers=Pair{String,String}[]) = begin
        request = HTTP.Request("GET", path, headers)
        handler = first(HTTP.Handlers.gethandler(
            HTMXObjects.CONTEXT[].service.router, request))
        handler(request)
    end
    try
        browser_headers = [
            "Accept" => "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        ]
        index = drive("/"; headers=browser_headers)
        index_body = String(index.body)
        @test index.status == 200
        @test occursin("text/html", HTTP.header(index, "Content-Type", ""))
        @test occursin(
            "https://cdn.jsdelivr.net/npm/htmx.org@2.0.8/dist/htmx.min.js",
            index_body,
        )
        @test occursin("data-htmxo-operation-load", index_body)
        @test occursin("hx-trigger=\"load\"", index_body)

        htmx_headers = vcat(browser_headers, ["HX-Request" => "true"])
        full = drive("/"; headers=htmx_headers)
        @test full.status == 200
        @test occursin("name=\"operation\"", String(full.body))

        transpiled = drive("/mount/execute?operation=transpile"; headers=htmx_headers)
        transpiled_body = String(transpiled.body)
        @test transpiled.status == 200
        @test occursin("generated quantities", transpiled_body)

        declined = drive("/mount/execute?operation=predict"; headers=htmx_headers)
        declined_body = String(declined.body)
        @test declined.status == 200
        @test occursin("draws", declined_body)

        formula = drive("/mount/formula"; headers=htmx_headers)
        @test formula.status == 200
        @test occursin("y ~", String(formula.body))
    finally
        terminate()
    end
end

println("offline descriptor-mount validation: ok")
