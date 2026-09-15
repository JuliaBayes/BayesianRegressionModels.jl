module BRMDocsComparisons

using Markdown
using BayesianRegressionModels
using Distributions
using StanBlocks
using Turing

const BRM = BayesianRegressionModels
const EXAMPLE_MODULES = Dict{Symbol,Module}()
const PANE_IDS = "brm|stanblocks|stan|turing"
const PANE_LABELS = "BRM authoring|StanBlocks model|Stan source|Turing model"
const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, ".."))

"""Return one stable evaluation module for all executable blocks on a page."""
example_module(name::Symbol) = get!(EXAMPLE_MODULES, name) do
    Module(name)
end

function evaluate_source(mod::Module, displayed::AbstractString)
    parsed = Meta.parseall(displayed; filename="brm-docs-example.jl")
    expressions = parsed.head === :toplevel ? parsed.args : Any[parsed]
    value = nothing
    for expression in expressions
        expression isa LineNumberNode && continue
        value = Core.eval(mod, expression)
    end
    return value
end

function definition_name(expression)
    expression isa Expr || return nothing
    if expression.head === :function
        signature = expression.args[1]
        signature isa Expr && signature.head === :call || return nothing
        name = first(signature.args)
        return name isa Symbol ? name : nothing
    end
    for argument in expression.args
        name = definition_name(argument)
        isnothing(name) || return name
    end
    return nothing
end

function repository_source(relative_path::AbstractString)
    path = normpath(joinpath(REPOSITORY_ROOT, relative_path))
    relative = relpath(path, REPOSITORY_ROOT)
    (!isempty(splitpath(relative)) && first(splitpath(relative)) == "..") &&
        error("repository source path escapes the repository root: $relative_path")
    return read(path, String)
end

"""Return one function declaration exactly as written in a repository source file."""
function source_function(relative_path::AbstractString, name::Symbol)
    source = repository_source(relative_path)
    marker = "function $name"
    matches = findall(marker, source)
    length(matches) == 1 || error(
        "expected one `$marker` declaration in $relative_path, found $(length(matches))")
    start = first(only(matches))
    expression, next_position = Meta.parse(source, start; greedy=true, raise=true)
    definition_name(expression) === name || error(
        "source at `$marker` in $relative_path did not parse as that function")
    stop = prevind(source, next_position)
    return strip(SubString(source, start, stop), '\n')
end

"""Return an exact, marker-bounded source region from a repository file."""
function source_region(relative_path::AbstractString;
                       starting_at::AbstractString,
                       ending_before::AbstractString)
    source = repository_source(relative_path)
    starts = findall(starting_at, source)
    length(starts) == 1 || error(
        "expected one `$starting_at` marker in $relative_path, found $(length(starts))")
    start = first(only(starts))

    stops = findall(ending_before, source)
    stops = filter(r -> first(r) > start, stops)
    length(stops) == 1 || error(
        "expected one `$ending_before` marker after `$starting_at` in " *
        "$relative_path, found $(length(stops))")
    stop = prevind(source, first(only(stops)))
    return strip(SubString(source, start, stop), '\n')
end

"""Render a drift-proof repository source region as a documentation code block."""
source_code_region(relative_path::AbstractString; language="julia", kwargs...) =
    Markdown.MD([Markdown.Code(language, source_region(relative_path; kwargs...))])

"""Evaluate a source file's shared prelude, stopping before a named function."""
function evaluate_source_prelude(mod::Module, relative_path::AbstractString;
                                 before::Symbol,
                                 starting_at::Union{Nothing,AbstractString}=nothing)
    source = repository_source(relative_path)
    if !isnothing(starting_at)
        matches = findall(starting_at, source)
        length(matches) == 1 || error(
            "expected one `$starting_at` marker in $relative_path, " *
            "found $(length(matches))")
        source = SubString(source, first(only(matches)), lastindex(source))
    end
    Core.eval(mod, :(using BayesianRegressionModels, Distributions, StanBlocks))
    parsed = Meta.parseall(source;
                           filename=relative_path)
    expressions = parsed.head === :toplevel ? parsed.args : Any[parsed]
    for expression in expressions
        expression isa LineNumberNode && continue
        definition_name(expression) === before && return nothing
        Core.eval(mod, expression)
    end
    error("$relative_path does not define `$before`; shared prelude was not bounded")
end

function exception_text(backend::AbstractString, err)
    reason = sprint(showerror, err)
    "$backend unsupported for this BRM example\n\n$reason"
end

function stan_emissions(brmi::BRM.BRMI, mod::Module;
                        required::Bool=false, total_groups=:auto)
    try
        sb = Base.invokelatest(BRM.SBBRMI, brmi; mod=mod, total_groups)
        sb_source = strip(sprint(show, sb), '\n')
        stan_source = strip(Base.invokelatest(BRM.stan_code, sb), '\n')
        return sb_source, stan_source
    catch err
        required && rethrow()
        reason = exception_text("StanBlocks", err)
        return reason, "Stan emission unavailable because StanBlocks lowering failed.\n\n" *
                       sprint(showerror, err)
    end
end

function turing_emission(brmi::BRM.BRMI)
    try
        backend = Base.invokelatest(BRM.TuringBRMI, brmi)
        source = Base.invokelatest(BRM.turing_model_source, backend)
        isnothing(source) && error("Turing lowering did not retain its generated source")
        programs = source isa Tuple ? source : (source,)
        return join((sprint(io -> Base.show_unquoted(io,
                     Base.remove_linenums!(deepcopy(program))))
                     for program in programs), "\n\n")
    catch err
        return exception_text("Turing", err)
    end
end

"""
Evaluate one displayed BRM example and emit the fixed four-pane comparison.
The StanBlocks, Stan, and Turing panes are always derived during this build.
"""
function comparison(mod::Module, code::AbstractString, brmi_name::Symbol;
                    title=replace(string(brmi_name), '_' => ' '),
                    require_stan::Bool=false, total_groups=:auto)
    displayed = strip(code, '\n')
    Core.eval(mod, :(using BayesianRegressionModels, Distributions))
    evaluate_source(mod, displayed)
    candidate = Core.eval(mod, brmi_name)
    brmi = candidate isa Function ? Base.invokelatest(candidate) : candidate
    brmi isa BRM.BRMI || error(
        "docs comparison `$brmi_name` did not evaluate to or construct a BRMI")
    sb_source, stan_source = stan_emissions(brmi, mod; required=require_stan, total_groups)
    turing_source = turing_emission(brmi)
    return Markdown.MD([
        Markdown.Code("brm-comparison", string(title)),
        Markdown.Code("julia", displayed),
        Markdown.Code("julia", sb_source),
        Markdown.Code("stan", stan_source),
        Markdown.Code("julia", turing_source),
    ])
end

"""
Render a StanBlocks-backend (`@slic`) model that is NOT expressed through the `@brm`
formula surface: its exact model source (read from the repository, drift-proof) plus
the generated Stan it emits. There is no `@brm` authoring pane or Turing pane because
such a model does not go through the formula DSL — this is the honest two-pane view
for a model that lives at BRM's backend layer. Errors if the emission fails.
"""
function slic_model_panes(mod::Module, code::AbstractString, model_name::Symbol;
                          title=replace(string(model_name), '_' => ' '))
    displayed = strip(code, '\n')
    Core.eval(mod, :(using BayesianRegressionModels, Distributions, StanBlocks))
    evaluate_source(mod, displayed)
    candidate = Core.eval(mod, model_name)
    model = candidate isa Function ? Base.invokelatest(candidate) : candidate
    stan_source = strip(Base.invokelatest(StanBlocks.stan_code, model), '\n')
    return Markdown.MD([
        Markdown.Paragraph([Markdown.Bold(string(title))]),
        Markdown.Code("julia", displayed),
        Markdown.Code("stan", stan_source),
    ])
end

"""
Render a StanBlocks `compile_slic_bundle` model (a multi-part `@slic`/`@deffun`
workspace that is not a single `@brm` or `@slic` function): `include` its source file,
show the parent-body source held in `body_name`, then the generated Stan from
`builder_name()` (a zero-arg builder returning the bundle result with a `.code`
field). Two panes — there is no `@brm` authoring pane or Turing pane. Errors closed.
"""
function bundle_model_panes(mod::Module, include_relpath::AbstractString,
                            body_name::Symbol, builder_name::Symbol; title)
    Core.eval(mod, :(using BayesianRegressionModels, Distributions, StanBlocks))
    path = normpath(joinpath(REPOSITORY_ROOT, include_relpath))
    Base.invokelatest(Base.include, mod, path)
    body = strip(Core.eval(mod, body_name), '\n')
    result = Base.invokelatest(Core.eval(mod, builder_name))
    stan_source = strip(result.code, '\n')
    return Markdown.MD([
        Markdown.Paragraph([Markdown.Bold(string(title))]),
        Markdown.Code("julia", body),
        Markdown.Code("stan", stan_source),
    ])
end

"""Assert that a generated page contains complete required Stan comparisons."""
function validate_required_stan_outputs(path::AbstractString,
                                        brmi_names::Tuple)
    source = read(path, String)
    expected = length(brmi_names)
    comparisons = length(collect(eachmatch(
        r"(?m)^```brm-comparison\s*$", source)))
    comparisons == expected || error(
        "$(basename(path)) generated $comparisons required comparisons; " *
        "expected $expected")

    stan_fences = length(collect(eachmatch(r"(?m)^```stan\s*$", source)))
    stan_fences == expected || error(
        "$(basename(path)) generated $stan_fences Stan panes; expected $expected")

    for name in brmi_names
        occursin("function $name", source) || error(
            "$(basename(path)) omitted the exact `$name` declaration")
    end

    for fallback in ("StanBlocks unsupported for this BRM example",
                     "Stan emission unavailable because StanBlocks lowering failed")
        occursin(fallback, source) && error(
            "$(basename(path)) contains forbidden required-backend fallback: " *
            fallback)
    end
    return nothing
end

function validate_generated_templates(paths)
    for path in paths
        source = read(path, String)
        generated = length(collect(eachmatch(
            r"Main\.BRMDocsComparisons\.comparison\(", source)))
        generated > 0 || error(
            "$(basename(path)) contains no build-generated comparison")
        occursin("data-backend-comparison", source) && error(
            "$(basename(path)) hand-authors the comparison wrapper; " *
            "BRMDocsComparisons.comparison owns the fixed four panes")
        occursin(r"(?m)^```stan\s*$", source) && error(
            "$(basename(path)) contains hand-authored Stan; generated examples " *
            "must call BRMDocsComparisons.comparison")
    end
    return nothing
end

function validate_no_bypasses(paths)
    executable_fences = Set((
        "julia", "jldoctest", "@eval", "@example", "@repl", "@setup"))
    for path in paths
        source = read(path, String)
        occursin("BRMDocsComparisons.source_excerpt", source) && error(
            "$path bypasses the build-generated four-pane comparison")
        for block in eachmatch(r"(?ms)^```([^\n]*)\n(.*?)^```\s*$", source)
            info, code = block.captures
            fence_parts = split(strip(info))
            language = isempty(fence_parts) ? "" : first(fence_parts)
            language in executable_fences || continue
            occursin("@brm", code) || continue
            occursin("Main.BRMDocsComparisons.comparison(", code) && continue
            error("$path contains a standalone executable `@brm` example " *
                  "in a `$language` fence; use the generated four-pane " *
                  "comparison")
        end
        if endswith(path, "turing-backend.md")
            occursin("Main.BRMDocsComparisons.comparison", source) && error(
                "backend-specific pages must be overview-only; move examples " *
                "to a backend-neutral page")
        end
    end
    return nothing
end

end
