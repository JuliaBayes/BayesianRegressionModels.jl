# Check generated docs pages WITHOUT a VitePress build: run the same validators
# `docs/make.jl` runs, then evaluate every ```@eval block of each page the way the
# docs build does (same `Main.BRMDocsComparisons`, same per-page example modules) and
# fail if a block throws or a comparison fell back to "StanBlocks unsupported".
#
# The full site build takes 25-60 minutes; this takes a few minutes per page and
# catches what breaks a page in CI: a bypassing `@brm` fence, a source marker that no
# longer matches the research file, a model that no longer lowers.
#
#     julia --project=test docs/check_page.jl docs/src/state-space-models.md [more pages]
#
# It does not check Documenter cross-references, VitePress dead links or rendering.

include(joinpath(@__DIR__, "backend_comparisons.jl"))
include(joinpath(@__DIR__, "centering_examples.jl"))
using Markdown

function check_page(page)
    BRMDocsComparisons.validate_no_bypasses([page])
    source = read(page, String)
    blocks = [m.captures[1] for m in eachmatch(r"(?ms)^```@eval\n(.*?)^```\s*$", source)]
    occursin("Main.BRMDocsComparisons.comparison(", source) &&
        BRMDocsComparisons.validate_generated_templates([page])
    println(basename(page), ": validators ok, ", length(blocks), " @eval blocks")
    failures = String[]
    for (i, block) in enumerate(blocks)
        started = time()
        value = try
            include_string(Main, block, "$(basename(page)):@eval-$i")
        catch err
            push!(failures, "block $i threw: " * first(sprint(showerror, err), 300))
            continue
        end
        comparison = value isa Markdown.MD && !isempty(value.content) &&
            value.content[1] isa Markdown.Code && value.content[1].language == "brm-comparison"
        note = ""
        if comparison
            text = join((c.code for c in value.content if c isa Markdown.Code), "\n")
            for fallback in ("StanBlocks unsupported for this BRM example",
                             "Stan emission unavailable because StanBlocks lowering failed")
                occursin(fallback, text) && push!(failures, "block $i: comparison `$(value.content[1].code)` contains `$fallback`")
            end
            note = "  comparison `$(value.content[1].code)`, $(length(value.content) - 1) panes"
        end
        println("  block ", i, ": ", round(time() - started; digits=1), " s", note)
    end
    return failures
end

function main(pages)
    isempty(pages) && error("usage: julia --project=test docs/check_page.jl docs/src/<page>.md [...]")
    failures = String[]
    for page in pages
        append!(failures, string(basename(page), ": ") .* check_page(abspath(page)))
    end
    foreach(f -> println("FAIL  ", f), failures)
    isempty(failures) || exit(1)
    println("PAGE_CHECK_OK")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
