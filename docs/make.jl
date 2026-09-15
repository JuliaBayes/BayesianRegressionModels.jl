using Documenter, DocumenterVitepress, BayesianRegressionModels

include("backend_comparisons.jl")

const GENERATED_EXAMPLE_PAGES = [
    joinpath(@__DIR__, "src", "index.md"),
    joinpath(@__DIR__, "src", "formula-terms.md"),
    joinpath(@__DIR__, "src", "likelihoods.md"),
    joinpath(@__DIR__, "src", "feature-atlas.md"),
    joinpath(@__DIR__, "src", "warfarin.md"),
    joinpath(@__DIR__, "src", "wastewater.md"),
    joinpath(@__DIR__, "src", "wastewater-cdc.md"),
    joinpath(@__DIR__, "src", "seal-brm.md"),
    joinpath(@__DIR__, "src", "adaptive-centering.md"),
    joinpath(@__DIR__, "src", "eight-schools-centering.md"),
    joinpath(@__DIR__, "src", "radon-centering.md"),
]
const DOCS_MARKDOWN_PAGES = sort!(String[
    joinpath(root, file)
    for (root, _, files) in walkdir(joinpath(@__DIR__, "src"))
    for file in files
    if endswith(file, ".md")
])
BRMDocsComparisons.validate_generated_templates(GENERATED_EXAMPLE_PAGES)
BRMDocsComparisons.validate_no_bypasses(DOCS_MARKDOWN_PAGES)

# The docs have one publication home. Keep the target literal: Documenter's
# GitHub Actions deploy criterion requires the workflow repository to match
# `repo`, and the CI-equivalent docs wrapper validates the same static target.
# A legacy JuliaBayes workflow therefore cannot create a drifting second site.
const DOCS_DEVBRANCH = "ns/devibe"

# Note: BRM's docs must build without the private HTMXObjects.jl repo — the
# CI cannot clone it without repository-specific credentials. So `htmxo-embed.ts`
# is committed in-tree (see docs/src/.vitepress/theme/) rather than synced from
# HTMXObjects via `HTMXObjects.vitepress_theme_install` at build time. Local
# edits to the canonical upstream file should be hand-copied; see
# /docs-vitepress §8.
@static if Base.find_package("HTMXObjects") !== nothing
    @eval import HTMXObjects
    @eval HTMXObjects.vitepress_theme_install(joinpath(@__DIR__, "src", ".vitepress", "theme"))
end

makedocs(
    sitename = "BayesianRegressionModels.jl",
    modules  = [BayesianRegressionModels],
    format   = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/nsiccha/BayesianRegressionModels.jl",
        devurl = "dev",
        devbranch = DOCS_DEVBRANCH,
        build_vitepress = false,
    ),
    pages = [
        "Home" => "index.md",
        "Formula terms" => "formula-terms.md",
        "Likelihoods" => "likelihoods.md",
        "Feature atlas" => "feature-atlas.md",
        "Warfarin PK/PD" => "warfarin.md",
        "Wastewater Rt" => "wastewater.md",
        "CDC ww-inference" => "wastewater-cdc.md",
        "Grey-seal IPM" => "seal-brm.md",
        "Adaptive HSGP centering" => "adaptive-centering.md",
        "Adaptive eight-schools centering" => "eight-schools-centering.md",
        "Adaptive radon centering" => "radon-centering.md",
        "Turing backend" => "turing-backend.md",
        "Formulas and custom models: deck" => "feature-deck.md",
        "Gallery" => "gallery.md",
        "API" => "api.md",
    ],
    checkdocs = :none,
    warnonly = true,
)

BRMDocsComparisons.validate_required_stan_outputs(
    joinpath(@__DIR__, "build", ".documenter", "warfarin.md"),
    (:warfarin_pk_brmi, :warfarin_pd_brmi, :warfarin_joint_brmi),
)

BRMDocsComparisons.validate_required_stan_outputs(
    joinpath(@__DIR__, "build", ".documenter", "wastewater.md"),
    (:wastewater_brm_model,),
)

BRMDocsComparisons.validate_required_stan_outputs(
    joinpath(@__DIR__, "build", ".documenter", "wastewater-cdc.md"),
    (:cdc_ww_brm_model,),
)

BRMDocsComparisons.validate_required_stan_outputs(
    joinpath(@__DIR__, "build", ".documenter", "seal-brm.md"),
    (:grey_seal_brm_model,),
)

BRMDocsComparisons.validate_required_stan_outputs(
    joinpath(@__DIR__, "build", ".documenter", "adaptive-centering.md"),
    (:adaptive_motorcycle_model,),
)

BRMDocsComparisons.validate_required_stan_outputs(
    joinpath(@__DIR__, "build", ".documenter", "eight-schools-centering.md"),
    (:eight_schools_model,),
)

BRMDocsComparisons.validate_required_stan_outputs(
    joinpath(@__DIR__, "build", ".documenter", "radon-centering.md"),
    (:adaptive_radon_model,),
)

# Copy committed live-brm recordings into the VitePress build tree so
# the gallery embed resolves the same URLs in production as in dev
# (where vite proxies /live-brm to the running app on :8121).
let src = joinpath(@__DIR__, "src", "public")
    dst = joinpath(@__DIR__, "build", ".documenter", "public")
    if isdir(src)
        cp(src, dst; force=true)
        @info "Copied public assets to $dst"
    end
end

DocumenterVitepress.build_docs(joinpath(@__DIR__, "build"))

let redirect = joinpath(@__DIR__, "build", "index.html")
    isfile(redirect) || write(redirect, """
    <!DOCTYPE html>
    <html><head>
    <meta http-equiv="refresh" content="0; url=dev/">
    </head><body>Redirecting to <a href="dev/">dev</a>...</body></html>
    """)
end

DocumenterVitepress.deploydocs(
    repo = "github.com/nsiccha/BayesianRegressionModels.jl",
    devbranch = DOCS_DEVBRANCH,
    push_preview = true,
)
