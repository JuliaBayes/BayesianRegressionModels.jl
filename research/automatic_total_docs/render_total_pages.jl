using Documenter, DocumenterVitepress, BayesianRegressionModels
const repo = joinpath(@__DIR__, "pupil4-study")
include(joinpath(repo, "docs/backend_comparisons.jl"))
include(joinpath(repo, "docs/centering_examples.jl"))
const root = joinpath(@__DIR__, "automatic-total-render-v1")
const source = joinpath(root, "source")
mkpath(source)
for name in ("pupil-centering.md", "pupil-scale-centering.md", "air-centering.md")
    cp(joinpath(repo, "docs/src", name), joinpath(source, name); force=true)
end
for name in (".vitepress", "assets")
    cp(joinpath(repo, "docs/src", name), joinpath(source, name); force=true)
end
makedocs(sitename="BayesianRegressionModels.jl", source=source,
    build=joinpath(root,"build"), modules=Module[], remotes=nothing,
    format=DocumenterVitepress.MarkdownVitepress(repo="github.com/nsiccha/BayesianRegressionModels.jl",
        devbranch="ns/devibe",devurl="dev",build_vitepress=false),
    pages=["Adaptive centering"=>[
        "Pupil: numeric scale predictor"=>"pupil-centering.md",
        "Pupil: hierarchical residual SD"=>"pupil-scale-centering.md",
        "Air pollution"=>"air-centering.md"]],
    checkdocs=:none,warnonly=true)
for (page, models) in (("pupil-centering",(:pupil_numeric_brm_model,)),
        ("pupil-scale-centering",(:pupil_hierarchical_brm_model,)),
        ("air-centering",(:air_intercept_brm_model,:air_independent_brm_model)))
    BRMDocsComparisons.validate_required_stan_outputs(
        joinpath(root,"build/.documenter",page*".md"), models)
end
cp(joinpath(repo,"docs/package.json"),joinpath(root,"package.json");force=true)
DocumenterVitepress.build_docs(joinpath(root,"build"))
println("THREE_AUTOMATIC_TOTAL_PAGES_RENDER_COMPLETE")
