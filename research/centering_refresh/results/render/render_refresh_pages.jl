using Documenter, DocumenterVitepress, BayesianRegressionModels
const repo = "/home/n/.local/state/kb-agents-worktrees/BayesianRegressionModels-docs-adaptive-centering"
include(joinpath(repo, "docs/backend_comparisons.jl"))
const root = joinpath(@__DIR__, "centering-refresh-render-v1")
const source = joinpath(root, "source")
mkpath(source)
for name in ("adaptive-centering.md", "eight-schools-centering.md", "radon-centering.md", "pupil-centering.md")
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
        "Motorcycle HSGP"=>"adaptive-centering.md",
        "Eight schools"=>"eight-schools-centering.md",
        "Radon"=>"radon-centering.md",
        "Pupil: exact marginalization"=>"pupil-centering.md"]],
    checkdocs=:none,warnonly=true)
for (page, model) in (("adaptive-centering",:adaptive_motorcycle_model),
        ("eight-schools-centering",:eight_schools_model),("radon-centering",:adaptive_radon_model))
    BRMDocsComparisons.validate_required_stan_outputs(
        joinpath(root,"build/.documenter",page*".md"), (model,))
end
DocumenterVitepress.build_docs(joinpath(root,"build"))
println("FOUR_PAGE_RENDER_COMPLETE")
