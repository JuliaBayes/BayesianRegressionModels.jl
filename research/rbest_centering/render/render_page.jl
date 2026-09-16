# Focused local render of the RBesT page: REPO OUT   (Documenter pane generation with required Stan outputs, then VitePress)
using Documenter, DocumenterVitepress, BayesianRegressionModels
repo, root = abspath(ARGS[1]), abspath(ARGS[2])
@assert pkgdir(BayesianRegressionModels) == repo "docs environment must develop the rendered checkout: $(pkgdir(BayesianRegressionModels)) != $repo"
include(joinpath(repo, "docs/backend_comparisons.jl"))
include(joinpath(repo, "docs/centering_examples.jl"))
source = joinpath(root, "source"); mkpath(source)
for name in ("rbest-centering.md", "air-centering.md")
    cp(joinpath(repo, "docs/src", name), joinpath(source, name); force=true)
end
for name in (".vitepress", "assets")
    cp(joinpath(repo, "docs/src", name), joinpath(source, name); force=true)
end
makedocs(sitename="BayesianRegressionModels.jl", source=source, build=joinpath(root, "build"), modules=Module[], remotes=nothing,
    format=DocumenterVitepress.MarkdownVitepress(repo="github.com/nsiccha/BayesianRegressionModels.jl", devbranch="ns/devibe", devurl="dev", build_vitepress=false),
    pages=["Adaptive centering" => ["Air pollution" => "air-centering.md", "RBesT MAP prior" => "rbest-centering.md"]],
    checkdocs=:none, warnonly=true)
for (page, models) in (("rbest-centering", (:rbest_as_brm_model, :rbest_crohn_brm_model)), ("air-centering", (:air_intercept_brm_model, :air_independent_brm_model)))
    BRMDocsComparisons.validate_required_stan_outputs(joinpath(root, "build/.documenter", page * ".md"), models)
end
cp(joinpath(repo, "docs/package.json"), joinpath(root, "package.json"); force=true)
DocumenterVitepress.build_docs(joinpath(root, "build"))
println("RBEST_PAGE_RENDER_COMPLETE")
