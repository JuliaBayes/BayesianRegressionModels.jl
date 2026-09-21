module BRMMacroWeb

using HTMXObjects
import HTMX
using DynamicObjects: fetchindex!, clear_mem_caches!, @memo!
using Treebars: polling_fetchindex, initialize_progress!,
                prepare_progress!, with_prepared_progress,
                htmx_treebar_styles, htmx_treebar_script,
                ThreadsafeDict
using Random
using Chairmarks
using DataFrames
using Statistics: quantile, median
using FiniteDifferences: FiniteDifferences, central_fdm
using BridgeStan: BridgeStan
using StanLogDensityProblems: StanProblem
using LogDensityProblems
# Formula presets reference `logistic` (Bernoulli / Binomial link); BRM imports
# LogExpFunctions internally but doesn't re-export it, so eval-context needs
# its own import.
using LogExpFunctions: logistic, logit, softmax, logsumexp
using StanBlocks
using Distributions
using OrderedCollections: OrderedDict
using WarmupHMC: initialize_mcmc, adaptive_warmup_mcmc
using JSON
using AlgebraOfVega: vega_head, vega_runtime, auto_remap_node, with_plot_caption,
    config, pointinterval, lineribbon, to_node, ppc_overlay,
    ECDFPlot, VLines, nonnumeric, Scatter
import AlgebraOfGraphics as AoG

# The @brm macro and VBRMI / SBBRMI implementations now live in the main
# `BayesianRegressionModels` package (moved out of web-macro in `ns/devibe`
# so a downstream package can consume them as a submodule without the web app deps).
using BayesianRegressionModels
# Pull in the types / functions the web app touches by unqualified name.
using BayesianRegressionModels: AbstractColumn, MissingColumn, DataColumn,
    NamedColumn, ExprColumn, LikelihoodColumn, MaterializedColumn,
    Data, MaybeData, BRMI, VBRMI, SBBRMI,
    assign, doublepipe, gr, mm, gp, hsgp, offset, zscale, center, standardize, protect,
    me, s, ar, OrderedLogistic, Ordinal, Cumulative, StoppingRatio,
    LogitLink, ProbitLink, CloglogLink,
    # Custom-distribution stub names. SLIC's symbol resolver checks
    # `isdefined(BRMMacroWeb, :name)`, which is false for names only
    # brought in via plain `using`; explicit-name imports register them
    # so the lookup succeeds.
    zero_inflated_poisson, zero_inflated_poisson_lpmf,
    zero_inflated_poisson_lpmfs, zero_inflated_poisson_rng,
    brm_ordinal, brm_ordinal_lpmf, brm_ordinal_lpmfs, brm_ordinal_rng,
    brm_ordinal_logcdf, brm_ordinal_logccdf, brm_ordinal_cdf,
    multi_std_normal, multi_std_normal_lpdf, ranef_b_matrix,
    hurdle_poisson, hurdle_poisson_lpmf,
    hurdle_poisson_lpmfs, hurdle_poisson_rng,
    brm_von_mises, brm_von_mises_lpdf,
    brm_von_mises_lpdfs, brm_von_mises_rng,
    Horseshoe, CategoricalLogit, ZeroInflatedPoisson, HurdlePoisson,
    NegativeBinomial2,
    BetaBinomial, BetaBinomial2, CircularVonMises,
    # Accessors used unqualified by html_expr.jl and stan_compile code.
    name, getargs, getf, getkwargs, getbroadcast, getop,
    # Macro / pipeline entry points called by Formula + stan_code.
    parse!, _brm, stan_code, maybedata,
    # Part machinery and push_parts!! (called by a confidential -ext.jl unqualified).
    Part, push_parts!!, vbroadcasted, llikelihood!
# Extension hooks — use `import` (not `using`) so a confidential -ext.jl can
# ADD methods to the same function binding rather than creating a shadowing
# local function.
import BayesianRegressionModels: _sb_submodel_rhs!, vmeta_sampling_rhs,
    nparams, lprior!

# Styled HTML rendering for BRMI / VBRMI cards stays web-side (pulls in
# HTMX builders).
include("html_expr.jl")

# Extension hook: extensions (e.g. a gitignored `<prefix>-ext.jl`) that need
# to contribute auxiliary data which doesn't fit as per-row DataFrame
# columns -- for example `dose_times::Vector{<:AbstractVector}` indexed by
# subject id -- add a method `dataset_extras(::Val{:ns}, df)` returning a
# NamedTuple of extras. The namespace is derived from the example
# label/slug (first dash/space-separated segment), so a `<prefix>-*` example
# dispatches to `::Val{:<prefix>}`. Default is no extras.
dataset_extras(::Val, df) = (;)

# DOs in dependency order. Every feature is a focused @dynamicstruct:
# - Backend/data lives on `AppData`: `dataset`, `run(text, ns)` with its pipeline
#   data, `step_chain` / `compute_steps` for polling_fetchindex, `context` for
#   per-request namespace/run bundles.
# - UI/HTML lives on the routes structs: `PipelineRoutes` owns the formula
#   editor page, per-step render dispatch, and `context!`; the `@include
#   examples` sub-struct owns the examples list/detail/mark routes plus the
#   `entries`/`find`/`find_by_slug`/`persist!` operations that construct
#   ExampleEntry instances with the right `__parent__` for URL construction.
include("security.jl")
include("formula_dataset.jl")
include("example_entry.jl")
include("draws_helpers.jl")
include("ppc_kinds.jl")
include("app_data.jl")
include("pipeline_routes.jl")
include("app_context.jl")
# BRMDescriptor → semantic_app mount (consumer-side adapter, decision 0lqlxi4).
# A self-contained submodule, also includable standalone (see
# ../validate_descriptor_mount.jl). Routes nothing on load.
include("descriptor_mount.jl")

function __init__()
    route!(AppContext())
end

# Confidential client-project extensions (gitignored `<prefix>-ext.jl`); discover
# and load each present one. Each adds a `dataset_extras(::Val{:<prefix>}, df)`
# method and pushes its formula calls into `_ALLOWED_CALLS`. The discovered
# prefixes are recorded in `CONFIDENTIAL_EXT_PREFIXES`, which drives the
# example-visibility gate (example_entry.jl) and gallery recording (app_data.jl),
# so BRM's tracked source names no specific client project.
const CONFIDENTIAL_EXT_PREFIXES = let dir = @__DIR__
    prefixes = String[]
    for _name in sort(readdir(dir))
        m = match(r"^(.+)-ext\.jl$", _name)
        m === nothing && continue
        path = joinpath(dir, _name)
        isfile(path) || continue
        include(path)
        push!(prefixes, String(m.captures[1]))
    end
    prefixes
end

end # module
