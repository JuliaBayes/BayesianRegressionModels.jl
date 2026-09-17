# DOs in dependency order. Every feature is a focused @dynamicstruct:
# - Backend/data lives on `AppData`: `dataset`, `run(text, ns)` with its pipeline
#   data, `step_chain` / `compute_steps` for polling_fetchindex, `context` for
#   per-request namespace/run bundles.
# - UI/HTML lives on the routes structs: `PipelineRoutes` owns the formula
#   editor page, per-step render dispatch, and `context!`; the `@include
#   examples` sub-struct owns the examples list/detail/mark routes plus the
#   `entries`/`find`/`find_by_slug`/`persist!` operations that construct
#   ExampleEntry instances with the right `__parent__` for URL construction.
struct FormulaSecurityError <: Exception
    msg::String
end
Base.showerror(io::IO, e::FormulaSecurityError) = print(io, "FormulaSecurityError: ", e.msg)

_ALLOWED_CALLS = Set{Symbol}([
    :~, :(+), :(-), :(*), :(/), :(^), :(|), :(||), :(&),
    :(==), :(!=), :(<), :(>), :(<=), :(>=),
    :log, :log2, :log10, :log1p, :exp, :exp2, :expm1,
    :sqrt, :cbrt, :abs, :abs2, :sign, :floor, :ceil, :round,
    :sin, :cos, :tan, :asin, :acos, :atan,
    :min, :max, :clamp, :mod, :rem, :div,
    :logistic, :logit, :softmax, :logsumexp,
    :log_abs_tanh, :log_square_tanh,
    :Normal, :Poisson, :Binomial, :BinomialLogit, :Bernoulli, :BernoulliLogit, :Beta, :Gamma,
    :Exponential, :Cauchy, :TDist, :LocationScale, :LogNormal, :Weibull,
    :NegativeBinomial, :NegativeBinomial2, :BetaBinomial, :BetaBinomial2,
    :Geometric, :Laplace, :Uniform,
    :MvNormal, :MixtureModel, :Dirichlet,
    :InverseGamma, :InverseGaussian, :VonMises, :CircularVonMises, :Pareto,
    :OrderedLogistic, :Ordinal, :Cumulative, :StoppingRatio,
    :LogitLink, :ProbitLink, :CloglogLink, :Categorical, :CategoricalLogit,
    :zscale, :center, :standardize, :factor, :offset, :protect,
    :s, :bs, :t2, :gp, :hsgp, :ar, :dar, :ar1, :mo, :mo1, :mi,
    :cbind, :mvbind, :mm, :gr, :dp, :me, :centered,
    :Horseshoe, :ZeroInflatedPoisson, :HurdlePoisson, :weighted,
    :length, :unique, :sort, :size, :eltype, :nrow, :ncol,
])

# NOTE: `:.` is deliberately NOT a safe head. A `:.` head admits a qualified
# reference like `Base.exit` / `Core.eval`; combined with the `:call` branch
# only allowlist-checking `fname` when `fname isa Symbol`, that let ANY
# `Module.func(args…)` slip through the screen (`Base.exit()`,
# `Base.open("/etc/passwd")`, `Core.eval(1)`, …). No legitimate formula uses a
# dotted head, so excluding it costs nothing and closes the bypass. See the
# explicit dotted-call rejection in the `:call` branch below.
_SAFE_HEADS = Set{Symbol}([
    :block, :call, :(=), :(||), :tuple, :vect, :ref,
    :kw, :parameters, :(...),
    :comparison, :&&,
])
walk(x) = _walk(x, false)
_walk(x, inside_nested_brm::Bool) = begin
    x isa Union{Number, AbstractString, Symbol, LineNumberNode,
                Nothing, Bool, QuoteNode} && return nothing
    x isa Expr || return nothing
    if x.head == :macrocall
        x.args[1] === Symbol("@brm") ||
            return FormulaSecurityError(
                "macro calls are not allowed in formulas (got $(x.args[1]))")
        inside_nested_brm &&
            return FormulaSecurityError(
                "a nested predictor formula cannot contain another nested `@brm`")
        length(x.args) == 3 ||
            return FormulaSecurityError(
                "nested `@brm` requires exactly one parenthesized predictor expression")
        payload = x.args[3]
        if payload isa Expr &&
           (payload.head == :block || payload.head == :(=) ||
            (payload.head == :call && !isempty(payload.args) && payload.args[1] === :~))
            return FormulaSecurityError(
                "nested `@brm(...)` accepts one predictor expression, not a block or statement")
        end
        return _walk(payload, true)
    end
    x.head in (:cmd, :string) &&
        return FormulaSecurityError(
            "`$(x.head)` expressions are not allowed in formulas")
    (x.head == :quote || x.head == :$) &&
        return FormulaSecurityError(
            "quote/interpolation expressions are not allowed in formulas")
    if x.head == :call
        fname = x.args[1]
        if fname isa Symbol && fname ∉ _ALLOWED_CALLS
            return FormulaSecurityError(
                "function `$fname` is not in the formula allowlist. " *
                "Allowed: arithmetic, math, distributions, DSL operators.")
        end
        # Qualified/dotted calls (`Module.func(args…)`) are never allowlisted:
        # `fname` is an `Expr(:., …)`, not a Symbol, so the check above cannot
        # see it. Reject it explicitly (belt-and-suspenders alongside `:.`
        # being absent from `_SAFE_HEADS`) with a clear message.
        if fname isa Expr && fname.head === :.
            return FormulaSecurityError(
                "qualified/dotted calls are not allowed in formulas (got `$fname`). " *
                "Use a bare allowlisted function name.")
        end
        if fname isa Expr && fname.head == :curly
            tname = fname.args[1]
            tname isa Symbol && tname ∉ _ALLOWED_CALLS &&
                return FormulaSecurityError(
                    "type constructor `$tname` is not in the formula allowlist")
        end
    end
    x.head ∉ _SAFE_HEADS &&
        return FormulaSecurityError(
            "expression type `:$(x.head)` is not allowed in formulas")
    for arg in x.args
        v = _walk(arg, inside_nested_brm); v === nothing || return v
    end
    nothing
end
