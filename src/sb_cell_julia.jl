# ---- Julia-first kernel cells on StanBlocks ---------------------------------
#
# A kernel cell (`@plate for`, or the deprecated `kernel(...) do`) is ordinary
# Julia source. RKBRMI evaluates it as written. StanBlocks traces most of the
# same source unchanged: operators, dotted broadcasts, indexing, and every
# function whose Stan builtin shares its Julia name (`exp`, `log`, `sum`,
# `abs2.`, ...). This pass supplies the SB backend's remaining Stan spellings:
#
# - a deterministic call whose Julia callable is a differently named Stan
#   builtin (`logistic` -> `inv_logit`, `cumsum` -> `cumulative_sum`,
#   `fill`/`zeros`/`ones` -> `rep_vector`/`rep_matrix`, `vcat` ->
#   `append_row`, ...);
# - an observation constructor from Distributions (`y ~ Normal(mu, s)`),
#   through the `_sb_stan_distribution_call` table the top-level likelihoods
#   use, also as the family of `censored`/`truncated`/`interval_censored`/
#   `weighted`.
#
# A call is matched by the callable the cell's own module binds, never by its
# spelling, so a caller function that happens to be named `logistic` is left
# to StanBlocks. Everything else passes through unchanged: StanBlocks
# spellings keep their current Stan source, and a caller helper resolves to the
# `@deffun` of the same name, whose Julia methods are what RKBRMI calls.

# The value `head` names in `scope`, or `nothing` when nothing binds it there.
_sb_cell_binding(scope, head) = nothing
_sb_cell_binding(scope::Module, head::Symbol) =
    isdefined(scope, head) ? getfield(scope, head) : nothing
_sb_cell_binding(scope, head::GlobalRef) =
    isdefined(head.mod, head.name) ? getfield(head.mod, head.name) : nothing
function _sb_cell_binding(scope::Module, head::Expr)
    Meta.isexpr(head, :., 2) && head.args[2] isa QuoteNode || return nothing
    owner = _sb_cell_binding(scope, head.args[1])
    field = head.args[2].value
    owner isa Module && field isa Symbol && isdefined(owner, field) ?
        getfield(owner, field) : nothing
end

# The Stan builtin a Julia callable is spelled as, for calls and dotted calls.
_sb_cell_stan_name(callable) = nothing
_sb_cell_stan_name(::typeof(LogExpFunctions.logistic)) = :inv_logit
_sb_cell_stan_name(::typeof(LogExpFunctions.log1pexp)) = :log1p_exp
_sb_cell_stan_name(::typeof(LogExpFunctions.log1mexp)) = :log1m_exp
_sb_cell_stan_name(::typeof(LogExpFunctions.logsumexp)) = :log_sum_exp
_sb_cell_stan_name(::typeof(cumsum)) = :cumulative_sum

# The Stan call for a Julia call, or `nothing` to keep the authored call.
function _sb_cell_stan_call(callable, args)
    name = _sb_cell_stan_name(callable)
    name === nothing ? nothing : Expr(:call, name, args...)
end
# Constructors whose Stan builtin depends on the result's rank.
_sb_cell_stan_call(::typeof(fill), args) =
    length(args) == 2 ? Expr(:call, :rep_vector, args...) :
    length(args) == 3 ? Expr(:call, :rep_matrix, args...) : nothing
_sb_cell_stan_call(::typeof(zeros), args) = _sb_cell_stan_call(fill, Any[0.0; args])
_sb_cell_stan_call(::typeof(ones), args) = _sb_cell_stan_call(fill, Any[1.0; args])
# Stan concatenates two values at a time.
_sb_cell_stan_call(::typeof(vcat), args) = length(args) < 2 ? nothing :
    foldl((a, b) -> Expr(:call, :append_row, a, b), args)

_sb_cell_combinator(head) =
    head in (:censored, :truncated, :interval_censored, :weighted)

_sb_cell_keyword_argument(arg) = Meta.isexpr(arg, (:parameters, :kw))

# One call argument in StanBlocks spelling; keyword names stay as written.
_sb_cell_argument(arg, scope) = _sb_cell_value(arg, scope)
function _sb_cell_argument(arg::Expr, scope)
    Meta.isexpr(arg, :kw, 2) &&
        return Expr(:kw, arg.args[1], _sb_cell_value(arg.args[2], scope))
    Meta.isexpr(arg, :parameters) &&
        return Expr(:parameters, (_sb_cell_argument(a, scope) for a in arg.args)...)
    _sb_cell_value(arg, scope)
end

_sb_cell_keywords!(keywords, arg) = keywords
_sb_cell_keywords!(keywords, arg::Symbol) = push!(keywords, arg => arg)
function _sb_cell_keywords!(keywords, arg::Expr)
    if Meta.isexpr(arg, :kw, 2)
        push!(keywords, arg.args[1] => arg.args[2])
    elseif Meta.isexpr(arg, :parameters)
        foreach(a -> _sb_cell_keywords!(keywords, a), arg.args)
    end
    keywords
end

# A deterministic cell expression in StanBlocks spelling.
_sb_cell_value(x, scope) = x
function _sb_cell_value(x::Expr, scope)
    if x.head === :call
        head = first(x.args)
        args = Any[_sb_cell_argument(a, scope) for a in x.args[2:end]]
        if !any(_sb_cell_keyword_argument, args)
            callable = _sb_cell_binding(scope, head)
            rewritten = callable === nothing ? nothing : _sb_cell_stan_call(callable, args)
            rewritten === nothing || return rewritten
        end
        return Expr(:call, head, args...)
    elseif _brm_is_broadcast_call(x)
        head = first(x.args)
        callable = _sb_cell_binding(scope, head)
        name = callable === nothing ? nothing : _sb_cell_stan_name(callable)
        return Expr(:., something(name, head),
            Expr(:tuple, (_sb_cell_argument(a, scope) for a in x.args[2].args)...))
    elseif x.head === :. && length(x.args) == 2 && x.args[2] isa QuoteNode
        return Expr(:., _sb_cell_value(x.args[1], scope), x.args[2])
    end
    Expr(x.head, (_sb_cell_value(a, scope) for a in x.args)...)
end

# An observation's law in StanBlocks spelling: a Distributions constructor
# becomes its Stan family call, and a combinator spells its family argument.
# Any other family head is a StanBlocks family token and stays as written
# (Stan's `logistic` family is not the Julia `logistic` function).
function _sb_cell_family(x, scope)
    Meta.isexpr(x, :call) || return _sb_cell_value(x, scope)
    head = first(x.args)
    callable = _sb_cell_binding(scope, head)
    args = Any[_sb_cell_argument(a, scope) for a in x.args[2:end]]
    if callable isa Type && callable <: Distribution
        positional = Tuple(filter(!_sb_cell_keyword_argument, args))
        keywords = foldl(_sb_cell_keywords!, filter(_sb_cell_keyword_argument, args);
            init=Pair{Symbol,Any}[])
        return _sb_stan_distribution_call(callable, positional, NamedTuple(keywords))
    elseif head isa Symbol && _sb_cell_combinator(head)
        family = findfirst(!_sb_cell_keyword_argument, x.args[2:end])
        family === nothing || (args[family] = _sb_cell_family(x.args[1 + family], scope))
    end
    Expr(:call, head, args...)
end

# An unbound family token (`normal`) names the StanBlocks family of that name.
function _sb_cell_family_binding(scope, head)
    callable = _sb_cell_binding(scope, head)
    callable === nothing && head isa Symbol && isdefined(StanBlocks, head) ?
        getfield(StanBlocks, head) : callable
end

_sb_cell_statement(x, scope) = _sb_cell_value(x, scope)
function _sb_cell_statement(x::Expr, scope)
    # A cell's `@scan` block and its step loop hold statements of their own.
    _is_brm_scan(x) &&
        return Expr(:macrocall, x.args[1:end-1]..., _sb_cell_block(x.args[end], scope))
    Meta.isexpr(x, :for) &&
        return Expr(:for, _sb_cell_value(x.args[1], scope), _sb_cell_block(x.args[2], scope))
    Meta.isexpr(x, :call, 3) && x.args[1] === :~ || return _sb_cell_value(x, scope)
    law = x.args[3]
    if Meta.isexpr(law, :call) && first(law.args) === :weighted
        law = _brm_cell_typed_weight(head -> _sb_cell_family_binding(scope, head),
            law, "sbimpl"; normal=c -> c === Normal || c === StanBlocks.normal)
    end
    Expr(:call, :~, x.args[2], _sb_cell_family(law, scope))
end

_sb_cell_statements(statements, scope) =
    Any[_sb_cell_statement(statement, scope) for statement in statements]
_sb_cell_block(x, scope) = Meta.isexpr(x, :block) ?
    Expr(:block, _sb_cell_statements(x.args, scope)...) : _sb_cell_statement(x, scope)

# ---- `AR1Normal` in a cell: the BRM-owned AR(1) submodel ---------------------
#
# The `@slic` submodels `y[p] ~ AR1Normal(...)` calls (`ar1_normal.jl`): the
# exact AR(1) density over one series, given one mean, scale and persistence
# per step, at the recorded integer times or one step apart. `obs` is the
# caller's observed series; without it the submodel draws the series.
const _sb_ar1_normal = StanBlocks.@slic begin
    @scan begin
        obs[1] ~ normal(mu[1], sigma[1] / sqrt(1 - phi[1] ^ 2))
        for t in 2:n
            k = time[t] - time[t - 1]
            obs[t] ~ normal(mu[t] + phi[t] ^ k * (obs[t - 1] - mu[t - 1]),
                sigma[t] * sqrt((1 - phi[t] ^ (2 * k)) / (1 - phi[t] ^ 2)))
        end
    end
end

const _sb_ar1_normal_steps = StanBlocks.@slic begin
    @scan begin
        obs[1] ~ normal(mu[1], sigma[1] / sqrt(1 - phi[1] ^ 2))
        for t in 2:n
            obs[t] ~ normal(mu[t] + phi[t] * (obs[t - 1] - mu[t - 1]), sigma[t])
        end
    end
end

# A cell's statements with each `y ~ AR1Normal(...)` calling the AR(1)
# submodel: on the observed series (`obs = y`), or drawing it when the
# response is omitted and `y` is no cell parameter.
function _sb_cell_ar1_normal_calls(statements, params)
    any(_is_brm_ar1_normal, statements) || return statements
    used = _brm_cell_value_refs!(Set{Symbol}(params), Expr(:block, statements...))
    map(statements) do statement
        _is_brm_ar1_normal(statement) || return statement
        spec = _brm_kernel_ar1_normal(statement, params, statements)
        observed = spec.response in params
        observed || spec.length != :(length($(spec.response))) || error(
            "sbimpl: `$(spec.response) ~ AR1Normal(...)` with `$(spec.response)` omitted " *
            "draws a series whose length nothing in the cell gives; pass " *
            "`time = t[p]`, or an argument read along the series")
        mu, sigma, phi = _brm_ar1_normal_series(spec, :fill)
        keywords = Any[Expr(:kw, :mu, mu), Expr(:kw, :sigma, sigma),
            Expr(:kw, :phi, phi), Expr(:kw, :n, spec.length)]
        spec.time === nothing || push!(keywords, Expr(:kw, :time, spec.time))
        node = spec.response
        if observed
            pushfirst!(keywords, Expr(:kw, :obs, spec.response))
            node = Symbol(spec.response, :_ar1)
            while node in used
                node = Symbol(node, :_)
            end
            push!(used, node)
        end
        submodel = spec.time === nothing ? _sb_ar1_normal_steps : _sb_ar1_normal
        Expr(:call, :~, node, Expr(:call, submodel, Expr(:parameters, keywords...)))
    end
end

# Whether an emitted statement is a cell's AR(1) submodel call.
# The plan reads a copy of the emitted body, so the submodel is matched by
# its model, not by identity.
_sb_plan_ar1_normal_call(x) = Meta.isexpr(x, :call, 3) && x.args[1] === :~ &&
    Meta.isexpr(x.args[3], :call) && (head = first(x.args[3].args);
        head isa StanBlocks.SlicModel && any(submodel -> isequal(head.model, submodel.model),
            (_sb_ar1_normal, _sb_ar1_normal_steps)))
