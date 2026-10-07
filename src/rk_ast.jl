# BRM-side `@rkppl` AST emission (phase 2 U3 retarget). Pure Julia: builds
# the `_RKEmittedProgram` (surface-spelling submodel `defs` + the `main`
# `begin ... end` block `Expr`) that the thin layer lowers via
# `lower_rkppl(main, data_names; mod)` after evaluating the defs through
# `@rkppl` — no ReactiveKernels dependency, so this file is
# committed-testable without the PPL.
#
# Formula predictors: offset-only predictors emit a bare data
# affine (`mu = z`, no coefficients, no priors), and a predictor sharing
# its name with a data column is alpha-renamed (the single program
# namespace cannot hold both bindings). The AST is the sole emission
# path; the extension holds no fallback serializer.

# Explicit graph definitions keep their macro in the source and replay. Bare
# assignment definitions retain the existing @rkppl submodel convention.
function _rk_source_definition(definition)
    kind = :rkppl
    body = definition
    if Meta.isexpr(definition, :macrocall)
        head = first(definition.args)
        kernel = head === Symbol("@kernel") ||
            (head isa GlobalRef && nameof(head.mod) === :ReactiveKernels &&
                head.name === Symbol("@kernel")) ||
            isequal(head, Expr(:., :ReactiveKernels, QuoteNode(Symbol("@kernel"))))
        kernel || error("RK source: explicit definitions must use ReactiveKernels.@kernel")
        values = filter(arg -> !(arg isa LineNumberNode), definition.args[2:end])
        length(values) == 1 || error("RK source: graph definition needs one named body")
        body = only(values)
        kind = :kernel
    elseif Meta.isexpr(definition, :function)
        kind = :function
    end
    body isa Expr && body.head in (:(=), :function) &&
        length(body.args) == 2 && Meta.isexpr(first(body.args), :call) ||
        error("RK source: definition must be an ordinary function, @rkppl or @kernel definition")
    name = first(first(body.args).args)
    name isa Symbol || error("RK source: definition requires a plain local callable name")
    (; name, kind)
end

function _rk_validate_source_definitions(emitted)
    bound = Set{Symbol}(first(binding) for binding in emitted.bindings)
    defined = Set{Symbol}()
    for definition in emitted.defs
        name = _rk_source_definition(definition).name
        name in defined && error("RK source: callable `$name` is defined more than once")
        push!(defined, name)
        name in bound && error(
            "RK source: callable `$name` is both bound and defined; use separate entry and leaf names")
    end
    nothing
end

# A planned column can describe geometry while its emitted assignment owns
# the executable value. Bind only source inputs, so a cached preparation can
# never shadow that assignment on replay or rebinding.
function _rk_source_data_columns(plan, emitted)
    computed = Set{Symbol}()
    for statement in emitted.main.args
        _rk_source_assignments!(computed, statement)
    end
    Dict{Symbol,Any}(name => value for (name, value) in plan.columns if !(name in computed))
end

function _rk_source_assignments!(names, statement)
    statement isa Expr || return names
    if statement.head === :(=)
        _rk_source_lhs!(names, first(statement.args))
    elseif statement.head in (:macrocall, :for, :block)
        foreach(arg -> _rk_source_assignments!(names, arg), statement.args)
    end
    names
end

# Frozen labels are values, not source identifiers or categorical pool objects.
_rk_ast_level_value(value) = value
_rk_ast_level_value(value::CA.CategoricalValue) = _rk_ast_level_value(CA.unwrap(value))
_rk_ast_level_value(value::Symbol) = QuoteNode(value)
_rk_ast_level_values(values) = Expr(:vect, (_rk_ast_level_value(value) for value in values)...)

# Whole numerical calls may return arrays. State each fitted data result's
# observation axis in source, so authoring does not mistake it for a scalar.
# The range is geometry metadata; the values still come from the graph call.
function _rk_source_data_axes(statements, columns)
    taken = Set{Symbol}(keys(columns))
    foreach(statement -> _rk_source_symbols!(taken, statement), statements)
    out = Expr[]
    for statement in statements
        if !Meta.isexpr(statement, :(=), 2)
            push!(out, statement)
            continue
        end
        name, value = statement.args
        if !(name isa Symbol && haskey(columns, name) && columns[name] isa AbstractVector)
            push!(out, statement)
            continue
        end
        values = _rk_ast_fresh_name(string(name, "_source_values"), taken)
        rows = _rk_ast_fresh_name(string(name, "_source_rows"), taken)
        range = Expr(:call, GlobalRef(Base, :eachindex), values)
        push!(out, Expr(:(=), values, value),
            Expr(:(=), rows, Expr(:call, GlobalRef(Base, :collect), range)),
            Expr(:(=), name, _rk_ast_dotted(:getindex,
                Expr(:call, :Ref, values), rows)))
    end
    out
end

function _rk_lower_assignment_expr(node, name::Symbol)
    node isa Number && return node
    node isa _BRMPreparedRef && return node.name
    node isa _BRMPreparedExpr || error(
        "RK backend: internal: assignment `$name` holds an unlowerable node")
    isempty(node.kwargs) || error(
        "RK backend: internal: assignment `$name` carries keywords " *
        "(slice 1 admits pure positional calls)")
    node.callable isa Function || error(
        "RK backend: internal: assignment `$name` calls a non-function " *
        "callable")
    lowered = map(node.args) do arg
        _rk_lower_assignment_expr(arg, name)
    end
    Expr(:call, nameof(node.callable), lowered...)
end

function _rk_ast_coef_name(base::String, taken::Set{Symbol})
    name = Symbol(base)
    while name in taken
        name = Symbol(string(name), "_")
    end
    push!(taken, name)
    name
end

function _rk_ast_statistical_call!(definitions, taken, name, args...;
        kernel=false, template=getproperty(_BRM_STATISTICAL_VALUES, name))
    signature, body = template.args
    # Reuse a definition across distinct statistical blocks. A collision with
    # authored data, parameters or callable names only renames the definition.
    callee = _rk_ast_shared_definition!(definitions, taken, name,
        signature.args[2:end], deepcopy(body); kernel, head=template.head)
    Expr(:call, callee, args...)
end

# The indicator column of a shared factor's `position`-th coefficient level.
# Only variance-based shrinkage allocations read it; the effect itself gathers
# coefficients by level (`_rk_ast_affine`).
_rk_ast_factor_indicator!(definitions, taken, term, position) =
    _rk_ast_statistical_call!(definitions, taken, :brm_factor_dummy, only(term.columns),
        _rk_ast_level_value(term.options.level_values[position]); kernel=true)

# A horseshoe coefficient, as StanBlocks' `_sb_horseshoe`: the local scale
# and standardized draw, scaled by the predictor's shared global scale `tau`.
function _rk_ast_horseshoe_block!(definitions, taken, local_scale, tau, ratio)
    block = _rk_block_body()
    global_scale = _rk_block_argument!(block, :tau, tau)
    lambda, raw = _rk_block_local!(block, :lambda), _rk_block_local!(block, :raw)
    push!(block.statements, Expr(:call, :~, lambda, Expr(:call, :HalfCauchy, local_scale)),
        Expr(:call, :~, raw, Expr(:call, :Normal, 0, 1)))
    _rk_ast_block_call!(definitions, taken, "brm_horseshoe", block,
        Expr(:call, :*, raw, lambda, global_scale, ratio))
end

# A monotonic effect, as StanBlocks' `_sb_mo`: the increment simplex with its
# Dirichlet prior, returning the cumulative contrast at each row's level.
function _rk_ast_monotonic_block!(definitions, taken, term)
    column = only(term.columns)
    block = _rk_block_body()
    c = _rk_block_argument!(block, :c, column)
    increments = _rk_block_local!(block, :simplex_incr)
    push!(block.statements, Expr(:call, :~, increments,
        Expr(:call, :Dirichlet, Expr(:vect, term.options.alpha...))))
    contrast = _rk_ast_statistical_call!(definitions, taken,
        :brm_monotonic_contrast, c, increments)
    _rk_ast_block_call!(definitions, taken, "brm_monotonic", block, contrast)
end

# A statistical block is one self-contained submodel, as in StanBlocks: it
# allocates its parameters with their priors and returns its post-processed
# value, so the caller writes one `lhs ~ block(...)` statement and the block's
# coordinates live under `lhs.`. Arguments are substituted by their caller
# expressions, so an argument must not shadow a caller name the body reads
# freely (`free`, e.g. a prior's hyperparameter), and a local must not
# capture a name from the free reads or from any argument's value.
struct _RKBlockBody
    free::Set{Symbol}
    captured::Set{Symbol}
    allocated::Set{Symbol}
    arguments::Vector{Symbol}
    values::Vector{Any}
    statements::Vector{Any}
end

function _rk_block_body(free...)
    names = _rk_source_symbols!(Set{Symbol}(), collect(Any, free))
    _RKBlockBody(names, copy(names), Set{Symbol}(), Symbol[], Any[], Any[])
end

function _rk_block_local!(block::_RKBlockBody, base)
    name = _rk_ast_fresh_name(string(base), union(block.captured, block.allocated))
    push!(block.allocated, name)
    name
end

function _rk_block_argument!(block::_RKBlockBody, base, value)
    symbols = _rk_source_symbols!(Set{Symbol}(), value)
    isempty(intersect(symbols, setdiff(block.allocated, block.arguments))) || error(
        "RK backend: internal: allocate block arguments before its locals")
    argument = _rk_ast_fresh_name(string(base), union(block.free, block.allocated))
    push!(block.allocated, argument)
    union!(block.captured, symbols)
    push!(block.arguments, argument)
    push!(block.values, value)
    argument
end

# Identical blocks share one definition.
function _rk_ast_block_call!(definitions, taken, base, block::_RKBlockBody, value)
    body = Expr(:block, block.statements..., Expr(:return, value))
    for definition in definitions
        Meta.isexpr(definition, :(=), 2) || continue
        call = first(definition.args)
        Meta.isexpr(call, :call) && isequal(call.args[2:end], block.arguments) &&
            isequal(last(definition.args), body) || continue
        return Expr(:call, first(call.args), block.values...)
    end
    name = _rk_ast_fresh_name(string(base), taken)
    push!(definitions, Expr(:(=), Expr(:call, name, block.arguments...), body))
    Expr(:call, name, block.values...)
end

function _rk_ast_affine(predictor::_RKPredictorSpec, coefs::Dict{Int,Symbol},
        colref::Dict{Int}, refref::Dict{Int}; values::Bool=false,
        replacements=Dict{Int,Any}())
    summands = Any[]
    intercept = nothing
    for (index, term) in enumerate(predictor.terms)
        if haskey(replacements, index)
            # A component value (or nothing, when an earlier one absorbed it).
            replacement = replacements[index]
            replacement === nothing || push!(summands, replacement)
        elseif term.kind === :intercept
            # A scalar; it broadcasts against the other (row-valued) summands.
            intercept = coefs[index]
            push!(summands, intercept)
        elseif term.kind === :continuous
            push!(summands, Expr(:call, :.*, coefs[index], colref[index]))
        elseif term.kind === :factor
            if haskey(term.options, :index)
                # Each row reads its level's coefficient; a treatment-coded
                # reference level reads the leading zero.
                coefficients = term.options.coding === :fullrank ? refref[index] :
                    Expr(:call, :vcat, 0.0, refref[index])
                push!(summands, Expr(:ref, coefficients, term.options.index))
                continue
            end
            # Factor use is always bare `c[g]`; the LevelMap (full cover
            # or subset) rides the broadcast prior, and unmapped rows
            # contribute 0.
            push!(summands, Expr(:ref, refref[index], colref[index]))
        elseif term.kind === :ranef_gather
            push!(summands, refref[index])
        elseif term.kind === :monotonic
            # The coefficient scales the explicitly computed cumulative weights.
            push!(summands, Expr(:call, :.*, coefs[index], refref[index]))
        elseif term.kind === :monotonic_summand
            # A coefficient-free cumulative-weight value.
            push!(summands, refref[index])
        elseif term.kind === :dar
            # A trajectory whose recurrence and axis are stated in the preamble.
            push!(summands, refref[index])
        elseif term.kind === :offset
            push!(summands, colref[index])
        elseif term.kind === :structured
            push!(summands, refref[index])
        elseif term.kind === :spline
            # The ordinary fitted-basis product is a predictor summand.
            push!(summands, refref[index])
        elseif term.kind === :hsgp
            # The ordinary Hilbert basis/spectral product is a summand.
            push!(summands, refref[index])
        elseif term.kind === :gp
            push!(summands, refref[index])
        elseif term.kind === :ar
            # Scaled scan summand: the thin layer classifies
            # `coef .* state` as a ScanSummandTerm (SB's `ar` latent
            # path with its free beta).
            push!(summands, Expr(:call, :.*, coefs[index], refref[index]))
        elseif term.kind === :me
            # Scaled latent summand: the thin layer classifies
            # `coef .* latent` as a ContinuousTerm over the plate
            # vector (SB's `me` true covariate with its free beta).
            push!(summands, Expr(:call, :.*, coefs[index], refref[index]))
        end
    end
    length(summands) == 1 || return Expr(:call, :.+, summands...)
    # A lone intercept still states the predictor's rows.
    only(summands) === intercept && predictor.row_source !== nothing ?
        Expr(:call, :.*, intercept, Expr(:call, :ones,
            Expr(:call, :length, predictor.row_source))) : only(summands)
end

function _rk_ast_r2d2_scale(r2d2, addressee; scalar=true, variance_values=nothing)
    shares, variances = r2d2.allocation[addressee]
    variance_values === nothing || (variances = variance_values)
    scales = Any[Expr(:call, :*, r2d2.tau,
        Expr(:call, :sqrt, Expr(:call, :/,
            Expr(:call, :*, Expr(:ref, r2d2.phi, shares[j]), r2d2.r2),
            variances[j]))) for j in eachindex(shares)]
    scalar ? only(scales) : Expr(:vect, scales...)
end

function _rk_ast_spline_ids(plan::_RKStructuralPlan)
    ids = Set{Symbol}()
    for predictor in plan.predictors, term in predictor.terms
        term.kind === :spline || continue
        push!(ids, term.options.id)
    end
    ids
end

function _rk_ast_hsgp_ids(plan::_RKStructuralPlan)
    ids = Set{Symbol}()
    for predictor in plan.predictors, term in predictor.terms
        term.kind === :hsgp || continue
        push!(ids, term.options.id)
    end
    ids
end

function _rk_ast_dotted(head::Symbol, args...)
    Expr(:., head, Expr(:tuple, args...))
end

# The `levels(g)[S]` subset literal dropping position `p` of `K`: edge
# drops spell as explicit literal ranges, middle drops as literal index
# lists (no `end` — the plan knows `K`, so the literal is exact).
function _rk_ast_subset_literal(p::Int, K::Int)
    p == 1 && return Expr(:call, :(:), 2, K)
    p == K && return Expr(:call, :(:), 1, K - 1)
    Expr(:vect, [1:p-1; p+1:K]...)
end

# A factor coefficient's broadcast prior: `c[levels(g)] .~ Fam.(...)`
# full-rank, `c[levels(g)[S]] .~ Fam.(...)` for a reference subset.
# Always stated (factors have no default prior); the scalar args
# broadcast over the LevelMap block. One family per block (the
# thin-layer wide-block rule); the plan family symbol is the head.
function _rk_ast_factor_prior(coef::Symbol, col::Symbol,
        options::NamedTuple, K::Int, family::Symbol, args::Tuple)
    index = if haskey(options, :index)
        Expr(:call, :(:), 1, length(options.labels))
    elseif options.coding === :fullrank
        Expr(:call, :levels, col)
    else
        Expr(:ref, Expr(:call, :levels, col),
            _rk_ast_subset_literal(options.drop, K))
    end
    Expr(:call, :.~, Expr(:ref, coef, index),
        _rk_ast_dotted(family, args...))
end

_rk_ast_response_uses_scale(family::Symbol) =
    family === :gaussian || family === :nb2_log ||
    family === :gamma_log || family === :beta_logit ||
    family === :beta_binomial_logit ||
    family === :student_t || family === :hurdle_poisson ||
    family === :wald || family === :von_mises ||
    family === :negative_binomial || family === :zero_inflated_poisson ||
    family === :lognormal || family === :weibull

# The scale-slot body spelling inside a bare response statement. A
# direct scale (outer name, literal, or the plan-forbidden nothing)
# passes through inline. A distributional scale predictor inverts its
# link on the predictor name exactly like a location predictor (`exp.`
# for log), so the response always reads the constrained vector.
function _rk_ast_response_scale(response::_RKLikelihoodSpec,
        rename::Dict{Symbol,Symbol}, predictor_link::Dict{Symbol,Symbol})
    name = response.scale_predictor
    name === nothing && return response.scale
    response.scale === nothing || error(
        "RK backend: internal: response `$(response.response)` carries " *
        "both a scalar scale and a scale predictor")
    actual = get(rename, name, name)
    link = predictor_link[name]
    link === :identity && return actual
    link === :log && return _rk_ast_dotted(:exp, actual)
    link === :logit && return _rk_ast_dotted(:logistic, actual)
    error("RK backend: internal: scale predictor `$name` has link `$link`")
end

# The Student-t degrees of freedom inside a bare response statement: a
# direct nu (outer name or literal) passes through inline, exactly one
# of the scalar/predictor pair set (the planner guarantees it). A
# modeled nu inverts its link on the predictor name exactly like a
# scale predictor (`exp.` for log), so the head always reads the
# constrained vector.
function _rk_ast_response_nu(response::_RKLikelihoodSpec,
        rename::Dict{Symbol,Symbol}, predictor_link::Dict{Symbol,Symbol})
    name = response.nu_predictor
    name === nothing || response.nu === nothing || error(
        "RK backend: internal: response `$(response.response)` carries " *
        "both a scalar nu and a nu predictor")
    name === nothing && return response.nu
    actual = get(rename, name, name)
    link = predictor_link[name]
    link === :identity && return actual
    link === :log && return _rk_ast_dotted(:exp, actual)
    link === :logit && return _rk_ast_dotted(:logistic, actual)
    error("RK backend: internal: nu predictor `$name` has link `$link`")
end

# The inverse-link spelling (`Bernoulli.(logistic.(η))`,
# `Poisson.(exp.(η))`, …) is what the `@rkppl` surface takes; the thin
# layer recovers the link-native HAVE from it — the lowered
# `LikelihoodSpec` carries the same family+link the link-faithful
# direct serializer produced, on all six slice-1 families (verified
# behaviorally against `lower_rkppl`, not assumed).
#
# `leaf` maps each role to its INLINE spelling: `:predictor` (the
# predictor name, possibly renamed), `:scale` (the scale value or
# link-inverted scale predictor), `:nu` (the Student-t degrees of
# freedom: the scalar value or the link-inverted nu predictor),
# `:zero_inflation` (the ZIP zero
# probability, literal or name, inline; a modeled-zi submodel rides
# `:scale` under `logistic.` instead), `:trials`/`:weights`/`:lower`/
# `:upper` (columns or literals inline),
# `:extra_predictors`/`:count_columns` (tail predictors / tail count
# columns inline). Evidence and weights STRUCTURE (which wrapper,
# whether weighted) still read from `response`.
#
# With `fused_heads`, the six families the thin layer desugars
# pre-spine (`BernoulliLogit.(eta)`, `PoissonLog.(eta)`,
# `BinomialLogit.(n, mu)`, `NegativeBinomial2Log.(eta, phi)`,
# `GammaLog.(alpha, eta)`, `BetaLogit.(mu, kappa)`) emit the fused
# spelling; the desugar rewrites each to exactly the decomposed twin
# below (same roles, same order), and evidence/weights wrappers
# recurse, so the lowered plan is identical by construction. Default
# on at the Digest-2 pin; `false` remains the decomposed-twin pin.
#
function _rk_ast_fresh_name(base::String, taken::Set{Symbol})
    name = Symbol(base)
    tail = ""
    while name in taken
        tail *= "_"
        name = Symbol(base, tail)
    end
    push!(taken, name)
    name
end

function _rk_ast_response_dist(response::_RKLikelihoodSpec,
        leaf::Dict{Symbol,Any}, fused_heads::Bool,
        wrap_location::Bool=true)
    predictor = leaf[:predictor]
    # Mixture components reuse these branches per component: predictor
    # locations wrap (the decomposed twin — fused heads desugar
    # pre-spine on whole responses only, so components always spell
    # the wrapper form), while param/literal locations ride the
    # constrained scale bare. Single-family responses always wrap.
    base = if response.family === :gaussian
        _rk_ast_dotted(:Normal, predictor, leaf[:scale])
    elseif response.family === :bernoulli_logit
        # Triple 2 and triple 3 both lower to the T2 shape: the affine
        # value feeds logistic either way.
        fused_heads && wrap_location ?
            _rk_ast_dotted(:BernoulliLogit, predictor) :
            wrap_location ? _rk_ast_dotted(:Bernoulli,
                _rk_ast_dotted(:logistic, predictor)) :
            _rk_ast_dotted(:Bernoulli, predictor)
    elseif response.family === :poisson_log
        fused_heads && wrap_location ?
            _rk_ast_dotted(:PoissonLog, predictor) :
            wrap_location ? _rk_ast_dotted(:Poisson,
                _rk_ast_dotted(:exp, predictor)) :
            _rk_ast_dotted(:Poisson, predictor)
    elseif response.family === :binomial_logit
        # Both triples lower to one spelling: `Binomial.(n,
        # logistic.(p))` with a column or literal `n`.
        fused_heads && wrap_location ?
            _rk_ast_dotted(:BinomialLogit, leaf[:trials], predictor) :
            wrap_location ? _rk_ast_dotted(:Binomial, leaf[:trials],
                _rk_ast_dotted(:logistic, predictor)) :
            _rk_ast_dotted(:Binomial, leaf[:trials], predictor)
    elseif response.family === :bernoulli_probit
        _rk_ast_dotted(:Bernoulli,
            _rk_ast_dotted(:brm_invprobit, predictor))
    elseif response.family === :bernoulli_cloglog
        _rk_ast_dotted(:Bernoulli,
            _rk_ast_dotted(:brm_invcloglog, predictor))
    elseif response.family === :binomial_probit
        _rk_ast_dotted(:Binomial, leaf[:trials],
            _rk_ast_dotted(:brm_invprobit, predictor))
    elseif response.family === :binomial_cloglog
        _rk_ast_dotted(:Binomial, leaf[:trials],
            _rk_ast_dotted(:brm_invcloglog, predictor))
    elseif response.family === :beta_binomial_logit
        # Twin head (thin-layer decision, pair fam-betabinom): the
        # plan's `BetaBinomial2(n, mu, phi)` maps to
        # `BetaBinomial2.(n, logistic.(mu), phi)` (hurdle precedent);
        # precision rides the scale slot, scalars inline bare. No
        # fused head: one spelling either way.
        wrap_location ? _rk_ast_dotted(:BetaBinomial2, leaf[:trials],
            _rk_ast_dotted(:logistic, predictor),
            leaf[:scale]) :
        _rk_ast_dotted(:BetaBinomial2, leaf[:trials], predictor,
            leaf[:scale])
    elseif response.family === :beta_logit
        # Mean-concentration form: the plan pins mu (the predictor
        # itself) and kappa identical in both positions, so the same
        # values emit twice. `probit`/`cloglog` are thin-layer link
        # words (peel-and-discard, like `logistic`/`exp`); the AST
        # never calls them.
        mu = wrap_location ? _rk_ast_dotted(:logistic, predictor) : predictor
        kappa = leaf[:scale]
        fused_heads && wrap_location ?
            _rk_ast_dotted(:BetaLogit, predictor, kappa) :
            _rk_ast_dotted(:Beta,
                Expr(:call, :.*, mu, kappa),
                Expr(:call, :.*, Expr(:call, :.-, 1, mu), kappa))
    elseif response.family === :nb2_log
        fused_heads && wrap_location ?
            _rk_ast_dotted(:NegativeBinomial2Log, predictor, leaf[:scale]) :
            wrap_location ? _rk_ast_dotted(:NegativeBinomial2,
                _rk_ast_dotted(:exp, predictor),
                leaf[:scale]) :
            _rk_ast_dotted(:NegativeBinomial2, predictor, leaf[:scale])
    elseif response.family === :hurdle_poisson
        # Twin head (thin-layer decision, pair fam-hurdle): the plan's
        # `HurdlePoisson(lambda, p_zero)` maps to
        # `HurdlePoisson.(exp.(eta), p_zero)` (NB2 precedent); the hu
        # submodel rides the scale slot under `logistic.`, scalars
        # inline bare. No fused head: one spelling either way.
        wrap_location ? _rk_ast_dotted(:HurdlePoisson,
            _rk_ast_dotted(:exp, predictor),
            leaf[:scale]) :
        _rk_ast_dotted(:HurdlePoisson, predictor, leaf[:scale])
    elseif response.family === :wald
        # Twin head (thin-layer decision, pair fam-inversegaussian):
        # the plan's `InverseGaussian(mu, lam)` maps to
        # `InverseGaussian.(exp.(eta), lam)` (NB2 precedent); scalars
        # inline bare. No fused head: one spelling either way.
        wrap_location ? _rk_ast_dotted(:InverseGaussian,
            _rk_ast_dotted(:exp, predictor),
            leaf[:scale]) :
        _rk_ast_dotted(:InverseGaussian, predictor, leaf[:scale])
    elseif response.family === :negative_binomial
        # Twin head (thin-layer decision, pair fam-nb1): the plan's
        # `NegativeBinomial(r, p)` maps to
        # `NegativeBinomial.(exp.(eta), p)` (NB2 precedent); the
        # modeled-p submodel rides the scale slot under `logistic.`
        # (pair nuisance-nb1p), scalars inline bare. No fused head:
        # one spelling either way.
        wrap_location ? _rk_ast_dotted(:NegativeBinomial,
            _rk_ast_dotted(:exp, predictor),
            leaf[:scale]) :
        _rk_ast_dotted(:NegativeBinomial, predictor, leaf[:scale])
    elseif response.family === :weibull
        # Twin head (thin-layer decision, pair fam-weibull): the
        # plan's `Weibull(k, theta)` maps to
        # `Weibull.(k, exp.(eta))` (Distributions `(shape, scale)`
        # order, NB2 precedent); scalars inline bare. No fused
        # head: one spelling either way.
        wrap_location ? _rk_ast_dotted(:Weibull,
            leaf[:scale],
            _rk_ast_dotted(:exp, predictor)) :
        _rk_ast_dotted(:Weibull, leaf[:scale], predictor)
    elseif response.family === :exponential_log
        # Twin head (thin-layer decision, pair fam-exp): the plan's
        # `Exponential(mu)` maps to `Exponential.(exp.(eta))`
        # (Poisson-shaped single-arg twin); no scale slot. No fused
        # head: one spelling either way.
        wrap_location ? _rk_ast_dotted(:Exponential,
            _rk_ast_dotted(:exp, predictor)) :
        _rk_ast_dotted(:Exponential, predictor)
    elseif response.family === :gamma_log
        # Mean-shape form: the plan pins both alpha positions identical,
        # so the same value emits twice.
        shape = leaf[:scale]
        loc = wrap_location ? _rk_ast_dotted(:exp, predictor) : predictor
        fused_heads && wrap_location ?
            _rk_ast_dotted(:GammaLog, shape, predictor) :
            _rk_ast_dotted(:Gamma, shape, Expr(:call, :./, loc, shape))
    elseif response.family === :student_t
        # Dedicated single head (thin-layer decision, pair fam-student):
        # the plan's `LocationScale(mu, s, TDist(nu))` maps to
        # `StudentT.(nu, mu, sigma)` by arg reorder (Stan
        # `student_t(nu, mu, sigma)` order), the same class of
        # normalization as the existing spelling maps. A modeled nu
        # rides under `exp.` (the scale-predictor precedent). No
        # `LocationScale` twin: the Normal single-head precedent
        # governs (no link wrap to bridge).
        _rk_ast_dotted(:StudentT, leaf[:nu], predictor, leaf[:scale])
    elseif response.family === :zero_inflated_poisson
        # Dedicated single head (thin-layer decision, pair fam-zip):
        # the plan's `ZeroInflatedPoisson(lambda, zi)` maps to
        # `ZeroInflatedPoisson.(exp.(lambda), zi)` (Julia/Stan
        # `(lambda, zi)` order); a modeled `logit(zi)` submodel rides
        # the scale slot under `logistic.` instead (the hurdle hu
        # precedent). No fused head and no decomposed twin: one
        # spelling either way, so the fused flag changes nothing.
        zi = response.scale_predictor === nothing ?
            leaf[:zero_inflation] : leaf[:scale]
        _rk_ast_dotted(:ZeroInflatedPoisson,
            _rk_ast_dotted(:exp, predictor), zi)
    elseif response.family === :von_mises
        # Twin heads (thin-layer decision, pair fam-vonmises): exact
        # `VonMises(mu, kappa)` maps to `VonMises.(mu, kappa)` and
        # `CircularVonMises` appends the literal principal interval
        # (Distributions `(mu, kappa)` order + `(lo, hi)`). kappa
        # rides the scale slot (scalars inline bare, the `log(kappa)`
        # submodel under `exp.`). No fused head: one spelling either
        # way.
        interval = response.interval
        interval === nothing ?
            _rk_ast_dotted(:VonMises, predictor, leaf[:scale]) :
            _rk_ast_dotted(:CircularVonMises, predictor, leaf[:scale],
                interval[1], interval[2])
    elseif response.family === :lognormal
        # Single head (thin-layer decision, pair fam-lognormal): the
        # plan's `LogNormal(mu, sigma)` maps to
        # `LogNormal.(mu, sigma)` (Distributions `(mu, sigma)`
        # order); sigma rides the scalar-only scale slot. No fused
        # head: one spelling either way.
        _rk_ast_dotted(:LogNormal, predictor, leaf[:scale])
    elseif response.family === :categorical_logit
        # Reference-coded: K−1 non-reference etas, class 1 the implicit
        # zero reference (class order follows predictor order).
        _rk_ast_dotted(:CategoricalLogit, predictor,
            leaf[:extra_predictors]...)
    elseif response.family === :ordered_logit
        # The cutpoint statement owns its ordered coordinates.
        _rk_ast_dotted(:OrderedLogistic, predictor, Expr(:call, :Ref, response.thresholds))
    elseif response.family === :ordinal
        # Thresholds, discrimination and category-specific effects are values
        # declared in this same source block.
        structure = response.ordinal_structure === :cumulative ?
            :Cumulative : :StoppingRatio
        linktag = response.link === :logit ? :LogitLink :
            response.link === :probit ? :ProbitLink : :CloglogLink
        _rk_ast_dotted(:Ordinal, Expr(:call, structure),
            Expr(:call, linktag), predictor,
            Expr(:call, :Ref, response.thresholds), leaf[:discrimination],
            (haskey(leaf, :threshold_effects) ? (leaf[:threshold_effects],) : ())...)
    elseif response.family === :multinomial
        # Lead count column (LHS) + trials + simplex + tail count columns.
        _rk_ast_dotted(:Multinomial, leaf[:trials], Expr(:call, :Ref, predictor))
    elseif response.family === :categorical
        Expr(:call, :Categorical, predictor)
    elseif response.family === :mixture
        _rk_ast_mixture_dist(response, leaf)
    end
    evidence = response.evidence
    dist = _rk_ast_response_modifier(base, evidence.kind,
        get(leaf, :lower, -Inf), get(leaf, :upper, Inf))
    response.weights === nothing ? dist :
        _rk_ast_dotted(:weighted, dist, leaf[:weights])
end

function _rk_ast_response_modifier(base, kind, lower, upper)
    kind === :none && return base
    kind in (:truncated, :censored) && return _rk_ast_dotted(kind, base, lower, upper)
    kind === :interval_censored && return _rk_ast_dotted(kind, base, upper)
    error("RK backend: unsupported response evidence `$kind`")
end

# The class count behind a leveled response: the planned `n_levels`
# when set, else the tail/count arity implies it (K−2 tails, K−1 tail
# counts). A mismatch is an internal error.
function _rk_ast_response_levels(response::_RKLikelihoodSpec)
    family = response.family
    implied = family === :categorical_logit ?
        length(response.extra_predictors) + 2 :
        length(response.count_columns) + 1
    n = response.n_levels
    n === nothing && return implied
    n == implied || error(
        "RK backend: internal: response `$(response.response)` plans " *
        "$n levels but carries $implied")
    n
end

# One mixture component's inline spelling: a synthesized single-family
# spec (the dist branches read family/evidence/weights from the spec —
# evidence/weights are always none here — and every role from the leaf,
# so the spec's own predictor slot is unread) plus that leaf plus the
# wrap flag (predictor locations wrap, params/literals ride bare).
function _rk_ast_mixture_leaves(response::_RKLikelihoodSpec,
        rename::Dict{Symbol,Symbol}, predictor_link::Dict{Symbol,Symbol})
    map(response.mixture_components) do comp
        wrap = comp.location_kind === :predictor
        loc = wrap ? get(rename, comp.location::Symbol, comp.location) :
            comp.location
        cspec = _RKLikelihoodSpec(comp.family, comp.link, response.response,
            response.response, comp.scale, comp.scale_predictor, nothing,
            _RKResponseEvidence(:none, nothing, nothing), response.label,
            response.trials, nothing, nothing, Symbol[], Symbol[], nothing,
            nothing, Symbol[], nothing, Symbol[], nothing,
            _RKMixtureComponent[], nothing, nothing, nothing, nothing,
            nothing, nothing)
        cleaf = Dict{Symbol,Any}(:predictor => loc)
        if _rk_ast_response_uses_scale(comp.family)
            cleaf[:scale] =
                _rk_ast_response_scale(cspec, rename, predictor_link)
        end
        comp.family === :binomial_logit &&
            (cleaf[:trials] = response.trials)
        (cspec, cleaf, wrap)
    end
end

# A finite mixture over the planned components: each component emits
# its decomposed twin (fused heads desugar pre-spine on whole
# responses only — components always spell the wrapper form), with
# predictor locations link-wrapped and param/literal locations bare.
# Literal weights inline; a Dirichlet simplex name rides bare.
function _rk_ast_mixture_dist(response::_RKLikelihoodSpec,
        leaf::Dict{Symbol,Any})
    comp_exprs = map(leaf[:mixture]) do (cspec, cleaf, wrap)
        _rk_ast_response_dist(cspec, cleaf, false, wrap)
    end
    weights = response.mixture_weights
    weights_expr = weights isa Vector ? Expr(:vect, weights...) :
        weights isa Symbol ? weights :
        error("RK backend: internal: response `$(response.response)` " *
              "mixture has no planned weights")
    _rk_ast_dotted(:MixtureModel, _rk_ast_dotted(:vcat, comp_exprs...),
        Expr(:call, :Ref, weights_expr))
end

# Build the bare response statement for a response: `resp .~ dist`
# with every role spelled inline (predictor and scale-predictor names
# renamed like any other use-site). This is exactly the statement a
# one-shot stream submodel would expand to (the old `*_glm` shells took
# a precomputed predictor while borrowing Stan/SB's fused
# design-matrix-plus-coefficients name, so they were renamed out of
# existence); emitting it directly keeps the lowered program identical
# while the surface stays honest. Missing evidence sides pass ∓Inf
# floats; the thin layer normalizes them back to nothing at bind.
function _rk_ast_response_stmt(response::_RKLikelihoodSpec,
        rename::Dict{Symbol,Symbol}, predictor_link::Dict{Symbol,Symbol},
        fused_heads::Bool, row_names=Set{Symbol}(), threshold_effects_name=nothing)
    family = response.family
    leaf = Dict{Symbol,Any}(
        :predictor => get(rename, response.predictor, response.predictor))
    if _rk_ast_response_uses_scale(family)
        leaf[:scale] =
            _rk_ast_response_scale(response, rename, predictor_link)
    end
    if family === :student_t
        # Scalar or modeled (sampled/assignment names pass through;
        # only predictor names alpha-rename and invert their link).
        response.nu === nothing && response.nu_predictor === nothing &&
            error("RK backend: internal: response `$(response.response)` " *
                  "plans Student-t without degrees of freedom")
        leaf[:nu] = _rk_ast_response_nu(response, rename, predictor_link)
    end
    if family === :zero_inflated_poisson
        # Scalar-only (sampled/assignment names pass through; only
        # predictor names alpha-rename) — except a modeled-zi
        # submodel, which rides the scale slot (`leaf[:scale]`)
        # instead and leaves this `nothing`.
        response.zero_inflation === nothing &&
            response.scale_predictor === nothing && error(
            "RK backend: internal: response `$(response.response)` plans " *
            "zero-inflated Poisson without a zero probability")
        leaf[:zero_inflation] = response.zero_inflation
    end
    if family === :binomial_logit || family === :binomial_probit ||
            family === :binomial_cloglog || family === :beta_binomial_logit ||
            family === :multinomial
        leaf[:trials] = response.trials
    end
    if response.weights !== nothing
        leaf[:weights] = response.weights
    end
    kind = response.evidence.kind
    if kind === :truncated || kind === :censored
        lower = response.evidence.lower
        upper = response.evidence.upper
        leaf[:lower] = lower === nothing ? -Inf : lower
        leaf[:upper] = upper === nothing ? Inf : upper
    elseif kind === :interval_censored
        leaf[:upper] = response.evidence.upper
    end
    if family === :categorical_logit
        _rk_ast_response_levels(response)
        leaf[:extra_predictors] =
            Any[get(rename, p, p) for p in response.extra_predictors]
    elseif family === :multinomial
        _rk_ast_response_levels(response)
        leaf[:count_columns] = response.count_columns
    elseif family === :mixture
        leaf[:mixture] =
            _rk_ast_mixture_leaves(response, rename, predictor_link)
    end
    if family === :ordinal
        d = response.discrimination
        leaf[:discrimination] = d === nothing ? 1.0 :
            d isa Symbol && haskey(predictor_link, d) ?
            _rk_ast_dotted(:exp, get(rename, d, d)) : d
        if response.threshold_coefs !== nothing
            leaf[:threshold_effects] = Expr(:call, :eachrow, threshold_effects_name)
        end
    end
    lhs = response.response
    if family === :multinomial
        lhs = Expr(:call, :eachrow, Expr(:call, :hcat, lhs, response.count_columns...))
    end
    if response.mi_jobs !== nothing
        jobs = response.mi_jobs
        for (role, value) in leaf
            leaf[role] = _rk_ast_observed_slice(value, jobs, row_names)
        end
    end
    Expr(:call, :.~, lhs,
        _rk_ast_response_dist(response, leaf, fused_heads))
end

# Count columns are graph values, so a row likelihood cannot require them to
# arrive as precomputed bound data. A numerical plate computes normalized
# pointwise scores; the ordinary scoring adapter reads those scores unchanged.
function _rk_ast_multinomial_response_stmt!(definitions, statements, taken,
        response, rename)
    response.evidence.kind === :none || error(
        "RK backend: multinomial count rows do not support response evidence modifiers")
    columns = Expr(:tuple, response.response, response.count_columns...)
    probabilities = get(rename, response.predictor, response.predictor)
    scores = _rk_ast_fresh_name(string(response.response, "_multinomial_scores"), taken)
    cell = first(_rk_ast_statistical_call!(definitions, taken, :brm_multinomial_cell).args)
    expression = _rk_ast_statistical_call!(definitions, taken, :brm_multinomial_scores,
        cell, columns, response.trials, probabilities; kernel=true)
    response.weights === nothing || (expression = Expr(:call, :.*, expression, response.weights))
    push!(statements, Expr(:(=), scores, expression))
    reader = _rk_ast_fresh_name("brm_logdensity_value", taken)
    push!(definitions, :(function $reader(observed, logdensity)
        return logdensity
    end))
    Expr(:call, :.~, response.response, _rk_ast_dotted(:LogDensity, reader, scores))
end

_rk_ast_observed_slice(value, jobs, rows) = value
_rk_ast_observed_slice(value::Symbol, jobs, rows) =
    value in rows ? Expr(:ref, value, jobs) : value
_rk_ast_observed_slice(value::Expr, jobs, rows) =
    Expr(value.head, (_rk_ast_observed_slice(arg, jobs, rows) for arg in value.args)...)

# A monotonic component is named after its increments vector, as SBBRMI
# names `mo_<c>` (first occurrence) and its repeats.
_rk_ast_monotonic_name(term, taken) = _rk_ast_fresh_name(
    replace(string(term.options.increments), "_simplex_incr" => ""), taken)

# Varying-name pre-pass (uniform split form): one `b_<suffix>` draws
# component per bucket (SBBRMI's name) plus one `ranef_<target>_<suffix>`
# per slice. The `ranef_` prefix cannot collide with the thin layer's
# implicit `r_<target>_<group>` term labels; `taken` dedups the rest.
# Rename-independent (targets use original predictor names), so names
# pre-mint before predictor emission.
function _rk_ast_ranef_names!(
        plan::_RKStructuralPlan, taken::Set{Symbol})
    draws = Dict{Int,Symbol}()
    effects = Dict{Tuple{Symbol,Symbol,Union{Symbol,Nothing}},Symbol}()
    for (bi, bucket) in enumerate(plan.ranef_buckets)
        suffix = bucket.id === nothing ? string(bucket.group) :
            string(bucket.id, "_", bucket.group)
        draws[bi] = _rk_ast_coef_name("b_" * suffix, taken)
        for (target, _) in bucket.slices
            effect = _rk_ast_coef_name(
                "ranef_" * string(target) * "_" * suffix, taken)
            effects[(target, bucket.group, bucket.id)] = effect
        end
    end
    draws, effects
end

function _rk_ast_sampled(parameter::_RKSampledParameter)
    name = parameter.name
    family, override = parameter.family, parameter.support_override
    if override === :positive
        head = family === :Cauchy ? :HalfCauchy : :HalfNormal
        return Expr(:call, :~, name,
            Expr(:call, head, parameter.args[2]))
    end
    if override === :interval
        # Unit-interval truncated-Normal (dar persistence): the thin-layer
        # screen takes `truncated(Normal(mu, s), 0, 1)` exactly.
        return Expr(:call, :~, name,
            Expr(:call, :truncated,
                Expr(:call, :Normal,
                    parameter.args[1], parameter.args[2]),
                0, 1))
    end
    if override isa Tuple && first(override) in (:truncated, :restricted)
        # Explicit normalized truncation and declaration-only support bounds
        # use different ordinary source forms, preserving absolute densities.
        wrapper, lower, upper = override
        return Expr(:call, :~, name,
            Expr(:call, wrapper,
                Expr(:call, family, parameter.args...),
                lower, upper))
    end
    family === :Flat && return Expr(:call, :~, name, Expr(:call, :Flat))
    family === :LKJCovarianceFactor && error(
        "RK backend: covariance factors require explicit scale and LKJ statements")
    Expr(:call, :~, name, Expr(:call, family, parameter.args...))
end

# Each constrained or unconstrained vector carries its own explicit prior.
function _rk_ast_vector_parameter(parameter::_RKVectorParameter)
    if parameter.family === :ordered_normal
        return Expr(:call, :~, parameter.name, Expr(:call, :Ordered,
            Expr(:call, :Normal, parameter.args...), parameter.size))
    elseif parameter.family === :vector_normal
        return Expr(:call, :.~, Expr(:ref, parameter.name,
            Expr(:call, :(:), 1, parameter.size)), _rk_ast_dotted(:Normal, parameter.args...))
    end
    parameter.family === :simplex_dirichlet || return nothing
    alpha = only(parameter.args)
    Expr(:call, :~, parameter.name,
        Expr(:call, :Dirichlet, Expr(:vect, alpha...)))
end

# A `@plate` block: `name[i] ~ Normal(loc, scale)` over the range
# column's index (length `n_obs`, like every column). GP latents take
# the standardized default; `me` latents take the shared-scalar args.
# The macrocall carries a synthetic line node; the surface reads only
# `args[3]`.
function _rk_ast_plate(name::Symbol, range::Symbol,
        loc::Float64=0.0, scale::Float64=1.0)
    cell = Expr(:call, :~,
        Expr(:ref, name, :i), Expr(:call, :Normal, loc, scale))
    loop = Expr(:for, Expr(:(=), :i, Expr(:call, :eachindex, range)),
        Expr(:block, cell))
    Expr(:macrocall, Symbol("@plate"), LineNumberNode(0), loop)
end

function _rk_ast_gp_pair_calls(node, callee)
    node isa Expr || return node
    args = map(arg -> _rk_ast_gp_pair_calls(arg, callee), node.args)
    Meta.isexpr(node, :call) && first(args) === :gp_pair_locations &&
        (args[1] = callee)
    Expr(node.head, args...)
end

function _rk_ast_gp_covariance!(definitions, taken, x, sigma, rho, period, jitter;
        periodic=false)
    templates = StatisticalPreparation._GP_COVARIANCE_MODELS
    pair = _rk_ast_statistical_call!(definitions, taken,
        :brm_gp_pair_locations, :x; kernel=true,
        template=templates.gp_pair_locations)
    template = periodic ? templates.gp_periodic_cov : templates.gp_exp_quad_cov
    template = _rk_ast_gp_pair_calls(template, first(pair.args))
    args = periodic ? (x, sigma, rho, period, jitter) : (x, sigma, rho, jitter)
    _rk_ast_statistical_call!(definitions, taken,
        periodic ? :brm_gp_periodic_cov : :brm_gp_exp_quad_cov, args...;
        kernel=true, template)
end

# An exact GP, as StanBlocks' `_sb_gp`/`_sb_gp_periodic`: the length-scale
# and marginal-scale priors with standardized innovations, returning the
# Cholesky-scaled latent draw at each location. The covariance arguments are
# (locations, sigma, rho, period, jitter); `period` is 0 for exp_quad.
function _rk_ast_gp_block!(definitions, taken, term)
    options = term.options
    rho_prior = last(_rk_ast_sampled(options.rho_param).args)
    sigma_prior = last(_rk_ast_sampled(options.sigma_param).args)
    block = _rk_block_body(rho_prior, sigma_prior)
    x = _rk_block_argument!(block, :x, only(term.columns))
    rho, sigma = _rk_block_local!(block, :rho), _rk_block_local!(block, :sigma)
    axis, z = _rk_block_local!(block, :axis), _rk_block_local!(block, :z)
    push!(block.statements, Expr(:call, :~, rho, rho_prior),
        Expr(:call, :~, sigma, sigma_prior),
        Expr(:(=), axis, Expr(:call, :eachindex, x)),
        Expr(:call, :.~, Expr(:ref, z, axis), _rk_ast_dotted(:Normal, 0, 1)))
    covariance = _rk_ast_gp_covariance!(definitions, taken, x, sigma, rho,
        options.cov === :periodic ? options.period : 0.0, options.jitter; periodic=options.cov === :periodic)
    _rk_ast_block_call!(definitions, taken, "brm_gp", block,
        Expr(:call, :brm_gp_latent, covariance, z))
end

function _rk_ast_gp_names(plan::_RKStructuralPlan)
    names = Set{Symbol}()
    for predictor in plan.predictors, term in predictor.terms
        term.kind === :gp || continue
        options = term.options
        push!(names, options.rho, options.sigma, options.z, options.f)
    end
    names
end

# An AR(1) latent path, as StanBlocks' `_sb_ar1`: the sampled
# `phi_raw ~ Normal(0, 1)`, the non-centered `@scan` recurrence and the
# `phi = tanh(phi_raw)` stationarity map, returning the path. The loop bound
# is the authored term's explicit data-axis length; the seed + innovation
# shape is SB's `ar1_recurse` verbatim (`u[1] = eps[1]`,
# `u[t] = phi*u[t-1] + eps[t]`).
function _rk_ast_ar_block!(definitions, taken, nsteps)
    block = _rk_block_body()
    T = _rk_block_argument!(block, :T, nsteps)
    phi_raw, phi = _rk_block_local!(block, :phi_raw), _rk_block_local!(block, :phi)
    state, eps = _rk_block_local!(block, :state), _rk_block_local!(block, :eps)
    t = _rk_block_local!(block, :t)
    carry = Expr(:(=), Expr(:ref, state, t), Expr(:call, :+,
        Expr(:call, :*, phi, Expr(:ref, state, Expr(:call, :-, t, 1))), eps))
    loop = Expr(:for, Expr(:(=), t, Expr(:call, :(:), 2, T)),
        Expr(:block, Expr(:call, :~, eps, Expr(:call, :Normal, 0.0, 1.0)), carry))
    push!(block.statements, Expr(:call, :~, phi_raw, Expr(:call, :Normal, 0.0, 1.0)),
        Expr(:macrocall, Symbol("@scan"), LineNumberNode(0), Expr(:block,
            Expr(:call, :~, Expr(:ref, state, 1), Expr(:call, :Normal, 0.0, 1.0)), loop)),
        Expr(:(=), phi, Expr(:call, :tanh, phi_raw)))
    _rk_ast_block_call!(definitions, taken, "brm_ar1", block, state)
end

function _rk_ast_ar_names(plan::_RKStructuralPlan)
    names = Set{Symbol}()
    for predictor in plan.predictors, term in predictor.terms
        term.kind === :ar || continue
        options = term.options
        push!(names, options.state, options.phi, options.phi_raw,
            options.eps)
    end
    names
end

# A differenced AR(1) path, as StanBlocks' `_sb_dar1`: persistence and scale
# priors with the zero-started recurrence over standardized innovations,
# returning the level path.
function _rk_ast_dar_block!(definitions, taken, term, nsteps)
    options = term.options
    beta_prior = last(_rk_ast_sampled(options.beta_param).args)
    sigma_prior = last(_rk_ast_sampled(options.sigma_param).args)
    block = _rk_block_body(beta_prior, sigma_prior)
    T = _rk_block_argument!(block, :T, nsteps)
    beta, sigma = _rk_block_local!(block, :beta), _rk_block_local!(block, :sigma)
    level, increment = _rk_block_local!(block, :level), _rk_block_local!(block, :increment)
    innovation, t = _rk_block_local!(block, :innovation), _rk_block_local!(block, :t)
    previous = Expr(:call, :-, t, 1)
    body = Expr(:block,
        Expr(:call, :~, innovation, Expr(:call, :Normal, 0, 1)),
        Expr(:(=), Expr(:ref, increment, t), Expr(:call, :+,
            Expr(:call, :*, beta, Expr(:ref, increment, previous)),
            Expr(:call, :*, sigma, innovation))),
        Expr(:(=), Expr(:ref, level, t), Expr(:call, :+,
            Expr(:ref, level, previous), Expr(:ref, increment, t))))
    push!(block.statements, Expr(:call, :~, beta, beta_prior),
        Expr(:call, :~, sigma, sigma_prior),
        Expr(:macrocall, Symbol("@scan"), LineNumberNode(0), Expr(:block,
            Expr(:(=), Expr(:ref, level, 1), 0.0),
            Expr(:(=), Expr(:ref, increment, 1), 0.0),
            Expr(:for, Expr(:(=), t, Expr(:call, :(:), 2, T)), body))))
    _rk_ast_block_call!(definitions, taken, "brm_differenced_ar1", block, level)
end

function _rk_ast_dar_names(plan::_RKStructuralPlan)
    names = Set{Symbol}()
    for predictor in plan.predictors, term in predictor.terms
        term.kind === :dar || continue
        options = term.options
        push!(names, options.beta, options.sigma)
    end
    names
end

# A joint correlated-outcomes response emits the plain-`~` vector form
# directly in main (row-grouped, never broadcast — no stream-submodel
# def): `[y1, y2] ~ MvNormalCholesky([mu1, mu2], L)`.
function _rk_ast_joint_response(response::_RKLikelihoodSpec,
        rename::Dict{Symbol,Symbol})
    outcomes = [response.response; response.extra_responses...]
    means = [get(rename, response.predictor, response.predictor);
        [get(rename, p, p) for p in response.extra_predictors]...]
    stem = response.factor
    stem === nothing && error(
        "RK backend: internal: joint response `$(response.label)` has " *
        "no factor stem")
    length(outcomes) == length(means) || error(
        "RK backend: internal: joint response `$(response.label)` has " *
        "$(length(outcomes)) outcomes but $(length(means)) means")
    Expr(:call, :~, Expr(:vect, outcomes...),
        Expr(:call, :MvNormalCholesky, Expr(:vect, means...), stem))
end

function _rk_ast_me_names(plan::_RKStructuralPlan)
    names = Set{Symbol}()
    for predictor in plan.predictors, term in predictor.terms
        term.kind === :me || continue
        push!(names, term.options.latent)
    end
    names
end

# `coordinates`, when a vector, receives one semantic record per sampled
# declaration this emitter can address across backends
# (`_rk_coordinate_record!`, read by src/coordinate_transport.jl). Emission is
# otherwise unchanged; an unrecorded declaration is refused by the transport.
function _rk_emit_ast(plan::_RKStructuralPlan, fused_heads::Bool=true;
        values::Bool=false, reserved=(), coordinates=nothing)
    taken = union(Set(keys(plan.columns)),
        Set(p.name for p in plan.parameters),
        Set(a.name for a in plan.assignments),
        Set(p.name for p in plan.predictors),
        Set(d.name for d in plan.derived),
        Set(v.name for v in plan.vector_parameters),
        _rk_ast_spline_ids(plan),
        _rk_ast_hsgp_ids(plan),
        _rk_ast_gp_names(plan),
        _rk_ast_dar_names(plan),
        _rk_ast_ar_names(plan),
        _rk_ast_me_names(plan))
    union!(taken, (t.options.id for p in plan.predictors for t in p.terms
        if t.kind === :structured))
    union!(taken, reserved)
    # A predictor sharing its name with a data column cannot keep it:
    # the program has one namespace, so the affine (definition and
    # response uses) is alpha-renamed. Unreachable via `@brm`
    # (observation discovery claims `P ~ …` as a likelihood whenever `P`
    # is observed data), but programmatic plans can still overlap.
    rename = Dict{Symbol,Symbol}()
    for predictor in plan.predictors
        (haskey(plan.columns, predictor.name) ||
            (values && predictor.link !== :identity)) || continue
        fresh = Symbol(string(predictor.name), "_")
        while fresh in taken
            fresh = Symbol(string(fresh), "_")
        end
        push!(taken, fresh)
        rename[predictor.name] = fresh
    end
    priors = Dict{Tuple{Symbol,Symbol},Tuple{Symbol,Tuple}}(
        (p.predictor, p.addressee) => (p.family, p.args)
        for p in plan.population_priors)
    ranef_draws, ranef_effects = _rk_ast_ranef_names!(plan, taken)
    r2d2s = Dict(rp.predictor => rp for rp in plan.r2d2_priors)
    hs_priors = Dict((p.predictor, p.addressee) =>
        (p.local_scale, p.global_scale) for p in plan.horseshoe_priors)
    hs_predictors =
        Set{Symbol}(p.predictor for p in plan.horseshoe_priors)
    response_for = Dict{Symbol,Symbol}()
    for response in plan.responses
        haskey(response_for, response.predictor) ||
            (response_for[response.predictor] = response.response)
    end
    defs = Expr[]
    bindings = Pair{Symbol,Any}[]
    stmts = Expr[]
    structured_blocks = Dict{Tuple{Symbol,Symbol},Any}()
    index_sources = Dict{Symbol,Tuple{Symbol,Any}}()
    for predictor in plan.predictors, term in predictor.terms
        if haskey(term.options, :zero_source)
            push!(stmts, Expr(:(=), only(term.columns),
                Expr(:call, :zeros, Expr(:call, :length, term.options.zero_source))))
        elseif term.kind === :factor && haskey(term.options, :index)
            index_sources[term.options.index] = (only(term.columns), term.options.index_levels)
        elseif term.kind in (:monotonic, :monotonic_summand)
            index_sources[only(term.columns)] = (term.options.source, term.options.levels)
        elseif term.kind === :hsgp && haskey(term.options, :group_index)
            index_sources[term.options.group_index] =
                (term.options.group_source, term.options.group_levels)
        end
    end
    for name in sort!(collect(keys(index_sources)); by=string)
        source, levels = index_sources[name]
        call = _rk_ast_statistical_call!(defs, taken, :brm_prepared_indices,
            source, _rk_ast_level_values(levels); kernel=true)
        push!(stmts, Expr(:(=), name, call))
    end
    for derived in plan.derived
        expression = _rk_ast_data_expr!(defs, stmts, bindings, taken, derived.expression)
        push!(stmts, Expr(:(=), derived.name, expression))
    end
    # Emit shared budgets once, before their population and group-level
    # consumers. All derived scales remain ordinary graph values.
    joint_priors = Dict{Tuple{Symbol,Symbol},Tuple{Symbol,Tuple}}()
    for (bi, bucket) in enumerate(plan.ranef_buckets)
        bucket.decomposition === nothing && continue
        append!(stmts, _rk_ast_value_bucket(defs, bucket, ranef_draws[bi],
            ranef_effects, taken, bindings; predictors=plan.predictors,
            population_priors=joint_priors, coordinates))
    end
    for (bi, bucket) in enumerate(plan.ranef_buckets)
        bucket.decomposition === nothing || continue
        append!(stmts, _rk_ast_value_bucket(defs, bucket, ranef_draws[bi],
            ranef_effects, taken, bindings; coordinates))
    end
    vector_priors = Dict(v.name => v for v in plan.vector_parameters)
    owned_vectors = Set{Symbol}()
    for predictor in plan.predictors
        lhs = get(rename, predictor.name, predictor.name)
        r2d2 = get(r2d2s, predictor.name, nothing)
        hs_tau = nothing
        hs_scale = 1.0
        if predictor.name in hs_predictors
            scales = [p.global_scale for p in plan.horseshoe_priors
                if p.predictor === predictor.name]
            hs_scale = all(==(first(scales)), scales) ? first(scales) : 1.0
            hs_tau = _rk_ast_fresh_name(string(predictor.name, "_tau"), taken)
            push!(stmts, Expr(:call, :~, hs_tau, Expr(:call, :HalfCauchy, hs_scale)))
        end
        coefs = Dict{Int,Symbol}()
        colactual = Dict{Int,Any}()
        refactual = Dict{Int,Any}()
        replacements = Dict{Int,Any}()
        population = Tuple{Int,Any,Tuple{Symbol,Tuple}}[]
        scalar_stmts = Expr[]
        monotonic_alpha(term) = only(vector_priors[term.options.increments].args)
        # An ordinary scalar coefficient under its resolved prior.
        function scalar_coefficient!(index, term, override)
            coef = _rk_ast_coef_name(
                string(predictor.name, "_", term.addressee), taken)
            coefs[index] = coef
            r2d2 === nothing &&
                !haskey(joint_priors, (predictor.name, term.addressee)) &&
                _rk_coordinate_record!(coordinates, (; kind=:population,
                    declaration=coef, predictor=predictor.name,
                    coefficient=term.addressee))
            Expr(:call, :~, coef, Expr(:call, override[1], override[2]...))
        end
        for (index, term) in enumerate(predictor.terms)
            kind = term.kind
            if kind in (:continuous, :factor, :monotonic, :monotonic_summand, :offset, :ar, :me)
                colactual[index] = only(term.columns)
            end
            if kind === :monotonic_summand
                refactual[index] = _rk_ast_monotonic_component!(defs, stmts, taken,
                    _rk_ast_monotonic_name(term, taken), only(term.columns),
                    monotonic_alpha(term), nothing)
                push!(owned_vectors, term.options.increments)
                _rk_coordinate_record!(coordinates, (; kind=:monotonic,
                    declaration=refactual[index], predictor=predictor.name,
                    head=:mo1, source=term.options.source, beta=false))
            elseif kind === :spline || kind === :hsgp
                refactual[index] = term.options.id
            elseif kind === :structured
                refactual[index] = term.options.id
                append!(stmts, _rk_ast_structured_term(defs, term, structured_blocks,
                    taken, bindings))
            elseif kind === :gp
                refactual[index] = term.options.f
            elseif kind === :ar
                refactual[index] = term.options.state
            elseif kind === :me
                refactual[index] = term.options.latent
            elseif kind === :dar
                refactual[index] = _rk_ast_fresh_name(string(term.label, "_level"), taken)
                push!(stmts, Expr(:call, :~, refactual[index], _rk_ast_dar_block!(defs,
                    taken, term, length(plan.columns[term.options.source]))))
            elseif kind === :ranef_gather
                refactual[index] = ranef_effects[(predictor.name,
                    term.options.bucket_group, term.options.bucket_id)]
            end
            (kind === :offset || kind === :ranef_gather ||
                kind === :spline || kind === :hsgp ||
                kind === :structured ||
                kind === :gp || kind === :dar ||
                kind === :monotonic_summand) && continue
            hs_spec = get(hs_priors, (predictor.name, term.addressee),
                nothing)
            override = if hs_spec !== nothing
                nothing
            elseif haskey(joint_priors, (predictor.name, term.addressee))
                joint_priors[(predictor.name, term.addressee)]
            elseif r2d2 === nothing
                key = (predictor.name, term.addressee)
                haskey(priors, key) || error(
                    "RK backend: internal: no population prior for " *
                    "`$(predictor.name)` addressee `$(term.addressee)`")
                priors[key]
            else
                # R2D2 overrides are Normal-only by construction
                # (planner gate); normalize to the family shape here.
                r2 = get(r2d2.overrides, term.addressee, nothing)
                r2 === nothing ? (kind === :intercept ? (:Normal, (0.0, 1.0)) :
                    (:Normal, (0.0, _rk_ast_r2d2_scale(r2d2, term.addressee;
                        scalar=kind !== :factor,
                        variance_values=kind === :factor ?
                            [Expr(:call, :var, _rk_ast_factor_indicator!(defs, taken, term, j))
                                for j in eachindex(term.options.labels)] :
                            [Expr(:call, :var, colactual[index])])))) : (:Normal, r2)
            end
            if kind === :factor
                col = only(term.columns)
                K = length(_rk_grouping_levels(plan.columns[col]))
                coef = _rk_ast_coef_name(
                    string(predictor.name, "_", term.addressee), taken)
                refactual[index] = coef
                override === nothing || push!(stmts,
                    _rk_ast_factor_prior(coef, col, term.options, K,
                        override[1], override[2]))
                ordinary = override !== nothing && r2d2 === nothing &&
                    !haskey(joint_priors, (predictor.name, term.addressee))
                ordinary && haskey(term.options, :level_values) &&
                    _rk_coordinate_record!(coordinates, (; kind=:population_block,
                        declaration=coef, predictor=predictor.name,
                        coefficient=term.addressee, labels=term.options.labels,
                        level_values=term.options.level_values,
                        coding=term.options.coding))
            elseif kind === :monotonic && hs_spec === nothing && override !== nothing
                # The monotonic component owns its simplex and coefficient.
                replacements[index] = _rk_ast_monotonic_component!(defs, stmts, taken,
                    _rk_ast_monotonic_name(term, taken), only(term.columns),
                    monotonic_alpha(term), override)
                push!(owned_vectors, term.options.increments)
                # Its coefficient pairs like a population coefficient only
                # under an ordinary prior; a shared budget leaves it unpaired.
                _rk_coordinate_record!(coordinates, (; kind=:monotonic,
                    declaration=replacements[index], predictor=predictor.name,
                    head=:mo, source=term.options.source,
                    beta=r2d2 === nothing &&
                        !haskey(joint_priors, (predictor.name, term.addressee))))
            elseif hs_spec === nothing && _rk_ast_population_family(override) &&
                    (kind === :intercept ? predictor.row_source !== nothing :
                        kind === :continuous)
                column = kind === :intercept ?
                    Expr(:call, :ones, Expr(:call, :length, predictor.row_source)) :
                    colactual[index]
                push!(population, (index, column, override))
            else
                if kind === :monotonic
                    # A horseshoe coefficient scales the owned contrast.
                    refactual[index] = _rk_ast_monotonic_component!(defs, stmts, taken,
                        _rk_ast_monotonic_name(term, taken), only(term.columns),
                        monotonic_alpha(term), nothing)
                    push!(owned_vectors, term.options.increments)
                    _rk_coordinate_record!(coordinates, (; kind=:monotonic,
                        declaration=refactual[index], predictor=predictor.name,
                        head=:mo, source=term.options.source, beta=false))
                end
                if hs_spec !== nothing
                    coef = _rk_ast_coef_name(
                        string(predictor.name, "_", term.addressee), taken)
                    coefs[index] = coef
                    push!(scalar_stmts, Expr(:call, :~, coef, _rk_ast_horseshoe_block!(
                        defs, taken, hs_spec[1], hs_tau, hs_spec[2] / hs_scale)))
                else
                    push!(scalar_stmts, scalar_coefficient!(index, term, override))
                end
            end
        end
        # A lone intercept is its scalar coefficient, not a one-column design
        # product.
        if length(population) == 1 &&
                predictor.terms[first(only(population))].kind === :intercept
            index, _, override = only(population)
            pushfirst!(scalar_stmts,
                scalar_coefficient!(index, predictor.terms[index], override))
            empty!(population)
        end
        for term in predictor.terms
            term.kind === :spline || continue
            append!(stmts, _rk_ast_value_spline(defs, term, taken))
        end
        for term in predictor.terms
            term.kind === :hsgp || continue
            append!(stmts, _rk_ast_value_hsgp(defs, term, taken, bindings))
            # Ungrouped HSGP components own scalar hyperparameters and basis
            # weights; grouped or hyper-predicted terms stay unrecorded.
            (haskey(term.options, :group_index) ||
                !isempty(get(term.options, :hyper_plans, ()))) && continue
            _rk_coordinate_record!(coordinates, (; kind=:hsgp,
                declaration=term.options.id, predictor=predictor.name,
                axes=Tuple(term.columns),
                iso=get(term.options, :cov, :exp_quad) === :periodic ||
                    term.options.iso))
        end
        for term in predictor.terms
            term.kind === :gp || continue
            isnothing(only(term.columns)) && error(
                "RK backend: internal: gp predictor `$(predictor.name)` " *
                "feeds no response")
            push!(stmts, Expr(:call, :~, term.options.f, _rk_ast_gp_block!(defs, taken, term)))
        end
        for term in predictor.terms
            term.kind === :ar || continue
            push!(stmts, Expr(:call, :~, term.options.state, _rk_ast_ar_block!(defs, taken,
                length(plan.columns[only(term.columns)]))))
        end
        for term in predictor.terms
            term.kind === :me || continue
            options = term.options
            push!(stmts, _rk_ast_plate(options.latent,
                only(term.columns), options.loc, options.scale))
        end
        append!(stmts, scalar_stmts)
        # The population component's value takes the place of the first
        # coefficient it absorbs.
        if !isempty(population)
            # SBBRMI's names: `X_<target>` and `pop_<target>`, where a linked
            # predictor's target carries its link (`log_v`).
            target = predictor.link === :identity ? string(predictor.name) :
                string(predictor.link, "_", predictor.name)
            name = _rk_ast_population_component!(defs, stmts, taken,
                _rk_ast_fresh_name(string("pop_", target), taken),
                string("X_", target),
                [entry[2] for entry in population], [entry[3] for entry in population])
            # The component owns each coefficient. Record the actual sampled
            # declaration and its index rather than its old caller-side name.
            mixed = length(unique(first(entry[3]) for entry in population)) > 1
            for (j, entry) in enumerate(population)
                term = predictor.terms[first(entry)]
                r2d2 === nothing &&
                    !haskey(joint_priors, (predictor.name, term.addressee)) &&
                    _rk_coordinate_record!(coordinates, (; kind=:population,
                        declaration=Symbol(name, mixed ? ".beta_pop_$j" : ".beta_pop"),
                        index=mixed ? () : (j,), predictor=predictor.name,
                        coefficient=term.addressee))
            end
            replacements[first(first(population))] = name
            foreach(entry -> replacements[first(entry)] = nothing, population[2:end])
        end
        affine = _rk_ast_affine(predictor, coefs, colactual, refactual; values, replacements)
        push!(stmts, Expr(:(=), lhs, affine))
        if values && (lhs !== predictor.name || predictor.link !== :identity)
            value = _rk_value_link!(bindings, predictor.link, lhs, taken)
            push!(stmts, Expr(:(=), predictor.name, value))
        end
    end
    for parameter in plan.parameters
        if parameter.family === :LKJCovarianceFactor
            K, theta, eta = parameter.args
            scales = _rk_ast_fresh_name(string(parameter.name, "_scales"), taken)
            L = _rk_ast_fresh_name(string(parameter.name, "_L_corr"), taken)
            push!(stmts, Expr(:call, :.~, Expr(:ref, scales, Expr(:call, :(:), 1, K)),
                _rk_ast_dotted(:Exponential, theta)))
            push!(stmts, Expr(:call, :~, L, Expr(:call, :LKJCholesky, K, eta)))
            push!(stmts, Expr(:(=), parameter.name, Expr(:call, :.*, scales, L)))
        else
            push!(stmts, _rk_ast_sampled(parameter))
            _rk_coordinate_record!(coordinates,
                (; kind=:scalar, declaration=parameter.name))
        end
    end
    # Monotonic increment simplexes belong to their monotonic blocks.
    monotonic = Set(term.options.increments for predictor in plan.predictors
        for term in predictor.terms if term.kind in (:monotonic, :monotonic_summand))
    # R2D2 allocation simplexes belong to their (unpaired) shrinkage prior.
    r2d2_phis = Set(rp.phi for rp in plan.r2d2_priors)
    for vector_parameter in plan.vector_parameters
        vector_parameter.name in monotonic && continue
        response_index = findfirst(r -> r.threshold_coefs === vector_parameter.name, plan.responses)
        if response_index !== nothing
            response = plan.responses[response_index]
            push!(stmts, Expr(:call, :.~, Expr(:ref, vector_parameter.name,
                Expr(:call, :(:), 1, length(response.threshold_columns)),
                Expr(:call, :(:), 1, response.n_levels - 1)),
                _rk_ast_dotted(:Normal, vector_parameter.args...)))
            _rk_coordinate_record!(coordinates, (; kind=:threshold_coefficients,
                declaration=vector_parameter.name, response=response.response,
                terms=Tuple(response.threshold_columns), stages=response.n_levels - 1))
            continue
        end
        vector_parameter.name in owned_vectors && continue
        stmt = _rk_ast_vector_parameter(vector_parameter)
        stmt === nothing && continue
        push!(stmts, stmt)
        # Response cutpoints/thresholds and authored `Dirichlet` simplexes
        # keep their declaration name on both backends, like scalars.
        vector_parameter.name in r2d2_phis ||
            _rk_coordinate_record!(coordinates, (; kind=:vector,
                declaration=vector_parameter.name, family=vector_parameter.family))
    end
    for assignment in plan.assignments
        push!(stmts, Expr(:(=), assignment.name,
            _rk_lower_assignment_expr(assignment.expression, assignment.name)))
    end
    predictor_link = Dict(spec.name => spec.link for spec in plan.predictors)
    for response in plan.responses
        if response.family === :multinomial
            push!(stmts, _rk_ast_multinomial_response_stmt!(defs, stmts, taken, response, rename))
            continue
        end
        if response.family === :mvnormal_cholesky
            push!(stmts, _rk_ast_joint_response(response, rename))
            continue
        end
        effects_name = nothing
        if response.threshold_coefs !== nothing
            design_name = _rk_ast_fresh_name(string(response.response, "_threshold_X"), taken)
            push!(stmts, Expr(:(=), design_name,
                Expr(:call, :hcat, response.threshold_columns...)))
            effects_name = _rk_ast_fresh_name(string(response.response, "_threshold_effects"), taken)
            push!(stmts, Expr(:(=), effects_name, Expr(:call, :*,
                design_name, response.threshold_coefs)))
        end
        push!(stmts,
            _rk_ast_response_stmt(response, rename, predictor_link,
                fused_heads, union(Set(keys(plan.columns)), Set(Base.values(rename)),
                    Set(p.name for p in plan.predictors)), effects_name))
    end
    values || (stmts = _rk_source_data_axes(stmts, plan.columns))
    _rk_fitted_source(_rk_source_program(defs, Expr(:block, stmts...), bindings),
        _rk_observed_names(plan))
end
