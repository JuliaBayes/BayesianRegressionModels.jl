"""
    _rk_callable_source!(definitions, bindings, entry, f)

Source provider for an exact ordinary callable identity used by an emitted
expression, including calls nested inside an anonymous kernel. Return
`nothing` to keep the existing callable binding. To claim it, append an
ordinary Julia `function entry(...) ... end` or an explicit
`ReactiveKernels.@kernel entry(...) = begin ... end` compatible with the emitted
call's inputs, optionally append helper definitions and separately named
leaf bindings, and return a non-`nothing` value. The emitted entry replaces
the original binding; existing call sites retain their arguments. Graph
definitions may expose named plate/scan recipes; ordinary Julia functions do
not make their internal loops visible to RK. Domain source may choose a new
scientifically equivalent API rather than retain a historical argument count.
Provider graph definitions precede their emitted reader/cell graphs; place
helper graph definitions before entries that compose them.

`entry` is the callable's own name (`nameof`), made fresh with trailing
underscores only where the emitted program already reads that name or the
name has another fixed meaning in RK source: an export of Core, Base,
Distributions or LogExpFunctions, RKPPL's built-in vocabulary, or an import
of the emission module. Closures, anonymous functions and callable objects
receive the generic `brm_value_function`.

The provider receives no values, lowered plan or derivative. It supplies its
own native mathematics under the caller's source namespace. An entry cannot
also be bound, and rebinding the original callable as its own leaf is refused.
Extend with `import BayesianRegressionModels: _rk_callable_source!`.
"""
_rk_callable_source!(definitions, bindings, entry, f) = nothing

function _rk_resolve_callable_sources(emitted::_RKEmittedProgram)
    definitions = copy(emitted.defs)
    pending = copy(emitted.bindings)
    bindings = Pair{Symbol,Any}[]
    claimed = IdDict{Any,Symbol}()
    while !isempty(pending)
        entry, callable = popfirst!(pending)
        haskey(claimed, callable) && error(
            "RK source: provider for `$(claimed[callable])` rebound its original callable as `$entry`")
        provider_definitions = Expr[]
        provider_bindings = Pair{Symbol,Any}[]
        result = _rk_callable_source!(provider_definitions, provider_bindings, entry, callable)
        if result === nothing
            isempty(provider_definitions) && isempty(provider_bindings) || error(
                "RK source: unclaimed provider for `$entry` changed its source collections")
            push!(bindings, entry => callable)
            continue
        end
        names = Symbol[]
        for definition in provider_definitions
            source = _rk_source_definition(definition)
            source.kind in (:function, :kernel) || error(
                "RK source: callable provider for `$entry` needs ordinary Julia functions or explicit @kernel definitions")
            push!(names, source.name)
        end
        entry in names || error("RK source: callable provider did not define its entry `$entry`")
        claimed[callable] = entry
        prepend!(definitions, provider_definitions)
        append!(pending, provider_bindings)
    end
    resolved = _RKEmittedProgram(definitions, emitted.main, bindings)
    _rk_validate_source_definitions(resolved)
    resolved
end

_rk_source_program(definitions, main, bindings) =
    _rk_resolve_callable_sources(_rk_name_callable_bindings(
        _RKEmittedProgram(definitions, main, bindings)))

# An opaque callable is bound under a placeholder while the program is
# emitted; `#` keeps it out of every authored and generated name.
const _RK_CALLABLE_PLACEHOLDER = "#brm_callable"
_rk_is_callable_placeholder(name::Symbol) =
    startswith(string(name), _RK_CALLABLE_PLACEHOLDER)

# What the RK emission module imports beside `using ReactiveKernelsPPL`.
# Emitted source spells these names directly.
const _RK_SOURCE_IMPORTS = (
    ReactiveKernels=(Symbol("@kernel"), :plate),
    BayesianRegressionModels=(:brm_tps_basis, :brm_t2_basis, :brm_hsgp_basis,
        :brm_hsgp_periodic_basis, :brm_hsgp_sqrt_spd, :brm_hsgp_periodic_sqrt_spd,
        :brm_gp_covariance, :brm_gp_latent, :brm_level_indices, :brm_ranef_column,
        :brm_dummy, :brm_panel_slice, :brm_flatten_cells,
        :brm_invprobit, :brm_invcloglog))

# RKPPL vocabulary that Base, Core, Distributions and LogExpFunctions do not
# define. RKPPL reads these heads by name whatever the model module binds
# (`rkppl-use` §2 and §9).
const _RK_SOURCE_VOCABULARY = Set{Symbol}([:ReactiveKernelsPPL,
    :normcdf, :probit, :cloglog, :treatment, :levels, :weighted, :restricted,
    :interval_censored, :Ordered, :LogDensity, :Flat, :Ordinal, :Cumulative,
    :StoppingRatio, :LogitLink, :ProbitLink, :CloglogLink, :PoissonLog,
    :CategoricalLogit, :OrderedLogistic, :MvNormalCholesky, :NormalIDGLM,
    :ZeroInflatedPoisson, :ZeroInflatedBinomial])

# A named function keeps its own name. Closures, anonymous functions and
# callable objects have none that source could read.
function _rk_callable_own_name(@nospecialize(callable))
    callable isa Function && Base.issingletontype(typeof(callable)) || return nothing
    name = nameof(callable)
    Base.isidentifier(name) && !Base.isoperator(name) ? name : nothing
end

# A name with a fixed meaning in RK source names only that meaning:
# `logistic` names the inverse-logit link, never a caller's `logistic`.
function _rk_callable_name_free(name::Symbol, @nospecialize(callable))
    name in _RK_SOURCE_VOCABULARY && return false
    (name in keys(_RK_SOURCE_IMPORTS) ||
        any(names -> name in names, _RK_SOURCE_IMPORTS)) && return false
    for mod in (Core, Base, Distributions, LogExpFunctions)
        Base.isexported(mod, name) && return getfield(mod, name) === callable
    end
    true
end

# Each opaque callable takes its own name once the whole program exists, made
# fresh against every name the program reads; an anonymous callable takes a
# generic one. A source provider receives the chosen name as its entry.
function _rk_name_callable_bindings(emitted::_RKEmittedProgram)
    any(binding -> _rk_is_callable_placeholder(first(binding)), emitted.bindings) ||
        return emitted
    used = Set{Symbol}(first.(emitted.bindings))
    _rk_source_symbols!(used, emitted.main)
    _rk_source_symbols!(used, emitted.defs)
    renames = Dict{Symbol,Symbol}()
    for (placeholder, callable) in emitted.bindings
        _rk_is_callable_placeholder(placeholder) || continue
        name = something(_rk_callable_own_name(callable), :brm_value_function)
        while name in used || !_rk_callable_name_free(name, callable)
            name = Symbol(name, "_")
        end
        push!(used, name)
        renames[placeholder] = name
    end
    _RKEmittedProgram(
        Expr[_rk_rename_symbols(definition, renames) for definition in emitted.defs],
        _rk_rename_symbols(emitted.main, renames),
        Pair{Symbol,Any}[get(renames, name, name) => callable
            for (name, callable) in emitted.bindings])
end

_rk_rename_symbols(value, renames) = value
_rk_rename_symbols(name::Symbol, renames) = get(renames, name, name)
function _rk_rename_symbols(value::Expr, renames)
    args = Any[_rk_rename_symbols(arg, renames) for arg in value.args]
    all(i -> args[i] === value.args[i], eachindex(args)) ? value : Expr(value.head, args...)
end
