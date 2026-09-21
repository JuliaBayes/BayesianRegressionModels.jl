# descriptor_mount.jl — BRMDescriptor → HTMXObjects `semantic_app` mount.
#
# `brm_descriptor` is the producer half BRM owns (brm-use); this module is the
# app-side adapter decision `0lqlxi4` assigns to the consumer. It turns one
# descriptor's derived operations + schema into a mountable semantic graph:
# a generated operation picker, one execution route, and reflection routes,
# with stable framework-owned result targets.
#
# Derivation rules (nothing here is a second registry):
# - The operation picker domain is read LIVE from `d.operations` (`name` as
#   the machine value, `title` as the label), so derivation-exactness,
#   `operations=…` suppression/addition, and `titles=…` relabelling all hold
#   without touching this file.
# - Whether an operation needs posterior draws is derived from its outputs'
#   `generative` flags (`:draw` / `:pointwise_loglik`); whether it needs a
#   new dataframe is derived from `origin == :brm`. No operation names are
#   hardcoded anywhere in this file.
# - Unknown operations fail closed through `brm_operation`, which errors and
#   names the offered operations.
#
# Boundaries (deliberate, not gaps):
# - Draws-needing and dataframe-needing operations DECLINE with an explicit
#   `SemanticUnavailable` reason plus the Julia call that supplies the input.
#   Draws matrices and dataframes are not form-collectible route inputs, and
#   cross-operation state (fit→predict chaining) is out of contract on the
#   `semantic_app` path. Heavy execution with progress/polling/PPC stays in
#   BRMMacroWeb's pipeline stages.
# - Only no-input operations execute here (`:transpile`, `:fit`-family
#   compilation). `:fit` compiles on first run (minutes); `:auto` polling
#   covers the wait, and `instantiate` caches by `d.id`.
# - Override operations run with zero kwargs and render generically (semantic
#   nodes pass through, anything else as plain-text `show`). Overrides
#   needing form-collected kwargs stay Julia-side.
# - This module routes NOTHING on load. The page carries required descriptor
#   state, so serve it behind a retaining `RootProvider` factory (htmxo-use
#   §10.6) — the default fresh-per-request factory cannot build it. See
#   `../validate_descriptor_mount.jl` for the serving pattern.
module DescriptorMount

using BayesianRegressionModels: BRMDescriptor, BRMOperation, brm_operation, brm_execute
using DynamicObjects
using HTMXObjects
import HTMXObjects: h, Raw
using LogDensityProblems

export BRMDescriptorMount, BRMDescriptorMountPage, mount_descriptor_page
export mount_operation_domain, mount_needs_draws, mount_needs_dataframe
export mount_execute_operation, mount_render_result, mount_problem_summary
export mount_formula_node, mount_schema_node
export mount_inputs_node, mount_outputs_node, mount_operations_node

# ---- derivation: what an operation needs, read off the descriptor -----------

"""
    mount_operation_domain(d) -> OptionDomain

The executable picker domain: one entry per DERIVED operation, machine value
`op.name`, human label `op.title`. A bare mount (no descriptor) gets the
empty domain — an empty control, never a crash.
"""
mount_operation_domain(::Nothing) = option_domain(Symbol[])
mount_operation_domain(d::BRMDescriptor) =
    option_domain([op.name => op.title for op in d.operations])

_generative_by_name(d::BRMDescriptor) =
    Dict{Symbol,Symbol}(o.name => o.generative for o in d.outputs)

"""
    mount_needs_draws(d, op) -> Bool

True when executing `op` needs posterior draws: one of its outputs is a
predictive draw or pointwise-loglikelihood twin. Derived from the
descriptor's `generative` flags, not from the operation's name. Outputs the
descriptor does not know (an override's private names) default to not
needing draws — the override's own runner owns that claim.
"""
mount_needs_draws(d::BRMDescriptor, op::BRMOperation) = begin
    generative = _generative_by_name(d)
    any(op.outputs) do name
        get(generative, name, :posterior) in (:draw, :pointwise_loglik)
    end
end

"""
    mount_needs_dataframe(op) -> Bool

True when executing `op` needs a new dataframe: `:brm`-origin operations
take it positionally (brm-use). Stan-origin operations run on bound data.
"""
mount_needs_dataframe(op::BRMOperation) = op.origin === :brm

_require_descriptor(::Nothing) = throw(ArgumentError(
    "BRMDescriptorMount has no descriptor — build it with mount_descriptor_page(d)."))
_require_descriptor(d::BRMDescriptor) = d

# ---- execution --------------------------------------------------------------

function mount_execute_operation(::Nothing, ::Symbol)
    throw(ArgumentError(
        "BRMDescriptorMount has no descriptor — build it with mount_descriptor_page(d)."))
end

"""
    mount_execute_operation(d, opname) -> SemanticNode

Run one derived operation with no form-collected inputs. Draws-needing and
dataframe-needing operations decline with an explicit `SemanticUnavailable`
naming the Julia call that supplies the missing input; anything else runs
through `brm_execute` and renders by result shape.
"""
function mount_execute_operation(d::BRMDescriptor, opname::Symbol)
    op = brm_operation(d, opname)
    if mount_needs_dataframe(op)
        columns = join(string.(d.columns), ", ")
        return SemanticUnavailable(
            "`$opname` needs a new dataframe with columns $columns — " *
            "supply it in Julia via `brm_execute(d, :$opname, new_df)`; " *
            "this mount collects only the operation choice.")
    end
    if mount_needs_draws(d, op)
        outs = join(string.(op.outputs), ", ")
        return SemanticUnavailable(
            "`$opname` needs posterior draws (it produces $outs) — " *
            "fit first, then call `brm_execute(d, :$opname; draws, seed)` " *
            "in Julia; this mount holds no draws.")
    end
    return mount_render_result(d, op, brm_execute(d, opname))
end

"""
    mount_render_result(d, op, result) -> SemanticNode

Render an executed operation's result. `:stan`-origin strings are Stan
source (the `:transpile` contract); `:stan`-origin non-strings are the
compiled sampling problem, summarized; overrides render generically.
"""
mount_render_result(::BRMDescriptor, op::BRMOperation, result::AbstractString) =
    op.origin === :override ? SemanticCode(:text, result) :
                              SemanticCode(:stan, result)

function mount_render_result(d::BRMDescriptor, op::BRMOperation, result)
    op.origin === :override && return _render_override_result(result)
    parameters = Symbol[o.name for o in d.outputs if o.kind === :parameter]
    return mount_problem_summary(op.title, d.id, LogDensityProblems.dimension(result), parameters)
end

_render_override_result(result::SemanticNode) = result
_render_override_result(result) =
    SemanticCode(:text, sprint(show, MIME"text/plain"(), result))

"""
    mount_problem_summary(title, id, dim, parameters) -> SemanticNode

Pure summary of a compiled sampling problem: unconstrained dimension plus
the descriptor's parameter names. Pure (no problem handle) so it is
testable without compiling Stan.
"""
mount_problem_summary(title::AbstractString, id::AbstractString, dim::Integer,
                      parameters::AbstractVector) = SemanticSection(
    String(title),
    SemanticStatus(:available;
        detail="Compiled sampling problem ($dim unconstrained dimensions)."),
    SemanticFields(;
        unconstrained_dimensions=dim,
        parameters=join(string.(parameters), ", "),
        descriptor_id=String(id),
    ),
)

# ---- reflection -------------------------------------------------------------

_dash(::Nothing) = "—"
_dash(x) = string(x)

mount_formula_node(d::BRMDescriptor) = SemanticSection(
    "Formula",
    SemanticFields(; model=string(d.name), descriptor_id=d.id),
    SemanticCode(:julia, d.formula),
)

_schema_rows(d::BRMDescriptor) = [(
    column=string(c),
    stan_inputs=join((string(i.name) for i in d.inputs if i.column === c), ", "),
    transforms=begin
        kinds = unique!(String[string(i.transform) for i in d.inputs
                               if i.column === c && !isnothing(i.transform)])
        isempty(kinds) ? "pass-through" : join(kinds, ", ")
    end,
) for c in d.columns]

mount_schema_node(d::BRMDescriptor) =
    SemanticSection("Schema", SemanticTable(_schema_rows(d)))

mount_inputs_node(d::BRMDescriptor) = SemanticSection(
    "Inputs",
    SemanticTable([(
        name=string(i.name),
        type=string(i.type),
        size=string(i.size),
        constraints=string(i.constraints),
        observed=i.observed,
        held_out=i.held_out,
        derived=i.derived,
        inlined=i.inlined,
        column=_dash(i.column),
        transform=_dash(i.transform),
    ) for i in d.inputs]),
)

mount_outputs_node(d::BRMDescriptor) = SemanticSection(
    "Outputs",
    SemanticTable([(
        name=string(o.name),
        kind=string(o.kind),
        type=string(o.type),
        generative=string(o.generative),
        source=_dash(o.source),
        role=string(o.role),
        declaration=_dash(isnothing(o.declaration) ? nothing : o.declaration.target),
        family=_dash(isnothing(o.declaration) ? nothing : o.declaration.family),
        logical=_dash(o.logical),
        labels=isnothing(o.labels) ? "—" : join(string.(o.labels), ", "),
        segments=isnothing(o.segments) ? "—" : join(string.(o.segments), ", "),
    ) for o in d.outputs]),
)

mount_operations_node(d::BRMDescriptor) = SemanticSection(
    "Operations",
    SemanticTable([(
        name=string(op.name),
        title=op.title,
        origin=string(op.origin),
        inputs=join(string.(op.inputs), ", "),
        outputs=join(string.(op.outputs), ", "),
    ) for op in d.operations]),
)

# ---- the semantic graph -----------------------------------------------------

"""
The mounted graph: one descriptor, one operation picker, one execution
route, and reflection routes. Built with a descriptor (see
`mount_descriptor_page`); a bare mount renders an empty picker and its
bodies throw loudly.
"""
@htmx struct BRMDescriptorMount
    descriptor::Union{BRMDescriptor,Nothing} = nothing

    @options(operation) = mount_operation_domain(descriptor)

    """Run one derived operation of the mounted descriptor (no form-collected inputs)."""
    @get execute(; operation::Symbol) = mount_execute_operation(descriptor, operation)

    """The canonical declaration rendering plus model identity."""
    @get formula() = mount_formula_node(_require_descriptor(descriptor))

    """The dataframe schema: every column the declaration reads, with its Stan inputs."""
    @get schema() = mount_schema_node(_require_descriptor(descriptor))

    """The Stan data block, with dataframe provenance per input."""
    @get inputs() = mount_inputs_node(_require_descriptor(descriptor))

    """Everything the model produces, with BRM roles attached."""
    @get outputs() = mount_outputs_node(_require_descriptor(descriptor))

    """The derived operation set: names, titles, origins, and arities."""
    @get operations() = mount_operations_node(_require_descriptor(descriptor))
end

function _mount_shell(content, d::BRMDescriptor)
    htmx(
        h.main(content);
        hyperscript_version=nothing,
        pico_version=nothing,
        feedback=false,
        compose=false,
        overlay=false,
        extra_head=(h.title("BRM descriptor $(d.name)"),),
    )
end

"""
The served page: owns the descriptor (required root state) and compiles the
mount graph through `semantic_app` — one operation card per route, a
generated picker form, and stable result targets.
"""
@htmx struct BRMDescriptorMountPage
    descriptor::BRMDescriptor

    @include mount = BRMDescriptorMount(; descriptor)

    __page__(content) = _mount_shell(content, descriptor)

    """Mount the descriptor's derived operations and schema as one semantic app."""
    @get index() = semantic_app(mount; title="BRM descriptor $(descriptor.name)", submit="Run")
end

"""
    mount_descriptor_page(d::BRMDescriptor) -> BRMDescriptorMountPage

Build the servable page for one descriptor. Serve it behind a retaining
`RootProvider` factory (the page's descriptor state is required, so the
default fresh-per-request factory cannot build it), or drive it in-process
(see `../validate_descriptor_mount.jl` for the serving pattern).
Extra kwargs (e.g. `__cache_base__`) pass through to the page constructor.
"""
mount_descriptor_page(d::BRMDescriptor; kwargs...) = BRMDescriptorMountPage(d; kwargs...)

end # module DescriptorMount
