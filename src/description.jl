"""A named value and its declared indexing axis, never guessed from its name."""
struct BRMDescriptionReference
    name::Symbol
    axis::Symbol
    logical::Union{Symbol,Tuple}
end
BRMDescriptionReference(name::Symbol,axis::Symbol)=BRMDescriptionReference(name,axis,name)

"""
    BRMPriorDescription

An effective prior, addressed by a logical tuple `id`. `distribution` preserves
the actual callable and arguments; `support` includes declared constraints.
`source` records the winning selector or the producer default. `anchor` is an
optional report-supplied URL. Numeric constants are snapshots of the selected
prepared model; symbolic hyperparameters remain named references.
"""
struct BRMPriorDescription
    id::Tuple
    distribution::Any
    support::NamedTuple
    source::Any
    anchor::Union{Nothing,String}
    provenance::NamedTuple
end

"""
    BRMDescriptionComponent

One semantic node. Its stable logical tuple `id` identifies this use of the
actual `callable`. `arguments` and `keywords` contain literals, references and
recursive component nodes. `children` is the inventory a custom hook must
inspect before claiming coverage. Axes, named outputs, effective priors and
fitted constants come from the selected descriptor. `provenance` ties the node
to the authored declaration and executable model identity. Inputs are read-only.
"""
struct BRMDescriptionComponent
    id::Tuple
    kind::Symbol
    callable::Any
    arguments::Tuple
    keywords::NamedTuple
    axes::Tuple
    outputs::Tuple
    priors::Tuple
    fitted_constants::Tuple
    children::Tuple
    provenance::NamedTuple
    notation::Tuple
    bindings::Tuple
end

"""The read-only semantic component supplied to a description hook."""
const BRMDescriptionContext = BRMDescriptionComponent

"""
    BRMDescriptionFragment(; prose=(), equations=(), covers=(), notation=())

A hook's scientific explanation. Equations are LaTeX without display delimiters.
`covers` lists exact component IDs; covering a parent never covers its children.
`notation` contains records with `name`, `meaning`, and optional `symbol`,
`axis`, `unit`. Hooks are responsible for the accuracy of their scientific math.
"""
struct BRMDescriptionFragment
    prose::Tuple
    equations::Tuple
    covers::Tuple
    notation::Tuple
end
BRMDescriptionFragment(; prose=(), equations=(), covers=(), notation=()) =
    BRMDescriptionFragment(Tuple(prose), Tuple(equations), Tuple(covers), Tuple(notation))

struct BRMDescriptionCoverage
    id::Tuple
    status::Symbol
    reason::String
    provenance::NamedTuple
end

"""
    BRMDescription

Deterministic description of an existing prepared model. `complete` is true
only when every inventoried semantic component is covered. It certifies inventory
coverage, not the independent scientific correctness of handwritten hooks.
`model_id` is the unchanged inference identity; labels and prose do not alter it.
"""
struct BRMDescription
    model_id::String
    components::Tuple
    prose::Tuple
    equations::Tuple
    notation::Tuple
    priors::Tuple
    coverage::Tuple
    diagnostics::Tuple
    complete::Bool
end

"""
    brm_describe_component(callable, context::BRMDescriptionContext)

Extend by dispatch on the actual callable or its type. Return a
`BRMDescriptionFragment`, or `nothing` when no scientific description is known.
Ordinary calls nested in assignments and kernel cells use this same hook.
Per-request `hooks=(callable => context -> fragment, ...)` override dispatch.
No model fitting, sampling or callable execution occurs during description.
"""
brm_describe_component(_callable, _context::BRMDescriptionContext) = nothing

"""
    brm_description_binding(context, name::Symbol)

Resolve an authored component-local name without parsing generated carriers.
Returns `(name, role, value, prior_ids)`; aliases point at public arguments,
parameters link to exact logical prior IDs. Missing/ambiguous names fail loudly.
"""
function brm_description_binding(c::BRMDescriptionContext,name::Symbol)
    hits=filter(b -> b.name===name,c.bindings)
    length(hits)==1 || throw(ArgumentError("description binding `$name` has $(length(hits)) matches in $(c.id)"))
    only(hits)
end
function brm_description_binding(c::BRMDescriptionContext,path::Tuple)
    hits=filter(b->get(b,:path,(b.name,))==path,c.bindings)
    length(hits)==1 || throw(ArgumentError("description binding path $path has $(length(hits)) matches in $(c.id)"))
    only(hits)
end

"""Resolve the exact logical prior ID returned by a component binding."""
function brm_description_prior(c::BRMDescriptionContext,id::Tuple)
    hits=filter(p -> p.id==id,c.priors)
    length(hits)==1 || throw(ArgumentError("description prior $id has $(length(hits)) matches"))
    only(hits)
end

"""
    brm_description_prior_anchor(description_or_context, id::Tuple; prefix=nothing)

Stable HTML target for an exact logical prior, scoped to the model identity.
Supply a unique `prefix` when mounting repeated instances of one model artifact.
The Markdown renderer creates this target in its complete effective-prior table.
"""
_brmd_prior_anchor(prefix::AbstractString,id::Tuple) =
    "brm-prior-"*bytes2hex(codeunits(prefix))*"-"*bytes2hex(codeunits(repr(id)))
brm_description_prior_anchor(d::BRMDescription,id::Tuple;prefix=nothing) =
    _brmd_prior_anchor(isnothing(prefix) ? d.model_id : String(prefix),id)
brm_description_prior_anchor(c::BRMDescriptionContext,id::Tuple;prefix=nothing) =
    _brmd_prior_anchor(isnothing(prefix) ? c.provenance.model_id : String(prefix),id)
brm_description_prior_anchor(p::BRMPriorDescription;prefix=nothing) =
    _brmd_prior_anchor(isnothing(prefix) ? p.provenance.model_id : String(prefix),p.id)

"""
    brm_description_prior_references(context)

Ordered logical prior IDs bound to this component, its quantities and descendants.
Resolve them with `brm_description_prior`; link their default rendered targets with
`brm_description_prior_anchor`. This does not parse generated parameter names.
"""
function brm_description_prior_references(c::BRMDescriptionContext)
    ids=Set{Tuple}()
    _brmd_prior_references!(ids,c)
    Tuple(p.id for p in c.priors if p.id in ids)
end
function _brmd_prior_references!(ids,c::BRMDescriptionComponent)
    for b in c.bindings
        union!(ids,b.prior_ids)
    end
    for p in c.priors
        if first(c.id)===:random_effect && length(p.id)>=length(c.id) && p.id[1:length(c.id)]==c.id ||
           first(p.id)===:population && length(p.id)>=2 && p.id[2]===c.provenance.owner
            push!(ids,p.id)
        end
    end
    foreach(x->_brmd_prior_references!(ids,c,x),(c.arguments...,values(c.keywords)...))
    foreach(x->_brmd_prior_references!(ids,x),c.children)
end
_brmd_prior_references!(_ids,_c,_x)=nothing
_brmd_prior_references!(_ids,_c,_x::BRMDescriptionComponent)=nothing
_brmd_prior_references!(ids,c,x::Union{Tuple,NamedTuple})=foreach(v->_brmd_prior_references!(ids,c,v),x)
function _brmd_prior_references!(ids,c,x::BRMDescriptionReference)
    resolved=_brmd_resolve_alias(x,c.bindings)
    resolved isa BRMDescriptionReference || return
    for p in c.priors
        if p.id==resolved.logical || get(p.source,:binding_id,nothing)==resolved.logical ||
           p.id==(:parameter,resolved.logical) ||
           first(p.id)===:population && length(p.id)>=2 && p.id[2]===resolved.logical
            push!(ids,p.id)
        end
    end
end

"""Return the complete, ordered recursive semantic inventory."""
function brm_description_components(component::BRMDescriptionComponent)
    (component, (node for child in component.children
                for node in brm_description_components(child))...)
end
brm_description_components(description::BRMDescription) =
    Tuple(node for root in description.components
          for node in brm_description_components(root))

include("description_semantics.jl")
include("description_priors.jl")
include("description_render.jl")
