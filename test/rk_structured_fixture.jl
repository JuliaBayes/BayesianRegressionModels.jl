# Independently authored public consumer extension, retained from the public repro.
module NeutralTerms
import BayesianRegressionModels
const BRM = BayesianRegressionModels
import BayesianRegressionModels: _sb_term_group_block, _sb_is_direct_term,
    _brm_prepares_term, _brm_prepare_term, _brm_replay_term,
    _brm_native_structured_effect, _sb_predictor_term!
function group_line end
_sb_is_direct_term(::typeof(group_line)) = true
_sb_term_group_block(::typeof(group_line)) = (; fields=[
    (; name=:line, n_per_group=2, group=(; kwarg=:group), prior=:correlated_normal)])
_brm_prepares_term(::BRM.ExprColumn{typeof(group_line)}) = true
function _brm_prepare_term(term::BRM.ExprColumn{typeof(group_line)}, target, context)
    base=BRM._brm_prepare_structured_term(term,target,context,_sb_term_group_block(group_line,term))
    xname,xraw=BRM._brm_term_data(:group_line,only(BRM.getargs(term)),context)
    BRM._BRMPreparedTerm(base.callable,base.source,
        merge(base.state,(;xname,x=collect(Float64,xraw))),base.dependencies)
end
function _brm_replay_term(::typeof(group_line),training,fresh,context)
    fields=map(training.state.fields) do field
        merge(field,(;idx=BRM._brm_apply_levels(field.levels,context.data[field.source])))
    end
    BRM._BRMPreparedTerm(training.callable,training.source,
        merge(training.state,(;fields=Tuple(fields),x=collect(Float64,context.data[training.state.xname]))),
        training.dependencies)
end
function _brm_native_structured_effect(term::BRM._BRMPreparedTerm{typeof(group_line)},block,field)
    [block[field.idx[i],1]+block[field.idx[i],2]*term.state.x[i] for i in eachindex(field.idx)]
end
function _sb_predictor_term!(stmts,data,::typeof(group_line),t;
        target::Symbol,group_block_lookup=Dict(),kwargs...)
    xname,xraw=BRM._sb_inner_data(:group_line,only(BRM.getargs(t)))
    data[xname]=collect(Float64,xraw)
    info=BRM._sb_find_group_block(group_line,t,group_block_lookup)
    isnothing(info) && error("no allocated neutral line block")
    (;block_name,idx_name)=info
    a=Symbol(:line_,target,:_a); b=Symbol(:line_,target,:_b)
    col=Symbol(:line_,target,:_,xname)
    push!(stmts,:($a=$(block_name)[$idx_name,1]))
    push!(stmts,:($b=$(block_name)[$idx_name,2]))
    push!(stmts,:($col=$a .+ $b .* $xname))
    col
end
end
using .NeutralTerms: group_line
