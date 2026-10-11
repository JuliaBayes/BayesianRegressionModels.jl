module BayesianRegressionModelsReactiveKernelsEnzymeExt

using BayesianRegressionModels
using DifferentiationInterface: AutoEnzyme, SecondOrder
using ReactiveKernels
using ReactiveKernelsPPL
import Enzyme

const BRM = BayesianRegressionModels

function BRM._rk_eta_hvp_query(backend::BRM.RKBRMI, unit::Vector{Float64})
    rkext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
    second_order = SecondOrder(
        AutoEnzyme(; mode=Enzyme.Reverse),
        AutoEnzyme(; mode=Enzyme.Forward))
    prepare_query_ad(prepare_ad_hvp, backend.model,
        rkext._rk_translated_plan(backend.plan), :sampler,
        second_order, (zeros(Float64, length(unit)),), unit)
end

BRM._rk_eta_hvp(query, direction::Vector{Float64}, u::Vector{Float64}) =
    only(ad_hvp(query, (direction,), u))

function BRM._rk_nlme_compact_cell_sampler(backend::BRM.RKBRMI,
        observations::Tuple, base::Vector{Float64}, selected::Vector{Int})
    rkext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
    prepare_compact_cell_sampler(backend.model,
        rkext._rk_translated_plan(backend.plan), observations, base, selected;
        backend=AutoEnzyme(; mode=Enzyme.set_runtime_activity(Enzyme.Reverse)))
end

BRM._rk_nlme_compact_cell_value_and_gradient!(sampler::CompactCellSampler,
        gradient::Vector{Float64}, base::Vector{Float64},
        active::Vector{Float64}, selected::Vector{Int}, row::Int) =
    cell_value_and_gradient!(sampler, gradient, base, active, selected, row)

end
