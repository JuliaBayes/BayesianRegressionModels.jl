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

function _nlme_cell_eta_value(query, unit, theta_positions, theta,
        sigma_positions, sigma, eta_positions, eta, row)
    point = BRM._BRMNLMEPoint(unit, theta_positions, theta,
        sigma_positions, sigma, eta_positions, eta)
    query(point, row)
end

function BRM._rk_nlme_cell_value_and_eta_gradient(query::CellQuery,
        point::BRM._BRMNLMEPoint, row::Int)
    gradient = zeros(Float64, length(point.eta))
    Enzyme.autodiff(Enzyme.set_runtime_activity(Enzyme.Reverse),
        _nlme_cell_eta_value, Enzyme.Active,
        Enzyme.Const(query), Enzyme.Const(point.unit),
        Enzyme.Const(point.theta_positions), Enzyme.Const(point.theta),
        Enzyme.Const(point.sigma_positions), Enzyme.Const(point.sigma),
        Enzyme.Const(point.eta_positions), Enzyme.Duplicated(point.eta, gradient),
        Enzyme.Const(row))
    query(point, row), gradient
end

end
