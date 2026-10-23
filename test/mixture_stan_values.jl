# Shared by the SBBRMI `MixtureModel` tests: instantiate a descriptor's Stan
# program and read its constrained values at an unconstrained point.
using BayesianRegressionModels, BridgeStan

const MIXTURE_CACHE = joinpath(tempdir(), "brm-mixtures")
isdir(MIXTURE_CACHE) || mkpath(MIXTURE_CACHE)

mixture_problem(d) = brm_execute(d, :instantiate;
    path=joinpath(MIXTURE_CACHE, string(d.id) * ".stan"))

# Constrained values of every parameter, transformed parameter and generated
# quantity at the unconstrained point `q`, by Stan name.
function mixture_values(problem, q)
    model = problem.model
    names = BridgeStan.param_names(model; include_tp=true, include_gq=true)
    values = BridgeStan.param_constrain(model, q; include_tp=true,
        include_gq=true, rng=BridgeStan.StanRNG(model, 20261023))
    Dict(zip(names, values))
end
