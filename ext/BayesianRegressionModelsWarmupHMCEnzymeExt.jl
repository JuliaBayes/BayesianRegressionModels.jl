module BayesianRegressionModelsWarmupHMCEnzymeExt

using BayesianRegressionModels
using Enzyme
using LogDensityProblems
using WarmupHMC

const AC_EXT = Base.get_extension(
    BayesianRegressionModels, :BayesianRegressionModelsWarmupHMCExt,
)
const SmallLocation = AC_EXT.BRMAdaptiveCenteringArgument{:location,true}
const SmallLogScale = AC_EXT.BRMAdaptiveCenteringArgument{:log_scale,true}
const SmallArguments = Tuple{SmallLocation,SmallLogScale}
const FloatCentering = WarmupHMC.PartiallyCentered{Float64}
const SmallReparametrization = WarmupHMC.Reparametrization{
    FloatCentering,FloatCentering,SmallArguments,
}
const SmallIndexedReparametrization = WarmupHMC.IndexedReparametrization{
    Vector{Pair{Int,SmallReparametrization}},
}

# This pullback deliberately mirrors Enzyme's own floating-point graph because
# WarmupHMC makes discrete centering decisions from these gradients. Enzyme
# 0.13.200 changed tanh's derivative from `seed / cosh(x)^2` to
# `(1 - tanh(x)^2) * seed`; the two are algebraically equal but not bit-equal.
const ENZYME_TANH_PULLBACK_FROM_PRIMAL = pkgversion(Enzyme) >= v"0.13.200"

# Differentiating the generic indexed loop still leaves one Enzyme rule call per
# accessor.  For BRM's scalar and intercept/slope blocks, run that same reverse
# pass directly.  The pair vector's exact concrete type makes this dispatch
# exclusive to the BRM K<=2 plan; wider blocks and every non-BRM reparametrizer
# retain WarmupHMC's generic DifferentiationInterface path.
function LogDensityProblems.logdensity_and_gradient(
    problem::WarmupHMC.ReparametrizedProblem{
        SmallIndexedReparametrization,P,B,
    },
    x::AbstractVector,
) where {P,B}
    ljac, y = problem.reparametrizer(x)
    ld, model_gradient =
        LogDensityProblems.logdensity_and_gradient(problem.problem, y)
    gradient = zeros(eltype(model_gradient), length(model_gradient))
    state = first(problem.reparametrizer.pairs).second.args[1].state
    T = eltype(x)
    # Pair construction is block-major, then group-major, then term-major.
    # Reverse that exact order, cache only values that the frozen forward graph
    # repeats identically, and retain its literal pullback operation order.
    for bi in Iterators.reverse(eachindex(state.blocks))
        block = state.blocks[bi]
        K = block.ranef.n_terms
        tau1 = exp(x[block.log_scales[1]])
        C11 = tau1 * one(T)
        if K == 2
            raw_index = only(block.cholesky_free)
            scale2_index = block.log_scales[2]
            raw = x[raw_index]
            stick = one(T)
            corr = tanh(raw)
            L21 = stick * corr
            variance = one(T) - corr * corr
            root = sqrt(variance)
            stick *= root
            tau2 = exp(x[scale2_index])
            C21 = tau2 * L21
            C22 = tau2 * stick
            tanh_pullback_factor = ENZYME_TANH_PULLBACK_FROM_PRIMAL ?
                one(T) - corr * corr : cosh(raw) * cosh(raw)
        end

        for group in block.ranef.n_groups:-1:1, term in K:-1:1
            pair_number = state.pair_lookup[bi][term, group]
            index, value = problem.reparametrizer.pairs[pair_number]
            source = value.source.c
            target = value.target.c
            model_seed = model_gradient[index]

            if term == 1
                location = zero(T)
                log_scale = log(C11)
            else
                c = state.sources[state.pair_lookup[bi][1, group]]
                u = x[block.effects[1, group]]
                m = zero(T)
                numerator = u - c * m
                denominator = C11^c
                innovation = numerator / denominator
                location = zero(T)
                location += C21 * innovation
                log_scale = log(C22)
            end

            delta = target - source
            local_ljac = log_scale * delta
            shifted = x[index] - source * location
            transport_scale = exp(local_ljac)

            dlocation = zero(T)
            dshifted = model_seed * transport_scale
            dtransport = model_seed * shifted
            dlocal_ljac = one(T)
            dlocal_ljac += dtransport * transport_scale
            dlog_scale = dlocal_ljac * delta
            dlocation += (-dshifted) * source
            dlocation += model_seed * target

            if term == 1
                dC11 = dlog_scale / C11
                dtau1 = dC11 * one(T)
                gradient[block.log_scales[1]] += dtau1 * tau1
            else
                dC22 = dlog_scale / C22
                dtau2 = dC22 * stick
                dstick = dC22 * tau2
                droot = dstick * one(T)
                dvariance = droot / (root * 2)
                dproduct = -dvariance
                dcorr = dproduct * corr
                dcorr += dproduct * corr
                draw = ENZYME_TANH_PULLBACK_FROM_PRIMAL ?
                    tanh_pullback_factor * dcorr : dcorr / tanh_pullback_factor
                gradient[raw_index] += draw
                gradient[scale2_index] += dtau2 * tau2

                effect_index = block.effects[1, group]
                dproduct = dlocation
                dC21 = dproduct * innovation
                dinnovation = dproduct * C21
                dnumerator = dinnovation / denominator
                dtau2 = dC21 * L21
                dL21 = dC21 * tau2
                dcorr = dL21 * one(T)
                draw = ENZYME_TANH_PULLBACK_FROM_PRIMAL ?
                    tanh_pullback_factor * dcorr : dcorr / tanh_pullback_factor
                dscale1 = ((((dlocation * C21) * (numerator * -c)) *
                    C11^(c - one(c))) * tau1) /
                    (denominator * denominator)
                gradient[raw_index] += draw
                gradient[block.log_scales[1]] += dscale1
                gradient[scale2_index] += dtau2 * tau2
                gradient[effect_index] += dnumerator
            end
            gradient[index] += dshifted
        end
    end
    for index in eachindex(gradient)
        index in state.effect_indices ||
            (gradient[index] += model_gradient[index])
    end
    ljac + ld, gradient
end

end
