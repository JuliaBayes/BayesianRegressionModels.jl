# Shared fixtures for the NLME tests (test/nlme_view.jl, test/nlme_estimation_ext.jl).
using Test, BayesianRegressionModels, Distributions
using ReactiveKernels, ReactiveKernelsPPL, Enzyme
using DifferentiationInterface: AutoEnzyme
const BRM = BayesianRegressionModels
include(joinpath(@__DIR__, "testset_filter.jl"))

# Four subjects, interleaved rows, unequal row counts.
const NLME_DATA = (;
    subject=["s1", "s2", "s3", "s1", "s4", "s2", "s3", "s1", "s4", "s2", "s3", "s4", "s1"],
    t=[0.5, 0.5, 0.5, 1.0, 0.5, 1.5, 2.0, 3.0, 2.0, 4.0, 6.0, 6.0, 8.0],
    dose=fill(100.0, 13),
    wt=[0.1, -0.3, 0.6, 0.1, -0.8, -0.3, 0.6, 0.1, -0.8, -0.3, 0.6, -0.8, 0.1],
    y=[18.0, 22.0, 15.0, 16.5, 25.0, 17.0, 10.0, 12.0, 14.0, 9.0, 4.0, 5.0, 3.5],
)

function nlme_pk(data)
    @brm data begin
        sigma ~ Exponential(1)
        log(CL) ~ 1 + wt + (1 | pk | subject)
        log(V) ~ 1 + (1 | pk | subject)
        m = dose / V * exp(-CL / V * t)
        y ~ Normal(m, sigma)
    end
end

const RKEXT = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)

pointwise_query(backend) = prepare_query(backend.model,
    RKEXT._rk_translated_plan(backend.plan), :pointwise)


# The current population-PK spelling: per-subject predictors read inside an
# indexed `@plate for` cell over pre-grouped observations.
const NLME_PLATE_DATA = (; subject=["a", "b", "c"], dose=[100.0, 80.0, 120.0],
    weight=[0.1, -0.2, 0.3], t=[[0.5, 1.0, 2.0], [0.5, 3.0], [1.0, 2.0, 4.0, 8.0]],
    dv=[[18.0, 15.0, 11.0], [20.0, 8.0], [16.0, 12.0, 7.0, 3.0]])

function nlme_plate_pk(data)
    @brm data begin
        sigma ~ Exponential(1)
        log_CL ~ 1 + weight + (1 | pk | subject)
        log_V ~ 1 + (1 | pk | subject)
        @plate for i in eachindex(log_CL)
            CL = exp(log_CL[i])
            Vc = exp(log_V[i])
            pred[i] = dose[i] / Vc .* exp.(-(CL / Vc) .* t[i])
            dv[i] ~ normal(pred[i], sigma)
        end
    end
end

likelihood_query(backend) = prepare_query(backend.model,
    RKEXT._rk_translated_plan(backend.plan), :likelihood)

const NLME_AD = AutoEnzyme(; mode=Enzyme.Reverse)

# Independent oracle for the @plate population-PK model: every subject's
# conditional log-likelihood, and its η gradient, written out from the model's
# equations. With CL = exp(a + η_CL), V = exp(b + η_V), k = CL / V and
# pred = dose / V * exp(-k t): ∂pred/∂η_CL = -k t pred, ∂pred/∂η_V = (k t - 1) pred,
# and ∂ log p / ∂pred = (dv - pred) / sigma².
function _plate_pk_oracle_terms(m, θ, σ, H)
    names = m.view.coordinates[m.theta]
    θn = Dict(zip(names, θ))
    cl0, cl_w = θn[Symbol("pop_log_CL.beta_pop.1")], θn[Symbol("pop_log_CL.beta_pop.2")]
    v0 = θn[:log_V_Intercept]
    sigma = exp(only(σ))
    margin(pred) = findfirst(b -> b.predictor === pred, m.view.block_margins)
    kcl, kv = margin(:log_CL), margin(:log_V)
    d = NLME_PLATE_DATA
    map(eachindex(m.view.levels)) do i
        r = only(m.view.rows[i])
        CL = exp(cl0 + cl_w * d.weight[r] + H[kcl, i])
        V = exp(v0 + H[kv, i])
        kt = (CL / V) .* d.t[r]
        pred = d.dose[r] / V .* exp.(-kt)
        value = sum(logpdf.(Normal.(pred, sigma), d.dv[r]))
        residual = (d.dv[r] .- pred) ./ sigma^2
        gradient = zeros(size(H, 1))
        gradient[kcl] = sum(residual .* pred .* -kt)
        gradient[kv] = sum(residual .* pred .* (kt .- 1))
        (; value, gradient)
    end
end

plate_pk_oracle(m, θ, σ, H) = [term.value for term in _plate_pk_oracle_terms(m, θ, σ, H)]
plate_pk_oracle_gradients(m, θ, σ, H) =
    reduce(hcat, [term.gradient for term in _plate_pk_oracle_terms(m, θ, σ, H)])
