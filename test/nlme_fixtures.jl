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

# The deprecated closure spelling of the same subject-indexed PK cell. It
# remains a value plan, with one in-cell observation array per subject.
function nlme_kernel_pk(data)
    @brm data begin
        sigma ~ Exponential(1)
        log_CL ~ 1 + weight + (1 | pk | subject)
        log_V ~ 1 + (1 | pk | subject)
        pred ~ kernel(t, dose, dv, log_CL, log_V) do ts, d, y, lCL, lV
            CL = exp(lCL)
            Vc = exp(lV)
            p = d / Vc .* exp.(-(CL / Vc) .* ts)
            y ~ normal(p, sigma)
            p
        end
    end
end

likelihood_query(backend) = prepare_query(backend.model,
    RKEXT._rk_translated_plan(backend.plan), :likelihood)

const NLME_AD = AutoEnzyme(; mode=Enzyme.Reverse)

# Independent oracle for the @plate population-PK model: every subject's
# conditional log-likelihood, η gradient, and η Hessian, written out from the model's
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
        population = Dict(Symbol("pop_log_CL.beta_pop.1") => gradient[kcl],
            Symbol("pop_log_CL.beta_pop.2") => d.weight[r] * gradient[kcl],
            :log_V_Intercept => gradient[kv])
        theta_gradient = [population[name] for name in names]
        sigma_gradient = [sum((d.dv[r] .- pred).^2 ./ sigma^2 .- 1)]
        # For p = dose/V * exp(-CL*t/V), with q = CL*t/V:
        # p_CL = -q*p, p_V = (q-1)*p, and
        # p_CL,CL = (q²-q)*p, p_CL,V = (2q-q²)*p,
        # p_V,V = (q²-3q+1)*p. The Normal Hessian is
        # Σ[(y-p)*p_ab - p_a*p_b] / sigma².
        dcl = -kt .* pred
        dv = (kt .- 1) .* pred
        hessian = zeros(size(H, 1), size(H, 1))
        hessian[kcl, kcl] = sum(residual .* pred .* (kt.^2 .- kt) .- dcl.^2 ./ sigma^2)
        hessian[kcl, kv] = sum(residual .* pred .* (2 .* kt .- kt.^2) .- dcl .* dv ./ sigma^2)
        hessian[kv, kcl] = hessian[kcl, kv]
        hessian[kv, kv] = sum(residual .* pred .* (kt.^2 .- 3 .* kt .+ 1) .- dv.^2 ./ sigma^2)
        (; value, gradient, hessian, theta_gradient, sigma_gradient)
    end
end

plate_pk_oracle(m, θ, σ, H) = [term.value for term in _plate_pk_oracle_terms(m, θ, σ, H)]
plate_pk_oracle_gradients(m, θ, σ, H) =
    reduce(hcat, [term.gradient for term in _plate_pk_oracle_terms(m, θ, σ, H)])
plate_pk_oracle_hessians(m, θ, σ, H) =
    [term.hessian for term in _plate_pk_oracle_terms(m, θ, σ, H)]

# A flat observation axis joined to the subject cells by `ragged(y, group)`:
# rows interleave subjects, counts differ, and the subject column is not in
# level order, so the joined (cell-by-cell) density order differs from both.
const NLME_RAGGED_DATA = (; subject=["s2", "s1", "s3"], dose=[80.0, 100.0, 120.0],
    obs_subject=["s1", "s2", "s3", "s1", "s3", "s2", "s1", "s3"],
    t=[0.5, 0.25, 0.5, 1.5, 2.0, 1.0, 3.0, 6.0],
    y=[8.1, 7.6, 9.0, 5.2, 6.3, 4.9, 2.1, 2.4])

function nlme_ragged_pk(data)
    @brm data begin
        sigma ~ Exponential(1)
        log(CL) ~ 1 + (1 | pk | subject)
        log(V) ~ 1 + (1 | pk | subject)
        @plate for i in eachindex(CL)
            conc[i] = dose[i] / V[i] .* exp.((-CL[i] / V[i]) .* ragged(t, obs_subject)[i])
        end
        ragged(y, obs_subject) ~ Normal(conc, sigma)
    end
end

# Independent oracle for the ragged-join model, by subject label from the flat
# rows: the same equations and η gradients as the @plate oracle above.
function _ragged_pk_oracle_terms(m, θ, σ, H, d)
    population(pred) = θ[findfirst(==(only(r.population for r in m.view.mu_references
        if r.predictor === pred)), m.theta)]
    a, b = population(:CL), population(:V)
    sigma = exp(only(σ))
    margin(pred) = findfirst(b -> b.predictor === pred, m.view.block_margins)
    kcl, kv = margin(:CL), margin(:V)
    map(eachindex(m.view.levels)) do i
        level = string(m.view.levels[i])
        dose = d.dose[only(findall(==(level), d.subject))]
        rows = findall(==(level), d.obs_subject)
        CL = exp(a + H[kcl, i])
        V = exp(b + H[kv, i])
        kt = (CL / V) .* d.t[rows]
        pred = dose / V .* exp.(-kt)
        value = sum(logpdf.(Normal.(pred, sigma), d.y[rows]))
        residual = (d.y[rows] .- pred) ./ sigma^2
        gradient = zeros(size(H, 1))
        gradient[kcl] = sum(residual .* pred .* -kt)
        gradient[kv] = sum(residual .* pred .* (kt .- 1))
        (; value, gradient)
    end
end

ragged_pk_oracle(m, θ, σ, H, d) = [term.value for term in _ragged_pk_oracle_terms(m, θ, σ, H, d)]
ragged_pk_oracle_gradients(m, θ, σ, H, d) =
    reduce(hcat, [term.gradient for term in _ragged_pk_oracle_terms(m, θ, σ, H, d)])

# A plate cell with its own observation (one marker array per subject) and a
# flat concentration column joined to the cells, left-censored at a per-row
# quantification limit with a combined additive and proportional scale. Rows
# interleave subjects; `s4` has no markers and one concentration, and two
# concentrations sit at their limit.
const NLME_CENSORED_JOIN_DATA = (; subject=["s2", "s1", "s3", "s4"],
    dose=[80.0, 100.0, 120.0, 90.0],
    marker=[[1.1, 0.9], [1.3], [0.8, 1.0, 1.2], Float64[]],
    obs_subject=["s1", "s2", "s3", "s1", "s3", "s2", "s1", "s3", "s4"],
    t=[0.5, 0.25, 0.5, 1.5, 2.0, 1.0, 3.0, 6.0, 1.0],
    conc=[8.1, 7.6, 9.0, 5.2, 6.3, 5.0, 2.1, 3.0, 4.0],
    lloq=[1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 1.0, 3.0, 0.5])

function nlme_censored_join(data)
    @brm data begin
        sigma_add ~ Exponential(1)
        sigma_prop ~ Exponential(1)
        marker_scale ~ Exponential(1)
        log(CL) ~ 1 + (1 | pk | subject)
        log(V) ~ 1 + (1 | pk | subject)
        @plate for i in eachindex(CL)
            pred[i] = dose[i] / V[i] .* exp.((-CL[i] / V[i]) .* ragged(t, obs_subject)[i])
            marker[i] ~ normal(fill(CL[i] / 10, length(marker[i])), marker_scale)
        end
        ragged(conc, obs_subject) ~ censored(
            Normal(pred, addprop(pred, sigma_add, sigma_prop)); lower=lloq)
    end
end

# A subject-level covariate with missing values, completed by the model and
# read by a subject predictor, beside an in-cell marker observation and a
# ragged concentration join. The covariate's observed values are subject data:
# `s1` and `s4` have one, `s2` and `s3` none.
const NLME_IMPUTED_JOIN_DATA = (; subject=["s2", "s1", "s3", "s4"],
    dose=[80.0, 100.0, 120.0, 90.0],
    w=Union{Missing,Float64}[missing, 0.3, missing, -0.5],
    marker=[[1.1, 0.9], [1.3], [0.8, 1.0, 1.2], Float64[]],
    obs_subject=["s1", "s2", "s3", "s1", "s3", "s2", "s1", "s3", "s4"],
    t=[0.5, 0.25, 0.5, 1.5, 2.0, 1.0, 3.0, 6.0, 1.0],
    conc=[8.1, 7.6, 9.0, 5.2, 6.3, 5.0, 2.1, 3.0, 4.0])

function nlme_imputed_join(data)
    @brm data begin
        mi(w) ~ Normal(0, 1)
        sigma ~ Exponential(1)
        marker_scale ~ Exponential(1)
        log(CL) ~ 1 + w + (1 | pk | subject)
        log(V) ~ 1 + (1 | pk | subject)
        @plate for i in eachindex(CL)
            pred[i] = dose[i] / V[i] .* exp.((-CL[i] / V[i]) .* ragged(t, obs_subject)[i])
            marker[i] ~ normal(fill(CL[i] / 10, length(marker[i])), marker_scale)
        end
        ragged(conc, obs_subject) ~ Normal(pred, sigma)
    end
end

# Independent oracle: each subject's marker and censored concentration
# densities, written out from the model's equations at the NLME point.
function censored_join_oracle(m, backend, θ, σ, H, d)
    c = constrain(backend.model.layout, BRM._brm_nlme_point(m, θ, σ, H))
    margin(pred) = findfirst(b -> b.predictor === pred, m.view.block_margins)
    kcl, kv = margin(:CL), margin(:V)
    map(eachindex(m.view.levels)) do i
        level = string(m.view.levels[i])
        r = only(findall(==(level), d.subject))
        rows = findall(==(level), d.obs_subject)
        CL = exp(c.CL_Intercept + H[kcl, i])
        V = exp(c.V_Intercept + H[kv, i])
        pred = d.dose[r] / V .* exp.(-(CL / V) .* d.t[rows])
        scale = sqrt.(c.sigma_add^2 .+ (pred .* c.sigma_prop) .^ 2)
        sum(logpdf.(censored.(Normal.(pred, scale), d.lloq[rows], Inf), d.conc[rows]);
            init=0.0) +
            sum(logpdf.(Normal(CL / 10, c.marker_scale), d.marker[r]); init=0.0)
    end
end

# A generic two-level random-effect model with interleaved observation rows:
# subject effects plus nested unit effects owned through `obs_subject`.
const NESTED_DATA = (;
    subject=["s1", "s2", "s3"],
    obs_subject=["s2", "s1", "s3", "s2", "s3", "s3", "s1", "s2", "s3"],
    unit=["u2", "u1", "u4", "u3", "u5", "u6", "u1", "u2", "u4"],
    y=[0.3, 0.2, -0.2, 0.5, 0.1, -0.4, 0.4, 0.6, -0.1],
)

function nested_model(data)
    @brm data begin
        sigma ~ Exponential(1)
        mu ~ 1 + (1 | p | subject)
        unit_shift ~ 0 + (1 | u | unit)
        @plate for i in eachindex(mu)
            pred[i] = mu[i] .+ ragged(unit_shift, obs_subject)[i]
        end
        ragged(y, obs_subject) ~ Normal(pred, sigma)
    end
end
