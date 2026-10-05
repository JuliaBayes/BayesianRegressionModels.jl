# GP statistical construction adopted from public RK 4d4b7608478a389c472dc14282c4053330517650.
# Dense factorization is a general numerical operation supplied by RK proper.
const Prep = BRM.StatisticalPreparation
Prep._gp_cholesky_lower(K) = ReactiveKernels.rk_cholesky_lower(K)

for name in (:gp_pair_locations, :gp_exp_quad_cov, :gp_periodic_cov)
    definition = Expr(:macrocall, GlobalRef(ReactiveKernels, Symbol("@kernel")),
        LineNumberNode(0), deepcopy(getproperty(Prep._GP_COVARIANCE_MODELS, name)))
    Core.eval(@__MODULE__, definition)
end

const _GP_EXP_QUAD_COV = prepare(gp_exp_quad_cov_graph)
const _GP_PERIODIC_COV = prepare(gp_periodic_cov_graph)

Prep._gp_exp_quad_cov_vector(x, sigma, rho, jitter) =
    _GP_EXP_QUAD_COV(x, sigma, rho, jitter)
Prep._gp_periodic_cov_vector(x, sigma, rho, period, jitter) =
    _GP_PERIODIC_COV(x, sigma, rho, period, jitter)

# Exact fixed-layout body: two independent HSGPs for mean and log scale.
# q = [log_rho_mu, log_sd_mu, v_mu[1:20]...,
#      log_rho_sigma, log_sd_sigma, v_sigma[1:20]...].
# c=0 is noncentered, c=1 centered; the Jacobian maps the live coordinates.
@kernel dual_hsgp(q::Vector{Float64}, x::Vector{Float64}, y::Vector{Float64},
              c::Vector{Float64}, modes::Vector{Float64}, half_width::Float64) = begin
    frequency = modes .* (pi / (2 * half_width))
    frequency_squared = frequency .^ 2
    basis = sin.((x .+ half_width) * transpose(frequency)) ./ sqrt(half_width)

    log_rho_mu = q[1]
    log_sd_mu = q[2]
    log_rho_sigma = q[23]
    log_sd_sigma = q[24]
    v_mu = q[3:22]
    v_sigma = q[25:44]
    c_mu = c[1:20]
    c_sigma = c[21:40]

    log_scale_mu = log_sd_mu .+ 0.5 * log_rho_mu .+ 0.25 * log(2pi) .-
        0.25 .* exp(2 * log_rho_mu) .* frequency_squared
    log_scale_sigma = log_sd_sigma .+ 0.5 * log_rho_sigma .+ 0.25 * log(2pi) .-
        0.25 .* exp(2 * log_rho_sigma) .* frequency_squared
    z_mu = v_mu .* exp.(-c_mu .* log_scale_mu)
    z_sigma = v_sigma .* exp.(-c_sigma .* log_scale_sigma)
    weights_mu = v_mu .* exp.((1 .- c_mu) .* log_scale_mu)
    weights_sigma = v_sigma .* exp.((1 .- c_sigma) .* log_scale_sigma)
    mu = basis * weights_mu
    log_sigma = basis * weights_sigma

    # LogNormal(0,4) priors plus the four positive-parameter Jacobians.
    hyperprior = -(log_rho_mu^2 + log_sd_mu^2 +
                   log_rho_sigma^2 + log_sd_sigma^2) / 32 -
        4 * log(4) - 2 * log(2pi)
    coordinate_jacobian = -sum(c_mu .* log_scale_mu) - sum(c_sigma .* log_scale_sigma)
    weight_prior = -0.5 * (sum(abs2, z_mu) + sum(abs2, z_sigma)) - 20 * log(2pi)
    pointwise = plate(y, mu, log_sigma) do yi, mui, lsi
        -0.5 * ((yi - mui) * exp(-lsi))^2 - lsi - 0.5 * log(2pi)
    end
    likelihood = sum(pointwise)
    posterior = hyperprior + weight_prior + coordinate_jacobian + likelihood
    return posterior
end

const _RK_STATISTICAL_GRAPHS = (
    gp_exp_quad_cov=gp_exp_quad_cov_graph,
    gp_periodic_cov=gp_periodic_cov_graph,
    dual_hsgp=dual_hsgp,
)
BRM.rk_model(name::Symbol) = hasproperty(_RK_STATISTICAL_GRAPHS, name) ?
    getproperty(_RK_STATISTICAL_GRAPHS, name) :
    throw(ArgumentError("unknown BRM native statistical model `$name`"))

function Prep.prepare_dual_hsgp(data; want=:posterior)
    length(data.x) == length(data.y) == 133 ||
        throw(DimensionMismatch("expected 133 rows"))
    prepare(dual_hsgp; have=(:q, :c, :x, :y, :modes, :half_width), want,
        bound=(; x=data.x, y=data.y, modes=collect(1.0:20.0), half_width=1.5))
end
