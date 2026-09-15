module PupilTotalEffects

using DelimitedFiles, Distributions, LinearAlgebra, LogDensityProblems, SHA, Statistics

const DATA_SHA256 = "27e73bec304bc68271b0a9d46b1736a7c8e62fd8dcfac7d20704524c47edc8e8"
const DATA_REVISION = "d90fc01e6f6fcdced7ee64c9d2ed607d212ec77c"
const INTERCEPT_MEAN = 5651.9
const INTERCEPT_SD = 2026.1
const GROUP_SCALE = 2026.1
const LOG_SIGMA_SCALE = 2.5

abstract type MeanPrior end
struct GaussianMean <: MeanPrior end
struct StudentMixtureMean <: MeanPrior end
extra_dimensions(::GaussianMean) = 0
extra_dimensions(::StudentMixtureMean) = 1
mean_variance(::GaussianMean, q) = INTERCEPT_SD^2
mean_variance(::StudentMixtureMean, q) = INTERCEPT_SD^2*exp(-q[end])
mixture_term!(g, ::GaussianMean, q, h, v) = 0.0
function mixture_term!(g, ::StudentMixtureMean, q, h, v)
    eta = q[end]
    lambda = exp(eta)
    variance = INTERCEPT_SD^2*exp(-eta)
    # lambda ~ Gamma(nu/2, rate=nu/2); beta0|lambda ~ Normal(m,s/sqrt(lambda)).
    g[end] = 1.5 - 1.5lambda + variance/2*(1/v-h^2/v^2)
    logpdf(Gamma(1.5,2/3),lambda) + eta
end

struct PupilData
    ids::Vector{Int}
    group::Vector{Int}
    x::Vector{Float64}
    y::Vector{Float64}
    xbar::Float64
    idbar::Float64
    n::Vector{Int}
    mean_x::Vector{Float64}
    mean_y::Vector{Float64}
    sxx::Vector{Float64}
    slope::Vector{Float64}
    sse::Vector{Float64}
end

function load_data(path=joinpath(@__DIR__, "reference", "pupil.csv"))
    bytes2hex(sha256(read(path))) == DATA_SHA256 || error("Pupil data hash mismatch")
    raw, header = readdlm(path, ',', Float64; header=true)
    vec(header) == ["subj", "trial", "load", "p_size"] || error("Unexpected columns")
    ids = sort(unique(Int.(raw[:, 1])))
    lookup = Dict(id => j for (j, id) in enumerate(ids))
    group = [lookup[Int(id)] for id in raw[:, 1]]
    x, y = raw[:, 3], raw[:, 4]
    length(y) == 2228 && length(ids) == 20 || error("Unexpected pupil dimensions")
    n, mx, my, sxx, slope, sse = Int[], Float64[], Float64[], Float64[], Float64[], Float64[]
    for j in eachindex(ids)
        mask = findall(==(j), group)
        xj, yj = x[mask], y[mask]
        xc, yc = xj .- mean(xj), yj .- mean(yj)
        b = dot(xc, yc) / dot(xc, xc)
        push!(n, length(mask)); push!(mx, mean(xj)); push!(my, mean(yj))
        push!(sxx, dot(xc, xc)); push!(slope, b); push!(sse, sum(abs2, yc .- b .* xc))
    end
    PupilData(ids, group, x, y, mean(x), mean(raw[:, 1]), n, mx, my, sxx, slope, sse)
end

"""Physical model frame: log(tau_a), log(tau_b), gamma0, gamma1, A[1:J], B[1:J]."""
struct PupilProblem{P<:MeanPrior}
    data::PupilData
    gradient_calls::Base.RefValue{Int}
    mean_prior::P
end
PupilProblem(data=load_data(), prior=GaussianMean()) = PupilProblem(data, Ref(0), prior)
LogDensityProblems.dimension(p::PupilProblem) = 4 + 2length(p.data.ids) + extra_dimensions(p.mean_prior)
LogDensityProblems.capabilities(::Type{<:PupilProblem}) = LogDensityProblems.LogDensityOrder{1}()

function coordinate_names(data)
    vcat(["log_tau_intercept", "log_tau_load", "log_sigma_intercept", "log_sigma_subj_slope"],
         ["total_intercept[$id]" for id in data.ids], ["total_load[$id]" for id in data.ids])
end
coordinate_names(data, ::GaussianMean) = coordinate_names(data)
coordinate_names(data, ::StudentMixtureMean) = vcat(coordinate_names(data),"log_intercept_mixture_precision")

student3_log(x, scale) = log(2 / (pi * sqrt(3) * scale)) - 2log1p(x^2 / (3scale^2))
student3_grad(x, scale) = -4x / (3scale^2 + x^2)

function evaluate(p::PupilProblem, q)
    d = p.data
    J = length(d.ids)
    length(q) == LogDensityProblems.dimension(p) || throw(DimensionMismatch())
    g = zeros(length(q))
    all(isfinite, q) || return -Inf, g
    la, lb, gamma0, gamma1 = q[1:4]
    va, vb = exp(2la), exp(2lb)
    if !(isfinite(va) && isfinite(vb) && va > 0 && vb > 0)
        return -Inf, g
    end
    A, B = @view(q[5:4+J]), @view(q[5+J:4+2J])
    ma, mb = mean(A), mean(B)
    qa, qb = sum(a -> (a-ma)^2, A), sum(b -> (b-mb)^2, B)
    # Original population design uses centered load; group design uses raw load.
    # beta_load has the source model's flat prior. Integrating both population
    # effects leaves a proper Gaussian factor on mean(A)+xbar*mean(B), and a
    # flat common-slope direction. No flat group-effect prior is introduced.
    h = ma + d.xbar * mb - INTERCEPT_MEAN
    v = mean_variance(p.mean_prior,q) + (va + d.xbar^2 * vb) / J
    isfinite(v) && v > 0 || return -Inf, g
    lp = -(J-1)*log(2pi) - log(J) - (J-1)*(la+lb) - (qa/va + qb/vb)/2
    lp += -(log(2pi*v) + h^2/v)/2
    g[1] = -(J-1) + qa/va + va/J*(h^2/v^2 - 1/v)
    g[2] = -(J-1) + qb/vb + d.xbar^2*vb/J*(h^2/v^2 - 1/v)
    lp += mixture_term!(g,p.mean_prior,q,h,v)
    for (k, variance, ell) in ((1, va, la), (2, vb, lb))
        tau = sqrt(variance)
        lp += log(2) + student3_log(tau, GROUP_SCALE) + ell
        g[k] += 1 - 4variance/(3GROUP_SCALE^2+variance)
    end
    lp += student3_log(gamma0, LOG_SIGMA_SCALE)
    g[3] = student3_grad(gamma0, LOG_SIGMA_SCALE)
    for j in 1:J
        g[4+j] = -(A[j]-ma)/va - h/(J*v)
        g[4+J+j] = -(B[j]-mb)/vb - d.xbar*h/(J*v)
        # sigma ~ subj is numeric-ID regression in the literal post-3 data.
        id_centered = d.ids[j] - d.idbar
        ls = gamma0 + gamma1*id_centered
        inv_variance = exp(-2ls)
        isfinite(inv_variance) || return -Inf, zeros(length(q))
        residual = A[j] + B[j]*d.mean_x[j] - d.mean_y[j]
        slope_residual = B[j] - d.slope[j]
        rss = d.n[j]*residual^2 + d.sxx[j]*slope_residual^2 + d.sse[j]
        lp += -d.n[j]*(log(2pi)/2 + ls) - rss*inv_variance/2
        g[4+j] -= d.n[j]*residual*inv_variance
        g[4+J+j] -= (d.n[j]*residual*d.mean_x[j] + d.sxx[j]*slope_residual)*inv_variance
        gs = -d.n[j] + rss*inv_variance
        g[3] += gs
        g[4] += gs*id_centered
    end
    isfinite(lp) && all(isfinite, g) || return -Inf, zeros(length(q))
    lp, g
end
LogDensityProblems.logdensity(p::PupilProblem, q) = first(evaluate(p, q))
function LogDensityProblems.logdensity_and_gradient(p::PupilProblem, q)
    p.gradient_calls[] += 1
    evaluate(p, q)
end

function initial_position(d::PupilData)
    A = d.mean_y .- d.slope .* d.mean_x
    sigmas = sqrt.(d.sse ./ d.n)
    vcat(log(max(std(A), 1.0)), log(max(std(d.slope), 1.0)), mean(log.(sigmas)),
         0.0, A, d.slope)
end
initial_position(d, ::GaussianMean) = initial_position(d)
initial_position(d, ::StudentMixtureMean) = vcat(initial_position(d),0.0)

function to_source(q, controls, d)
    J = length(d.ids)
    r = copy(q)
    for j in 1:2J
        idx = 4+j
        loc = j <= J ? INTERCEPT_MEAN : 0.0
        ell = q[j <= J ? 1 : 2]
        r[idx] = controls[j]*loc + (q[idx]-loc)*exp((controls[j]-1)*ell)
    end
    r
end

function to_model(r, controls, d)
    J = length(d.ids)
    q = copy(r)
    for j in 1:2J
        idx = 4+j
        loc = j <= J ? INTERCEPT_MEAN : 0.0
        ell = r[j <= J ? 1 : 2]
        q[idx] = loc + (r[idx]-controls[j]*loc)*exp((1-controls[j])*ell)
    end
    q
end

end
