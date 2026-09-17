"""
    EpiRenewal

An example package on top of BayesianRegressionModels.jl: the mechanics of a
renewal-equation epidemic model as operators for `@brm` top-level assignments.
BayesianRegressionModels knows nothing about epidemics; the statistical parts of a
model stay formula lines (`log_R ~ 1 + rw(time)`), and this package supplies what
sits between the reproduction number and the counts:

    past = seeded_history(log_I0, exp(log_R), gen_pmf, 14)   # infections before day 1
    I    = renewal(gen_pmf, exp(log_R), past)                # I[t] = R[t] * sum_s gen[s] * I[t - s]
    Y    = delay(delay_pmf, I, past)                         # Y[t] = sum_d pmf[d] * I[t - d]
    cases ~ nb_cases(Y, cluster, observed)

Every operator takes its kernel FIRST, as a data vector or as a function, so a
`do` block works and its body may read sampled parameters:

    Y = delay(I, past, 15) do d
        censored_lognormal_mass(d, mu_delay, sigma_delay, 15)
    end

The operators are Stan functions (`StanBlocks.@deffun`); a model that uses them
is built with `SBBRMI(model; mod=@__MODULE__)` from a module that has
`using EpiRenewal`.

## The row-order contract

The operators see plain vectors, so the frame has to be laid out the way they
read it: **one row per (group, day), ordered by group and then by day, every
group covering the same equally spaced days.** [`epi_frame`](@ref) sorts a table
into that order and refuses one that cannot satisfy it. With several groups
("patches") a row vector is read as a `T × P` matrix, patch `g` in column `g`.
"""
module EpiRenewal

using StanBlocks
using Distributions: UnivariateDistribution, cdf
using Statistics: mean

export censored_pmf, epi_frame
export growth_rate, seeded_history, renewal, delay, gravity, per_patch
export censored_lognormal_mass, nb_cases, censored_delay

# ── Julia side ────────────────────────────────────────────────────────────────

"""
    censored_pmf(dist; D, drop_zero=false, nq=400)

Daily probability masses of a continuous delay `dist` when both ends are only
known to the day: `p_d = P(d <= P + T < d + 1 | P + T < D)` for `d = 0, …, D - 1`,
with `P ~ Uniform(0, 1)` the time of the first event within its day and
`T ~ dist`. `drop_zero=true` removes lag 0 and renormalises (a generation
interval: nobody infects on the day of their own infection).
"""
function censored_pmf(dist::UnivariateDistribution; D::Integer, drop_zero::Bool=false, nq::Integer=400)
    D >= 1 || throw(ArgumentError("censored_pmf: `D` must be at least 1, got $D"))
    midpoints = ((1:nq) .- 0.5) ./ nq
    F(x) = mean(cdf(dist, max(x - p, 0.0)) for p in midpoints)        # P(P + T < x)
    pmf = [(F(d) - F(d - 1)) / F(D) for d in 1:D]
    drop_zero ? pmf[2:end] ./ sum(pmf[2:end]) : pmf
end

"""
    epi_frame(rows::NamedTuple; time::Symbol, by=nothing) -> NamedTuple

Sort the row columns of `rows` into the order the operators read — by `by`
(if given), then by `time` — and check the row-order contract: equal-length
columns, no duplicated `(by, time)` cell, every group covering the same days,
and equally spaced days. Anything else is an error that names the offending
group or day; a gap is never filled in silently.

Shared fields that are not one-per-row (a generation-interval vector, a distance
matrix) do not belong in `rows`; `merge` them into the result.
"""
function epi_frame(rows::NamedTuple; time::Symbol, by::Union{Nothing,Symbol}=nothing)
    haskey(rows, time) || throw(ArgumentError("epi_frame: no column `$time`; columns are $(keys(rows))"))
    isnothing(by) || haskey(rows, by) || throw(ArgumentError("epi_frame: no column `$by`; columns are $(keys(rows))"))
    n = length(rows[time])
    for (name, column) in pairs(rows)
        column isa AbstractVector && length(column) == n || throw(ArgumentError(
            "epi_frame: column `$name` is not a vector with one entry per row ($n rows); " *
            "merge shared fields in after `epi_frame`"))
    end
    groups = isnothing(by) ? fill(1, n) : rows[by]
    order = sortperm(collect(zip(groups, rows[time])))
    sorted_groups, sorted_times = groups[order], rows[time][order]
    for i in 2:n
        (sorted_groups[i], sorted_times[i]) == (sorted_groups[i-1], sorted_times[i-1]) && throw(ArgumentError(
            "epi_frame: more than one row for " * (isnothing(by) ? "" : "`$by` = $(sorted_groups[i]), ") *
            "`$time` = $(sorted_times[i])"))
    end
    days = sort!(unique(sorted_times))
    steps = unique(diff(days))
    length(steps) <= 1 || throw(ArgumentError(
        "epi_frame: `$time` is not equally spaced (steps $(steps)); the renewal recursion needs every day"))
    for group in unique(sorted_groups)
        mine = sorted_times[sorted_groups .== group]
        mine == days || throw(ArgumentError(
            "epi_frame: " * (isnothing(by) ? "the frame" : "`$by` = $group") * " does not cover the same " *
            "`$time` values as the rest (missing: $(setdiff(days, mine)))"))
    end
    NamedTuple{keys(rows)}(Tuple(column[order] for column in values(rows)))
end

# ── Stan side: the operators and the observation families ─────────────────────

StanBlocks.@deffun begin
    clamp_between(x::real, lo::real, hi::real)::real = x < lo ? lo : (x > hi ? hi : x)

    # growth rate r implied by a reproduction number R (Euler–Lotka, two Newton steps from r = 0)
    growth_rate(R::real, gen_pmf::vector[G])::real = begin
        r = 0.0
        for iter in 1:2
            f = -1.0 / R
            df = 0.0
            for s in 1:G
                f += gen_pmf[s] * exp(-r * s)
                df -= s * gen_pmf[s] * exp(-r * s)
            end
            r = r - f / df
        end
        clamp_between(r, -2.0, 2.0)
    end

    # a path at day s: the path itself from day 1 on, its history before;
    # `past[h]` is day h - H, so `past[H]` is day 0; beyond the history it is 0
    at_day(x::vector[T], past::vector[H], s::int)::real =
        s >= 1 ? x[s] : (s + H >= 1 ? past[s + H] : 0.0)

    # ── seeding: infections before day 1 grow at the rate the first R implies ──
    # past[h] = exp(log_I0) * exp(r * (s - 1)) at day s = h - H
    seeded_history(log_I0::real, R::vector[T], gen_pmf::vector[G], H::int)::vector[H] = begin
        r = growth_rate(R[1], gen_pmf)
        past::vector[H]
        for h in 1:H
            past[h] = exp(log_I0) * exp(r * (h - H - 1))
        end
        past
    end
    # several patches: one seed per patch, R read as T × P (rows ordered by patch, then day)
    seeded_history(log_I0::vector[P], R::vector[N], gen_pmf::vector[G], H::int)::matrix[H, P] = begin
        T = N / P
        past::matrix[H, P]
        for g in 1:P
            r = growth_rate(R[(g - 1) * T + 1], gen_pmf)
            for h in 1:H
                past[h, g] = exp(log_I0[g]) * exp(r * (h - H - 1))
            end
        end
        past
    end

    # ── renewal: I[t] = R[t] * sum_{s = 1}^{G} gen[s] * I[t - s] ──
    renewal(gen_pmf::vector[G], R::vector[T], past::vector[H])::vector[T] = begin
        I::vector[T]
        for t in 1:T
            force = 0.0
            for s in 1:G
                force += gen_pmf[s] * at_day(I, past, t - s)
            end
            I[t] = clamp_between(R[t] * force, 0.0, 1e15)
        end
        I
    end
    # the generation interval as a function of the lag s = 1 .. G
    renewal(gen, R::vector[T], past::vector[H], G::int)::vector[T] = begin
        I::vector[T]
        for t in 1:T
            force = 0.0
            for s in 1:G
                force += gen(s) * at_day(I, past, t - s)
            end
            I[t] = clamp_between(R[t] * force, 0.0, 1e15)
        end
        I
    end
    # coupled patches: I[t, g] = R[t, g] * sum_h K[g, h] * sum_s gen[s] * I[t - s, h]
    renewal(gen_pmf::vector[G], R::vector[N], past::matrix[H, P], K::matrix[P, P])::vector[N] = begin
        T = N / P
        Rm = to_matrix(R, T, P)
        I::matrix[T, P]
        force::vector[P]
        for t in 1:T
            for h in 1:P
                force[h] = 0.0
                for s in 1:G
                    force[h] += gen_pmf[s] * at_day(col(I, h), col(past, h), t - s)
                end
            end
            for g in 1:P
                pressure = 0.0
                for h in 1:P
                    pressure += K[g, h] * force[h]
                end
                I[t, g] = clamp_between(Rm[t, g] * pressure, 0.0, 1e15)
            end
        end
        to_vector(I)
    end

    # ── delay: Y[t] = sum_{d = 0}^{D - 1} pmf[d] * I[t - d] ──
    delay(pmf::vector[D], I::vector[T], past::vector[H])::vector[T] = begin
        Y::vector[T]
        for t in 1:T
            acc = 0.0
            for d in 0:(D - 1)
                acc += pmf[d + 1] * at_day(I, past, t - d)
            end
            Y[t] = acc
        end
        Y
    end
    # the delay distribution as a function of the lag d = 0 .. D - 1
    delay(pmf, I::vector[T], past::vector[H], D::int)::vector[T] = begin
        Y::vector[T]
        for t in 1:T
            acc = 0.0
            for d in 0:(D - 1)
                acc += pmf(d) * at_day(I, past, t - d)
            end
            Y[t] = acc
        end
        Y
    end
    # several patches, each delayed on its own
    delay(pmf::vector[D], I::vector[N], past::matrix[H, P])::vector[N] = begin
        T = N / P
        Im = to_matrix(I, T, P)
        Y::matrix[T, P]
        for g in 1:P
            for t in 1:T
                acc = 0.0
                for d in 0:(D - 1)
                    acc += pmf[d + 1] * at_day(col(Im, g), col(past, g), t - d)
                end
                Y[t, g] = acc
            end
        end
        to_vector(Y)
    end
    delay(pmf, I::vector[N], past::matrix[H, P], D::int)::vector[N] = begin
        T = N / P
        Im = to_matrix(I, T, P)
        Y::matrix[T, P]
        for g in 1:P
            for t in 1:T
                acc = 0.0
                for d in 0:(D - 1)
                    acc += pmf(d) * at_day(col(Im, g), col(past, g), t - d)
                end
                Y[t, g] = acc
            end
        end
        to_vector(Y)
    end

    # ── mixing between patches ──
    # gravity weights: K[g, h] is the share of patch g's infection pressure that comes from patch h;
    # `dist_flat` is the P × P distance matrix as a vector
    gravity(pop::vector[P], dist_flat::vector[PP], gamma::real)::matrix[P, P] = begin
        mean_pop = sum(pop) / P
        K::matrix[P, P]
        for g in 1:P
            total = 0.0
            for h in 1:P
                K[g, h] = g == h ? 1.0 : (pop[g] / mean_pop) * (pop[h] / mean_pop) / exp(gamma * log(dist_flat[g + (h - 1) * P]))
                total += K[g, h]
            end
            for h in 1:P
                K[g, h] = K[g, h] / total
            end
        end
        K
    end
    # a per-row predictor that is constant within a patch, as one value per patch
    # (`like` only supplies the number of patches)
    per_patch(x::vector[N], like::vector[P])::vector[P] = begin
        T = N / P
        out::vector[P]
        for g in 1:P
            out[g] = x[(g - 1) * T + 1]
        end
        out
    end

    # ── delay distributions with both ends known to the day ──
    # F(x) = P(P + T < x) for P ~ Uniform(0, 1), T ~ LogNormal(mu, sigma), in closed form: G(x) - G(x - 1)
    lnorm_G(a::real, mu::real, sigma::real)::real =
        a <= 0.0 ? 0.0 : a * Phi((log(a) - mu) / sigma) - exp(mu + 0.5 * sigma * sigma) * Phi((log(a) - mu) / sigma - sigma)
    lnorm_F(x::real, mu::real, sigma::real)::real = lnorm_G(x, mu, sigma) - lnorm_G(x - 1.0, mu, sigma)
    # P(delay = d | delay < D): the daily mass `censored_pmf(LogNormal(mu, sigma); D)` has at lag d
    censored_lognormal_mass(d::int, mu::real, sigma::real, D::int)::real =
        (lnorm_F(d + 1.0, mu, sigma) - lnorm_F(d + 0.0, mu, sigma)) / lnorm_F(D + 0.0, mu, sigma)

    # ── observation family: negative-binomial counts on the rows marked observed ──
    # cases ~ NegBin(mean Y, variance Y + cluster^2 Y^2); a row with observed == 0 is not scored,
    # and the generator draws every row, so an unobserved row's draw is its forecast
    @lhs @lpxf nb_cases_lpmf(cases::int[N], Y::vector[N], cluster::real, observed::int[N])::real = begin
        lp = 0.0
        for i in 1:N
            if observed[i] == 1
                lp += neg_binomial_2_lpmf(cases[i], Y[i], 1.0 / (cluster * cluster))
            end
        end
        lp
    end
    nb_cases_lpmfs(cases::int[N], Y::vector[N], cluster::real, observed::int[N])::vector[N] = begin
        lp::vector[N]
        for i in 1:N
            lp[i] = observed[i] == 1 ? neg_binomial_2_lpmf(cases[i], Y[i], 1.0 / (cluster * cluster)) : 0.0
        end
        lp
    end
    nb_cases_rng(int[N], Y::vector[N], cluster::real, observed::int[N])::int[N] = begin
        out::int[N]
        for i in 1:N
            out[i] = neg_binomial_2_rng(Y[i], 1.0 / (cluster * cluster))
        end
        out
    end

    # ── observation family: linelist delays, right-truncated at each event's window ──
    @lhs @lpxf censored_delay_lpmf(d::int[N], mu::real, sigma::real, window::int[N])::real = begin
        lp = 0.0
        for j in 1:N
            lp += log(censored_lognormal_mass(d[j], mu, sigma, window[j]))
        end
        lp
    end
    censored_delay_lpmfs(d::int[N], mu::real, sigma::real, window::int[N])::vector[N] = begin
        lp::vector[N]
        for j in 1:N
            lp[j] = log(censored_lognormal_mass(d[j], mu, sigma, window[j]))
        end
        lp
    end
    censored_delay_rng(int[N], mu::real, sigma::real, window::int[N])::int[N] = begin
        out::int[N]
        for j in 1:N
            W = window[j]
            probs::vector[W]
            for k in 1:W
                probs[k] = censored_lognormal_mass(k - 1, mu, sigma, W)
            end
            out[j] = categorical_rng(probs / sum(probs)) - 1
        end
        out
    end
end

end # module
