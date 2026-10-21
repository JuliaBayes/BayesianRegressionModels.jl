# Structural companion for the CDC `ww-inference-model`
# (github.com/CDCgov/ww-inference-model): a joint wastewater +
# hospital-admissions renewal model on BRM's StanBlocks backend. Authorized by
# decision `2026-08-26T16-44-33-638-1yz7ybe` (option B).
#
# The upstream model has four coupled pieces; the comments below identify which
# semantics this executable structural companion preserves.
#
#  1. INFECTION PROCESS. Reference-subpopulation weekly unadjusted log-Rᵘ follows a
#     differenced autoregression, `log Rᵘ(tᵢ)=log Rᵘ(tᵢ₋₁)+β·Δ+σ_r ε` (`dar_logru`).
#     The realized reproduction number carries INFECTION FEEDBACK,
#     `R(t)=Rᵘ(t)·exp(−γ Σ_τ I(t−τ)g(τ))`, and incidence follows the renewal
#     equation with a 50-day exponential seeding phase (`renewal_feedback`).
#
#  2. MULTI-SUBPOPULATION HIERARCHY. K catchment subpopulations (+ a reference)
#     each carry `log Rᵘ_k = log Rᵘ_0 + m + δ_k(t)` with δ_k an AR(1) deviation
#     (`ar1_dev`); each runs its own renewal, and jurisdiction incidence is the
#     population-weighted aggregate `I(t)=Σ_k w_k I_k(t)`.
#
#  3. WASTEWATER OBSERVATION (per subpopulation/site). Genome concentration is the
#     shedding-kinetics convolution of that subpopulation's infections scaled by
#     genomes-per-infection / per-person volume (`shed_convolve`, `log_shed_scale`);
#     site-lab measurements are log-normal with a site scaling `M_k` and a
#     hierarchical observation SD `σ_c,k`, with below-limit-of-detection
#     left-censoring.
#
#  4. HOSPITAL-ADMISSIONS OBSERVATION (jurisdiction). Expected admissions convolve
#     aggregate infections with an infection-to-hospitalization delay
#     (`delay_convolve`), scaled by a time-varying AR(1)-logit IHR and a day-of-week
#     Dirichlet effect; counts are negative-binomial.
#
# The carried-state parts (renewal, AR processes, all convolutions) live in
# `@deffun`s and are called loop-free from the `@slic` body — `plate` broadcasts the
# per-subpopulation renewal+wastewater cell over independent subpopulations, and the
# aggregate + hospitalizations are ordinary top-level statements on the collected
# infection matrix.
#
# Verified: transpile + stanc + finite BridgeStan density/gradient (dim=379 on the
# default fixture); see `test/cdc_ww_inference.jl`. NOT sampled — the deliverable is
# the model. The generation-interval / shedding / delay PMFs and priors are
# illustrative discretizations, not calibrated values.

using BayesianRegressionModels
using StanBlocks
using Distributions: Beta, Dirichlet, LogNormal

StanBlocks.@deffun begin
    # CDC's global process: an AR(1) on first differences followed by cumulative
    # summation. `eps` has one fewer element than the returned weekly trajectory.
    dar_logru(logru1::real, eps::vector[ni], beta::real, sigma::real,
              n_weeks::int)::vector[n_weeks] = begin
        x::vector[n_weeks]
        diff::vector[ni]
        x[1] = logru1
        if n_weeks >= 2
            diff[1] = sigma * eps[1]
            x[2] = x[1] + diff[1]
        end
        for i in 3:n_weeks
            diff[i - 1] = beta * diff[i - 2] + sigma * eps[i - 1]
            x[i] = x[i - 1] + diff[i - 1]
        end
        x
    end
    # Stationary AR(1) deviation around a supplied mean trajectory. CDC uses the
    # stationary initial scale for subpopulation R and IHR deviations.
    ar1_dev(eps::vector[nt], phi::real, sigma::real)::vector[nt] = begin
        d::vector[nt]
        d[1] = sigma * eps[1] / sqrt(1.0 - square(phi))
        for t in 2:nt
            d[t] = phi * d[t - 1] + sigma * eps[t]
        end
        d
    end
    # Renewal with infection feedback + exponential seeding.
    renewal_feedback(logRt::vector[nt], g::vector[ng], gamma::real, I0::real,
                     r::real, n_seed::int)::vector[nt] = begin
        I::vector[nt]
        for t in 1:nt
            if t <= n_seed
                I[t] = I0 * exp(r * (t - 1))
            else
                conv = 0.0
                for s in 1:ng
                    prev = t - s >= 1 ? I[t - s] : 0.0
                    conv = conv + g[s] * prev
                end
                Rt = exp(logRt[t]) * exp(-gamma * conv)
                I[t] = Rt * conv
            end
        end
        I
    end
    # CDC parameterizes incidence by the per-capita value on the first observed
    # day, then back-calculates the beginning of the unobserved growth period.
    renewal_from_first_observed(logRt::vector[nt], g::vector[ng], gamma::real,
                                i_first_obs::real, growth::real,
                                uot::int)::vector[nt] =
        renewal_feedback(logRt, g, gamma,
                         exp(log(i_first_obs) - uot * growth), growth, uot)
    # Normalized triangular shedding trajectory on the log10 viral-load scale,
    # matching CDC's `get_vl_trajectory` recurrence.
    viral_shedding_trajectory(t_peak::real, viral_peak::real,
                              duration_shedding::real, n::int)::vector[n] = begin
        s::vector[n]
        growth = viral_peak / t_peak
        wane = viral_peak / (duration_shedding - t_peak)
        for t in 1:n
            if t <= t_peak
                s[t] = exp(log(10.0) * growth * t)
            else
                log10_load = viral_peak + wane * t_peak - wane * t
                s[t] = exp(log(10.0) * (log10_load < 0.0 ? 0.0 : log10_load))
            end
        end
        s / sum(s)
    end
    # Shedding-load convolution: C(t)=Σ_{k=1}^{nsh} s(k) I(t-k+1).
    shed_convolve(I::vector[nt], s::vector[nsh])::vector[nt] = begin
        c::vector[nt]
        for t in 1:nt
            acc = 0.0
            for k in 1:nsh
                idx = t - k + 1
                acc = acc + (idx >= 1 ? s[k] * I[idx] : 0.0)
            end
            c[t] = acc
        end
        c
    end
    # Infection->outcome delay convolution: L(t)=Σ_{k=1}^{nd} d(k) I(t-k+1) (d[1]=lag 0).
    delay_convolve(I::vector[nt], d::vector[nd])::vector[nt] = begin
        l::vector[nt]
        for t in 1:nt
            acc = 0.0
            for k in 1:nd
                idx = t - k + 1
                acc = acc + (idx >= 1 ? d[k] * I[idx] : 0.0)
            end
            l[t] = acc
        end
        l
    end
    # Jurisdiction aggregate: population-weighted sum across subpopulation columns
    # of the collected infection matrix. `I_mat` is [nt x K] (one column per subpop),
    # `w` the population weights; `I_mat * w` is Stan matrix-vector product -> vector[nt].
    # (Used by the @brm surface form below, where `*` at the formula level is the
    # Wilkinson interaction operator, not matmul — so the matmul lives here.)
    wsum(I_mat::matrix[m, K], w::vector[K])::vector[m] = I_mat * w
    # Expand one weekly latent value onto a daily row axis through a data index.
    weekly_expand(x::vector[nw], week_idx::int[nt])::vector[nt] = x[week_idx]
    weekly_expand_columns(x::matrix[nw, S], week_idx::int[nt])::matrix[nt, S] = begin
        out::matrix[nt, S]
        for t in 1:nt
            for stream in 1:S
                out[t, stream] = x[week_idx[t], stream]
            end
        end
        out
    end
    # Build sparse wastewater-record means from the latent subpopulation matrix.
    # Records own independent time, subpopulation and lab-site mappings, so a
    # latent reference/uncovered population need not have a wastewater series.
    ww_expected_log(I_mat::matrix[nt, K], sh::vector[nsh],
                    sample_time::int[n], sample_subpop::int[n],
                    sample_lab::int[n], log_lab_mod::vector[nlab],
                    log10_g::real, mwpd::real)::vector[n] = begin
        out::vector[n]
        for i in 1:n
            shed = 0.0
            for lag in 1:nsh
                t = sample_time[i] - lag + 1
                shed = shed + (t >= 1 ? sh[lag] * I_mat[t, sample_subpop[i]] : 0.0)
            end
            out[i] = log(10.0) * log10_g + log(shed + 1e-8) - log(mwpd) +
                     log_lab_mod[sample_lab[i]]
        end
        out
    end
    gather_exp(x::vector[nlab], idx::int[n])::vector[n] = exp(x[idx])
    take_window(x::vector[n], start_idx::int, width::int)::vector[width] =
        x[start_idx:(start_idx + width - 1)]
    scale_simplex(x::vector[K], scale::real)::vector[K] = scale * x
    hospital_daily_mean(lat::vector[nt], logit_p::vector[nt], dow::int[nt],
                        dow_effect::vector[7], npop::real)::vector[nt] =
        npop * (dow_effect[dow] .* inv_logit(logit_p) .* lat)
    # Generic count observation component. Each record names a stream and an
    # inclusive time interval. Each stream supplies its own subpopulation
    # weights, delay PMF, population multiplier, and weekly rate trajectory.
    count_interval_mean(I_mat::matrix[nt, K], subpop_weights::matrix[K, S],
                        delay::matrix[nd, S], logit_rate::matrix[nt, S],
                        interval_start::int[n], interval_stop::int[n],
                        stream_idx::int[n], dow::int[nt],
                        dow_effect::vector[7], population::vector[S])::vector[n] = begin
        out::vector[n]
        for record in 1:n
            stream = stream_idx[record]
            expected = 0.0
            for outcome_time in interval_start[record]:interval_stop[record]
                delayed = 0.0
                for lag in 1:nd
                    infection_time = outcome_time - lag + 1
                    if infection_time >= 1
                        stream_incidence = 0.0
                        for subpop in 1:K
                            stream_incidence = stream_incidence +
                                I_mat[infection_time, subpop] *
                                subpop_weights[subpop, stream]
                        end
                        delayed = delayed + delay[lag, stream] *
                            inv_logit(logit_rate[infection_time, stream]) *
                            stream_incidence
                    end
                end
                expected = expected + population[stream] *
                    dow_effect[dow[outcome_time]] * delayed
            end
            out[record] = expected + 1e-8
        end
        out
    end
    # Expected admissions from the aggregate latent infections: day-of-week multiplier
    # exp(log_dow), AR(1)-logit IHR inv_logit(logit_p), and jurisdiction population.
    ihr_scale(lat::vector[nt], logit_p::vector[nt], log_dow::vector[nt],
              npop::real)::vector[nt] = npop * (exp(log_dow) .* inv_logit(logit_p) .* lat)
end

"""
    cdc_ww_inference_fixture(; K = 3, nt = 84, n_weeks = 12, n_seed = 15)

A minimal jurisdiction dataset: `K` catchment subpopulations over `nt` daily
timepoints. `ww` is a ragged per-subpopulation log-concentration series; `h` is the
daily hospital-admissions count. `g`/`s`/`d` are the (fixed) generation-interval,
shedding-kinetics and infection-to-hospitalization delay PMFs; `w` are the
population weights; `lloq` the log limit of detection; `n_pop` the jurisdiction
population; `week_idx`/`dow` map days to weeks / day-of-week.
"""
function cdc_ww_inference_fixture(; K = 3, nt = 84, n_weeks = 12, n_seed = 15)
    g = [0.02, 0.06, 0.12, 0.16, 0.16, 0.14, 0.11, 0.08, 0.06, 0.04, 0.03, 0.02]
    s = [0.01, 0.05, 0.12, 0.17, 0.17, 0.14, 0.11, 0.08, 0.06, 0.05, 0.03, 0.01]
    d = [0.01, 0.03, 0.06, 0.11, 0.14, 0.13, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.03, 0.01]
    (;
        ww = [[log(30.0) + 0.7 * exp(-((t - 45.0) / 15)^2) + 0.05 * sin(t / 4) for t in 1:nt]
              for _ in 1:K],
        h = [max(0, round(Int, 20 + 40 * exp(-((t - 50.0) / 14)^2))) for t in 1:nt],
        g = g, s = s, d = d,
        week_idx = [min(n_weeks, div(t - 1, 7) + 1) for t in 1:nt],
        dow = [((t - 1) % 7) + 1 for t in 1:nt],
        w = (ws = [K - i + 1.0 for i in 1:K]; ws ./ sum(ws)),   # normalized population weights
        lloq = log(5.0), n_pop = 500000.0,
        nt = nt, n_weeks = n_weeks, n_innov = n_weeks - 1,
        ng = length(g), nsh = length(s), nd = length(d),
        K = K, n_seed = n_seed,
    )
end

"""
    cdc_ww_inference_model([data])

A StanBlocks-native structural companion to CDC's `ww-inference-model`: multi-subpopulation
differenced-AR Rt with infection feedback, per-subpopulation censored wastewater
measurement, and a joint negative-binomial hospital-admissions stream. Swap data with
`cdc_ww_inference_model()(; h = new_h, …)`.
"""
function cdc_ww_inference_model(data = cdc_ww_inference_fixture())
    @slic data begin
        # reference-subpopulation weekly differenced-AR log-Rᵘ
        logru1 ~ normal(0.0, 0.5)
        beta   ~ normal(0.5, 0.2; lower = 0.0, upper = 1.0)
        sigma_r ~ normal(0.0, 0.2; lower = 0.0)
        eps_r::vector[n_innov] ~ std_normal()
        logru_daily = dar_logru(logru1, eps_r, beta, sigma_r, n_weeks)[week_idx]
        # shared subpopulation-hierarchy hyperparameters
        m_int ~ normal(0.0, 0.5)
        gamma ~ lognormal(-4.0, 0.5)                    # infection feedback
        phi_delta ~ normal(0.1, 0.05; lower = 0.0, upper = 1.0)
        sigma_delta ~ normal(0.0, 0.2; lower = 0.0)
        r_growth ~ normal(0.0, 0.05)                    # seeding growth rate
        # wastewater measurement hyperparameters
        log_shed_scale ~ normal(0.0, 1.0)               # log(genomes / per-person volume)
        sigma_m ~ normal(0.0, 0.25; lower = 0.0)
        log_sigma_c_hat ~ normal(0.0, 1.0)
        sigma_log_sigma_c ~ normal(0.0, 0.7; lower = 0.0)
        # per-subpopulation renewal + wastewater observation; collect I_k trajectories
        I_mat ~ plate(ww; outer = (K,)) do wwi
            log_I0 ~ normal(-6.0, 1.0)
            eps_d::vector[nt] ~ std_normal()
            delta_k = ar1_dev(eps_d, phi_delta, sigma_delta)
            logRt_k = logru_daily + (m_int + delta_k)
            I_k = renewal_feedback(logRt_k, g, gamma, exp(log_I0), r_growth, n_seed)
            logM_k ~ normal(0.0, sigma_m)
            log_sigma_c_k ~ normal(log_sigma_c_hat, sigma_log_sigma_c)
            C_k = shed_convolve(I_k, s) * exp(log_shed_scale)
            wwi ~ censored(normal, log(C_k) + logM_k, exp(log_sigma_c_k); lower = lloq)
            I_k
        end
        # jurisdiction aggregate infections
        I_agg = I_mat * w
        # hospital admissions: delay convolution + AR(1) logit IHR + day-of-week + NB
        logit_ihr0 ~ normal(-4.6, 0.3)
        phi_H ~ normal(0.01, 0.01; lower = 0.0, upper = 1.0)
        sigma_H ~ normal(0.0, 0.01; lower = 0.0)
        eps_H::vector[nt] ~ std_normal()
        logit_ph = logit_ihr0 + ar1_dev(eps_H, phi_H, sigma_H)
        p_hosp = inv_logit(logit_ph)
        omega_raw::simplex[7] ~ dirichlet(rep_vector(5.0, 7))
        omega = 7.0 * omega_raw                         # day-of-week effect, mean 1
        lat = delay_convolve(I_agg, d)
        H = (omega[dow] .* p_hosp) .* lat
        phi_h ~ normal(0.0, 1.0; lower = 0.0)
        h ~ neg_binomial_2(n_pop * H, phi_h)
    end
end

# ---------------------------------------------------------------------------
# The same four-component structural model expressed on the `@brm` FORMULA surface.
#
# Every coupled piece is composed at formula level, while deterministic scans live
# in typed `@deffun`s. The shared reference log-Rᵘ uses the same differenced-AR
# recurrence as the @slic companion. The weekday multiplier is a directly sampled
# simplex scaled to mean one.
#
# How each formula seam carries the model:
#   - shared weekly log-Rᵘ ............. `log_ru_week ~ 1 + dar(week_grid; p=1)`,
#                                        expanded daily and closed over in the cell;
#   - infection feedback + renewal ..... `renewal_from_first_observed` @deffun;
#   - multi-subpopulation hierarchy .... `kernel(...) do ... end` broadcasts the
#                                        per-subpop latent renewal cell; stationary
#                                        AR(1) deviations are drawn IN the cell
#                                        (`eps_d ~ std_normal()` + `ar1_dev`); the
#                                        hierarchical initial incidence and growth
#                                        LPs derive the grouping; cells collect;
#   - sparse wastewater ................ record mappings gather from `I_mat`, then
#                                        a lab-hierarchical censored Normal is fit;
#   - count streams .................... subpopulation-weight matrix + stream delay
#                                        PMF + weekly AR-logit rate + inclusive
#                                        interval aggregation + `NegativeBinomial2`;
#   - forecast carriers ................ named infection, count-mean, and WW-mean
#                                        generated quantities on the same mappings.
#
# Verified: transpile + stanc + finite BridgeStan density/gradient; see
# `test/cdc_ww_inference.jl`. NOT sampled — the deliverable is the model.
"""
    cdc_ww_brm_fixture(; K = 3, nt = 84, uot = 50, ht = 14)

Multi-frame fixture for [`cdc_ww_brm_model`](@ref). `K` is the number of latent
subpopulations and includes an uncovered reference population at index one. The
latent kernel has one row per subpopulation, while wastewater measurements occupy
their own sparse record frame with explicit time, subpopulation, lab-site and LOD
mappings. Several lab sites may map to one catchment and repeated sampling times are
valid. Count records occupy a second independent frame: each names a stream and an
inclusive interval, while stream-level matrices map latent subpopulations and delay
kernels. Matching forecast mappings reuse both observation components without
joining the calibration likelihood.
"""
function cdc_ww_brm_fixture(; K = 3, nt = 84, uot = 50, ht = 14)
    K >= 2 || error("cdc_ww_brm_fixture needs at least a reference and one sampled subpopulation")
    nt >= 14 || error("cdc_ww_brm_fixture needs at least 14 observed days")
    uot >= 1 || error("cdc_ww_brm_fixture needs a positive unobserved seeding window")
    ht >= 1 || error("cdc_ww_brm_fixture needs a positive forecast horizon")
    g  = [0.02, 0.06, 0.12, 0.16, 0.16, 0.14, 0.11, 0.08, 0.06, 0.04, 0.03, 0.02]
    sh = [0.01, 0.05, 0.12, 0.17, 0.17, 0.14, 0.11, 0.08, 0.06, 0.05, 0.03, 0.01]
    dl = [0.01, 0.03, 0.06, 0.11, 0.14, 0.13, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.03, 0.01]
    n_total = uot + nt + ht
    n_weeks = cld(n_total, 7)
    subpopulation = [i == 1 ? "reference-uncovered" : "catchment-$(i - 1)" for i in 1:K]
    subpop_size = collect(range(2.0, 1.0; length=K))
    subpop_weight = subpop_size ./ sum(subpop_size)

    # At least two lab sites map to the first sampled catchment. This creates
    # repeated (subpopulation, time) records without duplicating latent states.
    n_labs = K
    ww_lab = ["lab-$i" for i in 1:n_labs]
    lab_to_subpop = [i <= 2 ? 2 : min(K, i) for i in 1:n_labs]
    ww_time = Int[]
    ww_subpop = Int[]
    ww_obs_lab = Int[]
    ww_lod = Float64[]
    ww_log_conc = Float64[]
    for lab in 1:n_labs, observed_day in 1:7:nt
        push!(ww_time, uot + observed_day)
        push!(ww_subpop, lab_to_subpop[lab])
        push!(ww_obs_lab, lab)
        lod = log(2.0 + 0.25 * lab + 0.01 * observed_day)
        latent_signal = log(15.0 + 8.0 * exp(-((observed_day - nt / 2) / 15)^2)) +
                        0.08 * lab
        push!(ww_lod, lod)
        # A value exactly at its row-specific LOD selects the censored branch.
        push!(ww_log_conc, (observed_day + lab) % 4 == 0 ? lod : max(lod + 0.05, latent_signal))
    end

    # Count modules are separate from both the latent and wastewater axes. The
    # first stream is daily jurisdiction admissions; the second is a weekly
    # sampled-catchment stream with a distinct population map and delay PMF.
    count_stream = ["jurisdiction-hospital-daily", "sampled-catchments-weekly"]
    n_count_streams = length(count_stream)
    sampled_weights = vcat(0.0, subpop_size[2:end] ./ sum(subpop_size[2:end]))
    count_subpop_weights = hcat(subpop_weight, sampled_weights)
    shifted_delay = vcat(0.0, dl[1:(end - 1)])
    shifted_delay ./= sum(shifted_delay)
    count_delay = hcat(dl, shifted_delay)
    count_population = [500000.0, 280000.0]
    count_start = Int[]
    count_stop = Int[]
    count_obs_stream = Int[]
    count = Int[]
    for observed_day in 1:nt
        push!(count_start, uot + observed_day)
        push!(count_stop, uot + observed_day)
        push!(count_obs_stream, 1)
        push!(count, max(0, round(Int, 20 + 40 * exp(-((observed_day - nt / 2) / 14)^2))))
    end
    for observed_day in 1:7:nt
        width = min(7, nt - observed_day + 1)
        push!(count_start, uot + observed_day)
        push!(count_stop, uot + observed_day + width - 1)
        push!(count_obs_stream, 2)
        push!(count, max(0, round(Int, width * (8 + 14 * exp(-((observed_day - nt / 2) / 16)^2)))))
    end

    forecast_count_start = Int[]
    forecast_count_stop = Int[]
    forecast_count_obs_stream = Int[]
    for forecast_day in (nt + 1):(nt + ht)
        push!(forecast_count_start, uot + forecast_day)
        push!(forecast_count_stop, uot + forecast_day)
        push!(forecast_count_obs_stream, 1)
    end
    for forecast_day in (nt + 1):7:(nt + ht)
        width = min(7, nt + ht - forecast_day + 1)
        push!(forecast_count_start, uot + forecast_day)
        push!(forecast_count_stop, uot + forecast_day + width - 1)
        push!(forecast_count_obs_stream, 2)
    end

    forecast_ww_time = Int[]
    forecast_ww_subpop = Int[]
    forecast_ww_obs_lab = Int[]
    for lab in 1:n_labs, forecast_day in (nt + 1):7:(nt + ht)
        push!(forecast_ww_time, uot + forecast_day)
        push!(forecast_ww_subpop, lab_to_subpop[lab])
        push!(forecast_ww_obs_lab, lab)
    end

    (;
        subpopulation,
        is_reference = [i == 1 ? 1.0 : 0.0 for i in 1:K],
        t_grid = [collect(1.0:n_total) for _ in 1:K],
        time_grid = collect(1.0:n_total),
        week_grid = collect(1.0:n_weeks),
        week_idx = [min(n_weeks, div(t - 1, 7) + 1) for t in 1:n_total],
        dow = [mod(t - uot - 1, 7) + 1 for t in 1:n_total],
        ww_lab, lab_to_subpop, ww_time, ww_subpop, ww_obs_lab, ww_lod, ww_log_conc,
        forecast_ww_time, forecast_ww_subpop, forecast_ww_obs_lab,
        count_stream,
        count_week_grid = [collect(1.0:n_weeks) for _ in 1:n_count_streams],
        count_subpop_weights, count_delay, count_population,
        count_start, count_stop, count_obs_stream, count,
        forecast_count_start, forecast_count_stop, forecast_count_obs_stream,
        g = g, sh = sh, dl = dl, nsh = length(sh),
        w = subpop_weight,
        uot, nt, ht, n_total, n_weeks, K, n_labs, n_count_streams,
        mwpd = 757.0,
        n_pop = 500000.0,
    )
end

"""
    cdc_ww_brm_model([df]) -> BRMI

An executable structural port of CDC's four-component model on the `@brm` formula
surface, returned as a `BRMI`; lower with `SBBRMI(cdc_ww_brm_model())`. See the
block comment above for the current parity boundary. Re-bind data with
`cdc_ww_brm_model(cdc_ww_brm_fixture(; K = 2, nt = 56))`.
"""
function cdc_ww_brm_model(df = cdc_ww_brm_fixture())
    @brm df begin
        # Shared epidemic parameters. `dar` owns the zero-start differenced-AR
        # innovations while the intercept remains the initial weekly log-Rᵘ level.
        gamma       ~ LogNormal(-4.0, 0.5)
        phi_delta   ~ Beta(2.0, 8.0)
        sigma_delta ~ Exponential(4.0)
        log10_g     ~ Normal(12.0, 1.0)
        t_peak      ~ LogNormal(log(4.0), 0.25)
        viral_peak  ~ Normal(6.0, 1.0)
        shed_tail   ~ LogNormal(log(14.0), 0.3)
        dur_shed = t_peak + shed_tail
        shedding = viral_shedding_trajectory(t_peak, viral_peak, dur_shed, nsh)

        log_ru_week ~ 1 + dar(week_grid; p = 1)
        effect(log_ru_week, Intercept) ~ Normal(0.0, 0.5)
        log_ru = weekly_expand(log_ru_week, week_idx)

        # Hierarchical incidence and initial growth are defined over the latent
        # subpopulation axis, including the uncovered reference population.
        logit_I0      ~ 1 + (1 | initial | subpopulation)
        initial_growth ~ 1 + (1 | growth | subpopulation)
        effect(logit_I0, Intercept) ~ Normal(-9.0, 1.0)
        effect(initial_growth, Intercept) ~ Normal(0.0, 0.05)
        sd(:, initial) ~ Normal(0.0, 0.75)
        sd(:, growth) ~ Normal(0.0, 0.05)

        # The kernel produces latent trajectories only. Observation frames are
        # mapped onto this matrix after the cell has been collected.
        I_mat ~ kernel(t_grid, is_reference, logit_I0, initial_growth) do ts, isref, lI0, growth_i
            eps_d::vector[n_total] ~ std_normal()
            delta_k = ar1_dev(eps_d, phi_delta, sigma_delta)
            logRt_k = log_ru + (1.0 - isref) * delta_k
            renewal_from_first_observed(logRt_k, g, gamma,
                                        inv_logit(lI0), growth_i, uot)
        end

        # Sparse wastewater records have independent time, subpopulation, lab
        # and record-specific LOD mappings. Multiple labs may observe the same
        # catchment/time pair; the reference population may have no records.
        log_lab_mod ~ 0 + (1 | ww_scale | ww_lab)
        log_sigma_ww ~ 1 + (1 | ww_noise | ww_lab)
        effect(log_sigma_ww, Intercept) ~ Normal(log(0.35), 0.5)
        sd(:, ww_scale) ~ Normal(0.0, 0.5)
        sd(:, ww_noise) ~ Normal(0.0, 0.35)
        ww_mu = ww_expected_log(I_mat, shedding, ww_time, ww_subpop, ww_obs_lab,
                                log_lab_mod, log10_g, mwpd)
        ww_sigma = gather_exp(log_sigma_ww, ww_obs_lab)
        ww_log_conc ~ censored(Normal(ww_mu, ww_sigma); lower = ww_lod)

        # jurisdiction aggregate infections (population-weighted across subpops)
        I_agg = wsum(I_mat, w)

        # Count-stream module. Each stream gets a hierarchical rate center and a
        # stationary weekly AR deviation. Observation rows can be daily or span
        # arbitrary inclusive intervals and can map different latent catchments.
        phi_count ~ Beta(2.0, 8.0)
        sigma_count ~ Exponential(20.0)
        count_rate_center ~ 1 + (1 | count_rate | count_stream)
        effect(count_rate_center, Intercept) ~ Normal(-4.6, 0.3)
        sd(:, count_rate) ~ Normal(0.0, 0.5)
        count_rate_week ~ kernel(count_week_grid, count_rate_center) do weeks, center
            eps_count::vector[n_weeks] ~ std_normal()
            center + ar1_dev(eps_count, phi_count, sigma_count)
        end
        count_rate_daily = weekly_expand_columns(count_rate_week, week_idx)
        dow_share ~ Dirichlet(7, 5.0)
        dow_effect = scale_simplex(dow_share, 7.0)
        log_phi_count ~ 1 + (1 | count_dispersion | count_stream)
        effect(log_phi_count, Intercept) ~ Normal(log(15.0), 0.5)
        sd(:, count_dispersion) ~ Normal(0.0, 0.5)
        count_mu = count_interval_mean(I_mat, count_subpop_weights, count_delay,
                                       count_rate_daily, count_start, count_stop,
                                       count_obs_stream, dow, dow_effect,
                                       count_population)
        count_phi = gather_exp(log_phi_count, count_obs_stream)
        count ~ NegativeBinomial2(count_mu, count_phi)

        # Forecast contract: named deterministic carriers use the same fitted
        # latent state and observation mappings but are not calibration outcomes.
        forecast_infections = take_window(I_agg, uot + nt + 1, ht)
        forecast_count_mean = count_interval_mean(
            I_mat, count_subpop_weights, count_delay, count_rate_daily,
            forecast_count_start, forecast_count_stop,
            forecast_count_obs_stream, dow, dow_effect, count_population)
        forecast_ww_log_mean = ww_expected_log(
            I_mat, shedding, forecast_ww_time, forecast_ww_subpop,
            forecast_ww_obs_lab, log_lab_mod, log10_g, mwpd)
    end
end

"""Reusable generative declaration plan for [`cdc_ww_brm_model`](@ref)."""
cdc_ww_brm_plan(df = cdc_ww_brm_fixture()) =
    generative_plan(cdc_ww_brm_model, df; mod = @__MODULE__)

"""Executable semantic descriptor, including named forecast carriers."""
cdc_ww_brm_descriptor(df = cdc_ww_brm_fixture()) =
    brm_descriptor(cdc_ww_brm_model, df; mod = @__MODULE__, name = :cdc_wastewater)
