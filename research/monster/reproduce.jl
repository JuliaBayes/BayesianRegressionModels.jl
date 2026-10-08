using BayesianRegressionModels
using Distributions
using Random: AbstractRNG
using StanBlocks

# The Monster PBPK model (Gelman, Bois and Jiang 1996), as implemented in
# https://github.com/nsiccha/monster at a362738efd9c83525a2edb6db3e3646fba1a293f:
# `stan/unconstrained_monster.stan` with `cfg/nu=4/parallel_incremental_data.json`.
# README.md in this directory states the exact correspondence.

# Physiology: a line-by-line port of `simulate_person` and its helpers from
# `stan/flexible_monster.stan` (identical in `stan/unconstrained_monster.stan`),
# restricted to the Strang-splitting branch (`no_sub_steps > 0`) that the
# likelihood uses.
StanBlocks.@deffun begin
    monster_min_concentration()::real = 1e-12

    # lambert_w0(exp(earg)), linearized above 700 and below -40 to avoid
    # overflow/underflow of exp.
    monster_lambert_w0_exp(earg::real)::real = begin
        if is_nan(earg)
            return earg
        end
        if earg > 700
            y0 = lambert_w0(exp(700.0))
            return y0 + (earg - 700.0) * y0 / (y0 + 1)
        end
        if earg < -40
            return lambert_w0(exp(-40.0)) * exp(earg + 40.0)
        end
        return lambert_w0(exp(earg))
    end

    # Exact solution of dC/dt = V C / (K + C) over dt (V < 0 eliminates).
    monster_michaelis_menten_step(dt::real, C::real, V::real, K::real)::real = begin
        if C <= monster_min_concentration()
            return monster_min_concentration()
        end
        if K == 0
            return C - dt * V
        end
        earg = (dt * V + C) / K + log(C / K)
        return K * monster_lambert_w0_exp(earg)
    end

    # Geometric interpolation between two positive state vectors.
    monster_interpolate(xi::real, left::vector[n], right::vector[n])::vector[n] =
        exp((1 - xi) * log(monster_min_concentration() + left) +
            xi * log(monster_min_concentration() + right))

    # One exposure experiment for one person: inhalation until `times[1]`,
    # washout afterwards. Returns the venous-blood predictions at `times`
    # followed by the exhaled-air predictions at `times`.
    monster_experiment(
        times::vector[n_time], concentration_exposure::real,
        lean_body_mass::real, mass_fraction_fat::real,
        volume_flow_pulmonary::real,
        VPR::real, unit_volume_flow::vector[4], volume_fraction::vector[3],
        partition_coefficient_alveolar::real,
        partition_coefficient::vector[4], VMI_per_kg::real, KMI_per_l::real,
        no_sub_steps::int,
    )::vector[n_time + n_time] = begin
        no_exposure_times = 1
        body_mass = lean_body_mass / (1 - mass_fraction_fat)
        volume_fat = mass_fraction_fat * body_mass / 0.92
        volume_flow_alveolar = 0.7 * volume_flow_pulmonary
        mass_flow_exposure = volume_flow_alveolar * concentration_exposure

        volume_flow_venous = volume_flow_alveolar / VPR
        volume_flow = unit_volume_flow * volume_flow_venous
        volume::vector[4]
        volume[1] = lean_body_mass * volume_fraction[1]
        volume[2] = lean_body_mass * volume_fraction[2]
        volume[3] = volume_fat
        volume[4] = lean_body_mass * volume_fraction[3]
        effective_volume = volume .* partition_coefficient
        VMI = lean_body_mass^0.7 * VMI_per_kg / effective_volume[4]
        KMI = KMI_per_l / effective_volume[4]

        concentration_out = rep_vector(monster_min_concentration(), 4)
        FVP = volume_flow ./ effective_volume
        FPF = volume_flow_venous +
              volume_flow_alveolar / partition_coefficient_alveolar

        # Strang splitting: half a Michaelis-Menten step, an exact step of the
        # linear tissue transport (matrix exponential), another half step.
        A = add_diag(FVP * (volume_flow / FPF)', -FVP)
        A_source = mass_flow_exposure / FPF * (A \ FVP)
        last_concentration_out = concentration_out
        dt = times[1] / no_sub_steps
        transition_matrix = matrix_exp(dt * A)
        exp_A_source = transition_matrix * A_source - A_source

        all_concentration_out::vector[n_time, 4]
        last_time = 0.0
        next_time = 0.0
        time_idx = 1
        next_checkpoint = times[time_idx]
        while time_idx <= n_time
            next_time = last_time + dt
            concentration_out[4] = monster_michaelis_menten_step(
                dt / 2, concentration_out[4], -VMI, KMI)
            if time_idx <= no_exposure_times
                concentration_out = transition_matrix * concentration_out +
                                    exp_A_source
            else
                concentration_out = transition_matrix * concentration_out
            end
            concentration_out[4] = monster_michaelis_menten_step(
                dt / 2, concentration_out[4], -VMI, KMI)
            while next_time >= next_checkpoint
                all_concentration_out[time_idx] = monster_interpolate(
                    (next_checkpoint - last_time) / dt,
                    last_concentration_out, concentration_out)
                if time_idx == no_exposure_times
                    concentration_out = all_concentration_out[time_idx]
                    next_time = times[time_idx]
                end
                time_idx += 1
                if time_idx <= n_time
                    next_checkpoint = times[time_idx]
                else
                    break
                end
            end
            last_time = next_time
            last_concentration_out = concentration_out
        end

        prediction::vector[n_time + n_time]
        for t in 1:n_time
            concentration_venous = dot_product(
                unit_volume_flow, all_concentration_out[t])
            concentration_inhale = t <= no_exposure_times ?
                                   concentration_exposure : 0.0
            concentration_alveolar = (concentration_inhale + concentration_venous) /
                                     (VPR + partition_coefficient_alveolar)
            concentration_exhale = 0.7 * concentration_alveolar +
                                   0.3 * concentration_inhale
            prediction[t] = monster_min_concentration() + concentration_venous
            prediction[n_time + t] = monster_min_concentration() + concentration_exhale
        end
        prediction
    end
end

# Prior of a population geometric standard deviation on the log scale,
# tau = log(GSD), with tau^2 ~ Scaled-Inv-chi^2(nu, s^2). The source samples
# u = log(tau) - log(s) and adds scaled_inv_chi_square_lpdf(tau^2 | nu, s) +
# log(tau^2); that is exactly this density of tau, up to a constant.
struct ScaledInvChiScale <: ContinuousUnivariateDistribution
    nu::Float64
    s::Float64
end
_variance_law(d::ScaledInvChiScale) = InverseGamma(d.nu / 2, d.nu * d.s^2 / 2)
Distributions.logpdf(d::ScaledInvChiScale, tau::Real) =
    tau > 0 ? logpdf(_variance_law(d), tau^2) + log(2 * tau) : -Inf
Distributions.rand(rng::AbstractRNG, d::ScaledInvChiScale) =
    sqrt(rand(rng, _variance_law(d)))
Base.minimum(::ScaledInvChiScale) = 0.0
Base.maximum(::ScaledInvChiScale) = Inf

# Its Stan translation for SBBRMI.
BayesianRegressionModels._sb_stan_dist_name(::Type{<:ScaledInvChiScale}) =
    :scaled_inv_chi_scale
StanBlocks.@deffun begin
    @lpxf scaled_inv_chi_scale_lpdf(tau::real, nu::real, s::real)::real =
        scaled_inv_chi_square_lpdf(square(tau), nu, s) + log(2.0 * tau)
    scaled_inv_chi_scale_lpdfs(tau::real, nu::real, s::real)::real =
        scaled_inv_chi_scale_lpdf(tau, nu, s)
    scaled_inv_chi_scale_rng(nu::real, s::real)::real =
        sqrt(scaled_inv_chi_square_rng(nu, s))
end

# The six persons and two exposure experiments of the source's final data
# update. Exhaled concentrations are stored in the source's raw units and
# scaled by 1e-3 exactly as its `private_data.py` does; venous blood was not
# sampled at 245 and 270 minutes, so `venous_*_index` selects the sampled times.
function monster_data()
    t1 = [240.0, 245.0, 270.0, 360.0, 1320.0, 2760.0, 4260.0, 8580.0]
    t2 = [240.0, 245.0, 270.0, 360.0, 1320.0, 2760.0, 4260.0, 5700.0, 10020.0]
    venous_index(t) = [1; 4:length(t)]
    exhaled_index(t) = length(t) .+ (1:length(t))
    times_72 = [t1, t1, t1, t2, t2, t2]
    times_144 = [t1, t1, t2, t2, t1, t2]
    (;
        subject = ["person-1", "person-2", "person-3", "person-4", "person-5", "person-6"],
        lean_body_mass = [62.0, 71.0, 71.0, 74.0, 61.0, 61.0],
        fat_fraction = [0.114, 0.134, 0.134, 0.14, 0.09, 0.208],
        pulmonary_flow = [7.6, 11.6, 10.0, 11.3, 12.3, 8.8],
        exposure_72 = 0.488,
        exposure_144 = 0.976,
        n_substeps = 128,
        times_72,
        venous_72 = [
            [2.8, 0.92, 0.17, 0.082, 0.055, 0.018],
            [3.0, 1.2, 0.15, 0.066, 0.051, 0.020],
            [3.2, 1.16, 0.115, 0.048, 0.035, 0.015],
            [3.1, 1.3, 0.185, 0.068, 0.040, 0.037, 0.0065],
            [2.8, 1.12, 0.14, 0.068, 0.047, 0.036, 0.014],
            [2.6, 0.96, 0.105, 0.070, 0.051, 0.050, 0.025],
        ],
        venous_72_index = venous_index.(times_72),
        exhaled_72 = [
            1e-3 * [340, 99, 44, 33, 6.3, 3.45, 2.1, 0.76],
            1e-3 * [345, 101, 76, 49, 6.3, 2.7, 1.7, 0.78],
            1e-3 * [294, 118, 83, 48, 5.2, 2.55, 1.3, 0.65],
            1e-3 * [329, 117, 65, 30, 7.2, 2.6, 1.62, 1.08, 0.24],
            1e-3 * [360, 93, 44, 21, 6.8, 2.4, 1.4, 0.96, 0.38],
            1e-3 * [292, 64, 50, 23, 4.05, 2.5, 2.0, 1.45, 0.9],
        ],
        exhaled_72_index = exhaled_index.(times_72),
        times_144,
        venous_144 = [
            [5.7, 1.76, 0.36, 0.147, 0.106, 0.072],
            [8.8, 2.9, 0.36, 0.19, 0.12, 0.036],
            [6.4, 2.36, 0.260, 0.177, 0.085, 0.085, 0.024],
            [6.0, 2.48, 0.36, 0.165, 0.071, 0.064, 0.018],
            [6.4, 2.96, 0.35, 0.19, 0.105, 0.05],
            [6.0, 1.76, 0.245, 0.16, 0.12, 0.098, 0.052],
        ],
        venous_144_index = venous_index.(times_144),
        exhaled_144 = [
            1e-3 * [632, 219, 116, 58, 12.9, 5.2, 3.5, 1.2],
            1e-3 * [699, 241, 120, 75, 11.4, 6.7, 4.1, 1.3],
            1e-3 * [569, 178, 103, 64, 11, 5.4, 3.0, 2.0, 0.82],
            1e-3 * [646, 249, 126, 101, 11, 5.4, 2.7, 2.1, 0.6],
            1e-3 * [686, 108, 98, 65.5, 11.2, 6.2, 3.4, 1.4],
            1e-3 * [628, 193, 100, 56, 9.3, 6.0, 5.1, 3.2, 1.5],
        ],
        exhaled_144_index = exhaled_index.(times_144),
    )
end

function monster_brmi(data = monster_data())
    @brm data begin
        # p(sigma) proportional to 1 / sigma for both measurement channels.
        log_sigma_venous ~ Flat()
        log_sigma_exhaled ~ Flat()
        sigma_venous = exp(log_sigma_venous)
        sigma_exhaled = exp(log_sigma_exhaled)

        # Fifteen person-level quantities on the log scale. The four flow
        # fractions and the two lean-volume fractions are unnormalized here
        # and normalized inside the cell.
        log_VPR ~ 1 + (1 | VPR | subject)
        log_Fwp ~ 1 + (1 | Fwp | subject)
        log_Fpp ~ 1 + (1 | Fpp | subject)
        log_Ff ~ 1 + (1 | Ff | subject)
        log_Fl ~ 1 + (1 | Fl | subject)
        log_Vwp ~ 1 + (1 | Vwp | subject)
        log_Vpp ~ 1 + (1 | Vpp | subject)
        log_Vl ~ 1 + (1 | Vl | subject)
        log_Pba ~ 1 + (1 | Pba | subject)
        log_Pwp ~ 1 + (1 | Pwp | subject)
        log_Ppp ~ 1 + (1 | Ppp | subject)
        log_Pf ~ 1 + (1 | Pf | subject)
        log_Pl ~ 1 + (1 | Pl | subject)
        log_VMI ~ 1 + (1 | VMI | subject)
        log_KMI ~ 1 + (1 | KMI | subject)

        # Population geometric means: lognormal around the prior guess.
        effect(log_VPR, Intercept) ~ Normal(log(1.6), log(1.3))
        effect(log_Fwp, Intercept) ~ Normal(log(0.48), log(1.2))
        effect(log_Fpp, Intercept) ~ Normal(log(0.2), log(1.2))
        effect(log_Ff, Intercept) ~ Normal(log(0.07), log(1.2))
        effect(log_Fl, Intercept) ~ Normal(log(0.25), log(1.1))
        effect(log_Vwp, Intercept) ~ Normal(log(0.28), log(1.2))
        effect(log_Vpp, Intercept) ~ Normal(log(0.56), log(1.2))
        effect(log_Vl, Intercept) ~ Normal(log(0.033), log(1.1))
        effect(log_Pba, Intercept) ~ Normal(log(12.0), log(1.5))
        effect(log_Pwp, Intercept) ~ Normal(log(4.8), log(1.5))
        effect(log_Ppp, Intercept) ~ Normal(log(1.6), log(1.5))
        effect(log_Pf, Intercept) ~ Normal(log(125.0), log(1.5))
        effect(log_Pl, Intercept) ~ Normal(log(4.8), log(1.5))
        effect(log_VMI, Intercept) ~ Normal(log(0.042), log(10.0))
        effect(log_KMI, Intercept) ~ Normal(log(16.0), log(10.0))

        # Population geometric standard deviations, nu = 4.
        sd(:, VPR) ~ ScaledInvChiScale(4.0, log(1.3))
        sd(:, Fwp) ~ ScaledInvChiScale(4.0, log(1.2))
        sd(:, Fpp) ~ ScaledInvChiScale(4.0, log(1.2))
        sd(:, Ff) ~ ScaledInvChiScale(4.0, log(1.2))
        sd(:, Fl) ~ ScaledInvChiScale(4.0, log(1.1))
        sd(:, Vwp) ~ ScaledInvChiScale(4.0, log(1.2))
        sd(:, Vpp) ~ ScaledInvChiScale(4.0, log(1.2))
        sd(:, Vl) ~ ScaledInvChiScale(4.0, log(1.1))
        sd(:, Pba) ~ ScaledInvChiScale(4.0, log(1.3))
        sd(:, Pwp) ~ ScaledInvChiScale(4.0, log(1.3))
        sd(:, Ppp) ~ ScaledInvChiScale(4.0, log(1.3))
        sd(:, Pf) ~ ScaledInvChiScale(4.0, log(1.3))
        sd(:, Pl) ~ ScaledInvChiScale(4.0, log(1.3))
        sd(:, VMI) ~ ScaledInvChiScale(4.0, log(2.0))
        sd(:, KMI) ~ ScaledInvChiScale(4.0, log(1.5))

        @plate for i in eachindex(log_VPR)
            unit_volume_flow = softmax([log_Fwp[i], log_Fpp[i], log_Ff[i], log_Fl[i]])
            Vl = exp(log_Vl[i])
            volume_fraction = append_row(
                (0.837 - Vl) * softmax([log_Vwp[i], log_Vpp[i]]), Vl)
            partition_coefficient = exp([log_Pwp[i], log_Ppp[i], log_Pf[i], log_Pl[i]])
            pred_72[i] = monster_experiment(
                times_72[i], exposure_72,
                lean_body_mass[i], fat_fraction[i], pulmonary_flow[i],
                exp(log_VPR[i]), unit_volume_flow, volume_fraction,
                exp(log_Pba[i]), partition_coefficient,
                exp(log_VMI[i]), exp(log_KMI[i]), n_substeps)
            pred_144[i] = monster_experiment(
                times_144[i], exposure_144,
                lean_body_mass[i], fat_fraction[i], pulmonary_flow[i],
                exp(log_VPR[i]), unit_volume_flow, volume_fraction,
                exp(log_Pba[i]), partition_coefficient,
                exp(log_VMI[i]), exp(log_KMI[i]), n_substeps)
            venous_72[i] ~ lognormal(log(pred_72[i][venous_72_index[i]]), sigma_venous)
            exhaled_72[i] ~ lognormal(log(pred_72[i][exhaled_72_index[i]]), sigma_exhaled)
            venous_144[i] ~ lognormal(log(pred_144[i][venous_144_index[i]]), sigma_venous)
            exhaled_144[i] ~ lognormal(log(pred_144[i][exhaled_144_index[i]]), sigma_exhaled)
        end
    end
end

monster_sbbrmi(data = monster_data()) = SBBRMI(monster_brmi(data); mod=@__MODULE__)

"""
    write_monster_stan_files(dir; data=monster_data())

Write the generated Stan program and its Stan JSON data to `dir` as
`monster.stan` and `monster.data.json`, ready for CmdStan, BridgeStan or
nutpie. Returns the two paths.
"""
function write_monster_stan_files(dir; data=monster_data())
    sb = monster_sbbrmi(data)
    mkpath(dir)
    stan = joinpath(dir, "monster.stan")
    json = joinpath(dir, "monster.data.json")
    write(stan, BayesianRegressionModels.stan_code(sb))
    write(json, StanBlocks.bridgestan_data(BayesianRegressionModels.stan_data(sb)))
    (; stan, json)
end
