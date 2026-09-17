# The models of the documentation page "Epidemic renewal models", spelled with EpiRenewal's operators.
#
# Read by the page (its closing section shows two of them verbatim) and by test/runtests.jl, which
# requires each to have the same log density and gradient as the page's hand-written Stan functions.
# `page` is the module holding research/epi_renewal/renewal.jl: it supplies the simulated data.

function epirenewal_delay_model(data = page.reporting_delay_data())
    @brm data begin
        mu    ~ Normal(1.0, 0.5; lower=0.0, upper=3.0)
        sigma ~ Normal(0.5, 0.25; lower=0.1, upper=2.0)
        delay ~ censored_delay(mu, sigma, window)
    end
end

function epirenewal_single_model(data = page.renewal_single_data())
    @brm data begin
        log_I0  ~ Normal(log(50.0), 0.5)
        cluster ~ Normal(0.0, 0.1; lower=0.0)
        log_R   ~ 1 + rw(time)
        effect(log_R, Intercept) ~ Normal(log(1.3), 0.1)
        sd(:, rw(time)) ~ Normal(0.0, 0.05)
        past    = seeded_history(log_I0, exp(log_R), gen_pmf, 14)    # infections on days -13 … 0
        I       = renewal(gen_pmf, exp(log_R), past)
        Y       = delay(delay_pmf, I, past)
        cases   ~ nb_cases(Y, cluster, observed)
    end
end

# both kernels as functions of the lag; the bodies read the shared data vectors
function epirenewal_single_model_functions(data = page.renewal_single_data())
    @brm data begin
        log_I0  ~ Normal(log(50.0), 0.5)
        cluster ~ Normal(0.0, 0.1; lower=0.0)
        log_R   ~ 1 + rw(time)
        effect(log_R, Intercept) ~ Normal(log(1.3), 0.1)
        sd(:, rw(time)) ~ Normal(0.0, 0.05)
        past    = seeded_history(log_I0, exp(log_R), gen_pmf, 14)
        I       = renewal(exp(log_R), past, 13) do s
            gen_pmf[s]
        end
        Y       = delay(I, past, 15) do d
            delay_pmf[d + 1]
        end
        cases   ~ nb_cases(Y, cluster, observed)
    end
end

# the reporting delay ESTIMATED inside the renewal model: the `do` body reads sampled parameters
function epirenewal_estimated_delay_model(data = page.renewal_single_data())
    @brm data begin
        mu_delay    ~ Normal(1.5, 0.2)
        sigma_delay ~ Normal(0.5, 0.1; lower=0.0)
        log_I0  ~ Normal(log(50.0), 0.5)
        cluster ~ Normal(0.0, 0.1; lower=0.0)
        log_R   ~ 1 + rw(time)
        effect(log_R, Intercept) ~ Normal(log(1.3), 0.1)
        sd(:, rw(time)) ~ Normal(0.0, 0.05)
        past    = seeded_history(log_I0, exp(log_R), gen_pmf, 14)
        I       = renewal(gen_pmf, exp(log_R), past)
        Y       = delay(I, past, 15) do d                            # d = 0 … 14
            censored_lognormal_mass(d, mu_delay, sigma_delay, 15)
        end
        cases   ~ nb_cases(Y, cluster, observed)
    end
end

function epirenewal_patch_model(data = page.renewal_patch_data())
    @brm data begin
        gamma   ~ Normal(1.5, 0.5; lower=0.0)
        cluster ~ Normal(0.0, 0.1; lower=0.0)
        log_R   ~ 1 + rw(time) + cdar(week; by=patch, cor=C)
        effect(log_R, Intercept) ~ Normal(log(1.3), 0.1)
        sd(:, rw(time)) ~ Normal(0.0, 0.05)
        sd(:, cdar(week)) ~ Normal(0.0, 0.2)
        ar(:, cdar(week)) ~ Normal(0.8, 0.1)
        log_I0  ~ 0 + offset(seed_mean) + factor(patch)
        effect(log_I0, patch) ~ Normal(0.0, 0.5)
        seeds   = per_patch(log_I0, pop)                             # one seed per patch
        mixing  = gravity(pop, dist_flat, gamma)                     # P × P mixing weights
        past    = seeded_history(seeds, exp(log_R), gen_pmf, 14)     # 14 × P
        I       = renewal(gen_pmf, exp(log_R), past, mixing)
        Y       = delay(delay_pmf, I, past)
        cases   ~ nb_cases(Y, cluster, observed)
    end
end
