# test/plate_scan.jl — `@scan` recurrences inside `@brm` `@plate for` cells.
#
# Run: julia --startup-file=no --project=test test/plate_scan.jl
#
# A `@scan begin … end` block in a cell is one recurrence per cell, passed to
# StanBlocks unchanged apart from the cell's slice reads. Checked here: the
# printed source, the emitted Stan program, its density against an
# independent per-subject AR(1) reference, a random-effect panel, and the
# omitted-response program that forward-simulates the trajectories.

using Test, BayesianRegressionModels, Distributions, StanBlocks
import BridgeStan
const BRM = BayesianRegressionModels

const SCAN_PANEL = (;
    y=[[0.12, 0.31, 0.2], [-0.05, 0.1, 0.42], [0.3, -0.2, 0.05]],
    T=3,
)

# A per-subject latent AR(1) state added to a shared level.
latent_ar(data) = @brm data begin
    phi ~ Uniform(-1, 1)
    s ~ Exponential(1)
    sigma ~ Exponential(1)
    mu0 ~ Normal(0, 1)
    @plate for i in eachindex(y)
        @scan begin
            h[1] ~ Normal(0, 1)
            for t in 2:T
                h[t] ~ Normal(phi * h[t - 1], s)
            end
        end
        y[i] ~ Normal(mu0 .+ h, sigma)
    end
end

# The same state over a random-effect level: the cells are its subjects.
latent_ar_ranef(data) = @brm data begin
    phi ~ Uniform(-1, 1)
    s ~ Exponential(1)
    sigma ~ Exponential(1)
    level ~ 1 + (1 | subject)
    @plate for i in eachindex(level)
        @scan begin
            h[1] ~ Normal(0, 1)
            for t in 2:T
                h[t] ~ Normal(phi * h[t - 1], s)
            end
        end
        y[i] ~ Normal(level[i] .+ h, sigma)
    end
end

# The observed series as the scan's carried array: AR(1) errors around each
# subject's level.
ar_errors_cell(data) = @brm data begin
    phi ~ Uniform(-1, 1)
    sigma ~ Exponential(1)
    level ~ 1 + (1 | subject)
    @plate for i in eachindex(level)
        @scan begin
            y[i][1] ~ Normal(level[i], sigma / sqrt(1 - phi^2))
            for t in 2:T
                y[i][t] ~ Normal(level[i] + phi * (y[i][t - 1] - level[i]), sigma)
            end
        end
    end
end

# Independent reference: the model's full log density at constrained values,
# without Jacobian.
function latent_ar_logdensity(v, data)
    lp = logpdf(Uniform(-1, 1), v.phi) + logpdf(Exponential(1), v.s) +
         logpdf(Exponential(1), v.sigma) + logpdf(Normal(0, 1), v.mu0)
    for (j, y) in enumerate(data.y)
        h = v.h[:, j]
        lp += logpdf(Normal(0, 1), h[1])
        for t in 2:data.T
            lp += logpdf(Normal(v.phi * h[t - 1], v.s), h[t])
        end
        lp += sum(logpdf.(Normal.(v.mu0 .+ h, v.sigma), y))
    end
    lp
end

stanc_ok(code) = StanBlocks.stanc_check(code; warn_pedantic=false).ok

@testset "plate scan" begin
    brmi = latent_ar(SCAN_PANEL)

    @testset "source" begin
        @test occursin("""
            @plate for i in eachindex(y)
                @scan begin
                    h[1] ~ Normal(0, 1)
                    for t in 2:T
                        h[t] ~ Normal(phi * h[t - 1], s)
                    end
                end
                y[i] ~ Normal(mu0 .+ h, sigma)
            end""", sprint(show, brmi))
    end

    sb = SBBRMI(brmi; mod=@__MODULE__)
    code = BRM.stan_code(sb)

    @testset "emitted Stan" begin
        @test occursin(r"matrix\[T, \w+\] h;", code)
        @test occursin(r"for\(t in 2:T\) \{\s+h\[t, \w+\] ~ normal\(\(phi \* h\[\(t - 1\), \w+\]\), s\);", code)
        @test stanc_ok(code)
    end

    @testset "density against the reference" begin
        folder = joinpath(tempdir(), "brm-plate-scan")
        mkpath(folder)
        model = BRM.stan_instantiate(sb; path=joinpath(folder, "latent_ar.stan")).model
        n = BridgeStan.param_unc_num(model)
        names = BridgeStan.param_names(model)
        nsub = length(SCAN_PANEL.y)
        for u in (zeros(n), fill(0.1, n), collect(range(-0.4, 0.4; length=n)))
            values = Dict(zip(names, BridgeStan.param_constrain(model, u)))
            h = [values["h.$t.$j"] for t in 1:SCAN_PANEL.T, j in 1:nsub]
            v = (; phi=values["phi"], s=values["s"], sigma=values["sigma"],
                   mu0=values["mu0"], h)
            @test BridgeStan.log_density(model, u; propto=false, jacobian=false) ≈
                latent_ar_logdensity(v, SCAN_PANEL) atol=1e-10
        end
    end

    @testset "random-effect cells" begin
        data = (; SCAN_PANEL..., subject=["b", "a", "c"])
        ranef_code = BRM.stan_code(SBBRMI(latent_ar_ranef(data); mod=@__MODULE__))
        @test occursin(r"matrix\[T, \w+\] h;", ranef_code)
        @test stanc_ok(ranef_code)
    end

    # The observed series itself is the scan's carried array: AR(1) errors
    # around the subject's level, stationary at the first entry.
    @testset "observed AR(1) errors" begin
        data = (; SCAN_PANEL..., subject=["b", "a", "c"])
        # Omitted: the cell forward-simulates each subject's whole series.
        prior_code = BRM.stan_code(SBBRMI(
            ar_errors_cell((; subject=data.subject, T=data.T)); mod=@__MODULE__))
        @test occursin(r"y\[1, \w+\] = normal_rng\(level\[\w+\], \(sigma / sqrt\(\(1 - \(phi \^ 2\)\)\)\)\);", prior_code)
        @test occursin(r"y\[t, \w+\] = normal_rng\(\s*\(level\[\w+\] \+ \(phi \* \(y\[\(t - 1\), \w+\] - level\[\w+\]\)\)\),\s*sigma\s*\);", prior_code)
        @test stanc_ok(prior_code)
        # Bound: StanBlocks `76a5ebe` refuses a data-bound carried array
        # (`@scan: setup y[1]: y is data.`), snag StanBlocks
        # `observed-scan-sc-6bfe7467`.
        @test_broken try
            stanc_ok(BRM.stan_code(SBBRMI(ar_errors_cell(data); mod=@__MODULE__)))
        catch
            false
        end
    end

    # Without `y` the random-effect level still counts the cells.
    @testset "omitted response simulates the trajectories" begin
        prior_code = BRM.stan_code(SBBRMI(
            latent_ar_ranef((; subject=["b", "a", "c"], T=SCAN_PANEL.T)); mod=@__MODULE__))
        @test occursin(r"h\[t, \w+\] = normal_rng\(", prior_code)
        @test stanc_ok(prior_code)
    end
end
