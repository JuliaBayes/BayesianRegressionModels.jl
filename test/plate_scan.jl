# test/plate_scan.jl — `@scan` recurrences inside `@brm` `@plate for` cells.
#
# Run: julia --startup-file=no --project=test test/plate_scan.jl
#
# A `@scan begin … end` block in a cell is one recurrence per cell, passed to
# StanBlocks and RKPPL unchanged apart from the cell's slice reads. Checked
# here: the printed source, the emitted Stan program, its density against an
# independent per-subject AR(1) reference, a random-effect panel, the
# omitted-response program that forward-simulates the trajectories, and the
# RK log density and gradient of the latent and the observed recurrence
# against independent oracles.

include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using LinearAlgebra: Symmetric
using LogExpFunctions: logistic, log1pexp

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
        # Bound: StanBlocks `82a954da` observes a data-bound carried array, but
        # not through a do-cell parameter that shadows its data name, which is
        # how BRM spells every cell (`@scan: unsupported statement LHS
        # (y[plate_i__pl_1])[t]`), snag StanBlocks `observed-scan-in-63843c19`.
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

# RK: a bounded `Uniform(lo, hi)` scalar is `lo + (hi - lo) * logistic(u)`, a
# positive one `exp(u)`; each adds its log Jacobian.
uniform_value(u, lo, hi) = lo + (hi - lo) * logistic(u)
uniform_logjac(u, lo, hi) = log(hi - lo) - log1pexp(-u) - log1pexp(u)

@testset "plate scan on RK" begin
    @testset "latent recurrence: density and gradient" begin
        brmi = latent_ar(SCAN_PANEL)
        backend, problem = consumer_problem(brmi)
        names = coordinate_names(backend.model.layout)
        T, nsub = SCAN_PANEL.T, length(SCAN_PANEL.y)
        at(name) = coordinate(names, Symbol(name))
        # One seed per subject, then each subject's `T - 1` steps in plate order.
        seeds = [at("_rkppl_scan_h_seed_h_1.$j") for j in 1:nsub]
        steps = [at("_rkppl_scan_h_draw_h.$((j - 1) * (T - 1) + t - 1)")
            for t in 2:T, j in 1:nsub]
        @test length(names) == 4 + nsub * T
        oracle(u) = begin
            phi = uniform_value(u[at(:phi)], -1, 1)
            s, sigma = exp(u[at(:s)]), exp(u[at(:sigma)])
            h = [t == 1 ? u[seeds[j]] : u[steps[t - 1, j]] for t in 1:T, j in 1:nsub]
            latent_ar_logdensity((; phi, s, sigma, mu0=u[at(:mu0)], h), SCAN_PANEL) +
                uniform_logjac(u[at(:phi)], -1, 1) + u[at(:s)] + u[at(:sigma)]
        end
        n = length(names)
        for u in (zeros(n), fill(0.1, n), collect(range(-0.4, 0.4; length=n)))
            check_consumer_point(problem, u, oracle)
        end
    end

    # The observed series is the scan's carried array: RKPPL observes it in
    # place, one series per subject. Its density is each subject's stationary
    # AR(1) law, a multivariate normal with covariance `sigma^2 / (1 - phi^2) *
    # phi^|s - t|` around the subject's level.
    @testset "observed AR(1) errors: density and gradient" begin
        data = (; SCAN_PANEL..., subject=["b", "a", "c"])
        saved = deepcopy(data)
        backend, problem = consumer_problem(ar_errors_cell(data))
        names = coordinate_names(backend.model.layout)
        at(name) = coordinate(names, Symbol(name))
        tau = findfirst(n -> occursin(".tau.", string(n)), names)
        z = findall(n -> occursin(".z.", string(n)), names)
        @test length(names) == 4 + length(z)
        levels = sort(unique(data.subject))
        order = [findfirst(==(label), levels) for label in data.subject]
        oracle(u) = begin
            phi = uniform_value(u[at(:phi)], -1, 1)
            sigma, scale = exp(u[at(:sigma)]), exp(u[tau])
            level = u[at(:level_Intercept)] .+ scale .* u[z][order]
            lags = abs.((1:data.T) .- (1:data.T)')
            cov = Symmetric(sigma^2 / (1 - phi^2) .* phi .^ lags)
            logpdf(Normal(), u[at(:level_Intercept)]) + sum(logpdf.(Normal(), u[z])) +
                logpdf(LogNormal(0, 1), scale) + u[tau] +
                logpdf(Exponential(), sigma) + u[at(:sigma)] +
                logpdf(Uniform(-1, 1), phi) + uniform_logjac(u[at(:phi)], -1, 1) +
                sum(logpdf(MvNormal(fill(level[j], data.T), cov), data.y[j])
                    for j in eachindex(data.y))
        end
        n = length(names)
        for u in (zeros(n), fill(0.1, n), collect(range(-0.4, 0.4; length=n)))
            check_consumer_point(problem, u, oracle)
        end
        @test isequal(data, saved)
    end
end
