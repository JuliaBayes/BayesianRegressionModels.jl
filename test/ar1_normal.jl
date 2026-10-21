# test/ar1_normal.jl — per-group AR(1) residuals: in a
# `@plate for p in groups(probe; order = day)` cell,
# `y[p] ~ AR1Normal(mu[p], sigma, phi; time = day[p])`.
#
# Run: julia --startup-file=no --project=test test/ar1_normal.jl
#
# `groups` is an ordinary function (the indices of each group), and
# `AR1Normal` calls each backend's AR(1) submodel, the cell's observed `@scan`
# (decisions `0det9mo` and `1pnob5t`: the plate is the grouping). Checked here:
# `groups` itself, the statement as printed and the submodel each backend
# calls, the same density as the `@scan` written out, its refusals, the RK log density and gradient against independent oracles —
# a multivariate normal per group with covariance
# `sigma^2 / (1 - phi^2) * phi^|t_i - t_j|` for shared `sigma` and `phi`, and
# the time-ordered conditional recursion when both vary by row — on rows in
# any order, ragged groups and gaps in time, and the replay of a fitted model
# onto reordered and new rows.

include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using LinearAlgebra: Symmetric

# Three probes with different series lengths and missing days, rows
# interleaved and out of day order.
const PANEL = (;
    probe=["p2", "p1", "p3", "p2", "p1", "p3", "p2", "p3", "p2", "p1", "p3", "p2"],
    day=[3, 2, 2, 1, 1, 4, 2, 3, 6, 4, 7, 5],
    x=[0.3, -0.2, 0.5, 0.1, -0.4, 0.0, 0.8, -0.6, 0.2, 0.7, -0.1, 0.4],
    y=[0.41, -0.12, 0.66, 0.05, -0.31, 0.27, 0.92, -0.48, 0.18, 0.52, 0.03, 0.37],
)

shared_ar(data) = @brm data begin
    phi ~ Uniform(-1, 1)
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | probe)
    @plate for p in groups(probe; order = day)
        y[p] ~ AR1Normal(mu[p], sigma, phi; time = day[p])
    end
end

varying_ar(data) = @brm data begin
    atanh(phi) ~ 1 + (1 | probe)
    log(sigma) ~ 1 + (1 | probe)
    mu ~ 1 + x
    @plate for p in groups(probe; order = day)
        y[p] ~ AR1Normal(mu[p], sigma[p], phi[p]; time = day[p])
    end
end

# Without `time`, consecutive rows of a group are one step apart.
stepped_ar(data) = @brm data begin
    phi ~ Uniform(-1, 1)
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | probe)
    @plate for p in groups(probe; order = day)
        y[p] ~ AR1Normal(mu[p], sigma, phi)
    end
end

# Each group's rows in time order: the cells' `p`, computed by the same
# exported function outside the model.
group_rows(data) = groups(data.probe; order=data.day)

# Shared `sigma` and `phi`: one multivariate normal per group, the stationary
# AR(1) covariance at the group's observed times.
function ar_mvnormal_loglik(y, mu, sigma, phi, data; steps=false)
    sum(group_rows(data)) do rows
        t = steps ? eachindex(rows) : data.day[rows]
        cov = Symmetric(sigma^2 / (1 - phi^2) .* phi .^ abs.(t .- t'))
        logpdf(MvNormal(mu[rows], cov), y[rows])
    end
end

# Row-varying `sigma` and `phi`: the exact conditional recursion, step `k`
# apart, read at the later row as the expansion reads it.
function ar_recursion_loglik(y, mu, sigma, phi, data)
    sum(group_rows(data)) do rows
        r = first(rows)
        lp = logpdf(Normal(mu[r], sigma[r] / sqrt(1 - phi[r]^2)), y[r])
        for (a, b) in zip(rows[1:end-1], rows[2:end])
            k = data.day[b] - data.day[a]
            lp += logpdf(Normal(mu[b] + phi[b]^k * (y[a] - mu[a]),
                sigma[b] * sqrt((1 - phi[b]^(2k)) / (1 - phi[b]^2))), y[b])
        end
        lp
    end
end


# The same kind of series, pre-grouped: one vector per subject.
const SCAN_PANEL = (; y=[[0.1, 0.3], [0.2, -0.1, 0.4]], day=[[1, 2], [1, 2, 4]])

const PROBES = sort(unique(PANEL.probe))
probe_index(data) = [findfirst(==(g), PROBES) for g in data.probe]

rows_of(data, rows) = map(column -> column[rows], data)

rk_density(backend, u) = LogDensityProblems.logdensity(
    rk_logdensity_problem(backend; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse)), u)

@testset "AR1Normal" begin
    @testset "groups: the indices of each group" begin
        @test groups(["b", "a", "b", "a"]; order=[4, 9, 1, 2]) == [[4, 2], [3, 1]]
        @test groups(["b", "a", "b", "a"]) == [[2, 4], [1, 3]]
        # Ties keep row order.
        @test groups([1, 1, 1]; order=[2, 1, 2]) == [[2, 1, 3]]
        @test groups(PANEL.probe; order=PANEL.day) ==
            [[5, 2, 10], [4, 7, 1, 12, 9], [3, 8, 6, 11]]
        @test_throws ArgumentError groups(["a", missing])
        @test_throws DimensionMismatch groups(["a", "b"]; order=[1])
    end

    @testset "prints as written; each backend calls its AR(1) submodel" begin
        @test occursin("""
            @plate for p in groups(probe; order=day)
                y[p] ~ AR1Normal(mu[p], sigma, phi; time=day[p])
            end""", sprint(show, shared_ar(PANEL)))
        # RK: the cell observes `y` through BRM's `@rkppl` submodel, a shared
        # argument filled to the series' length.
        emitted = BRM._rk_emit_ast(RKBRMI(shared_ar(PANEL)).plan)
        defs = join((sprint(Base.show_unquoted, d) for d in emitted.defs), "\n")
        main = sprint(Base.show_unquoted, emitted.main)
        @test occursin("brm_ar1_normal(mu, sigma, phi, time) = begin", defs)
        @test occursin(r"y\w*\[\w+\] ~ brm_ar1_normal\(", main)
        @test occursin(r"fill\(sigma, length\(", main)
        @test !occursin("AR1Normal", main)
        stepped = BRM._rk_emit_ast(RKBRMI(stepped_ar(PANEL)).plan)
        @test any(d -> occursin("brm_ar1_normal_steps(mu, sigma, phi) = begin",
            sprint(Base.show_unquoted, d)), stepped.defs)
        # SB: the cell calls BRM's `@slic` submodel on the observed series.
        code = BRM.stan_code(SBBRMI(shared_ar(PANEL); mod=@__MODULE__))
        @test occursin("y_ar1", code)

        # The statement and the `@scan` it states are one model, whichever
        # arguments vary along the series: an `x[p]` read, or a cell local
        # computed from one, varies; anything else is shared.
        written(data) = @brm data begin
            phi ~ Uniform(-1, 1)
            sigma ~ Exponential(1)
            mu ~ 1 + x + (1 | probe)
            @plate for p in groups(probe; order = day)
                @scan begin
                    y[p][1] ~ Normal(mu[p][1], sigma / sqrt(1 - phi ^ 2))
                    for t in 2:length(day[p])
                        k = day[p][t] - day[p][t - 1]
                        y[p][t] ~ Normal(mu[p][t] + phi ^ k * (y[p][t - 1] - mu[p][t - 1]),
                            sigma * sqrt((1 - phi ^ (2 * k)) / (1 - phi ^ 2)))
                    end
                end
            end
        end
        written_varying(data) = @brm data begin
            atanh(phi) ~ 1 + (1 | probe)
            log(sigma) ~ 1 + (1 | probe)
            mu ~ 1 + x
            @plate for p in groups(probe; order = day)
                @scan begin
                    y[p][1] ~ Normal(mu[p][1], sigma[p][1] / sqrt(1 - phi[p][1] ^ 2))
                    for t in 2:length(day[p])
                        k = day[p][t] - day[p][t - 1]
                        y[p][t] ~ Normal(mu[p][t] + phi[p][t] ^ k * (y[p][t - 1] - mu[p][t - 1]),
                            sigma[p][t] * sqrt((1 - phi[p][t] ^ (2 * k)) / (1 - phi[p][t] ^ 2)))
                    end
                end
            end
        end
        local_ar(data) = @brm data begin
            phi ~ Uniform(-1, 1)
            log_s ~ Normal(0, 1)
            mu ~ 1 + x + (1 | probe)
            @plate for p in groups(probe; order = day)
                m = mu[p] .+ 0.5
                y[p] ~ AR1Normal(m, exp(log_s), phi; time = day[p])
            end
        end
        written_local(data) = @brm data begin
            phi ~ Uniform(-1, 1)
            log_s ~ Normal(0, 1)
            mu ~ 1 + x + (1 | probe)
            @plate for p in groups(probe; order = day)
                m = mu[p] .+ 0.5
                @scan begin
                    y[p][1] ~ Normal(m[1], exp(log_s) / sqrt(1 - phi ^ 2))
                    for t in 2:length(day[p])
                        k = day[p][t] - day[p][t - 1]
                        y[p][t] ~ Normal(m[t] + phi ^ k * (y[p][t - 1] - m[t - 1]),
                            exp(log_s) * sqrt((1 - phi ^ (2 * k)) / (1 - phi ^ 2)))
                    end
                end
            end
        end
        written_steps(data) = @brm data begin
            phi ~ Uniform(-1, 1)
            sigma ~ Exponential(1)
            mu ~ 1 + x + (1 | probe)
            @plate for p in groups(probe; order = day)
                @scan begin
                    y[p][1] ~ Normal(mu[p][1], sigma / sqrt(1 - phi ^ 2))
                    for t in 2:length(mu[p])
                        y[p][t] ~ Normal(mu[p][t] + phi * (y[p][t - 1] - mu[p][t - 1]), sigma)
                    end
                end
            end
        end
        for (statement, scan) in ((shared_ar, written), (varying_ar, written_varying),
                (local_ar, written_local), (stepped_ar, written_steps))
            backend, reference = RKBRMI(statement(PANEL)), RKBRMI(scan(PANEL))
            @test coordinate_names(backend.model.layout) ==
                coordinate_names(reference.model.layout)
            n = backend.model.layout.total
            for u in (zeros(n), collect(range(-0.3, 0.5; length=n)))
                @test rk_density(backend, u) ≈ rk_density(reference, u) rtol=1e-12
            end
        end
    end

    @testset "refusals at the formula boundary" begin
        @test_throws "AR1Normal(mu, sigma, phi)" @eval @brm PANEL begin
            phi ~ Uniform(-1, 1)
            @plate for p in groups(probe; order = day)
                y[p] ~ AR1Normal(0, phi; time = day[p])
            end
        end
        @test_throws "AR1Normal(mu, sigma, phi)" @eval @brm PANEL begin
            phi ~ Uniform(-1, 1)
            @plate for p in groups(probe; order = day)
                y[p] ~ AR1Normal(0, 1, phi; times = day[p])
            end
        end
        # refused: every read of a `groups(...)` cell is a group's rows, which
        # is what tells a series argument from a shared one; a cell over
        # pre-grouped data writes its @scan.
        @test_throws "write the `@scan` it stands for" @eval @brm SCAN_PANEL begin
            phi ~ Uniform(-1, 1)
            @plate for i in eachindex(y)
                y[i] ~ AR1Normal(0, 1, phi; time = day[i])
            end
        end
        @test_throws "reads the cell's time column" @eval @brm PANEL begin
            phi ~ Uniform(-1, 1)
            @plate for p in groups(probe; order = day)
                y[p] ~ AR1Normal(0, 1, phi; time = day)
            end
        end
        # refused: the grouping is the plate's, not a keyword (decision
        # 1pnob5t, guiding principle 198kldn).
        @test_throws "takes no `by=`" @eval @brm PANEL begin
            phi ~ Uniform(-1, 1)
            @plate for p in groups(probe; order = day)
                y[p] ~ AR1Normal(0, 1, phi; time = day[p], by = probe)
            end
        end
        @test_throws "one series per `@plate` cell" @eval @brm PANEL begin
            phi ~ Uniform(-1, 1)
            y ~ AR1Normal(0, 1, phi; time = day, by = probe)
        end
        # A repeated time has no AR(1) step; the gap rule needs whole steps.
        @test_throws "repeats time 3 in group \"p2\"" shared_ar(
            merge(PANEL, (; day=[3, 2, 2, 1, 1, 4, 2, 3, 6, 4, 7, 3])))
        @test_throws "holds 7.5 in group \"p3\"" shared_ar(
            merge(PANEL, (; day=Float64[3, 2, 2, 1, 1, 4, 2, 3, 6, 4, 7.5, 5])))
    end

    @testset "the two references state one law" begin
        mu = 0.1 .* PANEL.x
        for (sigma, phi) in ((0.7, 0.54), (1.3, -0.4), (0.4, 0.82))
            @test ar_mvnormal_loglik(PANEL.y, mu, sigma, phi, PANEL) ≈
                ar_recursion_loglik(PANEL.y, mu, fill(sigma, 12), fill(phi, 12), PANEL) atol=1e-12
        end
    end

    @testset "RK, shared sigma and phi: density and gradient ($model)" for (model, steps) in
            ((shared_ar, false), (stepped_ar, true))
        saved = deepcopy(PANEL)
        backend, problem = consumer_problem(model(PANEL))
        names = coordinate_names(backend.model.layout)
        at(name) = coordinate(names, Symbol(name))
        tau = at("b_probe.tau.1")
        z = [at("b_probe.z.$j.1") for j in eachindex(PROBES)]
        groups = probe_index(PANEL)
        @test length(names) == 5 + length(PROBES)
        oracle(u) = begin
            phi = uniform_value(u[at(:phi)], -1, 1)
            sigma, scale = exp(u[at(:sigma)]), exp(u[tau])
            mu = u[at("pop_mu.beta_pop.1")] .+ u[at("pop_mu.beta_pop.2")] .* PANEL.x .+
                scale .* u[z][groups]
            ar_mvnormal_loglik(PANEL.y, mu, sigma, phi, PANEL; steps) + logpdf(Uniform(-1, 1), phi) + uniform_logjac(u[at(:phi)], -1, 1) +
                logpdf(Exponential(), sigma) + u[at(:sigma)] +
                logpdf(LogNormal(0, 1), scale) + u[tau] + sum(logpdf.(Normal(), u[z])) +
                logpdf(Normal(), u[at("pop_mu.beta_pop.1")]) +
                logpdf(Normal(), u[at("pop_mu.beta_pop.2")])
        end
        n = length(names)
        for u in (zeros(n), fill(0.1, n), collect(range(-0.4, 0.4; length=n)))
            check_consumer_point(problem, u, oracle)
        end
        @test isequal(PANEL, saved)
    end

    @testset "RK, sigma and phi varying by group: density and gradient" begin
        model(data) = @brm data begin
            atanh(phi) ~ 1 + (1 | ph | probe)
            log(sigma) ~ 1 + (1 | sg | probe)
            mu ~ 1 + x
            @plate for p in groups(probe; order = day)
                y[p] ~ AR1Normal(mu[p], sigma[p], phi[p]; time = day[p])
            end
        end
        backend, problem = consumer_problem(model(PANEL))
        names = coordinate_names(backend.model.layout)
        at(name) = coordinate(names, Symbol(name))
        groups = probe_index(PANEL)
        effect(id, u) = exp(u[at("b_$(id)_probe.tau.1")]) .*
            u[[at("b_$(id)_probe.z.$j.1") for j in eachindex(PROBES)]][groups]
        # An `|ID|` block's SD keeps its standard-normal density on the
        # positive half line (`restricted(Normal(0, 1), 0, Inf)`).
        block_prior(id, u) = logpdf(Normal(), exp(u[at("b_$(id)_probe.tau.1")])) +
            u[at("b_$(id)_probe.tau.1")] +
            sum(logpdf(Normal(), u[at("b_$(id)_probe.z.$j.1")]) for j in eachindex(PROBES))
        oracle(u) = begin
            phi = tanh.(u[at(:phi_Intercept)] .+ effect(:ph, u))
            sigma = exp.(u[at(:sigma_Intercept)] .+ effect(:sg, u))
            mu = u[at("pop_mu.beta_pop.1")] .+ u[at("pop_mu.beta_pop.2")] .* PANEL.x
            ar_recursion_loglik(PANEL.y, mu, sigma, phi, PANEL) +
                logpdf(Normal(), u[at(:phi_Intercept)]) + logpdf(Normal(), u[at(:sigma_Intercept)]) +
                block_prior(:ph, u) + block_prior(:sg, u) +
                logpdf(Normal(), u[at("pop_mu.beta_pop.1")]) +
                logpdf(Normal(), u[at("pop_mu.beta_pop.2")])
        end
        n = length(names)
        @test n == 4 + 2 * (1 + length(PROBES))
        for u in (zeros(n), fill(0.1, n), collect(range(-0.4, 0.4; length=n)))
            check_consumer_point(problem, u, oracle)
        end
    end

    @testset "RK: rows in any order, and replay onto new rows" begin
        backend = RKBRMI(shared_ar(PANEL))
        u = collect(range(-0.4, 0.5; length=backend.model.layout.total))
        reference = rk_density(backend, u)
        shuffled = rows_of(PANEL, [7, 12, 1, 3, 10, 5, 9, 2, 11, 4, 8, 6])
        @test rk_density(RKBRMI(shared_ar(shuffled)), u) ≈ reference atol=1e-12
        # A fitted model replays onto reordered rows, and onto new rows of the
        # same probes, as a direct build on them does.
        @test rk_density(reprocess(backend, shuffled), u) ≈ reference atol=1e-12
        extended = (; probe=[PANEL.probe; "p1"; "p3"], day=[PANEL.day; 6; 5],
            x=[PANEL.x; 0.2; -0.3], y=[PANEL.y; 0.15; 0.44])
        replay = reprocess(backend, extended)
        @test replay.model === backend.model
        @test rk_density(replay, u) ≈ rk_density(RKBRMI(shared_ar(extended)), u) atol=1e-12
    end

    # The plate header the expansion uses is ordinary `@plate for` surface: a
    # cell per group, reading each group's rows.
    @testset "per-group plates: groups(g; order = t)" begin
        flat(data) = @brm data begin
            sigma ~ Exponential(1)
            mu ~ 1 + x + (1 | probe)
            y ~ Normal(mu, sigma)
        end
        grouped(data) = @brm data begin
            sigma ~ Exponential(1)
            mu ~ 1 + x + (1 | probe)
            @plate for p in groups(probe)
                y[p] ~ Normal(mu[p], sigma)
            end
        end
        ordered(data) = @brm data begin
            sigma ~ Exponential(1)
            mu ~ 1 + x + (1 | probe)
            @plate for p in groups(probe; order = day)
                y[p] ~ Normal(mu[p], sigma)
            end
        end
        # Independent rows: grouping and order change nothing.
        reference = RKBRMI(flat(PANEL))
        u = collect(range(-0.4, 0.5; length=reference.model.layout.total))
        for model in (grouped, ordered)
            backend = RKBRMI(model(PANEL))
            @test coordinate_names(backend.model.layout) ==
                coordinate_names(reference.model.layout)
            @test rk_density(backend, u) ≈ rk_density(reference, u) atol=1e-12
            @test StanBlocks.stanc_check(BRM.stan_code(SBBRMI(model(PANEL);
                mod=@__MODULE__)); warn_pedantic=false).ok
        end
        # refused: a collected per-group value could only be matched to its
        # group by position, the hazard `ragged(x, group => cells)` rules out
        # (3681f4a, decision 1pnob5t).
        @test_throws "collects no result" @eval @brm PANEL begin
            sigma ~ Exponential(1)
            @plate for p in groups(probe)
                loc[p] = 2 .* x[p]
                y[p] ~ Normal(loc[p], sigma)
            end
        end
        # refused: no action at a distance — a model-scope statement does not
        # set a plate's row order (guiding principle 1mwjios, decision 1pnob5t);
        # model-scope `ragged` statements are retired (todo 1nfukuv).
        @test_throws "are retired" @eval @brm PANEL begin
            sigma ~ Exponential(1)
            ragged(y, probe; order = day)
            @plate for p in groups(probe)
                y[p] ~ Normal(0, sigma)
            end
        end
    end

    @testset "SBBRMI: the same density as RK, descriptor and prior program" begin
        stanc_ok(code) = StanBlocks.stanc_check(code; warn_pedantic=false).ok
        @test stanc_ok(BRM.stan_code(SBBRMI(shared_ar(PANEL); mod=@__MODULE__)))
        # With `y` omitted the same block draws whole trajectories.
        omitted = (; probe=PANEL.probe, day=PANEL.day, x=PANEL.x)
        for model in (shared_ar, stepped_ar)
            prior_code = BRM.stan_code(SBBRMI(model(omitted); mod=@__MODULE__))
            @test occursin("normal_rng", prior_code)
            @test stanc_ok(prior_code)
        end
        # One scientific description of the expansion.
        d = brm_descriptor(SBBRMI(shared_ar(PANEL); mod=@__MODULE__))
        @test d isa BRMDescriptor
        @test brm_description(d) isa BayesianRegressionModels.BRMDescription
        # Stan's unnamed intercept block samples its log scale and
        # standardized effects directly: the RK coordinates, renamed.
        backend, problem = consumer_problem(shared_ar(PANEL))
        names = coordinate_names(backend.model.layout)
        at(name) = names[coordinate(names, Symbol(name))]
        stan = consumer_stan(shared_ar(PANEL), "ar-errors-shared")
        mapping = [at(:phi) => "phi"; at(:sigma) => "sigma";
            at("pop_mu.beta_pop.1") => "pop_mu_beta_pop.1";
            at("pop_mu.beta_pop.2") => "pop_mu_beta_pop.2";
            at("b_probe.tau.1") => "r_mu_probe_log_scale";
            [at("b_probe.z.$j.1") => "r_mu_probe_xi.$j" for j in eachindex(PROBES)]]
        n = length(names)
        for u in (zeros(n), fill(0.1, n), collect(range(-0.4, 0.4; length=n)))
            check_consumer_stan(problem, stan, mapping, backend, u)
        end
    end
end
