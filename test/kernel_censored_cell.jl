# test/kernel_censored_cell.jl — the StanBlocks distribution-HOF token spelling
# works inside a `kernel(...)` do-block cell (snag `censored-in-kern-87b50f51`).
#
# The `censored` / `truncated` / `interval_censored` formula markers are
# top-level-`@brm` only: a kernel cell is handed to the StanBlocks plate as-is
# (`_sb_kernel_doblock!`) and never reaches BRM's likelihood dispatch. In-cell,
# the supported spelling is the StanBlocks token form — a bare lowercase family
# token plus positional arguments (`yy ~ censored(normal, mu, sigma;
# lower=lloq)`), per stanblocks-use §8.1 — which lowers through the same
# `lower_clamping` / `lower_conditioning` / `interval_evidence_impl` builtins
# as plain-`@slic` use. The `Distributions.jl` call form is not valid there in
# either case (`Normal(...)` or `normal(...)`).

using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Exponential, Normal

knc_stanc_ok(model) =
    StanBlocks.stanc_check(StanBlocks.stan_code(model); warn_pedantic = false).ok

knc_df() = (;
    site = ["a", "b"],
    t_grid = [collect(1.0:4) for _ in 1:2],
    dv = [[1.0, 0.5, -0.5, -1.0], [0.8, 0.2, -0.2, -0.8]],
)

knc_censored(df) = @brm df begin
    sigma ~ Exponential(1)
    log_a ~ 1 + (1 | site)
    pred ~ kernel(t_grid, dv, log_a) do ts, yy, la
        mu = ts .* exp(la)
        yy ~ censored(normal, mu, sigma; lower=-1.0)
        mu
    end
end

knc_truncated(df) = @brm df begin
    sigma ~ Exponential(1)
    log_a ~ 1 + (1 | site)
    pred ~ kernel(t_grid, dv, log_a) do ts, yy, la
        mu = ts .* exp(la)
        yy ~ truncated(normal, mu, sigma; lower=-1.0)
        mu
    end
end

knc_interval(df) = @brm df begin
    sigma ~ Exponential(1)
    log_a ~ 1 + (1 | site)
    pred ~ kernel(t_grid, dv, log_a) do ts, yy, la
        mu = ts .* exp(la)
        yy ~ interval_censored(normal, -1.0, 1.0, mu, sigma)
        mu
    end
end

@testset "kernel(...) cell: censored token form transpiles + stanc" begin
    sb = SBBRMI(knc_censored(knc_df()); mod = @__MODULE__)
    @test StanBlocks.stan.transpiles(sb.model)
    code = StanBlocks.stan_code(sb.model)
    @test occursin("lower_clamping_normal_lpdf", code)
    @test occursin("lower_clamping_normal_lpdfs", code)
    @test occursin("lower_clamping_vector_normal_rng", code)
    @test occursin("~ lower_clamping_normal(", code)
    @test knc_stanc_ok(sb.model)
end

@testset "kernel(...) cell: truncated token form transpiles + stanc" begin
    sb = SBBRMI(knc_truncated(knc_df()); mod = @__MODULE__)
    @test StanBlocks.stan.transpiles(sb.model)
    code = StanBlocks.stan_code(sb.model)
    @test occursin("lower_conditioning_normal_lpdf", code)
    @test occursin("~ lower_conditioning_normal(", code)
    @test knc_stanc_ok(sb.model)
end

@testset "kernel(...) cell: interval_censored token form transpiles + stanc" begin
    sb = SBBRMI(knc_interval(knc_df()); mod = @__MODULE__)
    @test StanBlocks.stan.transpiles(sb.model)
    code = StanBlocks.stan_code(sb.model)
    @test occursin("interval_evidence_impl_normal_lpdf", code)
    @test knc_stanc_ok(sb.model)
end

@testset "kernel(...) cell: distribution call forms still fail loudly" begin
    # `Normal(...)` hits the deliberate UnionAll rejection (StanBlocks decision
    # `3bbtrv`); `normal(...)` traces to `anything` in the family-token
    # position. Neither message is matched here — the wording is owned by
    # StanBlocks — only that both stay loud instead of silently accepted.
    df = knc_df()
    m_cap = @brm df begin
        sigma ~ Exponential(1)
        log_a ~ 1 + (1 | site)
        pred ~ kernel(t_grid, dv, log_a) do ts, yy, la
            mu = ts .* exp(la)
            yy ~ censored(Normal(mu, sigma); lower=-1.0)
            mu
        end
    end
    sb_cap = SBBRMI(m_cap; mod = @__MODULE__)
    @test_throws Exception StanBlocks.stan_code(sb_cap.model)
    m_call = @brm df begin
        sigma ~ Exponential(1)
        log_a ~ 1 + (1 | site)
        pred ~ kernel(t_grid, dv, log_a) do ts, yy, la
            mu = ts .* exp(la)
            yy ~ censored(normal(mu, sigma); lower=-1.0)
            mu
        end
    end
    sb_call = SBBRMI(m_call; mod = @__MODULE__)
    @test_throws Exception StanBlocks.stan_code(sb_call.model)
end
