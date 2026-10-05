# Synthetic source-emission coverage: modeled secondary rows remain sampled
# when a kernel cell uses dotted calls. No numerical or performance claim.
using Test
using BayesianRegressionModels
using StanBlocks

const BRM = BayesianRegressionModels

const BROADCAST_DATA = (;
    subject = ["b", "empty", "a"],
    t = [[0.1, 0.7], Float64[], [0.2, 0.8, 1.3]],
    inner_y = [[0.3, 0.2], Float64[], [0.1, -0.1, 0.2]],
    row_subject = ["b", "a", "b", "a", "a"],
    item = ["b1", "a1", "b2", "a1", "a2"],
    raw_log = [0.1, -0.2, 0.3, -0.2, 0.4],
    y = [[0.2, -0.1], Float64[], [0.1, 0.3, -0.2]])

const BROADCAST_BODY = """
theta ~ 1 + (1 | p | subject)
effect(theta, Intercept) ~ Normal(0, 1)
sd(:, p) ~ Exponential(1)
sigma ~ Exponential(1)
item_log ~ 0 + (1 | l | item)
sd(:, l) ~ Exponential(1)
pred ~ kernel(t, inner_y, theta, ragged(item_log, row_subject)) do ts, cell_y, a, local_log
    loc = a .* exp(local_log)
    cell_y ~ normal(loc, sigma)
    ts * a
end
y ~ Normal(pred, sigma)
"""

function broadcast_backend(expression; data_only = false)
    body = replace(BROADCAST_BODY, "exp(local_log)" => expression)
    data_only && (body = replace(body,
        "ragged(item_log, row_subject)" => "ragged(raw_log, row_subject)"))
    brmi = Core.eval(@__MODULE__, BRM._brm(body; df = BROADCAST_DATA))
    SBBRMI(brmi; mod = @__MODULE__)
end

@testset "kernel modeled ragged arguments in dotted calls" begin
    original_data = deepcopy(BROADCAST_DATA)
    ordinary = broadcast_backend("exp(local_log)")
    ordinary_code = BRM.stan_code(ordinary)
    @test StanBlocks.stanc_check(ordinary_code; warn_pedantic = false).ok

    # Plain, nested and arithmetic-containing dotted calls all consume the
    # same sampled quantity. The row join is unsorted, interleaved and empty
    # for one subject; a repeated item still shares its random effect.
    for expression in ("exp.(local_log)", "exp.(local_log .+ 0.0)",
                       "exp.(log.(exp.(local_log)))")
        @testset "$expression" begin
            sb = broadcast_backend(expression)
            code = BRM.stan_code(sb)
            display = sprint(show, sb; context = :limit => false)
            @test !occursin(r"\blocal_log\b", display)
            @test occursin("item_log[kernel_rows_local_log]", display)
            @test !haskey(sb.data, :item_log)
            @test sb.data == ordinary.data
            @test sb.data[:kernel_pred_item_log_ragged] == [[1, 3], Int[], [2, 4, 5]]
            @test sb.data[:item_idx] == [3, 1, 4, 1, 2]
            @test StanBlocks.stanc_check(code; warn_pedantic = false).ok
        end
    end

    @testset "raw data ragged arguments retain their local binding" begin
        sb = broadcast_backend("exp.(local_log)"; data_only = true)
        @test sb.data[:kernel_pred_raw_log_ragged] ==
              [[0.1, 0.3], Float64[], [-0.2, -0.2, 0.4]]
        @test StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic = false).ok
    end
    @test isequal(BROADCAST_DATA, original_data)
end

@testset "kernel argument substitution preserves syntactic names" begin
    gathered = :(item_log[rows])
    # A field name and a keyword name are labels; the receiver and argument
    # expressions are values. Qualified dotted calls must traverse both.
    @test BRM._sb_subst_sym(:(local_log.local_log), :local_log, gathered) ==
          :($(gathered).local_log)
    @test BRM._sb_subst_sym(:(f(; local_log = local_log)), :local_log, gathered) ==
          :(f(; local_log = $gathered))
    @test BRM._sb_subst_sym(:(Base.exp.(local_log)), :local_log, gathered) ==
          :(Base.exp.($gathered))
    @test BRM._sb_subst_sym(QuoteNode(:local_log), :local_log, gathered) ==
          QuoteNode(:local_log)
end
