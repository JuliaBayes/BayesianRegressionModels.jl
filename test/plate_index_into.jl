# test/plate_index_into.jl — index columns that point into another flat axis
# (snag `axis-witness-and-6a7991c9`, design decision `0ycyfxr`).
#
# Run: julia --startup-file=no --project=test test/plate_index_into.jl
#
# An event schedule numbers its reads across every subject, and each
# observation stores the GLOBAL read it is taken at. A cell computes one value
# per LOCAL read, so it needs per-subject positions: with
# `reads = groups(read_subject => subject)` and `obs = groups(obs_subject =>
# subject)`, the cell reads `indexin(obs_read[obs[i]], reads[i])`, each
# observation's read as its position among the subject's reads, as Julia
# computes it (todo `1nfukuv`). An index that leaves its own subject is
# refused; `something.(indexin(…), 0)` keeps `0` as "no index". The result must match hand-rebased, pre-grouped
# columns exactly, on both backends and on replay.

include(joinpath(@__DIR__, "rk_consumer_support.jl"))

# Unsorted subjects; an interleaved read axis; observations naming global reads.
const READS = (;
    subject=["b", "a", "c"],
    weight=[60.0, 75.0, 90.0],
    read_subject=["a", "b", "a", "c", "b", "a", "c"],
    read_time=[0.5, 0.25, 1.0, 0.5, 2.0, 3.0, 1.5],
    obs_subject=["a", "c", "b", "a", "b", "c", "a"],
    obs_read=[3, 4, 5, 1, 2, 7, 6],
    y=[0.61, 0.52, 0.33, 0.81, 0.79, 0.41, 0.22],
)

# The same model with every secondary column grouped by hand in plate order
# (`subject`) and every index rebased by hand: a's reads are rows 1, 3, 6,
# b's 2, 5 and c's 4, 7 of the read axis.
const PREGROUPED = (;
    subject=READS.subject,
    weight=READS.weight,
    read_time=[[0.25, 2.0], [0.5, 1.0, 3.0], [0.5, 1.5]],
    obs_read=[[2, 1], [2, 1, 3], [1, 2]],
    y=[[0.33, 0.79], [0.61, 0.81, 0.22], [0.52, 0.41]],
)

declared(data) = @brm data begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    reads = groups(read_subject => subject)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        conc = exp.(-exp(log_CL[i]) .* read_time[reads[i]])
        loc[i] = conc[indexin(obs_read[obs[i]], reads[i])]
        y[obs[i]] ~ normal(loc[i], sigma)
    end
end

pregrouped(data) = @brm data begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    @plate for i in eachindex(log_CL)
        conc = exp.(-exp(log_CL[i]) .* read_time[i])
        loc[i] = conc[obs_read[i]]
        y[i] ~ normal(loc[i], sigma)
    end
end

@stestset "into= gathers per-subject read positions on Stan" begin
    saved = deepcopy(READS)
    sb = SBBRMI(declared(READS); mod=@__MODULE__, total_groups=())
    reference = SBBRMI(pregrouped(PREGROUPED); mod=@__MODULE__, total_groups=())
    @test sb.data[:kernel_loc_obs_read_ragged] == PREGROUPED.obs_read
    @test sb.data[:kernel_loc_read_time_ragged] == PREGROUPED.read_time
    @test !haskey(sb.data, :read_subject)
    @test StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic=false).ok
    stan = BRM.stan_instantiate(sb; path=joinpath(mkpath(joinpath(tempdir(),
        "brm-index-into")), "declared.stan"))
    stan_ref = BRM.stan_instantiate(reference; path=joinpath(tempdir(),
        "brm-index-into", "pregrouped.stan"))
    @test BridgeStan.param_unc_names(stan.model) ==
          BridgeStan.param_unc_names(stan_ref.model)
    n = BridgeStan.param_unc_num(stan.model)
    for u in (zeros(n), fill(0.1, n), collect(range(-0.4, 0.4; length=n)))
        @test BridgeStan.log_density(stan.model, u; propto=false) ≈
              BridgeStan.log_density(stan_ref.model, u; propto=false) atol=1e-12
    end
    @test isequal(READS, saved)
end

@stestset "into= gathers per-subject read positions on RK" begin
    backend, problem = consumer_problem(declared(READS))
    ref_backend, ref_problem = consumer_problem(pregrouped(PREGROUPED))
    @test coordinate_names(backend.model.layout) ==
          coordinate_names(ref_backend.model.layout)
    n = LogDensityProblems.dimension(problem)
    for u in (zeros(n), fill(0.1, n), collect(range(-0.4, 0.4; length=n)))
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        ref_value, ref_gradient = LogDensityProblems.logdensity_and_gradient(ref_problem, u)
        @test isfinite(value)
        @test value ≈ ref_value atol=1e-12
        @test gradient ≈ ref_gradient atol=1e-10
    end
end

@stestset "something.(indexin(…), 0) keeps 0 as no index and replays on new data" begin
    # Operation rows: `0` marks an operation that reads nothing.
    ops = merge(READS, (;
        op_subject=["b", "a", "a", "c", "b"],
        op_read=[0, 1, 6, 7, 5],
    ))
    ops_model(data) = @brm data begin
        sigma ~ Exponential(1)
        log_CL ~ 1 + weight + (1 | p | subject)
        reads = groups(read_subject => subject)
        ops = groups(op_subject => subject)
        obs = groups(obs_subject => subject)
        @plate for i in eachindex(log_CL)
            conc = exp.(-exp(log_CL[i]) .* read_time[reads[i]]) .+
                1e-3 * sum(something.(indexin(op_read[ops[i]], reads[i]), 0))
            loc[i] = conc[indexin(obs_read[obs[i]], reads[i])]
            y[obs[i]] ~ normal(loc[i], sigma)
        end
    end
    sb = SBBRMI(ops_model(ops); mod=@__MODULE__, total_groups=())
    @test sb.data[:kernel_loc_op_read_ragged] == [[0, 2], [1, 3], [2]]
    # The cell reads what Julia computes for the same loop.
    reads = groups(ops.read_subject => ops.subject)
    @test [something.(indexin(ops.op_read[r], reads[j]), 0)
           for (j, r) in enumerate(groups(ops.op_subject => ops.subject))] ==
          [[0, 2], [1, 3], [2]]
    # refused: in plain `indexin(…)` a 0 has no position.
    strict(data) = @brm data begin
        sigma ~ Exponential(1)
        log_CL ~ 1 + weight + (1 | p | subject)
        reads = groups(read_subject => subject)
        ops = groups(op_subject => subject)
        @plate for i in eachindex(log_CL)
            loc[i] = exp(log_CL[i]) .+ sum(indexin(op_read[ops[i]], reads[i]))
        end
    end
    @test_throws "`op_read[1]` is 0, which is no index" strict(ops)
    # Replay rebases again: c's two observations swap reads, so their
    # positions swap too.
    swapped = merge(ops, (; obs_read=[3, 7, 5, 1, 2, 4, 6]))
    replayed = reprocess(sb, swapped; freeze_constants=true)
    @test replayed.data[:kernel_loc_obs_read_ragged] == [[2, 1], [2, 1, 3], [2, 1]]
    @test replayed.data[:kernel_loc_op_read_ragged] == [[0, 2], [1, 3], [2]]
end

@stestset "into= refuses an index that leaves its subject" begin
    # refused: an index into another subject's reads would silently read the
    # wrong subject's value (snag axis-witness-and-6a7991c9).
    crossing = merge(READS, (; obs_read=[3, 4, 5, 1, 2, 7, 2]))
    @test_throws "belongs to subject \"a\" but points at row 2 of `read_subject`, which belongs to subject \"b\"" SBBRMI(
        declared(crossing); mod=@__MODULE__, total_groups=())
    @test_throws "belongs to subject \"a\" but points at row 2" RKBRMI(declared(crossing))
    # refused: a row number past the read axis has no read to point at.
    beyond = merge(READS, (; obs_read=[3, 4, 5, 1, 2, 7, 8]))
    @test_throws "row 7 of `obs_read` is 8, outside the 7 rows of `read_subject`" SBBRMI(
        declared(beyond); mod=@__MODULE__, total_groups=())
end

@testset "indexin reads are checked at the formula boundary" begin
    # refused: the two bindings list different cells, so `reads[i]` is not
    # the subject `obs[i]` belongs to.
    other_cells = :(@brm merge(READS, (; label=READS.subject)) begin
        log_CL ~ 1 + (1 | p | subject)
        reads = groups(read_subject => label)
        obs = groups(obs_subject => subject)
        @plate for i in eachindex(log_CL)
            loc[i] = sum(indexin(obs_read[obs[i]], reads[i]))
        end
    end)
    @test_throws "list different cells" macroexpand(@__MODULE__, other_cells)
    # refused: positions are among the target's indices in index order.
    ordered = :(@brm READS begin
        log_CL ~ 1 + (1 | p | subject)
        reads = groups(read_subject => subject; order = read_time)
        obs = groups(obs_subject => subject)
        @plate for i in eachindex(log_CL)
            loc[i] = sum(indexin(obs_read[obs[i]], reads[i]))
        end
    end)
    @test_throws "bind `obs` and `reads` without `order`" macroexpand(@__MODULE__, ordered)
    # refused: the formula-level grouped observation is retired (todo 1nfukuv).
    response = :(@brm READS begin
        mu ~ 1
        ragged(y, obs_subject; into=read_subject) ~ Normal(mu, 1)
    end)
    @test_throws "are retired" macroexpand(@__MODULE__, response)
end
