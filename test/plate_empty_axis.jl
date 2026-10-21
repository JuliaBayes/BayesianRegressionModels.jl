# test/plate_empty_axis.jl — a secondary axis with no rows (todo `0jp0opk`).
#
# Run: julia --startup-file=no --project=test test/plate_empty_axis.jl
#
# A design can leave a whole secondary axis empty: a placebo-only arm with no
# placebo rows, a schedule with no observations yet. `ragged(x, group)` over a
# zero-row `group` then gives every cell an empty slice, for a formula LP, a
# raw data column and an observed response alike. An empty slice contributes
# nothing, so each model must equal a twin that never reads the empty axis —
# density and gradient, on both backends — and the same program must replay
# onto data that has rows (and back) with `reprocess`, including the row
# count of a declared axis that sizes a predictor with no data column.

include(joinpath(@__DIR__, "rk_consumer_support.jl"))

# Unsorted subjects; the placebo axis has no rows.
const EMPTY_PLACEBO = (;
    subject=["b", "a"],
    weight=[0.6, 0.8],
    p_time=Float64[],
    p_subject=String[],
    obs_subject=["a", "b", "a"],
    y=[0.1, 0.2, 0.3],
)
# The same subjects with placebo rows: b owns row 2, a rows 1 and 3.
const PLACEBO_ROWS = merge(EMPTY_PLACEBO, (;
    p_time=[0.5, 1.0, 2.0],
    p_subject=["a", "b", "a"],
))

lp_gather(data) = @brm data begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    placebo ~ 0 + p_time
    placebo_rows = groups(p_subject => subject)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        loc[i] = exp(log_CL[i]) + sum(placebo[placebo_rows[i]])
        y[obs[i]] ~ normal(loc[i], sigma)
    end
end

# The placebo predictor and its coefficient exist, but no cell reads them.
lp_twin(data) = @brm data begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    placebo ~ 0 + p_time
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        loc[i] = exp(log_CL[i])
        y[obs[i]] ~ normal(loc[i], sigma)
    end
end

# A predictor with no data column takes its length from the declared axis.
intercept_gather(data) = @brm data begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    p_offset ~ 1
    placebo_rows = groups(p_subject => subject)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        loc[i] = exp(log_CL[i]) + sum(p_offset[placebo_rows[i]])
        y[obs[i]] ~ normal(loc[i], sigma)
    end
end

column_gather(data) = @brm data begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    placebo_rows = groups(p_subject => subject)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        loc[i] = exp(log_CL[i]) + sum(p_time[placebo_rows[i]])
        y[obs[i]] ~ normal(loc[i], sigma)
    end
end

column_twin(data) = @brm data begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        loc[i] = exp(log_CL[i])
        y[obs[i]] ~ normal(loc[i], sigma)
    end
end

# A response observed on the empty placebo axis, beside `y` on a populated one.
const PLACEBO_RESPONSE = merge(EMPTY_PLACEBO, (; py=Float64[]))

observed_in_cell(data) = @brm data begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    placebo_rows = groups(p_subject => subject)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        loc[i] = exp(log_CL[i])
        y[obs[i]] ~ normal(loc[i], sigma)
        py[placebo_rows[i]] ~ normal(loc[i] .* p_time[placebo_rows[i]], sigma)
    end
end

# The same response observed against a collected plate output.
observed_output(data) = @brm data begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    placebo_rows = groups(p_subject => subject)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        ploc[i] = exp(log_CL[i]) .* p_time[placebo_rows[i]]
        y[obs[i]] ~ normal(exp(log_CL[i]), sigma)
        py[placebo_rows[i]] ~ Normal(ploc[i], sigma)
    end
end

const STAN_DIR = mkpath(joinpath(tempdir(), "brm-plate-empty-axis"))

function stan_problem(brmi, name)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    @test StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic=false).ok
    sb, BRM.stan_instantiate(sb; path=joinpath(STAN_DIR, name * ".stan"))
end

points(n) = (zeros(n), fill(0.1, n), collect(range(-0.4, 0.4; length=n)))

# The twin never reads the empty axis, so a backend may drop the placebo
# predictor from it, and with it the predictor's coefficient. `extra` names
# those coordinates: they must be the only ones the twin lacks, and they add
# exactly their own standard-normal prior to the density and gradient.
function check_twin(names, twin_names, density, twin_density, extra)
    @test sort(setdiff(names, twin_names)) == sort(extra)
    @test setdiff(twin_names, names) == []
    shared = [findfirst(==(n), names) for n in twin_names]
    added = [findfirst(==(n), names) for n in extra]
    for u in points(length(names))
        value, gradient = density(u)
        twin_value, twin_gradient = twin_density(u[shared])
        @test isfinite(value)
        @test value ≈ twin_value + sum(logpdf.(Normal(), u[added]); init=0.0) atol=1e-12
        @test gradient[shared] ≈ twin_gradient atol=1e-10
        @test gradient[added] ≈ -u[added] atol=1e-10
    end
end

function check_stan_twin(stan, twin; extra=String[])
    check_twin(BridgeStan.param_unc_names(stan.model),
        BridgeStan.param_unc_names(twin.model),
        u -> BridgeStan.log_density_gradient(stan.model, u; propto=false),
        u -> BridgeStan.log_density_gradient(twin.model, u; propto=false), extra)
end

function check_rk_twin(brmi, twin_brmi; extra=Symbol[])
    backend, problem = consumer_problem(brmi)
    twin_backend, twin_problem = consumer_problem(twin_brmi)
    check_twin(coordinate_names(backend.model.layout),
        coordinate_names(twin_backend.model.layout),
        u -> LogDensityProblems.logdensity_and_gradient(problem, u),
        u -> LogDensityProblems.logdensity_and_gradient(twin_problem, u), extra)
end

@stestset "an empty axis gathers an empty slice per cell on Stan" begin
    saved = deepcopy(EMPTY_PLACEBO)
    sb, stan = stan_problem(lp_gather(EMPTY_PLACEBO), "lp-gather")
    @test sb.data[:kernel_loc_placebo_ragged] == [Int[], Int[]]
    _, twin = stan_problem(lp_twin(EMPTY_PLACEBO), "lp-twin")
    check_stan_twin(stan, twin; extra=["pop_placebo_beta_pop.1"])

    sb, stan = stan_problem(column_gather(EMPTY_PLACEBO), "column-gather")
    @test sb.data[:kernel_loc_p_time_ragged] == [Float64[], Float64[]]
    _, twin = stan_problem(column_twin(EMPTY_PLACEBO), "column-twin")
    check_stan_twin(stan, twin)

    sb, stan = stan_problem(intercept_gather(EMPTY_PLACEBO), "intercept-gather")
    @test sb.data[:n_rows_p_offset_p_subject] == 0
    check_stan_twin(stan, twin; extra=["pop_p_offset_beta_pop.1"])
    @test isequal(EMPTY_PLACEBO, saved)
end

@stestset "an empty axis gathers an empty slice per cell on RK" begin
    check_rk_twin(lp_gather(EMPTY_PLACEBO), lp_twin(EMPTY_PLACEBO);
        extra=[Symbol("pop_placebo.beta_pop.1")])
    check_rk_twin(column_gather(EMPTY_PLACEBO), column_twin(EMPTY_PLACEBO))
    check_rk_twin(intercept_gather(EMPTY_PLACEBO), column_twin(EMPTY_PLACEBO);
        extra=[:p_offset_Intercept])
end

@stestset "an empty observed axis observes nothing" begin
    # The twin observes only `y`.
    _, twin = stan_problem(column_twin(PLACEBO_RESPONSE), "observed-twin")
    for (model, name) in ((observed_in_cell, "observed-in-cell"),
                          (observed_output, "observed-output"))
        _, stan = stan_problem(model(PLACEBO_RESPONSE), name)
        check_stan_twin(stan, twin)
        check_rk_twin(model(PLACEBO_RESPONSE), column_twin(PLACEBO_RESPONSE))
    end
end

@stestset "reprocess moves a program between an empty and a populated axis" begin
    for (model, key, populated) in (
            (lp_gather, :kernel_loc_placebo_ragged, [[2], [1, 3]]),
            (column_gather, :kernel_loc_p_time_ragged, [[1.0], [0.5, 2.0]]),
            (intercept_gather, :kernel_loc_p_offset_ragged, [[2], [1, 3]]))
        empty = SBBRMI(model(EMPTY_PLACEBO); mod=@__MODULE__, total_groups=())
        direct = SBBRMI(model(PLACEBO_ROWS); mod=@__MODULE__, total_groups=())
        @test direct.data[key] == populated
        # The program does not depend on whether the axis has rows.
        @test BRM.stan_code(direct) == BRM.stan_code(empty)

        filled = reprocess(empty, PLACEBO_ROWS; freeze_constants=true)
        @test BRM.stan_code(filled) == BRM.stan_code(empty)
        @test filled.data == direct.data

        emptied = reprocess(direct, EMPTY_PLACEBO; freeze_constants=true)
        @test emptied.data[key] == empty.data[key]
        @test emptied.data == empty.data
    end
    # The replayed populated program evaluates like the one built on rows. The
    # intercept-only predictor is sized by the declared axis's row count, which
    # the replay recomputes (`:axis_extent`) rather than keeping the fit's 0.
    for (model, name) in ((lp_gather, "lp"), (intercept_gather, "intercept"))
        filled = reprocess(SBBRMI(model(EMPTY_PLACEBO); mod=@__MODULE__,
            total_groups=()), PLACEBO_ROWS; freeze_constants=true)
        replayed = BRM.stan_instantiate(filled;
            path=joinpath(STAN_DIR, name * "-replayed.stan"))
        _, direct = stan_problem(model(PLACEBO_ROWS), name * "-direct")
        for u in points(BridgeStan.param_unc_num(direct.model))
            @test BridgeStan.log_density(replayed.model, u; propto=false) ≈
                  BridgeStan.log_density(direct.model, u; propto=false) atol=1e-12
        end
    end
end

@stestset "missing grouping labels are still refused" begin
    # refused: a missing label names no subject, so its row would be dropped
    # from every cell without notice. The generic data-column check refuses it
    # before the gather's own missing-label check is reached.
    unlabelled = merge(PLACEBO_ROWS, (; p_subject=["a", missing, "a"]))
    @test_throws "data column `p_subject` contains `missing` values" SBBRMI(
        lp_gather(unlabelled); mod=@__MODULE__, total_groups=())
    @test_throws "data column `p_subject` contains `missing` values" RKBRMI(
        lp_gather(unlabelled))
end
