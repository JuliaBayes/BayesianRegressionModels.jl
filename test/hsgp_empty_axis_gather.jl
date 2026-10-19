# test/hsgp_empty_axis_gather.jl — an HSGP over an axis with no rows, gathered
# per subject through `ragged(drift, d_subject)` (decision 1fk9b29, `empty`
# prong; empty grouping columns: todo `0jp0opk`).
#
# Run: julia --startup-file=no --project=test test/hsgp_empty_axis_gather.jl
#
# An explicit `domain=` (or `cov=:periodic`) lets the basis exist without
# data, and `ragged` over a zero-row group gives every cell an empty slice. The
# term then contributes only its priors, so each gather model must equal a
# twin that declares the same term but never reads it: the same coordinates,
# density and gradient, on BridgeStan and RK. `reprocess` moves the Stan
# program onto data with rows.

include(joinpath(@__DIR__, "rk_consumer_support.jl"))

# Unsorted subjects; the drift axis has no rows.
const EMPTY_DRIFT = (;
    subject=["b", "a"],
    weight=[0.6, 0.8],
    d_time=Float64[],
    d_subject=String[],
    obs_subject=["a", "b", "a"],
    y=[0.1, 0.2, 0.3],
)
# The same subjects with drift rows inside the domain: b owns row 2, a rows 1 and 3.
const DRIFT_ROWS = merge(EMPTY_DRIFT, (;
    d_time=[0.5, 1.0, -0.7],
    d_subject=["a", "b", "a"],
))

exp_quad_gather(data) = @brm data begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    drift ~ 0 + hsgp(d_time; k=4, domain=(-1.5, 1.5))
    ragged(drift, d_subject)
    ragged(y, obs_subject)
    @plate for i in eachindex(log_CL)
        loc[i] = exp(log_CL[i]) + sum(drift[i])
        y[i] ~ normal(loc[i], sigma)
    end
end

exp_quad_twin(data) = @brm data begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    drift ~ 0 + hsgp(d_time; k=4, domain=(-1.5, 1.5))
    ragged(y, obs_subject)
    @plate for i in eachindex(log_CL)
        loc[i] = exp(log_CL[i])
        y[i] ~ normal(loc[i], sigma)
    end
end

periodic_gather(data) = @brm data begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    drift ~ 0 + hsgp(d_time; k=3, cov=:periodic, period=24.0)
    ragged(drift, d_subject)
    ragged(y, obs_subject)
    @plate for i in eachindex(log_CL)
        loc[i] = exp(log_CL[i]) + sum(drift[i])
        y[i] ~ normal(loc[i], sigma)
    end
end

periodic_twin(data) = @brm data begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    drift ~ 0 + hsgp(d_time; k=3, cov=:periodic, period=24.0)
    ragged(y, obs_subject)
    @plate for i in eachindex(log_CL)
        loc[i] = exp(log_CL[i])
        y[i] ~ normal(loc[i], sigma)
    end
end

const PAIRS = (
    ("exp-quad", exp_quad_gather, exp_quad_twin, 4),
    ("periodic", periodic_gather, periodic_twin, 6),
)

points(n) = (zeros(n), fill(0.1, n), collect(range(-0.4, 0.4; length=n)))

# A backend may drop the unread term from the twin, and with it the term's
# own coordinates (`extra`). They must be the only ones the twin lacks; the
# weights must enter through their standard-normal prior alone, the shared
# coordinates must have the twin's gradient, and moving only the shared
# coordinates must leave the density offset (the extras' own prior) unchanged.
function check_twin(names, twin_names, density, twin_density)
    @test setdiff(twin_names, names) == []
    shared = [findfirst(==(n), names) for n in twin_names]
    extra = setdiff(eachindex(names), shared)
    @test all(i -> occursin("hsgp_d_time", string(names[i])), extra)
    weights = filter(i -> occursin("beta_raw", string(names[i])), extra)
    @test isempty(extra) || !isempty(weights)
    for u in points(length(names))
        value, gradient = density(u)
        twin_value, twin_gradient = twin_density(u[shared])
        @test isfinite(value)
        @test gradient[shared] ≈ twin_gradient atol=1e-10
        @test gradient[weights] ≈ -u[weights] atol=1e-10
        moved = copy(u)
        moved[shared] .+= 0.05
        moved_value, _ = density(moved)
        moved_twin, _ = twin_density(moved[shared])
        @test moved_value - moved_twin ≈ value - twin_value atol=1e-10
        isempty(extra) && @test value ≈ twin_value atol=1e-12
    end
end

@stestset "an empty-axis HSGP gathered per subject on Stan" begin
    saved = deepcopy(EMPTY_DRIFT)
    for (label, gather, twin, B) in PAIRS
        @testset "$label" begin
            sb = SBBRMI(gather(EMPTY_DRIFT); mod=@__MODULE__, total_groups=())
            @test sb.data[:kernel_loc_drift_ragged] == [Int[], Int[]]
            @test size(sb.data[:PHI_hsgp_d_time]) == (0, B)
            stan = consumer_stan(gather(EMPTY_DRIFT), "hsgp-empty-gather-" * label)
            stan_twin = consumer_stan(twin(EMPTY_DRIFT), "hsgp-empty-twin-" * label)
            check_twin(BridgeStan.param_unc_names(stan.model),
                BridgeStan.param_unc_names(stan_twin.model),
                u -> BridgeStan.log_density_gradient(stan.model, u; propto=false),
                u -> BridgeStan.log_density_gradient(stan_twin.model, u; propto=false))
        end
    end
    @test isequal(EMPTY_DRIFT, saved)
end

@stestset "an empty-axis HSGP gathered per subject on RK" begin
    for (label, gather, twin, _) in PAIRS
        @testset "$label" begin
            backend, problem = consumer_problem(gather(EMPTY_DRIFT))
            twin_backend, twin_problem = consumer_problem(twin(EMPTY_DRIFT))
            check_twin(coordinate_names(backend.model.layout),
                coordinate_names(twin_backend.model.layout),
                u -> LogDensityProblems.logdensity_and_gradient(problem, u),
                u -> LogDensityProblems.logdensity_and_gradient(twin_problem, u))
        end
    end
end

@stestset "reprocess moves an empty-axis HSGP gather onto rows" begin
    for (label, gather, _, B) in PAIRS
        @testset "$label" begin
            sb = SBBRMI(gather(EMPTY_DRIFT); mod=@__MODULE__, total_groups=())
            filled = reprocess(sb, DRIFT_ROWS)
            @test BRM.stan_code(filled) == BRM.stan_code(sb)
            @test size(filled.data[:PHI_hsgp_d_time]) == (3, B)
            @test length.(filled.data[:kernel_loc_drift_ragged]) == [1, 2]
            problem = BRM.stan_instantiate(filled;
                path=joinpath(tempdir(), "brm-rk-consumer", "hsgp-empty-filled-" * label * ".stan"))
            n = BridgeStan.param_unc_num(problem.model)
            @test isfinite(BridgeStan.log_density(problem.model, fill(0.1, n); propto=false))
        end
    end
end
