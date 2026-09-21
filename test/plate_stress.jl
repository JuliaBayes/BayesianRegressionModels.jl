using Test
using BayesianRegressionModels
using BridgeStan
using LogDensityProblems
using StanBlocks

const STRESS_CACHE = joinpath(tempdir(), "brm-plate-stress")
const RUN_BRIDGESTAN = get(ENV, "BRM_PLATE_STRESS_RUNTIME", "1") != "0"
const RUN_GAPS = get(ENV, "BRM_PLATE_STRESS_GAPS", "1") != "0"
const STRESS_CASE = get(ENV, "BRM_PLATE_STRESS_CASE", "all")

function stanc_accepts(model)
    result = StanBlocks.stanc_check(StanBlocks.stan_code(model); warn_pedantic=false)
    result.ok || @error "stanc rejected BRM PLATE stress case" output=result.output
    result.ok
end

function bridgestan_accepts(model; expected_dimension=nothing)
    code = StanBlocks.stan_code(model)
    path = joinpath(STRESS_CACHE, string(hash(code)) * ".stan")
    problem = StanBlocks.stan_instantiate(model; path)
    dimension = LogDensityProblems.dimension(problem)
    q = zeros(dimension)
    lp = LogDensityProblems.logdensity(problem, q)
    lp_grad, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
    dimension_ok = isnothing(expected_dimension) || dimension == expected_dimension
    dimension_ok && isfinite(lp) && isfinite(lp_grad) &&
        length(gradient) == dimension && all(isfinite, gradient)
end

# Fixed-width correlated group effect: the first BRM migration target for
# `(1 + x | ID | group)` buckets and fixed-width biomarker parameter vectors.
StanBlocks.@slic brm_fixed_correlated_cell(
    k::int,
    L::matrix[k, k],
    tau::vector[k],
) = begin
    z::vector[k] ~ std_normal()
    return diag_pre_multiply(tau, L) * z
end

const fixed_correlated = StanBlocks.@slic (;
    n_groups=5,
    k=3,
    y=[0.2, -0.1, 0.3, 0.0, 0.4],
) begin
    L::cholesky_factor_corr[k] ~ lkj_corr_cholesky(2.0)
    tau::vector[k] ~ normal(0.0, 1.0; lower=0.0)
    b ~ plate(y; outer=(n_groups,)) do yi
        cell ~ brm_fixed_correlated_cell(k, L, tau)
        yi ~ normal(cell[1], 0.5)
        cell
    end
end

# Shipped ragged BRM composition: informative ragged Cholesky parameters live
# at the top level while the called PLATE cell consumes K[g] and L[g].
StanBlocks.@slic brm_ragged_latent_cell(k::int, L::matrix[k, k]) = begin
    z::vector[k] ~ std_normal()
    return L * z
end

const ragged_correlated = StanBlocks.@slic (;
    K=[1, 2, 4],
    y=0.2,
) begin
    L::cholesky_factor_corr[K] ~ lkj_corr_cholesky(2.0)
    b ~ plate(; outer=length(K)) do g
        cell ~ brm_ragged_latent_cell(K[g], L[g])
        cell
    end
    y ~ normal(sum(b[3]), 1.0)
end

# Likelihood-bearing scalar cells exercise captured shared parameters, sliced
# data, block routing, and deterministic PLATE results.
const scalar_likelihood = StanBlocks.@slic (;
    y=[0.2, -0.1, 0.3, 0.0, 0.4, -0.2],
    mu0=0.25,
) begin
    sigma ~ normal(0.0, 1.0; lower=0.0)
    theta ~ plate(y; outer=(6,)) do yi
        z ~ std_normal()
        mu = mu0 + sigma * z
        yi ~ normal(mu, sigma)
        mu
    end
end


# Dense N-D vector cells exercise logical-cell axes plus multiple PLATE axes.
const nd_vector = StanBlocks.@slic (;y=[0.2 -0.1 0.3; 0.0 0.4 -0.2], k=2) begin
    theta ~ plate(y; outer=(2, 3)) do yi
        z::vector[k] ~ std_normal()
        yi ~ normal(z[1], 1.0)
        z
    end
end

# Two independent grouping factors should remain two independent plates rather
# than forcing a monolithic crossed allocator.
const crossed_groups = StanBlocks.@slic (;
    y=[0.2, -0.1, 0.3, 0.0],
    subject=[1, 1, 2, 2],
    item=[1, 2, 1, 2],
) begin
    b_subject ~ plate(; outer=(2,)) do s
        z_subject ~ std_normal()
        z_subject
    end
    b_item ~ plate(; outer=(2,)) do i
        z_item ~ std_normal()
        z_item
    end
    y ~ normal(b_subject[subject] + b_item[item], 1.0)
end


# Known gaps. `@test_broken` makes the suite green today, but turns a newly
# supported capability into an unexpected pass that must be promoted above.
const matrix_cell_gap = StanBlocks.@slic (;n=3, k=2) begin
    theta ~ plate(; outer=(n,)) do g
        z::vector[k * k] ~ std_normal()
        to_matrix(z, k, k)
    end
end

const constrained_vector = StanBlocks.@slic (;n=3, k=3, y=[1, 2, 3]) begin
    p ~ plate(y; outer=(n,)) do yg
        cell::simplex[k] ~ dirichlet(rep_vector(1.0, k))
        yg ~ categorical(cell)
        cell
    end
end

# Fully dead plates lower WHOLE to generated quantities (`parameters {}` empty,
# stanblocks-use §28) — so the pre-September unobserved simplex spelling below
# is a GQ-routing pin, not a fitted stress. It keeps its own testset.
const prior_only_simplex_plate = StanBlocks.@slic (;n=3, k=3) begin
    p ~ plate(; outer=(n,)) do g
        cell::simplex[k] ~ dirichlet(rep_vector(1.0, k))
        cell
    end
end

const vararg_cell_gap = StanBlocks.@slic (;
    x=[0.1, 0.2, 0.3],
    y=[0.3, 0.2, 0.1],
) begin
    theta ~ plate(x, y; outer=(3,)) do cells...
        z ~ std_normal()
        z + sum(cells)
    end
end

# Ragged PLATE input slices are typed as per-cell vectors, so they can feed a
# typed BRM subject/series likelihood cell alongside ragged constrained inputs.
StanBlocks.@slic brm_ragged_observed_cell(
    k::int,
    L::matrix[k, k],
    y::vector[k],
) = begin
    z::vector[k] ~ std_normal()
    b = L * z
    y ~ normal(b, 0.5)
    return b
end

const ragged_input = StanBlocks.@slic (;
    K=[1, 2, 4],
    groups=[1, 2, 3],
    y=[[0.2], [-0.1, 0.3], [0.0, 0.4, -0.2, 0.1]],
) begin
    L::cholesky_factor_corr[K] ~ lkj_corr_cholesky(2.0)
    b ~ plate(groups, y; outer=length(K)) do g, yg
        cell ~ brm_ragged_observed_cell(K[g], L[g], yg)
        cell
    end
end

const crossed_local_reuse = StanBlocks.@slic (;
    y=[0.2, -0.1, 0.3, 0.0],
    subject=[1, 1, 2, 2],
    item=[1, 2, 1, 2],
) begin
    b_subject ~ plate(; outer=(2,)) do s
        z ~ std_normal()
        z
    end
    b_item ~ plate(; outer=(2,)) do i
        z ~ std_normal()
        z
    end
    y ~ normal(b_subject[subject] + b_item[item], 1.0)
end

# Uniform-but-ragged-typed per-group predictions aggregated downstream: the
# BRM per-group-prediction-then-aggregate shape through the `as_matrix` cast.
const as_matrix_agg = StanBlocks.@slic (;
    nsub=2,
    tcol=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    w=[1.0, 2.0],
    y=[0.1, 0.2, 0.3],
) begin
    a ~ std_normal()
    pred ~ plate(tcol; outer=(nsub,)) do t
        a .* t
    end
    I_agg = as_matrix(pred) * w
    y ~ normal(I_agg, 1.0)
end

# `@plate for` over BRM grouped intercepts: model-scope array plus per-cell
# observation, the annotated-loop spelling of the scalar-likelihood shape.
const annotated_plate_groups = StanBlocks.@slic (;
    y=[0.2, -0.1, 0.3, 0.0, 0.4, -0.2],
    mu0=0.25,
) begin
    sigma ~ normal(0.0, 1.0; lower=0.0)
    @plate for i in 1:6
        b[i] ~ normal(mu0, 1.0)
        y[i] ~ normal(b[i], sigma)
    end
end

# `@scan` AR(1) latent with the observation outside: the BRM `ar`-term shape,
# centered parameterization.
const annotated_scan_ar1 = StanBlocks.@slic (;
    y=[0.2, -0.1, 0.3, 0.0, 0.4],
    T=5,
) begin
    phi ~ normal(0.0, 0.5)
    s ~ exponential(1.0)
    @scan begin
        h[1] ~ normal(0.0, 1.0)
        for t in 2:T
            h[t] ~ normal(phi * h[t-1], s)
        end
    end
    y ~ normal(h, 1.0)
end

# `@scan` inside `@plate for`: per-subject AR(1) state space, the BRM
# hierarchical time-series shape, with per-column observation.
const annotated_nested_scan = StanBlocks.@slic (;
    y=[0.2 -0.1; 0.3 0.0; 0.4 -0.2],
    S=2,
    T=3,
) begin
    phi ~ normal(0.0, 0.5)
    s ~ exponential(1.0)
    @plate for j in 1:S
        @scan begin
            h[1] ~ normal(0.0, 1.0)
            for t in 2:T
                h[t] ~ normal(phi * h[t-1], s)
            end
        end
        y[:, j] ~ normal(h, 1.0)
    end
end

# Untyped fresh `~` cells with vector-shaped family arguments infer the
# broadcast shape, byte-identically to the typed spelling (regression pin for
# the plate-untyped-vector mis-typing fix).
const untyped_vector_cell = StanBlocks.@slic (;S=2, T=3) begin
    hvec ~ normal(0.0, 1.0; n=T)
    yy ~ plate(; outer=(S,)) do j
        yj ~ normal(hvec, 1.0)
        yj
    end
end

const typed_vector_cell = StanBlocks.@slic (;S=2, T=3) begin
    hvec ~ normal(0.0, 1.0; n=T)
    yy ~ plate(; outer=(S,)) do j
        yj::vector[T] ~ normal(hvec, 1.0)
        yj
    end
end

const EXPECTED_DIMENSIONS = Dict(
    "constrained vector" => 6,
    "as-matrix aggregation" => 1,
    "annotated plate groups" => 7,
    "annotated scan AR(1)" => 7,
    "annotated nested scan" => 8,
)


@info "BRM PLATE stress environment" StanBlocks=Base.pkgversion(StanBlocks) BridgeStan=Base.pkgversion(BridgeStan) RUN_BRIDGESTAN RUN_GAPS STRESS_CASE

@testset "StanBlocks PLATE — BRM acceptance stress" begin
    @testset "transpile and stanc" begin
        for (name, model) in (
            "scalar likelihood" => scalar_likelihood,
            "fixed correlated" => fixed_correlated,
            "ragged correlated" => ragged_correlated,
            "ragged input" => ragged_input,
            "constrained vector" => constrained_vector,
            "N-D vector" => nd_vector,
            "crossed groups" => crossed_groups,
            "crossed local reuse" => crossed_local_reuse,
            "as-matrix aggregation" => as_matrix_agg,
            "annotated plate groups" => annotated_plate_groups,
            "annotated scan AR(1)" => annotated_scan_ar1,
            "annotated nested scan" => annotated_nested_scan,
        )
            STRESS_CASE == "all" || STRESS_CASE == name || continue
            @testset "$name" begin
                @info "Running BRM PLATE compiler case" name
                transpiles = StanBlocks.transpiles(model; re=false)
                @test transpiles
                transpiles && @test stanc_accepts(model)
            end
        end
    end

    @testset "BridgeStan runtime" begin
        if RUN_BRIDGESTAN
            for (name, model) in (
                "scalar likelihood" => scalar_likelihood,
                "fixed correlated" => fixed_correlated,
                "ragged correlated" => ragged_correlated,
                "ragged input" => ragged_input,
                "constrained vector" => constrained_vector,
                "crossed local reuse" => crossed_local_reuse,
                "as-matrix aggregation" => as_matrix_agg,
                "annotated plate groups" => annotated_plate_groups,
                "annotated scan AR(1)" => annotated_scan_ar1,
                "annotated nested scan" => annotated_nested_scan,
            )
                STRESS_CASE == "all" || STRESS_CASE == name || continue
                @info "Running BRM PLATE BridgeStan case" name
                expected_dimension = get(EXPECTED_DIMENSIONS, name, nothing)
                @test bridgestan_accepts(model; expected_dimension)
            end
        else
            @info "Skipping BridgeStan runtime gate (BRM_PLATE_STRESS_RUNTIME=0)"
        end
    end

    @testset "prior-only plate routes to generated quantities" begin
        if STRESS_CASE == "all" || STRESS_CASE == "prior-only simplex"
            @info "Running BRM PLATE GQ-routing pin"
            transpiles = StanBlocks.transpiles(prior_only_simplex_plate; re=false)
            @test transpiles
            if transpiles
                @test stanc_accepts(prior_only_simplex_plate)
                code = StanBlocks.stan_code(prior_only_simplex_plate)
                @test occursin(r"parameters\s*\{\s*\}", code)
                @test occursin("dirichlet_vector_rng", code)
            end
        end
    end

    @testset "untyped vector cell infers broadcast shape" begin
        if STRESS_CASE == "all" || STRESS_CASE == "untyped vector cell"
            @info "Running BRM PLATE untyped-cell pin"
            @test StanBlocks.transpiles(untyped_vector_cell; re=false)
            @test stanc_accepts(untyped_vector_cell)
            @test StanBlocks.stan_code(untyped_vector_cell) ==
                StanBlocks.stan_code(typed_vector_cell)
        end
    end

    @testset "known capability gaps" begin
        if RUN_GAPS
            @test_broken StanBlocks.transpiles(matrix_cell_gap; re=false)
            @test_broken StanBlocks.transpiles(vararg_cell_gap; re=false)
            @test_broken occursin("reduce_sum", StanBlocks.stan_code(scalar_likelihood))
        else
            @info "Skipping expected-failure probes (BRM_PLATE_STRESS_GAPS=0)"
        end
    end
end
