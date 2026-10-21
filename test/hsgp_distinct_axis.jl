# An HSGP basis row is a function of its axis value alone, so the basis is
# evaluated once per distinct axis value and every row gathers its value's
# row (snag `hsgp-basis-evalu-e69381d8`): a prediction grid shared by many
# subjects repeats each time point once per subject. The result is the
# per-row evaluation bit for bit. RK evaluates its basis in its graph, so its
# planner prepares an HSGP term without evaluating one.
#
# Run: julia --startup-file=no --project=test test/hsgp_distinct_axis.jl
using Test
using BayesianRegressionModels
using Distributions: Exponential, Normal
using Random: Xoshiro, randperm
const BRM = BayesianRegressionModels

# The per-row laws, entry by entry.
row_basis(c, x, K) = let (mu, L) = c
    [1 / sqrt(L) * sin(sqrt((k * pi / (2 * L))^2) * ((xi - mu) + L)) for xi in x, k in 1:K]
end
row_periodic(period, x, K) = let w0 = 2pi / period
    hcat([cos(w0 * j * xi) for xi in x, j in 1:K], [sin(w0 * j * xi) for xi in x, j in 1:K])
end
bits(A) = reinterpret(UInt64, vec(Matrix{Float64}(A)))

const GRID = collect(range(-1.1, 1.3; length=97))
const AXES = (
    repeated=repeat(GRID, 7)[randperm(Xoshiro(3), 7 * 97)],
    distinct=2.4 .* rand(Xoshiro(4), 300) .- 1.1,
    signed_zeros=[0.0, -0.0, 0.5, 0.0, -0.0, 0.5],
    integers=[1, 2, 2, 3, 1],
    empty=Float64[],
)

@testset "a repeated axis value is evaluated once, with the per-row bits" begin
    c = (0.1, 1.7)
    for (label, x) in pairs(AXES), K in (1, 5, 16)
        PHI, lambda = BRM._brm_apply_hsgp(c, x, K)
        @test size(PHI) == (length(x), K)
        @test bits(PHI) == bits(row_basis(c, x, K))
        tensor, omega2 = BRM._brm_apply_hsgp((c,), (x,), (K,))
        @test bits(tensor) == bits(PHI)
        @test omega2 == reshape(lambda, K, 1)
        @test bits(BRM._brm_apply_hsgp_periodic(2.3, x, K)) == bits(row_periodic(2.3, x, K))
    end
    # A tensor basis multiplies each row's per-axis entries in axis order.
    x = AXES.repeated
    z = cos.(3 .* x) .+ 0.01 .* (eachindex(x) .% 3)
    fits, K = ((0.1, 1.7), (0.0, 1.5)), (3, 4)
    PHI, omega2 = BRM._brm_apply_hsgp(fits, (x, z), K)
    X, Z = row_basis(fits[1], x, K[1]), row_basis(fits[2], z, K[2])
    oracle = [1.0 * X[i, I[1]] * Z[i, I[2]] for i in eachindex(x), I in CartesianIndices(K)]
    @test bits(PHI) == bits(reshape(oracle, length(x), prod(K)))
    # The caller's axis is read, never written.
    saved = copy(x)
    BRM._brm_apply_hsgp(fits, (x, z), K)
    @test isequal(x, saved)
end

@testset "RK prepares an HSGP term without evaluating its basis" begin
    model = @brm begin
        sigma ~ Exponential(1)
        a ~ 0 + hsgp(t; k=8, domain=(-1.5, 1.5))
        b ~ 0 + hsgp(t; k=6, cov=:periodic, period=2.0)
        m = a + b
        y ~ Normal(m, sigma)
    end
    df = (; t=repeat(GRID, 3), y=sin.(repeat(GRID, 3)))
    brmi = model(df)
    context = BRM._brm_backend_context(brmi)
    for target in (:a, :b)
        kw = (; available_predictors=(:a, :b))
        rk = only(BRM._rk_model_predictor_geometry(brmi, context, target; kw...).terms)
        shared = only(BRM._brm_prepare_predictor_geometry(brmi, context, target; kw...).terms)
        @test !haskey(rk.state, :PHI)
        @test size(shared.state.PHI, 1) == length(df.t)
        # Everything else the planner reads is the shared preparation's.
        for field in keys(rk.state)
            @test isequal(getfield(rk.state, field), getfield(shared.state, field))
        end
    end
end
