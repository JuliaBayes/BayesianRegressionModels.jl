using Test
using BayesianRegressionModels
using WarmupHMC, Enzyme, BridgeStan
import StanBlocks
using DifferentiationInterface: AutoEnzyme
using LogDensityProblems, LinearAlgebra
using Distributions: Normal, Exponential

# Consumer acceptance: no extension internals or writes to reparametrizer fields.
const PUBLIC_CENTERING_DATA = (
    group = repeat(["a", "b", "c"], inner=2),
    x = [-0.8, 0.2, -0.3, 0.9, 0.4, -0.6],
    t = [0.1, 0.7, -0.5, 0.2, -0.9, 0.6],
    y = [-0.4, 0.5, -0.1, 0.8, 0.3, -0.2],
)
const PUBLIC_SCALAR_BUILDER = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | group)
    y ~ Normal(mu, sigma)
end
const PUBLIC_BUCKET_BUILDER = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | p | group)
    y ~ Normal(mu, sigma)
end
const PUBLIC_CORRELATED_BUILDER = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x + t | p | group)
    y ~ Normal(mu, sigma)
end
const PUBLIC_CENTERING_BUILDERS = (
    ("scalar intercept", PUBLIC_SCALAR_BUILDER),
    ("shared-ID K=1", PUBLIC_BUCKET_BUILDER),
    ("correlated K=3", PUBLIC_CORRELATED_BUILDER),
)

# Read the physical covariance factor from BridgeStan, independently of BRM's
# transform/accessors. At K=1 there is no sampled correlation factor.
function public_centering_factor(problem, block, q)
    K = block.ranef.n_terms
    names = BridgeStan.param_names(problem.model)
    values = BridgeStan.param_constrain(problem.model, q)
    binding = block.ranef.binding
    scales = if block.ranef.id === nothing # plain scalar-intercept fixture
        [exp(values[only(findall(==("$(binding)_log_scale"), names))])]
    else
        [values[only(findall(==("$(binding)_tau.$k"), names))] for k in 1:K]
    end
    K == 1 && return reshape(scales, 1, 1)
    L = [values[only(findall(==("$(binding)_L.$i.$j"), names))]
         for i in 1:K, j in 1:K]
    Diagonal(scales) * L
end

function public_centering_reference(q, block, controls, C)
    mapped = copy(q)
    logjac = 0.0
    K, G = size(block.effects)
    for g in 1:G
        z = zeros(K)
        for k in 1:K
            c = controls[(g - 1) * K + k]
            location = sum(C[k, j] * z[j] for j in 1:k-1; init=0.0)
            scale = C[k, k]
            z[k] = (q[block.effects[k, g]] - c * location) / scale^c
            mapped[block.effects[k, g]] =
                block.target_c * location + scale^block.target_c * z[k]
            logjac += (block.target_c - c) * log(scale)
        end
    end
    logjac, mapped
end

function public_centering_fd(problem, q)
    h = 1e-5
    [begin
        plus, minus = copy(q), copy(q)
        plus[i] += h
        minus[i] -= h
        (LogDensityProblems.logdensity(problem, plus) -
         LogDensityProblems.logdensity(problem, minus)) / (2h)
    end for i in eachindex(q)]
end

function public_weighted_correlation(x, y, weights)
    dx = x .- sum(weights .* x) / sum(weights)
    dy = y .- sum(weights .* y) / sum(weights)
    sum(weights .* dx .* dy) /
        sqrt(sum(weights .* dx.^2) * sum(weights .* dy.^2))
end

function public_centering_scores(sb, problem, block, q, backend)
    K, G = size(block.effects)
    n = 6
    weights = [1.0, 2.0, 1.0, 3.0, 2.0, 1.0]
    model_positions = [q .+ 0.09 .* sin.(i .* eachindex(q) .+ 0.3) for i in 1:n]
    factors = [public_centering_factor(problem, block, v) for v in model_positions]
    innovations = [block.target_c == 0.0 ? v[block.effects] : C \ v[block.effects]
                   for (v, C) in zip(model_positions, factors)]
    model_gradients = [last(LogDensityProblems.logdensity_and_gradient(problem, v))
                       for v in model_positions]
    hs = [block.target_c == 0.0 ? grad[block.effects] : C' * grad[block.effects]
          for (grad, C) in zip(model_gradients, factors)]
    expected = [begin
        k, g = Tuple(CartesianIndices(block.effects)[p])
        positions, gradients = zeros(n), zeros(n)
        for i in 1:n
            C, z = factors[i], innovations[i]
            m = sum(C[k, j] * z[j, g] for j in 1:k-1; init=0.0)
            s = C[k, k]
            positions[i] = t * m + s^t * z[k, g]
            gradients[i] = hs[i][k, g] / s^t
        end
        public_weighted_correlation(positions, gradients, weights)
    end for p in 1:length(block.effects), t in 0.0:0.1:1.0]

    for controls in (zeros(K * G), collect(range(0.17, 0.83, length=K * G)), ones(K * G))
        wrapped = adaptive_centering_problem(sb, problem, backend; centeredness=controls)
        source_positions = [begin
            source = copy(v)
            for g in 1:G, k in 1:K
                c = controls[(g - 1) * K + k]
                m = sum(C[k, j] * z[j, g] for j in 1:k-1; init=0.0)
                source[block.effects[k, g]] = c * m + C[k, k]^c * z[k, g]
            end
            source
        end for (v, C, z) in zip(model_positions, factors, innovations)]
        source_gradients = [last(LogDensityProblems.logdensity_and_gradient(wrapped, v))
                            for v in source_positions]
        rows = WarmupHMC.candidate_scoring_losses(wrapped,
            reduce(hcat, source_positions), reduce(hcat, source_gradients); weights)
        @test length(rows) == 11K * G
        for row in rows
            p = row.pair_number
            candidate_index = round(Int, 10row.candidate) + 1
            @test row.index == vec(block.effects)[p]
            @test row.groups == n
            @test isfinite(row.loss)
            @test row.loss ≈ expected[p, candidate_index] atol=2e-10 rtol=2e-10
        end
    end
end

@testset "public adaptive-centering constructor and controls" begin
    @test Base.get_extension(BayesianRegressionModels,
        :BayesianRegressionModelsWarmupHMCExt) !== nothing
    for (label, builder) in PUBLIC_CENTERING_BUILDERS, centered in (false, true)
        @testset "$label / target=$(Int(centered))" begin
            sb = SBBRMI(builder(PUBLIC_CENTERING_DATA); total_groups=(),
                centered_groups=centered ? [:group] : (), mod=@__MODULE__)
            problem = stan_instantiate(sb)
            names = BridgeStan.param_unc_names(problem.model)
            block = only(adaptive_centering_blocks(sb, names))
            @test block.target_c == Float64(centered)
            @test block.effects == ranef_coordinates(block.ranef, names)
            selected = vec(block.effects) # terms within each group
            untouched = setdiff(eachindex(names), selected)
            q = collect(range(-0.4, 0.6, length=length(names)))
            q[block.log_scales] .= log.([0.7 + 0.55k
                for k in 0:length(block.log_scales)-1])
            C = public_centering_factor(problem, block, q)
            # Cross-check marginal scales, not only the density identity.
            @test sqrt.(diag(C * C')) ≈ exp.(q[block.log_scales])
            backend = AutoEnzyme()
            public_centering_scores(sb, problem, block, q, backend)

            default = adaptive_centering_problem(sb, problem, backend)
            @test first.(WarmupHMC.reparam_sources(default)) == selected
            @test all(p -> last(p).c == block.target_c,
                WarmupHMC.reparam_sources(default))
            lp0, g0 = LogDensityProblems.logdensity_and_gradient(problem, q)
            lpd, gd = LogDensityProblems.logdensity_and_gradient(default, q)
            @test lpd ≈ lp0 atol=1e-10
            @test gd ≈ g0 atol=1e-10

            controls = collect(range(0.17, 0.83, length=length(selected)))
            for input in (0.35, controls)
                c = input isa Real ? fill(input, length(selected)) : input
                wrapped = adaptive_centering_problem(sb, problem, backend;
                    centeredness=input)
                @test [last(p).c for p in WarmupHMC.reparam_sources(wrapped)] == c
                @test first.(WarmupHMC.reparam_sources(wrapped)) == selected
                ir = WarmupHMC.reparametrizer(wrapped)
                ljac, mapped = ir(q)
                reference_ljac, reference_mapped =
                    public_centering_reference(q, block, c, C)
                @test ljac ≈ reference_ljac atol=1e-12
                @test mapped ≈ reference_mapped atol=1e-12
                @test mapped[untouched] == q[untouched]
                inverse_ljac, roundtrip = WarmupHMC.with_logabsdet_jacobian!(
                    similar(q), WarmupHMC.InverseFunctions.inverse(ir), mapped)
                @test roundtrip ≈ q atol=1e-12
                @test inverse_ljac ≈ -ljac atol=1e-12
                lp, gradient = LogDensityProblems.logdensity_and_gradient(wrapped, q)
                @test lp ≈ ljac + LogDensityProblems.logdensity(problem, mapped) atol=1e-10
                @test gradient ≈ public_centering_fd(wrapped, q) atol=2e-5 rtol=2e-5
                if !centered && block.ranef.n_terms == 1
                    sd = only(C)
                    @test mapped[selected] ≈ q[selected] ./ sd .^ c atol=1e-12
                    @test ljac ≈ -sum(c) * log(sd) atol=1e-12
                end

                # Restore through the public WarmupHMC route; it must also
                # synchronize BRM's scorer and correlated accessor state.
                copied = deepcopy(wrapped)
                restored = reverse(c)
                WarmupHMC.restore_reparam_sources!(copied,
                    [idx => WarmupHMC.PartiallyCentered(value)
                     for (idx, value) in zip(selected, restored)])
                fresh = adaptive_centering_problem(sb, problem, backend;
                    centeredness=restored)
                @test [last(p).c for p in WarmupHMC.reparam_sources(wrapped)] == c
                @test WarmupHMC.reparametrizer(copied)(q) ==
                    WarmupHMC.reparametrizer(fresh)(q)
                @test LogDensityProblems.logdensity_and_gradient(copied, q) ==
                    LogDensityProblems.logdensity_and_gradient(fresh, q)
                @test WarmupHMC.reparametrizer(wrapped)(q) == (ljac, mapped)
            end
            @test_throws DimensionMismatch adaptive_centering_problem(sb, problem,
                backend; centeredness=zeros(length(selected) + 1))
            for bad in (-0.1, 1.1, NaN, Inf)
                @test_throws ArgumentError adaptive_centering_problem(sb, problem,
                    backend; centeredness=bad)
            end
        end
    end
end
