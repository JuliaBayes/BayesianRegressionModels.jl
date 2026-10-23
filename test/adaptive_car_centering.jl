# test/adaptive_car_centering.jl — online adaptive-centering contract for the
# areal term `car(area; edges=(from, to), type=:icar | :bym2)`.
#
# Every area is a triangular pair of cells: its structured `phi` cell (location
# 0, scale `a = sigma * sqrt(rho / s_c)`, `sigma` under `:icar`) and, under
# `:bym2`, its unstructured `theta` cell (scale `sigma * sqrt(1 - rho)`, `sigma`
# for an island, location `intercept + a * phi`). `c=0` is the emitted
# standardized frame; `c=1` is the spatial component and the area's total
# `intercept + effects[k]`. Covers metadata
# resolution against the compiled model, the exact transform against an
# independent map, the c=1 meaning against BridgeStan's transformed `effects`,
# candidate-frame invariance, Enzyme gradients, the joint and learned-order
# wrappers beside a random-effect block, the controls report, the generated
# skip, and one short online warmup.
#
#     julia --project=test test/adaptive_car_centering.jl
using Test
using BayesianRegressionModels
using BridgeStan
using Distributions: Normal, Beta, Uniform, Exponential
import DifferentiationInterface as DI
import Enzyme
using LinearAlgebra
using LogDensityProblems
import StanBlocks
using WarmupHMC
using Random: Xoshiro, randn

const BRM = BayesianRegressionModels
const BS = BridgeStan
const CAR_AC_EXT = Base.get_extension(
    BayesianRegressionModels, :BayesianRegressionModelsWarmupHMCExt,
)
const CAR_AC_CACHE = joinpath(tempdir(), "brm-adaptive-car-centering")

# Areas "a".."h" (test/car_term.jl's graph). Component {a, b, c, d, h}: a–b,
# b–c, c–d, a–c, d–h; component {e, f}; "g" is an island with rows; "h" has
# no rows. Sorted areas a..h are 1..8; `struct_idx == [1, 2, 3, 4, 6, 7, 0, 5]`.
const CAR_AC_EDGES = (["a", "b", "c", "a", "d", "e"], ["b", "c", "d", "c", "h", "f"])
const CAR_AC_DF = (; area=["a", "b", "c", "d", "e", "f", "g", "a", "e"],
    grp=["u", "v", "u", "v", "u", "v", "u", "v", "u"],
    x=[0.1, -0.4, 0.8, 0.3, -1.0, 0.5, 0.2, -0.3, 0.6],
    y=[0.3, -0.2, 0.5, 1.1, -0.7, 0.0, 0.4, -0.1, 0.9],
    from=CAR_AC_EDGES[1], to=CAR_AC_EDGES[2])

car_ac_model(type) = type === :bym2 ? (@brm CAR_AC_DF begin
    mu ~ 1 + car(area; edges=(from, to), type=:bym2)
    y ~ Normal(mu, 1.0)
end) : (@brm CAR_AC_DF begin
    mu ~ 1 + car(area; edges=(from, to), type=:icar)
    y ~ Normal(mu, 1.0)
end)

function car_ac_instantiate(sb)
    code = StanBlocks.stan_code(sb.model)
    mkpath(CAR_AC_CACHE)
    StanBlocks.stan_instantiate(sb.model;
        path=joinpath(CAR_AC_CACHE, string(hash(code)) * ".stan"))
end

const CAR_AC_SB = Dict(type => SBBRMI(car_ac_model(type); mod=@__MODULE__)
                       for type in (:bym2, :icar))
const CAR_AC_PROBLEM = Dict(type => car_ac_instantiate(sb) for (type, sb) in CAR_AC_SB)
car_ac_names(type) = BS.param_unc_names(CAR_AC_PROBLEM[type].model)

index_of(names, name) = only(findall(==(name), names))
indices_of(names, prefix) = findall(startswith(prefix * "."), names)

inv_logit(x) = 1 / (1 + exp(-x))

# The independent map, written from the emitted law alone (`_sb_car_bym2`):
# per area, `phi` from its coordinate `a^c * phi`, then `theta` from
# `c * (intercept + a * phi) + s^c * theta`. Indices come from the compiled
# names, the graph from the Stan data. `controls` follow the wrapper's
# documented pair order: the `phi` cells in emitted order, then one `theta` cell
# per area.
function manual_car_map(x, names, data, type, controls)
    phi_idx = indices_of(names, "car_mu_area_phi")
    sigma = exp(x[index_of(names, "car_mu_area_sigma")])
    y = copy(x)
    ljac = 0.0
    struct_idx = data[:car_mu_area_struct_idx]
    if type === :icar
        for (j, i) in enumerate(phi_idx)
            y[i] = x[i] / sigma^controls[j]
            ljac -= controls[j] * log(sigma)
        end
        return ljac, y
    end
    rho = inv_logit(x[index_of(names, "car_mu_area_rho")])
    intercept = x[index_of(names, "pop_mu_beta_pop.1")]
    inv_scale = data[:car_mu_area_inv_scale]
    theta_idx = indices_of(names, "car_mu_area_theta")
    n = length(phi_idx)
    area_of = [findfirst(==(j), struct_idx) for j in 1:n]
    a = [sigma * sqrt(rho * inv_scale[area_of[j]]) for j in 1:n]
    phi = [x[phi_idx[j]] / a[j]^controls[j] for j in 1:n]
    for j in 1:n
        y[phi_idx[j]] = phi[j]
        ljac -= controls[j] * log(a[j])
    end
    for (k, i) in enumerate(theta_idx)
        j = struct_idx[k]
        c = controls[n + k]
        s = j > 0 ? sigma * sqrt(1 - rho) : sigma
        m = intercept + (j > 0 ? a[j] * phi[j] : 0.0)
        y[i] = (x[i] - c * m) / s^c
        ljac -= c * log(s)
    end
    ljac, y
end

function set_car_sources!(state, ir, controls)
    ir.pairs .= map(ir.pairs, controls) do (idx, value), c
        idx => WarmupHMC.Reparametrization(
            value.target, WarmupHMC.PartiallyCentered(Float64(c)), value.args...)
    end
    CAR_AC_EXT._sync_sources!(state, ir)
end

finite_difference(f, x; step=1e-6) = [begin
    plus, minus = copy(x), copy(x)
    plus[i] += step
    minus[i] -= step
    (f(plus) - f(minus)) / (2step)
end for i in eachindex(x)]

@testset "car metadata resolves the compiled graph and scales" begin
    names = car_ac_names(:bym2)
    sb = CAR_AC_SB[:bym2]
    block = only(BRM._adaptive_car_centering_blocks(sb, names))
    @test block.logical === :mu
    @test block.term === :car_mu_area
    @test block.bym2
    @test block.structured == indices_of(names, "car_mu_area_phi")
    @test block.unstructured == indices_of(names, "car_mu_area_theta")
    @test length(block.structured) == 7 && length(block.unstructured) == 8
    @test block.struct_idx == [1, 2, 3, 4, 6, 7, 0, 5]
    @test block.phi_area == [1, 2, 3, 4, 8, 5, 6]
    @test block.areas == ["a", "b", "c", "d", "e", "f", "g", "h"]
    @test block.log_inv_scale ≈ log.(sb.data[:car_mu_area_inv_scale][block.phi_area])
    @test block.sigma == index_of(names, "car_mu_area_sigma")
    @test block.rho == index_of(names, "car_mu_area_rho")
    @test (block.sigma_lower, block.sigma_upper) == (0.0, Inf)
    @test (block.rho_lower, block.rho_upper) == (0.0, 1.0)
    @test block.intercept == index_of(names, "pop_mu_beta_pop.1")

    # Cell scales against the emitted law at sigma = 0.7, rho = 0.3.
    x = zeros(length(names))
    x[block.sigma] = log(0.7)
    x[block.rho] = log(0.3 / 0.7)
    inv_scale = sb.data[:car_mu_area_inv_scale]
    expected = vcat([0.7 * sqrt(0.3 * inv_scale[k]) for k in block.phi_area],
                    [k == 7 ? 0.7 : 0.7 * sqrt(0.7) for k in 1:8])
    @test [exp(BRM._adaptive_car_log_scale(x, block, cell)) for cell in 1:15] ≈
          expected atol=1e-14
    # The shares stay finite as rho reaches either end of [0, 1].
    for r in (-800.0, 800.0)
        x[block.rho] = r
        @test all(isfinite, BRM._adaptive_car_log_scale(x, block, cell) for cell in 1:15)
    end

    icar = only(BRM._adaptive_car_centering_blocks(CAR_AC_SB[:icar], car_ac_names(:icar)))
    @test !icar.bym2
    @test isempty(icar.unstructured)
    @test icar.rho == 0
    @test icar.intercept == 0
    @test length(icar.structured) == 7

    # A configured scale and a narrowed share are read through their bounds.
    configured = SBBRMI(@brm(CAR_AC_DF, begin
        mu ~ 1 + car(area; edges=(from, to), type=:bym2)
        sd(mu, car(area)) ~ Normal(0.0, 0.5)
        mixing(mu, car(area)) ~ Uniform(0.2, 0.9)
        y ~ Normal(mu, 1.0)
    end); mod=@__MODULE__)
    narrowed = only(BRM._adaptive_car_centering_blocks(configured, names))
    @test (narrowed.rho_lower, narrowed.rho_upper) == (0.2, 0.9)
    x[narrowed.sigma] = log(0.7)
    x[narrowed.rho] = 0.4
    rho = 0.2 + 0.7 * inv_logit(0.4)
    @test exp(BRM._adaptive_car_log_scale(x, narrowed, 8)) ≈ 0.7 * sqrt(1 - rho) atol=1e-14
    @test exp(BRM._adaptive_car_log_scale(x, narrowed, 1)) ≈
          0.7 * sqrt(rho * inv_scale[1]) atol=1e-14

    missing_theta = filter(!=("car_mu_area_theta.8"), names)
    @test_throws "unstructured" BRM._adaptive_car_centering_blocks(sb, missing_theta)
end

@testset "car transform, Jacobian, inverse, frames, and Enzyme gradient ($type)" for type in (:bym2, :icar)
    names = car_ac_names(type)
    sb = CAR_AC_SB[type]
    blocks = BRM._adaptive_car_centering_blocks(sb, names)
    state, ir = CAR_AC_EXT._adaptive_car_centering_reparametrizer(blocks)
    block = only(blocks)
    n_cells = type === :bym2 ? 15 : 7
    @test first.(ir.pairs) == vcat(block.structured, block.unstructured)
    @test length(ir.pairs) == n_cells

    x = collect(range(-0.6, 0.7; length=length(names)))
    x[block.sigma] = log(0.8)
    type === :bym2 && (x[block.rho] = -0.3)
    zeros_c = zeros(n_cells)
    set_car_sources!(state, ir, zeros_c)
    @test isequal(ir(x), (0.0, x))

    controls = collect(range(0.05, 0.95; length=n_cells))
    ones_c = ones(n_cells)
    for c in (controls, ones_c, reverse(controls))
        set_car_sources!(state, ir, c)
        ljac, mapped = ir(x)
        expected_ljac, expected = manual_car_map(x, names, sb.data, type, c)
        @test ljac ≈ expected_ljac atol=1e-13
        @test mapped ≈ expected atol=1e-13
        inverse_ljac, roundtrip = WarmupHMC._inverse_with_logabsdet_jacobian(ir, mapped)
        @test inverse_ljac ≈ -ljac atol=1e-13
        @test roundtrip ≈ x atol=1e-13
    end

    set_car_sources!(state, ir, controls)
    weight = collect(range(0.3, 1.4; length=length(x)))
    objective(v, transform, w) = ((j, q) = transform(v); j + dot(w, q))
    ad_gradient = DI.gradient(objective, DI.AutoEnzyme(), x, DI.Constant(ir),
                              DI.Constant(weight))
    @test ad_gradient ≈ finite_difference(v -> objective(v, ir, weight), x) atol=3e-8 rtol=3e-8

    # Source-invariant frames: the same innovations and innovation gradient under
    # any source controls give the same frame and the same candidate scores.
    rng = Xoshiro(0x0a5e)
    innovations = randn(rng, n_cells)
    h = randn(rng, n_cells)
    scales = exp.(BRM._adaptive_car_log_scale.(Ref(x), Ref(block), 1:n_cells))
    n = length(block.structured)
    frames = map((zeros_c, controls, ones_c)) do c
        set_car_sources!(state, ir, c)
        position, gradient = copy(x), zeros(length(x))
        for cell in 1:n_cells
            j = BRM._adaptive_car_phi_cell(block, cell)
            intercept = BRM._adaptive_car_cell(block, cell).intercept
            m = (intercept == 0 ? 0.0 : x[intercept]) +
                (j == 0 ? 0.0 : scales[j] * innovations[j])
            position[BRM._adaptive_car_index(block, cell)] =
                c[cell] * m + scales[cell]^c[cell] * innovations[cell]
        end
        # g = A(c)^(-T) h for the triangular A: theta rows first.
        for cell in n_cells:-1:1
            g = h[cell]
            if cell <= n
                for k in n + 1:n_cells
                    BRM._adaptive_car_phi_cell(block, k) == cell || continue
                    g -= c[k] * scales[cell] * gradient[BRM._adaptive_car_index(block, k)]
                end
            end
            gradient[BRM._adaptive_car_index(block, cell)] = g / scales[cell]^c[cell]
        end
        frame = CAR_AC_EXT._prepare_frame(state, ir, position, gradient)
        @test frame.innovation ≈ innovations atol=1e-12
        @test frame.invariant_gradient ≈ h atol=1e-12
        @test frame.scale ≈ scales atol=1e-14
        frame
    end
    for p in 1:n_cells, t in 0.0:0.1:1.0
        candidate = WarmupHMC.PartiallyCentered(t)
        scores = [CAR_AC_EXT._score_candidate(f, p, first(ir.pairs[p]), last(ir.pairs[p]),
                                              candidate) for f in frames]
        @test all(isapprox(s[2], scores[1][2]; atol=1e-12) for s in scores)
        @test all(isapprox(s[3], scores[1][3]; atol=1e-12) for s in scores)
    end
end

@testset "car c=1 is the spatial component and the area's total ($type)" for type in (:bym2, :icar)
    problem = CAR_AC_PROBLEM[type]
    sm = problem.model
    names = car_ac_names(type)
    cnames = BS.param_names(sm; include_tp=true, include_gq=false)
    effects_idx = indices_of(cnames, "car_mu_area_effects")
    sb = CAR_AC_SB[type]
    wrapped = adaptive_centering_problem(sb, problem, DI.AutoEnzyme())
    ir = WarmupHMC.reparametrizer(wrapped)
    state = ir.pairs[1][2].args[1].state
    block = only(state.blocks)

    # The wrapper starts in the compiled frame.
    x = 0.3 .* randn(Xoshiro(7), length(names))
    plain = LogDensityProblems.logdensity_and_gradient(problem, x)
    adaptive = LogDensityProblems.logdensity_and_gradient(wrapped, x)
    @test adaptive[1] ≈ plain[1] atol=1e-11
    @test adaptive[2] ≈ plain[2] atol=1e-11

    set_car_sources!(state, ir, ones(length(ir.pairs)))
    ljac, model_position = ir(x)
    constrained = BS.param_constrain(sm, model_position; include_tp=true, include_gq=false)
    sigma = constrained[index_of(cnames, "car_mu_area_sigma")]
    phi = constrained[indices_of(cnames, "car_mu_area_phi")]
    effects = constrained[effects_idx]
    @test sigma ≈ exp(x[block.sigma])
    if type === :bym2
        rho = constrained[index_of(cnames, "car_mu_area_rho")]
        @test rho ≈ inv_logit(x[block.rho])
        inv_scale = sb.data[:car_mu_area_inv_scale]
        # theta sources are every area's total, islands and row-less areas included
        intercept = constrained[index_of(cnames, "pop_mu_beta_pop.1")]
        @test x[block.unstructured] ≈ intercept .+ effects atol=1e-12
        @test x[block.structured] ≈
              [sigma * sqrt(rho * inv_scale[k]) for k in block.phi_area] .* phi atol=1e-12
    else
        @test x[block.structured] ≈ effects[block.phi_area] atol=1e-12
        @test effects[7] == 0.0
    end

    # Exact density at mixed controls, and its gradient.
    controls = collect(range(0.1, 0.9; length=length(ir.pairs)))
    set_car_sources!(state, ir, controls)
    ljac, model_position = ir(x)
    wrapped_lp, wrapped_gradient = LogDensityProblems.logdensity_and_gradient(wrapped, x)
    @test wrapped_lp ≈ ljac + LogDensityProblems.logdensity(problem, model_position) atol=1e-10
    @test wrapped_gradient ≈ finite_difference(
        v -> LogDensityProblems.logdensity(wrapped, v), x; step=1e-5) atol=3e-5 rtol=3e-5
    @test WarmupHMC.candidate_scoring_plan(wrapped) isa WarmupHMC.CandidateScoringPlan

    rows = adaptive_centering_controls(wrapped, sb)
    @test length(rows) == length(ir.pairs)
    @test all(r -> r.family === :car && r.binding === :mu, rows)
    @test all(r -> r.margin == (; predictor=:mu, coefficient=:car_mu_area), rows)
    @test [r.c for r in rows] == controls
    @test [r.index for r in rows] == first.(ir.pairs)
    @test [r.level for r in rows if r.term == 1] == ["a", "b", "c", "d", "h", "e", "f"]
    type === :bym2 && @test [r.level for r in rows if r.term == 2] == block.areas
end

# A car field beside a correlated random-effect block: the joint wrapper (pair
# order ordinary, then car) and the learned-order wrapper.
const CAR_AC_JOINT_SB = SBBRMI(@brm(CAR_AC_DF, begin
    mu ~ 1 + x + (1 + x | grp) + car(area; edges=(from, to), type=:bym2)
    y ~ Normal(mu, 1.0)
end); total_groups=(), mod=@__MODULE__)

@testset "car cells join the ordinary block's wrappers" begin
    problem = car_ac_instantiate(CAR_AC_JOINT_SB)
    names = BS.param_unc_names(problem.model)
    ordinary = BRM.adaptive_centering_blocks(CAR_AC_JOINT_SB, names)
    car = BRM._adaptive_car_centering_blocks(CAR_AC_JOINT_SB, names)
    @test length(ordinary) == 1 && length(car) == 1
    n_ranef = 2 * 2
    n_car = 15

    wrapped = adaptive_centering_problem(CAR_AC_JOINT_SB, problem, DI.AutoEnzyme())
    ir = WarmupHMC.reparametrizer(wrapped)
    state = ir.pairs[1][2].args[1].state
    @test state isa CAR_AC_EXT.BRMJointAdaptiveCenteringState
    @test first.(ir.pairs) == vcat(vec(only(ordinary).effects), car[1].structured,
                                   car[1].unstructured)

    # Each family maps exactly as its own wrapper does at the same controls.
    controls = collect(range(0.15, 0.85; length=n_ranef + n_car))
    set_car_sources!(state, ir, controls)
    x = 0.4 .* randn(Xoshiro(11), length(names))
    ljac, mapped = ir(x)
    car_state, car_ir = CAR_AC_EXT._adaptive_car_centering_reparametrizer(car)
    set_car_sources!(car_state, car_ir, controls[n_ranef + 1:end])
    car_ljac, car_mapped = car_ir(x)
    ranef_state, ranef_ir = CAR_AC_EXT._adaptive_centering_reparametrizer(ordinary)
    set_car_sources!(ranef_state, ranef_ir, controls[1:n_ranef])
    ranef_ljac, ranef_mapped = ranef_ir(x)
    @test ljac ≈ car_ljac + ranef_ljac atol=1e-12
    car_coords = vcat(car[1].structured, car[1].unstructured)
    @test mapped[car_coords] ≈ car_mapped[car_coords] atol=1e-13
    ranef_coords = vec(only(ordinary).effects)
    @test mapped[ranef_coords] ≈ ranef_mapped[ranef_coords] atol=1e-13
    rest = setdiff(eachindex(x), car_coords, ranef_coords)
    @test mapped[rest] == x[rest]
    _, roundtrip = WarmupHMC._inverse_with_logabsdet_jacobian(ir, mapped)
    @test roundtrip ≈ x atol=1e-12

    wrapped_lp, wrapped_gradient = LogDensityProblems.logdensity_and_gradient(wrapped, x)
    @test wrapped_lp ≈ ljac + LogDensityProblems.logdensity(problem, mapped) atol=1e-10
    @test wrapped_gradient ≈ finite_difference(
        v -> LogDensityProblems.logdensity(wrapped, v), x; step=1e-5) atol=3e-5 rtol=3e-5
    rows = adaptive_centering_controls(wrapped, CAR_AC_JOINT_SB)
    @test [r.family for r in rows] == vcat(fill(:ranef, n_ranef), fill(:car, n_car))

    # Learned order: the car cells keep their per-cell maps inside it.
    learned = adaptive_centering_problem(CAR_AC_JOINT_SB, problem, DI.AutoEnzyme();
                                         ranef_order=:learned)
    lir = WarmupHMC.reparametrizer(learned)
    @test lir isa CAR_AC_EXT.BRMOrderedCenteringReparametrization
    @test length(WarmupHMC.reparam_controls(lir)) == n_ranef + n_car
    WarmupHMC.restore_reparam_controls!(lir, controls)
    learned_ljac, learned_mapped = lir(x)
    @test learned_mapped[car_coords] ≈ car_mapped[car_coords] atol=1e-13
    learned_ranef = BRM.AdaptiveCenteringBlock[only(ordinary)]
    _, ranef_only = CAR_AC_EXT._adaptive_ordered_centering_reparametrizer(
        learned_ranef, BRM._HSGPAdaptiveCenteringBlock[])
    WarmupHMC.restore_reparam_controls!(ranef_only, controls[1:n_ranef])
    ranef_ljac_learned, ranef_mapped_learned = ranef_only(x)
    @test learned_ljac ≈ car_ljac + ranef_ljac_learned atol=1e-12
    @test learned_mapped[ranef_coords] ≈ ranef_mapped_learned[ranef_coords] atol=1e-13
    inverse_ljac, learned_roundtrip = WarmupHMC._inverse_with_logabsdet_jacobian(
        lir, learned_mapped)
    @test inverse_ljac ≈ -learned_ljac atol=1e-12
    @test learned_roundtrip ≈ x atol=1e-12
    learned_lp, learned_gradient = LogDensityProblems.logdensity_and_gradient(learned, x)
    @test learned_lp ≈ learned_ljac +
        LogDensityProblems.logdensity(problem, learned_mapped) atol=1e-10
    @test learned_gradient ≈ finite_difference(
        v -> LogDensityProblems.logdensity(learned, v), x; step=1e-5) atol=3e-5 rtol=3e-5
    @test [r.family for r in adaptive_centering_controls(learned, CAR_AC_JOINT_SB)] ==
          vcat(fill(:ranef, n_ranef), fill(:car, n_car))
end

@testset "car fields refuse cdar plans; generated fields are skipped" begin
    @test_throws "cdar walk cells with random-effect, HSGP or car cells" begin
        cdar_df = merge(CAR_AC_DF, (; week=repeat(1:3; outer=3),
            patch=repeat(["p", "q", "r"]; inner=3), C=[1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]))
        mixed = SBBRMI(@brm(cdar_df, begin
            mu ~ 1 + car(area; edges=(from, to), type=:icar) + cdar(week; by=patch, cor=C)
            y ~ Normal(mu, 1.0)
        end); mod=@__MODULE__)
        adaptive_centering_problem(mixed, car_ac_instantiate(mixed), DI.AutoEnzyme())
    end
    # A car field that only an omitted response reads is emitted in generated
    # quantities: it has no cell, and a model left without cells names it.
    omitted = SBBRMI(@brm(begin
        mu ~ 1 + x + car(area; edges=(from, to), type=:bym2)
        y ~ Normal(mu, 1.0)
    end)(Base.structdiff(CAR_AC_DF, (; y=nothing))); mod=@__MODULE__)
    problem = car_ac_instantiate(omitted)
    names = BS.param_unc_names(problem.model)
    @test !any(startswith("car_mu_area_phi"), names)
    @test isempty(BRM._adaptive_car_centering_blocks(omitted, names))
    @test_throws "car_mu_area" adaptive_centering_problem(omitted, problem, DI.AutoEnzyme())
end

@testset "car online warmup on a data-rich field" begin
    # A 6 x 6 lattice of areas, 40 rows each, sd(field) = 0.6 against noise 0.5.
    side = 6
    label(i, j) = "r$(i)c$(j)"
    from = String[]; to = String[]
    for i in 1:side, j in 1:side
        i < side && (push!(from, label(i, j)); push!(to, label(i + 1, j)))
        j < side && (push!(from, label(i, j)); push!(to, label(i, j + 1)))
    end
    rng = Xoshiro(0x20261023)
    areas = [label(i, j) for i in 1:side for j in 1:side]
    effect = Dict(a => 0.6 * randn(rng) for a in areas)
    area = repeat(areas; inner=40)
    y = [1.0 + effect[a] + 0.5 * randn(rng) for a in area]
    df = (; area, y, from, to)
    sb = SBBRMI(@brm(df, begin
        sigma ~ Exponential(1)
        mu ~ 1 + car(area; edges=(from, to), type=:bym2)
        y ~ Normal(mu, sigma)
    end); mod=@__MODULE__)
    problem = car_ac_instantiate(sb)
    online = adaptive_centering_problem(sb, problem, DI.AutoEnzyme())
    result = WarmupHMC.adaptive_warmup_mcmc(
        Xoshiro(0x20261023), online;
        n_draws=100,
        n_evaluations=600,
        progress=nothing,
        monitor_ess=false,
    )
    adapted = [value.c for (_, value) in WarmupHMC.reparam_sources(online)]
    @test length(adapted) == 36 + 36
    @test all(c -> c in 0.0:0.1:1.0, adapted)
    @test size(result.posterior_position, 2) >= 100
    @test all(isfinite, result.posterior_position)
    # 40 rows per area pin each area's total effect: its theta cell centres.
    @test sum(adapted[37:end]) / 36 > 0.5
end

println("adaptive_car_centering.jl: all testsets passed")
