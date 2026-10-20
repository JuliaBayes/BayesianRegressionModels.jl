# HSGP options of decision 1fk9b29: a length-scale prior that keeps the
# validity floor (`length_scale_floor`), replay that refuses or clamps values
# outside the fitted basis (`extrapolate=`), `anchor=`, keyword-qualified term
# addresses, the `rows=` mask and empty axes with an explicit domain. The
# ReactiveKernels side is `test/rk_hsgp_options.jl`.
using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Exponential, LogNormal, Normal, Uniform, truncated
import BridgeStan
const BRM = BayesianRegressionModels

const HSGP_OPTIONS_RUNTIME = get(ENV, "BRM_HSGP_OPTIONS_RUNTIME", "1") != "0"
const HSGP_OPTIONS_CACHE = joinpath(tempdir(), "brm-hsgp-options")

options_df() = (; t=collect(range(0.0, 10.0; length=12)),
                  y=sin.(collect(range(0.0, 10.0; length=12)) ./ 3))
options_data(sb) = StanBlocks.stan_data(sb.model)
options_model(builder, df) = SBBRMI(builder(df); mod=@__MODULE__)
stan_lines(sb, pattern) = filter(l -> occursin(pattern, l), split(BRM.stan_code(sb), '\n'))

plain_builder = @brm begin
    mu ~ 1 + hsgp(t; k=5)
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end

@testset "length_scale_floor keeps the validity floor in an explicit prior" begin
    df = options_df()
    floored = options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=5)
        length_scale(:, hsgp(t)) ~ Uniform(length_scale_floor, 24.0)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), df)
    @test any(occursin("real<lower=rho_lower_hsgp_t, upper=24.0> hsgp_t_rho_iso", l)
              for l in stan_lines(floored, "rho_iso"))
    @test any(occursin("uniform(rho_lower_hsgp_t, 24.0)", l)
              for l in stan_lines(floored, "rho_iso"))
    # The floor is the same data value the default declaration bounds by.
    plain = options_model(plain_builder, df)
    @test options_data(floored)[:rho_lower_hsgp_t] == options_data(plain)[:rho_lower_hsgp_t]
    # A fresh reprocess re-derives it with the new basis; the source is unchanged.
    wider = (; t=collect(range(0.0, 20.0; length=12)), y=df.y)
    fresh = reprocess(floored, wider; freeze_constants=false)
    @test fresh.data[:rho_lower_hsgp_t] ≈ 2 * options_data(floored)[:rho_lower_hsgp_t]
    @test BRM.stan_code(fresh) == BRM.stan_code(floored)

    truncated_floor = options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=5)
        length_scale(:, hsgp(t)) ~ truncated(LogNormal(0, 1); lower=length_scale_floor, upper=24.0)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), df)
    @test any(l -> occursin("rho_lower_hsgp_t", l) && occursin("24.0", l),
              stan_lines(truncated_floor, "hsgp_t_rho_iso"))

    periodic = options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=3, cov=:periodic, period=24.0)
        length_scale(:, hsgp(t)) ~ Uniform(length_scale_floor, 5.0)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), df)
    @test any(occursin("uniform(rho_lower_hsgp_t, 5.0)", l)
              for l in stan_lines(periodic, "rho_iso"))

    # The `u` axis's fitted floor is about 94, so its upper bound sits above it.
    anisotropic = options_model(@brm(begin
        mu ~ 1 + hsgp(t, u; k=3, iso=false)
        length_scale(:, hsgp(t, u)) ~ Uniform(length_scale_floor, 200.0)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), merge(df, (; u=df.t .^ 2)))
    @test any(occursin("uniform(rho_lower_hsgp_t_u, 200.0)", l)
              for l in stan_lines(anisotropic, "hsgp_t_u_rho"))

    grouped = options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=4, by=g)
        length_scale(:, hsgp(t)) ~ Uniform(length_scale_floor, 20.0)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), merge(df, (; g=repeat(["a", "b"], 6))))
    @test occursin("uniform(rho_lower_hsgp_t_by_g, 20.0)", BRM.stan_code(grouped))

    # refused: the marker names a length scale's floor (decision 1fk9b29)
    @test_throws "only valid in its own" options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=5)
        sd(:, hsgp(t)) ~ Uniform(length_scale_floor, 2.0)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), df)

    if HSGP_OPTIONS_RUNTIME
        mkpath(HSGP_OPTIONS_CACHE)
        problem = BRM.stan_instantiate(floored;
            path=joinpath(HSGP_OPTIONS_CACHE, "floor.stan"))
        names = BridgeStan.param_unc_names(problem.model)
        index = only(findall(==("hsgp_t_rho_iso"), names))
        floor = options_data(floored)[:rho_lower_hsgp_t]
        for (u, expected) in ((-30.0, floor), (30.0, 24.0))
            point = zeros(length(names)); point[index] = u
            constrained = BridgeStan.param_constrain(problem.model, point)
            @test constrained[index] ≈ expected rtol=1e-8
        end
        @test isfinite(BridgeStan.log_density(problem.model, zeros(length(names))))
    end
end

# `t` spans [0, 10], so a k=5 basis has L = 1.5 * 5 = 7.5 and this floor.
const OPTIONS_FLOOR = (4 * 7.5 / π) * sqrt(log(100) / (5^2 - 1))

@testset "brm_length_scale_floor reads a term's fitted floor" begin
    df = options_df()
    plain = options_model(plain_builder, df)
    floor = brm_length_scale_floor(brm_descriptor(plain), :mu; term=:hsgp_t)
    @test floor isa Float64
    @test floor == options_data(plain)[:rho_lower_hsgp_t]
    @test floor ≈ OPTIONS_FLOOR
    # The same value under a floor-marked prior and a floor-replacing one.
    for prior in (:(Uniform(length_scale_floor, 24.0)), :(LogNormal(0, 1)))
        builder = eval(:(@brm begin
            mu ~ 1 + hsgp(t; k=5)
            length_scale(:, hsgp(t)) ~ $prior
            sigma ~ Exponential(1)
            y ~ Normal(mu, sigma)
        end))
        d = brm_descriptor(options_model(builder, df))
        @test brm_length_scale_floor(d, :mu; term=:hsgp_t) == floor
    end
    # A fresh reprocess re-derives it with the new basis.
    wider = (; t=collect(range(0.0, 20.0; length=12)), y=df.y)
    fresh = brm_descriptor(reprocess(plain, wider; freeze_constants=false))
    @test brm_length_scale_floor(fresh, :mu; term=:hsgp_t) ≈ 2 * floor

    anisotropic = options_model(@brm(begin
        mu ~ 1 + hsgp(t, u; k=3, iso=false)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), merge(df, (; u=df.t .^ 2)))
    floors = brm_length_scale_floor(brm_descriptor(anisotropic), :mu; term=:hsgp_t_u)
    @test floors isa Vector{Float64}
    @test floors == options_data(anisotropic)[:rho_lower_hsgp_t_u]
    @test length(floors) == 2

    periodic = options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=3, cov=:periodic, period=24.0)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), df)
    @test brm_length_scale_floor(brm_descriptor(periodic), :mu; term=:hsgp_t) ==
          BRM._brm_hsgp_periodic_rho_lower(3)

    with_g = merge(df, (; g=repeat(["a", "b"], 6)))
    grouped = options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=4, by=g)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), with_g)
    @test brm_length_scale_floor(brm_descriptor(grouped), :mu; term=:hsgp_t_by_g) ==
          options_data(grouped)[:rho_lower_hsgp_t_by_g]
    hyper = options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=4, by=g)
        log(length_scale(hsgp(t))) ~ 1 + (1 | g)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), with_g)
    @test brm_length_scale_floor(brm_descriptor(hyper), :mu; term=:hsgp_t_by_g) ==
          options_data(hyper)[:rho_lower_hsgp_t_by_g]

    # A one-function basis has no usable floor.
    single = options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=1)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), df)
    @test brm_length_scale_floor(brm_descriptor(single), :mu; term=:hsgp_t) == 0.0

    # A model-derived axis carries its floor as a literal of the fixed domain.
    latent = options_model(@brm(begin
        x ~ 1 + t
        mu ~ 1 + hsgp(x; k=6, domain=(0.02, 4.0))
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), df)
    @test brm_length_scale_floor(brm_descriptor(latent), :mu; term=:hsgp_x) ≈
          (4 * 1.99 / π) * sqrt(log(100) / (6^2 - 1))

    @test_throws "available term labels" brm_length_scale_floor(
        brm_descriptor(plain), :mu; term=:hsgp_u)
    mixed = options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=5) + gp(t)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), df)
    @test_throws "only an `hsgp` term" brm_length_scale_floor(
        brm_descriptor(mixed), :mu; term=Symbol("gp(t)"))
end

@testset "an upper bound at or below the fitted floor is refused" begin
    df = options_df()
    # refused: the declared length-scale support is empty, so the program
    # could never initialize (decision 0ddmtf8)
    message = "the fitted length-scale floor 4.183 (k=5, c=1.5, L=7.5) is not " *
              "below the upper bound 3.0"
    @test_throws message options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=5)
        length_scale(:, hsgp(t)) ~ Uniform(length_scale_floor, 3.0)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), df)
    @test_throws message options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=5)
        length_scale(:, hsgp(t)) ~ truncated(LogNormal(0, 1); lower=length_scale_floor, upper=3.0)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), df)
    @test_throws "length_scale(:, hsgp(t)) ~ Uniform(length_scale_floor, 3.0)" options_model(
        @brm(begin
            mu ~ 1 + hsgp(t; k=5)
            length_scale(:, hsgp(t)) ~ Uniform(length_scale_floor, 3.0)
            sigma ~ Exponential(1)
            y ~ Normal(mu, sigma)
        end), df)
    # A floor exactly equal to the bound leaves no interior either.
    exact = options_data(options_model(plain_builder, df))[:rho_lower_hsgp_t]
    @test_throws "is not below the upper bound" options_model(eval(:(@brm begin
        mu ~ 1 + hsgp(t; k=5)
        length_scale(:, hsgp(t)) ~ Uniform(length_scale_floor, $exact)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end)), df)
    # Each anisotropic axis is checked: `u`'s floor is about 94.
    @test_throws "floor of axis 2" options_model(@brm(begin
        mu ~ 1 + hsgp(t, u; k=3, iso=false)
        length_scale(:, hsgp(t, u)) ~ Uniform(length_scale_floor, 30.0)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), merge(df, (; u=df.t .^ 2)))
    @test_throws "(k=4, c=1.5, L=7.5)" options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=4, by=g)
        length_scale(:, hsgp(t)) ~ Uniform(length_scale_floor, 2.0)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), merge(df, (; g=repeat(["a", "b"], 6))))
    @test_throws "(k=3, period=24.0)" options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=3, cov=:periodic, period=24.0)
        length_scale(:, hsgp(t)) ~ Uniform(length_scale_floor, 1.0)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), df)
    @test_throws "(k=6, domain half-width L=" options_model(@brm(begin
        x ~ 1 + t
        mu ~ 1 + hsgp(x; k=6, domain=(0.02, 4.0))
        length_scale(:, hsgp(x)) ~ Uniform(length_scale_floor, 0.5)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), df)

    # A fresh reprocess re-derives the floor and checks it again; the frozen
    # replay keeps the fitted floor.
    bounded = options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=5)
        length_scale(:, hsgp(t)) ~ Uniform(length_scale_floor, 6.0)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), df)
    wider = (; t=collect(range(0.0, 20.0; length=12)), y=df.y)
    @test_throws "is not below the upper bound 6.0" reprocess(
        bounded, wider; freeze_constants=false)
    @test reprocess(bounded, (; t=[1.0, 9.0], y=zeros(2))).data[:rho_lower_hsgp_t] ==
          options_data(bounded)[:rho_lower_hsgp_t]

    # Without the marker an explicit prior below the floor is the user's choice.
    @test options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=5)
        length_scale(:, hsgp(t)) ~ Uniform(1.0, 3.0)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), df) isa SBBRMI
    # Literal bounds name the authored address, not the prior twice.
    @test_throws "`length_scale(:, hsgp(t)) ~ Uniform(5.0, 3.0)` needs" options_model(
        @brm(begin
            mu ~ 1 + hsgp(t; k=5)
            length_scale(:, hsgp(t)) ~ Uniform(5.0, 3.0)
            sigma ~ Exponential(1)
            y ~ Normal(mu, sigma)
        end), df)
end

@testset "replay outside the fitted basis interval" begin
    df = options_df()
    plain = options_model(plain_builder, df)
    # Training t is [0, 10]: the basis interval is [-2.5, 12.5].
    @test plain.preproc[:PHI_hsgp_t].const_.fits == ((5.0, 7.5),)
    inside = reprocess(plain, (; t=[-2.0, 11.0], y=zeros(2)))
    @test size(inside.data[:PHI_hsgp_t]) == (2, 5)
    # Past the interval the sines wrap around; replay refuses.
    @test_throws "outside the fitted basis interval" reprocess(
        plain, (; t=[-20.0, 40.0], y=zeros(2)))

    clamp_builder = @brm begin
        mu ~ 1 + hsgp(t; k=5, extrapolate=:clamp)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    clamped = options_model(clamp_builder, df)
    @test BRM.stan_code(clamped) == BRM.stan_code(plain)
    @test options_data(clamped)[:PHI_hsgp_t] == options_data(plain)[:PHI_hsgp_t]
    grid = collect(range(-20.0, 40.0; length=7))
    replay = reprocess(clamped, (; t=grid, y=zeros(7)))
    expected = reprocess(plain, (; t=clamp.(grid, 0.0, 10.0), y=zeros(7)))
    @test replay.data[:PHI_hsgp_t] ≈ expected.data[:PHI_hsgp_t]
    # Rows before and after the fitted range hold the curve at its end values.
    @test replay.data[:PHI_hsgp_t][1, :] ≈ options_data(plain)[:PHI_hsgp_t][1, :]
    @test replay.data[:PHI_hsgp_t][end, :] ≈ options_data(plain)[:PHI_hsgp_t][end, :]
    # The same record drives builder replay through a generative plan.
    plan = generative_plan(clamp_builder, df; mod=@__MODULE__)
    @test reprocess(plan, (; t=grid, y=zeros(7))).data[:PHI_hsgp_t] ≈
          expected.data[:PHI_hsgp_t]

    # refused: a periodic basis has no interval to leave (decision 1fk9b29)
    @test_throws "periodic" options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=3, cov=:periodic, period=24.0, extrapolate=:clamp)
        y ~ Normal(mu, 1)
    end), df)
    @test_throws "supports `:error` and `:clamp`" options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=3, extrapolate=:linear)
        y ~ Normal(mu, 1)
    end), df)
end

@testset "anchor subtracts the basis row at the anchor" begin
    df = options_df()
    plain = options_model(plain_builder, df)
    anchored = options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=5, anchor=0.0)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), df)
    PHI = options_data(plain)[:PHI_hsgp_t]
    # t[1] == 0: the anchored basis is the plain one minus its first row.
    @test options_data(anchored)[:PHI_hsgp_t] ≈ PHI .- PHI[1:1, :]
    @test all(iszero, options_data(anchored)[:PHI_hsgp_t][1, :])
    @test BRM.stan_code(anchored) == BRM.stan_code(plain)
    # An anchor outside the training values widens the interval to contain it.
    outside = options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=5, anchor=-4.0)
        y ~ Normal(mu, 1)
    end), df)
    @test outside.preproc[:PHI_hsgp_t].const_.fits == ((5.0, 1.5 * 9.0),)
    # Replay keeps the anchor: the value there stays zero.
    replay = reprocess(anchored, (; t=[0.0, 3.0], y=zeros(2)))
    @test all(iszero, replay.data[:PHI_hsgp_t][1, :])
    # Tensor and periodic bases anchor the same way.
    tensor = options_model(@brm(begin
        mu ~ 1 + hsgp(t, u; k=(3, 2), anchor=(0.0, 0.0))
        y ~ Normal(mu, 1)
    end), merge(df, (; u=df.t .^ 2)))
    @test all(iszero, options_data(tensor)[:PHI_hsgp_t_u][1, :])
    periodic = options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=3, cov=:periodic, period=24.0, anchor=0.0)
        y ~ Normal(mu, 1)
    end), df)
    @test all(iszero, options_data(periodic)[:PHI_hsgp_t][1, :])
    grouped = options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=4, by=g, anchor=0.0)
        y ~ Normal(mu, 1)
    end), merge(df, (; g=repeat(["a", "b"], 6))))
    @test all(iszero, options_data(grouped)[:PHI_hsgp_t_by_g][1, :])

    # refused: a fixed domain is not widened, and past it the basis wraps
    # around (decision 1fk9b29)
    @test_throws "outside its fixed domain" options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=4, domain=(-1.0, 11.0), anchor=-4.0)
        y ~ Normal(mu, 1)
    end), df)
    # refused: the projection centers every basis column, so the term would
    # no longer vanish at the anchor (decision 1fk9b29)
    @test_throws "orthogonal_to=:linear" options_model(@brm(begin
        mu ~ 1 + t + hsgp(t; k=4, anchor=0.0, orthogonal_to=:linear)
        y ~ Normal(mu, 1)
    end), df)
end

@testset "rows= fits the basis from the masked rows" begin
    df = options_df()
    carried = df.t .<= 6.0
    filled = ifelse.(carried, df.t, 123.0)
    masked = options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=5, rows=carried)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), (; t=filled, y=df.y, carried))
    reference = options_model(plain_builder, (; t=df.t[carried], y=df.y[carried]))
    PHI = options_data(masked)[:PHI_hsgp_t]
    @test PHI[carried, :] ≈ options_data(reference)[:PHI_hsgp_t]
    @test all(iszero, PHI[.!carried, :])
    @test options_data(masked)[:rho_lower_hsgp_t] ≈ options_data(reference)[:rho_lower_hsgp_t]
    # The mask is Julia-side preprocessing, not Stan data.
    @test !haskey(options_data(masked), :carried)
    # Replay rebinds the mask; masked-out rows are never evaluated or checked.
    replay = reprocess(masked, (; t=[1.0, 500.0], y=zeros(2), carried=[true, false]))
    @test all(iszero, replay.data[:PHI_hsgp_t][2, :])
    @test_throws "outside the fitted basis interval" reprocess(
        masked, (; t=[500.0, 1.0], y=zeros(2), carried=[true, false]))
    # Rows outside the mask may be missing; they contribute 0 all the same.
    gaps = options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=5, rows=carried)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), (; t=ifelse.(carried, df.t, missing), y=df.y, carried))
    @test options_data(gaps)[:PHI_hsgp_t] ≈ PHI
    @test BRM.stan_code(gaps) == BRM.stan_code(masked)
    @test all(iszero, reprocess(gaps, (; t=[1.0, missing], y=zeros(2),
                                       carried=[true, false])).data[:PHI_hsgp_t][2, :])
    @test_throws "marks as carrying the predictor" options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=5, rows=carried)
        y ~ Normal(mu, 1)
    end), (; t=ifelse.(carried, missing, df.t), y=df.y, carried))
    # A 0/1 numeric mask is the same mask.
    numeric = options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=5, rows=carried)
        y ~ Normal(mu, 1)
    end), (; t=filled, y=df.y, carried=Float64.(carried)))
    @test options_data(numeric)[:PHI_hsgp_t] ≈ PHI
    @test_throws "Bool (or 0/1)" options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=5, rows=carried)
        y ~ Normal(mu, 1)
    end), (; t=filled, y=df.y, carried=fill(0.5, 12)))
end

@testset "an empty axis with an explicit domain keeps a prior-only term" begin
    df = merge(options_df(), (; p=Float64[]))
    empty_domain = options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=5)
        drift ~ 0 + hsgp(p; k=4, domain=(-1.5, 1.5))
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), df)
    @test size(options_data(empty_domain)[:PHI_hsgp_p]) == (0, 4)
    @test !isempty(stan_lines(empty_domain, "hsgp_p_rho_iso"))
    empty_periodic = options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=5)
        drift ~ 0 + hsgp(p; k=3, cov=:periodic, period=24.0)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), df)
    @test size(options_data(empty_periodic)[:PHI_hsgp_p]) == (0, 6)
    # Replay onto rows again.
    replay = reprocess(empty_domain, merge(df, (; p=[0.0, 1.0])))
    @test size(replay.data[:PHI_hsgp_p]) == (2, 4)
    # A data-derived interval has nothing to fit on an empty axis.
    @test_throws "explicit `domain=" options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=5)
        drift ~ 0 + hsgp(p; k=4)
        y ~ Normal(mu, 1)
    end), df)
    if HSGP_OPTIONS_RUNTIME
        mkpath(HSGP_OPTIONS_CACHE)
        problem = BRM.stan_instantiate(empty_domain;
            path=joinpath(HSGP_OPTIONS_CACHE, "empty.stan"))
        # Nothing reads the empty term here, so StanBlocks draws its
        # prior-only parameters in generated quantities.
        names = BridgeStan.param_unc_names(problem.model)
        @test "hsgp_p_rho_iso" in BridgeStan.param_names(problem.model; include_gq=true)
        @test isfinite(BridgeStan.log_density(problem.model, zeros(length(names))))
    end
end

@testset "keyword-qualified addresses pick one of several same-axis terms" begin
    df = options_df()
    both = options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=5) + hsgp(t; k=3, cov=:periodic, period=24.0)
        length_scale(:, hsgp(t; cov=:exp_quad)) ~ Uniform(length_scale_floor, 10.0)
        sd(:, hsgp(t; cov=:periodic)) ~ Normal(0, 0.5)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end), df)
    code = BRM.stan_code(both)
    @test occursin("real<lower=rho_lower_hsgp_t, upper=10.0> hsgp_t_rho_iso", code)
    @test occursin("real<lower=rho_lower_hsgp_mu_t> hsgp_mu_t_rho_iso", code)
    @test occursin("hsgp_mu_t_sigma ~ normal(0.0, 0.5)", code)
    @test occursin("hsgp_t_sigma ~ lognormal(0.0, 1.0)", code)
    qualified = @brm df begin
        mu ~ 1 + hsgp(t; k=5) + hsgp(t; k=3, cov=:periodic, period=24.0)
        sd(:, hsgp(t; cov=:periodic)) ~ Normal(0, 0.5)
        y ~ Normal(mu, 1)
    end
    spec = only(term_priors(qualified))
    @test spec.term === Symbol("hsgp(t)")
    @test spec.qualifiers == (; cov=:periodic)
    @test_throws "is ambiguous" options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=5) + hsgp(t; k=3, cov=:periodic, period=24.0)
        length_scale(:, hsgp(t)) ~ Uniform(0.5, 10.0)
        y ~ Normal(mu, 1)
    end), df)
    @test_throws "matches none" options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=5) + hsgp(t; k=3, cov=:periodic, period=24.0)
        length_scale(:, hsgp(t; k=7)) ~ Uniform(0.5, 10.0)
        y ~ Normal(mu, 1)
    end), df)
    # A lone term must spell the qualifier too.
    lone = options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=5)
        length_scale(:, hsgp(t; k=5)) ~ Uniform(0.5, 10.0)
        y ~ Normal(mu, 1)
    end), df)
    @test occursin("uniform(0.5, 10.0)", BRM.stan_code(lone))
    @test_throws "matches none" options_model(@brm(begin
        mu ~ 1 + hsgp(t; k=5)
        length_scale(:, hsgp(t; k=20)) ~ Uniform(0.5, 10.0)
        y ~ Normal(mu, 1)
    end), df)
end
