using Test
using BayesianRegressionModels
using StanBlocks
using Distributions
using LogDensityProblems

const GP_TERM_N = 8
const GP_RUN_BRIDGESTAN = get(ENV, "BRM_GP_RUNTIME", "1") != "0"
const GP_TERM_CACHE = joinpath(tempdir(), "brm-gp-hsgp")

function gp_term_df(; shift_x=0.0, shift_x2=0.0)
    x = collect(range(-1.0, 1.0; length=GP_TERM_N)) .+ shift_x
    x2 = (x .- shift_x) .^ 2 .+ 0.1 .* (x .- shift_x) .+ shift_x2
    y = sin.(x)
    g = repeat(["a", "b"], inner=GP_TERM_N ÷ 2)
    (; x, x2, y, g)
end

function latent_hsgp_df(; shift=0.0)
    t = collect(range(-1.0, 1.0; length=GP_TERM_N)) .+ shift
    latent_conc = exp.(-1.4 .+ 0.8 .* t)
    lloq = fill(0.3, GP_TERM_N)
    c_obs = max.(latent_conc, lloq)
    zbl = Float64.(latent_conc .<= lloq)
    nominal_time = repeat([1, 2], outer=GP_TERM_N ÷ 2)
    subject = repeat(1:(GP_TERM_N ÷ 2), inner=2)
    y = 0.2 .* latent_conc
    (; t, nominal_time, subject, zbl, c_obs, lloq, y)
end

latent_hsgp_model(df) = @brm df begin
    log(x) ~ 1 + factor(nominal_time) + (1 | assay | subject)
    assay_sd ~ Exponential(1)
    c_obs ~ censored(LogNormal(log(x), assay_sd); lower=lloq)
    mu ~ 1 + factor(nominal_time) + zbl + x +
         hsgp(x; k=5, domain=(0.01, 5.0), orthogonal_to=:linear) +
         (1 + x | qt | subject)
    length_scale(:, hsgp(x)) ~ Uniform(1.5, 4.0)
    sd(:, hsgp(x)) ~ Normal(0, 0.5)
    sigma ~ Exponential(1)
    y ~ Normal(mu, sigma)
end

@testset "HSGP over a model-derived axis" begin
    df = latent_hsgp_df()
    brmi = latent_hsgp_model(df)
    @test :x in popcoefnames(brmi, :mu)
    @test :zbl in popcoefnames(brmi, :mu)
    @test ranefcoefnames(brmi, :qt) == [
        (predictor=:mu, coefficient=:Intercept),
        (predictor=:mu, coefficient=:x),
    ]

    sb = SBBRMI(brmi; mod=@__MODULE__)
    code = StanBlocks.stan_code(sb.model)
    @test !haskey(sb.data, :PHI_hsgp_x)
    @test size(sb.data[:omega2_hsgp_x]) == (5, 1)
    @test sb.preproc[:omega2_hsgp_x].kind === :static
    @test occursin("brm_hsgp_basis_1d", code)
    @test occursin("brm_hsgp_orthogonalize_linear", code)
    @test occursin(r"vector\[[^]]+\] hsgp_x =", code)
    @test occursin("x = exp(log_x);", code)
    @test occursin("hsgp_x_rho_iso ~ uniform(1.5, 4.0);", code)
    @test occursin("hsgp_x_sigma ~ normal(0.0, 0.5);", code)
    @test StanBlocks.stan.transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok

    @testset "public fixed-grid population curve matches the fitted summand" begin
        if GP_RUN_BRIDGESTAN
            isdir(GP_TERM_CACHE) || mkpath(GP_TERM_CACHE)
            problem = StanBlocks.stan_instantiate(
                sb.model;
                path=joinpath(GP_TERM_CACHE,
                              "latent-" * string(hash(code)) * ".stan"))
            dimension = LogDensityProblems.dimension(problem)
            q = [0.04 * ((i % 7) - 3) for i in 1:dimension]
            names = StanBlocks.BridgeStan.param_names(
                problem.model; include_tp=true, include_gq=false)
            constrained = StanBlocks.BridgeStan.param_constrain(
                problem.model, q; include_tp=true, include_gq=false)
            descriptor = brm_descriptor(sb)
            x_coordinates = brm_output_coordinates(
                descriptor, :x, names; role=:linear_predictor)
            grid = constrained[x_coordinates]
            curve = hsgp_population_curve(
                descriptor, reshape(constrained, 1, :), names, grid;
                predictor=:mu, coefficient=:x, term=:hsgp_x)

            hsgp_coordinates = brm_output_coordinates(
                descriptor, :hsgp_x, names)
            beta_coordinate = only(brm_population_effect_coordinates(
                descriptor, :mu, names; coefficient=:x).coordinates)
            @test vec(curve.hsgp) ≈ constrained[hsgp_coordinates] atol=1e-10
            @test maximum(abs, curve.hsgp) > 0
            @test vec(curve.linear) ≈ constrained[beta_coordinate] .* grid atol=1e-12
            @test curve.total ≈ curve.linear + curve.hsgp atol=1e-12
            @test collect(curve.domain) ≈ [0.01, 5.0]
            @test_throws "outside its fitted HSGP domain" hsgp_population_curve(
                descriptor, reshape(constrained, 1, :), names, [0.0];
                predictor=:mu, coefficient=:x, term=:hsgp_x)
        else
            @info "Skipping BridgeStan latent-HSGP curve gate (BRM_GP_RUNTIME=0)"
        end
    end

    replay = reprocess(sb, latent_hsgp_df(; shift=0.2))
    @test replay.data[:omega2_hsgp_x] == sb.data[:omega2_hsgp_x]
    @test StanBlocks.stan_code(replay.model) == code

    missing_domain = @brm df begin
        log_x ~ 1 + t
        x = exp(log_x)
        mu ~ 1 + x + hsgp(x; k=5, orthogonal_to=:linear)
        y ~ Normal(mu, 1)
    end
    @test_throws "requires an explicit fixed `domain" SBBRMI(
        missing_domain; mod=@__MODULE__)

    domain_and_c = @brm df begin
        log_x ~ 1 + t
        x = exp(log_x)
        mu ~ 1 + hsgp(x; k=5, c=1.5, domain=(0.01, 5.0))
        y ~ Normal(mu, 1)
    end
    @test_throws "cannot also specify" SBBRMI(domain_and_c; mod=@__MODULE__)

    invalid_domain = @brm df begin
        log_x ~ 1 + t
        x = exp(log_x)
        mu ~ 1 + hsgp(x; k=5, domain=(5.0, 0.01))
        y ~ Normal(mu, 1)
    end
    @test_throws "needs lower < upper" SBBRMI(invalid_domain; mod=@__MODULE__)
end

@testset "orthogonal raw HSGP uses the same public contract" begin
    df = gp_term_df()
    brmi = @brm df begin
        loc ~ 1 + x + hsgp(x; k=4, domain=(-2.0, 2.0),
                           orthogonal_to=:linear)
        y ~ Normal(loc, 1)
    end
    sb = SBBRMI(brmi; mod=@__MODULE__)
    PHI = sb.data[:PHI_hsgp_x]
    xc = df.x .- sum(df.x) / length(df.x)
    @test maximum(abs, vec(sum(PHI; dims=1))) < 1e-12
    @test maximum(abs, vec(transpose(xc) * PHI)) < 1e-12

    replay_df = gp_term_df(; shift_x=0.25)
    replay = reprocess(sb, replay_df; freeze_constants=false)
    replay_xc = replay_df.x .- sum(replay_df.x) / length(replay_df.x)
    replay_PHI = replay.data[:PHI_hsgp_x]
    @test maximum(abs, vec(sum(replay_PHI; dims=1))) < 1e-12
    @test maximum(abs, vec(transpose(replay_xc) * replay_PHI)) < 1e-12
    @test replay.preproc[:PHI_hsgp_x].const_.fits ==
          sb.preproc[:PHI_hsgp_x].const_.fits

    @test_throws "outside its fixed domain" reprocess(
        sb, gp_term_df(; shift_x=3.0))

    @test_throws "uses a raw-data axis" hsgp_population_curve(
        brm_descriptor(sb), zeros(1, 0), String[], [0.0];
        predictor=:loc, coefficient=:x, term=:hsgp_x)

    grouped_orthogonal = @brm df begin
        loc ~ 1 + x + hsgp(x; k=4, domain=(-2.0, 2.0),
                           orthogonal_to=:linear, by=g)
        y ~ Normal(loc, 1)
    end
    @test_throws "cannot be combined with `by=`" SBBRMI(
        grouped_orthogonal; mod=@__MODULE__)

    outside = merge(df, (; x=2 .* df.x))
    bad = @brm outside begin
        loc ~ 1 + x + hsgp(x; k=4, domain=(-1.5, 1.5),
                           orthogonal_to=:linear)
        y ~ Normal(loc, 1)
    end
    @test_throws "outside its fixed domain" SBBRMI(bad; mod=@__MODULE__)
end

exact_gp_model(df) = @brm df begin
    loc ~ 1 + gp(x, x2; iso=false)
    y ~ Normal(loc, 1)
end

hsgp_model(df) = @brm df begin
    loc ~ 1 + hsgp(x, x2; k=(3, 4), c=(1.5, 2.0), iso=false)
    y ~ Normal(loc, 1)
end

grouped_hsgp_model(df) = @brm df begin
    loc ~ 1 + hsgp(x; k=4, by=g)
    y ~ Normal(loc, 1)
end

bad_exact_k(df) = @brm df begin
    loc ~ gp(x; k=10)
    y ~ Normal(loc, 1)
end

bad_exact_by(df) = @brm df begin
    loc ~ gp(x; by=g)
    y ~ Normal(loc, 1)
end

bad_hsgp_k_arity(df) = @brm df begin
    loc ~ hsgp(x, x2; k=(3,))
    y ~ Normal(loc, 1)
end

@testset "exact gp and Hilbert-space hsgp are separate terms" begin
    df = gp_term_df()
    exact = exact_gp_model(df)
    approx = hsgp_model(df)
    grouped = grouped_hsgp_model(df)

    @testset "public contract and validation" begin
        @test popcoefnames(exact, :loc) == [:Intercept]
        @test popcoefnames(approx, :loc) == [:Intercept]
        @test grouping_factors(grouped, :loc) == [:g]
        @test_throws "Use `hsgp" bad_exact_k(df)
        @test_throws "Use `hsgp" bad_exact_by(df)
        @test_throws "one value per axis" SBBRMI(bad_hsgp_k_arity(df); mod=@__MODULE__)
    end

    sb_exact = SBBRMI(exact; mod=@__MODULE__)
    sb_hsgp = SBBRMI(approx; mod=@__MODULE__)
    sb_grouped = SBBRMI(grouped; mod=@__MODULE__)

    @testset "data shapes and replay" begin
        @test size(sb_exact.data[:X_gp_x_x2]) == (GP_TERM_N, 2)
        @test size(sb_hsgp.data[:PHI_hsgp_x_x2]) == (GP_TERM_N, 12)
        @test size(sb_hsgp.data[:omega2_hsgp_x_x2]) == (12, 2)
        @test !haskey(sb_grouped.data, :g)
        @test sb_grouped.data[:g_idx] == [1, 1, 1, 1, 2, 2, 2, 2]

        exact_input = only(i for i in brm_descriptor(sb_exact).inputs
                           if i.transform === :gp)
        hsgp_input = only(i for i in brm_descriptor(sb_hsgp).inputs
                          if i.transform === :hsgp)
        @test exact_input.name === :X_gp_x_x2
        @test hsgp_input.name === :PHI_hsgp_x_x2
        @test exact_input.column === nothing
        @test hsgp_input.column === nothing

        replay_df = gp_term_df(; shift_x=0.25, shift_x2=0.4)
        replay_exact = reprocess(sb_exact, replay_df)
        replay_hsgp = reprocess(sb_hsgp, replay_df)
        replay_grouped = reprocess(sb_grouped, replay_df)
        @test replay_exact.data[:X_gp_x_x2] != sb_exact.data[:X_gp_x_x2]
        @test replay_hsgp.data[:PHI_hsgp_x_x2] != sb_hsgp.data[:PHI_hsgp_x_x2]
        @test replay_grouped.data[:g_idx] == sb_grouped.data[:g_idx]
        @test replay_grouped.data[:PHI_hsgp_x_by_g] !=
              sb_grouped.data[:PHI_hsgp_x_by_g]
        @test replay_grouped.preproc[:PHI_hsgp_x_by_g].const_.fits ==
              sb_grouped.preproc[:PHI_hsgp_x_by_g].const_.fits
        @test StanBlocks.stan_code(replay_exact.model) == StanBlocks.stan_code(sb_exact.model)
        @test StanBlocks.stan_code(replay_hsgp.model) == StanBlocks.stan_code(sb_hsgp.model)
        @test StanBlocks.stan_code(replay_grouped.model) ==
              StanBlocks.stan_code(sb_grouped.model)
        @test :reprocess in Symbol[
            op.name for op in brm_descriptor(sb_grouped).operations]
        @test_throws "unseen level" reprocess(
            sb_grouped, merge(replay_df, (; g=vcat(fill("a", 4), fill("c", 4)))))
        @test_throws "no cv-contagious size" reprocess(
            sb_grouped, replay_df; resample_groups=:g)

        # The hsgp validity floor (decision 13keyez) is a function of the
        # fitted `L = c*max|x - mean(x)|`, so it tracks the BASIS, not the
        # source. Note `replay_df` is a pure SHIFT and `L` is shift-invariant,
        # so the bound is unchanged there even under a re-fit -- moving it
        # needs a covariate whose SPREAD changes.
        floor_key = :rho_lower_hsgp_x_x2
        @test replay_hsgp.data[floor_key] == sb_hsgp.data[floor_key]
        @test reprocess(sb_hsgp, replay_df; freeze_constants=false).data[floor_key] ==
              sb_hsgp.data[floor_key]

        # Doubling every axis doubles `L`, and so the floor -- but only on a
        # re-fit. This is the whole reason the bound is DATA and not a literal
        # in the emitted source: a literal could not track the new basis, and
        # would go on bounding `rho` for a basis this data no longer has.
        wide_df = let b = gp_term_df()
            (; x = 2 .* b.x, x2 = 2 .* b.x2, y = b.y, g = b.g)
        end
        refit_wide = reprocess(sb_hsgp, wide_df; freeze_constants=false)
        @test refit_wide.data[floor_key] ≈ 2 .* sb_hsgp.data[floor_key]
        @test reprocess(sb_hsgp, wide_df).data[floor_key] == sb_hsgp.data[floor_key]
        @test StanBlocks.stan_code(refit_wide.model) == StanBlocks.stan_code(sb_hsgp.model)
    end

    @testset "transpile and stanc" begin
        for (label, sb) in ((:exact, sb_exact), (:hsgp, sb_hsgp), (:grouped_hsgp, sb_grouped))
            @test StanBlocks.stan.transpiles(sb.model)
            code = StanBlocks.stan_code(sb.model)
            @test occursin(label === :exact ? "brm_exp_quad_cov" : "brm_hsgp_sqrt_spd", code)
            if label === :exact
                @test occursin("gp_exp_quad_cov", code)
                @test occursin("add_diag", code)
                @test occursin("brm_gp_locations", code)
                @test !occursin("sqdist", code)
                @test !occursin("K_axis", code)
            end
            @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
        end
    end

    @testset "BridgeStan finite density and gradient" begin
        if GP_RUN_BRIDGESTAN
            isdir(GP_TERM_CACHE) || mkpath(GP_TERM_CACHE)
            for (label, sb) in ((:exact, sb_exact), (:hsgp, sb_hsgp),
                                (:grouped_hsgp, sb_grouped))
                code = StanBlocks.stan_code(sb.model)
                problem = StanBlocks.stan_instantiate(
                    sb.model; path=joinpath(GP_TERM_CACHE, string(hash(code)) * ".stan"))
                dimension = LogDensityProblems.dimension(problem)
                q = [0.05 * ((i % 5) - 2) for i in 1:dimension]
                lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
                @test isfinite(lp)
                @test length(gradient) == dimension
                @test all(isfinite, gradient)
            end
        else
            @info "Skipping BridgeStan GP/HSGP runtime gate (BRM_GP_RUNTIME=0)"
        end
    end
end
