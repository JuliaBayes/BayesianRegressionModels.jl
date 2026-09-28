# S2Z SBBRMI emission: planner selection, stanc validity, fail-closed scope,
# and the BridgeStan joint = collapsed x recovery identity against the
# ordinary parameterization. Run with `julia --project=test test/s2z_emission.jl`.
using Test, Random, LinearAlgebra, Statistics, Distributions
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
const BRM = BayesianRegressionModels

data = (; x=[0., 1., 0., 1., 2., 3.], g=[1, 1, 2, 2, 3, 3], y=[0., 1., 2., 3., 1., 2.])
normal_builder = @brm begin
    mu ~ 1 + x + (1 + x || g)
    effect(mu, Intercept) ~ Normal(0, 5)
    effect(mu, x) ~ Normal(0, 2)
    y ~ Normal(mu, 1)
end
flat_builder = @brm begin
    mu ~ 1 + x + (1 + x || g)
    effect(mu, Intercept) ~ Normal(0, 5)
    effect(mu, x) ~ Flat()
    y ~ Normal(mu, 1)
end

s2z_stanc_ok(sb) = begin
    m = sb.model
    StanBlocks.transpiles(m) &&
        StanBlocks.stanc_check(StanBlocks.stan_code(m); warn_pedantic=false).ok
end

@testset "S2Z planner selection and scope" begin
    brmi = normal_builder(data)
    sb = SBBRMI(brmi; mod=@__MODULE__, s2z_groups=[:g], s2z_rho=0.4, total_groups=())
    block = only(s2z_effect_blocks(sb))
    @test block.group === :g
    @test block.columns == (:Intercept, :x)
    @test block.population_columns == (:Intercept, :x)
    @test block.B ≈ [1.0 0.0; 0.0 1.0]
    @test size(block.rho) == (3, 2)
    @test isempty(total_effect_blocks(sb))
    # Default and explicit opt-outs stay conventional.
    @test isempty(s2z_effect_blocks(SBBRMI(brmi; mod=@__MODULE__, total_groups=())))
    @test isempty(s2z_effect_blocks(SBBRMI(brmi; mod=@__MODULE__, s2z_groups=(), total_groups=())))
    # Vector rho resolves per coefficient.
    sbv = SBBRMI(brmi; mod=@__MODULE__, s2z_groups=:g, s2z_rho=[0.0, 1.0], total_groups=())
    @test only(s2z_effect_blocks(sbv)).rho == repeat([0.0 1.0], 3, 1)
    # stanc at both endpoints and an interior weight.
    for rho in (0.0, 0.4, 1.0)
        sbr = SBBRMI(brmi; mod=@__MODULE__, s2z_groups=[:g], s2z_rho=rho, total_groups=())
        @test s2z_stanc_ok(sbr)
    end
    @test brm_descriptor(sb) isa BRMDescriptor
    # Fail-closed scope.
    @test_throws ArgumentError SBBRMI(brmi; mod=@__MODULE__, s2z_groups=[:g], total_groups=())
    @test_throws ArgumentError SBBRMI(brmi; mod=@__MODULE__, s2z_groups=[:g], s2z_rho=2.0, total_groups=())
    @test_throws ArgumentError SBBRMI(brmi; mod=@__MODULE__, s2z_groups=[:g], s2z_rho=[0.5], total_groups=())
    @test_throws ArgumentError SBBRMI(brmi; mod=@__MODULE__, s2z_groups=[:missing], s2z_rho=0.5, total_groups=())
    @test_throws ArgumentError SBBRMI(brmi; mod=@__MODULE__, s2z_groups=[:g], s2z_rho=0.5,
        centered_groups=[:g], total_groups=())
    @test_throws ArgumentError SBBRMI(brmi; mod=@__MODULE__, s2z_groups=[:g], s2z_rho=0.5,
        cv_groups=[:g], total_groups=())
    @test_throws ArgumentError SBBRMI(brmi; mod=@__MODULE__, s2z_groups=[:g], s2z_rho=0.5,
        total_groups=[:g])
    student = @brm begin
        mu ~ 1 + x + (1 + x || g)
        effect(mu, Intercept) ~ LocationScale(2., 3., TDist(3))
        y ~ Normal(mu, 1)
    end
    @test_throws ArgumentError SBBRMI(student(data); mod=@__MODULE__,
        s2z_groups=[:g], s2z_rho=0.5, total_groups=())
    idmodel = @brm begin
        mu ~ 1 + x + (1 | a | g) + (0 + x | b | g)
        y ~ Normal(mu, 1)
    end
    @test_throws ArgumentError SBBRMI(idmodel(data); mod=@__MODULE__,
        s2z_groups=[:g], s2z_rho=0.5, total_groups=())
    unmatched = @brm begin
        mu ~ 1 + (1 + x || g)
        y ~ Normal(mu, 1)
    end
    @test_throws ArgumentError SBBRMI(unmatched(data); mod=@__MODULE__,
        s2z_groups=[:g], s2z_rho=0.5, total_groups=())
    onelevel = (; x=[0., 1.], g=[1, 1], y=[0., 1.])
    @test_throws ArgumentError SBBRMI(normal_builder(onelevel); mod=@__MODULE__,
        s2z_groups=[:g], s2z_rho=0.5, total_groups=())
end

@testset "S2Z BridgeStan identity audit" begin
    for (builder, tag) in ((normal_builder, :normal), (flat_builder, :flat))
        brmi = builder(data)
        sb = SBBRMI(brmi; mod=@__MODULE__, s2z_groups=[:g], s2z_rho=0.4, total_groups=())
        block = only(s2z_effect_blocks(sb))
        problem = StanBlocks.stan_instantiate(sb.model; path=joinpath(mktempdir(), "s2z_$tag.stan"))
        names = BridgeStan.param_unc_names(problem.model)
        println("S2Z $tag PARAMETERS ", names)
        coords = BRM._s2z_coordinates(sb, block, names)
        @test length(names) == 8
        conventional = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
        original = StanBlocks.stan_instantiate(conventional.model;
            path=joinpath(mktempdir(), "original_$tag.stan"))
        original_names = BridgeStan.param_unc_names(original.model)
        println("ORDINARY $tag PARAMETERS ", original_names)
        original_blocks = adaptive_centering_blocks(conventional, original_names)
        intercept_block = only(b for b in original_blocks if b.ranef.n_terms == 1 &&
            b.ranef.family in BRM._ADAPTIVE_INTERCEPT_FAMILIES)
        slope_block = only(b for b in original_blocks if !(b === intercept_block))
        pop_indices = [only(findall(==("pop_mu_beta_pop.$k"), original_names)) for k in 1:2]
        J, K = 3, 2
        m_offset = [0.3, -0.2]
        function original_joint(x)
            conditional = BRM._s2z_conditional(block, coords, x)
            tau = conditional.sd
            theta = conditional.theta
            delta = Matrix{Float64}(undef, J, K)
            for k in 1:K
                u = BRM._s2z_helmert_mul(Vector{Float64}(x[coords.contrasts[:, k]]))
                delta[:, k] = BRM._s2z_partial_forward(u, tau[k], block.rho[:, k])
            end
            m = conditional.mean + m_offset
            beta = theta - block.B * m
            b = delta .+ reshape(m, 1, K)
            x_ord = zeros(length(original_names))
            x_ord[pop_indices] = beta
            x_ord[vec(intercept_block.effects)] = b[:, 1] ./ tau[1]
            x_ord[intercept_block.log_scales] = x[coords.scales[1:1]]
            x_ord[vec(slope_block.effects)] = b[:, 2] ./ tau[2]
            x_ord[slope_block.log_scales] = x[coords.scales[2:2]]
            cov = isempty(findall(>(0), block.precision)) ? Diagonal(tau .^ 2 ./ J) :
                Diagonal(conditional.prior_var) -
                conditional.gain * conditional.B1 * Diagonal(conditional.prior_var)
            # No scale-prior adjustment: S2Z's LogNormal(tau) plus its lower-bound
            # +log(tau) Jacobian equals the ordinary log_scale ~ Normal density
            # exactly, and the slope taus match on both sides. The (beta,b) to
            # (theta,z,m) map contributes its full Jacobian: per coefficient a
            # sqrt(J) volume factor times the restricted partial-map Jacobian.
            BridgeStan.log_density(original.model, x_ord; propto=false) -
                J * sum(log, tau) +
                sum(BRM._s2z_partial_logjac(tau[k], block.rho[:, k]) for k in 1:K) +
                0.5 * K * log(J) -
                logpdf(MvNormal(conditional.mean, Symmetric(cov)), m)
        end
        q = zeros(length(names))
        q[coords.scales] .= log.([2.0, 0.5])
        q[coords.theta] .= [1.0, -0.5]
        for trial in 1:4
            x = q .+ 0.25 * randn(Xoshiro(100 + trial), length(q))
            @test BridgeStan.log_density(problem.model, x; propto=false) ≈
                original_joint(x) atol = 1e-9
        end
        # Generated recovery quantities evaluate finite.
        stanrng = BridgeStan.StanRNG(problem.model, 917)
        gq = BridgeStan.param_constrain(problem.model, q;
            include_tp=true, include_gq=true, rng=stanrng)
        @test all(isfinite, gq)
        # Gradients are finite and match finite differences of the Stan target.
        _, grad = BridgeStan.log_density_gradient(problem.model, q; propto=false)
        @test all(isfinite, grad)
        h = 1e-6
        fd = map(eachindex(q)) do i
            qp, qm = copy(q), copy(q)
            qp[i] += h
            qm[i] -= h
            (BridgeStan.log_density(problem.model, qp; propto=false) -
             BridgeStan.log_density(problem.model, qm; propto=false)) / (2h)
        end
        @test fd ≈ grad atol = 1e-5
    end
end

@testset "S2Z prediction and replay guards" begin
    brmi = normal_builder(data)
    sb = SBBRMI(brmi; mod=@__MODULE__, s2z_groups=[:g], s2z_rho=0.4, total_groups=())
    block = only(s2z_effect_blocks(sb))
    code = BRM.stan_code(sb)
    @test s2z_stanc_ok(sb)
    # Same-level replay rebinds; changed levels fail on the weights.
    replay = reprocess(sb, (; data..., x=data.x .+ 0.7))
    @test replay.data[block.group_index] == sb.data[block.group_index]
    @test BRM.stan_code(replay) == code
    newlevels = (; x=[9., 8., 7., 6.], g=[3, 3, 4, 4], y=[0., 0., 0., 0.])
    @test_throws ErrorException reprocess(sb, newlevels)
    reordered = (; x=[9., 8., 7., 6., 5., 4.], g=[2, 2, 1, 1, 3, 3], y=zeros(6))
    replay_reordered = reprocess(sb, reordered)
    @test replay_reordered.data[block.group_index] == [2, 2, 1, 1, 3, 3]
    subset = (; x=[9., 8., 7., 6.], g=[1, 1, 2, 2], y=zeros(4))
    @test_throws ArgumentError reprocess(sb, subset)
    @test_throws ArgumentError reprocess(sb, data; freeze_constants=false)
    plan = generative_plan(normal_builder, data; s2z_groups=[:g], s2z_rho=0.4,
        total_groups=())
    @test length(s2z_effect_blocks(plan)) == 1
    @test_throws ArgumentError generative_plan(plan, newlevels)
    names = ["s2z_contrast_mu.$r.$k" for k in 1:2 for r in 1:2]
    append!(names, ["s2z_theta_mu.$p" for p in 1:2])
    append!(names, ["s2z_scale_mu_tau.$k" for k in 1:2])
    draws = zeros(2, 8)
    @test_throws ArgumentError population_draws(sb, draws, names; groups=:g)
    @test_throws ArgumentError transport_draws(plan, plan, draws, names, names)
    @test_throws ArgumentError reprocess(sb, data; resample_groups=[:g])
end

@testset "S2Z Julia recovery statistics" begin
    brmi = normal_builder(data)
    sb = SBBRMI(brmi; mod=@__MODULE__, s2z_groups=[:g], s2z_rho=0.4, total_groups=())
    block = only(s2z_effect_blocks(sb))
    names = ["s2z_contrast_mu.$r.$k" for k in 1:2 for r in 1:2]
    append!(names, ["s2z_theta_mu.$p" for p in 1:2])
    append!(names, ["s2z_scale_mu_tau.$k" for k in 1:2])
    q = zeros(8)
    q[5:6] .= [1.0, -0.5]
    q[7:8] .= log.([2.0, 0.5])
    coords = BRM._s2z_coordinates(sb, block, names)
    conditional = BRM._s2z_conditional(block, coords, q)
    recovered = recover_s2z_draws(sb, repeat(q', 4000, 1), names; rng=Xoshiro(913))[block.predictor]
    @test vec(mean(recovered.population; dims=1)) ≈
        conditional.theta - block.B * conditional.mean atol = 0.08
    @test maximum(abs, recovered.effects .- recovered.deviations .-
        reshape(recovered.means, 4000, 1, 2)) < 1e-12
    @test maximum(abs, sum(recovered.deviations; dims=2)) < 1e-10
end

@testset "S2Z group coordinates match contrasts plus an auxiliary mean" begin
    brmi = normal_builder(data)
    cmat = [0.0 0.3; 0.7 1.0; 0.25 0.5]
    sbg = SBBRMI(brmi; mod=@__MODULE__, s2z_groups=[:g], s2z_coordinates=:groups,
        s2z_rho=cmat, total_groups=())
    block = only(s2z_effect_blocks(sbg))
    @test block.coordinates === :groups
    @test block.rho == cmat
    @test s2z_stanc_ok(sbg)
    # Group coordinates default to the noncentered frame.
    default = SBBRMI(brmi; mod=@__MODULE__, s2z_groups=[:g], s2z_coordinates=:groups,
        total_groups=())
    @test only(s2z_effect_blocks(default)).rho == zeros(3, 2)
    @test_throws ArgumentError SBBRMI(brmi; mod=@__MODULE__, s2z_groups=[:g],
        s2z_coordinates=:bogus, s2z_rho=0.0, total_groups=())
    sbc = SBBRMI(brmi; mod=@__MODULE__, s2z_groups=[:g], s2z_rho=0.0, total_groups=())
    pg = StanBlocks.stan_instantiate(sbg.model; path=joinpath(mktempdir(), "s2z_groups.stan"))
    pc = StanBlocks.stan_instantiate(sbc.model; path=joinpath(mktempdir(), "s2z_ncp.stan"))
    ng, nc = BridgeStan.param_unc_names(pg.model), BridgeStan.param_unc_names(pc.model)
    cg = BRM._s2z_coordinates(sbg, block, ng)
    cc = BRM._s2z_coordinates(sbc, only(s2z_effect_blocks(sbc)), nc)
    @test size(cg.contrasts) == (3, 2) && size(cc.contrasts) == (2, 2)
    @test all(n -> startswith(n, "s2z_level_mu."), ng[vec(cg.contrasts)])
    for trial in 1:4
        x = 0.5 .* randn(Xoshiro(200 + trial), length(ng))
        x[cg.scales] .+= log.([1.5, 0.6])
        tau = exp.(x[cg.scales])
        xc = zeros(length(nc))
        xc[cc.scales] = x[cg.scales]
        xc[cc.theta] = x[cg.theta]
        # Contrast target at w, plus the independent N(0, 1) auxiliary mean and
        # the s -> w Jacobian.
        extra = 0.0
        for k in 1:2
            w = x[cg.contrasts[:, k]] .* exp.(-cmat[:, k] .* log(tau[k]))
            xc[cc.contrasts[:, k]] = BRM._s2z_helmert_transpose_mul(w)
            extra += logpdf(Normal(), sum(w) / sqrt(3)) - sum(cmat[:, k]) * log(tau[k])
        end
        @test BridgeStan.log_density(pg.model, x; propto=false) ≈
            BridgeStan.log_density(pc.model, xc; propto=false) + extra atol = 1e-9
        dg = recover_s2z_draws(sbg, reshape(x, 1, :), ng; rng=Xoshiro(1))[:mu]
        dc = recover_s2z_draws(sbc, reshape(xc, 1, :), nc; rng=Xoshiro(1))[:mu]
        @test dg.deviations ≈ dc.deviations atol = 1e-12
        @test dg.population ≈ dc.population atol = 1e-12
        # Julia recovery matches Stan's transformed deviations.
        constrained = BridgeStan.param_constrain(pg.model, x; include_tp=true)
        tp = BridgeStan.param_names(pg.model; include_tp=true)
        dev = [constrained[only(findall(==("s2z_deviation_mu.$j.$k"), tp))] for j in 1:3, k in 1:2]
        @test dev ≈ dg.deviations[1, :, :] atol = 1e-12
    end
    q = 0.3 .* randn(Xoshiro(7), length(ng))
    _, grad = BridgeStan.log_density_gradient(pg.model, q; propto=false)
    fd = map(eachindex(q)) do i
        qp, qm = copy(q), copy(q)
        qp[i] += 1e-6; qm[i] -= 1e-6
        (BridgeStan.log_density(pg.model, qp; propto=false) -
         BridgeStan.log_density(pg.model, qm; propto=false)) / 2e-6
    end
    @test fd ≈ grad atol = 1e-5
    # Fisher weights parameterize the contrast map only.
    @test_throws ArgumentError select_s2z_rho(sbg, sbc, zeros(3, 1), ["a"];
        obs_prec=ones(3, 6), group=:g)
end
