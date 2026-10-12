# One NLME model's compiled Stan and RK targets, conditioned on a checked
# start: only the standardized subject draws are free.
# RUN: julia --project=test test/nlme_conditional.jl
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using BridgeStan, LinearAlgebra, LogDensityProblems, Random, WarmupHMC

const CONDITIONAL_DIR = mkpath(joinpath(tempdir(), "brm-nlme-conditional"))

# Every evaluation must equal the full target at the embedded point, with the
# gradient (and Hessian, where offered) restricted to the free coordinates.
function check_projection(c, x)
    full = copy(c.point)
    full[c.free] = x
    target = c.target
    @test LogDensityProblems.logdensity(c, x) ==
        LogDensityProblems.logdensity(target, full)
    value, gradient = LogDensityProblems.logdensity_and_gradient(c, x)
    full_value, full_gradient = LogDensityProblems.logdensity_and_gradient(target, full)
    @test value == full_value
    @test gradient == full_gradient[c.free]
    LogDensityProblems.capabilities(typeof(c)) isa
        LogDensityProblems.LogDensityOrder{2} || return
    _, _, hessian = LogDensityProblems.logdensity_gradient_and_hessian(c, x)
    _, _, full_hessian = LogDensityProblems.logdensity_gradient_and_hessian(target, full)
    @test hessian == full_hessian[c.free, c.free]
end

# The RK problem conditions the same model in RK packed order: at the same
# physical points its free coordinates, log-density differences and free
# gradients agree with the Stan problem's.
function check_rk_parity(c, r, xs1, xs2)
    t = c.map.transport
    @test r.backend === c.map.rk
    @test LogDensityProblems.dimension(r) == LogDensityProblems.dimension(c)
    @test sort(collect(zip(r.subjects, [m.level for m in r.margins]))) ==
        sort(collect(zip(c.subjects, [m.level for m in c.margins])))
    @test sort(r.names) == sort(String.(t.rk_names[t.permutation[c.free]]))
    rk_free(xs) = brm_stan_point_to_rk(t, brm_nlme_conditional_point(c, xs))[r.free]
    xr1, xr2 = rk_free(xs1), rk_free(xs2)
    @test brm_nlme_conditional_point(r, xr1) ≈
        brm_stan_point_to_rk(t, brm_nlme_conditional_point(c, xs1)) atol=1e-12
    check_projection(r, xr2)
    vs1, gs1 = LogDensityProblems.logdensity_and_gradient(c, xs1)
    vs2 = LogDensityProblems.logdensity(c, xs2)
    vr1, gr1 = LogDensityProblems.logdensity_and_gradient(r, xr1)
    vr2 = LogDensityProblems.logdensity(r, xr2)
    @test vr2 - vr1 ≈ vs2 - vs1 rtol=1e-8 atol=1e-10
    us1 = brm_nlme_conditional_point(c, xs1)
    _, full_gs1 = LogDensityProblems.logdensity_and_gradient(c.target, us1)
    @test brm_stan_gradient_to_rk(t, brm_stan_point_to_rk(t, us1), full_gs1)[r.free] ≈
        gr1 rtol=1e-8 atol=1e-10
    @test brm_nlme_joint_values(r, xr2).eta ≈ brm_nlme_joint_values(c, xs2).eta atol=1e-10
end

@stestset "flat NLME start conditions the exact compiled target" begin
    brmi = nlme_pk(NLME_DATA)
    rk = RKBRMI(brmi)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    map = brm_nlme_joint_map(rk, sb; path=joinpath(CONDITIONAL_DIR, "flat.stan"))
    θ = [0.32, -0.17, 0.48]
    σ = [-0.61]
    Ω = [0.36 0.105; 0.105 0.49]
    H = [0.12 -0.09 0.31 -0.22; -0.08 0.14 0.03 -0.17]
    start = brm_nlme_joint_start(map; theta=θ, sigma=σ, omega=Ω, eta=H,
        subjects=map.view.levels)
    saved = deepcopy(start)
    c = brm_nlme_conditional_problem(map, start)
    @test c.map === map
    @test c.backend === map.sb
    @test c.target === map.stan_problem
    @test LogDensityProblems.dimension(c) == length(H)
    @test LogDensityProblems.capabilities(typeof(c)) ==
        LogDensityProblems.capabilities(typeof(map.stan_problem))
    @test c.free == start.free.indices
    @test c.fixed == start.fixed.indices
    @test c.names == start.free.names
    @test c.position == start.free.values
    @test c.point == start.stan
    @test c.subjects == [map.view.levels[map.view.subject[map.transport.permutation[j]]]
        for j in c.free]
    @test sort(unique(c.subjects)) == sort(map.view.levels)
    @test all(m.block.group === :subject for m in c.margins)
    @test all(m.level == s for (m, s) in zip(c.margins, c.subjects))
    @test sort([(m.level, m.predictor) for m in c.margins]) ==
        sort([(m.level, m.predictor) for margins in map.view.subject_margins
            for m in margins])

    x = c.position .+ range(-0.4, 0.5; length=length(c.position))
    x_saved = copy(x)
    check_projection(c, x)
    @test x == x_saved
    point = brm_nlme_conditional_point(c, x)
    @test point[c.fixed] == start.fixed.values
    @test point[c.free] == x
    # The fixed coordinates are the NLME estimate; the free draws move η only.
    values = brm_nlme_joint_values(map, point)
    @test brm_nlme_joint_values(c, x) == values
    @test values.theta ≈ θ atol=1e-10
    @test values.sigma ≈ σ atol=1e-10
    @test values.omega ≈ Ω atol=1e-10
    @test values.eta != brm_nlme_joint_values(map, start.stan).eta
    X = [c.position x]
    @test brm_nlme_conditional_point(c, X) == [start.stan point]
    @test X == [c.position x_saved]
    # refused: a free point of the wrong length has no full Stan embedding.
    @test_throws DimensionMismatch LogDensityProblems.logdensity(c, x[2:end])
    @test_throws DimensionMismatch brm_nlme_conditional_point(c, X[2:end, :])

    # WarmupHMC samples the free coordinates alone; every draw keeps the
    # fitted population, scalar and covariance values.
    result = adaptive_warmup_mcmc(Xoshiro(2026), c;
        init=(; position=copy(c.position), squared_scale=ones(length(c.position))),
        n_draws=100)
    draws = convert(Matrix{Float64}, result.posterior_position)
    @test size(draws, 1) == length(c.free)
    @test size(draws, 2) >= 100
    full_draws = brm_nlme_conditional_point(c, draws)
    @test all(full_draws[c.fixed, :] .== start.fixed.values)
    last_values = brm_nlme_joint_values(map, full_draws[:, end])
    @test last_values.omega ≈ Ω atol=1e-10
    @test last_values.theta ≈ θ atol=1e-10
    @test start == saved

    r = brm_nlme_conditional_problem(map, start; backend=map.rk, ad_backend=NLME_AD)
    @test LogDensityProblems.capabilities(typeof(r)) ==
        LogDensityProblems.LogDensityOrder{1}()
    @test r.point ≈ brm_stan_point_to_rk(map.transport, start.stan) atol=1e-12
    check_rk_parity(c, r, c.position, x)
    rk_result = adaptive_warmup_mcmc(Xoshiro(2027), r;
        init=(; position=copy(r.position), squared_scale=ones(length(r.position))),
        n_draws=100)
    rk_draws = convert(Matrix{Float64}, rk_result.posterior_position)
    @test size(rk_draws, 1) == length(r.free)
    @test all(brm_nlme_conditional_point(r, rk_draws)[r.fixed, :] .== r.point[r.fixed])
    rk_values = brm_nlme_joint_values(r, rk_draws[:, end])
    @test rk_values.omega ≈ Ω atol=1e-10
    @test rk_values.theta ≈ θ atol=1e-10
    @test start == saved
    # refused: the backend must be one of this map's own backends.
    @test_throws BRMNLMEViewError brm_nlme_conditional_problem(map, start;
        backend=RKBRMI(brmi), ad_backend=NLME_AD)

    # refused: a start whose partition, names, values or dimension are not this
    # map's would condition the compiled target on coordinates the NLME fit
    # never produced (snag nlme-joint-condi-84096cce asks for exactly these
    # rejections).
    swapped = merge(start, (; free=merge(start.free,
        (; indices=[start.fixed.indices[1]; start.free.indices[2:end]]))))
    @test_throws BRMNLMEViewError brm_nlme_conditional_problem(map, swapped)
    renamed = merge(start, (; free=merge(start.free,
        (; names=reverse(start.free.names)))))
    @test_throws BRMNLMEViewError brm_nlme_conditional_problem(map, renamed)
    moved = copy(start.stan)
    moved[start.fixed.indices[1]] += 0.1
    @test_throws BRMNLMEViewError brm_nlme_conditional_problem(map,
        merge(start, (; stan=moved)))
    @test_throws DimensionMismatch brm_nlme_conditional_problem(map,
        merge(start, (; stan=start.stan[1:end-1])))
    nonfinite = copy(start.stan)
    nonfinite[start.free.indices[1]] = NaN
    @test_throws BRMNLMEViewError brm_nlme_conditional_problem(map,
        merge(start, (; stan=nonfinite,
            free=merge(start.free, (; values=nonfinite[start.free.indices])))))
    @test start == saved

    other_brmi = @brm NLME_DATA begin
        sigma ~ Exponential(1)
        mu ~ 1 + wt + (1 | subject)
        y ~ Normal(mu, sigma)
    end
    other = brm_nlme_joint_map(RKBRMI(other_brmi),
        SBBRMI(other_brmi; mod=@__MODULE__, total_groups=());
        path=joinpath(CONDITIONAL_DIR, "other.stan"))
    @test_throws DimensionMismatch brm_nlme_conditional_problem(other, start)
end

@stestset "fixed shared HSGP coordinates stay fixed" begin
    data = (; subject=["a", "b", "c"], t=[-0.8, 0.1, 0.7], y=[0.2, -0.1, 0.5])
    brmi = @brm data begin
        sigma ~ Exponential(1)
        mu ~ 1 + hsgp(t; k=3, domain=(-1.0, 1.0)) + (1 | subject)
        y ~ Normal(mu, sigma)
    end
    map = brm_nlme_joint_map(RKBRMI(brmi),
        SBBRMI(brmi; mod=@__MODULE__, total_groups=());
        path=joinpath(CONDITIONAL_DIR, "shared-hsgp.stan"))
    view = map.view
    shared = findall(==(:shared_latent), view.role)
    point = zeros(length(view.coordinates))
    point[shared] .= collect(0.1:0.1:0.1length(shared))
    start = brm_nlme_joint_start(map;
        theta=fill(0.2, count(==(:population), view.role)), sigma=[-0.3],
        omega=reshape([0.49], 1, 1), eta=reshape([0.12, -0.2, 0.3], 1, :),
        subjects=view.levels, shared_point=point)
    c = brm_nlme_conditional_problem(map, start)
    @test LogDensityProblems.dimension(c) == 3
    shared_stan = [j for (j, k) in enumerate(map.transport.permutation) if k in shared]
    @test !isempty(shared_stan)
    @test isempty(intersect(shared_stan, c.free))
    x = c.position .+ [0.3, -0.2, 0.1]
    check_projection(c, x)
    values = brm_nlme_joint_values(map, brm_nlme_conditional_point(c, x))
    @test values.shared.values ≈ point[shared] atol=1e-10
    r = brm_nlme_conditional_problem(map, start; backend=map.rk, ad_backend=NLME_AD)
    @test isempty(intersect(shared, r.free))
    check_rk_parity(c, r, c.position, x)
    @test brm_nlme_joint_values(r, r.position .- 0.2).shared.values ≈
        point[shared] atol=1e-10
end

@stestset "nested unit draws are free with their subject provenance" begin
    brmi = nested_model(NESTED_DATA)
    map = brm_nlme_joint_map(RKBRMI(brmi),
        SBBRMI(brmi; mod=@__MODULE__, total_groups=());
        path=joinpath(CONDITIONAL_DIR, "nested.stan"),
        subject_group=:subject, nested_groups=(unit=:obs_subject,))
    Ω = [reshape([0.49], 1, 1), reshape([0.36], 1, 1)]
    H = [[0.1, 0.2], [-0.2, 0.3, -0.1], [0.2, -0.1, 0.05, 0.3]]
    start = brm_nlme_joint_start(map; theta=[0.15], sigma=[log(0.7)],
        omega=Ω, eta=H, subjects=map.view.levels)
    c = brm_nlme_conditional_problem(map, start)
    @test LogDensityProblems.dimension(c) == sum(length, H)
    @test sort(collect(zip(c.subjects, [m.level for m in c.margins]))) ==
        sort([(level, m.level) for (level, margins) in
            zip(map.view.levels, map.view.subject_margins) for m in margins])
    @test count(m -> m.block.group === :unit, c.margins) == 6
    x = c.position .+ 0.05
    check_projection(c, x)
    values = brm_nlme_joint_values(map, brm_nlme_conditional_point(c, x))
    @test all(isapprox(values.omega[f], Ω[f]; atol=1e-10) for f in 1:2)
    @test values.theta ≈ [0.15] atol=1e-10
    @test values.sigma ≈ [log(0.7)] atol=1e-10
    r = brm_nlme_conditional_problem(map, start; backend=map.rk, ad_backend=NLME_AD)
    @test count(m -> m.block.group === :unit, r.margins) == 6
    check_rk_parity(c, r, c.position, x)
end
