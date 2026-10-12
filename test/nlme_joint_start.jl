# Complete NLME covariance/natural-effect estimate -> same BRMI's Stan point.
# RUN: julia --project=test test/nlme_joint_start.jl
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using BridgeStan, LinearAlgebra

@stestset "NLME estimate becomes a checked full Stan start" begin
    brmi = nlme_pk(NLME_DATA)
    rk = RKBRMI(brmi)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    mkpath(joinpath(tempdir(), "brm-nlme-joint-start"))
    map = brm_nlme_joint_map(rk, sb;
        path=joinpath(tempdir(), "brm-nlme-joint-start", "nlme-pk.stan"))
    stan = map.stan_problem
    view = map.view
    θ = [0.32, -0.17, 0.48]
    σ = [-0.61]
    Ω = [0.36 0.105; 0.105 0.49]
    H = [0.12 -0.09 0.31 -0.22; -0.08 0.14 0.03 -0.17]
    labels = reverse(view.levels)
    saved = (copy(θ), copy(σ), copy(Ω), copy(H), copy(labels))
    start = brm_nlme_joint_start(map; theta=θ, sigma=σ, omega=Ω,
        eta=H, subjects=labels)
    @test length(start.stan) == length(BridgeStan.param_unc_names(stan.model))
    @test sort(vcat(start.fixed.indices, start.free.indices)) ==
        collect(eachindex(start.stan))
    @test isempty(intersect(start.fixed.indices, start.free.indices))
    @test start.fixed.names == map.transport.stan_names[start.fixed.indices]
    @test start.free.names == map.transport.stan_names[start.free.indices]
    @test start.fixed.values == start.stan[start.fixed.indices]
    @test start.free.values == start.stan[start.free.indices]
    @test length(start.free.indices) == length(view.levels) * size(H, 1)
    @test start.checked.max_error <= 1e-9
    recovered = brm_nlme_joint_values(map, start.stan)
    @test recovered.theta ≈ θ atol=1e-10
    @test recovered.sigma ≈ σ atol=1e-10
    @test recovered.omega ≈ Ω atol=1e-10
    @test recovered.eta ≈ H[:, reverse(eachindex(labels))] atol=1e-10
    @test recovered.subjects == view.levels
    @test (θ, σ, Ω, H, labels) == saved
    # The tempting direct copy starts at unit covariance and treats natural
    # effects as standardized draws; it is a different physical point.
    copied = zeros(length(view.coordinates))
    copied[findall(==(:population), view.role)] = θ
    copied[findall(==(:scalar), view.role)] = σ
    for (i, j) in enumerate(reverse(eachindex(labels)))
        copied[view.subject_coordinates[i]] = H[:, j]
    end
    mistaken = brm_nlme_joint_values(map,
        brm_rk_point_to_stan(map.transport, copied))
    @test mistaken.omega != Ω
    @test mistaken.eta ≈ H[:, reverse(eachindex(labels))] atol=1e-10
    @test start.stan != brm_rk_point_to_stan(map.transport, copied)
    @test_throws BRMNLMEViewError brm_nlme_joint_start(map;
        theta=θ, sigma=σ, omega=Ω, eta=H,
        subjects=[labels[1], labels[1], labels[3], labels[4]])
    @test_throws BRMNLMEViewError brm_nlme_joint_start(map;
        theta=θ, sigma=σ, omega=[1.0 2.0; 2.0 1.0], eta=H,
        subjects=labels)
    @test_throws DimensionMismatch brm_nlme_joint_start(map;
        theta=θ[1:2], sigma=σ, omega=Ω, eta=H, subjects=labels)
    @test_throws ErrorException brm_nlme_joint_map(rk,
        SBBRMI(nlme_pk(merge(NLME_DATA, (; wt=NLME_DATA.wt .+ 0.1)));
            mod=@__MODULE__); path=joinpath(tempdir(), "unused.stan"))
end

@stestset "shared HSGP point survives the full Stan conversion" begin
    data = (; subject=["a", "b", "c"], t=[-0.8, 0.1, 0.7],
        y=[0.2, -0.1, 0.5])
    brmi = @brm data begin
        sigma ~ Exponential(1)
        mu ~ 1 + hsgp(t; k=3, domain=(-1.0, 1.0)) + (1 | subject)
        y ~ Normal(mu, sigma)
    end
    rk = RKBRMI(brmi)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    map = brm_nlme_joint_map(rk, sb;
        path=joinpath(tempdir(), "brm-nlme-joint-start", "shared-hsgp.stan"))
    view = map.view
    shared = findall(==(:shared_latent), view.role)
    @test !isempty(shared)
    point = zeros(length(view.coordinates))
    point[shared] .= collect(0.1:0.1:0.1length(shared))
    θ = fill(0.2, count(==(:population), view.role))
    σ = [-0.3]
    Ω = reshape([0.49], 1, 1)
    H = reshape([0.12, -0.2, 0.3], 1, :)
    @test_throws BRMNLMEViewError brm_nlme_joint_start(map;
        theta=θ, sigma=σ, omega=Ω, eta=H, subjects=view.levels)
    @test_throws DimensionMismatch brm_nlme_joint_start(map;
        theta=θ, sigma=σ, omega=Ω, eta=H, subjects=view.levels,
        shared_point=point[1:end-1])
    start = brm_nlme_joint_start(map;
        theta=θ, sigma=σ, omega=Ω, eta=H, subjects=view.levels,
        shared_point=point)
    recovered = brm_nlme_joint_values(map, start.stan)
    @test recovered.shared.names == view.coordinates[shared]
    @test recovered.shared.values ≈ point[shared] atol=1e-10
    @test recovered.omega ≈ Ω atol=1e-10
    @test recovered.eta ≈ H atol=1e-10
    @test start.checked.max_error <= 1e-9
    @test length(start.fixed.indices) + length(start.free.indices) ==
        length(start.stan)
end

@stestset "independent NLME covariance blocks stay independent in Stan" begin
    data = (; subject=["a", "b", "a", "b"], y=[0.4, -0.3, 0.2, -0.1],
        y2=[-0.2, 0.4, -0.1, 0.2])
    brmi = @brm data begin
        sigma ~ Exponential(1)
        a ~ 1 + (1 | aa | subject)
        b ~ 1 + (1 | bb | subject)
        y ~ Normal(a, sigma)
        y2 ~ Normal(b, sigma)
    end
    rk = RKBRMI(brmi)
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    mkpath(joinpath(tempdir(), "brm-nlme-joint-start"))
    map = brm_nlme_joint_map(rk, sb;
        path=joinpath(tempdir(), "brm-nlme-joint-start", "two-blocks.stan"))
    @test length(map.blocks) == 2
    Ω = Matrix(Diagonal([0.25, 0.64]))
    H = [0.12 -0.2; 0.03 0.15]
    θ = fill(0.2, count(==(:population), map.view.role))
    σ = [-0.7]
    start = brm_nlme_joint_start(map; theta=θ, sigma=σ, omega=Ω,
        eta=H, subjects=map.view.levels)
    recovered = brm_nlme_joint_values(map, start.stan)
    @test recovered.omega ≈ Ω atol=1e-10
    @test recovered.eta ≈ H atol=1e-10
    @test length(start.free.indices) == length(H)
    coupled = [0.25 0.02; 0.02 0.64]
    @test_throws BRMNLMEViewError brm_nlme_joint_start(map;
        theta=θ, sigma=σ, omega=coupled, eta=H, subjects=map.view.levels)
end
