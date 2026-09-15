using Test, Random, LinearAlgebra, Statistics, Distributions
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
using WarmupHMC, Enzyme
using DifferentiationInterface: AutoEnzyme
const BRM = BayesianRegressionModels

builder = @brm begin
    mu ~ 1 + (1|school_effect|school)
    effect(mu,Intercept) ~ Normal(2,3)
    sd(:,school_effect) ~ Normal(0,1)
    y ~ Normal(mu,0.5)
end
data = (;school=repeat(1:6;inner=4),
    y=repeat(collect(range(0.5,3.5;length=6));inner=4) .+ repeat([-0.3,-0.1,0.1,0.3],6))
sb = SBBRMI(builder(data);mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model;path=joinpath(mktempdir(),"whmc-total.stan"))
names = BridgeStan.param_unc_names(problem.model)
block = only(total_effect_blocks(sb))
coords = BRM._total_coordinates(sb,block,names)
physical = zeros(length(names))
physical[only(coords.scales)] = log(1.3)
physical[vec(coords.totals)] = collect(range(0.5,3.5;length=6))
println("TOTAL_WHMC_COMPILED");flush(stdout)

@testset "Exact total WHMC coordinate transport" begin
    for c in (0.,0.37,1.)
        rp = adaptive_centering_problem(sb,problem,AutoEnzyme();centeredness=c)
        ir = WarmupHMC.reparametrizer(rp)
        _,source = WarmupHMC._inverse_with_logabsdet_jacobian(ir,physical)
        lp,g = LogDensityProblems.logdensity_and_gradient(rp,source)
        jac,mapped = ir(source)
        @test mapped ≈ physical atol=1e-12
        @test lp ≈ LogDensityProblems.logdensity(problem,physical)+jac atol=1e-10
        fd = map(eachindex(source)) do i
            h=zeros(length(source));h[i]=1e-5
            (LogDensityProblems.logdensity(rp,source+h)-LogDensityProblems.logdensity(rp,source-h))/2e-5
        end
        @test g ≈ fd atol=1e-7 rtol=1e-7
        # Candidate coordinates and marginal gradients must be independent of
        # which source frame supplied the same physical observation.
        plan = WarmupHMC.candidate_scoring_plan(rp)
        frame = plan.prepare(ir,source,g)
        for (p,(index,value)) in enumerate(ir.pairs), candidate in (0.,0.5,1.)
            _,position,gradient = plan.score(frame,p,index,value,WarmupHMC.PartiallyCentered(candidate))
            mu=2.;ell=physical[only(coords.scales)]
            @test position ≈ candidate*mu+(physical[index]-mu)*exp((candidate-1)*ell) atol=1e-12
            @test gradient ≈ last(LogDensityProblems.logdensity_and_gradient(problem,physical))[index]*exp((1-candidate)*ell) atol=1e-10
        end
    end
end

@testset "Automatic online and post-hoc total WHMC fits" begin
    rp = adaptive_centering_problem(sb,problem,AutoEnzyme();centeredness=0.)
    _,initial = WarmupHMC._inverse_with_logabsdet_jacobian(WarmupHMC.reparametrizer(rp),physical)
    fit = adaptive_warmup_mcmc(Xoshiro(27),rp;init=initial,n_draws=2000,
        nonlinear_adapt=true,monitor_ess=false,
        callback=(state,stage)->begin
            println("TOTAL_WHMC_BOUNDARY ",stage," window=",state.outer_counter);flush(stdout);false
        end)
    @test size(fit.posterior_position,2) >= 2000
    @test all(isfinite,fit.posterior_position)
    @test fit.n_divergent_samples == 0
    @test all(0<=last(p).c<=1 for p in WarmupHMC.reparam_sources(rp))
    @test abs(mean(fit.posterior_position[coords.totals[1,1],:])-0.5) < 0.4
    println("TOTAL_ONLINE_COMPLETE sampling_gradients=",fit.sampling_evaluation_counter)
    pilot=permutedims(fit.posterior_position)
    gradients=permutedims(hcat([last(LogDensityProblems.logdensity_and_gradient(problem,collect(row))) for row in eachrow(pilot)]...))
    for criterion in (:position,:gradient)
        selected=select_total_centeredness(sb,pilot,names;criterion,gradients)
        fixed=adaptive_centering_problem(sb,problem,AutoEnzyme();centeredness=selected.centeredness)
        _,start=WarmupHMC._inverse_with_logabsdet_jacobian(WarmupHMC.reparametrizer(fixed),physical)
        refit=adaptive_warmup_mcmc(Xoshiro(29),fixed;init=start,n_draws=2000,
            nonlinear_adapt=false,monitor_ess=false)
        @test size(refit.posterior_position,2)>=2000
        @test all(isfinite,refit.posterior_position)
        @test refit.n_divergent_samples==0
        @test abs(mean(refit.posterior_position[coords.totals[1,1],:])-0.5)<0.4
        println("TOTAL_POSTHOC_COMPLETE criterion=",criterion," sampling_gradients=",refit.sampling_evaluation_counter);flush(stdout)
    end
end
