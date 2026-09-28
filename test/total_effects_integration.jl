using Test, Random, LinearAlgebra, Statistics, Distributions
using BayesianRegressionModels, StanBlocks, BridgeStan, LogDensityProblems
const BRM = BayesianRegressionModels

data = (;x=[0.,1.,0.,1.,2.,3.],g=[1,1,2,2,3,3],y=[0.,1.,2.,3.,1.,2.])
normal_builder = @brm begin
    mu ~ 1 + x + (1|a|g) + (0+x|b|g)
    effect(mu,Intercept) ~ Normal(0,5)
    effect(mu,x) ~ Normal(0,2)
    sd(:,a) ~ Cauchy(0,2)
    y ~ Normal(mu,1)
end
student_builder = @brm begin
    mu ~ 1 + center(x) + (1+x||g)
    effect(mu,Intercept) ~ LocationScale(2.,3.,TDist(3))
    effect(mu,center_x) ~ Flat()
    y ~ Normal(mu,1)
end

try
@testset "BRM total coefficients" begin
    for builder in (normal_builder,student_builder)
        brmi = builder(data)
        sb = SBBRMI(brmi;mod=@__MODULE__)
        block = only(total_effect_blocks(sb))
        @test block.group === :g
        @test block.columns == (:Intercept,:x)
        @test isempty(total_effect_blocks(SBBRMI(brmi;total_groups=())))
        @test isempty(total_effect_blocks(SBBRMI(brmi;centered_groups=[:g])))
        @test_throws ArgumentError SBBRMI(brmi;total_groups=:missing_group)
        code = BRM.stan_code(sb)
        @test StanBlocks.stanc_check(code;warn_pedantic=false).ok
        descriptor = brm_descriptor(sb)
        @test descriptor isa BRMDescriptor
        @test only(o for o in descriptor.outputs if o.name === block.population).role === :population_effect
        @test only(o for o in descriptor.outputs if o.name === block.population).labels == collect(block.population_columns)
        problem = StanBlocks.stan_instantiate(sb.model;path=joinpath(mktempdir(),"integrated.stan"))
        names = BridgeStan.param_unc_names(problem.model)
        println("PARAMETERS ", names)
        coords = BRM._total_coordinates(sb,block,names)
        @test length(names) == 8+length(block.mixture)
        q = zeros(length(names)); q[vec(coords.totals)] = collect(1.:6.)/3
        lp, grad = BridgeStan.log_density_gradient(problem.model,q;propto=false)
        @test isfinite(lp) && all(isfinite,grad)
        stanrng = BridgeStan.StanRNG(problem.model,917)
        gq = BridgeStan.param_constrain(problem.model,q;include_tp=true,include_gq=true,rng=stanrng)
        @test all(isfinite,gq)
        if builder === normal_builder
            conventional = SBBRMI(brmi;total_groups=())
            original = StanBlocks.stan_instantiate(conventional.model;
                path=joinpath(mktempdir(),"original.stan"))
            original_names = BridgeStan.param_unc_names(original.model)
            original_blocks = adaptive_centering_blocks(conventional,original_names)
            pop_indices = [only(findall(==("pop_mu_beta_pop.$k"),original_names)) for k in 1:2]
            function original_marginal(x)
                conditional = BRM._total_conditional(block,coords,x)
                beta = conditional.mean .+ [0.2,-0.1]
                original_q = zeros(length(original_names))
                original_q[pop_indices] = beta
                for (k,id) in enumerate((:a,:b))
                    original_block = only(b for b in original_blocks if b.ranef.id === id)
                    original_q[original_block.log_scales] = x[coords.scales[k:k]]
                    original_q[vec(original_block.effects)] =
                        (conditional.totals[:,k] .- (block.A*beta)[k])./conditional.tau[k]
                end
                BridgeStan.log_density(original.model,original_q;propto=false) -
                    3sum(log,conditional.tau) -
                    logpdf(MvNormal(conditional.mean,inv(conditional.factor)),beta)
            end
            for trial in 1:4
                x = q .+ 0.25randn(Xoshiro(32+trial),length(q))
                @test BridgeStan.log_density(problem.model,x;propto=false) ≈ original_marginal(x) atol=1e-10
            end
            prior = SBBRMI(builder((;x=data.x,g=data.g)))
            @test StanBlocks.stanc_check(BRM.stan_code(prior);warn_pedantic=false).ok
        end
        replay = reprocess(sb,(;data...,x=data.x.+0.7))
        @test replay.data[block.group_index] == sb.data[block.group_index]
        @test BRM.stan_code(replay) == code
        @test_throws ArgumentError reprocess(sb,data;freeze_constants=false)
        recovered = recover_population_draws(sb,repeat(q',4000,1),names;rng=Xoshiro(913))[block.predictor]
        conditional = BRM._total_conditional(block,coords,q)
        @test vec(mean(recovered.population;dims=1)) ≈ conditional.mean atol=0.06
        @test cov(recovered.population) ≈ inv(conditional.factor) atol=0.06
        @test maximum(abs,recovered.totals .- recovered.deviations .-
            reshape(recovered.population*block.A',4000,1,2)) < 1e-12
        plan = generative_plan(builder,data)
        newdata = (;x=[9.,8.,7.,6.],g=[3,3,4,4],y=[0.,0.,0.,0.])
        newplan = generative_plan(plan,newdata)
        newblock = only(total_effect_blocks(newplan))
        @test newblock.A == block.A
        newnames = vcat(names[1:2+length(block.mixture)],
            ["$(block.binding).$g.$k" for k in 1:2 for g in 1:2])
        moved = transport_draws(plan,newplan,repeat(q',4000,1),names,newnames;rng=Xoshiro(915))
        newcoords = BRM._total_coordinates(newplan,newblock,newnames)
        @test all(moved[:,newcoords.totals[1,:]] .== transpose(q[coords.totals[3,:]]))
        @test vec(mean(moved[:,newcoords.totals[2,:]];dims=1)) ≈ block.A*conditional.mean atol=0.08
        pop = population_draws(sb,repeat(q',4,1),names;groups=:g,rng=Xoshiro(13))
        @test pop[:,coords.totals[1,:]] == pop[:,coords.totals[2,:]]
        pilot = repeat(q',40,1) .+ 0.3randn(Xoshiro(9),40,length(q))
        gs = permutedims(hcat([last(LogDensityProblems.logdensity_and_gradient(problem,collect(row))) for row in eachrow(pilot)]...))
        for criterion in (:position,:gradient)
            selected = select_total_centeredness(sb,pilot,names;criterion,gradients=gs)
            @test length(selected.centeredness) == 6
            @test selected.indices == vec(coords.totals)
            @test all(0 .<= selected.centeredness .<= 1)
        end
    end
end
@testset "Total eligibility and unabsorbed fixed effects" begin
    random_intercept = @brm begin
        mu ~ 1 + x + (1|a|g)
        y ~ Normal(mu,1)
    end
    sb = SBBRMI(random_intercept(data))
    block = only(total_effect_blocks(sb))
    @test block.population_columns == (:Intercept,)
    @test size(block.A) == (1,1)
    @test StanBlocks.stanc_check(BRM.stan_code(sb);warn_pedantic=false).ok
    d = brm_descriptor(sb)
    @test only(o for o in d.outputs if o.name === :pop_mu_beta_pop).labels == [:x]
    correlated = @brm begin
        mu ~ 1 + x + (1+x|g)
        y ~ Normal(mu,1)
    end
    crossed = @brm begin
        mu ~ 1 + (1|g) + (1|g2)
        y ~ Normal(mu,1)
    end
    unsupported = @brm begin
        mu ~ 1 + (1|g)
        effect(mu,Intercept) ~ Cauchy(0,1)
        y ~ Normal(mu,1)
    end
    for builder in (correlated,crossed,unsupported)
        brmi = builder((;data...,g2=[1,2,1,2,1,2]))
        @test isempty(total_effect_blocks(SBBRMI(brmi)))
        @test_throws ArgumentError SBBRMI(brmi;total_groups=:g)
    end
    mixed = @brm begin
        mu ~ 1 + (1|g)
        eta ~ 1 + x + (1+x|g2)
        y ~ Normal(mu + eta,1)
    end
    mixed_brmi = mixed((;data...,g2=[1,2,1,2,1,2]))
    @test isempty(total_effect_blocks(SBBRMI(mixed_brmi)))
    @test only(total_effect_blocks(SBBRMI(mixed_brmi;total_groups=:g))).group === :g
    changed_student = @brm begin
        mu ~ 1 + center(x) + (1+x||g)
        effect(mu,Intercept) ~ LocationScale(2.,3.,TDist(5))
        effect(mu,center_x) ~ Flat()
        y ~ Normal(mu,1)
    end
    source_plan = generative_plan(student_builder,data)
    changed_plan = generative_plan(changed_student,data)
    # Prior compatibility is checked before any coordinate values are used.
    @test_throws ArgumentError transport_draws(source_plan,changed_plan,
        zeros(1,0),String[],String[])
end
catch e
    showerror(stdout,e); println()
    for frame in stacktrace(catch_backtrace())
        occursin("total_effect",String(frame.file)) && println(frame.file,':',frame.line,' ',frame.func)
    end
    exit(1)
end
