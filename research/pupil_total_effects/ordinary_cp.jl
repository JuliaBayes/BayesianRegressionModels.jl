include("brms_baseline.jl")

function cp_common(p,q,data)
    names=BridgeStan.param_names(p.model;include_tp=true)
    idx(name)=only(findall(==(name),names))
    x=BridgeStan.param_constrain(p.model,Vector{Float64}(q);include_tp=true)
    A=x[idx("Intercept")]-data.xbar*x[idx("b.1")].+x[idx.(["r_1_1.$j" for j in 1:20])]
    B=x[idx("b.1")].+x[idx.(["r_1_2.$j" for j in 1:20])]
    vcat(log.(x[idx.(["sd_1.1","sd_1.2"])]),x[idx.(["Intercept_sigma","b_sigma.1"])],A,B)
end

function cp_initial(p,l,data)
    physical=PTE.initial_position(data)
    x=BridgeStan.param_constrain(p.model,brms_initial(data,l,p))
    # Query the branch's actual population-location convention; center=TRUE
    # uses total slopes but leaves the random intercept near zero-centered.
    names=BridgeStan.param_names(p.model;include_tp=true)
    generated=BridgeStan.param_constrain(p.model,BridgeStan.param_unconstrain(p.model,x);include_tp=true)
    shifts=generated[[only(findall(==("mean_center_re_1.$k"),names)) for k in 1:2]]
    x[l.za]=physical[5:24].-mean(physical[5:24]).+shifts[1]
    x[l.zb]=physical[25:44].-mean(physical[25:44]).+shifts[2]
    q=BridgeStan.param_unconstrain(p.model,x)
    @test cp_common(p,q,data) ≈ physical
    q
end

function cp_audit(p,l,data)
    raw=PTE.PupilProblem(data)
    function reference(q)
        physical=cp_common(p,q,data)
        x=BridgeStan.param_constrain(p.model,Vector{Float64}(q))
        beta=x[[l.beta0,l.beta1]]
        D=Diagonal(exp.(2physical[1:2]))
        T=[1.0 -data.xbar;0.0 1.0]
        precision=Diagonal([inv(PTE.INTERCEPT_SD^2),0.0])+20T'*(D\T)
        covariance=inv(Symmetric(precision))
        natural=[PTE.INTERCEPT_MEAN/PTE.INTERCEPT_SD^2,0.0]+
            T'*(D\[sum(physical[5:24]),sum(physical[25:44])])
        first(PTE.evaluate(raw,physical))+logpdf(MvNormal(covariance*natural,covariance),beta)+
            logpdf(LocationScale(PTE.INTERCEPT_MEAN,PTE.INTERCEPT_SD,TDist(3)),beta[1])-
            logpdf(Normal(PTE.INTERCEPT_MEAN,PTE.INTERCEPT_SD),beta[1])
    end
    rng=Xoshiro(815)
    rows=NamedTuple[]
    for k in 1:8
        x=BridgeStan.param_constrain(p.model,cp_initial(p,l,data))
        x[l.beta0]+=50randn(rng);x[l.beta1]+=5randn(rng)
        x[l.za].+=50randn(rng,20);x[l.zb].+=5randn(rng,20)
        x[[l.taua,l.taub]].*=exp.(.1randn(rng,2))
        q=BridgeStan.param_unconstrain(p.model,x)
        lp,g=LogDensityProblems.logdensity_and_gradient(p,q)
        density_error=abs(lp-reference(q))
        gradient_error=maximum(abs.(g-numerical_gradient(reference,q))./(1 .+abs.(g)))
        @test density_error<2e-7
        @test gradient_error<5e-5
        push!(rows,(;point=k,density_error,gradient_error))
    end
    rows
end

function run_ordinary_cp(input,output)
    ispath(output) && error("Preserve previous output; choose a new directory")
    mkpath(output);BLAS.set_num_threads(1)
    for name in ("ordinary_cp.stan","data.json","brms-source-sha.txt")
        cp(joinpath(input,name),joinpath(output,name))
    end
    mkpath(joinpath(output,"source"))
    for name in ("model.jl","audit.jl","run.jl","stan_target.jl","brms_baseline.jl","ordinary_cp.jl")
        cp(joinpath(@__DIR__,name),joinpath(output,"source",name))
    end
    cp(joinpath(@__DIR__,"reference"),joinpath(output,"source","reference"))
    stan=BridgeStan.StanModel(joinpath(output,"ordinary_cp.stan"),read(joinpath(input,"data.json"),String))
    p=BrmsPupilProblem(stan,Ref(0);reject_numerical_errors=true)
    l=brms_layout(p);data=PTE.load_data()
    write_tsv(joinpath(output,"equivalence_audit.tsv"),cp_audit(p,l,data))
    @test p.numerical_rejections[]==0
    initial=cp_initial(p,l,data)
    p.gradient_calls[]=0
    callback=(state,stage)->begin
        println("boundary\tordinary_cp\t",stage,"\twindow=",state.outer_counter,"\tall_gradients=",p.gradient_calls[])
        flush(stdout);isfile(joinpath(output,"STOP"))
    end
    timed=@timed adaptive_warmup_mcmc(Xoshiro(1),p;init=initial,n_draws=2000,
        monitor_ess=true,nonlinear_adapt=false,callback,checkpoint_dir=joinpath(output,"checkpoints"))
    fit=timed.value;source=Matrix{Float64}(fit.posterior_position)
    @test size(source,2)>=2000
    checkpoint=deserialize(joinpath(output,"checkpoints","cp_latest.jls"))
    @test checkpoint.posterior_position ≈ source
    @test checkpoint.sampling_evaluation_counter==fit.sampling_evaluation_counter
    positions=reduce(hcat,(cp_common(p,q,data) for q in eachcol(source)))
    named=reduce(hcat,(BridgeStan.param_constrain(stan,Vector{Float64}(q)) for q in eachcol(source)))
    record=(;positions,original_positions=source,original_named_positions=named,original_named_names=l.names,
        divergences=fit.n_divergent_samples,sampling_gradients=fit.sampling_evaluation_counter,
        transition_gradients=fit.total_evaluation_counter,all_gradient_calls=p.gradient_calls[],
        numerical_rejections=p.numerical_rejections[],first_numerical_error=p.first_numerical_error[],
        fit_seconds=timed.time,compile_seconds=NaN,seed=1,
        model_frame="brms center=TRUE, without S2Z, plus physical totals")
    @test record.all_gradient_calls>=record.transition_gradients>=record.sampling_gradients>0
    serialize(joinpath(output,"brms_cp.jls"),record)
    rows,summary=diagnostic_rows("brms_cp_whmc",record,data)
    write_tsv(joinpath(output,"common_parameters.tsv"),rows)
    write_tsv(joinpath(output,"diagnostics.tsv"),[summary])
    println("ORDINARY_CP_RESULT\t",summary)
    println("ORDINARY_CP_COMPLETE\t",output)
end

if abspath(PROGRAM_FILE)==@__FILE__
    run_ordinary_cp(ARGS...)
end
