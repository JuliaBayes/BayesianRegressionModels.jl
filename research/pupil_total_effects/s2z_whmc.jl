include("recovery.jl")
include("stan_target.jl")
using DelimitedFiles, JSON

function helmert(J)
    Q = zeros(J,J-1)
    for k in 1:J-1
        Q[1:k,k] .= inv(sqrt(k*(k+1)))
        Q[k+1,k] = -k/sqrt(k*(k+1))
    end
    @test Q'Q ≈ I
    @test maximum(abs.(Q'ones(J))) < 1e-14
    Q
end

function s2z_layout(names,label,standata)
    idx(name)=only(findall(==(name),names))
    rho = label=="s2z_auto" ? permutedims(reduce(hcat,standata["rho_s2z_1"])) :
          fill(label=="s2z_cp" ? 1.0 : 0.0,20,2)
    @test size(rho)==(20,2)
    (;names,theta=idx.(["theta_s2z.1","theta_s2z.2"]),
      gamma=idx.(["Intercept_sigma","b_sigma.1"]),
      tau=idx.(["sd_1.1","sd_1.2"]),z=idx.(["z_s2z_1.$k" for k in 1:38]),
      u=idx("udf_b_s2z_1"),rho,Q=helmert(20))
end

"""Constrained native S2Z parameters -> physical totals and log mixture precision.

Written independently of generated Stan's sum-to-zero/recovery functions.
The determinant includes the mean/contrast-to-total volume factor J.
"""
function s2z_to_total(x,l,data)
    tau=x[l.tau]
    contrasts=zeros(20,2)
    logjac=log(20.0)
    for k in 1:2
        scale=1 .- l.rho[:,k] .+ l.rho[:,k].*tau[k]
        w=tau[k].*(l.Q*x[l.z[(k-1)*19+1:k*19]])./scale
        contrasts[:,k]=w.-mean(w)
        logjac += 19log(tau[k])-sum(log.(scale))+log(mean(scale))
    end
    theta=x[l.theta]
    q=vcat(log.(tau),x[l.gamma],
           theta[1]-data.xbar*theta[2].+contrasts[:,1],
           theta[2].+contrasts[:,2],-log(3.0)-log(x[l.u]))
    q,logjac
end

function s2z_from_total(q,l,data)
    x=zeros(length(l.names))
    tau=exp.(q[1:2])
    A,B=q[5:24],q[25:44]
    x[l.theta]=[mean(A)+data.xbar*mean(B),mean(B)]
    x[l.tau]=tau
    x[l.gamma]=q[3:4]
    x[l.u]=exp(-q[45])/3
    for (k,values) in enumerate((A,B))
        r=values.-mean(values)
        scale=1 .- l.rho[:,k] .+ l.rho[:,k].*tau[k]
        shift=-sum(r.*scale)/sum(scale)
        w=(r.+shift).*scale./tau[k]
        @test abs(sum(w)) < 1e-7
        x[l.z[(k-1)*19+1:k*19]]=l.Q'w
    end
    x
end

function audit_s2z(p,l,data,output)
    manual=PTE.PupilProblem(data,PTE.StudentMixtureMean())
    rng=Xoshiro(971)
    rows=NamedTuple[]
    allnames=BridgeStan.param_names(p.model;include_tp=true,include_gq=true)
    idx(name)=only(findall(==(name),allnames))
    stanrng=BridgeStan.StanRNG(p.model,771)
    reference(q)=begin
        physical,jac=s2z_to_total(BridgeStan.param_constrain(p.model,Vector{Float64}(q)),l,data)
        first(PTE.evaluate(manual,physical))+jac
    end
    for k in 1:8
        physical=PTE.initial_position(data,PTE.StudentMixtureMean())
        physical[1:2] .+= 0.1randn(rng,2)
        physical[3] += .02randn(rng)
        physical[4] += .001randn(rng)
        physical[5:24] .+= 50randn(rng,20)
        physical[25:44] .+= 5randn(rng,20)
        physical[45]=.2randn(rng)
        named=s2z_from_total(physical,l,data)
        q=BridgeStan.param_unconstrain(p.model,named)
        @test first(s2z_to_total(named,l,data)) ≈ physical
        @test BridgeStan.param_constrain(p.model,q) ≈ named
        lp,g=LogDensityProblems.logdensity_and_gradient(p,q)
        fd=numerical_gradient(reference,q)
        density_error=abs(lp-reference(q))
        gradient_error=maximum(abs.(g-fd)./(1 .+abs.(g)))
        @test density_error < 2e-7
        @test gradient_error < 5e-5
        generated=BridgeStan.param_constrain(p.model,q;include_tp=true,include_gq=true,rng=stanrng)
        A=generated[idx("theta_s2z.1")]-data.xbar*generated[idx("theta_s2z.2")].+
          generated[idx.(["r_s2z_1_1.$j" for j in 1:20])]
        B=generated[idx("theta_s2z.2")].+generated[idx.(["r_s2z_1_2.$j" for j in 1:20])]
        @test vcat(A,B) ≈ physical[5:44]
        A_recovered=generated[idx("Intercept")]-data.xbar*generated[idx("b.1")].+
          generated[idx.(["r_1_1.$j" for j in 1:20])]
        B_recovered=generated[idx("b.1")].+generated[idx.(["r_1_2.$j" for j in 1:20])]
        @test vcat(A_recovered,B_recovered) ≈ physical[5:44]
        push!(rows,(;point=k,density_error,gradient_error))
    end
    write_tsv(joinpath(output,"equivalence_audit.tsv"),rows)
    println("S2Z_AUDIT_PASS\t",output);flush(stdout)
end

function native_costs(path)
    row,head=readdlm(path,'\t',Float64;header=true)
    Dict(String(key)=>Int(row[1,k]) for (k,key) in enumerate(vec(head)))
end

function summarize_s2z(label,record,generated,names,data,output)
    parameters,summary=diagnostic_rows(label,record,data)
    write_tsv(joinpath(output,"common_parameters.tsv"),parameters[1:44])
    write_tsv(joinpath(output,"diagnostics.tsv"),[summary])
    idx(name)=only(findall(==(name),names))
    recovered=reduce(hcat,(original_position(record.positions[:,s],
        generated[idx.(["Intercept","b.1"]),s],data) for s in axes(generated,2)))
    @test recovered[7:26,:] ≈ generated[idx.(["r_1_1.$j" for j in 1:20]),:]
    @test recovered[27:46,:] ≈ generated[idx.(["r_1_2.$j" for j in 1:20]),:]
    write_tsv(joinpath(output,"original_scope_diagnostics.tsv"),
        summarize_scope(label,get(record,:recovery_seed,1),recovered,record,data))
    source_bulk=diagnostics(record.source_named_positions).bulk
    write_tsv(joinpath(output,"source_scope_diagnostics.tsv"),[(;
        scope="native_sampled_parameters45",min_bulk_ess=minimum(source_bulk),
        limiting_parameter=record.source_named_names[argmin(source_bulk)],
        min_bulk_ess_per_1000_sampling_gradients=1000minimum(source_bulk)/record.sampling_gradients)])
    serialize(joinpath(output,"fit.jls"),merge(record,(;
        generated_positions=generated,generated_names=names,original_physical_positions=recovered)))
    println("S2Z_RESULT\t",summary);flush(stdout)
end

function analyze_native_s2z(input,p,l,data,output)
    raw,head=readdlm(joinpath(input,"sampling.tsv"),'\t',Float64;header=true)
    names=String.(vec(head))
    idx(name)=only(findall(==(name),names))
    named=permutedims(raw[:,idx.(l.names)])
    positions=reduce(hcat,(first(s2z_to_total(q,l,data)) for q in eachcol(named)))
    costs=native_costs(joinpath(input,"gradient_counts.tsv"))
    record=(;positions,source_named_positions=named,source_named_names=l.names,
        mean_prior=PTE.StudentMixtureMean(),divergences=costs["divergences"],
        sampling_gradients=costs["sampling_gradients"],
        all_gradient_calls=costs["all_gradient_calls"],transition_gradients=costs["all_gradient_calls"],
        precursor_gradient_calls=costs["precursor_gradient_calls"],
        workflow_gradient_calls=costs["workflow_gradient_calls"],
        fit_seconds=NaN,compile_seconds=NaN,seed=1,
        sampler="native Stan NUTS",model_frame="total coefficients plus log mixture precision")
    mkpath(output)
    summarize_s2z(basename(input)*"_native",record,permutedims(raw),names,data,output)
end

function run_s2z(input,output)
    ispath(output) && error("Preserve previous output; choose a new directory")
    mkpath(output)
    BLAS.set_num_threads(1)
    label=basename(input)
    for file in ("clean.stan","resolved-data.json","init.json","provenance.json")
        cp(joinpath(input,file),joinpath(output,file))
    end
    mkpath(joinpath(output,"source"))
    for file in ("model.jl","audit.jl","run.jl","recovery.jl","stan_target.jl","s2z_whmc.jl")
        cp(joinpath(@__DIR__,file),joinpath(output,"source",file))
    end
    cp(joinpath(@__DIR__,"reference"),joinpath(output,"source","reference"))
    stan=BridgeStan.StanModel(joinpath(output,"clean.stan"),read(joinpath(input,"resolved-data.json"),String))
    p=BrmsPupilProblem(stan,Ref(0);reject_numerical_errors=true)
    l=s2z_layout(BridgeStan.param_names(stan),label,JSON.parsefile(joinpath(input,"resolved-data.json")))
    data=PTE.load_data()
    @test LogDensityProblems.dimension(p)==45
    audit_s2z(p,l,data,output)
    @test p.numerical_rejections[]==0
    analyze_native_s2z(input,p,l,data,joinpath(output,"native_analysis"))
    initial_dict=JSON.parsefile(joinpath(input,"init.json"))
    # The first R capsule used auto_unbox, which serialized the one-element
    # b_sigma vector as a scalar. The R fit itself received the proper list.
    # Restore that declared shape without changing the supplied numeric value.
    if initial_dict["b_sigma"] isa Number
        initial_dict["b_sigma"]=[initial_dict["b_sigma"]]
    end
    initial_json=JSON.json(initial_dict)
    write(joinpath(output,"stan-init.json"),initial_json)
    initial=BridgeStan.param_unconstrain_json(stan,initial_json)
    p.gradient_calls[]=0
    callback=(state,stage)->begin
        println("boundary\t",label,"\t",stage,"\twindow=",state.outer_counter,
                "\tall_gradients=",p.gradient_calls[]);flush(stdout)
        isfile(joinpath(output,"STOP"))
    end
    Base.cumulative_compile_timing(true)
    before=Base.cumulative_compile_time_ns()
    timed=try
        @timed adaptive_warmup_mcmc(Xoshiro(1),p;init=initial,n_draws=2000,
            monitor_ess=true,nonlinear_adapt=false,callback,checkpoint_dir=joinpath(output,"checkpoints"))
    finally
        Base.cumulative_compile_timing(false)
    end
    compilation=Base.cumulative_compile_time_ns().-before
    fit=timed.value
    source=Matrix{Float64}(fit.posterior_position)
    @test size(source,2)>=2000
    cp_final=deserialize(joinpath(output,"checkpoints","cp_latest.jls"))
    @test cp_final.posterior_position ≈ source
    @test cp_final.sampling_evaluation_counter==fit.sampling_evaluation_counter
    @test p.gradient_calls[]>=fit.total_evaluation_counter>=fit.sampling_evaluation_counter>0
    named=reduce(hcat,(BridgeStan.param_constrain(stan,Vector{Float64}(q)) for q in eachcol(source)))
    positions=reduce(hcat,(first(s2z_to_total(q,l,data)) for q in eachcol(named)))
    rng=BridgeStan.StanRNG(stan,101)
    generated=reduce(hcat,(BridgeStan.param_constrain(stan,Vector{Float64}(q);
        include_tp=true,include_gq=true,rng) for q in eachcol(source)))
    generated_names=BridgeStan.param_names(stan;include_tp=true,include_gq=true)
    precursor=native_costs(joinpath(input,"gradient_counts.tsv"))["precursor_gradient_calls"]
    record=(;positions,source_positions=source,source_named_positions=named,source_named_names=l.names,
        mean_prior=PTE.StudentMixtureMean(),divergences=fit.n_divergent_samples,
        sampling_gradients=fit.sampling_evaluation_counter,transition_gradients=fit.total_evaluation_counter,
        all_gradient_calls=p.gradient_calls[],precursor_gradient_calls=precursor,
        workflow_gradient_calls=p.gradient_calls[]+precursor,
        numerical_rejections=p.numerical_rejections[],first_numerical_error=p.first_numerical_error[],
        fit_seconds=timed.time,compile_seconds=first(compilation)/1e9,seed=1,recovery_seed=101,
        sampler="WarmupHMC nonlinear_adapt=false",model_frame="total coefficients plus log mixture precision")
    summarize_s2z(label*"_whmc",record,generated,generated_names,data,output)
    println("S2Z_WHMC_COMPLETE\t",output);flush(stdout)
end

if abspath(PROGRAM_FILE)==@__FILE__
    run_s2z(ARGS...)
end
