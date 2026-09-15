include("common.jl")
include("s2z.jl")
using Test, DelimitedFiles

function ordinary_start(model,label)
    q=BridgeStan.param_unconstrain_json(model,JSON.json(ordinary_init()))
    if label=="ordinary_cp"
        tn=BridgeStan.param_names(model;include_tp=true)
        t=BridgeStan.param_constrain(model,q;include_tp=true)
        names=BridgeStan.param_names(model);x=BridgeStan.param_constrain(model,q)
        for (group,margins) in ((1,2),(2,1)), k in 1:margins
            meanval=t[only(findall(==("mean_center_re_$group.$k"),tn))]
            for j in 1:J;x[only(findall(==("z_$group.$k.$j"),names))]=meanval;end
        end
        q=BridgeStan.param_unconstrain(model,x)
    end
    @assert ordinary_physical(model,q).totals≈ols_initial().totals
    q
end

function audit_ordinary(model,label,out)
    reference(q)=begin
        v=ordinary_physical(model,q)
        ordinary_reference(model,q)-(label=="ordinary_cp" ? J*sum(log.(v.tau)) : 0.)
    end
    q=ordinary_start(model,label)
    for trial in 0:3
        x=q.+(trial==0 ? zeros(length(q)) : 0.01max.(1.,abs.(q)).*randn(Xoshiro(trial),length(q)))
        lp,g=BridgeStan.log_density_gradient(model,x;propto=false)
        @assert abs(lp-reference(x))<2e-7
        @assert maximum(abs.(g-finite_gradient(reference,x))./max.(1.,abs.(g)))<5e-5
    end
    println("ORDINARY_AUDIT_PASS ",label);flush(stdout)
    q
end

function native_costs(dir)
    v,h=readdlm(joinpath(dir,"gradient_counts.tsv"),'\t',Float64;header=true)
    Dict(String(k)=>Int(v[1,i]) for (i,k) in enumerate(vec(h)))
end

function arm_model(label,native_root)
    if label=="s2z_auto"
        dir=joinpath(native_root,label)
        return BridgeStan.StanModel(joinpath(dir,"clean.stan"),joinpath(dir,"resolved-data.json");warn=false),
               JSON.parsefile(joinpath(dir,"resolved-data.json")),native_costs(dir)["precursor_gradient_calls"]
    end
    stan_model(label),JSON.parsefile(joinpath(REFERENCE,label*".json")),0
end

function run_brms(label,tm,out,native_root)
    model,data,pilot_cost=arm_model(label,native_root)
    raw=BrmsPupilProblem(model,Ref(0);reject_numerical_errors=true)
    is_s2z=startswith(label,"s2z")
    layout=is_s2z ? s2z_layout(model,label,data) : nothing
    if is_s2z
        audit_s2z(model,layout,tm,label,out)
        init=BridgeStan.param_unconstrain(model,s2z_from_total(total_initial(tm),layout,tm))
    else
        init=audit_ordinary(model,label,out)
    end
    checkpoint_dir=joinpath(out,label*"-checkpoints")
    callback=(state,stage)->begin
        println("BOUNDARY ",label," ",stage," window=",state.outer_counter," gradients=",raw.gradient_calls[])
        flush(stdout);isfile(joinpath(out,"STOP"))
    end
    raw_path=joinpath(out,label*"-raw.jls")
    if isfile(raw_path)
        base=deserialize(raw_path)
    else
        raw.gradient_calls[]=0
        timed=@timed adaptive_warmup_mcmc(Xoshiro(1),raw;init,n_draws=2000,
            nonlinear_adapt=false,monitor_ess=true,checkpoint_dir,callback)
        fit=timed.value;calls=raw.gradient_calls[]
        @assert size(fit.posterior_position,2)>=2000
        @assert calls>=fit.total_evaluation_counter>=fit.sampling_evaluation_counter>0
        base=(;positions=Matrix(fit.posterior_position),sampling_gradients=fit.sampling_evaluation_counter,
            all_gradient_calls=calls,total_gradient_calls=calls+pilot_cost,pilot_gradient_calls=pilot_cost,
            divergences=fit.n_divergent_samples,fit_seconds=timed.time,numerical_rejections=raw.numerical_rejections[])
        serialize(raw_path,base) # Preserve completed fitting before any postprocessing.
    end
    positions=base.positions
    total_positions=is_s2z ? hcat([first(s2z_to_total(BridgeStan.param_constrain(model,collect(q)),layout,tm))
        for q in eachcol(positions)]...) : nothing
    qois=is_s2z ? total_qois(tm,total_positions) : ordinary_qois(model,positions)
    record=(;base...,total_positions,names=BridgeStan.param_unc_names(model))
    serialize(joinpath(out,label*".jls"),record)
    serialize(joinpath(out,label*"-qois.jls"),qois)
    scientific_diagnostics(label*"_whmc",qois,record.sampling_gradients,record.total_gradient_calls,record.divergences,out)
end

function analyze_native(label,tm,input,out)
    model,data,_=arm_model(label,dirname(input))
    v,h=readdlm(joinpath(input,"sampling.tsv"),'\t',Float64;header=true)
    names=String.(vec(h));indices=[only(findall(==(n),names)) for n in BridgeStan.param_names(model)]
    constrained=permutedims(v[:,indices])
    if startswith(label,"s2z")
        layout=s2z_layout(model,label,data)
        audit_s2z(model,layout,tm,label,out)
        positions=hcat([first(s2z_to_total(x,layout,tm)) for x in eachcol(constrained)]...)
        qois=total_qois(tm,positions)
    else
        positions=hcat([BridgeStan.param_unconstrain(model,collect(x)) for x in eachcol(constrained)]...)
        qois=ordinary_qois(model,positions)
    end
    c=native_costs(input)
    serialize(joinpath(out,label*"_native-positions.jls"),positions)
    serialize(joinpath(out,label*"_native-qois.jls"),qois)
    scientific_diagnostics(label*"_native",qois,c["sampling_gradients"],c["workflow_gradient_calls"],c["divergences"],out)
end

function main(mode,out,audit_dir,native_root,labels...)
    @assert JSON.parsefile(joinpath(audit_dir,"audit.json"))["status"]=="passed"
    mkpath(out);BLAS.set_num_threads(1)
    tm=total_model(joinpath(out,"model"))
    for label in labels
        if mode=="fit";run_brms(label,tm,out,native_root)
        elseif mode=="analyze";analyze_native(label,tm,joinpath(native_root,label),out)
        elseif mode=="audit"
            model,data,_=arm_model(label,native_root)
            startswith(label,"s2z") ? audit_s2z(model,s2z_layout(model,label,data),tm,label,out) : audit_ordinary(model,label,out)
        else;error("Unknown mode $mode")
        end
    end
    println("PUPIL4_BRMS_COMPLETE ",mode);flush(stdout)
end
if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS...)
end
