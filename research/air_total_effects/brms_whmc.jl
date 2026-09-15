include("run.jl")
include("s2z.jl")
using DelimitedFiles

function costs(dir)
    values,header=readdlm(joinpath(dir,"gradient_counts.tsv"),'\t',Float64;header=true)
    Dict(String(k)=>Int(values[1,i]) for (i,k) in enumerate(vec(header)))
end

function target_model(d,label,native)
    if label=="s2z_auto"
        dir=joinpath(native,label)
        return BridgeStan.StanModel(joinpath(dir,"clean.stan"),joinpath(dir,"resolved-data.json");warn=false),
            JSON.parsefile(joinpath(dir,"resolved-data.json")),costs(dir)["precursor_gradient_calls"]
    end
    BridgeStan.StanModel(joinpath(d.dir,label*".stan"),joinpath(d.dir,label*".json");warn=false),
        JSON.parsefile(joinpath(d.dir,label*".json")),0
end

function fit_s2z(d,m,label,out,native)
    target,data,pilot=target_model(d,label,native);l=s2z_layout(d,target,label,data)
    audit_s2z(d,m,target,l,out,label)
    path=joinpath(out,label*".jls")
    fit=if isfile(path)
        deserialize(path)
    else
        raw=AIRTotals.BrmsPupilProblem(target,Ref(0);reject_numerical_errors=true)
        init=BridgeStan.param_unconstrain(target,s2z_from_total(d,m,l,AIRTotals.total_initial(d,m)))
        callback=(state,stage)->begin
            println("BOUNDARY ",label," ",stage," window=",state.outer_counter," gradients=",raw.gradient_calls[])
            flush(stdout);isfile(joinpath(out,"STOP"))
        end
        result=adaptive_warmup_mcmc(Xoshiro(1),raw;init,n_draws=2000,monitor_ess=true,
            nonlinear_adapt=false,checkpoint_dir=joinpath(out,label*"-checkpoints"),callback)
        @assert size(result.posterior_position,2)==2000
        value=(;positions=Matrix(result.posterior_position),sampling_gradients=result.sampling_evaluation_counter,
            all_gradient_calls=raw.gradient_calls[],total_gradient_calls=raw.gradient_calls[]+pilot,
            pilot_gradient_calls=pilot,divergences=result.n_divergent_samples)
        serialize(path,value);value
    end
    totals=hcat([first(s2z_to_total(d,m,l,BridgeStan.param_constrain(target,collect(q)))) for q in eachcol(fit.positions)]...)
    qois=total_qois(d,m,totals)
    serialize(joinpath(out,label*"-totals.jls"),totals);serialize(joinpath(out,label*"-qois.jls"),qois)
    diagnostics(d,label*"_whmc",qois,fit,out)
end

function analyze_native(d,m,label,out,native)
    target,data,_=target_model(d,label,native);dir=joinpath(native,label)
    values,header=readdlm(joinpath(dir,"sampling.tsv"),'\t',Float64;header=true)
    names=String.(vec(header));indices=[only(findall(==(n),names)) for n in BridgeStan.param_names(target)]
    constrained=permutedims(values[:,indices])
    qois=if label=="ordinary_ncp"
        positions=hcat([BridgeStan.param_unconstrain(target,collect(x)) for x in eachcol(constrained)]...)
        ordinary_qois(d,target,positions)
    else
        l=s2z_layout(d,target,label,data);audit_s2z(d,m,target,l,out,label)
        positions=hcat([first(s2z_to_total(d,m,l,x)) for x in eachcol(constrained)]...)
        total_qois(d,m,positions)
    end
    c=costs(dir)
    fit=(;sampling_gradients=c["sampling_gradients"],total_gradient_calls=c["workflow_gradient_calls"],divergences=c["divergences"])
    serialize(joinpath(out,label*"_native-qois.jls"),qois)
    diagnostics(d,label*"_native",qois,fit,out)
end

function main(mode,grouping,hierarchy,out,audit_dir,native,labels...)
    @assert JSON.parsefile(joinpath(audit_dir,"audit.json"))["status"]=="passed"
    mkpath(out);BLAS.set_num_threads(1)
    d=AIRTotals.load_data(grouping,hierarchy);m=AIRTotals.model(d,joinpath(out,"model"))
    for label in labels
        if mode=="fit";fit_s2z(d,m,label,out,native)
        elseif mode=="analyze";analyze_native(d,m,label,out,native)
        else;error("Unknown mode")
        end
    end
    println("AIR_BRMS_COMPLETE ",mode);flush(stdout)
end
if abspath(PROGRAM_FILE)==@__FILE__
    main(ARGS...)
end
