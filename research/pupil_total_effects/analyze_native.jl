include("recovery.jl")
using DelimitedFiles

function analyze_native(output,baseline_dir)
    raw,head = readdlm(joinpath(output,"sampling.tsv"),'\t',Float64;header=true)
    names = String.(vec(head))
    idx(name) = only(findall(==(name),names))
    # Use the named constrained serialization, avoiding assumptions about
    # the native unconstrained array-of-vectors layout.
    baseline = deserialize(joinpath(baseline_dir,"brms_ncp.jls"))
    named = permutedims(raw[:,idx.(baseline.original_named_names)])
    data = PTE.load_data()
    beta0,beta1 = raw[:,idx("Intercept")],raw[:,idx("b.1")]
    taua,taub = raw[:,idx("sd_1.1")],raw[:,idx("sd_1.2")]
    A = [beta0[s]-data.xbar*beta1[s]+taua[s]*raw[s,idx("z_1.1.$j")]
         for j in 1:20,s in 1:2000]
    B = [beta1[s]+taub[s]*raw[s,idx("z_1.2.$j")] for j in 1:20,s in 1:2000]
    common = vcat(log.(taua)',log.(taub)',raw[:,idx("Intercept_sigma")]',
                  raw[:,idx("b_sigma.1")]',A,B)
    @test size(common)==(44,2000)
    @test all(isfinite,common)
    costs,keys = readdlm(joinpath(output,"gradient_counts.tsv"),'\t',Float64;header=true)
    cost(key) = Int(costs[1,only(findall(==(key),vec(keys)))])
    record = (;positions=common,original_named_positions=named,
        original_named_names=baseline.original_named_names,
        divergences=cost("divergences"),sampling_gradients=cost("sampling_gradients"),
        all_gradient_calls=cost("all_gradient_calls"),transition_gradients=cost("all_gradient_calls"),
        fit_seconds=NaN,compile_seconds=NaN,seed=1,
        sampler="native Stan NUTS",model_frame="common total coefficients and original named NCP")
    serialize(joinpath(output,"native_ncp.jls"),record)
    parameters,summary = diagnostic_rows("brms_native_ncp",record,data)
    write_tsv(joinpath(output,"common_parameters.tsv"),parameters)
    write_tsv(joinpath(output,"diagnostics.tsv"),[summary])
    original = baseline_original(record,data)
    write_tsv(joinpath(output,"original_scope_diagnostics.tsv"),
              summarize_scope("brms_native_ncp",0,original,record,data))
    ds = diagnostics(original)
    write_tsv(joinpath(output,"original_physical_parameters.tsv"),[
        (;parameter=original_names(data)[j],bulk_ess=ds.bulk[j],mean_ess=ds.mean_ess[j],mcse=ds.mcse[j])
        for j in axes(original,1)])
    println("NATIVE_DIAGNOSTICS\t",summary)
end

if abspath(PROGRAM_FILE)==@__FILE__
    analyze_native(ARGS...)
end
