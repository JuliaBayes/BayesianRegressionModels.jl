include("run.jl")
using BridgeStan

struct BrmsPupilProblem{M}
    model::M
    gradient_calls::Base.RefValue{Int}
end
LogDensityProblems.dimension(p::BrmsPupilProblem) = length(BridgeStan.param_unc_names(p.model))
LogDensityProblems.capabilities(::Type{<:BrmsPupilProblem}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(p::BrmsPupilProblem,q) = BridgeStan.log_density(
    p.model,convert(Vector{Float64},q);propto=false,jacobian=true)
function LogDensityProblems.logdensity_and_gradient(p::BrmsPupilProblem,q)
    p.gradient_calls[] += 1
    BridgeStan.log_density_gradient(p.model,convert(Vector{Float64},q);propto=false,jacobian=true)
end

function brms_layout(p)
    # Name the constrained serialization, and use native conversions for the
    # sampler vector. In this generated array-of-vectors model the reported
    # param_unc_names ordering does not match the actual sampler input layout.
    names = BridgeStan.param_names(p.model)
    idx(name) = only(findall(==(name),names))
    (; names, beta0=idx("Intercept"), beta1=idx("b.1"), gamma0=idx("Intercept_sigma"),
        gamma1=idx("b_sigma.1"), taua=idx("sd_1.1"), taub=idx("sd_1.2"),
        za=[idx("z_1.1.$j") for j in 1:20], zb=[idx("z_1.2.$j") for j in 1:20])
end

function brms_initial(data, layout, p)
    physical = PTE.initial_position(data)
    q = zeros(46)
    A, B = physical[5:24], physical[25:44]
    q[layout.beta1] = mean(B)
    q[layout.beta0] = mean(A) + data.xbar*mean(B)
    q[layout.gamma0],q[layout.gamma1] = physical[3:4]
    q[layout.taua],q[layout.taub] = exp.(physical[1:2])
    q[layout.za] = (A.-mean(A))./exp(physical[1])
    q[layout.zb] = (B.-mean(B))./exp(physical[2])
    BridgeStan.param_unconstrain(p.model,q)
end

function common_position(q, layout, data, p)
    x = BridgeStan.param_constrain(p.model,convert(Vector{Float64},q))
    A = x[layout.beta0] - data.xbar*x[layout.beta1] .+ x[layout.taua].*x[layout.za]
    B = x[layout.beta1] .+ x[layout.taub].*x[layout.zb]
    vcat(log(x[layout.taua]),log(x[layout.taub]),x[layout.gamma0],x[layout.gamma1],A,B)
end

function conditional_beta_logpdf(q, layout, data, p)
    physical = common_position(q,layout,data,p)
    D = Diagonal(exp.(2physical[1:2]))
    T = [1.0 -data.xbar; 0.0 1.0]
    precision = Diagonal([inv(PTE.INTERCEPT_SD^2),0.0]) + 20T'*(D\T)
    covariance = inv(Symmetric(precision))
    natural = [PTE.INTERCEPT_MEAN/PTE.INTERCEPT_SD^2,0.0] +
        T'*(D\[sum(physical[5:24]),sum(physical[25:44])])
    logpdf(MvNormal(covariance*natural,covariance),q[[layout.beta0,layout.beta1]])
end

function baseline_audit(p,layout,data;student=false)
    rng = Xoshiro(719)
    initial = brms_initial(data,layout,p)
    raw = PTE.PupilProblem(data)
    function reference(q)
        lp = first(PTE.evaluate(raw,common_position(q,layout,data,p))) +
            conditional_beta_logpdf(q,layout,data,p) + 20(q[layout.taua]+q[layout.taub])
        if student
            beta0 = BridgeStan.param_constrain(p.model,convert(Vector{Float64},q))[layout.beta0]
            lp += logpdf(LocationScale(PTE.INTERCEPT_MEAN,PTE.INTERCEPT_SD,TDist(3)),beta0) -
                  logpdf(Normal(PTE.INTERCEPT_MEAN,PTE.INTERCEPT_SD),beta0)
        end
        lp
    end
    rows = NamedTuple[]
    for k in 1:8
        q = copy(initial)
        q[layout.beta0] += 50randn(rng)
        q[layout.beta1] += 5randn(rng)
        q[layout.za] .+= 0.03randn(rng,20)
        q[layout.zb] .+= 0.03randn(rng,20)
        q[[layout.taua,layout.taub]] .+= 0.1randn(rng,2)
        lp,g = LogDensityProblems.logdensity_and_gradient(p,q)
        fd = numerical_gradient(reference,q)
        density_error = abs(lp-reference(q))
        gradient_error = maximum(abs.(g-fd)./(1 .+ abs.(g)))
        @test density_error < 2e-7
        @test gradient_error < 5e-5
        push!(rows,(;point=k,density_error,gradient_error))
    end
    rows
end

function main_baseline()
    output = ENV["PUPIL_BASELINE_OUTPUT"]
    ispath(output) && error("Preserve previous output; choose a new directory")
    mkpath(output)
    BLAS.set_num_threads(1)
    prior_kind = get(ENV,"PUPIL_INTERCEPT_PRIOR","gaussian")
    prior_kind in ("gaussian","student_mixture") || error("Unknown intercept prior")
    student = prior_kind == "student_mixture"
    source = joinpath(@__DIR__,"reference",student ?
        "pupil-uncorrelated-student.stan" : "pupil-uncorrelated-gaussian.stan")
    model_file = joinpath(output,"pupil-brms.stan")
    cp(source,model_file)
    data_file = joinpath(@__DIR__,"reference","standata.json")
    stan = BridgeStan.StanModel(model_file,read(data_file,String))
    p = BrmsPupilProblem(stan,Ref(0))
    layout = brms_layout(p)
    data = PTE.load_data()
    write_tsv(joinpath(output,"brms_equivalence_audit.tsv"),baseline_audit(p,layout,data;student))
    mkpath(joinpath(output,"source"))
    for name in ("model.jl","audit.jl","run.jl","brms_baseline.jl")
        cp(joinpath(@__DIR__,name),joinpath(output,"source",name))
    end
    cp(joinpath(@__DIR__,"reference"),joinpath(output,"source","reference"))
    packages = [(;name=x.name,version=string(x.version),source=x.source,
        tree_hash=string(x.tree_hash)) for x in values(Pkg.dependencies())]
    write_tsv(joinpath(output,"packages.tsv"),sort(packages;by=x->x.name))
    open(joinpath(output,"provenance.toml"),"w") do io
        TOML.print(io,Dict("target"=>"conventional brms NCP; independent random effects",
            "intercept_prior"=>student ? "Student-t(3,5651.9,2026.1)" : "Normal(5651.9,2026.1)",
            "stan_sha256"=>bytes2hex(sha256(read(source))),
            "data_sha256"=>PTE.DATA_SHA256,"julia_version"=>string(VERSION),
            "git_commit"=>strip(read(`git -C $(@__DIR__) rev-parse HEAD`,String)),
            "sampled_dimensions"=>46,"common_dimensions"=>44,
            "seed"=>1,"chains"=>1,"draw_floor"=>2000))
    end
    p.gradient_calls[] = 0
    callback = (state,stage) -> begin
        println("boundary\tbrms_ncp\t",stage,"\twindow=",state.outer_counter,
                "\tall_gradients=",p.gradient_calls[])
        flush(stdout)
        isfile(joinpath(output,"STOP"))
    end
    Base.cumulative_compile_timing(true)
    before = Base.cumulative_compile_time_ns()
    timed = try
        @timed adaptive_warmup_mcmc(Xoshiro(1),p; init=brms_initial(data,layout,p),
            n_draws=2000,monitor_ess=true,nonlinear_adapt=false,callback,
            checkpoint_dir=joinpath(output,"checkpoints"))
    finally
        Base.cumulative_compile_timing(false)
    end
    compilation = Base.cumulative_compile_time_ns() .- before
    fit = timed.value
    original = convert(Matrix{Float64},fit.posterior_position)
    size(original,2)>=2000 || error("Baseline stopped early")
    common = reduce(hcat,(common_position(q,layout,data,p) for q in eachcol(original)))
    named = reduce(hcat,(BridgeStan.param_constrain(p.model,convert(Vector{Float64},q)) for q in eachcol(original)))
    # Rank-based ESS is unchanged by the positive SD transform. The other
    # named coordinates retain their conventional NCP values.
    record = (;positions=common,original_positions=original,
        original_named_positions=named,original_named_names=layout.names,
        divergences=fit.n_divergent_samples,sampling_gradients=fit.sampling_evaluation_counter,
        transition_gradients=fit.total_evaluation_counter,all_gradient_calls=p.gradient_calls[],
        fit_seconds=timed.time,compile_seconds=first(compilation)/1e9,seed=1,
        model_frame="brms conventional NCP plus derived common total coefficients")
    @test record.all_gradient_calls >= record.transition_gradients >= record.sampling_gradients > 0
    cp_final = deserialize(joinpath(output,"checkpoints","cp_latest.jls"))
    @test cp_final.posterior_position ≈ original
    @test cp_final.sampling_evaluation_counter == record.sampling_gradients
    serialize(joinpath(output,"brms_ncp.jls"),record)
    rows,summary = diagnostic_rows("brms_ncp",record,data)
    write_tsv(joinpath(output,"common_parameters.tsv"),rows)
    write_tsv(joinpath(output,"diagnostics.tsv"),[summary])
    samples = permutedims(reshape(named,46,size(named,2),1),(2,3,1))
    bulk = vec(MCMCDiagnosticTools.ess(samples;kind=:bulk))
    tail = vec(MCMCDiagnosticTools.ess(samples;kind=:tail))
    rhat = vec(MCMCDiagnosticTools.rhat(samples))
    write_tsv(joinpath(output,"original_ncp_parameters.tsv"),[
        (;parameter=layout.names[j],bulk_ess=bulk[j],tail_ess=tail[j],split_rhat=rhat[j]) for j in 1:46])
    write_tsv(joinpath(output,"original_ncp_minimum.tsv"),[(;
        min_bulk_ess=minimum(bulk),limiting_parameter=layout.names[argmin(bulk)],
        min_bulk_ess_per_1000_sampling_gradients=1000minimum(bulk)/record.sampling_gradients)])
    println("RESULT\t",summary)
    println("ORIGINAL_NCP_MINIMUM\t",minimum(bulk),"\t",layout.names[argmin(bulk)])
    println("COMPLETE\t",output)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_baseline()
end
