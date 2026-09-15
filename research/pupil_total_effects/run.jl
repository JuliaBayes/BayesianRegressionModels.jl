include("audit.jl")
import DifferentiationInterface as DI
import Enzyme
using MCMCDiagnosticTools, Serialization, SHA, TOML, WarmupHMC
import Pkg

const ENZYME_BACKEND = DI.AutoEnzyme(;
    mode=Enzyme.set_runtime_activity(Enzyme.Reverse), function_annotation=Enzyme.Const)

function write_tsv(path, rows)
    isempty(rows) && error("Empty table: $path")
    names = propertynames(first(rows))
    open(path, "w") do io
        println(io, join(names, '\t'))
        for row in rows
            println(io, join((getproperty(row, key) for key in names), '\t'))
        end
    end
end

function problem(controls, data)
    J = length(data.ids)
    raw = PTE.PupilProblem(data)
    specs = [4+j => Reparametrization(PartiallyCentered(1.0),
        PartiallyCentered(Float64(controls[j])),
        j <= J ? PTE.INTERCEPT_MEAN : 0.0,
        let idx = j <= J ? 1 : 2
            q -> q[idx]
        end) for j in 1:2J]
    rp = ReparametrizedProblem(IndexedReparametrization(specs), raw, ENZYME_BACKEND)
    (; raw, rp)
end

function audit_reparametrization(data)
    q = PTE.initial_position(data)
    controls = collect(range(0.0,1.0,length=40))
    p = problem(controls, data)
    source = PTE.to_source(q, controls, data)
    value, g = LogDensityProblems.logdensity_and_gradient(p.rp, source)
    jacobian = sum((1-controls[j])*q[j<=20 ? 1 : 2] for j in 1:40)
    @test value ≈ first(PTE.evaluate(p.raw,q)) + jacobian
    fd = numerical_gradient(source) do r
        physical = PTE.to_model(r, controls, data)
        reference_density(p.raw, physical) + sum((1-controls[j])*r[j<=20 ? 1 : 2] for j in 1:40)
    end
    error = maximum(abs.(g-fd) ./ (1 .+ abs.(g)))
    @test error < 3e-5
    println("reparametrization_gradient_relative_error=", error)
    error
end

function select_offline(draws)
    grid = collect(0.0:0.01:1.0)
    selected, rows = zeros(40), NamedTuple[]
    for j in 1:40
        ell = @view draws[j<=20 ? 1 : 2, :]
        values = @view draws[4+j, :]
        loc = j <= 20 ? PTE.INTERCEPT_MEAN : 0.0
        losses = map(grid) do c
            candidate = (values .- loc) .* exp.((c-1) .* ell)
            log(std(candidate)) + mean((1-c) .* ell)
        end
        all(isfinite, losses) || error("Nonfinite offline loss at coordinate $j")
        selected[j] = grid[argmin(losses)]
        append!(rows, [(; coordinate=j, centeredness=c, loss=losses[k]) for (k,c) in enumerate(grid)])
    end
    selected, rows
end

function diagnostic_rows(label, record, data)
    q = record.positions
    samples = permutedims(reshape(q, size(q,1),size(q,2),1), (2,3,1))
    bulk = vec(MCMCDiagnosticTools.ess(samples; kind=:bulk))
    tail = vec(MCMCDiagnosticTools.ess(samples; kind=:tail))
    rhat = vec(MCMCDiagnosticTools.rhat(samples))
    names = PTE.coordinate_names(data)
    rows = [(; parameter=names[j], mean=mean(q[j,:]), sd=std(q[j,:]),
        bulk_ess=bulk[j], tail_ess=tail[j], split_rhat=rhat[j]) for j in eachindex(names)]
    summary = (; arm=label, retained_draws=size(q,2), min_bulk_ess=minimum(bulk),
        limiting_parameter=names[argmin(bulk)], min_tail_ess=minimum(tail),
        max_split_rhat=maximum(rhat), divergences=record.divergences,
        divergence_percent=100record.divergences/size(q,2),
        sampling_gradients=record.sampling_gradients, transition_gradients=record.transition_gradients,
        all_gradient_calls=record.all_gradient_calls,
        min_bulk_ess_per_1000_sampling_gradients=1000minimum(bulk)/record.sampling_gradients,
        min_bulk_ess_per_1000_all_gradients=1000minimum(bulk)/record.all_gradient_calls,
        min_total_coefficient_ess_per_1000_gradients=1000minimum(bulk[5:end])/record.sampling_gradients,
        fit_seconds=record.fit_seconds, compile_seconds=record.compile_seconds)
    rows, summary
end

function run_arm(label, controls, data, output; online=false, n_draws=2000, seed=1)
    target = problem(controls, data)
    initial = PTE.to_source(PTE.initial_position(data),controls,data)
    # Init=vector deliberately retains Pathfinder; all arms start from the same
    # physical OLS point. This changes only initialization, not the posterior.
    callback = (state, stage) -> begin
        println("boundary\t",label,'\t',stage,"\twindow=",state.outer_counter,
            "\tall_gradients=",target.raw.gradient_calls[])
        flush(stdout)
        isfile(joinpath(output,"STOP"))
    end
    checkpoint_dir = joinpath(output, "checkpoints-"*label)
    target.raw.gradient_calls[] = 0
    Base.cumulative_compile_timing(true)
    before = Base.cumulative_compile_time_ns()
    timed = try
        @timed adaptive_warmup_mcmc(Xoshiro(seed), target.rp; init=initial,
            n_draws, monitor_ess=true, nonlinear_adapt=online, callback, checkpoint_dir)
    finally
        Base.cumulative_compile_timing(false)
    end
    compilation = Base.cumulative_compile_time_ns() .- before
    fit = timed.value
    record = (; positions=convert(Matrix{Float64},fit.posterior_position),
        divergences=fit.n_divergent_samples, sampling_gradients=fit.sampling_evaluation_counter,
        transition_gradients=fit.total_evaluation_counter,
        all_gradient_calls=target.raw.gradient_calls[], fit_seconds=timed.time,
        compile_seconds=first(compilation)/1e9, gc_seconds=timed.gctime,
        controls=[last(pair).c for pair in WarmupHMC.reparam_sources(target.rp)],
        seed, requested_draws=n_draws, model_frame="log_tau_a,log_tau_b,gamma0,gamma1,total_A,total_B")
    size(record.positions,2) >= n_draws || error("$label stopped before the retained-draw floor")
    checkpoint = deserialize(joinpath(checkpoint_dir,"cp_latest.jls"))
    @test checkpoint.sampling_evaluation_counter == record.sampling_gradients
    @test checkpoint.total_evaluation_counter == record.transition_gradients
    @test record.all_gradient_calls >= record.transition_gradients >= record.sampling_gradients > 0
    @test WarmupHMC.back_transform(checkpoint,target.rp,checkpoint.posterior_position) ≈ record.positions
    source = reduce(hcat, (PTE.to_source(q,record.controls,data) for q in eachcol(record.positions)))
    @test source ≈ checkpoint.posterior_position
    serialize(joinpath(output,label*".jls"),record)
    write_tsv(joinpath(output,label*"_controls.tsv"),[(; coordinate=j, centeredness=c) for (j,c) in enumerate(record.controls)])
    parameters, summary = diagnostic_rows(label,record,data)
    write_tsv(joinpath(output,label*"_parameters.tsv"),parameters)
    write_tsv(joinpath(output,label*"_diagnostics.tsv"),[summary])
    println("RESULT\t",summary)
    flush(stdout)
    record, summary
end

function main()
    output = get(ENV,"PUPIL_TOTAL_OUTPUT","")
    isempty(output) && error("Set PUPIL_TOTAL_OUTPUT to a new output directory")
    ispath(output) && error("Output already exists; preserve saved fits and choose a new directory")
    mkpath(output)
    BLAS.set_num_threads(1)
    data = PTE.load_data()
    write_tsv(joinpath(output,"density_audit.tsv"),audit_model(PTE.PupilProblem(data)))
    transform_error = audit_reparametrization(data)
    mkpath(joinpath(output,"source"))
    for name in ("model.jl","audit.jl","run.jl")
        cp(joinpath(@__DIR__,name),joinpath(output,"source",name))
    end
    packages = [(; name=p.name,version=string(p.version),source=p.source,
        tree_hash=string(p.tree_hash)) for p in values(Pkg.dependencies())]
    write_tsv(joinpath(output,"packages.tsv"),sort(packages;by=x->x.name))
    provenance = Dict("source_data_revision"=>PTE.DATA_REVISION,"data_sha256"=>PTE.DATA_SHA256,
        "source_post"=>"https://discourse.mc-stan.org/t/help-testing-brms-pr-for-sum-to-zero-and-partial-centering/41542/3",
        "changes_from_post"=>"independent random intercept/load effects; Normal(5651.9,2026.1) mean-intercept prior",
        "population_load_prior"=>"flat, as in brms source; integrated out",
        "residual_model"=>"log_sigma=gamma0+gamma1*(numeric_subj-mean_numeric_subj)",
        "sigma_intercept_prior"=>"student_t(3,0,2.5)","sigma_slope_prior"=>"flat",
        "group_sd_priors"=>"half student_t(3,0,2026.1)",
        "sampled_dimensions"=>44,"groups"=>20,"observations"=>2228,
        "integrated_dimensions"=>2,"original_parameters_recovered"=>false,
        "posterior_scope"=>"same modified pupil posterior in every arm; not exact original correlated/Student-t-intercept forum model",
        "baseline"=>"fixed c=0 scaled total coefficients; not independent-standard-normal NCP",
        "offline_criterion"=>"log standard deviation plus inverse log-Jacobian; grid 0:0.01:1",
        "online_criterion"=>"WarmupHMC default weighted position-gradient correlation; grid 0:0.1:1",
        "ess_scope"=>"all 44 model-frame coordinates, including total effects and four hyperparameters",
        "timing_scope"=>"full sampler call including Pathfinder, first-use compilation and checkpoint IO; not a timing benchmark",
        "seed_each_arm"=>1,"draw_floor_each_arm"=>2000,"chains_each_arm"=>1,
        "transform_gradient_error"=>transform_error,"julia_version"=>string(VERSION),
        "git_commit"=>strip(read(`git -C $(@__DIR__) rev-parse HEAD`,String)))
    open(io->TOML.print(io,provenance),joinpath(output,"provenance.toml"),"w")
    baseline, ds = run_arm("scaled_total",zeros(40),data,output)
    summaries = [ds]
    write_tsv(joinpath(output,"diagnostics.tsv"),summaries)
    selected, losses = select_offline(baseline.positions)
    write_tsv(joinpath(output,"offline_losses.tsv"),losses)
    write_tsv(joinpath(output,"offline_selected.tsv"),[(; coordinate=j,centeredness=c) for (j,c) in enumerate(selected)])
    partial, ds = run_arm("partial",selected,data,output)
    push!(summaries,ds); write_tsv(joinpath(output,"diagnostics.tsv"),summaries)
    online, ds = run_arm("online",zeros(40),data,output;online=true)
    push!(summaries,ds); write_tsv(joinpath(output,"diagnostics.tsv"),summaries)
    write_tsv(joinpath(output,"workflow_costs.tsv"),[
        (; workflow="baseline",all_gradient_calls=baseline.all_gradient_calls),
        (; workflow="pilot_plus_partial",all_gradient_calls=baseline.all_gradient_calls+partial.all_gradient_calls),
        (; workflow="online",all_gradient_calls=online.all_gradient_calls)])
    println("COMPLETE\t",output)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
