# Arm matrix for one case: CASE OUT AUDIT_DIR
# Twelve WarmupHMC arms, one chain, seed 1, 10,000 retained draws:
#   total_{ncp,cp,posthoc_position,posthoc_gradient,online_position,online_gradient}   (exact totals)
#   ordinary_{ncp,cp,posthoc_position,posthoc_gradient,online_position,online_gradient} (conventional block)
# Native RBesT arms are produced by native.R and analyzed by complete.jl.
isdefined(@__MODULE__, :RBesTCentering) || include(joinpath(@__DIR__, "model.jl"))
using .RBesTCentering, Serialization, Statistics, LinearAlgebra, Random, JSON, BridgeStan
using WarmupHMC, LogDensityProblems, MCMCDiagnosticTools
const RC = RBesTCentering
const N_DRAWS = 10_000

function write_tsv(path, rows)
    keys = propertynames(first(rows))
    open(path, "w") do io
        println(io, join(keys, '\t'))
        for row in rows; println(io, join((getproperty(row, k) for k in keys), '\t')); end
    end
end
qoi_names(c) = vcat(["population_mean", "between_trial_sd"], ["trial_$(j)_total" for j in 1:c.H])

function total_qois(c, m, positions)
    rows = permutedims(positions)
    recovered = RC.BRM.recover_population_draws(m.sb, rows, m.names; rng=Xoshiro(404))[c.lp]
    hcat(recovered.population[:, 1], exp.(rows[:, only(m.coords.scales)]), rows[:, vec(m.coords.totals)])
end
ordinary_qois(c, m, positions) = permutedims(RC.ordinary_physical(c, m, positions))

function diagnostics(c, label, qois, fit, out)
    names = qoi_names(c); @assert size(qois) == (N_DRAWS, length(names)) && all(isfinite, qois)
    cube = reshape(qois, N_DRAWS, 1, length(names))
    ess = vec(MCMCDiagnosticTools.ess(cube; kind=:bulk)); tail = vec(MCMCDiagnosticTools.ess(cube; kind=:tail))
    rh = vec(MCMCDiagnosticTools.rhat(cube)); mcse = vec(MCMCDiagnosticTools.mcse(cube; kind=mean))
    write_tsv(joinpath(out, label * "-qois.tsv"), [(; parameter=names[k], mean=mean(qois[:, k]), sd=std(qois[:, k]),
        bulk_ess=ess[k], tail_ess=tail[k], split_rhat=rh[k], mcse=mcse[k]) for k in eachindex(names)])
    s = (; arm=label, draws=N_DRAWS, min_bulk_ess=minimum(ess), limiting_qoi=names[argmin(ess)], min_tail_ess=minimum(tail),
        sampling_gradients=fit.sampling_gradients, fit_gradients=fit.all_gradient_calls, pilot_gradients=fit.pilot_gradient_calls,
        total_gradients=fit.total_gradient_calls,
        ess_per_1000_sampling_gradients=1000minimum(ess) / fit.sampling_gradients,
        ess_per_1000_total_gradients=1000minimum(ess) / fit.total_gradient_calls,
        divergences=fit.divergences, max_split_rhat=maximum(rh), fit_seconds=fit.fit_seconds,
        numerical_rejections=get(fit, :numerical_rejections, 0))
    write_tsv(joinpath(out, label * "-summary.tsv"), [s]); println("RBEST_SCIENTIFIC_RESULT ", s); flush(stdout)
end

"""One WarmupHMC arm. `m` is a total or ordinary model record; `wrapped=false` samples the raw compiled target."""
function run_arm(c, m, label, out; centeredness=0.0, online=false, pilot_cost=0, wrapped=true)
    # Stan-side numerical rejections (a NaN/infinite proposal) are treated as native Stan treats them:
    # the evaluation is charged, returns -Inf, and is counted in the record.
    raw = RC.BrmsPupilProblem(m.model, Ref(0); reject_numerical_errors=true)
    rp = wrapped ? RC.BRM.adaptive_centering_problem(m.sb, raw, RC.ENZYME_BACKEND; unc_names=m.names, centeredness) : raw
    start = m.kind == :total ? RC.total_initial(c, m) : RC.ordinary_initial(c, m)
    init = wrapped ? last(WarmupHMC._inverse_with_logabsdet_jacobian(WarmupHMC.reparametrizer(rp), start)) : start
    checkpoint_dir = joinpath(out, label * "-checkpoints")
    callback = (state, stage) -> begin
        println("BOUNDARY ", label, " ", stage, " window=", state.outer_counter, " gradients=", raw.gradient_calls[])
        flush(stdout); isfile(joinpath(out, "STOP"))
    end
    path = joinpath(out, label * ".jls")
    record = if isfile(path)
        deserialize(path)
    else
        raw.gradient_calls[] = 0
        timed = @timed adaptive_warmup_mcmc(Xoshiro(1), rp; init, n_draws=N_DRAWS, monitor_ess=true,
            nonlinear_adapt=online, checkpoint_dir, callback)
        fit = timed.value; cp = deserialize(joinpath(checkpoint_dir, "cp_latest.jls"))
        @assert size(fit.posterior_position, 2) >= N_DRAWS
        @assert raw.gradient_calls[] >= fit.total_evaluation_counter >= fit.sampling_evaluation_counter > 0
        positions = Matrix(fit.posterior_position)[:, 1:N_DRAWS]
        if wrapped
            @assert WarmupHMC.back_transform(cp, rp, cp.posterior_position) ≈ fit.posterior_position
        end
        value = (; positions, source_positions=Matrix(cp.posterior_position)[:, 1:N_DRAWS],
            source_gradients=Matrix(cp.posterior_gradient)[:, 1:N_DRAWS], names=m.names,
            controls=wrapped ? [last(p).c for p in WarmupHMC.reparam_sources(rp)] : Float64[],
            control_indices=wrapped ? first.(WarmupHMC.reparam_sources(rp)) : Int[],
            sampling_gradients=fit.sampling_evaluation_counter, all_gradient_calls=raw.gradient_calls[],
            total_gradient_calls=raw.gradient_calls[] + pilot_cost, pilot_gradient_calls=pilot_cost,
            divergences=fit.n_divergent_samples, fit_seconds=timed.time, kind=m.kind, online, seed=1,
            numerical_rejections=raw.numerical_rejections[], first_numerical_error=raw.first_numerical_error[])
        serialize(path, value); value
    end
    qois = m.kind == :total ? total_qois(c, m, record.positions) : ordinary_qois(c, m, record.positions)
    serialize(joinpath(out, label * "-qois.jls"), qois); diagnostics(c, label, qois, record, out)
    record
end

function online_loss!(kind)
    source = read(joinpath(pkgdir(WarmupHMC), "src", "Reparametrizations.jl"), String)
    for (start, finish) in (("reparametrization_loss((;ljac, cov)::OnlineReparametrizationLoss", "scale_estimate(loss::OnlineReparametrizationLoss)"),
            ("function reparametrization_loss(loss::WeightedReparametrizationLoss", "scale_estimate(loss::WeightedReparametrizationLoss)"))
        a = first(findfirst(start, source)); b = first(findnext(finish, source, a)) - 1
        code = source[a:b]; @assert occursin("w1=0", code)
        Base.include_string(WarmupHMC, replace(code, "w1=0" => (kind == :position ? "w1=1" : "w1=0")), "rbest-loss-selection")
    end
end

"""Post-hoc control selection for the ordinary block from NCP pilot draws (the four-page refresh rule)."""
function select_ordinary_controls(m, pilot, criterion, out, label)
    indices = pilot.control_indices; positions = pilot.positions; gradients = pilot.source_gradients
    ell = positions[m.log_scale, :]
    selected = zeros(length(indices)); rows = NamedTuple[]
    for (j, index) in enumerate(indices)
        z, g = positions[index, :], gradients[index, :]
        scores = Float64[]
        for cc in 0.0:0.01:1.0
            scale = exp.(cc .* ell); u = z .* scale; gu = g ./ scale
            score = criterion == :position ? log(std(u)) - mean(cc .* ell) : cor(u, gu)
            admissible = all(isfinite, u) && all(isfinite, gu) && isfinite(score)
            push!(scores, admissible ? score : Inf)
            push!(rows, (; index, centeredness=cc, loss=admissible ? score : missing, admissible))
        end
        any(isfinite, scores) || error("No admissible centering at $index")
        selected[j] = (argmin(scores) - 1) / 100
    end
    write_tsv(joinpath(out, label * "-selection_losses.tsv"), rows)
    selected
end

function run_matrix(c, out, t, o)
    mkpath(out); BLAS.set_num_threads(1)
    # Exact totals: NCP pilot through the wrapper at c=0, then transport its draws and gradients to the model frame.
    pilot = run_arm(c, t, "total_ncp", out)
    source = RC.BRM.adaptive_centering_problem(t.sb, RC.BrmsPupilProblem(t.model, Ref(0)), RC.ENZYME_BACKEND; unc_names=t.names, centeredness=0.0)
    target = RC.BRM.adaptive_centering_problem(t.sb, RC.BrmsPupilProblem(t.model, Ref(0)), RC.ENZYME_BACKEND; unc_names=t.names, centeredness=1.0)
    physical = similar(pilot.positions); gradients = similar(pilot.positions)
    WarmupHMC._jointly_transport_halo!(target, WarmupHMC.reparametrizer(source), pilot.source_positions, pilot.source_gradients, physical, gradients)
    @assert physical ≈ pilot.positions
    for draw in (1, N_DRAWS ÷ 2, N_DRAWS)
        @assert isapprox(gradients[:, draw], last(BridgeStan.log_density_gradient(t.model, physical[:, draw]; propto=false)); rtol=1e-8, atol=1e-7)
    end
    serialize(joinpath(out, "total-pilot-model-gradients.jls"), gradients)
    for criterion in (:position, :gradient)
        selected = RC.BRM.select_total_centeredness(t.sb, permutedims(physical), t.names; criterion, gradients=permutedims(gradients))
        serialize(joinpath(out, "total_selected_$(criterion).jls"), selected)
        run_arm(c, t, "total_posthoc_$(criterion)", out; centeredness=selected.centeredness, pilot_cost=pilot.all_gradient_calls)
    end
    run_arm(c, t, "total_cp", out; centeredness=1.0)
    for criterion in (:position, :gradient)
        online_loss!(criterion)
        Base.invokelatest(run_arm, c, t, "total_online_$(criterion)", out; online=true)
    end
    # Conventional block: NCP pilot through the wrapper at c=0 (records source gradients), CP at c=1, post hoc, online.
    opilot = run_arm(c, o, "ordinary_ncp", out)
    @assert opilot.positions ≈ opilot.source_positions   # c=0 is the compiled NCP frame
    for draw in (1, N_DRAWS ÷ 2, N_DRAWS)
        @assert isapprox(opilot.source_gradients[:, draw], last(BridgeStan.log_density_gradient(o.model, opilot.positions[:, draw]; propto=false)); rtol=1e-8, atol=1e-7)
    end
    for criterion in (:position, :gradient)
        selected = select_ordinary_controls(o, opilot, criterion, out, "ordinary_posthoc_$(criterion)")
        serialize(joinpath(out, "ordinary_selected_$(criterion).jls"), selected)
        run_arm(c, o, "ordinary_posthoc_$(criterion)", out; centeredness=selected, pilot_cost=opilot.all_gradient_calls)
    end
    run_arm(c, o, "ordinary_cp", out; centeredness=1.0)
    for criterion in (:position, :gradient)
        online_loss!(criterion)
        Base.invokelatest(run_arm, c, o, "ordinary_online_$(criterion)", out; online=true)
    end
    println("RBEST_MATRIX_COMPLETE ", c.name); flush(stdout)
end

if abspath(PROGRAM_FILE) == @__FILE__
    case, out, audit_dir = ARGS
    @assert JSON.parsefile(joinpath(audit_dir, "audit.json"))["status"] == "passed"
    c = RC.load_case(case); t = RC.total_model(c, joinpath(out, "model")); o = RC.ordinary_model(c, joinpath(out, "model"))
    run_matrix(c, out, t, o)
end
