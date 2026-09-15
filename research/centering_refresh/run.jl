# Full case-study refresh after the WarmupHMC active-position transport fix.
# Each invocation runs one arm; completed fits are immutable and reusable.
const CASE, REQUEST, OUTPUT = ARGS
CASE in ("hsgp", "eight", "radon") || error("Unknown case $CASE")
const CASE_DIR = CASE == "hsgp" ? "adaptive_centering" : CASE == "eight" ? "eight_schools_centering" : "radon_centering"
include(joinpath(@__DIR__, "..", CASE_DIR, "reproduce.jl"))
using Test

struct CountedDensity{P}
    problem::P
    gradient_calls::Base.RefValue{Int}
end
LogDensityProblems.dimension(p::CountedDensity) = LogDensityProblems.dimension(p.problem)
LogDensityProblems.capabilities(::Type{<:CountedDensity{P}}) where P = LogDensityProblems.capabilities(P)
LogDensityProblems.logdensity(p::CountedDensity, x) = LogDensityProblems.logdensity(p.problem, x)
function LogDensityProblems.logdensity_and_gradient(p::CountedDensity, x)
    p.gradient_calls[] += 1
    LogDensityProblems.logdensity_and_gradient(p.problem, x)
end

function select_online_loss!(position::Bool)
    # Existing internal selector, explicitly process-local until a public loss
    # keyword exists. The transport repair itself must already be in the package.
    source = read(joinpath(pkgdir(WarmupHMC), "src", "Reparametrizations.jl"), String)
    @test occursin("function _transport_active_evaluation", source)
    for (start, stop) in (
        ("reparametrization_loss((;ljac, cov)::OnlineReparametrizationLoss", "scale_estimate(loss::OnlineReparametrizationLoss)"),
        ("function reparametrization_loss(loss::WeightedReparametrizationLoss", "scale_estimate(loss::WeightedReparametrizationLoss)"))
        first_index = findfirst(start, source).start
        last_index = findnext(stop, source, first_index).start - 1
        body = source[first_index:last_index]
        @test occursin("w1=0", body)
        Base.include_string(WarmupHMC, replace(body, "w1=0" => (position ? "w1=1" : "w1=0")), "case-study-loss-selection")
    end
end

function make_stan(output)
    CASE == "hsgp" ? stan_density(build_brmi(prepared_data()), "model", output) : stan_density("model", output)
end

function candidate_inputs(stan, pilot, indices)
    positions = pilot.posterior_position
    logs = zeros(length(indices), size(positions, 2))
    if CASE == "hsgp"
        for gp in gp_draws(stan, pilot, prepared_data()), k in 1:DEFAULT_K
            # BRM exposes log spectral SD without exponentiating it.
            block = only(filter(b -> b.logical == (gp.name == "mu" ? :mu : :sigma),
                BRM._adaptive_hsgp_centering_blocks(stan.sb, BS.param_unc_names(stan.density.model))))
            i = only(findall(==(block.effects[k]), indices))
            logs[i, :] = gp.logs[:, k]
        end
    else
        for block in adaptive_centering_blocks(stan.sb, BS.param_unc_names(stan.density.model))
            @test size(block.effects, 1) == 1
            for index in vec(block.effects)
                i = only(findall(==(index), indices))
                logs[i, :] = positions[only(block.log_scales), :]
            end
        end
    end
    logs
end

function select_controls(stan, pilot, indices, gradients, output, arm)
    logs = candidate_inputs(stan, pilot, indices)
    selected = zeros(length(indices)); rows = NamedTuple[]
    for (j, index) in enumerate(indices)
        z, g, ell = pilot.posterior_position[index, :], gradients[index, :], logs[j, :]
        scores = Float64[]
        for c in 0.0:0.01:1.0
            scale = exp.(c .* ell)
            u = z .* scale
            gu = g ./ scale
            score = if arm == "posthoc_position"
                log(std(u)) - mean(c .* ell)
            else
                cor(u, gu)
            end
            admissible = all(isfinite, u) && all(isfinite, gu) && all(>(0), scale) && isfinite(score)
            push!(scores, admissible ? score : Inf)
            push!(rows, (; index, centeredness=c, loss=admissible ? score : missing, admissible))
        end
        any(isfinite, scores) || error("No admissible centering at $index")
        selected[j] = (argmin(scores)-1)/100
    end
    write_tsv(joinpath(output, "selection_losses.tsv"), rows)
    selected
end

function physical_qois(stan, positions)
    if CASE == "eight"
        names = String.(BS.param_unc_names(stan.density.model)); l = brm_layout(names)
        q = copy(positions)
        q[l.effects, :] .= positions[l.population:l.population, :] .+
            exp.(positions[l.scale:l.scale, :]) .* positions[l.effects, :]
        names[l.population] = "mu"; names[l.scale] = "log_tau"
        names[l.effects] = ["theta[$j]" for j in 1:8]
        return names, q
    elseif CASE == "radon"
        names = String.(BS.param_unc_names(stan.density.model)); q = copy(positions)
        for (role, entry) in enumerate(effect_blocks(stan.sb, names))
            population = only(findall(==("pop_mu_beta_pop.$role"), names))
            block = entry.block
            q[vec(block.effects), :] .= positions[population:population, :] .+
                exp.(positions[only(block.log_scales):only(block.log_scales), :]) .* positions[vec(block.effects), :]
            names[vec(block.effects)] = ["county_$(entry.role)[$j]" for j in 1:RADON_DATA.J]
        end
        return names, q
    else
        # Functions at each unique observed time, and all four GP hyperparameters.
        data = prepared_data(); draw = (; posterior_position=positions)
        times = unique(data.times); keep = [findfirst(==(t), data.times) for t in times]
        blocks = BRM._adaptive_hsgp_centering_blocks(stan.sb, BS.param_unc_names(stan.density.model))
        hyper = vcat([vcat(b.length_scales, b.sd) for b in blocks]...)
        qnames = String.(BS.param_unc_names(stan.density.model))[hyper]
        matrices = [positions[hyper, :]]
        basis = [sin(pi / (2L) * (data.x[i] + L) * j) / sqrt(L) for i in keep, j in 1:DEFAULT_K]
        for gp in gp_draws(stan, draw, data)
            logical = gp.name == "mu" ? :mu : :sigma
            values = basis * transpose(gp.weights)
            logical == :sigma && (values = exp.(values))
            push!(matrices, values)
            append!(qnames, ["$(logical)(time=$t)" for t in times])
        end
        return qnames, vcat(matrices...)
    end
end

function main(ARM)
    ARM in ("ncp", "cp", "posthoc_position", "posthoc_gradient", "online_position", "online_gradient") || error("Unknown arm $ARM")
    out = joinpath(OUTPUT, ARM)
    ispath(out) && error("Preserve existing $out")
    mkpath(out); BLAS.set_num_threads(1)
    whmc_source = read(joinpath(pkgdir(WarmupHMC), "src", "Reparametrizations.jl"), String)
    @test occursin("function _transport_active_evaluation", whmc_source)
    stan = make_stan(out)
    names = String.(BS.param_unc_names(stan.density.model))
    counted = CountedDensity(stan.density, Ref(0))
    adaptive = adaptive_centering_problem(stan.sb, counted, ENZYME_BACKEND; unc_names=names)
    indices = first.(WarmupHMC.reparam_sources(adaptive))
    controls = zeros(length(indices)); pilot_cost = 0
    if ARM == "cp"
        controls .= 1
    elseif startswith(ARM, "posthoc")
        pilot = deserialize(joinpath(OUTPUT, "ncp", "fit.jls"))
        checkpoint = deserialize(joinpath(OUTPUT, "ncp", "checkpoints", "cp_latest.jls"))
        @test checkpoint.posterior_position ≈ pilot.posterior_position
        for j in (1, 5000, 10000)
            @test last(LogDensityProblems.logdensity_and_gradient(stan.density, checkpoint.posterior_position[:,j])) ≈ checkpoint.posterior_gradient[:,j]
        end
        controls = select_controls(stan, pilot, indices, checkpoint.posterior_gradient, out, ARM)
        pilot_cost = pilot.all_gradient_calls
    end
    WarmupHMC.restore_reparam_sources!(adaptive, [idx => PartiallyCentered(c) for (idx,c) in zip(indices, controls)])
    target = ARM == "ncp" ? counted : adaptive
    online = startswith(ARM, "online")
    counted.gradient_calls[] = 0
    callback = (state, stage) -> begin
        println("BOUNDARY\t", CASE, '\t', ARM, '\t', stage, "\twindow=", state.outer_counter,
            "\tall_gradients=", counted.gradient_calls[])
        flush(stdout)
        isfile(joinpath(OUTPUT, "STOP"))
    end
    measured = @timed WarmupHMC.adaptive_warmup_mcmc(Xoshiro(1), target;
        n_draws=10_000, monitor_ess=true, nonlinear_adapt=online, callback,
        checkpoint_dir=joinpath(out, "checkpoints"))
    fit = measured.value
    positions = Matrix{Float64}(fit.posterior_position)
    checkpoint = deserialize(joinpath(out, "checkpoints", "cp_latest.jls"))
    @test size(positions, 2) >= 10_000
    @test checkpoint.sampling_evaluation_counter == fit.sampling_evaluation_counter
    @test checkpoint.total_evaluation_counter == fit.total_evaluation_counter
    @test counted.gradient_calls[] >= fit.total_evaluation_counter >= fit.sampling_evaluation_counter > 0
    mapped = copy(checkpoint.posterior_position)
    WarmupHMC.reparametrize!(target, mapped)
    @test mapped ≈ positions
    record = (; posterior_position=positions, n_divergent_samples=fit.n_divergent_samples,
        total_gradient_evaluations=fit.total_evaluation_counter,
        sampling_gradient_evaluations=fit.sampling_evaluation_counter,
        all_gradient_calls=counted.gradient_calls[], pilot_gradient_calls=pilot_cost,
        workflow_gradient_calls=counted.gradient_calls[]+pilot_cost,
        controls=WarmupHMC.reparam_sources(adaptive), nonlinear_adapt=online,
        complete=true, seed=1, requested_draws=10_000, fit_seconds=measured.time,
        stored_frame="NCP model", case=CASE, arm=ARM)
    serialize(joinpath(out, "fit.jls"), record)
    write_tsv(joinpath(out, "controls.tsv"), [(; index=idx, centeredness=c.c) for (idx,c) in record.controls])
    qnames, q = physical_qois(stan, positions)
    @test all(isfinite, q)
    samples = permutedims(reshape(q, size(q,1),size(q,2),1), (2,3,1))
    bulk = vec(MCMCDiagnosticTools.ess(samples; kind=:bulk))
    tail = vec(MCMCDiagnosticTools.ess(samples; kind=:tail))
    rhat = vec(MCMCDiagnosticTools.rhat(samples))
    write_tsv(joinpath(out, "parameters.tsv"), [(; qoi=qnames[i], mean=mean(q[i,:]), sd=std(q[i,:]),
        bulk_ess=bulk[i], tail_ess=tail[i], split_rhat=rhat[i]) for i in eachindex(qnames)])
    summary = (; case=CASE, arm=ARM, qois=length(qnames), draws=size(q,2),
        min_bulk_ess=minimum(bulk), limiting_qoi=qnames[argmin(bulk)], min_tail_ess=minimum(tail),
        max_split_rhat=maximum(rhat), divergences=fit.n_divergent_samples,
        sampling_gradients=record.sampling_gradient_evaluations,
        all_fit_gradients=record.all_gradient_calls, pilot_gradients=pilot_cost,
        total_gradients=record.workflow_gradient_calls,
        sampling_efficiency=minimum(bulk)/record.sampling_gradient_evaluations,
        total_efficiency=minimum(bulk)/record.workflow_gradient_calls)
    write_tsv(joinpath(out, "summary.tsv"), [summary])
    write_tsv(joinpath(out, "packages.tsv"), package_snapshot())
    cp(@__FILE__, joinpath(out, "run.jl"))
    println("CASE_ARM_COMPLETE\t", summary); flush(stdout)
end

if abspath(PROGRAM_FILE) == @__FILE__
    for arm in split(REQUEST, ',')
        startswith(arm, "online") && select_online_loss!(arm == "online_position")
        Base.invokelatest(main, arm)
    end
end
