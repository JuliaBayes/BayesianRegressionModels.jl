using BayesianRegressionModels
using BridgeStan
import DifferentiationInterface as DI
import Enzyme
using Distributions
using LinearAlgebra
using LogDensityProblems
using MCMCDiagnosticTools
using Random
using Serialization
using SHA
using StanBlocks
using Statistics
using TOML
using WarmupHMC
import Pkg

const BRM = BayesianRegressionModels
const BS = BridgeStan
const RESEARCH_DIR = @__DIR__
const SOURCE_MODEL_REVISION = "a42b3da85b7dc38f2745dde4fca197425f18c516"
const SOURCE_DATA_REVISION = "93b8b05cb7978952606f2043bec64d3b958b360c"
const SOURCE_MODEL_SHA256 = "1624c8770e8f08a90894f3417591eb45f93ccfb70cbe9870442ae2ab26343422"
const SOURCE_DATA_SHA256 = "fccd1624bd240b0c26a8b668f4d2181f0653000e4b169d393b3dc815a0b46952"
const SOURCE_DRAWS = 10_000
const SOURCE_SEED = 1
const ENZYME_BACKEND = DI.AutoEnzyme(;
    mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
    function_annotation=Enzyme.Const)

function checked_file(name, digest)
    path = joinpath(RESEARCH_DIR, name)
    bytes2hex(sha256(read(path))) == digest || error("$name source hash mismatch")
    path
end

function read_eight_schools()
    path = checked_file("eight_schools.data.R", SOURCE_DATA_SHA256)
    lines = readlines(path)
    values = Dict{String,Vector{Float64}}()
    for line in lines
        m = match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*<-\s*(?:c\()?([^)]*)\)?\s*$", line)
        m === nothing && error("unrecognized source data line: $line")
        values[m.captures[1]] = parse.(Float64, split(strip(m.captures[2]), ','; keepempty=false))
    end
    values["J"] == [8.0] || error("expected eight schools")
    length(values["y"]) == length(values["sigma"]) == 8 ||
        error("expected eight estimates and standard errors")
    all(isfinite, values["y"]) && all(>(0), values["sigma"]) ||
        error("eight-schools data must be finite with positive standard errors")
    (; J=Int(only(values["J"])), y=values["y"], sigma=values["sigma"])
end

"""A self-contained flat prior, without loading or sampling through Turing."""
struct EightSchoolsFlat <: ContinuousUnivariateDistribution end
struct EightSchoolsFlatPositive <: ContinuousUnivariateDistribution end

Distributions.logpdf(::EightSchoolsFlat, ::Real) = 0.0
Distributions.loglikelihood(::EightSchoolsFlat, ::AbstractVector{<:Real}) = 0.0
Base.minimum(::EightSchoolsFlat) = -Inf
Base.maximum(::EightSchoolsFlat) = Inf
Distributions.rand(rng::Random.AbstractRNG, ::EightSchoolsFlat) = randn(rng)

Distributions.logpdf(::EightSchoolsFlatPositive, x::Real) = x >= 0 ? 0.0 : -Inf
Distributions.loglikelihood(d::EightSchoolsFlatPositive, x::AbstractVector{<:Real}) =
    sum(Distributions.logpdf(d, x))
Base.minimum(::EightSchoolsFlatPositive) = 0.0
Base.maximum(::EightSchoolsFlatPositive) = Inf
Distributions.rand(rng::Random.AbstractRNG, ::EightSchoolsFlatPositive) = abs(randn(rng))

eight_schools_flat() = EightSchoolsFlat()
eight_schools_flat_positive() = EightSchoolsFlatPositive()
BRM.brm_distribution_type(::typeof(eight_schools_flat)) = EightSchoolsFlat
BRM.brm_distribution_type(::typeof(eight_schools_flat_positive)) = EightSchoolsFlatPositive
BRM._sb_stan_dist_name(::typeof(eight_schools_flat)) = :brm_eight_schools_flat
BRM._sb_stan_dist_name(::typeof(eight_schools_flat_positive)) = :brm_eight_schools_flat_positive
# Homogeneous positive vector-prior lowering asks for the mapped distribution
# TYPE as well as the factory callable.
BRM._sb_stan_dist_name(::Type{EightSchoolsFlat}) = :brm_eight_schools_flat
BRM._sb_stan_dist_name(::Type{EightSchoolsFlatPositive}) = :brm_eight_schools_flat_positive

StanBlocks.@deffun begin
    @lpxf brm_eight_schools_flat_lpdf(y::real)::real = 0.0
    brm_eight_schools_flat_rng()::real = normal_rng(0.0, 1.0)
    @lpxf brm_eight_schools_flat_positive_lpdf(y::real)::real = 0.0
    brm_eight_schools_flat_positive_rng()::real = abs(normal_rng(0.0, 1.0))
end

const EIGHT_SCHOOLS = @brm begin
    theta ~ 1 + (1 | eight_schools | school)
    effect(theta, Intercept) ~ eight_schools_flat()
    sd(:, eight_schools) ~ eight_schools_flat_positive()
    y ~ Normal(theta, sigma)
end

model_data(data=read_eight_schools()) = (; school=1:data.J, y=data.y, sigma=data.sigma)
build_brmi(data=read_eight_schools()) = EIGHT_SCHOOLS(model_data(data))

function write_tsv(path, rows)
    isempty(rows) && error("cannot write an empty table: $path")
    names = propertynames(first(rows))
    open(path, "w") do io
        println(io, join(names, '\t'))
        for row in rows
            println(io, join((getproperty(row, name) for name in names), '\t'))
        end
    end
end

function package_snapshot()
    map(sort!(collect(values(Pkg.dependencies())); by=p -> p.name)) do p
        git_sha = isdir(joinpath(p.source, ".git")) || isfile(joinpath(p.source, ".git")) ?
            strip(read(`git -C $(p.source) rev-parse HEAD`, String)) : ""
        (; package=p.name, version=string(p.version), tree_hash=string(p.tree_hash),
           source=p.source, git_sha)
    end
end

function run_provenance(output_dir)
    BLAS.set_num_threads(1)
    packages = package_snapshot()
    write_tsv(joinpath(output_dir, "packages.tsv"), packages)
    metadata = Dict(
        "source_model_revision" => SOURCE_MODEL_REVISION,
        "source_data_revision" => SOURCE_DATA_REVISION,
        "source_model_sha256" => SOURCE_MODEL_SHA256,
        "source_data_sha256" => SOURCE_DATA_SHA256,
        "julia_version" => string(VERSION),
        "brm_commit" => strip(read(`git -C $RESEARCH_DIR rev-parse HEAD`, String)),
        "script_sha256" => bytes2hex(sha256(read(@__FILE__))),
        "schools" => 8,
        "seed_each_fit" => SOURCE_SEED,
        "draws_requested_each_fit" => SOURCE_DRAWS,
        "chains_each_fit" => 1,
        "sampler" => "WarmupHMC.adaptive_warmup_mcmc",
        "sampler_config" => "defaults; n_draws=10000; monitor_ess=true; no initializer or NUTS overrides",
        "timed_scope" => "sampler call only: includes initialization, first-use Julia/AD compilation and checkpoint I/O; excludes Stan compilation, post-fit extraction, plotting and offline selection; counters follow WarmupHMC documented scopes",
        "turing_sampling" => "not run",
        "r_sampling" => "not run",
        "blas_threads" => BLAS.get_num_threads(),
        "ess_scope" => "all 10 unconstrained model coordinates: mu, log(tau), and eight school effects",
        "diagnostics" => "rank-normalized split Rhat; bulk ESS; tail ESS; retained divergences",
        "rhat_scope" => "within one split chain, not independent-chain convergence",
    )
    open(joinpath(output_dir, "provenance.toml"), "w") do io
        TOML.print(io, metadata)
    end
    packages
end

function stan_density(label, output_dir)
    mkpath(output_dir)
    sb = SBBRMI(build_brmi(); mod=@__MODULE__)
    code = BRM.stan_code(sb)
    checked = StanBlocks.stanc_check(code; warn_pedantic=false)
    checked.ok || error("stanc failed for $label\n$(checked.output)")
    path = joinpath(output_dir, "eight-schools-$label.stan")
    # The callable flat-prior families are generated during SBBRMI lowering.
    # Enter the newest Julia world for compilation so StanBlocks sees their
    # registered lpdf metadata when this helper is itself called from a function.
    problem = Base.invokelatest(StanBlocks.stan_instantiate, sb.model; path)
    q = zeros(LogDensityProblems.dimension(problem))
    value, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
    isfinite(value) && all(isfinite, gradient) || error("non-finite generated Stan density")
    (; density=problem, q, sb, code, path)
end

function physical_values(model, q)
    names = BS.param_names(model)
    values = BS.param_constrain(model, collect(q))
    Dict(zip(names, values))
end

function brm_layout(brm_names)
    population = only(findall(name -> occursin("pop_theta_beta_pop", name), brm_names))
    scale_candidates = findall(name -> occursin("log_scale", name) ||
                                        occursin(r"_tau(?:\.\d+)?$", name), brm_names)
    scale = only(scale_candidates)
    effect_names = filter(name -> occursin(r"_(?:xi|z_flat)\.\d+$", name), brm_names)
    effects = [only(findall(==(name), brm_names))
               for name in sort(effect_names; by=name -> parse(Int, last(split(name, '.'))))]
    length(effects) == 8 || error("expected eight school-effect coordinates")
    (; population, scale, effects)
end

function source_model(output_dir)
    mkpath(output_dir)
    source_path = checked_file("eight_schools.stan", SOURCE_MODEL_SHA256)
    data = read_eight_schools()
    array_json(x) = "[" * join(x, ",") * "]"
    source_data = "{" * join((
        "\"J\":8", "\"y\":" * array_json(data.y),
        "\"sigma\":" * array_json(data.sigma)), ",") * "}"
    BS.StanModel(source_path, source_data)
end

function brm_position(source_model, source_q, brm_names)
    p = physical_values(source_model, source_q)
    mu = p["mu"]
    tau = p["tau"]
    out = zeros(length(brm_names))
    layout = brm_layout(brm_names)
    for (i, name) in enumerate(brm_names)
        if i == layout.population
            out[i] = mu
        elseif i == layout.scale
            out[i] = log(tau)
        elseif i in layout.effects
            j = only(findall(==(i), layout.effects))
            out[i] = (p["theta.$j"] - mu) / tau
        else
            error("unmapped BRM coordinate $name")
        end
    end
    out
end

function brm_physical_gradient(gradient, brm_names, q_brm, source_model, source_q,
                               source_names)
    p = physical_values(source_model, source_q)
    tau = p["tau"]
    out = similar(gradient)
    z = Dict{Int,Float64}()
    logtau = 0.0
    layout = brm_layout(brm_names)
    scale_gradient = gradient[layout.scale]
    effect_index = Dict(layout.effects .=> eachindex(layout.effects))
    for i in eachindex(brm_names)
        if i == layout.population
            out[i] = gradient[i]
        elseif i == layout.scale
            logtau = q_brm[i]
            scale_gradient = gradient[i]
        elseif haskey(effect_index, i)
            j = only(findall(==(i), layout.effects))
            z[j] = q_brm[i]
            out[i] = gradient[i] / tau
        else
            error("unmapped BRM coordinate $name")
        end
    end
    scale_index = layout.scale
    scale_gradient -= dot(gradient[layout.effects], q_brm[layout.effects])
    out[layout.population] -= sum(gradient[layout.effects]) / tau
    # theta = mu + tau*z has an 8-dimensional source-to-BRM Jacobian tau^-8.
    # Include its log determinant before comparing centered-source gradients.
    scale_gradient -= length(layout.effects)
    out[scale_index] = scale_gradient
    @assert isapprox(logtau, log(tau); atol=1e-12)
    source_ordered = similar(out)
    for (i, name) in enumerate(source_names)
        source_ordered[i] = name == "mu" ? out[layout.population] :
            name == "tau" ? out[scale_index] :
            out[layout.effects[parse(Int, last(split(name, '.')))]]
    end
    source_ordered
end

function sample_source_fit(target, label, output_dir; nonlinear_adapt=true)
    result_path = joinpath(output_dir, "$label.jls")
    isfile(result_path) && error("A completed $label fit already exists; use a fresh directory.")
    callback = (state, stage) -> begin
        println("boundary\t", label, '\t', stage, "\twindow=", state.outer_counter)
        flush(stdout)
        isfile(joinpath(output_dir, "STOP"))
    end
    println("sampling\t", label, "\tseed=1\tn_draws=10000\tWarmupHMC defaults")
    flush(stdout)
    started = time_ns()
    fit = WarmupHMC.adaptive_warmup_mcmc(
        Xoshiro(SOURCE_SEED), target; n_draws=SOURCE_DRAWS, monitor_ess=true,
        nonlinear_adapt, callback, checkpoint_dir=joinpath(output_dir, "checkpoints-$label"))
    fit_seconds = (time_ns() - started) / 1e9
    retained = size(fit.posterior_position, 2)
    record = (; posterior_position=convert(Matrix{Float64}, fit.posterior_position),
        n_divergent_samples=fit.n_divergent_samples, seed=SOURCE_SEED, fit_seconds,
        total_gradient_evaluations=fit.total_evaluation_counter,
        sampling_gradient_evaluations=fit.sampling_evaluation_counter,
        requested_draws=SOURCE_DRAWS, complete=retained >= SOURCE_DRAWS)
    serialize(result_path, record)
    record.complete || error("$label stopped early with $retained draws")
    0 < record.sampling_gradient_evaluations <= record.total_gradient_evaluations ||
        error("invalid counters for $label")
    checkpoint = deserialize(joinpath(output_dir, "checkpoints-$label", "cp_latest.jls"))
    checkpoint.total_evaluation_counter == record.total_gradient_evaluations ||
        error("$label total counter differs from final checkpoint")
    checkpoint.sampling_evaluation_counter == record.sampling_gradient_evaluations ||
        error("$label sampling counter differs from final checkpoint")
    size(checkpoint.posterior_position, 2) == retained ||
        error("$label final-checkpoint retained draws differ")
    println("completed\t", label, "\tretained=", retained,
        "\tdivergences=", record.n_divergent_samples,
        "\ttotal_gradients=", record.total_gradient_evaluations,
        "\tsampling_gradients=", record.sampling_gradient_evaluations,
        "\tfit_seconds=", record.fit_seconds)
    flush(stdout)
    fit, record
end

function diagnostics(label, fit)
    q = fit.posterior_position
    samples = permutedims(reshape(q, size(q, 1), size(q, 2), 1), (2, 3, 1))
    (; fit=label, chains=1, retained_draws=size(q, 2),
       max_split_rhat=maximum(MCMCDiagnosticTools.rhat(samples)),
       min_bulk_ess=minimum(MCMCDiagnosticTools.ess(samples; kind=:bulk)),
       min_tail_ess=minimum(MCMCDiagnosticTools.ess(samples; kind=:tail)),
       divergences=fit.n_divergent_samples,
       divergence_percent=100fit.n_divergent_samples / size(q, 2),
       ess_coordinate_scope="mu, log(tau), theta_effect_1:8")
end

function coordinate_arrays(stan, fit)
    names = BS.param_unc_names(stan.density.model)
    index(name) = only(findall(==(name), names))
    block = only(adaptive_centering_blocks(stan.sb, names))
    effects = vec(block.effects)
    log_scale = only(block.log_scales)
    population = brm_layout(names).population
    (; names, effects=Int.(effects), log_scale, population,
       mu=fit.posterior_position[population, :], logtau=fit.posterior_position[log_scale, :],
       z=fit.posterior_position[effects, :])
end

function offline_selection(coordinates; candidates=0.0:0.01:1.0)
    J = size(coordinates.z, 1)
    losses = Matrix{Union{Missing,Float64}}(missing, length(candidates), J)
    admissible = Matrix{Bool}(undef, length(candidates), J)
    selected = Vector{Float64}(undef, J)
    for j in 1:J
        for (i, c) in enumerate(candidates)
            score = log(std(coordinates.z[j, :] .* exp.(c .* coordinates.logtau))) -
                    mean(c .* coordinates.logtau)
            losses[i, j] = if isfinite(score)
                admissible[i, j] = true
                score
            else
                admissible[i, j] = false
                missing
            end
        end
        finite = findall(admissible[:, j])
        isempty(finite) && error("school $j has no admissible offline candidate")
        selected[j] = candidates[finite[argmin(Vector(losses[finite, j]))]]
    end
    (; candidates=collect(candidates), selected, losses, admissible)
end

function export_coordinates(label, stan, fit, output_dir; controls=nothing)
    c = coordinate_arrays(stan, fit)
    # theta_effect is the physical school effect in every arm: mu + tau*z in a
    # noncentered frame, mu + tau^(1-c)*u in a selected-partial target frame.
    # The native sampler coordinate stays in noncentered_coordinate either way.
    physical = if isnothing(controls)
        c.z .* exp.(c.logtau)'
    else
        length(controls) == size(c.z, 1) ||
            error("centering controls must cover every school effect")
        stack([c.z[j, :] .* exp.((1 - controls[j]) .* c.logtau)
               for j in axes(c.z, 1)]; dims=1)
    end
    write_tsv(joinpath(output_dir, "$(label)_coordinates.tsv"), [
        (; draw=s, school=j, mu=c.mu[s], tau=exp(c.logtau[s]),
           theta_effect=c.mu[s] + physical[j, s],
           noncentered_coordinate=c.z[j, s], log_tau=c.logtau[s])
        for j in eachindex(c.effects) for s in eachindex(c.mu)])
    c
end

function export_selection(selection, output_dir)
    write_tsv(joinpath(output_dir, "offline_centeredness.tsv"), [
        (; school=j, centeredness=selection.selected[j]) for j in eachindex(selection.selected)])
    write_tsv(joinpath(output_dir, "offline_loss_profiles.tsv"), [
        (; school=j, centeredness=selection.candidates[i], loss=selection.losses[i, j],
           admissible=selection.admissible[i, j])
        for j in eachindex(selection.selected) for i in eachindex(selection.candidates)])
end

function fixed_partial_problem(stan, selected)
    adaptive = adaptive_centering_problem(stan.sb, stan.density, ENZYME_BACKEND)
    names = BS.param_unc_names(stan.density.model)
    block = only(adaptive_centering_blocks(stan.sb, names))
    by_index = Dict(idx => selected[j] for j in eachindex(selected)
                    for idx in (block.effects[1, j],))
    ir = WarmupHMC.reparametrizer(adaptive)
    restored = [idx => WarmupHMC.PartiallyCentered(by_index[idx]) for (idx, _) in ir.pairs]
    WarmupHMC.restore_reparam_sources!(adaptive, restored)
    [last(pair).c for pair in WarmupHMC.reparam_sources(adaptive)] == selected ||
        error("selected-partial controls were not restored in coordinate order")
    adaptive
end

function run_reproduction(; output_dir=get(ENV, "BRM_EIGHT_SCHOOLS_OUTPUT", mktempdir()))
    for label in ("noncentered", "partial", "online")
        isfile(joinpath(output_dir, "$label.jls")) && error(
            "Saved $label output exists; use a fresh directory or re-render existing figures.")
    end
    mkpath(output_dir)
    before = run_provenance(output_dir)
    data = read_eight_schools()
    write_tsv(joinpath(output_dir, "observations.tsv"), [
        (; school=j, estimate=data.y[j], standard_error=data.sigma[j])
        for j in 1:data.J])

    # One compiled BRM target serves all three arms. Its generated coordinates
    # are noncentered (`theta = mu + tau*z`); partial and online arms change only
    # WarmupHMC's sampler source frame.
    stan = stan_density("model", output_dir)
    pilot, pilot_record = sample_source_fit(stan.density, "noncentered", output_dir)
    pilot_coordinates = export_coordinates("noncentered", stan, pilot, output_dir)
    selection = offline_selection(pilot_coordinates)
    export_selection(selection, output_dir)

    partial_target = fixed_partial_problem(stan, selection.selected)
    partial, partial_record = sample_source_fit(
        partial_target, "partial", output_dir; nonlinear_adapt=false)
    # The serialized partial.jls is the actual selected source frame. Freeze it,
    # then apply WarmupHMC's one required source-to-target transform in place.
    partial_source = copy(partial.posterior_position)
    WarmupHMC.reparametrize!(partial_target, partial.posterior_position)
    partial_target_record = merge(partial_record,
        (; posterior_position=copy(partial.posterior_position)))
    serialize(joinpath(output_dir, "partial_target.jls"), partial_target_record)
    partial_coordinates = export_coordinates("partial", stan, partial, output_dir;
        controls=selection.selected)

    online_target = adaptive_centering_problem(stan.sb, stan.density, ENZYME_BACKEND)
    online, online_record = sample_source_fit(online_target, "online", output_dir)
    learned = [last(pair).c for pair in WarmupHMC.reparam_sources(online_target)]
    length(learned) == data.J || error("expected one adapted coordinate per school")
    all(c -> 0 <= c <= 1, learned) || error("online centering outside [0,1]")
    write_tsv(joinpath(output_dir, "online_centeredness.tsv"), [
        (; school=j, centeredness=learned[j]) for j in eachindex(learned)])
    online_coordinates = export_coordinates("online", stan, online, output_dir)

    rows = NamedTuple[]
    for (label, fit, record) in (
            ("noncentered", pilot, pilot_record),
            ("selected_partial", partial, partial_target_record),
            ("online", online, online_record))
        d = diagnostics(label, fit)
        push!(rows, merge(d,
            (; total_gradient_evaluations=record.total_gradient_evaluations,
              sampling_gradient_evaluations=record.sampling_gradient_evaluations,
              fit_seconds=record.fit_seconds,
              min_bulk_ess_per_total_gradient=d.min_bulk_ess /
                  record.total_gradient_evaluations,
              min_bulk_ess_per_sampling_gradient=d.min_bulk_ess /
                  record.sampling_gradient_evaluations,
              min_tail_ess_per_total_gradient=d.min_tail_ess /
                  record.total_gradient_evaluations,
              min_tail_ess_per_sampling_gradient=d.min_tail_ess /
                  record.sampling_gradient_evaluations)))
    end
    write_tsv(joinpath(output_dir, "diagnostics.tsv"), rows)
    write_tsv(joinpath(output_dir, "fit_costs.tsv"), [
        (; fit=row.fit, elapsed_seconds=row.fit_seconds,
         total_evaluation_counter=row.total_gradient_evaluations,
         sampling_evaluation_counter=row.sampling_gradient_evaluations,
         counter_scope="MCMC run total excludes Pathfinder/setup; sampling counts retained appended transitions")
        for row in rows])
    workflow = (; configuration="pilot_then_selected_partial_refit",
        elapsed_seconds=pilot_record.fit_seconds + partial_target_record.fit_seconds,
        total_evaluation_counter=pilot_record.total_gradient_evaluations +
            partial_target_record.total_gradient_evaluations,
        sampling_evaluation_counter=pilot_record.sampling_gradient_evaluations +
            partial_target_record.sampling_gradient_evaluations)
    write_tsv(joinpath(output_dir, "workflow_costs.tsv"), [workflow])
    write_tsv(joinpath(output_dir, "centeredness.tsv"), [
        (; school=j, offline=selection.selected[j], online=learned[j])
        for j in 1:data.J])

    # Bind the saved draws to the immutable source target as well as the synthetic audit.
    source = source_model(output_dir)
    source_names = BS.param_unc_names(source)
    brm_names = BS.param_unc_names(stan.density.model)
    saved_receipts = NamedTuple[]
    for draw in round.(Int, range(1, size(pilot.posterior_position, 2); length=16))
        q_brm = collect(pilot.posterior_position[:, draw])
        layout = brm_layout(brm_names)
        logtau = q_brm[layout.scale]
        tau = exp(logtau)
        mu = q_brm[layout.population]
        q_source = zeros(length(source_names))
        for (i, name) in enumerate(source_names)
            q_source[i] = name == "mu" ? mu :
                name == "tau" ? logtau :
                mu + tau * q_brm[layout.effects[parse(Int, last(split(name, '.')))]]
        end
        source_value, source_gradient = BS.log_density_gradient(
            source, q_source; propto=false, jacobian=true)
        brm_value, brm_gradient = BS.log_density_gradient(
            stan.density.model, q_brm; propto=false, jacobian=true)
        physical_gradient = brm_physical_gradient(
            brm_gradient, brm_names, q_brm, source, q_source, source_names)
        transformed_brm_value = brm_value - length(layout.effects) * logtau
        abs(transformed_brm_value - source_value) <= 1e-9 ||
            error("saved-pilot source density mismatch at draw $draw")
        maximum(abs.(physical_gradient .- source_gradient)) <= 1e-8 ||
            error("saved-pilot source gradient mismatch at draw $draw")
        push!(saved_receipts, (; draw, source_value, brm_value, transformed_brm_value,
            density_absolute_error=abs(transformed_brm_value - source_value),
            max_gradient_absolute_error=maximum(abs.(physical_gradient .- source_gradient))))
    end
    write_tsv(joinpath(output_dir, "saved_source_density_gradient_audit.tsv"), saved_receipts)

    before == package_snapshot() || error("a dependency checkout moved during the run")
    println("eight_schools_reproduction_complete\t", output_dir)
    foreach(row -> println("diagnostic\t", row), rows)
    (; stan, pilot, partial, partial_source, online, selection, learned, rows, output_dir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    if get(ENV, "BRM_EIGHT_SCHOOLS_RUNTIME", "0") == "1"
        run_reproduction()
    else
        println("No sampling requested. Set BRM_EIGHT_SCHOOLS_RUNTIME=1 for the full study.")
    end
end
