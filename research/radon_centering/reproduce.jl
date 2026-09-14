using BayesianRegressionModels
using BridgeStan
import DifferentiationInterface as DI
import Enzyme
using Distributions
using JSON
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
using ZipFile
import Pkg

const BRM = BayesianRegressionModels
const BS = BridgeStan
const RESEARCH_DIR = @__DIR__
const POSTERIORDB_REVISION = "5545a1dd07ae297c36edecbcd82aa49097b4c385"
const POSTERIOR_NAME = "radon_all-radon_variable_intercept_slope_noncentered"
const MODEL_NAME = "radon_variable_intercept_slope_noncentered"
const DATA_NAME = "radon_all"
const DATA_ZIP_SHA256 = "3f30c7909d530be01e70ab9e98f9f5d5e83371bb15c6dd168696aefd805b5672"
const DATA_JSON_SHA256 = "05cac39913090df5430e3f769d8eb8d2ee4e6f0a3c5286c9e767f37926d51ffd"
const SOURCE_STAN_SHA256 = "b37c4aebfe629591a306bbd630e74528ae9c9c1b23d00726c6633247927f34cb"
const N_DRAWS = 10_000
const SEED = 1
const OFFLINE_CANDIDATES = 0:0.01:1
const ONLINE_CANDIDATES = 0:0.1:1
const ENZYME_BACKEND = DI.AutoEnzyme(;
    mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
    function_annotation=Enzyme.Const)
function write_tsv(path, rows)
    isempty(rows) && error("cannot write an empty table: $path")
    keys = propertynames(first(rows))
    value(x) = ismissing(x) ? "missing" : string(x)
    open(path, "w") do io
        println(io, join(keys, '\t'))
        for row in rows
            println(io, join((value(getproperty(row, key)) for key in keys), '\t'))
        end
    end
end

function read_radon()
    zip_path = joinpath(RESEARCH_DIR, "reference", "$DATA_NAME.json.zip")
    bytes2hex(sha256(read(zip_path))) == DATA_ZIP_SHA256 ||
        error("PosteriorDB radon_all archive hash mismatch")
    reader = ZipFile.Reader(zip_path)
    try
        length(reader.files) == 1 || error("radon_all archive must contain one JSON file")
        raw = read(reader.files[1], String)
        bytes2hex(sha256(raw)) == DATA_JSON_SHA256 ||
            error("PosteriorDB radon_all JSON hash mismatch")
        parsed = JSON.parse(raw)
        required = ("N", "J", "floor_measure", "log_radon", "log_uppm", "county_idx")
        all(key -> haskey(parsed, key), required) || error("radon_all JSON is incomplete")
        N = parsed["N"]
        J = parsed["J"]
        floor_measure = Float64.(parsed["floor_measure"])
        log_radon = Float64.(parsed["log_radon"])
        log_uppm = Float64.(parsed["log_uppm"])
        county_idx = Int.(parsed["county_idx"])
        N == 12_573 && J == 386 || error("unexpected radon_all dimensions")
        all(length == N for length in (
            length(floor_measure), length(log_radon), length(log_uppm), length(county_idx))) ||
            error("radon_all vectors do not match N")
        all(isfinite, floor_measure) && all(isfinite, log_radon) && all(isfinite, log_uppm) ||
            error("radon_all contains non-finite covariates or responses")
        sort(unique(county_idx)) == collect(1:J) ||
            error("radon_all county indices are not dense over 1:J")
        (; N, J, floor_measure, log_radon, log_uppm, county_idx)
    finally
        close(reader)
    end
end

const RADON_DATA = read_radon()

# Fixed population coefficients are the source model's mu_alpha and mu_beta. The
# two scalar zerocorr blocks are its independent alpha and beta vectors. A
# constant-one slope column spells the intercept margin as the same direct-scale
# scalar family as the slope; both scales are half-normal(0, 1), as in PosteriorDB.
const RADON_NCP = @brm begin
    sigma_y ~ Normal(0, 1; lower=0)
    mu ~ 1 + floor_measure +
          (0 + intercept + floor_measure || county_idx)
    effect(mu, Intercept) ~ Normal(0, 10)
    effect(mu, floor_measure) ~ Normal(0, 10)
    log_radon ~ Normal(mu, sigma_y)
end

function build_model(data)
    RADON_NCP((; data.floor_measure, data.county_idx, data.log_radon,
               intercept=fill(1.0, data.N)))
end

function stan_density(label, output_dir)
    mkpath(output_dir)
    data = RADON_DATA
    sb = SBBRMI(build_model(data); mod=@__MODULE__)
    source = BRM.stan_code(sb)
    checked = StanBlocks.stanc_check(source)
    checked.ok || error("stanc failed for radon $label\n$(checked.output)")
    density = StanBlocks.stan_instantiate(sb.model;
        path=joinpath(output_dir, "radon-$label.stan"))
    q = zeros(LogDensityProblems.dimension(density))
    value, gradient = LogDensityProblems.logdensity_and_gradient(density, q)
    isfinite(value) && all(isfinite, gradient) || error("non-finite BRM Stan density")
    (; density, sb, data)
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
    data = read_radon()
    packages = package_snapshot()
    write_tsv(joinpath(output_dir, "packages.tsv"), packages)
    metadata = Dict(
        "posteriordb_revision" => POSTERIORDB_REVISION,
        "posterior_name" => POSTERIOR_NAME,
        "model_name" => MODEL_NAME,
        "data_name" => DATA_NAME,
        "data_zip_sha256" => DATA_ZIP_SHA256,
        "data_json_sha256" => DATA_JSON_SHA256,
        "source_stan_sha256" => SOURCE_STAN_SHA256,
        "observations" => data.N,
        "counties" => data.J,
        "hierarchical_effect_vectors" => 2,
        "hierarchical_effect_cells" => 2data.J,
        "total_scalar_parameters" => 2data.J + 5,
        "julia_version" => string(VERSION),
        "brm_commit" => strip(read(`git -C $RESEARCH_DIR rev-parse HEAD`, String)),
        "script_sha256" => bytes2hex(sha256(read(@__FILE__))),
        "sampler" => "WarmupHMC.adaptive_warmup_mcmc",
        "seed_each_fit" => SEED,
        "draws_requested_each_fit" => N_DRAWS,
        "chains_each_fit" => 1,
        "monitor_ess" => true,
        "normal_initializer_and_adaptation" => true,
        "offline_candidates" => "0:0.01:1",
        "online_candidates" => "0:0.1:1",
        "offline_objective" => "log sample standard deviation minus mean centered log scale",
        "online_objective" => "WarmupHMC weighted position-gradient correlation, default w1=0",
        "partial_refit_nonlinear_adapt" => false,
        "turing_sampling" => false,
        "r_sampling" => false,
        "blas_threads" => BLAS.get_num_threads(),
        "timing_scope" => "sampler call only; Stan compilation and setup are outside fit_seconds",
        "counter_scope" => "NUTS including warmup and discarded epochs; excludes Pathfinder/setup; sampling counter counts retained appended transitions",
        "diagnostics" => "rank-normalized split Rhat; bulk ESS; tail ESS; retained divergences",
        "rhat_scope" => "within one split chain, not independent-chain convergence",
    )
    open(joinpath(output_dir, "provenance.toml"), "w") do io
        TOML.print(io, metadata)
    end
    packages
end

function require_fresh_fit_outputs(output_dir, labels)
    for label in labels
        isfile(joinpath(output_dir, "$label.jls")) && error(
            "Saved $label draws exist; use a fresh output directory. Existing results are immutable.")
    end
end

function sample_fit(target, label, output_dir; nonlinear_adapt=true)
    result_path = joinpath(output_dir, "$label.jls")
    isfile(result_path) && error("A completed $label fit already exists; do not resample.")
    callback = (state, stage) -> begin
        println("boundary\t", label, '\t', stage,
                "\twindow=", state.outer_counter,
                "\trestart=", state.restart,
                "\tevaluations=", state.total_evaluation_counter)
        flush(stdout)
        isfile(joinpath(output_dir, "STOP"))
    end
    println("sampling\t", label, "\tseed=1\tn_draws=10000",
            "\tnonlinear_adapt=", nonlinear_adapt,
            "\tWarmupHMC defaults otherwise")
    flush(stdout)
    started_ns = time_ns()
    fit = WarmupHMC.adaptive_warmup_mcmc(
        Xoshiro(SEED), target; n_draws=N_DRAWS, monitor_ess=true, callback,
        checkpoint_dir=joinpath(output_dir, "checkpoints-$label"), nonlinear_adapt)
    fit_seconds = (time_ns() - started_ns) / 1e9
    retained = size(fit.posterior_position, 2)
    total = fit.total_evaluation_counter
    sampling = fit.sampling_evaluation_counter
    record = (; posterior_position=convert(Matrix{Float64}, fit.posterior_position),
        n_divergent_samples=fit.n_divergent_samples, seed=SEED, fit_seconds,
        total_gradient_evaluations=total,
        sampling_gradient_evaluations=sampling,
        requested_draws=N_DRAWS, complete=retained >= N_DRAWS,
        nonlinear_adapt)
    serialize(result_path, record)
    record.complete || error(
        "Stopped early; saved $retained draws, not a completed case study.")
    0 < sampling <= total || error("Invalid sampling/total gradient counters for $label")
    checkpoint = deserialize(joinpath(output_dir, "checkpoints-$label", "cp_latest.jls"))
    checkpoint.total_evaluation_counter == total ||
        error("Fit and final checkpoint total counters disagree for $label")
    checkpoint.sampling_evaluation_counter == sampling ||
        error("Fit and final checkpoint sampling counters disagree for $label")
    println("completed\t", label, "\tretained=", retained,
            "\tdivergences=", record.n_divergent_samples,
            "\ttotal_gradients=", total,
            "\tsampling_gradients=", sampling,
            "\tfit_seconds=", fit_seconds)
    flush(stdout)
    record
end

function diagnostics(label, fit)
    q = fit.posterior_position
    samples = permutedims(reshape(q, size(q, 1), size(q, 2), 1), (2, 3, 1))
    (; fit=label, chains=1, retained_draws=size(q, 2),
       max_split_rhat=maximum(MCMCDiagnosticTools.rhat(samples)),
       min_bulk_ess=minimum(MCMCDiagnosticTools.ess(samples; kind=:bulk)),
       min_tail_ess=minimum(MCMCDiagnosticTools.ess(samples; kind=:tail)),
       divergences=fit.n_divergent_samples,
       divergence_percent=100fit.n_divergent_samples / size(q, 2))
end

function effect_blocks(sb, unc_names)
    blocks = adaptive_centering_blocks(sb, unc_names)
    length(blocks) == 2 || error("expected independent intercept and slope adaptive blocks")
    roles = (:intercept, :slope)
    [(; role=roles[i], block=blocks[i]) for i in eachindex(blocks)]
end

function select_offline(sb, fit, output_dir, unc_names)
    rows = NamedTuple[]
    profiles = NamedTuple[]
    selected = Dict{Int,Float64}()
    for entry in effect_blocks(sb, unc_names)
        block = entry.block
        coordinates = vec(block.effects)
        log_scale_index = only(block.log_scales)
        z = fit.posterior_position[coordinates, :]
        log_scales = repeat(
            reshape(fit.posterior_position[log_scale_index, :], 1, :),
            length(coordinates), 1)
        selection = select_ranef_centeredness(
            permutedims(z), permutedims(log_scales); candidates=OFFLINE_CANDIDATES)
        for county in axes(block.effects, 2)
            index = block.effects[1, county]
            selected[index] = selection.centeredness[county]
            push!(rows, (; role=entry.role, county, centeredness=selection.centeredness[county]))
        end
        for candidate_index in eachindex(selection.candidates),
                county in axes(block.effects, 2)
            push!(profiles, (; role=entry.role, county,
                centeredness=selection.candidates[candidate_index],
                loss=selection.losses[candidate_index, county],
                admissible=selection.admissible[candidate_index, county]))
        end
    end
    length(selected) == 2RADON_DATA.J || error("offline selection did not cover every effect cell")
    write_tsv(joinpath(output_dir, "selected_centeredness.tsv"), rows)
    write_tsv(joinpath(output_dir, "offline_loss_profiles.tsv"), profiles)
    (; rows, profiles, selected)
end

function common_reference_online_losses(sb, density, fit, output_dir)
    unc_names = String.(BS.param_unc_names(density.model))
    adaptive = adaptive_centering_problem(sb, density, ENZYME_BACKEND)
    gradients = permutedims(reduce(hcat, [
        last(LogDensityProblems.logdensity_and_gradient(density, collect(q)))
        for q in eachcol(fit.posterior_position)]))
    all(isfinite, gradients) || error("non-finite common-reference pilot gradients")
    scored = candidate_scoring_losses(
        adaptive, fit.posterior_position, permutedims(gradients))
    rows = NamedTuple[]
    for score in scored, entry in effect_blocks(sb, unc_names)
        county = findfirst(isequal(score.index), vec(entry.block.effects))
        isnothing(county) && continue
        push!(rows, (; role=entry.role, county=Int(county),
            centeredness=score.candidate, loss=score.loss,
            groups=score.groups, effective_n=score.effective_n,
            evidence="retrospective_common_reference_pilot",
            weights="unit", objective="position_gradient_correlation_w1_0"))
    end
    length(rows) == 2RADON_DATA.J * length(ONLINE_CANDIDATES) ||
        error("common-reference online scorer did not cover every cell and candidate")
    write_tsv(joinpath(output_dir, "retrospective_online_losses.tsv"), rows)
    (; rows, gradients)
end

function fixed_partial_problem(sb, density, selected)
    target = adaptive_centering_problem(sb, density, ENZYME_BACKEND)
    sources = WarmupHMC.reparam_sources(target)
    length(sources) == length(selected) ||
        error("selected centering and adaptive coordinates differ")
    WarmupHMC.restore_reparam_sources!(target, [
        index => WarmupHMC.PartiallyCentered(selected[index])
        for (index, _) in sources])
    target
end

function run_reproduction(; output_dir=get(ENV, "BRM_RADON_OUTPUT", mktempdir()))
    require_fresh_fit_outputs(output_dir, ("noncentered", "partial"))
    mkpath(output_dir)
    before = run_provenance(output_dir)
    stan = stan_density("noncentered", output_dir)
    pilot = sample_fit(stan.density, "noncentered", output_dir)
    unc_names = String.(BS.param_unc_names(stan.density.model))
    offline = select_offline(stan.sb, pilot, output_dir, unc_names)
    common_reference_online_losses(stan.sb, stan.density, pilot, output_dir)
    partial_target = fixed_partial_problem(stan.sb, stan.density, offline.selected)
    cp(joinpath(output_dir, "radon-noncentered.stan"),
       joinpath(output_dir, "radon-partial.stan"); force=false)
    refit = sample_fit(partial_target, "partial", output_dir; nonlinear_adapt=false)
    rows = [diagnostics("noncentered", pilot), diagnostics("selected_partial", refit)]
    write_tsv(joinpath(output_dir, "diagnostics.tsv"), rows)
    before == package_snapshot() ||
        error("A dependency checkout moved during this run; inspect packages.tsv.")
    println("radon_reproduction_complete\t", output_dir)
    foreach(row -> println("diagnostic\t", row), rows)
    (; stan, pilot, refit, offline, output_dir)
end

function run_online(; output_dir=get(ENV, "BRM_RADON_OUTPUT", mktempdir()))
    require_fresh_fit_outputs(output_dir, ("online",))
    mkpath(output_dir)
    before = run_provenance(output_dir)
    stan = stan_density("online", output_dir)
    online = adaptive_centering_problem(stan.sb, stan.density, ENZYME_BACKEND)
    fit = sample_fit(online, "online", output_dir)
    learned = WarmupHMC.reparam_sources(online)
    length(learned) == 2RADON_DATA.J ||
        error("online run did not retain every effect-cell centering")
    rows = NamedTuple[]
    unc_names = String.(BS.param_unc_names(stan.density.model))
    for (index, value) in learned, entry in effect_blocks(stan.sb, unc_names)
        county = findfirst(isequal(index), vec(entry.block.effects))
        isnothing(county) && continue
        push!(rows, (; role=entry.role, county=Int(county), centeredness=value.c))
    end
    length(rows) == 2RADON_DATA.J || error("online centering export is incomplete")
    write_tsv(joinpath(output_dir, "online_centeredness.tsv"), rows)
    write_tsv(joinpath(output_dir, "online_diagnostics.tsv"), [diagnostics("online", fit)])
    before == package_snapshot() ||
        error("A dependency checkout moved during this run; inspect packages.tsv.")
    println("radon_online_complete\t", output_dir)
    (; stan, online, fit, rows, output_dir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    any(haskey(ENV, key) for key in ("BRM_RADON_DRAWS", "BRM_RADON_EVALS", "BRM_RADON_CHAINS")) &&
        error("This is the full 10,000-draw PosteriorDB case study; reduced overrides are rejected.")
    get(ENV, "BRM_RADON_TURING", "0") == "1" &&
        error("This study samples through BRM -> StanBlocks/BridgeStan, not Turing.")
    if get(ENV, "BRM_RADON_ONLINE", "0") == "1"
        run_online()
    elseif get(ENV, "BRM_RADON_RUNTIME", "0") == "1"
        run_reproduction()
    else
        println("No sampling requested. Set BRM_RADON_RUNTIME=1 for the full pilot/refit study.")
    end
end
