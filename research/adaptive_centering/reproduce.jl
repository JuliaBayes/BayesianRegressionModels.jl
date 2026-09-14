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
const SOURCE_REVISION = "0d00b8535e2c20c49017d03c7b060940eb8e7041"
const DATA_REVISION = "1dcc2bf5f955cc1224a3e1307256e1fe86b68dae"
const DATA_SHA256 = "b89a1e4eb0391a982b32be3e378df00e8593ff9971e9425e9c5d7929b74f9801"
const RESEARCH_DIR = @__DIR__
const DEFAULT_K = 20
const L = 1.5
const SOURCE_DRAWS = 10_000
const SOURCE_SEED = 1
const ENZYME_BACKEND = DI.AutoEnzyme(;
    mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
    function_annotation=Enzyme.Const)

function read_mcycle(path=joinpath(RESEARCH_DIR, "mcycle.csv"))
    bytes2hex(sha256(read(path))) == DATA_SHA256 || error("mcycle source hash mismatch")
    lines = readlines(path)
    first(lines) == "rownames,times,accel" || error("unexpected mcycle header")
    rows = split.(lines[2:end], ',')
    times = parse.(Float64, getindex.(rows, 2))
    accel = parse.(Float64, getindex.(rows, 3))
    length(times) == 133 || error("expected 133 motorcycle observations")
    (; times, accel)
end

function prepared_data(; k=DEFAULT_K, c_mu=zeros(k), c_sigma=zeros(k))
    k == DEFAULT_K || error("The source case study has 20 basis functions per GP.")
    source = read_mcycle()
    xmin, xmax = extrema(source.times)
    x = @. -1 + 2 * (source.times - xmin) / (xmax - xmin)
    y_scale = std(source.accel)
    y = source.accel ./ y_scale
    (; x, y, c_mu=Float64.(c_mu), c_sigma=Float64.(c_sigma),
       times=source.times, accel=source.accel, y_scale)
end

# BEGIN ADAPTIVE MOTORCYCLE MODEL
const MOTORCYCLE_NCP = @brm begin
    length_scale(mu, hsgp(x)) ~ LogNormal(0, 4)
    sd(mu, hsgp(x)) ~ LogNormal(0, 4)
    length_scale(sigma, hsgp(x)) ~ LogNormal(0, 4)
    sd(sigma, hsgp(x)) ~ LogNormal(0, 4)
    mu ~ hsgp(x; k=20, domain=(-1.5, 1.5))
    log(sigma) ~ hsgp(x; k=20, domain=(-1.5, 1.5))
    y ~ Normal(mu, sigma)
end

const MOTORCYCLE_PARTIAL = @brm begin
    length_scale(mu, hsgp(x)) ~ LogNormal(0, 4)
    sd(mu, hsgp(x)) ~ LogNormal(0, 4)
    length_scale(sigma, hsgp(x)) ~ LogNormal(0, 4)
    sd(sigma, hsgp(x)) ~ LogNormal(0, 4)
    mu ~ hsgp(x; k=20, domain=(-1.5, 1.5), centeredness=c_mu)
    log(sigma) ~ hsgp(x; k=20, domain=(-1.5, 1.5), centeredness=c_sigma)
    y ~ Normal(mu, sigma)
end
# END ADAPTIVE MOTORCYCLE MODEL

model_data(data) = (; x=data.x, y=data.y, c_mu=data.c_mu, c_sigma=data.c_sigma)
function build_brmi(data, k=DEFAULT_K; partial=false)
    k == DEFAULT_K || error("The source case study has k=20.")
    (partial ? MOTORCYCLE_PARTIAL : MOTORCYCLE_NCP)(model_data(data))
end

function stan_density(brmi, label, output_dir)
    mkpath(output_dir)
    sb = SBBRMI(brmi; mod=@__MODULE__)
    checked = StanBlocks.stanc_check(BRM.stan_code(sb))
    checked.ok || error("stanc failed for $label\n$(checked.output)")
    density = StanBlocks.stan_instantiate(sb.model;
        path=joinpath(output_dir, "motorcycle-$label.stan"))
    q = zeros(LogDensityProblems.dimension(density))
    value, grad = LogDensityProblems.logdensity_and_gradient(density, q)
    isfinite(value) && all(isfinite, grad) || error("non-finite Stan density")
    (; density, q, sb)
end

function write_tsv(path, rows)
    isempty(rows) && error("cannot write an empty table: $path")
    keys = propertynames(first(rows))
    open(path, "w") do io
        println(io, join(keys, '\t'))
        for row in rows
            println(io, join((getproperty(row, key) for key in keys), '\t'))
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
        "source_revision" => SOURCE_REVISION, "data_revision" => DATA_REVISION,
        "data_sha256" => DATA_SHA256, "julia_version" => string(VERSION),
        "brm_commit" => strip(read(`git -C $RESEARCH_DIR rev-parse HEAD`, String)),
        "script_sha256" => bytes2hex(sha256(read(@__FILE__))),
        "basis_functions_per_gp" => DEFAULT_K, "seed_each_fit" => SOURCE_SEED,
        "draws_requested_each_fit" => SOURCE_DRAWS, "chains_each_fit" => 1,
        "sampler" => "WarmupHMC.adaptive_warmup_mcmc",
        "sampler_config" => "defaults; n_draws=10000; monitor_ess=true (source progress enables this)",
        "turing_sampling" => "disabled pending measured value/runtime parity",
        "blas_threads" => BLAS.get_num_threads(),
        "diagnostics" => "rank-normalized split Rhat; bulk ESS; tail ESS; retained divergences",
        "rhat_scope" => "within one split chain, not independent-chain convergence",
    )
    open(joinpath(output_dir, "provenance.toml"), "w") do io
        TOML.print(io, metadata)
    end
    packages
end

# The article's only sampling options are seed 1 and n_draws=10000.
# Its progress display enables monitor_ess. Callback/checkpoints below are
# observational; no initializer, adaptation or NUTS settings are overridden.
function sample_source_fit(target, label, output_dir)
    result_path = joinpath(output_dir, "$label.jls")
    isfile(result_path) && error(
        "A completed $label fit already exists. Use render_results to replot it; do not resample.")
    callback = (state, stage) -> begin
        println("boundary\t", label, '\t', stage, "\twindow=", state.outer_counter,
            "\tretained=", size(state.posterior_position, 2))
        flush(stdout)
        isfile(joinpath(output_dir, "STOP"))
    end
    println("sampling\t", label, "\tseed=1\tn_draws=10000\tWarmupHMC defaults")
    flush(stdout)
    fit = WarmupHMC.adaptive_warmup_mcmc(
        Xoshiro(SOURCE_SEED), target; n_draws=SOURCE_DRAWS, monitor_ess=true,
        callback, checkpoint_dir=joinpath(output_dir, "checkpoints-$label"))
    retained = size(fit.posterior_position, 2)
    record = (; posterior_position=convert(Matrix{Float64}, fit.posterior_position),
        n_divergent_samples=fit.n_divergent_samples, seed=SOURCE_SEED,
        requested_draws=SOURCE_DRAWS, complete=retained >= SOURCE_DRAWS)
    serialize(result_path, record)
    record.complete || error("Stopped early; saved $retained draws, not a completed case study.")
    println("completed\t", label, "\tretained=", retained,
        "\tdivergences=", record.n_divergent_samples)
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

function constrained_draws(stan, fit)
    names = BS.param_names(stan.density.model)
    draws = reduce(hcat, (BS.param_constrain(stan.density.model, collect(q))
                         for q in eachcol(fit.posterior_position)))
    names, draws
end

log_spectral_scale(sigma, rho, omega2) =
    log(sigma) + 0.25log(2pi) + 0.5log(rho) - 0.25rho^2 * omega2

function gp_draws(stan, fit, data; partial=false)
    names, draws = constrained_draws(stan, fit)
    index = Dict(names .=> eachindex(names))
    map((("mu", "hsgp_x", data.c_mu), ("log_sigma", "hsgp_log_sigma_x", data.c_sigma))) do (name, prefix, c)
        rho = draws[index["$(prefix)_rho_iso"], :]
        sigma = draws[index["$(prefix)_sigma"], :]
        coordinate_name = partial ? "beta_partial" : "beta_raw"
        coordinates = hcat((draws[index["$(prefix)_$(coordinate_name).$j"], :]
                            for j in 1:DEFAULT_K)...)
        logs = hcat((log_spectral_scale.(sigma, rho, (j*pi/(2L))^2)
                     for j in 1:DEFAULT_K)...)
        weights = coordinates .* exp.(logs .* (1 .- c'))
        (; name, rho, sigma, coordinates, logs, weights)
    end
end

function pilot_selection(gps)
    map(gps) do gp
        select_hsgp_centeredness(gp.coordinates, gp.logs; candidates=0:0.01:1)
    end
end

function export_fit(stan, fit, data, label, output_dir; partial=false)
    gps = gp_draws(stan, fit, data; partial)
    phi = [sin(pi / (2L) * (x + L) * j) / sqrt(L) for x in data.x, j in 1:DEFAULT_K]
    curves = NamedTuple[]
    for gp in gps
        values = phi * gp.weights'
        gp.name == "log_sigma" && (values = exp.(values))
        # Exactly the source's standardized response units and observed times.
        for i in eachindex(data.times)
            qs = quantile(view(values, i, :), [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
            push!(curves, (; predictor=gp.name, time=data.times[i],
                q05=qs[1], q10=qs[2], q25=qs[3], q50=qs[4],
                q75=qs[5], q90=qs[6], q95=qs[7]))
        end
    end
    write_tsv(joinpath(output_dir, "$(label)_curves.tsv"), curves)
    # The same full draws feed all scatter panels; no independent CP fit.
    for gp in gps
        write_tsv(joinpath(output_dir, "$(label)_$(gp.name)_weights.tsv"), [
            (; draw=s, basis=j, rho=gp.rho[s], sigma=gp.sigma[s],
               coordinate=gp.coordinates[s, j], log_spectral_scale=gp.logs[s, j],
               physical_weight=gp.weights[s, j])
            for j in (1, 2, 19, 20) for s in axes(gp.coordinates, 1)])
    end
    gps
end

function export_selection(selected, output_dir)
    write_tsv(joinpath(output_dir, "centeredness.tsv"), [
        (; basis=j, mean=selected[1].centeredness[j], log_scale=selected[2].centeredness[j])
        for j in 1:DEFAULT_K])
    write_tsv(joinpath(output_dir, "loss_profiles.tsv"), [
        (; predictor=name, basis=j, centeredness=s.candidates[i],
           loss=s.losses[i, j], admissible=s.admissible[i, j])
        for (name, s) in zip(("mu", "log_sigma"), selected)
        for j in 1:DEFAULT_K for i in eachindex(s.candidates)])
end

function render_results(output_dir)
    run(`Rscript $(joinpath(RESEARCH_DIR, "plot_results.R")) $output_dir`)
end

function run_reproduction(; output_dir=get(ENV, "BRM_ADAPTIVE_OUTPUT", mktempdir()))
    mkpath(output_dir)
    before = run_provenance(output_dir)
    data = prepared_data()
    write_tsv(joinpath(output_dir, "observations.tsv"), [
        (; time=data.times[i], acceleration_scaled=data.y[i]) for i in eachindex(data.times)])
    ncp = stan_density(build_brmi(data), "noncentered", output_dir)
    pilot = sample_source_fit(ncp.density, "noncentered", output_dir)
    gps = export_fit(ncp, pilot, data, "noncentered", output_dir)
    selected = pilot_selection(gps)
    export_selection(selected, output_dir)
    partial_data = prepared_data(;
        c_mu=selected[1].centeredness, c_sigma=selected[2].centeredness)
    partial = stan_density(build_brmi(partial_data; partial=true), "partial", output_dir)
    refit = sample_source_fit(partial.density, "partial", output_dir)
    export_fit(partial, refit, partial_data, "partial", output_dir; partial=true)
    rows = [diagnostics("noncentered", pilot), diagnostics("selected_partial", refit)]
    write_tsv(joinpath(output_dir, "diagnostics.tsv"), rows)
    after = package_snapshot()
    before == after || error("A dependency checkout moved during this run; inspect packages.tsv.")
    render_results(output_dir)
    println("source_reproduction_complete\t", output_dir)
    foreach(row -> println("diagnostic\t", row), rows)
    (; pilot, refit, selected, ncp, partial, data, output_dir)
end

function run_online_stanblocks(; output_dir=get(ENV, "BRM_ADAPTIVE_OUTPUT", mktempdir()))
    mkpath(output_dir)
    run_provenance(output_dir)
    data = prepared_data()
    stan = stan_density(build_brmi(data), "online", output_dir)
    online = adaptive_centering_problem(stan.sb, stan.density, ENZYME_BACKEND)
    fit = sample_source_fit(online, "online", output_dir)
    learned = [v.c for (_, v) in WarmupHMC.reparam_sources(online)]
    length(learned) == 2DEFAULT_K || error("expected 40 target-scoped HSGP coordinates")
    write_tsv(joinpath(output_dir, "online_centeredness.tsv"), [
        (; predictor=i <= DEFAULT_K ? "mu" : "log_sigma", basis=mod1(i, DEFAULT_K),
           centeredness=learned[i]) for i in eachindex(learned)])
    # Returned online draws are already in the original NCP target frame.
    export_fit(stan, fit, data, "online", output_dir)
    write_tsv(joinpath(output_dir, "online_diagnostics.tsv"), [diagnostics("online", fit)])
    (; fit, learned, stan, online, output_dir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    any(haskey(ENV, key) for key in (
        "BRM_ADAPTIVE_K", "BRM_ADAPTIVE_DRAWS", "BRM_ADAPTIVE_EVALS", "BRM_ADAPTIVE_CHAINS")) &&
        error("This is the original k=20, 10000-draw case study. Reduced-budget overrides are no longer accepted.")
    get(ENV, "BRM_ADAPTIVE_TURING_ONLINE", "0") == "1" &&
        error("Turing sampling is disabled until numerical and runtime gradient parity is verified.")
    if get(ENV, "BRM_ADAPTIVE_ONLINE", "0") == "1"
        run_online_stanblocks()
    elseif get(ENV, "BRM_ADAPTIVE_RUNTIME", "0") == "1"
        run_reproduction()
    elseif get(ENV, "BRM_ADAPTIVE_RENDER", "0") == "1"
        render_results(ENV["BRM_ADAPTIVE_OUTPUT"])
    else
        println("No sampling requested. Set BRM_ADAPTIVE_RUNTIME=1 for the full source reproduction.")
    end
end
