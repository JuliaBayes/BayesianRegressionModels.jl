using BayesianRegressionModels
using BridgeStan
import DifferentiationInterface as DI
using Distributions: LogNormal, Normal
import Enzyme
using LinearAlgebra: BLAS
using LogDensityProblems
using Pkg
using Profile
using Serialization: deserialize
using SHA: sha256
using StanBlocks
using Statistics: median, quantile, std
using Test: @inferred
using Turing
using WarmupHMC

const BRM = BayesianRegressionModels
const BS = BridgeStan
const DP = Turing.DynamicPPL
const K = parse(Int, get(ENV, "BRM_HSGP_BENCH_K", "20"))
const WARMUP = parse(Int, get(ENV, "BRM_HSGP_BENCH_WARMUP", "1000"))
const SAMPLES = parse(Int, get(ENV, "BRM_HSGP_BENCH_SAMPLES", "21"))
const BATCH = parse(Int, get(ENV, "BRM_HSGP_BENCH_BATCH", "1000"))
const CACHE = get(
    ENV, "BRM_HSGP_BENCH_CACHE",
    joinpath(get(ENV, "TMPDIR", tempdir()), "brm-hsgp-gradient-benchmark"),
)
const OUTPUT = get(ENV, "BRM_HSGP_BENCH_OUTPUT", "")
const GEOMETRIES = Symbol.(split(get(
    ENV, "BRM_HSGP_BENCH_GEOMETRIES", "noncentered,centered,partial"), ','))
const DRAW_DIR = get(ENV, "BRM_HSGP_BENCH_DRAW_DIR", "")
const DRAW_COLUMNS = parse.(Int, split(get(
    ENV, "BRM_HSGP_BENCH_DRAW_COLUMNS", "1,2500,5000,7500,10000"), ','))
const PROFILE_ALLOCATIONS = get(
    ENV, "BRM_HSGP_BENCH_PROFILE_ALLOCATIONS", "false") == "true"
const PROFILE_CPU = get(
    ENV, "BRM_HSGP_BENCH_PROFILE_CPU", "false") == "true"
const RUNTIME_ACTIVITY = get(
    ENV, "BRM_HSGP_BENCH_RUNTIME_ACTIVITY", "true") == "true"
const ENZYME_BACKEND = RUNTIME_ACTIVITY ? DI.AutoEnzyme(;
    mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
    function_annotation=Enzyme.Const) :
    DI.AutoEnzyme(; function_annotation=Enzyme.Const)
const RUNTIME_RATIO_TARGET = 1.25

function dependency_receipt(name)
    info = only(info for info in values(Pkg.dependencies()) if info.name == name)
    revision = if !isnothing(info.tree_hash)
        string(info.tree_hash)
    elseif !isnothing(info.source) && isdir(info.source)
        try
            readchomp(`git -C $(info.source) rev-parse HEAD`)
        catch
            ""
        end
    else
        ""
    end
    (; version=string(info.version), revision)
end

const DEPENDENCIES = (;
    turing=dependency_receipt("Turing"),
    dynamicppl=dependency_receipt("DynamicPPL"),
    enzyme=dependency_receipt("Enzyme"),
    differentiationinterface=dependency_receipt("DifferentiationInterface"),
    warmuphmc=dependency_receipt("WarmupHMC"),
    mutatingfunctions=dependency_receipt("MutatingFunctions"),
    outputsignatures=dependency_receipt("OutputSignatures"),
)

function print_allocation_profile(label, f; repetitions=5, limit=24)
    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate=1.0 begin
        for _ in 1:repetitions
            f()
        end
    end
    allocations = Profile.Allocs.fetch().allocs
    println("allocation_profile=", (label, repetitions,
        count=length(allocations), bytes=sum(a -> a.size, allocations)))
    root = dirname(@__DIR__)
    groups = Dict{Tuple{String,String,Int,String},Tuple{Int,Int}}()
    for allocation in allocations
        frame_index = findfirst(allocation.stacktrace) do frame
            startswith(string(frame.file), root)
        end
        frame = isnothing(frame_index) ? last(allocation.stacktrace) :
            allocation.stacktrace[frame_index]
        key = (string(frame.func), string(frame.file), frame.line,
               string(allocation.type))
        count, bytes = get(groups, key, (0, 0))
        groups[key] = (count + 1, bytes + allocation.size)
    end
    sorted = sort(collect(groups); by=entry -> last(entry)[2], rev=true)
    for ((function_, file, line, type), (count, bytes)) in
            Iterators.take(sorted, limit)
        println("allocation_site=", (label, bytes, count, function_, file,
            line, type))
    end
    flush(stdout)
    nothing
end

function print_cpu_profile(label, f; repetitions=100_000)
    Profile.clear()
    @profile for _ in 1:repetitions
        f()
    end
    println("cpu_profile=", (label, repetitions))
    Profile.print(; format=:flat, sortedby=:count, mincount=20, C=true)
    flush(stdout)
    nothing
end

K in (8, 20) || error("BRM_HSGP_BENCH_K must be 8 or 20")
BLAS.set_num_threads(1)
mkpath(CACHE)

const MOTORCYCLE_NCP_K20 = @brm begin
    length_scale(mu, hsgp(x)) ~ LogNormal(0, 4)
    sd(mu, hsgp(x)) ~ LogNormal(0, 4)
    length_scale(sigma, hsgp(x)) ~ LogNormal(0, 4)
    sd(sigma, hsgp(x)) ~ LogNormal(0, 4)
    mu ~ hsgp(x; k=20, domain=(-1.5, 1.5))
    log(sigma) ~ hsgp(x; k=20, domain=(-1.5, 1.5))
    y ~ Normal(mu, sigma)
end

const MOTORCYCLE_PARTIAL_K20 = @brm begin
    length_scale(mu, hsgp(x)) ~ LogNormal(0, 4)
    sd(mu, hsgp(x)) ~ LogNormal(0, 4)
    length_scale(sigma, hsgp(x)) ~ LogNormal(0, 4)
    sd(sigma, hsgp(x)) ~ LogNormal(0, 4)
    mu ~ hsgp(x; k=20, domain=(-1.5, 1.5), centeredness=c_mu)
    log(sigma) ~ hsgp(x; k=20, domain=(-1.5, 1.5), centeredness=c_sigma)
    y ~ Normal(mu, sigma)
end

const MOTORCYCLE_NCP_K8 = @brm begin
    length_scale(mu, hsgp(x)) ~ LogNormal(0, 4)
    sd(mu, hsgp(x)) ~ LogNormal(0, 4)
    length_scale(sigma, hsgp(x)) ~ LogNormal(0, 4)
    sd(sigma, hsgp(x)) ~ LogNormal(0, 4)
    mu ~ hsgp(x; k=8, domain=(-1.5, 1.5))
    log(sigma) ~ hsgp(x; k=8, domain=(-1.5, 1.5))
    y ~ Normal(mu, sigma)
end

const MOTORCYCLE_PARTIAL_K8 = @brm begin
    length_scale(mu, hsgp(x)) ~ LogNormal(0, 4)
    sd(mu, hsgp(x)) ~ LogNormal(0, 4)
    length_scale(sigma, hsgp(x)) ~ LogNormal(0, 4)
    sd(sigma, hsgp(x)) ~ LogNormal(0, 4)
    mu ~ hsgp(x; k=8, domain=(-1.5, 1.5), centeredness=c_mu)
    log(sigma) ~ hsgp(x; k=8, domain=(-1.5, 1.5), centeredness=c_sigma)
    y ~ Normal(mu, sigma)
end

function motorcycle_data()
    path = joinpath(@__DIR__, "..", "research", "adaptive_centering", "mcycle.csv")
    lines = readlines(path)
    first(lines) == "rownames,times,accel" || error("unexpected mcycle header")
    rows = split.(lines[2:end], ',')
    times = parse.(Float64, getindex.(rows, 2))
    accel = parse.(Float64, getindex.(rows, 3))
    length(times) == 133 || error("expected all 133 motorcycle observations")
    xmin, xmax = extrema(times)
    x = @. -1 + 2 * (times - xmin) / (xmax - xmin)
    (; x, y=accel ./ std(accel))
end

function centeredness(k)
    # Exact source-faithful k=20 pilot selection (Xoshiro(1), 10_000 draws),
    # supplied with the immutable posterior receipt. The bounded k=8 regression
    # uses a deterministic heterogeneous geometry without running a pilot.
    if k == 20
        return (;
            mu=[1.0, 0.91, 0.60, 0.84, 0.52, 0.79, 0.71, 0.53, 0.61,
                0.17, 0.65, 0.42, 0.56, 0.48, 0.23, 0.33, 0.37, 0.24,
                0.17, 0.15],
            sigma=[0.94, 1.0, 0.82, 0.74, 0.47, 0.23, 0.45, 0.48, 0.23,
                   0.23, 0.15, 0.11, 0.07, 0.01, 0.03, 0.0, 0.01, 0.0,
                   0.0, 0.0],
        )
    end
    mu = collect(range(0.05, 0.95; length=k))
    sigma = reverse(mu)[mod1.(3 .* (0:(k - 1)) .+ 1, k)]
    (; mu, sigma)
end

function build_brmi(data, geometry)
    partial = geometry != :noncentered
    c = geometry == :noncentered ? (; mu=zeros(K), sigma=zeros(K)) :
        geometry == :centered ? (; mu=ones(K), sigma=ones(K)) : centeredness(K)
    model_data = (; data..., c_mu=c.mu, c_sigma=c.sigma)
    builder = if K == 20
        partial ? MOTORCYCLE_PARTIAL_K20 : MOTORCYCLE_NCP_K20
    else
        partial ? MOTORCYCLE_PARTIAL_K8 : MOTORCYCLE_NCP_K8
    end
    builder(model_data), c
end

function initial_parameters(backend, c)
    terms = only.(getproperty.(backend.plan.predictors, :terms))
    turing_ext = Base.get_extension(BRM, :BayesianRegressionModelsTuringExt)
    weights = (
        collect(range(-0.16, 0.18; length=K)),
        collect(range(0.12, -0.10; length=K)),
    )
    values = map(eachindex(terms)) do i
        term = terms[i]
        rho = term.state.rho_lower + (i == 1 ? 0.55 : 0.45)
        sigma = i == 1 ? 0.65 : 0.40
        z = weights[i]
        cb = i == 1 ? c.mu : c.sigma
        log_scale = turing_ext._brm_hsgp_log_sqrt_spd(term.state, sigma, rho)
        beta = exp.(cb .* log_scale) .* z
        any(!iszero, cb) ? (; rho, sigma, beta_partial=beta) :
            (; rho, sigma, beta_raw=z)
    end
    (; term_mu_1=values[1], term_sigma_1=values[2])
end

function turing_target(brmi, c)
    started = time_ns()
    backend = TuringBRMI(brmi)
    backend_ns = time_ns() - started
    parameters = initial_parameters(backend, c)
    started = time_ns()
    vi = DP.VarInfo(backend.model, DP.InitFromParams(parameters), DP.LinkAll())
    raw = DP.LogDensityFunction(backend.model, DP.getlogjoint_internal, vi)
    q = collect(DP.get_sample_input_vector(raw))
    construction_ns = time_ns() - started
    started = time_ns()
    value = LogDensityProblems.logdensity(raw, q)
    first_density_ns = time_ns() - started
    started = time_ns()
    turing_ext = Base.get_extension(
        BRM, :BayesianRegressionModelsTuringWarmupHMCExt)
    prepared = turing_ext._prepare_dynamicppl_gradient(raw, ENZYME_BACKEND)
    preparation_ns = time_ns() - started
    started = time_ns()
    first_value, first_gradient = @inferred(
        LogDensityProblems.logdensity_and_gradient(prepared, q))
    first_gradient_ns = time_ns() - started
    (; backend, raw, prepared, q, value, first_value, first_gradient,
       backend_ns, construction_ns, first_density_ns, preparation_ns,
       first_gradient_ns)
end

function stan_json(parameters, partial)
    mu = parameters.term_mu_1
    sigma = parameters.term_sigma_1
    weight_name = partial ? "beta_partial" : "beta_raw"
    mu_weights = getproperty(mu, Symbol(weight_name))
    sigma_weights = getproperty(sigma, Symbol(weight_name))
    "{" * join((
        "\"hsgp_x_rho_iso\":$(mu.rho)",
        "\"hsgp_x_sigma\":$(mu.sigma)",
        "\"hsgp_x_$weight_name\":[$(join(mu_weights, ','))]",
        "\"hsgp_log_sigma_x_rho_iso\":$(sigma.rho)",
        "\"hsgp_log_sigma_x_sigma\":$(sigma.sigma)",
        "\"hsgp_log_sigma_x_$weight_name\":[$(join(sigma_weights, ','))]",
    ), ',') * "}"
end

function stan_target(brmi, parameters, geometry)
    started = time_ns()
    sb = SBBRMI(brmi; mod=@__MODULE__)
    lowering_ns = time_ns() - started
    source_geometry = geometry == :noncentered ? :noncentered : :partial
    path = joinpath(CACHE, "motorcycle-$(source_geometry)-k$K.stan")
    started = time_ns()
    problem = StanBlocks.stan_instantiate(sb.model; path)
    instantiate_ns = time_ns() - started
    q = BS.param_unconstrain_json(
        problem.model, stan_json(parameters, geometry != :noncentered))
    started = time_ns()
    value, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
    first_gradient_ns = time_ns() - started
    (; sb, problem, q, value, gradient, path, lowering_ns, instantiate_ns,
       first_gradient_ns)
end

function set_sources!(problem, controls)
    ir = WarmupHMC.reparametrizer(problem)
    length(ir.pairs) == length(controls) || throw(DimensionMismatch())
    ir.pairs .= map(ir.pairs, controls) do (idx, value), c
        idx => WarmupHMC.Reparametrization(
            value.target, WarmupHMC.PartiallyCentered(c), value.args...)
    end
    problem.scoring_plan.synchronize!(ir)
    problem
end

function canonical_turing_ranges(target, partial)
    ranges = DP.get_all_ranges_and_transforms(target.raw)
    weight = partial ? :beta_partial : :beta_raw
    labels = Vector{String}(undef, length(target.q))
    for (logical, site) in ((:mu, :term_mu_1), (:log_sigma, :term_sigma_1))
        prefix = string(logical)
        for (field, suffix) in ((:rho, "rho"), (:sigma, "sd"))
            range = ranges[DP.VarName{site}(DP.Property{field}())].range
            labels[only(range)] = "$prefix.$suffix"
        end
        range = ranges[DP.VarName{site}(DP.Property{weight}())].range
        for (basis, idx) in enumerate(range)
            labels[idx] = "$prefix.beta.$basis"
        end
    end
    labels
end

function canonical_stan_names(problem)
    replace.(BS.param_unc_names(problem.model),
        r"^hsgp_x_rho_iso$" => "mu.rho",
        r"^hsgp_x_sigma$" => "mu.sd",
        r"^hsgp_x_beta_(?:raw|partial)\." => "mu.beta.",
        r"^hsgp_log_sigma_x_rho_iso$" => "log_sigma.rho",
        r"^hsgp_log_sigma_x_sigma$" => "log_sigma.sd",
        r"^hsgp_log_sigma_x_beta_(?:raw|partial)\." => "log_sigma.beta.",
    )
end

function benchmark_hsgp_blocks(turing, c)
    ranges = DP.get_all_ranges_and_transforms(turing.raw)
    blocks = BRM._HSGPAdaptiveCenteringBlock[]
    for (logical, site, component_c, component) in zip(
            (:mu, :sigma), (:term_mu_1, :term_sigma_1), (c.mu, c.sigma),
            turing.backend.plan.predictors)
        term = only(component.terms)
        field = any(!iszero, component_c) ? :beta_partial : :beta_raw
        property_range(name) = collect(ranges[
            DP.VarName{site}(DP.Property{name}())].range)
        push!(blocks, BRM._HSGPAdaptiveCenteringBlock(
            logical,
            Symbol(:hsgp_, join(string.(term.source), "_")),
            collect(Float64, component_c),
            property_range(field),
            property_range(:rho),
            [Float64(term.state.rho_lower)],
            only(property_range(:sigma)),
            0.0,
            Matrix{Float64}(term.state.omega2),
        ))
    end
    blocks
end

function reorder(values, source_names, target_names)
    by_name = Dict(source_names .=> values)
    [by_name[name] for name in target_names]
end

function central_difference(problem, x; step=1e-5)
    central_difference_function(
        q -> LogDensityProblems.logdensity(problem, q), x; step)
end

function central_difference_function(f, x; step=1e-5)
    gradient = similar(x)
    for i in eachindex(x)
        plus, minus = copy(x), copy(x)
        plus[i] += step
        minus[i] -= step
        gradient[i] = (f(plus) - f(minus)) / (2step)
    end
    gradient
end

function max_scaled_difference(a, b)
    maximum(zip(a, b); init=0.0) do (x, y)
        abs(x - y) / max(1.0, abs(x), abs(y))
    end
end

function require_leq(label, observed, threshold)
    observed <= threshold || error(
        "$label was $observed, exceeding the acceptance threshold $threshold")
end

function measure(f)
    for _ in 1:WARMUP
        f()
    end
    samples = Vector{Float64}(undef, SAMPLES)
    for sample in eachindex(samples)
        started = time_ns()
        for _ in 1:BATCH
            f()
        end
        samples[sample] = (time_ns() - started) / BATCH
    end
    (; median_ns=median(samples), q25_ns=quantile(samples, 0.25),
       q75_ns=quantile(samples, 0.75), minimum_ns=minimum(samples), samples)
end

function measure_pair(turing_call, turing_positions, stan_call, stan_positions)
    for iteration in 1:WARMUP
        index = mod1(iteration, length(turing_positions))
        turing_call(turing_positions[index])
        stan_call(stan_positions[index])
    end
    turing_samples = Vector{Float64}(undef, SAMPLES)
    stan_samples = similar(turing_samples)
    function measure_arm(f, positions)
        started = time_ns()
        for iteration in 1:BATCH
            f(positions[mod1(iteration, length(positions))])
        end
        (time_ns() - started) / BATCH
    end
    for sample in 1:SAMPLES
        if isodd(sample)
            turing_samples[sample] = measure_arm(turing_call, turing_positions)
            stan_samples[sample] = measure_arm(stan_call, stan_positions)
        else
            stan_samples[sample] = measure_arm(stan_call, stan_positions)
            turing_samples[sample] = measure_arm(turing_call, turing_positions)
        end
    end
    summarize(samples) = (;
        median_ns=median(samples), q25_ns=quantile(samples, 0.25),
        q75_ns=quantile(samples, 0.75), minimum_ns=minimum(samples), samples)
    (; turing=summarize(turing_samples), stan=summarize(stan_samples))
end

function allocation(f)
    f()
    @allocated f()
end

raw_logdensity(x, problem) = LogDensityProblems.logdensity(problem, x)

file_sha256(path) = open(path) do io
    bytes2hex(sha256(io))
end

struct NormalizedStanProblem{M}
    model::M
end
LogDensityProblems.dimension(problem::NormalizedStanProblem) =
    length(BS.param_unc_names(problem.model))
LogDensityProblems.capabilities(::Type{<:NormalizedStanProblem}) =
    LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(problem::NormalizedStanProblem, q) =
    BS.log_density(problem.model, q; propto=false, jacobian=true)
function LogDensityProblems.logdensity_and_gradient(
    problem::NormalizedStanProblem, q,
)
    gradient = similar(q)
    value, _ = BS.log_density_gradient!(
        problem.model, q, gradient; propto=false, jacobian=true)
    value, gradient
end

function recenter_position(position, blocks, source_controls, target_controls)
    target = copy(position)
    offset = 0
    for block in blocks
        for basis in eachindex(block.effects)
            source_c = source_controls[offset + basis]
            target_c = target_controls[offset + basis]
            log_scale = BRM._adaptive_hsgp_log_scale(
                position, block, basis)
            target[block.effects[basis]] *=
                exp((target_c - source_c) * log_scale)
        end
        offset += length(block.effects)
    end
    target
end

function benchmark_positions(
    geometry, turing, stan, blocks, turing_names, stan_names,
)
    turing_positions = [copy(turing.q)]
    stan_positions = [copy(stan.q)]
    labels = ["deterministic"]
    isempty(DRAW_DIR) && return (; turing_positions, stan_positions, labels,
                                  draw_sha256="", model_sha256="")
    K == 20 || error("posterior draw receipts apply only to k=20")
    draw_geometry = geometry == :noncentered ? "noncentered" : "partial"
    draw_path = joinpath(DRAW_DIR, "$draw_geometry.jls")
    isfile(draw_path) || error("missing posterior receipt $draw_path")
    receipt = deserialize(draw_path)
    receipt.complete || error("posterior receipt is incomplete")
    receipt.seed == 1 || error("posterior receipt did not use Xoshiro(1)")
    receipt.requested_draws == 10_000 || error(
        "posterior receipt did not retain the requested 10,000-draw fit")
    draws = receipt.posterior_position
    size(draws, 1) == length(stan.q) || error("posterior dimension mismatch")
    all(column -> column in axes(draws, 2), DRAW_COLUMNS) ||
        error("posterior draw column is out of range")
    source_c = draw_geometry == "noncentered" ? zeros(2K) :
        vcat(centeredness(K).mu, centeredness(K).sigma)
    target_c = geometry == :noncentered ? zeros(2K) :
        geometry == :centered ? ones(2K) :
        vcat(centeredness(K).mu, centeredness(K).sigma)
    for column in DRAW_COLUMNS
        # The byte-identical emitted model proves the serialized BridgeStan row
        # order. Canonical names then map that order into DynamicPPL coordinates.
        turing_position = reorder(
            view(draws, :, column), stan_names, turing_names)
        turing_position = recenter_position(
            turing_position, blocks, source_c, target_c)
        push!(turing_positions, turing_position)
        push!(stan_positions,
            reorder(turing_position, turing_names, stan_names))
        push!(labels, "posterior_$column")
    end
    (; turing_positions, stan_positions, labels,
       draw_sha256=file_sha256(draw_path), model_sha256=file_sha256(stan.path))
end

function write_tsv(path, rows)
    isempty(rows) && return
    keys_ = propertynames(first(rows))
    open(path, "w") do io
        println(io, join(keys_, '\t'))
        for row in rows
            println(io, join((getproperty(row, key) for key in keys_), '\t'))
        end
    end
end

function benchmark_fixed_late(result)
    turing_call = q -> LogDensityProblems.logdensity_and_gradient(
        result.turing_problem, q)
    stan_call = q -> LogDensityProblems.logdensity_and_gradient(
        result.stan.problem, q)
    measure_pair(
        turing_call, result.positions.turing_positions,
        stan_call, result.positions.stan_positions)
end

function benchmark_geometry(geometry)
    println("phase=build geometry=$geometry")
    flush(stdout)
    data = motorcycle_data()
    brmi, c = build_brmi(data, geometry)
    turing = turing_target(brmi, c)
    println("phase=turing_ready geometry=$geometry")
    flush(stdout)
    parameters = initial_parameters(turing.backend, c)
    stan = stan_target(brmi, parameters, geometry)
    println("phase=stan_ready geometry=$geometry")
    flush(stdout)
    partial = geometry != :noncentered
    turing_names = canonical_turing_ranges(turing, partial)
    stan_names = canonical_stan_names(stan.problem)
    stan_q_turing_order = reorder(stan.q, stan_names, turing_names)
    coordinate_gap = maximum(abs, turing.q .- stan_q_turing_order)
    if coordinate_gap >= 2e-12
        println("coordinate_debug=", (geometry, coordinate_gap, turing_names,
            turing_q=turing.q, stan_names, stan_q=stan.q,
            stan_q_turing_order))
        error("coordinate mismatch for $geometry")
    end

    blocks = benchmark_hsgp_blocks(turing, c)
    turing_problem = turing.prepared
    positions = benchmark_positions(
        geometry, turing, stan, blocks, turing_names, stan_names)
    turing_call = q ->
        LogDensityProblems.logdensity_and_gradient(turing_problem, q)
    stan_call = q -> LogDensityProblems.logdensity_and_gradient(stan.problem, q)
    density_gap = 0.0
    raw_turing_gap = 0.0
    gradient_gap = 0.0
    gradient_scaled_gap = 0.0
    for (turing_q, stan_q) in zip(
        positions.turing_positions, positions.stan_positions)
        turing_value, turing_gradient = turing_call(turing_q)
        stan_gradient_buffer = similar(stan_q)
        stan_value, _ = BS.log_density_gradient!(
            stan.problem.model, stan_q, stan_gradient_buffer;
            propto=false, jacobian=true)
        stan_gradient_turing_order = reorder(
            stan_gradient_buffer, stan_names, turing_names)
        density_gap = max(density_gap, abs(turing_value - stan_value))
        raw_turing_gap = max(raw_turing_gap, abs(turing_value -
            LogDensityProblems.logdensity(turing.raw, turing_q)))
        gradient_gap = max(gradient_gap, maximum(
            abs, turing_gradient .- stan_gradient_turing_order))
        gradient_scaled_gap = max(gradient_scaled_gap,
            max_scaled_difference(turing_gradient, stan_gradient_turing_order))
    end
    fd_indices = unique((1, length(positions.turing_positions)))
    turing_fd_gap = 0.0
    stan_fd_gap = 0.0
    turing_fd_scaled_gap = 0.0
    stan_fd_scaled_gap = 0.0
    for point in fd_indices
        turing_q = positions.turing_positions[point]
        stan_q = positions.stan_positions[point]
        _, turing_gradient = turing_call(turing_q)
        stan_gradient = similar(stan_q)
        BS.log_density_gradient!(
            stan.problem.model, stan_q, stan_gradient;
            propto=false, jacobian=true)
        fd_turing = central_difference(turing.raw, turing_q)
        fd_stan = central_difference_function(stan_q) do q
            BS.log_density(stan.problem.model, q; propto=false, jacobian=true)
        end
        turing_fd_gap = max(
            turing_fd_gap, maximum(abs, turing_gradient .- fd_turing))
        stan_fd_gap = max(
            stan_fd_gap, maximum(abs, stan_gradient .- fd_stan))
        turing_fd_scaled_gap = max(turing_fd_scaled_gap,
            max_scaled_difference(turing_gradient, fd_turing))
        stan_fd_scaled_gap = max(stan_fd_scaled_gap,
            max_scaled_difference(stan_gradient, fd_stan))
    end
    println("phase=equivalence_ready geometry=$geometry")
    flush(stdout)
    receipt = (;
        geometry=String(geometry),
        points=length(positions.labels),
        density_difference=density_gap,
        raw_turing_density_difference=raw_turing_gap,
        gradient_max_abs_difference=gradient_gap,
        gradient_max_scaled_difference=gradient_scaled_gap,
        turing_fd_max_abs_difference=turing_fd_gap,
        stan_fd_max_abs_difference=stan_fd_gap,
        turing_fd_max_scaled_difference=turing_fd_scaled_gap,
        stan_fd_max_scaled_difference=stan_fd_scaled_gap,
    )
    require_leq("$geometry normalized-density difference", density_gap, 1e-10)
    require_leq("$geometry raw-DynamicPPL density difference", raw_turing_gap, 1e-12)
    require_leq("$geometry scaled gradient difference", gradient_scaled_gap, 1e-9)
    require_leq("$geometry Turing scaled finite-difference error",
        turing_fd_scaled_gap, 1e-3)
    require_leq("$geometry Stan scaled finite-difference error",
        stan_fd_scaled_gap, 1e-3)
    allocations = (;
        turing=allocation(() -> turing_call(first(positions.turing_positions))),
        stan=allocation(() -> stan_call(first(positions.stan_positions))))
    PROFILE_ALLOCATIONS && print_allocation_profile(
        "fixed-$geometry",
        () -> turing_call(first(positions.turing_positions)))
    PROFILE_CPU && print_cpu_profile(
        "fixed-$geometry",
        () -> turing_call(first(positions.turing_positions)))
    raw_density_call = q -> LogDensityProblems.logdensity(turing.raw, q)
    raw_density_timing = measure(
        () -> raw_density_call(first(positions.turing_positions)))
    raw_density_allocations = allocation(
        () -> raw_density_call(first(positions.turing_positions)))
    (; receipt, allocations, turing, stan, c, turing_names, stan_names,
       positions, turing_problem, raw_density_timing, raw_density_allocations)
end

function benchmark_online(fixed_results)
    ncp = only(filter(r -> r.receipt.geometry == "noncentered", fixed_results))
    turing_online = adaptive_centering_problem(
        ncp.turing.backend, ncp.turing.raw, ENZYME_BACKEND)
    stan_online = adaptive_centering_problem(
        ncp.stan.sb, ncp.stan.problem, ENZYME_BACKEND)
    normalized_stan_online = adaptive_centering_problem(
        ncp.stan.sb, NormalizedStanProblem(ncp.stan.problem.model),
        ENZYME_BACKEND; unc_names=BS.param_unc_names(ncp.stan.problem.model))
    online_rows = NamedTuple[]
    for fixed in fixed_results
        controls = vcat(fixed.c.mu, fixed.c.sigma)
        set_sources!(turing_online, controls)
        set_sources!(stan_online, controls)
        set_sources!(normalized_stan_online, controls)
        turing_positions = [reorder(
            q, fixed.turing_names, ncp.turing_names)
            for q in fixed.positions.turing_positions]
        stan_positions = [reorder(
            q, fixed.stan_names, ncp.stan_names)
            for q in fixed.positions.stan_positions]
        turing_call = q -> LogDensityProblems.logdensity_and_gradient(
            turing_online, q)
        stan_call = q -> LogDensityProblems.logdensity_and_gradient(
            stan_online, q)
        normalized_stan_call = q -> LogDensityProblems.logdensity_and_gradient(
            normalized_stan_online, q)
        fixed_turing_call = q -> LogDensityProblems.logdensity_and_gradient(
            fixed.turing_problem, q)
        online_inner_call = q -> LogDensityProblems.logdensity_and_gradient(
            turing_online.problem, q)
        density_gap = 0.0
        gradient_gap = 0.0
        gradient_scaled_gap = 0.0
        fixed_density_gap = 0.0
        fixed_gradient_gap = 0.0
        fixed_gradient_scaled_gap = 0.0
        for point in eachindex(turing_positions)
            turing_value, turing_gradient = turing_call(turing_positions[point])
            normalized_stan_value, normalized_stan_gradient =
                normalized_stan_call(stan_positions[point])
            normalized_stan_gradient_turing_order = reorder(
                normalized_stan_gradient, ncp.stan_names, ncp.turing_names)
            fixed_turing_value, fixed_turing_gradient =
                fixed_turing_call(fixed.positions.turing_positions[point])
            fixed_turing_gradient_online_order = reorder(
                fixed_turing_gradient, fixed.turing_names, ncp.turing_names)
            density_gap = max(
                density_gap, abs(turing_value - normalized_stan_value))
            gradient_gap = max(gradient_gap, maximum(
                abs, turing_gradient .- normalized_stan_gradient_turing_order))
            gradient_scaled_gap = max(gradient_scaled_gap,
                max_scaled_difference(
                    turing_gradient, normalized_stan_gradient_turing_order))
            fixed_density_gap = max(
                fixed_density_gap, abs(turing_value - fixed_turing_value))
            fixed_gradient_gap = max(fixed_gradient_gap, maximum(
                abs, turing_gradient .- fixed_turing_gradient_online_order))
            fixed_gradient_scaled_gap = max(fixed_gradient_scaled_gap,
                max_scaled_difference(
                    turing_gradient, fixed_turing_gradient_online_order))
        end
        timing = measure_pair(
            turing_call, turing_positions, stan_call, stan_positions)
        require_leq("online $(fixed.receipt.geometry) normalized-density difference",
            density_gap, 1e-10)
        require_leq("online $(fixed.receipt.geometry) scaled gradient difference",
            gradient_scaled_gap, 1e-9)
        require_leq("online $(fixed.receipt.geometry) fixed-density difference",
            fixed_density_gap, 1e-10)
        require_leq("online $(fixed.receipt.geometry) fixed scaled-gradient difference",
            fixed_gradient_scaled_gap, 1e-9)
        native_control_timing = measure_pair(
            fixed_turing_call, turing_positions,
            online_inner_call, turing_positions)
        allocations = (;
            turing=allocation(() -> turing_call(first(turing_positions))),
            stan=allocation(() -> stan_call(first(stan_positions))))
        PROFILE_ALLOCATIONS && print_allocation_profile(
            "online-$(fixed.receipt.geometry)",
            () -> turing_call(first(turing_positions)))
        push!(online_rows, (;
            geometry=fixed.receipt.geometry,
            points=length(turing_positions),
            density_difference=density_gap,
            gradient_max_abs_difference=gradient_gap,
            gradient_max_scaled_difference=gradient_scaled_gap,
            fixed_density_difference=fixed_density_gap,
            fixed_gradient_max_abs_difference=fixed_gradient_gap,
            fixed_gradient_max_scaled_difference=fixed_gradient_scaled_gap,
            timing, native_control_timing, allocations,
            draw_sha256=fixed.positions.draw_sha256,
            model_sha256=fixed.positions.model_sha256,
        ))
    end
    online_rows
end

const REVISION = readchomp(`git -C $(joinpath(@__DIR__, "..")) rev-parse HEAD`)
const BENCHMARK_SHA256 = file_sha256(@__FILE__)
const PROVENANCE = (;
    brm_revision=REVISION,
    benchmark_sha256=BENCHMARK_SHA256,
    runtime_activity=RUNTIME_ACTIVITY,
    turing_version=DEPENDENCIES.turing.version,
    turing_revision=DEPENDENCIES.turing.revision,
    dynamicppl_version=DEPENDENCIES.dynamicppl.version,
    dynamicppl_revision=DEPENDENCIES.dynamicppl.revision,
    enzyme_version=DEPENDENCIES.enzyme.version,
    enzyme_revision=DEPENDENCIES.enzyme.revision,
    differentiationinterface_version=
        DEPENDENCIES.differentiationinterface.version,
    differentiationinterface_revision=
        DEPENDENCIES.differentiationinterface.revision,
    warmuphmc_version=DEPENDENCIES.warmuphmc.version,
    warmuphmc_revision=DEPENDENCIES.warmuphmc.revision,
    mutatingfunctions_revision=DEPENDENCIES.mutatingfunctions.revision,
    outputsignatures_revision=DEPENDENCIES.outputsignatures.revision,
)

println("context=", (;
    revision=REVISION,
    julia=VERSION, cpu=Sys.CPU_NAME, machine=Sys.MACHINE,
    threads=Threads.nthreads(), blas_threads=BLAS.get_num_threads(),
    affinity=readchomp(`taskset -pc $(getpid())`), rows=133, basis=K,
    runtime_activity=RUNTIME_ACTIVITY,
    runtime_ratio_target=RUNTIME_RATIO_TARGET,
    dependencies=DEPENDENCIES,
    benchmark_sha256=BENCHMARK_SHA256,
    protocol=(warmup=WARMUP, samples=SAMPLES, batch=BATCH),
))

const results = map(
    benchmark_geometry, Tuple(GEOMETRIES))
const online_results = any(r -> r.receipt.geometry == "noncentered", results) ?
    benchmark_online(results) : NamedTuple[]
const fixed_timings = map(benchmark_fixed_late, results)
for (result, timing) in zip(results, fixed_timings)
    println("equivalence=", result.receipt)
    println("timing=", (geometry=result.receipt.geometry, timing...))
    println("allocations=", (geometry=result.receipt.geometry, result.allocations...))
    println("setup=", (geometry=result.receipt.geometry,
        turing_backend_ns=result.turing.backend_ns,
        turing_construction_ns=result.turing.construction_ns,
        turing_preparation_ns=result.turing.preparation_ns,
        turing_first_gradient_ns=result.turing.first_gradient_ns,
        stan_lowering_ns=result.stan.lowering_ns,
        stan_instantiate_ns=result.stan.instantiate_ns,
        stan_first_gradient_ns=result.stan.first_gradient_ns))
    println("raw_density=", (geometry=result.receipt.geometry,
        timing=result.raw_density_timing,
        allocations=result.raw_density_allocations))
    flush(stdout)
end
for result in online_results
    println("online_equivalence=", (;
        geometry=result.geometry,
        density_difference=result.density_difference,
        gradient_max_abs_difference=result.gradient_max_abs_difference,
        gradient_max_scaled_difference=result.gradient_max_scaled_difference,
        fixed_density_difference=result.fixed_density_difference,
        fixed_gradient_max_abs_difference=result.fixed_gradient_max_abs_difference,
        fixed_gradient_max_scaled_difference=
            result.fixed_gradient_max_scaled_difference,
    ))
    println("online_timing=", (geometry=result.geometry, result.timing...))
    println("native_control_timing=", (geometry=result.geometry,
        original_prepared=result.native_control_timing.turing,
        online_inner=result.native_control_timing.stan))
    println("online_allocations=", (geometry=result.geometry, result.allocations...))
    flush(stdout)
end

const rows = [
    (; kind="fixed", geometry=result.receipt.geometry, backend=String(backend),
       median_ns=getproperty(timing, backend).median_ns,
       q25_ns=getproperty(timing, backend).q25_ns,
       q75_ns=getproperty(timing, backend).q75_ns,
       minimum_ns=getproperty(timing, backend).minimum_ns,
       allocations=getproperty(result.allocations, backend),
       points=result.receipt.points,
       density_difference=result.receipt.density_difference,
       raw_turing_density_difference=
           result.receipt.raw_turing_density_difference,
       fixed_density_difference=NaN,
       fixed_gradient_max_abs_difference=NaN,
       fixed_gradient_max_scaled_difference=NaN,
       gradient_max_abs_difference=result.receipt.gradient_max_abs_difference,
       gradient_max_scaled_difference=
           result.receipt.gradient_max_scaled_difference,
       fd_max_abs_difference=backend == :turing ?
           result.receipt.turing_fd_max_abs_difference :
           result.receipt.stan_fd_max_abs_difference,
       fd_max_scaled_difference=backend == :turing ?
           result.receipt.turing_fd_max_scaled_difference :
           result.receipt.stan_fd_max_scaled_difference,
       runtime_ratio=getproperty(timing, backend).median_ns /
           timing.stan.median_ns,
       runtime_pass=backend == :stan ||
           getproperty(timing, backend).median_ns <=
               RUNTIME_RATIO_TARGET * timing.stan.median_ns,
       draw_sha256=result.positions.draw_sha256,
       model_sha256=result.positions.model_sha256,
       PROVENANCE...)
    for (result, timing) in zip(results, fixed_timings)
        for backend in (:turing, :stan)
]
append!(rows, [
    (; kind="online", geometry=result.geometry, backend=String(backend),
       median_ns=getproperty(result.timing, backend).median_ns,
       q25_ns=getproperty(result.timing, backend).q25_ns,
       q75_ns=getproperty(result.timing, backend).q75_ns,
       minimum_ns=getproperty(result.timing, backend).minimum_ns,
       allocations=getproperty(result.allocations, backend),
       points=result.points,
       density_difference=result.density_difference,
       raw_turing_density_difference=NaN,
       fixed_density_difference=result.fixed_density_difference,
       fixed_gradient_max_abs_difference=
           result.fixed_gradient_max_abs_difference,
       fixed_gradient_max_scaled_difference=
           result.fixed_gradient_max_scaled_difference,
       gradient_max_abs_difference=result.gradient_max_abs_difference,
       gradient_max_scaled_difference=result.gradient_max_scaled_difference,
       fd_max_abs_difference=NaN,
       fd_max_scaled_difference=NaN,
       runtime_ratio=getproperty(result.timing, backend).median_ns /
           result.timing.stan.median_ns,
       runtime_pass=backend == :stan ||
           getproperty(result.timing, backend).median_ns <=
               RUNTIME_RATIO_TARGET * result.timing.stan.median_ns,
       draw_sha256=result.draw_sha256,
       model_sha256=result.model_sha256,
       PROVENANCE...)
    for result in online_results for backend in (:turing, :stan)
])
if !isempty(OUTPUT)
    mkpath(dirname(OUTPUT))
    write_tsv(OUTPUT, rows)
    println("output=", OUTPUT)
end
