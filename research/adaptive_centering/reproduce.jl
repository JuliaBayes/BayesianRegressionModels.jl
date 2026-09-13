using BayesianRegressionModels
using BridgeStan
import DifferentiationInterface as DI
using Distributions
import Enzyme
using LogDensityProblems
using Random
using StanBlocks
using Statistics
using Turing
using WarmupHMC

const BRM = BayesianRegressionModels
const BS = BridgeStan
const DP = Turing.DynamicPPL
const SOURCE_REVISION = "0d00b8535e2c20c49017d03c7b060940eb8e7041"
const DATA_REVISION = "1dcc2bf5f955cc1224a3e1307256e1fe86b68dae"
const DATA_SHA256 = "b89a1e4eb0391a982b32be3e378df00e8593ff9971e9425e9c5d7929b74f9801"
const RESEARCH_DIR = @__DIR__
const DEFAULT_K = 20
const L = 1.5
const ENZYME_BACKEND = DI.AutoEnzyme(;
    mode=Enzyme.set_runtime_activity(Enzyme.Reverse))

function read_mcycle(path=joinpath(RESEARCH_DIR, "mcycle.csv"))
    lines = readlines(path)
    first(lines) == "rownames,times,accel" || error("unexpected mcycle header")
    rows = split.(lines[2:end], ',')
    times = parse.(Float64, getindex.(rows, 2))
    accel = parse.(Float64, getindex.(rows, 3))
    length(times) == 133 || error("expected 133 motorcycle observations")
    (; times, accel)
end

function prepared_data(; k=DEFAULT_K, c_mu=zeros(k), c_sigma=zeros(k))
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

const MOTORCYCLE_NCP_BOUNDED = @brm begin
    length_scale(mu, hsgp(x)) ~ LogNormal(0, 4)
    sd(mu, hsgp(x)) ~ LogNormal(0, 4)
    length_scale(sigma, hsgp(x)) ~ LogNormal(0, 4)
    sd(sigma, hsgp(x)) ~ LogNormal(0, 4)
    mu ~ hsgp(x; k=8, domain=(-1.5, 1.5))
    log(sigma) ~ hsgp(x; k=8, domain=(-1.5, 1.5))
    y ~ Normal(mu, sigma)
end

const MOTORCYCLE_PARTIAL_BOUNDED = @brm begin
        length_scale(mu, hsgp(x)) ~ LogNormal(0, 4)
        sd(mu, hsgp(x)) ~ LogNormal(0, 4)
        length_scale(sigma, hsgp(x)) ~ LogNormal(0, 4)
        sd(sigma, hsgp(x)) ~ LogNormal(0, 4)
        mu ~ hsgp(x; k=8, domain=(-1.5, 1.5), centeredness=c_mu)
        log(sigma) ~ hsgp(x; k=8, domain=(-1.5, 1.5), centeredness=c_sigma)
        y ~ Normal(mu, sigma)
end

function motorcycle_builder(k, partial)
    k == DEFAULT_K && return partial ? MOTORCYCLE_PARTIAL : MOTORCYCLE_NCP
    k == 8 && return partial ? MOTORCYCLE_PARTIAL_BOUNDED : MOTORCYCLE_NCP_BOUNDED
    throw(ArgumentError("this audited reproduction supports k=8 or k=20, got $k"))
end

model_data(data) = (; x=data.x, y=data.y, c_mu=data.c_mu, c_sigma=data.c_sigma)
build_brmi(data, k; partial=false) = motorcycle_builder(k, partial)(model_data(data))

struct EnzymeTuringDensity{L,P,G}
    ldf::L
    preparation::P
    gradient::G
end

turing_logdensity(x, ldf) = LogDensityProblems.logdensity(ldf, x)
LogDensityProblems.dimension(d::EnzymeTuringDensity) = LogDensityProblems.dimension(d.ldf)
LogDensityProblems.capabilities(::Type{<:EnzymeTuringDensity}) =
    LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.logdensity(d::EnzymeTuringDensity, x) =
    LogDensityProblems.logdensity(d.ldf, x)
function LogDensityProblems.logdensity_and_gradient(d::EnzymeTuringDensity, x)
    value, gradient = DI.value_and_gradient!(
        turing_logdensity, d.gradient, d.preparation, ENZYME_BACKEND,
        x, DI.Constant(d.ldf))
    value, copy(gradient)
end

function turing_density(brmi, seed)
    Random.seed!(seed)
    backend = TuringBRMI(brmi)
    ldf = DP.LogDensityFunction(
        backend.model, DP.getlogjoint_internal, DP.LinkAll(); fix_transforms=true)
    q = collect(DP.get_sample_input_vector(ldf))
    preparation = DI.prepare_gradient(
        turing_logdensity, ENZYME_BACKEND, q, DI.Constant(ldf))
    gradient = similar(q)
    density = EnzymeTuringDensity(ldf, preparation, gradient)
    value, grad = LogDensityProblems.logdensity_and_gradient(density, q)
    isfinite(value) && all(isfinite, grad) || error("non-finite Turing density at initialization")
    (; density, q, backend)
end

function stan_density(brmi, label, output_dir)
    sb = SBBRMI(brmi; mod=@__MODULE__)
    checked = StanBlocks.stanc_check(BRM.stan_code(sb))
    checked.ok || error("stanc failed for $label\n$(checked.output)")
    path = joinpath(output_dir, "motorcycle-$label.stan")
    density = StanBlocks.stan_instantiate(sb.model; path)
    q = zeros(LogDensityProblems.dimension(density))
    value, grad = LogDensityProblems.logdensity_and_gradient(density, q)
    isfinite(value) && all(isfinite, grad) || error("non-finite Stan density at initialization")
    (; density, q, sb)
end

function sample_chains(make_target, seeds; n_draws, n_evaluations)
    fits = Vector{Any}(undef, length(seeds))
    elapsed = @elapsed for (i, seed) in pairs(seeds)
        target, q = make_target(seed)
        fits[i] = WarmupHMC.adaptive_warmup_mcmc(
            Xoshiro(seed), target;
            init=(; position=copy(q), squared_scale=ones(length(q))),
            n_draws,
            n_evaluations,
            recording_target=n_evaluations,
            stepsize_adaptation_limit=50,
            target_acceptance_rate=0.9,
            max_tree_depth=8,
            variance_cond_target=Inf,
            progress=nothing,
            monitor_ess=false,
            nonlinear_adapt=false,
        )
    end
    (; fits, elapsed)
end

function draws_cube(fits)
    n = minimum(size(f.posterior_position, 2) for f in fits)
    cat((f.posterior_position[:, end-n+1:end] for f in fits)...; dims=3)
end

function classical_rhat(x)
    n, m = size(x)
    m > 1 || return NaN
    within = mean(var(view(x, :, j); corrected=true) for j in 1:m)
    between = n * var(vec(mean(x; dims=1)); corrected=true)
    sqrt(((n - 1) / n * within + between / n) / within)
end

function initial_positive_ess(x)
    n, m = size(x)
    n < 4 && return n * m
    chain_means = vec(mean(x; dims=1))
    within = mean(var(view(x, :, j); corrected=true) for j in 1:m)
    between = n * var(chain_means; corrected=true)
    variance = (n - 1) / n * within + between / n
    variance <= 0 && return n * m
    rho = Float64[]
    for lag in 1:(n - 1)
        variogram = mean(
            sum((x[(lag + 1):n, j] .- x[1:(n - lag), j]).^2) / (n - lag)
            for j in 1:m)
        push!(rho, 1 - variogram / (2variance))
    end
    tau = 1.0
    pair = 1
    while pair + 1 <= length(rho)
        pair_sum = rho[pair] + rho[pair + 1]
        pair_sum > 0 || break
        tau += 2pair_sum
        pair += 2
    end
    min(n * m / tau, n * m)
end

function diagnostics(backend, geometry, sampled)
    cube = draws_cube(sampled.fits)
    rhats = [classical_rhat(permutedims(cube[i, :, :], (1, 2)))
             for i in axes(cube, 1)]
    esses = [initial_positive_ess(permutedims(cube[i, :, :], (1, 2)))
             for i in axes(cube, 1)]
    (; backend, geometry,
       chains=length(sampled.fits), draws_per_chain=size(cube, 2),
       parameters=size(cube, 1), max_rhat=maximum(rhats), min_ess=minimum(esses),
       divergences=sum(f.n_divergent_samples for f in sampled.fits),
       gradient_evaluations=sum(f.total_evaluation_counter for f in sampled.fits),
       elapsed_seconds=sampled.elapsed)
end

function constrained_draws(stan, fits)
    names = BS.param_names(stan.density.model)
    matrices = map(fits) do fit
        reduce(hcat, (BS.param_constrain(stan.density.model, q)
                      for q in eachcol(fit.posterior_position)))
    end
    names, reduce(hcat, matrices)
end

log_spectral_scale(sigma, rho, omega2) =
    @. log(sigma) + 0.25log(2pi) + 0.5log(rho) - 0.25rho^2 * omega2

function pilot_centeredness(stan, fits, brmi, k)
    names, draws = constrained_draws(stan, fits)
    index = Dict(names .=> eachindex(names))
    turing = TuringBRMI(brmi)
    predictors = Dict(p.name => only(p.terms) for p in turing.plan.predictors)
    function selection(prefix, target)
        unit = reduce(vcat, (
            reshape(draws[index["$(prefix)_beta_raw.$j"], :], 1, :) for j in 1:k))
        rho = draws[index["$(prefix)_rho_iso"], :]
        sigma = draws[index["$(prefix)_sigma"], :]
        omega2 = vec(predictors[target].state.omega2)
        logs = reduce(vcat, (
            reshape(log_spectral_scale.(sigma, rho, omega2[j]), 1, :) for j in 1:k))
        select_hsgp_centeredness(unit, logs; candidates=0:0.01:1)
    end
    (; mu=selection("hsgp_x", :mu),
       sigma=selection("hsgp_log_sigma_x", :sigma))
end

function write_tsv(path, rows)
    keys = propertynames(first(rows))
    open(path, "w") do io
        println(io, join(keys, '\t'))
        for row in rows
            println(io, join((getproperty(row, key) for key in keys), '\t'))
        end
    end
end

function run_reproduction(; k=parse(Int, get(ENV, "BRM_ADAPTIVE_K", "8")),
                          n_draws=parse(Int, get(ENV, "BRM_ADAPTIVE_DRAWS", "75")),
                          n_evaluations=parse(Int, get(ENV, "BRM_ADAPTIVE_EVALS", "350")),
                          n_chains=parse(Int, get(ENV, "BRM_ADAPTIVE_CHAINS", "4")),
                          output_dir=get(ENV, "BRM_ADAPTIVE_OUTPUT", mktempdir()))
    mkpath(output_dir)
    seeds = collect(0x6a09e667:(0x6a09e667 + n_chains - 1))
    data = prepared_data(; k)
    ncp_brmi = build_brmi(data, k)
    ncp_stan = stan_density(ncp_brmi, "ncp-k$k", output_dir)
    stan_ncp = sample_chains(seeds; n_draws, n_evaluations) do _
        ncp_stan.density, ncp_stan.q
    end
    selected = pilot_centeredness(ncp_stan, stan_ncp.fits, ncp_brmi, k)
    geometries = [
        (name="noncentered", c_mu=zeros(k), c_sigma=zeros(k), partial=false),
        (name="centered", c_mu=ones(k), c_sigma=ones(k), partial=true),
        (name="adaptive", c_mu=selected.mu.centeredness,
         c_sigma=selected.sigma.centeredness, partial=true),
    ]
    rows = NamedTuple[diagnostics("StanBlocks", "noncentered", stan_ncp)]
    stan_adaptive = nothing
    adaptive_fit = nothing
    for geometry in geometries[2:end]
        geometry_data = prepared_data(; k, c_mu=geometry.c_mu, c_sigma=geometry.c_sigma)
        brmi = build_brmi(geometry_data, k; partial=true)
        stan = stan_density(brmi, "$(geometry.name)-k$k", output_dir)
        sampled = sample_chains(seeds; n_draws, n_evaluations) do _
            stan.density, stan.q
        end
        push!(rows, diagnostics("StanBlocks", geometry.name, sampled))
        if geometry.name == "adaptive"
            stan_adaptive, adaptive_fit = stan, sampled
        end
    end
    for geometry in geometries
        geometry_data = prepared_data(; k, c_mu=geometry.c_mu, c_sigma=geometry.c_sigma)
        brmi = build_brmi(geometry_data, k; partial=geometry.partial)
        sampled = sample_chains(seeds; n_draws, n_evaluations) do seed
            td = turing_density(brmi, seed)
            td.density, td.q
        end
        push!(rows, diagnostics("Turing", geometry.name, sampled))
    end
    write_tsv(joinpath(output_dir, "diagnostics.tsv"), rows)
    write_tsv(joinpath(output_dir, "centeredness.tsv"), [
        (; basis=i, mean=selected.mu.centeredness[i],
           log_scale=selected.sigma.centeredness[i]) for i in 1:k])
    println("source_revision\t", SOURCE_REVISION)
    println("data_revision\t", DATA_REVISION)
    println("data_sha256\t", DATA_SHA256)
    println("basis_functions\t", k)
    println("draws_per_chain\t", n_draws)
    println("output_dir\t", output_dir)
    foreach(row -> println("result\t", row), rows)
    (; rows, selected, stan_adaptive, adaptive_fit, data, output_dir)
end

function validate_backends(; k=parse(Int, get(ENV, "BRM_ADAPTIVE_K", "8")))
    data = prepared_data(; k)
    brmi = build_brmi(data, k; partial=true)
    stan = stan_density(brmi, "validation-k$k", mktempdir())
    td = turing_density(brmi, 0x243f6a88)
    @assert LogDensityProblems.dimension(stan.density) ==
            LogDensityProblems.dimension(td.density)
    println("validation\tstanc+BridgeStan+Turing+Enzyme")
    println("dimension\t", LogDensityProblems.dimension(stan.density))
end

if abspath(PROGRAM_FILE) == @__FILE__
    if get(ENV, "BRM_ADAPTIVE_RUNTIME", "0") == "1"
        run_reproduction()
    else
        validate_backends()
    end
end
