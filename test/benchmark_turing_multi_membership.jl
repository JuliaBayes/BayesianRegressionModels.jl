using BayesianRegressionModels
using BridgeStan
import DifferentiationInterface as DI
import Enzyme
using LinearAlgebra: Cholesky
using Logging: NullLogger, with_logger
using LogDensityProblems
using Sockets: gethostname
using StanBlocks
using Statistics: median
using Test
using Turing

const BRM = BayesianRegressionModels
const BS = BridgeStan
const DP = Turing.DynamicPPL
const ENZYME_BACKEND = DI.AutoEnzyme(;
    mode=Enzyme.set_runtime_activity(Enzyme.Reverse))

function benchmark_call(f; warmup=20, samples=21, batch=20)
    for _ in 1:warmup
        f()
    end
    elapsed = Vector{Float64}(undef, samples)
    for sample in eachindex(elapsed)
        started = time_ns()
        for _ in 1:batch
            f()
        end
        elapsed[sample] = (time_ns() - started) / batch
    end
    f()
    GC.gc()
    allocated = @allocated begin
        for _ in 1:batch
            f()
        end
    end
    (; median_ns=median(elapsed), minimum_ns=minimum(elapsed),
       bytes_per_call=allocated / batch, warmup, samples, batch)
end

function show_benchmark(backend, path, result; rows, dimension)
    println(
        "BRM_TURING_MM_BENCH backend=", backend,
        " path=", path,
        " median_ns=", result.median_ns,
        " minimum_ns=", result.minimum_ns,
        " bytes_per_call=", result.bytes_per_call,
        " warmup=", result.warmup,
        " samples=", result.samples,
        " batch=", result.batch,
        " rows=", rows,
        " dimension=", dimension,
    )
end

function turing_density(backend, parameters)
    vi = DP.VarInfo(
        backend.model, DP.InitFromParams(parameters), DP.UnlinkAll())
    density = DP.LogDensityFunction(
        backend.model, DP.getlogjoint_internal, vi)
    position = collect(DP.get_sample_input_vector(density))
    preparation = DI.prepare_gradient(
        turing_logdensity_kernel, ENZYME_BACKEND, position,
        DI.Constant(density))
    gradient = similar(position)
    (; density, preparation, gradient, position)
end

turing_logdensity_kernel(position, density) =
    LogDensityProblems.logdensity(density, position)
turing_logdensity_and_gradient(turing) = DI.value_and_gradient!(
    turing_logdensity_kernel, turing.gradient, turing.preparation,
    ENZYME_BACKEND, turing.position, DI.Constant(turing.density))

const N = 128
const GROUPS = 8
const x = collect(range(-1.5, 1.5; length=N))
const g1 = [1 + mod(row - 1, GROUPS) for row in 1:N]
const g2 = [1 + mod(row + 2, GROUPS) for row in 1:N]
const w1 = [1.0 + mod(row, 3) for row in 1:N]
const w2 = [1.0 + mod(2row, 5) for row in 1:N]
const y = [mod(row + 1, 6) for row in 1:N]
const data = (; x, g1, g2, w1, w2, y)

const brmi = (@brm begin
    log(lambda) ~ 1 + x + (1 + x | mm(g1, g2; weights=(w1, w2)))
    y ~ Poisson(lambda)
end)(data)

const beta = [0.1, -0.2]
const rho = 0.25
const L_matrix = [1.0 0.0; rho sqrt(1 - rho^2)]
const tau = [0.4, 0.7]
const z = collect(range(-0.4, 0.4; length=2GROUPS))
const parameters = (;
    beta_pop=beta,
    group_1_1=(;
        L=Cholesky(copy(L_matrix), 'L', 0),
        tau=tau,
        z_flat=z,
    ),
)

const block_name = "b_log_lambda_mm__g1__g2__w__w1__w2"

function stan_value(name)
    if (m = match(r"^pop_log_lambda_beta_pop\.(\d+)$", name)) !== nothing
        beta[parse(Int, m.captures[1])]
    elseif name == "$(block_name)_L.1"
        atanh(rho)
    elseif (m = match(Regex("^$(block_name)_tau\\.(\\d+)\$"), name)) !== nothing
        log(tau[parse(Int, m.captures[1])])
    elseif (m = match(Regex("^$(block_name)_z_flat\\.(\\d+)\$"), name)) !== nothing
        z[parse(Int, m.captures[1])]
    else
        error("unexpected multi-membership Stan coordinate $name")
    end
end

function project_gradients(turing_gradient, stan_gradient, stan_names)
    stan_by_name = Dict(stan_names .=> stan_gradient)
    turing_rho_gradient = turing_gradient[4] -
        rho / sqrt(1 - rho^2) * turing_gradient[5]
    turing = vcat(
        turing_gradient[1:2], turing_rho_gradient,
        turing_gradient[6:7], turing_gradient[8:end])
    stan = vcat(
        [stan_by_name["pop_log_lambda_beta_pop.$i"] for i in 1:2],
        stan_by_name["$(block_name)_L.1"] / (1 - rho^2),
        [stan_by_name["$(block_name)_tau.$i"] / tau[i] for i in 1:2],
        [stan_by_name["$(block_name)_z_flat.$i"] for i in eachindex(z)],
    )
    turing, stan
end

const turing_construction = with_logger(NullLogger()) do
    benchmark_call(
        () -> DP.LogDensityFunction(TuringBRMI(brmi).model);
        warmup=2, samples=5, batch=1)
end
const stan_lowering = benchmark_call(
    () -> SBBRMI(brmi); warmup=2, samples=5, batch=1)

const backend = TuringBRMI(brmi)
const turing = turing_density(backend, parameters)
const sb = SBBRMI(brmi)
const stan_path = joinpath(
    tempdir(), "brm-turing-mm-$(UInt(hash(BRM.stan_code(sb)))).stan")
const stan_problem = StanBlocks.stan_instantiate(sb.model; path=stan_path)
const stan_names = BS.param_unc_names(stan_problem.model)
const stan_position = stan_value.(stan_names)
const stan_gradient = zeros(length(stan_position))

const turing_lp = LogDensityProblems.logdensity(turing.density, turing.position)
const turing_lp_grad, turing_gradient =
    turing_logdensity_and_gradient(turing)
const stan_lp = BS.log_density(
    stan_problem.model, stan_position; propto=false, jacobian=false)
const stan_lp_grad, _ = BS.log_density_gradient!(
    stan_problem.model, stan_position, stan_gradient;
    propto=false, jacobian=false)
const projected_turing, projected_stan = project_gradients(
    turing_gradient, stan_gradient, stan_names)

@test turing_lp == turing_lp_grad
@test stan_lp ≈ stan_lp_grad atol=1e-12 rtol=0
@test turing_lp ≈ stan_lp rtol=5e-11 atol=5e-9
@test projected_turing ≈ projected_stan rtol=2e-8 atol=2e-7

const turing_density_bench = benchmark_call(
    () -> LogDensityProblems.logdensity(turing.density, turing.position);
    warmup=30, samples=21, batch=50)
const turing_gradient_bench = benchmark_call(
    () -> turing_logdensity_and_gradient(turing);
    warmup=20, samples=21, batch=20)
const stan_density_bench = benchmark_call(
    () -> BS.log_density(
        stan_problem.model, stan_position; propto=false, jacobian=false);
    warmup=30, samples=21, batch=50)
const stan_gradient_bench = benchmark_call(
    () -> BS.log_density_gradient!(
        stan_problem.model, stan_position, stan_gradient;
        propto=false, jacobian=false);
    warmup=20, samples=21, batch=20)

println(
    "BRM_TURING_MM_PARITY lp_abs_diff=", abs(turing_lp - stan_lp),
    " projected_gradient_max_abs_diff=",
    maximum(abs, projected_turing .- projected_stan),
    " turing_dimension=", length(turing.position),
    " stan_dimension=", length(stan_position),
)
show_benchmark("Turing+Enzyme", "model_logdensity_construction",
               turing_construction; rows=N, dimension=length(turing.position))
show_benchmark("StanBlocks", "model_lowering", stan_lowering;
               rows=N, dimension=length(stan_position))
show_benchmark("Turing", "constrained_density", turing_density_bench;
               rows=N, dimension=length(turing.position))
show_benchmark("StanBlocks+BridgeStan", "constrained_density",
               stan_density_bench; rows=N, dimension=length(stan_position))
show_benchmark("Turing+Enzyme", "constrained_gradient",
               turing_gradient_bench; rows=N, dimension=length(turing.position))
show_benchmark("StanBlocks+BridgeStan", "constrained_gradient",
               stan_gradient_bench; rows=N, dimension=length(stan_position))
println(
    "BRM_TURING_MM_SPEED construction_ratio=",
    turing_construction.median_ns / stan_lowering.median_ns,
    " density_ratio=", turing_density_bench.median_ns / stan_density_bench.median_ns,
    " gradient_ratio=", turing_gradient_bench.median_ns / stan_gradient_bench.median_ns,
)
println(
    "BRM_TURING_MM_ENV host=", gethostname(),
    " machine=", Sys.MACHINE,
    " julia=", VERSION,
    " brm=", Base.pkgversion(BRM),
    " stanblocks=", Base.pkgversion(StanBlocks),
    " turing=", Base.pkgversion(Turing),
    " enzyme=", Base.pkgversion(Enzyme),
    " bridgestan=", Base.pkgversion(BS),
    " setup=gradient prepared once; caller-owned gradient reused; warmed ",
    "allocation-aware hot-call medians; density batch 50; gradient batch 20; ",
    "sampling omitted because this benchmark targets multi-membership lowering and density kernels",
)
