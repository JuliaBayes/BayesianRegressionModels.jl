using BayesianRegressionModels
using BridgeStan
import DifferentiationInterface as DI
import Enzyme
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
        "BRM_TURING_CROSSED_BENCH backend=", backend,
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

function turing_density(backend, params)
    vi = DP.VarInfo(backend.model, DP.InitFromParams(params), DP.UnlinkAll())
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
const SUBJECTS = 8
const ITEMS = 6
const x = collect(range(-1.5, 1.5; length=N))
const subject = [1 + mod(i, SUBJECTS) for i in 1:N]
const item = [1 + mod(5i, ITEMS) for i in 1:N]
const y = [mod(i + 1, 6) for i in 1:N]
const brmi = (@brm begin
    log(lambda) ~ 1 + x + (1 | subject) + (1 | item)
    y ~ Poisson(lambda)
end)((; x, subject, item, y))

const beta = [0.1, -0.2]
const subject_log_scale = log(0.6)
const item_log_scale = log(0.45)
const subject_z = collect(range(-0.4, 0.4; length=SUBJECTS))
const item_z = collect(range(0.3, -0.3; length=ITEMS))
const params = (;
    beta_pop=beta,
    group_1_1=(; log_scale=subject_log_scale, z=subject_z),
    group_1_2=(; log_scale=item_log_scale, z=item_z),
)

function stan_value(name)
    if (m = match(r"^pop_log_lambda_beta_pop\.(\d+)$", name)) !== nothing
        beta[parse(Int, m.captures[1])]
    elseif name == "r_log_lambda_subject_log_scale"
        subject_log_scale
    elseif (m = match(
            r"^r_log_lambda_subject_xi\.(\d+)$", name)) !== nothing
        subject_z[parse(Int, m.captures[1])]
    elseif name == "r_log_lambda_item_log_scale"
        item_log_scale
    elseif (m = match(r"^r_log_lambda_item_xi\.(\d+)$", name)) !== nothing
        item_z[parse(Int, m.captures[1])]
    else
        error("unexpected crossed-intercept Stan coordinate $name")
    end
end

const turing_construction = with_logger(NullLogger()) do
    benchmark_call(
        () -> DP.LogDensityFunction(TuringBRMI(brmi).model);
        warmup=2, samples=5, batch=1)
end
const stan_lowering = benchmark_call(
    () -> SBBRMI(brmi); warmup=2, samples=5, batch=1)

const backend = TuringBRMI(brmi)
const turing = turing_density(backend, params)
const sb = SBBRMI(brmi)
const stan_code = BRM.stan_code(sb)
const stan_path = joinpath(
    tempdir(), "brm-turing-crossed-$(UInt(hash(stan_code))).stan")
const stan_started = time_ns()
const stan_problem = StanBlocks.stan_instantiate(sb.model; path=stan_path)
const stan_instantiate_ms = (time_ns() - stan_started) / 1e6
const stan_names = BS.param_unc_names(stan_problem.model)
const stan_position = stan_value.(stan_names)
const stan_gradient = zeros(length(stan_position))

const turing_lp = LogDensityProblems.logdensity(
    turing.density, turing.position)
const turing_lp_gradient, turing_gradient =
    turing_logdensity_and_gradient(turing)
const stan_lp = BS.log_density(
    stan_problem.model, stan_position; propto=false, jacobian=false)
const stan_lp_gradient, _ = BS.log_density_gradient!(
    stan_problem.model, stan_position, stan_gradient;
    propto=false, jacobian=false)

@test turing_lp == turing_lp_gradient
@test stan_lp ≈ stan_lp_gradient atol=1e-12 rtol=0
@test turing_lp ≈ stan_lp rtol=5e-11 atol=5e-9
@test turing_gradient ≈ stan_gradient rtol=2e-8 atol=2e-7

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
    "BRM_TURING_CROSSED_PARITY lp_abs_diff=", abs(turing_lp - stan_lp),
    " gradient_max_abs_diff=",
    maximum(abs, turing_gradient .- stan_gradient),
    " stan_instantiate_ms=", stan_instantiate_ms,
)
show_benchmark("Turing", "model_logdensity_construction",
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
    "BRM_TURING_CROSSED_SPEED construction_ratio=",
    turing_construction.median_ns / stan_lowering.median_ns,
    " density_ratio=",
    turing_density_bench.median_ns / stan_density_bench.median_ns,
    " gradient_ratio=",
    turing_gradient_bench.median_ns / stan_gradient_bench.median_ns,
)
println(
    "BRM_TURING_CROSSED_ENV host=", gethostname(),
    " machine=", Sys.MACHINE,
    " julia=", VERSION,
    " brm=", Base.pkgversion(BRM),
    " turing=", Base.pkgversion(Turing),
    " enzyme=", Base.pkgversion(Enzyme),
    " bridgestan=", Base.pkgversion(BS),
    " setup=gradient prepared once; caller-owned gradient reused; warmed ",
    "allocation-aware hot-call medians; sampling omitted because this ",
    "benchmark targets crossed-group construction and density kernels",
)
