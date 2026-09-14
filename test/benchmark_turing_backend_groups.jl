using BayesianRegressionModels
using BridgeStan
import DifferentiationInterface as DI
import Enzyme
using LinearAlgebra: Cholesky, Diagonal, Symmetric, cholesky
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
    for sample in 1:samples
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
    (; median_ns=median(elapsed), min_ns=minimum(elapsed),
       bytes_per_call=allocated / batch, warmup, samples, batch)
end

function show_benchmark(case, backend, path, result; N, dimension)
    println(
        "BRM_TURING_SLOPE_BENCH case=", case,
        " backend=", backend,
        " path=", path,
        " median_ns=", result.median_ns,
        " min_ns=", result.min_ns,
        " bytes_per_call=", result.bytes_per_call,
        " warmup=", result.warmup,
        " samples=", result.samples,
        " batch=", result.batch,
        " N=", N,
        " dimension=", dimension,
    )
end

function turing_density(backend, params)
    vi = DP.VarInfo(
        backend.model, DP.InitFromParams(params), DP.UnlinkAll())
    ldf = DP.LogDensityFunction(
        backend.model, DP.getlogjoint_internal, vi)
    q = collect(DP.get_sample_input_vector(ldf))
    preparation = DI.prepare_gradient(
        turing_logdensity_kernel, ENZYME_BACKEND, q, DI.Constant(ldf))
    gradient = similar(q)
    (; ldf, preparation, q, gradient)
end

turing_logdensity_kernel(q, ldf) = LogDensityProblems.logdensity(ldf, q)
turing_logdensity_and_gradient(td) = DI.value_and_gradient!(
    turing_logdensity_kernel, td.gradient, td.preparation, ENZYME_BACKEND,
    td.q, DI.Constant(td.ldf))

function instantiate_stan(sb, label)
    code = BRM.stan_code(sb)
    path = joinpath(tempdir(), "brm-turing-$label-$(UInt(hash(code))).stan")
    started = time_ns()
    problem = StanBlocks.stan_instantiate(sb.model; path)
    (; problem, milliseconds=(time_ns() - started) / 1e6,
       names=BS.param_unc_names(problem.model))
end

function run_case(case, brmi, params, stan_position, project_gradients; N,
                  centered_groups=())
    turing_construction = with_logger(NullLogger()) do
        benchmark_call(
            () -> DP.LogDensityFunction(TuringBRMI(
                brmi; centered_groups).model);
            warmup=2, samples=5, batch=1)
    end
    stan_lowering = benchmark_call(
        () -> SBBRMI(brmi; centered_groups);
        warmup=2, samples=5, batch=1)

    backend = TuringBRMI(brmi; centered_groups)
    td = turing_density(backend, params)
    sb = SBBRMI(brmi; centered_groups)
    stan = instantiate_stan(sb, case)
    stan_q = stan_position(stan.problem.model, stan.names)
    stan_gradient = zeros(length(stan_q))

    turing_lp = LogDensityProblems.logdensity(td.ldf, td.q)
    turing_lp_grad, turing_gradient = turing_logdensity_and_gradient(td)
    stan_lp = BS.log_density(
        stan.problem.model, stan_q; propto=false, jacobian=false)
    stan_lp_grad, _ = BS.log_density_gradient!(
        stan.problem.model, stan_q, stan_gradient;
        propto=false, jacobian=false)
    projected_turing, projected_stan = project_gradients(
        turing_gradient, stan_gradient, stan.names)

    @test turing_lp == turing_lp_grad
    @test stan_lp ≈ stan_lp_grad atol=1e-12 rtol=0
    @test turing_lp ≈ stan_lp rtol=5e-11 atol=5e-9
    @test projected_turing ≈ projected_stan rtol=2e-8 atol=2e-7

    LogDensityProblems.logdensity(td.ldf, td.q)
    turing_logdensity_and_gradient(td)
    BS.log_density(stan.problem.model, stan_q; propto=false, jacobian=false)
    BS.log_density_gradient!(
        stan.problem.model, stan_q, stan_gradient;
        propto=false, jacobian=false)

    turing_density_bench = benchmark_call(
        () -> LogDensityProblems.logdensity(td.ldf, td.q);
        warmup=30, samples=21, batch=50)
    turing_gradient_bench = benchmark_call(
        () -> turing_logdensity_and_gradient(td);
        warmup=20, samples=21, batch=20)
    stan_density_bench = benchmark_call(
        () -> BS.log_density(
            stan.problem.model, stan_q; propto=false, jacobian=false);
        warmup=30, samples=21, batch=50)
    stan_gradient_bench = benchmark_call(
        () -> BS.log_density_gradient!(
            stan.problem.model, stan_q, stan_gradient;
            propto=false, jacobian=false);
        warmup=20, samples=21, batch=20)

    println(
        "BRM_TURING_SLOPE_PARITY case=", case,
        " lp_abs_diff=", abs(turing_lp - stan_lp),
        " projected_gradient_max_abs_diff=",
        maximum(abs, projected_turing .- projected_stan),
        " stan_instantiate_ms=", stan.milliseconds,
        " turing_dimension=", length(td.q),
        " stan_dimension=", length(stan_q),
    )
    show_benchmark(case, "Turing+Enzyme", "model_logdensity_construction",
                   turing_construction; N, dimension=length(td.q))
    show_benchmark(case, "StanBlocks", "model_lowering", stan_lowering;
                   N, dimension=length(stan_q))
    show_benchmark(case, "Turing", "constrained_density",
                   turing_density_bench; N, dimension=length(td.q))
    show_benchmark(case, "StanBlocks+BridgeStan", "constrained_density",
                   stan_density_bench; N, dimension=length(stan_q))
    show_benchmark(case, "Turing+Enzyme", "constrained_gradient",
                   turing_gradient_bench; N, dimension=length(td.q))
    show_benchmark(case, "StanBlocks+BridgeStan", "constrained_gradient",
                   stan_gradient_bench; N, dimension=length(stan_q))
    println(
        "BRM_TURING_SLOPE_SPEED case=", case,
        " construction_ratio=",
        turing_construction.median_ns / stan_lowering.median_ns,
        " density_ratio=",
        turing_density_bench.median_ns / stan_density_bench.median_ns,
        " gradient_ratio=",
        turing_gradient_bench.median_ns / stan_gradient_bench.median_ns,
    )
end

const N = 128
const G = 8
const x = collect(range(-1.5, 1.5; length=N))
const subject = [1 + mod(i, G) for i in 1:N]
const y = [mod(i + 1, 6) for i in 1:N]
const data = (; x, subject, y)
const beta = [0.1, -0.2]
const z = collect(range(-0.4, 0.4; length=2G))

const correlated_brmi = (@brm begin
    log(lambda) ~ 1 + x + (1 + x | subject)
    y ~ Poisson(lambda)
end)(data)
const rho = 0.25
const correlated_L_matrix = [1.0 0.0; rho sqrt(1 - rho^2)]
const correlated_tau = [0.4, 0.7]
const correlated_params = (;
    beta_pop=beta,
    L_group=Cholesky(copy(correlated_L_matrix), 'L', 0),
    tau_group=correlated_tau,
    z_group_flat=z,
)

function correlated_stan_value(name)
    if (m = match(r"^pop_log_lambda_beta_pop\.(\d+)$", name)) !== nothing
        beta[parse(Int, m.captures[1])]
    elseif name == "r_log_lambda_subject_L.1"
        atanh(rho)
    elseif (m = match(r"^r_log_lambda_subject_tau\.(\d+)$", name)) !== nothing
        log(correlated_tau[parse(Int, m.captures[1])])
    elseif (m = match(r"^r_log_lambda_subject_z_flat\.(\d+)$", name)) !== nothing
        z[parse(Int, m.captures[1])]
    else
        error("unexpected correlated Stan coordinate $name")
    end
end

function correlated_project(turing_gradient, stan_gradient, stan_names)
    stan_by_name = Dict(stan_names .=> stan_gradient)
    turing_rho_gradient = turing_gradient[4] -
        rho / sqrt(1 - rho^2) * turing_gradient[5]
    turing = vcat(
        turing_gradient[1:2], turing_rho_gradient,
        turing_gradient[6:7], turing_gradient[8:end])
    stan = vcat(
        [stan_by_name["pop_log_lambda_beta_pop.$i"] for i in 1:2],
        stan_by_name["r_log_lambda_subject_L.1"] / (1 - rho^2),
        [stan_by_name["r_log_lambda_subject_tau.$i"] /
         correlated_tau[i] for i in 1:2],
        [stan_by_name["r_log_lambda_subject_z_flat.$i"] for i in 1:length(z)],
    )
    turing, stan
end

if "correlated_poisson" in ARGS
    run_case("correlated_poisson", correlated_brmi, correlated_params,
             (_model, names) -> correlated_stan_value.(names),
             correlated_project; N)
end

const centered_coefficients = transpose(
    Diagonal(correlated_tau) * correlated_L_matrix * reshape(z, 2, G))
const centered_flat = vec(transpose(centered_coefficients))
const centered_params = (;
    beta_pop=beta,
    group_1_1=(;
        L=Cholesky(copy(correlated_L_matrix), 'L', 0),
        tau=correlated_tau,
        coefficients_flat=centered_flat,
    ),
)

json_array(values) = "[" * join(string.(values), ",") * "]"
json_matrix(matrix) = "[" * join(
    (json_array(@view matrix[i, :]) for i in axes(matrix, 1)), ",") * "]"

function centered_stan_position(model, _names)
    json = "{" * join([
        "\"pop_log_lambda_beta_pop\":" * json_array(beta),
        "\"r_log_lambda_subject_L\":" * json_matrix(correlated_L_matrix),
        "\"r_log_lambda_subject_tau\":" * json_array(correlated_tau),
        "\"r_log_lambda_subject_b\":" * json_matrix(centered_coefficients),
    ], ",") * "}"
    BS.param_unconstrain_json(model, json)
end

function centered_project(turing_gradient, stan_gradient, stan_names)
    stan_by_name = Dict(stan_names .=> stan_gradient)
    turing_rho_gradient = turing_gradient[4] -
        rho / sqrt(1 - rho^2) * turing_gradient[5]
    turing = vcat(
        turing_gradient[1:2], turing_rho_gradient,
        turing_gradient[6:7], turing_gradient[8:end])
    stan = vcat(
        [stan_by_name["pop_log_lambda_beta_pop.$i"] for i in 1:2],
        stan_by_name["r_log_lambda_subject_L.1"] / (1 - rho^2),
        [stan_by_name["r_log_lambda_subject_tau.$i"] /
         correlated_tau[i] for i in 1:2],
        # BridgeStan's names enumerate this array-of-vectors term-major, while
        # its unconstrained position and gradient are group-major. The JSON
        # position above establishes the semantic order; retain that order.
        stan_gradient[6:end],
    )
    turing, stan
end

if "centered_correlated_poisson" in ARGS
    run_case(
        "centered_correlated_poisson", correlated_brmi, centered_params,
        centered_stan_position, centered_project;
        N, centered_groups=(:subject,))
end

const zero_brmi = (@brm begin
    log(lambda) ~ 1 + x + (1 + x || subject)
    y ~ Poisson(lambda)
end)(data)
const intercept_log_scale = log(0.6)
const slope_tau = 0.7
const zero_params = (;
    beta_pop=beta,
    group_1_1=(;
        log_intercept_scale=intercept_log_scale,
        tau_slopes=[slope_tau],
        z_flat=z,
    ),
)

function zero_stan_value(name)
    if (m = match(r"^pop_log_lambda_beta_pop\.(\d+)$", name)) !== nothing
        beta[parse(Int, m.captures[1])]
    elseif name == "r_log_lambda_subject__nocor__1_log_scale"
        intercept_log_scale
    elseif (m = match(r"^r_log_lambda_subject__nocor__1_xi\.(\d+)$", name)) !== nothing
        z[2 * parse(Int, m.captures[1]) - 1]
    elseif name == "r_log_lambda_subject__nocor__2_tau.1"
        log(slope_tau)
    elseif (m = match(r"^r_log_lambda_subject__nocor__2_z_flat\.(\d+)$", name)) !== nothing
        z[2 * parse(Int, m.captures[1])]
    else
        error("unexpected zero-correlation Stan coordinate $name")
    end
end

function zero_project(turing_gradient, stan_gradient, stan_names)
    stan_by_name = Dict(stan_names .=> stan_gradient)
    turing = turing_gradient
    stan_z = collect(Iterators.flatten((
        (stan_by_name["r_log_lambda_subject__nocor__1_xi.$i"],
         stan_by_name["r_log_lambda_subject__nocor__2_z_flat.$i"])
        for i in 1:G)))
    stan = vcat(
        [stan_by_name["pop_log_lambda_beta_pop.$i"] for i in 1:2],
        stan_by_name["r_log_lambda_subject__nocor__1_log_scale"],
        stan_by_name["r_log_lambda_subject__nocor__2_tau.1"] / slope_tau,
        stan_z,
    )
    turing, stan
end

if "zero_correlation_poisson" in ARGS
    run_case("zero_correlation_poisson", zero_brmi, zero_params,
             (_model, names) -> zero_stan_value.(names), zero_project; N)
end

const shared_z = reverse(x)
const shared_brmi = (@brm begin
    log(mu) ~ 1 + x + (1 + x | joint | subject)
    log(phi) ~ 1 + shared_z + (1 + shared_z | joint | subject)
    y ~ BRM.NegativeBinomial2(mu, phi)
end)((; x, shared_z, subject, y))
const shared_beta_mean = [0.1, -0.2]
const shared_beta_precision = [-0.35, 0.15]
const shared_correlation = [
    1.0 0.20 -0.10 0.15
    0.20 1.0 0.25 -0.20
    -0.10 0.25 1.0 0.30
    0.15 -0.20 0.30 1.0
]
const shared_L = Cholesky(
    Matrix(cholesky(Symmetric(shared_correlation)).L), 'L', 0)
const shared_tau = [0.4, 0.7, 0.25, 0.45]
const shared_latent = collect(range(-0.6, 0.6; length=4G))
const shared_params = (;
    beta_pop=shared_beta_mean,
    beta_pop_phi=shared_beta_precision,
    shared_group_1=(;
        L=shared_L,
        tau=shared_tau,
        z_flat=shared_latent,
    ),
)

function shared_stan_position(model, _names)
    json = "{" * join([
        "\"b_joint_subject_L\":" * json_matrix(shared_L.L),
        "\"b_joint_subject_tau\":" * json_array(shared_tau),
        "\"b_joint_subject_z_flat\":" * json_array(shared_latent),
        "\"pop_log_mu_beta_pop\":" * json_array(shared_beta_mean),
        "\"pop_log_phi_beta_pop\":" * json_array(shared_beta_precision),
    ], ",") * "}"
    BS.param_unconstrain_json(model, json)
end

function shared_project(turing_gradient, stan_gradient, stan_names)
    stan_by_name = Dict(stan_names .=> stan_gradient)
    # DynamicPPL exposes the constrained Cholesky's ten lower-triangular
    # entries between the four population coefficients and four scales. The
    # covariance path is exercised at a non-identity L; this projection compares
    # every other shared semantic coordinate without conflating the backends'
    # different Cholesky transforms.
    @test length(turing_gradient) == 4 + 10 + 4 + 4G
    turing = vcat(
        turing_gradient[1:4], turing_gradient[15:18],
        turing_gradient[19:end])
    stan = vcat(
        [stan_by_name["pop_log_mu_beta_pop.$i"] for i in 1:2],
        [stan_by_name["pop_log_phi_beta_pop.$i"] for i in 1:2],
        [stan_by_name["b_joint_subject_tau.$i"] / shared_tau[i] for i in 1:4],
        [stan_by_name["b_joint_subject_z_flat.$i"] for i in 1:4G],
    )
    turing, stan
end

if isempty(ARGS) || "shared_distributional_id" in ARGS
    run_case("shared_distributional_id", shared_brmi, shared_params,
             shared_stan_position, shared_project; N)
end

println(
    "BRM_TURING_SLOPE_ENV host=", gethostname(),
    " machine=", Sys.MACHINE,
    " julia=", VERSION,
    " brm=", Base.pkgversion(BRM),
    " turing=", Base.pkgversion(Turing),
    " enzyme=", Base.pkgversion(Enzyme),
    " bridgestan=", Base.pkgversion(BS),
    " setup=gradient prepared once; caller-owned gradient reused; warmed ",
    "allocation-aware hot-call medians; density batch 50; gradient batch 20; ",
    "sampling omitted because this benchmark targets shared group lowering and density kernels",
)
