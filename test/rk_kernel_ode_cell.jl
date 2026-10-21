# Stan's ODE solvers in a kernel cell on RKBRMI. The cell SBBRMI compiles to
# Stan's `ode_rk45`/`ode_bdf` (stanblocks-use §29) runs RK's native ODE bridges
# unchanged. Needs ReactiveKernels with `rk_ode_rk45_tol` (ce48fc77 or later)
# and the solver packages, which test/Project.toml does not carry: run it from
# a copy of the test environment that adds them (test/README.md).
# Run: julia --startup-file=no --project=<env> test/rk_kernel_ode_cell.jl
using OrdinaryDiffEqLowOrderRK, OrdinaryDiffEqBDF, SciMLBase, SciMLSensitivity
include(joinpath(@__DIR__, "rk_consumer_support.jl"))

# One-compartment elimination, written once for both backends.
StanBlocks.@deffun @juliacompat elimination(t::real, y::vector[ny], k::real)::vector[ny] =
    -k * y

# Per-subject observation times of unequal lengths.
const ODE_DATA = (; subject=["a", "b", "c"], dose=[100.0, 80.0, 120.0],
    weight=[0.1, -0.2, 0.3], t=[[0.5, 1.0, 2.0], [0.5, 3.0], [1.0, 2.0, 4.0, 8.0]],
    dv=[[18.0, 15.0, 11.0], [20.0, 8.0], [16.0, 12.0, 7.0, 3.0]])

# One population-PK cell whose concentration is the closed form or an ODE
# solve of it. The tolerance-controlled solves ask for far more accuracy than
# Stan's defaults.
const ODE_PREDICTIONS = (
    closed=:(dose[i] / Vc .* exp.(-k .* t[i])),
    rk45=:(to_vector(ode_rk45(elimination, rep_vector(dose[i] / Vc, 1), 0.0,
        to_array_1d(t[i]), k)[:, 1])),
    rk45_tol=:(to_vector(ode_rk45_tol(elimination, rep_vector(dose[i] / Vc, 1), 0.0,
        to_array_1d(t[i]), 1e-10, 1e-10, 100000, k)[:, 1])),
    bdf=:(to_vector(ode_bdf(elimination, rep_vector(dose[i] / Vc, 1), 0.0,
        to_array_1d(t[i]), k)[:, 1])),
    bdf_tol=:(to_vector(ode_bdf_tol(elimination, rep_vector(dose[i] / Vc, 1), 0.0,
        to_array_1d(t[i]), 1e-10, 1e-10, 100000, k)[:, 1])),
)
for (name, prediction) in pairs(ODE_PREDICTIONS)
    @eval $(Symbol(:ode_pk_, name))(data) = @brm data begin
        sigma ~ Exponential(1)
        log_CL ~ 1 + weight + (1 | pk | subject)
        log_V ~ 1 + (1 | pk | subject)
        @plate for i in eachindex(log_CL)
            CL = exp(log_CL[i])
            Vc = exp(log_V[i])
            k = CL / Vc
            pred[i] = $prediction
            dv[i] ~ normal(pred[i], sigma)
        end
    end
end

const ODE_POINTS = (u -> zeros(length(u)), u -> [0.15 * sin(j) for j in eachindex(u)])

@stestset "ODE cells on RKBRMI solve to the closed-form density" begin
    closed_backend, closed = consumer_problem(ode_pk_closed(ODE_DATA))
    names = coordinate_names(closed_backend.model.layout)
    # Relative agreement with the closed form: about 3e-7 measured at Stan's
    # default solver tolerances (1e-6), about 1e-10 at 1e-10.
    for (name, rtol) in pairs((rk45=2e-6, bdf=2e-6, rk45_tol=1e-8, bdf_tol=1e-8))
        backend, problem = consumer_problem(getfield(@__MODULE__,
            Symbol(:ode_pk_, name))(ODE_DATA))
        @test coordinate_names(backend.model.layout) == names
        for point in ODE_POINTS
            u = point(names)
            before = copy(u)
            value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
            expected, expected_gradient = LogDensityProblems.logdensity_and_gradient(closed, u)
            println("ODE_DEVIATION ", name, " value=", abs(value - expected),
                " gradient=", maximum(abs, gradient - expected_gradient))
            @test value ≈ expected rtol=rtol
            @test gradient ≈ expected_gradient rtol=rtol
            @test isequal(u, before)
        end
    end
end

@stestset "ODE cells keep RK and compiled Stan densities" begin
    for name in (:rk45_tol, :bdf_tol)
        brmi = getfield(@__MODULE__, Symbol(:ode_pk_, name))(ODE_DATA)
        backend, problem = consumer_problem(brmi)
        sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
        code = BRM.stan_code(sb)
        @test occursin(string("ode_", replace(string(name), "_tol" => ""), "_tol("), code)
        stan = BRM.stan_instantiate(sb; path=joinpath(mktempdir(), "ode_$name.stan"))
        transport = brm_coordinate_transport(backend, sb,
            BridgeStan.param_unc_names(stan.model))
        for point in ODE_POINTS
            u = point(transport.rk_names)
            value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
            stan_u = brm_rk_point_to_stan(transport, u)
            stan_gradient = similar(stan_u)
            stan_value, _ = BridgeStan.log_density_gradient!(stan.model, stan_u,
                stan_gradient; propto=false, jacobian=true)
            stan_rk_gradient = brm_stan_gradient_to_rk(transport, u, stan_gradient)
            println("ODE_STAN_DEVIATION ", name, " value=",
                abs(value - stan_value - transport.logdensity_offset),
                " gradient=", maximum(abs, gradient - stan_rk_gradient))
            @test value ≈ stan_value + transport.logdensity_offset rtol=1e-8
            @test gradient ≈ stan_rk_gradient rtol=1e-8
        end
    end
end
