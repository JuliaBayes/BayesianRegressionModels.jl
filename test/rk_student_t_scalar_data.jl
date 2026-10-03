# Public synthetic scalar-data Student-t acceptance.
# Run: julia --project=test test/rk_student_t_scalar_data.jl
using Test, BayesianRegressionModels, Distributions, Enzyme, LogDensityProblems
using ReactiveKernels, ReactiveKernelsPPL, StanBlocks
using DifferentiationInterface: AutoEnzyme
import BridgeStan
const BRM = BayesianRegressionModels

const scalar_t_data = (; x=[-0.5, 0.0, 0.5], y=[-1.0, 0.1, 1.0],
    lo=fill(-1.0, 3), hi=fill(1.0, 3))
const scalar_t_builder = @brm begin
    mu ~ 1 + x
    y ~ censored(LocationScale(mu, 1, TDist(nu)); lower=lo, upper=hi)
end
const literal_t_builder = @brm begin
    mu ~ 1 + x
    y ~ censored(LocationScale(mu, 1, TDist(6.0)); lower=lo, upper=hi)
end
const unwrapped_t_builder = @brm begin
    mu ~ 1 + x
    y ~ LocationScale(mu, 1, TDist(nu))
end
const truncated_t_builder = @brm begin
    mu ~ 1 + x
    y ~ truncated(LocationScale(mu, 1, TDist(nu)); lower=lo, upper=hi)
end
const interval_t_builder = @brm begin
    mu ~ 1 + x
    y ~ interval_censored(LocationScale(mu, 1, TDist(nu)); upper=hi)
end
const literal_bounds_t_builder = @brm begin
    mu ~ 1 + x
    y ~ censored(LocationScale(mu, 1, TDist(nu)); lower=-1.0, upper=1.0)
end

function scalar_t_findiff(f, u)
    h = 1e-5
    map(eachindex(u)) do j
        up, down = copy(u), copy(u)
        up[j] += h
        down[j] -= h
        (f(up) - f(down)) / (2h)
    end
end

function scalar_t_parse_source(expression)
    source = sprint(Base.show_unquoted, expression)
    parsed = Base.JuliaSyntax.parseall(Expr, source)
    only(filter(x -> !(x isa LineNumberNode), parsed.args))
end

# Reparse the complete printed program, including every submodel definition,
# and use the public authoring API independently of BRM's translator.
function scalar_t_retyped(backend)
    emitted = BRM._rk_emit_ast(backend.plan)
    mod = Module(gensym(:ScalarStudentT))
    Core.eval(mod, :(using ReactiveKernelsPPL))
    @test isempty(emitted.bindings)
    for definition in emitted.defs
        Core.eval(mod, Expr(:macrocall, Symbol("@rkppl"), LineNumberNode(0),
            scalar_t_parse_source(definition)))
    end
    plan = bind_data(lower_rkppl(scalar_t_parse_source(emitted.main),
        backend.plan.columns; mod, conditioned=(:y,)), backend.plan.columns)
    built = build_kernel(plan)
    built, plan
end

function scalar_t_observation_oracle(backend, data, u, evidence)
    physical = constrain(backend.model.layout, u)
    mu = physical.mu_Intercept .+ physical.mu_x .* data.x
    map(eachindex(data.y)) do j
        distribution = LocationScale(mu[j], 1, TDist(data.nu))
        if evidence === :censored && data.y[j] == data.lo[j]
            logcdf(distribution, data.lo[j])
        elseif evidence === :censored && data.y[j] == data.hi[j]
            logccdf(distribution, data.hi[j])
        elseif evidence === :truncated
            logpdf(distribution, data.y[j]) -
                log(cdf(distribution, data.hi[j]) - cdf(distribution, data.lo[j]))
        elseif evidence === :interval_censored
            log(cdf(distribution, data.hi[j]) - cdf(distribution, data.y[j]))
        else
            logpdf(distribution, data.y[j])
        end
    end
end

@testset "scalar-data Student-t full emission and native execution" begin
    original = deepcopy(scalar_t_data)
    literal = RKBRMI(literal_t_builder(scalar_t_data))
    fixed = RKBRMI(scalar_t_builder(merge(scalar_t_data, (; nu=6.0))))
    fixed_program, literal_program = BRM._rk_emit_ast(fixed.plan), BRM._rk_emit_ast(literal.plan)
    @test fixed_program.defs == literal_program.defs
    @test fixed_program.main == literal_program.main
    @test fixed.plan.columns == literal.plan.columns
    @test coordinate_names(fixed.model.layout) == coordinate_names(literal.model.layout)
    changed = RKBRMI(scalar_t_builder(merge(scalar_t_data, (; nu=3.5))))
    stale_literal = RKBRMI(literal_t_builder(merge(scalar_t_data, (; nu=3.5))))
    @test BRM._rk_emit_ast(changed.plan).main != BRM._rk_emit_ast(stale_literal.plan).main
    @test only(changed.plan.responses).nu == 3.5
    @test only(stale_literal.plan.responses).nu == 6.0

    for (builder, nu, evidence) in ((scalar_t_builder, 6.0, :censored),
            (scalar_t_builder, 3.5, :censored), (unwrapped_t_builder, 12, :none),
            (truncated_t_builder, 6.0, :truncated),
            (interval_t_builder, 6.0, :interval_censored),
            (literal_bounds_t_builder, 6.0, :censored))
        @testset "nu=$nu evidence=$evidence" begin
            data = merge(scalar_t_data, (; nu))
            evidence === :interval_censored &&
                (data = merge(data, (; y=[-0.5, 0.0, 0.5])))
            brmi = builder(data)
            backend = RKBRMI(brmi)
            translated = Base.get_extension(BRM,
                :BayesianRegressionModelsReactiveKernelsExt)._rk_translated_plan(backend.plan)
            pointwise = prepare_query(backend.model, translated, :pointwise)
            likelihood = prepare_query(backend.model, translated, :likelihood)
            problem = rk_logdensity_problem(backend;
                ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
            retyped, retyped_plan = scalar_t_retyped(backend)
            retyped_query = prepare_sampler(retyped, retyped_plan, zeros(2);
                backend=AutoEnzyme(; mode=Enzyme.Reverse))
            @test LogDensityProblems.dimension(problem) == 2
            @test isempty(backend.plan.parameters)
            @test coordinate_names(retyped.layout) == coordinate_names(backend.model.layout)
            stan = if builder === scalar_t_builder && nu == 6.0
                # Compare the same BRMI, including normalization constants,
                # after matching named coordinates rather than their order.
                sb = SBBRMI(brmi; mod=@__MODULE__)
                cache = mktempdir(; prefix="brm-scalar-student-t-")
                stan_problem = BRM.stan_instantiate(sb; path=joinpath(cache, "model.stan"))
                names = BridgeStan.param_unc_names(stan_problem.model)
                mapping = Dict(:mu_Intercept => "pop_mu_beta_pop.1",
                    :mu_x => "pop_mu_beta_pop.2")
                indices = Int.(indexin(map(n -> mapping[n],
                    coordinate_names(backend.model.layout)), names))
                @test length(names) == length(unique(indices)) == 2
                (; problem=stan_problem, indices)
            else
                nothing
            end
            for u in ([0.0, 0.0], [0.2, -0.3], [-0.4, 0.5])
                original_u = copy(u)
                value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
                oracle = scalar_t_observation_oracle(backend, data, u, evidence)
                @test Base.invokelatest(pointwise, u).y ≈ oracle atol=1e-10 rtol=1e-10
                @test Base.invokelatest(likelihood, u) ≈ sum(oracle) atol=1e-10 rtol=1e-10
                @test isfinite(value)
                @test all(isfinite, gradient)
                @test value ≈ sum(oracle) + sum(logpdf.(Normal(), u)) atol=1e-10 rtol=1e-10
                @test gradient ≈ scalar_t_findiff(
                    v -> LogDensityProblems.logdensity(problem, v), u) atol=1e-7 rtol=1e-5
                @test retyped_query(u) == value
                replay_gradient = similar(u)
                replay_value, _ = sampler_value_and_gradient!(retyped_query, replay_gradient, u)
                @test replay_value == value
                @test replay_gradient == gradient
                if !isnothing(stan)
                    q, g = similar(u), similar(u)
                    q[stan.indices] = u
                    stan_value, _ = BridgeStan.log_density_gradient!(
                        stan.problem.model, q, g; propto=false, jacobian=true)
                    @test value ≈ stan_value atol=1e-10 rtol=1e-10
                    @test gradient ≈ g[stan.indices] atol=1e-9 rtol=1e-9
                end
                @test u == original_u
            end
        end
    end
    @test scalar_t_data == original
end
