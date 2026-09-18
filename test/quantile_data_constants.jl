# test/quantile_data_constants.jl — named numeric constants + builder hygiene gate.
#
# Snag bambi-quantile-r-36bd8217: `bmi ~ SkewDoubleExponential(mu, sigma, tau)`
# with `tau` a captured numeric (a global Float64, or a loop local) died at
# StanBlocks trace time. BRM classifies every bare RHS symbol as a data-side
# nonlocal and rebinds it via `@getproperty` (column-or-MissingColumn), so the
# scope binding is shadowed and a bare `tau` reaches the tracer — which refuses
# `Number` module bindings deliberately (StanBlocks decision `3bbtrv`), or
# cannot find a loop local at all. The sanctioned spelling is scalar-in-data:
# one builder, the constant supplied per fit through the data container.
# The same snag exposed a second defect: the builder body assigned formula
# names with no local scope, so a builder defined in local scope CAPTURED
# same-named enclosing bindings and overwrote them when it ran (an in-loop
# `@brm` silently replaced the loop variable with a NamedColumn). The builder
# body is now `let`-wrapped (src/macro.jl `_brm`).
# Transpile + stanc gate; no BridgeStan compile.

using Test
using BayesianRegressionModels
using StanBlocks
import StanBlocks.stan: transpiles

const BRM = BayesianRegressionModels

quantile_vecs() = (
    bmi=Float64[1.8, 2.0, 2.1, 2.3, 2.5],
    agez=Float64[-1.0, -0.5, 0.0, 0.5, 1.0],
)

# One builder, reused across fits: the constant arrives via data, never scope.
quantile_builder() = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + agez
    bmi ~ SkewDoubleExponential(mu, sigma, tau)
end

quantile_literal_builder() = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + agez
    bmi ~ SkewDoubleExponential(mu, sigma, 0.5)
end

function quantile_code(brmi)
    sb = SBBRMI(brmi; mod=@__MODULE__)
    code = BRM.stan_code(sb)
    @test transpiles(sb.model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    sb, code
end

# A data-constant program is the literal program with the data declaration
# plus symbol-for-literal substitution at the model and generated-quantities
# call sites — nothing else. The substitution stays out of the functions
# block: it declares and forwards a Stan parameter of the same name in BOTH
# programs, so touching it would corrupt the comparison.
function strip_data_constant(code, name)
    lines = filter!(split(code, '\n')) do line
        strip(line) != "real $name;"
    end
    code = join(lines, '\n')
    m = findfirst("\nmodel {", code)
    isnothing(m) && error("strip_data_constant: no model block in emitted Stan")
    head, tail = code[1:first(m)], code[first(m)+1:end]
    head * replace(tail, Regex(",\\s*$name\\)") => ", 0.5)")
end

@testset "scalar-in-data declares Stan data and matches the literal program" begin
    builder = quantile_builder()
    sb, code = quantile_code(builder(merge(quantile_vecs(), (; tau=0.5))))
    @test occursin("real tau;", code)
    @test occursin("skew_double_exponential(mu, sigma, tau);", code)
    @test occursin("skew_double_exponential_lpdfs(bmi, mu, sigma, tau)", code)
    @test occursin("skew_double_exponential_vector_rng(bmi_n, mu, sigma, tau)", code)
    @test sb.model.data[:tau] == 0.5

    _, literal_code = quantile_code(quantile_literal_builder()(quantile_vecs()))
    @test strip_data_constant(code, "tau") == literal_code
end

@testset "one builder serves a loop over constants; loop state survives" begin
    builder = quantile_builder()
    seen = Float64[]
    for tau in (0.1, 0.5, 0.9)
        sb, code = quantile_code(builder(merge(quantile_vecs(), (; tau))))
        @test sb.model.data[:tau] == tau
        @test occursin("skew_double_exponential(mu, sigma, tau);", code)
        # The builder ran above: the loop variable must be untouched.
        @test tau isa Float64
        push!(seen, tau)
    end
    @test seen == [0.1, 0.5, 0.9]
end

@testset "builder run never clobbers enclosing locals it shadows" begin
    # In-loop `@brm` mentioning the loop variable: BRMI construction succeeds
    # (the name binds MissingColumn inside the builder) and the loop variable
    # survives the run. Only the trace fails — loudly — since nothing declares
    # the name as data.
    for tau_loop in (0.1, 0.9)
        builder = @brm begin
            sigma ~ Exponential(1)
            mu ~ 1 + agez
            bmi ~ SkewDoubleExponential(mu, sigma, tau_loop)
        end
        brmi = builder(quantile_vecs())
        @test tau_loop isa Float64
        @test tau_loop in (0.1, 0.9)
        err = try
            quantile_code(brmi)
            nothing
        catch e
            e
        end
        @test err isa Exception
        @test occursin("tau_loop", sprint(showerror, err))
    end

    # Function-scope collision: a local named like a formula name survives.
    function shadowed_mu()
        mu = 42.0
        builder = @brm begin
            sigma ~ Exponential(1)
            mu ~ 1 + agez
            bmi ~ SkewDoubleExponential(mu, sigma, 0.5)
        end
        brmi = builder(quantile_vecs())
        mu, brmi
    end
    mu_after, brmi = shadowed_mu()
    @test mu_after == 42.0
    quantile_code(brmi)
end

# A scope binding the formula shadows is invisible to the builder: a global
# without a data column still fails loudly at trace time (StanBlocks `3bbtrv`
# refusal), never silently. Kept in its own testset after the data testsets
# so the global cannot leak into them.
tau_global = 0.5

@testset "unbound scope name still fails loudly, never silently" begin
    builder = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + agez
        bmi ~ SkewDoubleExponential(mu, sigma, tau_global)
    end
    err = try
        quantile_code(builder(quantile_vecs()))
        nothing
    catch e
        e
    end
    @test err isa Exception
    @test occursin("tau_global", sprint(showerror, err))
end
