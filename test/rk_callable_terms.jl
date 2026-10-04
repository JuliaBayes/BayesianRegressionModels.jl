include(joinpath(@__DIR__, "rk_consumer_support.jl"))

# Independently authored public reduction of the original callable-keyword
# formula route. The ordinary native methods retain the same scalar math.
StanBlocks.@deffun public_keyword_shift(x::vector[n]; shift::real) = x + rep_vector(shift, n)
public_keyword_shift(x::Real; shift) = x + shift
public_keyword_shift(x::AbstractVector; shift) = x .+ shift
StanBlocks.@deffun public_keyword_rows(x::vector[n]; shift::vector[k]) =
    k == 1 ? x + rep_vector(shift[1], n) : x + shift
public_keyword_rows(x; shift) = x + shift

@stestset "ordinary callable formula terms preserve keyword broadcasting" begin
    for mode in (:literal, :rows, :singleton, :nested)
        data = (; x=[0.2, -0.4, 1.1], y=[0.3, 0.1, 0.8],
            shift=mode === :singleton ? [0.25] : [0.1, -0.2, 0.3])
        before = deepcopy(data)
        brmi = if mode === :literal
            @brm data begin
                b ~ Normal(0.0, 1.0)
                mu ~ 0 + public_keyword_shift(x; shift=0.25)
                effect(mu, :) ~ Normal(0.0, 1.0)
                y ~ Normal(mu, 1.0)
            end
        elseif mode === :nested
            @brm data begin
                b ~ Normal(0.0, 1.0)
                mu ~ 0 + 3 * public_keyword_shift(x; shift=0.25)
                effect(mu, :) ~ Normal(0.0, 1.0)
                y ~ Normal(mu, 1.0)
            end
        else
            @brm data begin
                b ~ Normal(0.0, 1.0)
                mu ~ 0 + public_keyword_rows(x; shift=shift)
                effect(mu, :) ~ Normal(0.0, 1.0)
                y ~ Normal(mu, 1.0)
            end
        end
        backend, problem = consumer_problem(brmi)
        names = coordinate_names(backend.model.layout)
        @test length(names) == 1
        @test :b ∉ names
        coefficient = only(names)
        emitted = BRM._rk_emit_ast(backend.plan)
        @test !isempty(emitted.defs)
        @test any(binding -> last(binding) === (mode in (:literal, :nested) ?
            public_keyword_shift : public_keyword_rows), emitted.bindings)
        @test occursin("shift =", sprint(Base.show_unquoted, first(emitted.defs)))
        stan = consumer_stan(brmi, "callable-term-$mode")
        mapping = [coefficient => "pop_mu_beta_pop.1"]
        @test Set(BridgeStan.param_unc_names(stan.model)) == Set(last.(mapping))
        design = mode in (:literal, :nested) ? data.x .+ 0.25 : data.x .+ data.shift
        mode === :nested && (design .*= 3)
        function oracle(u)
            beta = u[findfirst(==(coefficient), names)]
            sum(logpdf.(Normal(), u)) + sum(logpdf.(Normal.(beta .* design, 1), data.y))
        end
        for u in ([0.0], [0.13], [-0.21])
            value, gradient = check_consumer_point(problem, u, oracle)
            check_consumer_stan(problem, stan, mapping, backend, u)
        end
        @test isequal(data, before)
    end
end
