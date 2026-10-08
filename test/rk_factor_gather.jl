# Categorical population effects gather their coefficients by level from one
# index column per factor coding, and identical design matrices are emitted once.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))

@stestset "factor effects gather by a shared level index" begin
    data = (; g=[2, 1, 3, 2, 3, 1], x=[-.4, .1, .3, .8, -.2, .5],
        w=[.3, -.1, .2, .6, -.5, .1],
        y1=[.2, -.1, .4, .9, .1, -.3], y2=[.1, .3, -.2, .5, .6, .2],
        y3=[.4, -.2, .1, .3, -.1, .2], y4=[-.3, .2, .5, -.1, .4, .1],
        y5=[.6, -.4, .2, .1, -.2, .3], y6=[.1, .5, -.3, .2, .0, -.1])
    before = deepcopy(data)
    brmi = @brm data begin
        mu ~ 1 + factor(g) + x
        nu ~ 1 + factor(g) + w
        eta ~ 1 + factor(g; ref=3)
        zeta ~ 0 + g + x + w
        pa ~ 1 + x + w
        pb ~ 1 + x + w
        s ~ Exponential(1)
        y1 ~ Normal(mu, s)
        y2 ~ Normal(nu, s)
        y3 ~ Normal(eta, s)
        y4 ~ Normal(zeta, s)
        y5 ~ Normal(pa, s)
        y6 ~ Normal(pb, s)
    end
    main = string(BRM._rk_emit_ast(BRM._brm_rk_plan(brmi)).main)
    # No indicator columns: every effect reads its level's coefficient.
    @test !occursin("brm_factor_dummy", main)
    @test !occursin(r"\* (mu|nu|eta|zeta)_g", main)
    # Treatment coding (`mu`, `nu`) and cell means (`zeta`) over the same level
    # order share one index; `ref=3` lists its reference first.
    @test count("brm_prepared_indices(g, [1, 2, 3])", main) == 1
    @test count("brm_prepared_indices(g, [3, 2, 1])", main) == 1
    @test occursin("(vcat(0.0, mu_g))[g_level]", main)
    @test occursin("(vcat(0.0, nu_g))[g_level]", main)
    @test occursin("(vcat(0.0, eta_g))[g__ref_3_level]", main)
    @test occursin("zeta_g[g_level]", main)
    # `pa` and `pb` share one design matrix.
    @test count("hcat(ones(length(x)), x, w)", main) == 1
    @test occursin("pop_pb ~ brm_population_effects(X_pa, 3", main)

    backend, problem = consumer_problem(brmi)
    names = coordinate_names(backend.model.layout)
    at(u, name) = u[findfirst(==(Symbol(name)), names)]
    coefs(u, block, k) = [at(u, "$block.$j") for j in 1:k]
    oracle(u) = begin
        s = exp(at(u, "s"))
        mu = [ones(6) data.x] * coefs(u, "pop_mu.beta_pop", 2) .+
            [0.0; coefs(u, "mu_g", 2)][data.g]
        nu = [ones(6) data.w] * coefs(u, "pop_nu.beta_pop", 2) .+
            [0.0; coefs(u, "nu_g", 2)][data.g]
        eg = coefs(u, "eta_g", 2)
        eta = at(u, "pop_eta.beta_pop.1") .+ [eg[2], eg[1], 0.0][data.g]
        bz = coefs(u, "pop_zeta.beta_pop", 2)
        zeta = coefs(u, "zeta_g", 3)[data.g] .+ bz[1] .* data.x .+ bz[2] .* data.w
        design = [ones(6) data.x data.w]
        pa = design * coefs(u, "pop_pa.beta_pop", 3)
        pb = design * coefs(u, "pop_pb.beta_pop", 3)
        likelihood = sum(sum(logpdf.(Normal.(m, s), y)) for (m, y) in
            ((mu, data.y1), (nu, data.y2), (eta, data.y3), (zeta, data.y4),
             (pa, data.y5), (pb, data.y6)))
        coefficients = filter(n -> n !== :s, names)
        likelihood + sum(logpdf(Normal(), at(u, n)) for n in coefficients) +
            logpdf(Exponential(1), s) + at(u, "s")
    end
    @test length(names) == 1 + 2 + 2 + 2 + 2 + 2 + 1 + 3 + 2 + 3 + 3
    for u in (zeros(length(names)), collect(range(-.4, .5; length=length(names))),
            [.3 * sin(3j) for j in eachindex(names)])
        check_consumer_point(problem, u, oracle)
    end
    @test isequal(data, before)
end
