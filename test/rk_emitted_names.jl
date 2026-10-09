# Names in the emitted RK main block say what each value is: a design matrix
# shared by several predictors is named for its columns, a grouping's level
# positions are computed once under that grouping's name, repeated monotonic
# effects are named for their predictor as SBBRMI names its carriers, and the
# linear predictor of `log(v) ~ …` is `log_v`.
#
#     julia --project=test test/rk_emitted_names.jl
include(joinpath(@__DIR__, "rk_consumer_support.jl"))

const NAMES_DATA = (;
    subject=repeat(["s1", "s2", "s3", "s4", "s5"]; inner=3),
    x1=[sin(1.3j) for j in 1:15], x2=[cos(0.7j) for j in 1:15],
    rank=[1, 2, 3, 2, 1, 3, 3, 2, 1, 1, 2, 3, 2, 3, 1],
    g1=repeat(["a", "b", "c"], 5), g2=repeat(["c", "a", "b"]; inner=5),
    y=[0.4 * sin(2.1j) + 1.5 for j in 1:15],
    y2=[0.3 * cos(1.7j) - 0.5 for j in 1:15])

emitted_main(brmi) = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi)).main
main_statements(brmi) = filter(s -> !(s isa LineNumberNode), emitted_main(brmi).args)

@stestset "value route: designs, level positions, monotonic effects, linked predictors" begin
    before = deepcopy(NAMES_DATA)
    brmi = @brm NAMES_DATA begin
        log(a) ~ 1 + x1 + x2 + (1 | p | subject)
        log(b) ~ 1 + x1 + x2 + (1 | p | subject)
        c ~ 0 + x1 + mo(rank) + (1 | subject)
        d ~ 0 + x2 + mo(rank)
        e ~ 0 + x2 + mo(rank) + (1 | mm(g1, g2))
        mu = a * b + c + d + e
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    stmts = main_statements(brmi)
    main = string(emitted_main(brmi))
    # One design per column set. `a` and `b` share theirs, so it belongs to
    # neither and is named for its columns; `d` and `e` share `hcat(x2)`.
    # `c` alone reads `hcat(x1)` and keeps its own name.
    @test :(X_Intercept_x1_x2 = hcat(ones(length(subject)), x1, x2)) in stmts
    @test occursin("pop_log_a ~ brm_population_effects(X_Intercept_x1_x2, 3", main)
    @test occursin("pop_log_b ~ brm_population_effects(X_Intercept_x1_x2, 3", main)
    @test :(X_x2 = hcat(x2)) in stmts
    @test occursin("pop_d ~ brm_population_effects(X_x2, 1", main)
    @test occursin("pop_e ~ brm_population_effects(X_x2, 1", main)
    @test :(X_c = hcat(x1)) in stmts
    @test !occursin(r"X_log_a\b|X_log_b\b|X_d\b|X_e\b", main)
    # The `p` and id-less buckets over `subject` read one level position.
    @test count("brm_level_indices(subject, subject)", main) == 1
    @test :(subject_level = brm_level_indices(subject, subject)) in stmts
    @test occursin("brm_ranef_column(b_p_subject, subject_level, 1)", main)
    @test occursin("brm_ranef_column(b_subject, subject_level, 1)", main)
    @test !occursin("_index_", main)
    # A multi-membership column's positions are into its bucket's joint levels.
    groups = :b_mm__g1__g2_groups
    @test :(b_mm__g1__g2_groups_g1_level = brm_level_indices(g1, $groups)) in stmts
    @test :(b_mm__g1__g2_groups_g2_level = brm_level_indices(g2, $groups)) in stmts
    # The first `mo(rank)` is `mo_rank`; its repeats are named for their
    # predictor, never numbered.
    @test occursin("mo_rank ~ brm_monotonic_effect(rank_idx", main)
    @test occursin("mo_d_rank ~ brm_monotonic_effect(rank_idx", main)
    @test occursin("mo_e_rank ~ brm_monotonic_effect(rank_idx", main)
    @test !occursin(r"mo_rank_\d", main)
    # `log(a) ~ …` sums into `log_a`, and `a` is its value.
    @test :(log_a = pop_log_a .+ ranef_a_p_subject) in stmts
    @test :(a = exp.(log_a)) in stmts
    @test :(b = exp.(log_b)) in stmts
    @test !occursin(r"\b[ab]_\b", main)

    # Coordinates follow the component names.
    backend, problem = consumer_problem(brmi)
    names = coordinate_names(backend.model.layout)
    for name in ("mo_rank.beta", "mo_d_rank.beta", "mo_e_rank.beta",
            "mo_rank.simplex_incr.1", "mo_d_rank.simplex_incr.1", "mo_e_rank.simplex_incr.1")
        @test Symbol(name) in names
    end
    for u in (zeros(length(names)), [0.3 * sin(3j) for j in eachindex(names)])
        value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
        @test isfinite(value)
        @test all(isfinite, gradient)
    end
    @test isequal(NAMES_DATA, before)
end

@stestset "monotonic carriers match SBBRMI's" begin
    brmi = @brm NAMES_DATA begin
        mu ~ 1 + x1 + mo(rank) + mo1(rank)
        log(sigma) ~ 1 + x2 + mo(rank)
        y ~ Normal(mu, sigma)
    end
    main = string(emitted_main(brmi))
    # A linked predictor's carrier carries its link, as `pop_log_sigma` does.
    @test occursin("mo_rank ~ brm_monotonic_effect(rank_idx", main)
    @test occursin("mo1_rank ~ brm_monotonic_value(rank_idx", main)
    @test occursin("mo_log_sigma_rank ~ brm_monotonic_effect(rank_idx", main)
    code = BRM.stan_code(SBBRMI(brmi; mod=@__MODULE__, total_groups=()))
    for carrier in ("mo_rank", "mo1_rank", "mo_log_sigma_rank")
        @test occursin("$(carrier)_simplex_incr ~ dirichlet", code)
    end
end

@stestset "structural route: shared designs and one level position" begin
    brmi = @brm NAMES_DATA begin
        mu ~ 1 + x1 + x2 + (1 | p | subject)
        nu ~ 1 + x1 + x2 + (1 | p | subject)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
        y2 ~ Normal(nu, sigma)
    end
    stmts = main_statements(brmi)
    main = string(emitted_main(brmi))
    @test :(X_Intercept_x1_x2 = hcat(ones(length(subject)), x1, x2)) in stmts
    @test count("hcat(", main) == 1
    @test occursin("pop_nu ~ brm_population_effects(X_Intercept_x1_x2, 3", main)
    @test :(subject_level = brm_level_indices(subject, subject)) in stmts
end
