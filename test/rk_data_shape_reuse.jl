# An RK build depends on the @brm body, not on the bound data's row counts:
# the emitted program carries no data-derived row geometry, so a graph built
# from one data set evaluates another data set of the same body exactly as
# that data set's own build does. Public synthetic models only.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))

module PublicShapeReuse
using BayesianRegressionModels, Distributions
# A monotonic predictor on a secondary row axis, read per subject through
# `ragged(...)`, and a ragged response observed through the kernel cell.
secondary_axis(data) = @brm data begin
    theta ~ 1 + (1 | p | subject)
    effect(theta, Intercept) ~ Normal(0, 0.7)
    sd(:, p) ~ Exponential(0.9)
    score ~ 1 + mo(rank)
    effect(score, Intercept) ~ Normal(0, 0.4)
    simplex(score, mo(rank)) ~ Dirichlet(1, 2)
    pred ~ kernel(t, theta, ragged(score, row_group)) do ts, a, scores
        a .+ ts .* sum(scores)
    end
    ragged(y, event_subject) ~ Normal(pred, 0.8)
end
end

emitted_source(brmi) = begin
    emitted = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    (join(sprint.(Base.show_unquoted, emitted.defs), '\n'),
        sprint(Base.show_unquoted, emitted.main), first.(emitted.bindings))
end

# `trained` builds the graph; `scored` supplies the new data. The retained
# graph and the scored data's own build must agree bit for bit.
function check_shape_reuse(build, trained, scored)
    saved = deepcopy((trained, scored))
    a, b = build(trained), build(scored)
    @test emitted_source(a) == emitted_source(b)
    retained = RKBRMI(a)
    own = RKBRMI(b)
    names = coordinate_names(own.model.layout)
    @test coordinate_names(retained.model.layout) == names
    bound = BRM.rk_translate_artifact(BRM.emit_rk_artifact(b; case_id="shape-reuse"))
    n = length(names)
    backend = AutoEnzyme(; mode=Enzyme.Reverse)
    reused = prepare_sampler(retained.model, bound, zeros(n); backend)
    fresh = prepare_sampler(own.model, bound, zeros(n); backend)
    for u in (zeros(n), fill(0.13, n), collect(range(-0.2, 0.3; length=n)))
        gr, gf = similar(u), similar(u)
        vr, _ = sampler_value_and_gradient!(reused, gr, u)
        vf, _ = sampler_value_and_gradient!(fresh, gf, u)
        @test vr == vf
        # Separately compiled reverse passes may associate adjoint sums
        # differently (measured at most 4.5e-16 here); values stay bit-equal.
        @test gr ≈ gf atol=1e-13 rtol=1e-13
        @test isfinite(vf) && all(isfinite, gf)
    end
    @test isequal((trained, scored), saved)
    own, names
end

secondary_data(dose_rows) = begin
    subjects = ["b", "empty", "a"]
    (; subject=subjects, t=[[0.2, 0.5], Float64[], [0.7]],
        row_group=[subjects[mod1(i, 3)] for i in 1:dose_rows],
        rank=[mod1(2i, 3) for i in 1:dose_rows],
        y=[0.1, -0.2, 0.4], event_subject=["b", "b", "a"])
end

@stestset "secondary-axis row counts do not enter the emitted program" begin
    trained, scored = secondary_data(5), secondary_data(8)
    own, names = check_shape_reuse(PublicShapeReuse.secondary_axis, trained, scored)
    # The scored build itself matches an independent density.
    index(n) = only(findall(==(Symbol(n)), names))
    a, b = index("pop_score.beta_pop.1"), index("mo_rank.beta")
    intercept, sd = index("pop_theta.beta_pop.1"), index("b_p_subject.tau.1")
    levels = CategoricalArrays.levels(scored.subject)
    z = [index("b_p_subject.z.$j.1") for j in eachindex(levels)]
    simplex = index("mo_rank.simplex_incr.1")
    @test length(names) == 8
    oracle(u) = begin
        q = 1/(1+exp(-u[simplex]))
        increments = [q, 1-q]
        contrast = [sum(increments[1:(rank-1)]) for rank in scored.rank]
        tau = exp(u[sd])
        score = u[a] .+ u[b] .* contrast
        rows = [only(findall(==(s), levels)) for s in scored.subject]
        theta = u[intercept] .+ tau .* u[z[rows]]
        likelihood = sum(enumerate(scored.subject)) do (j, s)
            loc = theta[j] .+ scored.t[j] .* sum(score[findall(==(s), scored.row_group)])
            sum(logpdf.(Normal.(loc, 0.8), scored.y[findall(==(s), scored.event_subject)]); init=0.0)
        end
        prior = logpdf(Normal(0, 0.4), u[a]) + logpdf(Normal(), u[b]) +
            logpdf(Normal(0, 0.7), u[intercept]) + logpdf(Exponential(0.9), tau) +
            sum(logpdf.(Normal(), u[z])) + logpdf(Dirichlet([1., 2.]), increments)
        prior + u[sd] + log(q) + log1p(-q) + likelihood
    end
    problem = rk_logdensity_problem(own; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    for u in (zeros(8), fill(0.13, 8), collect(range(-0.2, 0.3; length=8)))
        check_consumer_point(problem, u, oracle)
    end
end
