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
# Event rows joined to the kernel's subjects, with a per-row argument: the
# join's row partition orders both the response and the argument.
joined(data) = @brm data begin
    alpha ~ 1 + (1 | p | subject)
    effect(alpha, Intercept) ~ Normal(0, 0.7)
    sd(:, p) ~ Exponential(0.9)
    sigma ~ Exponential(0.9)
    loc ~ kernel(x, alpha) do xs, a
        xs * a
    end
    ragged(y, event_subject) ~ Normal(loc - reference, sigma)
end
# Per-subject response cells with a per-subject singleton argument, which
# each subject's cell length broadcasts.
nested(data) = @brm data begin
    alpha ~ 1 + (1 | p | subject)
    effect(alpha, Intercept) ~ Normal(0, 0.7)
    sd(:, p) ~ Exponential(0.9)
    sigma ~ Exponential(0.9)
    loc ~ kernel(x, alpha) do xs, a
        xs * a
    end
    y ~ Normal(loc - reference, sigma)
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

secondary_data(dose_rows, event_subject) = begin
    subjects = ["b", "empty", "a"]
    t = [[0.2 + 0.3k for k in 1:count(==(s), event_subject)] for s in subjects]
    (; subject=subjects, t,
        row_group=[subjects[mod1(i, 3)] for i in 1:dose_rows],
        rank=[mod1(2i, 3) for i in 1:dose_rows],
        y=[0.1 * mod1(3i, 5) - 0.2 for i in eachindex(event_subject)], event_subject)
end

@stestset "secondary-axis row counts do not enter the emitted program" begin
    trained = secondary_data(5, ["b", "b", "a"])
    scored = secondary_data(8, ["a", "b", "a", "b", "a"])
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

# Coordinates and independent density shared by the joined and nested models.
function grouped_kernel_oracle(names, data, rows_of)
    ia = only(findall(==(Symbol("pop_alpha.beta_pop.1")), names))
    it = only(findall(n -> occursin(".tau.", string(n)), names))
    iz = findall(n -> occursin(".z.", string(n)), names)
    is = only(findall(==(:sigma), names))
    @test length(names) == 6 && length(iz) == 3
    levels = sort(unique(data.subject))
    order = [findfirst(==(s), levels) for s in data.subject]
    u -> begin
        tau, sigma = exp(u[it]), exp(u[is])
        alpha = u[ia] .+ tau .* u[iz][order]
        likelihood = sum(eachindex(data.subject)) do j
            y, reference = rows_of(j)
            sum(logpdf.(Normal.(data.x[j] .* alpha[j] .- reference, sigma), y); init=0.0)
        end
        logpdf(Normal(0, 0.7), u[ia]) + sum(logpdf.(Normal(), u[iz])) +
            logpdf(Exponential(0.9), tau) + u[it] +
            logpdf(Exponential(0.9), sigma) + u[is] + likelihood
    end
end

function check_scored_density(own, names, oracle)
    problem = rk_logdensity_problem(own; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    n = length(names)
    for u in (zeros(n), fill(0.13, n), collect(range(-0.2, 0.3; length=n)))
        check_consumer_point(problem, u, oracle)
    end
end

joined_data(event_subject) = begin
    subjects = ["b", "empty", "a"]
    x = [[0.3k - 0.2 for k in 1:count(==(s), event_subject)] for s in subjects]
    (; subject=subjects, x, event_subject,
        y=[0.1 * mod1(2i, 7) - 0.3 for i in eachindex(event_subject)],
        reference=[0.01 * mod1(3i, 5) for i in eachindex(event_subject)])
end

@stestset "a ragged join's row partition is bound data, not source" begin
    trained = joined_data(["a", "b", "b", "a", "b", "a", "a"])
    # The same row count in another subject order: a stale partition would
    # pair rows with the wrong subjects and reference values.
    for scored in (joined_data(["b", "a", "a", "b", "a", "b", "a"]),
            joined_data(["a", "b", "a", "a", "a"]))
        own, names = check_shape_reuse(PublicShapeReuse.joined, trained, scored)
        oracle = grouped_kernel_oracle(names, scored, j -> begin
            rows = findall(==(scored.subject[j]), scored.event_subject)
            scored.y[rows], scored.reference[rows]
        end)
        check_scored_density(own, names, oracle)
    end
end

nested_data(lengths) = begin
    x = [[0.3k - 0.2 for k in 1:n] for n in lengths]
    (; subject=["b", "c", "a"], x,
        y=[[0.1 * mod1(j + k, 5) - 0.2 for k in 1:n] for (j, n) in enumerate(lengths)],
        reference=[[0.01j] for j in eachindex(lengths)])
end

@stestset "per-subject response lengths are bound data, not source" begin
    # Equal totals with different per-subject lengths: stale lengths would
    # broadcast each subject's argument over another subject's rows.
    trained, scored = nested_data([3, 0, 4]), nested_data([2, 1, 4])
    own, names = check_shape_reuse(PublicShapeReuse.nested, trained, scored)
    oracle = grouped_kernel_oracle(names, scored,
        j -> (scored.y[j], only(scored.reference[j])))
    check_scored_density(own, names, oracle)
end
