# test/rk_design_binder.jl — the graph inputs of new designs, built once per
# fit (snag `a-blessed-rk-sho-02370c78`; design decision `1x2h909`).
#
# `rk_design_binder(rk; resample_groups)` replays the first design of each kind
# in full and binds every later one into that replay's lowered program. Oracle:
# its inputs are those of `rk_bound_data(reprocess(rk, design; …))`, its graph
# values equal theirs, a later design emits no program, and a design
# `reprocess` refuses is refused with the same error. Public synthetic models
# only.
#
# Run: julia --startup-file=no --project=test test/rk_design_binder.jl

include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Random

module BinderModels
using BayesianRegressionModels, Distributions
function curve(t, dose_time, dose_amount, log_k, log_V, log_F, baseline)
    k, V = exp(log_k), exp(log_V)
    shift = isempty(baseline) ? 0.0 : sum(baseline) / length(baseline)
    map(t) do ti
        c = 0.0
        for (j, td) in enumerate(dose_time)
            ti > td && (c += exp(log_F[j]) * dose_amount[j] / V * exp(-k * (ti - td)))
        end
        c + shift
    end
end
# Subject predictors with a correlated block, dose-row and baseline-row
# predictors joined to their subject, and a per-subject cell.
pk(d) = @brm d begin
    sigma ~ Exponential(1)
    log_k ~ 1 + zscale(weight) + (1 | p | subject)
    log_V ~ 1 + diseased + (1 | p | subject)
    log_F ~ 0 + vessel + mo(diet)
    baseline ~ 0 + hsgp(base_time; k=5, domain=(-1.5, 1.5))
    effect(log_k, Intercept) ~ Normal(0, 1)
    effect(log_V, Intercept) ~ Normal(0, 1)
    effect(log_F, :) ~ Normal(0, 0.5)
    sd(:, p) ~ Exponential(1)
    doses = groups(dose_subject => subject)
    bases = groups(base_subject => subject)
    @plate for i in eachindex(log_k)
        mu = curve(t[i], dose_time[i], dose_amount[i], log_k[i], log_V[i],
            log_F[doses[i]], baseline[bases[i]])
        y[i] ~ censored(Normal(mu, sigma); lower=lloq[i])
        conc[i] = mu
    end
end
# Dose rows read in dose order (`order=dose_rank`), whatever order a design
# lists them in.
function ordered_curve(t, dose_time, log_k, log_F)
    k = exp(log_k)
    map(t) do ti
        c = 0.0
        for (j, td) in enumerate(dose_time)
            ti > td && (c += exp(log_F[j] - k * (ti - td)))
        end
        c
    end
end
pk_ordered(d) = @brm d begin
    sigma ~ Exponential(1)
    log_k ~ 1 + (1 | subject)
    log_F ~ 0 + vessel
    effect(log_k, Intercept) ~ Normal(0, 1)
    effect(log_F, :) ~ Normal(0, 0.5)
    doses = groups(dose_subject => subject; order=dose_rank)
    @plate for i in eachindex(log_k)
        mu = ordered_curve(t[i], dose_time[i], log_k[i], log_F[doses[i]])
        y[i] ~ Normal(mu, sigma)
        conc[i] = mu
    end
end
# A flat model: a string factor, an automatic HSGP interval and a zscale.
flat(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + g + hsgp(x; k=5) + zscale(z)
    y ~ Normal(mu, sigma)
end
fitted_constant(d) = @brm d begin
    sigma ~ Exponential(1)
    xbar = constant(mean(x))
    b ~ Normal(0, 1)
    mu = b * (x - xbar)
    y ~ Normal(mu, sigma)
end
# A weighted likelihood and an ordered outcome: their weights and outcomes
# bind as the design holds them.
weighted_normal(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + zscale(z)
    y ~ weighted(Normal(mu, sigma), fweights(n))
end
ordered(d) = @brm d begin
    eta ~ 0 + zscale(z)
    k ~ OrderedLogistic(eta)
end
ordered_authored(d) = @brm d begin
    slope ~ 0 + zscale(z)
    bias ~ Normal(0, 1)
    loc = slope + bias
    k ~ OrderedLogistic(loc)
end
# HSGPs over masked rows: the mask and the masked axis are graph values of the
# bound `keep` and `x` columns.
masked(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + hsgp(x; k=5, rows=keep)
    y ~ Normal(mu, sigma)
end
masked_periodic(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + hsgp(x; k=2, cov=:periodic, period=4.0, rows=keep)
    y ~ Normal(mu, sigma)
end
# An areal field: its graph is fitted from the training areas and edges.
areal(d) = @brm d begin
    mu ~ 1 + car(area; edges=(from, to), type=:bym2)
    effect(mu, Intercept) ~ Normal(0, 1)
    y ~ Normal(mu, 1.0)
end
# A cell whose curve a callable-source provider claims as an RK graph,
# counting its emissions: one per planned program.
import BayesianRegressionModels: _rk_callable_source!
using ReactiveKernels
dosed_curve(t, dose, log_k) = [dose * exp(-exp(log_k) * ti) for ti in t]
const EMISSIONS = Ref(0)
function _rk_callable_source!(definitions, bindings, entry, ::typeof(dosed_curve))
    EMISSIONS[] += 1
    push!(definitions, :(ReactiveKernels.@kernel $entry(t, dose, log_k) = begin
        k = exp(log_k)
        reads = ReactiveKernels.plate(eachindex(t)) do j
            dose * exp(-k * t[j])
        end
        return reads
    end))
    :done
end
provided(d) = @brm d begin
    sigma ~ Exponential(1)
    log_k ~ 1 + zscale(weight) + (1 | p | subject)
    effect(log_k, Intercept) ~ Normal(0, 1)
    sd(:, p) ~ Exponential(1)
    @plate for i in eachindex(log_k)
        conc[i] = dosed_curve(t[i], dose[i], log_k[i])
        y[i] ~ normal(conc[i], sigma)
    end
end
# A submodel hook scaling its column by two keyword constants, written into
# `data` (which the binder cannot tell from design data) or emitted as a
# statement of the program.
import BayesianRegressionModels: _rk_submodel_rhs!
function data_scaled end
function statement_scaled end
function _rk_scaled!(statements, data, target, rhs; as_data)
    x = name(only(getargs(rhs)))
    key = Symbol(target, :_scales)
    scales = Float64[getkwargs(rhs).lo, getkwargs(rhs).hi]
    as_data ? (data[key] = scales) :
        push!(statements, :($key = $(Expr(:vect, scales...))))
    push!(statements, :($target = $x .* $key[1] .+ $key[2]))
    :done
end
_rk_submodel_rhs!(definitions, statements, data, bindings, target::Symbol,
    ::typeof(data_scaled), rhs) = _rk_scaled!(statements, data, target, rhs; as_data=true)
_rk_submodel_rhs!(definitions, statements, data, bindings, target::Symbol,
    ::typeof(statement_scaled), rhs) = _rk_scaled!(statements, data, target, rhs; as_data=false)
hooked_data(d) = @brm d begin
    b ~ Normal(0, 1)
    loc ~ data_scaled(x; lo=0.5, hi=2.0)
    mu = loc + b
    y ~ Normal(mu, 0.8)
end
hooked_statement(d) = @brm d begin
    b ~ Normal(0, 1)
    loc ~ statement_scaled(x; lo=0.5, hi=2.0)
    mu = loc + b
    y ~ Normal(mu, 0.8)
end
# One model per value the program fits from the training data alone.
thin_plate(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + s(x)
    y ~ Normal(mu, sigma)
end
tensor(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + t2(x, z; k=(4, 4))
    y ~ Normal(mu, sigma)
end
interval(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + hsgp(x; k=5)
    y ~ Normal(mu, sigma)
end
zscaled(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + zscale(x)
    y ~ Normal(mu, sigma)
end
shrunk(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + g + (1 | h)
    effect(mu, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5, alpha=1)
    y ~ Normal(mu, sigma)
end
grouped(d) = @brm d begin
    sigma ~ Exponential(1)
    mu ~ 1 + (1 | h)
    y ~ Normal(mu, sigma)
end
end
const BM = BinderModels

# A PK design: `n` subjects with `reads` read times each, three doses and two
# baseline rows per subject.
function pk_data(rng, n, reads; labels=["s$(j)" for j in 1:n], responses=true)
    doses, bases = 3, 2
    data = (; subject=labels, weight=50 .+ 40 .* rand(rng, n),
        diseased=Float64.(rand(rng, 0:1, n)),
        t=[sort!(48 .* rand(rng, reads)) for _ in 1:n],
        lloq=[fill(0.01, reads) for _ in 1:n],
        dose_time=[[0.0, 12.0, 24.0] for _ in 1:n],
        dose_amount=[fill(100.0, doses) for _ in 1:n],
        dose_subject=repeat(labels; inner=doses),
        vessel=Float64.(rand(rng, 0:1, n * doses)),
        diet=categorical(rand(rng, 1:3, n * doses); levels=1:3, ordered=true),
        base_subject=repeat(labels; inner=bases),
        base_time=repeat([-1.0, 1.0], n))
    responses || return data
    merge(data, (; y=[0.02 .+ abs.(randn(rng, reads)) for _ in 1:n]))
end
# Three dose rows per subject, listed out of dose order.
function ordered_data(rng, n, reads; labels=["s$(j)" for j in 1:n], responses=true)
    data = (; subject=labels, t=[sort!(48 .* rand(rng, reads)) for _ in 1:n],
        dose_time=[[0.0, 12.0, 24.0] for _ in 1:n],
        dose_subject=repeat(labels; inner=3),
        dose_rank=reduce(vcat, [shuffle(rng, 1:3) for _ in 1:n]),
        vessel=Float64.(rand(rng, 0:1, 3n)))
    responses || return data
    merge(data, (; y=[0.02 .+ abs.(randn(rng, reads)) for _ in 1:n]))
end
flat_data(rng, n) = (; x=collect(range(0.3, 4.1; length=n)) .+ 0.01 .* rand(rng, n),
    g=["a", "b", "c"][mod1.(1:n, 3)], z=cos.(1:n) .+ rand(rng, n),
    keep=[isodd(i) || i > 8 for i in 1:n], y=randn(rng, n))
without(data, name) = NamedTuple{filter(!=(name), keys(data))}(data)
# Frequency weights and an ordered outcome with three levels.
leveled_data(rng, rows) = merge(flat_data(rng, rows),
    (; n=rand(rng, 1:3, rows), k=[1, 2, 3][mod1.(1:rows, 3)]))

absent(v) = occursin("Absent", string(nameof(typeof(v))))
same(a, b) = typeof(a) == typeof(b) && (absent(a) ?
    all(f -> isequal(getfield(a, f), getfield(b, f)), fieldnames(typeof(a))) : isequal(a, b))
same_inputs(x, y) = Set(keys(x)) == Set(keys(y)) && all(n -> same(x[n], y[n]), keys(y))

point(rk) = collect(range(-0.6, 0.8; length=rk.model.layout.total))
function graph_value(rk, inputs, want, runtime...)
    names = first.(runtime)
    query = ReactiveKernels.prepare(rk.model.spec;
        have=(:unconstrained, names..., keys(inputs)...), want=(want,), bound=inputs)
    value = Base.invokelatest(query, point(rk), last.(runtime)...)
    value isa Tuple ? only(value) : value
end

# Every design binds to what the planned replay binds, and its graph value is
# the planned replay's.
function check_binder(rk, designs; resample_groups=(), want, runtime=d -> ())
    binder = rk_design_binder(rk; resample_groups)
    for design in designs
        inputs = binder(design)
        planned = rk_bound_data(reprocess(rk, design; resample_groups))
        @test same_inputs(inputs, planned)
        @test graph_value(rk, inputs, want, runtime(design)...) ==
            graph_value(rk, planned, want, runtime(design)...)
    end
    binder
end

subject_effects(design) = (:b_p_subject => 0.1 .* reshape(
    collect(1.0:(2 * length(design.subject))), length(design.subject), 2),)

@stestset "binder inputs are the planned replay's" begin
    rng = Xoshiro(7)
    rk = RKBRMI(BM.pk(pk_data(rng, 6, 4)))
    # New subjects, with and without responses; one design per kind is planned.
    new = [pk_data(rng, n, r; labels=["n$(j)" for j in 1:n], responses=false)
        for (n, r) in ((2, 5), (3, 2), (1, 40))]
    binder = check_binder(rk, new; resample_groups=[:subject], want=:conc,
        runtime=subject_effects)
    @test length(binder.classes) == 1
    @test keys(binder.fitted) == (:weight_fit,)
    check_binder(rk, [pk_data(rng, 2, 3; labels=["n1", "n2"]),
        pk_data(rng, 3, 2; labels=["n1", "n2", "n3"])];
        resample_groups=[:subject], want=:conc, runtime=subject_effects)
    # The fitted subjects, their fitted effects (frozen grouping).
    check_binder(rk, [without(pk_data(rng, 2, 3; labels=["s2", "s5"]), :y),
        without(pk_data(rng, 3, 6; labels=["s1", "s3", "s6"]), :y)]; want=:conc)
    flat_rk = RKBRMI(BM.flat(flat_data(rng, 12)))
    designs = [map(v -> v[rows], without(flat_data(rng, 12), :y)) for rows in (2:5, [7], 1:9)]
    check_binder(flat_rk, designs; want=:mu)
    constant_rk = RKBRMI(BM.fitted_constant(flat_data(rng, 12)))
    binder = check_binder(constant_rk, designs; want=:mu)
    @test :xbar in keys(binder.fitted) || !(:xbar in keys(rk_bound_data(constant_rk)))
end

@stestset "fitted inputs bind once" begin
    rng = Xoshiro(11)
    rk = RKBRMI(BM.pk(pk_data(rng, 5, 4)))
    binder = rk_design_binder(rk; resample_groups=[:subject])
    design = pk_data(rng, 2, 6; labels=["n1", "n2"], responses=false)
    inputs = binder(design)
    rest = NamedTuple{Tuple(n for n in keys(inputs) if !haskey(binder.fitted, n))}(inputs)
    effects = subject_effects(design)
    once = ReactiveKernels.prepare(rk.model.spec;
        have=(:unconstrained, first.(effects)..., keys(rest)..., keys(binder.fitted)...),
        want=(:conc,), bound=binder.fitted)
    value = Base.invokelatest(once, point(rk), last.(effects)..., values(rest)...)
    @test (value isa Tuple ? only(value) : value) == graph_value(rk, inputs, :conc, effects...)
end

@stestset "a later design emits no program" begin
    rng = Xoshiro(3)
    train = (; subject=["s1", "s2", "s3", "s4"], weight=[60.0, 72.0, 81.0, 55.0],
        dose=[100.0, 120.0, 80.0, 100.0],
        t=[[0.5, 1.0], [0.25, 1.5, 2.0], [1.0], [0.5, 2.0]],
        y=[[1.0, 0.8], [1.2, 0.7, 0.6], [0.9], [0.6, 0.5]])
    rk = RKBRMI(BM.provided(train))
    binder = rk_design_binder(rk; resample_groups=[:subject])
    design(n) = (; subject=["n$(j)" for j in 1:n], weight=fill(70.0, n), dose=fill(100.0, n),
        t=[collect(range(0.1, 3.0; length=j + 2)) for j in 1:n])
    binder(design(2))
    before = BM.EMISSIONS[]
    for n in (1, 3, 5)
        @test same_inputs(binder(design(n)), rk_bound_data(reprocess(rk, design(n);
            resample_groups=[:subject])))
    end
    # Each planned replay emits once; the binder's three designs emitted nothing.
    @test BM.EMISSIONS[] == before + 3
end

@stestset "a design reprocess refuses is refused alike" begin
    rng = Xoshiro(5)
    rk = RKBRMI(BM.pk(pk_data(rng, 6, 4)))
    binder = rk_design_binder(rk; resample_groups=[:subject])
    base = pk_data(rng, 2, 3; labels=["n1", "n2"], responses=false)
    binder(base)
    function refused_alike(design, message)
        @test_throws message reprocess(rk, design; resample_groups=[:subject])
        @test_throws message binder(design)
    end
    # A baseline time past the fixed HSGP domain.
    refused_alike(merge(base, (; base_time=[2.0; base.base_time[2:end]])),
        "lies outside its fixed domain")
    # A dose row naming a subject the design does not have.
    refused_alike(merge(base, (; dose_subject=[base.dose_subject[1:end-1]; "zz"])),
        "name no subject")
    # A dose column on another axis than its subject labels.
    refused_alike(merge(base, (; vessel=base.vessel[1:end-1])), "but `dose_subject` has 6")
    # Per-subject inputs that disagree on the subject count.
    refused_alike(merge(base, (; t=base.t[1:1])), "disagree on subject count")
    # A subject listed twice.
    refused_alike(merge(base, (; subject=["n1", "n1"])), "unique subject per row")
    # An unseen `mo` level.
    refused_alike(merge(base, (; diet=categorical([1, 2, 4, 2, 1, 2]; levels=1:4, ordered=true))),
        "not a training level")
    flat_rk = RKBRMI(BM.flat(flat_data(rng, 12)))
    flat_binder = rk_design_binder(flat_rk)
    design = map(v -> v[2:5], without(flat_data(rng, 12), :y))
    flat_binder(design)
    @test_throws "not a training level" flat_binder(merge(design, (; g=["a", "zzz", "b", "c"])))
    @test_throws "outside the fitted basis interval" flat_binder(
        merge(design, (; x=[100.0, 1.0, 2.0, 3.0])))
    # A frozen grouping's unseen label.
    frozen = rk_design_binder(rk)
    fitted = without(pk_data(rng, 2, 3; labels=["s1", "s2"]), :y)
    frozen(fitted)
    @test_throws "not one of the" frozen(merge(fitted,
        (; subject=["s1", "zz"], dose_subject=repeat(["s1", "zz"]; inner=3),
           base_subject=repeat(["s1", "zz"]; inner=2))))
end

@stestset "an ordered join is checked per design as reprocess checks it" begin
    rng = Xoshiro(17)
    rk = RKBRMI(BM.pk_ordered(ordered_data(rng, 5, 4)))
    designs = [without(ordered_data(rng, n, r; labels=labels), :y)
        for (n, r, labels) in ((2, 3, ["s2", "s5"]), (3, 6, ["s1", "s3", "s4"]), (1, 2, ["s3"]))]
    binder = check_binder(rk, designs; want=:conc)
    # The partition is a graph value of the bound order column, so one
    # lowering serves every design.
    @test length(binder.classes) == 1 && !occursin("replayed in full", sprint(show, binder))
    @test :dose_rank in keys(binder(designs[1]))
    base = designs[1]
    @test_throws "must be a column on" reprocess(rk, merge(base, (; dose_rank=base.dose_rank[1:end-1])))
    @test_throws "must be a column on" binder(merge(base, (; dose_rank=base.dose_rank[1:end-1])))
    @test_throws "name no subject" binder(merge(base, (; dose_subject=[base.dose_subject[1:end-1]; "zz"])))
end

@stestset "weights and ordered outcomes bind as the design holds them" begin
    # Both used to be copied in Julia (`Float64` weights, `Int` outcomes), so
    # every design replayed in full; the ordered level count is the fit's.
    rng = Xoshiro(19)
    data = leveled_data(rng, 12)
    for (build, response, want) in ((BM.weighted_normal, :y, :mu), (BM.ordered, :k, :eta),
            (BM.ordered_authored, :k, :loc))
        rk = RKBRMI(build(data))
        # With the response and without it (todo `1u6wzfy`); row 7 holds one
        # level of `k`.
        for design_data in (data, without(data, response))
            designs = [map(v -> v[rows], design_data) for rows in (2:5, [7], 1:9)]
            binder = check_binder(rk, designs; want)
            @test length(binder.classes) == 1
            @test !occursin("replayed in full", sprint(show, binder))
        end
    end
    # A design is refused where `reprocess` refuses its weights or outcomes.
    for (build, bad, message) in (
            (BM.weighted_normal, (; n=[1, -1, 2, 1]), "nonnegative integer-valued counts"),
            (BM.weighted_normal, (; n=[1, 2, 3]), "has length 3 but response `y` has length 4"),
            (BM.ordered, (; k=[1, 4, 2, 1]), "outside the fitted levels 1..3"),
            (BM.ordered, (; k=[0, 1, 2, 1]), "outside the fitted levels 1..3"),
            (BM.ordered_authored, (; k=[1, 4, 2, 1]), "beyond its fitted levels"))
        rk = RKBRMI(build(data))
        binder = rk_design_binder(rk)
        design = map(v -> v[2:5], data)
        binder(design)
        @test_throws message reprocess(rk, merge(design, bad))
        @test_throws message binder(merge(design, bad))
    end
end

@stestset "a masked HSGP binds its raw mask and axis" begin
    rng = Xoshiro(9)
    # Rows 2, 4, 6 and 8 are outside the mask, so their axis may be missing.
    holes(x) = Union{Missing,Float64}[i in (2, 4) ? missing : v for (i, v) in enumerate(x)]
    for (build, fitted) in ((BM.masked, (:hsgp_x_rows_fit, :x_fit)), (BM.masked_periodic, ()))
        rk = RKBRMI(build(flat_data(rng, 12)))
        designs = [map(v -> v[rows], without(flat_data(rng, 12), :y)) for rows in (2:6, 3:9, 1:12)]
        missing_designs = [merge(d, (; x=holes(d.x))) for d in
            (map(v -> v[1:9], without(flat_data(rng, 12), :y)),
             map(v -> v[1:5], without(flat_data(rng, 12), :y)))]
        binder = check_binder(rk, [designs; missing_designs]; want=:mu)
        @test length(binder.classes) == 2
        # Every input but the fit-role ones is a design column.
        @test !occursin("replayed in full", sprint(show, binder))
        @test keys(binder.fitted) == fitted
        # Refused where `reprocess` refuses it, after the kind's first design.
        base = missing_designs[1]
        numeric = merge(base, (; keep=Float64.(base.keep)))
        binder(numeric)
        function refused_alike(design, message)
            @test_throws message reprocess(rk, design)
            @test_throws message binder(design)
        end
        refused_alike(merge(base, (; x=[missing; base.x[2:end]])),
            "missing in a row its `rows=` mask marks as carrying")
        refused_alike(merge(numeric, (; keep=[2.0; numeric.keep[2:end]])),
            "must hold Bool (or 0/1)")
        refused_alike(merge(base, (; keep=base.keep[2:end])), "has 8 rows but the axis has 9")
        refused_alike(merge(base, (; x=[base.x[1:5]; Inf; base.x[7:end]])),
            "`hsgp` axes must be finite")
        build === BM.masked && refused_alike(merge(base, (; x=[100.0; base.x[2:end]])),
            "outside the fitted basis interval")
    end
end

@stestset "an areal field binds its fitted graph once" begin
    rng = Xoshiro(19)
    edges = (; from=["a", "b", "c", "a", "d", "e"], to=["b", "c", "d", "c", "h", "f"])
    areas(n) = ["a", "b", "c", "d", "e", "f", "g"][mod1.(1:n, 7)]
    rk = RKBRMI(BM.areal((; area=areas(14), edges..., y=randn(rng, 14))))
    designs = [(; area=areas(n)[rows]) for (n, rows) in ((14, 2:5), (14, 4:12), (7, [7, 1]))]
    binder = check_binder(rk, designs; want=:mu)
    @test !occursin("replayed in full", sprint(show, binder))
    graph = (:car_mu_area_inv_scale, :car_mu_area_is_struct, :car_mu_area_node1,
        :car_mu_area_node2, :car_mu_area_struct_pos)
    @test keys(binder.fitted) == graph
    # Designs that also list edges keep the fitted graph, as a replay does.
    check_binder(rk, [(; d..., edges...) for d in designs]; want=:mu)
    @test_throws "not an area of the fitted graph" reprocess(rk, (; area=["a", "z"]))
    @test_throws "not an area of the fitted graph" binder((; area=["a", "z"]))
    # Resampled areas refit the graph from each design.
    resampled = rk_design_binder(rk; resample_groups=[:area])
    @test isempty(resampled.fitted)
end

@stestset "hook data replays every design; a hook statement does not" begin
    rng = Xoshiro(13)
    for (build, planned) in ((BM.hooked_data, true), (BM.hooked_statement, false))
        rk = RKBRMI(build((; x=randn(rng, 8), y=randn(rng, 8))))
        binder = rk_design_binder(rk)
        for n in (3, 5, 2)
            design = (; x=randn(rng, n))
            @test same_inputs(binder(design), rk_bound_data(reprocess(rk, design)))
        end
        @test occursin("replayed in full", sprint(show, binder)) == planned
    end
end

# 24 training rows: a thin-plate basis needs 10 unique axis values.
fit_data(rng, n) = (; x=collect(range(0.3, 4.1; length=n)) .+ 0.01 .* rand(rng, n),
    z=sin.(1:n) .+ 0.1 .* rand(rng, n), g=["a", "b", "c"][mod1.(1:n, 3)],
    h=["p", "q", "r", "s"][mod1.(1:n, 4)], y=randn(rng, n))

@stestset "values fitted from the training data alone bind frozen" begin
    rng = Xoshiro(19)
    train = fit_data(rng, 24)
    designs = [map(v -> v[rows], without(fit_data(rng, 24), :y)) for rows in (2:9, 5:20, 11:13)]
    # The definitions each program fits from its fit-role inputs alone.
    for (build, names) in ((BM.thin_plate, (:s_x_fit,)), (BM.tensor, (:t2_x_z_fit,)),
            (BM.interval, (:hsgp_x_fit, :hsgp_x_rho_floor)),
            (BM.zscaled, (:x_fit_mean, :x_fit_sd)),
            (BM.shrunk, (:x_fit_var, :h_fit_levels, :mu_g_lvl_2_indicator_var)),
            (BM.grouped, (:h_fit_levels,)))
        rk = RKBRMI(build(train))
        binder = check_binder(rk, designs; want=:mu)
        # One kind, bound into its lowering: no input forces a replay.
        @test length(binder.classes) == 1 && !occursin("replayed in full", sprint(show, binder))
        frozen = only(values(binder.classes)).frozen
        @test frozen !== nothing && all(name -> haskey(frozen.values, name), names)
        # A later design binds them by value: its binding evaluates none of them.
        @test all(name -> haskey(BRM._rk_design_plan(binder, designs[2]).frozen.values, name),
            names)
    end
    # A resampled grouping's levels are the design's own, never frozen.
    rk = RKBRMI(BM.grouped(train))
    binder = check_binder(rk, designs; resample_groups=[:h], want=:mu,
        runtime=d -> (:b_h => reshape(0.1 .* collect(1.0:length(unique(d.h))), :, 1),))
    frozen = only(values(binder.classes)).frozen
    @test frozen === nothing || !haskey(frozen.values, :h_fit_levels)
end

@stestset "a thin-plate fit is a packed vector the basis is evaluated from" begin
    rng = Xoshiro(23)
    x_fit, x = randn(rng, 30), randn(rng, 7)
    for k in (4, 10)
        fit = BRM.brm_tps_fit(x_fit, k)
        @test length(fit) == 1 + 30 * (k - 1)
        # The basis the emitted program evaluates from the packed fit is,
        # bit for bit, the one fitted from the axis.
        @test isequal(BRM.brm_tps_apply(x, fit, k),
            BRM._brm_apply_spline(BRM._brm_fit_spline(x_fit; k), x))
        @test isequal(BRM.brm_tps_basis(x, x_fit, k), BRM.brm_tps_apply(x, fit, k))
    end
    # refused: a packed fit's length fixes its knot count for its rank, so a
    # vector of another length is no fit of that rank (dev §1: no silent misread).
    @test_throws "packed thin-plate fit" BRM.brm_tps_apply(x, x_fit, 10)
end
