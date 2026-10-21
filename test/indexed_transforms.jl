# test/indexed_transforms.jl — a per-group field read onto rows by integer
# position, `x[g]`, and a fitted transform of it, `standardize(x)[g]`,
# `zscale(x)[g]` and `center(x)[g]` (snag `standardize-equa-a92391f2`, user
# decision `0siic7u`). The transform fits over `x`'s own values, one per
# group, so every group weighs once however many rows read it. Each indexed
# model is checked against its twin, which reads a hand-built row column.
#
# Run: julia --project=test test/indexed_transforms.jl

include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Statistics

module IndexedTransforms
using BayesianRegressionModels, Distributions
standardized = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + standardize(age)[subject] + (1 | subject)
    y ~ Normal(mu, sigma)
end
zscaled = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + zscale(age)[subject] + (1 | subject)
    y ~ Normal(mu, sigma)
end
centered = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + center(age)[subject] + (1 | subject)
    y ~ Normal(mu, sigma)
end
read = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + age[subject] + (1 | subject)
    y ~ Normal(mu, sigma)
end
row_weighted = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + standardize(age[subject]) + (1 | subject)
    y ~ Normal(mu, sigma)
end
twin = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + zage + (1 | subject)
    y ~ Normal(mu, sigma)
end
slope = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + standardize(age)[subject] | site)
    y ~ Normal(mu, sigma)
end
slope_twin = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + zage | site)
    y ~ Normal(mu, sigma)
end
row_slope = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + zscale(x) | site)
    y ~ Normal(mu, sigma)
end
row_slope_twin = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + zx | site)
    y ~ Normal(mu, sigma)
end
known_scale = @brm begin
    mu ~ 1 + x + standardize(age)[subject]
    y ~ Normal(mu, 1.0)
end
known_scale_twin = @brm begin
    mu ~ 1 + x + zage
    y ~ Normal(mu, 1.0)
end
keyword = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + standardize(age; by=subject) + (1 | subject)
    y ~ Normal(mu, sigma)
end
keyword_slope = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + standardize(age; by=subject) | site)
    y ~ Normal(mu, sigma)
end
end

const IT = IndexedTransforms
# Unequal rows per group, so a row-weighted fit differs from a per-group one.
const ROWS = [1, 2, 5, 8, 3, 10]
const AGE = [25.0, 31.0, 47.0, 52.0, 60.0, 38.0]
const SUBJECT = reduce(vcat, [fill(g, n) for (g, n) in enumerate(ROWS)])
const N = length(SUBJECT)
const X = sin.(1:N)
const Y = cos.(1:N)
const SITE = [isodd(i) ? 1 : 2 for i in 1:N]
indexed_data(; age=AGE, subject=SUBJECT, x=X, y=Y) = (; subject, age, x, y, site=SITE[1:length(subject)])
zscored(age) = (age .- mean(age)) ./ std(age)
twin_data(zage; subject=SUBJECT, x=X, y=Y) = (; subject, zage, x, y, site=SITE[1:length(subject)])

sbfit(builder, data; kwargs...) = SBBRMI(builder(data); mod=IT, kwargs...)
# The indexed model must be its twin's program with the twin's column renamed.
renamed(code, from, to) = replace(code, from => to)
# Equal Stan data up to that renaming; fitted values agree to rounding (BRM's
# overflow-safe mean is not `Statistics.mean` bit for bit).
same_value(a::AbstractArray{<:AbstractFloat}, b::AbstractArray{<:AbstractFloat}) =
    size(a) == size(b) && isapprox(a, b; rtol=1e-12, atol=1e-12)
same_value(a, b) = isequal(a, b)
function same_data(indexed, twin, label)
    a, b = BRM.stan_data(indexed), BRM.stan_data(twin)
    twin_key(k) = Symbol(replace(string(k), string(label) => "zage"))  # also `<label>_n`
    Set(twin_key.(keys(a))) == Set(keys(b)) || return false
    all(k -> same_value(a[k], b[twin_key(k)]), keys(a))
end
stanc_ok(sb) = StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic=false).ok
record(sb, kind) = only(e for e in values(sb.preproc) if e.kind === kind)

@stestset "standardize(x)[g] fits over one value per group" begin
    data = indexed_data()
    zage = zscored(AGE)[SUBJECT]
    @test !(mean(AGE[SUBJECT]) ≈ mean(AGE))   # the two weightings differ here
    for total_groups in ((), :auto)
        sb = sbfit(IT.standardized, data; total_groups)
        twin = sbfit(IT.twin, twin_data(zage); total_groups)
        @test renamed(BRM.stan_code(twin), "zage", "standardize_age_subject") ==
            BRM.stan_code(sb)
        @test same_data(sb, twin, :standardize_age_subject)
        @test stanc_ok(sb)
    end
    sb = sbfit(IT.standardized, data; total_groups=())
    e = sb.preproc[:standardize_age_subject]
    @test e.kind === :standardize
    @test collect(e.const_) ≈ [mean(AGE), std(AGE)]
    @test e.raw_ref isa BRM._BRMIndexedSource
    @test BRM.stan_data(sb)[:standardize_age_subject] ≈ zage
    @test !haskey(BRM.stan_data(sb), :age)   # Stan reads the prepared column only
    # The transform of an indexed vector fits over the rows it is given.
    rows = record(sbfit(IT.row_weighted, data; total_groups=()), :standardize)
    @test collect(rows.const_) ≈ [mean(AGE[SUBJECT]), std(AGE[SUBJECT])]
end

@stestset "zscale/center and the bare indexed read" begin
    data = indexed_data()
    zsb = sbfit(IT.zscaled, data; total_groups=())
    @test BRM.stan_data(zsb)[:zscale_age_subject] ≈ zscored(AGE)[SUBJECT]
    @test zsb.preproc[:zscale_age_subject].kind === :zscale
    csb = sbfit(IT.centered, data; total_groups=())
    @test csb.preproc[:center_age_subject].const_ ≈ mean(AGE)
    @test BRM.stan_data(csb)[:center_age_subject] ≈ (AGE .- mean(AGE))[SUBJECT]
    rsb = sbfit(IT.read, data; total_groups=())
    @test BRM.stan_data(rsb)[:age_subject] == AGE[SUBJECT]
    @test rsb.preproc[:age_subject].kind === :protect
    @test all(stanc_ok, (zsb, csb, rsb))
end

@stestset "replay freezes or refits the per-group fit" begin
    data = indexed_data()
    sb = sbfit(IT.standardized, data; total_groups=())
    mean_, sd_ = sb.preproc[:standardize_age_subject].const_
    new_age, new_subject = [70.0, 20.0, 33.0], [3, 1, 1, 2, 3]
    new = indexed_data(; age=new_age, subject=new_subject, x=zeros(5), y=zeros(5))
    frozen = reprocess(sb, new)
    @test BRM.stan_data(frozen)[:standardize_age_subject] ≈
        ((new_age .- mean_) ./ sd_)[new_subject]
    @test frozen.preproc[:standardize_age_subject].const_ == (mean_, sd_)
    refit = reprocess(sb, new; freeze_constants=false)
    @test BRM.stan_data(refit)[:standardize_age_subject] ≈ zscored(new_age)[new_subject]
    @test collect(refit.preproc[:standardize_age_subject].const_) ≈
        [mean(new_age), std(new_age)]
    resampled = reprocess(sb, new; resample_groups=[:subject])
    @test BRM.stan_data(resampled)[:standardize_age_subject] ≈
        ((new_age .- mean_) ./ sd_)[new_subject]
    read = reprocess(sbfit(IT.read, data; total_groups=()), new)
    @test BRM.stan_data(read)[:age_subject] == new_age[new_subject]
    # Default exact-totals design: the per-group values live in the fitted
    # totals, so a replay keeps them and refuses changed ones, as for the twin.
    totals = sbfit(IT.standardized, data)
    @test BRM.stan_data(reprocess(totals, indexed_data(; x=zeros(N), y=zeros(N))))[:total_W_mu] ==
        BRM.stan_data(totals)[:total_W_mu]
    unrepresented = "has no representation in the fitted totals"
    @test_throws unrepresented reprocess(totals, indexed_data(; age=AGE .+ 1))
    twin_totals = sbfit(IT.twin, twin_data(zscored(AGE)[SUBJECT]))
    @test_throws unrepresented reprocess(twin_totals,
        twin_data(((AGE .+ 1 .- mean(AGE)) ./ std(AGE))[SUBJECT]))
    # A position past the per-group field is refused, never wrapped.
    @test_throws "position 4 (row 1) is outside 1:3" reprocess(sb,
        indexed_data(; age=new_age, subject=[4, 1], x=zeros(2), y=zeros(2)))
end

@stestset "random-effect slope over another grouping" begin
    data = indexed_data()
    zage = zscored(AGE)[SUBJECT]
    sb = sbfit(IT.slope, data; total_groups=())
    @test BRM.stan_data(sb)[:standardize_age_subject] ≈ zage
    @test collect(sb.preproc[:standardize_age_subject].const_) ≈ [mean(AGE), std(AGE)]
    # The slope design's intercept column takes the row axis from the index,
    # never from the per-group field.
    @test occursin("rep_vector(1.0, num_elements(subject))", BRM.stan_code(sb))
    @test stanc_ok(sb)
    names, densities = stan_density_points(IT.slope(data), "indexed-slope")
    twin_names, twin_densities = stan_density_points(
        IT.slope_twin(twin_data(zage)), "indexed-slope-twin")
    @test names == twin_names
    @test densities ≈ twin_densities rtol=1e-12
    # RKBRMI gathers the same derived column for an indexed or transformed slope.
    for (model, twin) in ((IT.slope(data), IT.slope_twin(twin_data(zage))),
            (IT.row_slope(data), IT.row_slope_twin((; data..., zx=zscored(X)))))
        _, problem = consumer_problem(model)
        _, twin_problem = consumer_problem(twin)
        u = collect(range(-0.4, 0.4; length=LogDensityProblems.dimension(problem)))
        @test LogDensityProblems.dimension(problem) == LogDensityProblems.dimension(twin_problem)
        @test LogDensityProblems.logdensity(problem, u) ≈
            LogDensityProblems.logdensity(twin_problem, u) atol=1e-11 rtol=1e-11
    end
end

@stestset "RKBRMI matches the hand-built column and the Stan program" begin
    data = indexed_data()
    zage = zscored(AGE)[SUBJECT]
    brmi = IT.standardized(data)
    backend, problem = consumer_problem(brmi)
    _, twin_problem = consumer_problem(IT.twin(twin_data(zage)))
    names = coordinate_names(backend.model.layout)
    n = length(names)
    points = (zeros(n), fill(0.13, n), collect(range(-0.4, 0.4; length=n)))
    for u in points
        @test LogDensityProblems.logdensity(problem, u) ≈
            LogDensityProblems.logdensity(twin_problem, u) atol=1e-11 rtol=1e-11
    end
    stan = consumer_stan(brmi, "indexed-standardize"; mod=IT)
    transport = brm_coordinate_transport(backend,
        SBBRMI(brmi; mod=IT, total_groups=()), BridgeStan.param_unc_names(stan.model))
    mapping = [p.rk => p.stan for p in transport.pairs]
    @test length(mapping) == n
    for u in points
        check_consumer_stan(problem, stan, mapping, backend, u)
    end
    # A frozen replay keeps the training fit on the per-group field.
    new_age, new_subject = [70.0, 20.0, 33.0], [3, 1, 1, 2, 3]
    new = indexed_data(; age=new_age, subject=new_subject, x=collect(1.0:5), y=zeros(5))
    # Its twin replays the hand column computed with the training constants;
    # both keep the training groups.
    ad_backend = AutoEnzyme(; mode=Enzyme.Reverse)
    replay = reprocess(RKBRMI(brmi), new)
    replay_problem = rk_logdensity_problem(replay; ad_backend)
    frozen_zage = ((new_age .- mean(AGE)) ./ std(AGE))[new_subject]
    twin_replay = reprocess(RKBRMI(IT.twin(twin_data(zage))), twin_data(frozen_zage;
        subject=new_subject, x=collect(1.0:5), y=zeros(5)))
    frozen_twin = rk_logdensity_problem(twin_replay; ad_backend)
    m = LogDensityProblems.dimension(replay_problem)
    @test m == LogDensityProblems.dimension(frozen_twin)
    for u in (zeros(m), collect(range(-0.3, 0.3; length=m)))
        @test LogDensityProblems.logdensity(replay_problem, u) ≈
            LogDensityProblems.logdensity(frozen_twin, u) atol=1e-11 rtol=1e-11
    end
end

@stestset "VBRMI reads the per-group field" begin
    data = indexed_data()
    v = VBRMI(IT.known_scale(data))
    twin = VBRMI(IT.known_scale_twin(twin_data(zscored(AGE)[SUBJECT])))
    n = LogDensityProblems.dimension(v)
    @test n == LogDensityProblems.dimension(twin)
    u = collect(range(-0.4, 0.4; length=n))
    @test LogDensityProblems.logdensity(v, u) ≈ LogDensityProblems.logdensity(twin, u)
end

@stestset "descriptor and description" begin
    data = indexed_data()
    d = brm_descriptor(IT.standardized, data; mod=IT, name=:indexed)
    @test Set(d.columns) == Set((:age, :subject, :x, :y))
    sb = sbfit(IT.standardized, data; total_groups=())
    input = only(i for i in brm_descriptor(sb; name=:indexed).inputs
                 if i.name === :standardize_age_subject)
    @test input.transform === :standardize
    md = brm_description_markdown(brm_description(brm_descriptor(sb; name=:indexed)))
    @test occursin("\\frac{{\\mathrm{age}}_{\\mathrm{subject}}-c_{mu,3}}{s_{mu,3}}", md)
    @test occursin(string(first(sb.preproc[:standardize_age_subject].const_)), md)
end

# refused: a keyword on `center`/`zscale`/`standardize` used to be dropped on
# SBBRMI and VBRMI, silently keeping the row-level fit; the per-group fit is
# spelled `standardize(x)[g]` (snag `standardize-equa-a92391f2`, decision `0siic7u`).
@stestset "a keyword on a transform is refused, never ignored" begin
    data = (; subject=SUBJECT, age=AGE[SUBJECT], x=X, y=Y, site=SITE)
    message = "takes no keywords (got `by`)"
    @test_throws message sbfit(IT.keyword, data; total_groups=())
    @test_throws message sbfit(IT.keyword, data)
    @test_throws message sbfit(IT.keyword_slope, data; total_groups=())
    @test_throws message RKBRMI(IT.keyword(data))
    @test_throws message VBRMI(IT.keyword(data))
    @test_throws "standardize(x)[group]" sbfit(IT.keyword, data; total_groups=())
end

# refused: `x[g]` reads `x` at integer positions; any other index would have
# to guess an order (snag `standardize-equa-a92391f2`).
@stestset "an indexed read needs integer positions within the field" begin
    @test_throws "outside 1:6" sbfit(IT.standardized,
        indexed_data(; subject=[SUBJECT[1:end-1]; 7]); total_groups=())
    @test_throws "integer positions" sbfit(IT.standardized,
        indexed_data(; subject=string.(SUBJECT)); total_groups=())
    @test_throws "outside 1:6" RKBRMI(IT.standardized(
        indexed_data(; subject=[SUBJECT[1:end-1]; 7])))
end
