# test/model_family_composition.jl — composing a family of related models
# without string templating: optional prior addresses, the typed data-free
# builder (`BRMBuilder`, `brm_columns`, builder `merge`), source views of a
# model, string categorical columns on both backends, order-free structural
# `merge`, and the loud `effect(<unknown predictor>, :)`.
#
# Run: julia --startup-file=no --project=test test/model_family_composition.jl

include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Random
using BayesianRegressionModels: ExprColumn, getargs, getf

const FAMILY_N = 24
const FAMILY_DATA = let rng = Random.MersenneTwister(1)
    (; x=randn(rng, FAMILY_N), y=randn(rng, FAMILY_N), y2=randn(rng, FAMILY_N),
       subj=repeat(1:6, inner=4), z=repeat(1:3, 8), c=repeat([1, 2, 3], 8),
       ind=repeat(["a", "b", "c"], 8),
       zm=Union{Missing,Float64}[i == 3 ? missing : randn(rng) for i in 1:FAMILY_N])
end
const FAMILY_CATEGORICAL = merge(FAMILY_DATA, (; indc=categorical(FAMILY_DATA.ind)))

# A pre-grouped subject frame with a flat observation frame for `@plate` cells.
const PLATE_DATA = (;
    subject=["b", "a", "c"], weight=[60.0, 75.0, 90.0],
    obs_subject=["a", "c", "b", "a", "b", "c"],
    obs_time=[0.5, 0.25, 1.5, 1.0, 3.0, 2.0],
    y=[0.81, 0.76, 0.52, 0.49, 0.21, 0.33])
const PLATE_SOURCE = """
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    obs_rows = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        scaled = 2 .* obs_time[obs_rows[i]]
        loc[i] = exp.(-exp(log_CL[i]) .* scaled)
        y[obs_rows[i]] ~ normal(loc[i], sigma)
    end"""
const PLATE_BUILDER = @brm begin
    sigma ~ Exponential(1)
    log_CL ~ 1 + weight + (1 | p | subject)
    obs_rows = groups(obs_subject => subject)
    @plate for i in eachindex(log_CL)
        scaled = 2 .* obs_time[obs_rows[i]]
        loc[i] = exp.(-exp(log_CL[i]) .* scaled)
        y[obs_rows[i]] ~ normal(loc[i], sigma)
    end
end

sb_code(brmi) = BRM.stan_code(SBBRMI(brmi; mod=@__MODULE__))
stanc_ok(code) = StanBlocks.stanc_check(code; warn_pedantic=false).ok
has_optional(brmi) = any(BRM._brm_is_optional_prior ∘ first ∘ (op -> getargs(op, 2)),
    filter(op -> op isa ExprColumn && getf(op) === (~),
           map(BRM._named_op, BRM._brm_operation_values(brmi))))

@testset "optional prior addresses" begin
    # Every address class, present and absent. The absent ones are dropped
    # before lowering; the present ones lower exactly as unmarked statements.
    optional = @brm FAMILY_DATA begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 + x | p | subj)
        y ~ Normal(mu, sigma)
        optional(effect(mu, weight)) ~ Normal(0, 0.1)      # absent covariate
        optional(effect(mu, x)) ~ Normal(0, 0.5)
        optional(effect(nolp, :)) ~ Normal(0, 0.5)         # absent predictor
        optional(effect(nolp2, :)) ~ r2d2()
        optional(sd(:, q)) ~ Exponential(3)                # absent `|q|` block
        optional(sd(:, p)) ~ Exponential(2)
        optional(sd(mu, p, weight)) ~ Exponential(2)       # absent margin
        optional(sd(mu, p, x)) ~ Exponential(1.5)
        optional(cor(:, q)) ~ LKJCholesky(2, 2)
        optional(cor(:, p)) ~ LKJCholesky(2, 2)
        optional(ranef(:, q)) ~ TDist(4)
        optional(simplex(mu, mo1(z))) ~ Dirichlet(2, 1)    # absent term
        optional(sd(mu, s(x))) ~ Exponential(1)
    end
    plain = @brm FAMILY_DATA begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 + x | p | subj)
        y ~ Normal(mu, sigma)
        effect(mu, x) ~ Normal(0, 0.5)
        sd(:, p) ~ Exponential(2)
        sd(mu, p, x) ~ Exponential(1.5)
        cor(:, p) ~ LKJCholesky(2, 2)
    end
    resolved = BRM._brm_resolve_optional_priors(optional)
    @test !has_optional(resolved)
    @test sprint(show, resolved) == sprint(show, plain)
    @test BRM._brm_resolve_optional_priors(plain) === plain
    code = sb_code(optional)
    @test code == sb_code(plain)
    @test stanc_ok(code)
    # A backend keeps the model it lowered as its parent.
    @test !has_optional(parent(SBBRMI(optional; mod=@__MODULE__)))
    @test !has_optional(parent(RKBRMI(optional)))
    @test BRM.emit_rk_artifact(optional; case_id="optional").ast ==
          BRM.emit_rk_artifact(plain; case_id="optional").ast

    # A known value and a categorical block can be optional too.
    categorical_model = @brm FAMILY_DATA begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + factor(c) + (1 | p | subj)
        y ~ Normal(mu, sigma)
        optional(effect(mu, c)) ~ Normal(0, 0.3)
        optional(effect(mu, ind)) ~ Normal(0, 0.3)
        optional(sd(:, p)) = 0.4
        optional(sd(:, q)) = 0.4
    end
    categorical_plain = @brm FAMILY_DATA begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + factor(c) + (1 | p | subj)
        y ~ Normal(mu, sigma)
        effect(mu, c) ~ Normal(0, 0.3)
        sd(:, p) = 0.4
    end
    @test sb_code(categorical_model) == sb_code(categorical_plain)

    # Optional priors resolve on the composed model, so a prior fragment can be
    # merged onto any member of the family.
    structure = structure_of(plain)
    priors = @brm FAMILY_DATA begin
        optional(effect(mu, weight)) ~ Normal(0, 0.1)
        optional(effect(mu, x)) ~ Normal(0, 0.5)
        optional(sd(:, p)) ~ Exponential(2)
    end
    composed = sb_code(merge(structure, priors))
    @test composed == sb_code(merge(structure, @brm FAMILY_DATA begin
        effect(mu, x) ~ Normal(0, 0.5)
        sd(:, p) ~ Exponential(2)
    end))

    # Unmarked addresses still fail loudly; a kept optional address the backend
    # cannot place fails loudly too, so nothing is dropped on a guess.
    @test_throws "is not a population coefficient" sb_code(merge(plain, @brm FAMILY_DATA begin
        effect(mu, weight) ~ Normal(0, 0.1)
    end))
    no_intercept = @brm FAMILY_DATA begin
        sigma ~ Exponential(1)
        mu ~ 0 + x
        y ~ Normal(mu, sigma)
        optional(effect(mu, Intercept)) ~ Normal(0, 1)
    end
    @test_throws "Intercept" sb_code(no_intercept)

    # refused: one statement wraps exactly one prior address (parse contract)
    @test_throws "wraps exactly one prior address" BRM._brm(:(begin
        optional(mu) ~ Normal(0, 1)
    end))
    # refused: the marker and an unmarked statement on one address are two
    # statements for one parameter, as without the marker
    @test_throws "duplicate" BRM._brm(:(begin
        optional(effect(mu, x)) ~ Normal(0, 1)
        effect(mu, x) ~ Normal(0, 2)
    end))
end

@testset "source views print the model as written" begin
    model = @brm FAMILY_DATA begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + (1 + x | p | subj)
        y ~ Normal(mu, sigma)
        effect(mu, x) ~ Normal(0, 0.5)
        sd(:, p) ~ Exponential(2)
        sd(mu, p, x) ~ Exponential(1)
        cor(:, p) ~ LKJCholesky(2, 2)
        optional(effect(mu, weight)) ~ Normal(0, 0.1)
    end
    shown = sprint(show, model)
    @test !occursin("data (eltype", shown)
    @test !occursin("BRMI", shown)
    @test !occursin("effect(sd", shown) && !occursin("effect(cor", shown)
    @test occursin("sd(:, p) ~ Exponential(2)", shown)
    @test occursin("sd(mu, p, x) ~ Exponential(1)", shown)
    @test occursin("cor(:, p) ~ LKJCholesky(2, 2)", shown)
    @test occursin("optional(effect(mu, weight)) ~ Normal(0, 0.1)", shown)
    @test split(sprint(show, structure_of(model)), '\n') == [
        "sigma ~ Exponential(1)", "mu ~ 1 + x + (1 + x | p | subj)",
        "y ~ Normal(mu, sigma)"]
    @test sprint(show, structure_of(model)) * "\n" * sprint(show, priors_of(model)) ==
          shown
    # Term, interaction and standardized-law heads print as written.
    terms = @brm FAMILY_DATA begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + x & y2 + mo1(z) + (1 | p | subj)
        y ~ Normal(mu, sigma)
        effect(mu, x & y2) ~ Normal(0, 0.2)
        simplex(mu, mo1(z)) ~ Dirichlet(2, 1)
        ranef(:, p) ~ TDist(4)
        sd(:, p) = 0.3
    end
    shown_terms = sprint(show, priors_of(terms))
    for line in ("effect(mu, x & y2) ~ Normal(0, 0.2)", "simplex(mu, mo1(z)) ~ Dirichlet(2, 1)",
                 "ranef(:, p) ~ TDist(4)", "sd(:, p) = 0.3")
        @test occursin(line, shown_terms)
    end

    # Assignments and constants print as written.
    assigned = @brm FAMILY_DATA begin
        x_loc = constant(mean(x) + 1)
        sigma ~ Exponential(1)
        mu ~ 1 + x
        m = mu + x_loc
        y ~ Normal(m, sigma)
    end
    shown_assigned = sprint(show, assigned)
    @test occursin("\nm = mu + x_loc\n", shown_assigned)
    @test startswith(shown_assigned, "x_loc = constant(mean(x) + 1)\n")

    # A Symbol keyword value prints as the literal it is, on the built model and
    # on the builder (snag print-a-built-mo-d1f838f3).
    symbol_source = "sigma ~ Exponential(1)\n" *
        "mu ~ 1 + x + hsgp(x; k=5, cov=:exp_quad, extrapolate=:clamp)\n" *
        "y ~ Normal(mu, sigma)"
    symbol_builder = @brm begin
        sigma ~ Exponential(1)
        mu ~ 1 + x + hsgp(x; k=5, cov=:exp_quad, extrapolate=:clamp)
        y ~ Normal(mu, sigma)
    end
    symbols = symbol_builder(FAMILY_DATA)
    @test sprint(show, symbol_builder) == symbol_source
    @test sprint(show, symbols) == symbol_source
    @test sprint(show, structure_of(symbols)) == symbol_source

    # The printed source is the model: re-parsing it gives the same program.
    for m in (model, terms, assigned, symbols)
        reparsed = Core.eval(@__MODULE__,
            BRM._brm(Meta.parse("begin\n" * sprint(show, m) * "\nend"); df=FAMILY_DATA))
        @test sb_code(reparsed) == sb_code(m)
    end

    # A `@plate` cell prints as written, after the groupings it reads.
    plate = PLATE_BUILDER(PLATE_DATA)
    @test sprint(show, plate) == PLATE_SOURCE
    reparsed = Core.eval(@__MODULE__,
        BRM._brm(Meta.parse("begin\n" * sprint(show, plate) * "\nend"); df=PLATE_DATA))
    @test sb_code(reparsed) == sb_code(plate)
end

@testset "typed builder: data-free columns, merge, show" begin
    cases = [
        "simple" => (@brm begin sigma ~ Exponential(1); mu ~ 1 + x; y ~ Normal(mu, sigma) end),
        "ranef and string factor" => (@brm begin
            sigma ~ Exponential(1)
            mu ~ 1 + x + factor(ind; cmc=false) + (1 | p | subj)
            y ~ Normal(mu, sigma)
        end),
        "distributional" => (@brm begin mu ~ 1 + x; log(sigma) ~ 1 + c; y ~ Normal(mu, sigma) end),
        "imputed covariate" => (@brm begin
            sigma ~ Exponential(1)
            mi(zm) ~ Normal(0, 1)
            squared = zm * zm
            mu ~ 1 + x + zm + squared
            y ~ Normal(mu, sigma)
        end),
        "joint response" => (@brm begin
            mu1 ~ 1 + x; mu2 ~ 1 + x
            L ~ LKJCovarianceFactor(2; scale_prior=Exponential(1), shape=2)
            [y, y2] ~ MvNormalCholesky([mu1, mu2], L)
        end),
    ]
    for (label, builder) in cases
        @testset "$label" begin
            @test builder isa BRMBuilder
            @test builder(FAMILY_DATA) isa BRMI
            @test brm_columns(builder) ==
                  brm_columns(brm_descriptor(builder, FAMILY_DATA; mod=@__MODULE__))
        end
    end
    @test brm_columns(cases[1][2]) == (:x, :y)

    # A constant's inputs are columns a fresh build needs, unless the builder's
    # scope binds the name (`pi`, a captured local). The descriptor's replay
    # schema omits them, because replay keeps the fitted constant.
    calibrated = let k = 2.0
        @brm begin
            x_loc = constant(k * mean(y2) + pi)
            sigma ~ Exponential(1)
            mu ~ 1 + x
            m = mu + x_loc
            y ~ Normal(m, sigma)
        end
    end
    @test brm_columns(calibrated) == (:x, :y, :y2)
    @test :y2 ∉ brm_columns(brm_descriptor(calibrated, FAMILY_DATA; mod=@__MODULE__))
    @test stanc_ok(sb_code(calibrated(FAMILY_DATA)))

    @test brm_columns(PLATE_BUILDER) == (:obs_subject, :obs_time, :subject, :weight, :y)
    # The descriptor names the ragged response's flat column `y`, not the
    # cell's gathered `kernel_loc_y_ragged` key (todo `1dsty7q`).
    @test brm_columns(PLATE_BUILDER) ==
          brm_columns(brm_descriptor(PLATE_BUILDER, PLATE_DATA; mod=@__MODULE__))
    @test sprint(show, PLATE_BUILDER) == PLATE_SOURCE

    # Fragments compose into one builder exactly as one block written in the
    # fragments' order; the grouping and the cell may sit in different fragments.
    predictor = @brm begin mu ~ 1 + x end
    observation = @brm begin sigma ~ Exponential(1); y ~ Normal(mu, sigma) end
    priors = @brm begin
        optional(effect(mu, weight)) ~ Normal(0, 1)
        optional(effect(mu, x)) ~ Normal(0, 0.5)
    end
    inline = @brm begin
        mu ~ 1 + x
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
        effect(mu, x) ~ Normal(0, 0.5)
    end
    merged = merge(predictor, observation, priors)
    @test merged isa BRMBuilder
    @test brm_columns(merged) == (:x, :y)
    @test sb_code(merged(FAMILY_DATA)) == sb_code(inline(FAMILY_DATA))
    reversed = @brm begin
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
        mu ~ 1 + x
    end
    @test sb_code(merge(observation, predictor)(FAMILY_DATA)) ==
          sb_code(reversed(FAMILY_DATA))
    grouping = @brm begin obs_rows = groups(obs_subject => subject) end
    cell = @brm begin
        sigma ~ Exponential(1)
        log_CL ~ 1 + weight + (1 | p | subject)
        @plate for i in eachindex(log_CL)
            scaled = 2 .* obs_time[obs_rows[i]]
            loc[i] = exp.(-exp(log_CL[i]) .* scaled)
            y[obs_rows[i]] ~ normal(loc[i], sigma)
        end
    end
    @test sb_code(merge(cell, grouping)(PLATE_DATA)) == sb_code(PLATE_BUILDER(PLATE_DATA))
    @test sb_code(merge(grouping, cell)(PLATE_DATA)) == sb_code(PLATE_BUILDER(PLATE_DATA))
    # refused: built on its own, a binding no cell reads leaves its vectors
    # ungrouped (snag plate-body-reads-c69f8a88); only the composed model is
    # checked.
    @test_throws "no cell reads `obs_rows`" grouping(PLATE_DATA)
    # A merged builder is callable from code compiled before the merge.
    build_merged(df) = merge(predictor, observation)(df)
    @test build_merged(FAMILY_DATA) isa BRMI

    # `Expr` fragments compose into a builder without strings.
    covariates = (:x, :c)
    rhs = Expr(:call, :+, 1, covariates...)
    expr = Expr(:block, :(sigma ~ Exponential(1)), :(mu ~ $rhs), :(y ~ Normal(mu, sigma)))
    built = Core.eval(@__MODULE__, BRM._brm(expr))
    @test built isa BRMBuilder
    @test brm_columns(built) == (:c, :x, :y)
    @test sprint(show, built) == "sigma ~ Exponential(1)\nmu ~ 1 + x + c\ny ~ Normal(mu, sigma)"
end

@testset "string categorical columns" begin
    # SBBRMI: every String spelling lowers; `factor(...)` over strings codes
    # exactly like the same levels integer-coded in `sort(unique)` order.
    variants = [
        "1 + ind" => (@brm FAMILY_CATEGORICAL begin sigma ~ Exponential(1); mu ~ 1 + ind; y ~ Normal(mu, sigma) end),
        "1 + ind + x" => (@brm FAMILY_CATEGORICAL begin sigma ~ Exponential(1); mu ~ 1 + ind + x; y ~ Normal(mu, sigma) end),
        "0 + ind + x" => (@brm FAMILY_CATEGORICAL begin sigma ~ Exponential(1); mu ~ 0 + ind + x; y ~ Normal(mu, sigma) end),
        "factor(ind; ref)" => (@brm FAMILY_CATEGORICAL begin sigma ~ Exponential(1); mu ~ 1 + factor(ind; ref="b") + x; y ~ Normal(mu, sigma) end),
        "factor(ind; cmc=false)" => (@brm FAMILY_CATEGORICAL begin sigma ~ Exponential(1); mu ~ 0 + factor(ind; cmc=false) + x; y ~ Normal(mu, sigma) end),
        "factor(indc; cmc=false)" => (@brm FAMILY_CATEGORICAL begin sigma ~ Exponential(1); mu ~ 0 + factor(indc; cmc=false) + x; y ~ Normal(mu, sigma) end),
        "ranef + ref + prior" => (@brm FAMILY_CATEGORICAL begin
            sigma ~ Exponential(1)
            mu ~ 0 + factor(ind; cmc=false, ref="c") + x + (1 | p | subj)
            y ~ Normal(mu, sigma)
            effect(mu, ind) ~ Normal(0, 0.5)
        end),
    ]
    for (label, brmi) in variants
        @testset "SB $label" begin
            @test stanc_ok(sb_code(brmi))
        end
    end
    string_factor = @brm FAMILY_CATEGORICAL begin
        sigma ~ Exponential(1); mu ~ 1 + factor(ind; ref="b") + x; y ~ Normal(mu, sigma)
    end
    integer_twin = @brm merge(FAMILY_CATEGORICAL,
            (; code=[findfirst(==(v), ["a", "b", "c"]) for v in FAMILY_CATEGORICAL.ind])) begin
        sigma ~ Exponential(1); mu ~ 1 + factor(code; ref=2) + x; y ~ Normal(mu, sigma)
    end
    _, string_density = stan_density_points(string_factor, "family_string_factor")
    _, integer_density = stan_density_points(integer_twin, "family_integer_factor")
    @test string_density ≈ integer_density rtol=1e-12

    # A bare String column takes population priors on SBBRMI: its block address
    # claims every level column and a level address refines one of them.
    bare = @brm FAMILY_DATA begin
        sigma ~ Exponential(1)
        mu ~ 1 + ind + x
        y ~ Normal(mu, sigma)
        effect(mu, x) ~ Normal(0, 0.5)
        effect(mu, ind) ~ Normal(0, 0.7)
        effect(mu, ind_lvl_3) ~ Normal(1, 0.2)
    end
    @test popcoefnames(bare, :mu) == [:Intercept, :ind_lvl_2, :ind_lvl_3, :x]
    overrides = BRM._sb_effect_prior_overrides(bare)
    @test map(e -> isnothing(e) ? nothing : getargs(e), overrides[:mu].pop) ==
          [nothing, (0, 0.7), (1, 0.2), (0, 0.5)]
    @test stanc_ok(sb_code(bare))
    @test_throws "is not a population coefficient" sb_code(merge(bare, @brm FAMILY_DATA begin
        effect(mu, nope) ~ Normal(0, 1)
    end))

    # RKBRMI accepts the string-leveled `CategoricalVector` spelling.
    @test RKBRMI(@brm FAMILY_CATEGORICAL begin
        sigma ~ Exponential(1); mu ~ 0 + factor(indc; cmc=false) + x; y ~ Normal(mu, sigma)
    end) isa RKBRMI
end

@testset "structural merge is order-free" begin
    predictor = @brm FAMILY_DATA begin mu ~ 1 + x end
    observation = @brm FAMILY_DATA begin sigma ~ Exponential(1); y ~ Normal(mu, sigma) end
    inline = @brm FAMILY_DATA begin mu ~ 1 + x; sigma ~ Exponential(1); y ~ Normal(mu, sigma) end
    # Either order keeps `mu`'s formula (the reader's placeholder never
    # overrides it) and gives the inline program up to declaration order.
    lines(code) = sort(split(code, '\n'))
    for merged in (merge(predictor, observation), merge(observation, predictor))
        code = sb_code(merged)
        @test lines(code) == lines(sb_code(inline))
        @test stanc_ok(code)
    end
end

@testset "effect(<unknown predictor>, :) is refused on SBBRMI" begin
    model = @brm FAMILY_DATA begin
        sigma ~ Exponential(1); mu ~ 1 + x; y ~ Normal(mu, sigma)
        effect(nolp, :) ~ Normal(0, 0.5)
    end
    @test_throws "names no linear predictor" sb_code(model)
end
