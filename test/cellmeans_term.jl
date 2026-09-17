# test/cellmeans_term.jl — cell-mean coding of an intercept-free predictor's
# first categorical term (decision `0woa6hh`, brms / R `model.matrix`
# semantics): K addressable coefficients instead of K-1 treatment contrasts
# against a reference level pinned at zero. The same rule inside a random-effect
# block (decision `0wfo466`): `(0 + c | g)` gives every level of `c` its own
# group-level effect.

using Test
using BayesianRegressionModels
using StanBlocks
using LogDensityProblems
using Distributions: Cauchy, Exponential, Normal
import StanBlocks.stan: transpiles

const CELLS_RUNTIME = get(ENV, "BRM_CELLS_RUNTIME", "1") != "0"
const CELLS_CACHE = joinpath(tempdir(), "brm-cellmeans-term")
const BS = StanBlocks.BridgeStan

cells_df() = (; patch=[1, 2, 3, 1, 2, 3], arm=[1, 1, 2, 2, 1, 2],
                x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
                y=[-2.4, -2.2, -2.0, -1.8, -1.7, -1.5])

sbbrmi(brmi) = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
# BRM's own `stan_code` (it `invokelatest`s): a per-level prior of a new shape
# generates its vector-prior family during the call, inside this testset's world.
stan(sb::SBBRMI) = BayesianRegressionModels.stan_code(sb)
stan(brmi) = stan(sbbrmi(brmi))

# Exactly one declaration of `name`, so a constraint or size asserted below can
# never be satisfied by a different parameter of the same program.
declaration(code, name) = only(
    strip(l) for l in split(code, '\n')
    if occursin(Regex("\\b$(name);\$"), strip(l)) && !occursin("~", l))

@testset "no intercept: the first categorical term owns K cell means" begin
    df = cells_df()
    cells = @brm df begin
        mu ~ 0 + factor(patch)
        y ~ Normal(mu, 1.0)
    end
    code = stan(cells)
    @test declaration(code, "cat_mu_patch_beta") == "vector[patch_n_levels] cat_mu_patch_beta;"
    @test occursin("cat_mu_patch = cat_mu_patch_beta[patch_idx];", code)
    @test !occursin("append_row(0.0, cat_mu_patch_beta)", code)
    @test occursin("cat_mu_patch_beta ~ std_normal();", code)
    @test popcoefnames(cells, :mu) == Symbol[]
    @test transpiles(sbbrmi(cells).model)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok

    # `0` is only a marker and BRM has no implicit intercept: the bare column,
    # with or without the marker, is the same formula.
    bare = @brm df begin
        mu ~ patch
        y ~ Normal(mu, 1.0)
    end
    @test stan(bare) == code
end

@testset "an intercept keeps K-1 treatment contrasts" begin
    df = cells_df()
    treated = @brm df begin
        mu ~ 1 + factor(patch)
        y ~ Normal(mu, 1.0)
    end
    code = stan(treated)
    @test declaration(code, "cat_mu_patch_beta") ==
          "vector[(patch_n_levels - 1)] cat_mu_patch_beta;"
    @test occursin("append_row(0.0, cat_mu_patch_beta)[patch_idx]", code)
end

@testset "only the FIRST eligible categorical term is cell-mean coded" begin
    df = cells_df()
    two = @brm df begin
        mu ~ x + patch + arm
        y ~ Normal(mu, 1.0)
    end
    code = stan(two)
    @test declaration(code, "cat_mu_patch_beta") == "vector[patch_n_levels] cat_mu_patch_beta;"
    @test declaration(code, "cat_mu_arm_beta") == "vector[(arm_n_levels - 1)] cat_mu_arm_beta;"
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok

    # A random intercept is not a population intercept.
    grouped = @brm df begin
        mu ~ 0 + patch + (1 | arm)
        y ~ Normal(mu, 1.0)
    end
    @test declaration(stan(grouped), "cat_mu_patch_beta") ==
          "vector[patch_n_levels] cat_mu_patch_beta;"
end

@testset "`cmc=false` requests treatment coding (brms' switch)" begin
    df = cells_df()
    pinned = @brm df begin
        mu ~ 0 + factor(patch; cmc=false)
        y ~ Normal(mu, 1.0)
    end
    code = stan(pinned)
    @test declaration(code, "cat_mu_patch_beta") ==
          "vector[(patch_n_levels - 1)] cat_mu_patch_beta;"
    @test occursin("append_row(0.0, cat_mu_patch_beta)[patch_idx]", code)
    # `cmc=true` is the default, spelled out.
    explicit = @brm df begin
        mu ~ 0 + factor(patch; cmc=true)
        y ~ Normal(mu, 1.0)
    end
    @test declaration(stan(explicit), "cat_mu_patch_beta") ==
          "vector[patch_n_levels] cat_mu_patch_beta;"
    wrong = @brm df begin
        mu ~ 0 + factor(patch; cmc=0)
        y ~ Normal(mu, 1.0)
    end
    @test_throws "expects `true` or `false`" sbbrmi(wrong)

    # A reference level alone does NOT opt out: as in R, a releveled factor
    # without an intercept is still cell-mean coded (in its releveled order);
    # with `cmc=false` it is K-1 contrasts against that reference.
    releveled = @brm df begin
        mu ~ 0 + factor(patch; ref=3)
        y ~ Normal(mu, 1.0)
    end
    @test declaration(stan(releveled), "cat_mu_patch__ref_3_beta") ==
          "vector[patch__ref_3_n_levels] cat_mu_patch__ref_3_beta;"
    reffed = @brm df begin
        mu ~ 0 + factor(patch; ref=3, cmc=false)
        y ~ Normal(mu, 1.0)
    end
    @test declaration(stan(reffed), "cat_mu_patch__ref_3_beta") ==
          "vector[(patch__ref_3_n_levels - 1)] cat_mu_patch__ref_3_beta;"

    # The cell means then pass to the next categorical term.
    passed = @brm df begin
        mu ~ 0 + factor(patch; cmc=false) + arm
        y ~ Normal(mu, 1.0)
    end
    passed_code = stan(passed)
    @test declaration(passed_code, "cat_mu_patch_beta") ==
          "vector[(patch_n_levels - 1)] cat_mu_patch_beta;"
    @test declaration(passed_code, "cat_mu_arm_beta") == "vector[arm_n_levels] cat_mu_arm_beta;"
    @test StanBlocks.stanc_check(passed_code; warn_pedantic=false).ok
end

@testset "an ordinal model's thresholds are its location predictor's intercept" begin
    od = (; x=[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5], group=[1, 2, 1, 2, 1, 2],
            period=[1, 2, 3, 1, 2, 3], y=[1, 1, 2, 2, 3, 3])
    ordinal = @brm od begin
        eta ~ 0 + x + period
        log(disc) ~ 0 + factor(group; cmc=false)
        y ~ Ordinal(Cumulative(), ProbitLink(), eta; discrimination=disc)
    end
    code = stan(ordinal)
    @test declaration(code, "cat_eta_period_beta") ==
          "vector[(period_n_levels - 1)] cat_eta_period_beta;"
    @test declaration(code, "cat_log_disc_group_beta") ==
          "vector[(group_n_levels - 1)] cat_log_disc_group_beta;"
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok

    # The discrimination predictor is not threshold-located: without
    # `cmc=false` it is an ordinary intercept-free predictor.
    free_disc = @brm od begin
        eta ~ 0 + x
        log(disc) ~ 0 + group
        y ~ Ordinal(Cumulative(), ProbitLink(), eta; discrimination=disc)
    end
    @test declaration(stan(free_disc), "cat_log_disc_group_beta") ==
          "vector[group_n_levels] cat_log_disc_group_beta;"

    legacy = @brm od begin
        eta ~ 0 + period
        y ~ OrderedLogistic(eta)
    end
    @test declaration(stan(legacy), "cat_eta_period_beta") ==
          "vector[(period_n_levels - 1)] cat_eta_period_beta;"
end

@testset "prior addresses: the block, and each level on its own" begin
    df = cells_df()
    block = @brm df begin
        mu ~ 0 + factor(patch)
        effect(mu, patch) ~ Normal(0.0, 0.5)
        y ~ Normal(mu, 1.0)
    end
    # One shared Normal keeps the scalar statement a treatment block emits.
    @test occursin("cat_mu_patch_beta ~ normal(0.0, 0.5);", stan(block))

    levelled = @brm df begin
        mu ~ 0 + factor(patch)
        effect(mu, patch) ~ Normal(0.0, 2.0)
        effect(mu, patch_lvl_3) ~ Normal(log(50), 0.5)
        y ~ Normal(mu, 1.0)
    end
    code = stan(levelled)
    @test occursin("cat_mu_patch_beta ~ normal([0.0, 0.0, $(log(50))]', [2.0, 2.0, 0.5]');", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok

    # A level address alone leaves the other levels at the default Normal(0, 1).
    lone = @brm df begin
        mu ~ 0 + factor(patch)
        effect(mu, patch_lvl_2) ~ Normal(1.0, 0.25)
        y ~ Normal(mu, 1.0)
    end
    @test occursin("cat_mu_patch_beta ~ normal([0.0, 1.0, 0.0]', [1.0, 0.25, 1.0]');", stan(lone))

    # The `:` layers reach cell means too, and the level address outranks the
    # block address of the same statement shape instead of tying with it.
    layered = @brm df begin
        mu ~ 0 + factor(patch)
        effect(:, :) ~ Normal(0.0, 3.0)
        effect(:, patch_lvl_1) ~ Normal(0.0, 0.1)
        y ~ Normal(mu, 1.0)
    end
    @test occursin("cat_mu_patch_beta ~ normal([0.0, 0.0, 0.0]', [0.1, 3.0, 3.0]');", stan(layered))

    tied = @brm df begin
        mu ~ 0 + factor(patch)
        effect(:, patch) ~ Normal(0.0, 0.5)
        effect(mu, :) ~ Normal(0.0, 0.25)
        y ~ Normal(mu, 1.0)
    end
    @test_throws "equally specific" sbbrmi(tied)

    # A per-level family that is not Normal takes the generated vector prior.
    heavy = @brm df begin
        mu ~ 0 + factor(patch)
        effect(mu, patch_lvl_1) ~ Cauchy(0, 1)
        y ~ Normal(mu, 1.0)
    end
    heavy_code = stan(heavy)
    @test occursin(r"cat_mu_patch_beta ~ brm_vector_prior_[0-9a-f]+", heavy_code)
    @test count("cauchy_lpdf(x[", heavy_code) == 1
    @test StanBlocks.stanc_check(heavy_code; warn_pedantic=false).ok
end

@testset "level addresses exist only where cell means do" begin
    df = cells_df()
    treated = @brm df begin
        mu ~ 1 + factor(patch)
        effect(mu, patch_lvl_2) ~ Normal(0.0, 0.5)
        y ~ Normal(mu, 1.0)
    end
    @test_throws "not a population coefficient" sbbrmi(treated)

    beyond = @brm df begin
        mu ~ 0 + factor(patch)
        effect(mu, patch_lvl_4) ~ Normal(0.0, 0.5)
        y ~ Normal(mu, 1.0)
    end
    @test_throws "patch_lvl_4" sbbrmi(beyond)
end

@testset "replay keeps the frozen level set" begin
    df = cells_df()
    cells = @brm df begin
        mu ~ 0 + factor(patch)
        y ~ Normal(mu, 1.0)
    end
    sb = sbbrmi(cells)
    @test sb.data[:patch_n_levels] == 3
    replay = reprocess(sb, (; patch=[3, 1], y=[0.0, 0.0]))
    @test replay.data[:patch_idx] == [3, 1]
    @test replay.data[:patch_n_levels] == 3
    @test stan(replay) == stan(sb)
    @test_throws "not a training level" reprocess(sb, (; patch=[1, 9], y=[0.0, 0.0]))

    # A level address names a position in the FITTED level order. A replay
    # frame carrying only a subset of the levels must neither lose the address
    # nor shrink the per-level prior vector.
    levelled = @brm df begin
        mu ~ 0 + factor(patch)
        effect(mu, patch_lvl_3) ~ Normal(log(50), 0.5)
        y ~ Normal(mu, 1.0)
    end
    levelled_sb = sbbrmi(levelled)
    subset = reprocess(levelled_sb, (; patch=[3, 1], y=[0.0, 0.0]))
    @test subset.data[:patch_idx] == [3, 1]
    @test subset.data[:patch_n_levels] == 3
    @test stan(subset) == stan(levelled_sb)
    @test occursin("[1.0, 1.0, 0.5]'", stan(subset))

    # The same holds for a treatment block whose prior is a generated
    # per-contrast family: its length is the fitted K-1, not the replayed one.
    heavy = @brm df begin
        mu ~ 1 + factor(patch)
        effect(mu, patch) ~ Cauchy(0, 1)
        y ~ Normal(mu, 1.0)
    end
    heavy_sb = sbbrmi(heavy)
    @test stan(reprocess(heavy_sb, (; patch=[2, 2], y=[0.0, 0.0]))) == stan(heavy_sb)

    # A reusable generative plan is a FRESH build on the new frame (new
    # population): levels, and with them the level addresses, come from that
    # frame. A frame without the addressed level refuses loudly and names the
    # addresses that do exist, rather than re-pointing the prior.
    builder = @brm begin
        mu ~ 0 + factor(patch)
        effect(mu, patch_lvl_3) ~ Normal(log(50), 0.5)
        y ~ Normal(mu, 1.0)
    end
    plan = generative_plan(builder, df; mod=@__MODULE__, total_groups=())
    rebuilt = generative_plan(plan, (; patch=[3, 1, 2], y=[0.0, 0.0, 0.0]))
    @test rebuilt.data[:patch_idx] == [3, 1, 2]
    @test rebuilt.data[:patch_n_levels] == 3
    @test_throws "`patch_lvl_1`, `patch_lvl_2`" generative_plan(
        plan, (; patch=[3, 1], y=[0.0, 0.0]))
end

@testset "BridgeStan: K coordinates, exact cells, density and gradient" begin
    if CELLS_RUNTIME
        df = cells_df()
        cells = @brm df begin
            mu ~ 0 + factor(patch)
            effect(mu, patch_lvl_3) ~ Normal(log(50), 0.5)
            y ~ Normal(mu, 1.0)
        end
        sb = sbbrmi(cells)
        code = stan(sb)
        isdir(CELLS_CACHE) || mkpath(CELLS_CACHE)
        problem = StanBlocks.stan_instantiate(
            sb.model; path=joinpath(CELLS_CACHE, string(hash(code)) * ".stan"))
        sm = problem.model
        @test LogDensityProblems.dimension(problem) == 3
        names = String.(BS.param_names(sm))
        beta_i = [only(findall(==("cat_mu_patch_beta.$k"), names)) for k in 1:3]

        q = zeros(3)
        q[beta_i] .= [0.3, -0.7, 4.0]
        constrained_names = BS.param_names(sm; include_tp=true, include_gq=false)
        constrained = BS.param_constrain(sm, q; include_tp=true, include_gq=false)
        mu = [v for (nm, v) in zip(constrained_names, constrained)
              if startswith(String(nm), "mu.")]
        # Every row reads its own level's cell mean; no level is pinned at zero.
        @test mu ≈ [0.3, -0.7, 4.0, 0.3, -0.7, 4.0]

        lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
        @test isfinite(lp)
        @test all(isfinite, gradient)
        # The level-3 prior is Normal(log 50, 0.5): moving that one coordinate
        # to its prior mean changes the density by exactly the two kernels.
        q2 = copy(q); q2[beta_i[3]] = log(50)
        lp2 = LogDensityProblems.logdensity(problem, q2)
        kernel(b) = -0.5 * ((b - log(50)) / 0.5)^2 -
                    0.5 * sum(abs2, df.y[[3, 6]] .- b)
        @test lp2 - lp ≈ kernel(log(50)) - kernel(4.0)

        resolved = brm_population_effect_coordinates(
            brm_descriptor(sb), :mu, constrained_names; coefficient=:patch)
        @test resolved.coding === :cellmeans
        @test isnothing(resolved.reference_level)
        @test resolved.nonreference_levels == [1, 2, 3]
        @test isempty(resolved.contrasts)
        @test [c.level for c in resolved.cells] == [1, 2, 3]
        @test String.(constrained_names[[c.coordinate for c in resolved.cells]]) ==
              ["cat_mu_patch_beta.1", "cat_mu_patch_beta.2", "cat_mu_patch_beta.3"]

        treated = @brm df begin
            mu ~ 1 + factor(patch)
            y ~ Normal(mu, 1.0)
        end
        tsb = sbbrmi(treated)
        tcode = stan(tsb)
        tproblem = StanBlocks.stan_instantiate(
            tsb.model; path=joinpath(CELLS_CACHE, string(hash(tcode)) * ".stan"))
        tnames = BS.param_names(tproblem.model; include_tp=true, include_gq=false)
        tresolved = brm_population_effect_coordinates(
            brm_descriptor(tsb), :mu, tnames; coefficient=:patch)
        @test tresolved.coding === :treatment
        @test tresolved.reference_level == 1
        @test tresolved.nonreference_levels == [2, 3]
        @test isempty(tresolved.cells)
        @test length(tresolved.contrasts) == 2
    else
        @info "Skipping BridgeStan cell-means runtime gate (BRM_CELLS_RUNTIME=0)"
        @test true
    end
end


# ---- random-effect blocks (decision `0wfo466`) --------------------------------

ranef_df() = (; c=repeat([1, 2, 3], 4), g=repeat([1, 2, 3, 4]; inner=3),
                x=collect(range(-1.0, 1.0; length=12)),
                y=[-2.4, -2.2, -2.0, -1.8, -1.7, -1.5, -1.2, -1.0, -0.9, -0.6, -0.4, -0.1])

# The right-hand side of the one `Z_<lp>_<suffix> = hcat(...)` statement.
zcols(code, z) = only(
    String(strip(last(split(strip(l), " = "; limit=2)), ';'))
    for l in split(code, '\n') if occursin(" $z = hcat(", l))

@testset "random effects: an intercept-free LHS codes its first factor per level" begin
    df = ranef_df()
    cells = @brm df begin
        mu ~ 1 + (0 + c | g)
        y ~ Normal(mu, 1.0)
    end
    sb = sbbrmi(cells)
    code = stan(sb)
    @test zcols(code, "Z_mu_g") == "hcat(c_dummy_1, c_dummy_2, c_dummy_3)"
    @test sb.data[:n_terms_mu_g] == 3
    @test sb.data[:c_dummy_1] == Float64.(df.c .== 1)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok

    # An intercept keeps the K-1 dummies -- including one contributed by a
    # SIBLING term of the same block, since BRM merges `(1 | g) + (0 + c | g)`.
    treated = @brm df begin
        mu ~ 1 + (1 + c | g)
        y ~ Normal(mu, 1.0)
    end
    @test endswith(zcols(stan(treated), "Z_mu_g"), ", c_dummy_2, c_dummy_3)")
    @test !haskey(sbbrmi(treated).data, :c_dummy_1)
    merged = @brm df begin
        mu ~ 1 + (1 | g) + (0 + c | g)
        y ~ Normal(mu, 1.0)
    end
    @test endswith(zcols(stan(merged), "Z_mu_g"), ", c_dummy_2, c_dummy_3)")
    @test !haskey(sbbrmi(merged).data, :c_dummy_1)

    # `cmc=false` opts out here too; a continuous slope does not count as the
    # first FACTOR.
    pinned = @brm df begin
        mu ~ 1 + (0 + factor(c; cmc=false) | g)
        y ~ Normal(mu, 1.0)
    end
    @test zcols(stan(pinned), "Z_mu_g") == "hcat(c_dummy_2, c_dummy_3)"
    sloped = @brm df begin
        mu ~ 1 + (0 + x + c | g)
        y ~ Normal(mu, 1.0)
    end
    @test zcols(stan(sloped), "Z_mu_g") == "hcat(x, c_dummy_1, c_dummy_2, c_dummy_3)"

    # Zero-correlation `||` splits the LHS into single-term blocks; the decision
    # still belongs to the ORIGINAL left-hand side.
    free = @brm df begin
        mu ~ 1 + (0 + c || g)
        y ~ Normal(mu, 1.0)
    end
    @test zcols(stan(free), "Z_mu_g__nocor__1") == "hcat(c_dummy_1, c_dummy_2, c_dummy_3)"
    withint = @brm df begin
        mu ~ 1 + (1 + c || g)
        y ~ Normal(mu, 1.0)
    end
    @test zcols(stan(withint), "Z_mu_g__nocor__2") == "hcat(c_dummy_2, c_dummy_3)"
end

@testset "random effects: shared |ID| buckets size, name and address every level" begin
    df = ranef_df()
    shared = @brm df begin
        mu ~ 1 + (0 + c | p | g)
        sd(mu, p, c_dummy_1) ~ Exponential(0.5)
        y ~ Normal(mu, 1.0)
    end
    @test ranefcoefnames(shared, :p) == [
        (predictor=:mu, coefficient=:c_dummy_1),
        (predictor=:mu, coefficient=:c_dummy_2),
        (predictor=:mu, coefficient=:c_dummy_3)]
    sb = sbbrmi(shared)
    code = stan(sb)
    @test sb.data[:n_terms_p_g] == 3
    @test zcols(code, "Z_mu_p_g") == "hcat(c_dummy_1, c_dummy_2, c_dummy_3)"
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok

    treated = @brm df begin
        mu ~ 1 + (1 + c | p | g)
        y ~ Normal(mu, 1.0)
    end
    @test ranefcoefnames(treated, :p) == [
        (predictor=:mu, coefficient=:Intercept),
        (predictor=:mu, coefficient=:c_dummy_2),
        (predictor=:mu, coefficient=:c_dummy_3)]
    @test sbbrmi(treated).data[:n_terms_p_g] == 3
end

@testset "random effects: replay keeps every fitted level column" begin
    df = ranef_df()
    sb = sbbrmi(@brm df begin
        mu ~ 1 + (0 + c | g)
        y ~ Normal(mu, 1.0)
    end)
    subset = (; c=[3, 1], g=[2, 4], x=[0.0, 0.0], y=[0.0, 0.0])
    replay = reprocess(sb, subset)
    @test replay.data[:c_dummy_1] == [0.0, 1.0]
    @test replay.data[:c_dummy_2] == [0.0, 0.0]
    @test replay.data[:c_dummy_3] == [1.0, 0.0]
    @test stan(replay) == stan(sb)
    # A frozen RE-EMISSION for a new population carries only some levels too.
    resampled = reprocess(sb, subset; freeze_constants=true, resample_groups=[:g])
    @test zcols(stan(resampled), "Z_mu_g") == "hcat(c_dummy_1, c_dummy_2, c_dummy_3)"
    @test resampled.data[:n_terms_mu_g] == 3
    @test_throws "not a training level" reprocess(
        sb, (; c=[1, 9], g=[1, 2], x=[0.0, 0.0], y=[0.0, 0.0]))
end

@testset "random effects, BridgeStan: level-1 rows carry a group effect" begin
    if CELLS_RUNTIME
        df = ranef_df()
        sb = sbbrmi(@brm df begin
            mu ~ 1 + (0 + c | g)
            y ~ Normal(mu, 1.0)
        end)
        code = stan(sb)
        isdir(CELLS_CACHE) || mkpath(CELLS_CACHE)
        problem = StanBlocks.stan_instantiate(
            sb.model; path=joinpath(CELLS_CACHE, string(hash(code)) * ".stan"))
        sm = problem.model
        # UNCONSTRAINED names index `q`: the 3 x 3 Cholesky factor has 9
        # constrained entries but 3 free coordinates.
        names = String.(BS.param_unc_names(sm))
        # intercept + tau[3] + L (3 free) + z[3 x 4]
        @test LogDensityProblems.dimension(problem) == 1 + 3 + 3 + 12
        q = zeros(LogDensityProblems.dimension(problem))
        z_i = findall(startswith("r_mu_g_z"), names)
        @test length(z_i) == 12
        q[z_i] .= 1.0
        constrained_names = BS.param_names(sm; include_tp=true, include_gq=false)
        constrained = BS.param_constrain(sm, q; include_tp=true, include_gq=false)
        r = [v for (nm, v) in zip(constrained_names, constrained)
             if startswith(String(nm), "r_mu_g.")]
        # tau = 1, L = I, z = 1 => every per-level group effect is 1, and every
        # row reads exactly one of them -- level-1 rows included.
        @test r ≈ ones(12)
        lp, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
        @test isfinite(lp)
        @test all(isfinite, gradient)
    else
        @info "Skipping BridgeStan random-effect runtime gate (BRM_CELLS_RUNTIME=0)"
        @test true
    end
end
