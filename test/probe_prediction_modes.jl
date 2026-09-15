# test/probe_prediction_modes.jl — acceptance for src/prediction.jl: the two
# post-fit prediction modes (population-level "nore" and transported "recov").
#
# WHY THE SEMANTIC CHECKS ARE THE LOAD-BEARING ONES. It is easy to write a
# `nore` that zeroes *some* coordinates and looks right: the draw matrix has the
# expected shape, nothing errors, and the numbers are plausible. What proves it
# is reading the model's OWN transformed parameters back out of BridgeStan and
# asserting the per-row random-effect contributions `r_<lhs>_<g>` are EXACTLY
# zero — that is the model agreeing that it is at the population mean, rather
# than this file agreeing with itself about an index layout.
#
# Likewise for `recov`: the check that matters is not "the matrix has the right
# width" but that a transported draw evaluated on the NEW model reproduces the
# fitted model's log density when the new data IS the old data, and that shared
# coordinates survive a deliberately PERMUTED level order — which is exactly the
# case a positional splice gets silently wrong.
#
# Run: julia --project=<worktree> test/probe_prediction_modes.jl
# (needs BridgeStan reachable through StanBlocks; set BRM_PRED_RUNTIME=0 to run
# only the declaration-level half.)

using Test
using BayesianRegressionModels
using StanBlocks
using Random
using Distributions: Exponential, Normal

const PRED_RUNTIME = get(ENV, "BRM_PRED_RUNTIME", "1") != "0"
const BS = StanBlocks.BridgeStan

# ---- fixtures ---------------------------------------------------------------

# One `(… |p| g)` bucket shared by two sub-formulas, one plain correlated ranef
# on the same factor, and one plain intercept ranef on a second factor. That is
# every recognised non-stratified family in one model, so the block table is
# exercised against a single emission rather than three toy ones.
mixed_builder = @brm begin
    sigma  ~ Exponential(1)
    eta_CL ~ 0 + x + (1 | p | subject)
    eta_V  ~ 0 + x + (1 | p | subject)
    mu     ~ 1 + t + (1 | site) + (1 + x | subject)
    y      ~ Normal(mu + eta_CL + eta_V, sigma)
end

# `subjects` are LABELS, so a replay frame can introduce new ones and permute
# the old ones — the two things a positional splice cannot survive.
function mixed_df(subjects; seed = 1, n_per = 4)
    rng = MersenneTwister(seed)
    n = length(subjects) * n_per
    (; subject = repeat(subjects; inner = n_per),
       site    = repeat([10, 20]; outer = n ÷ 2),
       x       = randn(rng, n),
       t       = repeat(collect(1.0:n_per), length(subjects)),
       y       = randn(rng, n))
end

train_subjects = [11, 12, 13, 14]
train_df = mixed_df(train_subjects)
train_sb = SBBRMI(mixed_builder(train_df); mod = @__MODULE__)

# ---- 1. block description ---------------------------------------------------

@testset "ranef_blocks — description of the emitted blocks" begin
    blocks = ranef_blocks(train_sb)
    @test length(blocks) == 3
    by_binding = Dict(b.binding => b for b in blocks)

    bucket = by_binding[:b_p_subject]
    @test bucket.family === :ranef_correlated_draws
    @test bucket.group === :subject
    @test bucket.id === :p                     # the brms-style |ID| bucket
    @test bucket.by === nothing
    @test bucket.levels == train_subjects      # index order == sort(unique(raw))
    @test (bucket.n_terms, bucket.n_groups) == (2, 4)
    @test bucket.z === :b_p_subject_z_flat     # the standardised draw, not `b`
    @test bucket.noncentered

    plain = by_binding[:r_mu_subject]
    @test plain.family === :ranef_correlated
    @test plain.id === nothing                 # not a bucket
    @test (plain.n_terms, plain.n_groups) == (2, 4)
    @test plain.z === :r_mu_subject_z_flat

    site = by_binding[:r_mu_site]
    @test site.family === :ranef_intercept
    @test site.group === :site
    @test (site.n_terms, site.n_groups) == (1, 2)
    @test site.z === :r_mu_site_xi
    @test site.levels == [10, 20]

    # A plan is accepted wherever an SBBRMI is.
    @test [b.binding for b in ranef_blocks(generative_plan(train_sb))] ==
          [b.binding for b in blocks]
end

# `_RANEF_FAMILIES` is a hand-maintained mirror of the `@slic` submodels, and it
# has drifted once: 2ffac3c collapsed `ranef_correlated_draws` from a plate
# (`b_cols_z`) to a flat reshape (`z_flat`) without updating the table, which
# made `ranef_coordinates` error on every bucket model. That drift was only
# reachable by compiling a model — §3 below catches it, but §3 needs BridgeStan
# and was not run before the landing. This check needs neither: the emitted Stan
# SOURCE names the parameter, so declaring the wrong one is caught here.
@testset "block z names exist in the emitted Stan source" begin
    code = StanBlocks.stan_code(train_sb.model)
    for b in ranef_blocks(train_sb)
        @test occursin(string(b.z), code)
    end
    # The check discriminates: a name of the right shape that is NOT emitted
    # must fail it, otherwise `occursin` is passing on a substring accident.
    @test !occursin("b_p_subject_b_cols_z", code)
end

# The check above can only reach families that `ranef_blocks` SUCCEEDS on, so an
# emission with no table row is invisible to it — `ranef_blocks` raises before
# the loop ever sees the block. That is the shape of the drift, so close it
# directly: compare the table's keys against the `ranef_*` submodels the package
# actually exports. Needs no model, no data and no BridgeStan.
@testset "_RANEF_FAMILIES covers every emitted ranef_* submodel" begin
    emitted = Set(nm for nm in names(BayesianRegressionModels; all = true)
                  if startswith(String(nm), "ranef_") &&
                     isdefined(BayesianRegressionModels, nm) &&
                     getproperty(BayesianRegressionModels, nm) isa StanBlocks.SlicModel)
    tabled = Set(keys(BayesianRegressionModels._RANEF_FAMILIES))
    @test isempty(setdiff(emitted, tabled))   # an emission with no row
    @test isempty(setdiff(tabled, emitted))   # a row for a submodel that is gone
    # Pin the count too: both setdiffs stay empty if a submodel and its row are
    # deleted together, which is fine, but a silent DROP should still be read.
    @test length(emitted) == 15 # twelve established layouts + generic centered/noncentered + scalar slope
end

# The centered families are the reason `noncentered` is reported rather than
# assumed. `centered_groups=` ships, so a consumer may be handed one. Assert the
# designed split: DESCRIBED by `ranef_blocks` (you can list it and read its
# shape), REFUSED by the operations (the coordinate is the effect itself, so
# zeroing it would silently mean something else).
@testset "centered emissions — described, then refused at the operation" begin
    centered_int_builder = @brm begin
        sigma ~ Exponential(1)
        mu    ~ 1 + x + (1 | subject)
        y     ~ Normal(mu, sigma)
    end
    centered_corr_builder = @brm begin
        sigma ~ Exponential(1)
        mu    ~ 1 + x + (1 + x + t | subject)
        y     ~ Normal(mu, sigma)
    end
    centered_bucket_builder = @brm begin
        sigma  ~ Exponential(1)
        eta_CL ~ 0 + x + (1 | p | subject)
        eta_V  ~ 0 + x + (1 | p | subject)
        eta_Q  ~ 0 + x + (1 | p | subject)
        y      ~ Normal(eta_CL + eta_V + eta_Q, sigma)
    end

    # n_terms = 3 and n_groups = 2 throughout, so `.g.t` and `.t.g` are
    # DISTINGUISHABLE — a square fixture cannot discriminate the two layouts.
    cdf = mixed_df([11, 12])
    for (builder, family, binding, z, n_terms) in (
            (centered_int_builder,    :ranef_intercept_centered,        :r_mu_subject, :r_mu_subject_xi, 1),
            (centered_corr_builder,   :ranef_correlated_centered,       :r_mu_subject, :r_mu_subject_b,  3),
            (centered_bucket_builder, :ranef_correlated_draws_centered, :b_p_subject,  :b_p_subject_b,   3),
        )
        csb = SBBRMI(builder(cdf); mod = @__MODULE__, centered_groups = [:subject])
        blocks = ranef_blocks(csb)                  # DESCRIBES — must not raise
        b = only(filter(bb -> bb.binding === binding, blocks))
        @test b.family === family
        @test b.z === z
        @test b.n_terms == n_terms
        @test b.n_groups == 2
        @test !b.noncentered
        @test occursin(string(b.z), StanBlocks.stan_code(csb.model))

        # ...and REFUSES, naming the block and why.
        err = try
            population_draws(csb, zeros(1, 2), ["a", "b"]; groups = :subject)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("NON-CENTERED", err.msg)
        @test occursin(string(binding), err.msg)
    end
end

@testset "ranef_blocks — cv and stratified emissions" begin
    cv_builder = @brm begin
        sigma ~ Exponential(1)
        mu    ~ 1 + x + (1 + x | subject)
        y     ~ Normal(mu, sigma)
    end
    cv_sb = SBBRMI(cv_builder(train_df); mod = @__MODULE__, cv_groups = [:subject])
    cv = only(ranef_blocks(cv_sb))
    # cv is NOT a distinct family any more — same submodel, different size
    # expression at the call site. So the block description is identical to the
    # ordinary build's, and what has to be asserted is the emitted SIZE.
    @test cv.family === :ranef_correlated
    @test cv.z === :r_mu_subject_z_flat
    @test (cv.n_terms, cv.n_groups) == (2, 4)  # sized from maximum(group_idx)
    cv_code = StanBlocks.stan_code(cv_sb.model)
    @test occursin("r_mu_subject_n_g = max(subject_idx)", cv_code)
    plain_code = StanBlocks.stan_code(
        SBBRMI(cv_builder(train_df); mod = @__MODULE__).model)
    @test !occursin("max(subject_idx)", plain_code)   # the default keeps n_subject

    # REGRESSION. A cv-sized block's `n_groups=` names a Stan-side local
    # (`<binding>_n_g = maximum(<g>_idx)`), NOT a data key — that indirection is
    # what carries the taint. `ranef_blocks` used to look it up in `data` and
    # error out. The `(… |ID| g)` bucket path has passed such a local since
    # 2ffac3c, so `ranef_blocks` errored on every cv bucket model; no test
    # covered it because the only cv fixture was a plain ranef, which still
    # sized from `n_<g>` back then. Both paths are covered here now.
    bucket_cv_sb = SBBRMI(mixed_builder(train_df); mod = @__MODULE__,
                          cv_groups = [:subject])
    bucket_cv = only(filter(b -> b.id === :p, ranef_blocks(bucket_cv_sb)))
    @test bucket_cv.binding === :b_p_subject
    @test bucket_cv.n_groups == 4                      # from maximum(subject_idx)
    @test bucket_cv.z === :b_p_subject_z_flat

    int_cv_sb = SBBRMI((@brm begin
        sigma ~ Exponential(1)
        mu    ~ 1 + x + (1 | subject)
        y     ~ Normal(mu, sigma)
    end)(train_df); mod = @__MODULE__, cv_groups = [:subject])
    int_cv = only(ranef_blocks(int_cv_sb))
    @test int_cv.family === :ranef_intercept
    @test int_cv.z === :r_mu_subject_xi
    @test occursin("r_mu_subject_n_g = max(subject_idx)",
                   StanBlocks.stan_code(int_cv_sb.model))

    by_df = merge(train_df, (; stratum = repeat([1, 2]; inner = length(train_df.x) ÷ 2)))
    by_sb = SBBRMI((@brm begin
        sigma ~ Exponential(1)
        mu    ~ 1 + x + (1 + x | gr(subject, by = stratum))
        y     ~ Normal(mu, sigma)
    end)(by_df); mod = @__MODULE__)
    st = only(ranef_blocks(by_sb))
    @test st.family === :ranef_correlated_by
    @test st.group === :subject
    @test st.by === :stratum
    @test st.z === :r_mu_subject__by__stratum_z
end

# ---- 2. selector and shape guards (no runtime needed) -----------------------

@testset "loud edges" begin
    blocks = ranef_blocks(train_sb)
    fake = ["sigma", "pop_mu_beta_pop.1"]
    # A block whose coordinates are absent must NOT silently resolve.
    @test_throws ErrorException ranef_coordinates(blocks[1], fake)
    # A grouping factor with no block is a typo, not a no-op.
    @test_throws ErrorException population_draws(train_sb, zeros(2, 2),
                                                 ["a", "b"]; groups = :nope)
    # draws/unc_names shape mismatch is caught before any indexing happens.
    @test_throws ErrorException population_draws(train_sb, zeros(2, 3),
                                                 ["a", "b"]; groups = :subject)
end

if !PRED_RUNTIME
    @info "BRM_PRED_RUNTIME=0 — skipping the compiled-model half"
else

# ---- 3. coordinates against the real compiled model -------------------------

train_p = StanBlocks.stan_instantiate(train_sb.model)
train_sm = train_p.model
unc_train = BS.param_unc_names(train_sm)
blocks = ranef_blocks(train_sb)

@testset "ranef_coordinates — every coordinate resolved by name" begin
    all_coords = Int[]
    for b in blocks
        c = ranef_coordinates(b, unc_train)
        @test size(c) == (b.n_terms, b.n_groups)
        append!(all_coords, vec(c))
    end
    @test length(unique(all_coords)) == length(all_coords)   # no block overlaps
    # Exactly the standardised-draw coordinates, and nothing else: the L / tau
    # hyperparameters must NOT be claimed by a block.
    named = Set(unc_train[all_coords])
    @test all(n -> occursin("_z.", n) || occursin("_xi.", n) ||
                   occursin("_z_flat.", n), named)
    @test !any(n -> occursin("_tau.", n) || occursin("_L.", n), named)
    @test length(all_coords) == 2 * 4 + 2 * 4 + 2        # bucket + plain + site
end

# ---- 4. population_draws (nore) — the model's own answer ---------------------

# Read a transformed-parameter block out of BridgeStan by name, for one draw.
function constrained_by_name(sm, theta_unc)
    names = BS.param_names(sm; include_tp = true, include_gq = false)
    vals = BS.param_constrain(sm, theta_unc; include_tp = true, include_gq = false)
    Dict(n => v for (n, v) in zip(names, vals))
end

@testset "population_draws — the model reports zero random effects" begin
    rng = MersenneTwister(7)
    n_unc = BS.param_unc_num(train_sm)
    draws = randn(rng, 5, n_unc)

    pop = population_draws(train_sb, draws, unc_train; groups = :subject)
    @test size(pop) == size(draws)

    subj_coords = reduce(vcat,
        vec(ranef_coordinates(b, unc_train)) for b in blocks if b.group === :subject)
    site_coords = vec(ranef_coordinates(
        only(filter(b -> b.group === :site, blocks)), unc_train))
    @test all(iszero, pop[:, subj_coords])
    # Untouched: the other factor, and the hyperparameters of the zeroed blocks.
    @test pop[:, site_coords] == draws[:, site_coords]
    keep = setdiff(1:n_unc, subj_coords)
    @test pop[:, keep] == draws[:, keep]

    # THE SEMANTIC CHECK: the model itself says the per-row subject
    # contributions are exactly zero, and the site contribution is not.
    cons = constrained_by_name(train_sm, pop[1, :])
    for k in keys(cons)
        if startswith(k, "r_eta_CL_p_subject.") || startswith(k, "r_eta_V_p_subject.") ||
           startswith(k, "r_mu_subject.")
            @test cons[k] == 0.0
        end
    end
    @test any(v -> v != 0.0,
              [v for (k, v) in cons if startswith(k, "r_mu_site.")])
    # Zeroing a random effect must not disturb the fixed part.
    ref = constrained_by_name(train_sm, draws[1, :])
    @test all(k -> cons[k] == ref[k],
              [k for k in keys(cons) if startswith(k, "pop_mu.")])

    # Both factors at once.
    both = population_draws(train_sb, draws, unc_train; groups = (:subject, :site))
    cons2 = constrained_by_name(train_sm, both[1, :])
    @test all(v -> v == 0.0, [v for (k, v) in cons2 if startswith(k, "r_mu_site.")])
end

# ---- 5. transport_draws (recov) ---------------------------------------------

@testset "transport_draws — identity replay reproduces the fit exactly" begin
    rng = MersenneTwister(11)
    draws = randn(rng, 4, BS.param_unc_num(train_sm))
    same_plan = generative_plan(mixed_builder, train_df; mod = @__MODULE__)
    same_p = StanBlocks.stan_instantiate(same_plan.model)
    unc_same = BS.param_unc_names(same_p.model)

    moved = transport_draws(train_sb, same_plan, draws, unc_train, unc_same)
    @test size(moved) == size(draws)
    # Same declaration, same data ⇒ nothing is drawn fresh and the log density
    # is reproduced bit for bit. This is the end-to-end statement that the
    # by-name transport is an identity when it should be.
    for i in 1:size(draws, 1)
        @test BS.log_density(same_p.model, moved[i, :]) ≈
              BS.log_density(train_sm, draws[i, :])
    end
end

@testset "transport_draws — a shifted level index aligns by LABEL, not position" begin
    rng = MersenneTwister(13)
    draws = randn(rng, 3, BS.param_unc_num(train_sm))
    # THE CASE A POSITIONAL SPLICE GETS SILENTLY WRONG. Training levels are
    # [11, 12, 13, 14]; here a subject sorting BEFORE all of them is introduced
    # and two are dropped, so the retained subject 11 moves from block column 1
    # to column 2. Copying column-for-column would hand subject 11 subject 12's
    # effect and never error.
    shifted_subjects = [1, 11, 13]
    shifted_df = mixed_df(shifted_subjects; seed = 2)
    shifted_plan = generative_plan(mixed_builder, shifted_df; mod = @__MODULE__)
    shifted_p = StanBlocks.stan_instantiate(shifted_plan.model)
    unc_shifted = BS.param_unc_names(shifted_p.model)

    moved = transport_draws(train_sb, shifted_plan, draws, unc_train, unc_shifted;
                            rng = MersenneTwister(31))
    for b in ranef_blocks(shifted_plan)
        b.group === :subject || continue
        bt = only(filter(x -> x.binding === b.binding, ranef_blocks(train_sb)))
        @test b.levels == [1, 11, 13]           # the emitter's sorted order
        c_to = ranef_coordinates(b, unc_shifted)
        c_from = ranef_coordinates(bt, unc_train)
        # Subject 11 is at target column 2 and source column 1 — the retained
        # levels must follow their LABEL across that shift.
        @test moved[:, c_to[:, 2]] == draws[:, c_from[:, 1]]   # 11
        @test moved[:, c_to[:, 3]] == draws[:, c_from[:, 3]]   # 13
        # …and the positional answer for subject 11 must NOT be what we got.
        @test moved[:, c_to[:, 2]] != draws[:, c_from[:, 2]]   # would be 12
    end
end

@testset "transport_draws — new subjects are drawn fresh, old ones reused" begin
    rng = MersenneTwister(17)
    draws = randn(rng, 6, BS.param_unc_num(train_sm))
    # Two of the training subjects, plus two genuinely new ones.
    new_subjects = [12, 13, 90, 91]
    new_df = mixed_df(new_subjects; seed = 5)
    new_plan = generative_plan(mixed_builder, new_df; mod = @__MODULE__)
    new_p = StanBlocks.stan_instantiate(new_plan.model)
    unc_new = BS.param_unc_names(new_p.model)

    moved = transport_draws(train_sb, new_plan, draws, unc_train, unc_new;
                            rng = MersenneTwister(3))
    @test size(moved) == (6, length(unc_new))
    @test all(isfinite, moved)

    for b in ranef_blocks(new_plan)
        bt = only(filter(x -> x.binding === b.binding, ranef_blocks(train_sb)))
        c_to = ranef_coordinates(b, unc_new)
        c_from = ranef_coordinates(bt, unc_train)
        for (g, lvl) in enumerate(b.levels)
            gf = findfirst(==(lvl), bt.levels)
            if isnothing(gf)
                # A new level: fresh N(0,1), so it cannot equal any source column.
                @test all(t -> !any(gg -> moved[:, c_to[t, g]] == draws[:, c_from[t, gg]],
                                    1:bt.n_groups), 1:b.n_terms)
            else
                @test moved[:, c_to[:, g]] == draws[:, c_from[:, gf]]
            end
        end
    end

    # Non-ranef coordinates (pop coefficients, sigma, L, tau) are carried over
    # verbatim — that is what makes a fresh z a draw from the FITTED covariance.
    claimed = Set(reduce(vcat,
        vec(ranef_coordinates(b, unc_new)) for b in ranef_blocks(new_plan)))
    pos_train = Dict(String(n) => i for (i, n) in enumerate(unc_train))
    for (j, nm) in enumerate(unc_new)
        j in claimed && continue
        @test moved[:, j] == draws[:, pos_train[String(nm)]]
    end

    # The transported draws evaluate on the new model.
    @test all(i -> isfinite(BS.log_density(new_p.model, moved[i, :])), 1:size(moved, 1))

    # `resample=` additionally re-draws the EXISTING levels of that factor.
    loo = transport_draws(train_sb, new_plan, draws, unc_train, unc_new;
                          resample = :subject, rng = MersenneTwister(3))
    for b in ranef_blocks(new_plan)
        b.group === :subject || continue
        bt = only(filter(x -> x.binding === b.binding, ranef_blocks(train_sb)))
        c_to = ranef_coordinates(b, unc_new)
        c_from = ranef_coordinates(bt, unc_train)
        for (g, lvl) in enumerate(b.levels)
            gf = findfirst(==(lvl), bt.levels)
            isnothing(gf) && continue
            @test moved[:, c_to[:, g]] != loo[:, c_to[:, g]]
        end
    end
    # The site factor was NOT resampled, so it is untouched by `resample=:subject`.
    site_b = only(filter(b -> b.group === :site, ranef_blocks(new_plan)))
    c_site = vec(ranef_coordinates(site_b, unc_new))
    @test loo[:, c_site] == moved[:, c_site]
    @test_throws ErrorException transport_draws(train_sb, new_plan, draws,
                                                unc_train, unc_new; resample = :nope)
end

@testset "transport_draws — a changed design is refused, not spliced" begin
    rng = MersenneTwister(19)
    draws = randn(rng, 2, BS.param_unc_num(train_sm))
    # Same data, but `mu` gains a covariate: `pop_mu_beta_pop` grows a
    # coordinate the source fit has no value for. A positional splice would
    # shift every later coordinate; this must refuse.
    wider_builder = @brm begin
        sigma  ~ Exponential(1)
        eta_CL ~ 0 + x + (1 | p | subject)
        eta_V  ~ 0 + x + (1 | p | subject)
        mu     ~ 1 + t + x + (1 | site) + (1 + x | subject)
        y      ~ Normal(mu + eta_CL + eta_V, sigma)
    end
    wider = generative_plan(wider_builder, train_df; mod = @__MODULE__)
    wider_p = StanBlocks.stan_instantiate(wider.model)
    @test_throws ErrorException transport_draws(
        train_sb, wider, draws, unc_train, BS.param_unc_names(wider_p.model))

    # And a target that gained a whole random-effect block is refused too.
    extra_builder = @brm begin
        sigma  ~ Exponential(1)
        eta_CL ~ 0 + x + (1 | p | subject)
        eta_V  ~ 0 + x + (1 | p | subject)
        mu     ~ 1 + t + (1 | site) + (1 + x | subject)
        y      ~ Normal(mu + eta_CL + eta_V, sigma)
    end
    simpler = SBBRMI((@brm begin
        sigma ~ Exponential(1)
        mu    ~ 1 + t + (1 | site)
        y     ~ Normal(mu, sigma)
    end)(train_df); mod = @__MODULE__)
    simpler_p = StanBlocks.stan_instantiate(simpler.model)
    unc_simpler = BS.param_unc_names(simpler_p.model)
    extra = generative_plan(extra_builder, train_df; mod = @__MODULE__)
    extra_p = StanBlocks.stan_instantiate(extra.model)
    @test_throws ErrorException transport_draws(
        simpler, extra, randn(MersenneTwister(23), 2, length(unc_simpler)),
        unc_simpler, BS.param_unc_names(extra_p.model))
end

end # PRED_RUNTIME
