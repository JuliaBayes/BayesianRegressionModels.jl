# test/resample_effect_prior_labels.jl — a frozen resample replay sizes every
# population-coefficient prior from the FITTED design (snag
# `reprocess-resamp-0a765fc0`).
#
# `reprocess(fit, frame; freeze_constants=true, resample_groups=[g])` re-emits
# the model on the replay frame. Its design columns come from the fitted
# records (a `mo(c)` keeps its fitted level set, a string column its fitted
# level columns), but the prior resolvers named those columns with
# `popcoefnames`, which re-derived them from the replay rows. A frame holding
# one level of `c` dropped the `mo(c)` column from the labels only, so every
# `effect(lp, …)` / `r2d2` / Horseshoe prior on that predictor failed
# `internal … alignment error: n priors for n+1 columns`.
#
# Each case below replays a one-level frame and must emit exactly the Stan
# program an all-levels frame does; plain frozen replay keeps the fitted source.
#
# Run: julia --startup-file=no --project=test test/resample_effect_prior_labels.jl
# (set BRM_PRED_RUNTIME=0 to skip the BridgeStan half)

using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Normal, Beta

const BRM = BayesianRegressionModels
const BS = StanBlocks.BridgeStan
const PRED_RUNTIME = get(ENV, "BRM_PRED_RUNTIME", "1") != "0"

train = (; subject=repeat(1:6, inner=2), c=repeat([1, 2, 3], outer=4),
           h=repeat([1, 2, 3], inner=4), a=repeat([1.0, 0.0, 0.0], outer=4),
           b=repeat([0.0, 1.0], outer=6), x=collect(range(-1, 1; length=12)),
           g=repeat(["lo", "mid", "hi"], outer=4), y=zeros(12))
# A prediction schedule in which every row carries one level of each
# categorical, for new subjects.
one_level = (; subject=repeat([101, 102], inner=2), c=fill(1, 4), h=fill(2, 4),
               a=fill(1.0, 4), b=zeros(4), x=[-0.5, 0.0, 0.5, 1.0],
               g=fill("lo", 4), y=zeros(4))
all_levels = (; subject=repeat([101, 102, 103], inner=2),
                c=repeat([1, 2, 3], outer=2), h=repeat([1, 2, 3], outer=2),
                a=fill(1.0, 6), b=zeros(6), x=collect(range(-0.5, 1; length=6)),
                g=repeat(["lo", "mid", "hi"], outer=2), y=zeros(6))
fitted_subjects = merge(one_level, (; subject=repeat([1, 2], inner=2)))

cases = [
    # The reported shape: no intercept, indicator columns, a monotonic diet
    # effect and an HSGP, with a default Normal prior over every column and a
    # more specific one on the slope.
    ("Normal effect prior, mo + hsgp", @brm(begin
        mu ~ 0 + a + b + mo(c) + x + hsgp(x; k=5, extrapolate=:clamp) + (1 | subject)
        effect(mu, :) ~ Normal(0.0, 0.5)
        effect(mu, x) ~ Normal(0.0, 0.6)
        y ~ Normal(mu, 1.0)
    end), (;)),
    ("Normal effect prior, string level columns", @brm(begin
        mu ~ 0 + g + x + (1 | subject)
        effect(mu, x) ~ Normal(0, 1)
        y ~ Normal(mu, 1.0)
    end), (;)),
    ("Horseshoe, mo", @brm(begin
        mu ~ 1 + a + mo(c) + x + (1 | subject)
        effect(mu, x) ~ Horseshoe()
        y ~ Normal(mu, 1.0)
    end), (;)),
    # The flat decomposition lives on a predictor without the resampled group:
    # a group carrying a derived R2D2 scale refuses `cv_groups` on its own, and
    # an intercept-only group predictor would be a total-coefficient block.
    ("flat r2d2, mo", @brm(begin
        mu ~ 1 + (1 | subject)
        log_s ~ 1 + a + mo(c) + x
        effect(log_s, :) ~ r2d2(R2=Beta(1, 1))
        y ~ Normal(mu, exp(log_s))
    end), (; total_groups=())),
    ("joint r2d2 over population and contrasts, mo", @brm(begin
        mu ~ 1 + h + mo(c) + x + (1 | p | subject)
        sd(:, p) ~ r2d2(R2=Beta(1, 1), include=(:population, :contrasts))
        y ~ Normal(mu, 1.0)
    end), (;)),
]

# Both R2D2 forms divide each scale by its design column's variance, which a
# replay recomputes from its own rows: the one-level frame's `mo(c)` column and
# contrast dummies are constant, so the replay density is -Inf until those
# variances are frozen (todo `0a51w6j`; plain frozen replay shares the gap).
density_gap = Set(["flat r2d2, mo", "joint r2d2 over population and contrasts, mo"])

stan(sb) = StanBlocks.stan_code(sb.model)
population_labels(sb) = Dict(o.name => o.labels for o in brm_descriptor(sb).outputs
                             if o.role === :population_effect && !isnothing(o.labels))

for (label, builder, kwargs) in cases
    @testset "$label" begin
        fit = SBBRMI(builder(train); mod=@__MODULE__, kwargs...)
        # Plain frozen replay reuses the fitted program.
        @test stan(reprocess(fit, fitted_subjects; freeze_constants=true)) == stan(fit)

        replay = reprocess(fit, one_level; freeze_constants=true,
                           resample_groups=[:subject])
        reference = reprocess(fit, all_levels; freeze_constants=true,
                              resample_groups=[:subject])
        @test stan(replay) == stan(reference)
        @test StanBlocks.stanc_check(stan(replay); warn_pedantic=false).ok
        # Post-fit consumers read the emitted design, so labels match too.
        @test population_labels(replay) == population_labels(reference) ==
              population_labels(fit)

        if PRED_RUNTIME
            p = StanBlocks.stan_instantiate(replay.model)
            x = zeros(BS.param_unc_num(p.model))
            ld, grad = BS.log_density_gradient(p.model, x)
            if label in density_gap
                @test_broken isfinite(ld)
            else
                @test isfinite(ld)
                @test all(isfinite, grad)
            end
        end
    end
end
