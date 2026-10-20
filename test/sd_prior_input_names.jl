# test/sd_prior_input_names.jl — a scale prior's caller inputs keep their meaning
# whatever they are called.
#
# Run: julia --project=test test/sd_prior_input_names.jl
#
# A configured random-effect or smoothing scale prior is merged into the term's
# submodel, and the caller scalars it reads are passed in as kwargs. Under their
# own names they rebound the submodel's: `sd(:, ID) ~ Exponential(tau)` failed to
# transpile, and `z`, `lkj_eta`, `n_terms` or a Student-t block's `nu` silently
# emitted a different model (todo `0exxiqv`). Each spelling below must emit
# exactly the Stan program of its twin whose scalar is called `lam`, so it has
# the twin's density; the twin itself must pass stanc.
using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Exponential, Gamma, Normal, TDist

const BLOCK_DF = (; subject=[1, 1, 2, 2, 3, 3, 4, 4],
    x=[0.1, 0.4, -0.2, 0.3, 0.7, 0.5, 0.2, -0.1],
    y=[1.4, 0.9, 1.6, 0.7, 1.2, 0.5, 1.1, 0.4])
const SMOOTH_DF = (; x=[cos(i) for i in 1:40], y=[sin(i) for i in 1:40])
# No `yb` column: the margin `b_loc` reaches no likelihood, so the block is
# split and the inactive margins' prior merges into the completion submodel.
const MARGINAL_DF = (; subject=[1, 1, 2, 2, 3, 3, 4, 4],
    x=[-1.0, 0.5, -0.5, 1.0, 0.0, 1.5, -1.5, 0.25],
    ya=[0.3, 0.9, -0.2, 0.4, 0.1, 1.2, -0.8, 0.0],
    yc=[-0.4, -0.1, 0.8, 0.3, 0.5, 0.2, -0.6, 0.1])

block(name, law...; kwargs...) = @eval SBBRMI((@brm BLOCK_DF begin
    $name ~ Gamma(2, 2)
    dof ~ Gamma(2, 0.1)
    mu ~ 1 + x + (1 + x | effect | subject)
    $(law...)
    y ~ Normal(mu, 1.0)
end); mod=@__MODULE__, total_groups=(),
    $((Expr(:kw, k, v) for (k, v) in kwargs)...))

# name => (the model with that scalar name, names its submodel binds itself)
const SHAPES = [
    "Gaussian block" => (n -> block(n, :(sd(:, effect) ~ Exponential($n))),
        (:tau, :L, :z, :lkj_eta, :n_terms, :n_groups, :group_idx)),
    "general family (merged tau override)" =>
        (n -> block(n, :(sd(:, effect) ~ Gamma(2.0, $n))), (:tau, :L)),
    "per-margin vector prior" =>
        (n -> block(n, :(sd(:, effect) ~ Exponential($n)),
                    :(sd(mu, effect, x) ~ Exponential(2.0))), (:tau, :L)),
    "centered block" =>
        (n -> block(n, :(sd(:, effect) ~ Exponential($n));
                    centered_groups=(:subject,)), (:tau, :L)),
    "cv block" =>
        (n -> block(n, :(sd(:, effect) ~ Exponential($n));
                    cv_groups=(:subject,)), (:tau, :n_groups)),
    "Student-t block, constant df" =>
        (n -> block(n, :(sd(:, effect) ~ Exponential($n)),
                    :(ranef(:, effect) ~ TDist(4.0))), (:nu, :w)),
    "Student-t block, sampled df" =>
        (n -> block(n, :(sd(:, effect) ~ Exponential($n)),
                    :(ranef(:, effect) ~ TDist(dof))), (:tau, :nu, :w)),
    "smoothing scale" => (n -> @eval(SBBRMI((@brm SMOOTH_DF begin
            $n ~ Gamma(2, 2)
            mu ~ 1 + s(x)
            sd(:, s(x)) ~ Exponential($n)
            y ~ Normal(mu, 1.0)
        end); mod=@__MODULE__, total_groups=())),
        (:sd_pen, :b_pen, :Zpen, :Xnull)),
    "marginal completion" => (n -> @eval(SBBRMI((@brm MARGINAL_DF begin
            $n ~ Gamma(2, 2)
            a_loc ~ 1 + x + (1 + x | p | subject)
            b_loc ~ 1 + (1 | p | subject)
            c_loc ~ 1 + (1 | p | subject)
            sd(:, p) ~ Exponential($n)
            ya ~ Normal(a_loc, 0.5)
            yb ~ Normal(b_loc, 0.5)
            yc ~ Normal(c_loc, 0.5)
        end); mod=@__MODULE__, total_groups=())),
        (:tau_inactive, :tau_active, :b_active, :n_inactive, :z)),
]

emitted(sb) = BayesianRegressionModels.stan_code(sb)

@testset "a scale prior's input named like a submodel binding" begin
    for (label, (model, names)) in SHAPES
        @testset "$label" begin
            twin = emitted(model(:lam))
            @test occursin(r"\blam\b", twin)
            @test StanBlocks.stanc_check(twin; warn_pedantic=false).ok
            for name in names
                @test emitted(model(name)) ==
                      replace(twin, r"\blam\b" => string(name))
            end
        end
    end
end
