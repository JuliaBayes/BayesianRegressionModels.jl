# test/ranef_components.jl — `BRMOutput.component`: which part of its
# random-effect block each output is (snag `random-effect-ou-33fe1a8c`, user
# pick option A on decision `1t1id88`).
#
# The declaration half builds every block shape BRM emits and checks that each
# `:random_effect` output carries a component and that each block holds the
# components its parameterization defines. The runtime half reads draws
# through `brm_outputs(d; component=…)` alone and checks each component is the
# quantity it names against `brm_ranef_draws` (whose own oracles in
# test/ranef_draws.jl never read these carriers): `:scale` is the SD,
# `exp(:log_scale)` is the SD, `L L'` of `:correlation_cholesky` is the
# correlation, `diag(SD) L :standardized` is `:effects` (times the root of the
# `:mixing_weight` for Student-t) and `:contribution` is each row's design row
# times its level's effects.
#
# Run: julia --project=test test/ranef_components.jl
# (set BRM_RANEF_COMPONENTS_RUNTIME=0 to run only the declaration half.)

using Test
using BayesianRegressionModels
using StanBlocks
using Random
using LinearAlgebra: dot
using Distributions: Exponential, TDist, Beta, LKJCholesky

const BRM = BayesianRegressionModels
const BS = StanBlocks.BridgeStan
const RUNTIME = get(ENV, "BRM_RANEF_COMPONENTS_RUNTIME", "1") != "0"

rng = MersenneTwister(11)
const N = 24
data = (;
    subject = repeat(string.("s", 1:6); inner = 4),
    site = repeat(["a", "b", "c"], N ÷ 3),
    x = randn(rng, N),
    zage = randn(rng, N),
    y = randn(rng, N),
    z = randn(rng, N),
    omega_sd = 0.3,
    g1 = repeat(["a", "b", "c"], N ÷ 3),
    g2 = repeat(["b", "c", "a"], N ÷ 3),
    w1 = rand(rng, N),
    w2 = rand(rng, N),
)
strata = merge(data, (; group_of_site = [s == "c" ? "u" : "v" for s in data.site]))
no_z = Base.structdiff(data, NamedTuple{(:z,)})

# Two scalar `(1 | g)` blocks on two predictors (the reporter's shape).
scalar_builder = @brm begin
    sigma ~ Exponential(1)
    loc ~ 1 + zage + (1 | subject)
    slope ~ 1 + (1 | subject)
    mu = loc + slope * x
    y ~ Normal(mu, sigma)
end
shared_builder = @brm begin
    sigma ~ Exponential(1)
    tau_z ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject) + (1 | site)
    nu ~ 1 + (1 | p | subject)
    y ~ Normal(mu, sigma)
    z ~ Normal(nu, tau_z)
end
plain_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | subject) + (0 + x | site)
    y ~ Normal(mu, sigma)
end
generic_builder = @brm begin
    sigma ~ Exponential(1)
    tau_z ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject)
    nu ~ 1 + (1 | p | subject)
    sd(mu, p, x) ~ Exponential(1 / 4)
    cor(:, p) ~ LKJCholesky(3, 2)
    y ~ Normal(mu, sigma)
    z ~ Normal(nu, tau_z)
end
student_builder = @brm begin
    sigma ~ Exponential(1)
    tau_z ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject)
    nu ~ 1 + (1 | p | subject)
    ranef(:, p) ~ TDist(4)
    y ~ Normal(mu, sigma)
    z ~ Normal(nu, tau_z)
end
totals_builder = @brm begin
    sigma ~ Exponential(1)
    tau_z ~ Exponential(1)
    mu ~ 1 + x + (1 + x || subject)
    nu ~ 1 + (1 | q | subject)
    y ~ Normal(mu, sigma)
    z ~ Normal(nu, tau_z)
end
known_mixed_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject)
    sd(mu, p, Intercept) = sqrt(0.1)
    sd(mu, p, x) ~ Exponential(0.5)
    y ~ Normal(mu, sigma)
end
known_all_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | subject)
    sd(:, p) = omega_sd
    cor(:, p) = [1 0.3; 0.3 1]
    y ~ Normal(mu, sigma)
end
r2d2_bucket_builder = @brm begin
    sigma ~ Exponential(1)
    log_Vc ~ 1 + (1 | p | subject)
    log_k10 ~ 1 + (1 | p | subject)
    sd(:, p) ~ r2d2(mean_R2 = 0.5, prec_R2 = 2, concentration = 1,
                    reference_scale = sigma)
    cor(:, p) ~ LKJCholesky(2, 2)
    y ~ Normal(log_Vc + log_k10, sigma)
end
r2d2_flat_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 | subject)
    effect(mu, :) ~ r2d2(R2 = Beta(1, 1), tau_bsv = 0.5, alpha = 1)
    y ~ Normal(mu, sigma)
end
mm_builder = @brm begin
    sigma ~ Exponential(1)
    loc ~ 1 + (1 | mm(g1, g2; weights = (w1, w2)))
    y ~ Normal(loc, sigma)
end
stratified_builder = @brm begin
    sigma ~ Exponential(1)
    mu ~ 1 + x + (1 + x | gr(site, by = group_of_site))
    y ~ Normal(mu, sigma)
end
stratified_id_builder = @brm begin
    sigma ~ Exponential(1)
    tau_z ~ Exponential(1)
    mu ~ 1 + x + (1 + x | p | gr(site, by = group_of_site))
    nu ~ 1 + (1 | p | gr(site, by = group_of_site))
    y ~ Normal(mu, sigma)
    z ~ Normal(nu, tau_z)
end

build(builder, d = data; kw...) = SBBRMI(builder(d); mod = @__MODULE__, kw...)
none = (; total_groups = ())

# Every block shape: (label, SBBRMI thunk).
shapes = [
    "two scalar (1 | g) blocks" => () -> build(scalar_builder; none...),
    "shared |ID| and scalar" => () -> build(shared_builder; none...),
    "shared |ID| and scalar, centred" =>
        () -> build(shared_builder; none..., centered_groups = [:subject, :site]),
    "shared |ID| and scalar, cv" =>
        () -> build(shared_builder; none..., cv_groups = [:subject, :site]),
    "omitted response: marginal block + completion" => () -> build(shared_builder, no_z; none...),
    "omitted response, centred" =>
        () -> build(shared_builder, no_z; none..., centered_groups = [:subject]),
    "plain correlated and slope" => () -> build(plain_builder; none...),
    "plain correlated and slope, centred" =>
        () -> build(plain_builder; none..., centered_groups = [:subject, :site]),
    "configured sd/cor priors" => () -> build(generic_builder; none...),
    "configured sd/cor priors, centred" =>
        () -> build(generic_builder; none..., centered_groups = [:subject]),
    "Student-t" => () -> build(student_builder; none...),
    "exact totals" => () -> build(totals_builder),
    "known scale" => () -> build(known_mixed_builder; none...),
    "known scale and correlation" => () -> build(known_all_builder; none...),
    "R2D2 bucket" => () -> build(r2d2_bucket_builder; none...),
    "R2D2 flat" => () -> build(r2d2_flat_builder; none...),
    "multi-membership" => () -> build(mm_builder; none...),
    "stratified" => () -> build(stratified_builder, strata; none...),
    "stratified |ID|" => () -> build(stratified_id_builder, strata; none...),
    "stratified, cv" => () -> build(stratified_builder, strata; none..., cv_groups = [:site]),
    "population model" => () -> population_model(build(shared_builder; none...); groups = :subject),
]

# block binding => sorted components of its outputs
block_components(d) = Dict(
    target => sort!([o.component for o in d.outputs
                     if !isnothing(o.component) && !isnothing(o.declaration) &&
                        o.declaration.target === target])
    for target in unique(o.declaration.target for o in brm_outputs(d; role = :random_effect)
                         if !isnothing(o.declaration)))

@testset "every random-effect output has a component ($label)" for (label, make) in shapes
    d = brm_descriptor(make())
    re = brm_outputs(d; role = :random_effect)
    @test !isempty(re)
    @test all(o -> !isnothing(o.component), re)
    # Outputs outside every block carry none. Population blocks name their
    # parts too (`:coefficients`, `:contribution`; design decision `0yh0jw3`).
    @test all(o -> isnothing(o.component) ||
                   o.role in (:random_effect, :parameter, :population_effect), d.outputs)
end

@testset "components per block shape" begin
    d = brm_descriptor(build(scalar_builder; none...))
    bc = block_components(d)
    @test Set(keys(bc)) == Set(b.binding for b in ranef_blocks(d.plan))
    @test all(==([:contribution, :log_scale, :standardized]), values(bc))
    # The standardized carrier is the one `RanefBlock.z` already names.
    for b in ranef_blocks(d.plan)
        @test only(o.name for o in brm_outputs(d; component = :standardized)
                   if o.declaration.target === b.binding) === b.z
    end

    d = brm_descriptor(build(shared_builder; none...))
    p = only(b for b in ranef_blocks(d.plan) if b.id === :p)
    @test block_components(d)[p.binding] ==
        [:correlation_cholesky, :effects, :scale, :standardized, :standardized]
    # The scale is the carrier `brm_ranef_sd_coordinates` resolves.
    scale = only(o for o in brm_outputs(d; component = :scale)
                 if o.declaration.target === p.binding)
    names = ["$(scale.name).$k" for k in 1:3]
    @test brm_ranef_sd_coordinates(d, :mu, names; id = :p).output.name === scale.name

    d = brm_descriptor(build(plain_builder; none...))
    bc = block_components(d)
    subject = only(b for b in ranef_blocks(d.plan) if b.group === :subject)
    site = only(b for b in ranef_blocks(d.plan) if b.group === :site)
    @test bc[subject.binding] ==
        [:contribution, :correlation_cholesky, :effects, :scale, :standardized, :standardized]
    @test bc[site.binding] == [:contribution, :scale, :standardized]

    d = brm_descriptor(build(student_builder; none...))
    p = only(b for b in ranef_blocks(d.plan) if b.id === :p)
    @test :mixing_weight in block_components(d)[p.binding]

    # Centred: the sampled coordinate is the effect, never `:standardized`.
    d = brm_descriptor(build(shared_builder; none..., centered_groups = [:subject, :site]))
    for b in ranef_blocks(d.plan)
        @test :standardized ∉ block_components(d)[b.binding]
        @test only(o.component for o in d.outputs if o.name === b.z) === :effects
    end

    # Configured priors, centred: a plate draws the effects, and the covariance
    # factor `diag(tau) L` StanBlocks hoists out of that plate is `:auxiliary`.
    d = brm_descriptor(build(generic_builder; none..., centered_groups = [:subject]))
    p = only(b for b in ranef_blocks(d.plan) if b.id === :p)
    bc = block_components(d)[p.binding]
    @test count(==(:auxiliary), bc) == 1 && :standardized ∉ bc
    @test count(==(:effects), bc) == 3   # sampled cells, plate result, returned matrix

    d = brm_descriptor(build(stratified_builder, strata; none...))
    bc = only(values(block_components(d)))
    @test count(==(:scale), bc) == 2 && count(==(:correlation_cholesky), bc) == 2
    @test :level_scale in bc && :level_correlation_cholesky in bc

    d = brm_descriptor(build(totals_builder))
    for t in total_effect_blocks(d.plan)
        @test only(o for o in d.outputs if o.name === t.deviations).component === :effects
        @test only(o for o in d.outputs if o.name === t.scales).component === :scale
    end
end

@testset "brm_outputs component= filter" begin
    d = brm_descriptor(build(shared_builder; none...))
    both = brm_outputs(d; component = (:scale, :log_scale))
    @test Set(o.component for o in both) == Set([:scale, :log_scale])
    @test brm_outputs(d; component = :scale) ⊆ both
    @test isempty(brm_outputs(d; component = :scale, role = :posterior_predictive))
    @test isempty(brm_outputs(d; component = :no_such_component))
    @test length(brm_outputs(d)) == length(d.outputs)
end

@testset "an unlisted internal is refused, not left unlabelled" begin
    # refused: a random-effect carrier the emission table does not classify
    # would reach a consumer that groups by component as a silent `nothing`
    # (decision 1t1id88; dev §1).
    spec = BRM._RANEF_FAMILIES[:ranef_correlated_draws]
    try
        BRM._RANEF_FAMILIES[:ranef_correlated_draws] =
            merge(spec, (; internals = Base.structdiff(spec.internals, NamedTuple{(:L,)})))
        @test_throws "have no block component" brm_descriptor(build(shared_builder; none...))
    finally
        BRM._RANEF_FAMILIES[:ranef_correlated_draws] = spec
    end
    @test brm_descriptor(build(shared_builder; none...)) isa BRMDescriptor
end

if !RUNTIME
    @info "BRM_RANEF_COMPONENTS_RUNTIME=0 — skipping the compiled-model half"
else

function constrained_draws(sb; n_draws = 3, seed = 1)
    sm = BRM.stan_instantiate(sb).model
    names = BS.param_names(sm; include_tp = true, include_gq = true)
    r = MersenneTwister(seed)
    rows = map(1:n_draws) do i
        theta = 0.7 .* randn(r, BS.param_unc_num(sm))
        BS.param_constrain(sm, theta; include_tp = true, include_gq = true,
                           rng = BS.StanRNG(sm, seed + i))
    end
    (; names, draws = permutedims(reduce(hcat, rows)), d = brm_descriptor(sb))
end

# Draws × elements of every output of `block` holding `component`, located
# through the descriptor alone (several when a block emits one quantity in two
# layouts, e.g. a centred block's sampled effects and its returned matrix).
component_draws(c, block, component) =
    [c.draws[:, brm_output_coordinates(o, c.names)]
     for o in brm_outputs(c.d; component) if o.declaration.target === block.binding]
has(c, block, component) = !isempty(component_draws(c, block, component))
# Equal values, whatever the carrier's element layout.
same_values(x, y) = isapprox(sort(vec(x)), sort(vec(y)); atol = 1e-10)

# Checks one `brm_ranef_draws` entry `e` of an unstratified block against its
# components. `Z(r)` and `row_levels` (each row's level index) drive the
# `:contribution` check; omit them for a block that emits none.
function check_components(c, e; Z = nothing, row_levels = nothing)
    n, G, K = size(e.effects)
    effects_ok = all(x -> all(same_values(x[i, :], e.effects[i, :, :]) for i in 1:n),
                     component_draws(c, e, :effects))
    scale_ok = has(c, e, :log_scale) ?
        exp.(only(component_draws(c, e, :log_scale))) ≈ e.sd :
        only(component_draws(c, e, :scale)) ≈ e.sd
    L(i) = K == 1 ? ones(1, 1) :
        reshape(only(component_draws(c, e, :correlation_cholesky))[i, :], K, K)
    cor_ok = K == 1 || all(isapprox(L(i) * L(i)', e.cor[i, :, :]; atol = 1e-10) for i in 1:n)
    w = has(c, e, :mixing_weight) ? sqrt.(only(component_draws(c, e, :mixing_weight))) :
        ones(n, G)
    # Every standardized layout is K × G column-major (`z_flat`, `z`, `xi`).
    standardized_ok = all(component_draws(c, e, :standardized)) do z
        all(isapprox(e.effects[i, g, :],
                     w[i, g] .* (e.sd[i, :] .* (L(i) * reshape(z[i, :], K, G)[:, g]));
                     atol = 1e-10) for i in 1:n, g in 1:G)
    end
    contribution_ok = isnothing(Z) || all(1:n) do i
        rows = only(component_draws(c, e, :contribution))[i, :]
        all(isapprox(rows[r], dot(Z(r), e.effects[i, row_levels[r], :]); atol = 1e-10)
            for r in eachindex(rows))
    end
    (; effects_ok, scale_ok, cor_ok, standardized_ok, contribution_ok,
       has_effects = has(c, e, :effects), has_standardized = has(c, e, :standardized))
end
level_index(e, labels) = [findfirst(==(l), e.levels) for l in labels]

@testset "scalar (1 | g) blocks: log scale, standardized draw, contribution" begin
    c = constrained_draws(build(scalar_builder; none...))
    for e in brm_ranef_draws(c.d, c.draws, c.names)
        r = check_components(c, e; Z = _ -> [1.0], row_levels = level_index(e, data.subject))
        @test r.scale_ok && r.standardized_ok && r.contribution_ok && r.has_standardized
        @test !r.has_effects
    end
end

@testset "shared |ID| block: scale, correlation factor, standardized, effects" begin
    c = constrained_draws(build(shared_builder; none...))
    p = only(brm_ranef_draws(c.d, c.draws, c.names; id = :p))
    r = check_components(c, p)
    @test r.effects_ok && r.scale_ok && r.cor_ok && r.standardized_ok && r.has_standardized
    site = only(brm_ranef_draws(c.d, c.draws, c.names; group = :site))
    r = check_components(c, site; Z = _ -> [1.0], row_levels = level_index(site, data.site))
    @test r.scale_ok && r.standardized_ok && r.contribution_ok
end

@testset "plain correlated block: per-level effects and per-row contribution" begin
    c = constrained_draws(build(plain_builder; none...))
    e = only(brm_ranef_draws(c.d, c.draws, c.names; group = :subject))
    r = check_components(c, e; Z = r -> [1.0, data.x[r]],
                         row_levels = level_index(e, data.subject))
    @test r.effects_ok && r.scale_ok && r.cor_ok && r.standardized_ok && r.contribution_ok
end

@testset "Student-t block: the mixing weight scales the standardized draw" begin
    c = constrained_draws(build(student_builder; none...))
    p = only(brm_ranef_draws(c.d, c.draws, c.names; id = :p))
    r = check_components(c, p)
    @test r.effects_ok && r.scale_ok && r.cor_ok && r.standardized_ok
end

@testset "centred blocks: the sampled effects are `:effects`" begin
    c = constrained_draws(build(shared_builder; none..., centered_groups = [:subject, :site]))
    for e in brm_ranef_draws(c.d, c.draws, c.names)
        r = check_components(c, e)
        @test r.effects_ok && r.has_effects && r.scale_ok && r.cor_ok && !r.has_standardized
    end
end

@testset "brm-use labelling recipe: each block's scale with its predictors" begin
    c = constrained_draws(build(shared_builder; none...))
    d, constrained_draws_, constrained_names = c.d, c.draws, c.names
    # The snippet in brm-use references/descriptor.md, verbatim.
    blocks = Dict(e.binding => e for e in brm_ranef_draws(d, constrained_draws_, constrained_names))
    seen = Dict{Symbol,Any}()
    for o in brm_outputs(d; role=:random_effect, component=(:scale, :log_scale))
        predictors = unique(m.predictor for m in blocks[o.declaration.target].margins)
        cols = brm_output_coordinates(o, constrained_names)   # the scale's posterior columns
        seen[o.declaration.target] = (predictors, o.component, cols)
    end
    L = brm_outputs(d; component=:correlation_cholesky)       # every block's correlation factor
    p = only(b for b in ranef_blocks(d.plan) if b.id === :p)
    site = only(b for b in ranef_blocks(d.plan) if b.group === :site)
    @test seen[p.binding][1:2] == ([:mu, :nu], :scale)
    @test seen[site.binding][1:2] == ([:mu], :log_scale)
    @test constrained_draws_[:, seen[p.binding][3]] ≈ blocks[p.binding].sd
    @test exp.(constrained_draws_[:, seen[site.binding][3]]) ≈ blocks[site.binding].sd
    @test [o.declaration.target for o in L] == [p.binding]
end

end # RUNTIME
