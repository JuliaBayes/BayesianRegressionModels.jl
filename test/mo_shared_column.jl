# test/mo_shared_column.jl — REGRESSION for snag mo-term-in-sever-fe459870.
#
# Two population predictors carrying `mo(c)` over the SAME column used to emit
# one `mo_c ~ _sb_mo(...)` binding twice; StanBlocks tracing refused the
# shadowed binding (`AssertionError: name ∉ keys(info)`), so SBBRMI built but
# `brm_descriptor` died at trace time. The fix suffixes repeats per occurrence
# (`mo_<target>_<c>`, the `s`/`gp`/`hsgp` precedent) while the first keeps the
# historical `mo_<c>` carrier — and every public label (`popcoefnames`,
# `effect(lp, mo_c)`, `term=:mo_c`, `coefficient=:mo_c`) stays put.
#
# Run: julia --project=test test/mo_shared_column.jl
# Set BRM_MO_RUNTIME=0 to skip the BridgeStan coordinates gate.
using Test
using BayesianRegressionModels
using StanBlocks
using Distributions: Dirichlet, Normal

const BRM = BayesianRegressionModels
const BS = StanBlocks.BridgeStan
const MO_RUNTIME = get(ENV, "BRM_MO_RUNTIME", "1") != "0"

mo_df = (;
    x=[0.0, 1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 3.5],
    g=[1, 2, 3, 4, 1, 2, 3, 4],
    ya=zeros(8),
    yb=zeros(8),
    yp=zeros(8),
    yq=zeros(8),
)

@testset "shared mo column builds with independent carriers" begin
    builder = @brm begin
        a ~ 0 + x + mo(g)
        b ~ 0 + x + mo(g)
        ya ~ Normal(a, 1.0)
        yb ~ Normal(b, 1.0)
    end
    brmi = builder(mo_df)
    # Public beta labels are unchanged: one `mo_g` magnitude per predictor.
    @test popcoefnames(brmi, :a) == [:x, :mo_g]
    @test popcoefnames(brmi, :b) == [:x, :mo_g]
    sb = SBBRMI(brmi; mod=@__MODULE__)
    code = BRM.stan_code(sb)
    # First occurrence keeps the historical carrier; the repeat qualifies.
    @test occursin("mo_g_simplex_incr ~ dirichlet", code)
    @test occursin("mo_b_g_simplex_incr ~ dirichlet", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    # Descriptor tracing is what died pre-fix (`name ∉ keys(info)`).
    d = brm_descriptor(sb)
    byname = Dict(o.name => o for o in d.outputs)
    @test byname[:mo_g].declaration.target === :mo_g
    @test byname[:mo_b_g].declaration.target === :mo_b_g
    @test byname[:pop_a_beta_pop].labels == [:x, :mo_g]
    @test byname[:pop_b_beta_pop].labels == [:x, :mo_g]
end

@testset "per-predictor mo priors stay independent" begin
    builder = @brm begin
        a ~ 0 + x + mo(g)
        b ~ 0 + x + mo(g)
        effect(a, mo_g) ~ Normal(0.0, 0.5)
        effect(b, :) ~ Normal(0.0, 2.0)
        simplex(a, mo(g)) ~ Dirichlet(1, 2, 3)
        simplex(b, mo(g)) ~ Dirichlet(4, 5, 6)
        ya ~ Normal(a, 1.0)
        yb ~ Normal(b, 1.0)
    end
    sb = SBBRMI(builder(mo_df); mod=@__MODULE__)
    code = BRM.stan_code(sb)
    # Each predictor's own Dirichlet concentration reaches its own carrier.
    @test occursin("mo_g_simplex_incr ~ dirichlet([1.0, 2.0, 3.0]');", code)
    @test occursin("mo_b_g_simplex_incr ~ dirichlet([4.0, 5.0, 6.0]');", code)
    # Each predictor's own magnitude prior reaches its own beta vector:
    # `a` keeps the default x scale and takes 0.5 on `mo_g`; `b` takes 2.0
    # on both columns.
    @test occursin("[1.0, 0.5]'", code)
    @test occursin("[2.0, 2.0]'", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
end

@testset "shared mo1 column builds with independent carriers" begin
    builder = @brm begin
        p ~ 0 + x + mo1(g)
        q ~ 0 + x + mo1(g)
        yp ~ Normal(p, 1.0)
        yq ~ Normal(q, 1.0)
    end
    sb = SBBRMI(builder(mo_df); mod=@__MODULE__)
    code = BRM.stan_code(sb)
    @test occursin("mo1_g_simplex_incr ~ dirichlet", code)
    @test occursin("mo1_q_g_simplex_incr ~ dirichlet", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    d = brm_descriptor(sb)
    byname = Dict(o.name => o for o in d.outputs)
    @test byname[:mo1_g].declaration.target === :mo1_g
    @test byname[:mo1_q_g].declaration.target === :mo1_q_g
end

@testset "same-predictor repeats disambiguate" begin
    builder = @brm begin
        a ~ 0 + mo(g) + mo(g)
        ya ~ Normal(a, 1.0)
    end
    sb = SBBRMI(builder(mo_df); mod=@__MODULE__)
    code = BRM.stan_code(sb)
    # brms semantics: two occurrences, two independent increment simplexes.
    @test occursin("mo_g_simplex_incr ~ dirichlet", code)
    @test occursin("mo_a_g_simplex_incr ~ dirichlet", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    d = brm_descriptor(sb)
    # ... but one public term label cannot address two carriers: fail closed
    # with the entry count rather than picking by descriptor order. (The
    # entries check runs before constrained names are touched, so no
    # BridgeStan compile is needed for this refusal.)
    @test_throws "occurs 2 times" brm_term_coordinates(
        d, :a, String[]; term=:mo_g, parameter=:simplex)
end

@testset "shared mo column in random effects builds per predictor" begin
    rdf = (; mo_df..., subject=[1, 1, 2, 2, 3, 3, 4, 4])
    builder = @brm begin
        a ~ 0 + (mo(g) | subject)
        b ~ 0 + (mo(g) | subject)
        ya ~ Normal(a, 1.0)
        yb ~ Normal(b, 1.0)
    end
    sb = SBBRMI(builder(rdf); mod=@__MODULE__)
    code = BRM.stan_code(sb)
    @test occursin("mo_g_simplex_incr ~ dirichlet", code)
    @test occursin("mo_b_g_simplex_incr ~ dirichlet", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    # The descriptor traces; coordinates resolve per predictor like the
    # population path (declaration-level; compiled names in the runtime set).
    d = brm_descriptor(sb)
    byname = Dict(o.name => o for o in d.outputs)
    @test byname[:mo_b_g].declaration.target === :mo_b_g
end

# Strip the Stan `data` block: dict-ordered declarations (and only those) may
# differ between two programs whose densities are identical.
function _mo_shared_without_data(code::AbstractString)
    lines = split(code, "\n")
    start = findfirst(==("data {"), lines)
    isnothing(start) && return code
    stop = findfirst(==("}"), lines[(start + 1):end])
    isnothing(stop) && return code
    join(vcat(lines[1:(start - 1)], lines[(start + stop + 1):end]), "\n")
end

@testset "fix matches the renamed-column workaround" begin
    # The reporter's workaround (one renamed copy of the column per predictor)
    # and the fix emit the same Stan program modulo the predictable renames —
    # i.e. the fix delivers exactly the independent-simplexes model the
    # workaround hand-built, with no extra or missing density.
    fixed = @brm mo_df begin
        a ~ 0 + x + mo(g)
        b ~ 0 + x + mo(g)
        ya ~ Normal(a, 1.0)
        yb ~ Normal(b, 1.0)
    end
    work_df = (;
        x=mo_df.x, ga=mo_df.g, gb=mo_df.g, ya=mo_df.ya, yb=mo_df.yb)
    workaround = @brm work_df begin
        a ~ 0 + x + mo(ga)
        b ~ 0 + x + mo(gb)
        ya ~ Normal(a, 1.0)
        yb ~ Normal(b, 1.0)
    end
    sb_fixed = SBBRMI(fixed; mod=@__MODULE__)
    sb_work = SBBRMI(workaround; mod=@__MODULE__)
    # Same codes drive both programs.
    @test sb_fixed.data[:g_idx] == sb_work.data[:ga_idx] == sb_work.data[:gb_idx]
    code_fixed = BRM.stan_code(sb_fixed)
    code_work = replace(
        BRM.stan_code(sb_work),
        "mo_gb" => "mo_b_g", "mo_ga" => "mo_g",
        "gb_idx" => "g_idx", "ga_idx" => "g_idx")
    @test _mo_shared_without_data(code_work) ==
          _mo_shared_without_data(code_fixed)
end

@testset "reprocess preserves shared-mo carriers" begin
    builder = @brm begin
        a ~ 0 + x + mo(g)
        b ~ 0 + x + mo(g)
        ya ~ Normal(a, 1.0)
        yb ~ Normal(b, 1.0)
    end
    sb = SBBRMI(builder(mo_df); mod=@__MODULE__)
    again = reprocess(sb, mo_df)
    code = BRM.stan_code(again)
    @test occursin("mo_g_simplex_incr ~ dirichlet", code)
    @test occursin("mo_b_g_simplex_incr ~ dirichlet", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    d = brm_descriptor(again)
    byname = Dict(o.name => o for o in d.outputs)
    @test byname[:mo_b_g].declaration.target === :mo_b_g
end

@testset "per-predictor coordinates resolve distinct simplexes" begin
    if !MO_RUNTIME
        @info "Skipping shared-mo BridgeStan coordinates gate (BRM_MO_RUNTIME=0)"
    else
        builder = @brm begin
            a ~ 0 + x + mo(g)
            b ~ 0 + x + mo(g)
            p ~ 0 + x + mo1(g)
            q ~ 0 + x + mo1(g)
            ya ~ Normal(a, 1.0)
            yb ~ Normal(b, 1.0)
            yp ~ Normal(p, 1.0)
            yq ~ Normal(q, 1.0)
        end
        d = brm_descriptor(builder, mo_df; mod=@__MODULE__, name=:shared_mo)
        prob = brm_execute(d, :fit)
        names = BS.param_names(prob.model; include_tp=false, include_gq=false)
        # Each predictor addresses its own simplex under the SAME public
        # term label — the reporter's expected spelling.
        za = brm_term_coordinates(d, :a, names; term=:mo_g, parameter=:simplex)
        zb = brm_term_coordinates(d, :b, names; term=:mo_g, parameter=:simplex)
        @test length(za.coordinates) == 3 == length(zb.coordinates)
        @test za.output.name === :mo_g_simplex_incr
        @test zb.output.name === :mo_b_g_simplex_incr
        @test isempty(intersect(za.coordinates, zb.coordinates))
        zp = brm_term_coordinates(d, :p, names; term=:mo1_g, parameter=:simplex)
        zq = brm_term_coordinates(d, :q, names; term=:mo1_g, parameter=:simplex)
        @test length(zp.coordinates) == 3 == length(zq.coordinates)
        @test zp.output.name === :mo1_g_simplex_incr
        @test zq.output.name === :mo1_q_g_simplex_incr
        @test isempty(intersect(zp.coordinates, zq.coordinates))
        # ... and its own magnitude under the same public coefficient label.
        ma = brm_population_effect_coordinates(
            d, :a, names; coefficient=:mo_g)
        mb = brm_population_effect_coordinates(
            d, :b, names; coefficient=:mo_g)
        @test length(ma.coordinates) == 1 == length(mb.coordinates)
        @test only(ma.coordinates) != only(mb.coordinates)
    end
end
