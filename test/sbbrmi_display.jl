# Run: julia --project=test test/sbbrmi_display.jl
# Displayed configurations must reconstruct the executable model, including its
# priors and data bindings. No sampling is needed to verify this display path.
using Test, BayesianRegressionModels, StanBlocks, Distributions
const BRM = BayesianRegressionModels

function display_roundtrip(sb)
    before = BRM.stan_code(sb)
    data_before = deepcopy(sb.data)
    text = sprint(show, sb)
    @test !occursin("SlicModel(untraced", text)
    # Parse what the reader sees, rather than evaluating the printer's internal
    # AST: quoting and namespacing errors must fail this roundtrip too.
    sections = split(text, "emitted @slic body:\n"; limit=2)
    mod = Module(gensym(:DisplayedSBBRMI))
    Core.eval(mod, :(using BayesianRegressionModels, StanBlocks))
    header = split(sections[1], "configured submodels:\n"; limit=2)
    if length(header) == 2
        Core.eval(mod, Meta.parse("begin\n" * header[2] * "\nend"))
    end
    body = Meta.parse(sections[2])
    rebuilt = StanBlocks.SlicModel(body, sb.data, mod)
    @test StanBlocks.stan_code(rebuilt) == before
    @test BRM.stan_code(sb) == before
    @test sb.data == data_before
    @test StanBlocks.stanc_check(before; warn_pedantic=false).ok
    text
end

@testset "configured SBBRMI display" begin
    df = (; x=collect(range(-1., 1.; length=24)), y=sin.(1:24))
    shared = @brm df begin
        mu ~ hsgp(x; k=5)
        eta ~ hsgp(x; k=5)
        length_scale(:, hsgp(x)) ~ LogNormal(0, 4)
        sd(:, hsgp(x)) ~ LogNormal(0, 4)
        y ~ Normal(mu, exp(eta))
    end
    sb = SBBRMI(shared)
    text = display_roundtrip(sb)
    @test length(findall("Base.merge(", text)) == 1
    @test length(findall("_sb_hsgp_configured_1", text)) == 3 # definition + two calls
    @test occursin("rho_iso ~ lognormal(0.0, 4.0", text)
    @test occursin("sigma ~ lognormal(0.0, 4.0", text)
    @test occursin("brm_hsgp_sqrt_spd", string(BRM._sb_hsgp.model))
    @test !occursin("brm_hsgp_sqrt_spd", text) # no nested implementation expansion

    distinct = @brm df begin
        mu ~ hsgp(x; k=5)
        eta ~ hsgp(x; k=5)
        length_scale(mu, hsgp(x)) ~ LogNormal(0, 4)
        length_scale(eta, hsgp(x)) ~ Gamma(2, 3)
        y ~ Normal(mu, exp(eta))
    end
    text = display_roundtrip(SBBRMI(distinct))
    @test length(findall("Base.merge(", text)) == 2
    @test occursin("_sb_hsgp_configured_2", text)
    @test occursin("gamma(2.0, 1.0 ./ 3.0", text)

    unconfigured = @brm df begin
        mu ~ hsgp(x; k=5)
        y ~ Normal(mu, 1.)
    end
    default_sb = SBBRMI(unconfigured)
    text = display_roundtrip(default_sb)
    @test !occursin("configured submodels:", text)
    @test endswith(text, sprint(print, default_sb.model.model))

    # Native positive-prior vector configuration uses a different template and
    # must not be mistaken for a GP merely because a scale prior looks similar.
    grouped = @brm (; df..., group=repeat(1:4; inner=6)) begin
        mu ~ 1 + (1 | site | group)
        sd(:, site) ~ Cauchy(0, 2)
        y ~ Normal(mu, 1.)
    end
    text = display_roundtrip(SBBRMI(grouped))
    @test occursin("tau ~ cauchy(", text)
    @test occursin("ranef_correlated_draws_generic", text)

    # Replaying on new data retains the display behavior.
    display_roundtrip(reprocess(sb, (; x=df.x .+ 0.1, y=df.y)))

    # An input with the natural alias spelling must keep its identity.
    collision_data = merge(sb.data, Dict(:_sb_hsgp_configured_1 => 2.0))
    collision = SBBRMI(sb.parent,
        StanBlocks.SlicModel(sb.model.model, collision_data, sb.model.mod),
        collision_data, sb.preproc)
    parts = BRM._sb_display_parts(collision)
    @test only(parts.definitions).args[1] === :_sb_hsgp_configured_2

    templates = BRM._sb_display_templates()
    unrelated = StanBlocks.@slic begin
        fresh ~ normal(2, 7)
        return fresh
    end
    @test isnothing(BRM._sb_display_configuration(unrelated, templates))
    # A changed data dictionary cannot borrow a familiar name.
    base = BRM._sb_hsgp
    @test isnothing(BRM._sb_display_configuration(
        StanBlocks.SlicModel(base.model, Dict(:rho_lower=>1.), base.mod), templates))
end
