# test/periodic_hsgp_shared_axis.jl — REGRESSION for snag periodic-hsgp-sh-2b41252b.
#
# Two population predictors carrying `hsgp(z; cov=:periodic)` over the SAME
# axis used to emit one `hsgp_z ~ _sb_hsgp_periodic(...)` binding twice — the
# periodic emitter minted `hsgp_<x>` / `PHI_hsgp_<x>` directly instead of
# routing through `_sb_unique_structured_term_names` like the mainline
# `hsgp`/`s`/`gp` paths. StanBlocks tracing refused the shadowed binding
# (`AssertionError: name ∉ keys(info)`), so SBBRMI built but `brm_descriptor`
# died at trace time (the same signature as snag mo-term-in-sever-fe459870).
# The fix gives repeats the scoped `hsgp_<target>_<x>` carrier while the first
# keeps the historical `hsgp_<x>` spelling.
#
# Run: julia --project=test test/periodic_hsgp_shared_axis.jl
# Set BRM_PERIODIC_HSGP_RUNTIME=0 to skip the BridgeStan coordinates gate.
using Test
using BayesianRegressionModels
using StanBlocks

const BRM = BayesianRegressionModels
const BS = StanBlocks.BridgeStan
const PERIODIC_HSGP_RUNTIME = get(ENV, "BRM_PERIODIC_HSGP_RUNTIME", "1") != "0"

periodic_hsgp_df = (;
    x=[0.1, 0.5, 0.9, 0.2, 0.6, 1.0, 0.3, 0.7],
    z=collect(1.0:8.0),
    ya=zeros(8),
    yb=zeros(8),
)

@testset "shared periodic axis builds with independent carriers" begin
    builder = @brm begin
        a ~ 0 + x + hsgp(z; k=3, cov=:periodic, period=8.0)
        b ~ 0 + x + hsgp(z; k=3, cov=:periodic, period=8.0)
        ya ~ Normal(a, 1.0)
        yb ~ Normal(b, 1.0)
    end
    sb = SBBRMI(builder(periodic_hsgp_df); mod=@__MODULE__)
    # Each occurrence owns its basis data: same values, distinct keys.
    @test sb.data[:PHI_hsgp_z] == sb.data[:PHI_hsgp_b_z]
    @test sb.data[:harmonics_hsgp_z] == sb.data[:harmonics_hsgp_b_z]
    @test sb.data[:rho_lower_hsgp_z] == sb.data[:rho_lower_hsgp_b_z]
    code = BRM.stan_code(sb)
    # First occurrence keeps the historical carrier; the repeat qualifies —
    # and each owns an independent length-scale / sd / weight set.
    @test occursin("hsgp_z = (PHI_hsgp_z *", code)
    @test occursin("hsgp_b_z = (PHI_hsgp_b_z *", code)
    @test occursin("hsgp_z_rho_iso ~ lognormal", code)
    @test occursin("hsgp_b_z_rho_iso ~ lognormal", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    # Descriptor tracing is what died pre-fix (`name ∉ keys(info)`).
    d = brm_descriptor(sb)
    byname = Dict(o.name => o for o in d.outputs)
    @test byname[:hsgp_z].declaration.target === :hsgp_z
    @test byname[:hsgp_b_z].declaration.target === :hsgp_b_z
end

@testset "a lone periodic term keeps its historical carrier" begin
    builder = @brm begin
        mu ~ 1 + hsgp(z; k=3, cov=:periodic, period=8.0)
        ya ~ Normal(mu, 1.0)
    end
    sb = SBBRMI(builder(periodic_hsgp_df); mod=@__MODULE__)
    @test haskey(sb.data, :PHI_hsgp_z)
    code = BRM.stan_code(sb)
    @test occursin("hsgp_z = (PHI_hsgp_z *", code)
    @test !occursin("hsgp_mu_z", code)
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
end

@testset "per-predictor periodic coordinates resolve distinct weights" begin
    if !PERIODIC_HSGP_RUNTIME
        @info "Skipping shared-periodic BridgeStan coordinates gate (BRM_PERIODIC_HSGP_RUNTIME=0)"
    else
        builder = @brm begin
            a ~ 0 + x + hsgp(z; k=3, cov=:periodic, period=8.0)
            b ~ 0 + x + hsgp(z; k=3, cov=:periodic, period=8.0)
            ya ~ Normal(a, 1.0)
            yb ~ Normal(b, 1.0)
        end
        d = brm_descriptor(builder, periodic_hsgp_df; mod=@__MODULE__,
            name=:shared_periodic_hsgp)
        prob = brm_execute(d, :fit)
        names = BS.param_names(prob.model; include_tp=false, include_gq=false)
        # Each predictor addresses its own 2k=6 weights under the SAME public
        # term label — the reporter's expected spelling.
        za = brm_term_coordinates(d, :a, names; term=:hsgp_z, parameter=:basis_weights)
        zb = brm_term_coordinates(d, :b, names; term=:hsgp_z, parameter=:basis_weights)
        @test length(za.coordinates) == 6 == length(zb.coordinates)
        @test za.output.name === :hsgp_z_beta_raw
        @test zb.output.name === :hsgp_b_z_beta_raw
        @test isempty(intersect(za.coordinates, zb.coordinates))
    end
end
