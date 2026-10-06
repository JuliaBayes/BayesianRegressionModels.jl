# test/rk_artifact.jl — fusion-append artifacts + SB-leg machinery.
#
# Run: julia --project=test test/rk_artifact.jl
#
# Fixture-tests the BRM side of the parity-closeout append sweep (pair
# closeout-appends): v4 artifact emit → serialize → read (RK-free),
# artifact → bound-plan translation through the production route
# (behavioral equivalence with the live RKBRMI path), the Layer-2 plan
# dump, the explicit SB name-map machinery (pure), and the SB Stan
# codegen level (no BridgeStan compile — the heavy Stan leg runs at
# fusion under a compute token; see test/rk_append_worker.jl).

using Test
using BayesianRegressionModels
using Distributions: Exponential, Normal, logpdf
using ReactiveKernelsPPL: build_kernel, coordinate_names, prepare_query
using StanBlocks

const BRM = BayesianRegressionModels

module ArtifactGensymBindingSource
import BayesianRegressionModels: _rk_callable_source!

function original_affine end
native_affine(x, offset) = x .+ offset

function _rk_callable_source!(definitions, bindings, entry,
        ::typeof(original_affine))
    leaf = gensym(:artifact_native_affine)
    push!(bindings, leaf => native_affine)
    push!(definitions, :(function $entry(x, offset)
        return $leaf(x, offset)
    end))
    :done
end
end

const _DF = (;
    x=[0.5, -1.0, 1.5, 0.0, -0.5, 1.0],
    y=[1.0, 2.0, 1.5, 2.5, 3.0, 2.0],
)

function _gaussian_brmi()
    return @brm _DF begin
        mu ~ 1 + x
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
end

# Structural equality for plan round-trips: plain composite structs
# inherit egal `==` (distinct heap objects never compare equal), so the
# round-trip assert walks leaves with `==` and composites fieldwise.
function _structural_equal(a, b)
    typeof(a) === typeof(b) || return false
    T = typeof(a)
    if T <: Union{Number,Symbol,Bool,Nothing,AbstractString,Expr,DataType,
            Module,Function}
        return a == b
    elseif a isa AbstractArray
        return size(a) == size(b) &&
            all(_structural_equal(x, y) for (x, y) in zip(a, b))
    elseif a isa AbstractDict
        return length(a) == length(b) && all(
            haskey(b, k) && _structural_equal(v, b[k])
            for (k, v) in a)
    elseif a isa Tuple
        return length(a) == length(b) &&
            all(_structural_equal(x, y) for (x, y) in zip(a, b))
    elseif a isa NamedTuple
        return propertynames(a) == propertynames(b) &&
            all(_structural_equal(getfield(a, f), getfield(b, f))
                for f in propertynames(a))
    elseif isconcretetype(T) && fieldcount(T) > 0
        return all(_structural_equal(getfield(a, f), getfield(b, f))
            for f in fieldnames(T))
    else
        return a == b
    end
end

# Use the bound plan: it carries the emitted data-only definitions as well
# as the original columns, exactly as the production sampler does.
function _spec_posterior(model, translated, u)
    Base.invokelatest(prepare_query(model, translated, :sampler), Vector{Float64}(u))
end

@testset "artifact emit shape" begin
    brmi = _gaussian_brmi()
    a = BRM.emit_rk_artifact(brmi; case_id="fixture-gauss",
        provenance="test/rk_artifact.jl", brm_pin="test-pin")
    @test keys(a) == (:case_id, :ast, :defs, :bindings, :plan, :meta)
    @test a.case_id == "fixture-gauss"
    @test a.ast isa Expr && a.ast.head === :block
    # The population effects are one self-contained component submodel.
    @test a.defs isa Vector{Expr} &&
        [first(first(d.args).args) for d in a.defs] == [:brm_population_effects]
    @test a.bindings isa Vector{Pair{Symbol,Any}}
    @test a.plan isa BRM._RKStructuralPlan
    @test a.plan.n_obs == 6
    @test sort!(collect(keys(a.plan.columns))) == [:x, :y]
    @test a.meta.case_id == "fixture-gauss"
    @test a.meta.generator_version == BRM.rk_artifact_version() == 4
    @test a.meta.brm_pin == "test-pin"
    @test a.meta.emitted_at isa Int && a.meta.emitted_at > 0
    @test a.meta.julia_version == string(VERSION)
    @test_throws ErrorException BRM.emit_rk_artifact(brmi; case_id="")
end

@testset "artifact write/read round-trip" begin
    brmi = _gaussian_brmi()
    a = BRM.emit_rk_artifact(brmi; case_id="fixture-gauss")
    path = joinpath(mktempdir(), "fixture-gauss.jls")
    @test BRM.write_rk_artifact(path, a) == path
    b = BRM.read_rk_artifact(path)
    @test keys(b) == keys(a)
    @test b.case_id == a.case_id
    @test b.ast == a.ast
    @test b.defs == a.defs
    @test _structural_equal(b.bindings, a.bindings)
    @test _structural_equal(b.plan, a.plan)
    @test b.meta == a.meta
end

@testset "artifact retains gensym provider bindings from its source emission" begin
    data = (; x=[-0.5, 0.25, 0.75], y=[-0.2, 0.4, 1.1])
    brmi = @brm data begin
        offset ~ Normal(0, 1)
        mu = ArtifactGensymBindingSource.original_affine(x, offset)
        y ~ Normal(mu, 0.5)
    end
    artifact = BRM.emit_rk_artifact(brmi; case_id="gensym-provider-binding")
    @test length(artifact.bindings) == 1
    stored_name = first(artifact.bindings).first
    @test occursin(string(stored_name), sprint(show, artifact.defs))
    fresh = BRM._rk_emit_ast(artifact.plan)
    @test length(fresh.bindings) == 1
    @test first(fresh.bindings).first != stored_name

    roundtripped = BRM.read_rk_artifact(BRM.write_rk_artifact(
        joinpath(mktempdir(), "gensym-provider-binding.jls"), artifact))
    translated = BRM.rk_translate_artifact(roundtripped)
    model = build_kernel(translated)
    for u in ([0.0], [0.3], [-0.4])
        value = _spec_posterior(model, translated, u)
        oracle = logpdf(Normal(), only(u)) +
            sum(logpdf.(Normal.(data.x .+ only(u), 0.5), data.y))
        @test value ≈ oracle atol=1e-12
    end
end

@testset "artifact weighted predictor round-trip" begin
    # The complete ordinary program owns its matrix product and weighting.
    wdf = merge(_DF, (; n=[1, 2, 1, 2, 1, 2]))
    brmi = @brm wdf begin
        mu ~ 1 + x
        sigma ~ Exponential(1)
        y ~ weighted(Normal(mu, sigma), fweights(n))
    end
    a = BRM.emit_rk_artifact(brmi; case_id="fixture-weighted")
    @test [first(first(d.args).args) for d in a.defs] == [:brm_population_effects]
    b = BRM.read_rk_artifact(
        BRM.write_rk_artifact(joinpath(mktempdir(), "w.jls"), a))
    @test b.defs == a.defs
    translated = BRM.rk_translate_artifact(b)
    @test b.plan.n_obs == 6
    u = [0.5, -0.25, 0.1]
    model = build_kernel(translated)
    v = _spec_posterior(model, translated, u)
    coordinates = Dict(coordinate_names(model.layout) .=> u)
    coefficients = [coordinates[Symbol("pop_mu.beta_pop.1")],
        coordinates[Symbol("pop_mu.beta_pop.2")]]
    log_sigma = coordinates[:sigma]
    sigma = exp(log_sigma)
    oracle = sum(wdf.n .* logpdf.(
        Normal.(coefficients[1] .+ coefficients[2] .* wdf.x, sigma), wdf.y)) +
        sum(logpdf.(Normal(), coefficients)) +
        logpdf(Exponential(1), sigma) + log_sigma
    @test v ≈ oracle atol=1e-12
end

@testset "artifact read fails closed on skew" begin
    brmi = _gaussian_brmi()
    a = BRM.emit_rk_artifact(brmi; case_id="fixture-gauss")
    dir = mktempdir()
    garbage = joinpath(dir, "garbage.jls")
    open(garbage, "w") do io
        write(io, "not an artifact")
    end
    @test_throws Exception BRM.read_rk_artifact(garbage)
    old_meta = merge(a.meta, (; generator_version=1))
    old = merge(a, (; meta=old_meta))
    old_path = joinpath(dir, "old.jls")
    BRM.write_rk_artifact(old_path, old)
    err = try
        BRM.read_rk_artifact(old_path)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("generator_version", err.msg)
end

@testset "artifact translates through the production route" begin
    brmi = _gaussian_brmi()
    a = BRM.emit_rk_artifact(brmi; case_id="fixture-gauss")
    roundtripped = BRM.read_rk_artifact(
        BRM.write_rk_artifact(joinpath(mktempdir(), "rt.jls"), a))
    translated = BRM.rk_translate_artifact(roundtripped)
    @test roundtripped.plan.n_obs == 6
    # Behavioral equivalence with the live RKBRMI production path
    # (`_brm_rk_plan` → fresh emit → `_rk_translated_plan` →
    # `build_kernel`): same posterior, bit-equal, at the origin and a
    # nonzero probe.
    backend = BRM.RKBRMI(brmi)
    rt_model = build_kernel(translated)
    for u in (zeros(3), [0.5, -0.25, 0.1])
        live = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)._rk_translated_plan(backend.plan)
        v_live = _spec_posterior(backend.model, live, u)
        v_rt = _spec_posterior(rt_model, translated, u)
        @test v_rt == v_live
        @test isfinite(v_rt)
    end
    @test _structural_equal(backend.plan, a.plan)
end

@testset "Layer-2 plan dump" begin
    brmi = _gaussian_brmi()
    a = BRM.emit_rk_artifact(brmi; case_id="fixture-gauss")
    text = BRM.show_rk_plan(a.plan)
    @test occursin("n_obs = 6", text)
    @test occursin("gaussian", text)
    @test occursin("response y", text)
    @test occursin("predictor mu", text)
    @test occursin("columns     = [x, y]", text)
    # A ranef plan exercises the bucket/vector dump lines.
    ranef_df = merge(_DF, (; g=[1, 1, 2, 2, 3, 3]))
    ranef_brmi = @brm ranef_df begin
        mu ~ 1 + x + (1 + x | ID | g)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    ranef_plan = BRM._brm_rk_plan(ranef_brmi)
    @test length(ranef_plan.ranef_buckets) == 1
    ranef_text = BRM.show_rk_plan(ranef_plan)
    @test occursin("ranef       = ", ranef_text)
end

@testset "SB map resolution is a validated bijection" begin
    rk_names = [:mu_b1, :mu_b2, :sigma]
    stan_names = ["sigma", "b_mu_1", "b_mu_2"]
    sb_map = [:mu_b1 => "b_mu_1", :mu_b2 => "b_mu_2", :sigma => "sigma"]
    perm = BRM.resolve_sb_map(sb_map, rk_names, stan_names; case_id="t")
    @test perm == [3, 1, 2]
    u_rk = [0.5, -0.25, 0.1]
    @test BRM.apply_sb_map(u_rk, perm) == [0.1, 0.5, -0.25]
    g_stan = [1.0, 2.0, 3.0]
    @test BRM.unmap_sb_grad(g_stan, perm) == [2.0, 3.0, 1.0]
    # Fail-closed: duplicates, unknown names, partial coverage.
    dup_rk = [:mu_b1 => "b_mu_1", :mu_b1 => "b_mu_2", :sigma => "sigma"]
    @test_throws ErrorException BRM.resolve_sb_map(
        dup_rk, rk_names, stan_names; case_id="t")
    unknown = [:mu_b1 => "b_mu_1", :mu_b2 => "nope", :sigma => "sigma"]
    @test_throws ErrorException BRM.resolve_sb_map(
        unknown, rk_names, stan_names; case_id="t")
    short = [:mu_b1 => "b_mu_1", :sigma => "sigma"]
    @test_throws ErrorException BRM.resolve_sb_map(
        short, rk_names, stan_names; case_id="t")
    @test_throws ErrorException BRM.unmap_sb_grad([1.0, 2.0], perm)
end

@testset "SB Stan codegen level (no compile)" begin
    brmi = _gaussian_brmi()
    sb = SBBRMI(brmi; mod=@__MODULE__)
    code = StanBlocks.stan_code(sb.model)
    @test code isa AbstractString
    @test length(code) > 1000
    @test occursin("normal_id_glm", code)
end

# ---------------------------------------------------------------- worker
# The worker script is includable (main() runs only as a script); its
# pure machinery is fixture-tested here, and the end-to-end path adapts:
# full outputs when the twin's transpile_report_v3 has landed, else the
# loud missing-v3 error (never a silent skip).

include("rk_append_worker.jl")

using TOML: TOML

function _write_probe(dir::AbstractString)
    path = joinpath(dir, "probe.jl")
    open(path, "w") do io
        println(io, "# APPEND-LAYER1-BEGIN")
        println(io, "data = (; x=[0.5, -1.0, 1.5, 0.0, -0.5, 1.0],")
        println(io, "          y=[1.0, 2.0, 1.5, 2.5, 3.0, 2.0])")
        println(io, "builder = @brm begin")
        println(io, "    mu ~ 1 + x")
        println(io, "    sigma ~ Exponential(1)")
        println(io, "    y ~ Normal(mu, sigma)")
        println(io, "end")
        println(io, "# APPEND-LAYER1-END")
        println(io, "PROBE = (; builder, data,")
        println(io, "    sb_map=[1 => \"a\"], oracle=nothing)")
    end
    return path
end

@testset "worker argv/spec/probe validation" begin
    opts = _parse_argv(["--spec", "a.toml", "--out", "d"])
    @test opts.spec == "a.toml" && opts.out == "d"
    @test !opts.no_sb && !opts.print_coords && !opts.no_token
    @test _parse_argv(["--spec", "a", "--out", "d", "--no-token"]).no_token
    @test_throws ErrorException _parse_argv(["--out", "d"])
    @test_throws ErrorException _parse_argv(["--spec", "a", "--out", "d", "--bogus"])
    @test_throws ErrorException _parse_argv(
        ["--spec", "a", "--out", "d", "--no-sb", "--print-coords"])
    dir = mktempdir()
    spec_path = joinpath(dir, "in.toml")
    open(spec_path, "w") do io
        println(io, "id = \"t\"")
        println(io, "brm_inputs = \"p.jl\"")
        println(io, "manifest = \"m\"")
    end
    spec = _read_spec(spec_path)
    @test spec.id == "t" && spec.u_probes === nothing
    u_spec = joinpath(dir, "u.toml")
    open(u_spec, "w") do io
        println(io, "id = \"t\"")
        println(io, "brm_inputs = \"p.jl\"")
        println(io, "u_probes = [[0.5, -0.25, 0.1], [0, 0, 0]]")
    end
    spec2 = _read_spec(u_spec)
    @test spec2.u_probes == [[0.5, -0.25, 0.1], [0.0, 0.0, 0.0]]
    bad = joinpath(dir, "bad.toml")
    open(bad, "w") do io
        println(io, "id = \"t\"")
        println(io, "brm_inputs = \"p.jl\"")
        println(io, "u_probes = [[0.5], [0.0, 0.0]]")
    end
    @test_throws ErrorException _read_spec(bad)
    probe_path = _write_probe(dir)
    layer1 = _extract_layer1(probe_path)
    @test occursin("mu ~ 1 + x", layer1)
    @test !occursin("PROBE", layer1)
    nomark = joinpath(dir, "nomark.jl")
    write(nomark, "x = 1\n")
    @test_throws ErrorException _extract_layer1(nomark)
end

@testset "worker numbers.toml round-trips through TOML" begin
    rows = [(; posterior=-15.5, grad="PASS", grad_maxdiff=1.2e-16,
        sb_value=-15.5000000001, sb_grad_maxdiff=2.3e-14,
        sb_fd_maxdiff=1e-9, oracle=(; value=-15.5, diff_vs_sb=1e-10))]
    path = joinpath(mktempdir(), "numbers.toml")
    _write_numbers(path, "t", rows, Dict("brm" => "abc", "julia" => "1.10"))
    back = TOML.parsefile(path)
    @test back["probe"] isa Vector && length(back["probe"]) == 1
    p = only(back["probe"])
    @test p["posterior"] == -15.5
    @test p["grad"] == "PASS"
    @test p["grad_maxdiff"] == 1.2e-16
    @test p["sb_value"] == -15.5000000001
    @test p["sb_fd_maxdiff"] == 1e-9
    @test p["oracle"]["diff_vs_sb"] == 1e-10
    @test back["pins"]["brm"] == "abc"
end

@testset "worker end-to-end (--no-sb, reporter v2 interface)" begin
    dir = mktempdir()
    probe_path = _write_probe(dir)
    spec_path = joinpath(dir, "in.toml")
    open(spec_path, "w") do io
        println(io, "id = \"fixture-gauss\"")
        println(io, "brm_inputs = \"$probe_path\"")
        println(io, "u_probes = [[0.5, -0.25, 0.1]]")
        println(io, "manifest = \"fixture\"")
    end
    outdir = joinpath(dir, "out")
    try
        ret = _run_case(spec_path, outdir; no_sb=true, print_coords=false)
        @test ret == 0
        # Full path: all three outputs, machine
        # readable, posterior finite.
        for f in ("artifact.jls", "sections.md", "numbers.toml")
            @test isfile(joinpath(outdir, f))
        end
        sections = read(joinpath(outdir, "sections.md"), String)
        for h in ("## Layer 1", "## Layer 2", "## Layer 3",
                "## Boundary", "## Layer 4", "## Verification")
            @test occursin(h, sections)
        end
        @test !occursin("## Layer SB", sections)
        @test occursin("mu ~ 1 + x", sections)
        nums = TOML.parsefile(joinpath(outdir, "numbers.toml"))
        @test length(nums["probe"]) == 1
        @test isfinite(nums["probe"][1]["posterior"])
        @test nums["probe"][1]["grad"] == "PASS"
        println("ARTIFACT_REPORTER posterior=", nums["probe"][1]["posterior"],
            " gradient=", nums["probe"][1]["grad"])
    catch e
        # Pre-landing seam: the worker must fail closed LOUDLY on the
        # missing reporter v2 interface (never a silent or divergent render), and
        # the RK-free prefix (emit) must already have produced the
        # artifact. The backtrace prints so a non-seam failure is
        # diagnosable from the log.
        showerror(stderr, e, catch_backtrace())
        println(stderr)
        @test occursin("transpile_report_v2", sprint(showerror, e))
        @test isfile(joinpath(outdir, "artifact.jls"))
    end
end
