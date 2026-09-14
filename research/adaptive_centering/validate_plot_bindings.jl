# Bind every plotted coordinate to its named saved fit, independently of BRM's
# extraction/transport helpers. No sampler, native target or plotting load.
using Serialization, SHA, TOML, Test

function binding_table(path)
    lines = split.(readlines(path), '\t')
    names = Symbol.(first(lines))
    [NamedTuple{Tuple(names)}(Tuple(row)) for row in lines[2:end]]
end

function validate_plot_bindings(offline_dir, online_dir, diagnostics_dir,
                                source_audit_dir, partial_audit_dir)
    mapping = binding_table(joinpath(source_audit_dir, "source_coordinate_map.tsv"))
    partial_mapping = binding_table(joinpath(partial_audit_dir, "partial_source_coordinate_map.tsv"))
    indices = Dict(r.brm => i for (i, r) in enumerate(mapping))
    partial_indices = Dict(r.brm => i for (i, r) in enumerate(partial_mapping))
    selected = binding_table(joinpath(offline_dir, "centeredness.tsv"))
    learned = binding_table(joinpath(online_dir, "online_centeredness.tsv"))
    controls = Dict((r.predictor, parse(Int, r.basis)) => parse(Float64, r.centeredness)
                    for r in learned)
    fits = Dict(label => deserialize(joinpath(dir, "$label.jls"))
                for (label, dir) in (("noncentered", offline_dir),
                                    ("partial", offline_dir), ("online", online_dir)))
    # These are the immutable source model's spectral formula and the audit's
    # explicit coordinate order, not the producer's hsgp_coordinate_draws call.
    function saved_point(label, predictor, basis, draw)
        prefix = predictor == "mu" ? "hsgp_x" : "hsgp_log_sigma_x"
        index = label == "partial" ? partial_indices : indices
        q = fits[label].posterior_position
        log_rho = q[index["$(prefix)_rho_iso"], draw]
        log_sigma = q[index["$(prefix)_sigma"], draw]
        log_s = -.25 * (basis * exp(log_rho) * pi / 3)^2 +
                log_sigma + .45946926660233633 + .5 * log_rho
        coefficient = label == "partial" ? "beta_partial" : "beta_raw"
        value = q[index["$(prefix)_$(coefficient).$basis"], draw]
        c = label == "partial" ? parse(Float64,
            getproperty(selected[basis], predictor == "mu" ? :mean : :log_scale)) : 0.0
        (; value, log_s, c, rho=exp(log_rho), sigma=exp(log_sigma))
    end
    coordinate_count = 0
    pair_count = 0
    @testset "HSGP plotted rows belong to their named saved fits" begin
        @test length(mapping) == length(partial_mapping) == 44
        @test length(controls) == 40
        @test length(selected) == 20
        for fit in values(fits)
            @test fit.complete && size(fit.posterior_position) == (44, 10_000)
            @test all(isfinite, fit.posterior_position)
        end
        # Pair inputs retain each fit's compiled target coordinates. Online
        # returns NCP model coordinates; its learned display is checked below.
        for (label, dir) in (("noncentered", offline_dir), ("partial", offline_dir),
                             ("online", online_dir)), predictor in ("mu", "log_sigma")
            rows = binding_table(joinpath(dir, "$(label)_$(predictor)_weights.tsv"))
            @test length(rows) == 40_000
            seen = Set{Tuple{Int,Int}}()
            for r in rows
                draw, basis = parse(Int, r.draw), parse(Int, r.basis)
                push!(seen, (draw, basis))
                point = saved_point(label, predictor, basis, draw)
                @test parse(Float64, r.coordinate) == point.value
                @test parse(Float64, r.rho) ≈ point.rho
                @test parse(Float64, r.sigma) ≈ point.sigma
                @test isapprox(parse(Float64, r.log_spectral_scale), point.log_s;
                               atol=1e-10, rtol=1e-12)
                @test isapprox(parse(Float64, r.physical_weight),
                    point.value * exp((1-point.c)*point.log_s); atol=1e-12, rtol=1e-10)
            end
            @test seen == Set((draw, basis) for draw in 1:10_000 for basis in (1, 2, 19, 20))
            pair_count += length(rows)
        end
        rows = binding_table(joinpath(diagnostics_dir, "coordinate_gradients.tsv"))
        @test length(rows) == 240_000
        labels = Dict("NCP" => "noncentered", "Post-hoc" => "partial", "Online" => "online")
        seen = Set{Tuple{String,String,Int,Int}}()
        for r in rows
            draw, basis = parse(Int, r.draw), parse(Int, r.basis)
            push!(seen, (r.configuration, r.predictor, draw, basis))
            label = labels[r.configuration]
            point = saved_point(label, r.predictor, basis, draw)
            display_c = label == "online" ? controls[(r.predictor, basis)] : point.c
            expected = point.value * exp((display_c-point.c)*point.log_s)
            @test parse(Float64, r.centeredness) == display_c
            @test isapprox(parse(Float64, r.coordinate), expected; atol=1e-12, rtol=1e-10)
            @test isfinite(parse(Float64, r.gradient))
        end
        @test seen == Set((c, p, d, b) for c in keys(labels)
            for p in ("mu", "log_sigma") for d in 1:10_000 for b in (1, 2, 19, 20))
        coordinate_count = length(rows)
    end
    receipt = Dict("pair_input_rows" => pair_count,
        "gradient_coordinate_rows" => coordinate_count,
        "coordinate_gradients_sha256" => bytes2hex(sha256(read(
            joinpath(diagnostics_dir, "coordinate_gradients.tsv")))),
        "validator_sha256" => bytes2hex(sha256(read(@__FILE__))),
        "scope" => "Saved-fit coordinate and physical-weight bindings; gradient values have separate finite-difference checks",
        "fit_sha256" => Dict(label => bytes2hex(sha256(read(joinpath(dir, "$label.jls"))))
            for (label, dir) in (("noncentered", offline_dir), ("partial", offline_dir),
                                 ("online", online_dir))))
    TOML.print(stdout, receipt)
    receipt
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 5 || error("usage: validate_plot_bindings.jl OFFLINE ONLINE DIAGNOSTICS SOURCE_AUDIT PARTIAL_AUDIT")
    validate_plot_bindings(ARGS...)
end
