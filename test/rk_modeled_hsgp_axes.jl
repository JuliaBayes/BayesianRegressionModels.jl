# Public synthetic HSGP terms over model-derived axes and linear
# orthogonalization: the basis is evaluated in the RK graph from the current
# sampled axis, on its fixed formula domain, and matches compiled Stan.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))

module PublicModeledHSGPAxis
using BayesianRegressionModels, Distributions
function build(data)
    @brm data begin
        eta ~ 1 + (1 | location | subject)
        effect(eta, :) ~ Normal(-0.4, 0.7)
        sd(:, location) ~ Exponential(0.8)
        x = exp(eta)
        assay_scale ~ Exponential(0.6)
        assay ~ censored(LogNormal(eta, assay_scale); lower=limit)
        mu ~ 1 + x + hsgp(x; k=3, domain=(0.0, 4.0), orthogonal_to=:linear) +
            (1 + x | effect | subject)
        length_scale(:, hsgp(x)) ~ Uniform(0.4, 3.0)
        sd(:, hsgp(x)) ~ Normal(0.0, 0.7)
        effect(mu, :) ~ Normal(0.0, 1.2)
        sd(:, effect) ~ Exponential(0.9)
        cor(:, effect) ~ LKJCholesky(2, 2.5)
        sigma ~ Exponential(0.7)
        y ~ Normal(mu, sigma)
    end
end
end

# Independent fixed-domain eigenbasis, optionally centered and orthogonal to
# the centered axis. It never calls BRM basis helpers.
function modeled_hsgp_basis(x, k, center, width; orthogonal=false)
    omega2 = [(j * pi / (2width))^2 for j in 1:k]
    PHI = [sin(sqrt(omega2[j]) * (v - center + width)) / sqrt(width) for v in x, j in 1:k]
    if orthogonal
        xc = x .- sum(x) / length(x)
        ss = sum(abs2, xc)
        PHI = PHI .- sum(PHI; dims=1) ./ size(PHI, 1)
        ss > 1e-12 && (PHI = PHI .- xc .* (transpose(xc) * PHI) ./ ss)
    end
    floor = k == 1 ? 0.0 : 4width / pi * sqrt(log(100) / (k^2 - 1))
    PHI, omega2, floor
end

modeled_hsgp_weights(omega2, sigma, rho) =
    [sigma * sqrt(rho * sqrt(2pi)) * exp(-0.25 * rho^2 * w) for w in omega2]

# A constant sampled axis selects the degenerate-axis branch of the
# orthogonalization, where the law has a kink; finite differences are
# taken only at points whose subjects differ.
modeled_hsgp_points(N) = (zeros(N), collect(range(-0.2, 0.3; length=N)),
    0.13 .+ 0.07 .* sin.(1:N))

function modeled_hsgp_outputs(graph)
    [repr(entry.recipe.outputs) for entry in recipe_inventory(graph)]
end


@stestset "model-derived HSGP axis keeps fixed domain, orthogonal basis and sampled values" begin
    original = (; subject=[1,1,2,2,3,3],
        assay=[0.2,0.7,0.5,0.2,0.8,0.9], limit=fill(0.2,6),
        y=[0.1,0.4,-0.2,0.3,0.7,0.5])
    for permutation in ([1,2,3,4,5,6], [2,3,5,1,6,4])
        data = map(column -> column[permutation], original)
        saved = deepcopy(data)
        brmi = PublicModeledHSGPAxis.build(data)
        artifact = BRM.emit_rk_artifact(brmi; case_id="modeled-hsgp-axis")
        backend, problem = consumer_problem(brmi)
        names = coordinate_names(backend.model.layout)
        @test length(names) == 23
        # The axis is a graph value, never bound or frozen as data.
        @test !haskey(backend.plan.columns, :x)
        @test !haskey(backend.plan.columns, :eta)
        term = only(t for p in backend.plan.regression.predictors
            for t in p.terms if t.kind === :hsgp)
        @test term.columns == [:x]
        @test term.options.latent && term.options.orthogonal === :linear
        @test term.options.fixed_fits == ((2.0, 2.0),)
        emitted = BRM._rk_emit_ast(backend.plan)
        main = sprint(Base.show_unquoted, emitted.main)
        blocks = join((sprint(Base.show_unquoted, d) for d in emitted.defs
            if BRM._rk_source_definition(d).kind === :rkppl), "\n")
        # A bounded authored length-scale prior reads no floor, so the basis
        # graph returns only the matrix and squared frequencies.
        @test occursin("(hsgp_x_PHI, hsgp_x_omega2) = brm_hsgp_basis_graph(x)", main)
        @test !occursin("rho_floor", main)
        definitions = join(sprint(Base.show_unquoted, d) for d in emitted.defs)
        @test occursin("beta_raw[1:nbasis] .~ Normal.(0, 1)", definitions)
        @test occursin("hsgp_x ~ brm_hsgp_effect(hsgp_x_PHI, hsgp_x_omega2, 3)", main)
        @test occursin("rho_iso ~ Uniform(0.4, 3.0)", definitions)
        # Basis, centering, projection and spectral weights are numerical
        # intermediates of the actual built posterior graph.
        outputs = join(modeled_hsgp_outputs(kernel_graph(backend.model.spec)), "\n")
        for intermediate in ("basis_rows", "raw_basis", "axis_centered", "axis_ss",
                "basis_columns", "omega2", "weights")
            @test occursin(intermediate, outputs)
        end
        # The emitted basis graph is the fixed-domain orthogonal law at any axis.
        ext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
        mod = ext._rk_emit_module(emitted)
        owner = getfield(mod, :brm_hsgp_basis_graph)
        reader(axis) = first(Base.invokelatest(owner, axis))
        for axis in ([0.4, 1.1, 2.6, 0.9, 3.3, 1.7], [0.3, 0.8, 2.2, 1.0])
            @test reader(axis) ≈
                first(modeled_hsgp_basis(axis, 3, 2.0, 2.0; orthogonal=true)) atol=2e-14 rtol=2e-14
        end
        constant_axis = fill(1.3, 4)
        @test Base.invokelatest(reader, constant_axis) ≈
            first(modeled_hsgp_basis(constant_axis, 3, 2.0, 2.0; orthogonal=true)) atol=2e-14
        index(n) = only(findall(==(Symbol(n)), names))
        beta = index.(["eta_Intercept", "pop_mu.beta_pop.1", "pop_mu.beta_pop.2"])
        location_scale = index("b_location_subject.tau.1")
        location_z = [index("b_location_subject.z.$j.1") for j in 1:3]
        scales = index.(["b_effect_subject.tau.1", "b_effect_subject.tau.2"])
        z = [index("b_effect_subject.z.$j.$k") for j in 1:3, k in 1:2]
        correlation = index("b_effect_subject.L.1")
        assay_scale, sigma = index.(["assay_scale", "sigma"])
        rho_u, hsgp_sigma = index.(["hsgp_x.rho_iso", "hsgp_x.sigma"])
        weights_z = [index("hsgp_x.beta_raw.$b") for b in 1:3]
        function components(u)
            rho, tau = tanh(u[correlation]), exp.(u[scales])
            L = [1.0 0.0; rho sqrt(1-rho^2)]
            random = u[z] * transpose(tau .* L)
            eta = u[beta[1]] .+ exp(u[location_scale]) .* u[location_z][data.subject]
            x = exp.(eta)
            probability = 1 / (1 + exp(-u[rho_u]))
            length_scale = 0.4 + 2.6 * probability
            PHI, omega2, _ = modeled_hsgp_basis(x, 3, 2.0, 2.0; orthogonal=true)
            smooth = PHI * (modeled_hsgp_weights(omega2, exp(u[hsgp_sigma]),
                length_scale) .* u[weights_z])
            mu = u[beta[2]] .+ u[beta[3]] .* x .+ smooth .+
                random[data.subject,1] .+ random[data.subject,2] .* x
            (; eta, x, mu, rho, tau, length_scale, probability,
                assay_scale=exp(u[assay_scale]), sigma=exp(u[sigma]))
        end
        function assay_parts(c)
            [data.assay[j] <= data.limit[j] ?
                logcdf(LogNormal(c.eta[j],c.assay_scale),data.limit[j]) :
                logpdf(LogNormal(c.eta[j],c.assay_scale),data.assay[j]) for j in 1:6]
        end
        function oracle(u)
            c = components(u)
            logpdf(Normal(-0.4,0.7),u[beta[1]]) +
                sum(logpdf.(Normal(0,1.2),u[beta[2:3]])) +
                sum(logpdf.(Normal(),u[location_z])) + sum(logpdf.(Normal(),u[z])) +
                logpdf(Exponential(0.8),exp(u[location_scale])) + u[location_scale] +
                sum(logpdf.(Exponential(0.9),c.tau) .+ u[scales]) +
                logpdf(Beta(2.5,2.5),(c.rho+1)/2) - log(2) + log1p(-c.rho^2) +
                logpdf(Exponential(0.6),c.assay_scale) + u[assay_scale] +
                logpdf(Exponential(0.7),c.sigma) + u[sigma] +
                logpdf(Uniform(0.4,3.0),c.length_scale) + log(2.6) +
                    log(c.probability) + log1p(-c.probability) +
                logpdf(Normal(0,0.7),exp(u[hsgp_sigma])) + u[hsgp_sigma] +
                sum(logpdf.(Normal(),u[weights_z])) +
                sum(assay_parts(c)) + sum(logpdf.(Normal.(c.mu,c.sigma),data.y))
        end
        stan = consumer_stan(brmi, "modeled-hsgp-axis-$(first(permutation))";
            mod=PublicModeledHSGPAxis)
        mapping = [names[beta[1]]=>"pop_eta_beta_pop.1",
            names[beta[2]]=>"pop_mu_beta_pop.1", names[beta[3]]=>"pop_mu_beta_pop.2",
            names[location_scale]=>"b_location_subject_tau.1",
            names[correlation]=>"b_effect_subject_L.1",
            names[assay_scale]=>"assay_scale", names[sigma]=>"sigma",
            names[rho_u]=>"hsgp_x_rho_iso", names[hsgp_sigma]=>"hsgp_x_sigma"]
        append!(mapping,[names[location_z[j]]=>"b_location_subject_z_flat.$j" for j in 1:3])
        append!(mapping,[names[scales[k]]=>"b_effect_subject_tau.$k" for k in 1:2])
        append!(mapping,[names[z[j,k]]=>"b_effect_subject_z_flat.$(k+2*(j-1))" for j in 1:3 for k in 1:2])
        append!(mapping,[names[weights_z[b]]=>"hsgp_x_beta_raw.$b" for b in 1:3])
        bound = ext._rk_translated_plan(backend.plan)
        pointwise = prepare_query(backend.model, bound, :pointwise)
        translated = BRM.rk_translate_artifact(artifact)
        rebuilt = Base.invokelatest(build_kernel, translated)
        replay_pointwise = prepare_query(rebuilt, translated, :pointwise)
        replay = prepare_sampler(rebuilt, translated, zeros(23);
            backend=AutoEnzyme(; mode=Enzyme.Reverse))
        # Equal subject draws make x constant: the value still follows the
        # oracle, and value and gradient follow compiled Stan's same branch.
        degenerate = fill(0.13, 23)
        @test allequal(components(degenerate).x)
        @test LogDensityProblems.logdensity(problem, degenerate) ≈ oracle(degenerate) atol=2e-11 rtol=2e-11
        check_consumer_stan(problem, stan, mapping, backend, degenerate)
        for u in modeled_hsgp_points(23)
            value, gradient = check_consumer_point(problem, u, oracle)
            check_consumer_stan(problem, stan, mapping, backend, u)
            c, parts = components(u), pointwise(u)
            @test parts.assay ≈ assay_parts(c)
            @test parts.y ≈ logpdf.(Normal.(c.mu,c.sigma),data.y)
            @test isequal(parts, Base.invokelatest(replay_pointwise, u))
            replay_gradient = similar(u)
            replay_value, _ = sampler_value_and_gradient!(replay, replay_gradient, u)
            @test isequal(value, replay_value)
            @test gradient ≈ replay_gradient atol=2e-13 rtol=2e-13
        end
        @test isequal(data, saved)
    end
end

@stestset "model-derived and raw HSGP axes match compiled Stan and independent laws" begin
    data = (; subject=[1,1,2,2,3,3], y=[0.1,0.4,-0.2,0.3,0.7,0.5],
        w=[-0.6,0.1,0.5,-0.2,0.9,0.3])
    saved = deepcopy(data)
    groups = data.subject
    cases = (
        # Default priors keep the fixed-domain validity floor on the length scale.
        ("default-floor", @brm(data, begin
            eta ~ 1 + (1 | location | subject)
            x = exp(eta)
            mu ~ 1 + hsgp(x; k=3, domain=(0.0, 4.0))
            y ~ Normal(mu, 1.0)
        end)),
        # A log-link predictor supplies the axis on its response scale.
        ("link-predictor", @brm(data, begin
            log(x) ~ 1 + (1 | location | subject)
            mu ~ 1 + x + hsgp(x; k=4, domain=(0.02, 4.0), orthogonal_to=:linear)
            y ~ Normal(mu, 1.0)
        end)),
        # Raw-data axes now carry the same orthogonalization in the graph.
        ("raw-automatic", @brm(data, begin
            mu ~ 1 + w + hsgp(w; k=3, orthogonal_to=:linear)
            y ~ Normal(mu, 1.0)
        end)),
        ("raw-domain", @brm(data, begin
            mu ~ 1 + w + hsgp(w; k=3, domain=(-1.0, 1.5), orthogonal_to=:linear)
            y ~ Normal(mu, 1.0)
        end)))
    for (label, brmi) in cases
        backend, problem = consumer_problem(brmi)
        names = coordinate_names(backend.model.layout)
        index(n) = only(findall(==(Symbol(n)), names))
        axis = label in ("raw-automatic", "raw-domain") ? :w : :x
        k = label == "link-predictor" ? 4 : 3
        @test haskey(backend.plan.columns, :w) == (axis === :w)
        @test !haskey(backend.plan.columns, :x)
        id = "hsgp_$axis"
        rho_u, hsgp_sigma = index.([id * ".rho_iso", id * ".sigma"])
        weights_z = [index(id * ".beta_raw.$b") for b in 1:k]
        mapping = Pair{Symbol,String}[names[rho_u]=>id * "_rho_iso",
            names[hsgp_sigma]=>id * "_sigma"]
        append!(mapping, [names[weights_z[b]]=>id * "_beta_raw.$b" for b in 1:k])
        intercept = label == "default-floor" ? index("mu_Intercept") :
            index("pop_mu.beta_pop.1")
        push!(mapping, names[intercept]=>"pop_mu_beta_pop.1")
        slope = label == "default-floor" ? nothing : index("pop_mu.beta_pop.2")
        slope === nothing || push!(mapping, names[slope]=>"pop_mu_beta_pop.2")
        if axis === :x
            parent = label == "link-predictor" ? "x" : "eta"
            parent_beta = index("$(parent)_Intercept")
            location_scale = index("b_location_subject.tau.1")
            location_z = [index("b_location_subject.z.$j.1") for j in 1:3]
            push!(mapping, names[parent_beta]=>
                (label == "link-predictor" ? "pop_log_x_beta_pop.1" : "pop_eta_beta_pop.1"),
                names[location_scale]=>"b_location_subject_tau.1")
            append!(mapping, [names[location_z[j]]=>"b_location_subject_z_flat.$j" for j in 1:3])
        end
        function oracle(u)
            value = logpdf(Normal(), u[intercept]) +
                logpdf(LogNormal(0,1), exp(u[hsgp_sigma])) + u[hsgp_sigma] +
                sum(logpdf.(Normal(), u[weights_z]))
            slope === nothing || (value += logpdf(Normal(), u[slope]))
            axis_value = if axis === :x
                value += logpdf(Normal(), u[parent_beta]) +
                    logpdf(Normal(), exp(u[location_scale])) + u[location_scale] +
                    sum(logpdf.(Normal(), u[location_z]))
                exp.(u[parent_beta] .+ exp(u[location_scale]) .* u[location_z][groups])
            else
                data.w
            end
            center, width = label == "raw-automatic" ?
                (sum(data.w)/6, 1.5maximum(abs.(data.w .- sum(data.w)/6))) :
                label == "raw-domain" ? (0.25, 1.25) :
                label == "link-predictor" ? (2.01, 1.99) : (2.0, 2.0)
            PHI, omega2, floor = modeled_hsgp_basis(axis_value, k, center, width;
                orthogonal=label != "default-floor")
            rho = floor + exp(u[rho_u])
            value += logpdf(LogNormal(0,1), rho) + u[rho_u]
            smooth = PHI * (modeled_hsgp_weights(omega2, exp(u[hsgp_sigma]), rho) .* u[weights_z])
            mu = u[intercept] .+ smooth
            slope === nothing || (mu = mu .+ u[slope] .* axis_value)
            value + sum(logpdf.(Normal.(mu, 1), data.y))
        end
        stan = consumer_stan(brmi, "modeled-hsgp-control-$label")
        N = length(names)
        for u in modeled_hsgp_points(N)
            check_consumer_point(problem, u, oracle)
            check_consumer_stan(problem, stan, mapping, backend, u)
        end
    end
    @test isequal(data, saved)
end
