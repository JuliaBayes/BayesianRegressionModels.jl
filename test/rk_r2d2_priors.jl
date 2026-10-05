# Synthetic shared R2D2M2 laws: normalized density, all native Reverse
# coordinates, physical same-BRMI Stan parity and complete printed-source replay.
include(joinpath(@__DIR__, "rk_consumer_support.jl"))

module PublicR2D2
using BayesianRegressionModels, Distributions
const DATA = (; subject=[1,1,2,2,3,3], x=[-0.6,0.2,0.4,0.8,0.3,1.0],
    ya=[0.2,-0.1,0.4,0.3,-0.2,0.6], yb=[-0.3,0.2,0.1,0.5,0.4,-0.1])
function build(data=DATA; joint=false, categorical=false, latent=false, partial=false,
        design=false, extra_prior="")
    extra = joint ? ", include=(:population, :contrasts)" : ""
    covariate = categorical ? (design ? " + factor(c)" : " + c") : ""
    reference = latent ? "" : ", reference_scale=scale_a"
    budget = partial ? "sd(mu_a, shared)" : "sd(:, shared)"
    override = latent || partial ? "" :
        "sd(mu_b, shared) ~ r2d2(reference_scale=scale_b)"
    body = """
    scale_a ~ Exponential(0.8)
    scale_b ~ Exponential(0.9)
    mu_a ~ 1 + x$covariate + (1 | shared | subject)
    mu_b ~ 1 + x$covariate + (1 | shared | subject)
    $budget ~ r2d2(mean_R2=0.4, prec_R2=3.0, concentration=1.2$reference$extra)
    $override
    $extra_prior
    cor(:, shared) ~ LKJCholesky(2, 2.0)
    ya ~ Normal(mu_a, scale_a)
    yb ~ Normal(mu_b, scale_b)
    """
    Core.eval(@__MODULE__, BayesianRegressionModels._brm(body; df=data))
end
end

function r2d2_components(u, names; nphi, latent=false, partial=false)
    index(name) = only(findall(==(Symbol(name)), names))
    beta = index.(["mu_a_Intercept", "mu_a_x", "mu_b_Intercept", "mu_b_x"])
    z = [index("ranef_draws_shared_subject_z.$j.$k") for j in 1:3, k in 1:2]
    ir2 = index("ranef_draws_shared_subject_sd_r2d2_1_R2")
    iphi = index.(["ranef_draws_shared_subject_sd_r2d2_1_phi.$j" for j in 1:nphi-1])
    iscales = index.(["scale_a", "scale_b"])
    rho = tanh(u[index("ranef_draws_shared_subject_L.1")])
    r2 = 1 / (1 + exp(-u[ir2]))
    phi, remaining, jac = Float64[], 1.0, 0.0
    for j in 1:nphi-1
        fraction = 1 / (1 + exp(-(u[iphi[j]] + log(nphi-j))))
        push!(phi, remaining * fraction)
        jac += log(remaining) + log(fraction) + log1p(-fraction)
        remaining *= 1-fraction
    end
    push!(phi, remaining)
    scales = exp.(u[iscales])
    reference = latent ? exp.(u[index.([
        "ranef_draws_shared_subject_sd_r2d2_1_ref_$j" for j in 1:2])]) : scales
    free = partial ? exp(u[index("ranef_draws_shared_subject_sd_r2d2_free_2")]) : 0.
    tau = partial ? [reference[1]*sqrt(r2/(1-r2)),free] :
        reference .* sqrt.(phi[1:2] .* r2 ./ (1-r2))
    L = [1.0 0.0; rho sqrt(1-rho^2)]
    random = u[z] * transpose(tau .* L)
    jac += sum(u[iscales]) + log(r2) + log1p(-r2) + log1p(-rho^2)
    latent && (jac += sum(log,reference))
    partial && (jac += log(free))
    (; beta=u[beta], z=u[z], r2, phi, scales, reference, free, tau, L, random, rho, jac)
end

@stestset "shared R2D2M2 original laws and allocation controls" begin
    for mode in (:margins, :joint, :contrasts, :design, :latent, :partial)
        joint = mode in (:joint, :contrasts, :design, :latent)
        categorical = mode in (:contrasts, :design)
        design = mode === :design
        latent, partial = mode === :latent, mode === :partial
        data = categorical ? merge(PublicR2D2.DATA,(;c=[1,1,1,2,2,3])) : PublicR2D2.DATA
        saved = deepcopy(data)
        brmi = PublicR2D2.build(data; joint, categorical, design, latent, partial)
        backend, problem = consumer_problem(brmi)
        names = coordinate_names(backend.model.layout)
        nphi = categorical ? 8 : joint ? 4 : partial ? 1 : 2
        @test length(names) == 13+nphi+4categorical+2latent+partial
        @test count(n -> occursin("_R2",string(n)), names) == 1
        @test count(n -> occursin("_phi",string(n)), names) == nphi-1
        @test !any(n -> occursin(r"_sd\.[12]$",string(n)), names)
        emitted = BRM._rk_emit_ast(backend.plan)
        source = sprint(Base.show_unquoted, emitted.main)
        @test !occursin("brm_value_function", source)
        @test occursin("Dirichlet", source)
        graph = ReactiveKernels.kernel_graph(backend.model.spec)
        ports = Set(string(p.name) for recipe in graph.recipes for p in recipe.outputs)
        @test "ranef_draws_shared_subject_sd" in ports
        @test !joint || "mu_a_x_r2d2_variance" in ports
        function components(u)
            c = r2d2_components(u, names; nphi, latent, partial)
            mu_a = c.beta[1] .+ c.beta[2] .* data.x .+ c.random[data.subject,1]
            mu_b = c.beta[3] .+ c.beta[4] .* data.x .+ c.random[data.subject,2]
            slope_shares = categorical ? [3,6] : [3,4]
            slope_sd = joint ? c.reference .* sqrt.(c.phi[slope_shares] .* c.r2 ./
                ((1-c.r2) * sum((data.x .- sum(data.x)/6).^2)/5)) : ones(2)
            contrasts, contrast_sd = zeros(2,0), zeros(2,0)
            if categorical
                contrasts = [u[only(findall(==(Symbol("mu_$(j==1 ? "a" : "b")_c.$k")),names))]
                    for j in 1:2, k in 1:2]
                # Unequal level counts make using population variance instead
                # of sample variance observably wrong in the normalized prior.
                dummy_variances = [2*4/(6*5), 1*5/(6*5)]
                contrast_sd = [c.reference[j]*sqrt(c.phi[j==1 ? 3+k : 6+k]*c.r2/
                    ((1-c.r2)*dummy_variances[k])) for j in 1:2, k in 1:2]
                mu_a .+= [k==1 ? 0. : contrasts[1,k-1] for k in data.c]
                mu_b .+= [k==1 ? 0. : contrasts[2,k-1] for k in data.c]
            end
            (; c..., mu_a, mu_b, slope_sd, contrasts, contrast_sd)
        end
        function oracle(u)
            c = components(u)
            sum(logpdf.(Normal(),c.beta[[1,3]])) +
                sum(logpdf.(Normal.(0,c.slope_sd),c.beta[[2,4]])) +
                sum(logpdf.(Normal.(0,c.contrast_sd),c.contrasts)) +
                (latent ? sum(logpdf.(Normal(),c.reference)) : 0.) +
                (partial ? logpdf(Normal(),c.free) : 0.) +
                sum(logpdf.(Normal(),c.z)) + log(3/4) + log1p(-c.rho^2) +
                logpdf(Beta(0.4*3,(1-0.4)*3),c.r2) +
                logpdf(Dirichlet(fill(1.2,nphi)),c.phi) +
                logpdf(Exponential(0.8),c.scales[1]) +
                logpdf(Exponential(0.9),c.scales[2]) + c.jac +
                sum(logpdf.(Normal.(c.mu_a,c.scales[1]),data.ya)) +
                sum(logpdf.(Normal.(c.mu_b,c.scales[2]),data.yb))
        end
        stan = consumer_stan(brmi,"r2d2-$(mode)"; mod=PublicR2D2)
        stan_names = BridgeStan.param_names(stan.model)
        function physical(u)
            c = components(u)
            values = Dict("scale_a"=>c.scales[1], "scale_b"=>c.scales[2],
                "b_shared_subject_r2d2_1_R2"=>c.r2,
                "b_shared_subject_L.1.1"=>1., "b_shared_subject_L.2.1"=>c.rho,
                "b_shared_subject_L.1.2"=>0., "b_shared_subject_L.2.2"=>c.L[2,2])
            for j in 1:2, k in 1:2
                values["pop_mu_$(j==1 ? "a" : "b")_beta_pop.$k"] = c.beta[2(j-1)+k]
            end
            for j in 1:3, k in 1:2
                values["b_shared_subject_z_flat.$(2(j-1)+k)"] = c.z[j,k]
            end
            for j in 1:nphi
                values["b_shared_subject_r2d2_1_phi.$j"] = c.phi[j]
            end
            if latent
                for j in 1:2
                    values["b_shared_subject_r2d2_1_ref_$j"] = c.reference[j]
                end
            elseif partial
                values["b_shared_subject_r2d2_free_tau_2"] = c.free
            end
            if categorical
                for j in 1:2, k in 1:2
                    name = design ? "pop_mu_$(j==1 ? "a" : "b")_beta_pop.$(k+2)" :
                        "cat_mu_$(j==1 ? "a" : "b")_c_beta.$k"
                    values[name] = c.contrasts[j,k]
                end
            end
            @test Set(keys(values)) == Set(stan_names)
            [values[n] for n in stan_names]
        end
        function stan_oracle(u)
            su = BridgeStan.param_unconstrain(stan.model, physical(u))
            value, _ = BridgeStan.log_density_gradient!(stan.model, su, similar(su);
                propto=false, jacobian=false)
            value+components(u).jac
        end
        ext = Base.get_extension(BRM,:BayesianRegressionModelsReactiveKernelsExt)
        translated = ext._rk_translated_plan(backend.plan)
        prior = prepare_query(backend.model,translated,:prior)
        jacobian = prepare_query(backend.model,translated,:log_jacobian)
        pointwise = prepare_query(backend.model,translated,:pointwise)
        retyped = BRM._RKEmittedProgram(
            [Meta.parse(sprint(Base.show_unquoted,d)) for d in emitted.defs],
            Meta.parse(source),emitted.bindings)
        replay_plan = ext._rk_translate_from_emitted(backend.plan,retyped)
        replay_model = Base.invokelatest(build_kernel,replay_plan)
        replay = prepare_sampler(replay_model,replay_plan,zeros(length(names));
            backend=AutoEnzyme(;mode=Enzyme.Reverse))
        for u in (zeros(length(names)), fill(.13,length(names)),
                collect(range(-.2,.3;length=length(names))))
            value, gradient = check_consumer_point(problem,u,oracle)
            c = components(u)
            @test jacobian(u) ≈ c.jac atol=2e-11
            @test value ≈ stan_oracle(u) atol=2e-10 rtol=2e-10
            step = 1e-5
            sg = map(eachindex(u)) do j
                plus, minus = copy(u), copy(u)
                plus[j] += step; minus[j] -= step
                (stan_oracle(plus)-stan_oracle(minus))/(2step)
            end
            @test gradient ≈ sg atol=2e-8 rtol=2e-8
            parts = pointwise(u)
            @test parts.ya ≈ logpdf.(Normal.(c.mu_a,c.scales[1]),data.ya)
            @test parts.yb ≈ logpdf.(Normal.(c.mu_b,c.scales[2]),data.yb)
            @test value ≈ prior(u)+jacobian(u)+sum(parts.ya)+sum(parts.yb) atol=2e-11
            rg = similar(u)
            rv, _ = sampler_value_and_gradient!(replay,rg,u)
            @test isequal(rv,value)
            @test rg ≈ gradient atol=2e-12 rtol=2e-12
        end
        @test isequal(data,saved)
    end
end

@stestset "shared R2D2M2 conflicting allocations remain explicit" begin
    for extra_prior in ("effect(mu_a, x) ~ Normal(0, .3)",
            "effect(mu_a, :) ~ r2d2(tau_bsv=.6)",
            "sd(mu_a, shared) ~ Exponential(1.)")
        brmi = PublicR2D2.build(;joint=true,extra_prior)
        @test_throws ErrorException RKBRMI(brmi)
        @test_throws ErrorException SBBRMI(brmi;mod=PublicR2D2)
    end
    brmi = PublicR2D2.build(;extra_prior="effect(mu_a, :) ~ r2d2(tau_bsv=.6)\neffect(mu_b, :) ~ r2d2(tau_bsv=.6)")
    @test_throws ErrorException RKBRMI(brmi)
    @test_throws ErrorException SBBRMI(brmi;mod=PublicR2D2)
end
