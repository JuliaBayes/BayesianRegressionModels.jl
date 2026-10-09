# A caller value must remain the input to its authored prior when a component
# allocates a draw with the same name. Check independent normalized laws,
# every native Reverse coordinate and complete printed-source replay.
using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using Distributions, Enzyme, Statistics
using DifferentiationInterface: AutoEnzyme
include(joinpath(@__DIR__, "testset_filter.jl"))
include(joinpath(@__DIR__, "rk_source_roundtrip.jl"))
const BRM = BayesianRegressionModels
const EXT = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
const DATA = (; g=[1,1,2,2], x=[0.1,0.4,0.6,1.0], y=[0.2,-0.1,0.3,0.4])
const CASES = (
    "group caller tau" => @brm(DATA, begin
        tau ~ Exponential(1.0)
        mu ~ 1 + (1 | p | g)
        sd(:,p) ~ Exponential(tau)
        y ~ Normal(mu,tau)
    end),
    "HSGP caller sigma" => @brm(DATA, begin
        sigma ~ Exponential(1.0)
        mu ~ 1 + hsgp(x;k=3)
        sd(:,hsgp(x)) ~ Exponential(sigma)
        y ~ Normal(mu,1.0)
    end),

)

@stestset "component priors preserve colliding caller inputs" begin
    saved = deepcopy(DATA)
    for (label, brmi) in CASES
        @testset "$label" begin
            backend = check_rk_source_roundtrip(RKBRMI(brmi);
                ad_backend=AutoEnzyme(;mode=Enzyme.Reverse))
            layout = backend.model.layout
            bound = EXT._rk_translated_plan(backend.plan)
            prior = prepare_query(backend.model,bound,:prior)
            query = prepare_sampler(backend.model,bound,zeros(layout.total);
                backend=AutoEnzyme(;mode=Enzyme.Reverse))
            # Independent eigenbasis and spectral weights on the stated k=3
            # centered domain, without BRM's fitted basis or helper values.
            xc = DATA.x .- mean(DATA.x)
            width = 1.5maximum(abs,xc)
            omega2 = [(j*pi/(2width))^2 for j in 1:3]
            PHI = [sin(sqrt(w)*(x+width))/sqrt(width) for x in xc,w in omega2]
            floor = 4width/pi*sqrt(log(100)/(3^2-1))
            function law(u)
                p = constrain(layout,u)
                if label == "group caller tau"
                    b = p.b_p_g
                    eta = p.mu_Intercept .+ (b.z .* b.tau[1])[DATA.g]
                    lp = logpdf(Exponential(),p.tau) +
                        logpdf(Exponential(p.tau),only(b.tau)) +
                        sum(logpdf.(Normal(),b.z)) + logpdf(Normal(),p.mu_Intercept)
                    jac = log(p.tau) + log(only(b.tau))
                    response_scale = p.tau
                elseif label == "HSGP caller sigma"
                    h = p.hsgp_x
                    weights = h.sigma*sqrt(h.rho_iso*sqrt(2pi)) .* exp.(-h.rho_iso^2 .* omega2 ./ 4)
                    eta = p.mu_Intercept .+ PHI*(weights .* h.beta_raw)
                    lp = logpdf(Exponential(),p.sigma) + logpdf(Exponential(p.sigma),h.sigma) +
                        logpdf(LogNormal(),h.rho_iso) + sum(logpdf.(Normal(),h.beta_raw)) +
                        logpdf(Normal(),p.mu_Intercept)
                    jac = log(p.sigma) + log(h.sigma) + log(h.rho_iso-floor)
                    response_scale = 1.0

                end
                (;prior=lp, posterior=lp+jac+sum(logpdf.(Normal.(eta,response_scale),DATA.y)))
            end
            for u in (zeros(layout.total),collect(range(-0.2,0.3;length=layout.total)))
                @test Base.invokelatest(prior,u) ≈ law(u).prior atol=2e-12 rtol=2e-12
                g = similar(u)
                value,_ = sampler_value_and_gradient!(query,g,u)
                @test value ≈ law(u).posterior atol=2e-12 rtol=2e-12
                h = cbrt(eps(Float64))
                finite = map(eachindex(u)) do j
                    plus,minus = copy(u),copy(u)
                    plus[j]+=h; minus[j]-=h
                    (law(plus).posterior-law(minus).posterior)/(2h)
                end
                @test g ≈ finite atol=2e-7 rtol=2e-7
            end
        end
    end
    @test isequal(DATA,saved)
end
