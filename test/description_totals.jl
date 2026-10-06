using Test, BayesianRegressionModels, Distributions
import StanBlocks

const BRM = BayesianRegressionModels
const TOTALS_DATA = (; x=[-2.0,-0.5,0.7,2.5,3.0,4.2,1.0,-1.0,0.2,1.5],
    g=[1,1,2,2,3,3,4,4,5,5], y=[0.4,0.1,1.1,1.7,2.4,2.8,1.9,0.3,1.2,0.8])

# A consumer scalar family inside a BRM-generated vector prior.
positive_factory() = Exponential(1.0)
BRM.brm_distribution_type(::typeof(positive_factory)) = Exponential
BRM._sb_stan_dist_name(::typeof(positive_factory)) = :consumer_positive
StanBlocks.@deffun begin
    @lpxf consumer_positive_lpdf(y::real)::real = y >= 0.0 ? -y : negative_infinity()
    consumer_positive_rng()::real = exponential_rng(1.0)
end

totals_prior(r, id) = only(p for p in r.priors if p.id == id)
totals_law(p) = (p.distribution.callable, p.distribution.arguments, p.support)

# The same declaration under automatic exact totals and conventional emission.
function totals_pair(m)
    auto = SBBRMI(m; mod=@__MODULE__)
    conventional = SBBRMI(m; mod=@__MODULE__, total_groups=())
    @test length(total_effect_blocks(auto)) == 1
    @test isempty(total_effect_blocks(conventional))
    (; auto, r=brm_description(auto), c=brm_description(conventional))
end

@testset "exact totals describe the authored hierarchy" begin
    m = @brm TOTALS_DATA begin
        mu ~ 1 + (1 | shift | g)
        sd(:, shift) ~ Normal(0.0, 0.5)
        effect(mu, Intercept) ~ Normal(0.0, 2.0)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    (; r, c) = totals_pair(m)
    @test r.complete && c.complete
    population = totals_prior(r, (:population, :mu, :Intercept))
    @test totals_law(population) == totals_law(totals_prior(c, (:population, :mu, :Intercept)))
    @test population.source.integrated == (:total_effect, :mu)
    sd = totals_prior(r, (:random_effect, :shift, :g, :sd, 1))
    @test totals_law(sd) == (Normal, (0.0, 0.5), (; lower=0.0))
    @test totals_law(sd) == totals_law(totals_prior(c, (:random_effect, :shift, :g, :sd, 1)))
    @test sd.source.kind === :selector
    totals = totals_prior(r, (:total_effect, :mu, :totals))
    @test totals.distribution.callable === brm_total
    @test totals.source.integrated == ((:population, :mu, :Intercept),)
    # Generated carrier names are not logical prior identities.
    @test !any(p -> first(p.id) === :parameter && p.id[2] in (:total_mu, :total_scale_mu), r.priors)
    root = only(x for x in r.components if x.id == (:total_effect, :mu))
    @test brm_description_prior_references(root) ==
        ((:population, :mu, :Intercept), (:total_effect, :mu, :totals))
    random = only(x for x in r.components if x.kind === :random_effect)
    @test random.keywords.total == (:total_effect, :mu)
    @test brm_description_prior_references(random) == ((:random_effect, :shift, :g, :sd, 1),)
    md = brm_description_markdown(r)
    @test startswith(md, "Complete model description.")
    @test !occursin("brm_vector_prior", md) && !occursin("total_scale", md) && !occursin("total\\_scale", md)
    @test occursin("\\mathbf t_{i}=A\\boldsymbol\\beta+\\mathbf b_{i}", md)
    @test occursin("A=\\begin{bmatrix}1.0\\end{bmatrix},\\quad m=\\left[0.0\\right],\\quad q=\\left[0.25\\right]", md)
    @test occursin("\\operatorname{diag}\\left(\\left[\\mathrm{SD}_{1}\\right]\\right)", md)
    # The predictor term and the block's deviation equation share one block.
    @test occursin("\\mathbf b_{1,g_j}", md) && occursin("\\mathbf b_{1,i}\\sim\\mathcal N_{1}", md)
end

@testset "default random intercept keeps conventional identities and priors" begin
    m = @brm TOTALS_DATA begin
        mu ~ 1 + (1 | g)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    (; r, c) = totals_pair(m)
    @test r.complete
    key = (:random_effect, (:independent, 1), :g)
    sd = totals_prior(r, (key..., :sd, 1))
    @test totals_law(sd) == (LogNormal, (0.0, 1.0), (; lower=0.0))
    @test totals_law(sd) == totals_law(totals_prior(c, (key..., :sd, 1)))
    @test sd.source.kind === :default
    @test totals_law(totals_prior(r, (:population, :mu, :Intercept))) ==
        totals_law(totals_prior(c, (:population, :mu, :Intercept)))
    # Coordinates sampled only by the conventional program are not inventoried.
    @test !any(p -> p.id in ((key..., :log_scale), (key..., :standardized_deviations)), r.priors)
end

@testset "remaining population columns keep their own priors" begin
    m = @brm TOTALS_DATA begin
        mu ~ 1 + x + (1 | g)
        effect(mu, Intercept) ~ Normal(5.0, 2.0)
        effect(mu, x) ~ Normal(0.0, 0.1)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    (; auto, r, c) = totals_pair(m)
    @test occursin("pop_mu_beta_pop ~ normal([0.0]', [0.1]');", stan_code(auto))
    @test r.complete
    for label in (:Intercept, :x)
        @test totals_law(totals_prior(r, (:population, :mu, label))) ==
            totals_law(totals_prior(c, (:population, :mu, label)))
    end
    @test totals_prior(r, (:population, :mu, :x)).distribution.arguments == (0.0, 0.1)
    @test !haskey(totals_prior(r, (:population, :mu, :x)).source, :integrated)
    @test totals_prior(r, (:population, :mu, :Intercept)).source.integrated == (:total_effect, :mu)
end

@testset "independent total columns keep separate block identities" begin
    m = @brm TOTALS_DATA begin
        mu ~ 1 + x + (1 + x || g)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    (; r, c) = totals_pair(m)
    @test r.complete
    blocks(d) = [x.id for x in d.components if x.kind === :random_effect]
    @test blocks(r) == blocks(c) ==
        [(:random_effect, (:independent, 1), :g), (:random_effect, (:independent, 2), :g)]
    for key in blocks(r)
        @test totals_law(totals_prior(r, (key..., :sd, 1))) == totals_law(totals_prior(c, (key..., :sd, 1)))
    end
    md = brm_description_markdown(r)
    @test occursin("\\mathbf b_{i}=\\left[\\mathbf b_{1,i},\\mathbf b_{2,i}\\right]^{\\mathsf T}", md)
    @test occursin("A=\\begin{bmatrix}1.0&0.0\\\\0.0&1.0\\end{bmatrix}", md)
end

@testset "Student-t and flat population priors" begin
    t = @brm TOTALS_DATA begin
        mu ~ 1 + (1 | g)
        effect(mu, Intercept) ~ TDist(3)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    r = brm_description(SBBRMI(t; mod=@__MODULE__))
    @test r.complete
    @test totals_prior(r, (:population, :mu, :Intercept)).distribution.callable === TDist
    mixture = totals_prior(r, (:total_effect, :mu, :scale_mixture))
    @test mixture.distribution.callable === StanBlocks.stan.builtin.gamma
    @test mixture.distribution.arguments == ((1.5,), (1.5,))
    @test totals_prior(r, (:total_effect, :mu, :totals)).distribution.keywords.mixture_coefficients == (1,)
    @test any(p -> occursin("Gaussian scale-mixture", p), r.prose)
    flat = @brm TOTALS_DATA begin
        mu ~ 1 + (1 | g)
        effect(mu, Intercept) ~ Flat()
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    (; r, c) = totals_pair(flat)
    @test r.complete && c.complete
    @test totals_law(totals_prior(r, (:population, :mu, :Intercept))) ==
        totals_law(totals_prior(c, (:population, :mu, :Intercept)))
    @test any(p -> occursin("improper flat prior and is integrated against constant density", p), r.prose)
end

@testset "generated vector priors decompose into coordinate families" begin
    m = @brm TOTALS_DATA begin
        mu ~ 1 + s(x)
        sd(:, s(x)) ~ Exponential(2)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    r = brm_description(SBBRMI(m; mod=@__MODULE__, total_groups=()))
    @test r.complete
    p = totals_prior(r, (:parameter, :s_x, :sd_pen))
    coordinates = [x for x in brm_description_components(p.distribution)
                   if length(x.id) > 2 && x.id[end-1] === :coordinate]
    @test only(coordinates).callable === StanBlocks.stan.builtin.exponential
    @test only(coordinates).arguments == (0.5,)
    md = brm_description_markdown(r)
    @test occursin("coordinatewise `exponential`", md)
    @test occursin("\\operatorname{Exponential}_{\\mathrm{rate}}(0.5)", md)
    @test !occursin("\\mathrm{brm\\_vector\\_prior", md)

    # A consumer family remains an explicit gap with its actual callable.
    custom = @brm TOTALS_DATA begin
        mu ~ 1 + s(x)
        sd(:, s(x)) ~ positive_factory()
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    sb = SBBRMI(custom; mod=@__MODULE__, total_groups=())
    gap = brm_description(sb)
    @test !gap.complete
    @test only(gap.diagnostics) ==
        "No scientific description for consumer_positive at (:prior, :parameter, :s_x, :sd_pen, :coordinate, 1)"
    hooked = brm_description(sb; hooks=(consumer_positive => c -> BRMDescriptionFragment(covers=(c.id,)),))
    @test hooked.complete
end

@testset "sum-to-zero blocks retain their explicit gap" begin
    m = @brm TOTALS_DATA begin
        mu ~ 1 + (1 | g)
        sigma ~ Exponential(1)
        y ~ Normal(mu, sigma)
    end
    r = brm_description(SBBRMI(m; mod=@__MODULE__, s2z_groups=[:g], s2z_rho=0.0))
    @test !r.complete
    @test r.diagnostics == ("No scientific description for brm_s2z_contrast at (:prior, :parameter, :s2z_contrast_mu)",)
end

@testset "exact totals feeding a kernel cell" begin
    builder = @brm begin
        sigma ~ Exponential(1)
        log_CL ~ 1 + weight + (1 | p | subject)
        sd(:, p) ~ Normal(0.0, 0.5)
        effect(log_CL, Intercept) ~ Normal(0.0, 2.0)
        pred ~ kernel(ragged(pk_time, pk_subject), log_CL) do ts, lCL
            exp(-exp(lCL) .* ts)
        end
        ragged(pk_y, pk_subject) ~ Normal(pred, sigma)
    end
    data = (; subject=["s1", "s2", "s3"], weight=[60.0, 75.0, 90.0],
        pk_subject=["s1", "s1", "s2", "s2", "s3", "s3"], pk_time=[0.5, 1.0, 0.5, 1.0, 0.5, 1.0],
        pk_y=[0.7, 0.5, 0.6, 0.4, 0.8, 0.6])
    (; r, c) = totals_pair(builder(data))
    @test r.complete && c.complete
    for id in ((:population, :log_CL, :Intercept), (:population, :log_CL, :weight), (:random_effect, :p, :subject, :sd, 1))
        @test totals_law(totals_prior(r, id)) == totals_law(totals_prior(c, id))
    end
    @test totals_prior(r, (:population, :log_CL, :Intercept)).source.integrated == (:total_effect, :log_CL)
end
