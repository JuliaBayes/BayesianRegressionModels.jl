# Run: julia --startup-file=no --project=test test/description_cell_combinators.jl
#
# A `@plate for` cell resolves StanBlocks' builtin vocabulary first, so an
# in-cell `censored`/`truncated`/`weighted`/`interval_censored` is StanBlocks'
# distribution combinator, not the formula-level wrapper. Both spellings it
# accepts are built-in scientific descriptions: the call form, whose family is
# an argument child, and the token form, whose family is a `:family` child over
# the token's arguments (snag in-cell-censored-aaf99cff).
using Test, BayesianRegressionModels
using Distributions: Exponential, Normal
import StanBlocks

const combinator_data = (;
    subject=[1, 2], obs_subject=[1, 2, 1],
    time=[0.2, 0.3, 0.8], lloq=[0.1, 0.1, 0.1], weight=[1.0, 0.5, 0.25],
    lo=[0.0, 0.1, 0.2], hi=[0.5, 0.6, 0.9], y=[[0.4, 1.2], [0.6]],
)
const SB = StanBlocks.stan.builtin

describe(builder, data=combinator_data) =
    brm_description(SBBRMI(builder(data); mod=@__MODULE__, total_groups=()))
components(r, f) = filter(c -> c.callable === f, brm_description_components(r))
covered(r, c) = only(filter(x -> x.id == c.id, r.coverage)).status === :covered

const call_censored = @brm begin
    sigma_add ~ Exponential(1)
    sigma_prop ~ Exponential(1)
    eta ~ 1 + (1 | p | subject)
    obs = groups(obs_subject => subject)
    @plate for i in eachindex(subject)
        loc[i] = exp(eta[i]) .+ time[obs[i]]
        y[i] ~ censored(Normal(loc[i], addprop(loc[i], sigma_add, sigma_prop));
            lower=lloq[obs[i]])
    end
end

@testset "in-cell call-form censored is a built-in description" begin
    for data in (combinator_data, Base.structdiff(combinator_data, NamedTuple{(:y,)}))
        r = describe(call_censored, data)
        @test r.complete
        @test isempty(r.diagnostics)
        c = only(components(r, SB.censored))
        @test SB.censored !== censored
        @test covered(r, c)
        @test only(c.children).callable === Normal
        @test covered(r, only(c.children))
        cases = only(filter(e -> startswith(e, "p(y)=\\begin{cases}"), r.equations))
        @test occursin("F(", cases) && occursin("&y=", cases)
        # Keyword-only bounds: no upper row is read from the family's arguments.
        @test !occursin("1-F(", cases)
    end
end

@testset "in-cell censoring shares the formula wrapper's law" begin
    # The formula wrapper over the collected plate result and per-subject data.
    formula = @brm begin
        sigma_add ~ Exponential(1)
        sigma_prop ~ Exponential(1)
        eta ~ 1 + (1 | p | subject)
        obs = groups(obs_subject => subject)
        @plate for i in eachindex(subject)
            loc[i] = exp(eta[i]) .+ time[obs[i]]
        end
        y ~ censored(Normal(loc, addprop(loc, sigma_add, sigma_prop)); lower=lloq)
    end
    data = merge(combinator_data, (; lloq=[[0.1, 0.1], [0.1]]))
    top = describe(formula, data)
    cell = describe(call_censored)
    @test top.complete && cell.complete
    prose(r) = filter(p -> startswith(p, "Censoring contributes"), r.prose)
    @test !isempty(prose(top)) && prose(top) == prose(cell)
end

@testset "token-form combinators describe their family as a child" begin
    censored_token = @brm begin
        eta ~ 1 + (1 | p | subject)
        obs = groups(obs_subject => subject)
        @plate for i in eachindex(subject)
            loc[i] = exp(eta[i]) .+ time[obs[i]]
            y[i] ~ censored(normal, loc[i], 0.3; lower=lloq[obs[i]])
        end
    end
    r = describe(censored_token)
    @test r.complete
    c = only(components(r, SB.censored))
    family = only(filter(x -> last(x.id) === :family, c.children))
    @test family.callable === SB.normal
    @test family.id == (c.id..., :family)
    @test length(family.arguments) == 2 && last(family.arguments) == 0.3
    @test covered(r, c) && covered(r, family)
    cases = only(filter(e -> startswith(e, "p(y)=\\begin{cases}"), r.equations))
    # The token's arguments are not bounds: 0.3 is the SD, never an upper limit.
    @test !occursin("1-F(", cases) && !occursin("0.3", cases)

    # A family hook sees the family by its actual callable and token arguments.
    hook = SB.normal => x -> BRMDescriptionFragment(prose=("family hook",), covers=(x.id,))
    hooked = brm_description(SBBRMI(censored_token(combinator_data); mod=@__MODULE__,
        total_groups=()); hooks=(hook,))
    @test hooked.complete
    @test "family hook" in hooked.prose
end

@testset "in-cell truncated, weighted and interval_censored" begin
    truncated_call = @brm begin
        eta ~ 1 + (1 | p | subject)
        obs = groups(obs_subject => subject)
        @plate for i in eachindex(subject)
            loc[i] = exp(eta[i]) .+ time[obs[i]]
            y[i] ~ truncated(Normal(loc[i], 0.3); lower=0.0)
        end
    end
    weighted_token = @brm begin
        eta ~ 1 + (1 | p | subject)
        obs = groups(obs_subject => subject)
        @plate for i in eachindex(subject)
            loc[i] = exp(eta[i]) .+ time[obs[i]]
            y[i] ~ weighted(normal, weight[obs[i]], loc[i], 0.3)
        end
    end
    interval_token = @brm begin
        eta ~ 1 + (1 | p | subject)
        obs = groups(obs_subject => subject)
        @plate for i in eachindex(subject)
            loc[i] = exp(eta[i]) .+ time[obs[i]]
            y[i] ~ interval_censored(normal, lo[obs[i]],
                hi[obs[i]], loc[i], 0.3)
        end
    end
    r = describe(truncated_call)
    @test r.complete
    @test covered(r, only(components(r, SB.truncated)))
    @test any(e -> occursin("\\le y\\le \\infty", e), r.equations)

    r = describe(weighted_token)
    @test r.complete
    w = only(components(r, SB.weighted))
    @test only(filter(x -> last(x.id) === :family, w.children)).callable === SB.normal
    @test any(e -> startswith(e, "\\log\\mathcal L=\\sum_j "), r.equations)

    r = describe(interval_token)
    @test r.complete
    i = only(components(r, SB.interval_censored))
    family = only(filter(x -> last(x.id) === :family, i.children))
    @test family.callable === SB.normal && length(family.arguments) == 2
    @test any(e -> startswith(e, "\\mathcal L=F("), r.equations)
end

@testset "weighted over an in-cell censored law keeps its bounds in both spellings" begin
    weighted_call = @brm begin
        eta ~ 1 + (1 | p | subject)
        obs = groups(obs_subject => subject)
        @plate for i in eachindex(subject)
            loc[i] = exp(eta[i]) .+ time[obs[i]]
            y[i] ~ weighted(censored(Normal(loc[i], 0.3); lower=lloq[obs[i]]),
                weight[obs[i]])
        end
    end
    weighted_token = @brm begin
        eta ~ 1 + (1 | p | subject)
        obs = groups(obs_subject => subject)
        @plate for i in eachindex(subject)
            loc[i] = exp(eta[i]) .+ time[obs[i]]
            y[i] ~ weighted(censored, weight[obs[i]], normal, loc[i], 0.3;
                lower=lloq[obs[i]])
        end
    end
    cases(r) = only(filter(e -> startswith(e, "p(y)=\\begin{cases}"), r.equations))
    call, token = describe(weighted_call), describe(weighted_token)
    @test call.complete && token.complete
    # The token's `lower=` rides on `weighted` but bounds its censored family.
    @test occursin("F(", cases(token)) && occursin("&y=", cases(token))
    @test cases(token) == cases(call)
    @test any(e -> startswith(e, "\\log\\mathcal L=\\sum_j "), token.equations)
    w = only(components(token, SB.weighted))
    c = only(filter(x -> last(x.id) === :family, w.children))
    @test c.callable === SB.censored && haskey(c.keywords, :lower)
    family = only(filter(x -> last(x.id) === :family, c.children))
    @test family.callable === SB.normal && length(family.arguments) == 2
    @test covered(token, c) && covered(token, family)
end
