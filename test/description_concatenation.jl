# Run: julia --startup-file=no --project=test test/description_concatenation.jl
#
# SLIC's `vcat` is Julia's `vcat` in two roles. As the distribution of `~`, or a
# prior, it is the block-wise prior family `vcat(prior₁, prior₂, …)`, which
# decomposes into its coordinates' families or stays a gap. As a value in a
# `@plate for` cell, which is Julia source, it concatenates (the Stan backend
# spells it `append_row`), a built-in scientific description. The gather of
# `dose_axis_operation_gather.jl` reads `vcat(0.0, log_F[doses[i]])` this way
# (snag brm-description-8fde84e0).
using Test, BayesianRegressionModels, Distributions
import StanBlocks

const SB = StanBlocks.stan.builtin
components(r, f) = filter(c -> c.callable === f, brm_description_components(r))
covered(r, c) = only(filter(x -> x.id == c.id, r.coverage)).status === :covered

module DescriptionConcatenation
using BayesianRegressionModels, StanBlocks
function padded_term end
const cell=@slic begin
    scales ~ vcat(exponential(1.0), normal(0.0, 1.0; lower=0.0))
    return input .* scales[1] .+ scales[2]
end
function BayesianRegressionModels._sb_submodel_rhs!(stmts,data,target::Symbol,::typeof(padded_term),rhs)
    input=only(getargs(rhs)); key=name(input)
    data[key]=collect(parent(parent(input)))
    push!(stmts,Expr(:call,:~,target,Expr(:call,cell,Expr(:parameters,Expr(:kw,:input,key)))))
    :done
end
end
const padded_term=DescriptionConcatenation.padded_term

const cell_data = (; subject=["a", "b"], dose_subject=["a", "a", "b"],
    log_dose=[0.1, 0.4, 0.2], op_subject=["a", "a", "b", "a", "b"],
    op_dose=[1, 0, 3, 2, 0], amount=[1.0, 0.0, 2.0, 1.5, 0.0],
    obs_subject=["a", "b"], y=[0.3, 0.5])

@testset "a cell's concatenation is a described value" begin
    # One callable: only the position tells the roles apart.
    @test SB.vcat === vcat
    model = @brm cell_data begin
        log_F ~ 0 + log_dose
        log_CL ~ 1 + (1 | p | subject)
        events = groups(op_subject => subject)
        doses = groups(dose_subject => subject)
        obs = groups(obs_subject => subject)
        @plate for i in eachindex(log_CL)
            F = vcat(0.0, log_F[doses[i]])
            slot = something.(indexin(op_dose[events[i]], doses[i]), 0) .+ 1
            dose = sum(amount[events[i]] .* exp.(F[slot]))
            y[obs[i]] ~ normal(dose .* exp(-exp(log_CL[i])), 0.3)
        end
    end
    r = brm_description(SBBRMI(model; mod=@__MODULE__, total_groups=()))
    @test r.complete
    @test isempty(r.diagnostics)
    c = only(components(r, vcat))
    @test c.provenance.concatenation
    @test covered(r, c)
    math = brm_description_math(c, c)
    @test startswith(math, "\\left[0.0;\\ ") && endswith(math, "\\right]")
    @test occursin("\\mathrm{log\\_F}", math)
    @test any(p -> startswith(p, "Concatenation lays its arguments end to end"), r.prose)
end

@testset "an included model's `~ vcat(…)` stays the prior family" begin
    data = (; input=[0.1, 0.4, 0.8], y=[0.2, 0.5, 0.9])
    model = @brm data begin
        mu ~ padded_term(input)
        y ~ Normal(mu, 1)
    end
    r = brm_description(SBBRMI(model; mod=@__MODULE__, total_groups=()))
    families = components(r, vcat)
    @test !isempty(families)
    @test all(c -> [a.callable for a in c.arguments] == [SB.exponential, SB.normal], families)
    @test !any(c -> get(c.provenance, :concatenation, false), families)
    @test !any(p -> startswith(p, "Concatenation lays its arguments end to end"), r.prose)
end
