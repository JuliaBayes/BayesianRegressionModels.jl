# test/exported_families.jl — `using BayesianRegressionModels` binds every
# distribution family SBBRMI lowers, so a model module needs no
# `using Distributions` on top (decision `0w84aut`; snag
# `using-bayesianre-4995ff0b`).
#
# Run: julia --startup-file=no --project=test test/exported_families.jl

using Test
using BayesianRegressionModels
using Distributions: Distributions
import StanBlocks

const BRM = BayesianRegressionModels

# A concrete Distribution subtype. Walks the supertype chain by name, since
# subtyping answers `LocationScale <: Distribution` false for its dependent
# parameter bounds.
function is_family(F)
    T = Base.unwrap_unionall(F)
    T isa DataType && !isabstracttype(T) || return false
    while T !== Any
        T.name === Base.typename(Distributions.Distribution) && return true
        T = supertype(T)
    end
    false
end

# The family types a lowering function dispatches on through `::Type{<:F}`.
function lowered_families(f)
    families = Set{Any}()
    for m in methods(f), q in Base.unwrap_unionall(m.sig).parameters
        p = Base.unwrap_unionall(q)
        p isa DataType && p.name === Base.typename(Type) || continue
        F = p.parameters[1]
        F = F isa TypeVar ? F.ub : F
        F isa Type && is_family(F) && push!(families, F)
    end
    families
end

# Every SBBRMI family-lowering seam: the shared Stan name table, the scalar
# and vector prior emitters and the likelihood emitter.
const LOWERED_FAMILIES = collect(union(
    lowered_families(BRM._sb_stan_dist_name),
    lowered_families(BRM._sb_emit_prior!),
    lowered_families(BRM._sb_emit_vector_prior!),
    lowered_families(BRM._sb_lik_family!),
))

# Compared as bindings, not names: `Categorical` is an alias whose `nameof`
# is `DiscreteNonParametric`.
const EXPORTED = [getfield(BRM, n) for n in names(BRM) if isdefined(BRM, n)]

@testset "every lowered family is exported" begin
    # The reflection must see each seam; a renamed seam would read as empty.
    for F in (Distributions.LogNormal, Distributions.Uniform,
              Distributions.Dirichlet, Distributions.LocationScale,
              Distributions.Categorical, BRM.NegativeBinomial2)
        @test F in LOWERED_FAMILIES
    end
    @test isempty(filter(F -> !(F in EXPORTED), LOWERED_FAMILIES))
end

# The reporter's shape: a module that only does `using BayesianRegressionModels`,
# a bounded Uniform scalar prior and a LogNormal observation in a plate cell.
module OnlyBRMFamilies
using BayesianRegressionModels
const data = (;
    time=[[0.5, 1.0, 2.0], [0.5, 1.5]],
    dv=[[1.2, 0.9, 0.5], [1.4, 0.7]],
)
build() = @brm data begin
    tv ~ Uniform(log(1), log(1000))
    lnorm_sd ~ Exponential(1)
    @plate for i in eachindex(time)
        cp[i] = exp(tv) .* exp.(-0.1 .* time[i])
        dv[i] ~ LogNormal(log.(cp[i]), lnorm_sd)
    end
end
end

@testset "a model module needs only using BayesianRegressionModels" begin
    sb = SBBRMI(OnlyBRMFamilies.build(); mod=OnlyBRMFamilies)
    code = stan_code(sb)
    @test occursin("lognormal", code)
    @test occursin(r"real<lower=0(\.0)?, upper=6\.9", code)
    @test occursin("uniform", code)
    @test StanBlocks.stan.compiles(sb.model)
end

# Importing Distributions as well stays unambiguous: the bindings are identical.
module BRMAndDistributions
using BayesianRegressionModels, Distributions
const resolved = (LogNormal, Uniform, Gamma, Dirichlet, LocationScale)
end

@testset "using both BRM and Distributions" begin
    @test BRMAndDistributions.resolved === (Distributions.LogNormal,
        Distributions.Uniform, Distributions.Gamma, Distributions.Dirichlet,
        Distributions.LocationScale)
end
