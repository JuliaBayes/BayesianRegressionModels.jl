# Public synthetic acceptance for statistical implementations adopted by BRM.
using Test, BayesianRegressionModels, ReactiveKernels, ReactiveKernelsPPL
using Distributions, LinearAlgebra, Statistics, Enzyme, LogDensityProblems
using DifferentiationInterface: AutoEnzyme
include(joinpath(@__DIR__, "testset_filter.jl"))
const BRM = BayesianRegressionModels
const Prep = BRM.StatisticalPreparation

function library_build(name, body, columns)
    mod = Module(gensym(:BRMLibraryTest))
    Core.eval(mod, :(using ReactiveKernelsPPL))
    Core.eval(mod, :(const owned = $(BRM.rkppl_model(name))))
    names = Tuple(keys(columns))
    bound = bind_data(lower_rkppl(body, names; mod, conditioned=(:y,)), columns)
    bound, build_kernel(bound)
end

function library_check(name, body, columns)
    bound, built = library_build(name, body, columns)
    u = built.layout.total <= 1 ? fill(0.13, built.layout.total) :
        collect(range(-0.17, 0.21; length=built.layout.total))
    q = prepare_sampler(built, bound, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
    g = similar(u)
    saved = copy(u)
    value, _ = sampler_value_and_gradient!(q, g, u)
    @test isfinite(value)
    @test isequal(u, saved)
    h = 1e-5
    fd = map(eachindex(u)) do j
        plus, minus = copy(u), copy(u)
        plus[j] += h; minus[j] -= h
        (q(plus) - q(minus)) / (2h)
    end
    @test g ≈ fd atol=2e-5 rtol=2e-5
    # Reparse the adopted definition itself, then use the same public lowering.
    mod = Module(gensym(:BRMRetypedLibrary))
    Core.eval(mod, :(using ReactiveKernelsPPL))
    for helper in (:hsgp_rho_floors, :hsgp_periodic_rho_floor, :hsgp_sqrt_spd,
            :hsgp_periodic_sqrt_spd, :hsgp_grouped_sqrt_spd)
        Core.eval(mod, :(const $helper = $(getproperty(Prep, helper))))
    end
    definition = Meta.parse(sprint(Base.show_unquoted, getproperty(BRM._BRM_STATISTICAL_MODELS, name)))
    Core.eval(mod, Expr(:macrocall, Symbol("@rkppl"), LineNumberNode(0), definition))
    Core.eval(mod, :(const owned = $(getproperty(mod, name))))
    rebound = bind_data(lower_rkppl(Meta.parse(sprint(Base.show_unquoted, body)),
        Tuple(keys(columns)); mod, conditioned=(:y,)), columns)
    rebuilt = build_kernel(rebound)
    @test coordinate_names(rebuilt.layout) == coordinate_names(built.layout)
    for preset in (:sampler, :prior, :likelihood)
        @test isequal(Base.invokelatest(prepare_query(built, bound, preset), u),
            Base.invokelatest(prepare_query(rebuilt, rebound, preset), u))
    end
    bound, built, u
end

@stestset "adopted BRM statistical library" begin
    n = 12
    x = collect(range(-1, 1; length=n))
    g = repeat(["b", "a", "c"], 4)
    s = repeat(["two", "one"], 6)
    base = (; x, g, s, y=fill(0.2, n))
    cases = (
        (:varying_coefs, :(begin r ~ owned(g); y .~ Normal.(r[g], 1) end)),
        (:varying_coefs_correlated, :(begin r ~ owned(g, 2); y .~ Normal.(r[g, 1] .+ x .* r[g, 2], 1) end)),
        (:varying_coefs_centered, :(begin r ~ owned(g); y .~ Normal.(r[g], 1) end)),
        (:varying_coefs_centered_correlated, :(begin r ~ owned(g, 2); y .~ Normal.(r[g, 1] .+ x .* r[g, 2], 1) end)),
        (:varying_stratified, :(begin r ~ owned(g, s); y .~ Normal.(r, 1) end)),
        (:varying_stratified_correlated, :(begin r ~ owned(g, s, 2); y .~ Normal.(r[:, 1] .+ x .* r[:, 2], 1) end)),
        (:monotonic, :(begin phi ~ Dirichlet([1.0, 1.0]); r ~ owned(c, phi); y .~ Normal.(r, 1) end)),
        (:differenced_ar1, :(begin beta ~ Beta(2, 2); sigma ~ Exponential(1); r ~ owned(beta, sigma); y .~ Normal.(r, 1) end)),
        (:r2d2_coefs, :(begin r ~ owned(X, [1.0, 1.0]); mu = X * r; y .~ Normal.(mu, 1) end)),
        (:horseshoe_coefs, :(begin r ~ owned(X); mu = X * r; y .~ Normal.(mu, 1) end)),
        (:penalized_smooth, :(begin r ~ owned(X, Z); y .~ Normal.(r, 1) end)),
        (:t2_smooth, :(begin r ~ owned(X, Zrr, Zrn, Znr); y .~ Normal.(r, 1) end)),
        (:hsgp_effect, :(begin r ~ owned(PHI, lambda); y .~ Normal.(r, 1) end)),
        (:hsgp_periodic_effect, :(begin r ~ owned(PHI, harmonics); y .~ Normal.(r, 1) end)),
        (:hsgp_grouped_effect, :(begin r ~ owned(PHI, lambda, g); y .~ Normal.(r, 1) end)),
        (:ordered_logistic, :(begin y ~ owned(x) end)),
    )
    for (name, body) in cases
        filter = get(ENV, "BRM_LIBRARY_FILTER", "")
        isempty(filter) || occursin(filter, string(name)) || continue
        @testset "$name" begin
            columns = Dict{Symbol,Any}(pairs(base))
            if name in (:r2d2_coefs, :horseshoe_coefs)
                columns[:X] = hcat(x, sin.(2x))
            elseif name === :penalized_smooth
                columns[:X], columns[:Z] = Prep.tps_basis(x; k=5)
            elseif name === :t2_smooth
                columns[:X], columns[:Zrr], columns[:Zrn], columns[:Znr] = Prep.t2_basis(x, sin.(2x); k=(3, 3))
            elseif name === :hsgp_periodic_effect
                columns[:PHI], columns[:harmonics] = Prep.hsgp_periodic_basis(x; k=3, period=2.0)
            elseif name in (:hsgp_effect, :hsgp_grouped_effect)
                columns[:PHI], columns[:lambda] = Prep.hsgp_basis(x; k=3,
                    by=name === :hsgp_grouped_effect ? g : nothing)
            elseif name === :ordered_logistic
                columns[:y] = repeat([1, 2, 3], 4)
            elseif name === :monotonic
                columns[:c] = repeat([1, 2, 3], 4)
            end
            bound, built, u = library_check(name, body, columns)
            if name === :varying_coefs_correlated
                nt = constrain(built.layout, u)
                B = nt.r.z * (Diagonal(nt.r.sd) * nt.r.L)'
                gi = [findfirst(==(v), sort(unique(g))) for v in g]
                mu = B[gi, 1] .+ x .* B[gi, 2]
                @test Base.invokelatest(prepare_query(built, bound, :likelihood), u) ≈ sum(logpdf.(Normal.(mu, 1), columns[:y]))
                prior = sum(logpdf.(truncated(Normal(), 0, Inf), nt.r.sd)) +
                    logpdf(LKJCholesky(2, 1), Cholesky(LowerTriangular(nt.r.L))) + sum(logpdf.(Normal(), nt.r.z))
                @test Base.invokelatest(prepare_query(built, bound, :prior), u) ≈ prior
            end
        end
    end
end
