# A `_sb_submodel_rhs!` hook may store and read data under the name of a column
# the same model uses as a ranef grouping factor, in any formula order. BRM reads
# the grouping column only through `<g>_idx` / `n_<g>` and drops the raw column
# from the Stan data only when no emitted statement reads it
# (`_sb_settle_group_columns!`; snag `submodel-hook-da-23edf0b4`).
using Test, BayesianRegressionModels, Distributions
import StanBlocks
import BayesianRegressionModels: _sb_submodel_rhs!
using BayesianRegressionModels: name, getargs

module HookDataOwnership
using BayesianRegressionModels, StanBlocks
function group_term end
const cell = @slic begin
    a ~ normal(0.0, 1.0)
    return a .* z[subjects]
end
end
const group_term = HookDataOwnership.group_term

# How the hook stores the grouping column its statement reads:
# `:codes` integer codes under the column's own name, `:raw` the raw column
# object itself, `:none` nothing (it relies on the raw column BRM was given),
# `:per_target` integer codes under `subjects_<target>`.
const STORE = Ref(:codes)
# Extra data the hook writes before emitting, to probe BRM-derived names.
const EXTRA = Ref{Any}(nothing)

_codes(raw::AbstractVector{<:Integer}) = Int.(raw)
_codes(raw) = [parse(Int, s[2:end]) for s in raw]

function _sb_submodel_rhs!(stmts, data, target::Symbol, ::typeof(group_term), rhs)
    sa, za = getargs(rhs)
    raw = parent(parent(sa))
    key = STORE[] === :per_target ? Symbol(name(sa), :_, target) : name(sa)
    STORE[] === :none || (data[key] = STORE[] === :raw ? raw : _codes(raw))
    isnothing(EXTRA[]) || (data[first(EXTRA[])] = last(EXTRA[]))
    data[name(za)] = Float64.(parent(parent(za)))
    push!(stmts, Expr(:call, :~, target, Expr(:call, HookDataOwnership.cell,
        Expr(:parameters, Expr(:kw, :subjects, key), Expr(:kw, :z, name(za))))))
    :done
end

const N, S = 12, 4
const SUBJECTS = repeat(1:S; inner=N ÷ S)
base_data(subjects) = (; subjects, arm=repeat([1, 2]; inner=N ÷ 2),
    z=collect(range(0.1, 1.2; length=N)), y=collect(range(1.0, 2.0; length=N)),
    y2=collect(range(0.5, 1.5; length=N)))

build(m) = SBBRMI(m; mod=@__MODULE__, total_groups=())
# Each shape the ranef pass can visit `subjects` through, before and after the
# hook in formula order.
shapes = (
    hook_first = d -> @brm(d, begin
        eff ~ group_term(subjects, z)
        mu ~ 1 + (1 | subjects)
        y ~ Normal(mu, 1.0)
        y2 ~ Normal(eff, 1.0)
    end),
    group_first = d -> @brm(d, begin
        mu ~ 1 + (1 | subjects)
        eff ~ group_term(subjects, z)
        y ~ Normal(mu, 1.0)
        y2 ~ Normal(eff, 1.0)
    end),
    id_bucket = d -> @brm(d, begin
        eff ~ group_term(subjects, z)
        mu ~ 1 + (1 | p | subjects)
        y ~ Normal(mu, 1.0)
        y2 ~ Normal(eff, 1.0)
    end),
    by_stratum = d -> @brm(d, begin
        eff ~ group_term(subjects, z)
        mu ~ 1 + (1 | gr(subjects, by=arm))
        y ~ Normal(mu, 1.0)
        y2 ~ Normal(eff, 1.0)
    end),
)

function stan_of(m)
    sb = build(m)
    sb, String(StanBlocks.stan_code(sb.model))
end

@testset "hook data under a grouping column's own name" begin
    for (shape, model) in pairs(shapes), store in (:codes, :raw, :none)
        STORE[] = store
        @testset "$shape, integer labels, store=$store" begin
            sb, code = stan_of(model(base_data(SUBJECTS)))
            @test sb.data[:subjects] == SUBJECTS
            @test haskey(sb.data, shape === :by_stratum ?
                :subjects__by__arm_idx : :subjects_idx)
            @test occursin(r"\bsubjects\b", code)
        end
    end
    STORE[] = :codes
    labels = string.("s", SUBJECTS)
    for (shape, model) in pairs(shapes)
        @testset "$shape, string labels, integer codes" begin
            sb, _ = stan_of(model(base_data(labels)))
            @test sb.data[:subjects] == SUBJECTS
        end
    end
end

@testset "a statement reading raw string labels is refused at build" begin
    labels = string.("s", SUBJECTS)
    for store in (:raw, :none)
        STORE[] = store
        # refused: string labels are not Stan data, and the generic non-Stan
        # drop would otherwise delete them under the reading statement
        # (snag submodel-hook-da-23edf0b4).
        err = try
            build(shapes.hook_first(base_data(labels))); nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("reads `subjects`, which is also a ranef grouping column",
                       sprint(showerror, err))
    end
    STORE[] = :codes
end

@testset "per-target keys and hook-free models are unchanged" begin
    STORE[] = :per_target
    for (shape, model) in pairs(shapes)
        sb, _ = stan_of(model(base_data(SUBJECTS)))
        @test !haskey(sb.data, :subjects)
        @test sb.data[:subjects_eff] == SUBJECTS
    end
    STORE[] = :codes
    plain = d -> @brm(d, begin
        mu ~ 1 + (1 | subjects)
        y ~ Normal(mu, 1.0)
    end)
    for labels in (SUBJECTS, string.("s", SUBJECTS))
        sb, code = stan_of(plain(base_data(labels)))
        @test !haskey(sb.data, :subjects)
        @test !occursin(r"\bsubjects\b", code)
    end
end

@testset "BRM-derived index names stay BRM's" begin
    STORE[] = :codes
    # refused: `subjects_idx` is the index BRM derives and reads; a different
    # value from a hook would be overwritten under the hook's statement
    # (snag submodel-hook-da-23edf0b4).
    EXTRA[] = :subjects_idx => reverse(SUBJECTS)
    err = try
        build(shapes.hook_first(base_data(SUBJECTS))); nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("`subjects_idx` is the index BRM derives", sprint(showerror, err))
    # The same value BRM derives is not a conflict.
    EXTRA[] = :subjects_idx => SUBJECTS
    @test build(shapes.hook_first(base_data(SUBJECTS))).data[:subjects_idx] == SUBJECTS
    EXTRA[] = nothing
end

# `factor(g)` also writes `<g>_idx`; it must agree with the ranef index rather
# than trip the derived-name refusal.
@testset "a categorical predictor on the grouping column still builds" begin
    sb = build(@brm(base_data(SUBJECTS), begin
        mu ~ 1 + factor(subjects) + (1 | subjects)
        y ~ Normal(mu, 1.0)
    end))
    @test sb.data[:subjects_idx] == SUBJECTS
    @test !isempty(String(StanBlocks.stan_code(sb.model)))
end
