# test/submodel_scalar_keywords.jl — scalar keyword columns of a custom term
# broadcast; they do not define the term's row axis.
#
# A custom term (`_sb_submodel_rhs!` / `_rk_submodel_rhs!`) reads one row-wise
# positional column on a flat event axis and two scalar keyword columns, and
# `ragged(target, group)` joins its per-event value to the subject plate. A
# scalar is broadcast data: the target lives on the frame of its row-wise
# columns, whether the scalar is a `name = constant(expr)` declaration or a
# scalar field of the data container. Before, the ragged row check counted a
# scalar as a one-row column and refused the join on both backends ("declared
# over a 1-row axis (data column `shift_loc`)"; snag submodel-keyword-f4a3876e).
#
# Run: julia --project=test test/submodel_scalar_keywords.jl [<testset substring>...]
# Set BRM_SCALAR_KEYWORDS_RUNTIME=0 to skip the compiled-model sets.

include(joinpath(@__DIR__, "rk_consumer_support.jl"))

const RUNTIME = get(ENV, "BRM_SCALAR_KEYWORDS_RUNTIME", "1") != "0"

module ScalarKeywords
using BayesianRegressionModels, StanBlocks, Statistics
import BayesianRegressionModels: _sb_submodel_rhs!, _rk_submodel_rhs!
using BayesianRegressionModels: getargs, getkwargs, name

# A per-event shift with a location prior calibrated on the event table:
# target[j] = x[j] - location, location ~ normal(loc, scale).
const shift_cell = StanBlocks.@slic begin
    location ~ normal(loc, scale)
    return x .- location
end

function shifted end
function shifted_inline end

# The calibration arrives as two keyword columns.
function _sb_submodel_rhs!(stmts, data, target::Symbol, ::typeof(shifted), rhs)
    x = only(getargs(rhs))
    kw = getkwargs(rhs)
    push!(stmts, :($target ~ $shift_cell(; x=$(name(x)),
        loc=$(name(kw[:loc])), scale=$(name(kw[:scale])))))
    :done
end

# The same term with the calibration computed inside the hook: the control
# whose row frame comes from the positional column alone.
function _sb_submodel_rhs!(stmts, data, target::Symbol, ::typeof(shifted_inline), rhs)
    x = only(getargs(rhs))
    values = parent(parent(x))
    loc, scale = Symbol(target, :_loc), Symbol(target, :_scale)
    data[loc] = mean(values)
    data[scale] = std(values)
    push!(stmts, :($target ~ $shift_cell(; x=$(name(x)), loc=$loc, scale=$scale)))
    :done
end

# The native RK source of the same two terms.
function native_shift!(definitions, statements, data, target, rhs, loc, scale)
    key, helper = Symbol(target, :_source_x), Symbol(target, :_shift)
    data[key] = copy(parent(parent(only(getargs(rhs)))))
    push!(definitions, :($helper(x, loc, scale) = begin
        location ~ Normal(loc, scale)
        return x .- location
    end))
    push!(statements, :($target ~ $helper($key, $loc, $scale)))
    :done
end
_rk_submodel_rhs!(definitions, statements, data, bindings, target::Symbol,
        ::typeof(shifted), rhs) =
    native_shift!(definitions, statements, data, target, rhs,
        name(getkwargs(rhs)[:loc]), name(getkwargs(rhs)[:scale]))
function _rk_submodel_rhs!(definitions, statements, data, bindings, target::Symbol,
        ::typeof(shifted_inline), rhs)
    values = parent(parent(only(getargs(rhs))))
    loc, scale = Symbol(target, :_loc), Symbol(target, :_scale)
    data[loc], data[scale] = mean(values), std(values)
    native_shift!(definitions, statements, data, target, rhs, loc, scale)
end

# Keywords from `constant(...)` declarations.
constant_builder = @brm begin
    sigma ~ Exponential(1)
    shift_loc = constant(mean(log_dose))
    shift_scale = constant(std(log_dose))
    log_F ~ shifted(log_dose; loc=shift_loc, scale=shift_scale)
    log_CL ~ 1 + (1 | p | subject)
    ragged(log_F, dose_subject)
    @plate for i in eachindex(log_CL)
        dose = sum(dose_amount[i] .* exp.(log_F[i]))
        mu = 1e-2 * dose .* exp.(-exp(log_CL[i]) .* t_obs[i])
        dv[i] ~ normal(mu, sigma)
    end
end

# Keywords from scalar fields of the data container.
field_builder = @brm begin
    sigma ~ Exponential(1)
    log_F ~ shifted(log_dose; loc=shift_loc, scale=shift_scale)
    log_CL ~ 1 + (1 | p | subject)
    ragged(log_F, dose_subject)
    @plate for i in eachindex(log_CL)
        dose = sum(dose_amount[i] .* exp.(log_F[i]))
        mu = 1e-2 * dose .* exp.(-exp(log_CL[i]) .* t_obs[i])
        dv[i] ~ normal(mu, sigma)
    end
end

# No keywords: the hook writes the same calibration itself.
inline_builder = @brm begin
    sigma ~ Exponential(1)
    log_F ~ shifted_inline(log_dose)
    log_CL ~ 1 + (1 | p | subject)
    ragged(log_F, dose_subject)
    @plate for i in eachindex(log_CL)
        dose = sum(dose_amount[i] .* exp.(log_F[i]))
        mu = 1e-2 * dose .* exp.(-exp(log_CL[i]) .* t_obs[i])
        dv[i] ~ normal(mu, sigma)
    end
end

# 3 subjects; a flat 7-row event table whose rows interleave the subjects.
events = (;
    dose_subject=["s2", "s1", "s1", "s3", "s2", "s3", "s2"],
    log_dose=log.([50.0, 100.0, 100.0, 200.0, 50.0, 200.0, 50.0]),
)
train = (;
    subject=["s1", "s2", "s3"],
    t_obs=[[0.5, 1.0, 2.0], [0.5, 1.5], [0.25, 1.0, 3.0]],
    dv=[[0.8, 0.6, 0.7], [1.5, 1.0], [0.9, 0.5, 0.9]],
    dose_amount=[[100.0, 100.0], [50.0, 50.0, 50.0], [200.0, 200.0]],
    events...,
)
with_fields(data) = (; data..., shift_loc=mean(data.log_dose), shift_scale=std(data.log_dose))

# The fitted subjects with a different event table.
refit = (;
    subject=["s1", "s2", "s3"],
    t_obs=[[0.5, 2.5], [1.0], [0.5, 1.5, 2.0]],
    dv=[zeros(2), zeros(1), zeros(3)],
    dose_amount=[[80.0], [80.0, 80.0, 80.0], [80.0]],
    dose_subject=["s2", "s1", "s2", "s3", "s2"],
    log_dose=log.([80.0, 80.0, 80.0, 80.0, 80.0]),
)

# New subjects with a different event table.
future = (;
    subject=["n1", "n2"],
    t_obs=[[0.5, 2.5], [1.0, 2.0]],
    dv=[zeros(2), zeros(2)],
    dose_amount=[[80.0], [80.0, 80.0, 80.0]],
    dose_subject=["n2", "n1", "n2", "n2"],
    log_dose=log.([80.0, 80.0, 80.0, 80.0]),
)
end

const SK = ScalarKeywords

build(builder, data) = SBBRMI(builder(data); mod=SK, total_groups=())
stanc_ok(sb) = StanBlocks.stanc_check(StanBlocks.stan_code(sb.model); warn_pedantic=false)

@stestset "SBBRMI: scalar keyword columns do not set the term's row axis" begin
    for (label, builder, data) in (
            ("constant(...)", SK.constant_builder, SK.train),
            ("data field", SK.field_builder, SK.with_fields(SK.train)))
        @testset "$label" begin
            sb = build(builder, data)
            @test sb.data[:shift_loc] == SK.mean(SK.train.log_dose)
            @test sb.data[:shift_scale] == SK.std(SK.train.log_dose)
            @test !haskey(sb.data, :log_F)
            @test stanc_ok(sb).ok
        end
    end
    # A constant is Stan data exactly like the same scalar field.
    @test StanBlocks.stan_code(build(SK.constant_builder, SK.train).model) ==
          StanBlocks.stan_code(build(SK.field_builder, SK.with_fields(SK.train)).model)
end

@stestset "SBBRMI: the term's value sits on the event axis" begin
    if RUNTIME
        points(n) = ([0.1 * ((i % 5) - 2) for i in 1:n], [0.3 * sin(i) for i in 1:n])
        inline = StanBlocks.stan_instantiate(build(SK.inline_builder, SK.train).model)
        for (builder, data) in ((SK.constant_builder, SK.train),
                                (SK.field_builder, SK.with_fields(SK.train)))
            prob = StanBlocks.stan_instantiate(build(builder, data).model)
            dim = LogDensityProblems.dimension(prob)
            @test dim == LogDensityProblems.dimension(inline)
            pnames = StanBlocks.BridgeStan.param_names(prob.model; include_tp=true)
            @test pnames == StanBlocks.BridgeStan.param_names(inline.model; include_tp=true)
            for q in points(dim)
                lp, grad = LogDensityProblems.logdensity_and_gradient(prob, q)
                @test isfinite(lp) && all(isfinite, grad)
                # Same density as the hook that writes the calibration itself.
                @test lp == LogDensityProblems.logdensity(inline, q)
                values = StanBlocks.BridgeStan.param_constrain(prob.model, q; include_tp=true)
                location = values[only(findall(endswith("location"), pnames))]
                log_F = values[[findfirst(==("log_F.$j"), pnames) for j in 1:7]]
                @test log_F ≈ SK.train.log_dose .- location atol=1e-12
            end
        end
    else
        @info "Skipping compiled-model set (BRM_SCALAR_KEYWORDS_RUNTIME=0)"
    end
end

@stestset "SBBRMI: replay keeps the constants and re-joins the new event table" begin
    sb = build(SK.constant_builder, SK.train)
    # One dose level on replay: re-evaluating `std` would give 0.
    for (replay, nevents) in ((reprocess(sb, SK.refit), 5),
                              (reprocess(sb, SK.future; resample_groups=:subject), 4))
        @test replay.data[:shift_loc] == sb.data[:shift_loc]
        @test replay.data[:shift_scale] == sb.data[:shift_scale]
        @test length(replay.data[:log_dose]) == nevents
        @test stanc_ok(replay).ok
    end
end

@stestset "RKBRMI: the term joins the subject plate with the Stan density" begin
    for (label, builder, data) in (
            ("constant", SK.constant_builder, SK.train),
            ("field", SK.field_builder, SK.with_fields(SK.train)),
            ("inline", SK.inline_builder, SK.train))
        brmi = builder(data)
        rk, problem = consumer_problem(brmi)
        RUNTIME || continue
        stan = consumer_stan(brmi, "scalar-keywords-$label"; mod=SK)
        transport = brm_coordinate_transport(rk, SBBRMI(brmi; mod=SK, total_groups=()),
            BridgeStan.param_unc_names(stan.model))
        n = length(transport)
        @test n == length(coordinate_names(rk.model.layout))
        for u in (zeros(n), collect(range(-0.43, 0.51; length=n)), [0.37sin(i) for i in 1:n])
            value, gradient = LogDensityProblems.logdensity_and_gradient(problem, u)
            stan_u = brm_rk_point_to_stan(transport, u)
            stan_gradient = similar(stan_u)
            stan_value, _ = BridgeStan.log_density_gradient!(stan.model, stan_u,
                stan_gradient; propto=false, jacobian=true)
            @test value ≈ stan_value + transport.logdensity_offset atol=2e-11 rtol=2e-11
            @test gradient ≈ brm_stan_gradient_to_rk(transport, u, stan_gradient) atol=2e-10 rtol=2e-10
        end
    end
end
