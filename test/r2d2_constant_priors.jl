# test/r2d2_constant_priors.jl — a `name = constant(expr)` or a scalar data
# field as the location or scale of a per-column Normal prior that an R2D2
# budget leaves outside its simplex (snag `effect-qt-base-i-5680ac28`).
#
# An `r2d2` predictor keeps every excluded column's Normal loc/scale — always
# the intercept's — in per-column vectors beside the allocated scales, on both
# the whole-predictor `effect(lp, :) ~ r2d2(...)` form and the joint
# `sd(:, ID) ~ r2d2(...; include=...)` budget. Those vectors held numbers only,
# so a declared constant, which brm-use promises behaves exactly like the same
# scalar data field, was refused there by both backends. Now SBBRMI keeps a
# vector with a named entry as a Stan expression over that data (an
# all-numeric vector stays the frozen data it was), and RKBRMI reads a scalar
# data field's value, as its ordinary population priors do.
#
# Run: julia --startup-file=no --project=test test/r2d2_constant_priors.jl
# Set BRM_R2D2_RUNTIME=0 to skip the compiled Stan densities.

include(joinpath(@__DIR__, "rk_consumer_support.jl"))
using Statistics: mean

const R2D2C_RUNTIME = get(ENV, "BRM_R2D2_RUNTIME", "1") != "0"
const stan_code = BRM.stan_code
const stan_data = BRM.stan_data

const PANEL = (;
    conc       = [2.4, 2.2, 2.0, 1.8, 1.7, 1.5, 1.6, 1.9, 2.1, 1.4, 1.3, 1.8],
    wt         = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -0.25, 0.75, 0.3, -0.8, 1.2, -0.1],
    age        = [0.2, -0.4, 1.1, -0.9, 0.3, 0.7, -1.2, 0.1, 0.5, -0.6, 0.9, -0.3],
    indication = [1, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 2],
    subject    = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
    qt         = [0.3, -0.2, 0.5, 0.1, -0.4, 0.2, 0.0, 0.6, -0.1, 0.4, -0.3, 0.2],
)
const LOC = mean(PANEL.conc)
const SCALE = 10.0 / maximum(PANEL.conc)
const SUPPLIED_PANEL = merge(PANEL, (; qt_loc=LOC, qt_scale=SCALE))

# The joint budget allocates both predictors' slopes and contrasts; each
# intercept keeps its own Normal. `log_Vc`'s location and `qt_base`'s scale
# are fitted constants, so each predictor mixes a named and a numeric vector.
const JOINT = @brm begin
    qt_loc = constant(mean(conc))
    qt_scale = constant(10.0 / maximum(conc))
    sigma_pk ~ Exponential(1)
    sigma_qt ~ Exponential(1)
    log_Vc  ~ 1 + wt + indication + (1 | p | subject)
    qt_base ~ 1 + wt + indication + (1 | p | subject)
    effect(log_Vc, Intercept) ~ Normal(qt_loc, 2.0)
    effect(qt_base, Intercept) ~ Normal(0.0, qt_scale)
    sd(:, p) ~ r2d2(mean_R2=0.5, prec_R2=4, concentration=0.7,
                    reference_scale=sigma_pk, include=(:population, :contrasts))
    sd(qt_base, p) ~ r2d2(reference_scale=sigma_qt)
    cor(:, p) ~ LKJCholesky(2, 2)
    conc ~ Normal(log_Vc, sigma_pk)
    qt ~ Normal(qt_base, sigma_qt)
end
const JOINT_SUPPLIED = @brm begin
    sigma_pk ~ Exponential(1)
    sigma_qt ~ Exponential(1)
    log_Vc  ~ 1 + wt + indication + (1 | p | subject)
    qt_base ~ 1 + wt + indication + (1 | p | subject)
    effect(log_Vc, Intercept) ~ Normal(qt_loc, 2.0)
    effect(qt_base, Intercept) ~ Normal(0.0, qt_scale)
    sd(:, p) ~ r2d2(mean_R2=0.5, prec_R2=4, concentration=0.7,
                    reference_scale=sigma_pk, include=(:population, :contrasts))
    sd(qt_base, p) ~ r2d2(reference_scale=sigma_qt)
    cor(:, p) ~ LKJCholesky(2, 2)
    conc ~ Normal(log_Vc, sigma_pk)
    qt ~ Normal(qt_base, sigma_qt)
end
joint_literal(loc, scale) = eval(:(@brm begin
    sigma_pk ~ Exponential(1)
    sigma_qt ~ Exponential(1)
    log_Vc  ~ 1 + wt + indication + (1 | p | subject)
    qt_base ~ 1 + wt + indication + (1 | p | subject)
    effect(log_Vc, Intercept) ~ Normal($loc, 2.0)
    effect(qt_base, Intercept) ~ Normal(0.0, $scale)
    sd(:, p) ~ r2d2(mean_R2=0.5, prec_R2=4, concentration=0.7,
                    reference_scale=sigma_pk, include=(:population, :contrasts))
    sd(qt_base, p) ~ r2d2(reference_scale=sigma_qt)
    cor(:, p) ~ LKJCholesky(2, 2)
    conc ~ Normal(log_Vc, sigma_pk)
    qt ~ Normal(qt_base, sigma_qt)
end))

# The whole-predictor form: the intercept and `wt` keep their own Normal and
# leave the simplex, so `age` is the single share.
const FLAT = @brm begin
    qt_loc = constant(mean(conc))
    qt_scale = constant(10.0 / maximum(conc))
    log_CL ~ 1 + wt + age + (1 | p | subject)
    effect(log_CL, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5)
    effect(log_CL, Intercept) ~ Normal(qt_loc, qt_scale)
    effect(log_CL, wt) ~ Normal(0.0, qt_scale)
    conc ~ Normal(log_CL, 1)
end
const FLAT_SUPPLIED = @brm begin
    log_CL ~ 1 + wt + age + (1 | p | subject)
    effect(log_CL, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5)
    effect(log_CL, Intercept) ~ Normal(qt_loc, qt_scale)
    effect(log_CL, wt) ~ Normal(0.0, qt_scale)
    conc ~ Normal(log_CL, 1)
end
flat_literal(loc, scale) = eval(:(@brm begin
    log_CL ~ 1 + wt + age + (1 | p | subject)
    effect(log_CL, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5)
    effect(log_CL, Intercept) ~ Normal($loc, $scale)
    effect(log_CL, wt) ~ Normal(0.0, $scale)
    conc ~ Normal(log_CL, 1)
end))

const FORMS = (
    (; name="joint", constant=JOINT, supplied=JOINT_SUPPLIED, literal=joint_literal),
    (; name="flat", constant=FLAT, supplied=FLAT_SUPPLIED, literal=flat_literal),
)

sb(brmi) = SBBRMI(brmi; mod=@__MODULE__, total_groups=())

# The compiled program's full log density at fixed unconstrained points.
function sb_density_points(model, name)
    folder = joinpath(tempdir(), "brm-r2d2-constant-priors")
    mkpath(folder)
    stan = BRM.stan_instantiate(model; path=joinpath(folder, name * ".stan"))
    n = BridgeStan.param_unc_num(stan.model)
    points = (zeros(n), fill(0.1, n), collect(range(-0.4, 0.4; length=n)))
    BridgeStan.param_unc_names(stan.model),
        [BridgeStan.log_density(stan.model, u; propto=false) for u in points]
end

@stestset "SBBRMI: a constant in an R2D2 override is the data field it names" begin
    for form in FORMS
        constant = sb(form.constant(PANEL))
        code = stan_code(constant)
        @test code == stan_code(sb(form.supplied(SUPPLIED_PANEL)))
        @test stan_data(constant)[:qt_loc] == LOC
        @test stan_data(constant)[:qt_scale] == SCALE
        @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
        # The literal twin keeps the frozen numeric vectors it always had.
        literal = sb(form.literal(LOC, SCALE)(PANEL))
        @test StanBlocks.stanc_check(stan_code(literal); warn_pedantic=false).ok
        for target in (form.name == "joint" ? (:log_Vc, :qt_base) : (:log_CL,))
            for vector in (:beta_loc, :fallback)
                @test haskey(stan_data(literal), Symbol(:r2d2_, target, :_, vector))
            end
        end
    end
    # A vector naming data is a Stan expression over it; a numeric one stays
    # data, so one predictor can carry one of each.
    joint = sb(JOINT(PANEL))
    code = stan_code(joint)
    @test !haskey(stan_data(joint), :r2d2_log_Vc_beta_loc)
    @test stan_data(joint)[:r2d2_log_Vc_fallback] == [2.0, 1.0]
    @test stan_data(joint)[:r2d2_qt_base_beta_loc] == [0.0, 0.0]
    @test !haskey(stan_data(joint), :r2d2_qt_base_fallback)
    @test occursin("qt_loc", code) && occursin("qt_scale", code)
    flat = sb(FLAT(PANEL))
    @test !haskey(stan_data(flat), :r2d2_log_CL_beta_loc)
    @test !haskey(stan_data(flat), :r2d2_log_CL_fallback)
    @test stan_data(flat)[:r2d2_log_CL_share_idx] == [0, 0, 1]
end

@stestset "SBBRMI: the override vector may read a sampled scalar" begin
    # The ordinary population path takes any prior expression; an R2D2
    # predictor's excluded columns now do too.
    sampled = @brm begin
        intercept_scale ~ Exponential(1)
        log_CL ~ 1 + wt + age + (1 | p | subject)
        effect(log_CL, :) ~ r2d2(R2=Beta(1, 1), tau_bsv=0.5)
        effect(log_CL, Intercept) ~ Normal(0.0, intercept_scale)
        conc ~ Normal(log_CL, 1)
    end
    code = stan_code(sb(sampled(PANEL)))
    @test StanBlocks.stanc_check(code; warn_pedantic=false).ok
    @test occursin("intercept_scale", code)
end

if R2D2C_RUNTIME
    @stestset "SBBRMI: densities equal the literal twin, also after replay" begin
        for form in FORMS
            constant = sb(form.constant(PANEL))
            names, values = sb_density_points(constant, form.name * "_constant")
            literal_names, literal_values = sb_density_points(
                sb(form.literal(LOC, SCALE)(PANEL)), form.name * "_literal")
            @test names == literal_names
            @test values ≈ literal_values atol=1e-10 rtol=1e-12
            # Frozen replay keeps the fitted constants; a fresh replay
            # evaluates them on the new data, and the override vectors follow.
            conc = 2 .* PANEL.conc
            replay = merge(PANEL, (; conc))
            frozen = reprocess(constant, replay)
            @test stan_data(frozen)[:qt_scale] == SCALE
            fresh = reprocess(constant, replay; freeze_constants=false)
            @test stan_data(fresh)[:qt_scale] == 10.0 / maximum(conc)
            @test stan_data(fresh)[:qt_loc] == mean(conc)
            fresh_names, fresh_values = sb_density_points(
                fresh, form.name * "_constant")
            twin_names, twin_values = sb_density_points(
                sb(form.literal(mean(conc), 10.0 / maximum(conc))(replay)),
                form.name * "_literal")
            @test fresh_names == twin_names
            @test fresh_values ≈ twin_values atol=1e-10 rtol=1e-12
            @test !(fresh_values ≈ values)
        end
    end
else
    @info "Skipping the compiled R2D2 constant-prior densities (BRM_R2D2_RUNTIME=0)"
end

@stestset "RKBRMI: a constant in an R2D2 override is the number it holds" begin
    ad_backend = AutoEnzyme(; mode=Enzyme.Reverse)
    for form in FORMS
        rk = rk_logdensity_problem(RKBRMI(form.constant(PANEL)); ad_backend)
        supplied = rk_logdensity_problem(
            RKBRMI(form.supplied(SUPPLIED_PANEL)); ad_backend)
        literal = rk_logdensity_problem(
            RKBRMI(form.literal(LOC, SCALE)(PANEL)); ad_backend)
        n = LogDensityProblems.dimension(rk)
        @test LogDensityProblems.dimension(literal) == n
        for u in (zeros(n), fill(0.1, n), collect(range(-0.4, 0.4; length=n)))
            @test LogDensityProblems.logdensity(rk, u) ==
                LogDensityProblems.logdensity(supplied, u)
            @test LogDensityProblems.logdensity(rk, u) ≈
                LogDensityProblems.logdensity(literal, u) atol=1e-12
        end
    end
end
