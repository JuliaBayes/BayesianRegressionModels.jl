include(joinpath(@__DIR__, "rk_consumer_support.jl"))

module PublicKernelScope
using BayesianRegressionModels, Distributions, StanBlocks
StanBlocks.@deffun public_cell(ts::vector[n], a::real) = ts * a
public_cell(ts::AbstractVector, a::Real) = ts .* a
function build(data)
    @brm data begin
        theta ~ 1 + (1 | p | subject)
        effect(theta, Intercept) ~ Normal(0, 1)
        sd(:, p) ~ Exponential(1)
        sigma ~ Exponential(1)
        pred ~ kernel(t, theta) do ts, a
            return public_cell(ts, a)
        end
        y ~ Normal(pred, sigma)
    end
end
end

module DottedCellScope
using BayesianRegressionModels, Distributions
halfsq(x) = 0.5 * x * x
function build(data)
    @brm data begin
        s ~ Normal(0, 1)
        sigma ~ Exponential(1)
        @plate for i in eachindex(x)
            m = halfsq.(exp(s) .* x[i])
            y[i] ~ normal(m, sigma)
            pred[i] = x[i]
        end
    end
end
end

@stestset "kernel response aliases withhold one likelihood on independent axes" begin
    data = (; index=[[1, 2, 1]], y=[[0.1, -0.2, 0.4]],
        z=[[0.2, -0.3]], catalog=[[0.2, 0.7]])
    saved = deepcopy(data)
    brmi = @brm data begin
        shift ~ Normal(0, 1)
        pred ~ kernel(index, y, z, catalog) do ii, yy, zz, cc
            mu_y = cc[ii] .+ shift
            mu_z = cc .+ shift
            yy ~ normal(mu_y, 1)
            zz ~ normal(mu_z, 1)
            mu_y
        end
    end
    copy_brmi = deepcopy(brmi)
    @test copy_brmi !== brmi
    for (selection, active) in ((:yy, :z), (:z, :y))
        backend = check_rk_source_roundtrip(RKBRMI(brmi; held_out=selection))
        @test coordinate_names(backend.model.layout) == [:shift]
        @test BRM._rk_observed_names(backend.plan) == (active,)
        problem = rk_logdensity_problem(backend; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=(), held_out=selection)
        stan = BRM.stan_instantiate(sb; path=joinpath(tempdir(), "brm-rk-consumer",
            "kernel-held-out-$selection.stan"))
        oracle(u) = logpdf(Normal(), u[1]) +
            sum(logpdf.(Normal.(
                (active === :y ? data.catalog[1][data.index[1]] : data.catalog[1]) .+ u[1], 1),
                getproperty(data, active)[1]))
        for u in ([0.0], [0.2], [-0.3])
            check_consumer_point(problem, u, oracle)
            check_consumer_stan(problem, stan, [:shift => "shift"], backend, u)
        end
    end
    @test_throws "covers every observation" RKBRMI(brmi; held_out=(:yy, :z))
    @test_throws "unknown response" RKBRMI(brmi; held_out=:typo)
    @test isequal(data, saved)
end

@stestset "ragged kernel readers preserve subject joins and original axes" begin
    data = (; subject=["b", "empty", "a"],
        t=[[0.1, 0.7], Float64[], [0.2, 0.8, 1.3]],
        y=[[0.2, -0.1], Float64[], [0.1, 0.3, -0.2]],
        z=[0.1, -0.2, 0.3], event_subject=["a", "b", "b", "a", "a"],
        event_time=[0.2, 0.1, 0.7, 0.8, 1.3])
    saved = deepcopy(data)
    for mode in (:grouped, :ragged, :scope)
        brmi = if mode === :grouped
            @brm data begin
                theta ~ 1 + (1 | p | subject)
                effect(theta, Intercept) ~ Normal(0, 1)
                sd(:, p) ~ Exponential(1)
                sigma ~ Exponential(1)
                pred ~ kernel(t, theta) do ts, a
                    ts * a
                end
                y ~ Normal(pred, sigma)
            end
        elseif mode === :ragged
            @brm data begin
                theta ~ 1 + (1 | p | subject)
                effect(theta, Intercept) ~ Normal(0, 1)
                sd(:, p) ~ Exponential(1)
                sigma ~ Exponential(1)
                pred ~ kernel(ragged(event_time, event_subject), theta) do ts, a
                    ts * a
                end
                y ~ Normal(pred, sigma)
            end
        else
            PublicKernelScope.build(data)
        end
        backend, problem = consumer_problem(brmi)
        names = coordinate_names(backend.model.layout)
        ia = findfirst(==(Symbol("pop_theta.beta_pop.1")), names)
        it = findfirst(n -> occursin(".tau.", string(n)), names)
        iz = findall(n -> occursin(".z.", string(n)), names)
        is = findfirst(==(:sigma), names)
        @test length(names) == 6
        @test length(iz) == 3
        levels = sort(unique(data.subject))
        order = [findfirst(==(label), levels) for label in data.subject]
        oracle(u) = begin
            tau, sigma = exp(u[it]), exp(u[is])
            theta = u[ia] .+ tau .* u[iz][order]
            means = reduce(vcat, [data.t[i] .* theta[i] for i in eachindex(data.t)])
            logpdf(Normal(), u[ia]) + sum(logpdf.(Normal(), u[iz])) +
                logpdf(Exponential(), tau) + u[it] +
                logpdf(Exponential(), sigma) + u[is] +
                sum(logpdf.(Normal.(means, sigma), reduce(vcat, data.y)))
        end
        stan = consumer_stan(brmi, "ragged-reader-$mode";
            mod=mode === :scope ? PublicKernelScope : @__MODULE__)
        mapping = [names[ia] => "pop_theta_beta_pop.1";
            names[it] => "b_p_subject_tau.1"; names[is] => "sigma";
            [names[iz[i]] => "b_p_subject_z_flat.$i" for i in 1:3]]
        for u in (zeros(6), collect(range(-0.2, 0.3; length=6)), fill(-0.1, 6))
            check_consumer_point(problem, u, oracle)
            check_consumer_stan(problem, stan, mapping, backend, u)
        end
        artifact = BRM.emit_rk_artifact(brmi; case_id="ragged-reader-$mode")
        @test length(artifact.defs) >= 2
        @test all(d -> BRM._rk_source_definition(d).kind in (:function, :kernel, :rkppl), artifact.defs)
        @test isequal(data, saved)
    end
end

@stestset "kernel cell inputs keep distinct vector axes" begin
    data = (; index=[[1, 2, 1]], y=[[0.1, -0.2, 0.4]], catalog=[[0.2, 0.7]])
    saved = deepcopy(data)
    brmi = @brm data begin
        shift ~ Normal(0, 1)
        pred ~ kernel(index, y, catalog) do ii, yy, cc
            mu = cc[ii] .+ shift
            yy ~ normal(mu, 1)
            mu
        end
    end
    backend, problem = consumer_problem(brmi)
    @test coordinate_names(backend.model.layout) == [:shift]
    stan = consumer_stan(brmi, "kernel-distinct-axes")
    oracle(u) = logpdf(Normal(), u[1]) +
        sum(logpdf.(Normal.(data.catalog[1][data.index[1]] .+ u[1], 1), data.y[1]))
    for u in ([0.0], [0.2], [-0.3])
        check_consumer_point(problem, u, oracle)
        check_consumer_stan(problem, stan, [:shift => "shift"], backend, u)
    end
    @test isequal(data, saved)
end

# A cell value read only inside a broadcast call `f.(args...)` is captured like
# an ordinary call argument: a model parameter (keeping its prior), a shared data
# column, and a Julia callee (snag plate-cell-drops-4ab1adb2). The Julia-only
# callee has no Stan counterpart, so that mode checks the oracle alone.
@stestset "plate cells read model values inside broadcast calls" begin
    refs(ex) = BRM._brm_cell_value_refs!(Set{Symbol}(), ex)
    @test refs(:(tanh.(0.5 .* (exp(s) .* x)))) == Set([:s, :x])
    @test refs(:(M.g.(u, w))) == Set([:u, :w])
    @test refs(:(f(a.b; k = v))) == Set([:a, :v])

    data = (; x=[[1.0, 2.0], [3.0], [0.5, 1.5]], y=[[0.1, 0.2], [0.4], [0.3, 0.5]],
        w=[0.2, -0.3, 0.5, 0.1])
    saved = deepcopy(data)
    halfsq = DottedCellScope.halfsq
    for mode in (:parameter, :data, :callable)
        brmi = if mode === :parameter
            @brm data begin
                s ~ Normal(0, 1)
                sigma ~ Exponential(1)
                @plate for i in eachindex(x)
                    m = tanh.(0.5 .* (exp(s) .* x[i]))
                    y[i] ~ normal(m, sigma)
                    pred[i] = x[i]
                end
            end
        elseif mode === :data
            @brm data begin
                s ~ Normal(0, 1)
                sigma ~ Exponential(1)
                @plate for i in eachindex(x)
                    m = s .* x[i] .+ sum(tanh.(w))
                    y[i] ~ normal(m, sigma)
                    pred[i] = x[i]
                end
            end
        else
            DottedCellScope.build(data)
        end
        mean_of(s, x) = mode === :parameter ? tanh.(0.5 .* (exp(s) .* x)) :
            mode === :data ? s .* x .+ sum(tanh.(data.w)) : halfsq.(exp(s) .* x)
        backend, problem = consumer_problem(brmi)
        names = coordinate_names(backend.model.layout)
        @test sort(names) == [:s, :sigma]
        is, iσ = findfirst(==(:s), names), findfirst(==(:sigma), names)
        oracle(u) = begin
            s, sigma = u[is], exp(u[iσ])
            logpdf(Normal(), s) + logpdf(Exponential(), sigma) + u[iσ] +
                sum(sum(logpdf.(Normal.(mean_of(s, x), sigma), y))
                    for (x, y) in zip(data.x, data.y))
        end
        points = ([0.0, 0.0], [0.3, -0.2], [-0.4, 0.25])
        for u in points
            check_consumer_point(problem, u, oracle)
        end
        if mode !== :callable
            stan = consumer_stan(brmi, "plate-broadcast-$mode")
            for u in points
                check_consumer_stan(problem, stan, [:s => "s", :sigma => "sigma"], backend, u)
            end
        end
    end
    @test isequal(data, saved)
end

# The reader of a kernel cell is the authored cell as one subject plate over
# its per-subject inputs, then one flatten of the subject cells (snag
# rk-kernel-reader-9a48d7b7). It was a `1:count` plate passing every port as
# `Ref`, rebinding `cell_input_<k>` locals for a forwarding cell kernel, and
# flattening with `reduce(vcat, cells; init=…)`, which reallocates once per
# subject on every evaluation and reverse pass.
@stestset "kernel readers are the authored cell, flattened in linear time" begin
    flat = BRM.brm_flatten_cells
    @test isequal(flat(Vector{Float64}[]), Float64[])
    @test isequal(flat([[1.0, 2.0], Float64[], [3.0]]), [1.0, 2.0, 3.0])
    @test isequal(flat([2.5]), [2.5])
    @test isequal(flat([[1, 2], [3]]), [1, 2, 3])
    @test isequal(flat(Any[[1.0], 2.0]), Any[1.0, 2.0])
    cells = [fill(0.5, 10) for _ in 1:400]
    flat(cells)
    @test @allocated(flat(cells)) < 2 * sizeof(Float64) * 4000

    build(data) = @brm data begin
        log_k ~ Normal(0, 1)
        sigma ~ Exponential(1)
        @plate for i in eachindex(t)
            loc[i] = dose[i] .* exp.(-exp(log_k) .* t[i])
        end
        y ~ Normal(loc, sigma)
    end
    data = (; t=[[0.5, 1.5], Float64[], [0.7, 1.1, 2.0]], dose=[1.0, 2.0, 1.5],
        y=[[0.6, 0.2], Float64[], [1.0, 0.8, 0.3]])
    saved = deepcopy(data)
    brmi = build(data)
    emitted = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    source = join((sprint(Base.show_unquoted, d) for d in emitted.defs), "\n")
    @test occursin("loc_reader(loc_input_t, loc_input_dose, log_k)", source)
    @test occursin("ReactiveKernels.plate(loc_input_t, loc_input_dose, Ref(log_k)) do t, dose, log_k", source)
    # The reader returns its subject cells; the per-subject response reads
    # them cell by cell, so nothing is flattened.
    @test occursin("return cells", source)
    @test occursin("y[i] .~ Normal.(loc_cells[i], sigma)",
        sprint(Base.show_unquoted, emitted.main))
    for retired in ("cell_input", "subject_count", "loc_cell(", "init =", "reduce(vcat",
            "brm_flatten")
        @test !occursin(retired, source * sprint(Base.show_unquoted, emitted.main))
    end
    backend, problem = consumer_problem(brmi)
    names = coordinate_names(backend.model.layout)
    @test sort(names) == [:log_k, :sigma]
    ik, is = findfirst(==(:log_k), names), findfirst(==(:sigma), names)
    oracle(u) = begin
        k, sigma = exp(u[ik]), exp(u[is])
        logpdf(Normal(), u[ik]) + logpdf(Exponential(), sigma) + u[is] +
            sum(sum(logpdf.(Normal.(dose .* exp.(-k .* t), sigma), y); init=0.0)
                for (t, dose, y) in zip(data.t, data.dose, data.y))
    end
    stan = consumer_stan(brmi, "kernel-reader-shape")
    points = ([0.0, 0.0], [0.3, -0.2], [-0.4, 0.25])
    for u in points
        check_consumer_point(problem, u, oracle)
        check_consumer_stan(problem, stan, [:log_k => "log_k", :sigma => "sigma"], backend, u)
    end
    @test isequal(data, saved)

    # Evaluation cost is linear in the number of subjects: four times the
    # subjects allocate about four times as much, value and gradient alike.
    sized(groups) = (; t=[collect(range(0.5, 2.0; length=5)) for _ in 1:groups],
        dose=collect(range(1.0, 2.0; length=groups)), y=[fill(0.4, 5) for _ in 1:groups])
    allocations = map((100, 400)) do groups
        p = rk_logdensity_problem(RKBRMI(build(sized(groups)));
            ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        u = [0.1, -0.3]
        LogDensityProblems.logdensity(p, u)
        LogDensityProblems.logdensity_and_gradient(p, u)
        (@allocated(LogDensityProblems.logdensity(p, u)),
            @allocated(LogDensityProblems.logdensity_and_gradient(p, u)))
    end
    println("READER_ALLOCATIONS=", allocations); flush(stdout)
    @test first(allocations[2]) < 6 * first(allocations[1])
    @test last(allocations[2]) < 6 * last(allocations[1])

    # Cell locals named like the reader's own locals stay the cell's, and an
    # in-cell observation's argument readers carry only what they read.
    shadowing = @brm data begin
        log_k ~ Normal(0, 1)
        sigma ~ Exponential(1)
        @plate for i in eachindex(t)
            values = t[i] .* exp(log_k)
            cells = dose[i] .* exp.(-values)
            y[i] ~ normal(cells, sigma)
            loc[i] = cells
        end
    end
    shadowed = BRM._rk_emit_ast(BRM._brm_rk_plan(shadowing))
    source = join((sprint(Base.show_unquoted, d) for d in shadowed.defs), "\n")
    @test occursin("return cells_", source)
    @test occursin("y[i] .~ Normal.(loc_argument_y_1[i], sigma)",
        sprint(Base.show_unquoted, shadowed.main))
    backend, problem = consumer_problem(shadowing)
    stan = consumer_stan(shadowing, "kernel-reader-shadowing")
    for u in points
        check_consumer_point(problem, u, oracle)
        check_consumer_stan(problem, stan, [:log_k => "log_k", :sigma => "sigma"], backend, u)
    end

    # Scalar cells hold one value per subject, including a single subject;
    # their responses hold one number per subject and stay flat.
    for scalar_data in ((; dose=[1.0, 2.0, 1.5], y=[0.6, 1.1, 0.7]), (; dose=[1.2], y=[0.5]))
        scalar = @brm scalar_data begin
            log_k ~ Normal(0, 1)
            sigma ~ Exponential(1)
            @plate for i in eachindex(dose)
                loc[i] = dose[i] * exp(-exp(log_k))
            end
            y ~ Normal(loc, sigma)
        end
        backend, problem = consumer_problem(scalar)
        names = coordinate_names(backend.model.layout)
        jk, js = findfirst(==(:log_k), names), findfirst(==(:sigma), names)
        scalar_oracle(u) = logpdf(Normal(), u[jk]) + logpdf(Exponential(), exp(u[js])) +
            u[js] + sum(logpdf.(Normal.(scalar_data.dose .* exp(-exp(u[jk])), exp(u[js])),
                scalar_data.y))
        stan = consumer_stan(scalar, "kernel-reader-scalar-$(length(scalar_data.dose))")
        for u in points
            check_consumer_point(problem, u, scalar_oracle)
            check_consumer_stan(problem, stan, [:log_k => "log_k", :sigma => "sigma"], backend, u)
        end
    end
end

# A response holding one array per subject is observed per subject: RKPPL
# nested subject and entry plates over the kernel's subject cells, with the
# response bound as authored and no flattened response, row geometry port or
# row-aligned argument reader (todo 1jocaai). Covers a formula observation of
# a cell output and an in-cell observation, each against an independent
# oracle, finite differences, the compiled Stan program, its per-subject
# pointwise densities and withholding. Grouped-data arithmetic in a nested
# formula observation is `rk_grouped_observation_arguments.jl`'s `ordinary`.
@stestset "per-subject observations are RK nested plates" begin
    data = (; t=[[0.5, 1.5], Float64[], [0.7, 1.1, 2.0]], dose=[1.0, 2.0, 1.5],
        y=[[0.6, 0.2], Float64[], [1.0, 0.8, 0.3]])
    saved = deepcopy(data)
    models = (
        formula=(@brm data begin
            log_k ~ Normal(0, 1)
            sigma ~ Exponential(1)
            @plate for i in eachindex(t)
                loc[i] = dose[i] .* exp.(-exp(log_k) .* t[i])
            end
            y ~ Normal(loc, sigma)
        end),
        cell=(@brm data begin
            log_k ~ Normal(0, 1)
            sigma ~ Exponential(1)
            @plate for i in eachindex(t)
                m = dose[i] .* exp.(-exp(log_k) .* t[i])
                y[i] ~ normal(m, sigma)
                loc[i] = m
            end
        end))
    observed = (formula="y[i] .~ Normal.(loc_cells[i], sigma)",
        cell="y[i] .~ Normal.(loc_argument_y_1[i], sigma)")
    ext = Base.get_extension(BRM, :BayesianRegressionModelsReactiveKernelsExt)
    structure(program) = [(e.kind, e.depth) for e in recipe_inventory(program)
        if e.kind !== :ordinary]
    for (label, brmi) in pairs(models)
        emitted = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
        main = sprint(Base.show_unquoted, emitted.main)
        everything = main * join((sprint(Base.show_unquoted, d) for d in emitted.defs), "\n")
        @test occursin("@plate for i = eachindex(y)", main)
        @test occursin(observed[label], main)
        for retired in ("brm_flatten", "y_raw_response", "y_rows", "ones(length(")
            @test !occursin(retired, everything)
        end
        backend, problem = consumer_problem(brmi)
        @test isequal(backend.plan.columns[:y], data.y)
        names = coordinate_names(backend.model.layout)
        @test sort(names) == [:log_k, :sigma]
        ik, is = findfirst(==(:log_k), names), findfirst(==(:sigma), names)
        means(u) = [dose .* exp.(-exp(u[ik]) .* t) for (t, dose) in zip(data.t, data.dose)]
        prior(u) = logpdf(Normal(), u[ik]) + logpdf(Exponential(), exp(u[is])) + u[is]
        pointwise(u) = [logpdf.(Normal.(mu, exp(u[is])), y) for (mu, y) in zip(means(u), data.y)]
        oracle(u) = prior(u) + sum(sum(cell; init=0.0) for cell in pointwise(u))
        stan = consumer_stan(brmi, "nested-plates-$label")
        points = ([0.0, 0.0], [0.3, -0.2], [-0.4, 0.25])
        for u in points
            check_consumer_point(problem, u, oracle)
            check_consumer_stan(problem, stan, [:log_k => "log_k", :sigma => "sigma"], backend, u)
        end
        # The prepared density keeps the group plate with its observation
        # plate inside, one retained body whatever the number of subjects.
        translated = ext._rk_translated_plan(backend.plan)
        sampler = prepare_sampler(backend.model, translated, zeros(2);
            backend=AutoEnzyme(; mode=Enzyme.Reverse))
        @test (:plate, 1) in structure(sampler.kernel)
        # One array of densities per subject, empty subjects included.
        query = prepare_query(backend.model, translated, :pointwise)
        for u in points
            densities = Base.invokelatest(query, u).y
            @test length(densities) == length(data.y)
            @test all(isequal.(length.(densities), length.(data.y)))
            @test all(map((a, b) -> isapprox(a, b; atol=2e-12, rtol=2e-12),
                densities, pointwise(u)))
        end
        @test isequal(data, saved)
    end

    # Withholding the per-subject response drops its nested plate and density;
    # withholding the other response keeps it.
    z = [0.2, -0.1]
    both = @brm (; data..., z) begin
        log_k ~ Normal(0, 1)
        sigma ~ Exponential(1)
        @plate for i in eachindex(t)
            loc[i] = dose[i] .* exp.(-exp(log_k) .* t[i])
        end
        y ~ Normal(loc, sigma)
        z ~ Normal(log_k, 1)
    end
    for (hold, active) in ((:y, :z), (:z, :y))
        held = check_rk_source_roundtrip(RKBRMI(both; held_out=hold))
        @test BRM._rk_observed_names(held.plan) == (active,)
        @test occursin("@plate", sprint(Base.show_unquoted, BRM._rk_emit_ast(held.plan).main)) ==
            (active === :y)
        names = coordinate_names(held.model.layout)
        ik, is = findfirst(==(:log_k), names), findfirst(==(:sigma), names)
        oracle(u) = logpdf(Normal(), u[ik]) + logpdf(Exponential(), exp(u[is])) + u[is] +
            (active === :z ? sum(logpdf.(Normal(u[ik], 1), z)) :
                sum(sum(logpdf.(Normal.(dose .* exp.(-exp(u[ik]) .* t), exp(u[is])), y); init=0.0)
                    for (t, dose, y) in zip(data.t, data.dose, data.y)))
        problem = rk_logdensity_problem(held; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
        for u in ([0.0, 0.0], [0.3, -0.2])
            check_consumer_point(problem, u, oracle)
        end
    end
    @test isequal(data, saved)
end
