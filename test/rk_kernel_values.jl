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

@stestset "unbound leading response keeps subject likelihood and Stan inv_logit" begin
    data = (; subject=[1, 2], x=[[0.5, 1.0], [1.5]],
        y=[[0.2, 0.7], [-0.3]])
    brmi = @brm data begin
        slope ~ 1 + (1 | p | subject)
        effect(slope, Intercept) ~ Normal(0, 1)
        sd(:, p) ~ Exponential(1)
        loc ~ kernel(z, x, y, slope) do zz, xx, yy, b
            base = inv_logit(xx .* b)
            shifted = b + base
            shifted_twice = shifted + b
            shifted_back = shifted_twice - b
            mu = shifted_back + 0.5 .* (b - base)
            sigma = inv_logit(b)
            zz ~ normal(mu, sigma)
            yy ~ normal(mu, sigma)
            mu
        end
    end
    sb = SBBRMI(brmi; mod=@__MODULE__, total_groups=())
    @test StanBlocks.stanc_check(BRM.stan_code(sb); warn_pedantic=false).ok
    @test any(d -> d.role === :observation && d.data_source === nothing &&
        !isempty(d.context), generative_plan(sb).declarations)
    backend = check_rk_source_roundtrip(RKBRMI(brmi))
    @test backend.plan isa BRM._RKValuePlan
    @test BRM._rk_observed_names(backend.plan) == (:y,)
    @test !haskey(backend.plan.columns, :z)
    view = brm_nlme_view(backend)
    @test view.group === :subject
    @test view.levels == [1, 2]
    @test view.rows == [[1], [2]]
    model = brm_nlme_model(backend; ad_backend=AutoEnzyme(; mode=Enzyme.Reverse))
    θ, σ, H = [0.4], Float64[], reshape([0.2, -0.3], 1, 2)
    values = brm_nlme_loglikelihoods(model, θ, σ, H)
    expected = [sum(logpdf.(Normal.(0.5 .* BRM.logistic.(data.x[i] .* (θ[1] + H[1, i])) .+
        1.5 * (θ[1] + H[1, i]),
        BRM.logistic(θ[1] + H[1, i])), data.y[i]))
        for i in eachindex(data.subject)]
    @test values ≈ expected rtol=1e-12
end

@stestset "only operator calls with two operands take Stan's scalar-with-array semantics" begin
    function bind(ex)
        bindings = Pair{Symbol,Any}[]
        BRM._rk_kernel_bind_calls(ex, @__MODULE__, bindings, Set{Symbol}()), last.(bindings)
    end
    # Unary `+`/`-` mean the same in Julia and Stan and keep their spelling.
    for ex in (:(-x), :(+x), :((-).(x)), :(exp.(-x)))
        @test bind(ex) == (ex, [])
    end
    for (ex, callable) in ((:(a + b), BRM._rk_kernel_stan_add), (:(a - b), BRM._rk_kernel_stan_sub),
            (:(a - b - c), BRM._rk_kernel_stan_sub), (:(-(xs...)), BRM._rk_kernel_stan_sub),
            (:((-).(a, b)), BRM._rk_kernel_stan_sub))
        bound, callables = bind(ex)
        @test callables == [callable]
        @test bound != ex
    end
    @test BRM._rk_kernel_stan_sub(1.0, [2.0, 3.0]) == [-1.0, -2.0]
    @test BRM._rk_kernel_stan_sub([1.0]...) == -1.0
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

@stestset "unbound in-cell response leaves the other likelihood active" begin
    data = (; index=[[1, 2, 1]], y=[[0.1, -0.2, 0.4]],
        catalog=[[0.2, 0.7]])
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
    backend, problem = consumer_problem(brmi)
    @test BRM._rk_observed_names(backend.plan) == (:y,)
    @test coordinate_names(backend.model.layout) == [:shift]
    stan = consumer_stan(brmi, "kernel-unbound-response")
    oracle(u) = logpdf(Normal(), u[1]) +
        sum(logpdf.(Normal.(data.catalog[1][data.index[1]] .+ u[1], 1), data.y[1]))
    for u in ([0.0], [0.2], [-0.3])
        check_consumer_point(problem, u, oracle)
        check_consumer_stan(problem, stan, [:shift => "shift"], backend, u)
    end
end

@stestset "ragged kernel cells preserve subject joins and original axes" begin
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
        ia = findfirst(==(Symbol("theta_Intercept")), names)
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
        stan = consumer_stan(brmi, "ragged-cell-$mode";
            mod=mode === :scope ? PublicKernelScope : @__MODULE__)
        mapping = [names[ia] => "pop_theta_beta_pop.1";
            names[it] => "b_p_subject_tau.1"; names[is] => "sigma";
            [names[iz[i]] => "b_p_subject_z_flat.$i" for i in 1:3]]
        for u in (zeros(6), collect(range(-0.2, 0.3; length=6)), fill(-0.1, 6))
            check_consumer_point(problem, u, oracle)
            check_consumer_stan(problem, stan, mapping, backend, u)
        end
        artifact = BRM.emit_rk_artifact(brmi; case_id="ragged-cell-$mode")
        # The cell is an inline plate in the main block: the artifact defines
        # the group-effect submodel and no reader kernel.
        @test !isempty(artifact.defs)
        @test all(d -> BRM._rk_source_definition(d).kind in (:function, :kernel, :rkppl), artifact.defs)
        @test !any(d -> endswith(string(BRM._rk_source_definition(d).name), "_reader"), artifact.defs)
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

# The `@plate for` statements of an emitted main block, and the indexed
# outputs each one writes.
emitted_plates(main) = [s for s in main.args
    if Meta.isexpr(s, :macrocall) && first(s.args) === Symbol("@plate")]
plate_writes(plate) = Symbol[first(first(cell.args).args)
    for cell in last(last(plate.args).args).args
    if Meta.isexpr(cell, :(=)) && Meta.isexpr(first(cell.args), :ref)]

# A kernel cell is emitted once, as the authored `@plate for` loop in the model
# body, with one indexed write per value the program reads (decision
# 1dsco4y). It was a `ReactiveKernels.@kernel <name>_reader` per output, each
# re-running the whole cell; earlier still a `1:count` plate passing every
# port as `Ref` and flattening with `reduce(vcat, cells; init=…)`, which
# reallocates once per subject on every evaluation and reverse pass.
@stestset "kernel cells are one inline plate, flattened in linear time" begin
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
    plan = BRM._brm_rk_plan(brmi)
    emitted = BRM._rk_emit_ast(plan)
    main = sprint(Base.show_unquoted, emitted.main)
    # Each data input is read through its own column, and the per-subject
    # response is bound as authored with no flattened copy (snag
    # rk-emission-grou-9dac9bc6, todo 1jocaai).
    @test sort!(collect(keys(plan.columns)); by=string) == [:dose, :t, :y]
    @test isequal(plan.columns[:y], data.y)
    @test isempty(emitted.defs)
    # The cell reads its per-subject inputs at the subject index and the model
    # value `log_k` whole; no do-block `Ref` operand (todo 0t3q6dl).
    cell = only(p for p in emitted_plates(emitted.main) if :loc_cells in plate_writes(p))
    @test plate_writes(cell) == [:loc_cells]
    @test occursin("@plate for i = eachindex(t)", main)
    @test occursin(r"loc_cells\[i\] = dose\[i\] \.\* exp\.\(.*log_k.*t\[i\]", main)
    # The per-subject response reads the cells cell by cell, so it is never
    # flattened. The authored `loc` stays a named value over the cells,
    # planned only when queried.
    @test occursin("loc = brm_flatten_cells(loc_cells)", main)
    @test occursin("y[i] .~ Normal.(loc_cells[i], sigma)", main)
    for retired in ("_reader", "ReactiveKernels.plate", "Ref(", "cell_input", "subject_count",
            "init =", "reduce(vcat", "brm_flatten_response", "raw_response")
        @test !occursin(retired, main)
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
    stan = consumer_stan(brmi, "kernel-cell-shape")
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
    println("CELL_ALLOCATIONS=", allocations); flush(stdout)
    @test first(allocations[2]) < 6 * first(allocations[1])
    @test last(allocations[2]) < 6 * last(allocations[1])

    # One plate writes every value the program reads from a cell, so the cell
    # runs once per subject however many values that is; its locals keep their
    # authored names, including ones the retired reader kernels used.
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
    shadowing_plan = BRM._brm_rk_plan(shadowing)
    shadowed = BRM._rk_emit_ast(shadowing_plan)
    shadowed_main = sprint(Base.show_unquoted, shadowed.main)
    @test isempty(shadowed.defs)
    cell = only(p for p in emitted_plates(shadowed.main) if :loc_cells in plate_writes(p))
    @test plate_writes(cell) == [:loc_cells, :loc_argument_y_1]
    @test occursin("values = t[i] .* exp(log_k)", shadowed_main)
    @test occursin("cells = dose[i] .* exp.(-values)", shadowed_main)
    @test occursin("loc_argument_y_1[i] = cells", shadowed_main)
    @test occursin("y[i] .~ Normal.(loc_argument_y_1[i], sigma)", shadowed_main)
    # The observed cell response keeps its kernel input port beside its own
    # column; observed per subject, both hold the authored arrays.
    @test sort!(collect(keys(shadowing_plan.columns)); by=string) == [:dose, :loc_input_y, :t, :y]
    @test isequal(shadowing_plan.columns[:y], data.y)
    backend, problem = consumer_problem(shadowing)
    stan = consumer_stan(shadowing, "kernel-cell-shadowing")
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
        stan = consumer_stan(scalar, "kernel-cell-scalar-$(length(scalar_data.dose))")
        for u in points
            check_consumer_point(problem, u, scalar_oracle)
            check_consumer_stan(problem, stan, [:log_k => "log_k", :sigma => "sigma"], backend, u)
        end
    end
end

module CountedCell
using BayesianRegressionModels, Distributions
const calls = Ref(0)
counted_live(t, picks, rate) = (calls[] += 1; t[picks] .* exp(rate))
build(data) = @brm data begin
    log_a ~ Normal(0, 1)
    sigma ~ Exponential(1)
    @plate for i in eachindex(t)
        m = counted_live(t[i], picks[i], log_a)
        z[i] ~ normal(2 .* m, sigma)
        w[i] ~ normal(m .- 1, sigma)
        loc[i] = m
    end
    y ~ Normal(loc, sigma)
end
end

# Every value the program reads from a cell is written by the one plate BRM
# emits for it, so each subject's cell should run once per density and per
# value and gradient. The retired per-output reader kernels ran it once per
# reader, three times here (co-report RKPPLBench:flag:emission-hunt on snag
# rk-emission-lift-0a8bc777). The shared-cell lowering from RK snag
# rkppl-plate-for-5f5a19db evaluates all outputs together.
@stestset "a multi-output cell is one plate evaluated per subject" begin
    data = (; t=[[0.5, 1.0, 1.5], [0.7, 1.3, 2.0], [0.4, 0.9], [1.1, 0.2]],
        picks=[[1, 2], [2], [1, 2], [1]],
        y=[[0.6, 0.2], [0.3], [0.8, 0.1], [0.5]], z=[[1.2, 0.4], [0.6], [1.6, 0.2], [1.0]],
        w=[[0.0, -0.4], [-0.7], [-0.2, -0.9], [-0.5]])
    saved = deepcopy(data)
    brmi = CountedCell.build(data)
    emitted = BRM._rk_emit_ast(BRM._brm_rk_plan(brmi))
    @test isempty(emitted.defs)
    cell = only(p for p in emitted_plates(emitted.main) if :loc_cells in plate_writes(p))
    @test plate_writes(cell) == [:loc_cells, :loc_argument_z_1, :loc_argument_w_1]
    backend, problem = consumer_problem(brmi)
    names = coordinate_names(backend.model.layout)
    @test sort(names) == [:log_a, :sigma]
    ia, is = findfirst(==(:log_a), names), findfirst(==(:sigma), names)
    oracle(u) = begin
        sigma = exp(u[is])
        cells = [t[p] .* exp(u[ia]) for (t, p) in zip(data.t, data.picks)]
        logpdf(Normal(), u[ia]) + logpdf(Exponential(), sigma) + u[is] +
            sum(sum(logpdf.(Normal.(m, sigma), y)) + sum(logpdf.(Normal.(2 .* m, sigma), z)) +
                sum(logpdf.(Normal.(m .- 1, sigma), w))
                for (m, y, z, w) in zip(cells, data.y, data.z, data.w))
    end
    for u in ([0.0, 0.0], [0.3, -0.2], [-0.4, 0.25])
        check_consumer_point(problem, u, oracle)
    end
    u = [0.1, -0.2]
    LogDensityProblems.logdensity_and_gradient(problem, u)
    CountedCell.calls[] = 0
    LogDensityProblems.logdensity(problem, u)
    println("CELL_CALLS_PER_DENSITY=", CountedCell.calls[]); flush(stdout)
    @test CountedCell.calls[] == length(data.t)
    CountedCell.calls[] = 0
    LogDensityProblems.logdensity_and_gradient(problem, u)
    println("CELL_CALLS_PER_GRADIENT=", CountedCell.calls[]); flush(stdout)
    @test CountedCell.calls[] == length(data.t)
    @test isequal(data, saved)
end

# A response holding one array per subject is observed per subject: RKPPL
# nested subject and entry plates over the kernel's subject cells, with the
# response bound as authored and no flattened response, row geometry port or
# row-aligned argument reader (todo 1jocaai). Covers a formula observation of
# a cell output, the same with grouped-data arithmetic, and an in-cell
# observation, each against an independent oracle, finite differences, the
# compiled Stan program, its per-subject pointwise densities and withholding.
@stestset "per-subject observations are RK nested plates" begin
    data = (; t=[[0.5, 1.5], Float64[], [0.7, 1.1, 2.0]], dose=[1.0, 2.0, 1.5],
        y=[[0.6, 0.2], Float64[], [1.0, 0.8, 0.3]],
        reference=[[0.1, 0.0], Float64[], [0.2, -0.1, 0.05]])
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
        arithmetic=(@brm data begin
            log_k ~ Normal(0, 1)
            sigma ~ Exponential(1)
            @plate for i in eachindex(t)
                loc[i] = dose[i] .* exp.(-exp(log_k) .* t[i])
            end
            y ~ Normal(loc - reference, sigma)
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
        arithmetic="y[i] .~ Normal.(loc_argument_y_1[i], sigma)",
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
        for retired in ("brm_flatten_response", "y_raw_response", "y_rows", "ones(length(")
            @test !occursin(retired, everything)
        end
        backend, problem = consumer_problem(brmi)
        @test isequal(backend.plan.columns[:y], data.y)
        names = coordinate_names(backend.model.layout)
        @test sort(names) == [:log_k, :sigma]
        ik, is = findfirst(==(:log_k), names), findfirst(==(:sigma), names)
        shift = label === :arithmetic ? data.reference : [zero(t) for t in data.t]
        means(u) = [dose .* exp.(-exp(u[ik]) .* t) .- r
            for (t, dose, r) in zip(data.t, data.dose, shift)]
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
        @test occursin("y[i] .~", sprint(Base.show_unquoted, BRM._rk_emit_ast(held.plan).main)) ==
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
