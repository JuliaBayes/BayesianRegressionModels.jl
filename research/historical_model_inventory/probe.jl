#!/usr/bin/env julia

using BayesianRegressionModels
using Distributions
using LogDensityProblems
using LogExpFunctions: logit, logistic
using Random
using SHA
using StanBlocks
using WarmupHMC

const DEFAULT_INPUT = joinpath(@__DIR__, "translations.tsv")
const DEFAULT_OUTPUT = joinpath(@__DIR__, "capability_results.tsv")
const DEFAULT_CACHE = joinpath(tempdir(), "brm-historical-model-inventory")

tsv_unescape(value) = replace(value,
    "\\t" => "\t", "\\n" => "\n", "\\r" => "\r", "\\\\" => "\\")
tsv_escape(value) = replace(string(value),
    '\\' => "\\\\", '\t' => "\\t", '\n' => "\\n", '\r' => "\\r")

function read_tsv(path)
    lines = readlines(path)
    columns = split(first(lines), '\t'; keepempty=true)
    [Dict(columns .=> tsv_unescape.(split(line, '\t'; keepempty=true)))
     for line in Iterators.drop(lines, 1)]
end

function write_tsv(path, columns, rows)
    open(path, "w") do io
        println(io, join(columns, '\t'))
        for row in rows
            println(io, join((tsv_escape(get(row, column, "")) for column in columns), '\t'))
        end
    end
end

sha16(value) = bytes2hex(sha256(value))[1:16]

function compact_error(err, bt=catch_backtrace())
    text = sprint(showerror, err, bt)
    text = replace(text, '\t' => ' ', '\r' => ' ', '\n' => ' ')
    length(text) <= 1200 ? text : first(text, 1200) * "…"
end

split_csv(value) = isempty(value) ? String[] : split(value, ',')

function outcome_name(body)
    matches = collect(eachmatch(
        r"(?m)^\s*([A-Za-z_]\w*)\s*~\s*(?:Normal|LogNormal|BernoulliLogit|BinomialLogit|BetaBinomial2|Poisson|ZeroInflatedPoisson|NegativeBinomial2|OrderedLogistic|Beta|Gamma|Weibull|LocationScale)\(",
        body,
    ))
    isempty(matches) ? "" : last(matches).captures[1]
end

function categorical_columns(formula)
    values = String[]
    for m in eachmatch(r"\b(?:factor|C|S|mo)\(\s*([A-Za-z_]\w*)", formula)
        value = m.captures[1]
        value in values || push!(values, value)
    end
    values
end

function trial_columns(body)
    m = match(r"(?:BinomialLogit|BetaBinomial2)\(\s*([^,]+),", body)
    isnothing(m) && return String[]
    unique([x.match for x in eachmatch(r"\b[A-Za-z_]\w*\b", m.captures[1])])
end

function scalar_trials(body)
    m = match(r"BinomialLogit\(\s*(\d+)\s*,\s*log_odds\)", body)
    isnothing(m) ? nothing : parse(Int, m.captures[1])
end

function synthetic_data(row)
    columns = sort(unique(split_csv(row["data_columns"])))
    groups = Set(split_csv(row["group_columns"]))
    categoricals = Set(categorical_columns(row["formula_claim"]))
    trials = Set(trial_columns(row["current_brm_body"]))
    scalar_trial_count = scalar_trials(row["current_brm_body"])
    outcome = outcome_name(row["current_brm_body"])
    family = row["family_selected"]
    n = 12
    rng = Xoshiro(0x20260728)

    data = Dict{String,Any}()
    for column in columns
        if column in groups
            data[column] = repeat(1:3, inner=4)
        elseif column in categoricals
            data[column] = repeat(1:3, inner=4)
        elseif column in trials
            data[column] = fill(8, n)
        else
            # Positive values keep historical log()/sqrt()/exposure expressions in
            # domain while remaining non-constant for population effects.
            data[column] = 0.25 .+ rand(rng, n)
        end
    end

    if !isempty(outcome)
        data[outcome] = if family == "gaussian"
            randn(rng, n)
        elseif family in ("lognormal", "gamma", "weibull")
            0.25 .+ rand(rng, n)
        elseif family == "bernoulli"
            collect(Int, isodd.(1:n))
        elseif family == "binomial"
            isnothing(scalar_trial_count) ? fill(2, n) :
                scalar_trial_count == 1 ? collect(Int, isodd.(1:n)) :
                fill(min(2, scalar_trial_count), n)
        elseif family == "beta_binomial"
            fill(2, n)
        elseif family in ("poisson", "negativebinomial", "zero_inflated_poisson")
            collect(mod.(1:n, 4))
        elseif family == "ordered_logistic"
            repeat(1:4, inner=3)
        elseif family == "beta"
            collect(range(0.1, 0.9; length=n))
        else
            randn(rng, n)
        end
    end

    names = Tuple(Symbol.(sort(collect(keys(data)))))
    NamedTuple{names}(Tuple(data[string(name)] for name in names))
end

function empty_state(probe_id)
    Dict{String,String}(
        "probe_id" => probe_id,
        "brmi_parse" => "not-run",
        "brmi_eval" => "not-run",
        "sbbrmi_lower" => "not-run",
        "descriptor" => "not-run",
        "stan_transpile" => "not-run",
        "stanc" => "not-run",
        "stan_code_sha256" => "",
        "stan_data_sha256" => "",
        "static_error_stage" => "",
        "static_error" => "",
        "bridgestan_instantiate" => "not-run",
        "dimension" => "",
        "log_density" => "",
        "gradient_finite" => "not-run",
        "runtime_error" => "",
        "warmuphmc" => "not-run",
        "warmuphmc_draws" => "",
        "warmuphmc_error" => "",
    )
end

function build_static(row, state)
    body = row["current_brm_body"]
    try
        Meta.parse("begin\n$body\nend")
        state["brmi_parse"] = "pass"
    catch err
        state["brmi_parse"] = "fail"
        state["static_error_stage"] = "brmi_parse"
        state["static_error"] = compact_error(err)
        return nothing
    end

    data = synthetic_data(row)
    brmi = try
        wrapped = BayesianRegressionModels._brm(body; df=data)
        value = Core.eval(@__MODULE__, wrapped)
        state["brmi_eval"] = "pass"
        value
    catch err
        state["brmi_eval"] = "fail"
        state["static_error_stage"] = "brmi_eval"
        state["static_error"] = compact_error(err)
        return nothing
    end

    sb = try
        value = SBBRMI(brmi; mod=@__MODULE__)
        state["sbbrmi_lower"] = "pass"
        value
    catch err
        state["sbbrmi_lower"] = "fail"
        state["static_error_stage"] = "sbbrmi_lower"
        state["static_error"] = compact_error(err)
        return nothing
    end

    descriptor = try
        name = Symbol(replace(row["source"] * "_" * row["key"], r"\W" => "_"))
        value = brm_descriptor(sb; name)
        state["descriptor"] = "pass"
        value
    catch err
        state["descriptor"] = "fail"
        state["static_error_stage"] = "descriptor"
        state["static_error"] = compact_error(err)
        return nothing
    end

    code = try
        value = brm_execute(descriptor, :transpile)
        state["stan_transpile"] = "pass"
        value
    catch err
        state["stan_transpile"] = "fail"
        state["static_error_stage"] = "stan_transpile"
        state["static_error"] = compact_error(err)
        return nothing
    end
    state["stan_code_sha256"] = bytes2hex(sha256(code))
    # Model construction registers composed families (e.g. `brm_vector_prior_*`)
    # via `Core.eval`; a trace in this same frame would resolve methods at the
    # frame's world age and die with "missing `lpxf_expr`" on first use (see the
    # `stan_code(::SBBRMI)` convention in src/sbimpl.jl). Re-enter in the
    # current world — there is no `stan_data(::SBBRMI)` wrapper to forward to.
    stan_data = Base.invokelatest(StanBlocks.stan_data, sb.model)
    state["stan_data_sha256"] = bytes2hex(sha256(repr(sort(collect(stan_data); by=first))))

    try
        result = StanBlocks.stanc_check(code; warn_pedantic=false)
        state["stanc"] = result.ok ? "pass" : "fail"
        if !result.ok
            state["static_error_stage"] = "stanc"
            state["static_error"] = first(replace(result.output, '\n' => ' '), min(1200, length(result.output)))
            return nothing
        end
    catch err
        state["stanc"] = "fail"
        state["static_error_stage"] = "stanc"
        state["static_error"] = compact_error(err)
        return nothing
    end
    (; sb, descriptor, code)
end

function run_runtime!(built, state; cache_dir, sample=false, sample_draws=25)
    isnothing(built) && return
    mkpath(cache_dir)
    path = joinpath(cache_dir, state["stan_code_sha256"] * ".stan")
    problem = try
        # Same-frame world-age convention as `build_static` above: the BRM
        # wrapper re-enters `StanBlocks.stan_instantiate` in the current world.
        value = BayesianRegressionModels.stan_instantiate(
            built.sb;
            path,
            make_args=["O=0", "STAN_THREADS=true"],
        )
        state["bridgestan_instantiate"] = "pass"
        value
    catch err
        state["bridgestan_instantiate"] = "fail"
        state["runtime_error"] = compact_error(err)
        return
    end

    dimension = LogDensityProblems.dimension(problem)
    state["dimension"] = string(dimension)
    q = zeros(dimension)
    try
        lp = LogDensityProblems.logdensity(problem, q)
        lp2, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
        state["log_density"] = string(lp)
        state["gradient_finite"] = string(isfinite(lp) && isfinite(lp2) &&
                                                   all(isfinite, gradient))
        if !(isfinite(lp) && isfinite(lp2) && all(isfinite, gradient))
            state["runtime_error"] = "non-finite density or gradient at zero initialization"
            return
        end
    catch err
        state["gradient_finite"] = "false"
        state["runtime_error"] = compact_error(err)
        return
    end

    sample || return
    try
        fit = WarmupHMC.adaptive_warmup_mcmc(
            Xoshiro(0x20260728),
            problem;
            init=q,
            n_draws=sample_draws,
        )
        state["warmuphmc"] = "pass"
        state["warmuphmc_draws"] = string(size(fit.posterior_position, 2))
    catch err
        state["warmuphmc"] = "fail"
        state["warmuphmc_error"] = compact_error(err)
    end
end

function expanded_results(rows, states, direct_rows)
    outputs = Dict{String,String}[]
    for (index, row) in enumerate(rows)
        output = Dict(
            "row_index" => row["row_index"],
            "deployed" => row["deployed"],
            "source" => row["source"],
            "key" => row["key"],
            "variant" => row["variant"],
            "family_selected" => row["family_selected"],
            "family_selected_provenance" => row["family_selected_provenance"],
            "semantic_route" => row["semantic_route"],
            "translation_status" => row["translation_status"],
            "surface_support_class" => row["surface_support_class"],
            "surface_secondary_gap" => row["surface_secondary_gap"],
            "historical_formula" => row["formula_claim"],
            "current_brm_body" => row["current_brm_body"],
            "data_shape_assumptions" => row["data_shape_assumptions"],
        )
        if row["translation_status"] == "ready"
            probe_id = sha16(row["current_brm_body"] * "\0" * row["data_columns"] * "\0" *
                             row["group_columns"] * "\0" * row["family_selected"])
            if haskey(states, probe_id)
                merge!(output, states[probe_id])
                direct = direct_rows[probe_id]
                output["evidence_kind"] = direct == index ? "direct" :
                    direct == 0 ? "not-run" : "inherited-identical-probe"
                output["evidence_from"] = direct == 0 ? "" : string(direct)
            else
                merge!(output, empty_state(probe_id))
                output["evidence_kind"] = "not-run"
                output["evidence_from"] = ""
            end
        else
            merge!(output, empty_state(""))
            output["evidence_kind"] = "not-probed"
            output["evidence_from"] = ""
        end
        push!(outputs, output)
    end
    outputs
end

function probe(; input=DEFAULT_INPUT, output=DEFAULT_OUTPUT, runtime=false,
               sample=false, sample_draws=25, cache_dir=DEFAULT_CACHE, limit=typemax(Int))
    rows = read_tsv(input)
    states = Dict{String,Dict{String,String}}()
    built_models = Dict{String,Any}()
    direct_rows = Dict{String,Int}()
    ready_seen = 0

    for (index, row) in enumerate(rows)
        row["translation_status"] == "ready" || continue
        probe_id = sha16(row["current_brm_body"] * "\0" * row["data_columns"] * "\0" *
                         row["group_columns"] * "\0" * row["family_selected"])
        haskey(states, probe_id) && continue
        ready_seen += 1
        ready_seen > limit && break
        state = empty_state(probe_id)
        states[probe_id] = state
        direct_rows[probe_id] = index
        println("STATIC\t$ready_seen\t$probe_id\t$(row["source"]):$(row["key"])\t$(row["variant"])")
        built = build_static(row, state)
        built_models[probe_id] = built
        if runtime && state["stanc"] == "pass"
            println("RUNTIME\t$ready_seen\t$probe_id\t$(state["stan_code_sha256"])")
            run_runtime!(built, state; cache_dir, sample, sample_draws)
        end
        results = expanded_results(rows, states, direct_rows)
        write_tsv(output, collect(keys(first(results))), results)
    end

    # Rows beyond a debugging limit still need a state so expansion remains total.
    for row in rows
        row["translation_status"] == "ready" || continue
        probe_id = sha16(row["current_brm_body"] * "\0" * row["data_columns"] * "\0" *
                         row["group_columns"] * "\0" * row["family_selected"])
        haskey(states, probe_id) && continue
        states[probe_id] = empty_state(probe_id)
        direct_rows[probe_id] = 0
    end
    results = expanded_results(rows, states, direct_rows)
    write_tsv(output, collect(keys(first(results))), results)

    unique_ready = length(states)
    stanc_pass = count(state -> state["stanc"] == "pass", values(states))
    runtime_pass = count(state -> state["gradient_finite"] == "true", values(states))
    sample_pass = count(state -> state["warmuphmc"] == "pass", values(states))
    println("unique_ready_probes=$unique_ready")
    println("stanc_pass=$stanc_pass")
    println("finite_bridgestan=$runtime_pass")
    println("warmuphmc_pass=$sample_pass")
    println("output=$(abspath(output))")
    results
end

if abspath(PROGRAM_FILE) == @__FILE__
    runtime = "--runtime" in ARGS
    sample = "--sample" in ARGS
    input_arg = findfirst(startswith("--input="), ARGS)
    output_arg = findfirst(startswith("--output="), ARGS)
    limit_arg = findfirst(startswith("--limit="), ARGS)
    draws_arg = findfirst(startswith("--sample-draws="), ARGS)
    input = isnothing(input_arg) ? DEFAULT_INPUT : split(ARGS[input_arg], '='; limit=2)[2]
    output = isnothing(output_arg) ? DEFAULT_OUTPUT : split(ARGS[output_arg], '='; limit=2)[2]
    limit = isnothing(limit_arg) ? typemax(Int) : parse(Int, split(ARGS[limit_arg], '='; limit=2)[2])
    sample_draws = isnothing(draws_arg) ? 25 : parse(Int, split(ARGS[draws_arg], '='; limit=2)[2])
    probe(; input, output, runtime, sample, sample_draws, limit)
end
