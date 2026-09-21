#!/usr/bin/env julia

# Focused executable refresh for the four Student-t catalogue rows (authoritative
# nu prior recovered from ASKurz f90517976) and the three ZIP rows (split
# count/psi components paired into joint declarations). This deliberately reuses
# the corpus probe builder but updates no unaffected capability row.
#
# Pairing invariant: bambi:zip_mu and bambi:zip_psi carry the SAME joint body,
# data columns, and family, so they share one probe_id. The joint program is
# built and compiled ONCE; zip_mu records direct evidence and zip_psi records
# inherited-identical-probe with evidence_from pointing at zip_mu's row. The
# script asserts the shared probe_id rather than assuming the pairing held.

using BayesianRegressionModels
using LogDensityProblems
using StanBlocks

include(joinpath(@__DIR__, "probe.jl"))

const TARGETS = Set([
    "kruschke:income_famsize",
    "kruschke:guber1999_base",
    "kruschke:guber1999_complement",
    "kruschke:guber1999_interaction",
    "bambi:zip_mu",
    "bambi:zip_psi",
    "bambi:plot_comp_zip",
])
const PAIRED_JOINT = Set(["bambi:zip_mu", "bambi:zip_psi"])

length(ARGS) >= 1 || error(
    "usage: refresh_student_zip.jl <expected-brm-sha> [refresh-output.tsv] [capability-results.tsv]",
)

expected_brm_sha = ARGS[1]
refresh_output = length(ARGS) >= 2 ? ARGS[2] :
    joinpath(@__DIR__, "student_zip_refresh.tsv")
capability_path = length(ARGS) >= 3 ? ARGS[3] :
    joinpath(@__DIR__, "capability_results.tsv")
translation_path = joinpath(@__DIR__, "translations.tsv")

brm_root = dirname(dirname(pathof(BayesianRegressionModels)))
brm_sha = readchomp(`git -C $brm_root rev-parse HEAD`)
brm_sha == expected_brm_sha ||
    error("active BRM SHA $brm_sha != expected $expected_brm_sha")
stanblocks_root = dirname(dirname(pathof(StanBlocks)))
stanblocks_sha = readchomp(`git -C $stanblocks_root rev-parse HEAD`)

translations = read_tsv(translation_path)
targets = filter(translations) do row
    row["variant"] == "inferred-family" &&
        string(row["source"], ':', row["key"]) in TARGETS
end
length(targets) == length(TARGETS) ||
    error("expected $(length(TARGETS)) target rows, found $(length(targets))")
Set(string(row["source"], ':', row["key"]) for row in targets) == TARGETS ||
    error("target row-key mismatch")

function translation_contract(row)
    row_key = string(row["source"], ':', row["key"])
    row["translation_status"] == "ready" ||
        error("$row_key: translation_status=$(row["translation_status"]) != ready")
    if startswith(row_key, "kruschke:")
        row["family_selected"] == "student_t" ||
            error("$row_key: family_selected=$(row["family_selected"]) != student_t")
        occursin("TDist(nu)", row["current_brm_body"]) ||
            error("$row_key: body lost free-nu LocationScale-TDist")
        occursin("nu ~ Exponential(29)", row["current_brm_body"]) ||
            error("$row_key: body lost authoritative nu ~ Exponential(29)")
    else
        row["family_selected"] == "zero_inflated_poisson" ||
            error("$row_key: family_selected=$(row["family_selected"])")
        occursin("count ~ ZeroInflatedPoisson(lambda, zi)", row["current_brm_body"]) ||
            error("$row_key: body lost joint ZeroInflatedPoisson declaration")
    end
end
foreach(translation_contract, targets)

# The pairing invariant: both halves of the split ZIP card share one probe.
probe_ids = Dict(
    string(row["source"], ':', row["key"]) => sha16(
        row["current_brm_body"] * "\0" * row["data_columns"] * "\0" *
        row["group_columns"] * "\0" * row["family_selected"],
    ) for row in targets
)
length(unique(values(probe_ids))) == length(TARGETS) - 1 ||
    error("expected exactly one shared probe_id, got $(probe_ids)")
probe_ids["bambi:zip_mu"] == probe_ids["bambi:zip_psi"] ||
    error("zip_mu/zip_psi probe_id mismatch: pairing broken")

capability_columns = split(first(readlines(capability_path)), '\t'; keepempty=true)
capabilities = read_tsv(capability_path)
capability_by = Dict(
    (row["source"], row["key"], row["variant"]) => row for row in capabilities
)

finite_values(values_iter) = all(
    value -> value isa Number ? isfinite(value) : all(isfinite, value),
    values_iter,
)

refresh_rows = Dict{String,String}[]
failed_rows = String[]
# One runtime per unique probe; the paired card reuses its partner's state.
probed_state = Dict{String,Dict{String,String}}()
probed_meta = Dict{String,Dict{String,String}}()
probed_row_index = Dict{String,String}()

for row in sort(targets; by=row -> string(row["source"], ':', row["key"]))
    row_key = string(row["source"], ':', row["key"])
    prior = capability_by[(row["source"], row["key"], row["variant"])]
    probe_id = probe_ids[row_key]

    if haskey(probed_state, probe_id)
        # Paired card: inherit the partner's already-recorded runtime state.
        # The count-submodel card carries the direct evidence; the psi card
        # inherits it (its historical formula names the submodel, not the
        # joint outcome).
        row_key == "bambi:zip_psi" ||
            error("unexpected inheritance direction for $row_key")
        state = copy(probed_state[probe_id])
        meta = probed_meta[probe_id]
        for (key, value) in state
            prior[key] = value
        end
        for key in (
            "row_index", "deployed", "source", "key", "variant",
            "family_selected", "family_selected_provenance", "semantic_route",
            "translation_status", "surface_support_class",
            "current_brm_body", "data_shape_assumptions",
        )
            prior[key] = row[key]
        end
        prior["surface_secondary_gap"] = ""
        prior["historical_formula"] = row["formula_claim"]
        prior["evidence_kind"] = "inherited-identical-probe"
        prior["evidence_from"] = probed_row_index[probe_id]
        push!(refresh_rows, Dict(
            "row_key" => row_key,
            "row_index" => row["row_index"],
            "probe_id" => probe_id,
            "evidence_kind" => "inherited-identical-probe",
            "evidence_from" => probed_row_index[probe_id],
            "candidate_sha" => brm_sha,
            "stanblocks_sha" => stanblocks_sha,
            "descriptor" => state["descriptor"],
            "stanc" => state["stanc"],
            "stan_code_sha256" => state["stan_code_sha256"],
            "stan_data_sha256" => state["stan_data_sha256"],
            "bridgestan_instantiate" => state["bridgestan_instantiate"],
            "dimension" => state["dimension"],
            "log_density_zero" => state["log_density"],
            "gradient_finite" => state["gradient_finite"],
            "descriptor_operations" => meta["operations"],
            "prediction_outputs" => meta["prediction_outputs"],
            "prediction_rows" => meta["prediction_rows"],
            "prediction_finite" => meta["prediction_finite"],
            "pointwise_outputs" => meta["pointwise_outputs"],
            "pointwise_rows" => meta["pointwise_rows"],
            "pointwise_finite" => meta["pointwise_finite"],
            "result" => "pass",
            "error" => "",
        ))
        continue
    end

    state = empty_state(probe_id)
    built = try
        build_static(row, state)
    catch err
        # A lowering-path escape (e.g. an uncaught StanBlocks trace error)
        # fails this row's receipt, not the whole refresh batch.
        state["static_error_stage"] = "escaped-build_static"
        state["static_error"] = compact_error(err)
        nothing
    end
    operations = ""
    prediction_outputs = ""
    pointwise_outputs = ""
    prediction_rows = ""
    pointwise_rows = ""
    prediction_finite = "not-run"
    pointwise_finite = "not-run"
    operation_error = ""

    if !isnothing(built) && state["stanc"] == "pass"
        try
            operation_names = Symbol[operation.name for operation in built.descriptor.operations]
            operations = join(string.(operation_names), ',')
            all(operation -> operation in operation_names,
                (:fit, :predict, :pointwise_loglik)) ||
                error("missing required operation; offered=$operation_names")

            problem = brm_execute(built.descriptor, :fit)
            state["bridgestan_instantiate"] = "pass"
            dimension = LogDensityProblems.dimension(problem)
            state["dimension"] = string(dimension)
            q = zeros(dimension)
            log_density, gradient = LogDensityProblems.logdensity_and_gradient(problem, q)
            state["log_density"] = string(log_density)
            state["gradient_finite"] = string(
                isfinite(log_density) && all(isfinite, gradient),
            )
            state["gradient_finite"] == "true" ||
                error("non-finite BridgeStan density/gradient at zero")

            prediction = brm_execute(
                built.descriptor, :predict; problem, draws=q, seed=20260729,
            )
            pointwise = brm_execute(
                built.descriptor, :pointwise_loglik; problem, draws=q, seed=20260729,
            )
            outcome = outcome_name(row["current_brm_body"])
            expected_prediction = Set([Symbol(outcome * "_gen")])
            expected_pointwise = Set([Symbol(outcome * "_likelihood")])
            Set(keys(prediction)) == expected_prediction ||
                error("unexpected prediction outputs $(keys(prediction))")
            Set(keys(pointwise)) == expected_pointwise ||
                error("unexpected pointwise outputs $(keys(pointwise))")

            prediction_outputs = join(sort!(string.(collect(keys(prediction)))), ',')
            pointwise_outputs = join(sort!(string.(collect(keys(pointwise)))), ',')
            prediction_rows = join(sort!(string.(length.(collect(values(prediction))))), ',')
            pointwise_rows = join(sort!(string.(length.(collect(values(pointwise))))), ',')
            prediction_finite = string(finite_values(values(prediction)))
            pointwise_finite = string(finite_values(values(pointwise)))
            prediction_finite == "true" || error("non-finite prediction output")
            pointwise_finite == "true" || error("non-finite pointwise output")
        catch err
            state["bridgestan_instantiate"] == "not-run" &&
                (state["bridgestan_instantiate"] = "fail")
            state["runtime_error"] = compact_error(err)
            operation_error = state["runtime_error"]
        end
    end

    passed = state["descriptor"] == "pass" && state["stanc"] == "pass" &&
        state["bridgestan_instantiate"] == "pass" &&
        state["gradient_finite"] == "true" && prediction_finite == "true" &&
        pointwise_finite == "true"
    passed || push!(failed_rows, row_key)
    probed_state[probe_id] = copy(state)
    probed_row_index[probe_id] = row["row_index"]
    probed_meta[probe_id] = Dict(
        "operations" => operations,
        "prediction_outputs" => prediction_outputs,
        "prediction_rows" => prediction_rows,
        "prediction_finite" => prediction_finite,
        "pointwise_outputs" => pointwise_outputs,
        "pointwise_rows" => pointwise_rows,
        "pointwise_finite" => pointwise_finite,
    )

    for (key, value) in state
        prior[key] = value
    end
    for key in (
        "row_index", "deployed", "source", "key", "variant",
        "family_selected", "family_selected_provenance", "semantic_route",
        "translation_status", "surface_support_class",
        "current_brm_body", "data_shape_assumptions",
    )
        prior[key] = row[key]
    end
    # The ZIP dispatcher defect is landed; the student rows carry no gap text.
    prior["surface_secondary_gap"] = ""
    prior["historical_formula"] = row["formula_claim"]
    prior["evidence_kind"] = "direct"
    prior["evidence_from"] = ""

    push!(refresh_rows, Dict(
        "row_key" => row_key,
        "row_index" => row["row_index"],
        "probe_id" => probe_id,
        "evidence_kind" => "direct",
        "evidence_from" => "",
        "candidate_sha" => brm_sha,
        "stanblocks_sha" => stanblocks_sha,
        "descriptor" => state["descriptor"],
        "stanc" => state["stanc"],
        "stan_code_sha256" => state["stan_code_sha256"],
        "stan_data_sha256" => state["stan_data_sha256"],
        "bridgestan_instantiate" => state["bridgestan_instantiate"],
        "dimension" => state["dimension"],
        "log_density_zero" => state["log_density"],
        "gradient_finite" => state["gradient_finite"],
        "descriptor_operations" => operations,
        "prediction_outputs" => prediction_outputs,
        "prediction_rows" => prediction_rows,
        "prediction_finite" => prediction_finite,
        "pointwise_outputs" => pointwise_outputs,
        "pointwise_rows" => pointwise_rows,
        "pointwise_finite" => pointwise_finite,
        "result" => passed ? "pass" : "fail",
        "error" => isempty(operation_error) ?
            (isempty(state["static_error"]) ? state["runtime_error"] : state["static_error"]) :
            operation_error,
    ))
end

refresh_columns = [
    "row_key", "row_index", "probe_id", "evidence_kind", "evidence_from",
    "candidate_sha", "stanblocks_sha",
    "descriptor", "stanc", "stan_code_sha256", "stan_data_sha256",
    "bridgestan_instantiate", "dimension", "log_density_zero",
    "gradient_finite", "descriptor_operations", "prediction_outputs",
    "prediction_rows", "prediction_finite", "pointwise_outputs",
    "pointwise_rows", "pointwise_finite", "result", "error",
]
write_tsv(refresh_output, refresh_columns, refresh_rows)
write_tsv(capability_path, capability_columns, capabilities)

println("candidate_sha=$brm_sha")
println("stanblocks_sha=$stanblocks_sha")
println("target_rows=$(length(refresh_rows))")
passed_count = count(row -> row["result"] == "pass", refresh_rows)
println("passed=$passed_count")
println("refresh_output=$(abspath(refresh_output))")
println("capability_output=$(abspath(capability_path))")
isempty(failed_rows) || error(
    "focused Student-t/ZIP refresh failed for $(join(failed_rows, ','))",
)
