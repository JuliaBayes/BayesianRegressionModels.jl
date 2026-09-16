# Scatter data for the centering figures: CASE MATRIX_DIR OUT
# Total pairs: the same post-hoc-position draws of three trials (min, mid, max selected centeredness)
# shown in CP, NCP and ACP coordinates against log tau. Ordinary pairs: the same for the conventional block.
isdefined(@__MODULE__, :RBesTCentering) || include(joinpath(@__DIR__, "model.jl"))
using .RBesTCentering, Serialization, Statistics, Printf
const RC = RBesTCentering
include(joinpath(@__DIR__, "run.jl"))

function extreme_indices(controls)
    selected = [argmin(controls)]
    remaining = setdiff(eachindex(controls), selected)
    push!(selected, remaining[argmin(abs.(controls[remaining] .- 0.5))])
    remaining = setdiff(eachindex(controls), selected)
    push!(selected, remaining[argmax(controls[remaining])])
end

function total_pairs(c, t, matrix, out)
    fit = deserialize(joinpath(matrix, "total_posthoc_position.jls"))
    selected = deserialize(joinpath(matrix, "total_selected_position.jls"))
    chosen = extreme_indices(selected.centeredness); rows = NamedTuple[]; labels = NamedTuple[]; error = 0.0
    location = only(t.block.A * t.block.location); scale = only(t.coords.scales)
    for (panel, k) in enumerate(chosen)
        i = selected.indices[k]; j = only(findall(==(i), vec(t.coords.totals))); cc = selected.centeredness[k]
        title = @sprintf("%d: %s, c=%.2f", panel, c.labels[j], cc)
        push!(labels, (; panel=title, trial=j, label=c.labels[j], centeredness=cc, parameter=fit.names[i]))
        for (column, weight) in (("1. CP", 1.0), ("2. NCP", 0.0), ("3. ACP (position)", cc)), s in axes(fit.positions, 2)
            ell = fit.positions[scale, s]; value = fit.positions[i, s]
            coordinate = weight * location + (value - location) * exp((weight - 1) * ell)
            weight == cc && column == "3. ACP (position)" && (error = max(error, abs(coordinate - fit.source_positions[i, s])))
            push!(rows, (; panel=title, column, draw=s, log_group_sd=ell, coordinate))
        end
    end
    @assert error < 1e-8 && length(rows) == 3 * 3 * N_DRAWS
    write_tsv(joinpath(out, "total_pairs.tsv"), rows); write_tsv(joinpath(out, "total_pairs_selection.tsv"), labels)
    write_tsv(joinpath(out, "total_pairs_audit.tsv"), [(; draws=N_DRAWS, source_coordinate_error=error)])
end

function ordinary_pairs(c, o, matrix, out)
    fit = deserialize(joinpath(matrix, "ordinary_posthoc_position.jls"))
    selected = deserialize(joinpath(matrix, "ordinary_selected_position.jls"))
    chosen = extreme_indices(selected); rows = NamedTuple[]; labels = NamedTuple[]; error = 0.0
    for (panel, k) in enumerate(chosen)
        i = fit.control_indices[k]; j = only(findall(==(i), o.effects)); cc = selected[k]
        title = @sprintf("%d: %s, c=%.2f", panel, c.labels[j], cc)
        push!(labels, (; panel=title, trial=j, label=c.labels[j], centeredness=cc, parameter=fit.names[i]))
        for (column, weight) in (("1. CP", 1.0), ("2. NCP", 0.0), ("3. ACP (position)", cc)), s in axes(fit.positions, 2)
            ell = fit.positions[o.log_scale, s]; z = fit.positions[i, s]   # model frame is NCP: z
            coordinate = z * exp(weight * ell)
            weight == cc && column == "3. ACP (position)" && (error = max(error, abs(coordinate - fit.source_positions[i, s])))
            push!(rows, (; panel=title, column, draw=s, log_group_sd=ell, coordinate))
        end
    end
    @assert error < 1e-8 && length(rows) == 3 * 3 * N_DRAWS
    write_tsv(joinpath(out, "ordinary_pairs.tsv"), rows); write_tsv(joinpath(out, "ordinary_pairs_selection.tsv"), labels)
    write_tsv(joinpath(out, "ordinary_pairs_audit.tsv"), [(; draws=N_DRAWS, source_coordinate_error=error)])
end

if abspath(PROGRAM_FILE) == @__FILE__
    case, matrix, out = ARGS
    c = RC.load_case(case); mkpath(out)
    t = RC.total_model(c, joinpath(matrix, "model")); o = RC.ordinary_model(c, joinpath(matrix, "model"))
    total_pairs(c, t, matrix, out); ordinary_pairs(c, o, matrix, out)
    println("RBEST_PAIRS_COMPLETE ", case)
end
