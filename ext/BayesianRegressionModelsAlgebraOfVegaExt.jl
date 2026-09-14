module BayesianRegressionModelsAlgebraOfVegaExt

using AlgebraOfVega
using BayesianRegressionModels
using Statistics

const BRM = BayesianRegressionModels

const COLORS = ["#0B7BEC", "#E67E22", "#16877A", "#984EA3"]
const BANDS = [:q05 => :q95, :q10 => :q90, :q25 => :q75]

"""Posterior quantile ribbons; these are not automatically predictive intervals."""
function posteriorplot(curves; x=:time, ylabel="Response", title="",
                       observations=nothing, observed_y=:response, logscale=false,
                       xlabel="Observation", bands=BANDS)
    ribbon = data(curves) * mapping(x => xlabel, :q50 => ylabel) *
        lineribbon(bands=bands)
    layers = isnothing(observations) ? ribbon : ribbon +
        data(observations) * mapping(x, observed_y) *
        visual(Scatter; color="#252525", opacity=0.65, markersize=3)
    layers * config(width=620, height=320, title=title,
        scales=scales(Y=(; scale=logscale ? log10 : identity)))
end

"""Coordinates versus hyperparameters, with an independently scaled panel per pair."""
function coordinateplot(rows; title="", logx=true)
    data(rows) * mapping(:hyperparameter => "Hyperparameter position",
        :coordinate => "Coordinate position"; col=:parameter, row=:basis_label,
        color=:basis_label) * visual(Scatter; opacity=0.12, markersize=2) *
        config(width=260, height=190, title=title,
            facet=(; linkxaxes=:none, linkyaxes=:none),
            scales=scales(X=(; scale=logx ? log10 : identity),
                          Color=(; palette=COLORS)))
end

"""Compare centering profiles without knowing a model's carrier names."""
function centerednessplot(rows; title="Selected centeredness", compare=false)
    axes = compare ? mapping(:basis => "Basis frequency",
        :centeredness => "Centeredness"; col=:predictor, color=:configuration) :
        mapping(:basis => "Basis frequency", :centeredness => "Centeredness";
                color=:predictor)
    (data(rows) * axes * visual(Lines; linewidth=2) +
     data(rows) * axes * visual(Scatter; markersize=5)) *
        config(width=compare ? 460 : 760, height=300, title=title,
               axis=(; limits=(nothing, (0, 1))),
               scales=scales(Color=(; palette=COLORS)))
end

"""Candidate losses supplied by the selector, with no loss formula in the renderer."""
function lossplot(rows; ylabel="Loss", title="Centering objective")
    data(rows) * mapping(:centeredness => "Candidate centeredness", :loss => ylabel;
                        col=:predictor, color=:basis_label, group=:segment) *
        visual(Lines; linewidth=2) * config(width=460, height=300, title=title,
            facet=(; linkyaxes=:none), scales=scales(Color=(; palette=COLORS)))
end

function BRM.brm_posteriorplot(draws::AbstractMatrix; x=axes(draws, 2),
        probs=[0.95, 0.8, 0.5], kwargs...)
    length(x) == size(draws, 2) || throw(DimensionMismatch(
        "Posterior plot x values must match the output elements (matrix columns)"))
    all(p -> isfinite(p) && 0 < p < 1, probs) || error(
        "Posterior interval probabilities must lie strictly between zero and one")
    !isempty(probs) && issorted(probs; rev=true) || error(
        "Posterior interval probabilities must be nonempty and ordered outermost first")
    bands = [Symbol(:lower_, i) => Symbol(:upper_, i) for i in eachindex(probs)]
    interval_names = Tuple(Iterators.flatten((first(b), last(b)) for b in bands))
    rows = map(eachindex(x)) do i
        values = view(draws, :, i)
        all(isfinite, values) || error("Posterior plot has non-finite values at element $i")
        intervals = Tuple(Iterators.flatten((quantile(values, (1-p)/2),
                                             quantile(values, (1+p)/2)) for p in probs))
        merge((; x=x[i], q50=quantile(values, 0.5)), NamedTuple{interval_names}(intervals))
    end
    posteriorplot(rows; x=:x, bands, kwargs...)
end

function BRM.brm_posteriorplot(d::BRM.BRMDescriptor, draws::AbstractMatrix, names;
        logical::Symbol, role=nothing, kwargs...)
    BRM.brm_posteriorplot(BRM.brm_output_draws(d, draws, names; logical, role);
                         ylabel=string(logical), kwargs...)
end

function BRM.brm_ppcplot(d::BRM.BRMDescriptor, draws::AbstractMatrix;
        problem, seed::Integer, response::Symbol, x=nothing, kwargs...)
    predicted = BRM.brm_predictive_draws(d, draws; problem, seed)
    haskey(predicted, response) || error("No predictive output for response $response")
    y = BRM.column_data(d.plan.parent, response)
    y isa AbstractVector || error("PPC overlay needs a vector response")
    xs = isnothing(x) ? collect(eachindex(y)) : collect(x)
    length(xs) == length(y) || throw(DimensionMismatch("PPC x and observations differ"))
    BRM.brm_posteriorplot(getproperty(predicted, response); x=xs,
        observations=(; x=xs, response=y), ylabel=string(response), kwargs...)
end

function BRM.brm_pairplot(gp::NamedTuple; bases=axes(gp.coordinates, 2), title="")
    rows = [(; basis_label="Basis $(lpad(b, 2, '0'))",
              parameter=axis == 0 ? "Marginal SD" : "Length scale $(axis)",
              hyperparameter=axis == 0 ? gp.marginal_sd[i] : gp.length_scales[i, axis],
              coordinate=gp.coordinates[i, b])
            for b in bases for axis in 0:size(gp.length_scales, 2)
            for i in axes(gp.coordinates, 1)]
    all(row -> isfinite(row.coordinate), rows) || error(
        "Pair plot contains non-finite transformed coordinates; inspect the diagnostic finite mask")
    coordinateplot(rows; title)
end

function BRM.brm_pairplot(d::BRM.BRMDescriptor, draws::AbstractMatrix, names;
        predictor::Symbol, term::Symbol, centeredness=nothing, kwargs...)
    gp = BRM.hsgp_coordinate_draws(d, draws, names; predictor, term, centeredness)
    BRM.brm_pairplot(gp; kwargs...)
end

BRM.brm_centerednessplot(rows; kwargs...) = centerednessplot(rows; kwargs...)
BRM.brm_centering_lossplot(rows; kwargs...) = lossplot(rows; kwargs...)

"""Local prototype: actual coordinate gradients, with unlinked axes in every facet."""
function BRM.brm_gradientplot(rows; title="Coordinate–gradient diagnostic",
                              opacity=0.15, markersize=2)
    data(rows) * mapping(:coordinate => "Coordinate position",
        :gradient => "Log-density gradient"; col=:configuration, row=:basis_label,
        color=:basis_label) * visual(Scatter; opacity, markersize) *
        config(width=300, height=210, title=title,
            facet=(; linkxaxes=:none, linkyaxes=:none),
            scales=scales(Color=(; palette=COLORS)))
end

end
