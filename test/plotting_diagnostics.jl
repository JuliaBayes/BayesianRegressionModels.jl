# Run with --project=research/adaptive_centering/plots: AoV remains optional.
using Test, BayesianRegressionModels, AlgebraOfVega, JSON

function objects(value)
    if value isa AbstractDict
        vcat([value], reduce(vcat, (objects(v) for v in values(value)); init=Any[]))
    elseif value isa AbstractVector
        reduce(vcat, (objects(v) for v in value); init=Any[])
    else
        Any[]
    end
end

@testset "optional AoV diagnostic specifications" begin
    @test !isnothing(Base.get_extension(BayesianRegressionModels,
                                       :BayesianRegressionModelsAlgebraOfVegaExt))
    draws = reshape(collect(1.0:15.0), 5, 3)
    posterior = to_vegalite(brm_posteriorplot(draws; x=[2, 4, 6], probs=[0.9, 0.8, 0.5]);
                            interactive=false)
    tables = filter(d -> haskey(d, "values"), objects(posterior))
    @test !isempty(tables)
    @test all(d -> !haskey(d, "params"), objects(posterior))
    @test_throws DimensionMismatch brm_posteriorplot(draws; x=[1, 2])
    @test_throws ErrorException brm_posteriorplot(draws; probs=[1.0])
    @test_throws ErrorException brm_posteriorplot(draws; probs=[0.5, 0.9])
    @test_throws ErrorException brm_posteriorplot(draws; probs=Float64[])
    @test_throws ErrorException brm_posteriorplot(fill(NaN, 2, 3))

    observations = (; x=[2, 4, 6], response=[5.0, 8.0, 11.0])
    observed = to_vegalite(brm_posteriorplot(draws; x=observations.x, observations);
                            interactive=false)
    @test any(d -> get(d, "size", nothing) == 24.5, objects(observed))
    larger_observed = to_vegalite(brm_posteriorplot(draws; x=observations.x,
        observations, observation_markersize=10); interactive=false)
    @test any(d -> get(d, "size", nothing) == 50, objects(larger_observed))
    @test filter(d -> haskey(d, "values"), objects(observed)) ==
          filter(d -> haskey(d, "values"), objects(larger_observed))

    rows = [(; coordinate=Float64(i), gradient=-2.0i,
               configuration=c, basis_label="Basis 01")
            for c in ("NCP", "Post-hoc", "Online") for i in 1:2000]
    gradient = to_vegalite(brm_gradientplot(rows); interactive=false)
    @test any(d -> get(d, "x", nothing) == "independent" &&
                   get(d, "y", nothing) == "independent", objects(gradient))
    @test all(d -> !haskey(d, "params"), objects(gradient))
    @test any(d -> haskey(d, "values") && length(d["values"]) == 6000, objects(gradient))
    @test sizeof(JSON.json(gradient)) > 48 * 1024 # No scientific data cap in BRM.

    pair_rows = [(; hyperparameter=exp(i/100), coordinate=Float64(i),
                   parameter=p, basis_label="Basis 20")
                 for p in ("Length scale", "Marginal SD") for i in 1:5]
    pair = to_vegalite(brm_pairplot(pair_rows); interactive=false)
    @test any(d -> get(d, "type", nothing) == "log", objects(pair))
    @test any(d -> get(d, "x", nothing) == "independent" &&
                   get(d, "y", nothing) == "independent", objects(pair))
    @test any(d -> haskey(d, "values") && length(d["values"]) == 10, objects(pair))
    larger_pair = to_vegalite(brm_pairplot(pair_rows; markersize=12, opacity=0.2);
                               interactive=false)
    @test any(d -> get(d, "size", nothing) == 72, objects(larger_pair))
    @test any(d -> get(d, "opacity", nothing) == 0.2, objects(larger_pair))
    @test only(filter(d -> haskey(d, "values"), objects(larger_pair)))["values"] ==
          only(filter(d -> haskey(d, "values"), objects(pair)))["values"]
    gp = (; coordinates=reshape(collect(1.0:10), 5, 2), marginal_sd=ones(5),
             length_scales=fill(0.5, 5, 1))
    gp_pair = to_vegalite(brm_pairplot(gp; bases=[2], markersize=12, opacity=0.2);
                          interactive=false)
    @test any(d -> get(d, "size", nothing) == 72, objects(gp_pair))
    @test all(r -> r["basis_label"] == "Basis 02",
        only(filter(d -> haskey(d, "values"), objects(gp_pair)))["values"])

    centering = [(; basis=b, predictor=p, centeredness=0.1b,
                    configuration=c) for b in 1:3 for p in ("Mean", "Log-SD")
                   for c in ("Online", "Post-hoc")]
    selected = to_vegalite(brm_centerednessplot(centering; compare=true); interactive=false)
    @test any(d -> get(d, "field", nothing) == "configuration", objects(selected))
    selected_values = only(filter(d -> haskey(d, "values"), objects(selected)))["values"]
    # The line and point layers may each carry a copy of the input table.
    @test Set((r["basis"], r["predictor"], r["centeredness"], r["configuration"])
              for r in selected_values) == Set(Tuple(r) for r in centering)

    losses = [(; centeredness=0.1i, loss=Float64(i), predictor="Mean",
                 basis_label="Basis 01", segment="one") for i in 0:10]
    loss_spec = to_vegalite(brm_centering_lossplot(losses); interactive=false)
    @test any(d -> haskey(d, "values") && length(d["values"]) == 11, objects(loss_spec))
    @test any(d -> get(d, "field", nothing) == "loss", objects(loss_spec))
    bounded_loss = to_vegalite(brm_centering_lossplot(losses; ylimits=(-1, 0));
                               interactive=false)
    @test any(d -> get(d, "domain", nothing) == [-1, 0], objects(bounded_loss))
    @test only(filter(d -> haskey(d, "values"), objects(bounded_loss)))["values"] ==
          only(filter(d -> haskey(d, "values"), objects(loss_spec)))["values"]

    loss_rows = Base.get_extension(BayesianRegressionModels,
        :BayesianRegressionModelsAlgebraOfVegaExt).loss_plot_rows
    inputs = [(; configuration=c, predictor="Mean", basis_label="Basis 01",
                 segment=string(i), centeredness=0.5(i-1), loss=v)
              for (c, vs) in (("A", [-0.8, -0.5, -0.2]), ("B", [2.0, 3.0, 4.0]))
              for (i, v) in enumerate(vs)]
    saved = copy(inputs)
    normalized = loss_rows(inputs, :minmax)
    @test inputs == saved
    @test getproperty.(normalized, :raw_loss) == getproperty.(inputs, :loss)
    @test getproperty.(normalized, :loss) ≈ [0.0, 0.5, 1.0, 0.0, 0.5, 1.0]
    @test argmin(getproperty.(normalized[1:3], :loss)) == argmin(getproperty.(inputs[1:3], :loss))
    @test loss_rows(inputs, :none) === inputs
    @test_throws ArgumentError loss_rows(inputs, :unknown)
    special = [merge(first(inputs), (; loss=v)) for v in (2.0, 2.0, missing, Inf)]
    result = loss_rows(special, :minmax)
    @test getproperty.(result[1:2], :loss) == [0.0, 0.0]
    @test ismissing(result[3].loss)
    @test result[4].loss == Inf
end
