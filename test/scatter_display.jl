# Run with --project=research/adaptive_centering/plots.
using Test, BayesianRegressionModels, AlgebraOfVega, CairoMakie, Statistics
include("../research/adaptive_centering/plots/scatter_display.jl")

@testset "central marginal scatter display, not posterior filtering" begin
    values = collect(1.0:1000)
    original = copy(values)
    @test collect(scatter_display_limits(values)) ≈ quantile(values, [0.0125, 0.9875])
    @test values == original
    @test scatter_display_limits(values; mass=1) == (1, 1000)
    @test isnothing(scatter_display_limits(fill(2.0, 10)))
    @test_throws ArgumentError scatter_display_limits(values; mass=0)
    @test_throws ArgumentError scatter_display_limits(values; mass=1.01)
    @test_throws ArgumentError scatter_display_limits(Float64[])
    @test_throws ArgumentError scatter_display_limits([1.0, Inf])

    rows = [(; coordinate=s*i, gradient=-i/s,
        configuration=c, basis_label=b)
        for (b, s) in (("Basis 01", 1.0), ("Basis 20", 100.0))
        for c in ("NCP", "Online") for i in values]
    plot = brm_gradientplot(rows)
    spec_before = to_vegalite(plot; interactive=false)
    fig = Figure(size=(900, 700))
    grid = sdraw!(fig[1, 1], plot)
    receipt = zoom_scatter_axes!(grid)
    @test size(grid) == (2, 2)
    @test length(receipt) == 4
    @test all(r -> r.points == 1000 && r.visible == 974, receipt)
    @test Set(r.xhigh for r in receipt) == Set([987.5125, 98751.25])
    @test to_vegalite(plot; interactive=false) == spec_before
    for cell in grid
        entry = only(cell.entries)
        @test length(entry.positional[1]) == 1000
        @test cell.axis.limits[] == (scatter_display_limits(entry.positional[1]),
                                     scatter_display_limits(entry.positional[2]))
    end

    pair_rows = [(; hyperparameter=exp(i/100), coordinate=i,
        parameter=p, basis_label="Basis 01") for p in ("SD", "Length") for i in values]
    pairgrid = sdraw!(Figure()[1, 1], brm_pairplot(pair_rows))
    @test all(r -> r.xlow > 0, zoom_scatter_axes!(pairgrid))
end
