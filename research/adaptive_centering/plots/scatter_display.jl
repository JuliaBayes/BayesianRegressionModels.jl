using Statistics

"""Central marginal display limits, without modifying or filtering observations.

Each axis retains `mass` of its own empirical marginal range. This is not a
joint probability region. A constant marginal keeps automatic axis limits.
"""
function scatter_display_limits(values; mass=0.975)
    isfinite(mass) && 0 < mass <= 1 || throw(ArgumentError("mass must be in (0, 1]"))
    isempty(values) && throw(ArgumentError("Cannot zoom an empty scatter"))
    all(isfinite, values) || throw(ArgumentError("Nonfinite scatter values"))
    lo, hi = quantile(values, [(1-mass)/2, (1+mass)/2])
    lo == hi ? nothing : (lo, hi)
end

"""Zoom native AoV/AoG scatter axes using the actual data in each drawn facet.

Only Makie axis limits change. The AoG entries, all underlying points and the
input AoV specification remain intact. The returned receipt records the exact
limits and visible-point counts; tails are not clamped onto the boundary.
See AoG's documented AxisEntries/Entry API and Makie's limits! API.
"""
function zoom_scatter_axes!(grid; mass=0.975)
    receipts = NamedTuple[]
    for index in CartesianIndices(grid)
        cell = grid[index]
        entries = cell.entries
        length(entries) == 1 || error("Expected exactly one scatter layer per facet")
        entry = only(entries)
        entry.plottype <: Scatter || error("Display zoom is only for scatter plots")
        x, y = entry.positional
        length(x) == length(y) || error("Scatter coordinates are not paired")
        xlim = scatter_display_limits(x; mass)
        ylim = scatter_display_limits(y; mass)
        limits!(cell.axis, xlim, ylim)
        inrange(v, range) = isnothing(range) || first(range) <= v <= last(range)
        visible = count(i -> inrange(x[i], xlim) && inrange(y[i], ylim), eachindex(x))
        push!(receipts, (; row=index[1], column=index[2], points=length(x), visible,
            marginal_mass=mass, xlow=isnothing(xlim) ? missing : first(xlim),
            xhigh=isnothing(xlim) ? missing : last(xlim),
            ylow=isnothing(ylim) ? missing : first(ylim),
            yhigh=isnothing(ylim) ? missing : last(ylim)))
    end
    receipts
end
