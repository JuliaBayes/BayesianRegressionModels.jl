include(joinpath(@__DIR__, "model.jl"))
using .RBesTCentering, BridgeStan
const RC = RBesTCentering
out = ARGS[1]
for name in ("AS", "crohn")
    c = RC.load_case(name)
    t = RC.total_model(c, joinpath(out, name))
    println("TOTAL ", name, " names=", t.names)
    for centered in (false, true)
        o = RC.ordinary_model(c, joinpath(out, name); centered)
        println("ORDINARY centered=", centered, " ", name, " names=", o.names, " lp_cols=", o.lp_cols, " beta=", o.beta_col, " tau=", o.tau_col)
        q = RC.ordinary_initial(c, o)
        lp, g = BridgeStan.log_density_gradient(o.model, q; propto=false, jacobian=true)
        println("  lp=", lp, " ref=", RC.ordinary_reference(c, o, q), " diff=", lp - RC.ordinary_reference(c, o, q))
    end
    q = RC.total_initial(c, t)
    lp, g = BridgeStan.log_density_gradient(t.model, q; propto=false, jacobian=true)
    println("  total lp=", lp, " ref=", RC.total_reference(c, t, q), " diff=", lp - RC.total_reference(c, t, q))
end
println("SMOKE_OK")
