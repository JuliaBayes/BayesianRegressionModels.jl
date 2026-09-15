include("prepare_pairs.jl")
include("validate_saved.jl")
root,case,out=ARGS
@assert case in ("cluster-independent","cluster-intercept")
hierarchy=case=="cluster-independent" ? "independent" : "intercept_only"
audit=joinpath(root,case=="cluster-independent" ? "air-totals-audit-independent-v2" : "air-totals-audit-intercept-v3")
totals=joinpath(root,"air-totals-"*case*"-v1")
brms=joinpath(root,"air-brms-whmc-"*case*"-v1")
native=joinpath(root,"air-native-"*case*"-v1")
summary=joinpath(root,"air-summary-"*case*"-v1")
for label in ("ordinary_ncp","s2z_cp","s2z_ncp","s2z_auto")
    @assert isfile(joinpath(native,label,"provenance.json"))
end
main("analyze","cluster_region",hierarchy,summary,audit,native,
    "ordinary_ncp","s2z_cp","s2z_ncp","s2z_auto")
# Completed fit files are reused. Only the remaining auto arm is newly sampled.
main("fit","cluster_region",hierarchy,brms,audit,native,"s2z_cp","s2z_ncp","s2z_auto")
mkpath(out);d=AIRTotals.load_data("cluster_region",hierarchy)
m=AIRTotals.model(d,joinpath(totals,"model"))
total_pairs(d,m,totals,out);s2z_pairs(d,m,brms,native,out)
validate_case(d,m,totals,brms,summary,out)
println("AIR_COMPLETE ",case)
