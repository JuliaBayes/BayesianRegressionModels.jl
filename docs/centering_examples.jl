module BRMCenteringExamples

"""Read the actual fitted BRM declarations and show their data construction."""
function authoring(which::Symbol)
    root=normpath(joinpath(@__DIR__,".."))
    if which in (:pupil_numeric,:pupil_hierarchical)
        folder=which==:pupil_numeric ? "pupil_builtin_totals" : "pupil_scale_totals"
        source=read(joinpath(root,"research",folder,"common.jl"),String)
        start=first(findfirst("const BUILDER",source))
        finish=first(findnext("\nfunction total_model",source,start))-1
        builder=replace(strip(source[start:finish]),r"^const BUILDER\s*="=>"builder = ")
        name=which==:pupil_numeric ? "pupil_numeric_brm_model" : "pupil_hierarchical_brm_model"
        data="""
        reference = JSON.parsefile(joinpath(pkgdir(BayesianRegressionModels),
            "research", "pupil_scale_totals", "reference", "ordinary_ncp.json"))
        data = (;p_size=Float64.(reference["Y"]), load=Float64.(reference["Z_1_2"]),
                 subj=Int.(reference["J_1"]))
        """
        which==:pupil_numeric && (data*="data = merge(data, (;subject_id=700 .+ data.subj))\n")
    elseif which in (:rbest_as,:rbest_crohn)
        source=read(joinpath(root,"research","rbest_centering","model.jl"),String)
        marker=which==:rbest_as ? "const AS_MODEL =" : "const CROHN_MODEL ="
        start=first(findfirst(marker,source))
        endpoint=which==:rbest_as ? "\n# gMAP with the documented crohn" : "\nfunction read_dataset"
        finish=first(findnext(endpoint,source,start))-1
        builder=replace(strip(source[start:finish]),marker=>"builder =";count=1)
        name=which==:rbest_as ? "rbest_as_brm_model" : "rbest_crohn_brm_model"
        dataset=which==:rbest_as ? "AS" : "crohn"
        columns=which==:rbest_as ? "n=Int.(column(\"n\")), r=Int.(column(\"r\"))" :
            "y=Float64.(column(\"y\")), y_se=88 ./ sqrt.(Float64.(column(\"n\")))"
        data="""
        table, header = readdlm(joinpath(pkgdir(BayesianRegressionModels),
            "research", "rbest_centering", "reference", "datasets", "$dataset.tsv"), '\\t'; header=true)
        column(name) = table[:, only(findall(==(name), vec(header)))]
        data = (;study=collect(1:size(table, 1)), $columns)
        """
    else
        which in (:air_intercept,:air_independent) || error("Unknown centering model")
        source=read(joinpath(root,"research","air_total_effects","model.jl"),String)
        marker=which==:air_intercept ? "const INTERCEPT =" : "const INDEPENDENT ="
        start=first(findfirst(marker,source))
        endpoint=which==:air_intercept ? "\nconst INDEPENDENT" : "\nfunction load_data"
        finish=first(findnext(endpoint,source,start))-1
        builder=replace(strip(source[start:finish]),marker=>"builder =";count=1)
        hierarchy=which==:air_intercept ? "intercept_only" : "independent"
        name=which==:air_intercept ? "air_intercept_brm_model" : "air_independent_brm_model"
        data="""
        reference = JSON.parsefile(joinpath(pkgdir(BayesianRegressionModels),
            "research", "air_total_effects", "reference", "cluster_region",
            "$hierarchy", "ordinary_ncp.json"))
        data = (;log_pm25=Float64.(reference["Y"]),
                 log_sat=Float64.(getindex.(reference["X"], 2)),
                 region=Int.(reference["J_1"]))
        """
    end
    body=join(("    "*line for line in split(strip(data)*"\n\n"*builder*"\n\nbuilder(data)",'\n')),'\n')
    "using BayesianRegressionModels, Distributions, JSON, DelimitedFiles\n\nfunction $name()\n$body\nend"
end

end
