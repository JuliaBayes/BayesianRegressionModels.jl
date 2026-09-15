include("recovery.jl")

function run_fixed_cp(output)
    ispath(output) && error("Preserve previous output; choose a new directory")
    mkpath(joinpath(output,"source"))
    BLAS.set_num_threads(1)
    data=PTE.load_data()
    prior=PTE.StudentMixtureMean()
    write_tsv(joinpath(output,"density_audit.tsv"),audit_model(PTE.PupilProblem(data,prior)))
    audit_reparametrization(data;mean_prior=prior)
    for name in ("model.jl","audit.jl","run.jl","recovery.jl","fixed_cp.jl")
        cp(joinpath(@__DIR__,name),joinpath(output,"source",name))
    end
    cp(joinpath(@__DIR__,"reference"),joinpath(output,"source","reference"))
    record,summary=run_arm("centered_total",ones(40),data,output;mean_prior=prior)
    write_tsv(joinpath(output,"diagnostics.tsv"),[summary])
    recover_fit("centered_total",record,data,output)
    println("FIXED_CP_COMPLETE\t",output)
end

if abspath(PROGRAM_FILE)==@__FILE__
    run_fixed_cp(only(ARGS))
end
