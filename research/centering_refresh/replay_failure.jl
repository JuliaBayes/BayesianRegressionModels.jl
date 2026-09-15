# Replay a failed arm with an external call counter and the last physical target
# evaluation preserved. This changes bookkeeping only, not the target or sampler.
include("run.jl")
using JSON
const CALLS=Ref(0)
const LAST_RAW=Ref{Any}(nothing)
function LogDensityProblems.logdensity_and_gradient(p::CountedDensity,x)
    p.gradient_calls[] += 1
    CALLS[] += 1
    result=LogDensityProblems.logdensity_and_gradient(p.problem,x)
    LAST_RAW[]=(;q=copy(x),value=first(result),gradient=copy(last(result)))
    result
end
function replay_failure()
    select_online_loss!(REQUEST=="online_position")
    try
        Base.invokelatest(main,REQUEST)
    catch err
        record=(;case=CASE,arm=REQUEST,total_gradients=CALLS[],error=sprint(showerror,err),
            last_raw_value=LAST_RAW[].value,last_raw_gradient_finite=all(isfinite,LAST_RAW[].gradient),
            last_raw_position_finite=all(isfinite,LAST_RAW[].q))
        out=joinpath(OUTPUT,REQUEST)
        serialize(joinpath(out,"last_raw.jls"),LAST_RAW[])
        println("FAILURE_RECORDED ",record)
        open(io->JSON.print(io,record),joinpath(out,"failure.json"),"w")
        rethrow()
    end
end
if abspath(PROGRAM_FILE)==@__FILE__
    replay_failure()
end
