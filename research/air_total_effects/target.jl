# Stan rejects numerically invalid proposals. AIR's total prior has a positive
# definite conditional precision for every finite positive scale, but extreme
# Pathfinder trial steps can lose that property in floating point arithmetic.
# Preserve the first failed point and reject only this specific numerical case.
struct AirStanProblem{P}
    base::P
    first_spd_point::Base.RefValue{Union{Nothing,Vector{Float64}}}
end
AirStanProblem(model)=AirStanProblem(BrmsPupilProblem(model,Ref(0);reject_numerical_errors=true),
    Ref{Union{Nothing,Vector{Float64}}}(nothing))
function Base.getproperty(p::AirStanProblem,s::Symbol)
    s in (:base,:first_spd_point) ? getfield(p,s) : getproperty(getfield(p,:base),s)
end
LogDensityProblems.dimension(p::AirStanProblem)=LogDensityProblems.dimension(p.base)
LogDensityProblems.capabilities(::Type{<:AirStanProblem})=LogDensityProblems.LogDensityOrder{1}()
function reject_spd(p,e,q)
    message=sprint(showerror,e)
    numerical=e isa ErrorException && occursin("failed with exception: Exception:",message) &&
        occursin("mdivide_left_spd: Matrix A is not positive definite",message)
    numerical || return false
    if isnothing(p.first_spd_point[])
        p.first_spd_point[]=collect(q)
        println(stderr,"AIR_SPD_NUMERICAL_REJECTION ",message)
    end
    p.numerical_rejections[]+=1
    true
end
function LogDensityProblems.logdensity(p::AirStanProblem,q)
    try
        LogDensityProblems.logdensity(p.base,q)
    catch e
        reject_spd(p,e,q) || rethrow()
        -Inf
    end
end
function LogDensityProblems.logdensity_and_gradient(p::AirStanProblem,q)
    try
        LogDensityProblems.logdensity_and_gradient(p.base,q)
    catch e
        reject_spd(p,e,q) || rethrow()
        (-Inf,fill(NaN,length(q)))
    end
end
