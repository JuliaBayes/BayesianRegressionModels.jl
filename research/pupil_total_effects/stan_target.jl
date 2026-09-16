using BridgeStan

struct BrmsPupilProblem{M}
    model::M
    gradient_calls::Base.RefValue{Int}
    reject_numerical_errors::Bool
    numerical_rejections::Base.RefValue{Int}
    first_numerical_error::Base.RefValue{String}
end
BrmsPupilProblem(model,calls;reject_numerical_errors=false) =
    BrmsPupilProblem(model,calls,reject_numerical_errors,Ref(0),Ref(""))
LogDensityProblems.dimension(p::BrmsPupilProblem) = length(BridgeStan.param_unc_names(p.model))
LogDensityProblems.capabilities(::Type{<:BrmsPupilProblem}) = LogDensityProblems.LogDensityOrder{1}()
function reject_stan_numerical_error(p,e)
    message=sprint(showerror,e)
    # Stan's native sampler rejects out-of-domain numerical proposals. Expose
    # the same outcome to Julia's sampler/Pathfinder, while preserving the
    # first full cause and counting every rejected evaluation. Other errors
    # (including serialization, API and model-structure errors) propagate.
    recognized=e isa ErrorException && occursin("failed with exception: Exception:",message) &&
        occursin(r"(normal|student_t|inv_chi_square|gamma)_lpdf:|binomial_logit_lpmf:",message) &&
        occursin(r"must be (not nan|positive|finite|greater than 0)",message)
    p.reject_numerical_errors && recognized || return false
    p.numerical_rejections[]+=1
    if isempty(p.first_numerical_error[])
        p.first_numerical_error[]=message
        println(stderr,"STAN_NUMERICAL_REJECTION\t",message)
    end
    true
end
function LogDensityProblems.logdensity(p::BrmsPupilProblem,q)
    try
        BridgeStan.log_density(p.model,convert(Vector{Float64},q);propto=false,jacobian=true)
    catch e
        reject_stan_numerical_error(p,e) || rethrow()
        -Inf
    end
end
function LogDensityProblems.logdensity_and_gradient(p::BrmsPupilProblem,q)
    p.gradient_calls[] += 1
    try
        BridgeStan.log_density_gradient(p.model,convert(Vector{Float64},q);propto=false,jacobian=true)
    catch e
        reject_stan_numerical_error(p,e) || rethrow()
        (-Inf,fill(NaN,length(q)))
    end
end
