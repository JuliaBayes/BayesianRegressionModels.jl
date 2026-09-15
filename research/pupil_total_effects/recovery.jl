include("run.jl")

"""Population coefficients conditional on totals/scales (and Student precision).

The first coefficient is the population intercept at mean load, matching
brms's temporary Intercept. All groups enter through their means symmetrically.
"""
function population_conditional(q,data,prior)
    J = length(data.ids)
    va,vb = exp.(2q[1:2])
    s2 = PTE.mean_variance(prior,q)
    ma,mb = mean(q[5:24]),mean(q[25:44])
    observed = ma+data.xbar*mb
    c00 = (va+data.xbar^2*vb)/J
    c01 = data.xbar*vb/J
    c11 = vb/J
    denominator = s2+c00
    mu = [PTE.INTERCEPT_MEAN+s2/denominator*(observed-PTE.INTERCEPT_MEAN),
          mb+c01/denominator*(PTE.INTERCEPT_MEAN-observed)]
    covariance = [s2*c00/denominator s2*c01/denominator;
                  s2*c01/denominator c11-c01^2/denominator]
    mu,Symmetric(covariance)
end

function original_names(data)
    vcat(["population_intercept_at_mean_load","population_load"],
         PTE.coordinate_names(data)[1:4],
         ["deviation_intercept[$id]" for id in data.ids],
         ["deviation_load[$id]" for id in data.ids],
         ["population_intercept_at_load0","log_sigma_intercept_at_subject0"])
end

function original_position(q,beta,data)
    alpha = beta[1]-data.xbar*beta[2]
    vcat(beta,q[1:4],q[5:24].-alpha,q[25:44].-beta[2],
         alpha,q[3]-data.idbar*q[4])
end

function original_ncp(q)
    result = copy(q)
    result[7:26,:] ./= exp.(q[3:3,:])
    result[27:46,:] ./= exp.(q[4:4,:])
    result
end

function conditional_original(q,data,prior)
    beta,covariance = population_conditional(q,data,prior)
    position = original_position(q,beta,data)
    va = covariance[1,1]-2data.xbar*covariance[1,2]+data.xbar^2*covariance[2,2]
    variances = vcat(diag(covariance),zeros(4),fill(va,20),
                    fill(covariance[2,2],20),va,0.0)
    position,variances,beta,cholesky(covariance).L
end

function audit_recovery(fit,data)
    prior = get(fit,:mean_prior,PTE.GaussianMean())
    rows = NamedTuple[]
    rng = Xoshiro(1701)
    @testset "Independent conditional recovery algebra" begin
        for s in round.(Int,range(1,size(fit.positions,2),length=12))
            q = fit.positions[:,s]
            mu,covariance = population_conditional(q,data,prior)
            D = Diagonal(exp.(2q[1:2]))
            T = [1.0 -data.xbar;0.0 1.0]
            s2 = PTE.mean_variance(prior,q)
            precision = Diagonal([inv(s2),0.0])+20T'*(D\T)
            natural = [PTE.INTERCEPT_MEAN/s2,0.0]+T'*(D\[sum(q[5:24]),sum(q[25:44])])
            expected_cov = inv(Symmetric(precision))
            expected_mu = precision\natural
            mean_error = maximum(abs.(mu-expected_mu))
            covariance_error = maximum(abs.(covariance-expected_cov))
            @test mean_error < 1e-7
            @test covariance_error < 1e-7
            beta = mu+cholesky(covariance).L*randn(rng,2)
            recovered = original_position(q,beta,data)
            @test recovered[1]-data.xbar*recovered[2].+recovered[7:26] ≈ q[5:24]
            @test recovered[2].+recovered[27:46] ≈ q[25:44]
            # Conditional normalization: ratio of joint priors at two beta
            # values must equal the ratio of the proposed Normal conditional.
            prior_joint(b) = logpdf(Normal(PTE.INTERCEPT_MEAN,sqrt(s2)),b[1]) +
                sum(logpdf.(Normal(b[1]-data.xbar*b[2],exp(q[1])),q[5:24])) +
                sum(logpdf.(Normal(b[2],exp(q[2])),q[25:44]))
            density_error = abs((prior_joint(beta)-prior_joint(mu))-
                (logpdf(MvNormal(mu,covariance),beta)-logpdf(MvNormal(mu,covariance),mu)))
            @test density_error < 1e-7
            push!(rows,(;draw=s,mean_error,covariance_error,density_error))
        end
    end
    rows
end

function diagnostics(q)
    samples = permutedims(reshape(q,size(q,1),size(q,2),1),(2,3,1))
    (;bulk=vec(MCMCDiagnosticTools.ess(samples;kind=:bulk)),
      mean_ess=vec(MCMCDiagnosticTools.ess(samples;kind=mean)),
      mcse=vec(MCMCDiagnosticTools.mcse(samples;kind=mean)))
end

function baseline_original(record,data)
    names = record.original_named_names
    idx(name)=only(findall(==(name),names))
    beta_indices = [idx("Intercept"),idx("b.1")]
    reduce(hcat,(original_position(record.positions[:,s],
        record.original_named_positions[beta_indices,s],data) for s in axes(record.positions,2)))
end

function summarize_scope(label,seed,q,record,data)
    names = original_names(data)
    rows = NamedTuple[]
    for (scope,values) in (("original_physical46",q[1:46,:]),
                          ("original_physical_plus_raw_intercepts48",q),
                          ("original_stan_ncp46",original_ncp(q)[1:46,:]))
        ds = diagnostics(values)
        j = argmin(ds.bulk)
        parameter = names[j]
        if scope=="original_stan_ncp46" && 7<=j<=46
            parameter = replace(parameter,"deviation_"=>"z_")
        end
        push!(rows,(;arm=label,recovery_seed=seed,scope,min_bulk_ess=ds.bulk[j],
            limiting_parameter=parameter,
            min_bulk_ess_per_1000_sampling_gradients=1000ds.bulk[j]/record.sampling_gradients,
            min_bulk_ess_per_1000_all_gradients=1000ds.bulk[j]/record.all_gradient_calls))
    end
    rows
end

function recover_fit(label,fit,data,output;seeds=101:120)
    prior = get(fit,:mean_prior,PTE.GaussianMean())
    write_tsv(joinpath(output,label*"_audit.tsv"),audit_recovery(fit,data))
    components = [conditional_original(q,data,prior) for q in eachcol(fit.positions)]
    means = reduce(hcat,(x[1] for x in components))
    conditional_variances = reduce(hcat,(x[2] for x in components))
    ds_mean = diagnostics(means)
    names = original_names(data)
    variance_of_means = vec(var(means;dims=2))
    mean_variance = vec(mean(conditional_variances;dims=2))
    N = size(means,2)
    details = NamedTuple[]
    summaries = NamedTuple[]
    for seed in seeds
        rng = Xoshiro(seed)
        recovered = reduce(hcat,(original_position(fit.positions[:,s],
            components[s][3]+components[s][4]*randn(rng,2),data) for s in 1:N))
        ds = diagnostics(recovered)
        append!(summaries,summarize_scope(label,seed,recovered,fit,data))
        for j in eachindex(names)
            total_variance = variance_of_means[j]+mean_variance[j]
            # Exact added term in Var(sample mean), conditional on the saved
            # marginal path. The MCSE for the conditional-mean path is estimated.
            predicted_mcse = sqrt(ds_mean.mcse[j]^2+mean_variance[j]/N)
            push!(details,(;arm=label,recovery_seed=seed,parameter=names[j],
                conditional_noise_variance_fraction=mean_variance[j]/total_variance,
                conditional_mean_bulk_ess=ds_mean.bulk[j],recovered_bulk_ess=ds.bulk[j],
                conditional_mean_mean_ess=ds_mean.mean_ess[j],recovered_mean_ess=ds.mean_ess[j],
                conditional_mean_mcse=ds_mean.mcse[j],recovered_mcse=ds.mcse[j],
                predicted_recovered_mcse=predicted_mcse,
                recovered_mean=mean(recovered[j,:]),conditional_mean=mean(means[j,:]),
                posterior_sd_from_total_variance=sqrt(total_variance)))
        end
        seed==first(seeds) && serialize(joinpath(output,label*"_recovered_seed$seed.jls"),
            (;positions=recovered,names,source_fit=label,recovery_seed=seed,source_draws=N,
              frame="original physical parameters plus two raw-design intercepts"))
    end
    write_tsv(joinpath(output,label*"_parameter_diagnostics.tsv"),details)
    write_tsv(joinpath(output,label*"_summaries.tsv"),summaries)
    println("RECOVERY_COMPLETE\t",label)
    flush(stdout)
    summaries
end

function main_recovery(integrated_dir,baseline_dir,output)
    ispath(output) && error("Preserve previous recovery output")
    mkpath(output)
    cp(@__FILE__,joinpath(output,"recovery.jl"))
    data = PTE.load_data()
    baseline = deserialize(joinpath(baseline_dir,"brms_ncp.jls"))
    baseline_positions = baseline_original(baseline,data)
    rows = summarize_scope("brms_ncp",0,baseline_positions,baseline,data)
    ds = diagnostics(baseline_positions)
    names = original_names(data)
    write_tsv(joinpath(output,"brms_original_parameters.tsv"),[
        (;parameter=names[j],bulk_ess=ds.bulk[j],mean_ess=ds.mean_ess[j],mcse=ds.mcse[j],
          posterior_mean=mean(baseline_positions[j,:]),posterior_sd=std(baseline_positions[j,:]))
        for j in eachindex(names)])
    for label in ("scaled_total","partial")
        fit = deserialize(joinpath(integrated_dir,label*".jls"))
        append!(rows,recover_fit(label,fit,data,output))
    end
    # For the Gaussian case, also refresh the SAME brms draws. This isolates
    # the effect of recovery without changing a single HMC transition.
    control = deserialize(joinpath(integrated_dir,"scaled_total.jls"))
    if get(control,:mean_prior,PTE.GaussianMean()) isa PTE.GaussianMean
        append!(rows,recover_fit("brms_gaussian_refreshed",baseline,data,output))
    end
    write_tsv(joinpath(output,"summaries.tsv"),rows)
    println("COMPLETE\t",output)
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==3 || error("Expected integrated, baseline and new recovery directories")
    main_recovery(ARGS...)
end
