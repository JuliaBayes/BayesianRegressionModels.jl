using Serialization, Test, MCMCDiagnosticTools, JSON, SHA
using WarmupHMC # Load checkpoint types; all validation below uses independent scalar algebra.

# Read-only acceptance over the saved fits and derived tables: no BRM/WHMC
# transform helpers, target recompilation, plotting, or sampling.
length(ARGS) == 4 || error("usage: audit_saved.jl OFFLINE ONLINE DIAGNOSTICS COSTS")
offline, online_dir, diagnostics_dir, costs_dir = abspath.(ARGS)
study = dirname(@__DIR__)
Sys.which("unzip") === nothing && error("The read-only archive audit requires unzip on PATH.")
function rows(path)
    lines = readlines(path)
    names = split(first(lines), '\t')
    [Dict(zip(names, split(line, '\t'))) for line in lines[2:end]]
end
pilot = deserialize(joinpath(offline, "noncentered.jls"))
partial = deserialize(joinpath(offline, "partial.jls"))
online = deserialize(joinpath(online_dir, "online.jls"))
mapped = deserialize(joinpath(diagnostics_dir, "partial_model_frame.jls"))
mapping = rows(joinpath(study, "results", "source_coordinate_map.tsv"))
at = Dict(r["original"] => i for (i,r) in enumerate(mapping))
ia = [at["alpha_raw.$j"] for j in 1:386]
ib = [at["beta_raw.$j"] for j in 1:386]
ca = at["sigma_alpha"]; cb = at["sigma_beta"]
ma = at["mu_alpha"]; mb = at["mu_beta"]; sy = at["sigma_y"]
selected = Dict((r["role"], parse(Int,r["county"])) => parse(Float64,r["centeredness"])
    for r in rows(joinpath(offline,"selected_centeredness.tsv")))
learned = Dict((r["role"], parse(Int,r["county"])) => parse(Float64,r["centeredness"])
    for r in rows(joinpath(online_dir,"online_centeredness.tsv")))
q0, qo, qz = (f.posterior_position for f in (pilot,online,mapped))
qu = deserialize(joinpath(offline, "checkpoints-partial", "cp_latest.jls")).posterior_position

@testset "stored frames and independent scalar map" begin
    @test partial.posterior_position == qz
    @test length(mapping) == 777
    @test length(at) == 777
    @test pilot.nonlinear_adapt && !partial.nonlinear_adapt && online.nonlinear_adapt
    for q in (q0,qu,qo,qz)
        @test size(q) == (777,10000)
        @test all(isfinite,q)
    end
    for (role,indices,scale) in (("intercept",ia,ca),("slope",ib,cb)), j in 1:386
        expected = qu[indices[j],:] .* exp.(-selected[(role,j)] .* qu[scale,:])
        @test maximum(abs.(qz[indices[j],:] .- expected)) < 1e-9
    end
    @test qz[[ma,mb,ca,cb,sy],:] == qu[[ma,mb,ca,cb,sy],:]
end

representatives = rows(joinpath(diagnostics_dir,"representative_coordinates.tsv"))
@testset "plotted coordinates selected by inferred centeredness" begin
    @test length(representatives) == 6
    for role in ("intercept", "slope")
        candidates = collect(1:386)
        lo = first(sort(candidates; by=j -> (selected[(role,j)],j)))
        hi = first(sort(candidates; by=j -> (-selected[(role,j)],j)))
        mid = first(sort(filter(j -> j ∉ (lo,hi),candidates); by=j -> (abs(selected[(role,j)]-0.5),j)))
        actual = filter(r -> r["role"] == role, representatives)
        @test parse.(Int,getindex.(actual,"county")) == [lo,mid,hi]
        @test parse.(Float64,getindex.(actual,"centeredness")) == [selected[(role,j)] for j in (lo,mid,hi)]
    end
end

pairs = rows(joinpath(diagnostics_dir,"coordinate_pairs.tsv"))
seen = Dict{Tuple{String,String,Int},Int}()
@testset "every pair row bound to its named saved fit" begin
    @test length(pairs) == 300000
    for r in pairs
        config,role = r["configuration"],r["role"]
        j = parse(Int,r["county"])
        key=(config,role,j); draw=get(seen,key,0)+1; seen[key]=draw
        index = role == "intercept" ? ia[j] : ib[j]
        scale = role == "intercept" ? ca : cb
        q = startswith(config,"4") ? qu : startswith(config,"5") ? qo : q0
        c = startswith(config,"2") ? 1.0 : startswith(config,"3") ? selected[(role,j)] : startswith(config,"5") ? learned[(role,j)] : 0.0
        expected = q[index,draw] * exp(c*q[scale,draw])
        @test parse(Float64,r["coordinate"]) ≈ expected atol=1e-9 rtol=1e-12
        @test parse(Float64,r["hyperparameter"]) ≈ exp(q[scale,draw]) atol=1e-12
    end
    @test length(seen)==30 && all(==(10000),values(seen))
end

# Independent likelihood/prior derivative from immutable data, without BRM,
# its block discovery, or WarmupHMC's coordinate/gradient transport helpers.
archive=joinpath(study,"reference","radon_all.json.zip")
@assert bytes2hex(sha256(read(archive))) == "3f30c7909d530be01e70ab9e98f9f5d5e83371bb15c6dd168696aefd805b5672"
d=JSON.parse(read(`unzip -p $archive`,String))
@testset "PPC preserves every original observation and category" begin
    ppc=rows(joinpath(diagnostics_dir,"ppc_curves.tsv"))
    @test length(ppc)==length(d["log_radon"])==12573
    for (i,r) in enumerate(ppc)
        @test parse(Int,r["index"])==i
        @test parse(Int,r["county"])==d["county_idx"][i]
        @test parse(Float64,r["floor"])==d["floor_measure"][i]
        @test parse(Float64,r["observation"])==d["log_radon"][i]
        qs=[parse(Float64,r[k]) for k in ("q05","q25","q50","q75","q95")]
        @test all(isfinite,qs) && issorted(qs)
    end
end
nj=zeros(386); sx=zeros(386); sx2=zeros(386); ys=zeros(386); xy=zeros(386)
for (j,x,y) in zip(d["county_idx"],d["floor_measure"],d["log_radon"])
    nj[j]+=1; sx[j]+=x; sx2[j]+=x*x; ys[j]+=y; xy[j]+=x*y
end
gradient_rows=rows(joinpath(diagnostics_dir,"coordinate_gradients.tsv"))
gradient_errors=Float64[]
@testset "saved coordinate-gradient rows versus hand Gaussian gradient" begin
    @test length(gradient_rows)==18000
    for r in gradient_rows
        config,role=r["configuration"],r["role"]
        j=parse(Int,r["county"]); draw=parse(Int,r["draw"])
        post=startswith(config,"2"); on=startswith(config,"3")
        q=post ? qu : on ? qo : q0
        sa,sb,sigma=exp(q[ca,draw]),exp(q[cb,draw]),exp(q[sy,draw])
        za=q[ia[j],draw] * (post ? exp(-selected[("intercept",j)]*q[ca,draw]) : 1.0)
        zb=q[ib[j],draw] * (post ? exp(-selected[("slope",j)]*q[cb,draw]) : 1.0)
        a=q[ma,draw]+sa*za; b=q[mb,draw]+sb*zb
        g=role=="intercept" ? -za+sa*(ys[j]-nj[j]*a-sx[j]*b)/sigma^2 : -zb+sb*(xy[j]-sx[j]*a-sx2[j]*b)/sigma^2
        c=post ? selected[(role,j)] : on ? learned[(role,j)] : 1.0
        scale=role=="intercept" ? ca : cb; z=role=="intercept" ? za : zb
        factor=exp(c*q[scale,draw]); expected_g=g/factor
        @test parse(Float64,r["coordinate"]) ≈ z*factor atol=1e-9 rtol=1e-12
        @test parse(Float64,r["gradient"]) ≈ expected_g atol=1e-8 rtol=1e-9
        push!(gradient_errors,abs(parse(Float64,r["gradient"])-expected_g))
    end
end

costs=rows(joinpath(costs_dir,"fit_costs.tsv"))
diagnostics=rows(joinpath(offline,"diagnostics.tsv"))
@testset "model-frame ESS and exact retained/run-total cost ratios" begin
    for (i,(fit,q)) in enumerate(((pilot,q0),(partial,qz),(online,qo)))
        samples=permutedims(reshape(q,777,10000,1),(2,3,1))
        rh=maximum(MCMCDiagnosticTools.rhat(samples))
        bulk=minimum(MCMCDiagnosticTools.ess(samples;kind=:bulk))
        tail=minimum(MCMCDiagnosticTools.ess(samples;kind=:tail))
        @test rh == parse(Float64,diagnostics[i]["max_split_rhat"])
        @test bulk == parse(Float64,costs[i]["min_bulk_ess"])
        @test tail == parse(Float64,costs[i]["min_tail_ess"])
        @test fit.total_gradient_evaluations == parse(Int,costs[i]["total_nuts_gradient_evaluations"])
        @test fit.sampling_gradient_evaluations == parse(Int,costs[i]["sampling_gradient_evaluations"])
        @test fit.fit_seconds == parse(Float64,costs[i]["fit_seconds"])
        @test bulk/fit.total_gradient_evaluations == parse(Float64,costs[i]["min_bulk_ess_per_total_gradient"])
        @test bulk/fit.sampling_gradient_evaluations == parse(Float64,costs[i]["min_bulk_ess_per_sampling_gradient"])
        println("saved_fit_diagnostics\t",costs[i]["fit"],"\t",rh,"\t",bulk,"\t",tail)
    end
end
println("radon_independent_saved_audit_complete\tpair_rows=",length(pairs),"\tgradient_rows=",length(gradient_rows),"\tmax_hand_gradient_error=",maximum(gradient_errors))
