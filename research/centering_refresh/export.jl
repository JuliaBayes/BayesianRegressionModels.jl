# Export geometry and predictive quantities from completed refreshed fits.
const CASE, ROOT = ARGS
const CASE_DIR = CASE == "hsgp" ? "adaptive_centering" : CASE == "eight" ? "eight_schools_centering" : "radon_centering"
include(joinpath(@__DIR__, "..", CASE_DIR, "reproduce.jl"))
using Test
const ARMS = ("ncp", "cp", "posthoc_position", "posthoc_gradient", "online_position", "online_gradient")
const LABELS = Dict("ncp"=>"NCP", "cp"=>"CP", "posthoc_position"=>"Post-hoc position",
    "posthoc_gradient"=>"Post-hoc gradient", "online_position"=>"Online position", "online_gradient"=>"Online gradient")

function export_refresh()
    out = joinpath(ROOT, "export"); ispath(out) && error("Preserve previous export")
    mkpath(out); BLAS.set_num_threads(1)
    stan = CASE == "hsgp" ? stan_density(build_brmi(prepared_data()), "export", out) : stan_density("export", out)
    names = String.(BS.param_unc_names(stan.density.model))
    fits = Dict(arm=>deserialize(joinpath(ROOT,arm,"fit.jls")) for arm in ARMS if isfile(joinpath(ROOT,arm,"summary.tsv")))
    @test all(haskey(fits, arm) for arm in ("ncp", "posthoc_position", "posthoc_gradient", "online_position"))
    pilot = fits["ncp"]
    controls = Dict(arm=>Dict(i=>c.c for (i,c) in fit.controls) for (arm,fit) in fits)
    selected = controls["posthoc_position"]
    entries = NamedTuple[]
    if CASE == "hsgp"
        for b in BRM._adaptive_hsgp_centering_blocks(stan.sb,names), j in 1:DEFAULT_K
            push!(entries, (; group=b.logical == :mu ? "Mean GP" : "Log-SD GP",
                cell="Basis $(lpad(j,2,'0'))", id=j, index=b.effects[j], scale_index=b.sd,
                length_index=only(b.length_scales), selected=j in (1,2,19,20)))
        end
    else
        blocks = adaptive_centering_blocks(stan.sb,names)
        for (k,b) in enumerate(blocks)
            ids = collect(1:size(b.effects,2))
            chosen = if CASE == "eight"
                ids
            else
                lo = first(sort(ids;by=j -> (selected[b.effects[1,j]],j)))
                hi = first(sort(filter(!=(lo),ids);by=j -> (-selected[b.effects[1,j]],j)))
                mid = first(sort(filter(j->j ∉ (lo,hi),ids);by=j -> (abs(selected[b.effects[1,j]]-.5),j)))
                [lo,mid,hi]
            end
            for j in ids
                push!(entries, (;group=CASE == "eight" ? "School effects" : k==1 ? "County intercepts" : "County slopes",
                    cell=CASE == "eight" ? "School $j" : "$(findfirst(==(j),chosen)) County $(lpad(j,3,'0'))",
                    id=j,index=b.effects[1,j],scale_index=only(b.log_scales),length_index=0,selected=j in chosen))
            end
        end
    end
    write_tsv(joinpath(out,"coordinates.tsv"),[(;e...,c=selected[e.index]) for e in entries])
    rows = NamedTuple[]; gradient_rows = NamedTuple[]; checks = NamedTuple[]
    centering = NamedTuple[]
    for arm in filter(a->haskey(fits,a),ARMS)
        fit = fits[arm]; positions = fit.posterior_position
        checkpoint = deserialize(joinpath(ROOT,arm,"checkpoints","cp_latest.jls"))
        adaptive = adaptive_centering_problem(stan.sb,stan.density,ENZYME_BACKEND)
        WarmupHMC.restore_reparam_sources!(adaptive,fit.controls)
        for draw in (1,5000,10000)
            source = checkpoint.posterior_position[:,draw]
            ljac, model = WarmupHMC.reparametrizer(adaptive)(source)
            model_error = maximum(abs.(model .- positions[:,draw]) ./ (1 .+ abs.(positions[:,draw])))
            value, gradient = LogDensityProblems.logdensity_and_gradient(adaptive,source)
            gradient_error = maximum(abs.(gradient .- checkpoint.posterior_gradient[:,draw]) ./ (1 .+ abs.(gradient)))
            @test model_error < 1e-9
            @test gradient_error < 1e-7
            @test value ≈ LogDensityProblems.logdensity(stan.density,model) + ljac
            push!(checks,(;arm,draw,model_error,gradient_error))
        end
        arm == "cp" && continue # The CP visual reference uses the same pilot draws.
        logs = Dict{Int,Vector{Float64}}()
        if CASE == "hsgp"
            gps=gp_draws(stan,fit,prepared_data())
            for e in entries
                gp=only(filter(g->g.name==(e.group=="Mean GP" ? "mu" : "log_sigma"),gps))
                logs[e.index]=gp.logs[:,e.id]
            end
        else
            for e in entries
                logs[e.index]=positions[e.scale_index,:]
            end
        end
        for e in entries
            arm != "ncp" && push!(centering,(;arm=LABELS[arm],group=e.group,id=e.id,centeredness=controls[arm][e.index]))
            e.selected || continue
            displays = arm == "ncp" ? [("CP reference",1.0),("NCP reference",0.0)] : [(LABELS[arm],controls[arm][e.index])]
            for (display,c) in displays
                scale=exp.(c .* logs[e.index]); u=positions[e.index,:] .* scale
                if arm != "ncp"
                    @test u ≈ checkpoint.posterior_position[e.index,:] rtol=1e-8 atol=1e-10
                end
                all(isfinite,u) || error("Invalid displayed coordinates: $arm $(e.cell)")
                hypers = e.length_index == 0 ? [("Group SD",e.scale_index)] : [("Marginal SD",e.scale_index),("Length scale",e.length_index)]
                for (hyper,idx) in hypers, draw in axes(positions,2)
                    push!(rows,(;configuration=display,group=e.group,cell=e.cell,hyper,draw,
                        hyperparameter=exp(positions[idx,draw]),coordinate=u[draw]))
                end
                for draw in round.(Int,range(1,size(positions,2);length=1000))
                    g = checkpoint.posterior_gradient[e.index,draw]
                    arm == "ncp" && (g /= scale[draw])
                    isfinite(g) || error("Nonfinite displayed gradient")
                    push!(gradient_rows,(;configuration=display,group=e.group,cell=e.cell,draw,coordinate=u[draw],gradient=g))
                end
            end
        end
    end
    write_tsv(joinpath(out,"pairs.tsv"),rows)
    write_tsv(joinpath(out,"gradients.tsv"),gradient_rows)
    write_tsv(joinpath(out,"controls.tsv"),centering)
    write_tsv(joinpath(out,"frame_checks.tsv"),checks)
    if CASE == "eight"
        for (arm,fit) in fits
            export_coordinates(arm,stan,fit,out)
        end
        y=brm_predictive_draws(brm_descriptor(stan.sb),permutedims(pilot.posterior_position);problem=stan.density,seed=1).y
        d=read_eight_schools()
        write_tsv(joinpath(out,"ppc.tsv"),[let q=quantile(y[:,j],[.05,.25,.5,.75,.95])
            (;index=j,group="All schools",category="Reported estimate",observation=d.y[j],q05=q[1],q25=q[2],q50=q[3],q75=q[4],q95=q[5])
            end for j in 1:8])
    elseif CASE == "radon"
        d=RADON_DATA
        y=brm_predictive_draws(brm_descriptor(stan.sb),permutedims(pilot.posterior_position);problem=stan.density,seed=1).log_radon
        write_tsv(joinpath(out,"ppc.tsv"),[let q=quantile(y[:,j],[.05,.25,.5,.75,.95])
            (;index=j,group="County $(d.county_idx[j])",category="Floor code $(Int(d.floor_measure[j]))",observation=d.log_radon[j],q05=q[1],q25=q[2],q50=q[3],q75=q[4],q95=q[5])
            end for j in 1:d.N])
    else
        d=prepared_data()
        write_tsv(joinpath(out,"observations.tsv"),[(;time=d.times[i],acceleration_scaled=d.y[i]) for i in eachindex(d.times)])
        # Retain the source-style posterior function intervals for each method.
        curves=NamedTuple[]
        basis=[sin(pi/(2L)*(x+L)*j)/sqrt(L) for x in d.x,j in 1:DEFAULT_K]
        for (arm,fit) in fits, gp in gp_draws(stan,fit,d)
            values=basis*transpose(gp.weights)
            gp.name=="log_sigma" && (values=exp.(values))
            subset=(;posterior_position=fit.posterior_position[:,[1,5000,10000]])
            native_names,native_draws=constrained_draws(stan,subset;include_tp=true)
            native=permutedims(brm_output_draws(brm_descriptor(stan.sb),permutedims(native_draws),native_names;
                logical=gp.name=="mu" ? :mu : :sigma))
            @test values[:,[1,5000,10000]] ≈ native
            for i in eachindex(d.times)
                q=quantile(values[i,:],[.05,.1,.25,.5,.75,.9,.95])
                push!(curves,(;arm=LABELS[arm],predictor=gp.name,time=d.times[i],q05=q[1],q10=q[2],q25=q[3],q50=q[4],q75=q[5],q90=q[6],q95=q[7]))
            end
        end
        write_tsv(joinpath(out,"curves.tsv"),curves)
    end
    println("REFRESH_EXPORT_COMPLETE case=",CASE," pairs=",length(rows)," gradients=",length(gradient_rows)," checks=",length(checks))
end

export_refresh()
