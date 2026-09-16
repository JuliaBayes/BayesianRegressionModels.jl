# Exact-target audit for one case: CASE OUT LEGACY_CAPTURE_DIR [S2Z_CAPTURE_DIR|none]
# 1. BridgeStan densities/gradients of the total, ordinary-NCP and ordinary-CP BRM targets against
#    hand-written references (constant offset = the dropped half-Normal log 2).
# 2. BRM adaptive wrappers (totals; ordinary block) at c in {0, 0.5, 1}: exact transport, density
#    with Jacobian, finite-difference gradients.
# 3. NCP/CP consistency: the centered emission equals the non-centered one up to the H log tau Jacobian.
# 4. RBesT's own gMAP program (captured by capture.R) evaluated at the mapped point: constant offset
#    across points, and identical trial-level theta.
include(joinpath(@__DIR__, "model.jl"))
using .RBesTCentering, Test, Random, JSON, BridgeStan, Serialization, LogDensityProblems, WarmupHMC, LinearAlgebra, Statistics
const RC = RBesTCentering
case, out, capture_dir = ARGS[1:3]
c = RC.load_case(case); mkpath(out)
t = RC.total_model(c, out); o = RC.ordinary_model(c, out); oc = RC.ordinary_model(c, out; centered=true)
starts = Dict(:total => RC.total_initial(c, t), :ordinary_ncp => RC.ordinary_initial(c, o), :ordinary_cp => RC.ordinary_initial(c, oc))
refs = Dict(:total => q -> RC.total_reference(c, t, q), :ordinary_ncp => q -> RC.ordinary_reference(c, o, q), :ordinary_cp => q -> RC.ordinary_reference(c, oc, q))
models = Dict(:total => t.model, :ordinary_ncp => o.model, :ordinary_cp => oc.model)
offsets = Dict{Symbol,Float64}()
lp(m, q) = BridgeStan.log_density(m, collect(q); propto=false, jacobian=true)
lpg(m, q) = BridgeStan.log_density_gradient(m, collect(q); propto=false, jacobian=true)
constants = Dict{String,Float64}()
@testset "RBesT $case exact targets" begin
    for kind in (:total, :ordinary_ncp, :ordinary_cp)
        m = models[kind]; start = starts[kind]; ref = refs[kind]
        offsets[kind] = ref(start) - lp(m, start)
        @test abs(offsets[kind] - log(2)) < 1e-9
        for trial in 0:3
            point = start .+ (trial == 0 ? zeros(length(start)) : 0.05randn(Xoshiro(trial), length(start)))
            l, g = lpg(m, point)
            @test isapprox(l + log(2), ref(point); atol=1e-7, rtol=1e-12)
            fd = RC.finite_gradient(ref, point)
            @test maximum(abs.(g - fd) ./ max.(1, abs.(g))) < 2e-5
        end
    end
    # NCP/CP consistency at the same physical point: log p_u = log p_z - H log tau.
    q = starts[:ordinary_ncp]; beta = q[o.population]; ltau = q[o.log_scale]; z = q[o.effects]
    qc = copy(starts[:ordinary_cp]); qc[oc.population] = beta; qc[oc.log_scale] = ltau; qc[oc.effects] = exp(ltau) .* z
    @test isapprox(lp(oc.model, qc), lp(o.model, q) - c.H * ltau; atol=1e-8, rtol=1e-12)
    pc = BridgeStan.param_constrain(oc.model, qc; include_tp=true); pn = BridgeStan.param_constrain(o.model, q; include_tp=true)
    @test pc[oc.lp_cols] ≈ pn[o.lp_cols] && pc[oc.beta_col] ≈ pn[o.beta_col] && pc[oc.tau_col] ≈ pn[o.tau_col]
    # Adaptive wrappers.
    for (label, m, start, ref) in (("total", t, starts[:total], refs[:total]), ("ordinary", o, starts[:ordinary_ncp], refs[:ordinary_ncp]))
        for cc in (0.0, 0.5, 1.0)
            raw = RC.BrmsPupilProblem(m.model, Ref(0))
            rp = RC.BRM.adaptive_centering_problem(m.sb, raw, RC.ENZYME_BACKEND; unc_names=m.names, centeredness=cc)
            ir = WarmupHMC.reparametrizer(rp); _, x = WarmupHMC._inverse_with_logabsdet_jacobian(ir, start)
            jac, y = ir(x); @test y ≈ start
            l, g = LogDensityProblems.logdensity_and_gradient(rp, x)
            @test isapprox(l + log(2), ref(y) + jac; atol=1e-7, rtol=1e-12)
            @test maximum(abs.(g - RC.finite_gradient(zz -> LogDensityProblems.logdensity(rp, zz), x)) ./ max.(1, abs.(g))) < 3e-5
        end
    end
    # RBesT's own program at the mapped point.
    phys = JSON.parsefile(joinpath(capture_dir, "init-physical.json"))
    gm, gs = Float64(phys["beta_raw_guess"][1][1]), Float64(phys["beta_raw_guess"][2][1])
    t1, t2 = Float64.(phys["tau_raw_guess"]); gi = Int.(phys["group_index"])
    for (param, m, start) in (("ncp", o, starts[:ordinary_ncp]), ("cp", oc, starts[:ordinary_cp]))
        native = BridgeStan.StanModel(joinpath(capture_dir, "clean.stan"), joinpath(capture_dir, "data-$param.json"); warn=false)
        nn = BridgeStan.param_unc_names(native)
        @test nn == vcat(["beta_raw.1", "tau_raw.1"], ["xi_eta.$g" for g in 1:c.H])
        tpn = BridgeStan.param_names(native; include_tp=true, include_gq=false)
        theta_cols = [only(findall(==("theta.$h"), tpn)) for h in 1:c.H]
        diffs = Float64[]
        for trial in 0:3
            q = start .+ (trial == 0 ? zeros(length(start)) : 0.05randn(Xoshiro(10 + trial), length(start)))
            beta = q[m.population]; ltau = q[m.log_scale]; u = q[m.effects]
            # RBesT legacy CP samples theta_j itself in the rescaled intercept frame; BRM CP samples b_j = theta_j - beta.
            xi = zeros(c.H); xi[gi] = param == "ncp" ? u : (beta .+ u .- gm) ./ gs
            raw = vcat((beta - gm) / gs, (ltau - t1) / t2, xi)
            pb = BridgeStan.param_constrain(m.model, q; include_tp=true); pr = BridgeStan.param_constrain(native, raw; include_tp=true)
            @test pr[theta_cols] ≈ pb[m.lp_cols]
            push!(diffs, lp(native, raw) - lp(m.model, q))
        end
        @test maximum(abs.(diffs .- diffs[1])) < 1e-8
        constants["rbest_$(param)_minus_brm"] = diffs[1]
    end
end
# Optional: pull request 64's sum-to-zero program against BRM's exact totals at mapped points.
s2z_dir = length(ARGS) >= 4 ? ARGS[4] : "none"
if s2z_dir != "none"
    zero_sum_basis(J) = begin
        Q = zeros(J, J - 1)
        for k in 1:J-1; s = 1 / sqrt(k * (k + 1.0)); Q[1:k, k] .= s; Q[k+1, k] = -k * s; end; Q
    end
    phys = JSON.parsefile(joinpath(s2z_dir, "init-physical.json"))
    gm, gs = Float64(phys["beta_raw_guess"][1][1]), Float64(phys["beta_raw_guess"][2][1])
    t1, t2 = Float64.(phys["tau_raw_guess"]); gi = Int.(phys["group_index"]); Q = zero_sum_basis(c.H)
    @testset "RBesT $case sum-to-zero program vs BRM exact totals" begin
        for param in ("ncp", "cp")
            native = BridgeStan.StanModel(joinpath(s2z_dir, "clean.stan"), joinpath(s2z_dir, "data-$param.json"); warn=false)
            nn = BridgeStan.param_unc_names(native)
            @test nn == vcat(["beta_raw.1", "tau_raw.1"], ["xi_eta.$g" for g in 1:c.H-1], ["xi_abar.1"])
            tpn = BridgeStan.param_names(native; include_tp=true, include_gq=false)
            theta_cols = [only(findall(==("theta.$h"), tpn)) for h in 1:c.H]
            diffs = Float64[]
            for trial in 0:3
                q = starts[:total] .+ (trial == 0 ? zeros(length(starts[:total])) : 0.05randn(Xoshiro(20 + trial), length(starts[:total])))
                T = q[vec(t.coords.totals)]; ltau = q[only(t.coords.scales)]; tau = exp(ltau)
                alpha = mean(T); eps = zeros(c.H); eps[gi] = T .- alpha
                xi = (Q' * eps) ./ (param == "ncp" ? tau : gs)
                raw = vcat((alpha - gm) / gs, (ltau - t1) / t2, xi, 0.0)
                pb = BridgeStan.param_constrain(t.model, q; include_tp=true); pr = BridgeStan.param_constrain(native, raw; include_tp=true)
                @test pr[theta_cols] ≈ T[gi] rtol=1e-10
                # T = alpha 1 + Q * (tau * xi) under NCP: the map from RBesT's sampled contrasts to the totals
                # carries the Jacobian (H - 1) log tau; under CP the contrast scale is the constant guess g_s.
                jacobian = param == "ncp" ? (c.H - 1) * ltau : 0.0
                push!(diffs, lp(native, raw) - lp(t.model, q) - jacobian)
            end
            @test maximum(abs.(diffs .- diffs[1])) < 1e-8
            constants["rbest_s2z_$(param)_minus_brm_total_minus_jacobian"] = diffs[1]
        end
    end
end
open(io -> JSON.print(io, Dict("status" => "passed", "case" => case, "H" => c.H, "family" => String(c.family),
    "total_dimension" => length(t.names), "ordinary_dimension" => length(o.names),
    "constant_offset" => log(2), "rbest_constants" => constants,
    "total_names" => t.names, "ordinary_names" => o.names), 2), joinpath(out, "audit.json"), "w")
serialize(joinpath(out, "initial.jls"), (; total=starts[:total], ordinary_ncp=starts[:ordinary_ncp], ordinary_cp=starts[:ordinary_cp],
    total_names=t.names, ordinary_names=o.names))
println("RBEST_AUDIT_COMPLETE ", case)
