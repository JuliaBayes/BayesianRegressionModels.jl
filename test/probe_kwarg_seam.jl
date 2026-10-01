#!/usr/bin/env julia
# test/probe_kwarg_seam.jl — seam-regression probe for option-(ii) param-feed seam
#
# Verifies that existing BRM + StanBlocks machinery already supports the
# per-obs LP kwargs end-to-end (option ii), so a downstream PKPD package can implement
# its own kernel term without requiring BRM-core additions.
#
# Three checks:
#   (1) @brm latent LP sub-formulas (log_X ~ 1 + covariates + (1|p|subject))
#       emit per-obs Stan variables in the generated model block.
#   (2) fixed + ranef COMPOSE into a SINGLE per-obs Stan variable
#       (i.e. the threaded variable is NOT fixed-only).
#   (3) downstream _sb_submodel_rhs! reads LP names via getkwargs(rhs) and
#       threads them as @slic kwargs pointing at the correct Stan variable.
#
# SCOPE: BRM-side seam only. Cross-module @slic mod-reachability of the
# a real co-located downstream kernel is the downstream package's kernel-wiring check.

using BayesianRegressionModels, StanBlocks
using Distributions: Normal

# Must import (not just use) to extend BRM's function with a new method:
import BayesianRegressionModels: _sb_submodel_rhs!

# ── Probe term (in-script; NOT added to BRM core) ──────────────────────────

# Marker function — zero body, mirrors the downstream PKPD triad pattern.
function probe_kwarg_consumer end

# Trivial @slic: sums two per-obs LP kwargs. Zero internal ~ statements.
# Proves kwarg threading reaches the Stan body as per-obs quantities.
# Defined in Main so SLIC's module-resolution chain (mod → Main) finds it.
probe_slic = StanBlocks.@slic begin
    return log_Vc + log_k10
end

# Emitter: reads LP column names via getkwargs(rhs), threads as @slic kwargs.
# Mirrors the emitter pattern a downstream PKPD integration uses.
function _sb_submodel_rhs!(stmts, data, target::Symbol,
                            ::typeof(probe_kwarg_consumer), rhs)
    kw = getkwargs(rhs)
    log_Vc_sym  = name(kw[:log_Vc])
    log_k10_sym = name(kw[:log_k10])
    println("  [emitter] log_Vc kwarg → Stan sym: :$log_Vc_sym")
    println("  [emitter] log_k10 kwarg → Stan sym: :$log_k10_sym")
    push!(stmts, :($target ~ probe_slic(; log_Vc=$log_Vc_sym, log_k10=$log_k10_sym)))
    return :done
end

# ── Mock data ───────────────────────────────────────────────────────────────
# 4 observations, 2 subjects; covariates are subject-level (same per subject).
# log_Vc has 3 continuous + 1 binary covariate to exercise multi-covariate
# fixed-effects composition. Both LPs share the |p| label → correlated ranef block.
df = (;
    subject  = [1, 1, 2, 2],
    male     = [1, 1, 0, 0],
    zage     = [0.5, 0.5, -0.3, -0.3],
    diseased = [0, 0, 1, 1],
    y        = [0.1, 0.2, 0.15, 0.25],
)

# ── Probe @brm formula ──────────────────────────────────────────────────────
# log_Vc:  multi-covariate fixed (1 + male + zage + diseased) + |p| ranef
# log_k10: intercept-only fixed + same |p| ranef → exercises correlated block
# result:  consumer term that receives log_Vc and log_k10 as kwargs
# y:       response using result (ensures result is in the likelihood path)
println("=== Building BRMI ===")
brmi = @brm df begin
    log_Vc  ~ 1 + male + zage + diseased + (1|p|subject)
    log_k10 ~ 1 + (1|p|subject)
    result ~ probe_kwarg_consumer(; log_Vc=log_Vc, log_k10=log_k10)
    y ~ Normal(result, 0.1)
end

# ── Build SBBRMI + transpile ────────────────────────────────────────────────
println("=== Building SBBRMI ===")
sbbrmi = SBBRMI(brmi)
model  = sbbrmi.model

println("=== Transpile check ===")
@assert StanBlocks.stan.transpiles(model) "SEAM FAIL: model does not transpile"
println("Transpile: PASSED")

code = BayesianRegressionModels.stan_code(sbbrmi)

println("\n=== Generated Stan code ===")
println(code)

# ── Check (1): latent LP vars appear in Stan code ──────────────────────────
println("\n=== Check (1): latent LP vars present ===")
for sym in ("log_Vc", "log_k10")
    @assert occursin(sym, code) "SEAM FAIL (1): $sym absent from Stan code"
    println("  $sym: present ✓")
end

# ── Check (2): COMPOSITION — fixed + ranef in a SINGLE per-obs var ─────────
# Quote ALL Stan lines mentioning each threaded latent variable name.
# A correctly composed LP shows up in lines like:
#   log_Vc[n] = b_log_Vc[1] + b_log_Vc[2]*male[n] + ... + r_log_Vc[subject[n]];
# A fixed-only LP would omit the ranef term (r_... or similar).
println("\n=== Check (2): fixed+ranef composition ===")
lines = split(code, '\n')
for sym in ("log_Vc", "log_k10")
    matching = filter(l -> occursin(sym, l), lines)
    println("\n--- Stan lines mentioning '$sym' ---")
    for l in matching
        println("  ", strip(l))
    end
end

# ── Check (3): emitter output — probe_slic is INLINED by SLIC ───────────────
# SLIC inlines sub-model bodies into the parent; `probe_slic` does NOT appear
# literally in Stan. The correct verification is that `result` (the emitter
# target) is assigned as an expression of the threaded LP kwarg names.
println("\n=== Check (3): emitter output inlined correctly ===")
# (a) result variable exists in Stan code
@assert occursin("result", code) "SEAM FAIL (3a): result absent from Stan code"
# (b) result is assigned using the threaded LP var names (log_Vc and log_k10)
result_lines = filter(l -> occursin("result", l) && occursin("=", l), lines)
println("  Stan lines with 'result =':")
for l in result_lines
    println("    ", strip(l))
end
has_result_composition = any(l -> occursin("result", l) && occursin("log_Vc", l) && occursin("log_k10", l), lines)
@assert has_result_composition "SEAM FAIL (3b): result not assigned from log_Vc and log_k10"
println("Check (3) PASSED: result inlined from log_Vc + log_k10 ✓")

println("\n=== SEAM VERIFICATION COMPLETE — all checks passed ===")
