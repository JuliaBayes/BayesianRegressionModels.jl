using Pkg; Pkg.activate(@__DIR__)
using Printf
include("benchmark.jl")
include("../examples/all.jl")

formula, data = get_examples(:bambi, :escs)

println("Formula : ", formula)
println("N       : ", nrow(data))
println("Dim     : ", dimension(make_problem(formula, data, :brms; source=:bambi, key=:escs)))
println()

function fmt_time(seconds::Float64)
    if seconds < 1e-6
        return @sprintf("%.1f ns", seconds * 1e9)
    elseif seconds < 1e-3
        return @sprintf("%.1f μs", seconds * 1e6)
    else
        return @sprintf("%.1f ms", seconds * 1e3)
    end
end

function print_result(label, b)
    r = minimum(b)
    @printf("  %-40s  %s  (%d allocs)\n", label, fmt_time(r.time), r.allocs)
end

# ── Gradient correctness ─────────────────────────────────────────────────────

function compute_gradient(problem::AnyJuliaProblem, ad, q)
    prep = DI.prepare_gradient(problem._logdensity, ad, q)
    grad = similar(q)
    DI.value_and_gradient!(problem._logdensity, grad, prep, ad, q)
    return grad
end

function try_benchmark(label, problem, ad, ref_grad, q; rtol=1e-6)
    # Check correctness against reference gradient
    local grad
    try
        grad = compute_gradient(problem, ad, q)
    catch e
        @printf("  %-40s  FAILED\n", label)
        return (label=label, error=sprint(showerror, e, catch_backtrace()))
    end

    maxdiff = maximum(abs.(grad .- ref_grad))
    scale   = max(maximum(abs.(ref_grad)), 1.0)
    reldiff = maxdiff / scale
    if reldiff > rtol
        msg = @sprintf("max|Δ|=%.2e, rel=%.2e", maxdiff, reldiff)
        @printf("  %-40s  WRONG (%s)\n", label, msg)
        return (label=label, error=msg)
    end

    # Benchmark
    try
        print_result(label, _benchmark(problem, :gradient; ad))
        return nothing
    catch e
        @printf("  %-40s  FAILED\n", label)
        return (label=label, error=sprint(showerror, e, catch_backtrace()))
    end
end

enzyme_rev = AutoEnzyme(; mode=Enzyme.Reverse, function_annotation=Enzyme.Duplicated)
enzyme_fwd = AutoEnzyme(; mode=Enzyme.Forward, function_annotation=Enzyme.Duplicated)
errors = []

# ── Reference gradient from brms (Stan) ──────────────────────────────────────
brms_problem = make_problem(formula, data, :brms; source=:bambi, key=:escs)
q_test = randn(dimension(brms_problem))
_, ref_grad = logdensity_and_gradient(brms_problem, q_test)

println("Reference: brms/Stan gradient at random q")
@printf("  max|grad| = %.4f\n\n", maximum(abs.(ref_grad)))

# ── brms (Stan) ──────────────────────────────────────────────────────────────
println("brms (Stan):")
print_result("  primal",   _benchmark(brms_problem, :primal))
print_result("  gradient", _benchmark(brms_problem, :gradient))
println()

# ── Julia hand-written variants ──────────────────────────────────────────────
ad_backends = [
    (enzyme_fwd,                "Enzyme fwd"),
    (enzyme_rev,                "Enzyme rev"),
    (DI.AutoMooncakeForward(),  "Mooncake fwd"),
    (AutoMooncake(),            "Mooncake rev"),
]

for (backend, desc) in [
    (:julia3, "julia3 (BLAS mul!)"),
    (:julia5, "julia5 (5-arg mul!)"),
    (:julia6, "julia6 (manual gemv)"),
]
    println("$desc:")
    problem = make_problem(formula, data, backend)
    print_result("  primal", _benchmark(problem, :primal))

    for (ad, ad_name) in ad_backends
        err = try_benchmark("  gradient ($ad_name)", problem, ad, ref_grad, q_test)
        err !== nothing && push!(errors, err)
    end
    println()
end

# ── julia7: fine-grained Enzyme annotations ──────────────────────────────────
println("julia7 (split Enzyme: Const closure + Duplicated buffer):")
jp7 = make_problem(formula, data, :julia7)
print_result("  primal", _benchmark(jp7, :primal))

# Enzyme split: Const(ℓ_inner) + Duplicated(mu) + Duplicated(q)
# Enzyme split reverse
let mode_name = "Enzyme split rev"
    dmu = zeros(length(jp7._mu)); grad = zeros(dimension(jp7))
    try
        Enzyme.autodiff(Enzyme.Reverse, Enzyme.Const(jp7._logdensity_split), Enzyme.Active,
            Enzyme.Duplicated(jp7._mu, dmu), Enzyme.Duplicated(copy(q_test), grad))
        maxdiff = maximum(abs.(grad .- ref_grad))
        reldiff = maxdiff / max(maximum(abs.(ref_grad)), 1.0)
        if reldiff > 1e-6
            msg = @sprintf("max|Δ|=%.2e, rel=%.2e", maxdiff, reldiff)
            @printf("  %-40s  WRONG (%s)\n", "gradient ($mode_name)", msg)
            push!(errors, (label="gradient ($mode_name)", error=msg))
        else
            print_result("  gradient ($mode_name)", _benchmark_enzyme_split(jp7, :gradient; enzyme_mode=Enzyme.Reverse))
        end
    catch e
        @printf("  %-40s  FAILED\n", "gradient ($mode_name)")
        push!(errors, (label="gradient ($mode_name)", error=sprint(showerror, e, catch_backtrace())))
    end
end

# Also test Mooncake on julia7 for comparison
for (ad, ad_name) in [(AutoMooncake(), "Mooncake rev")]
    err = try_benchmark("  gradient ($ad_name)", jp7, ad, ref_grad, q_test)
    err !== nothing && push!(errors, err)
end
println()

# ── Turing (DynamicPPL) ───────────────────────────────────────────────────────
println("turing (DynamicPPL):")
tp = make_problem(formula, data, :turing)
print_result("  primal (loop ~)", _benchmark(tp, :primal))
tp2 = make_problem(formula, data, :turing2)
print_result("  primal (@addlogprob!)", _benchmark(tp2, :primal))
println()

# ── Error report ─────────────────────────────────────────────────────────────
if !isempty(errors)
    println("=" ^ 72)
    println("ERRORS ($(length(errors))):")
    println("=" ^ 72)
    for e in errors
        println()
        println("── $(e.label) ──")
        println(first(e.error, 500))
    end
end
