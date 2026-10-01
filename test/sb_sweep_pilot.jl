# test/sb_sweep_pilot.jl — v1 SB-parity sweep PILOT (3 cases).
#
# Run: julia --project=test test/sb_sweep_pilot.jl
#
# Proves the harness end to end: @brm counterpart → SBBRMI → BridgeStan
# compile → unconstrained-posterior records at sweep-defined parity points.
# Cases: `blr`, `radon_pooled`, `earn_height` (plain Gaussian regressions).
# Priors transcribed from the RK SOURCE at pin `8a51702` (comments there pin
# Stan semantics, incl. "Stan drops the log2 constant" on half-normals).
#
# Records append to $SB_SWEEP_OUT (default: tempdir()/sb-sweep-records.jsonl).
# Verified records are promoted to test/sb_sweep_records.jsonl at closeout.

include(joinpath(@__DIR__, "sb_sweep_common.jl"))

using Distributions: Normal
using ReactiveKernelsPPLExamples:
    BLRExample, RadonPooledExample, EarnHeightExample

const OUT = get(ENV, "SB_SWEEP_OUT",
    joinpath(tempdir(), "sb-sweep-records.jsonl"))

# posteriordb blr: beta_j ~ Normal(0,10), sigma ~ Normal(0,10) plain
# (lower=0 carries the half), y ~ Normal(X*beta, sigma). No intercept.
function blr_sb()
    X = BLRExample.BLR_X
    df = (; y=BLRExample.BLR_Y, x1=X[:, 1], x2=X[:, 2], x3=X[:, 3])
    builder = @brm begin
        sigma ~ Normal(0, 10; lower=0.0)
        mu ~ 0 + x1 + x2 + x3
        effect(mu, x1) ~ Normal(0, 10)
        effect(mu, x2) ~ Normal(0, 10)
        effect(mu, x3) ~ Normal(0, 10)
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

# radon_mn-radon_pooled: alpha/beta ~ Normal(0,10), sigma_y ~ Normal(0,1)
# plain; mu = alpha + beta*floor; N=60 representative subset (model's own
# demo-bound data, hashed in the record).
function radon_pooled_sb()
    df = (;
        y=RadonPooledExample.RADON_POOLED_LOG,
        floor=RadonPooledExample.RADON_POOLED_FLOOR,
    )
    builder = @brm begin
        sigma ~ Normal(0, 1; lower=0.0)
        mu ~ 1 + floor
        effect(mu, Intercept) ~ Normal(0, 10)
        effect(mu, floor) ~ Normal(0, 10)
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

# earnings-earn_height: FLAT improper priors (no ~ statements in .stan);
# mu = beta1 + beta2*height; full posteriordb data.
function earn_height_sb()
    df = (;
        y=EarnHeightExample.EARN_HEIGHT_EARN,
        height=EarnHeightExample.EARN_HEIGHT_HEIGHT,
    )
    builder = @brm begin
        sigma ~ Flat(; lower=0.0)
        mu ~ 1 + height
        effect(mu, Intercept) ~ Flat()
        effect(mu, height) ~ Flat()
        y ~ Normal(mu, sigma)
    end
    return SBBRMI(builder(df); mod=@__MODULE__)
end

open(OUT, "a") do io
    for (case, build) in (
        ("blr", blr_sb),
        ("radon_pooled", radon_pooled_sb),
        ("earn_height", earn_height_sb),
    )
        recs = sweep_case(io, case, build)
        for r in recs
            println("SB_SWEEP case=$(r["case"]) label=$(r["label"]) lp=$(r["lp"]) names=$(r["stan_names"])")
        end
    end
end
println("SB_SWEEP wrote $OUT")
