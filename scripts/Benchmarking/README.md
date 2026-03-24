# Bayesian Regression Model Benchmarks

Benchmarking log-density and gradient evaluation for a Bayesian linear regression across multiple backends and AD systems.

**Problem**: `drugs ~ o + c + e + a + n` (ESCS dataset, N=604, K=5, 7 unconstrained parameters).

## Running

```bash
cd scripts/Benchmarking
julia run_benchmark.jl
```

Gradient correctness is verified against brms/Stan as the reference.

## Results

### Primal (log-density evaluation)

| Variant | Time | Allocs | vs Stan |
|---|---|---|---|
| julia3 (BLAS mul!) | 823 ns | 0 | 3.0x faster |
| julia5 (5-arg mul!) | 930 ns | 0 | 2.7x faster |
| julia6 (manual gemv) | 943 ns | 0 | 2.6x faster |
| julia7 (split, BLAS mul!) | 1.0 μs | 0 | 2.5x faster |
| julia1 (allocating Xc*b) | 1.8 μs | 2 | 1.4x faster |
| **brms/Stan (glm fused)** | **2.5 μs** | **1** | **1.0x** |
| stan (matvec + vec normal) | 2.9 μs | 1 | 0.86x |
| stan (loop + vec normal) | 3.1 μs | 1 | 0.81x |
| turing2 (@addlogprob!) | 6.7 μs | 24 | 0.37x |
| julia2 (broadcasted row-dot) | 6.7 μs | 0 | 0.37x |
| turing (loop ~) | 13.5 μs | 20 | 0.19x |
| bambi/PyMC (pytensor) | 34 μs | 9 | 0.07x |

### Gradient (log-density + gradient, correct results only)

| Variant | Time | Allocs | vs Stan |
|---|---|---|---|
| **brms/Stan** | **4.2 μs** | **1** | **1.0x** |
| julia7 Enzyme split rev | 4.6 μs | 0 | 0.91x |
| julia3 Enzyme Dup rev | 4.8 μs | 0 | 0.88x |
| julia5 Enzyme Dup rev | 5.1 μs | 0 | 0.82x |
| julia7 Enzyme split fwd | 8.6 μs | 0 | 0.49x |
| julia6 Enzyme Dup rev | 13.4 μs | 0 | 0.31x |
| julia3 Mooncake rev | 15.7 μs | 1 | 0.27x |
| julia7 Mooncake rev | 16.2 μs | 1 | 0.26x |
| julia6 Mooncake rev | 91.4 μs | 0 | 0.05x |

## Implementations

### Hand-written Julia variants

All implement the same log-density as the brms-generated Stan model: centered predictors, student-t priors, half-student-t on sigma.

- **julia1**: `Xc * b` allocating, `dot(r, r)` for sum of squares
- **julia2**: Lazy `Base.broadcasted` with per-row dot products, custom `mysum` with `preprocess`/`instantiate`
- **julia3**: Pre-allocated buffer + `mul!` (BLAS gemv) + SIMD residual loop
- **julia4**: Broadcasted `.=` into pre-allocated buffer (same speed as lazy)
- **julia5**: 5-arg `mul!` (`r = Y - Xc*b` in one BLAS call) + SIMD loop
- **julia6**: Manual gemv (column-by-column axpy, no BLAS) + fused residual
- **julia7**: Same as julia3 but with split `ℓ_inner(mu, q)` for fine-grained Enzyme annotations

### Stan variants

- **brms (glm fused)**: Uses `normal_id_glm_lpdf` — hand-optimized fused likelihood with custom adjoint
- **stan (matvec + vec)**: Separate `Xc * b` + vectorized `normal_lpdf`
- **stan (loop + vec)**: Precomputed `mu` + vectorized `normal_lpdf`

### Turing/DynamicPPL variants

- **turing**: Standard `@model` with per-observation `Y[i] ~ Normal(mu[i], sigma)` loop
- **turing2**: Same priors but likelihood via `Turing.@addlogprob!`, bypassing tilde processing

## AD Backend Notes

### Enzyme

- **`function_annotation=Enzyme.Const`**: Produces **wrong gradients** on julia3/5/6/7. `Const` prevents shadow memory allocation for captured mutable buffers (`mu`, `r_buf`), so adjoints can't propagate through `mul!`.
- **`function_annotation=Enzyme.Duplicated`**: Correct gradients but creates shadows for ALL captured variables including constant data arrays (Y: 4.8KB, Xc: 24KB), adding ~1 μs overhead from zeroing ~30KB per call.
- **Fine-grained split (julia7)**: `ℓ_inner(mu, q)` takes the buffer as an explicit argument. Called with `Const(ℓ_inner)` + `Duplicated(mu, dmu)` + `Duplicated(q, grad)`. Only ~5KB shadow for the buffer. **Matches Stan performance.**

### Mooncake

- **Reverse mode**: Correct gradients, ~15–16 μs on julia3/5/7. Automatically determines constness (uses `NoRData` for Y/Xc) but has higher per-call overhead than Enzyme's LLVM-level codegen.
- **Forward mode**: Correct but extremely slow (milliseconds, 60k–500k allocs).
- The split approach doesn't help Mooncake — it traces through the full closure graph regardless.

### ForwardDiff

Fails on julia3/5/6/7 because `mul!` writes into pre-allocated `Float64` buffers that can't hold `Dual` numbers. Works on julia1 (allocating version).

## Key Insights

### Why hand-written Julia primals beat Stan

Stan evaluates with autodiff-ready `var` types even for primal-only calls (`propto=false`). Julia operates on plain `Float64` with zero-allocation BLAS + SIMD loops.

### Why Stan's gradient is hard to beat

Stan's `normal_id_glm_lpdf` has a hand-written adjoint that fuses primal and gradient into one pass, reusing intermediates. Gradient/primal ratio is 1.7x. Julia + Enzyme achieves 5.8x (whole-closure `Duplicated`) and comes close at 4.6x with fine-grained `Const`/`Duplicated` annotations.

### Why broadcasted row-dots are slow

`eachrow(Xc)` + per-row `dot`: 604 dot products of length 5, each too short for SIMD. BLAS `gemv` processes column-by-column — each axpy touches 604 contiguous elements, perfect for vectorization.

### DynamicPPL overhead breakdown

| Component | Time | Allocs |
|---|---|---|
| Empty model | 0 μs | 0 |
| Parameter management + transforms + priors | 1.0 μs | 18 |
| + manual likelihood (`@addlogprob!`) | 6.7 μs | 24 |
| + loop with 604 `~` tilde statements | 13.5 μs | 20 |

DynamicPPL v0.40 is already type-stable (`@code_warntype` shows concrete types). The overhead is inherent to its generality: parameter unpacking, bijector transforms, accumulator bookkeeping.

### DynamicPPL main branch

The main branch of DynamicPPL adds an `adtype` keyword to `LogDensityFunction` for integrated gradient computation. This passes model internals as `DI.Constant` contexts — conceptually similar to our manual Enzyme `Const`/`Duplicated` split. However, the gradient path doesn't work yet with current Mooncake/DI versions (fails at `prepare_gradient` with multi-argument `Constant` contexts).
