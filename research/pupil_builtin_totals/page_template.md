# Pupil: numeric scale predictor and automatic totals

## Result

BRM can express the pupil model from [post 3 of the Stan discussion](https://discourse.mc-stan.org/t/help-testing-brms-pr-for-sum-to-zero-and-partial-centering/41542/3), integrate its population mean coefficients, and expose its total coefficients to WHMC adaptation automatically.

All six total-coefficient variants completed with zero sampling divergences. In this pilot, **online position adaptation gave 318× the total-gradient efficiency of ordinary brms NCP + native Stan**, and 1.64× that of brms S2Z auto + WHMC. Online gradient adaptation gave 224× the native baseline. Fixed CP and S2Z auto were also effective; neither marginalization made fixed NCP competitive with those centered alternatives.

The table compares one common set of 46 scientific quantities. These are single-chain experiments; the numerical ordering of the two losses needs replication.

## Model and data

The data contain 2,228 observations, 20 subjects with IDs 701–720, and numeric load values 0–5. We preserve observation order and use independent subject intercept and slope deviations:

```r
bf(p_size ~ load + (load || subj), sigma ~ subj)
```

In these data `subj` is numeric. Thus the residual model is a regression of **log residual SD on subject ID**. The [second pupil study](pupil-scale-centering.md) instead assigns each subject a hierarchical residual SD.

```math
y_n\sim N(\mu_n,\sigma_n^2),\qquad
\mu_n=\beta_0+\beta_1(x_n-\bar x)+a_{j[n]}+b_{j[n]}x_n,
\qquad\log\sigma_n=\gamma_0+\gamma_1(s_n-\bar s).
```

The population load and subject-ID predictors use observation-weighted centering. The random slope uses raw load. Priors match the generated brms target:

- Population intercept: Student-t(3, location 5651.9, scale 2026.1).
- Population load slope: flat.
- Independent deviations: `a[j] ~ Normal(0, tau_a)`, `b[j] ~ Normal(0, tau_b)`.
- Both group SDs: half-Student-t(3, 0, 2026.1).
- Log-SD intercept: Student-t(3, 0, 2.5); numeric-ID slope: flat.

Flat priors apply to these population slopes because that is the source model's specification. The group effects retain their Gaussian hierarchical priors. Independence of the mean random effects is an explicit simplification of the forum model, shared by every row below.

## BRM implementation

The authoring pane reads the actual declarations used for the fits. Its backend panes are generated during the documentation build. The measurements use StanBlocks/BridgeStan and WHMC; the Turing pane is a generated-model comparison only.

```@eval
Main.BRMDocsComparisons.comparison(@__MODULE__,
    Main.BRMCenteringExamples.authoring(:pupil_numeric),
    :pupil_numeric_brm_model;
    title="Pupil means with a numeric subject-ID scale predictor", require_stan=true)
```

BRM recognizes that the population intercept and slope share their design with the subject effects. It replaces them with subject totals

```math
A_j=\beta_0-\bar x\beta_1+a_j,\qquad B_j=\beta_1+b_j,
\qquad\mu_n=A_{j[n]}+B_{j[n]}x_n.
```

The Student-t intercept uses an exact Gaussian scale mixture,

```math
\lambda\sim\operatorname{Gamma}(3/2,\mathrm{rate}=3/2),\qquad
\beta_0\mid\lambda\sim N(5651.9,2026.1^2/\lambda).
```

Conditional on `lambda`, both population mean coefficients are integrated analytically. All subjects are treated symmetrically. The two residual-model coefficients remain explicit parameters because this predictor has no random-effects block. Ordinary brms samples 46 parameters; BRM totals and brms S2Z sample 45, including one mixture variable.

The induced prior depends on the sums and centered sums of squares of the totals, so evaluation is O(J) for fixed coefficient dimension. BRM's generated likelihood evaluates the observations. No numerical quadrature is used during sampling.

## Centering and automatic WHMC integration

For each total `q[j]`, BRM supplies the family

```math
r_j(c_j)=c_jm_j+(q_j-m_j)\tau_{k(j)}^{c_j-1},\qquad 0\le c_j\le1.
```

The fixed reference location is 5651.9 for intercept totals and zero for slopes. CP uses `c=1`, NCP uses `c=0`, and partial centering chooses an intermediate value. Scaling the totals does not whiten their integrated joint prior; total NCP is a different construction from ordinary standardized-deviation NCP.

```julia
using StanBlocks, BridgeStan, WarmupHMC, Enzyme, Random
using DifferentiationInterface: AutoEnzyme

sb = SBBRMI(pupil_numeric_brm_model(); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model)
names = BridgeStan.param_unc_names(problem.model)
adaptive = adaptive_centering_problem(sb, problem, AutoEnzyme(); centeredness=0.0)
fit = adaptive_warmup_mcmc(Xoshiro(1), adaptive;
    n_draws=2000, monitor_ess=true, nonlinear_adapt=true)

draws = permutedims(fit.posterior_position) # compiled-model coordinates
recovered = recover_population_draws(sb, draws, names; rng=Xoshiro(101))
```

Fixed CP/NCP uses the corresponding `centeredness` and `nonlinear_adapt=false`. Post-hoc selection uses a completed NCP pilot:

```julia
selection = select_total_centeredness(sb, pilot_draws, names; criterion=:position)
partial = adaptive_centering_problem(sb, problem, AutoEnzyme();
    centeredness=selection.centeredness)
partial_fit = adaptive_warmup_mcmc(Xoshiro(1), partial;
    n_draws=2000, monitor_ess=true, nonlinear_adapt=false)
```

Both post-hoc arms use the same pilot and the built-in grid `0:0.1:1`. The position loss minimizes log coordinate SD plus the mean inverse log-Jacobian contribution. The gradient loss minimizes the correlation between candidate position and its density gradient. It uses the pilot's saved gradients after transport into the compiled model frame. No new target evaluations are needed for selection. Neither loss directly optimizes ESS.

Online adaptation uses warmup windows and freezes the centering controls for retained sampling. Both losses are tested; the harness selects WHMC's existing position-loss weights locally, while its default is the position–gradient criterion. This loss switch is not a new public BRM option.

## Matched comparisons and gradient accounting

Every row measures the same **46 scientific quantities**: the two population mean coefficients, two group SDs, two residual-model coefficients, and 20 subject total intercepts and slopes. Population coefficients are sampled directly by ordinary brms and recovered conditionally by the marginalized methods. Totals are deterministic functions of each sampled state. Deviations, standardized coordinates and mixture variables do not enter the minimum ESS.

Both efficiency columns are relative to **ordinary brms NCP + native Stan**. Sampling efficiency is minimum bulk ESS divided by sampling gradients. Total efficiency includes initialization, warmup and all required pilots. Each post-hoc workflow is charged its full 84,335-gradient NCP pilot. S2Z auto includes its 789-gradient precursor.

{{TABLE}}

[![Total gradient costs and relative efficiencies for 46 scientific quantities.](assets/pupil-builtin-centering/efficiency.png)](assets/pupil-builtin-centering/efficiency.png)

The post-hoc rows improve sampling efficiency but pay for a separate pilot. Online adaptation avoids that cost. The strong CP and S2Z-auto controls show that a comparison against ordinary NCP alone would give an incomplete picture.

Each arm uses one chain, seed 1 and 2,000 retained draws. Native Stan uses 1,000 warmup iterations, target acceptance 0.8 and maximum depth 10. WHMC uses its adaptive warmup and Pathfinder initialization. Ordinary and total arms start at the same physical per-subject OLS coefficients; S2Z starts at pooled coefficients with zero contrasts and the OLS scales. Initialization matches within native/WHMC pairs but is not fully controlled across parameterization families.

The nine brms fits are reused from the verified comparison because their targets and sampler configurations are unchanged. All six BRM-total fits use the built-in implementation and repaired WHMC transport. The comparison measures gradient work, not identical wall cost per gradient.

## Saved-draw geometry

The total-coefficient figure shows the **same 2,000 post-hoc position draws** in CP, NCP and ACP coordinates. Rows choose distinct coordinates by minimum selected centeredness, closest to 0.5, and maximum. CP is the visualization baseline; the ACP values were checked against the saved source coordinates.

[![Identical saved total-coefficient draws viewed in CP, NCP and partial coordinates.](assets/pupil-builtin-centering/total_pairs.png)](assets/pupil-builtin-centering/total_pairs.png)

The S2Z figure reuses the unchanged auto-WHMC fit. It shows centered contrasts, standardized contrasts, and the branch's subject-labelled `Q*z` coordinates. Those J displayed values represent J−1 independent contrast directions. The auto transform includes its weighted centering correction.

[![Identical S2Z-auto draws viewed in centered, noncentered and auto contrast coordinates.](assets/adaptive-pupil/s2z_pairs.png)](assets/adaptive-pupil/s2z_pairs.png)

brms S2Z auto uses the pinned branch's Pathfinder/Fisher selection, then holds its weights fixed. WHMC receives that same resolved target and weights. These weights differ from BRM's per-total centering controls and loss functions; S2Z auto here is not online WHMC adaptation.

## Recovery and validation

BRM recovers population coefficients from their exact Gaussian conditional given the totals, scales and mixture variable. For total block T with population-to-group design matrix A, its precision is

```math
Q=P+J A^\mathsf T\operatorname{diag}(\tau^{-2})A,
```

and its mean is `Q \ (P*location + A' * diag(tau^-2) * sum(T))`. Adding an independent conditional draw recovers the original population parameters; the subject totals stay fixed.

Conditional recovery adds genuine posterior variation and can raise ESS by diluting autocorrelation. It also adds noise to posterior-mean estimation. We therefore retain per-quantity MCSEs, repeat recovery across ten seeds, and compare the 44 quantities that require no stochastic recovery as a sensitivity. The main table consistently includes all 46 scientific quantities.

{{VALIDATION}}

The automatic target passed 25 density, gradient, centering-coordinate and recovery checks against the independent analytic implementation. Adding two constant half-Student-t normalizers aligns the absolute density; gradients and the posterior already agree. Generated S2Z and ordinary brms targets have their separate retained audits. These checks establish target equivalence, not a replicated sampling-performance ranking.

## Inspect and reproduce

{{LINKS}}

The data revision is `d90fc01e6f6fcdced7ee64c9d2ed607d212ec77c`; brms is pinned to PR #1919 revision `73cf607889879cb2a55f50b88d8141d76ff43279`. New total fits use BRM's built-in planner with the retained-flat-prior correction and WarmupHMC `7aed40b18bd4cdabb75330f285d5c9b355575ab9`, Julia 1.10.11. Native fits use CmdStan 2.39.0 and CmdStanR 0.9.0. The archive manifests retain the exact source and raw-draw provenance for reused and new fits.
