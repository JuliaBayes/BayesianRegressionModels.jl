# Pupil model: exact marginalization, centering and sampler comparisons

Prepared for discussion with Aki · 15 September 2026

## Summary

We compared three ways of representing the **same pupil-model posterior**:
ordinary brms population coefficients plus subject deviations; the exact
sum-to-zero (S2Z) marginalization in brms PR #1919; and a manual marginalization
that samples symmetric, subject-labelled **total coefficients**. The last two
integrate out two population/group-mean directions and permit exact conditional
recovery of the original parameters.

For the total-coefficient representation we tested fixed centered (CP), fixed
noncentered (NCP), and offline adaptively partially centered (ACP) coordinates
with WarmupHMC (WHMC). We tested brms S2Z CP/NCP/auto with both WHMC and native
Stan. Ordinary brms NCP under both samplers, and ordinary brms CP under WHMC,
provide additional controls.

In these one-chain pilots, our ACP has the best **sampling-epoch** minimum
ESS per gradient. After charging its pilot and all warmup, **our fixed CP and
brms S2Z auto + WHMC are essentially tied for the best total efficiency**.
Neither S2Z nor exact total-coefficient marginalization makes the NCP endpoint
efficient here. Centering is a major part of the observed improvement.

These are measured smoke tests, not a replicated performance ranking. One
table compares the **same 46 scientific quantities** throughout: population
coefficients, group scales, residual-model coefficients and subject-specific
total coefficients. Marginalized methods conditionally recover the population
coefficients; that recovery and its effect on ESS are explained below.

## 1. Exact model and data

The starting point is the pupil example in [post 3 of the Stan discussion](https://discourse.mc-stan.org/t/help-testing-brms-pr-for-sum-to-zero-and-partial-centering/41542/3).
The dataset contains **2,228 observations, 20 subjects (IDs 701–720), and numeric
load values 0–5**, kept in their original order. We deliberately simplify the
forum model by making random intercepts and slopes independent:

```r
bf(p_size ~ load + (load || subj), sigma ~ subj)
```

An important literal-data detail: `subj` is numeric. Thus `sigma ~ subj` is a
linear regression of **log residual SD on numeric subject ID**, not a separate
residual SD for each subject. We preserve that post-3 specification; we do not
substitute the different residual model discussed later in the thread.

Writing $x_n$ for load, $s_n$ for numeric subject ID, and $j[n]$ for its group,

$$
y_n\sim N(\mu_n,\sigma_n^2),\qquad
\mu_n=\beta_0+\beta_1(x_n-\bar x)+a_{j[n]}+b_{j[n]}x_n,
$$

$$
\log\sigma_n=\gamma_0+\gamma_1(s_n-\bar s).
$$

Here $\bar x$ and $\bar s$ are observation-weighted means. Fixed-effect load is
centered by brms; random-effect load uses the raw predictor. The priors are:

| Quantity | Prior |
|---|---|
| Population intercept $\beta_0$ at mean load | Student-$t_3$(location 5651.9, scale 2026.1) |
| Population slope $\beta_1$ | Flat, as in the generated brms model |
| Subject deviations | $a_j\sim N(0,\tau_a^2)$, $b_j\sim N(0,\tau_b^2)$, conditionally independent |
| Group SDs $\tau_a,\tau_b$ | Half Student-$t_3$(0, 2026.1) |
| Log-residual-SD intercept $\gamma_0$ | Student-$t_3$(0, 2.5) |
| Log-residual-SD slope $\gamma_1$ | Flat, as in the generated brms model |

This is the same posterior in every Student-t row below. It differs from the
forum's correlated-random-effects fit. Earlier Gaussian-intercept sensitivity
runs are retained separately; they are not mixed into the main table.

## 2. Exact total-coefficient marginalization

Define the complete subject-specific coefficients

$$
A_j=\beta_0-\bar x\beta_1+a_j,\qquad B_j=\beta_1+b_j.
$$

The likelihood becomes $\mu_n=A_{j[n]}+B_{j[n]}x_n$. Instead of sampling the
two population coefficients and 40 deviations, integrate out the two
population coefficients and sample the 40 totals. All subjects are treated
symmetrically; no subject is chosen as a dependent last group and there is no
superfluous direction.

For the Student-t intercept use the exact scale mixture

$$
\lambda\sim\operatorname{Gamma}(3/2,\text{rate}=3/2),\qquad
\beta_0\mid\lambda\sim N(m,s^2/\lambda),
\quad m=5651.9,\ s=2026.1.
$$

The fitted target retains $\eta=\log\lambda$; the population coefficients
are integrated analytically conditional on it. Let

$$
\bar A=J^{-1}\sum_j A_j,\quad \bar B=J^{-1}\sum_j B_j,\quad
Q_A=\sum_j(A_j-\bar A)^2,\quad Q_B=\sum_j(B_j-\bar B)^2,
$$

$$
h=\bar A+\bar x\bar B-m,\qquad
V=s^2/\lambda+(\tau_a^2+\bar x^2\tau_b^2)/J.
$$

Up to constants independent of all parameters, the induced log prior is

$$
-(J-1)(\log\tau_a+\log\tau_b)
-\frac{Q_A}{2\tau_a^2}-\frac{Q_B}{2\tau_b^2}
-\frac12\log V-\frac{h^2}{2V}.
$$

The implementation includes normalization constants, hyperpriors, the mixture
prior and log-transform Jacobians. Evaluating this prior and its gradient is
**O(J)**. No numerical quadrature is performed while sampling. This particular
Gaussian likelihood also admits exact per-subject sufficient statistics, so
our manual likelihood evaluation is O(J) after data preparation.

The dimension is **45**: 40 totals, two group scales, two residual-model
coefficients and one mixture precision. Ordinary brms has **46** sampled
parameters, with the Student-t intercept evaluated directly. The dimensional
reduction is exact marginalization, not just a bijective change of all 46
original coordinates. Original parameters remain recoverable conditionally.

## 3. What CP, NCP and ACP mean here

For each total coefficient $q_j$, use the independently specified family

$$
r_j(c_j)=c_jm_j+(q_j-m_j)\tau_{k(j)}^{c_j-1},\qquad 0\le c_j\le1,
$$

where $m_j=5651.9$ for an intercept total and zero for a slope total;
$\tau_{k(j)}$ is the corresponding group SD. These are fixed reference
locations, not a claim that the marginal prior on each total has that mean.

- **CP:** $c_j=1$, so the sampler uses the physical total coefficient.
- **NCP:** $c_j=0$, so it uses $(q_j-m_j)/\tau_{k(j)}$.
- **ACP:** choose each $c_j$ from a pilot, then hold it fixed during a fresh fit.

The NCP endpoint does **not** whiten the induced joint prior. The total
coefficients remain coupled by the integrated population-intercept factor.
It differs from ordinary brms NCP, which samples standardized deviations
$a_j/\tau_a$ and $b_j/\tau_b$ alongside the population coefficients.

Our offline choice minimizes, separately for each of the 40 total coordinates,

$$
L_j(c)=\log\operatorname{sd}_{\mathrm{pilot}}(r_j(c))
       +\operatorname{mean}_{\mathrm{pilot}}[(1-c)\log\tau_{k(j)}],
$$

over the fixed grid $0,0.01,\ldots,1$. The second term is the inverse
log-Jacobian contribution. This is a variance/Jacobian criterion, not a direct
optimization of ESS. We used a 2,000-draw total-NCP WHMC pilot, then a fresh
2,000-draw fit with the selected controls. Its full pilot cost is charged.
WHMC's ordinary linear and step-size adaptation remains enabled in every
WHMC arm. **Nonlinear online centering adaptation is disabled in this matrix.**

![The same 2,000 retained ACP-fit draws viewed as total CP, total NCP and offline ACP. Rows select distinct coordinates by minimum selected centeredness, nearest 0.5 among remaining coordinates, and maximum among the rest. A denotes intercept total, B slope total; numeric suffixes are subject IDs. CP is the visualization baseline.](results/student_mixture/pairs.png)

This figure compares coordinates of the same draws, not three different
posteriors or three independently fitted datasets.

## 4. The brms S2Z comparison

We use the pinned implementation in [brms PR #1919](https://github.com/paul-buerkner/brms/pull/1919),
commit `73cf607889879cb2a55f50b88d8141d76ff43279`:

```r
bf(p_size ~ load +
   (load | gr(subj, cor = FALSE, s2z = TRUE, center = TRUE)),
   sigma ~ subj)
```

Replace `TRUE` by `FALSE` or `"auto"` for the other arms. For each coefficient
block, an orthonormal $J\times(J-1)$ matrix $Q$ represents zero-sum deviations.
CP uses $r=Qz$ and NCP uses $r=\tau Qz$. Two finite-population coefficients
$\theta$ retain the overall means; the original group-mean directions are
integrated out. Original population coefficients and deviations are recovered
in generated quantities. The Student-t mixture again gives 45 sampled
dimensions. Its mixing convention is $u\sim\mathrm{InvChiSquare}(3)$ with
intercept scale $s\sqrt{3u}$, equivalent to our $\lambda=1/(3u)$.

For auto, the branch supplies fixed group/term weights $\rho_{jk}$ from its
own Pathfinder precursor and likelihood-Fisher calculation. Its transform is

$$
d_j=1-\rho_j+\rho_j\tau,\qquad
w_j=\tau(Qz)_j/d_j,\qquad r_j=w_j-\bar w.
$$

For this Gaussian model, the Fisher calculation accumulates within-subject
design information weighted by inverse residual variance, computes local
prior-scaled Gaussian covariance approximations, and adjusts those covariances
for the sum constraint. The resulting variance fractions are converted to
the branch's $\rho$ convention. These weights are **not the same numerical
controls or objective as our $c_j$**.

We let brms resolve its auto target once, then give **the identical final Stan
source and resolved data/weights** to native Stan and BridgeStan + WHMC.
The precursor cost was 789 gradient calls, charged to both workflows. There
is no retuning of brms's weights using the WHMC result.

![The same 2,000 retained brms S2Z auto + WHMC draws viewed in CP, NCP and brms auto coordinates. Rows select distinct subject/term weights by minimum rho, nearest 0.5 among remaining weights, and maximum among the rest. Lowercase a and b denote zero-sum intercept and slope contrasts. Each displayed vector has 20 subject-labelled entries but only 19 independent directions.](results/student_mixture/qoi46/s2z_pairs.png)

The CP column shows physical zero-sum contrasts $r_j$, NCP shows $r_j/\tau$,
and ACP shows the subject-labelled source coordinates $(Qz)_j$ of the auto
fit. The latter uses the full inverse transformation, including the shared
centering correction. These are constrained coordinates for visualization;
the sampler uses the 19 independent entries of $z$ per term. This plot's
contrasts differ from the complete subject totals in the preceding plot.

The ordinary brms CP control uses the same branch's `center=TRUE` without
S2Z. Its generated convention incorporates the population slope into the
centered slope coordinates while leaving the random intercept coordinates
near zero-centered deviations; it does not absorb the population intercept.
We evaluate its actual generated transform, rather than assuming its sampled
variables are identical to our total coefficients.

## 5. Sampling and accounting

Each completed fit has **one chain, seed 1 and 2,000 retained draws**.

- **Native Stan:** CmdStan 2.39.0, CmdStanR 0.9.0, diagonal-metric NUTS,
  1,000 warmup iterations, target acceptance 0.8, maximum tree depth 10.
- **WHMC:** WarmupHMC commit `deeea1d128d5235ad0ecb2fd911a6d881f1ac2c2`,
  Julia 1.10.11, default adaptive warmup and Pathfinder initialization,
  `monitor_ess=true`, `nonlinear_adapt=false`, retained-draw floor 2,000.
- **Initialization:** ordinary and total-coefficient fits start from the same
  physical per-subject OLS coefficients and scales. The S2Z fits use pooled
  finite-population coefficients, zero contrasts and the OLS scales. The
  supplied physical point matches within each native/WHMC pair. Initialization
  is therefore not fully controlled across parametrization families; WHMC
  additionally runs Pathfinder from that point.

The estimands are the **same 46 scientific quantities** in every row:

- Population intercept $\beta_0$ at mean load and population slope $\beta_1$.
- Group SDs $\tau_a,\tau_b$ and residual-model coefficients $\gamma_0,\gamma_1$.
- Subject-specific intercept totals $A_1,\ldots,A_{20}$ and slope totals
  $B_1,\ldots,B_{20}$.

Group SDs are stored in log units; rank-based ESS is unchanged by this monotone
transformation. Population coefficients come directly from ordinary brms and
from exact conditional recovery for both marginalized representations. Totals
are deterministic functions of the sampled state in every arm. Deviations,
standardized sampler coordinates and the auxiliary mixture precision are not
additional quantities in the minimum.

We report minimum rank-normalized bulk ESS with these three cost quantities:

1. **Total gradients:** every fitted-target gradient call in initialization,
   warmup and sampling, plus any required precursor or offline pilot.
2. **Min ESS / sampling gradients:** sampling-epoch efficiency, relevant to
   a longer run after adaptation.
3. **Min ESS / total gradients:** end-to-end gradient efficiency for this run.

In the table, plot and numerical comparisons below, both efficiencies are expressed
as ratios to **ordinary brms NCP + native Stan**. Each metric is divided by its
own baseline value, using the same parameter scope, so the baseline is **1×**
in both efficiency columns. For example, 10× means ten times the baseline's
minimum ESS per gradient. Total gradient counts remain absolute.

WHMC target wrappers count actual density-and-gradient requests. For native
Stan, we use CmdStan's user-defined C++ function support (`user_header` and
`allow-undefined`) to add an integer counter to the generated model. It
increments on gradient evaluations without changing the model's log density
or gradient; the CmdStan NUTS sampler itself is unchanged. We record its value
throughout each run and at the end of each process, including the auto-centering
pilot. As a check, every retained NUTS iteration used one gradient evaluation
at the start of its trajectory plus one per leapfrog step. Required pilot cost
is included, while model compilation and research/debugging attempts are not
gradient costs of the completed workflow.

Gradient count is not literal wall time. Our manual target uses sufficient
statistics while generated brms evaluates the observations, so the time per
gradient differs. These numbers primarily compare sampling/adaptation work;
we do not claim a controlled wall-time ranking.

## 6. Sampling efficiency for the 46 scientific quantities

Each row takes the minimum bulk ESS over the population coefficients, scales,
residual-model coefficients and subject totals listed above. The baseline's
minimum ESS is **170.3**, limited by its population intercept at mean load.
The identical estimands and baseline denominator apply to every row.

**All efficiency entries below are relative to brms NCP + native Stan (1×).**
“Totals” denotes our exact total-coefficient marginalization.

@@PRIMARY@@

![The table plotted with logarithmic axes. Total gradient cost is better to the left; relative sampling and total efficiencies are better to the right. Orange denotes native Stan and blue WHMC. The baseline is ordinary brms NCP + native Stan; its two efficiencies equal 1. Points are single-chain pilot measurements, without uncertainty intervals.](results/student_mixture/qoi46/efficiency.png)

Several distinctions matter:

- **Adaptation quality versus up-front cost:** total ACP gives @@ACP_SAMPLING@@
  the baseline's sampling efficiency, compared with total CP's @@CP_SAMPLING@@. But its NCP pilot
  costs 134,652 gradients and the refit 34,280, for 168,932 overall. Its total
  efficiency is consequently @@ACP_TOTAL@@ the baseline, versus fixed CP's @@CP_TOTAL@@.
- **No clear end-to-end winner between the two best pilots:** total CP and
  brms S2Z auto + WHMC give @@CP_TOTAL@@ and @@S2Z_AUTO_TOTAL@@ the baseline's total efficiency. That difference is negligible
  relative to the uncertainty of single-chain runs.
- **Centering matters strongly:** both marginalized NCP endpoints remain
  inefficient here. Ordinary brms CP improves on the ordinary NCP
  baseline, but its population intercept still limits efficiency. Thus a comparison only against ordinary NCP would
  give an incomplete account of the gains.
- **Sampler choice and initialization matter:** native and WHMC sampling
  efficiencies differ, and total cost can reverse the impression. For ordinary
  Student-t NCP, WHMC achieves 1.43× the native baseline's sampling efficiency
  and 1.91× its total efficiency on these quantities.

All Student-t rows have zero sampling divergences. This does not certify
convergence: the total-NCP control has within-chain split $\hat R$ up to
1.030, and some NCP minima are below 100 effective draws. Native ordinary NCP
hit depth 10 on 14 retained iterations. Across the common-coordinate mean
checks, the largest difference from the total-ACP fit was 4.42 estimated
combined MCSEs, for native S2Z NCP's `total_load[716]`. These descriptive checks
and the low-ESS arms motivate replicated convergence checks before strong
performance claims.

## 7. Conditional recovery of population coefficients and its effect on ESS

The population coefficients are scientific quantities in the main table.
Ordinary brms samples them; the two marginalized representations recover them
using additional conditional random draws. No further HMC transitions are run.
Subject deviations can also be recovered, but they are not additional headline
quantities: the table already includes the complete subject-specific effects.

### Exact conditional recovery

Both exact marginalizations permit recovery of the original population
coefficients and subject deviations. For our representation, condition on
the totals, scales and mixture precision. Let

$$
C=\frac1J\begin{pmatrix}
\tau_a^2+\bar x^2\tau_b^2 & \bar x\tau_b^2\\
\bar x\tau_b^2 & \tau_b^2
\end{pmatrix},\quad v_0=s^2/\lambda,\quad M=\bar A+\bar x\bar B.
$$

Then the population coefficients have a two-dimensional Gaussian conditional
with mean

$$
\mu_0=m+\frac{v_0}{v_0+C_{00}}(M-m),\qquad
\mu_1=\bar B+\frac{C_{01}}{v_0+C_{00}}(m-M)
$$

and covariance

$$
\Sigma_{00}=\frac{v_0C_{00}}{v_0+C_{00}},\quad
\Sigma_{01}=\frac{v_0C_{01}}{v_0+C_{00}},\quad
\Sigma_{11}=C_{11}-\frac{C_{01}^2}{v_0+C_{00}}.
$$

After drawing $(\beta_0,\beta_1)$, recover
$a_j=A_j-\beta_0+\bar x\beta_1$ and $b_j=B_j-\beta_1$.
This needs no additional HMC. The brms rows use its actual generated-quantity
recovery; our rows use recovery seed 101. The independent 20-seed
sensitivity for our fits is retained in the accompanying results.

To see why, write a recovered quantity as
$X_t=m(S_t)+\epsilon_t$, where recovery noise has conditional mean zero and is
drawn independently across iterations. Then

$$
\operatorname{Var}(X)=\operatorname{Var}(m(S))+
E[\operatorname{Var}(X\mid S)],\quad
\operatorname{Cov}(X_t,X_{t+k})=\operatorname{Cov}(m(S_t),m(S_{t+k}))\quad(k>0).
$$

Added recovery variance dilutes positive autocorrelation and can raise ESS.
Yet for estimating a posterior mean,

$$
\operatorname{Var}(\bar X)=\operatorname{Var}(\overline{m(S)})+
E[\operatorname{Var}(X\mid S)]/N.
$$

The conditional-mean (Rao–Blackwell) estimate therefore avoids that added
Monte Carlo noise. This is not invalid posterior recovery; it is a reason to
compare the same estimands and examine MCSE as well as ESS. These equations
describe covariance/mean ESS; rank-normalized bulk ESS is measured separately.

In our Student-t ACP fit, about **99.966%** of the recovered population-
intercept variance is conditional recovery variance. Its conditional-mean
MCSE is **0.225**, while recovered-mean MCSE is approximately **12.08**
(median over 20 recovery seeds). Both bulk ESS values are about 2,000.
Thus ESS alone does not reveal the difference in mean-estimation precision.
The main table includes recovered population coefficients because they are
quantities of interest, and includes totals without recovery noise. Its minimum
can therefore be limited by either kind of quantity. Per-quantity ESS and MCSE
are retained so this distinction can be examined directly. See the
[Stan discussion of ESS and MCSE](https://mc-stan.org/docs/reference-manual/analysis.html#effective-sample-size).

## 8. Verification, reproducibility and limits

The manual target was checked against an independent dense Gaussian
conditioning calculation and the original observation-level likelihood.
Each generated S2Z target passed eight nontrivial coordinate, density and
finite-difference gradient checks against that manual target, including
normalization constants and change-of-variable determinants. Maximum absolute
log-density discrepancy was below $4.8\times10^{-11}$ and maximum normalized
coordinate-gradient discrepancy below $2\times10^{-7}$. Generated quantities
recover exactly the same totals before and after adding conditional noise.
Ordinary brms CP/NCP and our recovery conditional have separate audits.

Full draws, native CSVs, source capsules, resolved auto weights, per-parameter
diagnostics, gradient receipts and process logs are retained. The matrix is
computed from saved draws; no posterior refits are needed to change diagnostic
scopes. The implementation is a manual research prototype, not a new production
BRM interface. The exact formula, recovered-parameter scope and gradient
denominators are recorded with the results.

Pinned sources:

- Data: [`bnicenboim/bcogsci`, revision `d90fc01e6f6fcdced7ee64c9d2ed607d212ec77c`](https://github.com/bnicenboim/bcogsci/blob/d90fc01e6f6fcdced7ee64c9d2ed607d212ec77c/data/df_pupil_complete.rda).
  The Rda SHA-256 is `ab45331f4d2be447211832bd6cf13501b31032837ddb16acada6d414fbf46042`;
  the lossless CSV SHA-256 is `27e73bec304bc68271b0a9d46b1736a7c8e62fd8dcfac7d20704524c47edc8e8`.
- brms 2.23.1, source commit `73cf607889879cb2a55f50b88d8141d76ff43279` from the PR branch.
- WarmupHMC commit `deeea1d128d5235ad0ecb2fd911a6d881f1ac2c2`; Julia 1.10.11;
  CmdStan 2.39.0 and CmdStanR 0.9.0.
- Research code and saved-result manifest: `research/pupil_total_effects/`
  in the working BayesianRegressionModels repository, snapshot **@@SOURCE_SHA@@**.

What remains unestablished is a replicated ranking across seeds/chains,
matched-initialization performance across all representations, controlled wall
time, behavior of the original correlated-effects pupil model, and behavior
with multiple crossed group structures. No online-centering result or native
Stan implementation of our manual total target is included here.

## 9. Inspect the harness and exact generated Stan files

@@CODE_LINKS@@

The most useful next comparison would repeat the promising CP/ACP/S2Z-auto
arms with matched physical starting states and multiple chains, keeping this
scientific scope fixed and checking conditional-mean MCSE for recovered quantities.
The present result is narrower: **symmetric total coefficients can preserve
the exact posterior and sample efficiently; a strong CP baseline and full
pilot accounting materially change the apparent benefit of adaptation.**
