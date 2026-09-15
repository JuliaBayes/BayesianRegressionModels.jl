# S2Z in BRM and its composition with adaptive centering

Research/design, 2026-09-15. No production S2Z implementation or performance claim is made here. The numerical companion is `verify_math.jl`; `math_receipt.toml` and `verify_math.log` record its checks.

## Result

Posterior-preserving sum-to-zero (S2Z) coordinates can compose with both our pilot/refit selection and online adaptation. There are two implementation levels:

1. Express each group block in orthonormal contrasts and adapt those free coordinates using the existing centering family. This is the smallest route to a working implementation.
2. Preserve a centeredness control for every original group and term. This requires a coupled block transform, whose forward map, inverse and Jacobian are available in closed form below. The existing independent per-coordinate selector cannot be applied unchanged.

The S2Z model transformation and the choice of centering remain separate operations. Neither requires adopting brms's Fisher-based automatic rule.

## 1. Preserve the original posterior

Take one group block with J levels and K coefficients:

\[
b_j\mid\Sigma\sim N_K(0,\Sigma),\qquad \Sigma=LL^T.
\]

Let Q be a J by J-1 orthonormal contrast matrix: Q'Q=I, Q'1=0. Let P=QQ'=I-11'/J. Write

\[
m=J^{-1}\sum_j b_j,\qquad \delta_j=b_j-m.
\]

Conditional on Sigma, m and delta are independent:

\[
m\sim N_K(0,\Sigma/J),\qquad
\operatorname{Cov}(\delta_j,\delta_l)=(\mathbb{1}_{j=l}-1/J)\Sigma.
\]

The mean here is the arithmetic mean across group levels, regardless of observation counts. For J=1 there are no contrast coordinates to adapt. The free contrast coefficients have J-1 independent N_K(0,Sigma) rows. Equivalently, draw a (J-1) by K standard-normal matrix E and set the J by K deviation matrix to Q E L'. Implement Q and Q' as implicit Helmert operations, not dense J by J matrices.

### The population coefficients must change coordinates too

Suppose the likelihood's linear predictor is X beta + Z b. The omitted mean contributes W m, where W=Z(1 tensor I_K). If W=X A for a known coefficient map A, define

\[
\alpha=\beta+A m,\qquad \eta=X\alpha+Z\delta.
\]

This equality holds for Gaussian, Bernoulli/logit, Poisson and other likelihoods depending on the same predictor. It does not require a Gaussian likelihood. Each linked distributional predictor must obey the same design identity.

**General exact route:** retain m and evaluate its Gaussian prior together with the original population prior p_beta(alpha-A m). This is an invertible reparameterization with the original dimension. The shear has determinant one; the fixed orthonormal mean/contrast change has only a parameter-independent normalization when m rather than sqrt(J)m is used. It can improve geometry but does not remove the mean direction from sampling.

**Exact collapsed route for Gaussian population priors:** if beta ~ N(mu,V), integrate m out:

\[
\alpha\mid\Sigma\sim N\left(\mu,\ C\right),\qquad
C=V+A(\Sigma/J)A^T.
\]

Keep all covariance-dependent normalizing terms. For a scalar random intercept this becomes alpha ~ N(mu, s_beta^2 + tau^2/J), where the second argument here is the variance.

For several independent random-effect blocks, concatenate their means, set S=blockdiag(Sigma_g/J_g), and collect their maps into A. Then C=V+A S A'. Blocks sharing a population intercept must be integrated jointly. For two crossed random intercepts, its variance becomes s_beta^2 + tau_1^2/J_1 + tau_2^2/J_2. Marginalization removes the sum of the block mean dimensions from the sampled state.

### Exact recovery and prediction

After a collapsed draw (alpha,delta,Sigma), recover the original mean vector from

\[
m\mid\alpha,\Sigma\sim N\left(SA^TC^{-1}(\alpha-\mu),\ S-SA^TC^{-1}AS\right),
\]

then beta=alpha-A m and b_j=delta_j+m_g. Use Cholesky solves; do not explicitly invert C. Conditional recovery is a random draw, not substitution of the conditional mean. A stable simulation alternative draws independent m0~N(0,S) and beta0~N(mu,V), then applies m=m0+SA'C^-1[alpha-(beta0+A m0)]. This also avoids constructing a potentially delicate conditional covariance difference.

Existing-level predictions can use alpha and delta directly. Population-coefficient summaries, random-effect summaries and new-group predictions require the original recovered quantities. A new group gets a new independent N(0,Sigma) effect with the recovered beta; do not append it to the fitted zero-sum vector and re-center everything. Preserve the training J and basis in saved-fit/replay metadata.

### Priors and scope limits

- Flat population directions remain flat after convolution, and their omitted means can be recovered from the appropriate conditional prior system. Do not approximate a flat prior with an arbitrarily large Gaussian or omit normalizers from proper directions.
- Student-t population priors can be made conditionally Gaussian with an exact scale-mixture auxiliary variable. Perform the calculation conditional on that variable, retaining its prior. Arbitrary other priors can use the retained-mean route; unsupported analytic collapse must be explicit.
- Merely replacing b by a zero-sum vector and keeping beta's old prior gives a different posterior. Strict zero-sum effects are a separate model feature.
- Do not multiply the projected deviations by sqrt(J/(J-1)) in the posterior-preserving construction. Their smaller marginal variance is intentional: the missing Sigma/J belongs to m. Restoring it to delta as well double-counts it.
- A centered zero-sum Gaussian density has J-1, not J, scale normalizers. For a scalar, summing J normal_lpdf(delta_j | 0,tau) requires adding log(tau). For a K-dimensional block the analogous correction is log|L|. Independent free contrasts avoid this bookkeeping trap.
- A group slope with no matching population design cannot use the simple absorbed-mean collapse. Retaining its mean is exact. Do not silently add an ordinary independent fixed-effect prior.
- Shared covariance across levels is the first useful scope. Varying level covariances or conditionally non-Gaussian group priors generally couple m and delta; the simple Sigma/J independent decomposition no longer applies. A larger Gaussian complete-square calculation conditional on mixture variables can extend it later.

## 2. Smallest adaptation route: free contrasts

For each contrast r, use our current triangular centering matrix

\[
(A_r)_{kk}=L_{kk}^{c_{rk}},\qquad
(A_r)_{kl}=c_{rk}L_{kl}\quad(l<k).
\]

Sample v_r=A_r z_r with standard-normal innovations z_r. Recover physical contrasts L A_r^-1 v_r, then apply Q to obtain delta. c=0 is NCP; c=1 is CP of the contrast coefficients. The source-to-NCP Jacobian is -sum_{r,k} c_{rk} log L_kk, exactly the current form.

This can reuse the existing fixed and online centering machinery after BRM exposes J-1 free contrast coordinates rather than pretending it has J unconstrained group coordinates. Per-contrast choices depend on the chosen basis and no longer mean 'this particular group is centered.' Shared choices across all contrasts avoid that group-basis dependence. Pair plots can still display physical groups, but contrast centeredness must not be labeled as per-group centeredness.

## 3. Per-group adaptation on the constrained space

For scalar effects choose our exponent interpolation d_j=tau^c_j, with c_j in [0,1]. Let u be a J-vector summing to zero, represented by free coordinates v=Q'u. Define

\[
\boxed{\delta=\tau P D^{-1}u,\qquad D=\operatorname{diag}(d_1,\ldots,d_J).}
\]

At c=0 this is delta=tau*u. At c=1 it is delta=u. Intermediate group-specific choices remain invertible on the zero-sum subspace:

\[
z_j=\delta_j/\tau,\qquad
u_j=d_j\left(z_j-\frac{\sum_l d_l z_l}{\sum_l d_l}\right).
\]

This weighted subtraction is the coupling that independent rescaling would miss.

The determinant on the J-1 free dimensions is

\[
\boxed{\ell J(c)=(J-1)\log\tau-\sum_j\log d_j+\log\left(J^{-1}\sum_jd_j\right).}
\]

It follows from det(Q'D^-1 Q)=prod_j(d_j^-1)*mean_j(d_j). The shared log-mean term is essential whenever group weights differ. With h=log(tau),

\[
\partial_h\ell J=(J-1)-\sum_jc_j+
\frac{\sum_j c_j\exp(c_jh)}{\sum_j\exp(c_jh)}.
\]

Evaluate log-mean-exp stably. These expressions hold with the candidate weights fixed; their dependence on the current scale is still differentiated.

### Correlated intercept/slope blocks

For each original group j use the same triangular A_j defined above, with its own c_jk. For a zero-sum K-vector collection u_j, set

\[
\tilde\delta_j=L A_j^{-1}u_j,\qquad
\delta_j=\tilde\delta_j-J^{-1}\sum_l\tilde\delta_l.
\]

Its inverse is

\[
z_j=L^{-1}\delta_j,\qquad
s=\left(\sum_j A_j\right)^{-1}\sum_j A_jz_j,\qquad
u_j=A_j(z_j-s).
\]

Every A_j and their sum are lower triangular with positive diagonals, so all solves exist. The restricted log-Jacobian is

\[
\boxed{\ell J=(J-1)\log|L|-\sum_j\log|A_j|+
\log\left|J^{-1}\sum_j A_j\right|.}
\]

Because A_j is triangular, this is the sum of the scalar boxed formulas across k, using tau=L_kk and c_j=c_jk. Off-diagonal entries still matter in the map and its derivatives. Forward/inverse evaluation costs O(J K^2), with one small shared triangular solve; no dense matrix over all J levels is needed. A dimension-independent proof uses the complementary-minor identity det(H'TH)=det(T)det(E'T^-1 E), where H spans the zero-sum subspace and E=(1/sqrt(J)) tensor I spans its orthogonal complement.

The brms commit inspected earlier uses the same projected structure but A_j=diag(1-rho_j)+diag(rho_j)L, whose scalar diagonal is 1-rho_j+rho_j*tau. We can retain our power interpolation; the determinant identity does not depend on choosing their interpolation or their Fisher centering rule.

## 4. Our offline loss after S2Z

An existing ordinary-model pilot can be reused: compute m=mean(b), delta=b-m and alpha=beta+A m for each saved physical draw. These are already draws from the collapsed posterior. No new pilot is required just to obtain S2Z draws, provided the original posterior and priors match exactly. Reconstruct physical effects first if the saved fit used NCP or another source frame. Old joint-target gradients cannot in general be converted to collapsed-target gradients merely by dropping the omitted-mean components; gradient-based diagnostics need the collapsed target evaluated at the mapped draws.

The current scalar selector minimizes log sd(z*tau^c)-E[c log tau]. It is separable because both the map and the diagonal Gaussian approximation separate over free coordinates.

For the coupled S2Z map let x be the fixed physical contrast coordinates plus unchanged global parameters, v_c=T_c^-1(x), and ell J_c=log|dx/dv_c|. The direct diagonal-Gaussian KL analogue, up to terms independent of c, is

\[
\boxed{\mathcal L_{diag}(c)=\sum_a\log\operatorname{sd}(v_{c,a})+
\mathbb E[\ell J_c].}
\]

Unchanged global-coordinate variance terms can be omitted. This follows from H(v_c)=H(x)-E[ell J_c], after optimizing the approximating Gaussian's mean and diagonal variance. For a full Gaussian approximation, use one half log det Cov(v_c) instead of the sum of log standard deviations; include all coordinates whose Gaussian approximation is intended. Full covariance is costly and can be singular with too few pilot draws. Neither objective directly optimizes ESS or guarantees a speedup.

Use the J-1 free contrast coordinates. The J displayed group coordinates are singular: applying a full-dimensional entropy formula to them or using J independent scalar losses is incorrect. The log-mean term and the recentering make group-specific selection coupled. Start with shared block/term weights or coordinate-descent sweeps over the existing 11-point grid while holding other controls fixed. For every candidate, compute the whole affected block's free-coordinate objective, not a single group's old scalar loss. Joint winners require joint scoring; independently minimizing locally computed curves does not solve this objective.

## 5. Online adaptation

Choose the S2Z/collapsed target when constructing the model. Its dimension stays fixed while centeredness adapts. Do not switch an ordinary target into a dimension-reduced target mid-warmup or treat a checkpoint with the old dimension as resumable in the collapsed model.

The current WarmupHMC default minimizes weighted Cor(position,gradient), with the KL/log-variance contribution given zero weight (w1=0). Preserve that identity when calling this 'our online approach.'

For each recorded physical state and each candidate block frame, transform the whole affected block to free v_c and transform its gradient consistently. If x=M_c(theta)v,

\[
g_v=M_c(\theta)^T g_x,\qquad
g_\theta^{source}=g_\theta^{target}+
(\partial_\theta[M_cv])^Tg_x+\partial_\theta\ell J_c.
\]

The first equation applies to the contrast coordinates conditional on theta; the second includes all hyperparameter corrections. In the full state this is the ordinary change-of-variables gradient J_T'g_target + grad log|J_T|. A block criterion can aggregate the existing coordinate correlations over its free coordinates. That aggregation and its update schedule are a new selection policy to specify and measure, even though the underlying position-gradient criterion remains familiar. It is not supplied by the present scalar CandidateScoringPlan callback.

Update weights at existing warmup boundaries, using old-source -> fixed-target -> new-source to preserve the same physical state, and transport/reset cached gradients, halo, metric and accumulators according to the sampler's existing lifecycle. Freeze weights for retained final sampling. A new c does not need an acceptance correction if the density and state transport are exact and adaptation is confined to warmup; changing c inside a trajectory or continuing unrestricted adaptation in final sampling would be a different algorithm.

For the per-contrast route, most existing machinery can remain. For actual per-group S2Z controls, the mutable per-index pair interface, recorder, winner application and BRM Enzyme specialization need an explicit block path. Hiding the shared correction in an index-local accessor would leave scoring/state assumptions inconsistent. Although one map costs O(J K^2), naively trying 11 candidates for every one of J K controls and rescoring the full block on every leaf can be quadratic in J. Start with shared controls or window-level coordinate-descent updates; a faster per-group proxy or low-rank scoring implementation would require its own derivation and measurement. A deliberately approximate fixed-reference scorer is possible, but must be labeled as a proxy rather than the coupled KL objective. Report per-second performance too: ESS per target gradient alone does not expose this extra scoring cost.

## 6. Concrete implementation paths

| Layer | Work |
| --- | --- |
| BRM normalized model / SBBRMI | Add an opt-in posterior-preserving geometry choice; initially ordinary exchangeable Gaussian random effects and matching population columns. Capture the population/group design map, resolved priors, training levels and shared covariance blocks before emitting any local ranef submodel. A constructor option parallel to centered_groups is a possible first surface; naming is not decided. Strict S2Z priors should be a separately explicit statistical choice. |
| BRM emission (`src/sbimpl.jl`) | Add contrast draw families, retained-mean oracle and collapsed Gaussian/conditional-Gaussian population prior. Coordinate all blocks whose means affect the same population vector. Emit generated recovery quantities and preserve logical coefficient addresses. Existing scalar, correlated and shared-ID dispatches are the seams. |
| StanBlocks consumption | Use J-1 ordinary vectors and a typed implicit Helmert helper initially. Native sum_to_zero_vector exposure is not required for this route; its name in BRM's reserved-keyword set is not evidence of existing support. A generic transform primitive can be considered separately if useful. |
| Metadata / prediction (`src/prediction.jl`, `src/adaptive_centering.jl`) | Represent constrained physical effects, free contrast indices, basis, mean recovery and source frame separately. Current RanefBlock/ranef_coordinates assumes K by J raw coordinates, and BRMAdaptiveCenteringState iterates n_groups; neither can simply be reused as-is. Preserve new-level prediction and replay semantics. |
| Fixed/offline path | Implement and verify the block transport and inverse in BRM; wrap the fixed compiled target with its exact Jacobian and Enzyme differentiation. Add the coupled pilot objective and save selected controls plus basis. An offline refit does not require an online block recorder. |
| Online WarmupHMC path | Reuse the existing per-coordinate lifecycle for an initial contrast implementation. Genuine group controls need block candidate/state/recorder support and atomic winner transport; BRM provides geometry metadata and exact maps. This is a possible cross-package extension, not an already-supported API. Keep existing ordinary/HSGP paths and their specialized derivatives unchanged. |

Recommended order: (1) scalar retained-mean oracle plus exact Gaussian collapse, (2) fixed CP/NCP and shared-weight/contrast adaptation, (3) heterogeneous scalar block transport plus offline selector, (4) online block lifecycle, (5) correlated and crossed blocks, conditional Student-t population priors and prediction/replay coverage. This stages implementation risk, not mathematical feasibility: the correlated formulas are already derived above.

Before sampling, compare the retained-mean model against the original density/gradient under an invertible map. For the collapsed model, dimension has changed: verify original joint = collapsed density times conditional recovery density, rather than demanding a nonexistent square Jacobian to the original full parameter vector. Then verify every fixed endpoint/intermediate map, all hyperparameter gradients, zero-sum/inverse invariants, posterior quantities and new-group predictions. Only then measure NCP vs selected refit vs online ESS per gradient and per second, charging pilots/selection/setup honestly.

## Evidence and sources

`julia --project=test research/s2z_design/verify_math.jl` exited 0: 565 assertions across 36 heterogeneous-transform cases, scalar full-target gradient checks, joint Gaussian marginalization/recovery and original prior-covariance reconstruction. Maximum inverse error 1.12e-15; restricted log-determinant error 1.78e-15; full finite-difference Jacobian log-determinant error 5.03e-11; transformed analytic-gradient scaled error 1.90e-9. These are independent mathematical checks, not an Enzyme, Stan, sampler or BRM integration test.

BRM source inspected at 7541607ed1e562898618d26dce898413c1ea369d: src/adaptive_centering.jl, ext/BayesianRegressionModelsWarmupHMCExt.jl, src/prediction.jl, src/sbimpl.jl and the radon offline driver. WarmupHMC public consumer contract was fetched 2026-09-15. The existing online BRM criterion is a fixed-frame scoring proxy even though its actual joint transport is exact.

- [Stan reference manual: sum-to-zero transforms](https://mc-stan.org/docs/reference-manual/transforms.html#sum-to-zero-vector) supports the orthonormal representation and J-1 scale-normalization issue.
- [brms R/stan-predictor.R at 7aed341](https://github.com/paul-buerkner/brms/blob/7aed341b80e92e71ca4c6c7304f9718ab181779f/R/stan-predictor.R#L1365) provides the projected partial transform and joint omitted-mean systems used as a comparison. Our power-interpolation extension, loss derivation and staged BRM design are the derivations in this note.
- [Full testing thread](https://discourse.mc-stan.org/t/help-testing-brms-pr-for-sum-to-zero-and-partial-centering/41542) provides motivation and models, not evidence that this proposed BRM implementation will achieve its reported speedups. Its later pilot-based automatic centering is a different revision from the cited Fisher commit.
