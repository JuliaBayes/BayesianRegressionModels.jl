"""Generate the refreshed narrative around the previously audited model examples."""
from pathlib import Path
import csv
import subprocess

BASE = Path(__file__).resolve().parents[2]
RESULTS = BASE / "research/centering_refresh/results"

def original(page):
    return subprocess.check_output(["git", "show", f"b7fd3f5:docs/src/{page}.md"], cwd=BASE).decode()

def image(case, name, caption):
    path = f"assets/centering-refresh-{case}/{name}.png"
    return f"\n[![{caption}]({path})]({path})\n"

def common(case):
    return r"""
## Both losses, post-hoc and online

For a zero-mean random effect with log scale $\ell$, the centering family is
$u_c=\exp(c\ell)z$: $c=0$ is NCP and $c=1$ is CP. Its transformed effect
gradient is $g_c=\exp(-c\ell)g_z$. We compare two criteria, minimized separately
for each effect:

```math
L_{\mathrm{position}}(c)=\log\operatorname{sd}(u_c)-\operatorname{mean}(c\ell),
\qquad
L_{\mathrm{gradient}}(c)=\operatorname{cor}(u_c,g_c).
```

The second is a **signed** correlation: an independent Gaussian coordinate
has correlation $-1$ with its log-density gradient. The first uses positions
and the Jacobian, without a gradient term.

Both post-hoc arms use the same NCP pilot, select on `0:0.01:1`, then fit
afresh with the controls fixed. The gradient selector uses the pilot's saved
gradients. Both online arms select on the native `0:0.1:1` grid during warmup,
using the sampler's trajectory evidence and weights. They have no separate
pilot. Controls are frozen for the retained sampling phase.

The online gradient criterion is WarmupHMC's default. The research harness
selects the position criterion through the existing internal loss functions;
there is currently no public loss-selection keyword. The model, initialization
policy, seed and requested draw count are otherwise shared across the arms.

These runs use the active-position transport implementation published in
[WarmupHMC `6b377cb`](https://github.com/nsiccha/WarmupHMC.jl/commit/6b377cb23934022af5879d199a7c57abfac54c70).
When centering changes, the active position and the adaptation sample now
represent the same physical points before and after the change.

WarmupHMC returns **model coordinates** in `posterior_position`; checkpoints
retain sampler coordinates. Export checks their mapping, Jacobian-adjusted
density and saved gradients before making figures or scientific summaries.
""" + image(case, "centeredness", "Centering selected with both losses, post-hoc and online")

def efficiency(case, quantities):
    with (RESULTS / case / "comparison.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    table = (RESULTS / case / "table.md").read_text()
    divs = "; ".join(f'{r["method"]}: {r["divergences"]}' for r in rows)
    return f"""
## Sampling efficiency and full workflow cost

Each completed arm has one chain, seed 1 and 10,000 retained draws. Every row
uses the same scientific quantities: **{quantities}**. Standardized effects
are excluded from the minimum. Positive scales may be stored as logs;
rank-normalized bulk ESS is invariant under that monotone change.

{table}

Both efficiency columns are relative to this study's **NCP + WarmupHMC**
baseline. Sampling efficiency is minimum bulk ESS divided by retained-sampling
gradient calls. Total efficiency divides that same minimum ESS by the full
workflow's gradient calls. The total includes initialization, all warmup and
adaptation, active-state reevaluations, and sampling. For each post-hoc row it
also includes the entire NCP pilot; the pilot's ESS is not added to the refit's.

Gradient counts measure target evaluations, a proxy for compute cost rather
than a wall-clock speed ratio. Compilation, plotting and independent audits
are outside the fitting counts. The complete numerical summaries, including
absolute ESS and both denominators, are in the linked result files.

Sampling divergences: **{divs}**. These are one-chain comparisons, so neither
the ranking nor a within-chain split R-hat establishes cross-chain convergence.
""" + image(case, "efficiency", "Full gradient cost and the two relative sampling efficiencies")

def geometry(case, groups):
    text = """
## Geometry of the fitted coordinates

The left column is always the **centered visualization baseline**, obtained
from the NCP pilot. The pilot comparison uses those same draws in CP and NCP.
The post-hoc and online panels each show their two newly fitted loss variants
in the coordinates actually used by the sampler. CP is the visual reference;
NCP remains the efficiency baseline. Axes are independent across panels.
"""
    for slug, label in groups:
        text += image(case, "pairs-pilot-" + slug, label + ": CP and NCP pilot coordinates")
        text += image(case, "pairs-posthoc-" + slug, label + ": CP reference and both post-hoc losses")
        text += image(case, "pairs-online-" + slug, label + ": CP reference and completed online losses")
    return text

def gradients(case, groups):
    text = """
## Position and gradient in the displayed coordinates

These panels pair each displayed effect coordinate with its own log-density
gradient, using 1,000 evenly spaced retained draws. The CP reference transforms
the pilot's positions and gradients together. The fitted panels use gradients
saved in their actual sampler frame; they do not attach an NCP gradient to a
centered position.
"""
    for slug, label in groups:
        for family in ("posthoc", "online"):
            text += image(case, f"gradients-{family}-{slug}", f"{label}: positions and gradients, {family}")
    return text

def reproduce(case):
    return f"""
## Reproduce and inspect

The [refresh harness](https://github.com/nsiccha/BayesianRegressionModels.jl/tree/ns/devibe/research/centering_refresh)
contains the driver, both loss selectors, saved-frame audits, export and AoV
plotting code. Its [results for this study](https://github.com/nsiccha/BayesianRegressionModels.jl/tree/ns/devibe/research/centering_refresh/results/{case})
contain the full efficiency denominators, per-quantity ESS and selected controls.
The original model directory retains the source specification and independent
density/gradient audit. The refresh uses those same model definitions.

Run `run.jl {case} ncp OUTPUT` first, then request
`cp,posthoc_position,posthoc_gradient,online_position,online_gradient` with the
same output root. Completed arm directories are immutable. See the harness
README for the environment and full commands.
"""

# Preserve the executable model and manual fully-centered example.
old = original("eight-schools-centering")
text = old.split("[![Centered and noncentered")[0]
text = text.replace("all four fits", "all six fits")
start = text.index("coordinate Jacobian. At 16")
end = text.index("\n\n## Choose", start)
text = text[:start] + "coordinate Jacobian. The independent audit is retained with the source model." + text[end:]
text += r"""
## Posterior effects and predictive checks

Thin intervals contain 90% of draws and thick intervals 50%. Each school is
a category with its own interval. Treatment effects are $\theta_j=\mu+\tau z_j$.
""" + image("eight", "posterior-effects", "School treatment effects from NCP and CP fits")
text += """
The predictive check uses the NCP fit and includes the known standard error
of each reported estimate. Red points are the observations in their original
school order. There are no ribbons between categorical schools.
""" + image("eight", "ppc", "Replicated-estimate intervals and observed school estimates")
text += "## Select centering from a pilot\n\n" + old.split("## Select centering from a pilot\n\n")[1].split("[![Offline centering")[0]
text += common("eight")
text += "\nBoth criteria favor coordinates close to noncentering in this weakly informed hierarchy.\n"
text += efficiency("eight", "population mean, group SD and eight school treatment effects (10 quantities)")
text += "\nThe position-loss post-hoc refit modestly improves sampling efficiency here, but its pilot makes total efficiency lower than NCP. Full centering has 39 divergences and poor efficiency; its intervals require that qualification.\n"
text += geometry("eight", [("school-effects", "School effects")])
text += gradients("eight", [("school-effects", "School effects")]) + reproduce("eight")
(BASE / "docs/src/eight-schools-centering.md").write_text(text)

old = original("radon-centering")
text = old.split("## Compare centered and noncentered geometry")[0]
text = text.replace("assets/adaptive-radon/data-ppc.png", "assets/centering-refresh-radon/ppc.png")
text += """
## Select one centering per county effect

Each county deviation has coordinates $u=s^c z$. Its population coefficient
remains separate. The following position-loss selection freezes an independent
control for every county intercept and slope:

""" + old.split("```julia\nusing Enzyme, BridgeStan")[1].split("\n```", 1)[0].join(["```julia\nusing Enzyme, BridgeStan", "\n```\n"])
text += common("radon")
text += efficiency("radon", "two population coefficients, two group SDs, residual SD, and all 386 county intercept totals and 386 slope totals (777 quantities)")
text += "\nThe population floor slope limits minimum ESS in every arm. A more favorable local effect geometry does not necessarily improve this global bottleneck. In this run, neither online loss beats NCP in total efficiency.\n"
text += """
For each random-effect role, the scatter rows select the minimum post-hoc
position-loss centeredness, the value nearest `0.5`, and the maximum. Ties use
the lowest county index, with distinct counties in the three rows. These same
coordinates are used throughout. The selection and full fits include all
386 counties; the selected county labels appear in the figures and exported
coordinate table.
"""
groups = [("county-intercepts", "County intercepts"), ("county-slopes", "County slopes")]
text += geometry("radon", groups) + gradients("radon", groups) + reproduce("radon")
(BASE / "docs/src/radon-centering.md").write_text(text)

old = original("adaptive-centering")
text = old.split("## 1. Fit the noncentered model")[0]
text = text.replace("The source workflow has two fits: a noncentered pilot and a selected-partial\nrefit. The centered plots transform the pilot draws. A third fit extends the\ncomparison with online adaptation during warmup.", "The source workflow has a noncentered pilot and a selected-partial refit.\nThis extension compares both adaptation losses, post-hoc and online, alongside\nfixed CP and NCP controls. Centered geometry remains the visual reference.")
text += """
## Fit the noncentered model

The mean and log residual-SD GPs each use 20 basis weights. NCP samples
standardized weights; CP samples physical weights. If $s_j$ is the spectral
SD of basis $j$, intermediate coordinates are $u_j=s_j^{c_j}z_j$.
The log spectral scale depends on both GP hyperparameters.

```julia
using Random, WarmupHMC
sb = SBBRMI(adaptive_motorcycle_model(); mod=@__MODULE__)
problem = StanBlocks.stan_instantiate(sb.model)
pilot = WarmupHMC.adaptive_warmup_mcmc(
    Xoshiro(1), problem; n_draws=10_000, monitor_ess=true)
```

The posterior function plot uses the native BRM prediction contract. The
left panel is the conditional mean; the right is the conditional residual SD.
The ribbons summarize continuous functions of time, rather than categorical
effects or replicated observations.
""" + image("hsgp", "posterior-ncp", "NCP mean and conditional-noise posterior functions")
text += common("hsgp")
text += """
## Fit with selected coordinates

All refreshed arms wrap the **same compiled noncentered Stan target**. The
post-hoc fits fix the selected centering controls; the online fits adapt them
during warmup. This keeps the target implementation shared across the matrix.
The original reproduction directory also demonstrates compiling selected
partial coordinates directly into a BRM model.
""" + image("hsgp", "posterior-post-hoc-position", "Post-hoc position-loss posterior functions")
text += image("hsgp", "posterior-post-hoc-gradient", "Post-hoc gradient-loss posterior functions")
text += image("hsgp", "posterior-online-position", "Online position-loss posterior functions")
text += image("hsgp", "posterior-online-gradient", "Online gradient-loss posterior functions")
text += efficiency("hsgp", "four GP hyperparameters, plus mean and conditional SD at each of the 94 distinct observed times (192 quantities)")
text += """
The remaining divergences are material: these runs do not establish reliable
performance rankings for this broad-prior HSGP. Full centering is particularly
poor. Both online losses improve total efficiency over NCP in this run. The
post-hoc arms have a much cheaper sampling phase, but must amortize a large pilot.

The online gradient arm uses an additional numerical admissibility check in
[WarmupHMC `7aed40b`](https://github.com/nsiccha/WarmupHMC.jl/commit/7aed40b18bd4cdabb75330f285d5c9b355575ab9):
if a proposed centering cannot represent the stored adaptation points and
gradients with finite values, it keeps the previous coordinates and state.
One update was rejected in this run. The remaining arms use the same active-state
transport implementation without encountering this representability limit.
"""
groups = [("mean-gp", "Mean GP"), ("log-sd-gp", "Log-SD GP")]
text += geometry("hsgp", groups)
text += r"""
Rows show frequencies 1, 2, 19 and 20 against each GP hyperparameter: $\rho$
is the length scale and $\sigma$ the marginal SD. The
display zooms to the central 97.5% extent for readability; no observations are
deleted from the underlying scatter data or diagnostics. Display limits are
recorded alongside the figures. Each online loss has its own fitted column.
"""
text += gradients("hsgp", groups)
text += "\n### Separate backend gradient comparison\n\n" + old.split("### Backend gradient comparison\n\n")[1].split("## Compute cost")[0]
text += reproduce("hsgp")
(BASE / "docs/src/adaptive-centering.md").write_text(text)
print("REFRESH_PAGES_COMPLETE")
