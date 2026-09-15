"""Build the docs page and the KB brief from the same measured tables and text."""
from pathlib import Path
from urllib.parse import urlencode
import json
import re
import shutil
import sys

root = Path(__file__).resolve().parent
repo = root.parent.parent
body = (root / "page_template.md").read_text()
body = body.replace("{{RESULT}}", "For **regional intercepts**, S2Z CP + WHMC gave 279× the native-NCP baseline's total-gradient efficiency; fixed BRM totals gave 166× and both online total variants 128×. With **independent intercepts and slopes**, fixed total CP gave 179×, online gradient 140× and S2Z CP + WHMC 95.5×. Thus fixed total CP was about 1.87× as efficient as S2Z CP + WHMC in the latter pilot, while S2Z CP was about 1.68× as efficient as total CP in the intercept-only pilot.")
body = body.replace("{{VALIDATION}}", "The intercept-only ordinary NCP fits had 6 native and 68 WHMC divergences; S2Z NCP had 35 and 22, and total NCP had six. Other intercept-only arms had zero. Ordinary and S2Z NCP under WHMC also show the largest mean discrepancies from total CP: 4.52 and 5.89 combined estimated MCSEs for group SD. Their within-chain split R-hat maxima are 1.037 and 1.043. These weak arms need longer or replicated convergence checks.\n\nFor independent intercepts and slopes, ordinary NCP had 20 native and six WHMC divergences, S2Z NCP 37 and 60, total NCP eleven, and post-hoc gradient two. All other arms had zero. Native ordinary NCP hit maximum tree depth on 1,375 of 2,000 retained transitions. Mean differences from total CP are below three combined MCSEs except two S2Z-auto WHMC quantities, with a maximum of 3.56. These are descriptive checks across many quantities, not proof of convergence.\n\nRemoving the recovered population coefficients leaves nine invariant quantities for the intercept-only model and 15 for the independent model. The latter still gives 169× total efficiency for total CP, 132× for online gradient and 90.4× for S2Z CP + WHMC against that baseline scope. All twelve total-model minimum ESS values are unchanged across ten recovery seeds.")

for key, case in (("INTERCEPT", "cluster-intercept"), ("INDEPENDENT", "cluster-independent")):
    results = root / "results" / case
    assert json.loads((results / "status.json").read_text())["completed"] == 14
    body = body.replace("{{" + key + "_TABLE}}", (results / "table.md").read_text().strip())

links = [("BRM models, priors and independent reference density", "model.jl"),
         ("Density, gradient and recovery audit", "audit.jl"),
         ("Six total fits and ordinary WHMC baseline", "run.jl"),
         ("Native Stan driver", "native.R"), ("S2Z WHMC and analysis", "brms_whmc.jl"),
         ("S2Z coordinate and generated-recovery audit", "s2z.jl"),
         ("Saved-draw validation", "validate_saved.jl"), ("Numerical proposal audit", "numerical_audit.jl"),
         ("Full raw-fit archive manifest", "results/fits/manifest.json"), ("Reproduction instructions", "README.md")]
for case in ("cluster-intercept", "cluster-independent"):
    for file, label in (("comparison.tsv", "comparison table"), ("per_quantity_mcse.tsv", "per-quantity ESS and MCSE"),
                        ("recovery_seed_sensitivity.tsv", "recovery sensitivity")):
        links.append((case + ": " + label, f"results/{case}/{file}"))
    for arm in ("ordinary_ncp", "s2z_cp", "s2z_ncp", "s2z_auto"):
        links.extend([(f"{case}, {arm}: {label}", f"reference/native/{case}/{arm}/{file}")
                      for label, file in (("Stan", "clean.stan"), ("data", "resolved-data.json"), ("gradient counts", "gradient_counts.tsv"))])
for hierarchy in ("intercept_only", "independent"):
    links.extend([(f"Automatic {hierarchy}: {label}", f"reference/automatic_totals/{hierarchy}/{file}")
                  for label, file in (("Stan", "automatic_totals.stan"), ("data", "automatic_totals.json"))])

docs, brief = body, body
for key, case in (("INTERCEPT", "cluster-intercept"), ("INDEPENDENT", "cluster-independent")):
    results = root / "results" / case
    for suffix, name, description in (("EFFICIENCY", "efficiency", "Total gradient costs and relative scientific-QOI efficiencies"),
                                      ("TOTAL_PAIRS", "total_pairs", "The same 2,000 total-coefficient draws in CP, NCP and partial coordinates"),
                                      ("S2Z_PAIRS", "s2z_pairs", "The same 2,000 S2Z-auto draws in centered, noncentered and auto contrast coordinates")):
        token = "{{" + key + "_" + suffix + "}}"
        asset = repo / "docs/src/assets/air-centering" / f"{case}-{name}.png"
        asset.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(results / f"{name}.png", asset)
        path = f"assets/air-centering/{asset.name}"
        docs = docs.replace(token, f"[![{description}.]({path})]({path})")
        figure = dict(schema="kb-aov/v1", title=f"AIR {case}: {description}", alt=description,
                      spec=json.loads((results / f"{name}.aov.json").read_text()),
                      provenance=dict(mode="preliminary", producer="BayesianRegressionModels:docs:adaptive-centering",
                                      base_commit="5f30e53e62fd1ac3bb43675c40eb0ab9b97f0945", run=f"air-{case}-v1", references=[dict(kind="data", label="Source model discussion", url="https://discourse.mc-stan.org/t/help-testing-brms-pr-for-sum-to-zero-and-partial-centering/41542")]))
        brief = brief.replace(token, "```kb-aov\n" + json.dumps(figure, separators=(",", ":")) + "\n```")

public = "https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/air_total_effects/"
docs = docs.replace("{{LINKS}}", "\n".join(f"- [{label}]({public}{path})." for label, path in links))
brief = brief.replace("{{LINKS}}", "\n".join(f"- [{label}](/code?{urlencode({'path':str(root/path), 'host':'strato2'})})." for label, path in links))
source = (root / "model.jl").read_text()
for which, marker, finish in (("air_intercept", "const INTERCEPT =", "\nconst INDEPENDENT"),
                              ("air_independent", "const INDEPENDENT =", "\nfunction load_data")):
    code = source[source.index(marker):source.index(finish, source.index(marker))].strip()
    pattern = r"```@eval\nMain\.BRMDocsComparisons\.comparison\(@__MODULE__,\n    Main\.BRMCenteringExamples\.authoring\(:" + which + r"\),.*?\n```"
    brief, count = re.subn(pattern, lambda _: "```julia\n" + code + "\n```", brief, flags=re.S)
    assert count == 1
brief = brief.replace("The authoring panes below read the exact fitted declarations. The backend views are regenerated during the docs build. Sampling uses StanBlocks/BridgeStan and WHMC; the generated Turing pane is for inspection only.", "These are the exact fitted BRM declarations. The linked source includes data construction, generated Stan and all sampling and diagnostic code.")
brief = brief.replace("sb = SBBRMI(air_independent_brm_model(); mod=@__MODULE__)", '''reference = JSON.parsefile("research/air_total_effects/reference/cluster_region/independent/ordinary_ncp.json")
data = (;log_pm25=Float64.(reference["Y"]),
         log_sat=Float64.(getindex.(reference["X"], 2)),
         region=Int.(reference["J_1"]))
sb = SBBRMI(INDEPENDENT(data); mod=@__MODULE__)''')
brief = brief.replace("using StanBlocks, BridgeStan, WarmupHMC, Enzyme, Random", "using BayesianRegressionModels, Distributions, JSON\nusing StanBlocks, BridgeStan, WarmupHMC, Enzyme, Random")
assert "{{" not in docs and "{{" not in brief
(repo / "docs/src/air-centering.md").write_text(docs)
Path(sys.argv[1]).write_text(brief)
print("AIR_DOCS_AND_BRIEF_BUILT", len(brief.encode()), "bytes; 2 tables, 6 AoV figures", flush=True)
