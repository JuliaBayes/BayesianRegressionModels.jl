"""Build the docs page and the KB brief from the same measured tables and text: build_pages.py BRIEF_OUT"""
from pathlib import Path
from urllib.parse import urlencode
import json
import re
import shutil
import sys

root = Path(__file__).resolve().parent
repo = root.parent.parent
text = json.loads((root / "results" / "text.json").read_text())   # RESULT and VALIDATION paragraphs, written after analysis
body = (root / "page_template.md").read_text()
body = body.replace("{{RESULT}}", text["result"]).replace("{{VALIDATION}}", text["validation"])

cases = (("AS", "AS"), ("CROHN", "crohn"))
for key, case in cases:
    results = root / "results" / case
    status = json.loads((results / "status.json").read_text())
    assert status["completed"] == status["planned"], status
    body = body.replace("{{" + key + "_TABLE}}", (results / "table.md").read_text().strip())

links = [("BRM models, priors and reference densities", "model.jl"),
         ("Density, gradient, wrapper and RBesT cross-check audit", "audit.jl"),
         ("Twelve WarmupHMC arms", "run.jl"), ("RBesT data and program capture", "capture.R"),
         ("Native RBesT arms under CmdStan", "native.R"), ("Native-arm diagnostics", "native_analysis.jl"),
         ("Recovery sensitivity and MCSE checks", "complete.jl"), ("Saved-draw pairs", "pairs.jl"),
         ("Full raw-fit archive manifest", "results/fits/manifest.json"), ("Primary-source model check and reproduction", "README.md")]
for _, case in cases:
    for file, label in (("comparison.tsv", "comparison table"), ("per_quantity_mcse.tsv", "per-quantity MCSE"),
                        ("recovery_seed_sensitivity.tsv", "recovery sensitivity"), ("mcse_summary.tsv", "MCSE summary")):
        links.append((case + ": " + label, f"results/{case}/{file}"))
    for variant in ("legacy", "s2z"):
        links.extend([(f"{case}, RBesT {variant}: {label}", f"reference/native/{case}/{variant}/{file}")
                      for label, file in (("Stan", "clean.stan"), ("data (NCP)", "data-ncp.json"), ("data (CP)", "data-cp.json"))])
    for arm in ("rbest_ncp", "rbest_cp", "s2z_ncp", "s2z_cp", "stan_ncp", "stan_cp"):
        links.append((f"{case}, {arm}: gradient counts", f"reference/native/{case}/{arm}/gradient_counts.tsv"))
    links.append((f"{case}: BRM exact-totals Stan", f"reference/automatic_totals/{case}/automatic_totals.stan"))
    links.append((f"{case}: BRM conventional Stan", f"reference/automatic_totals/{case}/ordinary_ncp.stan"))

docs, brief = body, body
for key, case in cases:
    results = root / "results" / case
    for suffix, name, description in (("EFFICIENCY", "efficiency", "Total gradient costs and relative scientific-QOI efficiencies"),
                                      ("TOTAL_PAIRS", "total_pairs", "The same 10,000 total-coefficient draws in CP, NCP and partial coordinates"),
                                      ("ORDINARY_PAIRS", "ordinary_pairs", "The same 10,000 conventional-block draws in CP, NCP and partial coordinates")):
        token = "{{" + key + "_" + suffix + "}}"
        asset = repo / "docs/src/assets/rbest-centering" / f"{case}-{name}.png"
        asset.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(results / f"{name}.png", asset)
        path = f"assets/rbest-centering/{asset.name}"
        docs = docs.replace(token, f"[![{description}.]({path})]({path})")
        figure = dict(schema="kb-aov/v1", title=f"RBesT {case}: {description}", alt=description,
                      spec=json.loads((results / f"{name}.aov.json").read_text()),
                      provenance=dict(mode="preliminary", producer="BayesianRegressionModels:docs:adaptive-centering",
                                      base_commit=text["base_commit"], run=f"rbest-{case}-v1",
                                      references=[dict(kind="data", label="RBesT", url="https://github.com/Novartis/RBesT"),
                                                  dict(kind="data", label="Weber, Discourse post 32", url="https://discourse.mc-stan.org/t/help-testing-brms-pr-for-sum-to-zero-and-partial-centering/41542/32")]))
        brief = brief.replace(token, "```kb-aov\n" + json.dumps(figure, separators=(",", ":")) + "\n```")

public = "https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/rbest_centering/"
docs = docs.replace("{{LINKS}}", "\n".join(f"- [{label}]({public}{path})." for label, path in links))
brief = brief.replace("{{LINKS}}", "\n".join(f"- [{label}](/code?{urlencode({'path': str(root / path), 'host': 'strato2'})})." for label, path in links))
source = (root / "model.jl").read_text()
for which, marker, finish in (("rbest_as", "const AS_MODEL =", "\n# gMAP with the documented crohn"),
                              ("rbest_crohn", "const CROHN_MODEL =", "\nfunction read_dataset")):
    code = source[source.index(marker):source.index(finish, source.index(marker))].strip()
    pattern = r"```@eval\nMain\.BRMDocsComparisons\.comparison\(@__MODULE__,\n    Main\.BRMCenteringExamples\.authoring\(:" + which + r"\),.*?\n```"
    brief, count = re.subn(pattern, lambda _: "```julia\n" + code + "\n```", brief, flags=re.S)
    assert count == 1
brief = brief.replace("The authoring panes below read the exact fitted declarations. The backend views are regenerated during the docs build. Sampling uses StanBlocks/BridgeStan and WHMC; the generated Turing pane is for inspection only.",
                      "These are the exact fitted BRM declarations. The linked source includes data construction, generated Stan and all sampling and diagnostic code.")
brief = brief.replace("sb = SBBRMI(rbest_as_brm_model(); mod=@__MODULE__)          # total_groups=:auto",
                      'AS = read_dataset("AS")   # research/rbest_centering/reference/datasets/AS.tsv\nsb = SBBRMI(AS_MODEL((; study=1:8, n=AS.n, r=AS.r)); mod=@__MODULE__)   # total_groups=:auto')
assert "{{" not in docs and "{{" not in brief
(repo / "docs/src/rbest-centering.md").write_text(docs)
Path(sys.argv[1]).write_text(brief)
print("RBEST_DOCS_AND_BRIEF_BUILT", len(brief.encode()), "bytes; 2 tables, 6 AoV figures", flush=True)
