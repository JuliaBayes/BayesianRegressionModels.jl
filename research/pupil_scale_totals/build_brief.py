"""Produce the KB brief from the single comparison table and AoV exports."""
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
results = root / "results"
body = (root / "brief_template.md").read_text()
body = body.replace("{{TABLE}}", (results / "table.md").read_text().strip())

def figure(name, title, alt):
    envelope = {
        "schema": "kb-aov/v1", "title": title, "alt": alt,
        "spec": json.loads((results / name).read_text()),
        "provenance": {
            "mode": "preliminary",
            "producer": "BayesianRegressionModels:docs:adaptive-centering",
            "base_commit": "54cbe3fae68cd60f1026d74f4b0fdbb74eb76aec",
            "run": "pupil4-15-arm-v1",
            "references": [{"kind": "spec", "label": "BRM total-coefficient implementation",
                "url": "https://github.com/nsiccha/BayesianRegressionModels.jl/tree/5f30e53e62fd1ac3bb43675c40eb0ab9b97f0945"},
                {"kind": "data", "label": "Source pupil model and data link",
                 "url": "https://discourse.mc-stan.org/t/help-testing-brms-pr-for-sum-to-zero-and-partial-centering/41542/4"}]
        }
    }
    return "```kb-aov\n" + json.dumps(envelope, separators=(",", ":"), ensure_ascii=False) + "\n```"

for token, name, title, alt in (
    ("EFFICIENCY_PLOT", "efficiency.aov.json", "Pupil: 15 matched sampler and parametrization arms", "Total gradients, relative sampling efficiency and relative total efficiency for 66 common scientific quantities."),
    ("TOTAL_PAIRS", "total_pairs.aov.json", "BRM totals: CP, NCP and adaptive coordinates", "The same 2000 posterior draws in each column, for total coordinates selected at low, middle and high inferred centeredness."),
    ("S2Z_PAIRS", "s2z_pairs.aov.json", "brms S2Z: CP, NCP and auto coordinates", "The same 2000 S2Z-auto WHMC draws represented as centered, noncentered and partially centered contrasts."),
):
    body = body.replace("{{" + token + "}}", figure(name, title, alt))

def link(label, path):
    return f"[{label}]({root / path})"

groups = [
    ("BRM formula, automatic target and independent audit", [("model and priors", "common.jl"), ("density/gradient/recovery audit", "audit.jl"), ("generated Stan", "reference/automatic_totals/automatic_totals.stan"), ("exact data", "reference/automatic_totals/automatic_totals.json")]),
    ("WHMC adaptation and diagnostics", [("six total arms", "total_whmc.jl"), ("ordinary and S2Z arms", "brms_whmc.jl"), ("S2Z transformation/audit", "s2z.jl"), ("common scientific quantities", "diagnostics.jl")]),
    ("Native Stan and gradient counting", [("native driver", "native.R"), ("native support", "support/native_tools.R"), ("C++ counter", "support/native_gradient_counter.hpp"), ("WHMC counter wrapper", "support/stan_target.jl")]),
]
for arm, label in (("ordinary_ncp", "Ordinary brms NCP"), ("s2z_cp", "S2Z CP"), ("s2z_ncp", "S2Z NCP"), ("s2z_auto", "S2Z auto")):
    p = "reference/native/" + arm + "/"
    groups.append((label, [("clean Stan", p+"clean.stan"), ("counted Stan", p+"instrumented.stan"), ("resolved data", p+"resolved-data.json"), ("initial values", p+"init.json"), ("gradient counts", p+"gradient_counts.tsv")]))
groups += [
    ("Ordinary brms CP", [("generated Stan", "reference/ordinary_cp.stan"), ("data", "reference/ordinary_cp.json")]),
    ("Numerical results", [("single comparison table", "results/comparison.tsv"), ("per-quantity ESS, means and MCSE", "results/per_quantity_mcse.tsv"), ("posterior agreement and invariant-QOI checks", "results/saved_draw_checks.tsv"), ("recovery-seed sensitivity", "results/recovery_seed_sensitivity.tsv")]),
    ("Saved-draw figures", [("total coordinate preparation", "prepare_total_pairs.jl"), ("S2Z coordinate preparation", "prepare_s2z_pairs.jl"), ("AoV plotting", "plot.jl")]),
    ("Inputs and reproducibility", [("original-order data", "reference/pupil.csv"), ("study README", "README.md"), ("file hashes", "results/source_manifest.json")]),
]
links = "These links open the saved text and generated models for inspection without rerunning the experiment.\n\n"
links += "\n".join("- **" + label + ":** " + ", ".join(link(name, path) for name, path in files) + "." for label, files in groups)
body = body.replace("{{SOURCE_LINKS}}", links)
assert "{{" not in body
Path(sys.argv[1]).write_text(body)
print(f"BRIEF_BUILT bytes={len(body.encode())} tables={body.count('| Method |')} figures={body.count('```kb-aov')}")
