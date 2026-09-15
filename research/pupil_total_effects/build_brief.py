"""Build the shareable narrative from the measured matrix, without refitting."""
import csv
import gzip
import json
from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parent
rows = list(csv.DictReader((root / "results/student_mixture/matrix/all_scopes.tsv").open(), delimiter="\t"))


def relative_efficiency(row, metric):
    baseline = [r for r in rows if r["scope"] == row["scope"]
                and r["model"] == "brms NCP" and r["sampler"] == "Native Stan"]
    assert len(baseline) == 1 and float(baseline[0][metric]) > 0
    return float(row[metric]) / float(baseline[0][metric])


def table(scope):
    lines = [
        "| Model | Sampler | Total gradients | Relative sampling efficiency | Relative total efficiency |",
        "|---|---|---:|---:|---:|",
    ]
    selected = [r for r in rows if r["scope"] == scope]
    assert len(selected) == 12
    for r in selected:
        label = r["model"].replace("total coefficients", "Totals")
        lines.append(f'| {label} | {r["sampler"]} | {int(r["total_gradients"]):,} | '
                     f'{relative_efficiency(r, "min_ess_per_sampling_gradient"):.4g}× | '
                     f'{relative_efficiency(r, "min_ess_per_total_gradient"):.4g}× |')
    return "\n".join(lines)


source_sha = sys.argv[1]
assert len(source_sha) >= 7 and all(c in "0123456789abcdef" for c in source_sha)
body = (root / "aki_brief_template.md").read_text()
body = body.replace("@@PRIMARY@@", table("common44"))
body = body.replace("@@RECOVERED@@", table("original_physical46"))
body = body.replace("@@SOURCE_SHA@@", source_sha)
for token, model, metric in [
    ("ACP_SAMPLING", "total coefficients ACP", "min_ess_per_sampling_gradient"),
    ("CP_SAMPLING", "total coefficients CP", "min_ess_per_sampling_gradient"),
    ("ACP_TOTAL", "total coefficients ACP", "min_ess_per_total_gradient"),
    ("CP_TOTAL", "total coefficients CP", "min_ess_per_total_gradient"),
    ("S2Z_AUTO_TOTAL", "brms S2Z auto", "min_ess_per_total_gradient"),
]:
    match = [r for r in rows if r["scope"] == "common44" and r["model"] == model and r["sampler"] == "WHMC"]
    assert len(match) == 1
    body = body.replace("@@" + token + "@@", f"{relative_efficiency(match[0], metric):.1f}×")
assert "@@" not in body
spec = json.loads(gzip.decompress((root / "results/student_mixture/pairs.aov.json.gz").read_bytes()))
envelope = {
    "schema": "kb-aov/v1",
    "title": "Total-coefficient posterior geometry: CP, NCP and offline ACP",
    "alt": "The same 2000 retained partial-refit draws in each column. Rows select distinct coordinates by minimum centeredness, nearest 0.5, and maximum centeredness. A is an intercept total; B is a slope total.",
    "spec": spec,
    "provenance": {
        "mode": "preliminary", "producer": "BayesianRegressionModels:docs:adaptive-centering",
        "base_commit": "e51ee53", "run": "pupil-student-total-v1",
        "references": [
            {"kind": "data", "label": "Pinned pupil source data",
             "url": "https://github.com/bnicenboim/bcogsci/blob/d90fc01e6f6fcdced7ee64c9d2ed607d212ec77c/data/df_pupil_complete.rda",
             "commit": "d90fc01e6f6fcdced7ee64c9d2ed607d212ec77c",
             "path": "data/df_pupil_complete.rda",
             "sha256": "ab45331f4d2be447211832bd6cf13501b31032837ddb16acada6d414fbf46042"},
            {"kind": "spec", "label": "Producer repository; research artifacts at " + source_sha,
             "url": "https://github.com/nsiccha/BayesianRegressionModels.jl"},
        ],
    },
}
body, count = re.subn(r"!\[([^\n]+)\]\(results/student_mixture/pairs.png\)",
                      lambda m: "```kb-aov\n" + json.dumps(envelope, separators=(",", ":")) + "\n```\n\n" + m[1], body)
assert count == 1
output = root / "aki-pupil.md"
output.write_text(body)
print(output)
