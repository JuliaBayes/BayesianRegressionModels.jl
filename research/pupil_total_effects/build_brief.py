"""Build the shareable narrative from the measured matrix, without refitting."""
import csv
import gzip
import json
from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parent
rows = list(csv.DictReader((root / "results/online_adaptation/brief_matrix/qoi46.tsv").open(), delimiter="\t"))


def relative_efficiency(row, metric):
    baseline = [r for r in rows if r["model"] == "brms NCP" and r["sampler"] == "Native Stan"]
    assert len(baseline) == 1 and float(baseline[0][metric]) > 0
    return float(row[metric]) / float(baseline[0][metric])


def table():
    lines = [
        "| Model | Sampler | Total gradients | Relative sampling efficiency | Relative total efficiency |",
        "|---|---|---:|---:|---:|",
    ]
    selected = rows
    assert len(selected) == 15
    for r in selected:
        label = r["model"].replace("total coefficients", "Totals")
        lines.append(f'| {label} | {r["sampler"]} | {int(r["total_gradients"]):,} | '
                     f'{relative_efficiency(r, "sampling_efficiency"):.3g}× | '
                     f'{relative_efficiency(r, "total_efficiency"):.3g}× |')
    return "\n".join(lines)


source_sha = sys.argv[1]
assert len(source_sha) >= 7 and all(c in "0123456789abcdef" for c in source_sha)
body = (root / "aki_brief_template.md").read_text()
body = body.replace("@@PRIMARY@@", table())
links = (root / "results/student_mixture/qoi46/inspectable_sources.md").read_text()
links = links.replace("Complete 46-QOI efficiency table (TSV)", "Original 12-arm 46-QOI table (archived TSV)")
body = body.replace("@@CODE_LINKS@@", links)
body = body.replace("@@SOURCE_SHA@@", source_sha)
for token, model, metric in [
    ("ACP_SAMPLING", "Totals post-hoc position", "sampling_efficiency"),
    ("CP_SAMPLING", "total coefficients CP", "sampling_efficiency"),
    ("ACP_TOTAL", "Totals post-hoc position", "total_efficiency"),
    ("CP_TOTAL", "total coefficients CP", "total_efficiency"),
    ("S2Z_AUTO_TOTAL", "brms S2Z auto", "total_efficiency"),
]:
    match = [r for r in rows if r["model"] == model and r["sampler"] == "WHMC"]
    assert len(match) == 1
    body = body.replace("@@" + token + "@@", f"{relative_efficiency(match[0], metric):.1f}×")
assert "@@" not in body
def envelope(name, title, caption):
    path = root / ("results/student_mixture/" + name + ".aov.json.gz")
    if name == "qoi46/efficiency":
        path = root / "results/online_adaptation/brief_matrix/efficiency.aov.json.gz"
    spec = json.loads(gzip.decompress(path.read_bytes()))
    return {
    "schema": "kb-aov/v1",
    "title": title,
    "alt": caption,
    "spec": spec,
    "provenance": {
        "mode": "preliminary", "producer": "BayesianRegressionModels:docs:adaptive-centering",
        "base_commit": source_sha,
        "run": "pupil-online-brief-matrix-v1" if name == "qoi46/efficiency" else "pupil-qoi46-plots-v1",
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
titles = {
    "pairs": "Total-coefficient posterior geometry: CP, NCP and offline ACP",
    "qoi46/s2z_pairs": "brms S2Z posterior geometry: CP, NCP and auto",
    "qoi46/efficiency": "The same 46 scientific quantities: cost and relative efficiency",
}

def embed(match):
    caption, name = match.groups()
    payload = envelope(name, titles[name], caption)
    return "```kb-aov\n" + json.dumps(payload, separators=(",", ":")) + "\n```\n\n" + caption

body, count = re.subn(r"!\[([^\n]+)\]\(results/student_mixture/([^\n]+).png\)", embed, body)
assert count == 3
output = root / "aki-pupil.md"
output.write_text(body)
print(output)
