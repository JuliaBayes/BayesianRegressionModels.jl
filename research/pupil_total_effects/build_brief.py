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

if "--docs" in sys.argv:
    import shutil
    repo = root.parents[1]
    body = body.replace("Prepared for discussion with Aki · 15 September 2026", "Case study · 15 September 2026")
    body = body.replace("observed sampling and total efficiency**, after repairing a WHMC bug in the\ntransport of the active sampler position during centering changes.", "observed sampling and total efficiency**.")
    body = body.replace("The three added arms (post-hoc gradient and both online losses) use the\n  tested local active-state transport fix", "The post-hoc gradient and both online arms use the\n  active-state transport implementation")
    body = body.replace("The three added Student-t arms use local fix", "The post-hoc gradient and both online Student-t arms use")
    start = body.index("### Online adaptation correction")
    end = body.index("Full draws, native CSVs", start)
    body = body[:start] + """### Online adaptation implementation

The online arms preserve the physical active position when the centering
coordinates change and reevaluate its density and gradient in the new frame.
The online Student-t runs use WarmupHMC's implementation published as
[`6b377cb`](https://github.com/nsiccha/WarmupHMC.jl/commit/6b377cb23934022af5879d199a7c57abfac54c70).
Both online losses have zero divergences in 2,000 retained Student-t draws.
Separate Gaussian sensitivity fits also have zero divergences with both losses.

""" + body[end:]
    body = body.split("## 9. Inspect the harness and exact generated Stan files")[0]
    base = "https://github.com/nsiccha/BayesianRegressionModels.jl/blob/ns/devibe/research/pupil_total_effects/"
    def link(label, path):
        assert (root / path).is_file(), path
        return f"[{label}]({base}{path})"
    groups = [
        ("Model, sampling and independent checks", [("manual total-coefficient target", "model.jl"), ("independent density/gradient audit", "audit.jl"), ("NCP pilot and post-hoc position refit", "run.jl"), ("fixed CP", "fixed_cp.jl")]),
        ("Adaptation and recovery", [("online experiment", "online_transport_trial.jl"), ("post-hoc gradient selection", "offline_gradient_trial.jl"), ("conditional recovery", "recovery.jl"), ("46-QOI comparison for the added arms", "compare_online_trials.jl")]),
        ("brms and native Stan harness", [("ordinary brms + WHMC", "brms_baseline.jl"), ("S2Z preparation/native sampling", "s2z_native.R"), ("S2Z + WHMC", "s2z_whmc.jl"), ("native gradient counter", "native_gradient_counter.hpp"), ("counter integration", "native_tools.R")]),
        ("Exact generated Stan programs", [("ordinary NCP", "results/student_mixture/native_ncp/pupil-original.stan"), ("ordinary CP", "results/student_mixture/ordinary_cp/ordinary_cp.stan"), ("S2Z CP", "results/student_mixture/s2z_cp/clean.stan"), ("S2Z NCP", "results/student_mixture/s2z_ncp/clean.stan"), ("S2Z auto", "results/student_mixture/s2z_auto/clean.stan")]),
        ("Numerical results", [("complete 15-row table", "results/online_adaptation/brief_matrix/qoi46.tsv"), ("original twelve arms, per QOI", "results/student_mixture/qoi46/parameters.tsv"), ("additional arms, per QOI", "results/online_adaptation/comparison/parameters.tsv"), ("online-fix and run manifest", "results/online_adaptation/receipt.json")]),
    ]
    body += "## 9. Inspect and reproduce the calculation\n\nThe saved source files can be inspected without rerunning the experiments.\n\n"
    for title, files in groups:
        body += f"- **{title}:** " + ", ".join(link(label, path) for label, path in files) + ".\n"
    body += "\nOur total target is implemented in Julia; the linked Stan programs are the exact brms comparison targets. The research README describes the preserved run artifacts and environments.\n"
    assets = repo / "docs/src/assets/adaptive-pupil"
    assets.mkdir(exist_ok=True)
    for filename, source in [
        ("pairs.png", "results/student_mixture/pairs.png"),
        ("s2z_pairs.png", "results/student_mixture/qoi46/s2z_pairs.png"),
        ("efficiency.png", "results/online_adaptation/brief_matrix/efficiency.png"),
    ]:
        shutil.copyfile(root / source, assets / filename)
    body = re.sub(r"!\[([^\n]+)\]\(results/student_mixture/(?:qoi46/)?([^/]+\.png)\)",
                  r"![\1](assets/adaptive-pupil/\2)", body)
    # Documenter parses Julia Markdown before VitePress. Display-dollar fences
    # from the KB brief must become its native math blocks at that boundary.
    body, math_blocks = re.subn(r"(?m)^\$\$\n(.*?)\n\$\$$",
                               r"```math\n\1\n```", body, flags=re.S)
    assert math_blocks == 16 and "\n$$\n" not in body
    assert "results/student_mixture/" not in re.sub(r"https://[^\s)]+", "", body)
    output = repo / "docs/src/pupil-centering.md"
    output.write_text(body)
    print(output)
    sys.exit(0)

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
