"""Build one comparable efficiency table per case from immutable run records."""
import csv
import json
from pathlib import Path
import sys

CASE, ROOT = sys.argv[1:]
ROOT = Path(ROOT)
OUT = Path(__file__).resolve().parent / "results" / CASE
OUT.mkdir(parents=True, exist_ok=True)
ARMS = {"ncp": "NCP", "cp": "CP", "posthoc_position": "Post-hoc position",
        "posthoc_gradient": "Post-hoc gradient", "online_position": "Online position",
        "online_gradient": "Online gradient"}
rows = []
for arm, label in ARMS.items():
    path = ROOT / arm / "summary.tsv"
    if path.exists():
        row = next(csv.DictReader(path.open(), delimiter="\t"))
        row["method"] = label
        rows.append(row)
        target = OUT / arm
        target.mkdir(exist_ok=True)
        for name in ("summary.tsv", "parameters.tsv", "controls.tsv", "packages.tsv", "selection_losses.tsv"):
            src = ROOT / arm / name
            if src.exists():
                (target / name).write_bytes(src.read_bytes())
    elif (ROOT / arm / "failure.json").exists():
        row = json.loads((ROOT / arm / "failure.json").read_text())
        (OUT / f"{arm}-failure.json").write_text(json.dumps(row, indent=2) + "\n")
base = rows[0]
assert base["arm"] == "ncp"
plot = []
lines = ["| WHMC method | Total gradients | Sampling efficiency | Total efficiency |",
         "|:--|--:|--:|--:|"]
for row in rows:
    sampling = float(row["sampling_efficiency"]) / float(base["sampling_efficiency"])
    total = float(row["total_efficiency"]) / float(base["total_efficiency"])
    row.update(relative_sampling_efficiency=sampling, relative_total_efficiency=total)
    lines.append(f'| {row["method"]} | {int(row["total_gradients"]):,} | {sampling:.3g}× | {total:.3g}× |')
    for metric, value in (("Total gradients", float(row["total_gradients"])),
                          ("Relative sampling efficiency", sampling),
                          ("Relative total efficiency", total)):
        plot.append(dict(method=row["method"], metric=metric, value=value))
for arm, label in ARMS.items():
    failure = ROOT / arm / "failure.json"
    if failure.exists() and not (ROOT / arm / "summary.tsv").exists():
        f = json.loads(failure.read_text())
        cost = f.get("total_gradients")
        cost = f"{cost:,}" if cost is not None else "Unavailable"
        lines.append(f"| {label} (failed during adaptation) | {cost} | — | — |")
for name, table in (("comparison.tsv", rows), ("efficiency.tsv", plot)):
    with (OUT / name).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(table)
(OUT / "table.md").write_text("\n".join(lines) + "\n")
if (ROOT / "export").is_dir():
    for name in ("coordinates.tsv", "controls.tsv", "frame_checks.tsv"):
        src = ROOT / "export" / name
        if src.exists():
            (OUT / name).write_bytes(src.read_bytes())
print("\n".join(lines))
