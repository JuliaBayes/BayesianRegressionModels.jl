"""Assemble one common-QOI table from completed saved-fit diagnostics."""
import csv
import json
from pathlib import Path
import sys

scratch, output = map(Path, sys.argv[1:3])
case = sys.argv[3]
assert case in ("cluster-independent", "cluster-intercept")
output.mkdir(parents=True, exist_ok=True)
labels = [
    ("ordinary_ncp_native", "brms NCP", "Native Stan"),
    ("ordinary_ncp_whmc", "brms NCP", "WHMC"),
    ("s2z_cp_native", "brms S2Z CP", "Native Stan"),
    ("s2z_cp_whmc", "brms S2Z CP", "WHMC"),
    ("s2z_ncp_native", "brms S2Z NCP", "Native Stan"),
    ("s2z_ncp_whmc", "brms S2Z NCP", "WHMC"),
    ("s2z_auto_native", "brms S2Z auto", "Native Stan"),
    ("s2z_auto_whmc", "brms S2Z auto", "WHMC"),
    ("total_ncp", "BRM total NCP", "WHMC"),
    ("total_cp", "BRM total CP", "WHMC"),
    ("total_posthoc_position", "BRM total post-hoc position", "WHMC"),
    ("total_posthoc_gradient", "BRM total post-hoc gradient", "WHMC"),
    ("total_online_position", "BRM total online position", "WHMC"),
    ("total_online_gradient", "BRM total online gradient", "WHMC"),
]
directories = [scratch / f"air-{kind}-{case}-v1" for kind in ("summary", "totals", "brms-whmc", "brms-auto-whmc")]
records = {}
for directory in directories:
    for p in directory.glob("*-summary.tsv"):
        row = next(csv.DictReader(p.open(), delimiter="\t"))
        if row["arm"] in records:
            assert records[row["arm"]] == row, f"Conflicting diagnostic: {p}"
        records[row["arm"]] = row
baseline = records["ordinary_ncp_native"]
rows, plot = [], []
for key, model, sampler in labels:
    if key not in records:
        continue
    row = dict(records[key], model=model, sampler=sampler)
    for metric in ("sampling", "total"):
        field = f"ess_per_1000_{metric}_gradients"
        row[f"relative_{metric}_efficiency"] = float(row[field]) / float(baseline[field])
    rows.append(row)
    for metric, value in (("1. Total gradients", float(row["total_gradients"])),
                          ("2. Sampling efficiency", row["relative_sampling_efficiency"]),
                          ("3. Total efficiency", row["relative_total_efficiency"])):
        plot.append(dict(model=model, sampler=sampler, metric=metric, value=value))

def write(name, items):
    with (output / name).open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(items[0]), delimiter="\t", lineterminator="\n")
        w.writeheader(); w.writerows(items)

write("comparison.tsv", rows)
write("efficiency_plot.tsv", plot)
table = ["| Method | Total gradients | Sampling efficiency | Total efficiency |", "|---|---:|---:|---:|"]
for r in rows:
    table.append(f"| {r['model']} · {r['sampler']} | {int(r['total_gradients']):,} | {r['relative_sampling_efficiency']:.3g}× | {r['relative_total_efficiency']:.3g}× |")
(output / "table.md").write_text("\n".join(table) + "\n")
(output / "status.json").write_text(json.dumps({"completed": len(rows), "planned": len(labels), "missing": [key for key, _, _ in labels if key not in records]}, indent=2))
print(f"ASSEMBLED {len(rows)}/{len(labels)} rows")
