"""Assemble one common-QOI table for a case from completed arm diagnostics: SCRATCH OUTPUT CASE"""
import csv
import json
from pathlib import Path
import sys

scratch, output = map(Path, sys.argv[1:3])
case = sys.argv[3]
assert case in ("AS", "crohn")
output.mkdir(parents=True, exist_ok=True)
labels = [
    ("rbest_ncp_native", "RBesT 1.11 NCP (its control)", "Native Stan"),
    ("rbest_cp_native", "RBesT 1.11 CP (its control)", "Native Stan"),
    ("stan_ncp_native", "RBesT 1.11 NCP (Stan defaults)", "Native Stan"),
    ("stan_cp_native", "RBesT 1.11 CP (Stan defaults)", "Native Stan"),
    ("s2z_ncp_native", "RBesT PR64 S2Z NCP", "Native Stan"),
    ("s2z_cp_native", "RBesT PR64 S2Z CP", "Native Stan"),
    ("ordinary_ncp", "BRM ordinary NCP", "WHMC"),
    ("ordinary_cp", "BRM ordinary CP", "WHMC"),
    ("ordinary_posthoc_position", "BRM ordinary post-hoc position", "WHMC"),
    ("ordinary_posthoc_gradient", "BRM ordinary post-hoc gradient", "WHMC"),
    ("ordinary_online_position", "BRM ordinary online position", "WHMC"),
    ("ordinary_online_gradient", "BRM ordinary online gradient", "WHMC"),
    ("total_ncp", "BRM total NCP", "WHMC"),
    ("total_cp", "BRM total CP", "WHMC"),
    ("total_posthoc_position", "BRM total post-hoc position", "WHMC"),
    ("total_posthoc_gradient", "BRM total post-hoc gradient", "WHMC"),
    ("total_online_position", "BRM total online position", "WHMC"),
    ("total_online_gradient", "BRM total online gradient", "WHMC"),
]
directories = [scratch / f"rbest-{kind}-{case}-v1" for kind in ("summary", "matrix")]
records = {}
for directory in directories:
    for p in directory.glob("*-summary.tsv"):
        row = next(csv.DictReader(p.open(), delimiter="\t"))
        if row["arm"] in records:
            assert records[row["arm"]] == row, f"Conflicting diagnostic: {p}"
        records[row["arm"]] = row
baseline = records["rbest_ncp_native"]
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
    fields = list(dict.fromkeys(k for item in items for k in item))   # union, first-seen order
    with (output / name).open("w") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n", restval="0")
        w.writeheader(); w.writerows(items)

write("comparison.tsv", rows)
write("efficiency_plot.tsv", plot)
table = ["| Method | Total gradients | Sampling efficiency | Total efficiency | Divergences |", "|---|---:|---:|---:|---:|"]
for r in rows:
    table.append(f"| {r['model']} · {r['sampler']} | {int(float(r['total_gradients'])):,} | {r['relative_sampling_efficiency']:.3g}× | {r['relative_total_efficiency']:.3g}× | {int(float(r['divergences']))} |")
(output / "table.md").write_text("\n".join(table) + "\n")
(output / "status.json").write_text(json.dumps({"completed": len(rows), "planned": len(labels), "missing": [key for key, _, _ in labels if key not in records]}, indent=2))
print(f"ASSEMBLED {len(rows)}/{len(labels)} rows")
