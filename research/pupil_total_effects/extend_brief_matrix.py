"""Append the three completed adaptation arms to the original 12-arm matrix."""
import csv
from pathlib import Path

root = Path(__file__).resolve().parent
original = root / "results/student_mixture/qoi46"
new = root / "results/online_adaptation/comparison"
out = root / "results/online_adaptation/brief_matrix"
out.mkdir(exist_ok=True)

def read(path):
    return list(csv.DictReader(path.open(), delimiter="\t"))

def write(path, rows):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

rows = read(original / "qoi46.tsv")
assert len(rows) == 12
old = next(r for r in rows if r["model"] == "total coefficients ACP")
old["model"] = "Totals post-hoc position"
measured = [r for r in read(new / "qoi46.tsv") if r["prior"] == "Student-t"]
control = next(r for r in measured if r["arm"] == "Post-hoc position / WHMC")
assert float(old["min_ess"]) == float(control["min_bulk_ess"])
assert old["total_gradients"] == control["total_gradients"]
added = ["Post-hoc gradient", "Online position", "Online gradient"]
for arm in added:
    r = next(r for r in measured if r["arm"] == arm + " / WHMC")
    result = {k: r[k] for k in rows[0] if k not in ("model", "sampler", "min_ess")}
    result.update(model="Totals " + arm.lower(), sampler="WHMC", min_ess=r["min_bulk_ess"])
    rows.append(result)
assert len(rows) == 15
write(out / "qoi46.tsv", rows)

baseline = rows[0]
plot = []
for r in rows:
    label = r["model"].replace("total coefficients", "Totals")
    for title, value in (
        ("1. Total gradients ↓", float(r["total_gradients"])),
        ("2. Sampling efficiency ↑", float(r["sampling_efficiency"]) / float(baseline["sampling_efficiency"])),
        ("3. Total efficiency ↑", float(r["total_efficiency"]) / float(baseline["total_efficiency"])),
    ):
        plot.append(dict(model=label, sampler=r["sampler"], metric=title, value=value))
write(out / "efficiency_plot.tsv", plot)
print(f"BRIEF_MATRIX_COMPLETE rows={len(rows)} plot_values={len(plot)}")
