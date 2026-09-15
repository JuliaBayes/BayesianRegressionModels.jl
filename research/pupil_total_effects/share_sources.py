"""Freeze an inspectable source snapshot and write KB-native file links.

No execution or publication. The KB file viewer/gist exporter resolves these
absolute producer paths; SHA-256 identifies the exact text in each snapshot.
"""
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
destination = Path(sys.argv[1]).resolve()
output = root / "results/student_mixture/qoi46"
output.mkdir(parents=True, exist_ok=True)

files = sorted({p.relative_to(root).as_posix()
                for pattern in ("*.jl", "*.R", "*.hpp", "*.py")
                for p in root.glob(pattern)})
files += ["reference/pupil.csv"]
result = "results/student_mixture/"
files += [result + p for p in (
    "native_ncp/pupil-original.stan", "native_ncp/pupil-counted.stan",
    "native_ncp/data.json", "native_ncp/init.json",
    "ordinary_cp/ordinary_cp.stan", "ordinary_cp/data.json",
    "s2z_auto/native/centering-weights.txt",
    "qoi46/qoi46.tsv", "qoi46/parameters.tsv", "qoi46/efficiency_plot.tsv",
    "qoi46/s2z_pairs_selection.tsv", "qoi46/s2z_pairs_audit.tsv",
)]
for arm in ("s2z_cp", "s2z_ncp", "s2z_auto"):
    files += [result + arm + "/" + p for p in (
        "clean.stan", "native/instrumented.stan", "resolved-data.json", "init.json",
        "native/gradient_counts.tsv", "whmc/equivalence_audit.tsv")]
manifest = []
for name in files:
    source = root / name
    target = destination / name
    body = source.read_bytes()
    if target.exists():
        assert target.read_bytes() == body, f"Snapshot changed: {name}; use a new destination"
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    manifest.append({"path": name, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()})
manifest_body = json.dumps(manifest, indent=2) + "\n"
(destination / "manifest.json").write_text(manifest_body)
(output / "source_manifest.json").write_text(manifest_body)


def link(label, name):
    return f"[{label}]({destination / name})"


lines = [
    "These links open the saved source text directly. No local execution is needed.",
    "The generated Stan files are the targets used in the completed fits; the",
    "counted variants include the zero-contribution gradient counter. Clean and",
    "counted files target the same posterior. Our total-coefficient target is",
    "implemented manually in Julia, so it has no generated Stan file.", "",
    "### Test harness and analysis", "",
    "| Inspect | Source files |", "|---|---|",
]
groups = [
    ("Manual total target, priors and independent audits", [("model.jl", "model.jl"), ("audit.jl", "audit.jl")]),
    ("Total NCP pilot, offline selection/refit and fixed CP", [("run.jl", "run.jl"), ("fixed_cp.jl", "fixed_cp.jl")]),
    ("Ordinary brms + WHMC", [("brms_baseline.jl", "brms_baseline.jl"), ("ordinary_cp.jl", "ordinary_cp.jl"), ("stan_target.jl", "stan_target.jl")]),
    ("Native Stan and actual gradient counting", [("native_stan.R", "native_stan.R"), ("native_tools.R", "native_tools.R"), ("C++ counter", "native_gradient_counter.hpp")]),
    ("brms S2Z generation, native sampling and WHMC/audits", [("s2z_native.R", "s2z_native.R"), ("capture_s2z.R", "capture_s2z.R"), ("s2z_whmc.jl", "s2z_whmc.jl")]),
    ("Conditional recovery and the 46-QOI comparison", [("recovery.jl", "recovery.jl"), ("compare_qois.jl", "compare_qois.jl"), ("matrix/cost assembly", "compare_matrix.jl")]),
    ("Figures from saved draws", [("total-coordinate preparation", "prepare_pairs.jl"), ("S2Z-coordinate preparation", "prepare_s2z_pairs.jl"), ("total-coordinate plot", "plot_saved.jl"), ("comparison plots", "plot_comparison.jl")]),
]
for label, sources in groups:
    lines.append("| " + label + " | " + ", ".join(link(text, name) for text, name in sources) + " |")
lines += ["", "### Exact generated Stan targets and inputs", "",
          "| Model | Generated Stan | Inputs |", "|---|---|---|"]
for label, folder, clean, counted in [
    ("Ordinary NCP", "native_ncp", "pupil-original.stan", "pupil-counted.stan"),
    ("Ordinary CP (WHMC)", "ordinary_cp", "ordinary_cp.stan", None),
    ("S2Z CP", "s2z_cp", "clean.stan", "native/instrumented.stan"),
    ("S2Z NCP", "s2z_ncp", "clean.stan", "native/instrumented.stan"),
    ("S2Z auto", "s2z_auto", "clean.stan", "native/instrumented.stan"),
]:
    prefix = result + folder + "/"
    sources = link("clean model", prefix + clean)
    if counted:
        sources += ", " + link("counted native model", prefix + counted)
    inputs = link("resolved data", prefix + ("resolved-data.json" if folder.startswith("s2z") else "data.json"))
    if folder != "ordinary_cp":
        inputs += ", " + link("initialization", prefix + "init.json")
    if folder == "s2z_auto":
        inputs += ", " + link("auto weights", prefix + "native/centering-weights.txt")
    lines.append(f"| {label} | {sources} | {inputs} |")
lines += ["", "### Data and numerical results", "",
    "- " + link("Original-order pupil data (CSV)", "reference/pupil.csv") + ".",
    "- " + link("Complete 46-QOI efficiency table (TSV)", result + "qoi46/qoi46.tsv") + ".",
    "- " + link("Per-quantity ESS and MCSE (TSV)", result + "qoi46/parameters.tsv") + ".",
    "- " + link("S2Z scatter selection", result + "qoi46/s2z_pairs_selection.tsv") + " and " + link("coordinate audit", result + "qoi46/s2z_pairs_audit.tsv") + ".",
    "- " + link("Source snapshot manifest with byte counts and SHA-256", "manifest.json") + ".", "",
]
(output / "inspectable_sources.md").write_text("\n".join(lines))
print(f"SOURCE_SNAPSHOT_COMPLETE files={len(manifest)} bytes={sum(r['bytes'] for r in manifest)} destination={destination}")
