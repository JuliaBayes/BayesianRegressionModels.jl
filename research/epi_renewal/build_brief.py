#!/usr/bin/env python3
"""Assemble the KB brief for research/epi_renewal: wrap each results/<name>.aov.json into a
`kb-aov/v1` fence (preliminary mode: the run's rows are not yet checked in at the base commit)
and splice the fences into the brief template at `{{fig:<name>}}` markers.

  python3 research/epi_renewal/build_brief.py <template.md> <out.md> --base-commit <sha> --run <name>
"""
import argparse, hashlib, json, pathlib, re, sys

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
REPO = "https://github.com/nsiccha/BayesianRegressionModels.jl"
PRODUCER = "BayesianRegressionModels:epi"

FIGURES = {
    "single_R":        ("Single patch: R_t recovered", "Posterior 50/80/95 % bands of the daily reproduction number over 56 days with the simulated truth as a dashed line."),
    "forecast":        ("Forecast from a 42-day fit", "Posterior predictive case bands through day 56 from a fit that observes days 1–42 only; points are the simulated counts coloured by fitted/forecast; dashed line is the true expected count."),
    "patch_R":         ("Six patches: R_{g,t}", "Per-patch posterior bands of the reproduction number with the truth dashed, one panel per patch."),
    "patch_cases":     ("Six patches: expected vs simulated cases", "Per-patch posterior bands of expected daily cases on a symlog axis with the simulated counts as points."),
    "patch_recovery":  ("Six patches: K_mix, delta, log I0 recovered", "Posterior median and 95 % interval against the truth for the 36 mixing weights, 48 weekly deviations and 6 seeds; dashed identity line."),
    "delay_pmf":       ("Reporting-delay PMF", "Daily reporting-delay masses: bands over posterior draws of the censored LogNormal fit, truth dashed, PMF at the posterior means as points."),
    "prior_predictive":("Prior predictive cases", "50/80/95 % bands of daily cases drawn from the prior (held_out=:all) on a symlog axis with the simulated series as points."),
    "scalars":         ("Scalar parameters", "Posterior median with 50/95 % intervals for every scalar parameter of the single-patch, 42-day, delay and six-patch fits; crosses mark the truth."),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def envelope(name, base_commit, run):
    title, alt = FIGURES[name]
    spec_path = RESULTS / f"{name}.aov.json"
    spec = json.loads(spec_path.read_text())
    return {
        "schema": "kb-aov/v1",
        "title": title,
        "alt": alt,
        "spec": spec,
        "provenance": {
            "mode": "preliminary",
            "producer": PRODUCER,
            "base_commit": base_commit,
            "run": run,
            "references": [
                {"kind": "spec", "label": "self-contained model, fit and summaries (wren16.jl) and the figure script (plots.jl)",
                 "url": f"{REPO}/tree/ns/devibe/research/epi_renewal", "commit": base_commit, "path": "research/epi_renewal"},
                {"kind": "data", "label": f"{name}.aov.json (inline rows; sha256 {sha256(spec_path)[:12]}…)",
                 "url": f"{REPO}/tree/ns/devibe/research/epi_renewal/results", "path": f"research/epi_renewal/results/{name}.aov.json",
                 "sha256": sha256(spec_path)},
            ],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("template"); ap.add_argument("out")
    ap.add_argument("--base-commit", required=True); ap.add_argument("--run", required=True)
    a = ap.parse_args()
    text = pathlib.Path(a.template).read_text(encoding="utf-8")
    used = []
    def sub(m):
        name = m.group(1)
        used.append(name)
        payload = envelope(name, a.base_commit, a.run)
        return "```kb-aov\n" + json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n```"
    out = re.sub(r"\{\{fig:([A-Za-z_]+)\}\}", sub, text)
    missing = [n for n in FIGURES if n not in used]
    pathlib.Path(a.out).write_text(out, encoding="utf-8")
    print(f"wrote {a.out}: {len(used)} figures embedded ({len(out)} bytes); unused figures: {missing or 'none'}")


if __name__ == "__main__":
    main()
