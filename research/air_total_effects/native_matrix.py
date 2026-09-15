"""Run the remaining native arms sequentially, preserving completed outputs."""
import json
import pathlib
import subprocess
import sys

cmdstan, output, grouping, hierarchy, audit = sys.argv[1:]
output, audit = pathlib.Path(output), pathlib.Path(audit)
assert json.loads((audit / "audit.json").read_text())["status"] == "passed"
for label in ("s2z_cp", "s2z_ncp", "s2z_auto"):
    dest = output / label
    if (dest / "provenance.json").is_file():
        assert (dest / "gradient_counts.tsv").is_file() and (dest / "fit.rds").is_file()
        print("REUSE_COMPLETE_NATIVE", label, flush=True)
        continue
    assert not dest.exists(), f"Incomplete output must be inspected before retrying: {dest}"
    subprocess.run(["Rscript", str(pathlib.Path(__file__).with_name("native.R")), cmdstan,
                    str(dest), grouping, hierarchy, label,
                    str(audit / "ordinary-init.json"), str(audit / "audit.json")], check=True)
print("AIR_NATIVE_MATRIX_COMPLETE", grouping, hierarchy, flush=True)
