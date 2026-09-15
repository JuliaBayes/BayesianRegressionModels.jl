"""Preserve completed raw draws, final checkpoints and cost evidence in git artifacts."""
from pathlib import Path
import hashlib
import json
import tarfile
import sys

scratch, output = map(Path, sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)
groups = {}
for folder in ("pupil4-totals-v1", "pupil4-brms-whmc-v1", "pupil4-summary-v1"):
    base = scratch / folder
    files = list(base.glob("*.jls")) + list(base.glob("*.tsv")) + list(base.glob("*.json"))
    files += list(base.glob("*-checkpoints/cp_latest.jls"))
    if folder == "pupil4-totals-v1":
        files += [p for p in (base / "harness").rglob("*") if p.is_file() and p.suffix != ".so"]
    groups[folder] = files
for arm in ("ordinary_ncp", "s2z_cp", "s2z_ncp", "s2z_auto"):
    base = scratch / "pupil4-native-v1" / arm
    files = [p for p in base.iterdir() if p.is_file() and p.suffix in (".R", ".hpp", ".json", ".tsv", ".rds", ".txt", ".stan")]
    for sub in ("sampling", "precursor", "counter-receipts"):
        files += [p for p in (base / sub).rglob("*") if p.is_file()]
    groups["native-" + arm] = files

manifest = []
for group, files in groups.items():
    archive = output / (group + ".tar.gz")
    if archive.exists():
        raise SystemExit(f"Preserve existing archive: {archive}")
    members = []
    with tarfile.open(archive, "w:gz") as target:
        for p in sorted(set(files)):
            name = str(p.relative_to(scratch))
            data = p.read_bytes()
            target.add(p, arcname=name, recursive=False)
            members.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    with tarfile.open(archive) as saved:
        for member in members:
            data = saved.extractfile(member["path"]).read()
            assert len(data) == member["bytes"] and hashlib.sha256(data).hexdigest() == member["sha256"]
    manifest.append(dict(archive=archive.name, bytes=archive.stat().st_size,
                         sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), members=members))
(output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"RAW_ARCHIVES_VERIFIED count={len(manifest)} bytes={sum(x['bytes'] for x in manifest)}")
