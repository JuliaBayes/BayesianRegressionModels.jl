"""Retain fit arrays, final checkpoints and native process evidence, with hashes."""
from pathlib import Path
import hashlib
import json
import sys
import tarfile

scratch, output = map(Path, sys.argv[1:3])
folders = sys.argv[3:]
output.mkdir(parents=True, exist_ok=True)
manifest = []
for folder in folders:
    base = scratch / folder
    assert base.is_dir(), base
    if (base / "sampling").is_dir():
        assert (base / "provenance.json").is_file() and (base / "gradient_counts.tsv").is_file()
        files = [p for p in base.iterdir() if p.is_file() and p.suffix in
                 (".R", ".hpp", ".json", ".tsv", ".rds", ".txt", ".stan")]
        for child in ("sampling", "precursor", "counter-receipts"):
            files.extend(p for p in (base / child).rglob("*") if p.is_file())
    else:
        files = [p for p in base.iterdir() if p.is_file() and p.suffix in (".jls", ".tsv", ".json")]
        files.extend(base.glob("*-checkpoints/cp_latest.jls"))
        for child in ("model", "harness"):
            files.extend(p for p in (base / child).rglob("*") if p.is_file()
                         and p.suffix in (".jl", ".R", ".py", ".hpp", ".json", ".tsv", ".stan", ".md"))
    assert files, folder
    archive = output / (folder.replace("/", "-") + ".tar.gz")
    assert not archive.exists(), f"Preserve the existing archive: {archive}"
    members = []
    with tarfile.open(archive, "w:gz") as target:
        for path in sorted(set(files)):
            data = path.read_bytes()
            name = str(path.relative_to(scratch))
            target.add(path, arcname=name, recursive=False)
            members.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    with tarfile.open(archive) as saved:
        for member in members:
            data = saved.extractfile(member["path"]).read()
            assert len(data) == member["bytes"] and hashlib.sha256(data).hexdigest() == member["sha256"]
    manifest.append(dict(archive=archive.name, bytes=archive.stat().st_size,
                         sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), members=members))
(output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("RAW_ARCHIVES_VERIFIED", len(manifest), sum(x["bytes"] for x in manifest), flush=True)
