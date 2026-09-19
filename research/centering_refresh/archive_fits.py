"""Build centering-refresh capsules; upload them through the shared module.

Thin wrapper over `research/capsule_upload.py`: study-specific capsule
enumeration and construction stay here (verbatim from the original
per-study script); chunking, upload, read-back verification, resume, and
manifest emission all live in the shared module. Record key order and
manifest formatting are unchanged, so re-emitting `capsules.json`
through this path is byte-identical.
"""
import gzip
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from capsule_upload import digest_file, emit_manifest, new_record, upload_file  # noqa: E402

import json

BASE = Path('/home/n/scratch/kb-agent-tmp/BayesianRegressionModels-docs-adaptive-centering')
RESULTS = Path(__file__).resolve().parent / 'results'
DEST = BASE / 'centering-refresh-capsules-v1'
DEST.mkdir(exist_ok=True)
MANIFEST = RESULTS / 'capsules.json'
AGENT_ID = 'BayesianRegressionModels:docs:adaptive-centering'
records = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}

for case in ('eight', 'radon', 'hsgp'):
    root = BASE / f'centering-refresh-{case}-v1'
    for arm in sorted(p for p in root.iterdir() if (p / 'fit.jls').exists() or (p / 'failure.json').exists()):
        key = f'{case}/{arm.name}'
        if key in records and records[key].get('complete'):
            continue
        capsule = DEST / f'{case}-{arm.name}.tar.gz'
        if not capsule.exists():
            paths = list(arm.glob('*.jls')) + list(arm.glob('*.tsv')) + list(arm.glob('*.jl'))
            paths += list(arm.glob('*.stan')) + list(arm.glob('*.json'))
            paths += [arm / 'checkpoints/cp_latest.jls']
            if (arm / 'failure.json').exists():
                paths += list((arm / 'checkpoints').glob('*.jls'))
            paths = sorted(set(p for p in paths if p.is_file()))
            with capsule.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=1, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode='w|') as tar:
                    for path in paths:
                        tar.add(path, arcname=str(path.relative_to(root)))
                    env = 'finite-transport-env' if (arm / 'packages.tsv').exists() and 'warmuphmc-finite-transport-fix' in (arm / 'packages.tsv').read_text() else 'pupil-online-fix-env'
                    for path in (BASE / env).glob('*toml'):
                        tar.add(path, arcname='environment/' + path.name)
            file_hashes = {str(p.relative_to(root)): digest_file(p) for p in paths}
            records[key] = new_record(capsule.name, digest_file(capsule),
                                      capsule.stat().st_size,
                                      {'files': file_hashes})
            emit_manifest(MANIFEST, records)
        record = records[key]
        upload_file(capsule, AGENT_ID, record,
                    on_part=lambda: emit_manifest(MANIFEST, records))
        emit_manifest(MANIFEST, records)
        print('CAPSULE_COMPLETE', key, record['bytes'], record['sha256'], flush=True)
print('RAW_FITS_ARCHIVED', len(records), flush=True)
