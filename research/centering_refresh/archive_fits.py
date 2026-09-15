"""Preserve raw fits/checkpoints in verified KB uploads; keep hashes in Git."""
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.parse
import urllib.request

BASE = Path('/home/n/scratch/kb-agent-tmp/BayesianRegressionModels-docs-adaptive-centering')
RESULTS = Path(__file__).resolve().parent / 'results'
DEST = BASE / 'centering-refresh-capsules-v1'
DEST.mkdir(exist_ok=True)
MANIFEST = RESULTS / 'capsules.json'
records = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
HEADERS = {'X-KB-Agent-ID': 'BayesianRegressionModels:docs:adaptive-centering',
           'Content-Type': 'application/octet-stream'}

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

for case in ('eight', 'radon', 'hsgp'):
    root = BASE / f'centering-refresh-{case}-v1'
    for arm in sorted(p for p in root.iterdir() if (p/'fit.jls').exists() or (p/'failure.json').exists()):
        key = f'{case}/{arm.name}'
        if key in records and records[key].get('complete'):
            continue
        capsule = DEST / f'{case}-{arm.name}.tar.gz'
        if not capsule.exists():
            paths = list(arm.glob('*.jls')) + list(arm.glob('*.tsv')) + list(arm.glob('*.jl'))
            paths += list(arm.glob('*.stan')) + list(arm.glob('*.json'))
            paths += [arm/'checkpoints/cp_latest.jls']
            if (arm/'failure.json').exists():
                paths += list((arm/'checkpoints').glob('*.jls'))
            paths = sorted(set(p for p in paths if p.is_file()))
            with capsule.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=1, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode='w|') as tar:
                    for path in paths:
                        tar.add(path, arcname=str(path.relative_to(root)))
                    env = 'finite-transport-env' if (arm/'packages.tsv').exists() and 'warmuphmc-finite-transport-fix' in (arm/'packages.tsv').read_text() else 'pupil-online-fix-env'
                    for path in (BASE/env).glob('*toml'):
                        tar.add(path, arcname='environment/'+path.name)
            file_hashes = {str(p.relative_to(root)): digest(p) for p in paths}
            records[key] = dict(archive=capsule.name, sha256=digest(capsule), bytes=capsule.stat().st_size,
                                files=file_hashes, parts=[], complete=False)
            MANIFEST.write_text(json.dumps(records, indent=2)+'\n')
        record = records[key]
        with capsule.open('rb') as source:
            index = 0
            while block := source.read(20*1024*1024):
                if index < len(record['parts']):
                    assert record['parts'][index]['sha256'] == hashlib.sha256(block).hexdigest()
                    index += 1
                    continue
                request = urllib.request.Request('http://localhost:4200/code/upload?ext=bin', data=block,
                                                 headers=HEADERS, method='POST')
                with urllib.request.urlopen(request, timeout=120) as response:
                    remote = response.read().decode().strip()
                assert remote.startswith('/home/niko/.local/state/kb-agents/uploads/'), remote
                url = 'http://localhost:4200/code?' + urllib.parse.urlencode({'path':remote,'raw':'1'})
                with urllib.request.urlopen(url, timeout=120) as response:
                    actual = hashlib.sha256(response.read()).hexdigest()
                expected = hashlib.sha256(block).hexdigest()
                assert actual == expected
                record['parts'].append(dict(index=index,path=remote,bytes=len(block),sha256=expected))
                MANIFEST.write_text(json.dumps(records, indent=2)+'\n')
                print('CAPSULE_PART_VERIFIED',key,index,expected,flush=True)
                index += 1
        record['complete'] = True
        MANIFEST.write_text(json.dumps(records, indent=2)+'\n')
        print('CAPSULE_COMPLETE',key,record['bytes'],record['sha256'],flush=True)
print('RAW_FITS_ARCHIVED',len(records),flush=True)
