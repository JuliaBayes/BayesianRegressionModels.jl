"""Shared KB capsule uploader: chunked verified uploads + manifest records.

One transport for every fit-artifact capsule flow: study-specific drivers
enumerate and (when needed) build their archives; everything from the
archive bytes onward — chunking, upload, read-back verification, resume,
and manifest emission — lives here, so all capsules share one retrieval
path and one record schema.

Record schema (the `capsules.json` shape, key order significant):
    {archive, sha256, bytes, files?, members?, parts, complete}
    parts[i] = {index, path, bytes, sha256}
`files` (dict) and `members` (list) are driver-supplied member hashes and
pass through untouched; `parts`/`complete` are owned by this module.
Manifests are rewritten incrementally after every verified part, so a
killed run resumes without re-uploading: recorded parts are re-read
locally and hash-asserted, never re-sent.

CLI:
    python capsule_upload.py upload --agent-id ID --manifest PATH FILE...
    python capsule_upload.py verify --manifest PATH [--rewrite]
`upload` merges capsule records for FILEs into the manifest (dict-shape
manifests keyed by archive stem, or list-shape fits manifests matched on
`archive` — extended in place, existing keys byte-identical). `verify`
re-downloads every part, reassembles, SHA-checks the archive, and with
`--rewrite` re-emits the manifest for byte comparison.
"""

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

CHUNK = 20 * 1024 * 1024
UPLOAD_URL = 'http://localhost:4200/code/upload?ext=bin'
REMOTE_PREFIX = '/home/niko/.local/state/kb-agents/uploads/'


def digest_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def upload_block(block, agent_id):
    headers = {'X-KB-Agent-ID': agent_id,
               'Content-Type': 'application/octet-stream'}
    request = urllib.request.Request(UPLOAD_URL, data=block, headers=headers,
                                     method='POST')
    with urllib.request.urlopen(request, timeout=300) as response:
        remote = response.read().decode().strip()
    assert remote.startswith(REMOTE_PREFIX), remote
    return remote


def read_back(remote):
    url = 'http://localhost:4200/code?' + urllib.parse.urlencode(
        {'path': remote, 'raw': '1'})
    with urllib.request.urlopen(url, timeout=300) as response:
        return response.read()


def emit_manifest(path, records):
    Path(path).write_text(json.dumps(records, indent=2) + '\n')


def _verify_part_local(block, recorded):
    assert recorded['sha256'] == hashlib.sha256(block).hexdigest()
    assert recorded['bytes'] == len(block)


def upload_file(path, agent_id, record, on_part=None):
    """Upload `path` in verified 20 MB parts, resuming `record['parts']`.

    Mutates and returns `record` (sets `complete`). `on_part`, when
    given, runs after every newly verified part (the caller persists
    the manifest there for crash-safe resume).
    """
    path = Path(path)
    parts = record.setdefault('parts', [])
    index = 0
    with path.open('rb') as source:
        while block := source.read(CHUNK):
            if index < len(parts):
                _verify_part_local(block, parts[index])
                index += 1
                continue
            remote = upload_block(block, agent_id)
            actual = hashlib.sha256(read_back(remote)).hexdigest()
            expected = hashlib.sha256(block).hexdigest()
            assert actual == expected, (index, remote)
            parts.append(dict(index=index, path=remote, bytes=len(block),
                              sha256=expected))
            print('CAPSULE_PART_VERIFIED', path.name, index, expected,
                  flush=True)
            if on_part is not None:
                on_part()
            index += 1
    assert index == len(parts)
    record['complete'] = True
    return record


def new_record(archive_name, sha256, size, extra=None):
    """A fresh capsule record with the canonical key order."""
    record = dict(archive=archive_name, sha256=sha256, bytes=size)
    if extra:
        record.update(extra)
    record['parts'] = []
    record['complete'] = False
    return record


def verify_record(record, workdir=None):
    """Re-download every part, reassemble, SHA-check the archive.

    Returns the reassembled bytes. Raises on any mismatch.
    """
    blob = b''.join(read_back(p['path']) for p in record['parts'])
    assert len(blob) == record['bytes'], (record['archive'], len(blob))
    assert hashlib.sha256(blob).hexdigest() == record['sha256'], \
        record['archive']
    return blob


def _load_manifest(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def cmd_upload(args):
    manifest_path = Path(args.manifest)
    records = _load_manifest(manifest_path)
    if records is None:
        records = {} if args.shape == 'dict' else []
    changed = False
    for filename in args.files:
        path = Path(filename)
        key = path.stem
        if isinstance(records, dict):
            record = records.get(key)
            if record is not None and record.get('complete'):
                print('CAPSULE_SKIP_COMPLETE', key, flush=True)
                continue
            if record is None:
                record = new_record(path.name, digest_file(path),
                                    path.stat().st_size)
                records[key] = record
        else:
            record = next((r for r in records if r.get('archive') == path.name),
                          None)
            if record is not None and record.get('complete'):
                print('CAPSULE_SKIP_COMPLETE', path.name, flush=True)
                continue
            if record is None:
                record = new_record(path.name, digest_file(path),
                                    path.stat().st_size)
                records.append(record)
        emit_manifest(manifest_path, records)
        upload_file(path, args.agent_id, record,
                    on_part=lambda: emit_manifest(manifest_path, records))
        emit_manifest(manifest_path, records)
        changed = True
        print('CAPSULE_COMPLETE', record['archive'], record['bytes'],
              record['sha256'], flush=True)
    if changed:
        print('CAPSULES_DONE', manifest_path, flush=True)


def cmd_verify(args):
    records = _load_manifest(args.manifest)
    assert records is not None, args.manifest
    items = records.values() if isinstance(records, dict) else records
    total = 0
    for record in items:
        blob = verify_record(record)
        total += len(blob)
        print('CAPSULE_VERIFIED', record['archive'], len(blob),
              record['sha256'], flush=True)
    if args.rewrite:
        emit_manifest(args.manifest, records)
    print('CAPSULES_VERIFIED', len(items) if not isinstance(items, list)
          else len(items), total, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='verb', required=True)
    up = sub.add_parser('upload', help='upload files as verified capsules')
    up.add_argument('--agent-id', required=True)
    up.add_argument('--manifest', required=True)
    up.add_argument('--shape', choices=('dict', 'list'), default='dict')
    up.add_argument('files', nargs='+')
    vf = sub.add_parser('verify', help='re-download and SHA-check capsules')
    vf.add_argument('--manifest', required=True)
    vf.add_argument('--rewrite', action='store_true')
    args = parser.parse_args(argv)
    if args.verb == 'upload':
        cmd_upload(args)
    else:
        cmd_verify(args)


if __name__ == '__main__':
    main()
