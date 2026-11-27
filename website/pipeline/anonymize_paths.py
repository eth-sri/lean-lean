"""Anonymize home-directory usernames in published JSON and indexed gzip packs."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import re

SITE_DATA = Path(__file__).resolve().parents[1] / 'site/data'
HOME_PATH = re.compile(r'(?<![\w:/])((?:file://)?/(?:local/home|home|Users)/)[\w.-]+')
WINDOWS_HOME = re.compile(r'(?i)\b([a-z]:\\Users\\)[\w.-]+')


def anonymize_text(text):
    return WINDOWS_HOME.sub(lambda m: m[1] + 'user',
                            HOME_PATH.sub(lambda m: m[1] + 'user', text))


def anonymize(value):
    if isinstance(value, str):
        return anonymize_text(value)
    if isinstance(value, list):
        return [anonymize(item) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            new_key = anonymize_text(key)
            if new_key in result:
                raise ValueError('Anonymization would merge distinct JSON keys')
            result[new_key] = anonymize(item)
        return result
    return value


def encoded(value):
    return gzip.compress(json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode(),
                         compresslevel=9, mtime=0)


def possible_path(raw):
    return b'/home/' in raw or b'/Users/' in raw or b'\\\\users\\\\' in raw.lower()


def clean_json(raw):
    if not possible_path(raw):
        return None
    old = json.loads(raw)
    new = anonymize(old)
    return new if new != old else None


def clean_pack(path, check=False):
    head_path = path.with_name(path.stem + '.json.gz')
    head = json.loads(gzip.decompress(head_path.read_bytes()))
    source = path.name == 'source.bin'
    entries = list(head.items()) if source else list(enumerate(head['pack']))
    pack = path.read_bytes()
    members, offsets = [], []
    changed = False
    end = 0
    new_end = 0
    for _, (offset, length) in entries:
        if offset != end or offset + length > len(pack):
            raise ValueError(f'Invalid pack offsets: {path}')
        member = pack[offset:offset + length]
        raw = gzip.decompress(member)
        clean = clean_json(raw)
        if clean is not None:
            member = encoded(clean)
            changed = True
        offsets.append([new_end, len(member)])
        new_end += len(member)
        members.append(member)
        end = offset + length
    if end != len(pack):
        raise ValueError(f'Unindexed pack bytes: {path}')
    new_head = anonymize(head)
    if changed:
        if source:
            new_head = {anonymize_text(key): loc for (key, _), loc in zip(entries, offsets)}
            if len(new_head) != len(entries):
                raise ValueError(f'Anonymization would merge source paths: {path}')
        else:
            new_head['pack'] = offsets
    if not changed and new_head == head:
        return False
    if not check:
        # All members and the corresponding index are validated before writing.
        path.write_bytes(b''.join(members))
        head_path.write_bytes(encoded(new_head))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='report without changing files')
    parser.add_argument('--data', type=Path, default=SITE_DATA)
    args = parser.parse_args()
    changed = []
    heads = set()
    for path in sorted(args.data.rglob('*.bin')):
        heads.add(path.with_name(path.stem + '.json.gz'))
        if clean_pack(path, args.check):
            changed.append(path)
    for path in sorted(args.data.rglob('*')):
        if path in heads or not path.is_file() or not path.name.endswith(('.json', '.json.gz')):
            continue
        data = path.read_bytes()
        raw = gzip.decompress(data) if path.suffix == '.gz' else data
        clean = clean_json(raw)
        if clean is not None:
            changed.append(path)
            if not args.check:
                path.write_bytes(encoded(clean) if path.suffix == '.gz' else
                                 json.dumps(clean, separators=(',', ':'), ensure_ascii=False).encode())
    for path in changed:
        print(path.relative_to(args.data))
    print(f'{len(changed)} files or packs {"need anonymization" if args.check else "anonymized"}')
    if args.check and changed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
