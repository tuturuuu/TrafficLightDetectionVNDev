#!/usr/bin/env python3
"""Download and extract the official BSTLD RGB train/test archives from Zenodo.

Files are resumed through .part files and checked against published MD5 hashes.
Source: https://zenodo.org/records/12706046
Run: python -u scripts/data/prepare_bstld.py
Progress goes to stdout; preparation_report.json is written after both succeed.
"""
import concurrent.futures
import hashlib
import fcntl
import json
from pathlib import Path
import shutil
import time
import zipfile
import requests
ROOT = Path(__file__).resolve().parents[2] / 'data/BSTLD'
FILES = [('dataset_train_rgb.zip', 6179029947, 'cf0ac345bcd9793c77ec1b6ce3b6fa32'), ('dataset_test_rgb.zip', 12160942862, 'ecea5fae3532c32608def305e66fdf8f')]

def prepare(item):
    name, size, digest = item
    dst = ROOT / 'downloads' / name
    partial = dst.with_suffix('.zip.part')
    if not dst.exists():
        for attempt in range(6):
            start = partial.stat().st_size if partial.exists() else 0
            print(f'{name}: resuming at {start / 1000000000.0:.2f}/{size / 1000000000.0:.2f} GB', flush=True)
            if start == size:
                break
            if start > size:
                raise ValueError('Partial file larger than expected')
            try:
                url = 'https://zenodo.org/records/12706046/files/' + name + '?download=1&resume=' + str(start)
                with requests.get(url, headers={'Range': f'bytes={start}-'}, stream=True, timeout=(30, 90)) as r:
                    r.raise_for_status()
                    if r.status_code == 206:
                        if not r.headers.get('Content-Range', '').startswith(f'bytes {start}-'):
                            raise ValueError('Unexpected range')
                    elif r.status_code != 200 or start:
                        raise ValueError('Server did not honor resume')
                    n = start
                    last = time.monotonic()
                    last_n = n
                    with partial.open('ab') as f:
                        for chunk in r.iter_content(4 * 1024 * 1024):
                            f.write(chunk)
                            n += len(chunk)
                            if n > size:
                                raise ValueError('Download larger than expected')
                            now = time.monotonic()
                            if now - last >= 30:
                                print(f'{name}: {n / 1000000000.0:.2f}/{size / 1000000000.0:.2f} GB ({(n - last_n) / (now - last) / 1000000.0:.1f} MB/s)', flush=True)
                                last, last_n = (now, n)
                if partial.stat().st_size != size:
                    raise IOError('Incomplete download')
                break
            except requests.RequestException as e:
                print(name, 'retry', attempt + 1, type(e).__name__, flush=True)
                time.sleep(3)
        if not partial.exists() or partial.stat().st_size != size:
            raise IOError(name + ' download incomplete')
        source = partial
    else:
        source = dst
    print(name, 'verifying MD5', flush=True)
    md5 = hashlib.md5()
    with source.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            md5.update(chunk)
    if md5.hexdigest() != digest:
        raise ValueError(name + ' checksum mismatch')
    if source != dst:
        partial.rename(dst)
    destination = ROOT / 'raw' / name.removesuffix('.zip')
    marker = destination / '.extraction_complete.json'
    if marker.exists():
        return json.loads(marker.read_text())
    if destination.exists():
        raise FileExistsError(destination)
    staging = destination.with_name(destination.name + '.extracting')
    with zipfile.ZipFile(dst) as z:
        members = z.infolist()
        total = sum((m.file_size for m in members))
        if total > shutil.disk_usage(ROOT).free:
            raise RuntimeError('Insufficient disk space to extract')
        for m in members:
            path = destination / m.filename
            if not path.resolve().is_relative_to(destination.resolve()):
                raise ValueError('Unsafe archive path')
            if m.external_attr >> 16 & 61440 == 40960:
                raise ValueError('Archive contains symlink')
        staging.mkdir(parents=True, exist_ok=True)
        print(name, 'extracting', len(members), 'entries', round(total / 1000000000.0, 2), 'GB', flush=True)
        for m in members:
            z.extract(m, staging)
    staging.rename(destination)
    print(name, 'complete', flush=True)
    result = {'archive': name, 'source': 'https://zenodo.org/records/12706046', 'md5': digest, 'bytes': size, 'extracted_to': str(destination), 'entries': len(members), 'uncompressed_bytes': total}
    marker.write_text(json.dumps(result, indent=2) + '\n')
    return result

def main():
    (ROOT / 'downloads').mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'downloads' / 'prepare.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        results = list(ex.map(prepare, FILES))
    (ROOT / 'preparation_report.json').write_text(json.dumps(results, indent=2) + '\n')
    print('BSTLD preparation complete', flush=True)
if __name__ == '__main__':
    main()
