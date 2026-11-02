#!/usr/bin/env python3
"""Download a public Google Drive folder recursively, without login.

gdown --folder stops at 50 files per folder, so this lists folders through the
public embeddedfolderview page and downloads each file by
ID in parallel. That page caps at 5500 entries; for bigger folders pass
--api-key (Drive API v3, paginated). Files already present with nonzero size are skipped, so a rerun
resumes an interrupted download.

    python scripts/data/fetch_gdrive_folder.py <folder_id_or_url> <dest> [--workers 16] [--api-key KEY]
"""
import argparse
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

LIST_URL = 'https://drive.google.com/embeddedfolderview?id={}'
FILE_URL = 'https://drive.usercontent.google.com/download?id={}&export=download&confirm=t'
ENTRY = re.compile(r'<a href="([^"]+)"[^>]*>.*?<div class="flip-entry-title">([^<]*)</div>', re.S)
API_URL = 'https://www.googleapis.com/drive/v3/files'
FOLDER_MIME = 'application/vnd.google-apps.folder'
UA = {'User-Agent': 'Mozilla/5.0'}


def fetch(url, timeout=300):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read(), r.headers.get_content_type()


def with_retry(fn, what, tries=6):
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:  # network errors and Drive throttling both retry
            if attempt == tries - 1:
                raise RuntimeError(f'{what}: {e}') from e
            time.sleep(2 ** attempt)


EMBED_CAP = 5500  # embeddedfolderview silently returns at most this many entries
API_KEY = None


def list_folder_api(folder_id):
    """Drive API v3 with an API key: works for public folders and paginates."""
    token = ''
    while True:
        q = urllib.parse.urlencode({
            'q': f"'{folder_id}' in parents and trashed=false", 'key': API_KEY, 'pageSize': 1000,
            'fields': 'nextPageToken,files(id,name,mimeType)', 'pageToken': token,
            'supportsAllDrives': 'true', 'includeItemsFromAllDrives': 'true'})
        page = json.loads(with_retry(lambda: fetch(f'{API_URL}?{q}')[0], f'list {folder_id}'))
        for f in page['files']:
            yield ('dir' if f['mimeType'] == FOLDER_MIME else 'file'), f['name'], f['id']
        if not (token := page.get('nextPageToken')):
            return


def list_folder(folder_id):
    if API_KEY:
        yield from list_folder_api(folder_id)
        return
    page = with_retry(lambda: fetch(LIST_URL.format(folder_id))[0].decode(), f'list {folder_id}')
    entries = ENTRY.findall(page)
    if len(entries) >= EMBED_CAP:
        raise RuntimeError(f'folder {folder_id} lists {len(entries)} entries, the public page cap; '
                           'listing is truncated, rerun with --api-key')
    for url, name in entries:
        url, name = html.unescape(url), html.unescape(name)
        if m := re.search(r'/folders/([\w-]+)', url):
            yield 'dir', name, m.group(1)
        elif m := re.search(r'/file/d/([\w-]+)', url):
            yield 'file', name, m.group(1)


def walk(folder_id, rel=Path(), seen=None):
    seen = set() if seen is None else seen
    if folder_id in seen:  # Drive shortcuts can alias a folder (e.g. val/ and valid/)
        return
    seen.add(folder_id)
    for kind, name, fid in list_folder(folder_id):
        if kind == 'dir':
            yield from walk(fid, rel / name, seen)
        else:
            yield rel / name, fid


def download(fid, path):
    def go():
        data, ctype = fetch(FILE_URL.format(fid))
        if ctype == 'text/html':  # quota/virus-scan page instead of the file
            raise IOError('Drive returned an HTML page (quota or permission)')
        return data
    data = with_retry(go, str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.part')
    tmp.write_bytes(data)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('folder', help='folder ID or https://drive.google.com/drive/folders/<ID> URL')
    p.add_argument('dest', type=Path)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--api-key', help='Google Cloud API key with Drive API enabled; needed for folders over 5500 files')
    p.add_argument('--include', default='', help='only paths matching this regex, e.g. "^val/"')
    args = p.parse_args()
    global API_KEY
    API_KEY = args.api_key

    folder_id = re.sub(r'.*/folders/([\w-]+).*', r'\1', args.folder)
    pattern = re.compile(args.include)
    print('Listing folder...', flush=True)
    files = [(rel, fid) for rel, fid in walk(folder_id) if pattern.search(rel.as_posix())]
    todo = [(rel, fid) for rel, fid in files
            if not ((args.dest / rel).is_file() and (args.dest / rel).stat().st_size > 0)]
    print(f'{len(files)} files, {len(files) - len(todo)} already present, {len(todo)} to download', flush=True)

    failed = []
    with ThreadPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(download, fid, args.dest / rel): rel for rel, fid in todo}
        for i, job in enumerate(as_completed(jobs), 1):
            if job.exception():
                failed.append(jobs[job])
                print(f'FAIL {job.exception()}', file=sys.stderr, flush=True)
            if i % 500 == 0 or i == len(jobs):
                print(f'{i}/{len(jobs)} done, {len(failed)} failed', flush=True)
    if failed:
        sys.exit(f'{len(failed)} files failed; rerun the same command to retry them')


if __name__ == '__main__':
    main()
