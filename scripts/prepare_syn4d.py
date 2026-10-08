"""Extract completed Syn4D scene archives once, outside DataLoader workers.

An unfinished archive/extraction is never published as a ready scene. Existing
scene directories are preserved. Videos in subset/mp4/scene need no copying.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile


def extract_scene(archive: str | Path) -> dict:
    archive = Path(archive).resolve()
    if not archive.name.endswith('.tar.zst') or not archive.is_file():
        raise ValueError(f'expected a completed .tar.zst archive: {archive}')
    scene = archive.name[:-8]
    destination = archive.parent / scene
    if destination.exists():
        return dict(archive=str(archive), scene=str(destination), status='existing_preserved')
    before = archive.stat()
    signature = (before.st_size, before.st_mtime_ns)
    files = size = 0
    with tempfile.TemporaryDirectory(prefix='.syn4d-extract-', dir=archive.parent) as staging:
        process = subprocess.Popen(['zstd', '-dc', '--', str(archive)], stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=process.stdout, mode='r|') as tar:
                for member in tar:
                    path = PurePosixPath(member.name)
                    if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != scene:
                        raise ValueError(f'archive member outside scene {scene}: {member.name}')
                    if not (member.isdir() or member.isfile()):
                        raise ValueError(f'unsupported archive member: {member.name}')
                    target = Path(staging).joinpath(*path.parts)
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as source, target.open('xb') as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
                    if target.stat().st_size != member.size:
                        raise ValueError(f'truncated archive member: {member.name}')
                    files += 1
                    size += member.size
            # Drain the compressed stream so zstd also verifies its checksum.
            while process.stdout.read(1024 * 1024):
                pass
            error = process.stderr.read().decode(errors='replace')
            if process.wait() != 0:
                raise ValueError(f'zstd rejected {archive}: {error}')
            after = archive.stat()
            if signature != (after.st_size, after.st_mtime_ns):
                raise ValueError(f'archive changed during extraction: {archive}')
            extracted = Path(staging) / scene
            if not (extracted / 'ground_truth/meta_exr_csv').is_dir():
                raise ValueError(f'archive lacks Syn4D camera metadata: {archive}')
            report = dict(archive=str(archive), scene=str(destination), status='ready',
                          archive_bytes=signature[0], archive_mtime_ns=signature[1],
                          files=files, extracted_bytes=size)
            (extracted / '.syn4d_ready.json').write_text(json.dumps(report, indent=2) + '\n')
            extracted.rename(destination)
            return report
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
            process.stdout.close()
            process.stderr.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archives', nargs='+', help='Completed scene .tar.zst files; shell globs supported')
    args = parser.parse_args()
    for archive in args.archives:
        print(json.dumps(extract_scene(archive)), flush=True)


if __name__ == '__main__':
    main()
