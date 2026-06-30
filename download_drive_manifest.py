import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "test images"
MANIFEST = TARGET / "drive_manifest.json"
CHUNK_SIZE = 1024 * 1024
NUMBERED_FILE = re.compile(r"^(\d+)\.(bmp|json)$", re.IGNORECASE)


def load_manifest():
    text = MANIFEST.read_text(encoding="utf-8-sig")
    return json.loads(text)


def already_done(path):
    return path.exists() and path.stat().st_size > 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--pair-start-index", type=int)
    parser.add_argument("--pair-end-index", type=int)
    return parser.parse_args()


def in_requested_range(path, start, end):
    if start is None and end is None:
        return True

    match = NUMBERED_FILE.match(Path(path).name)
    if not match:
        return False

    number = int(match.group(1))
    if start is not None and number < start:
        return False
    if end is not None and number > end:
        return False
    return True


def file_number(path):
    match = NUMBERED_FILE.match(Path(path).name)
    if not match:
        return None
    return int(match.group(1))


def pair_index_numbers(entries, start_index, end_index):
    if start_index is None and end_index is None:
        return None

    numbers = sorted(
        {
            number
            for entry in entries
            if (number := file_number(entry["path"])) is not None
        }
    )
    start = 0 if start_index is None else max(start_index - 1, 0)
    end = len(numbers) if end_index is None else end_index
    return set(numbers[start:end])


def download_one(session, url, path):
    part = path.with_suffix(path.suffix + ".part")
    if part.exists():
        part.unlink()

    path.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=(20, 120), allow_redirects=True) as response:
        response.raise_for_status()
        ctype = response.headers.get("content-type", "")
        if "text/html" in ctype.lower():
            preview = response.text[:500].replace("\n", " ")
            raise RuntimeError(f"Google returned HTML instead of a file: {preview}")

        with part.open("wb") as handle:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    handle.write(chunk)

    if part.stat().st_size == 0:
        part.unlink(missing_ok=True)
        raise RuntimeError("Downloaded zero bytes")
    os.replace(part, path)


def main():
    args = parse_args()
    all_entries = load_manifest()
    wanted_numbers = pair_index_numbers(
        all_entries,
        args.pair_start_index,
        args.pair_end_index,
    )
    entries = [
        entry
        for entry in all_entries
        if in_requested_range(entry["path"], args.start, args.end)
        and (
            wanted_numbers is None
            or file_number(entry["path"]) in wanted_numbers
        )
    ]
    session = requests.Session()
    done = 0
    skipped = 0
    failed = []

    for index, entry in enumerate(entries, 1):
        rel_path = entry["path"].replace("/", os.sep)
        target_path = TARGET / rel_path

        if already_done(target_path):
            skipped += 1
            continue

        for attempt in range(1, 4):
            try:
                download_one(session, entry["url"], target_path)
                done += 1
                break
            except Exception as exc:
                if attempt == 3:
                    failed.append((rel_path, str(exc)))
                    print(f"FAILED {rel_path}: {exc}", flush=True)
                else:
                    time.sleep(5 * attempt)

        if (done + skipped) % 25 == 0 or index == len(entries):
            print(
                f"{index}/{len(entries)} processed, {done} downloaded, "
                f"{skipped} skipped, {len(failed)} failed",
                flush=True,
            )

    if failed:
        fail_path = TARGET / "download_failures.txt"
        fail_path.write_text(
            "\n".join(f"{path}\t{error}" for path, error in failed),
            encoding="utf-8",
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
