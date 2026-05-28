#!/usr/bin/env python3
"""Download model files listed in a JSON manifest.

Manifest format (array of objects):
[
  {"url": "https://.../cnn_model.pth", "path": "CNN_module/cnn_model.pth", "sha256": "..."},
  {"url": "https://.../transformer_model.pth", "path": "Transformer_module/transformer_model.pth"}
]

Usage examples:
  python scripts/download_models.py --manifest scripts/models_example.json --dry-run
  python scripts/download_models.py --manifest scripts/models_example.json

The script supports `--dry-run` to only print actions and `--verify` to check SHA256 when provided in manifest.
"""
from __future__ import annotations
import argparse
import json
import os
import hashlib
import sys

try:
    import requests
except Exception:
    requests = None


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: str, dry_run: bool = False) -> None:
    print(f"-> {url} -> {dest}")
    if dry_run:
        return
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    if requests:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    else:
        # fallback to urllib
        from urllib.request import urlopen

        with urlopen(url) as r, open(dest, "wb") as f:
            f.write(r.read())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download model files from a manifest JSON file")
    p.add_argument("--manifest", required=True, help="Path to manifest JSON file")
    p.add_argument("--dry-run", action="store_true", help="Only print planned actions")
    p.add_argument("--verify", action="store_true", help="Verify SHA256 when present in manifest")
    args = p.parse_args(argv)

    if not os.path.exists(args.manifest):
        print("Manifest not found:", args.manifest, file=sys.stderr)
        return 2

    with open(args.manifest, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Manifest must be a JSON array", file=sys.stderr)
        return 3

    for entry in data:
        url = entry.get("url")
        dest = entry.get("path")
        if not url or not dest:
            print("Skipping invalid entry (missing url or path):", entry)
            continue
        download(url, dest, dry_run=args.dry_run)
        if args.verify and entry.get("sha256") and not args.dry_run:
            got = sha256_of_file(dest)
            if got.lower() != entry.get("sha256").lower():
                print(f"SHA256 mismatch for {dest}: expected {entry.get('sha256')}, got {got}", file=sys.stderr)
                return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
