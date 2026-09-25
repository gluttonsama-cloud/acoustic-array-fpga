"""Explicitly fetch three pinned, attributed development examples (<10 MB each)."""

import hashlib
import json
import urllib.request
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "data/manifests/libri_examples.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        destination = (manifest_path.parent / entry["path"]).resolve()
        destination.relative_to(root / "data/raw")
        if not entry["url"].startswith("https://librosa.org/data/audio/"):
            raise ValueError("Only the official pinned example host is permitted")
        if destination.exists():
            data = destination.read_bytes()
        else:
            with urllib.request.urlopen(entry["url"], timeout=60) as response:
                data = response.read(10 * 1024 * 1024 + 1)
            if len(data) > 10 * 1024 * 1024:
                raise ValueError("Example exceeds download size limit")
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {destination.name}; existing file not overwritten")
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        print(f"Verified {destination.name}: {len(data)} bytes")


if __name__ == "__main__":
    main()
