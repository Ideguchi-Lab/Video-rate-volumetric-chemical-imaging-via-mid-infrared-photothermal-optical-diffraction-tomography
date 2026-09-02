"""Download the dataset into $MIPODT_DATA_DIR (default: ./data).

Fill in ``FILES`` once the data is deposited (Zenodo / figshare). Each entry
is (relative path, URL, sha256).
"""

from __future__ import annotations

import hashlib
import urllib.request

from mipodt_paper.paths import DATA_DIR

FILES: list[tuple[str, str, str]] = [
    # ("fig1/raw_holograms.tif", "https://zenodo.org/record/XXXX/files/...", "<sha256>"),
]


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not FILES:
        raise SystemExit("FILES is empty: the dataset has not been published yet.")
    for rel, url, digest in FILES:
        dest = DATA_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and sha256(dest) == digest:
            print(f"ok       {rel}")
            continue
        print(f"download {rel}")
        urllib.request.urlretrieve(url, dest)
        if sha256(dest) != digest:
            raise SystemExit(f"checksum mismatch for {rel}")


if __name__ == "__main__":
    main()
