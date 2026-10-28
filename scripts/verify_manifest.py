
from __future__ import annotations

import sys

from cqfa.paths import ROOT

from build_manifest import MANIFEST, digest, iter_files  # noqa: E402


def main() -> int:
    if not MANIFEST.exists():
        print(f"  {MANIFEST.name} not found; run scripts/build_manifest.py first")
        return 1

    recorded = {}
    for line in MANIFEST.read_text().splitlines():
        if line.strip():
            h, _, rel = line.partition("  ")
            recorded[rel] = h

    on_disk = {p.relative_to(ROOT).as_posix(): p for p in iter_files()}

    changed = [r for r in recorded.keys() & on_disk.keys()
               if digest(on_disk[r]) != recorded[r]]
    missing = sorted(recorded.keys() - on_disk.keys())
    extra = sorted(on_disk.keys() - recorded.keys())

    print(f"  {len(recorded)} files in manifest, {len(on_disk)} on disk")
    print(f"  changed: {len(changed)}   missing: {len(missing)}   unlisted: {len(extra)}")
    for r in (changed + missing + extra)[:20]:
        tag = "CHANGED" if r in changed else ("MISSING" if r in missing else "UNLISTED")
        print(f"    {tag}  {r}")

    ok = not (changed or missing or extra)
    print("\n  manifest verified" if ok else "\n  manifest does NOT match")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.exit(main())
