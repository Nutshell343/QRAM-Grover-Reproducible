"""Check release hashes, or explicitly refresh them after intentional edits."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parent
EXCLUDED = {"runs", ".venv", "venv", "__pycache__", ".git", ".idea", ".vscode"}


def inventory():
    result = {}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.name == "bundle_manifest.json":
            continue
        relative = path.relative_to(ROOT)
        if EXCLUDED.intersection(relative.parts) or path.suffix == ".pyc":
            continue
        result[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--refresh", action="store_true", help="Accept intentional file edits as a new local inventory; does not alter original provenance or reference results")
    args = cli.parse_args()
    current = inventory()
    path = ROOT / "bundle_manifest.json"
    if args.refresh:
        path.write_text(json.dumps({"version": "2026-11-04", "scope": "Portable frozen coherent QRAM/Grover simulation", "sha256": current}, ensure_ascii=False, indent=2), encoding="utf8")
        print("Refreshed local inventory:", len(current), "files. Use a new run-name after scientific source/calibration edits.")
    else:
        expected = json.loads(path.read_text(encoding="utf8"))["sha256"]
        changed = [key for key in set(current) | set(expected) if current.get(key) != expected.get(key)]
        if changed:
            raise SystemExit("Missing, changed or extra release files: " + ", ".join(sorted(changed)))
        print("All", len(expected), "release files match.")


if __name__ == "__main__":
    main()
