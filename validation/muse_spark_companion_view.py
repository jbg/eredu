"""Create a partial local view of the pinned released media-source members.

The view uses symbolic links in the model repository's snapshot layout. No tensor
payloads are copied or read, and no existing files are removed. Admission and
payload verification remain separate operations.
"""
import argparse
import json
from pathlib import Path
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    evidence = json.loads((Path(__file__).parent / "fixtures/muse-spark/bf16-media-companion-admission.json").read_text())
    members = set()
    for item in evidence["retained_files"]:
        member = Path(item["member"])
        if member.is_absolute() or ".." in member.parts:
            raise ValueError("selected member escapes its checkpoint directory")
        if not (source / member).is_file():
            raise FileNotFoundError(source / member)
        members.add(member)
    repository = source.parent.parent if source.parent.name == "snapshots" else source
    (repository / "blobs").mkdir(exist_ok=True)
    snapshots = repository / "snapshots"
    snapshots.mkdir(exist_ok=True)
    view = Path(tempfile.mkdtemp(prefix="media-companion-", dir=snapshots))
    for member in sorted(members):
        (view / member).parent.mkdir(parents=True, exist_ok=True)
        (view / member).symlink_to(source / member)
    print(view)


if __name__ == "__main__":
    main()
