#!/usr/bin/env python3
"""Conservative release scope; unknown changes always require the full matrix."""
from __future__ import annotations

import argparse
import fnmatch
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tomllib
import zipfile

from validate_release_packages import cargo_metadata, publishable_packages, RELEASE_ORDER

ROOT = Path(__file__).resolve().parents[1]
# Bootstrap from the last full main gate before scoped verification existed.
# New successful full runs supersede it through their release-plan artifact.
BOOTSTRAP_BASE = "084458c34eabd80eb3bf8d85d13d3c7ea1c5b499"
SCOPED = (
    "eredu-architectures/src/qwen/hybrid/checkpoint.rs",
    "eredu-architectures/src/replicated_text.rs",
    "eredu-backend-mlx/src/backend/nn/grouped/gated_product.rs",
    "eredu-backend-mlx/src/backend/nn/shared/extensions.rs",
    "eredu-backend-mlx/src/backend/runtime/checkpoint/binding.rs",
    "eredu-backend-mlx/src/backend/runtime/checkpoint/binding/*.rs",
    "eredu-backend-mlx/src/composition/mlx/replicated_text/lowering.rs",
    "eredu-backend-mlx/src/composition/mlx/replicated_text/tests/*.rs",
    "eredu-backend-mlx/src/tests/*.rs",
)
# This implementation is behind pub(crate) mod grouped. Changing that module
# boundary or adding a re-export changes a file outside SCOPED and forces full.
PRIVATE = "eredu-backend-mlx/src/backend/nn/grouped/gated_product.rs"
TOKEN = re.compile(r'//[^\n]*|/\*.*?\*/|r\#*".*?"\#*|"(?:\\.|[^"\\])*"|\w+|[^\s]', re.S)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def public_surface(source: str) -> list[list[str]]:
    """Compare public declarations, including multiline signatures and fields.

    Deliberately over-selects: a public struct's private field edit also requires
    full verification. This is a scope guard, not a Rust API compatibility proof.
    """
    tokens = [m.group() for m in TOKEN.finditer(source)
              if not m.group().startswith(("//", "/*"))]
    declarations = []
    for start, token in enumerate(tokens):
        if token != "pub" or tokens[start + 1:start + 2] == ["("]:
            continue
        # Attributes can hide or change a public item even when its signature
        # stays identical (for example adding #[test] or #[cfg(test)]).
        attribute_start = start
        while attribute_start > 0 and tokens[attribute_start - 1] == "]":
            position = attribute_start - 1
            nested = 1
            while position > 0 and nested:
                position -= 1
                nested += (tokens[position] == "]") - (tokens[position] == "[")
            if position == 0 or tokens[position - 1] != "#":
                break
            attribute_start = position - 1
        declaration = tokens[attribute_start:start]
        depth = 0
        brackets = 0
        parentheses = 0
        angles = 0
        aggregate = False
        for item in tokens[start:]:
            declaration.append(item)
            aggregate |= item in ("struct", "enum", "trait", "union")
            brackets += (item == "[") - (item == "]")
            parentheses += (item == "(") - (item == ")")
            if depth == 0:
                angles = max(0, angles + (item == "<") - (item == ">"))
            if item == "{":
                if not aggregate:
                    if brackets or parentheses or angles:
                        # Const expressions in signatures need a real Rust
                        # parser. Conservatively include the remainder so any
                        # edit here requires full verification.
                        declaration.extend(tokens[attribute_start + len(declaration):])
                    break
                depth += 1
            elif item == "}":
                depth -= 1
                if depth == 0:
                    break
            elif item == ";" and depth == 0 and brackets == 0 and parentheses == 0:
                break
        declarations.append(declaration)
    return declarations


def opaque_surface(source: str) -> list[list[str]]:
    """Macro definitions and extern blocks cannot use the ordinary API guard."""
    tokens = [m.group() for m in TOKEN.finditer(source)
              if not m.group().startswith(("//", "/*"))]
    blocks = []
    for start, token in enumerate(tokens):
        if token not in ("macro_rules", "extern"):
            continue
        block = []
        stack = []
        for item in tokens[start:]:
            block.append(item)
            if item in ("{", "(", "["):
                stack.append(item)
            elif item in ("}", ")", "]"):
                if stack:
                    stack.pop()
                if not stack:
                    break
            elif item == ";" and not stack:
                break
        blocks.append(block)
    return blocks


def normalized_manifest(text: str, members: set[str]) -> dict:
    value = tomllib.loads(text)
    package = value.get("package", {})
    if "version" in package:
        package["version"] = patch_identity(package["version"])
    sections = [value, value.get("workspace", {})]
    sections.extend(value.get("target", {}).values())
    for section in sections:
        for kind in ("dependencies", "dev-dependencies", "build-dependencies"):
            for name, dep in section.get(kind, {}).items():
                if isinstance(dep, dict) and dep.get("package", name) in members:
                    if "version" in dep:
                        dep["version"] = patch_identity(dep["version"])
    return value


def patch_identity(version):
    # Internal patch minimums can follow a bounded fix; changing a crate's
    # compatibility line can change public dependency type identities.
    if isinstance(version, str) and (match := re.fullmatch(r"([\^=]?)(\d+)\.(\d+)\.\d+", version)):
        prefix, major, minor = match.groups()
        if major != "0" or minor != "0":
            return f"{prefix}{major}.{minor}.*"
    return version


def scoped_change(path: str, before: str | None, after: str | None, members: set[str]) -> bool:
    if path.startswith("doc/") or path.endswith(".md"):
        return True
    if before is None or after is None:
        return False
    if path.endswith("Cargo.toml"):
        return normalized_manifest(before, members) == normalized_manifest(after, members)
    if path == "Cargo.lock":
        def normalized(text):
            data = tomllib.loads(text)
            for package in data.get("package", []):
                if package["name"] in members and "source" not in package:
                    package["version"] = patch_identity(package["version"])
            return data
        return normalized(before) == normalized(after)
    if not any(fnmatch.fnmatchcase(path, pattern) for pattern in SCOPED):
        return False
    if opaque_surface(before) != opaque_surface(after):
        return False
    if re.findall(r"cfg\s*!\s*\([^;]*?\)", before) != re.findall(r"cfg\s*!\s*\([^;]*?\)", after):
        return False
    # Conditional compilation and macros can change platform/feature surfaces.
    def attributes(text):
        return re.findall(r"#\s*\[[\s\S]*?\]", text)
    if attributes(before) != attributes(after):
        # New tests are expected; other attribute changes require full coverage.
        def production_attrs(text):
            return [a for a in attributes(text) if a != "#[test]"]
        if production_attrs(before) != production_attrs(after):
            return False
    if path == PRIVATE or "/tests/" in path or path.endswith("_tests.rs"):
        return True
    return public_surface(before) == public_surface(after)


def full_baseline(head: str) -> str:
    """Only successful full main runs can certify unchanged platform coverage."""
    runs = json.loads(subprocess.check_output([
        "gh", "run", "list", "--workflow", "native-release-gate.yml", "--branch", "main",
        "--status", "success", "--limit", "50", "--json", "databaseId,headSha,event",
    ], cwd=ROOT))
    for run in runs:
        sha = run["headSha"]
        if run["event"] not in ("push", "workflow_dispatch", "schedule"):
            continue
        if subprocess.run(["git", "merge-base", "--is-ancestor", sha, head], cwd=ROOT).returncode:
            continue
        if sha == BOOTSTRAP_BASE:
            return sha
        artifacts = json.loads(subprocess.check_output([
            "gh", "api", f"repos/{{owner}}/{{repo}}/actions/runs/{run['databaseId']}/artifacts",
        ], cwd=ROOT))
        for artifact in artifacts["artifacts"]:
            if artifact["name"] == "release-plan" and not artifact["expired"]:
                archive = subprocess.check_output([
                    "gh", "api", f"repos/{{owner}}/{{repo}}/actions/artifacts/{artifact['id']}/zip",
                ], cwd=ROOT)
                with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
                    plan = json.loads(zipped.read("release-plan.json"))
                if plan["full"] and plan["head"] == sha:
                    return sha
    return BOOTSTRAP_BASE


def plan(base: str, head: str, packages: dict, force_full: bool = False) -> dict:
    head = git("rev-parse", head)
    if subprocess.run(["git", "merge-base", "--is-ancestor", base, head], cwd=ROOT).returncode:
        raise RuntimeError("full verification baseline must be an ancestor of the candidate")
    paths = git("diff", "--name-only", base, head).splitlines()
    members = set(packages)
    reasons = []
    selected = set()
    # Scope is cumulative since full verification, but archive roots are pending
    # changes since each crate's own release. A previous scoped release must not
    # force its already-published crates into every subsequent patch release.
    for name, package in packages.items():
        directory = Path(package["manifest_path"]).parent.relative_to(ROOT).as_posix()
        tag = subprocess.run(["git", "rev-parse", "--verify", f"refs/tags/{name}-v{package['version']}^{{commit}}"],
                             cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        reference = base
        if tag.returncode == 0:
            released = tag.stdout.strip()
            if released != head and subprocess.run(
                ["git", "merge-base", "--is-ancestor", released, head], cwd=ROOT
            ).returncode == 0:
                reference = released
        if git("diff", "--name-only", reference, head, "--", directory):
            selected.add(name)
    for path in paths:
        def read(revision):
            result = subprocess.run(["git", "show", f"{revision}:{path}"], cwd=ROOT,
                                    text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            return result.stdout if result.returncode == 0 else None
        if not scoped_change(path, read(base), read(head), members):
            reasons.append(path)
    changed_lines = sum(int(value) for row in git("diff", "--numstat", base, head).splitlines()
                        for value in row.split("\t")[:2] if value.isdigit())
    if changed_lines > 1000 or len(paths) > 20:
        reasons.append("change exceeds the scoped limit (1,000 changed lines / 20 files)")
    full = force_full or bool(reasons)
    # Infrastructure-only changes and scheduled audits still exercise packaging.
    if force_full or (not selected and full):
        selected = members
    return {"base": base, "head": head, "full": full,
            "reasons": reasons or (["explicit full audit"] if force_full else []),
            "packages": [name for name in RELEASE_ORDER if name in selected]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="override the last successful full main baseline")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("release-plan.json"))
    args = parser.parse_args()
    head = git("rev-parse", args.head)
    result = plan(args.base or full_baseline(head), head,
                  publishable_packages(cargo_metadata(ROOT)), args.full)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a") as stream:
            stream.write(f"full={str(result['full']).lower()}\n")
            stream.write(f"packages={json.dumps(result['packages'])}\n")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a") as stream:
            stream.write("### Release verification plan\n```json\n" + json.dumps(result, indent=2) + "\n```\n")


if __name__ == "__main__":
    main()
