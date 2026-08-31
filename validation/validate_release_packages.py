#!/usr/bin/env python3
"""Build every publishable workspace crate against an ephemeral local registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Any


# This is also the documented publication order. Dependency order includes
# development dependencies because crates.io validates their version metadata.
RELEASE_ORDER = (
    "eredu-gguf",
    "safemlx-internal-macros",
    "eredu-backend-mlx-macros",
    "safemlx-sys",
    "eredu-nn-macros",
    "eredu-checkpoint",
    "eredu-core",
    "eredu-text",
    "safemlx",
    "eredu-nn",
    "eredu-runtime",
    "eredu-architectures",
    "eredu-codec",
    "eredu-evaluation",
    "eredu-backend-mlx",
    "eredu",
    "eredu-cli",
)

MAX_ARCHIVE_BYTES = 10 * 1024 * 1024


def run(
    command: list[str],
    *,
    cwd: Path,
    capture: bool = False,
    env: dict[str, str] | None = None,
) -> str:
    print(f"+ {shlex.join(command)}", flush=True)
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        env=env,
    )
    return result.stdout if capture else ""


def cargo_metadata(workspace: Path) -> dict[str, Any]:
    return json.loads(
        run(
            ["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"],
            cwd=workspace,
            capture=True,
        )
    )


def publishable_packages(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    members = set(metadata["workspace_members"])
    return {
        package["name"]: package
        for package in metadata["packages"]
        if package["id"] in members and package["publish"] != []
    }


def validate_release_order(packages: dict[str, dict[str, Any]]) -> None:
    expected = set(packages)
    actual = set(RELEASE_ORDER)
    if len(actual) != len(RELEASE_ORDER):
        raise RuntimeError("release order contains a duplicate crate")
    if expected != actual:
        missing = sorted(expected - actual)
        stale = sorted(actual - expected)
        raise RuntimeError(
            f"release order does not match publishable crates; missing={missing}, stale={stale}"
        )

    positions = {name: position for position, name in enumerate(RELEASE_ORDER)}
    for package_name, package in packages.items():
        for dependency in package["dependencies"]:
            dependency_name = dependency["name"]
            if dependency_name in packages:
                if positions[dependency_name] >= positions[package_name]:
                    raise RuntimeError(
                        f"{dependency_name} must precede {package_name} in the release order"
                    )
            elif dependency.get("path") is not None:
                raise RuntimeError(
                    f"publishable crate {package_name} depends on non-publishable "
                    f"workspace crate {dependency_name}"
                )


def copy_release_source(source: Path, destination: Path) -> None:
    """Copy Git package candidates without the enormous build directory."""
    files = run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=source,
        capture=True,
    ).split("\0")
    for relative_text in files:
        if not relative_text:
            continue
        relative = Path(relative_text)
        source_file = source / relative
        if not source_file.exists() and not source_file.is_symlink():
            continue
        destination_file = destination / relative
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        if source_file.is_symlink():
            destination_file.symlink_to(os.readlink(source_file))
        else:
            shutil.copy2(source_file, destination_file)


def index_path(crate_name: str) -> Path:
    name = crate_name.lower()
    if len(name) == 1:
        return Path("1") / name
    if len(name) == 2:
        return Path("2") / name
    if len(name) == 3:
        return Path("3") / name[0] / name
    return Path(name[:2]) / name[2:4] / name


def split_features(
    features: dict[str, list[str]],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Split index-v1 features from names requiring the index-v2 grammar."""
    modern = {
        name
        for name, values in features.items()
        if any(value.startswith("dep:") or "?/" in value for value in values)
    }
    changed = True
    while changed:
        changed = False
        for name, values in features.items():
            feature_references = {
                value for value in values if "/" not in value and not value.startswith("dep:")
            }
            if name not in modern and feature_references & modern:
                modern.add(name)
                changed = True
    old = {name: values for name, values in features.items() if name not in modern}
    new = {name: values for name, values in features.items() if name in modern}
    return old, new


def registry_dependency(
    dependency: dict[str, Any], publishable_names: set[str]
) -> dict[str, Any]:
    actual_name = dependency["name"]
    renamed = dependency.get("rename")
    source = dependency.get("source")
    if actual_name in publishable_names:
        registry = None
    elif source and source.startswith("registry+"):
        registry = source.removeprefix("registry+")
    elif dependency.get("path") is not None:
        raise RuntimeError(f"cannot stage non-publishable path dependency {actual_name}")
    else:
        raise RuntimeError(f"cannot stage non-registry dependency {actual_name}: {source}")

    result: dict[str, Any] = {
        "name": renamed or actual_name,
        "req": dependency["req"],
        "features": dependency["features"],
        "optional": dependency["optional"],
        "default_features": dependency["uses_default_features"],
        "target": dependency.get("target"),
        "kind": dependency.get("kind") or "normal",
        "registry": registry,
    }
    if renamed:
        result["package"] = actual_name
    return result


def registry_record(
    package: dict[str, Any], checksum: str, publishable_names: set[str]
) -> dict[str, Any]:
    old_features, new_features = split_features(package["features"])
    record: dict[str, Any] = {
        "name": package["name"],
        "vers": package["version"],
        "deps": [
            registry_dependency(dependency, publishable_names)
            for dependency in package["dependencies"]
        ],
        "cksum": checksum,
        "features": old_features,
        "yanked": False,
    }
    if package.get("links") is not None:
        record["links"] = package["links"]
    if package.get("rust_version") is not None:
        record["rust_version"] = package["rust_version"]
    if new_features:
        record["features2"] = new_features
        record["v"] = 2
    return record


def write_cargo_config(config: Path, index: Path, staged: list[dict[str, Any]]) -> None:
    lines = [
        "[registries.staged]",
        f'index = {json.dumps(index.as_uri())}',
    ]
    if staged:
        lines.extend(["", "[patch.crates-io]"])
        for package in staged:
            lines.append(
                f'{json.dumps(package["name"])} = '
                f'{{ version = "={package["version"]}", registry = "staged" }}'
            )
    config.write_text("\n".join(lines) + "\n", encoding="utf-8")


def git_commit_index(index: Path, relative_path: Path, message: str) -> None:
    run(["git", "add", "config.json", str(relative_path)], cwd=index)
    run(
        [
            "git",
            "-c",
            "user.name=Eredu release validation",
            "-c",
            "user.email=release-validation@invalid",
            "commit",
            "--quiet",
            "-m",
            message,
        ],
        cwd=index,
    )


def stage_package(
    package: dict[str, Any],
    archive: Path,
    index: Path,
    downloads: Path,
    publishable_names: set[str],
) -> None:
    name = package["name"]
    version = package["version"]
    download = downloads / name / version / "download"
    download.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(archive, download)
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()

    relative_index_path = index_path(name)
    index_file = index / relative_index_path
    index_file.parent.mkdir(parents=True, exist_ok=True)
    record = registry_record(package, checksum, publishable_names)
    with index_file.open("a", encoding="utf-8") as handle:
        json.dump(record, handle, separators=(",", ":"), sort_keys=True)
        handle.write("\n")
    git_commit_index(index, relative_index_path, f"Stage {name} {version}")


def validate_packaged_tests(
    package: dict[str, Any],
    archive: Path,
    destination: Path,
    config: Path,
    environment: dict[str, str],
) -> None:
    library_kinds = {"lib", "proc-macro"}
    if not any(library_kinds.intersection(target["kind"]) for target in package["targets"]):
        return

    shutil.unpack_archive(archive, destination, "gztar")
    package_root = destination / f'{package["name"]}-{package["version"]}'
    run(
        [
            "cargo",
            "test",
            "--no-run",
            "--lib",
            "--no-default-features",
            "--config",
            str(config),
        ],
        cwd=package_root,
        env=environment,
    )
    run(
        [
            "cargo",
            "test",
            "--doc",
            "--no-default-features",
            "--config",
            str(config),
        ],
        cwd=package_root,
        env=environment,
    )


def validate_downstream_consumer(
    package: dict[str, Any],
    destination: Path,
    config: Path,
    environment: dict[str, str],
) -> None:
    library_kinds = {"lib", "proc-macro"}
    if not any(library_kinds.intersection(target["kind"]) for target in package["targets"]):
        return

    consumer = destination / package["name"]
    source = consumer / "src"
    source.mkdir(parents=True)
    (consumer / "Cargo.toml").write_text(
        "\n".join(
            [
                "[package]",
                f'name = "{package["name"]}-release-consumer"',
                'version = "0.0.0"',
                'edition = "2021"',
                "",
                "[dependencies]",
                f'{package["name"]} = {{ version = "={package["version"]}", '
                'registry = "staged", default-features = false }',
                "",
            ]
        ),
        encoding="utf-8",
    )
    (source / "lib.rs").write_text("", encoding="utf-8")
    run(
        ["cargo", "check", "--config", str(config)],
        cwd=consumer,
        env=environment,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="validate tracked and untracked working-tree source instead of requiring a clean tree",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    workspace = Path(__file__).resolve().parents[1]
    status = run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=workspace,
        capture=True,
    )
    if status and not arguments.allow_dirty:
        raise RuntimeError("working tree is dirty; commit changes or pass --allow-dirty")

    metadata = cargo_metadata(workspace)
    packages = publishable_packages(metadata)
    validate_release_order(packages)
    publishable_names = set(packages)

    sizes: list[tuple[str, int]] = []
    with tempfile.TemporaryDirectory(prefix="eredu-release-packages-") as temporary:
        root = Path(temporary)
        release_workspace = root / "workspace"
        index = root / "index"
        downloads = root / "downloads"
        target_dir = root / "target"
        release_workspace.mkdir()
        index.mkdir()
        downloads.mkdir()
        target_dir.mkdir()
        copy_release_source(workspace, release_workspace)

        (index / "config.json").write_text(
            json.dumps({"dl": downloads.as_uri(), "api": "http://127.0.0.1:9"}) + "\n",
            encoding="utf-8",
        )
        run(["git", "init", "--quiet", "--initial-branch=main"], cwd=index)
        run(["git", "add", "config.json"], cwd=index)
        run(
            [
                "git",
                "-c",
                "user.name=Eredu release validation",
                "-c",
                "user.email=release-validation@invalid",
                "commit",
                "--quiet",
                "-m",
                "Initialize staged registry",
            ],
            cwd=index,
        )

        config = root / "cargo-config.toml"
        staged: list[dict[str, Any]] = []
        write_cargo_config(config, index, staged)
        environment = os.environ.copy()
        environment["CARGO_TARGET_DIR"] = str(target_dir)

        for crate_name in RELEASE_ORDER:
            package = packages[crate_name]
            print(f"\n==> Packaging {crate_name} {package['version']}", flush=True)
            run(
                [
                    "cargo",
                    "package",
                    "--quiet",
                    "-p",
                    crate_name,
                    "--config",
                    str(config),
                ],
                cwd=release_workspace,
                env=environment,
            )
            archive = target_dir / "package" / f"{crate_name}-{package['version']}.crate"
            size = archive.stat().st_size
            sizes.append((crate_name, size))
            if size > MAX_ARCHIVE_BYTES:
                raise RuntimeError(
                    f"{archive.name} is {size:,} bytes; crates.io permits at most "
                    f"{MAX_ARCHIVE_BYTES:,} bytes"
                )
            print(f"    archive size: {size / 1024:.1f} KiB", flush=True)
            validate_packaged_tests(
                package,
                archive,
                root / "unit-tests",
                config,
                environment,
            )
            stage_package(package, archive, index, downloads, publishable_names)
            staged.append(package)
            write_cargo_config(config, index, staged)
            validate_downstream_consumer(
                package,
                root / "downstream-consumers",
                config,
                environment,
            )

    print("\nValidated publishable crate archives:")
    for crate_name, size in sizes:
        print(f"  {crate_name:<28} {size / 1024:>9.1f} KiB")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
