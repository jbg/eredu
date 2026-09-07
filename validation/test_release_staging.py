"""Exercise real Cargo resolution without external dependencies or native builds."""
from contextlib import redirect_stdout
import gzip
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest

from validate_release_packages import (
    copy_release_source, index_path, normalize_source_times,
    stage_package, staging_directory, write_cargo_config,
)


class ReleaseStagingTests(unittest.TestCase):
    def package(self, name, dependencies=()):
        return {"name": name, "version": "1.0.0", "features": {},
                "dependencies": [{"name": dependency, "path": "/fixture/" + dependency,
                                  "req": "=1.0.0", "features": [], "optional": False,
                                  "uses_default_features": True}
                                 for dependency in dependencies]}

    def archive(self, root, package, source):
        manifest = (f'[package]\nname="{package["name"]}"\nversion="1.0.0"\n'
                    'edition="2021"\n[dependencies]\n')
        manifest += "\n".join(f'{d["name"]}="=1.0.0"' for d in package["dependencies"])
        archive = root / (package["name"] + ".crate")
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            for name, text in [("Cargo.toml", manifest), ("src/lib.rs", source)]:
                data = text.encode()
                info = tarfile.TarInfo(f'{package["name"]}-1.0.0/{name}')
                info.size = len(data)
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(data))
        archive.write_bytes(gzip.compress(buffer.getvalue(), mtime=0))
        return archive

    def test_exact_bytes_metadata_and_dependency_identity_determine_registry(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            leaf = self.package("fixture-leaf")
            parent = self.package("fixture-parent", ["fixture-leaf"])
            leaf_archive = self.archive(root, leaf, "pub fn value() -> u8 { 1 }")
            parent_archive = self.archive(root, parent, "pub use fixture_leaf::value;")
            first = stage_package(leaf, leaf_archive, root / "registries", [])
            self.assertEqual(first, stage_package(leaf, leaf_archive, root / "registries", []))
            first_parent = stage_package(parent, parent_archive, root / "registries", [first])
            self.archive(root, leaf, "pub fn value() -> u8 { 2 }")
            changed = stage_package(leaf, leaf_archive, root / "registries", [])
            self.assertNotEqual(first["index"], changed["index"])
            changed_parent = stage_package(parent, parent_archive, root / "registries", [changed])
            self.assertNotEqual(first_parent["index"], changed_parent["index"])
            record = json.loads((changed_parent["index"] / index_path(parent["name"])).read_text())
            self.assertEqual(record["deps"][0]["registry"], changed["index"].as_uri())
            leaf["features"] = {"extra": []}
            self.assertNotEqual(changed["index"], stage_package(leaf, leaf_archive, root / "registries", [])["index"])

    def test_source_identity_ignores_mtime_but_covers_contents_modes_and_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            subprocess.run(["git", "init", "-q", str(source)], check=True)
            file = source / "input.rs"
            file.write_text("first")
            def snapshot(number):
                return copy_release_source(source, root / f"snapshot-{number}")
            first = snapshot(1)
            os.utime(file, (1, 1))
            self.assertEqual(first, snapshot(2))
            file.write_text("second")
            second = snapshot(3)
            self.assertNotEqual(first, second)
            file.chmod(0o755)
            self.assertNotEqual(second, snapshot(4))
            link = source / "link"
            link.symlink_to("input.rs")
            linked = snapshot(5)
            link.unlink()
            link.symlink_to("missing.rs")
            self.assertNotEqual(linked, snapshot(6))

    def test_real_cargo_reuses_identical_archives_and_loads_changed_same_version(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            leaf = self.package("fixture-leaf")
            parent = self.package("fixture-parent", ["fixture-leaf"])
            environment = dict(os.environ, CARGO_HOME=str(root / "cargo-home"),
                               CARGO_TARGET_DIR=str(root / "target"), CARGO_INCREMENTAL="0")
            identities = []
            for iteration, value in enumerate([1, 1, 2]):
                leaf_archive = self.archive(root, leaf, f"pub fn value() -> u8 {{ {value} }}")
                parent_archive = self.archive(root, parent, "pub use fixture_leaf::value;")
                # Recreate both the stage and registry downloads as on a fresh
                # runner, retaining only target artifacts between iterations.
                shutil.rmtree(root / "cargo-home" / "registry", ignore_errors=True)
                with staging_directory(root / "stage") as stage:
                    first = stage_package(leaf, leaf_archive, stage / "registries", [])
                    second = stage_package(parent, parent_archive, stage / "registries", [first])
                    identities.append(second["index"])
                    config = stage / "config.toml"
                    write_cargo_config(config, [first, second])
                    consumer = stage / "consumer"
                    (consumer / "src").mkdir(parents=True)
                    (consumer / "Cargo.toml").write_text(
                        '[package]\nname="consumer"\nversion="1.0.0"\nedition="2021"\n'
                        '[dependencies]\nfixture-parent={version="=1.0.0",registry="fixture-parent"}\n')
                    (consumer / "src/main.rs").write_text('fn main() { println!("{}", fixture_parent::value()); }')
                    normalize_source_times(consumer)
                    result = subprocess.run(["cargo", "build", "--message-format=json", "--config", str(config)],
                                            cwd=consumer, env=environment, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    artifacts = [message for line in result.stdout.splitlines()
                                 if (message := json.loads(line)).get("reason") == "compiler-artifact"]
                    self.assertGreaterEqual(len(artifacts), 3)
                    if iteration == 1:
                        self.assertTrue(all(a["fresh"] for a in artifacts), result.stdout + result.stderr)
                    binary = next(a["executable"] for a in artifacts if a.get("executable"))
                    self.assertEqual(subprocess.check_output([binary], text=True).strip(), str(value))
                self.assertFalse((root / "stage" / "active").exists())
            self.assertEqual(identities[0], identities[1])
            self.assertNotEqual(identities[1], identities[2])

    def test_normal_cargo_package_retains_verification_and_reproducible_archives(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            root = Path(temporary)
            environment = dict(os.environ, CARGO_HOME=str(root / "cargo-home"),
                               CARGO_TARGET_DIR=str(root / "target"), CARGO_INCREMENTAL="0")
            packages = [self.package("fixture-leaf"), self.package("fixture-parent", ["fixture-leaf"])]
            # Cargo package consults the original registry even with patches.
            # Supply an empty local index so this fixture never needs network.
            empty = root / "empty-index"
            empty.mkdir()
            (empty / "config.json").write_text(json.dumps({"dl": root.as_uri()}))
            for args in [("init", "-q"), ("add", "config.json"),
                         ("-c", "user.name=Test", "-c", "user.email=test@invalid", "commit", "-qm", "empty")]:
                subprocess.run(["git", *args], cwd=empty, check=True)
            checksums = []
            for iteration in range(2):
                with staging_directory(root / "stage") as stage:
                    workspace = stage / "workspace"
                    workspace.mkdir()
                    (workspace / "Cargo.toml").write_text(
                        '[workspace]\nmembers=["fixture-leaf","fixture-parent"]\nresolver="2"\n')
                    for package in packages:
                        crate = workspace / package["name"]
                        (crate / "src").mkdir(parents=True)
                        manifest = f'[package]\nname="{package["name"]}"\nversion="1.0.0"\nedition="2021"\n'
                        if package["dependencies"]:
                            manifest += '[dependencies]\nfixture-leaf={path="../fixture-leaf",version="=1.0.0"}\n'
                        (crate / "Cargo.toml").write_text(manifest)
                        (crate / "src/lib.rs").write_text(
                            "pub use fixture_leaf::value;" if package["dependencies"] else "pub fn value() -> u8 { 1 }")
                    normalize_source_times(workspace)
                    config = stage / "config.toml"
                    staged = []
                    current = []
                    for package in packages:
                        write_cargo_config(config, staged)
                        with config.open("a") as handle:
                            handle.write('\n[source.crates-io]\nreplace-with="empty-fixture"\n'
                                         f'[source.empty-fixture]\nregistry="{empty.as_uri()}"\n')
                        result = subprocess.run(["cargo", "package", "-p", package["name"], "--config", str(config)],
                                                cwd=workspace, env=environment, capture_output=True, text=True)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertIn("Verifying", result.stderr)
                        archive = root / "target/package" / (package["name"] + "-1.0.0.crate")
                        entry = stage_package(package, archive, stage / "registries", staged)
                        staged.append(entry)
                        current.append(entry["index"])
                    checksums.append(current)
            self.assertEqual(checksums[0], checksums[1])


if __name__ == "__main__":
    unittest.main()
