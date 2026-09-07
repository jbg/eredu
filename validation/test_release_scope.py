import tempfile
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from release_scope import PRIVATE, plan, public_surface, scoped_change
from rust_cache import expired, prune
from validate_release_packages import release_closure, registry_dependency


class ReleaseScopeTests(unittest.TestCase):
    path = "eredu-architectures/src/qwen/hybrid/checkpoint.rs"

    def scoped(self, before, after, path=None):
        return scoped_change(path or self.path, before, after, {"eredu", "eredu-architectures"})

    def test_body_fix_retains_public_signature(self):
        self.assertTrue(self.scoped("pub fn load(x: i32) -> i32 { x }",
                                    "pub fn load(x: i32) -> i32 { x + 1 }"))

    def test_multiline_api_type_change_requires_full(self):
        self.assertFalse(self.scoped("pub fn load(\n x: i32,\n) { }",
                                     "pub fn load(\n x: u32,\n) { }"))
        self.assertFalse(self.scoped("pub fn load() -> [u8; 4] { todo!() }",
                                     "pub fn load() -> [u8; 8] { todo!() }"))

    def test_enum_variant_and_struct_field_changes_require_full(self):
        for before, after in [("pub enum E { A }", "pub enum E { A, B }"),
                              ("pub struct S { pub x: i32 }", "pub struct S { pub x: u32 }")]:
            self.assertFalse(self.scoped(before, after))

    def test_scoped_private_helper_can_change(self):
        self.assertTrue(self.scoped("pub fn old() {}", "pub fn new(x: u8) {}", PRIVATE))
        # Opening or re-exporting the enclosing private module is unrecognized.
        self.assertFalse(self.scoped("pub(crate) mod grouped;", "pub mod grouped;",
                                     "eredu-backend-mlx/src/backend/nn/mod.rs"))

    def test_platform_feature_and_unknown_changes_require_full(self):
        self.assertFalse(self.scoped('#[cfg(feature = "a")] fn f() {}',
                                     '#[cfg(feature = "b")] fn f() {}'))
        self.assertFalse(self.scoped('fn f() { if cfg!(unix) {} }',
                                     'fn f() { if cfg!(windows) {} }'))
        for path in ("safemlx-sys/build.rs", "rust-toolchain.toml", ".github/workflows/linux.yml",
                     "eredu-core/src/lib.rs", "eredu-architectures/src/new.rs"):
            self.assertFalse(self.scoped("before", "after", path))

    def test_additions_and_deletions_require_full(self):
        self.assertFalse(self.scoped(None, "fn f() {}"))
        self.assertFalse(self.scoped("fn f() {}", None))

    def test_version_bump_does_not_hide_feature_change(self):
        old = '[package]\nname="eredu"\nversion="0.3.0"\n[features]\nmlx=[]\n'
        new = old.replace("0.3.0", "0.3.1")
        self.assertTrue(self.scoped(old, new, "eredu/Cargo.toml"))
        self.assertFalse(self.scoped(old, new.replace("mlx=[]", 'mlx=["dep:backend"]'),
                                     "eredu/Cargo.toml"))

    def test_external_dependency_updates_require_full(self):
        old = '[dependencies]\nserde="1"\n'
        self.assertFalse(self.scoped(old, old.replace('"1"', '"2"'), "eredu/Cargo.toml"))

    def test_public_dependency_type_identity_changes_require_full(self):
        old = '[package]\nname="eredu"\nversion="0.3.0"\n'
        self.assertFalse(self.scoped(old, old.replace("0.3.0", "0.4.0"), "eredu/Cargo.toml"))
        old = '[workspace.dependencies]\neredu={version="0.3.0",path="eredu"}\n'
        self.assertTrue(self.scoped(old, old.replace("0.3.0", "0.3.1"), "Cargo.toml"))
        self.assertFalse(self.scoped(old, old.replace("0.3.0", "0.4.0"), "Cargo.toml"))

    def test_new_test_does_not_force_full(self):
        self.assertTrue(self.scoped("fn f() {}", "fn f() {}\n#[test]\nfn checks() {}"))

    def test_test_attributes_cannot_hide_production_api_changes(self):
        for attribute in ("#[test]", "#[cfg(test)]"):
            self.assertFalse(self.scoped("pub fn f() {}", attribute + "\npub fn f() {}"))

    def test_existing_macro_does_not_force_full_but_changing_it_does(self):
        macro = "macro_rules! m { () => { 1 }; } "
        self.assertTrue(self.scoped(macro + "fn f() { 1 }", macro + "fn f() { 2 }"))
        self.assertFalse(self.scoped(macro, macro.replace("1", "2")))

    def test_comments_do_not_change_api(self):
        self.assertEqual(public_surface("// pub fn fake() {}\npub fn f() {}"),
                         public_surface("/* pub fn fake() {} */ pub fn f() { 42; }"))


class ReleaseArchiveSelectionTests(unittest.TestCase):
    def packages(self):
        return {"eredu-architectures": {"version": "0.3.1", "dependencies": []},
                "eredu-backend-mlx": {"version": "0.3.1", "dependencies": [{"name": "eredu-architectures"}]},
                "eredu": {"version": "0.3.1", "dependencies": [{"name": "eredu-backend-mlx"}]},
                "eredu-cli": {"version": "0.1.4", "dependencies": [{"name": "eredu"}]}}

    def test_published_dependencies_are_not_repackaged(self):
        self.assertEqual(release_closure(self.packages(), ["eredu-cli"], lambda *_: True), ["eredu-cli"])

    def test_unpublished_dependency_closure_is_topological(self):
        self.assertEqual(release_closure(self.packages(), ["eredu-cli"], lambda *_: False),
                         list(self.packages()))

    def test_closure_stops_at_published_dependencies(self):
        self.assertEqual(release_closure(self.packages(), ["eredu-cli"],
                                        lambda name, _: name == "eredu-backend-mlx"),
                         ["eredu", "eredu-cli"])

    def test_empty_and_unknown_release_sets(self):
        self.assertEqual(release_closure(self.packages(), []), [])
        with self.assertRaisesRegex(RuntimeError, "unknown release"):
            release_closure(self.packages(), ["misspelled"])

    def test_unchanged_path_dependency_uses_crates_io(self):
        dependency = {"name": "eredu-core", "path": "/workspace/eredu-core", "req": "^0.3.0",
                      "features": [], "optional": False, "uses_default_features": True, "target": None}
        record = registry_dependency(dependency, {"eredu"})
        self.assertEqual(record["registry"], "https://github.com/rust-lang/crates.io-index")
        self.assertIsNone(registry_dependency(dependency, {"eredu-core"})["registry"])

    def test_past_scoped_release_is_not_repackaged_and_version_followup_keeps_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def git(*args):
                return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
            def write(path, contents):
                file = root / path
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text(contents)
            def commit():
                git("add", ".")
                git("-c", "user.name=Test", "-c", "user.email=test@invalid", "commit", "-qm", "fixture")
                return git("rev-parse", "HEAD")
            git("init", "-q")
            architecture = "eredu-architectures/src/qwen/hybrid/checkpoint.rs"
            binding = "eredu-backend-mlx/src/backend/runtime/checkpoint/binding.rs"
            write(architecture, "pub fn recipe() -> u8 { 0 }")
            write(binding, "fn bind() { }")
            write("eredu-backend-mlx/Cargo.toml", '[package]\nname="eredu-backend-mlx"\nversion="0.3.1"\n')
            base = commit()
            write(architecture, "pub fn recipe() -> u8 { 1 }")
            commit()
            git("tag", "eredu-architectures-v0.3.1")
            write(binding, "fn bind() { let _ = 1; }")
            commit()
            write("eredu-backend-mlx/src/platform.rs", "fn new_platform_code() {}")
            commit()
            write("eredu-backend-mlx/Cargo.toml", '[package]\nname="eredu-backend-mlx"\nversion="0.3.2"\n')
            head = commit()
            packages = {name: {"manifest_path": str(root / name / "Cargo.toml"), "version": version}
                        for name, version in [("eredu-architectures", "0.3.1"), ("eredu-backend-mlx", "0.3.2")]}
            with patch("release_scope.ROOT", root):
                result = plan(base, head, packages)
            self.assertEqual(result["packages"], ["eredu-backend-mlx"])
            # A later version-only commit cannot turn the preceding
            # unrecognized platform change into scoped verification.
            self.assertTrue(result["full"])


class RustCacheTests(unittest.TestCase):
    def test_snapshot_bound_preserves_source_outside_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "source.rs").write_text("source")
            target = root / "target"
            target.mkdir()
            (target / "old.rlib").write_bytes(b"x" * 32)
            (target / "new.rlib").write_bytes(b"y" * 32)
            (target / "incremental").mkdir()
            (target / "incremental" / "big").write_bytes(b"z" * 128)
            self.assertLessEqual(prune(target, 32), 32)
            self.assertEqual((root / "source.rs").read_text(), "source")
            self.assertFalse((target / "incremental").exists())

    def test_retention_keeps_latest_per_configuration_and_bounds_total(self):
        def cache(i, key, size):
            return {"id": i, "key": key, "size_in_bytes": size, "created_at": f"2026-09-{i:02}"}
        caches = [cache(1, "native-v1-native", 9999), cache(2, "rust-v2-old", 100),
                  cache(3, "rust-v3-mac--old--sha", 100), cache(4, "rust-v3-mac--new--sha", 100),
                  cache(5, "rust-v3-ios--new--sha", 100)]
        self.assertEqual(set(expired(caches, 200)), {2, 3})
        self.assertEqual(set(expired(caches, 100)), {2, 3, 4})


if __name__ == "__main__":
    unittest.main()
