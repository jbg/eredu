"""Check exact coverage and failure propagation using a real Rust test harness."""

from pathlib import Path
import os
import json
import time
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from run_native_unit_tests import run_batches, select_tests, test_names


class NativeUnitBatchTests(unittest.TestCase):
    def test_exact_names_ignore_policy_and_failure_propagation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "tests.rs"
            source.write_text('''
                fn record(name: &str) {
                    use std::io::Write;
                    let mut file = std::fs::OpenOptions::new().create(true).append(true)
                        .open(std::env::var("EREDU_BATCH_TEST_LOG").unwrap()).unwrap();
                    writeln!(file, "{name}").unwrap();
                }
                fn cache_marker() -> std::path::PathBuf {
                    std::path::PathBuf::from(std::env::var("EREDU_PREPARED_CHECKPOINT_CACHE").unwrap())
                        .join("marker")
                }
                #[test] fn a() { record("a"); std::fs::write(cache_marker(), "warm").unwrap(); }
                #[test] fn a_longer() { record("a_longer"); panic!("must propagate"); }
                #[test] #[ignore] fn ignored() { panic!("must stay ignored"); }
                #[test] fn z_after_failure() {
                    assert!(!cache_marker().exists(), "each batch starts with a fresh cache");
                    record("after_failure");
                }
            ''')
            executable = str(root / "tests")
            subprocess.run(["rustc", "--test", str(source), "-o", executable], check=True)
            listing = subprocess.check_output([executable, "--list"], text=True)
            self.assertEqual(test_names(listing), ["a", "a_longer", "ignored", "z_after_failure"])
            log = root / "executed"
            with patch.dict(os.environ, {"EREDU_BATCH_TEST_LOG": str(log)}):
                self.assertEqual(run_batches(executable, 1), 1)
                # Every ordinary test executes once, including after a failure;
                # exact names prevent `a` from also selecting `a_longer`.
                self.assertEqual(log.read_text().splitlines(), ["a", "a_longer", "after_failure"])
                log.unlink()
                self.assertEqual(run_batches(executable, 1, fail_fast=True), 1)
                self.assertEqual(log.read_text().splitlines(), ["a", "a_longer"])


class RingBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        source = cls.root / "ring.rs"
        source.write_text('''
            #[test] fn ordinary() {}
            mod tests {
                pub mod distributed_fixture_ring {
                    #[test] #[ignore] fn a() {}
                    #[test] #[ignore] fn b() {}
                    #[test] #[ignore] fn c() {}
                    #[test] fn worker() {}
                    #[test] #[ignore] fn timeout() {
                        use std::process::Command;
                        let marker = std::env::var("EREDU_ORPHAN_MARKER").unwrap();
                        let _child = Command::new("sh").arg("-c")
                            .arg(r#"sleep 1; echo orphan > "$1""#).arg("sh").arg(marker)
                            .spawn().unwrap();
                        println!("worker started");
                        std::thread::sleep(std::time::Duration::from_secs(20));
                    }
                }
                mod distributed_realtime_ring {
                    #[test] #[ignore] fn moshi_ring_tp2_native_model_parity() { panic!(); }
                    #[test] #[ignore] fn moshi_ring_tp2_personaplex_model_parity() { panic!(); }
                }
            }
        ''')
        cls.executable = str(cls.root / "ring")
        subprocess.run(["rustc", "--test", str(source), "-o", cls.executable], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_ring_shards_cover_only_ignored_synthetic_tests_once(self):
        shards = [select_tests(self.executable, "ring", i, 3, []) for i in range(3)]
        names = [name for shard in shards for name in shard]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(sorted(name.rsplit("::", 1)[1] for name in names),
                         ["a", "b", "c", "timeout"])
        ordinary = select_tests(self.executable, "unit", 0, 1, [])
        self.assertEqual(ordinary, ["ordinary", "tests::distributed_fixture_ring::worker"])
        with self.assertRaisesRegex(RuntimeError, "matched nothing"):
            select_tests(self.executable, "ring", 0, 1, ["nonexistent"])
        with self.assertRaisesRegex(RuntimeError, "missing ignored Ring smoke tests"):
            select_tests(self.executable, "ring-smoke", 0, 1, [])

    def test_ring_runs_ignored_tests_and_records_exact_selection(self):
        results = self.root / "results"
        self.assertEqual(run_batches(self.executable, 1, suite="ring",
                                     filters=["::a", "::b"], results_dir=results), 0)
        selected = json.loads((results / "selected.json").read_text())
        records = json.loads((results / "results.json").read_text())
        self.assertEqual([r["tests"][0] for r in records], selected)
        self.assertEqual(len(records), 2)
        self.assertIn("1 passed; 0 failed; 0 ignored", (results / "0000.log").read_text())

    def test_timeout_fails_and_kills_worker_descendants(self):
        results = self.root / "timeout-results"
        marker = self.root / "orphan"
        with patch.dict(os.environ, {"EREDU_ORPHAN_MARKER": str(marker)}):
            self.assertEqual(run_batches(self.executable, 1, suite="ring",
                                         filters=["::timeout"], test_timeout=0.3,
                                         results_dir=results), 1)
        records = json.loads((results / "results.json").read_text())
        self.assertTrue(records[0]["timed_out"])
        self.assertIn("worker started", (results / "0000.log").read_text())
        time.sleep(1)
        self.assertFalse(marker.exists(), "a timed-out worker must not survive its parent")

    def test_suite_budget_does_not_report_unrun_tests_as_success(self):
        self.assertEqual(run_batches(self.executable, 1, suite="ring",
                                     filters=["::a", "::b"], suite_timeout=1e-9), 1)


if __name__ == "__main__":
    unittest.main()
