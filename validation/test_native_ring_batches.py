"""Check exact coverage and failure propagation using a real Rust test harness."""

from pathlib import Path
import os
import json
import time
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from run_native_ring_tests import run_batches, select_tests


class RingBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        source = cls.root / "ring.rs"
        fixtures = {
            "partition": [
                ("a", ""),
                ("b", ""),
                ("c", ""),
                ("timeout", """
                    let marker = std::env::var("EREDU_ORPHAN_MARKER").unwrap();
                    let _child = std::process::Command::new("sh").arg("-c")
                        .arg(r#"sleep 1; echo orphan > "$1""#).arg("sh").arg(marker)
                        .spawn().unwrap();
                    println!("worker started");
                    std::thread::sleep(std::time::Duration::from_secs(20));
                """),
            ],
            "realtime": [("released", "panic!();")],
        }
        source.write_text("\n".join(
            f"mod {module} {{ " + " ".join(
                f"#[test] #[ignore] fn {name}() {{ {body} }}" for name, body in tests
            ) + " }" for module, tests in fixtures.items()
        ))
        cls.executable = str(cls.root / "ring")
        subprocess.run(["rustc", "--test", str(source), "-o", cls.executable], check=True)

    def setUp(self):
        selection = patch("run_native_ring_tests.RING_TESTS",
                          ["partition::a", "partition::b", "partition::c", "partition::timeout"])
        selection.start()
        self.addCleanup(selection.stop)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_ring_shards_cover_only_ignored_synthetic_tests_once(self):
        shards = [select_tests(self.executable, i, 3, []) for i in range(3)]
        names = [name for shard in shards for name in shard]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(sorted(name.rsplit("::", 1)[1] for name in names),
                         ["a", "b", "c", "timeout"])
        with self.assertRaisesRegex(RuntimeError, "matched nothing"):
            select_tests(self.executable, 0, 1, ["nonexistent"])
        with patch("run_native_ring_tests.RING_TESTS", ["missing"]):
            with self.assertRaisesRegex(RuntimeError, "missing ignored Ring integration tests"):
                select_tests(self.executable, 0, 1, [])

    def test_ring_runs_ignored_tests_and_records_exact_selection(self):
        results = self.root / "results"
        self.assertEqual(run_batches(self.executable, 1,
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
            self.assertEqual(run_batches(self.executable, 1,
                                         filters=["::timeout"], test_timeout=0.3,
                                         results_dir=results), 1)
        records = json.loads((results / "results.json").read_text())
        self.assertTrue(records[0]["timed_out"])
        self.assertIn("worker started", (results / "0000.log").read_text())
        time.sleep(1)
        self.assertFalse(marker.exists(), "a timed-out worker must not survive its parent")

    def test_suite_budget_does_not_report_unrun_tests_as_success(self):
        self.assertEqual(run_batches(self.executable, 1,
                                     filters=["::a", "::b"], suite_timeout=1e-9), 1)


if __name__ == "__main__":
    unittest.main()
