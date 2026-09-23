"""Check exact coverage and failure propagation using a real Rust test harness."""

from pathlib import Path
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from run_native_unit_tests import run_batches, test_names


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
                #[test] fn a() { record("a"); }
                #[test] fn a_longer() { record("a_longer"); panic!("must propagate"); }
                #[test] #[ignore] fn ignored() { panic!("must stay ignored"); }
                #[test] fn z_after_failure() { record("after_failure"); }
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


if __name__ == "__main__":
    unittest.main()
