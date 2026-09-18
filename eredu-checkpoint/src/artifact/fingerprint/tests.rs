use super::*;
use std::cell::Cell;

#[derive(Debug, thiserror::Error)]
#[error("refused original fingerprint destination")]
struct Refused;
struct Account {
    calls: Cell<usize>,
    stop: usize,
}
impl ArtifactFingerprintAllocation for Account {
    type Error = Refused;
    fn reserve(&self, _: usize) -> Result<(), Refused> {
        let n = self.calls.get();
        assert!(n <= self.stop, "producer after refusal");
        self.calls.set(n + 1);
        if n == self.stop { Err(Refused) } else { Ok(()) }
    }
}
#[test]
fn original_file_pass_preserves_bytes_paths_errors_and_every_refusal() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("weights-東京-ß.tensor");
    let bytes = vec![29; 1024 * 1024 + 17];
    std::fs::write(&path, &bytes).unwrap();
    let source = ArtifactFingerprintSource::new([ArtifactFile::new("weights", &path)]).unwrap();
    let expected = source.fingerprint().unwrap();
    assert_eq!(expected[0].length(), bytes.len() as u64);
    assert_eq!(
        expected[0].digest(),
        <[u8; 32]>::from(Sha256::digest(&bytes))
    );
    for missing in [false, true] {
        if missing {
            std::fs::remove_file(&path).unwrap();
        }
        let account = Account {
            calls: Cell::new(0),
            stop: usize::MAX,
        };
        let actual = source.fingerprint_with_allocations(&account);
        if missing {
            assert!(
                matches!(actual, Err(ArtifactFingerprintPreparationError::Source(
                ArtifactFingerprintError::Io { action: "open", ref path, ref source }
            )) if path.file_name() == Some(std::ffi::OsStr::new("weights-東京-ß.tensor"))
                && source.kind() == std::io::ErrorKind::NotFound)
            );
        } else {
            assert_eq!(actual.unwrap(), expected);
        }
        let calls = account.calls.get();
        assert!(calls > 5);
        for stop in 0..calls {
            let account = Account {
                calls: Cell::new(0),
                stop,
            };
            assert!(matches!(
                source.fingerprint_with_allocations(&account),
                Err(ArtifactFingerprintPreparationError::Funding(Refused))
            ));
            assert_eq!(account.calls.get(), stop + 1);
        }
    }
}

#[test]
fn prepared_open_retries_interruptions_and_preserves_terminal_error_and_refusal() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("open-retry");
    std::fs::write(&path, [3, 5, 8]).unwrap();
    use std::os::unix::ffi::OsStrExt;
    let prepared_path = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
    for terminal in [false, true] {
        let run = |stop| {
            let attempts = Cell::new(0);
            let account = Account {
                calls: Cell::new(0),
                stop,
            };
            let result = Destination(&account).open_prepared(&path, || {
                attempts.set(attempts.get() + 1);
                if attempts.get() <= 2 {
                    return Err(rustix::io::Errno::INTR);
                }
                if terminal {
                    return Err(rustix::io::Errno::ACCESS);
                }
                rustix::fs::open(
                    prepared_path.as_c_str(),
                    rustix::fs::OFlags::RDONLY,
                    rustix::fs::Mode::empty(),
                )
            });
            (result, account.calls.get(), attempts.get())
        };
        let (result, count, attempts) = run(usize::MAX);
        assert_eq!(attempts, 3);
        if terminal {
            assert!(
                matches!(result, Err(ArtifactFingerprintPreparationError::Source(
                ArtifactFingerprintError::Io { action: "open", path: error_path, source }
            )) if error_path == path && source.kind() == std::io::ErrorKind::PermissionDenied)
            );
        } else {
            assert_eq!(result.unwrap().metadata().unwrap().len(), 3);
            assert_eq!(
                count, 1,
                "retries reuse the original controls and prepared path"
            );
        }
        for stop in 0..count {
            let (result, calls, attempts) = run(stop);
            assert!(matches!(
                result,
                Err(ArtifactFingerprintPreparationError::Funding(Refused))
            ));
            assert_eq!(calls, stop + 1);
            assert_eq!(attempts, if stop == 0 { 0 } else { 3 });
        }
    }
}
