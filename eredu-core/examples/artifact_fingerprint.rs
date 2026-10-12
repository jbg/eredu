//! Time a fresh deferred artifact fingerprint and its memoized lookup.
//!
//! cargo run -p eredu-core --example artifact_fingerprint -- FILE_OR_DIRECTORY [ITERATIONS]

use eredu_core::artifact::{ArtifactFile, DeferredArtifactIdentity};
use std::{error::Error, path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let path = PathBuf::from(
        arguments
            .next()
            .ok_or("expected FILE_OR_DIRECTORY [ITERATIONS]")?,
    );
    let iterations = arguments
        .next()
        .map(|value| {
            value
                .to_str()
                .ok_or("invalid iteration count")?
                .parse::<usize>()
                .map_err(|_| "invalid iteration count")
        })
        .transpose()?
        .unwrap_or(3);
    if iterations == 0 || arguments.next().is_some() {
        return Err(
            "expected FILE_OR_DIRECTORY [ITERATIONS], with a positive iteration count".into(),
        );
    }
    let files = if path.is_dir() {
        let mut files = Vec::new();
        for entry in std::fs::read_dir(&path)? {
            let entry = entry?;
            if entry.path().is_file() {
                let name = entry
                    .file_name()
                    .into_string()
                    .map_err(|_| "member name is not UTF-8")?;
                files.push(ArtifactFile::new(name, entry.path()));
            }
        }
        files.sort_by(|a, b| a.logical_role().cmp(b.logical_role()));
        if files.is_empty() {
            return Err("directory has no regular files".into());
        }
        files
    } else {
        vec![ArtifactFile::new("weights", &path)]
    };
    let bytes = files.iter().try_fold(0u64, |sum, file| {
        std::fs::metadata(file.path()).map(|metadata| sum + metadata.len())
    })?;
    let mut expected = None;
    for iteration in 1..=iterations {
        let deferred =
            DeferredArtifactIdentity::filesystem("eredu.fingerprint-benchmark.v1", files.clone())?;
        let start = Instant::now();
        let identity = deferred.resolve()?;
        let elapsed = start.elapsed();
        let start = Instant::now();
        let cached_identity = std::hint::black_box(&deferred).resolve()?;
        let cached = start.elapsed();
        assert_eq!(identity, cached_identity);
        if let Some(previous) = expected {
            assert_eq!(identity, previous, "file identity changed between runs");
        }
        expected = Some(identity);
        println!(
            "run={iteration} bytes={bytes} seconds={:.6} MiB/s={:.1} cached_ns={} {identity}",
            elapsed.as_secs_f64(),
            bytes as f64 / (1024.0 * 1024.0) / elapsed.as_secs_f64(),
            cached.as_nanos(),
        );
    }
    Ok(())
}
