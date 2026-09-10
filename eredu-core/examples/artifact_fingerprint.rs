//! Time a fresh deferred file fingerprint and its memoized lookup.
//!
//! cargo run -p eredu-core --example artifact_fingerprint -- FILE [ITERATIONS]

use eredu_core::artifact::{ArtifactFile, DeferredArtifactIdentity};
use std::{error::Error, path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let path = PathBuf::from(arguments.next().ok_or("expected FILE [ITERATIONS]")?);
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
        return Err("expected FILE [ITERATIONS], with a positive iteration count".into());
    }
    let bytes = std::fs::metadata(&path)?.len();
    let mut expected = None;
    for iteration in 1..=iterations {
        let deferred = DeferredArtifactIdentity::filesystem(
            "eredu.fingerprint-benchmark.v1",
            [ArtifactFile::new("weights", &path)],
        )?;
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
