//! Deterministic host draws for backend-neutral execution policy.

use std::num::NonZeroUsize;

use rand::{distr::Distribution, distr::Uniform, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// An explicitly seeded host random stream, stored entirely inline.
///
/// Cloning copies the complete generator state, including buffered draws. A
/// caller can draw from an unpublished clone and adopt it only when its
/// transition commits; discarding that clone leaves the retained stream intact.
/// No clock or operating-system entropy participates in construction or draws.
///
/// Replay uses ChaCha8 and the `rand` 0.9 uniform distribution. Reproducibility
/// is scoped to this algorithm and dependency version; this type is not a
/// version-independent serialized random-state format.
#[derive(Debug, Clone)]
pub struct HostRandomState {
    generator: ChaCha8Rng,
}

impl HostRandomState {
    /// Create a stream from an explicit seed.
    pub fn with_seed(seed: u64) -> Self {
        Self {
            generator: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Draw an unbiased index in `0..choices`.
    ///
    /// Sampling uses a `u64` distribution so the same seed and bounds produce
    /// the same sequence on hosts with different pointer widths.
    pub fn index(&mut self, choices: NonZeroUsize) -> usize {
        let distribution = Uniform::new(0_u64, choices.get() as u64)
            .expect("a nonzero choice count defines a nonempty range");
        distribution.sample(&mut self.generator) as usize
    }
}
