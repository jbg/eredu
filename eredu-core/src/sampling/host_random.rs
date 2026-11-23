//! Deterministic host draws for backend-neutral execution policy.

use std::num::NonZeroUsize;

use rand::{distr::Distribution, distr::Open01, distr::Uniform, Rng, SeedableRng};
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

    /// Draws a full-width random word from the selected stream.
    pub fn next_u64(&mut self) -> u64 {
        self.generator.random()
    }

    /// Selects a numbered stream within the given seed.
    pub fn with_stream(seed: u64, stream: u64) -> Self {
        let mut generator = Self::with_seed(seed);
        generator.generator.set_stream(stream);
        generator
    }

    /// Derives an independent substream from this stream's original seed.
    pub fn split(&self, stream: u64) -> Self {
        let mut derived = self.clone();
        derived.generator.set_stream(stream);
        derived.generator.set_word_pos(0);
        derived
    }

    /// Draws an FP32 value in the half-open unit interval.
    pub fn unit_interval_f32(&mut self) -> f32 {
        self.generator.random::<f32>()
    }

    /// Draw a uniform value in the half-open unit interval.
    pub fn unit_interval(&mut self) -> f64 {
        self.generator.random::<f64>()
    }

    /// Draws a uniform value strictly between zero and one.
    pub fn open_unit_interval(&mut self) -> f64 {
        Open01.sample(&mut self.generator)
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

impl crate::execution_control::SnapshotValue for HostRandomState {
    fn estimate_snapshot_value(
        &self,
    ) -> Result<crate::execution_control::SnapshotEstimate, crate::BackendFailure> {
        crate::execution_control::snapshot_value_estimate::<Self>([])
    }

    fn copy_snapshot_value(
        &self,
        _: &crate::execution_control::SnapshotRetention,
    ) -> Result<Self, crate::BackendFailure> {
        Ok(self.clone())
    }
}
