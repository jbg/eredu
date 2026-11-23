//! Native score, key, and transfer operations for speculative sampling.

use std::sync::Arc;

use crate::{
    capture::{CaptureTokenDomain, SpeculativeCapturePrimitives},
    intervention::{InterventionBackend, InterventionEstimator},
    sampling::SamplingBackend,
    speculative::SpeculativeQueueMechanisms,
    SpeculativeDraftRandomPosition,
};

type Logits<M> = <<M as SpeculativeSamplingPrimitives>::Backend as SamplingBackend>::Logits;
type Random<M> = <<M as SpeculativeSamplingPrimitives>::Backend as SamplingBackend>::RandomState;
type Queue<M> = <<M as SpeculativeSamplingPrimitives>::Backend as SamplingBackend>::Context;

type Root<M> = <<<M as SpeculativeSamplingPrimitives>::CaptureBinding as SpeculativeCapturePrimitives>::Queues as SpeculativeQueueMechanisms>::Root;

/// Native score, random-state, transport, and capture operations used by the portable sampler.
pub trait SpeculativeSamplingPrimitives: Clone + 'static {
    /// Sampling and score primitives shared with ordinary generation.
    type Backend: SamplingBackend<
            Error = Self::Error,
            RandomState: Clone,
            Logits: eredu_nn::Tensor<Context = Queue<Self>>,
        > + crate::refinement::RefinementFinalizationBackend<
            CanvasScores = Logits<Self>,
            CanvasTokens = <Self::Backend as SamplingBackend>::Token,
            PositionScores = Logits<Self>,
            FinalizationContext = Queue<Self>,
            FinalizationError = Self::Error,
        >;
    /// Retained executable score value.
    type Value;
    /// Explicit seed material for the native random stream.
    type Seed;
    /// Position-addressable draft key.
    type DraftRandomness: Clone;
    /// Native root key used to derive distinct streams.
    type RandomnessRoot;
    /// Physical placement and capture capabilities.
    type CaptureBinding: SpeculativeCapturePrimitives<
        InvocationError = Self::Error,
        Queues: SpeculativeQueueMechanisms<Queue = Queue<Self>>,
    >;
    /// Native operation failure.
    type Error: std::error::Error + Send + Sync + 'static;
    /// Native transformations for one selected score row.
    type Capture<'a>: InterventionBackend<Tensor = Logits<Self>, Error: Send + Sync + 'static>;

    /// Borrows a retained score as an operator tensor.
    fn wrap(value: &Self::Value) -> Logits<Self>;
    /// Returns the operator tensor as a retained score.
    fn unwrap(value: Logits<Self>) -> Self::Value;
    /// Reports the physical vocabulary width.
    fn vocabulary(value: &Self::Value) -> usize;
    /// Materializes a selected token for native constrained finalization.
    fn candidate(token: u32) -> <Self::Backend as SamplingBackend>::Token;
    /// Retained bytes backing the currently initialized keys.
    fn snapshot_randomness(
        target: Option<&Random<Self>>,
        draft: Option<&Self::DraftRandomness>,
    ) -> Option<u64>;
    /// Additional native capacity for key replacement.
    fn continuation_randomness_bytes() -> u64;
    /// Preserves a portable error at the native invocation boundary.
    fn from_error(error: impl std::error::Error + Send + Sync + 'static) -> Self::Error;
    /// Estimates native work for admitted interventions.
    fn intervention_estimator() -> Arc<dyn InterventionEstimator>;
    /// Binds native transforms to the selected queue and logical vocabulary.
    fn score_capture<'a>(
        queue: &'a Queue<Self>,
        domain: Option<CaptureTokenDomain<'a>>,
    ) -> Self::Capture<'a>;
    /// Exposes the vocabulary row with the model-logits observation shape.
    fn capture_scores(
        value: &Logits<Self>,
        queue: &Queue<Self>,
    ) -> Result<Logits<Self>, Self::Error>;
    /// Restores the physical sampling shape of an intervened row.
    fn restore_scores(
        value: Logits<Self>,
        original: &Logits<Self>,
        queue: &Queue<Self>,
    ) -> Result<Logits<Self>, Self::Error>;
    /// Settles one sampled token and reads its scalar identity.
    fn sample_token(
        token: <Self::Backend as SamplingBackend>::Token,
        queue: &Queue<Self>,
    ) -> Result<u32, Self::Error>;
    /// Materializes the explicit seed.
    fn seed(seed: u64) -> Result<Self::Seed, Self::Error>;
    /// Constructs the native root key from admitted seed material.
    fn randomness_root(seed: Option<Self::Seed>) -> Result<Self::RandomnessRoot, Self::Error>;
    /// Derives a target key on the specified queue.
    fn target_randomness_from_root(
        root: &mut Self::RandomnessRoot,
        queue: &Queue<Self>,
    ) -> Result<Random<Self>, Self::Error>;
    /// Derives a position-addressable draft key on the specified queue.
    fn draft_randomness_from_root(
        root: &mut Self::RandomnessRoot,
        queue: &Queue<Self>,
    ) -> Result<Self::DraftRandomness, Self::Error>;
    /// Borrows the draft key for queue dependency submission.
    fn draft_key_root(key: &Self::DraftRandomness) -> &Root<Self>;
    /// Copies a draft key to the nominated queue and settles the copy.
    fn copy_draft_key(
        key: &Self::DraftRandomness,
        destination: &Queue<Self>,
    ) -> Result<Self::DraftRandomness, Self::Error>;
    /// Derives the independent key for one exact draft position.
    fn draft_randomness_at(
        root: &Self::DraftRandomness,
        position: SpeculativeDraftRandomPosition,
        queue: &Queue<Self>,
    ) -> Result<Random<Self>, Self::Error>;
    /// Reads the normalized probability of one token.
    fn probability_at(
        distribution: &Self::Value,
        token: u32,
        queue: &Queue<Self>,
    ) -> Result<f32, Self::Error>;
    /// Draws one acceptance random variate.
    fn sample_unit_interval(
        randomness: Option<&mut Random<Self>>,
        queue: &Queue<Self>,
    ) -> Result<f32, Self::Error>;
    /// Materializes the positive part of the difference of two distributions.
    fn positive_probability_difference(
        left: &Self::Value,
        right: &Self::Value,
        queue: &Queue<Self>,
    ) -> Result<Option<Self::Value>, Self::Error>;
    /// Borrows one distribution for queue dependency submission.
    fn distribution_root(value: &Self::Value) -> &Root<Self>;
    /// Copies distributions to the nominated destination.
    fn copy_distributions(
        distributions: &mut [&mut Self::Value],
        destination: &Queue<Self>,
    ) -> Result<(), Self::Error>;
}
