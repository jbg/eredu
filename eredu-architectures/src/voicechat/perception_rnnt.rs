//! One streaming perception invocation shared by transcription and Nemotron.
//!
//! Each accepted mel window produces exactly one live-audio frame. Learned
//! parameters remain independent of candidate caches, and returning a candidate
//! does not establish native completion or publish either component's state.

use eredu_core::{execution_control::SnapshotEstimate, BackendFailure};
use eredu_runtime::execution_control::{
    snapshot_value_estimate, SnapshotReservation, SnapshotValue,
};

use eredu_nn::RetainedTensorRoots;

use eredu_nn::{NeuralBackend, Parameterized, Tensor};

use super::{
    perception::{Perception, PerceptionDimensions, PerceptionError, PerceptionState},
    perception_layered::PerceptionExecution,
    rnnt::{Rnnt, RnntDimensions, RnntError, RnntExecution, RnntFrame, RnntLimits, RnntState},
};

/// Bounds checked before the perception invocation or recurrent initialization.
#[derive(Debug, Clone, Copy)]
pub struct PerceptionRnntLimits {
    /// Maximum retained live-audio frame frontier for both components.
    pub frame_capacity: u64,
    /// Per-frame symbol and retained hypothesis bounds.
    pub rnnt: RnntLimits,
}

impl PerceptionRnntLimits {
    /// Admits at least one acoustic frame before allocating neural state.
    pub fn validate_startup(self) -> Result<(), PerceptionRnntError> {
        if self.frame_capacity == 0 {
            return Err(PerceptionRnntError::Capacity {
                required: 1,
                capacity: 0,
            });
        }
        if self.rnnt.max_symbols_per_frame == 0 {
            return Err(
                RnntError::InvalidConfiguration("RNNT symbol limit must be positive").into(),
            );
        }
        Ok(())
    }
}

/// Recorded acoustic frontier of ordinary or stage-owned perception caches.
pub trait PerceptionCacheState {
    /// Exclusive count of accepted live-audio frames.
    fn frames(&self) -> u64;
}
impl<T> PerceptionCacheState for PerceptionState<T> {
    fn frames(&self) -> u64 {
        self.frames()
    }
}

/// Paired candidate caches, separate from parameters and conversational policy.
#[derive(Debug, Clone)]
pub struct PerceptionRnntState<T, C = PerceptionState<T>> {
    dimensions: Option<(PerceptionDimensions, RnntDimensions)>,
    perception: C,
    rnnt: RnntState<T>,
}

impl<T: SnapshotValue, C: SnapshotValue> SnapshotValue for PerceptionRnntState<T, C> {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        snapshot_value_estimate::<Self>([
            self.perception.estimate_snapshot_value(),
            self.rnnt.estimate_snapshot_value(),
        ])
    }
    fn copy_snapshot_value(&self, retention: &SnapshotReservation) -> Result<Self, BackendFailure> {
        Ok(Self {
            perception: self.perception.copy_snapshot_value(retention)?,
            rnnt: self.rnnt.copy_snapshot_value(retention)?,
            dimensions: self.dimensions,
        })
    }
}

impl<T, C: RetainedTensorRoots<T>> RetainedTensorRoots<T> for PerceptionRnntState<T, C> {
    fn visit_retained_tensors<'a>(&'a self, visitor: &mut dyn FnMut(&'a T)) {
        self.perception.visit_retained_tensors(visitor);
        self.rnnt.visit_retained_tensors(visitor);
    }
}
impl<T, C: Default> Default for PerceptionRnntState<T, C> {
    fn default() -> Self {
        Self {
            dimensions: None,
            perception: C::default(),
            rnnt: RnntState::default(),
        }
    }
}
impl<T, C: PerceptionCacheState> PerceptionRnntState<T, C> {
    /// Creates a paired empty state from exact stage-owned initial perception caches.
    pub fn from_perception(perception: C) -> Result<Self, PerceptionRnntError> {
        if perception.frames() != 0 {
            return Err(PerceptionRnntError::Geometry(
                "initial perception caches already contain audio history",
            ));
        }
        Ok(Self {
            dimensions: None,
            perception,
            rnnt: RnntState::default(),
        })
    }
    /// Joins cache owners at one frame frontier under exact connected model geometry.
    /// Selected execution validates each component's local cache ownership before use.
    pub fn from_parts(
        dimensions: (PerceptionDimensions, RnntDimensions),
        perception: C,
        rnnt: RnntState<T>,
    ) -> Result<Self, PerceptionRnntError> {
        dimensions.0.validate()?;
        dimensions.1.validate()?;
        if dimensions.0.block.width != dimensions.1.encoder_width
            || perception.frames() != rnnt.frames()
            || rnnt
                .predictor()
                .is_some_and(|state| state.dimensions() != dimensions.1)
        {
            return Err(PerceptionRnntError::Geometry(
                "perception/RNNT component geometry or frame frontiers differ",
            ));
        }
        Ok(Self {
            dimensions: Some(dimensions),
            perception,
            rnnt,
        })
    }
    /// Shared live-audio frontier; tool-only model steps do not advance it.
    pub fn frames(&self) -> u64 {
        self.perception.frames()
    }
    /// Perception caches from the shared invocation.
    pub fn perception(&self) -> &C {
        &self.perception
    }
    /// Retained prediction network and complete transcription hypothesis.
    pub fn rnnt(&self) -> &RnntState<T> {
        &self.rnnt
    }
}

/// Unpublished outputs and both candidate states from one admitted audio frame.
#[derive(Debug, Clone)]
pub struct PerceptionRnntFrame<T, C = PerceptionState<T>> {
    /// The exact unprojected `[1, 1, encoder_width]` features supplied to RNNT.
    pub encoder: T,
    /// `[1, 1, output_width]` features available for Nemotron channel fusion.
    pub projected: T,
    /// Candidate caches, adopted only at the shared transaction's commit boundary.
    pub state: PerceptionRnntState<T, C>,
    /// Newly decoded transcription tokens, in order.
    pub emitted: Vec<usize>,
    /// First joint decision was blank, independently of the final loop decision.
    pub first_blank: bool,
    /// Final evaluated joint decision was blank.
    pub ended_blank: bool,
    /// Decoding stopped on a nonblank decision at the symbol bound.
    pub hit_symbol_limit: bool,
}

/// Invalid composition, retained state, explicit bound or component failure.
#[derive(Debug, thiserror::Error)]
pub enum PerceptionRnntError {
    /// Disconnected raw feature widths, invalid input or incompatible caches.
    #[error("{0}")]
    Geometry(&'static str),
    /// The caller's frame capacity cannot admit this invocation.
    #[error("perception/RNNT frame {required} exceeds capacity {capacity}")]
    Capacity {
        /// Required exclusive frame frontier.
        required: u64,
        /// Selected maximum frontier.
        capacity: u64,
    },
    /// Perception failed without adopting its candidate caches.
    #[error(transparent)]
    Perception(#[from] PerceptionError),
    /// RNNT failed without adopting either component's candidate caches.
    #[error(transparent)]
    Rnnt(#[from] RnntError),
    /// A shared coordination mechanism failed, preserving its native source.
    #[error(transparent)]
    Backend(#[from] BackendFailure),
    /// The perception candidate did not observe the exact expected committed epoch.
    #[error("perception expected committed epoch {expected:?}, observed {observed:?}")]
    PerceptionCommit {
        /// Caller-recorded component epoch.
        expected: eredu_core::DistributedCommitEpoch,
        /// Exact observed decision, including indeterminate completion.
        observed: eredu_core::DistributedCommitOutcome,
    },
    /// A neutral tensor operation failed, preserving its original source.
    #[error(transparent)]
    Neural(#[from] eredu_nn::Error),
}

/// Ordinary perception and RNNT parameters with one shared encoder invocation.
#[derive(Debug, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct PerceptionRnnt<B: NeuralBackend, P = Perception<B>, J = Rnnt<B>> {
    #[parameter(skip)]
    dimensions: (PerceptionDimensions, RnntDimensions),
    perception: P,
    rnnt: J,
    #[parameter(skip)]
    backend: std::marker::PhantomData<B>,
}
impl<B: NeuralBackend> PerceptionRnnt<B> {
    /// Declares ordinary parameter slots without reopening artifacts.
    pub fn new(
        perception: PerceptionDimensions,
        rnnt: RnntDimensions,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, PerceptionRnntError> {
        if perception.block.width != rnnt.encoder_width {
            return Err(PerceptionRnntError::Geometry(
                "RNNT must consume the unprojected perception width",
            ));
        }
        Self::from_parts(
            Perception::new(perception, context)?,
            Rnnt::new(rnnt, context)?,
        )
    }
}

impl<B: NeuralBackend, P: PerceptionExecution<B>, J: RnntExecution<B>> PerceptionRnnt<B, P, J> {
    /// Separates learned components for rebinding an execution provider.
    /// No cache or live frame belongs to this parameter owner.
    pub fn into_parts(self) -> (P, J) {
        (self.perception, self.rnnt)
    }

    /// Joins an admitted perception realization and RNNT without replacement parameters.
    pub fn from_parts(perception: P, rnnt: J) -> Result<Self, PerceptionRnntError> {
        let dimensions = (perception.dimensions(), rnnt.dimensions());
        if dimensions.0.block.width != dimensions.1.encoder_width {
            return Err(PerceptionRnntError::Geometry(
                "RNNT must consume the unprojected perception width",
            ));
        }
        Ok(Self {
            dimensions,
            perception,
            rnnt,
            backend: std::marker::PhantomData,
        })
    }

    /// Prepares fixed tool-channel silence without running RNNT or adopting live caches.
    pub fn prepare_silence(
        &mut self,
        prepared: &super::silence::PreparedSilenceConditioning,
        output_dtype: eredu_nn::TensorElementType,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, super::silence::SilenceConditioningError> {
        prepared.execute::<B, P>(&mut self.perception, output_dtype, context)
    }

    /// Advances one streaming window atomically with respect to its borrowed parent.
    ///
    /// `drop_preencoded` discards the released subsampler overlap. The remaining
    /// geometry must be exactly one frame, so no hidden batch loop can advance
    /// the live-audio frontier beyond the caller's scheduled invocation. RNNT
    /// consumes a view of the raw encoder result; projection never substitutes
    /// for transcription input and the perception stack is not run a second time.
    pub fn advance(
        &mut self,
        mel: &B::Tensor,
        drop_preencoded: usize,
        previous: &PerceptionRnntState<B::Tensor>,
        limits: PerceptionRnntLimits,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PerceptionRnntFrame<B::Tensor>, PerceptionRnntError> {
        let next_frame = self.validate_advance(mel.shape(), drop_preencoded, previous, limits)?;
        let perception =
            self.perception
                .forward(mel, drop_preencoded, &previous.perception, context)?;
        validate_perception_result(
            self.dimensions,
            next_frame,
            &perception.encoder,
            &perception.projected,
            &perception.state,
        )?;
        let raw = perception
            .encoder
            .reshape(&[1, self.dimensions.1.encoder_width], context)?;
        let rnnt = self
            .rnnt
            .decode_frame(&raw, &previous.rnnt, limits.rnnt, context)?;
        compose_frame(
            self.dimensions,
            next_frame,
            perception.encoder,
            perception.projected,
            perception.state,
            rnnt,
        )
    }
    /// Shared cold preflight for existing tensors and host mel materialization.
    pub(crate) fn validate_advance(
        &self,
        shape: &[i32],
        drop_preencoded: usize,
        previous: &PerceptionRnntState<B::Tensor>,
        limits: PerceptionRnntLimits,
    ) -> Result<u64, PerceptionRnntError> {
        validate_advance(self.dimensions, shape, drop_preencoded, previous, limits)
    }
}

pub(crate) fn validate_advance<T, C: PerceptionCacheState>(
    dimensions: (PerceptionDimensions, RnntDimensions),
    shape: &[i32],
    drop_preencoded: usize,
    previous: &PerceptionRnntState<T, C>,
    limits: PerceptionRnntLimits,
) -> Result<u64, PerceptionRnntError> {
    if previous.dimensions.is_some_and(|d| d != dimensions)
        || previous.perception.frames() != previous.rnnt.frames()
    {
        return Err(PerceptionRnntError::Geometry(
            "perception/RNNT state has incompatible geometry or frame frontiers",
        ));
    }
    if shape.len() != 3 || shape[0] != 1 || shape[2] != dimensions.0.subsampling.features {
        return Err(PerceptionRnntError::Geometry(
            "streaming perception/RNNT requires [1, mel_time, features] input",
        ));
    }
    let frames = dimensions
        .0
        .subsampling
        .output_frames(shape[1])
        .map_err(PerceptionError::from)?;
    if usize::try_from(frames)
        .ok()
        .and_then(|n| n.checked_sub(drop_preencoded))
        != Some(1)
    {
        return Err(PerceptionRnntError::Geometry(
            "streaming perception/RNNT window must retain exactly one encoded frame",
        ));
    }
    let next_frame = previous
        .frames()
        .checked_add(1)
        .ok_or(PerceptionRnntError::Geometry(
            "perception/RNNT frame counter exhausted",
        ))?;
    if next_frame > limits.frame_capacity {
        return Err(PerceptionRnntError::Capacity {
            required: next_frame,
            capacity: limits.frame_capacity,
        });
    }
    limits.validate_startup()?;
    if previous.rnnt.hypothesis().len() > limits.rnnt.hypothesis_capacity {
        return Err(RnntError::Capacity {
            capacity: limits.rnnt.hypothesis_capacity,
        }
        .into());
    }
    Ok(next_frame)
}

pub(crate) fn validate_perception_result<T: Tensor, C: PerceptionCacheState>(
    dimensions: (PerceptionDimensions, RnntDimensions),
    next_frame: u64,
    encoder: &T,
    projected: &T,
    perception: &C,
) -> Result<(), PerceptionRnntError> {
    if encoder.shape() != [1, 1, dimensions.1.encoder_width]
        || projected.shape() != [1, 1, dimensions.0.output_width]
        || perception.frames() != next_frame
    {
        return Err(PerceptionRnntError::Geometry(
            "perception result differs from the admitted frame geometry",
        ));
    }
    Ok(())
}

pub(crate) fn compose_frame<T, C: PerceptionCacheState>(
    dimensions: (PerceptionDimensions, RnntDimensions),
    next_frame: u64,
    encoder: T,
    projected: T,
    perception: C,
    rnnt: RnntFrame<T>,
) -> Result<PerceptionRnntFrame<T, C>, PerceptionRnntError> {
    if rnnt.state.frames() != next_frame || perception.frames() != next_frame {
        return Err(PerceptionRnntError::Geometry(
            "RNNT result differs from the admitted frame frontier",
        ));
    }
    Ok(PerceptionRnntFrame {
        encoder,
        projected,
        state: PerceptionRnntState {
            dimensions: Some(dimensions),
            perception,
            rnnt: rnnt.state,
        },
        emitted: rnnt.emitted,
        first_blank: rnnt.first_blank,
        ended_blank: rnnt.ended_blank,
        hit_symbol_limit: rnnt.hit_symbol_limit,
    })
}
