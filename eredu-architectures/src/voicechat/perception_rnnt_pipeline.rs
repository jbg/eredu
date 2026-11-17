//! One atomic input candidate over shared perception and RNNT component pipelines.
//!
//! Perception runs once. Its unprojected output enters the ordinary coordinated
//! RNNT label loop; its projected output remains available to Nemotron. Component
//! completion never adopts either cache into the caller's conversational parent.

use super::{
    perception::{PerceptionDimensions, PerceptionError},
    perception_layered::{PerceptionInput, PerceptionRuntimeState},
    perception_rnnt::{
        self, PerceptionCacheState, PerceptionRnntError, PerceptionRnntFrame, PerceptionRnntLimits,
        PerceptionRnntState,
    },
    perception_rnnt_preparation::PerceptionRnntPipelineResources,
    perception_stage::{PerceptionPipeline, PerceptionPipelineArchitecture},
    rnnt::RnntProjectionExecution,
    rnnt_pipeline::{self, RnntPipelineCursor, RnntPredictorExecution},
    rnnt_predictor::RnntPredictorRuntimeState,
    rnnt_predictor_stage::{RnntPredictorPipeline, RnntPredictorPipelineArchitecture},
};
use eredu_core::{BackendFailure, DistributedCommitEpoch, DistributedCommitOutcome};
use eredu_nn::{NeuralBackend, Tensor, TensorElementType};
use eredu_runtime::{
    replicated_execution::ReplicatedCoordinator, BoundComponentPipeline, LayerWeightResidency,
    LayerwiseModelMetadata, ResidencyReport, UnitPolicyResidency,
};

/// Independent parameter residency selections for the three learned components.
pub struct PerceptionRnntResidency {
    /// Learned perception units.
    pub perception: LayerWeightResidency,
    /// Recurrent predictor layers.
    pub predictor: LayerWeightResidency,
    /// Embedding and joint projections.
    pub projections: LayerWeightResidency,
}

/// Materialization evidence from the three independently selected policies.
pub struct PerceptionRnntMetadata {
    /// Perception materialization and selected residency.
    pub perception: LayerwiseModelMetadata,
    /// Predictor materialization and selected residency.
    pub predictor: LayerwiseModelMetadata,
    /// Projection materialization and selected residency.
    pub projections: LayerwiseModelMetadata,
}

/// Current physical residency of each independently budgeted provider.
pub struct PerceptionRnntResidencyReports {
    /// Perception residency.
    pub perception: ResidencyReport,
    /// Predictor residency.
    pub predictor: ResidencyReport,
    /// Projection residency.
    pub projections: ResidencyReport,
}

/// Bound components and decision authority for the portable paired-input driver.
pub struct PerceptionRnntPipeline<B: NeuralBackend, P, R, J, D> {
    pub(super) perception: PerceptionPipeline<B, P>,
    pub(super) predictor: RnntPredictorPipeline<R>,
    pub(super) projections: J,
    pub(super) decision: D,
    pub(super) resources: PerceptionRnntPipelineResources,
}

impl<B: NeuralBackend, P, R, J, D> PerceptionRnntPipeline<B, P, R, J, D> {
    /// Exact prepared rank-local parameter and initialized cache payload census.
    pub fn resources(&self) -> PerceptionRnntPipelineResources {
        self.resources
    }

    /// Borrows the learned components and decision authority for paired execution.
    pub fn parts_mut(
        &mut self,
    ) -> (
        &mut PerceptionPipeline<B, P>,
        &mut J,
        &mut RnntPredictorPipeline<R>,
        &mut D,
    ) {
        (
            &mut self.perception,
            &mut self.projections,
            &mut self.predictor,
            &mut self.decision,
        )
    }

    /// Creates an empty pair with exact local perception ownership.
    pub fn initial_state(
        &self,
    ) -> Result<
        PerceptionRnntState<B::Tensor, PerceptionRuntimeState<B::Tensor>>,
        PerceptionRnntError,
    > {
        PerceptionRnntState::from_perception(self.perception.initial_state())
    }

    /// Reads the three independently configured physical reports.
    pub fn residency_reports(&self) -> Result<PerceptionRnntResidencyReports, BackendFailure>
    where
        B: eredu_nn::DistributedNeuralBackend,
        P: BoundComponentPipeline<
            PerceptionPipelineArchitecture<B>,
            B,
            PerceptionRuntimeState<B::Tensor>,
        >,
        R: BoundComponentPipeline<
            RnntPredictorPipelineArchitecture<B>,
            B,
            RnntPredictorRuntimeState<B::Tensor>,
        >,
        J: UnitPolicyResidency,
    {
        Ok(PerceptionRnntResidencyReports {
            perception: self.perception.pipeline().residency()?,
            predictor: self.predictor.pipeline().residency()?,
            projections: self.projections.residency()?,
        })
    }
}

const PROTOCOL: u64 = 0x50455243524e0001;

/// Stage-owned perception execution over the shared component pipeline mechanism.
pub trait PerceptionPipelineExecution<B: NeuralBackend> {
    /// Only this stage's mutable caches, retained independently of parameters.
    type State: PerceptionCacheState + Clone;
    /// Exact global perception geometry.
    fn dimensions(&self) -> PerceptionDimensions;
    /// Cold validation of stage ownership, F32 caches, requested batch and frontier.
    fn validate_parent(&self, parent: &Self::State, batch: i32) -> Result<(), PerceptionError>;
    /// Executes one immutable-parent candidate and publishes both perception readouts.
    fn forward(
        &mut self,
        input: PerceptionInput<'_, B::Tensor>,
        parent: &Self::State,
        descriptor: [u64; 8],
        epoch: DistributedCommitEpoch,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, B::Tensor, Self::State, DistributedCommitOutcome), PerceptionError>;
}
fn agreed<T, B, C>(
    coordinator: &mut C,
    descriptor: [u64; 8],
    local: Result<T, PerceptionRnntError>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<T, PerceptionRnntError>
where
    B: NeuralBackend,
    C: ReplicatedCoordinator<B::Tensor, <B::Tensor as Tensor>::Context>,
{
    if !coordinator.agree(descriptor, local.is_ok(), context)? {
        return Err(local.err().unwrap_or(PerceptionRnntError::Geometry(
            "perception/RNNT participant failed or invocation identity differs",
        )));
    }
    local
}

/// Advances one recorded mel window with stage-local perception and recurrent caches.
///
/// Every rank supplies matching request/turn/invocation identity, perception epoch,
/// mel geometry and limits. The perception epoch is fresh caller-owned authority;
/// retries and forks must not reuse it. The RNNT cursor reserves the complete
/// worst-case frame before perception reads and is not refunded if perception fails.
/// The returned paired state remains unpublished until the enclosing session commits.
#[allow(clippy::too_many_arguments)]
pub fn advance<B, P, J, R, C>(
    perception: &mut P,
    projections: &mut J,
    predictor: &mut R,
    coordinator: &mut C,
    identity: [u64; 3],
    perception_epoch: DistributedCommitEpoch,
    cursor: &mut RnntPipelineCursor,
    mel: &B::Tensor,
    drop_preencoded: usize,
    previous: &PerceptionRnntState<B::Tensor, P::State>,
    limits: PerceptionRnntLimits,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<PerceptionRnntFrame<B::Tensor, P::State>, PerceptionRnntError>
where
    B: NeuralBackend,
    P: PerceptionPipelineExecution<B>,
    J: RnntProjectionExecution<B>,
    R: RnntPredictorExecution<B>,
    C: ReplicatedCoordinator<B::Tensor, <B::Tensor as Tensor>::Context>,
{
    let dimensions = (perception.dimensions(), projections.dimensions());
    let descriptor = [
        PROTOCOL,
        identity[0],
        identity[1],
        identity[2],
        perception_epoch.value(),
        previous.frames(),
        drop_preencoded as u64,
        mel.shape().get(1).copied().unwrap_or(-1) as u64,
    ];
    let local = (|| {
        dimensions.0.validate()?;
        dimensions.1.validate()?;
        if dimensions.0.block.width != dimensions.1.encoder_width {
            return Err(PerceptionRnntError::Geometry(
                "RNNT must consume the unprojected perception width",
            ));
        }
        if mel.element_type() != Some(TensorElementType::F32) {
            return Err(PerceptionRnntError::Geometry(
                "perception pipeline requires F32 mel input",
            ));
        }
        perception.validate_parent(previous.perception(), 1)?;
        perception_rnnt::validate_advance(
            dimensions,
            mel.shape(),
            drop_preencoded,
            previous,
            limits,
        )
    })();
    let next_frame = agreed::<_, B, C>(coordinator, descriptor, local, context)?;
    let d = dimensions.0;
    agreed::<_, B, C>(
        coordinator,
        [
            PROTOCOL + 1,
            d.subsampling.features as u64,
            d.subsampling.channels as u64,
            d.subsampling.width as u64,
            d.block.width as u64,
            d.block.heads as u64,
            d.block.ff_hidden as u64,
            d.layers as u64,
        ],
        Ok(()),
        context,
    )?;
    agreed::<_, B, C>(
        coordinator,
        [
            PROTOCOL + 2,
            d.block.conv_kernel as u64,
            d.block.left_context as u64,
            d.output_width as u64,
            limits.frame_capacity,
            0,
            0,
            0,
        ],
        Ok(()),
        context,
    )?;
    // This reserves the recurrent epoch range and agrees all RNNT dimensions,
    // hypothesis contents, precision and capacity before the first perception lease.
    let rnnt = rnnt_pipeline::prepare_frame(
        projections,
        predictor,
        coordinator,
        identity,
        cursor,
        previous.rnnt(),
        limits.rnnt,
        context,
    )?;
    let output = perception
        .forward(
            PerceptionInput {
                mel,
                drop_preencoded,
            },
            previous.perception(),
            descriptor,
            perception_epoch,
            context,
        )
        .map_err(PerceptionRnntError::from)
        .and_then(|(encoder, projected, state, outcome)| {
            if outcome != DistributedCommitOutcome::Committed(perception_epoch) {
                return Err(PerceptionRnntError::PerceptionCommit {
                    expected: perception_epoch,
                    observed: outcome,
                });
            }
            perception_rnnt::validate_perception_result(
                dimensions, next_frame, &encoder, &projected, &state,
            )?;
            if encoder.element_type() != Some(TensorElementType::F32)
                || projected.element_type() != Some(TensorElementType::F32)
            {
                return Err(PerceptionRnntError::Geometry(
                    "perception pipeline published a non-F32 readout",
                ));
            }
            perception.validate_parent(&state, 1)?;
            let raw = encoder.reshape(&[1, dimensions.1.encoder_width], context)?;
            Ok((encoder, projected, state, raw))
        });
    let mut completed = descriptor;
    completed[0] = PROTOCOL + 3;
    let (encoder, projected, state, raw) =
        agreed::<_, B, C>(coordinator, completed, output, context)?;
    let decoded = rnnt_pipeline::decode_reserved(
        rnnt,
        coordinator,
        &raw,
        previous.rnnt(),
        limits.rnnt,
        context,
    )?;
    let result =
        perception_rnnt::compose_frame(dimensions, next_frame, encoder, projected, state, decoded);
    completed[0] = PROTOCOL + 4;
    agreed::<_, B, C>(coordinator, completed, result, context)
}
