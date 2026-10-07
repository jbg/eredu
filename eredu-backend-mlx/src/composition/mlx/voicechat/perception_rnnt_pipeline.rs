//! Native resources for an already-admitted perception/transcription pipeline.

use super::{
    materialize_voicechat_perception_pipeline, materialize_voicechat_rnnt_predictor_pipeline,
    materialize_voicechat_rnnt_projection, materialize_voicechat_rnnt_projection_parallel,
    MlxPerceptionPipeline, MlxPerceptionPipelineState, MlxRnntPredictorPipeline, MlxRnntProjection,
};
use crate::{backend::runtime::distributed::coordinator::MlxReplicatedCoordinator, MlxTensor};
use eredu_architectures::voicechat::{
    perception_rnnt::{PerceptionRnntError, PerceptionRnntState},
    perception_rnnt_preparation::{
        PerceptionRnntPipelineResources, PreparedPerceptionRnntPipelineBindings,
    },
};
use eredu_core::BackendFailure;
use eredu_runtime::{LayerWeightResidency, LayerwiseModelMetadata, ResidencyReport};
use safemlx::Stream;

/// Reports from three independently budgeted parameter providers.
pub struct MlxPerceptionRnntResidencyReports {
    /// Learned perception units owned by this rank.
    pub perception: ResidencyReport,
    /// Recurrent layers owned by this rank.
    pub predictor: ResidencyReport,
    /// Vocabulary shards and replicated joint affines.
    pub projections: ResidencyReport,
}
/// Materialization evidence from three independently selected providers.
pub struct MlxPerceptionRnntMetadata {
    /// Perception materialization and residency selection.
    pub perception: LayerwiseModelMetadata,
    /// Local recurrent layer materialization and residency selection.
    pub predictor: LayerwiseModelMetadata,
    /// Nonrecurrent RNNT materialization and residency selection.
    pub projections: LayerwiseModelMetadata,
}

/// Retained native components for the ordinary portable paired-input driver.
///
/// There is no cursor, epoch allocator or generation loop in this owner. Borrow
/// its components into `input_pipeline::PipelineInput` with caller-owned identity.
pub struct MlxPerceptionRnntPipeline {
    perception: MlxPerceptionPipeline,
    predictor: MlxRnntPredictorPipeline,
    projections: MlxRnntProjection,
    decision: MlxReplicatedCoordinator,
    resources: PerceptionRnntPipelineResources,
}
impl MlxPerceptionRnntPipeline {
    /// Exact prepared rank-local parameter and initialized cache payload census.
    /// This excludes scratch, activations, transport and host frontend windows.
    pub fn resources(&self) -> PerceptionRnntPipelineResources {
        self.resources
    }

    /// Borrows learned components and authority without moving their sources.
    pub fn parts_mut(
        &mut self,
    ) -> (
        &mut MlxPerceptionPipeline,
        &mut MlxRnntProjection,
        &mut MlxRnntPredictorPipeline,
        &mut MlxReplicatedCoordinator,
    ) {
        (
            &mut self.perception,
            &mut self.projections,
            &mut self.predictor,
            &mut self.decision,
        )
    }
    /// Empty paired state with exact local perception ownership.
    pub fn initial_state(
        &self,
    ) -> Result<PerceptionRnntState<MlxTensor, MlxPerceptionPipelineState>, PerceptionRnntError>
    {
        PerceptionRnntState::from_perception(self.perception.initial_state())
    }
    /// Physical reports preserve each independently configured provider budget.
    pub fn residency_reports(&self) -> Result<MlxPerceptionRnntResidencyReports, BackendFailure> {
        Ok(MlxPerceptionRnntResidencyReports {
            perception: self.perception.residency_report()?,
            predictor: self.predictor.residency_report()?,
            projections: self.projections.residency_report()?,
        })
    }
}

/// Materializes retained components with three explicitly independent policies.
/// No policy is copied as a shared allowance. Aggregate scratch/activation/host
/// capacity remains the caller's responsibility, alongside these provider reports.
#[allow(clippy::too_many_arguments)]
pub fn materialize_voicechat_perception_rnnt_pipeline(
    prepared: PreparedPerceptionRnntPipelineBindings,
    world: &safemlx::distributed::Group,
    perception_residency: LayerWeightResidency,
    predictor_residency: LayerWeightResidency,
    projection_residency: LayerWeightResidency,
    weights_stream: &Stream,
    execution_stream: &Stream,
) -> Result<(MlxPerceptionRnntPipeline, MlxPerceptionRnntMetadata), BackendFailure> {
    for policy in [
        perception_residency,
        predictor_residency,
        projection_residency,
    ] {
        policy.offload().map_err(BackendFailure::from_error)?;
    }
    let resources = prepared.resources();
    let projection_owner = prepared.projection_owner();
    let (perception, predictor, projections, manifest, decision_group, owner, projection_group) =
        prepared.into_parts();
    for (manifest, group) in [
        (perception.manifest(), perception.tensor_group()),
        (predictor.manifest(), predictor.tensor_group()),
    ] {
        crate::backend::runtime::execution::pipeline::validate_component_communication(
            manifest, group,
        )?;
    }
    let mut selections = vec![(decision_group, owner)];
    if let Some(group) = projection_group {
        selections.push((
            group,
            projection_owner.expect("prepared projection authority has an owner"),
        ));
    }
    let mut coordinators = MlxReplicatedCoordinator::from_manifest_selections(
        manifest,
        &selections,
        world,
        execution_stream,
    )?;
    let projection_coordinator = projection_group.map(|_| coordinators.pop().unwrap());
    let decision = coordinators
        .pop()
        .expect("prepared decision authority exists");
    let (perception, _, perception_metadata) = materialize_voicechat_perception_pipeline(
        perception,
        world,
        perception_residency,
        weights_stream,
        execution_stream,
    )?;
    let (predictor, predictor_metadata) = materialize_voicechat_rnnt_predictor_pipeline(
        predictor,
        world,
        predictor_residency,
        weights_stream,
        execution_stream,
    )?;
    let (projections, projection_metadata) = match projection_coordinator.as_ref() {
        Some(coordinator) => materialize_voicechat_rnnt_projection_parallel(
            projections,
            coordinator,
            projection_residency,
            weights_stream,
            execution_stream,
        ),
        None => materialize_voicechat_rnnt_projection(
            projections,
            projection_residency,
            weights_stream,
            execution_stream,
        ),
    }?;
    Ok((
        MlxPerceptionRnntPipeline {
            perception,
            predictor,
            projections,
            decision,
            resources,
        },
        MlxPerceptionRnntMetadata {
            perception: perception_metadata,
            predictor: predictor_metadata,
            projections: projection_metadata,
        },
    ))
}
