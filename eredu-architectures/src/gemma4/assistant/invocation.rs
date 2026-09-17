//! Shared sequential assistant operation and its exact borrowed-source projection.
use super::*;
use crate::external_assistant::{
    Gemma4AssistantArchitecture, invocation::ExternalAssistantOperation,
};
use eredu_nn::workspace::{
    WorkspaceBackend, WorkspaceContext, WorkspaceMetadataError, WorkspaceTensor,
};
use std::mem::{size_of, size_of_val};

/// The existing target-feature-driven sequential proposal equation.
pub struct DraftStep;
/// Actual arguments, lent without copying native state or source ownership.
pub struct DraftStepArguments<'a, T> {
    /// Target token embedding for this proposal.
    pub embedding: &'a T,
    /// Actual branch-local hidden state and shared target K/V.
    pub state: &'a mut AssistantState<T>,
}
/// Paid finite metadata state for that same operation.
pub struct ProjectedDraftStep {
    embedding: WorkspaceTensor,
    hidden: WorkspaceTensor,
    shared: ProjectedShared,
    offset: i32,
}
struct ProjectedShared {
    entries: Vec<(AttentionPolicy, (WorkspaceTensor, WorkspaceTensor))>,
    context: WorkspaceContext,
}
impl super::super::SharedAttentionStore<WorkspaceTensor> for ProjectedShared {
    fn get(&self, policy: &AttentionPolicy) -> Option<&(WorkspaceTensor, WorkspaceTensor)> {
        self.entries
            .iter()
            .find(|(key, _)| key == policy)
            .map(|(_, values)| values)
    }
    fn publish(
        &mut self,
        _: AttentionPolicy,
        _: (WorkspaceTensor, WorkspaceTensor),
    ) -> Result<(), Error> {
        // Validated external Gemma layers only consume the retained target KV.
        Err(self.context.metadata_error(format_args!(
            "assistant attempted to publish into borrowed target K/V"
        )))
    }
}
impl ExternalAssistantOperation<Gemma4AssistantArchitecture> for DraftStep {
    type Arguments<'a, T: Tensor + 'a> = DraftStepArguments<'a, T>;
    type Output<T: Tensor> = T;
    type Projected = ProjectedDraftStep;

    fn invocation_kind() -> eredu_runtime::speculative::external_occurrence::ExternalInvocationKind {
        eredu_runtime::speculative::external_occurrence::ExternalInvocationKind::AssistantStep
    }
    fn workspace_module(
        config: &AssistantConfig,
        context: &WorkspaceContext,
    ) -> Result<Assistant<WorkspaceBackend>, Error> {
        Assistant::from_config(config, context)
    }
    fn execute<B, C>(
        module: &mut Assistant<B>,
        arguments: DraftStepArguments<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + Clone,
        C: AttentionCache<B::Tensor>,
    {
        module.draft_step::<C>(arguments.embedding, arguments.state, context)
    }
    fn reborrow<'a, 'state, T: Tensor>(arguments: &'a mut DraftStepArguments<'state,T>)
        -> DraftStepArguments<'a,T> where 'state: 'a {
        DraftStepArguments { embedding: arguments.embedding, state: &mut *arguments.state }
    }
    fn visit_evidence<T: Tensor>(arguments: &DraftStepArguments<'_,T>,
        visit: &mut dyn FnMut(&crate::speculative_execution::PreparedEmbeddedEvidence)) {
        if let Some(evidence) = &arguments.state.evidence { visit(evidence); }
    }
    fn retain_evidence<T: Tensor>(arguments: &mut DraftStepArguments<'_,T>,
        evidence: crate::speculative_execution::PreparedEmbeddedEvidence) {
        arguments.state.evidence = Some(evidence);
    }
    fn project<'a, T: Tensor>(
        arguments: &'a DraftStepArguments<'_, T>,
        mut project: impl FnMut(&'a T) -> Result<WorkspaceTensor, Error>,
        context: &WorkspaceContext,
    ) -> Result<ProjectedDraftStep, Error> {
        let controls = [
            size_of::<ProjectedDraftStep>(),
            size_of::<Result<ProjectedDraftStep, Error>>(),
            size_of_val(&project),
            size_of::<Result<WorkspaceTensor, Error>>(),
            size_of::<super::shared::Iter<'_, T>>(),
            size_of::<(&AttentionPolicy, &(T, T))>(),
        ];
        context.charge_metadata(
            controls
                .into_iter()
                .try_fold(size_of_val(&controls), usize::checked_add)
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        let embedding = project(arguments.embedding)?;
        let hidden = project(&arguments.state.hidden)?;
        let mut entries = context.metadata_vec(arguments.state.shared_kv.len())?;
        for (policy, (keys, values)) in &arguments.state.shared_kv {
            entries.push((*policy, (project(keys)?, project(values)?)));
        }
        Ok(ProjectedDraftStep {
            embedding,
            hidden,
            shared: ProjectedShared {
                entries,
                context: context.clone(),
            },
            offset: arguments.state.kv_offset,
        })
    }
    fn geometry(
        projected: &ProjectedDraftStep,
        context: &WorkspaceContext,
    ) -> Result<eredu_core::InferenceGeometry, Error> {
        let shape = projected.embedding.shape();
        if shape.len() != 3 || shape[0] <= 0 || shape[1] != 1 || projected.offset < 0 {
            return Err(context.metadata_error(format_args!(
                "sequential assistant invocation has invalid input/frontier geometry"
            )));
        }
        let geometry = eredu_core::InferenceGeometry {
            batch_size: shape[0] as u64,
            cached_positions: projected.offset as u64,
            input_positions: 1,
            max_output_tokens: 0,
            prefill_chunk_positions: 1,
            output: eredu_core::OutputDemand::Sequence,
        };
        geometry
            .validate_fixed()
            .map_err(|cause| context.metadata_source(cause))?;
        Ok(geometry)
    }
    fn trace(
        module: &mut Assistant<WorkspaceBackend>,
        projected: &mut ProjectedDraftStep,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        module.draft_step_parts::<eredu_runtime::working_memory::WorkspaceConcatLayerState>(
            &projected.embedding,
            &mut projected.hidden,
            &mut projected.offset,
            &mut projected.shared,
            context,
        )
    }
    fn visit_projected(projected: &ProjectedDraftStep, visit: &mut dyn FnMut(&WorkspaceTensor)) {
        visit(&projected.embedding);
        visit(&projected.hidden);
        for (_, (keys, values)) in &projected.shared.entries {
            visit(keys);
            visit(values);
        }
    }
    fn visit_arguments<T: Tensor>(
        arguments: &DraftStepArguments<'_, T>,
        visit: &mut dyn FnMut(&T),
    ) {
        visit(arguments.embedding);
        visit(&arguments.state.hidden);
        for (keys, values) in arguments.state.shared_kv.values() {
            visit(keys);
            visit(values);
        }
    }
    fn visit_output<T: Tensor>(output: &T, visit: &mut dyn FnMut(&T)) {
        visit(output);
    }
}
