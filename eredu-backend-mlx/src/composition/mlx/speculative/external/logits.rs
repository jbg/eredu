//! External readouts reuse the completed model source and numerical row worker.
use super::*;
use eredu_architectures::speculative_execution::PreparedEmbeddedEvidence;
use eredu_nn::workspace::WorkspaceMetadataFundingError;
use eredu_runtime::working_memory::WorkingMemoryError;
use std::mem::{size_of, size_of_val};

pub(super) fn selected(value: MlxTensor, evidence: Option<&PreparedEmbeddedEvidence>,
    context: SpeculativeExecutionStreams<'_>) -> Result<IndependentLogits, Error> {
    selected_at(value,evidence,ExternalAssistantTensorPlacement::Target,context)
}
pub(super) fn selected_at(value:MlxTensor,evidence:Option<&PreparedEmbeddedEvidence>,
    placement:ExternalAssistantTensorPlacement,context:SpeculativeExecutionStreams<'_>,
)->Result<IndependentLogits,Error>{
    let selected_placement=match placement {
        ExternalAssistantTensorPlacement::Target=>eredu_core::speculative::SamplingPlacement::Target,
        ExternalAssistantTensorPlacement::Draft=>eredu_core::speculative::SamplingPlacement::Draft,
    };
    let Some((sources, _environment)) = context.original_numerical_for(selected_placement) else {
        return Ok(IndependentLogits::Ordinary(value.into_array()));
    };
    let run = || {
        let controls = [size_of::<MlxTensor>(), size_of::<IndependentLogits>(),
            size_of::<Result<IndependentLogits, Error>>(), size_of::<SpeculativeExecutionStreams<'_>>(),
            size_of::<Option<&PreparedEmbeddedEvidence>>(),
            size_of::<ExternalAssistantTensorPlacement>(),size_of::<eredu_core::speculative::SamplingPlacement>(),
            size_of::<(MlxTensor,Option<&PreparedEmbeddedEvidence>,ExternalAssistantTensorPlacement,
                SpeculativeExecutionStreams<'_>)>(),
            size_of::<(MlxTensor,Option<&PreparedEmbeddedEvidence>,SpeculativeExecutionStreams<'_>)>(),
            Array::descriptor_control_bytes().ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?];
        sources.metadata_funding().reserve_metadata(controls.into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?)
            .map_err(Error::WorkspacePlanning)?;
        if context.original_external().is_none() {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        let descriptor = value.as_array().try_descriptor()?;
        let selected = match descriptor.shape() {
            [batch, vocabulary] if *batch > 0 && *vocabulary > 0 => None,
            [batch, positions, vocabulary] if *batch > 0 && *positions > 0 && *vocabulary > 0 =>
                Some(usize::try_from(*positions - 1).map_err(|_| Error::PrefillControl(WorkingMemoryError::Overflow))?),
            _ => return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch)),
        };
        drop(descriptor);
        let evidence=evidence.ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
        match selected {
            Some(row)=>super::super::sampling::numerical::logits_row_at(&value,row,evidence,context,
                selected_placement),
            None=>super::super::sampling::numerical::completed_logits_at(&value,evidence,context,
                selected_placement),
        }
    };
    run().map_err(|cause| sources.retain_error(cause))
}
pub(super) fn row(value: &MlxTensor, row: usize, evidence: Option<&PreparedEmbeddedEvidence>,
    placement: ExternalAssistantTensorPlacement, context: SpeculativeExecutionStreams<'_>,
) -> Result<IndependentLogits, Error> {
    let stream = match placement {
        ExternalAssistantTensorPlacement::Target => context.target(),
        ExternalAssistantTensorPlacement::Draft => context.draft(),
    };
    let selected_placement=match placement {
        ExternalAssistantTensorPlacement::Target=>eredu_core::speculative::SamplingPlacement::Target,
        ExternalAssistantTensorPlacement::Draft=>eredu_core::speculative::SamplingPlacement::Draft,
    };
    let Some((sources, environment)) = context.original_numerical_for(selected_placement) else {
        let index = i32::try_from(row).map_err(|_| ordinary_error("external assistant row exceeds i32"))?;
        return value.as_array().try_index_device((.., index, ..), stream)
            .map(IndependentLogits::Ordinary).map_err(Error::from);
    };
    let run = || {
        let controls = [size_of::<(&MlxTensor, usize, Option<&PreparedEmbeddedEvidence>, ExternalAssistantTensorPlacement,
                SpeculativeExecutionStreams<'_>)>(), size_of::<Result<IndependentLogits, Error>>()];
        sources.metadata_funding().reserve_metadata(controls.into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .ok_or(Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow))?)
            .map_err(Error::WorkspacePlanning)?;
        if context.original_external().is_none() || stream != environment.stream() {
            return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
        }
        super::super::sampling::numerical::logits_row_at(value,row,
            evidence.ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?,context,selected_placement)
    };
    run().map_err(|cause| sources.retain_error(cause))
}
