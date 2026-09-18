//! Final native carrier binding; public native observers still borrow Arrays.
use super::super::speculative::LogitsSource;
use super::*;
use eredu_architectures::speculative_execution::EmbeddedPredictionObservers;
use eredu_runtime::ActivationObserver;

/// Same ordinary static Slice/Squeeze readout used inside the quoted model
/// phase. Its original caller validates its actual scope and prepared slot first.
pub(crate) fn row(value: &Array, row: usize, stream: &Stream) -> Result<Array, Error> {
    let row =
        i32::try_from(row).map_err(|_| Error::InvalidOperation("prediction row exceeds i32"))?;
    value
        .try_index_device((.., row, ..), stream)
        .map_err(Error::from)
}
struct BorrowedArrayObserver(Box<dyn ActivationObserver<Array, Error>>);
impl ActivationObserver<IndependentLogits, Error> for BorrowedArrayObserver {
    fn observe(&mut self, path: &str, value: &IndependentLogits) -> Result<(), Error> {
        let value = value.ordinary().ok_or(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
        ).at_speculative_stage("borrowed prediction logits callback"))?;
        self.0.observe(path, value)
    }
    fn intervene(
        &mut self,
        path: &str,
        value: &IndependentLogits,
    ) -> Result<Option<IndependentLogits>, Error> {
        let value = value.ordinary().ok_or(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
        ).at_speculative_stage("borrowed prediction logits callback"))?;
        self.0
            .intervene(path, value)
            .map(|replacement| replacement.map(IndependentLogits::Ordinary))
    }
}
/// Only the shared outer row observer invokes this adapter (observe/intervene).
/// Unknown callback ownership remains ordinary. The original empty observer set
/// is constructed directly for the closed carrier and needs no Array escape.
pub(in crate::composition::mlx) fn observers(
    observers: EmbeddedPredictionObservers<MlxTensor, Array, Error>,
) -> EmbeddedPredictionObservers<MlxTensor, IndependentLogits, Error> {
    observers.map_logits(|logits| Box::new(BorrowedArrayObserver(logits)))
}

use crate::backend::runtime::cache::state::CompletedResidentSource;
use eredu_architectures::speculative_execution::{
    EmbeddedPredictionLogitBlock, PreparedEmbeddedEvidence,
};
use eredu_core::HostPreparationAuthority;
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use eredu_runtime::working_memory::{SpeculativeNumericalSource, WorkingMemoryError};
use std::mem::{size_of, size_of_val};
fn mismatch() -> Error {
    Error::PrefillControl(WorkingMemoryError::IdentityMismatch)
}
fn source(value: Option<&PreparedEmbeddedEvidence>) -> Result<&CompletedResidentSource, Error> {
    value
        .and_then(PreparedEmbeddedEvidence::get::<CompletedResidentSource>)
        .ok_or_else(mismatch)
}
pub(super) fn selected(
    value: MlxTensor,
    evidence: Option<&PreparedEmbeddedEvidence>,
    context: SpeculativeExecutionStreams<'_>,
) -> Result<IndependentLogits, Error> {
    let Some((sources, environment)) = context.original_numerical() else {
        return Ok(IndependentLogits::Ordinary(value.into_array()));
    };
    if context.embedded_invocation().is_some() || environment.stream() != context.target() {
        return Err(mismatch());
    }
    // The shared prefill driver preserves this exact completed target output.
    // Its final row belongs to the original numerical consumer, including the
    // view descriptor and native completion; no ordinary post-phase index runs.
    if value.as_array().ndim() != 3 || value.as_array().dim(1) <= 0 {
        return Err(mismatch());
    }
    let row = usize::try_from(value.as_array().dim(1) - 1).map_err(|_| mismatch())?;
    super::super::speculative::completed_prediction_logits_row(
        value.as_array(),
        row,
        source(evidence)?,
        sources,
        environment,
    )
}
pub(super) fn source_row(
    value: &MlxTensor,
    index: usize,
    evidence: Option<&PreparedEmbeddedEvidence>,
    context: SpeculativeExecutionStreams<'_>,
    draft: bool,
) -> Result<IndependentLogits, Error> {
    let stream = if draft {
        context.draft()
    } else {
        context.target()
    };
    let Some((sources, environment)) = context.original_numerical() else {
        return row(value.as_array(), index, stream).map(IndependentLogits::Ordinary);
    };
    if context.embedded_invocation().is_some() || environment.stream() != stream {
        return Err(mismatch());
    }
    super::super::speculative::completed_prediction_logits_row(
        value.as_array(),
        index,
        source(evidence)?,
        sources,
        environment,
    )
}
pub(super) fn retain_block(
    value: MlxTensor,
    evidence: Option<PreparedEmbeddedEvidence>,
    context: SpeculativeExecutionStreams<'_>,
) -> Result<EmbeddedPredictionLogitBlock<MlxTensor>, Error> {
    let Some((sources, environment)) = context.original_numerical() else {
        return Ok(EmbeddedPredictionLogitBlock::ordinary(value));
    };
    if context.embedded_invocation().is_some() || environment.stream() != context.draft() {
        return Err(mismatch());
    }
    sources.validate_environment(environment)?;
    let completed = source(evidence.as_ref())?;
    let overflow = || Error::WorkspacePlanning(HostMetadataFundingError::Overflow);
    let parts = [
        EmbeddedPredictionLogitBlock::<MlxTensor>::retained_control_bytes().ok_or_else(overflow)?,
        HostPreparationAuthority::retention_bytes::<HostMetadataFunding>()
            .ok_or_else(overflow)?,
        size_of::<Result<EmbeddedPredictionLogitBlock<MlxTensor>, Error>>(),
        size_of::<Option<PreparedEmbeddedEvidence>>(),
        size_of::<HostPreparationAuthority>(),
        CompletedResidentSource::array_source_control_bytes().ok_or_else(overflow)?,
        completed
            .completed_stream_control_bytes()
            .ok_or_else(mismatch)?,
    ];
    let funding = sources.metadata_funding();
    funding
        .reserve_metadata(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or_else(overflow)?,
        )
        .map_err(Error::WorkspacePlanning)?;
    completed.validate_completed_stream(environment.stream())?;
    let (_, account) = completed.array_source(value.as_array(), funding)?;
    if !SpeculativeNumericalSource::Model(account).belongs_to_request(sources.request()) {
        return Err(mismatch());
    }
    Ok(EmbeddedPredictionLogitBlock::from_prepared(
        value,
        evidence.expect("validated output evidence"),
        HostPreparationAuthority::retain(funding.clone()),
    ))
}

/// Callback ownership is checked before capability queries or actual callbacks.
pub(super) fn validate_observer(
    has_caller: bool,
    context: SpeculativeExecutionStreams<'_>,
) -> Result<(), Error> {
    if let Some((sources, environment)) = context.original_numerical() {
        sources.validate_environment(environment)?;
        if context.embedded_invocation().is_some() {
            return Err(mismatch());
        }
        if has_caller {
            return Err(Error::PrefillControl(WorkingMemoryError::UnknownBound).at_speculative_stage("outer prediction callback ownership"));
        }
    }
    Ok(())
}

/// A move creates no tensor, source registration or callback qualification.
/// Only the structural empty slot can use it on the original execution path.
pub(super) fn observe_owned(
    value: MlxTensor,
    path: &str,
    chunk: Option<&eredu_runtime::prefill::PrefillChunk>,
    observer: Option<&mut dyn ActivationObserver<MlxTensor, Error>>,
    context: SpeculativeExecutionStreams<'_>,
) -> Result<MlxTensor, Error> {
    let Some((sources, _environment)) = context.original_numerical() else {
        return eredu_architectures::speculative_execution::observe_embedded_tensor_ordinary(
            value, path, chunk, observer,
        );
    };
    validate_observer(observer.is_some(), context)?;
    let parts = [
        size_of::<MlxTensor>(),
        size_of::<Result<MlxTensor, Error>>(),
        size_of::<Option<&mut dyn ActivationObserver<MlxTensor, Error>>>(),
        size_of::<Option<&eredu_runtime::prefill::PrefillChunk>>(),
        size_of::<&str>(),
        size_of::<SpeculativeExecutionStreams<'_>>(),
    ];
    let bytes = parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
        .ok_or(Error::WorkspacePlanning(
            HostMetadataFundingError::Overflow,
        ))?;
    sources
        .metadata_funding()
        .reserve_metadata(bytes)
        .map_err(Error::WorkspacePlanning)?;
    Ok(value)
}
