//! Source-bound lazy readouts use the existing original numerical compiler.
use super::*;
use crate::backend::{OriginalCopyEnvironment, runtime::cache::state::CompletedResidentSource};
use eredu_nn::{
    Index, Tensor,
    workspace::{WorkspaceContext, WorkspaceMetadataError, WorkspaceTensor},
};
use eredu_runtime::working_memory::WorkingMemoryError;
use safemlx::{
    OriginalBufferBudget, PreparedArrayClone, PreparedArrayCloneCause, PreparedStreamCopy,
};

fn invalid() -> Error {
    Error::PrefillControl(WorkingMemoryError::IdentityMismatch)
}
fn overflow() -> Error {
    Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow)
}
struct ModelSource {
    stream: PreparedStreamCopy<model::StreamOwner>,
    budget: OriginalBufferBudget,
    custody: OriginalSpeculativeBudgetCustody,
    funding: WorkspaceMetadataFunding,
}
impl ModelSource {
    fn prepare(
        array: &Array,
        completed: &CompletedResidentSource,
        sources: &OriginalSpeculativeNumericalSources,
        environment: &OriginalCopyEnvironment<'_>,
    ) -> Result<Self, Error> {
        sources.validate_environment(environment)?;
        let funding = sources.metadata_funding();
        let controls = [
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<Result<OriginalNumericalValue, Error>>(),
            size_of::<(
                &Array,
                &CompletedResidentSource,
                &OriginalSpeculativeNumericalSources,
                &OriginalCopyEnvironment<'_>,
            )>(),
            size_of::<(OriginalBufferBudget, OriginalSpeculativeBudgetCustody)>(),
            value_control_bytes().ok_or_else(overflow)?,
            CompletedResidentSource::array_source_control_bytes().ok_or_else(overflow)?,
            completed
                .completed_stream_control_bytes()
                .ok_or_else(invalid)?,
        ];
        funding
            .reserve_metadata(
                controls
                    .into_iter()
                    .try_fold(size_of_val(&controls), usize::checked_add)
                    .ok_or_else(overflow)?,
            )
            .map_err(Error::WorkspacePlanning)?;
        completed.validate_completed_stream(environment.stream())?;
        if !(2..=3).contains(&array.ndim())
            || array.dim(0) != 1
            || array.shape().iter().any(|&extent| extent <= 0)
            || array.dtype() != safemlx::Dtype::Float32
        {
            return Err(invalid());
        }
        let (budget, custody) = completed.array_source(array, funding)?;
        if !SpeculativeNumericalSource::Model(custody).belongs_to_request(sources.request()) {
            return Err(invalid());
        }
        let stream = model::prepare_stream(custody.clone(), environment.stream(), funding)?;
        Ok(Self {
            stream,
            budget: budget.clone(),
            custody: custody.clone(),
            funding: funding.clone(),
        })
    }
    // Private callers either move the exact inspected array or fill a prepared
    // clone from that same borrowed descriptor. No arbitrary raw-array adoption.
    fn finish(self, array: Array) -> OriginalNumericalValue {
        OriginalNumericalValue(
            Some(Rc::new(Value {
                array,
                original_budget: Some(self.budget),
                stream: ValueStream::Embedded(self.stream),
                meaning: Meaning::Logits,
                provenance: Provenance::Model(self.custody),
                funding: self.funding,
                _copy: None,
                _snapshot_host: None,
                _readout_source: None,
            })),
            None,
        )
    }
}
/// Selected prefill scores already own their exact completed native array; this
/// move constructs only the paid closed numerical owner and stream custody.
pub(crate) fn completed_logits(
    array: Array,
    completed: &CompletedResidentSource,
    sources: &OriginalSpeculativeNumericalSources,
    environment: &OriginalCopyEnvironment<'_>,
) -> Result<super::super::logits::IndependentLogits, Error> {
    let owner = ModelSource::prepare(&array, completed, sources, environment)?;
    Ok(super::super::logits::IndependentLogits::Original(
        owner.finish(array),
    ))
}
fn borrow_completed(
    array: &Array,
    completed: &CompletedResidentSource,
    sources: &OriginalSpeculativeNumericalSources,
    environment: &OriginalCopyEnvironment<'_>,
) -> Result<OriginalNumericalValue, Error> {
    let owner = ModelSource::prepare(array, completed, sources, environment)?;
    #[derive(Debug, thiserror::Error)]
    #[error("{cause}")]
    struct Failure {
        #[source]
        cause: PreparedArrayCloneCause,
        _funding: WorkspaceMetadataFunding,
    }
    let controls = [
        PreparedArrayClone::control_bytes().ok_or_else(overflow)?,
        Array::inspection_clone_handle_bytes(),
        size_of::<Failure>(),
        eredu_core::BackendFailure::source_retention_peak_bytes::<Failure>()
            .ok_or_else(overflow)?,
        size_of::<Result<Array, PreparedArrayCloneCause>>(),
        size_of::<OriginalNumericalValue>(),
    ];
    owner
        .funding
        .reserve_metadata(
            controls
                .into_iter()
                .try_fold(size_of_val(&controls), usize::checked_add)
                .ok_or_else(overflow)?,
        )
        .map_err(Error::WorkspacePlanning)?;
    let fail = |cause| {
        Error::StorageSource(eredu_core::BackendFailure::from_error(Failure {
            cause,
            _funding: owner.funding.clone(),
        }))
    };
    let mut slot = PreparedArrayClone::try_prepare_for_inspection().map_err(&fail)?;
    let array = slot.fill_for_inspection(array).map_err(fail)?;
    Ok(owner.finish(array))
}
/// Called only for the requested row after the source model phase completed.
/// It creates one ordinary static-index equation inside the shared numerical
/// plan/claim/Scope/completion driver, with the actual source account retained.
pub(crate) fn logits_row(
    array: &Array,
    row: usize,
    completed: &CompletedResidentSource,
    sources: &OriginalSpeculativeNumericalSources,
    environment: &OriginalCopyEnvironment<'_>,
) -> Result<super::super::logits::IndependentLogits, Error> {
    let row = u32::try_from(row).map_err(|_| invalid())?;
    let kind = program::SpeculativeNumericalKind::LogitsRow { row };
    let source = borrow_completed(array, completed, sources, environment)?;
    let (roots, mechanisms) = sources.numerical_prerequisites();
    match NumericalProducer::execute(sources, environment, roots, mechanisms, kind, &source, None)?
    {
        NumericalOutput::Logits(value) => {
            Ok(super::super::logits::IndependentLogits::Original(value))
        }
        _ => Err(invalid()),
    }
}
/// Imports a completed model/numerical/copy source with its actual stream and
/// full evidence custody. No numerical operation or completion is fabricated.
pub(crate) fn completed_logits_at(value:&crate::MlxTensor,
    evidence:&eredu_architectures::speculative_execution::PreparedEmbeddedEvidence,
    context:crate::composition::mlx::speculative::SpeculativeExecutionStreams<'_>,
    placement:eredu_core::speculative::SamplingPlacement,
)->Result<super::super::logits::IndependentLogits,Error>{
    readout_controls(context,placement)?;
    tensor::borrow_logits(value,evidence,context,placement)
        .map(super::super::logits::IndependentLogits::Original)
}
/// One actual static row executes on the selected destination after importing
/// the closed completed source at its original placement.
pub(crate) fn logits_row_at(value:&crate::MlxTensor,row:usize,
    evidence:&eredu_architectures::speculative_execution::PreparedEmbeddedEvidence,
    context:crate::composition::mlx::speculative::SpeculativeExecutionStreams<'_>,
    placement:eredu_core::speculative::SamplingPlacement,
)->Result<super::super::logits::IndependentLogits,Error>{
    readout_controls(context,placement)?;
    let row=u32::try_from(row).map_err(|_|invalid())?;
    let source=tensor::borrow_logits(value,evidence,context,placement)?;
    match NumericalProducer::execute_at(context,placement,
        program::SpeculativeNumericalKind::LogitsRow{row},&source,None)? {
        NumericalOutput::Logits(value)=>Ok(super::super::logits::IndependentLogits::Original(value)),
        _=>Err(invalid()),
    }
}
fn readout_controls(context:crate::composition::mlx::speculative::SpeculativeExecutionStreams<'_>,
    placement:eredu_core::speculative::SamplingPlacement)->Result<(),Error>{
    let (sources,_)=context.original_numerical_for(placement).ok_or_else(invalid)?;
    let frames=[size_of::<(&crate::MlxTensor,usize,
            &eredu_architectures::speculative_execution::PreparedEmbeddedEvidence,
            crate::composition::mlx::speculative::SpeculativeExecutionStreams<'_>,
            eredu_core::speculative::SamplingPlacement)>(),
        size_of::<OriginalNumericalValue>(),size_of::<Result<OriginalNumericalValue,Error>>(),
        size_of::<super::super::logits::IndependentLogits>(),
        size_of::<Result<super::super::logits::IndependentLogits,Error>>(),
        size_of::<NumericalOutput>(),size_of::<Result<NumericalOutput,Error>>(),
        size_of::<u32>(),size_of::<program::SpeculativeNumericalKind>(),
        size_of::<(crate::composition::mlx::speculative::SpeculativeExecutionStreams<'_>,
            eredu_core::speculative::SamplingPlacement)>(),size_of::<Result<(),Error>>()];
    sources.metadata_funding().reserve_metadata(frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
        .ok_or_else(overflow)?).map_err(Error::WorkspacePlanning)
}
pub(super) fn metadata_row(
    value: &WorkspaceTensor,
    row: u32,
    context: &WorkspaceContext,
) -> Result<WorkspaceTensor, eredu_nn::Error> {
    let parts = [
        size_of::<(&WorkspaceTensor, u32, &WorkspaceContext)>(),
        size_of::<[Index; 3]>(),
        size_of::<Result<WorkspaceTensor, eredu_nn::Error>>(),
    ];
    context.charge_metadata(
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or(WorkspaceMetadataError::Overflow)?,
    )?;
    value.index(
        &[
            Index::Full,
            Index::At(i32::try_from(row).map_err(|_| WorkspaceMetadataError::Overflow)?),
            Index::Full,
        ],
        context,
    )
}
