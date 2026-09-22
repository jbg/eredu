//! Closed receipt owner lent to the ordinary funded observer transaction seams.
use super::*;
mod interventions;
mod local_hook;
mod run_identity;
pub use interventions::PreparedPartitionInterventionEvidence;
mod contiguous;
mod projected;
mod routed_hooks;
use crate::working_memory::{
    PartitionCaptureTensorDeliveryError, PreparedPartitionTensorDelivery, ScheduledCaptureStep,
};
pub use contiguous::{
    PartitionCaptureLocalSource, PartitionCaptureRoutedLocalSource,
    PreparedPartitionContiguousSource, PreparedPartitionRoutedSource,
};
pub(crate) use contiguous::{PreparedPartitionContiguousRow, PreparedPartitionRoutedLocalSource};
use eredu_core::{Completion, DistributedCommitEpoch, checkpoint::TensorDtype};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
pub use local_hook::PartitionCaptureLocalHook;
pub use routed_hooks::PartitionCaptureRoutedHooks;
pub use run_identity::PreparedPartitionCaptureRunIdentity;
use std::mem::{size_of, size_of_val};

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("{0}")]
    Coordinates(#[from] eredu_core::component::ComponentCoordinateConstructionError),

    #[error("original partition observer source: {0}")]
    Source(&'static str),
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
    #[error(transparent)]
    Destination(#[from] eredu_nn::Error),
    #[error(transparent)]
    Receipt(#[from] PartitionCaptureReceiptConstructionError),
    #[error(transparent)]
    Allowance(#[from] PartitionCaptureAllowanceError),
    #[error(transparent)]
    FragmentAllowance(#[from] PartitionCaptureFragmentAllowanceError),
    #[error(transparent)]
    Coordination(#[from] PartitionCaptureCoordinationError),
    #[error(transparent)]
    Exchange(#[from] PartitionCaptureExchangeError),
    #[error(transparent)]
    Evidence(#[from] PartitionCaptureEvidenceError),
    #[error(transparent)]
    Delivery(#[from] PartitionCaptureTensorDeliveryError),
    #[error(transparent)]
    Host(#[from] crate::working_memory::CaptureRunHostError),
    #[error(transparent)]
    Capture(#[from] CaptureError),
    #[error(transparent)]
    Local(eredu_core::BackendFailure),
    #[error(transparent)]
    Intervention(#[from] PartitionInterventionSourceError),
    #[error(transparent)]
    Progress(#[from] crate::capture::CapturePrefillProgressError),
}
/// Keeps the selected admission and paid constructor/transport error alive on
/// every refusal. The observer retains the partial original frame separately.
#[derive(Debug, thiserror::Error)]
#[error("original partition observer: {cause}")]
pub struct PartitionCaptureProgramError {
    #[source]
    cause: Cause,
    _source: SharedCapturePlan,
    _metadata: HostMetadataFunding,
}
mod sealed {
    pub trait Sealed {}
}
/// A runtime-owned program around the existing receipt transaction. Backends
/// lend this owner and separately prove the loaded placement, transform and
/// native transport. External implementations cannot replace its quota worker.
pub trait ScheduledPartitionCapture: sealed::Sealed {
    /// Prepare once under the same logical frame and actual first forward epoch.
    fn prepare(
        &mut self,
        source: &SharedCapturePlan,
        phase: CapturePhase,
        prediction: u64,
        epoch: DistributedCommitEpoch,
        frame: &mut ScheduledCaptureStep<'_>,
        ledger: &mut CaptureLedger,
    ) -> Result<(), PartitionCaptureProgramError>;
    /// Compare the prepared transcript after the shared local preparation vote.
    fn coordinate(
        &mut self,
        epoch: DistributedCommitEpoch,
        ledger: &CaptureLedger,
    ) -> Result<(), PartitionCaptureProgramError>;
    /// Borrow the original companion receipt worker. The caller separately
    /// lends its already allocated evidence frame and actual before/after value.
    fn intervention_evidence(
        &mut self,
        index: usize,
    ) -> Result<Option<&mut (dyn ScheduledPartitionCapture + '_)>, PartitionCaptureProgramError>;
    /// Actual operation membership; absent pipeline ranks do not enter member votes.
    fn intervention_member(
        &self,
        _index: usize,
        _window: Option<crate::intervention::InterventionPrefillWindow>,
    ) -> Result<Option<bool>, PartitionCaptureProgramError> {
        Ok(None)
    }
    /// Issue only the prepaid local loan after the existing preflight member vote.
    fn begin_intervention(
        &mut self,
        index: usize,
        window: Option<crate::intervention::InterventionPrefillWindow>,
        valid: bool,
    ) -> Result<Option<PartitionInterventionLocalAllowance>, PartitionCaptureProgramError>;
    /// Return this exact local loan for the existing second member vote.
    fn finish_intervention(
        &mut self,
        index: usize,
        loan: PartitionInterventionLocalAllowance,
        success: bool,
    ) -> Result<(), PartitionCaptureProgramError>;
    /// Validate actual member cursors; absent ranks await the final world receipt.
    fn validate_intervention_prefill_end(
        &self,
        _frame: &ScheduledCaptureStep<'_>,
        _window: crate::intervention::InterventionPrefillWindow,
    ) -> Result<bool, PartitionCaptureProgramError> {
        Ok(false)
    }
    /// Whether this selected hook has an actual tensor producer on this rank.
    fn produces(&self, index: usize) -> Result<bool, PartitionCaptureProgramError>;
    /// Spend one reached local replica hook in the current coordinated epoch.
    /// None denotes an inactive selection or the producer, whose transform owns
    /// its own source completion. No source is acquired for an absent PP rank.
    fn take_receiver_source(
        &mut self,
        index: usize,
    ) -> Result<Option<TensorDtype>, PartitionCaptureProgramError>;
    /// Spend an actual empty-overlap producer's local prefill source once for
    /// this coordinated epoch. Complete/global receivers keep the existing path.
    fn take_projected_receiver_source(
        &mut self,
        _index: usize,
        _inference: eredu_core::InferenceGeometry,
        _chunk: u64,
    ) -> Result<Option<PartitionPrefillReceiverSource>, PartitionCaptureProgramError> {
        Ok(None)
    }
    /// The same original local receipt loan for a single decode invocation.
    fn take_invocation_projection(
        &mut self,
        _index: usize,
    ) -> Result<Option<PartitionCaptureLocalHook>, PartitionCaptureProgramError> {
        Ok(None)
    }
    /// Return the same original local owner before ordinary delivery resumes.
    fn return_invocation_projection(
        &mut self,
        index: usize,
        hook: PartitionCaptureLocalHook,
    ) -> Result<(), PartitionCaptureProgramError> {
        self.return_prefill_projection(index, hook)
    }
    /// Actual local decode source whose selected overlap produces no fragment.
    fn take_invocation_receiver_source(
        &mut self,
        _index: usize,
    ) -> Result<Option<super::PartitionInvocationReceiverSource>, PartitionCaptureProgramError>
    {
        Ok(None)
    }
    /// Exact runtime sparse program; presence grants no native capability.
    fn routed_source(&self, _index: usize) -> Result<bool, PartitionCaptureProgramError> {
        Ok(false)
    }
    /// Suspend the existing receipt owners while the native provider is borrowed.
    fn take_routed_hooks(
        &mut self,
        _index: usize,
        _frame: &mut ScheduledCaptureStep<'_>,
        _prefill: Option<(&crate::prefill::PrefillChunk, eredu_core::InferenceGeometry)>,
    ) -> Result<Option<PartitionCaptureRoutedHooks>, PartitionCaptureProgramError> {
        Ok(None)
    }
    /// Return the same finished original hook table before transport resumes.
    fn return_routed_hooks(
        &mut self,
        hooks: PartitionCaptureRoutedHooks,
        _frame: &mut ScheduledCaptureStep<'_>,
        _prefill: Option<(&crate::prefill::PrefillChunk, eredu_core::InferenceGeometry)>,
    ) -> Result<(), PartitionCaptureProgramError> {
        Err(hooks.reject("routed hook returned to another program"))
    }
    /// Validate the actual dtype and estimate, then lend only this local quota.
    fn reservation(
        &mut self,
        index: usize,
        dtype: &TensorDtype,
        usage: CaptureUsage,
    ) -> Result<&mut CaptureQuota, PartitionCaptureProgramError>;
    /// Lend the actual projected host/receipt owner after the canonical global
    /// hook has begun. Complete-producer rows retain their ordinary destination.
    fn take_prefill_projection(
        &mut self,
        _index: usize,
        _frame: &mut ScheduledCaptureStep<'_>,
        _first: bool,
    ) -> Result<Option<PartitionCaptureLocalHook>, PartitionCaptureProgramError> {
        Ok(None)
    }
    /// Return the same original owner before delivery can resume. This consumes
    /// a rejected owner as well; it can never become another row's destination.
    fn return_prefill_projection(
        &mut self,
        index: usize,
        hook: PartitionCaptureLocalHook,
    ) -> Result<(), PartitionCaptureProgramError>;
    /// Account real announced prefill progress on receivers without inspecting
    /// a tensor or creating an independent native usage reservation.
    fn remote_prefill(
        &mut self,
        frame: &mut ScheduledCaptureStep<'_>,
        bound: crate::layered::BoundCaptureSelection<'_>,
        chunk: &crate::prefill::PrefillChunk,
    ) -> Result<(), PartitionCaptureProgramError>;
    /// Validate companion hooks after the actual model chunk returns. Early
    /// receiver source progress does not complete local before/after callbacks.
    fn complete_prefill_evidence(
        &mut self,
        frame: &mut ScheduledCaptureStep<'_>,
        bound: crate::layered::BoundCaptureSelection<'_>,
        chunk: &crate::prefill::PrefillChunk,
    ) -> Result<(), PartitionCaptureProgramError>;
    /// Fill receivers through the ordinary all-rank exchange before frame seal.
    fn deliver(
        &mut self,
        frame: &mut ScheduledCaptureStep<'_>,
    ) -> Result<(), PartitionCaptureProgramError>;
}
/// Payload-free facts from the selected architecture/native cold source. These
/// do not themselves grant placement, scalar conversion or communication.
#[derive(Debug)]
pub struct PartitionCaptureProducerSource {
    /// Exact complete-global producer selected by the architecture plan.
    pub producer: usize,
    /// Actual source floating representation from the cold native witness.
    /// Absent on inactive or already-skipped hooks; a reached admitted tensor
    /// row still requires the exact witness before any native work.
    pub dtype: Option<TensorDtype>,
    /// Existing full-selection transform and generated-source usage.
    pub estimate: PartitionCaptureNativeEstimate,
}
/// Cold row choice for the same receipt program. A contiguous slot must be
/// filled with its actual retained per-rank source and original Host owner
/// before preparation; it does not impersonate a complete-global producer.
#[derive(Clone, Copy)]
pub enum PreparedPartitionCaptureRow<'a> {
    /// Exact schedule proves this row has no invocation in this frame.
    Inactive,
    Complete(&'a PartitionCaptureProducerSource),
    Contiguous,
    /// Exact sparse source and original fragment Host must be attached before preparation.
    Routed,
}
struct Entry<'t, T: PartitionCaptureTransport> {
    delivery: Option<PreparedPartitionTensorDelivery<'t, T>>,
    allowance: PreparedPartitionCaptureAllowance,
}
/// One fixed selection table for a logical original frame. Every row follows
/// the same ordinary receipt protocol; no collective occurs in a tensor hook.
pub struct PreparedPartitionCaptureProgram<'t, T: PartitionCaptureTransport> {
    transport: &'t T,
    run_identity: Option<PreparedPartitionCaptureRunIdentity>,
    context: PartitionCaptureContext,
    limits: PartitionCaptureReceiptLimits,
    limit_policy: CaptureLimitPolicy,
    rows: Vec<Option<PartitionCaptureProducerSource>>,
    entries: Vec<Option<Entry<'t, T>>>,
    projected: Vec<Option<projected::Row<'t, T>>>,
    routed_hooks: Option<Vec<routed_hooks::Slot>>,
    interventions: Option<interventions::Rows<'t, T>>,
    skipped: Vec<Option<CaptureSkipReason>>,
    receiver_epochs: Vec<Option<DistributedCommitEpoch>>,
    coordination: Option<PreparedPartitionCaptureCoordination<'t, T>>,
    first_epoch: Option<DistributedCommitEpoch>,
    last_epoch: Option<DistributedCommitEpoch>,
    coordination_complete: bool,
    delivered: bool,
    source: SharedCapturePlan,
    metadata: HostMetadataFunding,
}
impl<T: PartitionCaptureTransport> std::fmt::Debug for PreparedPartitionCaptureProgram<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedPartitionCaptureProgram")
            .field("rows", &self.rows.len())
            .field("prepared", &self.first_epoch.is_some())
            .field("delivered", &self.delivered)
            .finish_non_exhaustive()
    }
}
impl<T: PartitionCaptureTransport> sealed::Sealed for PreparedPartitionCaptureProgram<'_, T> {}
impl<'t, T: PartitionCaptureTransport> PreparedPartitionCaptureProgram<'t, T>
where
    T::Error: Send + Sync + 'static,
    <T::Completion as Completion>::Error: Send + Sync + 'static,
{
    /// Retain paid host destinations for an exact shared plan and loaded context.
    /// Native source/transport qualification remains mandatory before lending
    /// the owner to a managed model callback.
    pub fn new(
        transport: &'t T,
        source: &SharedCapturePlan,
        context: &PartitionCaptureContext,
        rows: &[PartitionCaptureProducerSource],
        limits: PartitionCaptureReceiptLimits,
        metadata: &HostMetadataFunding,
    ) -> Result<Self, PartitionCaptureProgramError> {
        Self::new_rows(
            transport,
            source,
            context,
            rows.iter().map(PreparedPartitionCaptureRow::Complete),
            limits,
            metadata,
        )
    }
    /// Retain complete and projected row slots without manufacturing a global
    /// source for a partition. Every projected slot must be attached once.
    pub fn new_selected(
        transport: &'t T,
        source: &SharedCapturePlan,
        context: &PartitionCaptureContext,
        rows: &[PreparedPartitionCaptureRow<'_>],
        limits: PartitionCaptureReceiptLimits,
        metadata: &HostMetadataFunding,
    ) -> Result<Self, PartitionCaptureProgramError> {
        Self::new_rows(
            transport,
            source,
            context,
            rows.iter().copied(),
            limits,
            metadata,
        )
    }
    fn row_constructor_control_bytes<I>() -> Option<usize> {
        let parts = [
            size_of::<Self>() * 2,
            size_of::<Result<Self, PartitionCaptureProgramError>>(),
            size_of::<PartitionCaptureProgramError>(),
            size_of::<Cause>(),
            size_of::<PartitionCaptureContext>() * 2,
            size_of::<PartitionCaptureProducerSource>(),
            size_of::<I>(),
            size_of::<Option<PartitionCaptureProducerSource>>(),
            size_of::<PreparedPartitionCaptureRow<'_>>(),
            size_of::<(
                &T,
                &SharedCapturePlan,
                &PartitionCaptureContext,
                &[PreparedPartitionCaptureRow<'_>],
                PartitionCaptureReceiptLimits,
                &HostMetadataFunding,
            )>(),
            size_of::<Option<Entry<'t, T>>>(),
            size_of::<Vec<Option<DistributedCommitEpoch>>>(),
            size_of::<Option<DistributedCommitEpoch>>(),
            size_of::<(
                &T,
                &SharedCapturePlan,
                &PartitionCaptureContext,
                &[PartitionCaptureProducerSource],
                &HostMetadataFunding,
            )>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Metadata for attaching each actual contiguous/routed source row to a
    /// freshly constructed program. Counts share one projected table; routed
    /// rows also share one hook table. Source and Host owners are supplied and
    /// priced separately, as are later receipt preparation and delivery.
    pub fn projected_rows_metadata_bytes(
        rows: usize,
        contiguous: usize,
        routed: usize,
    ) -> Option<usize> {
        use eredu_nn::workspace::WorkspaceContext;
        let projected = contiguous.checked_add(routed)?;
        if projected > rows {
            return None;
        }
        if projected == 0 {
            return Some(0);
        }
        let mut bytes = projected::control_bytes::<T>()?
            .checked_mul(projected)?
            .checked_add(WorkspaceContext::metadata_vec_bytes::<
                Option<projected::Row<'t, T>>,
            >(rows)?)?;
        if routed != 0 {
            bytes = bytes
                .checked_add(
                    PartitionCaptureRoutedHooks::table_control_bytes()?.checked_mul(routed)?,
                )?
                .checked_add(WorkspaceContext::metadata_vec_bytes::<routed_hooks::Slot>(
                    rows,
                )?)?;
        }
        Some(bytes)
    }
    fn selected_rows_metadata_bytes(count: usize) -> Option<usize> {
        use eredu_nn::workspace::WorkspaceContext;
        type Rows<'a> = std::iter::Copied<std::slice::Iter<'a, PreparedPartitionCaptureRow<'a>>>;
        Self::row_constructor_control_bytes::<Rows<'_>>()?
            .checked_add(WorkspaceContext::metadata_vec_bytes::<
                Option<PartitionCaptureProducerSource>,
            >(count)?)?
            .checked_add(WorkspaceContext::metadata_vec_bytes::<Option<Entry<'t, T>>>(count)?)?
            .checked_add(WorkspaceContext::metadata_vec_bytes::<
                Option<CaptureSkipReason>,
            >(count)?)?
            .checked_add(WorkspaceContext::metadata_vec_bytes::<
                Option<DistributedCommitEpoch>,
            >(count)?)
    }
    fn new_rows<'a, I: ExactSizeIterator<Item = PreparedPartitionCaptureRow<'a>>>(
        transport: &'t T,
        source: &SharedCapturePlan,
        context: &PartitionCaptureContext,
        rows: I,
        limits: PartitionCaptureReceiptLimits,
        metadata: &HostMetadataFunding,
    ) -> Result<Self, PartitionCaptureProgramError> {
        let error = |cause| PartitionCaptureProgramError {
            cause,
            _source: source.clone(),
            _metadata: metadata.clone(),
        };
        let bytes = Self::row_constructor_control_bytes::<I>()
            .ok_or_else(|| error(Cause::Source("constructor controls overflow")))?;
        metadata
            .reserve_metadata(bytes)
            .map_err(|cause| error(cause.into()))?;
        if rows.len() != source.admission().plan().selections.len()
            || context.capture_plan_identity != source.admission().identity()
            || transport.participant_count() == 0
            || transport.capture_rank() >= transport.participant_count()
            || limits.max_producers == 0
            || limits.max_fragments == 0
            || limits.max_record_bytes == 0
        {
            return Err(error(Cause::Source(
                "selected rows, context or world differs",
            )));
        }
        context.validate().map_err(|cause| error(cause.into()))?;
        source
            .admission()
            .geometry_at(context.phase, context.prediction, context.invocation)
            .map_err(|cause| error(cause.into()))?;
        let count = rows.len();
        let mut owned_rows = metadata
            .metadata_vec(count)
            .map_err(|cause| error(cause.into()))?;
        let mut entries = metadata
            .metadata_vec(count)
            .map_err(|cause| error(cause.into()))?;
        let mut skipped = metadata
            .metadata_vec(count)
            .map_err(|cause| error(cause.into()))?;
        let mut receiver_epochs = metadata
            .metadata_vec(count)
            .map_err(|cause| error(cause.into()))?;
        for (selection, row) in source.admission().plan().selections.iter().zip(rows) {
            let row = match row {
                PreparedPartitionCaptureRow::Inactive => {
                    if selection
                        .schedule
                        .includes(context.phase, context.prediction)
                    {
                        return Err(error(Cause::Source(
                            "active capture row has no source declaration",
                        )));
                    }
                    None
                }
                PreparedPartitionCaptureRow::Complete(row) => {
                    if row.producer >= transport.participant_count()
                        || !matches!(
                            row.dtype,
                            None | Some(TensorDtype::F32 | TensorDtype::F16 | TensorDtype::Bf16)
                        )
                        || row.estimate.generated_creation_bytes != 0
                        || !matches!(
                            selection.transform,
                            CaptureTransform::FullTensor
                                | CaptureTransform::Slice
                                | CaptureTransform::Preview { .. }
                                | CaptureTransform::Summary
                                | CaptureTransform::Histogram { .. }
                                | CaptureTransform::TopCandidates { .. }
                                | CaptureTransform::TokenScores { .. }
                        )
                    {
                        return Err(error(Cause::Source(
                            "selected row lacks a complete tensor source",
                        )));
                    }
                    Some(PartitionCaptureProducerSource {
                        producer: row.producer,
                        dtype: row.dtype.clone(),
                        estimate: row.estimate,
                    })
                }
                PreparedPartitionCaptureRow::Routed => {
                    if context.invocation.is_none()
                        && !matches!(
                            (context.phase, context.prediction),
                            (CapturePhase::Prefill, 0) | (CapturePhase::Decode, 1..)
                        )
                        || !selection
                            .schedule
                            .includes(context.phase, context.prediction)
                        || !matches!(selection.transform, CaptureTransform::RoutedUnits)
                    {
                        return Err(error(Cause::Source(
                            "selected row lacks a sparse invocation declaration",
                        )));
                    }
                    None
                }
                PreparedPartitionCaptureRow::Contiguous => {
                    if context.invocation.is_none()
                        && !matches!(
                            (context.phase, context.prediction),
                            (CapturePhase::Prefill, 0) | (CapturePhase::Decode, 1..)
                        )
                        || !selection
                            .schedule
                            .includes(context.phase, context.prediction)
                        || !matches!(
                            selection.transform,
                            CaptureTransform::FullTensor
                                | CaptureTransform::Slice
                                | CaptureTransform::Preview { .. }
                                | CaptureTransform::Summary
                                | CaptureTransform::Histogram { .. }
                        )
                    {
                        return Err(error(Cause::Source(
                            "selected row lacks a contiguous invocation declaration",
                        )));
                    }
                    None
                }
            };
            skipped.push(None);
            receiver_epochs.push(None);
            owned_rows.push(row);
            entries.push(None);
        }
        let context =
            receipt::copy_context(context, metadata).map_err(|cause| error(cause.into()))?;
        Ok(Self {
            transport,
            run_identity: None,
            context,
            limits,
            limit_policy: source.admission().plan().limits.on_limit,
            rows: owned_rows,
            entries,
            projected: Vec::new(),
            routed_hooks: None,
            interventions: None,
            skipped,
            receiver_epochs,
            coordination: None,
            first_epoch: None,
            last_epoch: None,
            coordination_complete: false,
            delivered: false,
            source: source.clone(),
            metadata: metadata.clone(),
        })
    }
    fn error(&self, cause: Cause) -> PartitionCaptureProgramError {
        PartitionCaptureProgramError {
            cause,
            _source: self.source.clone(),
            _metadata: self.metadata.clone(),
        }
    }
    /// One fixed debit of the program's shared entry worker. Nested method
    /// calls retain their own debit; receiver lookup also calls `produces`.
    pub fn entry_control_bytes() -> Option<usize> {
        Some(
            size_of::<(&mut Self, usize, &TensorDtype, CaptureUsage)>()
                + size_of::<PartitionCaptureProgramError>()
                + size_of::<Cause>()
                + size_of::<Result<(), PartitionCaptureProgramError>>()
                + size_of::<Option<Entry<'t, T>>>(),
        )
    }

    /// Extra fixed debit for one scheduled complete producer terminal-row lookup.
    pub fn terminal_row_control_bytes() -> Option<usize> {
        Some(size_of::<(
            &ScheduledCaptureStep<'_>,
            &CaptureSelection,
            usize,
            eredu_core::InferenceGeometry,
            u64,
            Option<usize>,
            Result<Option<usize>, crate::working_memory::CaptureRunHostError>,
        )>())
    }

    /// Extra receiver lookup debit, separate from both entry and produces debits.
    pub fn receiver_control_bytes() -> Option<usize> {
        Some(size_of::<(
            &mut Self,
            usize,
            DistributedCommitEpoch,
            Option<TensorDtype>,
            Result<Option<TensorDtype>, PartitionCaptureProgramError>,
        )>())
    }

    /// Extra remote prefill policy debit, separate from entry and projected workers.
    pub fn remote_prefill_control_bytes() -> Option<usize> {
        Some(size_of::<(
            crate::capture::CapturePrefillObservationPolicy<'_>,
            crate::capture::CapturePrefillObservationRow<'_>,
            eredu_core::capture::CapturePrefillTransformFragment<'_, '_>,
            Option<&eredu_core::capture::CapturePrefillTransformPlan<'_>>,
            Option<&TensorDtype>,
            CaptureUsage,
        )>())
    }
    /// Maximum controls for one selected invocation hook, including receiver
    /// fallback and both projected epoch lookups. The caller retains the actual
    /// source and row census; frame preparation and delivery are separate.
    pub fn invocation_hook_control_bytes() -> Option<usize> {
        Self::entry_control_bytes()?
            .checked_mul(4)?
            .checked_add(Self::projected_entry_control_bytes()?.checked_mul(2)?)?
            .checked_add(Self::receiver_control_bytes()?)
    }
    fn charge(&self) -> Result<(), PartitionCaptureProgramError> {
        let bytes = Self::entry_control_bytes()
            .ok_or_else(|| self.error(Cause::Source("entry controls overflow")))?;
        self.metadata
            .reserve_metadata(bytes)
            .map_err(|cause| self.error(cause.into()))
    }
    fn allowed_skip(&self, reason: Option<CaptureSkipReason>) -> Option<CaptureSkipReason> {
        (self.limit_policy == CaptureLimitPolicy::Skip)
            .then_some(reason)
            .flatten()
    }
}
impl<'t, T: PartitionCaptureTransport> ScheduledPartitionCapture
    for PreparedPartitionCaptureProgram<'t, T>
where
    T::Error: Send + Sync + 'static,
    <T::Completion as Completion>::Error: Send + Sync + 'static,
{
    fn prepare(
        &mut self,
        source: &SharedCapturePlan,
        phase: CapturePhase,
        prediction: u64,
        epoch: DistributedCommitEpoch,
        frame: &mut ScheduledCaptureStep<'_>,
        ledger: &mut CaptureLedger,
    ) -> Result<(), PartitionCaptureProgramError> {
        self.charge()?;
        if self.first_epoch.is_some()
            || !source.same_storage(&self.source)
            || phase != self.context.phase
            || prediction != self.context.prediction
            || !frame.matches_partition_context(source, &self.context)
            || frame.records().len() != self.rows.len()
        {
            return Err(self.error(Cause::Source(
                "frame source, coordinate or preparation attempt differs",
            )));
        }
        self.first_epoch = Some(epoch);
        self.context.forward_epoch = epoch.value();
        if let Some(run) = &self.run_identity {
            self.context.run_identity = run.bind_epoch(epoch).map_err(|cause| self.error(cause))?;
        }
        frame
            .prepare_partition_evidence(&self.metadata)
            .map_err(|cause| self.error(cause.into()))?;
        let coordination = PreparedPartitionCaptureCoordination::prepare(
            self.transport,
            &self.source,
            &self.context,
            &self.metadata,
            ledger,
        )
        .map_err(|cause| self.error(cause.into()))?;
        for index in 0..self.rows.len() {
            if !self.source.admission().plan().selections[index]
                .schedule
                .includes(phase, prediction)
            {
                continue;
            }
            self.context.selection_index = index;
            if self.prepare_projected(index, epoch, frame, ledger)? {
                continue;
            }
            let row = self.rows[index].as_ref().ok_or_else(|| {
                self.error(Cause::Source("projected row has no original source owner"))
            })?;
            self.metadata
                .reserve_metadata(
                    Self::terminal_row_control_bytes()
                        .ok_or_else(|| self.error(Cause::Source("entry controls overflow")))?,
                )
                .map_err(|cause| self.error(cause.into()))?;
            let terminal_rows = frame
                .partition_terminal_rows(index)
                .map_err(|cause| self.error(cause.into()))?;
            let receipt = PartitionCaptureReceiptPlan::new_complete_shared_funded_terminal_global(
                &self.source,
                &self.context,
                row.producer,
                self.transport.participant_count(),
                self.limits,
                &self.metadata,
                ledger,
                terminal_rows,
            );
            let (mut receipt, _preparation) = match receipt {
                Ok(receipt) => receipt,
                Err(cause) => {
                    if let Some(reason) = self.allowed_skip(cause.limit_skip()) {
                        self.skipped[index] = Some(reason.clone());
                        frame
                            .record_partition_skip(index, reason)
                            .map_err(|cause| self.error(cause.into()))?;
                        continue;
                    }
                    return Err(self.error(cause.into()));
                }
            };
            let allowance = PreparedPartitionCaptureAllowance::prepare(
                self.transport,
                &mut receipt,
                row.estimate,
                &self.metadata,
                ledger,
            );
            let mut allowance = match allowance {
                Ok(allowance) => allowance,
                Err(cause) => {
                    if let Some(reason) = self.allowed_skip(cause.limit_skip()) {
                        self.skipped[index] = Some(reason.clone());
                        frame
                            .record_partition_skip(index, reason)
                            .map_err(|cause| self.error(cause.into()))?;
                        continue;
                    }
                    return Err(self.error(cause.into()));
                }
            };

            let charged = frame.records()[index]
                .charged
                .checked_add(row.estimate.capture)
                .map_err(|cause| self.error(cause.into()))?;
            let evidence = PreparedPartitionCaptureEvidence::prepare(
                &receipt,
                charged,
                &self.metadata,
                allowance.quota_mut(),
            )
            .map_err(|cause| self.error(cause.into()))?;
            let exchange =
                PartitionCaptureExchange::admit(self.transport, receipt, allowance.quota_mut())
                    .map_err(|cause| self.error(cause.into()))?;
            let delivery = PreparedPartitionTensorDelivery::prepare_with_source(
                exchange,
                row.dtype.clone(),
                charged,
                &self.metadata,
            )
            .map_err(|cause| self.error(cause.into()))?
            .with_evidence(evidence);
            self.entries[index] = Some(Entry {
                delivery: Some(delivery),
                allowance,
            });
        }
        self.prepare_interventions(epoch, frame, ledger)?;
        self.coordination = Some(coordination);
        Ok(())
    }
    fn coordinate(
        &mut self,
        epoch: DistributedCommitEpoch,
        ledger: &CaptureLedger,
    ) -> Result<(), PartitionCaptureProgramError> {
        // An attempted epoch is not evidence that either the source frames or
        // final receipt/ledger vote succeeded. Consume readiness before any
        // fallible work and never restore it on a refusal or partial resolution.
        let previously_complete = std::mem::replace(&mut self.coordination_complete, false);
        self.charge()?;
        let first = self
            .first_epoch
            .ok_or_else(|| self.error(Cause::Source("observer was not prepared")))?;
        if epoch < first
            || self.last_epoch.is_some_and(|last| last >= epoch)
            || self.delivered
            || (self.last_epoch.is_some() && !previously_complete)
        {
            return Err(self.error(Cause::Source("coordination epoch is unavailable or spent")));
        }
        self.last_epoch = Some(epoch);
        if let Some(mut coordination) = self.coordination.take() {
            if epoch != first {
                return Err(self.error(Cause::Source("first preparation vote was omitted")));
            }
            for index in 0..self.rows.len() {
                if !self.source.admission().plan().selections[index]
                    .schedule
                    .includes(self.context.phase, self.context.prediction)
                {
                    continue;
                }
                if self.coordinate_projected(index, &mut coordination, ledger)? {
                    continue;
                }
                let row = self.rows[index].as_ref().ok_or_else(|| {
                    self.error(Cause::Source("projected row has no original source owner"))
                })?;
                if let Some(entry) = &self.entries[index] {
                    let delivery = entry.delivery.as_ref().ok_or_else(|| {
                        self.error(Cause::Source(
                            "capture delivery was consumed before source vote",
                        ))
                    })?;
                    let dtype = coordination
                        .coordinate_source(
                            index,
                            Some(delivery.receipt_plan()),
                            row.producer,
                            row.dtype.as_ref(),
                            None,
                            row.estimate.capture,
                            ledger,
                        )
                        .map_err(|cause| self.error(cause.into()))?
                        .ok_or_else(|| {
                            self.error(Cause::Source("active producer scalar is absent"))
                        })?;
                    coordination
                        .include(delivery.receipt_plan(), dtype.clone(), row.estimate.capture)
                        .map_err(|cause| self.error(cause.into()))?;
                    self.entries[index]
                        .as_mut()
                        .expect("checked active row")
                        .delivery
                        .as_mut()
                        .expect("checked delivery")
                        .bind_source_dtype(dtype.clone())
                        .map_err(|cause| self.error(cause.into()))?;
                    self.rows[index]
                        .as_mut()
                        .expect("checked complete row")
                        .dtype = Some(dtype);
                } else {
                    let reason = self.skipped[index].as_ref().ok_or_else(|| {
                        self.error(Cause::Source("scheduled source has no receipt or skip"))
                    })?;
                    coordination
                        .coordinate_source(
                            index,
                            None,
                            row.producer,
                            None,
                            Some(reason),
                            row.estimate.capture,
                            ledger,
                        )
                        .map_err(|cause| self.error(cause.into()))?;
                    coordination
                        .include_skip(index, reason)
                        .map_err(|cause| self.error(cause.into()))?;
                }
            }
            self.coordinate_interventions(&mut coordination)?;
            coordination
                .coordinate(ledger)
                .map_err(|cause| self.error(cause.into()))?;
        } else if !previously_complete {
            return Err(self.error(Cause::Source(
                "capture preparation did not establish a coordination vote",
            )));
        }
        self.coordinate_intervention_evidence(epoch, ledger)?;
        self.coordination_complete = true;
        Ok(())
    }
    fn intervention_evidence(
        &mut self,
        index: usize,
    ) -> Result<Option<&mut (dyn ScheduledPartitionCapture + '_)>, PartitionCaptureProgramError>
    {
        PreparedPartitionCaptureProgram::intervention_evidence(self, index)
    }
    fn intervention_member(
        &self,
        index: usize,
        window: Option<crate::intervention::InterventionPrefillWindow>,
    ) -> Result<Option<bool>, PartitionCaptureProgramError> {
        PreparedPartitionCaptureProgram::intervention_member(self, index, window)
    }
    fn begin_intervention(
        &mut self,
        index: usize,
        window: Option<crate::intervention::InterventionPrefillWindow>,
        valid: bool,
    ) -> Result<Option<PartitionInterventionLocalAllowance>, PartitionCaptureProgramError> {
        PreparedPartitionCaptureProgram::begin_intervention(self, index, window, valid)
    }
    fn finish_intervention(
        &mut self,
        index: usize,
        loan: PartitionInterventionLocalAllowance,
        success: bool,
    ) -> Result<(), PartitionCaptureProgramError> {
        PreparedPartitionCaptureProgram::finish_intervention(self, index, loan, success)
    }
    fn validate_intervention_prefill_end(
        &self,
        frame: &ScheduledCaptureStep<'_>,
        window: crate::intervention::InterventionPrefillWindow,
    ) -> Result<bool, PartitionCaptureProgramError> {
        PreparedPartitionCaptureProgram::validate_intervention_prefill_end(self, frame, window)
    }
    fn produces(&self, index: usize) -> Result<bool, PartitionCaptureProgramError> {
        self.charge()?;
        let row = self
            .rows
            .get(index)
            .ok_or_else(|| self.error(Cause::Source("selection index differs")))?;
        if !self.coordination_complete || self.delivered {
            return Err(self.error(Cause::Source(
                "hook preceded coordination or followed delivery",
            )));
        }
        if let Some(projected) = self.projected.get(index).and_then(Option::as_ref) {
            return match projected {
                projected::Row::Ready(value) => Ok(value.produces()),
                projected::Row::Skipped { .. } => Ok(false),
                _ => Err(self.error(Cause::Source("projected row has no ready source"))),
            };
        }
        let row = row.as_ref().ok_or_else(|| {
            self.error(Cause::Source("projected row has no original source owner"))
        })?;
        Ok(self.entries[index].is_some() && row.producer == self.transport.capture_rank())
    }
    fn take_receiver_source(
        &mut self,
        index: usize,
    ) -> Result<Option<TensorDtype>, PartitionCaptureProgramError> {
        self.charge()?;
        self.metadata
            .reserve_metadata(
                Self::receiver_control_bytes()
                    .ok_or_else(|| self.error(Cause::Source("entry controls overflow")))?,
            )
            .map_err(|cause| self.error(cause.into()))?;
        let producer = self.produces(index)?;
        if producer
            || self.projected.get(index).is_some_and(Option::is_some)
            || self.entries[index].is_none()
        {
            return Ok(None);
        }
        let epoch = self
            .last_epoch
            .ok_or_else(|| self.error(Cause::Source("receiver hook has no coordinated epoch")))?;
        if self.receiver_epochs[index] == Some(epoch) {
            return Err(self.error(Cause::Source("receiver source attempt is already spent")));
        }
        self.receiver_epochs[index] = Some(epoch);
        self.rows[index]
            .as_ref()
            .and_then(|row| row.dtype.clone())
            .map(Some)
            .ok_or_else(|| {
                self.error(Cause::Source(
                    "receiver source has no agreed scalar witness",
                ))
            })
    }
    fn routed_source(&self, index: usize) -> Result<bool, PartitionCaptureProgramError> {
        self.has_routed_source(index)
    }
    fn take_routed_hooks(
        &mut self,
        index: usize,
        frame: &mut ScheduledCaptureStep<'_>,
        prefill: Option<(&crate::prefill::PrefillChunk, eredu_core::InferenceGeometry)>,
    ) -> Result<Option<PartitionCaptureRoutedHooks>, PartitionCaptureProgramError> {
        self.lend_routed_hooks(index, frame, prefill).map(Some)
    }
    fn return_routed_hooks(
        &mut self,
        hooks: PartitionCaptureRoutedHooks,
        frame: &mut ScheduledCaptureStep<'_>,
        prefill: Option<(&crate::prefill::PrefillChunk, eredu_core::InferenceGeometry)>,
    ) -> Result<(), PartitionCaptureProgramError> {
        self.restore_routed_hooks(hooks, frame, prefill)
    }
    fn reservation(
        &mut self,
        index: usize,
        dtype: &TensorDtype,
        usage: CaptureUsage,
    ) -> Result<&mut CaptureQuota, PartitionCaptureProgramError> {
        if !self.produces(index)? {
            return Err(self.error(Cause::Source("nonproducer attempted a native transform")));
        }
        if self.projected.get(index).is_some_and(Option::is_some) {
            return Err(self.error(Cause::Source(
                "projected source requires its typed local hook loan",
            )));
        }
        let row = self.rows[index].as_ref().ok_or_else(|| {
            self.error(Cause::Source("projected row has no original source owner"))
        })?;
        if row.dtype.as_ref() != Some(dtype) || usage != row.estimate.capture {
            return Err(self.error(Cause::Source(
                "actual tensor representation or usage differs",
            )));
        }
        Ok(self.entries[index]
            .as_mut()
            .expect("checked prepared row")
            .allowance
            .quota_mut())
    }
    fn take_invocation_projection(
        &mut self,
        index: usize,
    ) -> Result<Option<PartitionCaptureLocalHook>, PartitionCaptureProgramError> {
        self.take_projected_invocation(index)
    }
    fn take_invocation_receiver_source(
        &mut self,
        index: usize,
    ) -> Result<Option<super::PartitionInvocationReceiverSource>, PartitionCaptureProgramError>
    {
        self.take_projected_invocation_receiver(index)
    }
    fn take_prefill_projection(
        &mut self,
        index: usize,
        frame: &mut ScheduledCaptureStep<'_>,
        first: bool,
    ) -> Result<Option<PartitionCaptureLocalHook>, PartitionCaptureProgramError> {
        self.take_projected_hook(index, frame, first)
    }
    fn return_prefill_projection(
        &mut self,
        index: usize,
        hook: PartitionCaptureLocalHook,
    ) -> Result<(), PartitionCaptureProgramError> {
        match self.projected.get_mut(index).and_then(Option::as_mut) {
            Some(projected::Row::Ready(row)) => row.return_local(hook),
            _ => Err(hook.reject()),
        }
    }
    fn take_projected_receiver_source(
        &mut self,
        index: usize,
        inference: eredu_core::InferenceGeometry,
        chunk: u64,
    ) -> Result<Option<PartitionPrefillReceiverSource>, PartitionCaptureProgramError> {
        self.take_projected_receiver(index, inference, chunk)
    }
    fn remote_prefill(
        &mut self,
        frame: &mut ScheduledCaptureStep<'_>,
        bound: crate::layered::BoundCaptureSelection<'_>,
        chunk: &crate::prefill::PrefillChunk,
    ) -> Result<(), PartitionCaptureProgramError> {
        self.charge()?;
        if !bound.selection().source().same_storage(&self.source)
            || !self.coordination_complete
            || self.delivered
        {
            return Err(self.error(Cause::Source(
                "remote progress differs from its bound frame",
            )));
        }
        self.progress_intervention_evidence(frame, bound, chunk)?;
        self.metadata
            .reserve_metadata(
                Self::remote_prefill_control_bytes()
                    .ok_or_else(|| self.error(Cause::Source("entry controls overflow")))?,
            )
            .map_err(|cause| self.error(cause.into()))?;
        let policy = crate::capture::CapturePrefillObservationPolicy::from_bound(bound)
            .map_err(|cause| self.error(cause.into()))?;
        let source = self.source.clone();
        let metadata = self.metadata.clone();
        let error = |cause| PartitionCaptureProgramError {
            cause,
            _source: source.clone(),
            _metadata: metadata.clone(),
        };
        for index in 0..self.rows.len() {
            if self.progress_projected_receiver(index, frame, bound, chunk)? {
                continue;
            }
            let complete = self.rows[index].as_ref().ok_or_else(|| {
                error(Cause::Source("projected row has no original source owner"))
            })?;
            if complete.producer == self.transport.capture_rank() {
                continue;
            }
            let Some(entry) = self.entries[index].as_mut() else {
                continue;
            };
            let path = &source.admission().plan().selections[index].path;
            let decision = frame
                .begin_prefill_hook(index, chunk, path)
                .map_err(|cause| error(cause.into()))?;
            if decision == crate::capture::CapturePrefillHookDecision::Ignore {
                continue;
            }
            let row = policy.row(index).map_err(|cause| error(cause.into()))?;
            if row.terminal() {
                let dtype = complete
                    .dtype
                    .as_ref()
                    .ok_or_else(|| error(Cause::Source("terminal row has no scalar source")))?;
                let charge = entry
                    .allowance
                    .take_remote_charge()
                    .map_err(|cause| error(cause.into()))?;
                frame
                    .reserve_remote_prefill_hook(index, dtype.clone(), charge)
                    .map_err(|cause| error(cause.into()))?;
                if row.candidate().is_some() {
                    frame.finish_candidate_prefill_hook(index, chunk)
                } else {
                    frame.finish_token_scores_prefill_hook(index, chunk)
                }
                .map_err(|cause| error(cause.into()))?;
                continue;
            }
            if let Some(plan) = row.transform_plan().filter(|plan| {
                matches!(
                    plan.selection().transform,
                    CaptureTransform::Summary | CaptureTransform::Histogram { .. }
                )
            }) {
                let fragment = plan
                    .fragment(chunk.input.start / bound.geometry().prefill_chunk_positions)
                    .map_err(|cause| error(Cause::Progress(cause.into())))?;
                if decision == crate::capture::CapturePrefillHookDecision::First {
                    let dtype = complete.dtype.as_ref().ok_or_else(|| {
                        error(Cause::Source(
                            "reached remote reduced row has no actual scalar source",
                        ))
                    })?;
                    let charge = entry
                        .allowance
                        .take_remote_charge()
                        .map_err(|cause| error(cause.into()))?;
                    frame
                        .reserve_remote_prefill_hook(index, dtype.clone(), charge)
                        .map_err(|cause| error(cause.into()))?;
                }
                match plan.selection().transform {
                    CaptureTransform::Histogram { .. } => {
                        frame.finish_histogram_prefill_hook(index, &fragment)
                    }
                    _ => frame.finish_summary_prefill_hook(index, &fragment),
                }
                .map_err(|cause| error(cause.into()))?;
                continue;
            }
            let assembly = row
                .assembly()
                .ok_or_else(|| error(Cause::Source("remote row has no tensor assembly")))?;
            let fragment = assembly
                .fragment(chunk.input.start / bound.geometry().prefill_chunk_positions)
                .map_err(|cause| error(Cause::Progress(cause.into())))?;
            if decision == crate::capture::CapturePrefillHookDecision::First {
                let dtype = complete.dtype.as_ref().ok_or_else(|| {
                    error(Cause::Source(
                        "reached remote capture row has no actual scalar source",
                    ))
                })?;
                let charge = entry
                    .allowance
                    .take_remote_charge()
                    .map_err(|cause| error(cause.into()))?;
                frame
                    .reserve_remote_prefill_hook(index, dtype.clone(), charge)
                    .map_err(|cause| error(cause.into()))?;
            }
            frame
                .finish_prefill_hook(index, &fragment)
                .map_err(|cause| error(cause.into()))?;
        }
        Ok(())
    }
    fn complete_prefill_evidence(
        &mut self,
        frame: &mut ScheduledCaptureStep<'_>,
        bound: crate::layered::BoundCaptureSelection<'_>,
        chunk: &crate::prefill::PrefillChunk,
    ) -> Result<(), PartitionCaptureProgramError> {
        if self
            .interventions
            .as_ref()
            .is_none_or(|rows| !rows.has_evidence())
        {
            return Ok(());
        }
        self.charge()?;
        if !bound.selection().source().same_storage(&self.source)
            || !self.coordination_complete
            || self.delivered
        {
            return Err(self.error(Cause::Source(
                "evidence completion differs from its bound frame",
            )));
        }
        self.complete_intervention_evidence(frame, bound, chunk)
    }
    fn deliver(
        &mut self,
        frame: &mut ScheduledCaptureStep<'_>,
    ) -> Result<(), PartitionCaptureProgramError> {
        self.charge()?;
        if !self.coordination_complete || self.coordination.is_some() || self.delivered {
            return Err(self.error(Cause::Source("receipt delivery is unavailable or spent")));
        }
        self.delivered = true;
        self.deliver_interventions(frame)?;
        self.deliver_intervention_evidence(frame)?;
        for index in 0..self.entries.len() {
            if self.deliver_projected(index, frame)? {
                continue;
            }
            if let Some(entry) = &mut self.entries[index] {
                let delivery = entry.delivery.take().expect("one prepared delivery");
                delivery
                    .deliver(frame)
                    .map_err(|cause| self.error(cause.into()))?;
            }
        }
        Ok(())
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
