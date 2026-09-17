//! Existing fixed record builders used as attributed intervention companions.
use super::*;
use crate::working_memory::{
    CaptureSummaryHostPlan, CaptureTensorHostPlan, OriginalInterventionSource,
};

const MASKS:[[bool;2];4]=[[false,false],[true,false],[false,true],[true,true]];
fn child_plan<'a>(companion:&'a InterventionEvidenceCompanion,phase:CapturePhase,prediction:u64,invocation:Option<CaptureInvocationShape>,window:Option<CaptureInvocationWindow>,selected:bool,skipped:Option<&'a [Option<CaptureSkipReason>;2]>)->Result<CaptureStepHostPlan<'a>,CaptureStepError> {
    let before=selected && skipped.is_none_or(|rows|rows[0].is_none());
    let after=selected && skipped.is_none_or(|rows|rows[1].is_none());
    let mask=&MASKS[usize::from(before)+2*usize::from(after)];
    CaptureStepHostPlan::prepare_selected_window_skips(companion.geometry_source(),phase,prediction,invocation,Some(mask),window,skipped.map(|rows|rows.as_slice()))
}
#[derive(Debug)]
pub(in crate::working_memory) struct PreparedInterventionEvidence<'a> {
    pub(in crate::working_memory) frame: Option<PreparedCaptureStep<'a>>,
    pub(in crate::working_memory) spent: [bool; 2],
    pub(in crate::working_memory) partition: bool,
    pub(in crate::working_memory) partition_claims: Option<[crate::working_memory::capture_run::ClaimState; 3]>,
}
impl<'a> PreparedInterventionEvidence<'a> {
    pub(in crate::working_memory) fn tensor_geometry(&self,index:usize)->Result<Option<CaptureTensorGeometry<'a>>,CaptureStepError> {self.frame()?.plan.geometry(index)}
    pub(in crate::working_memory) fn summary_geometry(&self,index:usize)->Result<Option<CaptureSummaryGeometry<'a>>,CaptureStepError> {self.frame()?.plan.summary_geometry(index)}

    pub(in crate::working_memory) fn frame(&self) -> Result<&PreparedCaptureStep<'a>,CaptureStepError> {
        self.frame.as_ref().ok_or(CaptureStepError::InvalidCompletion)
    }
    pub(in crate::working_memory) fn frame_mut(&mut self) -> Result<&mut PreparedCaptureStep<'a>,CaptureStepError> {
        self.frame.as_mut().ok_or(CaptureStepError::InvalidCompletion)
    }

    pub(super) fn prepare(
        companion: &'a InterventionEvidenceCompanion,
        phase: CapturePhase,
        prediction: u64,
        invocation: Option<CaptureInvocationShape>,
        window: Option<CaptureInvocationWindow>,
        selected: bool,
        skipped: Option<&'a [Option<CaptureSkipReason>;2]>,
        custody: CaptureTensorCustody,
    ) -> Result<Self, CaptureStepError> {
        let plan=child_plan(companion,phase,prediction,invocation,window,selected,skipped)?;
        if plan.len() != 2 {
            return Err(CaptureStepError::InvalidCompletion);
        }
        Ok(Self {
            frame: Some(builder::allocate(plan, custody)?),
            spent: [false; 2],
            partition: false,
            partition_claims: None,
        })
    }
}
pub(in crate::working_memory) fn source_peak(
    source: &OriginalInterventionSource,
    phase: CapturePhase,
    prediction: u64,
    invocation: Option<CaptureInvocationShape>,
    window: Option<CaptureInvocationWindow>,
    selected: Option<&[bool]>,
    skipped: Option<&[[Option<CaptureSkipReason>;2]]>,
) -> Result<u64, CaptureStepError> {
    let count = source.plan().admission().plan().operations.len();
    let mut bytes =
        super::super::plan::extent(count, size_of::<Option<PreparedInterventionEvidence>>())?;
    bytes = bytes
        .checked_add(size_of::<Vec<Option<PreparedInterventionEvidence>>>() as u64)
        .ok_or(WorkingMemoryError::Overflow)?;
    for index in 0..count {
        let Some(companion) = source.plan().evidence(index) else {
            continue;
        };
        let plan=child_plan(companion,phase,prediction,invocation,window,selected.is_none_or(|mask|mask[index]),skipped.map(|rows|&rows[index]))?;
        if plan.len() != 2 {
            return Err(CaptureStepError::InvalidCompletion);
        }
        if (phase,prediction)==(CapturePhase::Prefill,0) && invocation.is_none() && window.is_none()
            && source.plan().admission().points().get(index)
                .is_some_and(crate::intervention::InterventionPrefillWindow::row_axis) {
            bytes=bytes.checked_add(crate::working_memory::capture_run::prefill_target_bytes(plan.len())?)
                .ok_or(WorkingMemoryError::Overflow)?;
        }
        bytes = bytes
            .checked_add(plan.initialization_peak_bytes())
            .ok_or(WorkingMemoryError::Overflow)?;
        for side in 0..2 {
            if let Some(geometry) = plan.geometry(side)? {
                bytes = bytes
                    .checked_add(
                        CaptureTensorHostPlan::prepare(geometry)?.initialization_peak_bytes(),
                    )
                    .ok_or(WorkingMemoryError::Overflow)?;
            }
            if let Some(geometry) = plan.summary_geometry(side)? {
                bytes = bytes
                    .checked_add(
                        CaptureSummaryHostPlan::prepare(geometry)?.initialization_peak_bytes(),
                    )
                    .ok_or(WorkingMemoryError::Overflow)?;
            }
        }
    }
    let controls = [
        size_of::<PreparedInterventionEvidence<'static>>(),
        size_of::<crate::working_memory::capture_run::PartitionInterventionEvidenceFrame<'static>>(),
        size_of::<crate::working_memory::capture_run::CaptureClaimRow<'static>>(),
        size_of::<Option<crate::working_memory::capture_run::PartitionInterventionEvidenceFrame<'static>>>(),
        size_of::<Result<crate::working_memory::capture_run::PartitionInterventionEvidenceFrame<'static>,crate::working_memory::CaptureRunHostError>>(),
        size_of::<Result<(),crate::working_memory::CaptureRunHostError>>(),
        size_of::<(&mut crate::working_memory::ScheduledCaptureStep<'static>,usize)>(),
        size_of::<CaptureStepHostPlan<'static>>(),
        size_of::<crate::working_memory::CaptureInterventionEvidenceClaim<'static, 'static>>(),
        size_of::<crate::working_memory::CaptureInterventionEvidenceKind<'static, 'static>>(),
        size_of::<crate::working_memory::InterventionEvidenceReceipt<'static>>(),
        size_of::<crate::working_memory::ClaimedInterventionEvidence<'static>>(),
        size_of::<Option<crate::working_memory::ClaimedInterventionEvidence<'static>>>(),
        size_of::<
            Result<
                Option<crate::working_memory::ClaimedInterventionEvidence<'static>>,
                crate::capture::FundedCaptureError<WorkingMemoryError>,
            >,
        >(),
        size_of::<
            Result<
                Option<crate::working_memory::CaptureInterventionEvidenceClaim<'static, 'static>>,
                crate::working_memory::CaptureRunHostError,
            >,
        >(),
        size_of::<[bool; 2]>(),
        size_of::<crate::capture::CaptureObservationStep<'static>>(),
        size_of::<Result<Option<CaptureUsage>,CaptureError>>(),
        size_of::<(Option<TensorDtype>,CaptureUsage,Option<CaptureSkipReason>)>(),
        size_of::<Option<CaptureInvocationShape>>(),
        size_of::<Option<CaptureInvocationWindow>>(),
        size_of::<Option<&[Option<CaptureSkipReason>;2]>>(),
        size_of::<(&InterventionEvidenceCompanion,CapturePhase,u64,Option<CaptureInvocationShape>,Option<CaptureInvocationWindow>,bool,Option<&[Option<CaptureSkipReason>;2]>)>(),
        size_of::<Vec<Option<PreparedInterventionEvidence<'static>>>>(),
    ];
    let controls = controls
        .into_iter()
        .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(WorkingMemoryError::Overflow)?;
    bytes
        .checked_add(controls)
        .ok_or(WorkingMemoryError::Overflow.into())
}
impl PreparedCaptureStep<'_> {
    /// Move-only, infallible detachment also used on an aborted transaction.
    /// Every child has its own retained custody; the parent remains alive while
    /// unused buffers retire, then protects all moved evidence records.
    pub(in crate::working_memory) fn flush_intervention_evidence(&mut self) {
        for (record, slot) in self
            .frame
            .interventions
            .iter_mut()
            .zip(&mut self.intervention_evidence)
        {
            // A lent or abandoned child remains an explicit incomplete slot.
            // In particular it cannot become an empty successful evidence list.
            if slot.as_ref().is_some_and(|child| child.frame.is_none()) { continue; }
            if let Some(mut child) = slot.take() {
                debug_assert!(record.evidence.is_empty());
                let frame = child.frame.as_mut().expect("checked evidence frame");
                record.evidence = std::mem::take(&mut frame.frame.records);
                drop(child);
            }
        }
    }
}
