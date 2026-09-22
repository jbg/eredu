//! Physical source windows retain ordinary claim identity and shared native work.
use super::*;
use eredu_runtime::{
    intervention::InterventionPrefillWindow as Window, working_memory::InterventionPrefillFragment,
};
impl PreparedModelInterventions {
    pub(in super::super) fn prepare_scheduled_fragment(
        source: &OriginalInterventionSource,
        selected: &[bool],
        span: Window,
        skips: &[[Option<CaptureSkipReason>; 2]],
        context: &WorkspaceContext,
    ) -> Result<Self, eredu_nn::Error> {
        context
            .charge_metadata(Window::control_bytes().ok_or(WorkspaceMetadataError::Overflow)?)?;
        span.validate(source.plan().admission())
            .map_err(|cause| context.metadata_source(cause))?;
        for (index, selected) in selected.iter().copied().enumerate() {
            if selected {
                let valid = if source.plan().admission().plan().operations[index].evidence
                    != InterventionEvidence::None
                {
                    Window::validate_evidence_operation(source, index)
                } else {
                    Window::validate_operation(source.plan().admission(), index)
                };
                valid.map_err(|cause| context.metadata_source(cause))?;
            }
        }
        let mut prepared = Self::prepare_scheduled_with_evidence(
            source,
            selected,
            CapturePhase::Prefill,
            0,
            Some(skips),
            context,
        )?;
        prepared.scheduled_span = Some(span);
        Ok(prepared)
    }
    pub(in super::super) fn scheduled_span(&self) -> Option<Window> {
        self.scheduled_span
    }
    pub(crate) fn validate_prefill_span(&self, span: Window) -> Result<(), Failure> {
        if self.scheduled_span != Some(span) {
            return Err(Failure::SourceChanged);
        }
        span.validate(self.source.plan().admission())
            .map_err(|_| Failure::ClaimMismatch)
    }
    pub(crate) fn execute_prefill_scheduled(
        &self,
        value: &Array,
        fragment: InterventionPrefillFragment<'_, '_>,
        charged: CaptureUsage,
        projection: [CaptureUsage; 2],
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
        scope: &WorkingMemoryFundingScope,
        observer: &OriginalScopeObserver,
    ) -> Result<Option<Array>, FundedCaptureError<Failure>> {
        self.validate_prefill_span(fragment.source())
            .map_err(FundedCaptureError::Backend)?;
        let (edit, program) = self
            .checked(value, fragment.claim(), NativeCustody::Scheduled(scope))
            .map_err(FundedCaptureError::Backend)?;
        if edit.usage != charged || edit.projection != projection {
            return Err(FundedCaptureError::Backend(Failure::ClaimMismatch));
        }
        let output = Self::execute_program(program, value, stream, observer, roots)?;
        fragment.finish()?;
        Ok(output)
    }
}

pub(super) fn execution_control_bytes() -> Option<usize> {
    let frames = [
        size_of::<Window>(),
        size_of::<InterventionPrefillFragment<'_, '_>>(),
        size_of::<(
            &PreparedModelInterventions,
            &Array,
            InterventionPrefillFragment<'_, '_>,
            CaptureUsage,
            [CaptureUsage; 2],
            &Stream,
            &RefCell<Vec<Array>>,
            &WorkingMemoryFundingScope,
            &OriginalScopeObserver,
        )>(),
        size_of::<Result<Option<Array>, FundedCaptureError<Failure>>>(),
        Window::control_bytes()?,
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
