//! Each physical source is retained beneath one logical prediction owner.
use super::*;
use eredu_core::InferenceGeometry;
use eredu_runtime::{intervention::InterventionPrefillWindow as Window, prefill::PrefillChunk};
impl PreparedTextInterventions {
    pub(crate) fn begin_prefill(
        &mut self,
        inference: InferenceGeometry,
        chunk: &PrefillChunk,
        context: &WorkspaceContext,
    ) -> Result<()> {
        if self.partition.is_some() {
            return Err(context.metadata_source(CaptureProtocolError::Invocation));
        }
        self.begin_prefill_with_partition(inference, chunk, None, context)
    }
    pub(super) fn begin_prefill_with_partition(
        &mut self,
        inference: InferenceGeometry,
        chunk: &PrefillChunk,
        placement: Option<(
            &eredu_architectures::component_partition::ComponentPartitionLayouts,
            usize,
            &eredu_runtime::RetainedCommunicationSource,
        )>,
        context: &WorkspaceContext,
    ) -> Result<()> {
        self.check_context(context)?;
        context.charge_metadata(control_bytes().ok_or(WorkspaceMetadataError::Overflow)?)?;
        if !self.prefill_selected.iter().any(|v| *v) {
            return Ok(());
        }
        if self.first_prediction != 0
            || self.active != Some(0)
            || self.prefill_active.is_some()
            || self.prefill_next != chunk.input.start
        {
            return Err(context.metadata_source(CaptureProtocolError::Transaction));
        }
        let span = Window::new(self.source.plan().admission(), inference, chunk)
            .map_err(|cause| context.metadata_source(cause))?;
        if self.prefill_evidence.is_empty() && placement.is_none() {
            context
                .reserve_metadata_vec(&mut self.prefill_evidence, self.prefill_selected.len())?;
            for (index, selected) in self.prefill_selected.iter().copied().enumerate() {
                let progress = if selected && self.source.plan().evidence(index).is_some() {
                    let policy = span
                        .evidence_policy(&self.source, index)
                        .map_err(|cause| context.metadata_source(cause))?;
                    let first = policy
                        .row(0)
                        .map_err(|cause| context.metadata_source(cause))?;
                    let mut values = std::array::from_fn(|_| first.initial_progress());
                    let count = self
                        .source
                        .plan()
                        .evidence(index)
                        .expect("selected companion")
                        .geometry_source()
                        .points()
                        .len();
                    for (index, value) in values.iter_mut().take(count).enumerate() {
                        *value = policy
                            .row(index)
                            .map_err(|cause| context.metadata_source(cause))?
                            .initial_progress();
                    }
                    Some(values)
                } else {
                    None
                };
                self.prefill_evidence.push(progress);
            }
        }
        let mut skips = context.metadata_vec(self.prefill_selected.len())?;
        skips.resize(self.prefill_selected.len(), [None, None]);
        let mut row = PreparedModelInterventions::prepare_scheduled_fragment(
            &self.source,
            &self.prefill_selected,
            span,
            &skips,
            context,
        )?;
        if let Some((layouts, rank, communication)) = placement {
            row.bind_partition(layouts, rank, communication, context)?;
        }
        context.reserve_metadata_vec(&mut self.prefill_rows, 1)?;
        let index = self.prefill_rows.len();
        self.prefill_rows.push(row);
        self.prefill_active = Some(index);
        self.prefill_rows[index].begin_prepaid(CapturePhase::Prefill, 0, context)
    }
    pub(crate) fn end_prefill(
        &mut self,
        inference: InferenceGeometry,
        chunk: &PrefillChunk,
        context: &WorkspaceContext,
    ) -> Result<()> {
        self.check_context(context)?;
        context.charge_metadata(control_bytes().ok_or(WorkspaceMetadataError::Overflow)?)?;
        if !self.prefill_selected.iter().any(|v| *v) {
            return Ok(());
        }
        let span = Window::new(self.source.plan().admission(), inference, chunk)
            .map_err(|cause| context.metadata_source(cause))?;
        let index = self
            .prefill_active
            .ok_or_else(|| context.metadata_source(CaptureProtocolError::Transaction))?;
        if self.prefill_rows[index].scheduled_span() != Some(span) {
            return Err(context.metadata_source(CaptureProtocolError::Transaction));
        }
        self.prefill_rows[index].finish(context)?;
        if self.partition.is_none() {
            let ordinal = span.range()[0] / inference.prefill_chunk_positions;
            for (operation, progress) in self.prefill_evidence.iter().enumerate() {
                if let Some(progress) = progress {
                    let policy = span
                        .evidence_policy(&self.source, operation)
                        .map_err(|cause| context.metadata_source(cause))?;
                    for (side, progress) in progress
                        .iter()
                        .take(
                            self.source
                                .plan()
                                .evidence(operation)
                                .expect("progress source")
                                .geometry_source()
                                .points()
                                .len(),
                        )
                        .enumerate()
                    {
                        policy
                            .row(side)
                            .map_err(|cause| context.metadata_source(cause))?
                            .validate_chunk_end(progress, ordinal)
                            .map_err(|cause| context.metadata_source(cause))?;
                    }
                }
            }
            for (operation, progress) in self.prefill_evidence.iter_mut().enumerate() {
                if let Some(progress) = progress {
                    let policy = span
                        .evidence_policy(&self.source, operation)
                        .map_err(|cause| context.metadata_source(cause))?;
                    for (side, progress) in progress
                        .iter_mut()
                        .take(
                            self.source
                                .plan()
                                .evidence(operation)
                                .expect("progress source")
                                .geometry_source()
                                .points()
                                .len(),
                        )
                        .enumerate()
                    {
                        policy
                            .row(side)
                            .map_err(|cause| context.metadata_source(cause))?
                            .advance_chunk(progress, ordinal)
                            .map_err(|cause| context.metadata_source(cause))?;
                    }
                }
            }
        }

        self.prefill_next = span.range()[1];
        self.prefill_active = None;
        Ok(())
    }
    pub(crate) fn row_for_prefill(
        &self,
        phase: CapturePhase,
        prediction: u64,
        span: Window,
    ) -> std::result::Result<&PreparedModelInterventions, Failure> {
        if self.active.is_some()
            || self.prefill_active.is_some()
            || self.next_prediction != self.end_prediction
            || phase != CapturePhase::Prefill
            || prediction != 0
            || self.first_prediction != 0
        {
            return Err(Failure::ClaimMismatch);
        }
        span.validate(self.source.plan().admission())
            .map_err(|_| Failure::ClaimMismatch)?;
        self.prefill_rows
            .iter()
            .find(|row| row.scheduled_span() == Some(span))
            .ok_or(Failure::SourceChanged)
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    let frames = [
        size_of::<Vec<PreparedModelInterventions>>(),
        size_of::<[Vec<bool>; 2]>(),
        size_of::<Option<usize>>(),
        size_of::<Option<[eredu_runtime::capture::CapturePrefillRowProgress; 4]>>(),
        size_of::<eredu_runtime::capture::CapturePrefillObservationPolicy<'_>>(),
        size_of::<eredu_runtime::capture::CapturePrefillObservationRow<'_>>(),
        size_of::<Vec<[Option<CaptureSkipReason>; 2]>>(),
        size_of::<[Window; 2]>(),
        Window::control_bytes()?,
        size_of::<(
            &mut PreparedTextInterventions,
            InferenceGeometry,
            &PrefillChunk,
            &WorkspaceContext,
        )>(),
        size_of::<Result<()>>(),
        size_of::<Result<PreparedModelInterventions>>(),
        size_of::<std::result::Result<&PreparedModelInterventions, Failure>>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
