//! Dense operation rows in the existing original capture transaction owner.
use super::*;
mod evidence;
use crate::intervention::InterventionPrefillWindow;
use crate::working_memory::OriginalInterventionSource;
pub use evidence::PreparedPartitionInterventionEvidence;

type Prepare<'t, T> =
    fn(
        PreparedPartitionInterventionSource,
        &'t T,
        DistributedCommitEpoch,
        &mut CaptureLedger,
    ) -> Result<PreparedPartitionIntervention<'t, T>, PartitionInterventionSourceError>;
enum Entry<'t, T: PartitionCaptureTransport> {
    Inactive,
    Source(Option<PreparedPartitionInterventionSource>),
    Ready {
        work: PreparedPartitionIntervention<'t, T>,
        baseline: CaptureUsage,
    },
}
pub(super) struct Rows<'t, T: PartitionCaptureTransport> {
    rows: Vec<Entry<'t, T>>,
    source: OriginalInterventionSource,
    prepare: Prepare<'t, T>,
    evidence: Vec<Option<PreparedPartitionInterventionEvidence<'t, T>>>,
}
impl<T: PartitionCaptureTransport> Rows<'_, T> {
    pub(super) fn has_evidence(&self) -> bool {
        self.evidence.iter().any(Option::is_some)
    }
}
impl<'t, T: PartitionCaptureTransport> PreparedPartitionCaptureProgram<'t, T>
where
    T::Error: Send + Sync + 'static,
    <T::Completion as Completion>::Error: Send + Sync + 'static,
{
    /// The same fixed row-constructor transports; table allocations are quoted separately.
    fn intervention_source_control_bytes<I, J>() -> Option<usize> {
        [
            size_of::<Rows<'t, T>>(),
            size_of::<I>(),
            size_of::<J>(),
            size_of::<(&mut Self, &OriginalInterventionSource)>(),
            size_of::<Result<(), PartitionCaptureProgramError>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }
    /// Prospective constructor metadata for the exact consumed source/evidence iterators.
    /// This neither constructs a transcript nor admits an operation.
    pub fn intervention_sources_metadata_bytes<I, J>(rows: usize) -> Option<usize>
    where
        I: ExactSizeIterator<Item = Option<PreparedPartitionInterventionSource>>,
        J: ExactSizeIterator<Item = Option<PreparedPartitionInterventionEvidence<'t, T>>>,
    {
        Self::intervention_source_control_bytes::<I, J>()?
            .checked_add(eredu_nn::workspace::WorkspaceContext::metadata_vec_bytes::<
                Entry<'t, T>,
            >(rows)?)?
            .checked_add(eredu_nn::workspace::WorkspaceContext::metadata_vec_bytes::<
                Option<PreparedPartitionInterventionEvidence<'t, T>>,
            >(rows)?)
    }
    /// Attach the exact original operation table before the first frame. The
    /// native rows remain backend-owned; this retains only their paid transcript.
    pub fn set_intervention_sources<I>(
        &mut self,
        source: &OriginalInterventionSource,
        rows: I,
    ) -> Result<(), PartitionCaptureProgramError>
    where
        T: PartitionCaptureHookTransport,
        I: ExactSizeIterator<Item = Option<PreparedPartitionInterventionSource>>,
    {
        let count = rows.len();
        self.set_intervention_sources_with_evidence(
            source,
            rows,
            std::iter::repeat_with(|| None).take(count),
        )
    }
    /// Attach original operation and companion sources atomically. Active
    /// requested evidence must have its exact retained two-side receipt worker.
    pub fn set_intervention_sources_with_evidence<I, J>(
        &mut self,
        source: &OriginalInterventionSource,
        rows: I,
        evidence: J,
    ) -> Result<(), PartitionCaptureProgramError>
    where
        T: PartitionCaptureHookTransport,
        I: ExactSizeIterator<Item = Option<PreparedPartitionInterventionSource>>,
        J: ExactSizeIterator<Item = Option<PreparedPartitionInterventionEvidence<'t, T>>>,
    {
        let result = (|| -> Result<(), Cause> {
            if self.first_epoch.is_some()
                || self.interventions.is_some()
                || source.plan().admission().request() != self.source.admission().request()
                || source.plan().admission().invocation_bounds()
                    != self.source.admission().invocation_bounds()
                || source.plan().admission().text_origin() != self.source.admission().text_origin()
                || rows.len() != source.plan().admission().plan().operations.len()
                || evidence.len() != rows.len()
            {
                return Err(Cause::Source(
                    "original intervention table is missing, repeated or belongs to another request",
                ));
            }
            self.metadata.reserve_metadata(
                Self::intervention_source_control_bytes::<I, J>().ok_or(CaptureError::Overflow)?,
            )?;
            let mut owned = self.metadata.metadata_vec(rows.len())?;
            let mut companions = self.metadata.metadata_vec(rows.len())?;
            for (index, (row, mut companion)) in rows.zip(evidence).enumerate() {
                let active = source.plan().admission().plan().operations[index]
                    .schedule
                    .includes(self.context.phase, self.context.prediction);
                let needs = active
                    && source.plan().admission().plan().operations[index].evidence
                        != eredu_core::intervention::InterventionEvidence::None;
                let model = self.context.invocation.is_some();
                if (!model && needs != companion.is_some())
                    || (model && companion.is_some() && (!needs || row.is_none()))
                    || companion
                        .as_ref()
                        .is_some_and(|child| !child.matches(source, index, self))
                {
                    return Err(Cause::Source(
                        "partition intervention evidence has no exact original delivery source",
                    ));
                }
                if let Some(child) = &mut companion {
                    // Same original run/epoch and user limit policy, independent
                    // of the companion's descriptive zero-budget admission.
                    child.program.run_identity = self.run_identity.clone();
                    child.program.limit_policy = self.limit_policy;
                    child.bind_budget(&self.source)?;
                }
                companions.push(companion);
                match row {
                    Some(row)
                        if active
                            && row.original().same_source(source)
                            && row.operation() == index
                            && row.coordinate()
                                == (self.context.phase, self.context.prediction) =>
                    {
                        owned.push(Entry::Source(Some(row)))
                    }
                    None if !active || model => owned.push(Entry::Inactive),
                    _ => {
                        return Err(Cause::Source(
                            "original intervention operation or schedule differs",
                        ));
                    }
                }
            }
            self.interventions = Some(Rows {
                rows: owned,
                source: source.clone(),
                prepare: PreparedPartitionInterventionSource::prepare::<T>,
                evidence: companions,
            });
            Ok(())
        })();
        result.map_err(|cause| self.error(cause))
    }
    pub(super) fn prepare_interventions(
        &mut self,
        epoch: DistributedCommitEpoch,
        frame: &mut ScheduledCaptureStep<'_>,
        ledger: &mut CaptureLedger,
    ) -> Result<(), PartitionCaptureProgramError> {
        let result = (|| -> Result<(), Cause> {
            match (&mut self.interventions, frame.intervention_admission()) {
                (None, None) => Ok(()),
                (Some(rows), Some(_)) => {
                    frame.validate_partition_intervention_source(&rows.source)?;
                    for (index, row) in rows.rows.iter_mut().enumerate() {
                        let (active, evidence) = frame.partition_intervention_selection(index)?;
                        if active != matches!(row, Entry::Source(_))
                            || evidence != rows.evidence.get(index).is_some_and(Option::is_some)
                        {
                            return Err(Cause::Source(
                                "partition source selection differs from the admitted frame",
                            ));
                        }
                        if let Entry::Source(source) = row {
                            let source = source
                                .take()
                                .ok_or(Cause::Source("intervention source preparation is spent"))?;
                            if source.model_invocation()
                                != frame.partition_intervention_invocation()
                            {
                                return Err(Cause::Source(
                                    "intervention frame physical axes or window differs from its source",
                                ));
                            }
                            let record = frame.interventions().get(index).ok_or(Cause::Source(
                                "original operation outcome row is missing",
                            ))?;
                            if record.outcome
                                != eredu_core::intervention::InterventionOutcome::Missing
                            {
                                return Err(Cause::Source(
                                    "original operation outcome is already completed",
                                ));
                            }
                            let baseline = record.charged;
                            let work = (rows.prepare)(source, self.transport, epoch, ledger)?;
                            *row = Entry::Ready { work, baseline };
                        }
                    }
                    Ok(())
                }
                _ => Err(Cause::Source(
                    "partition operation has no original source table",
                )),
            }
        })();
        result.map_err(|cause| self.error(cause))?;
        self.prepare_intervention_evidence(epoch, frame, ledger)
    }
    pub(super) fn coordinate_interventions(
        &self,
        coordination: &mut PreparedPartitionCaptureCoordination<'t, T>,
    ) -> Result<(), PartitionCaptureProgramError> {
        let result = (|| -> Result<(), Cause> {
            if let Some(rows) = &self.interventions {
                for row in &rows.rows {
                    match row {
                        Entry::Inactive => (),
                        Entry::Ready { work, .. } => coordination.include_intervention(work)?,
                        _ => {
                            return Err(Cause::Source(
                                "original operation source was not prepared",
                            ));
                        }
                    }
                }
            }
            Ok(())
        })();
        result.map_err(|cause| self.error(cause))
    }
    pub(super) fn intervention_member(
        &self,
        index: usize,
        window: Option<InterventionPrefillWindow>,
    ) -> Result<Option<bool>, PartitionCaptureProgramError> {
        if !self.coordination_complete || self.delivered {
            return Err(self.error(Cause::Source(
                "operation hook is outside its coordinated frame",
            )));
        }
        let Some(rows) = &self.interventions else {
            return Ok(None);
        };
        match rows.rows.get(index) {
            Some(Entry::Ready { work, .. }) => work
                .member(window)
                .map(Some)
                .map_err(|cause| self.error(cause.into())),
            _ => Err(self.error(Cause::Source("operation is inactive or lacks its source"))),
        }
    }
    pub(super) fn begin_intervention(
        &mut self,
        index: usize,
        window: Option<InterventionPrefillWindow>,
        valid: bool,
    ) -> Result<Option<PartitionInterventionLocalAllowance>, PartitionCaptureProgramError> {
        if self.intervention_member(index, window)?.is_none() {
            return Err(self.error(Cause::Source("operation has no partition source")));
        }
        let result = match self
            .interventions
            .as_mut()
            .and_then(|rows| rows.rows.get_mut(index))
        {
            Some(Entry::Ready { work, .. }) => work.begin(window, valid).map_err(Cause::from),
            _ => Err(Cause::Source("operation source is not ready")),
        };
        result.map_err(|cause| self.error(cause))
    }
    pub(super) fn finish_intervention(
        &mut self,
        index: usize,
        loan: PartitionInterventionLocalAllowance,
        success: bool,
    ) -> Result<(), PartitionCaptureProgramError> {
        if !self.coordination_complete || self.delivered {
            return Err(self.error(Cause::Source("operation return is outside its frame")));
        }
        let result = match self
            .interventions
            .as_mut()
            .and_then(|rows| rows.rows.get_mut(index))
        {
            Some(Entry::Ready { work, .. }) => work.finish(loan, success).map_err(Cause::from),
            _ => Err(Cause::Source("operation return has no original owner")),
        };
        result.map_err(|cause| self.error(cause))
    }
    pub(super) fn deliver_interventions(
        &mut self,
        frame: &mut ScheduledCaptureStep<'_>,
    ) -> Result<(), PartitionCaptureProgramError> {
        let result = (|| -> Result<(), Cause> {
            if let Some(rows) = &mut self.interventions {
                frame.validate_partition_intervention_source(&rows.source)?;
                for row in &mut rows.rows {
                    if let Entry::Ready { work, baseline } = row {
                        let receipt = work.deliver(*baseline)?;
                        frame.record_partition_intervention(receipt)?;
                    } else if !matches!(row, Entry::Inactive) {
                        return Err(Cause::Source(
                            "intervention delivery has no original source",
                        ));
                    }
                }
            }
            Ok(())
        })();
        result.map_err(|cause| self.error(cause))
    }
    pub(super) fn validate_intervention_prefill_end(
        &self,
        frame: &ScheduledCaptureStep<'_>,
        window: InterventionPrefillWindow,
    ) -> Result<bool, PartitionCaptureProgramError> {
        let Some(rows) = &self.interventions else {
            return Ok(false);
        };
        let result = (|| -> Result<bool, Cause> {
            if !self.coordination_complete || self.delivered {
                return Err(Cause::Source(
                    "prefill source is outside its coordinated frame",
                ));
            }
            frame.validate_partition_intervention_source(&rows.source)?;
            for (index, row) in rows.rows.iter().enumerate() {
                if !InterventionPrefillWindow::row_axis(
                    &rows.source.plan().admission().points()[index],
                ) {
                    continue;
                }
                match row {
                    Entry::Inactive => (),
                    Entry::Ready { work, .. } => {
                        if !work.local_window_complete(Some(window))? {
                            return Err(Cause::Source(
                                "local partition intervention chunk is incomplete",
                            ));
                        }
                        if work.member(Some(window))? {
                            frame.validate_prefill_intervention_operation_end(index, window)?;
                        } else if frame.interventions()[index].outcome
                            != eredu_core::intervention::InterventionOutcome::Missing
                        {
                            return Err(Cause::Source(
                                "absent partition operation acquired a local outcome",
                            ));
                        }
                    }
                    _ => return Err(Cause::Source("prefill source has not been prepared")),
                }
            }
            Ok(true)
        })();
        result.map_err(|cause| self.error(cause))
    }
}
