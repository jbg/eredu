use super::*;
use crate::api::{PreparedChatResumeSettings, PreparedChatSnapshot};
use eredu_runtime::{
    execution_control::TextSnapshotBackend,
    working_memory::{OriginalChatBackend, WorkspaceCopyLimits},
};
pub(super) mod journal;
pub(super) use journal::{CaptureJournal, CopiedJournal, RestoreJournal, RestoredJournal};

/// Descriptive output boundary. Deserialization conveys no source authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationOutputCheckpointData {
    pub run_id: String,
    pub epoch: u64,
    pub next_sequence: u64,
    pub next_prediction: u64,
}
/// Independently escaping output checkpoint and its original construction account.
/// The event sequence remains monotone across epochs. Fields are read-only.
#[derive(Debug)]
pub struct GenerationOutputCheckpoint {
    data: GenerationOutputCheckpointData,
    _funding: HostMetadataFunding,
}
impl std::ops::Deref for GenerationOutputCheckpoint {
    type Target = GenerationOutputCheckpointData;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}
impl Serialize for GenerationOutputCheckpoint {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.data.serialize(serializer)
    }
}
impl PartialEq for GenerationOutputCheckpoint {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}
impl Eq for GenerationOutputCheckpoint {}
impl GenerationOutputCheckpoint {
    pub(super) fn prepare(
        run_id: &str,
        epoch: u64,
        next_sequence: u64,
        next_prediction: u64,
        funding: &HostMetadataFunding,
    ) -> Result<Self, RecordConstructionError> {
        (|| {
            records::construction::controls(
                funding,
                &[
                    size_of::<Self>(),
                    size_of::<Result<Self, RecordConstructionError>>(),
                ],
            )?;
            let data = output_data(run_id, epoch, next_sequence, next_prediction, funding)?;
            Ok(Self {
                data,
                _funding: funding.clone(),
            })
        })()
        .map_err(|cause| RecordConstructionError::retain(cause, funding))
    }
}
fn output_data(
    run_id: &str,
    epoch: u64,
    next_sequence: u64,
    next_prediction: u64,
    funding: &HostMetadataFunding,
) -> records::construction::Result<GenerationOutputCheckpointData> {
    records::construction::controls(funding, &[size_of::<GenerationOutputCheckpointData>()])?;
    Ok(GenerationOutputCheckpointData {
        run_id: records::construction::string(run_id, funding)?,
        epoch,
        next_sequence,
        next_prediction,
    })
}
/// One canonical native/cursor/parser snapshot with its original record journal.
pub struct ControlledGenerationSnapshot<B: TextSnapshotBackend> {
    pub(super) snapshot: PreparedChatSnapshot<B>,
    pub(super) journal: CopiedJournal,
}
impl<B: TextSnapshotBackend> ControlledGenerationSnapshot<B> {
    pub fn metadata(&self) -> &GenerationSnapshotMetadata {
        &self.journal.metadata
    }
    pub fn token_ids(&self) -> &[u32] {
        self.snapshot.token_ids()
    }
    pub fn prompt_attribution(&self) -> &PreparedPromptAttribution {
        self.journal.prompt.prepared().attribution()
    }
    pub fn complete_token_ids(&self) -> Option<&[u32]> {
        self.prompt_attribution().complete_token_ids()
    }
}

pub(super) fn copy_output(
    source: &GenerationOutputCheckpointData,
    funding: &HostMetadataFunding,
) -> Result<GenerationOutputCheckpointData, RecordConstructionError> {
    output_data(
        &source.run_id,
        source.epoch,
        source.next_sequence,
        source.next_prediction,
        funding,
    )
    .map_err(|cause| RecordConstructionError::retain(cause, funding))
}

pub(super) fn copy_info(
    source: &SnapshotInfo,
    funding: &HostMetadataFunding,
) -> Result<SnapshotInfo, RecordConstructionError> {
    use records::construction::{controls, optional, string};
    (|| {
        controls(
            funding,
            &[
                size_of::<SnapshotInfo>(),
                size_of::<GenerationOutputCheckpointData>(),
            ],
        )?;
        Ok(SnapshotInfo {
            snapshot_id: string(&source.snapshot_id, funding)?,
            session_id: string(&source.session_id, funding)?,
            artifact_identity: optional(&source.artifact_identity, funding)?,
            capture_plan_id: optional(&source.capture_plan_id, funding)?,
            intervention_plan_id: optional(&source.intervention_plan_id, funding)?,
            tokenizer_identity: source.tokenizer_identity,
            configuration_identity: source.configuration_identity,
            pending_forced_token: source.pending_forced_token,
            status: source.status,
            output: GenerationOutputCheckpointData {
                run_id: string(&source.output.run_id, funding)?,
                epoch: source.output.epoch,
                next_sequence: source.output.next_sequence,
                next_prediction: source.output.next_prediction,
            },
            retained_bytes: source.retained_bytes,
        })
    })()
    .map_err(|cause| RecordConstructionError::retain(cause, funding))
}
impl<B: TextGenerationBackend> ControlledGenerationSession<'_, B> {
    /// Copies a consumer checkpoint prospectively under the record producer's account.
    pub fn output_checkpoint(&self) -> Result<GenerationOutputCheckpoint, RecordConstructionError> {
        GenerationOutputCheckpoint::prepare(
            &self.delivery.template.run_id,
            self.delivery.epoch,
            self.delivery.sequence,
            self.next_prediction(),
            &self.delivery.funding,
        )
    }
    pub fn snapshot_usage(&self) -> Option<SnapshotUsage> {
        self.snapshot_budget.as_ref().map(SnapshotBudget::usage)
    }
    pub(super) fn require_snapshot_budget(
        &self,
    ) -> Result<SnapshotBudget, ControlledGenerationError> {
        self.snapshot_budget
            .clone()
            .ok_or(ControlledGenerationError::Rejected(
                "snapshot limits have not been admitted",
            ))
    }
}
impl<B: TextSnapshotBackend> ControlledGenerationSession<'_, B> {
    /// Admits the nonrefundable snapshot budget once, with explicit original host
    /// and native destination capacities. Copy providers recheck exact geometry.
    pub fn enable_snapshots(
        &mut self,
        limits: SnapshotLimits,
        host_capacity: eredu_core::MemoryLimitDeclarations,
        native: WorkspaceCopyLimits,
    ) -> Result<(), ControlledGenerationError>
    where
        B: OriginalChatBackend
            + TextResumeBackend<
                ResumeSource = B::SavedTextComponents,
                DisplacedState = <B as NativeTextStateBackend>::NativeTextState,
            >,
    {
        self.require_live_control()?;
        if self.snapshot_budget.is_some() {
            return Err(ControlledGenerationError::Rejected(
                "snapshot limits are already configured",
            ));
        }
        self.active_mut()?
            .snapshot_estimate(host_capacity.clone())?;
        let budget = SnapshotBudget::prepare(limits, &self.delivery.funding).map_err(|cause| {
            RecordConstructionError::retain(cause.into(), &self.delivery.funding)
        })?;
        self.snapshot_budget = Some(budget);
        self.snapshot_host_capacity = host_capacity;
        self.native_copy_limits = native;
        self.capabilities.snapshot = ControlSupport::Supported;
        self.capabilities.restore = ControlSupport::Supported;
        self.capabilities.fork = ControlSupport::Supported;
        self.capabilities.isolation = Some(SnapshotIsolation::DeepCopy);
        Ok(())
    }
    pub fn snapshot(
        &mut self,
        emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<ControlledGenerationSnapshot<B>, ControlledGenerationError> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.snapshot_inner(emit))) {
            Ok(result) => result,
            Err(payload) => {
                self.failed = true;
                std::panic::resume_unwind(payload)
            }
        }
    }
    fn snapshot_inner(
        &mut self,
        mut emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<ControlledGenerationSnapshot<B>, ControlledGenerationError> {
        self.require_live_control()?;
        records::construction::controls(
            &self.delivery.funding,
            &[
                size_of::<CaptureJournal<'_, B>>(),
                size_of::<CopiedJournal>(),
                size_of::<ControlledGenerationSnapshot<B>>(),
                size_of::<Result<ControlledGenerationSnapshot<B>, ControlledGenerationError>>(),
            ],
        )
        .map_err(|cause| RecordConstructionError::retain(cause, &self.delivery.funding))?;
        let budget = self.require_snapshot_budget()?;
        let session = self.session.as_mut().expect("validated live session");
        let journal = CaptureJournal::<B>::new(
            &self.delivery,
            self.tokenizer_identity,
            session.pending_forced_token(),
            session.status(),
            session.next_prediction(),
        );
        let (snapshot, mut journal) = session.snapshot_with_host(
            &budget,
            self.snapshot_host_capacity.clone(),
            self.native_copy_limits.clone(),
            journal,
        )?;
        journal.metadata.retain_destination(
            snapshot.host_preparation().clone(),
            snapshot.storage_reservation().clone(),
        );
        let record = copy_info(&journal.info, &self.delivery.funding)?;
        self.delivery.send(
            ControlEvent::SnapshotCreated { metadata: record },
            &mut emit,
        );
        self.delivery_result()?;
        Ok(ControlledGenerationSnapshot { snapshot, journal })
    }
}
impl<B> ControlledGenerationSession<'_, B>
where
    B: OriginalChatBackend
        + TextSnapshotBackend
        + TextResumeBackend<
            ResumeSource = <B as TextSnapshotBackend>::SavedTextComponents,
            DisplacedState = <B as NativeTextStateBackend>::NativeTextState,
        >,
{
    /// Reinstalls the canonical saved boundary and copied journal. Transport and
    /// copy consumption are cumulative; terminal snapshots remain terminal.
    pub fn restore(
        &mut self,
        snapshot: &ControlledGenerationSnapshot<B>,
        emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<(), ControlledGenerationError> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.restore_inner(snapshot, emit)
        })) {
            Ok(result) => result,
            Err(payload) => {
                self.failed = true;
                std::panic::resume_unwind(payload)
            }
        }
    }
    fn restore_inner(
        &mut self,
        snapshot: &ControlledGenerationSnapshot<B>,
        mut emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<(), ControlledGenerationError> {
        self.require_live_control()?;
        self.require_snapshot_budget()?;
        let info = &snapshot.journal.info;
        if info.output.run_id != self.delivery.template.run_id
            || info.session_id != self.delivery.template.session_id
            || info.tokenizer_identity != self.tokenizer_identity
            || info.configuration_identity != self.delivery.configuration_identity
        {
            return Err(ControlledGenerationError::Rejected(
                "snapshot belongs to a different recorded run",
            ));
        }
        let next_epoch = self
            .delivery
            .epoch
            .checked_add(1)
            .ok_or(ExecutionControlError::Overflow)?;
        let event = (|| {
            let snapshot_id =
                records::construction::string(&info.snapshot_id, &self.delivery.funding).map_err(
                    |cause| RecordConstructionError::retain(cause, &self.delivery.funding),
                )?;
            let output = copy_output(&info.output, &self.delivery.funding)?;
            Ok::<_, ControlledGenerationError>(ObservedGenerationEvent::Restored {
                snapshot_id,
                output,
            })
        })()?;
        let cancellation = self.delivery.control.cancellation().clone();
        let capacity = self.snapshot_host_capacity.clone();
        let Some(copied) = self.active_mut()?.restore_with_host(
            &snapshot.snapshot,
            PreparedChatResumeSettings::default(),
            capacity,
            &cancellation,
            RestoreJournal {
                semantic_prefix: &snapshot.journal.semantic_prefix,
                prompt: &snapshot.journal.prompt,
            },
        )?
        else {
            return Ok(());
        };
        let RestoredJournal {
            semantic_prefix,
            prompt,
            destination,
        } = copied;
        // Replace the prior payloads before retiring their old destination.
        self.delivery.semantic_prefix = semantic_prefix;
        self.delivery.prompt = prompt;
        self.journal_destination = Some(destination);
        self.delivery.epoch = next_epoch;
        self.delivery.prediction = self.next_prediction();
        self.delivery.timing = self.active()?.timing();
        self.delivery.send(event, &mut emit);
        self.delivery_result()
    }
}
