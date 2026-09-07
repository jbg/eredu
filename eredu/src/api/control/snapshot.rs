use super::*;
use eredu_runtime::execution_control::{
    GenerationBoundary, SnapshotBudget, SnapshotTokenController, TextContinuationSnapshot,
    TextSnapshotBackend,
};

/// A delivered output prefix. Consumers retain events strictly before
/// `next_sequence` from this run/epoch when reconciling a restore; the new stream
/// continues with its current monotone sequence and a new epoch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationOutputCheckpoint {
    /// Source logical run.
    pub run_id: String,
    /// Epoch in which the prefix was delivered.
    pub epoch: u64,
    /// Exclusive end of the delivered prefix in the run's monotone event journal.
    pub next_sequence: u64,
    /// Next absolute prediction, including the committed inherited prefix.
    pub next_prediction: u64,
}

/// Versioned metadata for an opaque in-process continuation. Serialized metadata
/// cannot recreate a snapshot, admission, native handle or compatibility proof.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationSnapshotMetadata {
    /// Metadata version.
    pub schema_version: u32,
    /// Fresh snapshot identity.
    pub snapshot_id: String,
    /// Loaded session that owns the executable and prepared sources.
    pub session_id: String,
    /// Source artifact attribution, when present on the prepared request.
    pub artifact_identity: Option<String>,
    /// Immutable capture admission retained with this continuation.
    pub capture_plan_id: String,
    /// Immutable intervention admission retained with this continuation.
    pub intervention_plan_id: Option<String>,
    /// Exact immutable tokenizer fingerprint of this run.
    pub tokenizer_identity: [u8; 32],
    /// Initial generation and semantic-policy identity, including constraint
    /// blueprint, termination, tokenizer and seed. Later choices are retained
    /// exactly in the opaque saved state and attributed through control records.
    pub configuration_identity: [u8; 32],
    /// Prospective canonical choice saved before its ordinary commitment.
    pub pending_forced_token: Option<u32>,
    /// Prepared, paused, or normally completed state preserved by restoration.
    pub status: GenerationStatus,
    /// Delivered prefix paired with incremental semantic buffers.
    pub output: GenerationOutputCheckpoint,
    /// Complete charged logical native/controller/host state, excluding shared
    /// immutable weights/tokenizer data and allocator overhead.
    pub retained_bytes: u64,
}

/// Reusable deep snapshot of the ordinary native, sampler, constraint, semantic,
/// termination, pending-input and capture/intervention continuation. It is usable
/// only with the source logical run and loaded driver; metadata may outlive that
/// run, but cannot authorize restoration into a new model or process.
pub struct ControlledGenerationSnapshot<B: TextSnapshotBackend> {
    pub(super) continuation: TextContinuationSnapshot<B, ControlConstraints>,
    pub(super) cursor: CommittedGenerationCursor,
    pub(super) pipeline: CommittedTokenPipeline<PreparedChatTokenDecoder>,
    pub(super) lifecycle: GenerationBoundary,
    pub(super) metadata: GenerationSnapshotMetadata,
    pub(super) semantic_prefix: Vec<SemanticEvent>,
    pub(super) prompt_token_ids: std::sync::Arc<[u32]>,
}
impl<B: TextSnapshotBackend> ControlledGenerationSnapshot<B> {
    /// Exact immutable canonical prompt, shared safely by the branch tree.
    pub fn prompt_token_ids(&self) -> &[u32] {
        &self.prompt_token_ids
    }
    /// Stable metadata with no native handles.
    pub fn metadata(&self) -> &GenerationSnapshotMetadata {
        &self.metadata
    }
    /// Exact committed canonical prefix, including any terminal special token.
    pub fn token_ids(&self) -> &[u32] {
        self.cursor.token_ids()
    }
}

impl<B: TextGenerationBackend> ControlledGenerationSession<'_, B> {
    /// Current consumer checkpoint; capturing it alone does not save native state.
    pub fn output_checkpoint(&self) -> GenerationOutputCheckpoint {
        GenerationOutputCheckpoint {
            run_id: self.delivery.template.run_id.clone(),
            epoch: self.delivery.epoch,
            next_sequence: self.delivery.sequence,
            next_prediction: self.next_prediction(),
        }
    }
    /// Non-rewindable resource consumption, absent until limits are configured.
    pub fn snapshot_usage(&self) -> Option<SnapshotUsage> {
        self.snapshot_budget.as_ref().map(SnapshotBudget::usage)
    }
}

impl<B: TextSnapshotBackend> ControlledGenerationSession<'_, B> {
    pub(super) fn snapshot_result<T>(
        &mut self,
        result: Result<T, ControlledGenerationError<B::Error>>,
    ) -> Result<T, ControlledGenerationError<B::Error>> {
        if matches!(
            &result,
            Err(ControlledGenerationError::Snapshot(
                eredu_runtime::execution_control::TextSnapshotError::Backend(_)
            ))
        ) && !matches!(
            B::estimate_native_text_state(self.driver.runtime(), None),
            Ok(Some(_))
        ) {
            // Preserve known, safely completed copy failures. An unresolved or
            // poisoned native owner must not leave a resumable facade lifecycle.
            self.lifecycle.fail();
        }
        result
    }
    /// Configures resource limits exactly once. Unknown required native, grammar,
    /// or semantic storage fails explicitly before copying. Limits remain outside
    /// snapshots and cannot be reset by pause, restore or repeated configuration.
    pub fn enable_snapshots(
        &mut self,
        limits: SnapshotLimits,
    ) -> Result<(), ControlledGenerationError<B::Error>> {
        self.lifecycle.checkpoint()?;
        if self.snapshot_budget.is_some() {
            return Err(
                CaptureError::Invalid("snapshot limits are already configured".into()).into(),
            );
        }
        if let ControlSupport::Unsupported { reason } =
            B::native_text_state_support(self.driver.runtime())
        {
            return Err(CaptureError::Unsupported(reason).into());
        }
        if self.state.controller().snapshot_storage_bytes().is_none() {
            return Err(CaptureError::Unsupported(
                "complete active grammar storage estimate is unavailable".into(),
            )
            .into());
        }
        self.semantic_snapshot_bytes()
            .ok_or(ExecutionControlError::UnknownEstimate)?;
        let boundary = self.state.boundary(&mut self.driver)?;
        let (runtime, state, pending) = boundary.parts();
        B::estimate_native_text_state(runtime, None)
            .map_err(eredu_runtime::execution_control::TextSnapshotError::Backend)?
            .ok_or(ExecutionControlError::UnknownEstimate)?;
        B::estimate_sampling_state(runtime, B::sampling_state(state))
            .map_err(eredu_runtime::execution_control::TextSnapshotError::Backend)?
            .ok_or(ExecutionControlError::UnknownEstimate)?;
        B::estimate_pending_input(runtime, pending)
            .map_err(eredu_runtime::execution_control::TextSnapshotError::Backend)?
            .ok_or(ExecutionControlError::UnknownEstimate)?;
        drop(boundary);
        self.snapshot_budget = Some(SnapshotBudget::new(limits));
        Ok(())
    }

    pub(super) fn require_snapshot_budget(
        &self,
    ) -> Result<SnapshotBudget, ControlledGenerationError<B::Error>> {
        self.snapshot_budget.clone().ok_or_else(|| {
            CaptureError::Invalid("configure snapshot limits before retaining state".into()).into()
        })
    }

    /// Captures the initial, paused or normally completed boundary. No prompt
    /// replay, RNG draw, semantic finalization or admission reset occurs.
    pub fn snapshot(
        &mut self,
        emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<ControlledGenerationSnapshot<B>, ControlledGenerationError<B::Error>> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.snapshot_inner(emit))) {
            Ok(result) => self.snapshot_result(result),
            Err(payload) => {
                self.lifecycle.fail();
                std::panic::resume_unwind(payload)
            }
        }
    }

    fn snapshot_inner(
        &mut self,
        mut emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<ControlledGenerationSnapshot<B>, ControlledGenerationError<B::Error>> {
        if self.delivery.control.cancellation().is_cancelled() {
            return Err(
                CaptureError::Invalid("cancelled generation cannot be snapshotted".into()).into(),
            );
        }
        let lifecycle = self.lifecycle.checkpoint()?;
        let budget = self.require_snapshot_budget()?;
        let mut metadata = GenerationSnapshotMetadata {
            schema_version: EXECUTION_CONTROL_SCHEMA_VERSION,
            snapshot_id: new_identity("snapshot"),
            session_id: self.delivery.template.session_id.clone(),
            artifact_identity: self.delivery.template.artifact_identity.clone(),
            capture_plan_id: self.delivery.template.capture_plan_id.clone(),
            intervention_plan_id: self.delivery.template.intervention_plan_id.clone(),
            tokenizer_identity: self.tokenizer_identity,
            configuration_identity: self.delivery.configuration_identity,
            pending_forced_token: self.pending_forced_token(),
            status: self.status(),
            output: self.output_checkpoint(),
            retained_bytes: 0,
        };
        let host_bytes = self
            .semantic_snapshot_bytes()
            .ok_or(ExecutionControlError::UnknownEstimate)?
            .checked_add(std::mem::size_of::<ControlledGenerationSnapshot<B>>() as u64)
            .and_then(|n| n.checked_add(metadata.snapshot_id.len() as u64))
            .and_then(|n| n.checked_add(metadata.session_id.len() as u64))
            .and_then(|n| {
                n.checked_add(
                    metadata
                        .artifact_identity
                        .as_ref()
                        .map_or(0, |id| id.len() as u64),
                )
            })
            .and_then(|n| n.checked_add(metadata.output.run_id.len() as u64))
            .and_then(|n| n.checked_add(metadata.capture_plan_id.len() as u64))
            .and_then(|n| {
                n.checked_add(
                    metadata
                        .intervention_plan_id
                        .as_ref()
                        .map_or(0, |id| id.len() as u64),
                )
            })
            .ok_or(ExecutionControlError::Overflow)?;
        let continuation = TextContinuationSnapshot::capture(
            &mut self.state.boundary(&mut self.driver)?,
            &budget,
            Some(host_bytes),
        )?;
        let pipeline = self
            .pipeline
            .fork()
            .map_err(eredu_runtime::execution_control::TextSnapshotError::Host)?;
        let cursor = self.cursor.clone();
        metadata.retained_bytes = continuation.retained_bytes();
        let snapshot = ControlledGenerationSnapshot {
            continuation,
            pipeline,
            cursor,
            lifecycle,
            metadata,
            semantic_prefix: self.delivery.semantic_prefix.clone(),
            prompt_token_ids: std::sync::Arc::clone(&self.delivery.prompt_token_ids),
        };
        let max_predictions = snapshot.cursor.max_predictions();
        self.branch_growth_known = matches!(
            B::text_sampling_control_support(self.driver.runtime()),
            ControlSupport::Supported
        ) && snapshot
            .pipeline
            .continuation_storage_bytes(max_predictions)
            .is_some()
            && snapshot
                .continuation
                .controller()
                .inner()
                .continuation_storage_bytes(max_predictions)
                .is_some()
            && snapshot
                .continuation
                .native_continuation_growth(self.driver.runtime(), max_predictions)
                .is_ok();
        self.delivery.send(
            ObservedGenerationEvent::SnapshotCreated {
                metadata: snapshot.metadata.clone(),
            },
            &mut emit,
        );
        self.delivery_result()?;
        Ok(snapshot)
    }

    /// Restores the same run from a reusable complete snapshot. Sequence, epoch,
    /// capture consumption, transport and copy accounting never rewind. Consumers
    /// reconcile to the `Restored` prefix; old text/tokens are not delivered again.
    /// A normally completed snapshot remains terminal. Cancellation/failure cannot
    /// be revived by restoration.
    pub fn restore(
        &mut self,
        snapshot: &ControlledGenerationSnapshot<B>,
        emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<(), ControlledGenerationError<B::Error>> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.restore_inner(snapshot, emit)
        })) {
            Ok(result) => self.snapshot_result(result),
            Err(payload) => {
                self.lifecycle.fail();
                std::panic::resume_unwind(payload)
            }
        }
    }

    fn restore_inner(
        &mut self,
        snapshot: &ControlledGenerationSnapshot<B>,
        mut emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<(), ControlledGenerationError<B::Error>> {
        if self.delivery.control.cancellation().is_cancelled() {
            return Err(
                CaptureError::Invalid("cancelled generation cannot be restored".into()).into(),
            );
        }
        self.lifecycle.validate_restore()?;
        if snapshot.metadata.output.run_id != self.delivery.template.run_id
            || snapshot.metadata.session_id != self.delivery.template.session_id
            || snapshot.metadata.tokenizer_identity != self.tokenizer_identity
            || snapshot.metadata.configuration_identity != self.delivery.configuration_identity
        {
            return Err(
                eredu_runtime::execution_control::TextSnapshotError::IncompatibleRun.into(),
            );
        }
        let budget = self.require_snapshot_budget()?;
        let (pipeline, cursor, semantic_prefix) = snapshot.continuation.restore_with(
            &mut self.state.boundary(&mut self.driver)?,
            &budget,
            || {
                Ok((
                    snapshot.pipeline.fork()?,
                    snapshot.cursor.clone(),
                    snapshot.semantic_prefix.clone(),
                ))
            },
        )?;
        self.pipeline = pipeline;
        self.cursor = cursor;
        self.delivery.semantic_prefix = semantic_prefix;
        // Validated before native installation, with no intervening transition.
        self.lifecycle
            .restore(&snapshot.lifecycle)
            .expect("validated lifecycle restore");
        self.delivery.epoch = self.lifecycle.epoch();
        self.delivery.prediction = self.next_prediction();
        self.delivery.send(
            ObservedGenerationEvent::Restored {
                snapshot_id: snapshot.metadata.snapshot_id.clone(),
                output: snapshot.metadata.output.clone(),
            },
            &mut emit,
        );
        self.delivery_result()
    }
}
