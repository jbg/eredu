use super::*;
use crate::api::TraceLimits;
use eredu_runtime::execution_control::{
    apply_prepared_sampling_override, TextBranchRequest, TextContinuationBranch,
    TextSnapshotBackend, TextSnapshotError,
};

/// Explicit budgets and future choices for a serial isolated child. The child
/// keeps the snapshot's absolute generation limit and all earlier native state.
pub struct GenerationBranchOptions {
    /// Fresh child transport budget, including the self-contained prefix record.
    pub trace_limits: TraceLimits,
    /// Required for captured/intervened sources; includes inherited consumption.
    pub capture_limits: Option<CaptureLimits>,
    /// Prospective temperature/reseed; absence preserves the saved sampler exactly.
    pub sampling: Option<SamplingOverride>,
    /// None inherits; Some re-admits a replacement plan for future positions only.
    pub intervention: Option<eredu_core::intervention::InterventionPlan>,
}

/// Serializable provenance for an independent logical child run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationBranchMetadata {
    /// Metadata version.
    pub schema_version: u32,
    /// Fresh logical child run identity.
    pub run_id: String,
    /// Exact parent snapshot and inherited absolute/output boundary.
    pub parent: GenerationSnapshotMetadata,
    /// Capture consumption already inherited before the child executes.
    pub inherited_capture_usage: CaptureUsage,
    /// Explicit child transport limit, independent of the parent's consumption.
    pub trace_limits: TraceLimits,
    /// Prospective sampler change, with inherited randomness unless explicitly reseeded.
    pub sampling_override: Option<SamplingOverride>,
    /// Effective sampler facts inherited from the exact saved boundary.
    pub sampling_before: SamplingStateFacts,
    /// Effective child sampler facts after the prospective override.
    pub sampling_after: SamplingStateFacts,
    /// Prospective immutable request re-admitted under this child's session.
    pub intervention_override: Option<eredu_core::intervention::InterventionPlan>,
}

/// Inactive continuation for serial exchange with one exclusively borrowed model.
/// After an exchange this handle holds the previously active run. Dropping a
/// handle releases that inactive state; the active child's lease follows its run.
pub struct ControlledGenerationBranch<B: TextSnapshotBackend> {
    continuation: TextContinuationBranch<B, ControlConstraints>,
    cursor: CommittedGenerationCursor,
    pipeline: CommittedTokenPipeline<PreparedChatTokenDecoder>,
    lifecycle: GenerationLifecycle,
    delivery: Delivery,
}

impl<B: TextSnapshotBackend> ControlledGenerationBranch<B> {
    /// Identity of the run currently held by this inactive slot.
    pub fn run_id(&self) -> &str {
        &self.delivery.template.run_id
    }
    /// Canonical committed prefix in this inactive slot.
    pub fn token_ids(&self) -> &[u32] {
        self.cursor.token_ids()
    }
    /// Lifecycle retained with the inactive run.
    pub fn status(&self) -> GenerationStatus {
        self.lifecycle.status()
    }
    /// Requests affect this logical run even while it is inactive.
    pub fn control_handle(&self) -> GenerationControlHandle {
        self.delivery.control.clone()
    }
}

impl<B: TextSnapshotBackend + TextSamplingControlBackend> ControlledGenerationSession<'_, B> {
    /// Forks a complete saved boundary without installing it, replaying input or
    /// loading weights. The emitted BranchStarted belongs to the child and
    /// contains its inherited semantic prefix. Children execute through `exchange`.
    pub fn fork(
        &mut self,
        snapshot: &ControlledGenerationSnapshot<B>,
        options: GenerationBranchOptions,
        emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<ControlledGenerationBranch<B>, ControlledGenerationError> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.fork_inner(snapshot, options, emit)
        })) {
            Ok(result) => self.snapshot_result(result),
            Err(payload) => {
                self.lifecycle.fail();
                std::panic::resume_unwind(payload)
            }
        }
    }

    fn fork_inner(
        &mut self,
        snapshot: &ControlledGenerationSnapshot<B>,
        options: GenerationBranchOptions,
        mut emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<ControlledGenerationBranch<B>, ControlledGenerationError> {
        self.lifecycle.checkpoint()?;
        if self.delivery.control.cancellation().is_cancelled() {
            return Err(CaptureError::Invalid(
                "cancelled generation cannot create branches".into(),
            )
            .into());
        }
        if snapshot.metadata.tokenizer_identity != self.tokenizer_identity {
            return Err(TextSnapshotError::<eredu_core::BackendFailure>::IncompatibleRun.into());
        }
        let budget = self.require_snapshot_budget()?;
        let max_predictions = snapshot.cursor.max_predictions();
        let native_growth = snapshot
            .continuation
            .native_continuation_growth(self.driver.runtime(), max_predictions)?;
        let host_growth = snapshot
            .pipeline
            .continuation_storage_bytes(max_predictions)
            .and_then(|bytes| {
                bytes.checked_add(
                    snapshot
                        .continuation
                        .controller()
                        .inner()
                        .continuation_storage_bytes(max_predictions)?,
                )
            })
            .and_then(|bytes| {
                bytes.checked_add(
                    options
                        .trace_limits
                        .total_bytes
                        .checked_mul(std::mem::size_of::<SemanticEvent>() as u64 + 1)?,
                )
            })
            .ok_or(ExecutionControlError::UnknownEstimate)?;
        let growth = native_growth
            .checked_add(host_growth)
            .ok_or(ExecutionControlError::Overflow)?;
        let run_id = new_identity("run");
        let session_id = new_identity("branch-session");
        // The complete source reservation covers each inherited host component.
        // Add the owned override DTO and fixed child identity/delivery storage.
        let host = snapshot
            .metadata
            .retained_bytes
            .checked_mul(3)
            .and_then(|bytes| {
                bytes.checked_add(std::mem::size_of::<ControlledGenerationBranch<B>>() as u64)
            })
            .and_then(|bytes| bytes.checked_add((run_id.len() + session_id.len()) as u64))
            .and_then(|bytes| {
                bytes.checked_add(match &options.intervention {
                    Some(plan) => {
                        eredu_runtime::execution_control::intervention_plan_storage_bytes(plan)?
                    }
                    None => 0,
                })
            })
            .ok_or(ExecutionControlError::Overflow)?;
        let replacement_schema = options
            .intervention
            .as_ref()
            .map(|plan| plan.schema_version);
        let (continuation, (pipeline, cursor, mut delivery, lineage)) =
            snapshot.continuation.fork_with(
                &mut self.state.boundary(&mut self.driver)?,
                &budget,
                TextBranchRequest {
                    session_id: &session_id,
                    max_predictions,
                    capture_limits: options.capture_limits,
                    intervention: options.intervention,
                    host_bytes: Some(host),
                    continuation_growth_bytes: Some(growth),
                },
                |runtime, generation| {
                    let sampling_before = B::sampling_control_facts(generation);
                    if let Some(request) = options.sampling {
                        apply_prepared_sampling_override::<B>(runtime, generation, request)
                            .map_err(|error| match error {
                                SamplingOverrideError::Backend(error) => {
                                    TextSnapshotError::Backend(error)
                                }
                                SamplingOverrideError::Invalid(reason) => {
                                    TextSnapshotError::Unsupported(reason)
                                }
                            })?;
                    }
                    let capture = B::capture_run(generation);
                    let lineage = GenerationBranchMetadata {
                        schema_version: EXECUTION_CONTROL_SCHEMA_VERSION,
                        run_id: run_id.clone(),
                        parent: snapshot.metadata.clone(),
                        inherited_capture_usage: snapshot
                            .continuation
                            .capture_checkpoint()
                            .map_or_else(CaptureUsage::default, |capture| {
                                capture.inherited_usage()
                            }),
                        trace_limits: options.trace_limits,
                        sampling_override: options.sampling,
                        sampling_before,
                        sampling_after: B::sampling_control_facts(generation),
                        intervention_override: replacement_schema.map(|schema_version| {
                            capture
                                .and_then(|run| run.intervention_plan())
                                .map(|plan| plan.plan().clone())
                                .unwrap_or_else(|| eredu_core::intervention::InterventionPlan {
                                    schema_version,
                                    operations: Vec::new(),
                                })
                        }),
                    };
                    let template = ObservedGenerationRecord {
                        schema_version: CAPTURE_SCHEMA_VERSION,
                        run_id: run_id.clone(),
                        artifact_identity: snapshot.metadata.artifact_identity.clone(),
                        session_id: session_id.clone(),
                        capture_plan_id: capture.map_or_else(
                            || snapshot.metadata.capture_plan_id.clone(),
                            |run| run.plan().identity().into(),
                        ),
                        intervention_plan_id: capture
                            .and_then(|run| run.intervention_plan())
                            .map(|plan| plan.identity().into()),
                        event: ObservedGenerationEvent::Lifecycle {
                            status: snapshot.metadata.status,
                            next_prediction: snapshot.metadata.output.next_prediction,
                        },
                    };
                    Ok((
                        snapshot.pipeline.fork().map_err(TextSnapshotError::Host)?,
                        snapshot.cursor.clone(),
                        Delivery {
                            configuration_identity: snapshot.metadata.configuration_identity,
                            template,
                            budget: TraceBudget::new(options.trace_limits),
                            control: GenerationControlHandle::default(),
                            sequence: 0,
                            epoch: 0,
                            prediction: snapshot.metadata.output.next_prediction,
                            prompt_length: self.delivery.prompt_length,
                            prompt_token_ids: std::sync::Arc::clone(&snapshot.prompt_token_ids),
                            started: Instant::now(),
                            preparation_elapsed: std::time::Duration::ZERO,
                            timing: GenerationTiming::default(),
                            closed: false,
                            failure: None,
                            semantic_prefix: snapshot.semantic_prefix.clone(),
                        },
                        lineage,
                    ))
                },
            )?;
        delivery.send(
            ObservedGenerationEvent::BranchStarted {
                lineage,
                prompt_token_ids: snapshot.prompt_token_ids.to_vec(),
                inherited_token_ids: snapshot.cursor.token_ids().to_vec(),
                inherited_semantics: snapshot.semantic_prefix.clone(),
            },
            &mut emit,
        );
        if let Some(error) = delivery.failure.take() {
            return Err(error.into());
        }
        Ok(ControlledGenerationBranch {
            continuation,
            pipeline,
            cursor,
            lifecycle: GenerationLifecycle::fork(&snapshot.lifecycle),
            delivery,
        })
    }
}

impl<B: TextSnapshotBackend> ControlledGenerationSession<'_, B> {
    /// Exchanges the active run with a saved branch slot. Native state, semantic
    /// buffers, lifecycle, identity, cancellation and cumulative budgets move
    /// together. No prediction, copy, random draw or semantic replay occurs.
    pub fn exchange(
        &mut self,
        branch: &mut ControlledGenerationBranch<B>,
        emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<(), ControlledGenerationError> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.exchange_inner(branch, emit)
        })) {
            Ok(result) => {
                if result.is_err()
                    && !matches!(
                        B::estimate_native_text_state(self.driver.runtime(), None),
                        Ok(Some(_))
                    )
                {
                    self.lifecycle.fail();
                }
                result
            }
            Err(payload) => {
                self.lifecycle.fail();
                std::panic::resume_unwind(payload)
            }
        }
    }

    fn exchange_inner(
        &mut self,
        branch: &mut ControlledGenerationBranch<B>,
        mut emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<(), ControlledGenerationError> {
        branch
            .continuation
            .exchange(&mut self.driver, &mut self.state)?;
        std::mem::swap(&mut self.pipeline, &mut branch.pipeline);
        std::mem::swap(&mut self.cursor, &mut branch.cursor);
        std::mem::swap(&mut self.lifecycle, &mut branch.lifecycle);
        std::mem::swap(&mut self.delivery, &mut branch.delivery);
        self.lifecycle_record(&mut emit);
        self.delivery_result()
    }
}
