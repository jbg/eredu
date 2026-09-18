use super::*;
use crate::api::{PreparedChatBranch, PreparedChatResumeSettings};
use eredu_runtime::working_memory::OriginalChatBackend;
use snapshot::journal::{BranchJournal, BranchedJournal};

/// Independent child transport limits and prospective original source policy.
pub struct GenerationBranchOptions {
    pub trace_limits: TraceLimits,
    pub capture_limits: Option<CaptureLimits>,
    pub sampling: Option<SamplingOverride>,
    pub intervention: Option<eredu_core::intervention::InterventionPlan>,
}
/// Inactive canonical machine and its independently delivered record journal.
pub struct ControlledGenerationBranch<B: TextSnapshotBackend> {
    branch: PreparedChatBranch<B>,
    failure: ControlledSessionFailure,
    tokenizer_identity: [u8; 32],
    snapshot_budget: Option<SnapshotBudget>,
    snapshot_host_capacity: u64,
    native_copy_limits: eredu_runtime::working_memory::WorkspaceCopyLimits,
    capabilities: ExecutionControlCapabilities,
    delivery: Delivery,
    journal_destination: Option<HostPreparationAuthority>,
}
impl<B: TextSnapshotBackend> ControlledGenerationBranch<B> {
    pub fn run_id(&self) -> &str {
        &self.delivery.template.run_id
    }
    pub fn token_ids(&self) -> &[u32] {
        self.branch.token_ids()
    }
    pub fn status(&self) -> GenerationStatus {
        self.branch.status()
    }
    pub fn control_handle(&self) -> GenerationControlHandle {
        self.delivery.control.clone()
    }
}
fn identity(
    kind: &'static str,
    funding: &HostMetadataFunding,
) -> records::construction::Result<String> {
    let identity = crate::api::observed::PreparedIdentity::new(kind);
    let mut output = records::construction::string_capacity(
        identity
            .bytes()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or(HostMetadataFundingError::Overflow)?,
        funding,
    )?;
    identity
        .write_into(&mut output)
        .expect("funded String writer");
    Ok(output)
}
fn copy_capabilities(
    source: &ExecutionControlCapabilities,
    funding: &HostMetadataFunding,
) -> records::construction::Result<ExecutionControlCapabilities> {
    use records::construction::{string, vector};
    let support = |source: &ControlSupport| -> records::construction::Result<ControlSupport> {
        Ok(match source {
            ControlSupport::Supported => ControlSupport::Supported,
            ControlSupport::Unsupported { reason } => ControlSupport::Unsupported {
                reason: string(reason, funding)?,
            },
        })
    };
    let mut conditions = vector(source.conditions.len(), funding)?;
    for condition in &source.conditions {
        conditions.push(string(condition, funding)?)
    }
    Ok(ExecutionControlCapabilities {
        schema_version: source.schema_version,
        step: support(&source.step)?,
        pause_resume: support(&source.pause_resume)?,
        snapshot: support(&source.snapshot)?,
        restore: support(&source.restore)?,
        fork: support(&source.fork)?,
        force_next_token: support(&source.force_next_token)?,
        sampling_overrides: support(&source.sampling_overrides)?,
        isolation: source.isolation,
        conditions,
    })
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
    pub fn fork(
        &mut self,
        snapshot: &ControlledGenerationSnapshot<B>,
        options: GenerationBranchOptions,
        emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<ControlledGenerationBranch<B>, ControlledGenerationError> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.fork_inner(snapshot, options, emit)
        })) {
            Ok(result) => result,
            Err(payload) => {
                self.failed = true;
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
        self.require_live_control()?;
        let budget = self.require_snapshot_budget()?;
        if snapshot.journal.info.tokenizer_identity != self.tokenizer_identity {
            return Err(ControlledGenerationError::Rejected(
                "snapshot tokenizer differs from recorded run",
            ));
        }
        if options.trace_limits.per_record_bytes == 0 || options.trace_limits.total_bytes == 0 {
            return Err(ControlledGenerationError::Rejected(
                "trace byte limits must be positive",
            ));
        }
        let source_facts =
            snapshot
                .snapshot
                .resume_source_facts()
                .ok_or(ControlledGenerationError::Rejected(
                    "saved source has no qualified lineage projection",
                ))?;
        let funding = self.delivery.funding.clone();
        let prepare = (|| -> records::construction::Result<_> {
            use records::construction::{controls, intervention_request, optional, string, vector};
            controls(
                &funding,
                &[
                    size_of::<ControlledGenerationBranch<B>>(),
                    size_of::<GenerationBranchOptions>(),
                    size_of::<BranchJournal<'_>>(),
                    size_of::<BranchedJournal>(),
                    size_of::<Delivery>(),
                    size_of::<BranchInfo>(),
                    size_of::<ExecutionControlCapabilities>(),
                    size_of::<RecordContext>(),
                    size_of::<TextResumeSourceFacts>(),
                    size_of::<TextResumeFacts<'_>>(),
                    size_of::<OriginalTextResumeOptions<'_>>(),
                    size_of::<Result<ControlledGenerationBranch<B>, ControlledGenerationError>>(),
                    size_of::<crate::api::observed::PreparedIdentity<'_>>(),
                ],
            )?;
            let run_id = identity("run", &funding)?;
            let session_id = identity("branch-session", &funding)?;
            let template = RecordContext {
                run_id,
                session_id,
                artifact_identity: optional(&snapshot.journal.info.artifact_identity, &funding)?,
                parameter_overlay_id: optional(
                    &self.delivery.template.parameter_overlay_id,
                    &funding,
                )?,
                capture_plan_id: None,
                intervention_plan_id: None,
            };
            let mut inherited_token_ids = vector(snapshot.token_ids().len(), &funding)?;
            inherited_token_ids.extend_from_slice(snapshot.token_ids());
            let intervention = options
                .intervention
                .as_ref()
                .map(|request| intervention_request(request, &funding))
                .transpose()?;
            let lineage_run = string(&template.run_id, &funding)?;
            Ok((
                template,
                copy_capabilities(&self.capabilities, &funding)?,
                inherited_token_ids,
                intervention,
                lineage_run,
            ))
        })()
        .map_err(|cause| RecordConstructionError::retain(cause, &funding))?;
        let (mut template, capabilities, inherited_token_ids, intervention_override, lineage_run) =
            prepare;
        let parent = snapshot::copy_info(&snapshot.journal.info, &funding)?;
        let inherited_semantics =
            snapshot::journal::funded_events(&snapshot.journal.semantic_prefix, &funding)?;
        let failure = ControlledSessionFailure::prepare(&funding)?;
        let options_source = OriginalTextResumeOptions {
            kind: OriginalTextResumeKind::Branch,
            terminal: false,
            session_id: Some(&template.session_id),
            capture_limits: options.capture_limits.as_ref(),
            intervention: options.intervention.as_ref(),
            sampling: options.sampling,
        };
        let cancellation = self.delivery.control.cancellation().clone();
        let capacity = self.snapshot_host_capacity;
        let child = self.active_mut()?.fork_with_host(
            &snapshot.snapshot,
            PreparedChatResumeSettings::default(),
            options_source,
            capacity,
            &cancellation,
            BranchJournal {
                restored: snapshot::RestoreJournal {
                    semantic_prefix: &snapshot.journal.semantic_prefix,
                    prompt: &snapshot.journal.prompt,
                },
            },
        )?;
        let Some((branch, BranchedJournal { restored, control })) = child else {
            return Err(ControlledGenerationError::Rejected(
                "branch preparation was cancelled",
            ));
        };
        let facts = branch.resume_facts();
        template.capture_plan_id = facts
            .capture_plan_id
            .map(|id| records::construction::string(id, &funding))
            .transpose()
            .map_err(|cause| RecordConstructionError::retain(cause, &funding))?;
        template.intervention_plan_id = facts
            .intervention_plan_id
            .filter(|_| facts.has_interventions)
            .map(|id| records::construction::string(id, &funding))
            .transpose()
            .map_err(|cause| RecordConstructionError::retain(cause, &funding))?;
        let lineage = BranchInfo {
            run_id: lineage_run,
            parent,
            inherited_capture_usage: source_facts.inherited_capture_usage,
            trace_limits: options.trace_limits,
            sampling_override: options.sampling,
            sampling_before: source_facts.sampling_before,
            sampling_after: facts.sampling_after,
            intervention_override,
        };
        let snapshot::RestoredJournal {
            semantic_prefix,
            prompt,
            destination,
        } = restored;
        let mut delivery = Delivery {
            configuration_identity: snapshot.journal.info.configuration_identity,
            template,
            budget: TraceBudget::new(options.trace_limits),
            control,
            sequence: 0,
            epoch: 0,
            prediction: branch.next_prediction(),
            prompt,
            started: Instant::now(),
            preparation_elapsed: std::time::Duration::ZERO,
            timing: GenerationTiming::default(),
            closed: false,
            failure: None,
            record_failure: None,
            semantic_prefix,
            funding,
        };
        delivery.send(
            ControlEvent::BranchStarted {
                lineage,
                inherited_token_ids,
                inherited_semantics,
            },
            &mut emit,
        );
        let local = if let Some(error) = delivery.record_failure.take() {
            Err(error.into())
        } else if let Some(error) = delivery.failure.take() {
            Err(error.into())
        } else {
            Ok((!delivery.control.cancellation().is_cancelled()).then_some(()))
        };
        if self
            .active()?
            .finish_record_delivery(local, ControlledGenerationError::Backend)?
            .is_none()
        {
            delivery.control.cancel();
        }
        Ok(ControlledGenerationBranch {
            branch,
            failure,
            tokenizer_identity: self.tokenizer_identity,
            snapshot_budget: Some(budget),
            snapshot_host_capacity: capacity,
            native_copy_limits: self.native_copy_limits,
            capabilities,
            delivery,
            journal_destination: Some(destination),
        })
    }
}
impl<B: TextSnapshotBackend> ControlledGenerationSession<'_, B> {
    /// Exchanges complete machine and journal ownership; neither side advances.
    pub fn exchange(
        &mut self,
        branch: &mut ControlledGenerationBranch<B>,
        emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<(), ControlledGenerationError> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.exchange_inner(branch, emit)
        })) {
            Ok(result) => result,
            Err(payload) => {
                self.failed = true;
                std::panic::resume_unwind(payload)
            }
        }
    }
    fn exchange_inner(
        &mut self,
        branch: &mut ControlledGenerationBranch<B>,
        mut emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<(), ControlledGenerationError> {
        self.active_mut()?.exchange(&mut branch.branch)?;
        std::mem::swap(&mut self.failure, &mut branch.failure);
        std::mem::swap(&mut self.tokenizer_identity, &mut branch.tokenizer_identity);
        std::mem::swap(&mut self.snapshot_budget, &mut branch.snapshot_budget);
        std::mem::swap(
            &mut self.snapshot_host_capacity,
            &mut branch.snapshot_host_capacity,
        );
        std::mem::swap(&mut self.native_copy_limits, &mut branch.native_copy_limits);
        std::mem::swap(&mut self.capabilities, &mut branch.capabilities);
        std::mem::swap(&mut self.delivery, &mut branch.delivery);
        std::mem::swap(
            &mut self.journal_destination,
            &mut branch.journal_destination,
        );
        self.lifecycle_record(&mut emit);
        self.delivery_result()
    }
}
