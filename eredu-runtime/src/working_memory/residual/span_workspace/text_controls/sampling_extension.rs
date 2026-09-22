//! A finite sampling program under a fresh protected account and an old run origin.
use super::*;
use crate::working_memory::{
    PendingSamplingExtension, SamplingWorkspacePhase, SamplingWorkspaceReport,
    WorkspaceReportMetadata, text_preparation::SamplingExtensionBinding,
};
use eredu_core::{EstimationCompleteness, StateMemoryAssumptions};
use std::num::NonZeroU8;

/// Unadmitted actual sampling phases, tied to one pending original run boundary.
/// Native mechanisms attach their qualified storage/graph/record facts before
/// this candidate can reserve a new protected account.
#[derive(Debug)]
pub struct SamplingExtensionQuote {
    pending: PendingSamplingExtension,
    quote: IncrementalInferenceQuote,
}

/// Accepted finite replacement program. Its original run remains paused until
/// the native provider completes preparation and commits this owner. Old model,
/// capture and sampling spending is never reset or attached to this account.
#[derive(Debug)]
pub struct OriginalTextSamplingExtension {
    pending: Option<PendingSamplingExtension>,
    binding: SamplingExtensionBinding,
    reseed_claimed: bool,
    workspace: OwnedTextSpanWorkspace,
    run: WorkingMemoryFundingRun,
}

fn report_error(
    cause: crate::working_memory::WorkspaceReportError,
    funding: &eredu_core::HostMetadataFunding,
) -> WorkingMemoryError {
    match cause {
        crate::working_memory::WorkspaceReportError::Domain(error) => error.into(),
        crate::working_memory::WorkspaceReportError::Policy(
            eredu_core::AdmissionPolicyError::ArithmeticOverflow { .. },
        ) => WorkingMemoryError::Overflow,
        crate::working_memory::WorkspaceReportError::Policy(_) => {
            WorkingMemoryError::IdentityMismatch
        }
        crate::working_memory::WorkspaceReportError::Metadata(error) => {
            crate::working_memory::reservation_metadata::neural_error(error, funding)
        }
    }
}
impl PendingSamplingExtension {
    /// Compose only the actual sampled phase reports. The existing model state
    /// is retained by the source request and no new decoder work is described.
    pub fn compose(
        self,
        report: &SamplingWorkspaceReport,
        plan: InferenceSpanWorkspacePlan,
    ) -> Result<SamplingExtensionQuote, WorkingMemoryError> {
        self.binding.validate_pending()?;
        let funding = self.binding.funding().clone();
        let metadata = WorkspaceReportMetadata::with_funding(&funding);
        metadata
            .admit::<(
                SamplingExtensionQuote,
                Result<SamplingExtensionQuote, WorkingMemoryError>,
                RuntimeStateEstimate,
                ExecutionWorkspaceEstimate,
                InferenceSpanWorkspace,
            )>()
            .map_err(|cause| report_error(cause, &funding))?;
        if plan.geometry() != self.geometry()
            || report.steps != self.remaining_steps()
            || !plan
                .metadata_funding()
                .is_some_and(|actual| actual.same_account(&funding))
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let expected = usize::try_from(report.steps)
            .ok()
            .and_then(|n| n.checked_add(1))
            .ok_or(WorkingMemoryError::Overflow)?;
        if plan.records().len() != expected
            || !plan.records().iter().enumerate().all(|(index, row)| {
                row.span()
                    == &InferenceWorkspaceSpan::Sampling(if index == 0 {
                        SamplingWorkspacePhase::Preparation
                    } else {
                        SamplingWorkspacePhase::Step {
                            index: (index - 1) as u64,
                        }
                    })
            })
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let physical = report
            .physical_domains
            .as_ref()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        let peak = report.peak.bytes();
        let tensor = report.tensor_peak_bytes;
        if report.first_gap.is_some() {
            return Err(WorkingMemoryError::UnknownBound);
        }
        let geometry = self.geometry();
        let zero = || {
            metadata.bounded(0, format_args!("sampling extension constructs no model state, attention, or materialization"))
            .map_err(|cause| report_error(cause, &funding))
        };
        let workspace = ExecutionWorkspaceEstimate {
            physical_domains: Some(eredu_core::DomainExecutionWorkspaceEstimate {
                geometry,
                activations: metadata
                    .empty_domain_requirements(physical)
                    .map_err(|e| report_error(e, &funding))?,
                attention: metadata
                    .empty_domain_requirements(physical)
                    .map_err(|e| report_error(e, &funding))?,
                vocabulary: metadata
                    .clone_domain_requirements(physical)
                    .map_err(|e| report_error(e, &funding))?,
                state_update: metadata
                    .empty_domain_requirements(physical)
                    .map_err(|e| report_error(e, &funding))?,
                materialization: metadata
                    .empty_domain_requirements(physical)
                    .map_err(|e| report_error(e, &funding))?,
                retained: metadata
                    .empty_domain_requirements(physical)
                    .map_err(|e| report_error(e, &funding))?,
            }),
            geometry,
            activations: zero()?,
            attention: zero()?,
            vocabulary: metadata
                .clone_bound(&report.peak)
                .map_err(|cause| report_error(cause, &funding))?,
            state_update: zero()?,
            materialization: zero()?,
            retained: zero()?,
        };
        let positions = geometry
            .cached_positions
            .checked_add(geometry.input_positions)
            .and_then(|n| n.checked_add(geometry.max_output_tokens))
            .ok_or(WorkingMemoryError::Overflow)?;
        let state = RuntimeStateEstimate {
            physical_domains: Some(eredu_core::DomainRuntimeStateEstimate {
                geometry,
                decoder_state: metadata
                    .empty_domain_requirements(physical)
                    .map_err(|e| report_error(e, &funding))?,
                media_embeddings: metadata
                    .empty_domain_requirements(physical)
                    .map_err(|e| report_error(e, &funding))?,
                media_workspace: metadata
                    .empty_domain_requirements(physical)
                    .map_err(|e| report_error(e, &funding))?,
            }),
            fixed_state_bytes: 0,
            bytes_per_position_per_batch: 0,
            context_state_bytes: 0,
            selected_state_backing: None,
            multimodal_embedding_bytes: 0,
            media_execution_workspace_bytes: 0,
            requested_state_bytes: 0,
            execution_workspace: Some(workspace),
            persistent_state_completeness: EstimationCompleteness::Complete,
            assumptions: StateMemoryAssumptions {
                floating_state_dtype_bytes: NonZeroU8::new(1).unwrap(),
                batch_size: geometry.batch_size,
                requested_positions: positions,
                sliding_window_bounds: Vec::new(),
                allocation_granularity: 1,
            },
            completeness: EstimationCompleteness::Complete,
        };
        let reservation = self.binding.request().memory_reservation();
        let quote = IncrementalInferenceQuote {
            state: super::super::super::QuoteDiagnostics::new(state, metadata)
                .map_err(|cause| report_error(cause, &funding))?,
            geometry,
            incremental_bytes: peak,
            incremental_requirements: Some(
                metadata
                    .clone_domain_requirements(physical)
                    .map_err(|e| report_error(e, &funding))?,
            ),
            equation_incremental_bytes: tensor,
            pool: reservation.0.pool.clone(),
            pin: None,
            controller: None,
            sources: None,
            span_workspace: InferenceSpanWorkspace {
                plan,
                preparation: Some(0),
                attention: Some(0),
                materialization: Some(0),
                sampling: peak,
                physical_outside: Some({
                    metadata
                        .admit::<(eredu_core::DomainMemoryRequirements, [usize; 2])>()
                        .map_err(|e| report_error(e, &funding))?;
                    Arc::new(
                        metadata
                            .empty_domain_requirements(physical)
                            .map_err(|e| report_error(e, &funding))?,
                    )
                }),
                text_controls: None,
            },
            span_seal: None,
        };
        Ok(SamplingExtensionQuote {
            pending: self,
            quote,
        })
    }
}
impl SamplingExtensionQuote {
    /// Exact phase plan/full bounds before native facts are installed.
    pub fn workspace(&self) -> &InferenceSpanWorkspace {
        self.quote.span_workspace()
    }
    /// Prepare the same control owner with this genuine pending origin.
    pub fn prepare_controls(
        &self,
        facts: TextHostControlFacts,
    ) -> Result<PreparedTextControlWorkspace, WorkingMemoryError> {
        self.pending.binding.validate_pending()?;
        let mut controls = PreparedTextControlWorkspace::prepare_controls(
            self.pending.geometry(),
            self.workspace().plan(),
            facts,
        )?;
        controls.binding.sampling_extension = Some(self.pending.binding.clone());
        Ok(controls)
    }
    /// Seal the complete independently priced native program once.
    pub fn with_controls(
        mut self,
        controls: PreparedTextControlWorkspace,
    ) -> Result<Self, WorkingMemoryError> {
        self.pending.binding.validate_pending()?;
        if !controls
            .binding
            .sampling_extension
            .as_ref()
            .is_some_and(|origin| origin.same(&self.pending.binding))
            || controls.binding.sequence.is_some()
            || controls.binding.source.is_some()
            || controls.binding.prefill_scopes.is_some()
            || controls.binding.prediction_scopes.is_none()
            || controls.binding.native_storage.is_none()
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.pending.binding.validate_policy(
            controls
                .binding
                .graph_metadata
                .and_then(|f| f.requested_ceiling()),
            controls
                .binding
                .tracking
                .and_then(|f| f.requested_ceiling()),
        )?;
        self.quote = self
            .quote
            .with_span_workspace_and_text_controls(controls)
            .map_err(|error| match error {
                ResidualQuoteError::Storage(cause) => cause,
                _ => WorkingMemoryError::UnknownBound,
            })?;
        Ok(self)
    }
    /// Complete currently composed requirement; reading it grants no capacity.
    pub fn required_bytes(&self) -> Option<u64> {
        self.quote.incremental_bytes()
    }
    /// Reserve the complete fresh program against the same pool and execution,
    /// preserving the original request and every old spent role.
    pub fn admit(
        self,
        capacity: eredu_core::MemoryLimits,
    ) -> Result<OriginalTextSamplingExtension, WorkingMemoryError> {
        self.pending.binding.validate_pending()?;
        if self.quote.span_seal.is_none() || self.quote.span_workspace.text_controls.is_none() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let funding = self.pending.binding.funding().clone();
        let metadata = WorkspaceReportMetadata::with_funding(&funding);
        metadata
            .admit::<(
                OriginalTextSamplingExtension,
                Result<OriginalTextSamplingExtension, WorkingMemoryError>,
                Admission,
            )>()
            .map_err(|cause| report_error(cause, &funding))?;
        let original = self.pending.binding.request().memory_reservation();
        let admission = Admission {
            memory_limits: original.admission().memory_limits.clone(),
            additional_headroom: original.admission().additional_headroom.clone(),
            requested_positions: original.admission().requested_positions,
            state: metadata
                .clone_state(self.quote.state())
                .map_err(|cause| report_error(cause, &funding))?,
            incremental_required_bytes: self.quote.incremental_bytes(),
        };
        let reservation = self.quote.reserve(
            &original.0.pool,
            &original.0.execution,
            &admission,
            capacity,
            &[],
        )?;
        let (reservation, run) = reservation.into_funding()?;
        let (workspace, witness) = self
            .quote
            .into_funded_text_span_workspace(&run, &reservation)
            .map_err(|error| error.cause)?;
        if witness.is_some() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let binding = self.pending.binding.clone();
        Ok(OriginalTextSamplingExtension {
            pending: Some(self.pending),
            binding,
            reseed_claimed: false,
            workspace,
            run,
        })
    }
}
impl OriginalTextSamplingExtension {
    /// Original protected funding run, distinct from the model's old account.
    pub fn funding(&self) -> &WorkingMemoryFundingRun {
        &self.run
    }
    /// Same exact newly accepted host/native control association.
    pub fn control_guard(&self) -> OriginalTextControlGuard {
        self.workspace.control_guard()
    }
    /// Extract the existing qualified native storage bank once.
    pub fn take_native_storage_bank<M: OriginalNativeStorageMechanism>(
        &mut self,
        selection: &NativeStorageSelection,
    ) -> Result<Option<OriginalNativeStorageBank<M>>, WorkingMemoryError> {
        self.binding.validate_pending()?;
        self.workspace
            .take_native_storage_bank(&self.run, selection)
    }
    /// Extract the same graph metadata capsule after pending-origin validation.
    pub fn take_graph_metadata(
        &mut self,
    ) -> Result<Option<OriginalGraphMetadata>, WorkingMemoryError> {
        self.binding.validate_pending()?;
        self.workspace.take_extension_graph_metadata(&self.binding)
    }
    /// Extract the same submission tracking capsule after pending-origin validation.
    pub fn take_submission_tracking(
        &mut self,
    ) -> Result<Option<OriginalSubmissionTracking>, WorkingMemoryError> {
        self.binding.validate_pending()?;
        self.workspace
            .take_extension_submission_tracking(&self.binding)
    }
    /// Extract the same finite role bank, with only sampling/event roles present.
    pub fn take_prediction_scopes(
        &mut self,
    ) -> Result<Option<OriginalTextPredictionScopes>, WorkingMemoryError> {
        self.binding.validate_pending()?;
        self.workspace.take_prediction_scopes()
    }
    /// Claim the one optional reseed construction before native entry.
    pub fn claim_reseed(&mut self) -> Result<OriginalPreparationScopeCustody, WorkingMemoryError> {
        self.binding.validate_pending()?;
        if self.reseed_claimed
            || self
                .workspace
                .workspace()
                .control_binding()
                .and_then(|b| b.preparation_scopes)
                .is_none()
        {
            return Err(WorkingMemoryError::AlreadyStarted);
        }
        self.reseed_claimed = true;
        Ok(OriginalPreparationScopeCustody::sampling(
            &self.workspace.controls,
        ))
    }
    /// Publish the fully prepared replacement before issuing another prediction.
    /// Native completion and atomic sampler installation remain provider duties.
    pub fn commit(&mut self) -> Result<(), WorkingMemoryError> {
        let pending = self
            .pending
            .take()
            .ok_or(WorkingMemoryError::AlreadyStarted)?;
        self.binding = pending.commit()?;
        Ok(())
    }
}
