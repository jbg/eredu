//! Saved-source admission with credit confined to a sealed copy program.

use super::*;
use crate::working_memory::{
    RegisteredSavedSamplingSource, RegisteredWorkspaceCopy, Usage, WorkingMemoryFundingRun,
    saved_source::SavedSourceValidation,
};

/// Failure while constructing the same closed copy quote with finite metadata.
/// Ordinary callers retain their existing ResidualQuoteError adapter.
#[derive(Debug, thiserror::Error)]
pub enum WorkspaceCopyCompositionError {
    /// Original source/coverage or association failure.
    #[error(transparent)]
    Quote(#[from] ResidualQuoteError),
    /// Fixed or already paid report construction failure.
    #[error(transparent)]
    Report(#[from] crate::working_memory::WorkspaceReportError),
}
impl WorkspaceCopyCompositionError {
    /// Preserves the ordinary source/coverage error adapter.
    pub fn into_legacy(self) -> ResidualQuoteError {
        match self {
            Self::Quote(error) => error,
            Self::Report(error) => ResidualQuoteError::Estimate(error.into_capability()),
        }
    }
}
impl WorkspaceCopyCompositionError {
    /// Keeps paid metadata errors intact and pays before retaining other causes.
    pub fn into_workspace(
        self,
        metadata: crate::working_memory::WorkspaceReportMetadata<'_>,
    ) -> eredu_nn::Error {
        match self {
            Self::Report(error) => metadata.error(error),
            Self::Quote(error) => metadata.source(error),
        }
    }
}

impl From<WorkingMemoryError> for WorkspaceCopyCompositionError {
    fn from(error: WorkingMemoryError) -> Self {
        Self::Quote(error.into())
    }
}
impl From<IncompleteWorkspace> for WorkspaceCopyCompositionError {
    fn from(error: IncompleteWorkspace) -> Self {
        Self::Quote(error.into())
    }
}

/// Full inference diagnostics with an identity-bound incremental preparation.
///
/// The fixed isolated-copy program receives its own registered-source credit.
/// A separately registered completed prepared input can additionally supply
/// exact equation-root credit. Copied destination state, sampling and all other
/// supplied enclosing resources remain fully priced. This proof retains accounting pins, not
/// numerical payload or an inference grant. Its concrete provider must bind
/// and execute the same ordered copy operands under settled source custody.
#[derive(Debug)]
pub struct CopyPreparationInferenceQuote<K: Ord + Send + 'static> {
    proof: IncrementalInferenceQuote,
    copy: RegisteredWorkspaceCopy<K>,
}

impl<K: Clone + Ord + Send + Sync + 'static> RegisteredWorkspaceCopy<K> {
    /// Adds this closed copy's full and incremental terms to otherwise identical
    /// complete request contributions. `outside_without_copy` must include all
    /// preparation, sampling, controller and transfer costs except this exact
    /// isolated-copy program. In particular, independently copied host payload
    /// and any numerical suffix of the copy require their own complete bounds.
    ///
    /// A mutable trace report or arbitrary byte discount cannot replace the
    /// stored plan. Original complete diagnostics survive unchanged by credit.
    pub fn compose_inference(
        self,
        equations: &InferenceWorkspaceReport,
        state: RuntimeStateEstimate,
        outside_without_copy: ExecutionWorkspaceEstimate,
    ) -> Result<CopyPreparationInferenceQuote<K>, ResidualQuoteError> {
        self.compose_inference_metadata(
            equations,
            state,
            outside_without_copy,
            crate::working_memory::WorkspaceReportMetadata::ordinary(),
        )
        .map_err(WorkspaceCopyCompositionError::into_legacy)
    }

    /// The same source-credit composition with counted report destinations.
    /// This neither expands copy credit nor changes native source qualification.
    pub fn compose_inference_metadata(
        self,
        equations: &InferenceWorkspaceReport,
        state: RuntimeStateEstimate,
        outside_without_copy: ExecutionWorkspaceEstimate,
        metadata: crate::working_memory::WorkspaceReportMetadata<'_>,
    ) -> Result<CopyPreparationInferenceQuote<K>, WorkspaceCopyCompositionError> {
        self.compose_inference_source_metadata(
            equations,
            state,
            outside_without_copy,
            None,
            metadata,
        )
    }

    /// Adds completed prepared-input credit only through its actual registered
    /// root association. The copied decoder/key proof remains independent and
    /// both account pins survive admission; no byte subtraction is accepted.
    pub fn compose_inference_with_prepared_source_metadata(
        self,
        equations: &InferenceWorkspaceReport,
        state: RuntimeStateEstimate,
        outside_without_copy: ExecutionWorkspaceEstimate,
        prepared: &RegisteredPreparedWorkspaceStorage<K>,
        metadata: crate::working_memory::WorkspaceReportMetadata<'_>,
    ) -> Result<CopyPreparationInferenceQuote<K>, WorkspaceCopyCompositionError> {
        self.compose_inference_source_metadata(
            equations,
            state,
            outside_without_copy,
            Some(prepared),
            metadata,
        )
    }

    fn compose_inference_source_metadata(
        self,
        equations: &InferenceWorkspaceReport,
        state: RuntimeStateEstimate,
        outside_without_copy: ExecutionWorkspaceEstimate,
        prepared: Option<&RegisteredPreparedWorkspaceStorage<K>>,
        metadata: crate::working_memory::WorkspaceReportMetadata<'_>,
    ) -> Result<CopyPreparationInferenceQuote<K>, WorkspaceCopyCompositionError> {
        metadata.admit::<CopyPreparationInferenceQuote<K>>()?;
        metadata.admit::<WorkspaceCopyCompositionError>()?;
        metadata.admit::<(
            Option<&RegisteredPreparedWorkspaceStorage<K>>,
            Option<RegisteredStoragePin>,
        )>()?;
        metadata.admit::<IncrementalInferenceQuote>()?;
        metadata.admit::<InferenceSpanWorkspace>()?;
        let geometry = equations.geometry();
        validate_workspace_fixed(&outside_without_copy, geometry)
            .map_err(crate::working_memory::WorkspaceReportError::from)?;
        if !equations.has_complete_state_spans() {
            return Err(WorkingMemoryError::UnknownBound.into());
        }
        let span = self
            .report()
            .state
            .as_ref()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        let full_copy = span
            .retained_bytes
            .zip(span.transient_bytes)
            .and_then(|(retained, transient)| retained.checked_add(transient));
        if full_copy.is_none() && self.report().physical_domains.is_none() {
            return Err(WorkingMemoryError::UnknownBound.into());
        }
        let mut full_outside = metadata.clone_execution(&outside_without_copy)?;
        let domains_complete = self.report().physical_domains.is_some()
            && outside_without_copy.physical_domains.is_some();
        add_copy(
            &mut full_outside.state_update,
            full_copy,
            domains_complete,
            metadata,
        )?;
        let mut incremental_outside = outside_without_copy;
        add_copy(
            &mut incremental_outside.state_update,
            self.incremental_bytes(),
            domains_complete,
            metadata,
        )?;
        match self.report().physical_domains.as_ref() {
            Some(copy) => {
                if let Some(domains) = &mut full_outside.physical_domains {
                    let full = metadata.combine_domain_requirements(
                        copy.state_transient
                            .as_ref()
                            .ok_or(WorkingMemoryError::UnknownBound)?,
                        copy.retained_state
                            .as_ref()
                            .ok_or(WorkingMemoryError::UnknownBound)?,
                        true,
                    )?;
                    domains.state_update =
                        metadata.combine_domain_requirements(&domains.state_update, &full, true)?;
                }
                if let Some(domains) = &mut incremental_outside.physical_domains {
                    domains.state_update = metadata.combine_domain_requirements(
                        &domains.state_update,
                        copy.residual
                            .as_ref()
                            .ok_or(WorkingMemoryError::UnknownBound)?,
                        true,
                    )?;
                }
            }
            None => {
                full_outside.physical_domains = None;
                incremental_outside.physical_domains = None;
            }
        }
        if let Some(prepared) = prepared {
            if !prepared.pool().same_ledger(self.source().pool()) {
                return Err(WorkingMemoryError::IdentityMismatch.into());
            }
            let mut proof = prepared.compose_copied_source_metadata(
                equations,
                state,
                full_outside,
                incremental_outside,
                metadata,
            )?;
            let copy_pin =
                RegisteredStoragePin::new_metadata(self.source().registration().clone(), metadata)
                    .map_err(crate::working_memory::WorkspaceReportError::from)?;
            proof.pin = Some(match proof.pin.take() {
                Some(prepared_pin) => {
                    RegisteredStoragePin::pair_metadata(prepared_pin, copy_pin, metadata)
                        .map_err(crate::working_memory::WorkspaceReportError::from)?
                }
                None => return Err(WorkingMemoryError::IdentityMismatch.into()),
            });
            return Ok(CopyPreparationInferenceQuote { proof, copy: self });
        }
        let state = equations.refine_state_backing_metadata(state, metadata)?;
        let span_workspace = InferenceSpanWorkspace::new_metadata(
            equations.span_workspace_plan(),
            &full_outside,
            metadata,
        )
        .map_err(crate::working_memory::WorkspaceReportError::from)?;
        let full =
            equations.compose_metadata(metadata.clone_state(&state)?, full_outside, metadata)?;
        // Preserve provable full overflow and required-domain gaps even when
        // a separately charged source could make the reduced number small.
        full_requirement_with(&full, geometry, self.source().pool())?;
        let incremental = equations.compose_metadata(state, incremental_outside, metadata)?;
        let incremental_bytes =
            full_requirement_with(&incremental, geometry, self.source().pool())?;
        let incremental_requirements = full_domain_requirements(&incremental, metadata)?;
        let pin =
            RegisteredStoragePin::new_metadata(self.source().registration().clone(), metadata)
                .map_err(crate::working_memory::WorkspaceReportError::from)?;
        let proof = IncrementalInferenceQuote {
            state: super::QuoteDiagnostics::new(full, metadata)?,
            geometry,
            incremental_bytes,
            incremental_requirements,
            // The closed source discount applies only to the separate copy.
            // Future equations remain fully priced, so the named transient is
            // the exact term later replaced by retained native generations.
            equation_incremental_bytes: equations.transient().bytes(),
            pool: self.source().pool().clone(),
            pin: Some(pin),
            controller: None,
            sources: None,
            span_workspace,
            span_seal: None,
        };
        Ok(CopyPreparationInferenceQuote { proof, copy: self })
    }
}

fn add_copy(
    bound: &mut WorkspaceBound,
    copy: Option<u64>,
    domains_complete: bool,
    metadata: crate::working_memory::WorkspaceReportMetadata<'_>,
) -> Result<(), WorkspaceCopyCompositionError> {
    if matches!(bound, WorkspaceBound::Unknown { .. }) {
        return Ok(());
    }
    let diagnostic = match (&*bound, copy) {
        (WorkspaceBound::Bounded { bytes, .. }, Some(copy)) => bytes.checked_add(copy),
        _ => None,
    };
    if let Some(total) = diagnostic {
        if let WorkspaceBound::Bounded { bytes, assumptions } = bound {
            *bytes = total;
            metadata.append(
                assumptions,
                "; closed isolated-copy preparation with exact registered source association",
            )?;
        }
    } else if domains_complete {
        *bound = metadata.per_domain(format_args!(
            "closed isolated-copy requirements are resolved independently per physical domain"
        ))?;
    } else if copy.is_none() || matches!(bound, WorkspaceBound::PerDomain { .. }) {
        *bound = metadata.unknown(format_args!(
            "copy requirements lack complete physical attribution"
        ))?;
    } else {
        return Err(WorkingMemoryError::Overflow.into());
    }
    Ok(())
}

impl<K: Clone + Ord + Send + Sync + 'static> CopyPreparationInferenceQuote<K> {
    /// Opt in to original full span diagnostics before saved-source admission.
    pub fn with_span_workspace(mut self) -> Result<Self, ResidualQuoteError> {
        self.proof = self.proof.with_span_workspace()?;
        Ok(self)
    }
    /// The actual equation span plan, without exposing or replacing copy credit.
    pub fn span_workspace(&self) -> &InferenceSpanWorkspace {
        self.proof.span_workspace()
    }

    /// Seal original request controls on the same source-bound copy proof.
    /// No copied source receives credit for these new request-owned controls.
    pub fn with_span_workspace_and_text_controls(
        mut self,
        controls: PreparedTextControlWorkspace,
    ) -> Result<Self, ResidualQuoteError> {
        self.proof = self.proof.with_span_workspace_and_text_controls(controls)?;
        Ok(self)
    }

    /// Reuse actual installed capture source custody without rebuilding a pin
    /// inventory. New controls/C publication remain part of this fresh quote.
    pub fn with_saved_capture_sources(
        mut self,
        witness: &RegisteredInferenceSourceWitness,
        source: &eredu_core::capture::SharedCapturePlan,
    ) -> Result<Self, WorkingMemoryError> {
        self.proof = self.proof.with_saved_capture_sources(witness, source)?;
        Ok(self)
    }

    /// Promote the same accepted proof through its original capture publication
    /// worker. The saved copy plan retires after the proof has retained its source
    /// pin; it cannot supply new destination or publication permission.
    pub fn begin_capture_plan_publication(
        self,
        run: &WorkingMemoryFundingRun,
        reservation: &WorkingMemoryReservation,
        source: &eredu_core::capture::SharedCapturePlan,
    ) -> Result<crate::working_memory::PendingCapturePlanPublication<K>, SpanWorkspaceOwnerError>
    where
        K: crate::working_memory::CapturePlanStorageKey,
    {
        let Self { proof, copy } = self;
        let result = proof.begin_capture_plan_publication::<K>(run, reservation, source);
        drop(copy);
        result
    }

    /// Consume the accepted saved-source quote into its fresh text request.
    /// The ordinary span promotion checks the exact reservation/run and retains
    /// the registered copy-source pin. The numerical copy plan grants nothing
    /// after admission and retires outside the account lock.
    pub fn into_funded_text_span_workspace(
        self,
        run: &WorkingMemoryFundingRun,
        reservation: &WorkingMemoryReservation,
    ) -> Result<
        (
            OwnedTextSpanWorkspace,
            Option<RegisteredInferenceSourceWitness>,
        ),
        SpanWorkspaceOwnerError,
    > {
        let Self { proof, copy } = self;
        let result = proof.into_funded_text_span_workspace(run, reservation);
        drop(copy);
        result
    }

    /// Borrow the exact accepted plan without exporting copy-source authority.
    pub fn reserved_span_workspace<'a>(
        &'a self,
        reservation: &'a WorkingMemoryReservation,
    ) -> Result<ReservedInferenceSpanWorkspace<'a>, WorkingMemoryError> {
        self.proof.reserved_span_workspace(reservation)
    }
    /// Original complete diagnostics; source credit does not edit their fields.
    pub fn state(&self) -> &RuntimeStateEstimate {
        self.proof.state()
    }
    /// Exact request geometry inspected by the shared equation traversal.
    pub fn geometry(&self) -> InferenceGeometry {
        self.proof.geometry()
    }
    /// Full new demand, excluding only the fixed copy's registered old roots.
    pub fn incremental_bytes(&self) -> Option<u64> {
        self.proof.incremental_bytes()
    }
    /// Placed incremental requirement retaining exact allocation provenance.
    pub fn incremental_requirements(&self) -> Option<&eredu_core::DomainMemoryRequirements> {
        self.proof.incremental_requirements()
    }
    /// Domain retaining the independently charged old roots.
    pub fn pool(&self) -> &MemoryLedger {
        self.proof.pool()
    }

    /// Reserves the exact proved request while validating both the actual saved
    /// sampler/native owners and every credited root under the commit lock.
    /// Ordinary core policy must have produced `admission` from these unchanged
    /// full diagnostics and this incremental bound. A scalar or edited report
    /// cannot lower the minimum. Source pins survive funding and quarantine.
    pub fn reserve_saved_source_with_capacity_handoff(
        &self,
        pool: &MemoryLedger,
        execution: &InferenceExecutionIdentity,
        admission: &Admission,
        capacity: eredu_core::MemoryLimits,
        handoffs: &[WorkingMemoryCapacityHandoff],
        saved: &RegisteredSavedSamplingSource<'_, K>,
    ) -> Result<WorkingMemoryReservation, WorkingMemoryError> {
        if !pool.same_ledger(self.pool())
            || admission.state != *self.state()
            || admission.incremental_required_bytes != self.incremental_bytes()
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let joined = CopySourceValidation {
            saved,
            copy: self.copy.source().registration(),
        };
        let pin = self
            .proof
            .pin
            .clone()
            .expect("closed copy always retains its source association");
        let requirements = self
            .proof
            .incremental_requirements()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        // The future-generation proof owns its dynamic equation plan. Its
        // planning account is distinct from the sealed copied-source H/credit.
        let reservation = pool.reserve_limited_with_source_metadata(
            execution,
            admission,
            Some(capacity),
            Some((requirements, pin)),
            handoffs,
            Some(&joined),
            self.proof.metadata_funding(),
        )?;
        Ok(self.proof.attach_span_identity(reservation))
    }
}

struct CopySourceValidation<'a, 's, K: Ord + Send + 'static> {
    saved: &'a RegisteredSavedSamplingSource<'s, K>,
    copy: &'a WorkingMemoryStorage<K>,
}
impl<K: Clone + Ord + Send + Sync + 'static> SavedSourceValidation
    for CopySourceValidation<'_, '_, K>
{
    fn validate(&self, pool: &MemoryLedger, usage: &Usage) -> Result<(), WorkingMemoryError> {
        self.saved.validate(pool, usage)?;
        self.copy.validate_copy_source(pool, usage)
    }
    fn pin(&self) -> RegisteredStoragePin {
        // The copy's own pin is independently carried by the sealed residual.
        self.saved.pin()
    }
    fn pin_control_bytes(&self) -> Result<usize, WorkingMemoryError> {
        self.saved.pin_control_bytes()
    }
}
