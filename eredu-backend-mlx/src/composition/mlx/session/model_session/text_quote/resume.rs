//! Fresh incremental admission for an exact independently saved dense source.

use super::*;
use crate::backend::runtime::residency::storage::StorageIdentity;
use eredu_runtime::working_memory::WorkingMemoryStorage;
mod finish;
mod native_quote;
mod paged;
use crate::composition::mlx::session::model_session::saved_array_copy::decoder::{
    CopiedTextComponentsOwner, PreparedSavedTextResumeQuote, ResumeCapture,
};
use eredu_core::{AdmissionResult, TextStepContext};
use eredu_runtime::working_memory::{PrefillPlanningError, WorkingMemoryFundingRun};
pub(in crate::composition::mlx::session) use native_quote::seal_saved_native_quote;

/// Closed cold admission, before any history/native copy or preparation claim.
/// Its source is immutable saved data; its request and core run are newly bound.
/// No SessionPayload owner or old source request/grant is retained here.
///
/// Full source-inclusive diagnostics remain intact. Only the sealed decoder/key
/// copy program receives exact registered-root credit; its source residency
/// stays independently charged and atomically health-validated. Only the eventual checked installer may seal this pending quote.
pub(in crate::composition::mlx::session) struct PendingSavedTextAdmission {
    source: CopiedTextComponentsOwner,
    child_capture: Option<ResumeCapture>,
    sampling_change: Option<eredu_runtime::execution_control::ValidatedSamplingOverride>,
    source_native: Option<crate::backend::nn::workspace::ProjectedNativeStorage>,
    source_registration: eredu_runtime::working_memory::WorkingMemoryStorage<
        crate::backend::runtime::residency::storage::StorageIdentity,
    >,
    ordinary: Option<MlxTextPreparation>,
    context: TextStepContext,
    initial_revision: eredu_runtime::working_memory::InferenceStateRevision,
    installed: Cell<bool>,
    kind: eredu_core::OriginalTextResumeKind,
    original_exchange: Option<super::super::control_slot::PreparedControlExchange>,
    // Exact copied-manager construction only; this is not a decode-input loan.
    paged_source_metadata: Option<eredu_nn::workspace::WorkspaceContext>,
    paged_host_facts: Option<eredu_runtime::working_memory::HostSourceConstructionFacts>,
    // New constructor H is independent of the saved source's copy account.
    _host_preparation: Option<eredu_core::HostPreparationAuthority>,
}

impl PendingSavedTextAdmission {
    pub(in crate::composition::mlx::session) fn apply_child_sampling(
        &self,
        runtime: &ModelRuntime<MlxBackend<'_>>,
        state: &mut generation::MlxTextSamplingState,
    ) -> Result<(), Error> {
        let Some(change) = self.sampling_change else {
            return Ok(());
        };
        if !state
            .quote
            .as_ref()
            .is_some_and(|quote| quote.same_owner(self.quote()))
        {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        self.source.validate_resume_origin(runtime)?;
        self.quote()
            .replace_saved_sampling(runtime, state, &self.context, change)
    }
    pub(in crate::composition::mlx::session) fn resumed_has_rng(&self) -> bool {
        self.sampling_change
            .is_some_and(|change| change.reseed().is_some())
            || self.source.sampling().sampling_state_facts().has_rng
    }
    pub(in crate::composition::mlx::session) fn take_capture(
        &self,
        runtime: &ModelRuntime<MlxBackend<'_>>,
    ) -> Result<Option<super::super::text_capture::InstalledCapture>, Error> {
        match (
            self.child_capture
                .as_ref()
                .map(ResumeCapture::checkpoint)
                .or_else(|| self.source.capture_checkpoint()),
            self.quote().capture.as_ref(),
        ) {
            (Some(checkpoint), Some(capture)) => capture
                .take_saved_installed(runtime.session(), checkpoint, self.kind)
                .map(Some),
            (None, None) => Ok(None),
            _ => Err(memory(WorkingMemoryError::IdentityMismatch)),
        }
    }

    pub(in crate::composition::mlx::session) fn take_original_exchange(
        &mut self,
    ) -> Result<Option<super::super::control_slot::PreparedControlExchange>, Error> {
        if self._host_preparation.is_some() {
            self.original_exchange
                .take()
                .map(Some)
                .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))
        } else {
            Ok(None)
        }
    }

    pub(in crate::composition::mlx::session) fn host_preparation(
        &self,
    ) -> Option<&eredu_core::HostPreparationAuthority> {
        self._host_preparation.as_ref()
    }
    pub(in crate::composition::mlx::session) fn registered_source(
        &self,
    ) -> &WorkingMemoryStorage<StorageIdentity> {
        &self.source_registration
    }
    pub(in crate::composition::mlx::session) fn source_native(
        &self,
    ) -> Option<&crate::backend::nn::workspace::ProjectedNativeStorage> {
        self.source_native.as_ref()
    }
    pub(in crate::composition::mlx::session) fn source(&self) -> &CopiedTextComponentsOwner {
        &self.source
    }
    pub(in crate::composition::mlx::session) fn planning_metadata(
        &self,
    ) -> Option<&eredu_nn::workspace::HostMetadataFunding> {
        self.quote().planning_metadata.as_ref()
    }

    pub(in crate::composition::mlx::session) fn preparation(&self) -> &InferenceTextPreparation {
        self.ordinary
            .as_ref()
            .expect("unfinished saved admission")
            .request
            .as_ref()
            .expect("admitted request")
    }

    pub(in crate::composition::mlx::session) fn quote(&self) -> &TextExecutionQuoteOwner {
        self.ordinary
            .as_ref()
            .expect("unfinished saved admission")
            .quote
            .as_ref()
            .expect("admitted quote")
    }

    /// Rechecks the exact untouched installation target and final controller.
    /// No lease, stage claim, callback or native operation is acquired here.
    pub(in crate::composition::mlx::session) fn validate_before_install<
        C: TokenFilterController,
    >(
        &self,
        runtime: &ModelRuntime<MlxBackend<'_>>,
        controller: &C,
        context: &TextStepContext,
    ) -> Result<(), Error> {
        if &self.context != context || self.installed.get() {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        self.quote().opening.require_pending().map_err(memory)?;
        self.source.validate_resume_origin(runtime)?;
        self.source
            .validate_resume_account(&self.source_registration)?;
        self.quote()
            .validate_binding(runtime, self.preparation().request())?;
        let retained = runtime
            .session()
            .payload
            .model
            .erased()
            .retained_inference_authority()?;
        retained
            .validate_revision(&self.initial_revision)
            .map_err(memory)?;
        self.quote()
            .storage_contract
            .validate(controller)
            .map_err(|error| Error::Other(Box::new(error)))?;
        let workspace = controller
            .inference_workspace(self.quote().request.geometry().max_output_tokens)
            .ok_or_else(|| unknown())?;
        self.quote()
            .storage_contract
            .validate_workspace(workspace)
            .map_err(|error| Error::Other(Box::new(error)))?;
        let contract = TextControllerContract::from_workspace(
            workspace,
            self.quote().controller.output_width(),
        )
        .map_err(|error| Error::Other(Box::new(error)))?;
        if contract != self.quote().controller {
            return Err(memory(WorkingMemoryError::IdentityMismatch));
        }
        Ok(())
    }

    /// The caller has exchanged and published its exact completed copied state.
    /// This reads that installed target and seals the quote once; it never
    /// supplies copy provenance or performs the exchange itself.
    pub(in crate::composition::mlx::session) fn seal_installed(
        &self,
        runtime: &ModelRuntime<MlxBackend<'_>>,
        transition: Option<&eredu_runtime::working_memory::CopiedMediaStateBinding>,
    ) -> Result<(), Error> {
        self.quote().seal_installed_opening(runtime)?;
        // The copy, exchange and native publication are complete. Bind the
        // actual installed independent manager before exposing this run.
        self.install_paged_sources(runtime)?;
        match (
            self.source
                .sampling()
                .pending_media()
                .filter(|_| self.quote().request().geometry().max_output_tokens > 0),
            transition,
        ) {
            (Some(media), Some(transition)) => {
                self.quote()
                    .seal_copied_media(runtime, media.semantics(), transition)?
            }
            (None, None) => {}
            _ => return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch)),
        }
        self.installed.set(true);
        Ok(())
    }

    /// Finish-only transfer after successful installation and shared readiness.
    /// The enclosing private driver owns readiness/failure custody. This moves
    /// existing owners and constructs no payload or execution permission.
    pub(in crate::composition::mlx::session) fn finish(&mut self) -> MlxTextPreparation {
        assert!(
            self.installed.get(),
            "saved admission must be installed before finish"
        );
        self.ordinary.take().expect("saved admission finishes once")
    }

    // Private composition bridge: only this closed admitted owner supplies the
    // run to the prompt worker. It cannot substitute a geometry-only request.
    pub(in crate::composition::mlx::session) fn with_funding<T>(
        &self,
        worker: impl FnOnce(&WorkingMemoryFundingRun) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let funding = self
            .quote()
            .funding
            .try_borrow()
            .map_err(|_| memory(WorkingMemoryError::ExecutionFenced))?;
        worker(
            funding
                .as_ref()
                .ok_or_else(|| memory(WorkingMemoryError::IdentityMismatch))?,
        )
    }
}

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct OriginalAdmissionFailure {
    #[source]
    cause: Error,
    _host: eredu_core::HostPreparationAuthority,
}

pub(in crate::composition::mlx::session) fn original_resume_admission_control_bytes()
-> Option<usize> {
    [
        std::mem::size_of::<PendingSavedTextAdmission>(),
        std::mem::size_of::<Option<PendingSavedTextAdmission>>(),
        std::mem::size_of::<Result<PendingSavedTextAdmission, Error>>(),
        std::mem::size_of::<Result<PendingSavedTextAdmission, BackendFailure>>(),
        std::mem::size_of::<OriginalAdmissionFailure>(),
        std::mem::size_of::<Option<eredu_core::HostPreparationAuthority>>(),
        BackendFailure::source_retention_peak_bytes::<OriginalAdmissionFailure>()?,
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
}

pub(in crate::composition::mlx::session) fn admit_original_saved<C: TokenFilterController>(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    source: &CopiedTextComponentsOwner,
    config: TextGenerationConfig,
    controller: &C,
    context: &TextStepContext,
    host: &eredu_core::HostPreparationAuthority,
    options: &eredu_core::OriginalTextResumeOptions<'_>,
) -> Result<PendingSavedTextAdmission, BackendFailure> {
    admit_saved_inner(
        runtime,
        source,
        config,
        controller,
        context,
        Some(host),
        options,
    )
    .map_err(|cause| {
        BackendFailure::from_error(OriginalAdmissionFailure {
            cause,
            _host: host.clone(),
        })
    })
}

fn admit_saved_inner<C: TokenFilterController>(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    source: &CopiedTextComponentsOwner,
    config: TextGenerationConfig,
    controller: &C,
    context: &TextStepContext,
    host: Option<&eredu_core::HostPreparationAuthority>,
    options: &eredu_core::OriginalTextResumeOptions<'_>,
) -> Result<PendingSavedTextAdmission, Error> {
    let mut planning_metadata = None;
    let result = (|| {
        let original = host.is_some();
        PreparedSavedTextResumeQuote::validate_resume_options(source, config.clone(), options)
            .map_err(memory)?;
        if source.capture_checkpoint().is_some() && !original {
            return Err(unknown());
        }
        if context.attempt() != 0 {
            return Err(memory(WorkingMemoryError::TextStepOrdinalMismatch {
                expected: 0,
                actual: context.attempt(),
            }));
        }
        source.validate_resume_origin(runtime)?;
        let sampling_change = options
            .sampling
            .map(|request| {
                eredu_runtime::execution_control::validate_sampling_override::<Error>(
                    source.sampling().sampling_state_facts(),
                    request,
                )
                .map_err(|cause| Error::Other(Box::new(cause)))
            })
            .transpose()?;
        let session = runtime.session();
        session.ensure_healthy()?;
        let original_exchange = host
            .map(|host| {
                super::super::control_slot::OriginalControlExchangePlan::inspect(runtime)
                    .map_err(super::super::control_slot::error)?
                    .construct(host)
            })
            .transpose()?;
        let policy = config.inference_policy();
        if !original && policy.submission_tracking_capacity_bytes.is_some() {
            return Err(capability(
                eredu_core::CapabilityError::InvalidConfiguration {
                    field: "submission_tracking_capacity_bytes",
                    detail: "saved/copy tracking quota and role integration is not implemented"
                        .into(),
                },
            ));
        }
        if !original && policy.graph_metadata_capacity_bytes.is_some() {
            return Err(capability(
                eredu_core::CapabilityError::InvalidConfiguration {
                    field: "graph_metadata_capacity_bytes",
                    detail: "saved/copy graph quota and role integration is not implemented".into(),
                },
            ));
        }
        policy
            .validate(config.sampling().max_new_tokens)
            .map_err(capability)?;
        let capacity = policy
            .memory_limits
            .resolve(runtime.backend().memory_ledger().topology())
            .map_err(|cause| memory(cause.into()))?;
        let outputs = u64::try_from(config.sampling().max_new_tokens.ok_or_else(|| unknown())?)
            .map_err(|_| memory(WorkingMemoryError::Overflow))?;
        let workspace = controller
            .inference_workspace(outputs)
            .ok_or_else(|| unknown())?;
        let storage_contract = if original {
            ControllerStorageContract::inspect_original_retained(
                controller,
                workspace,
                runtime.backend().memory_ledger(),
                session
                    .payload
                    .model
                    .erased()
                    .inference_execution_identity(),
                outputs,
            )
            .map_err(Error::StorageSource)?
        } else {
            ControllerStorageContract::inspect(controller)
                .map_err(|error| Error::Other(Box::new(error)))?
        };
        storage_contract
            .validate_workspace(workspace)
            .map_err(|error| Error::Other(Box::new(error)))?;
        let diagnostic = match host {
            Some(host) => PreparedSavedTextResumeQuote::prepare_original(
                runtime,
                source,
                config.clone(),
                workspace,
                host,
                options,
            )?,
            None => {
                PreparedSavedTextResumeQuote::prepare(runtime, source, config.clone(), workspace)?
            }
        };
        planning_metadata = diagnostic.planning_metadata();
        // Owner assembly has large by-value transports. It begins only after
        // the recursive cold quote returns, with its own prepaid controls.
        if let Some(funding) = planning_metadata.as_ref() {
            funding
                .reserve_metadata(
                    finish::control_bytes::<C>()
                        .ok_or_else(|| memory(WorkingMemoryError::Overflow))?,
                )
                .map_err(Error::WorkspacePlanning)?;
        }
        finish::admit(
            runtime,
            source,
            config,
            controller,
            context,
            host,
            options.kind,
            sampling_change,
            diagnostic,
            workspace,
            storage_contract,
            original_exchange,
            capacity,
            outputs,
            &planning_metadata,
        )
    })();
    result.map_err(|cause| match planning_metadata {
        Some(funding) => crate::composition::mlx::model::retain_planning_error(cause, funding),
        None => cause,
    })
}
