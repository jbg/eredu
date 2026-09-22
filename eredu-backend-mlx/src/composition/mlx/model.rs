//! Architecture-erased executable and generation dispatch.

use std::path::Path;

use eredu_core::cache::{PromptCacheDescriptor, PromptCacheOptions, SharedPromptCacheManifest};
use eredu_core::{SpeculativeCapability, SpeculativeDraftSource};
use eredu_runtime::{CacheResidencyPolicy, CacheResidencyReport, layered::BoundCaptureSelection};
use safemlx::{Array, Stream, error::Exception};

mod cache_persistence;
mod loaded_helpers;
pub(in crate::composition) use loaded_helpers::settle_loaded_numerical_values;
mod recipe_planning;
pub(in crate::composition::mlx) use recipe_planning::capture::CaptureRecorder;
pub(super) use recipe_planning::{
    FundedResidentRecipePlanning, PreparedPlanningError, RecipeWorkspace,
    planning_error_has_funding, retain_planning_error, retain_planning_error_with_kind,
    retain_planning_failure,
};

use crate::backend::error::Error;
use crate::backend::runtime::media::input;
use crate::backend::runtime::residency::storage::{RetainedStorage, StorageIdentity};

/// Cold retained storage evidence for one idle executable and its enclosing
/// model/session. Decoder state stays separate because the request equation
/// quote already includes its complete backing and growth overlap.
///
/// This report grants no allocation authority and does not release an unquoted
/// owner. The caller must preserve the idle owner while consuming its evidence.
#[derive(Debug)]
pub(crate) struct RetainedIdleModelStorage {
    nonstate: RetainedStorage,
    decoder_state: RetainedStorage,
}

impl RetainedIdleModelStorage {
    pub(crate) fn validate_physical_attribution(
        &self,
        pool: &eredu_runtime::working_memory::MemoryLedger,
    ) -> Result<(), Error> {
        self.nonstate.validate_physical_attribution(pool)?;
        self.decoder_state.validate_physical_attribution(pool)
    }

    pub(crate) fn nonstate_bytes(&self) -> Result<Option<u64>, Error> {
        self.nonstate.byte_bound().map_err(Into::into)
    }

    pub(crate) fn decoder_state_bytes(&self) -> Result<Option<u64>, Error> {
        self.decoder_state.byte_bound().map_err(Into::into)
    }

    /// Certifies absence of retained decoder payload, including selected cache
    /// managers. This does not infer a zero frontier: geometry must separately
    /// match the actual state projection, including host-only position state.
    pub(crate) fn has_empty_decoder_storage(&self) -> Result<bool, Error> {
        self.decoder_state.has_no_payload()
    }

    /// Authenticates an existing decoder independently of any previous request.
    /// Imported state retains its original allocation charges; this pin grants
    /// no capacity, state succession, or native execution authority.
    pub(crate) fn pin_registered_decoder_source(
        &self,
        pool: &eredu_runtime::working_memory::MemoryLedger,
        funding: &eredu_core::HostMetadataFunding,
    ) -> Result<eredu_runtime::working_memory::WorkingMemoryStorage<StorageIdentity>, Error> {
        use eredu_core::{HostMetadataFunding, HostPreparationAuthority};
        use eredu_nn::workspace::WorkspaceMetadataAllocation;
        use std::mem::size_of;

        funding
            .reserve_metadata(
                HostPreparationAuthority::retention_bytes::<HostMetadataFunding>()
                    .and_then(|bytes| {
                        bytes.checked_add(size_of::<(
                            &Self,
                            &eredu_runtime::working_memory::MemoryLedger,
                            &HostMetadataFunding,
                            HostPreparationAuthority,
                            Option<
                                eredu_runtime::working_memory::WorkingMemoryStorage<
                                    StorageIdentity,
                                >,
                            >,
                            Result<
                                eredu_runtime::working_memory::WorkingMemoryStorage<
                                    StorageIdentity,
                                >,
                                Error,
                            >,
                        )>())
                    })
                    .ok_or(Error::PrefillControl(
                        eredu_runtime::working_memory::WorkingMemoryError::Overflow,
                    ))?,
            )
            .map_err(Error::WorkspacePlanning)?;
        let plan = self
            .decoder_state
            .source_pin_plan(pool)
            .map_err(|cause| Error::Neural(funding.metadata_source(cause)))?;
        funding
            .reserve_metadata(plan.requested_bytes())
            .map_err(Error::WorkspacePlanning)?;
        let host = HostPreparationAuthority::retain(funding.clone());
        self.decoder_state
            .pin_registered_with_host(pool, &host)
            .map_err(|cause| Error::Neural(funding.metadata_source(cause)))
    }

    /// Complete source validation for the single original whole-KV table.
    /// Decoder arrays cannot smuggle a second original fixed-table population.
    pub(crate) fn validate_original_table(
        &self,
        pool: &eredu_runtime::working_memory::MemoryLedger,
        expected: Option<&eredu_runtime::working_memory::OriginalResidentResetSource>,
    ) -> Result<(), Error> {
        self.nonstate.validate_original_table(pool, expected)?;
        self.decoder_state.validate_original_table(pool, None)
    }

    /// Returns nonstate and decoder inventories without adding their byte
    /// totals. Merge before registering a full inventory so aliases across the
    /// two roles retain one physical charge.
    pub(crate) fn into_parts(self) -> (RetainedStorage, RetainedStorage) {
        (self.nonstate, self.decoder_state)
    }
}

/// Exact source whose descriptive rows were installed on the fresh saved
/// workspace before copy/state spans. Consumed by the existing native quote;
/// no alternate source query or numerical authority is carried by this owner.
pub(in crate::composition::mlx) struct PreparedSavedParameterSource {
    pub(in crate::composition::mlx) layerwise:
        Option<crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
    pub(in crate::composition::mlx) backings:
        crate::backend::nn::workspace::ParameterWorkspaceBackings,
}

/// The single architecture-erased outer boundary for complete MLX execution.
pub(crate) struct Executable {
    inner: Box<dyn super::replicated_text::ErasedReplicatedTextExecutable>,
    parameter_sources: crate::backend::nn::workspace::CompletedParameterSources,
    inference: Option<eredu_architectures::prepared_execution::PreparedInferenceBlueprint>,
    workspace: Option<crate::backend::nn::workspace::ResidentExecutionMechanisms>,
    _storage_publication:
        Option<crate::backend::runtime::residency::storage::RetainedStoragePublication>,
    // Keep accounting alive when the executable is moved out of its model.
    _memory_owner: Option<crate::backend::managed_memory::NativeMemoryOwner>,
}

/// Borrowed selection for the existing equation program. The bound branch keeps
/// the exact retained plan/path association and candidate; neither variant owns
/// native storage, a reservation or an installed capture session.
#[derive(Clone, Copy)]
enum CaptureQuoteSource<'capture> {
    Raw(&'capture eredu_core::capture::SharedCapturePlan),
    Bound(BoundCaptureSelection<'capture>),
}

impl Executable {
    pub(super) fn new(
        inner: Box<dyn super::replicated_text::ErasedReplicatedTextExecutable>,
    ) -> Self {
        Self {
            inner,
            parameter_sources: Default::default(),
            inference: None,
            workspace: None,
            _storage_publication: None,
            _memory_owner: None,
        }
    }

    pub(crate) fn parameter_sources(
        &self,
    ) -> &crate::backend::nn::workspace::CompletedParameterSources {
        &self.parameter_sources
    }

    /// The prepared transaction swaps this alongside the immutable values.
    /// Displaced source retirement remains with the transaction after unlocking.
    pub(crate) fn exchange_parameter_sources(
        &mut self,
        sources: &mut crate::backend::nn::workspace::CompletedParameterSources,
    ) {
        std::mem::swap(&mut self.parameter_sources, sources);
    }

    pub(crate) fn retain_memory_owner(
        &mut self,
        owner: crate::backend::managed_memory::NativeMemoryOwner,
    ) -> Result<(), Error> {
        let storage = {
            // The sole caller still owns a fresh, unexposed loaded model. Its
            // own workers have no forward windows, so the ordinary entry does
            // not invert a live manager-worker lock. Release before attachment.
            let _inspection = safemlx::OrdinaryArrayMetadataGuard::enter()?;
            self.erased().retained_target_nonstate_storage()?
        };
        if storage.byte_bound()?.is_some() {
            self._storage_publication = Some(storage.publish_unquoted(&owner)?);
        }
        // Missing inventory coverage remains protected by this owner. Even a
        // complete nonstate projection cannot certify state, processors, future
        // work or independent copies, so publication never clears the barrier.
        self._memory_owner = Some(owner);
        Ok(())
    }

    /// Publishes all declared storage of a fresh idle executable before its
    /// enclosing session releases local loading handles. This transition does
    /// not authorize future allocation and cannot release descendant leases.
    /// Nonempty decoder state needs a separate request-overlap handoff.
    pub(crate) fn publish_initial_idle_storage(
        &mut self,
        enclosing: RetainedStorage,
        owner: &crate::backend::managed_memory::NativeMemoryOwner,
    ) -> Result<bool, Error> {
        if !self
            ._memory_owner
            .as_ref()
            .is_some_and(|existing| existing.same_authority(owner))
        {
            return Ok(false);
        }
        let storage = match self.retained_idle_storage(Some(enclosing)) {
            Ok(storage) => storage,
            // This optional publication never clears loading authority unless
            // its entire cold inventory succeeds. Foreign runtime contention
            // leaves that original owner in place; no retry or bound is inferred.
            Err(error) if initial_publication_inspection_busy(&error) => return Ok(false),
            Err(error) => return Err(error),
        };
        if storage.nonstate_bytes()?.is_none() || !storage.has_empty_decoder_storage()? {
            return Ok(false);
        }
        let (mut nonstate, decoder) = storage.into_parts();
        nonstate.merge(decoder)?;
        // Complete registration precedes every attachment. A failed attachment
        // leaves any published charges valid and the loading owner untouched.
        let publication = nonstate.publish_unquoted(owner)?;
        self._storage_publication = Some(publication);
        self._memory_owner = None;
        Ok(true)
    }

    /// Collects existing storage without native evaluation or state cloning.
    /// `enclosing` must cover processor, communication, parameter-overlay and
    /// other model/session payload outside this executable. Absence means
    /// unknown coverage, while an empty inventory certifies no such payload.
    /// Independent outputs, snapshots and active requests keep their own
    /// accounting authorities; this inventory cannot release those owners.
    pub(crate) fn retained_idle_storage(
        &self,
        enclosing: Option<RetainedStorage>,
    ) -> Result<RetainedIdleModelStorage, Error> {
        let mut nonstate = RetainedStorage::default();
        self.collect_retained_idle_nonstate_storage(&mut nonstate)?;
        match enclosing {
            Some(enclosing) => nonstate.merge(enclosing)?,
            None => nonstate.mark_incomplete(),
        }
        // Each retained module supplies explicit numerical coverage,
        // including nonparameter helpers. Unknown leaves propagate through the
        // inventory; complete parameter topology alone never upgrades it.
        let decoder_state = self.erased().retained_decoder_state_storage()?;
        Ok(RetainedIdleModelStorage {
            nonstate,
            decoder_state,
        })
    }

    /// Fills separately supplied nonstate and decoder inventories while idle.
    /// The caller covers enclosing session owners in the nonstate destination.
    pub(crate) fn collect_retained_idle_storage(
        &self,
        nonstate: &mut RetainedStorage,
        decoder: &mut RetainedStorage,
    ) -> Result<(), Error> {
        self.collect_retained_idle_nonstate_storage(nonstate)?;
        self.erased()
            .collect_retained_decoder_state_storage(decoder)
    }

    fn collect_retained_idle_nonstate_storage(
        &self,
        nonstate: &mut RetainedStorage,
    ) -> Result<(), Error> {
        self.erased()
            .collect_retained_target_nonstate_storage(nonstate)?;
        self.erased()
            .collect_retained_prediction_storage(nonstate)?;
        self.erased()
            .collect_retained_idle_auxiliary_storage(nonstate)?;
        if let Some(blueprint) = &self.inference {
            if !nonstate.requires_borrowed_source_storage() {
                nonstate.include_sources(blueprint.source_storage()?)?;
                return Ok(());
            }
            let mut failure = None;
            let complete = blueprint.visit_source_storage(&mut |source| {
                if failure.is_none() {
                    failure = nonstate.include_source_ref(source).err();
                }
            });
            if let Some(error) = failure {
                return Err(error.into());
            }
            if !complete? {
                nonstate.mark_incomplete();
            }
        }
        Ok(())
    }

    pub(super) fn with_inference_blueprint(
        mut self,
        inference: eredu_architectures::prepared_execution::PreparedInferenceBlueprint,
        workspace: Option<crate::backend::nn::workspace::ResidentExecutionMechanisms>,
    ) -> Self {
        self.inference = Some(inference);
        self.workspace = workspace;
        self
    }

    pub(crate) fn prepare_autoregressive_media_semantics(
        &mut self,
        source: &eredu_runtime::working_memory::OriginalPreparedHostInput,
        cache: &mut super::replicated_text::MlxPredictionTargetState,
        pool: &eredu_runtime::working_memory::MemoryLedger,
        funding: &eredu_nn::workspace::HostMetadataFunding,
        stream: &safemlx::Stream,
    ) -> Result<eredu_architectures::media_plan::BoundPreparedMediaSemantics, Error> {
        let blueprint = self
            .inference
            .as_ref()
            .ok_or(Error::PrefillScopeUnavailable)?;
        self.inner
            .prepare_autoregressive_media_semantics(source, cache, blueprint, pool, funding, stream)
    }

    pub(crate) fn inference_blueprint(
        &self,
    ) -> Option<&eredu_architectures::prepared_execution::PreparedInferenceBlueprint> {
        self.inference.as_ref()
    }

    /// Complete initial publication, after the executable's loading authority
    /// has retired. A target-only publication under that authority is not enough.
    pub(crate) fn has_published_idle_storage(&self) -> bool {
        self._storage_publication.is_some() && self._memory_owner.is_none()
    }

    pub(crate) fn native_storage_mechanism(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::storage::native_storage::MlxNativeStorage>,
        Error,
    > {
        Ok(self.erased().native_storage_mechanism()?.map(|mechanism| {
            let mechanism = mechanism.with_parameter_sources(self.parameter_sources.clone());
            match self
                ._storage_publication
                .as_ref()
                .filter(|_| self._memory_owner.is_none())
            {
                Some(publication) => mechanism.with_initial_publication(publication.clone()),
                None => mechanism,
            }
        }))
    }

    /// Borrows the actual publication retained by this executable, not an
    /// inventory estimate or a new registration. The source remains installed
    /// when a reset retires a later enclosing-session publication.
    pub(crate) fn covers_nonstate_publication(
        &self,
        publication: &crate::backend::runtime::residency::storage::RetainedStoragePublication,
    ) -> bool {
        self._memory_owner.is_none()
            && self
                ._storage_publication
                .as_ref()
                .is_some_and(|retained| publication.can_retire_with(retained))
    }

    pub(crate) fn has_workspace_mechanisms(&self) -> bool {
        self.workspace.is_some()
    }

    pub(crate) fn resident_workspace_mechanisms(
        &self,
    ) -> Option<crate::backend::nn::workspace::ResidentExecutionMechanisms> {
        self.workspace
    }

    /// Shared allocator/native host facts for source-bound copy and numerical
    /// programs, which separately select their actual stream. Equation contexts
    /// and recorders must consume resident_workspace_mechanisms instead.
    pub(crate) fn workspace_mechanisms(
        &self,
    ) -> Option<crate::backend::nn::workspace::MlxMetalWorkspaceMechanisms> {
        self.workspace
            .map(crate::backend::nn::workspace::ResidentExecutionMechanisms::ordinary)
    }

    pub(crate) fn layerwise_workspace(
        &self,
    ) -> Result<Option<crate::backend::runtime::execution::generic::LayerwiseWorkspace>, Error>
    {
        let selected = self
            .inference_blueprint()
            .ok_or_else(|| {
                Error::Other(Box::new(
                    eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
                ))
            })?
            .selected();
        match selected.text_realization().residency() {
            eredu_runtime::LayerWeightResidency::FullyResident => Ok(None),
            eredu_runtime::LayerWeightResidency::LayerwiseHost(_)
            | eredu_runtime::LayerWeightResidency::DenseDiskStream(_) => {
                let facts = self.workspace.ok_or_else(|| {
                    Error::Other(Box::new(
                        eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
                    ))
                })?;
                self.erased()
                    .layerwise_workspace(facts.ordinary_storage().allocation())
                    .map(Some)
            }
            _ => Err(Error::Other(Box::new(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            ))),
        }
    }

    pub(crate) fn prepared_layerwise_workspace(
        &self,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Option<crate::backend::runtime::execution::generic::LayerwiseWorkspace>, Error>
    {
        let unknown = || {
            Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::UnknownBound)
        };
        let selected = self.inference_blueprint().ok_or_else(unknown)?.selected();
        match selected.text_realization().residency() {
            eredu_runtime::LayerWeightResidency::FullyResident => Ok(None),
            eredu_runtime::LayerWeightResidency::LayerwiseHost(_)
            | eredu_runtime::LayerWeightResidency::DenseDiskStream(_) => self
                .erased()
                .prepared_layerwise_workspace(
                    self.workspace.ok_or_else(unknown)?.allocation(),
                    context,
                )
                .map(Some),
            _ => Err(unknown()),
        }
    }

    pub(crate) fn quote_replicated_resident_text(
        &self,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<eredu_runtime::working_memory::InferenceWorkspaceReport, Error> {
        let blueprint = self.inference_blueprint().ok_or_else(|| {
            Error::ArchitectureModel("executable has no retained inference blueprint".into())
        })?;
        let batch = u32::try_from(geometry.batch_size)
            .ok()
            .and_then(std::num::NonZeroU32::new)
            .ok_or_else(|| {
                Error::ArchitectureModel("workspace batch exceeds native tensor extent".into())
            })?;
        let facts = self.workspace.ok_or_else(|| {
            Error::ArchitectureModel(
                "selected native execution has no retained workspace mechanism facts".into(),
            )
        })?;
        let context = eredu_nn::workspace::WorkspaceContext::new(facts.ordinary_storage());
        self.erased()
            .install_workspace_parameter_representations(&context)
            .map_err(|cause| Error::Other(Box::new(cause)))?;
        let state = self.erased().project_resident_workspace(batch, &context)?;
        blueprint
            .quote_replicated_resident_text(geometry, &state, &context)
            .map_err(|error| Error::Other(Box::new(error)))
    }

    pub(crate) fn erased(
        &self,
    ) -> &(dyn super::replicated_text::ErasedReplicatedTextExecutable + 'static) {
        self.inner.as_ref()
    }

    pub(crate) fn quote_replicated_resident_text_with_sampling<'a>(
        &self,
        geometry: eredu_core::InferenceGeometry,
        config: eredu_core::TextGenerationConfig,
        filter: impl Into<eredu_core::TextFilterWorkspace<'a>>,
    ) -> Result<eredu_architectures::prepared_execution::PreparedTextGenerationWorkspace, Error>
    {
        self.quote_resident_sampling_impl(geometry, config, filter.into(), None, None)
            .map(|(quote, _)| quote)
    }

    /// Additive cold capture equations. H is returned separately and has not
    /// been admitted, held or included a second time inside numerical spans.
    pub(crate) fn quote_replicated_resident_text_with_sampling_and_capture<'a, 'capture>(
        &self,
        geometry: eredu_core::InferenceGeometry,
        config: eredu_core::TextGenerationConfig,
        filter: impl Into<eredu_core::TextFilterWorkspace<'a>>,
        capture: &'capture eredu_core::capture::SharedCapturePlan,
    ) -> Result<
        (
            eredu_architectures::prepared_execution::PreparedTextGenerationWorkspace,
            eredu_runtime::working_memory::CaptureRunHostPlan<'capture>,
        ),
        Error,
    > {
        let (quote, host) = self.quote_resident_sampling_impl(
            geometry,
            config,
            filter.into(),
            Some(CaptureQuoteSource::Raw(capture)),
            None,
        )?;
        Ok((
            quote,
            host.expect("capture input creates its closed host plan"),
        ))
    }

    /// Quotes an already bound prefill selection through the same resident or
    /// layerwise equation program. The returned H plan borrows the companion's
    /// exact source; admission, source publication and installation remain with
    /// the caller. A different candidate or path owner is rejected before work.
    pub(crate) fn quote_replicated_resident_text_with_sampling_and_prefill_capture<'a, 'capture>(
        &self,
        geometry: eredu_core::InferenceGeometry,
        config: eredu_core::TextGenerationConfig,
        filter: impl Into<eredu_core::TextFilterWorkspace<'a>>,
        capture: BoundCaptureSelection<'capture>,
        publications: Option<&std::cell::Cell<usize>>,
    ) -> Result<
        (
            eredu_architectures::prepared_execution::PreparedTextGenerationWorkspace,
            eredu_runtime::working_memory::CaptureRunHostPlan<'capture>,
        ),
        Error,
    > {
        let (quote, host) = self.quote_resident_sampling_impl(
            geometry,
            config,
            filter.into(),
            Some(CaptureQuoteSource::Bound(capture)),
            publications,
        )?;
        Ok((
            quote,
            host.expect("bound capture creates its closed host plan"),
        ))
    }

    fn quote_resident_sampling_impl<'a, 'capture>(
        &self,
        geometry: eredu_core::InferenceGeometry,
        config: eredu_core::TextGenerationConfig,
        filter: eredu_core::TextFilterWorkspace<'a>,
        capture: Option<CaptureQuoteSource<'capture>>,
        publications: Option<&std::cell::Cell<usize>>,
    ) -> Result<
        (
            eredu_architectures::prepared_execution::PreparedTextGenerationWorkspace,
            Option<eredu_runtime::working_memory::CaptureRunHostPlan<'capture>>,
        ),
        Error,
    > {
        self.validate_capture_candidate(geometry, capture)?;
        // Preserve ordinary prerequisite/error ordering before state projection.
        self.inference_blueprint().ok_or_else(|| {
            Error::ArchitectureModel("executable has no retained inference blueprint".into())
        })?;
        let batch = u32::try_from(geometry.batch_size)
            .ok()
            .and_then(std::num::NonZeroU32::new)
            .ok_or_else(|| {
                Error::ArchitectureModel("workspace batch exceeds native tensor extent".into())
            })?;
        let facts = self.workspace.ok_or_else(|| {
            Error::ArchitectureModel(
                "selected native execution has no retained workspace mechanism facts".into(),
            )
        })?;
        let context = eredu_nn::workspace::WorkspaceContext::new(facts.ordinary_storage());
        let layerwise = self.layerwise_workspace()?;
        self.install_parameter_source(&context, layerwise.as_ref())
            .map_err(|cause| Error::Other(Box::new(cause)))?;
        let state = self.erased().project_resident_workspace(batch, &context)?;
        self.quote_sampling_program(
            geometry,
            &state,
            &context,
            config,
            filter,
            capture,
            layerwise.as_ref(),
            publications,
        )
    }

    /// Quotes an actual supplied saved-state projection and populated sampler
    /// through the retained selected resident or layerwise equation driver.
    /// The returned temporary workspace owns the exact ready host sources or
    /// direct-read plan used for parameter projection and materialization. Its
    /// caller must retain/pin those sources and the matching receipt at admission.
    /// This creates no request, native work or mutable-state copy.
    pub(crate) fn quote_replicated_text_with_existing_sampling(
        &self,
        geometry: eredu_core::InferenceGeometry,
        state: &eredu_runtime::DeviceState<
            eredu_nn::workspace::WorkspaceBackend,
            eredu_runtime::working_memory::WorkspaceResidentLayerState,
        >,
        context: &eredu_nn::workspace::WorkspaceContext,
        sampling: eredu_architectures::prepared_execution::BorrowedTextSamplingWorkspace<'_>,
    ) -> Result<
        (
            eredu_architectures::prepared_execution::PreparedTextGenerationWorkspace,
            Option<crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
        ),
        Error,
    > {
        let blueprint = self.inference_blueprint().ok_or_else(|| {
            Error::ArchitectureModel("executable has no retained inference blueprint".into())
        })?;
        let layerwise = self.layerwise_workspace()?;
        let generation = match layerwise.as_ref() {
            Some(workspace) => blueprint.quote_replicated_layerwise_text_with_existing_sampling(
                geometry,
                state,
                context,
                sampling,
                &NativeLayerwiseParameters(workspace),
            ),
            None => blueprint.quote_replicated_resident_text_with_existing_sampling(
                geometry, state, context, sampling,
            ),
        }
        .map_err(|error| Error::Other(Box::new(error)))?;
        Ok((generation, layerwise))
    }

    /// Installs actual static and prepared unit representations before the
    /// saved workspace starts copying state. The returned owner remains with
    /// that preparation until its original native recipe takes the same source.
    pub(in crate::composition::mlx) fn prepare_saved_parameter_source(
        &self,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<PreparedSavedParameterSource, Error> {
        context
            .charge_metadata(std::mem::size_of::<(
                PreparedSavedParameterSource,
                Result<PreparedSavedParameterSource, Error>,
                Option<crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
                &Self,
                &eredu_nn::workspace::WorkspaceContext,
            )>())
            .map_err(|cause| recipe_planning::context_source(context, cause))?;
        let layerwise = self.prepared_layerwise_workspace(context)?;
        let backings = self
            .install_parameter_source(context, layerwise.as_ref())
            .map_err(|cause| recipe_planning::context_source(context, cause))?;
        Ok(PreparedSavedParameterSource {
            layerwise,
            backings,
        })
    }

    /// Installs static and actual retained replacement rows into one allocation
    /// map before state projection or equation spans begin.
    pub(in crate::composition::mlx) fn install_parameter_source(
        &self,
        context: &eredu_nn::workspace::WorkspaceContext,
        layerwise: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
    ) -> Result<crate::backend::nn::workspace::ParameterWorkspaceBackings, eredu_nn::Error> {
        let mut backings = self
            .erased()
            .install_workspace_parameter_representations(context)?;
        if let Some(layerwise) = layerwise {
            layerwise.install_replacement_parameter_representations(context, &mut backings)?;
        }
        backings.classify_completed(&self.parameter_sources, context)?;
        Ok(backings)
    }

    /// Native recipe of the exact saved state and populated sampler, observed
    /// through the same architecture equation/binding driver as ordinary resume.
    /// The caller keeps source pins and admits the resulting fresh request.
    pub(in crate::composition::mlx) fn quote_replicated_text_with_existing_sampling_recipe(
        &self,
        geometry: eredu_core::InferenceGeometry,
        state: &eredu_runtime::DeviceState<
            eredu_nn::workspace::WorkspaceBackend,
            eredu_runtime::working_memory::WorkspaceResidentLayerState,
        >,
        context: &eredu_nn::workspace::WorkspaceContext,
        sampling: eredu_architectures::prepared_execution::BorrowedTextSamplingWorkspace<'_>,
        pool: &eredu_runtime::working_memory::MemoryLedger,
        capture: Option<(
            &eredu_runtime::capture::FundedCaptureCheckpoint,
            &eredu_runtime::layered::PreparedCaptureSelection,
        )>,
        addressable: Option<&crate::backend::nn::workspace::AddressableSources>,
        interventions: Option<
            crate::composition::mlx::session::intervention::TextInterventionQuote<'_>,
        >,
        prepared_parameters: Option<PreparedSavedParameterSource>,
    ) -> Result<
        (
            eredu_architectures::prepared_execution::PreparedTextGenerationWorkspace,
            Option<crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
            crate::backend::nn::workspace::ResidentNativeRecipe,
        ),
        Error,
    > {
        let blueprint = self.inference_blueprint().ok_or(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
        ))?;
        let facts = self.workspace.ok_or(Error::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
        ))?;
        let layerwise = prepared_parameters
            .ok_or_else(|| {
                recipe_planning::context_source(
                    context,
                    eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                )
            })?
            .layerwise;
        // The capture trace consumes the retained source at ordinary unit
        // binding. Replaced constructor nodes are authenticated here and priced
        // once with the source-copy recipe after numerical quotation.
        let mut recorder = facts
            .recorder(geometry, context)
            .map_err(|cause| recipe_planning::context_source(context, cause))?;
        if let Some(source) = addressable {
            recorder
                .bind_addressable_sources(source.clone())
                .map_err(|cause| recipe_planning::context_source(context, cause))?;
        }
        context
            .charge_metadata(std::mem::size_of::<(
                Option<PreparedSavedParameterSource>,
                Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
                Option<NativeLayerwiseParameters<'_>>,
                Option<crate::composition::mlx::session::intervention::TextInterventionQuote<'_>>,
                Option<&dyn eredu_architectures::prepared_execution::WorkspaceLayerwiseParameters>,
            )>())
            .map_err(|cause| recipe_planning::context_source(context, cause))?;
        if let Some(source) = layerwise.as_ref() {
            recorder
                .bind_layerwise_span_constructor_source(source)
                .map_err(|cause| recipe_planning::context_source(context, cause))?;
        }
        let parameters = layerwise.as_ref().map(NativeLayerwiseParameters);
        if capture.is_none() && interventions.is_some() {
            return Err(recipe_planning::context_source(
                context,
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ));
        }
        let generation = match capture {
            Some((checkpoint, selection)) => recipe_planning::capture::quote_saved(
                self,
                blueprint,
                geometry,
                state,
                context,
                sampling,
                checkpoint,
                selection,
                interventions,
                &mut recorder,
                parameters.as_ref().map(|source| {
                    source as
                    &dyn eredu_architectures::prepared_execution::WorkspaceLayerwiseParameters
                }),
            ),
            None => blueprint.quote_replicated_text_with_existing_sampling_and_trace(
                geometry,
                state,
                context,
                sampling,
                parameters.as_ref().map(|source| {
                    source as
                    &dyn eredu_architectures::prepared_execution::WorkspaceLayerwiseParameters
                }),
                &mut recorder,
            ),
        }
        .map_err(|error| recipe_planning::context_source(context, error))?;
        let mut recipe = recorder
            .finish(generation.equations.span_workspace_plan())
            .map_err(|error| recipe_planning::context_source(context, error))?;
        self.erased().bind_layerwise_neural_recipe(
            pool,
            &mut recipe,
            context.metadata_funding().as_ref(),
        )?;
        if let Some(sources) = &layerwise {
            sources.bind_native_host_copies(&mut recipe)?;
        }
        Ok((generation, layerwise, recipe))
    }

    /// Binds exact registered opening storage before quoting the same retained
    /// architecture equations. Temporary native witnesses remain here until
    /// inspection finishes; the returned pin contains accounting metadata only.
    pub(crate) fn quote_registered_resident_text_with_sampling<'a>(
        &self,
        geometry: eredu_core::InferenceGeometry,
        config: eredu_core::TextGenerationConfig,
        filter: impl Into<eredu_core::TextFilterWorkspace<'a>>,
        pool: &eredu_runtime::working_memory::MemoryLedger,
    ) -> Result<
        (
            eredu_architectures::prepared_execution::PreparedTextGenerationWorkspace,
            eredu_runtime::working_memory::RegisteredWorkspaceStorage<StorageIdentity>,
        ),
        Error,
    > {
        self.quote_registered_sampling_impl(geometry, config, filter.into(), pool, None, None)
            .map(|(quote, storage, _)| (quote, storage))
    }

    /// Same exact registered decoder roots and selected materialization route,
    /// with additional capture equations in that very context. No source pin
    /// for the capture/path owners, physical admission or installation is made.
    pub(crate) fn quote_registered_resident_text_with_sampling_and_capture<'a, 'capture>(
        &self,
        geometry: eredu_core::InferenceGeometry,
        config: eredu_core::TextGenerationConfig,
        filter: impl Into<eredu_core::TextFilterWorkspace<'a>>,
        pool: &eredu_runtime::working_memory::MemoryLedger,
        capture: &'capture eredu_core::capture::SharedCapturePlan,
    ) -> Result<
        (
            eredu_architectures::prepared_execution::PreparedTextGenerationWorkspace,
            eredu_runtime::working_memory::RegisteredWorkspaceStorage<StorageIdentity>,
            eredu_runtime::working_memory::CaptureRunHostPlan<'capture>,
        ),
        Error,
    > {
        let (quote, storage, host) = self.quote_registered_sampling_impl(
            geometry,
            config,
            filter.into(),
            pool,
            Some(CaptureQuoteSource::Raw(capture)),
            None,
        )?;
        Ok((
            quote,
            storage,
            host.expect("capture input creates its closed host plan"),
        ))
    }

    /// Registered opening-state version of the bound prefill route. The source
    /// checks do not register the capture/path owners or grant native capacity.
    pub(crate) fn quote_registered_resident_text_with_sampling_and_prefill_capture<'a, 'capture>(
        &self,
        geometry: eredu_core::InferenceGeometry,
        config: eredu_core::TextGenerationConfig,
        filter: impl Into<eredu_core::TextFilterWorkspace<'a>>,
        pool: &eredu_runtime::working_memory::MemoryLedger,
        capture: BoundCaptureSelection<'capture>,
        publications: Option<&std::cell::Cell<usize>>,
    ) -> Result<
        (
            eredu_architectures::prepared_execution::PreparedTextGenerationWorkspace,
            eredu_runtime::working_memory::RegisteredWorkspaceStorage<StorageIdentity>,
            eredu_runtime::working_memory::CaptureRunHostPlan<'capture>,
        ),
        Error,
    > {
        let (quote, storage, host) = self.quote_registered_sampling_impl(
            geometry,
            config,
            filter.into(),
            pool,
            Some(CaptureQuoteSource::Bound(capture)),
            publications,
        )?;
        Ok((
            quote,
            storage,
            host.expect("bound capture creates its closed host plan"),
        ))
    }

    fn validate_capture_candidate(
        &self,
        geometry: eredu_core::InferenceGeometry,
        capture: Option<CaptureQuoteSource<'_>>,
    ) -> Result<(), Error> {
        let Some(CaptureQuoteSource::Bound(bound)) = capture else {
            return Ok(());
        };
        if bound.geometry() != geometry {
            return Err(Error::Other(Box::new(
                eredu_runtime::layered::PreparedCaptureSelectionError::Identity,
            )));
        }
        let paths = self.erased().shared_observation_paths().ok_or_else(|| {
            Error::Other(Box::new(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            ))
        })?;
        bound
            .selection()
            .validate_sources(bound.selection().source(), paths)
            .map_err(|error| Error::Other(Box::new(error)))
    }

    fn quote_registered_sampling_impl<'a, 'capture>(
        &self,
        geometry: eredu_core::InferenceGeometry,
        config: eredu_core::TextGenerationConfig,
        filter: eredu_core::TextFilterWorkspace<'a>,
        pool: &eredu_runtime::working_memory::MemoryLedger,
        capture: Option<CaptureQuoteSource<'capture>>,
        publications: Option<&std::cell::Cell<usize>>,
    ) -> Result<
        (
            eredu_architectures::prepared_execution::PreparedTextGenerationWorkspace,
            eredu_runtime::working_memory::RegisteredWorkspaceStorage<StorageIdentity>,
            Option<eredu_runtime::working_memory::CaptureRunHostPlan<'capture>>,
        ),
        Error,
    > {
        self.validate_capture_candidate(geometry, capture)?;
        // Preserve ordinary prerequisite/error ordering before state projection.
        self.inference_blueprint().ok_or_else(|| {
            Error::ArchitectureModel("executable has no retained inference blueprint".into())
        })?;
        let batch = u32::try_from(geometry.batch_size)
            .ok()
            .and_then(std::num::NonZeroU32::new)
            .ok_or_else(|| {
                Error::ArchitectureModel("workspace batch exceeds native tensor extent".into())
            })?;
        let facts = self.workspace.ok_or_else(|| {
            Error::ArchitectureModel(
                "selected native execution has no retained workspace mechanism facts".into(),
            )
        })?;
        let funding = pool
            .prepare_workspace_metadata(
                self.erased().inference_execution_identity(),
                pool.configured_limits().clone(),
            )
            .map_err(Error::WorkspacePlanning)?;
        let context = eredu_nn::workspace::WorkspaceContext::new_with_metadata_funding(
            facts.ordinary_storage(),
            funding,
        )
        .map_err(|cause| Error::Neural(cause.into()))?;
        let layerwise = self.layerwise_workspace()?;
        let parameter_backings = self
            .install_parameter_source(&context, layerwise.as_ref())
            .map_err(|cause| Error::Other(Box::new(cause)))?;
        let projected = self
            .erased()
            .project_resident_workspace_with_storage(batch, &context)?;
        if !projected.storage.is_complete() {
            return Err(Error::Other(Box::new(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            )));
        }
        let roots = || {
            projected
                .storage
                .iter()
                .filter(|(_, bytes, _)| *bytes != 0)
                .map(|(identity, _, root)| {
                    crate::backend::nn::workspace::registered_storage_row(identity, root)
                })
                .chain(parameter_backings.roots())
        };
        let completed = parameter_backings
            .completed_source(&context)
            .map_err(Error::Neural)?;
        let count = projected
            .storage
            .iter()
            .filter(|(_, bytes, _)| *bytes != 0)
            .count()
            .checked_add(parameter_backings.len())
            .ok_or(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::Overflow,
            ))?;
        let layout = match &completed {
            Some(source) => eredu_runtime::working_memory::RegisteredWorkspaceStorageLayout::<
                StorageIdentity,
            >::new_with_completed_source(count, source),
            None => eredu_runtime::working_memory::RegisteredWorkspaceStorageLayout::<
                StorageIdentity,
            >::new(count),
        }
        .map_err(Error::PrefillControl)?;
        context
            .charge_metadata(layout.requested_bytes())
            .map_err(|cause| Error::Neural(cause.into()))?;
        let storage = match completed {
            Some(source) => layout.construct_with_completed_source(pool, &context, roots(), source),
            None => layout.construct(pool, &context, roots()),
        }
        .map_err(|error| Error::Other(Box::new(error)))?;
        let (quote, host) = self.quote_sampling_program(
            geometry,
            &projected.state,
            &context,
            config,
            filter,
            capture,
            layerwise.as_ref(),
            publications,
        )?;
        Ok((quote, storage, host))
    }

    fn quote_sampling_program<'a, 'capture>(
        &self,
        geometry: eredu_core::InferenceGeometry,
        state: &eredu_runtime::DeviceState<
            eredu_nn::workspace::WorkspaceBackend,
            eredu_runtime::working_memory::WorkspaceResidentLayerState,
        >,
        context: &eredu_nn::workspace::WorkspaceContext,
        config: eredu_core::TextGenerationConfig,
        filter: eredu_core::TextFilterWorkspace<'a>,
        capture: Option<CaptureQuoteSource<'capture>>,
        layerwise: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
        publications: Option<&std::cell::Cell<usize>>,
    ) -> Result<
        (
            eredu_architectures::prepared_execution::PreparedTextGenerationWorkspace,
            Option<eredu_runtime::working_memory::CaptureRunHostPlan<'capture>>,
        ),
        Error,
    > {
        let blueprint = self.inference_blueprint().ok_or_else(|| {
            Error::ArchitectureModel("executable has no retained inference blueprint".into())
        })?;
        if let Some(capture) = capture {
            let paths = self.erased().shared_observation_paths().ok_or_else(|| {
                Error::Other(Box::new(
                    eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
                ))
            })?;
            use super::session::capture_workspace::CaptureWorkspaceObserver;
            let counter = std::cell::Cell::new(
                crate::backend::array_copy::CaptureNativePopulation::default(),
            );
            let (observer, host) = match capture {
                CaptureQuoteSource::Raw(source) => {
                    CaptureWorkspaceObserver::new(source, geometry, context)
                }
                CaptureQuoteSource::Bound(bound) => {
                    CaptureWorkspaceObserver::with_prefill(bound, context)
                }
            }
            .map_err(|e| Error::Other(Box::new(e)))?;
            let mut observer = if publications.is_some() {
                observer.with_transfer_counter(&counter)
            } else {
                observer
            };
            let quote = match layerwise {
                Some(parameters) => blueprint
                    .quote_replicated_layerwise_text_with_sampling_observed(
                        geometry,
                        state,
                        context,
                        config,
                        filter,
                        &NativeLayerwiseParameters(parameters),
                        paths,
                        &mut observer,
                    ),
                None => blueprint.quote_replicated_resident_text_with_sampling_observed(
                    geometry,
                    state,
                    context,
                    config,
                    filter,
                    paths,
                    &mut observer,
                ),
            }
            .map_err(|e| Error::Other(Box::new(e)))?;
            if let Some(publications) = publications {
                publications.set(observer.publication_count().map_err(Error::Neural)?);
            }
            Ok((quote, Some(host)))
        } else {
            let quote = match layerwise {
                Some(parameters) => blueprint.quote_replicated_layerwise_text_with_sampling(
                    geometry,
                    state,
                    context,
                    config,
                    filter,
                    &NativeLayerwiseParameters(parameters),
                ),
                None => blueprint.quote_replicated_resident_text_with_sampling(
                    geometry, state, context, config, filter,
                ),
            }
            .map_err(|e| Error::Other(Box::new(e)))?;
            Ok((quote, None))
        }
    }

    pub(crate) fn quote_original_token_prompt_workspace(
        &self,
        geometry: eredu_core::InferenceGeometry,
        input: &eredu_runtime::working_memory::OriginalTokenInputLayout,
    ) -> Result<eredu_runtime::working_memory::TextPromptWorkspaceReport, Error> {
        let facts = self.workspace.ok_or_else(|| {
            Error::ArchitectureModel(
                "selected native execution has no retained workspace mechanism facts".into(),
            )
        })?;
        eredu_runtime::working_memory::quote_original_token_prompt_workspace(
            geometry,
            input,
            &eredu_nn::workspace::WorkspaceContext::new(facts),
        )
        .map_err(|error| Error::Other(Box::new(error)))
    }

    pub(crate) fn quote_text_prompt_workspace(
        &self,
        geometry: eredu_core::InferenceGeometry,
        source_capacity_bytes: Option<u64>,
    ) -> Result<eredu_runtime::working_memory::TextPromptWorkspaceReport, Error> {
        let facts = self.workspace.ok_or_else(|| {
            Error::ArchitectureModel(
                "selected native execution has no retained workspace mechanism facts".into(),
            )
        })?;
        eredu_runtime::working_memory::quote_text_prompt_workspace(
            geometry,
            source_capacity_bytes,
            &eredu_nn::workspace::WorkspaceContext::new(facts.ordinary_storage()),
        )
        .map_err(|error| Error::Other(Box::new(error)))
    }

    pub(crate) fn erased_mut(
        &mut self,
    ) -> &mut (dyn super::replicated_text::ErasedReplicatedTextExecutable + 'static) {
        self.inner.as_mut()
    }

    /// Cold composition for fresh host-token resident generation. Existing
    /// weights/sources and the enclosing domain remain separately registered;
    /// callers must supply every resource absent from the prepared trace.
    pub(crate) fn quote_text_preparation_workspace(
        &self,
        geometry: eredu_core::InferenceGeometry,
        source_capacity_bytes: Option<u64>,
        config: eredu_core::TextGenerationConfig,
        controller: eredu_core::TextControllerWorkspace<'_>,
        outside: eredu_core::ExecutionWorkspaceEstimate,
    ) -> Result<eredu_core::RuntimeStateEstimate, Error> {
        let invalid = |error| Error::Other(Box::new(error));
        geometry.validate().map_err(invalid)?;
        if config
            .sampling()
            .max_new_tokens
            .and_then(|count| u64::try_from(count).ok())
            != Some(geometry.max_output_tokens)
        {
            return Err(invalid(eredu_core::CapabilityError::InvalidConfiguration {
                field: "text_preparation_workspace",
                detail: "sampling allowance differs from the quoted request".into(),
            }));
        }
        let blueprint = self.inference_blueprint().ok_or_else(|| {
            Error::ArchitectureModel("executable has no retained inference blueprint".into())
        })?;
        let dtype = blueprint
            .selected()
            .text_realization()
            .state()
            .floating_dtype()
            .unwrap_or(eredu_runtime::StateStorageDtype::F32);
        let state = eredu_core::estimate_runtime_state(
            self.erased().capability_estimate().state_layout(),
            eredu_core::InputTokenCount::text(geometry.cached_positions + geometry.input_positions),
            geometry.max_output_tokens,
            geometry.batch_size,
            dtype.bytes(),
        )
        .map_err(invalid)?;
        let generation =
            self.quote_replicated_resident_text_with_sampling(geometry, config, controller.filter)?;
        let prompt = self.quote_text_prompt_workspace(geometry, source_capacity_bytes)?;
        generation
            .compose_preparation(state, &prompt, controller, outside)
            .map_err(invalid)
    }

    pub(crate) fn install_embedded_prediction_observers(
        &mut self,
        observers: eredu_architectures::speculative_execution::EmbeddedPredictionObservers<
            crate::MlxTensor,
            Array,
            Error,
        >,
    ) -> bool {
        self.erased_mut().install_embedded_prediction_observers(
            super::prepared_speculative::embedded_logits::observers(observers),
        )
    }

    pub(crate) fn has_neutral_partitioned_control(&self) -> bool {
        self.erased().has_partition_control()
    }

    pub(crate) fn reset_cache_distributed(&mut self) -> Result<(), Exception> {
        self.erased_mut()
            .reset_cache_distributed()
            .map_err(|error| Exception::custom(error.to_string()))
    }

    pub(crate) fn load_prompt_cache_distributed(
        &mut self,
        funding: &eredu_runtime::cache::PromptCachePersistenceFunding,
        materialization: &crate::backend::runtime::cache::residency::PromptCacheMaterialization,
        control: Option<&crate::backend::runtime::distributed::topology::original_source::control::cache::CacheControlOwner>,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
    ) -> Result<Option<SharedPromptCacheManifest>, Error> {
        self.erased_mut().load_prompt_cache_distributed(
            funding,
            materialization,
            control,
            directory,
            expected,
            prefix_token_ids,
        )
    }

    pub(crate) fn load_prompt_cache_for_input_distributed(
        &mut self,
        funding: &eredu_runtime::cache::PromptCachePersistenceFunding,
        materialization: &crate::backend::runtime::cache::residency::PromptCacheMaterialization,
        control: Option<&crate::backend::runtime::distributed::topology::original_source::control::cache::CacheControlOwner>,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        input_identity: eredu_runtime::SharedPreparedInputCacheIdentity,
    ) -> Result<Option<SharedPromptCacheManifest>, Error> {
        self.erased_mut().load_prompt_cache_for_input_distributed(
            funding,
            materialization,
            control,
            directory,
            expected,
            prefix_token_ids,
            input_identity,
        )
    }

    pub(crate) fn save_prompt_cache_distributed(
        &mut self,
        funding: &eredu_runtime::cache::PromptCachePersistenceFunding,
        control: Option<&crate::backend::runtime::distributed::topology::original_source::control::cache::CacheControlOwner>,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<Option<SharedPromptCacheManifest>, Error> {
        self.erased_mut().save_prompt_cache_distributed(
            funding,
            control,
            destination,
            descriptor,
            prefix_token_ids,
            options,
        )
    }

    pub fn speculative_capability(&self) -> SpeculativeCapability {
        if self.erased().has_embedded_prediction() {
            return SpeculativeCapability::Ready {
                draft_source: SpeculativeDraftSource::Embedded,
            };
        }
        match self
            .erased()
            .capability_estimate()
            .speculative_draft_source()
        {
            Some(draft_source @ SpeculativeDraftSource::Separate) => {
                SpeculativeCapability::Declared { draft_source }
            }
            Some(draft_source) => SpeculativeCapability::Unsupported {
                draft_source,
                architecture: self.effective_model_type().to_owned(),
            },
            None => SpeculativeCapability::Unavailable,
        }
    }

    pub fn residency_report(&self) -> Result<Option<eredu_runtime::ResidencyReport>, Error> {
        self.erased().residency_report()
    }

    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        self.erased().dense_stream_report()
    }

    pub fn materialization_report(&self) -> Option<&eredu_runtime::WeightMaterializationReport> {
        self.erased().materialization_report()
    }

    pub fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBanksResidencyReport>,
        Error,
    > {
        self.erased().parameter_bank_report()
    }

    pub fn effective_model_type(&self) -> &str {
        self.erased().effective_model_type()
    }

    pub fn prompt_cache_model_identity(
        &self,
    ) -> Result<eredu_core::cache::PromptCacheModelIdentity, Exception> {
        Ok(self.erased().prompt_cache_model_identity().clone())
    }

    pub fn reset_cache_with_options(
        &mut self,
        _policy: CacheResidencyPolicy,
    ) -> Result<(), Exception> {
        self.reset_cache()
    }

    pub fn reset_cache(&mut self) -> Result<(), Exception> {
        self.erased_mut().reset_cache()
    }

    pub fn load_prompt_cache(
        &mut self,
        funding: &eredu_runtime::cache::PromptCachePersistenceFunding,
        materialization: &crate::backend::runtime::cache::residency::PromptCacheMaterialization,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
    ) -> Result<SharedPromptCacheManifest, Error> {
        self.erased_mut().load_prompt_cache(
            funding,
            materialization,
            None,
            directory.as_ref(),
            expected,
            prefix_token_ids,
        )
    }

    pub fn load_prompt_cache_for_input(
        &mut self,
        funding: &eredu_runtime::cache::PromptCachePersistenceFunding,
        materialization: &crate::backend::runtime::cache::residency::PromptCacheMaterialization,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        input_identity: impl Into<eredu_runtime::SharedPreparedInputCacheIdentity>,
    ) -> Result<SharedPromptCacheManifest, Error> {
        self.erased_mut().load_prompt_cache_for_input(
            funding,
            materialization,
            None,
            directory.as_ref(),
            expected,
            prefix_token_ids,
            input_identity.into(),
        )
    }

    pub fn save_prompt_cache(
        &mut self,
        funding: &eredu_runtime::cache::PromptCachePersistenceFunding,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<SharedPromptCacheManifest, Error> {
        self.erased_mut().save_prompt_cache(
            funding,
            None,
            destination.as_ref(),
            descriptor,
            prefix_token_ids,
            options,
        )
    }

    pub fn cache_residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        self.erased().cache_residency_report()
    }

    pub(crate) fn prefill(
        &mut self,
        input: input::ModelInput<'_>,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.erased_mut().prefill(input, stream)
    }
}

/// Mechanical projection of the already selected native parameter population.
pub(crate) struct NativeLayerwiseParameters<'a>(
    pub(crate) &'a crate::backend::runtime::execution::generic::LayerwiseWorkspace,
);
impl eredu_architectures::prepared_execution::WorkspaceLayerwiseParameters
    for NativeLayerwiseParameters<'_>
{
    fn observe_acquire(
        &self,
        ordinal: usize,
        address: eredu_runtime::ExecutionUnitAddress,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<(), eredu_nn::Error> {
        eredu_architectures::prepared_execution::WorkspaceLayerwiseParameters::observe_acquire(
            self.0, ordinal, address, context,
        )
    }
    fn excludes_parameter(&self, name: &str) -> bool {
        self.0.excludes_parameter(name)
    }
    fn layout(&self) -> &eredu_runtime::ExecutionUnitLayout {
        self.0.layout()
    }
    fn execution_address(&self, ordinal: usize) -> Option<eredu_runtime::ExecutionUnitAddress> {
        self.0.execution_address(ordinal)
    }
    fn parameter_source(
        &self,
    ) -> Result<
        eredu_runtime::working_memory::WorkspaceParameterSourceLoan<'_>,
        eredu_runtime::working_memory::WorkspaceParameterSourceError,
    > {
        // Borrow the actual retained workspace, not this forwarding wrapper.
        Ok(self.0.parameter_source())
    }
    fn parameters(
        &self,
        ordinal: usize,
        address: eredu_runtime::ExecutionUnitAddress,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<
        std::collections::BTreeMap<eredu_nn::ParameterId, eredu_nn::workspace::WorkspaceTensor>,
        eredu_nn::Error,
    > {
        self.0.parameters(ordinal, address, context)
    }
}

fn initial_publication_inspection_busy(error: &Error) -> bool {
    // Runtime mechanisms can transport this same cause through the neutral
    // session error and Error::Other. Classify the actual first metadata cause,
    // never diagnostic text or a nested source inside a Native query failure.
    let mut source: &(dyn std::error::Error + 'static) = error;
    loop {
        if let Some(metadata) = source.downcast_ref::<safemlx::ArrayMetadataError>() {
            return matches!(metadata, safemlx::ArrayMetadataError::RuntimeBusy);
        }
        let Some(next) = source.source() else {
            return false;
        };
        source = next;
    }
}

#[cfg(test)]
#[path = "model/initial_publication_tests.rs"]
mod initial_publication_tests;
