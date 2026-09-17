//! The actual original media source reduced by the existing native recipe worker.
use super::*;
use crate::backend::nn::workspace::{
    ProjectedResidentState, ResidentNativeRecipe, ResidentRecipeRecorder,
};
use crate::backend::runtime::residency::storage::StorageIdentity;
use crate::composition::mlx::model::{retain_planning_error, retain_planning_error_with_kind};
use eredu_core::{
    BackendFailureKind, InferenceGeometry, PreparedRequestRejection, TextFilterWorkspace,
    TextGenerationConfig,
};
use eredu_nn::workspace::{WorkspaceMetadataFunding, WorkspaceMetadataFundingError};
use eredu_runtime::working_memory::{
    RegisteredPreparedWorkspaceStorage, RegisteredWorkspaceStorageLayout,
};
use std::mem::{size_of, size_of_val};
use crate::composition::mlx::session::model_session::text_quote::CaptureAdmission;
use eredu_runtime::layered::{BoundCaptureSelection, PreparedCaptureSelection};
use crate::composition::mlx::session::intervention::TextInterventionQuote;
use eredu_runtime::working_memory::OriginalInterventionSource;

/// No Q is issued here. The existing packet and complete metadata source remain
/// borrowed/owned until the later quote consumes their genuinely observed facts.
pub(crate) struct OriginalMediaRecipe {
    report: OriginalMediaWorkspaceReport,
    recipe: Option<ResidentNativeRecipe>,
    projected: Option<ProjectedResidentState>,
    storage: RegisteredPreparedWorkspaceStorage<StorageIdentity>,
    capture_selection: Option<(PreparedCaptureSelection, InferenceGeometry)>,
    intervention_source: Option<OriginalInterventionSource>,
    source: CompletedOriginalModelInput,
    parameter_epoch: u64,
    context: WorkspaceContext,
    funding: WorkspaceMetadataFunding,
}
#[derive(thiserror::Error)]
#[error("{detail}")]
struct IncompleteMediaRecipe {
    detail: String,
    #[source]
    reason: PreparedRequestRejection,
}
impl std::fmt::Debug for IncompleteMediaRecipe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IncompleteMediaRecipe")
            .field("detail", &self.detail)
            .field("reason", &self.reason)
            .finish_non_exhaustive()
    }
}
/// Preserve the actual refusal while identifying its source construction step.
/// The existing planning error adapter pays and retains this typed transport.
#[derive(Debug, thiserror::Error)]
#[error("original media {stage}: {cause}")]
struct MediaSourceFailure<E: std::error::Error + Send + Sync + 'static> {
    stage: &'static str,
    #[source]
    cause: E,
}
fn source_failure<E: std::error::Error + Send + Sync + 'static>(
    stage: &'static str,
    cause: E,
    funding: &WorkspaceMetadataFunding,
) -> Error {
    let native = (&cause as &dyn std::error::Error).downcast_ref::<Error>();
    let kind = native
        .and_then(Error::retained_backend_failure_kind)
        .or_else(|| {
            (&cause as &dyn std::error::Error)
                .downcast_ref::<eredu_core::BackendFailure>()
                .map(eredu_core::BackendFailure::kind)
        })
        .unwrap_or(BackendFailureKind::Other);
    let preserved = native.is_some_and(Error::model_state_preserved);
    retain_planning_error_with_kind(
        MediaSourceFailure { stage, cause },
        funding.clone(),
        kind,
        preserved,
    )
}
impl std::fmt::Debug for OriginalMediaRecipe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OriginalMediaRecipe")
            .field("parameter_epoch", &self.parameter_epoch)
            .field(
                "spans",
                &self
                    .recipe
                    .as_ref()
                    .map_or(0, |recipe| recipe.records().len()),
            )
            .finish_non_exhaustive()
    }
}
impl OriginalMediaRecipe {
    pub(crate) fn has_missing_operation(&self) -> bool {
        let recipe = self.recipe.as_ref().expect("unconsumed media recipe");
        recipe
            .records()
            .iter()
            .any(|record| record.first_missing_operation().is_some())
            || recipe
                .sampling_records()
                .iter()
                .any(|record| record.first_missing_operation().is_some())
    }

    /// Build the observed candidate for the shared completeness/selection worker.
    /// This creates no reservation, no native work and no second B upload.
    pub(in crate::composition::mlx::session::model_session) fn quote_complete(
        &mut self,
        runtime: &ModelRuntime<MlxBackend<'_>>,
        config: TextGenerationConfig,
        workspace: eredu_core::TextControllerWorkspace<'_>,
        storage: &eredu_runtime::working_memory::ControllerStorageContract,
        claim: &eredu_core::GenerationSequencePreparation<'_, '_>,
        original_table: bool,
        retained_sources: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
        capture: Option<&CaptureAdmission<'_>>,
    ) -> Result<
        crate::composition::mlx::session::model_session::text_quote::TextWorkspaceCandidate,
        Error,
    > {
        use eredu_runtime::working_memory::ControllerStorageContract;
        let result = (|| {
            let session = runtime.session();
            if self.recipe.is_none() || self.projected.is_none()
                || !session.parameter_epoch_matches(self.parameter_epoch) {
                return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
            }
            self.context.charge_metadata(size_of::<(
                &ControllerStorageContract,
                Option<&CaptureAdmission<'_>>,
                [&PreparedCaptureSelection; 2], &InferenceGeometry, Result<(), Error>,
                Result<(), eredu_runtime::layered::PreparedCaptureSelectionError>,
                ProjectedResidentState, Option<ProjectedResidentState>,
                Option<crate::backend::nn::workspace::ProjectedPagedSources>,
                Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
                bool,
                crate::composition::mlx::session::model_session::text_quote::TextWorkspaceCandidate,
                Result<crate::composition::mlx::session::model_session::text_quote::TextWorkspaceCandidate, Error>,
                Option<crate::composition::mlx::session::model_session::text_quote::original_prepared::PreparedMediaQuoteSource>,
                eredu_runtime::working_memory::IncrementalInferenceQuote,
                Result<eredu_runtime::working_memory::IncrementalInferenceQuote, eredu_runtime::working_memory::IncompleteWorkspace>,
                eredu_core::InputTokenCount,
                eredu_core::TextControllerWorkspace<'_>,
                &eredu_core::GenerationSequencePreparation<'_, '_>,
            )>()).map_err(|cause| Error::Neural(cause.into()))?;
            if !self
                .source
                .matches_workspace_source(self.storage.prepared_source())
            {
                return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
            }
            let state_input = self
                .source
                .borrowed_semantics()
                .request_position_facts()
                .and_then(|positions| {
                    positions.legacy_input_accounting(std::num::NonZeroU8::new(4).unwrap())
                })
                .map_err(|cause| retain_planning_error(cause, self.funding.clone()))?;
            let geometry = self.report.equations().geometry();
            match (&self.capture_selection, capture) {
                (Some((traced, traced_geometry)), Some(admission)) => {
                    if *traced_geometry != geometry {
                        return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
                    }
                    admission.validate_intervention_source(self.intervention_source.as_ref())?;
                    admission.validate_media_trace(traced)
                        .map_err(|cause| source_failure("capture source association", cause, &self.funding))?;
                }
                (None, None) if self.intervention_source.is_none() => {}
                _ => return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch)),
            }
            let sampling = self
                .report
                .sampling()
                .ok_or(Error::PrefillControl(WorkingMemoryError::UnknownBound))?;
            let mut projected = self.projected.take()
                .ok_or(Error::PrefillControl(WorkingMemoryError::IdentityMismatch))?;
            let mut paged_sources = projected.storage
                .take_paged_sources(&self.context)
                .map_err(|cause| source_failure("paged source handoff", cause, &self.funding))?;
            // The canonical source owns its pins. As in the token and saved
            // workers, retire temporary native aliases before another fallible
            // constructor can retire that source. Registered metadata roots and
            // the completed media input keep their separate existing owners.
            drop(projected);
            if let Some(source) = &mut paged_sources {
                source
                    .prepare_catalogs(self.report.equations().span_workspace_plan(), &self.context)
                    .map_err(|cause| {
                        source_failure("paged catalog preparation", cause, &self.funding)
                    })?;
                source
                    .prepare_host_program(&session.payload.memory_pool, &self.context)
                    .map_err(|cause| source_failure("paged Host program", cause, &self.funding))?;
            }
            let quote =
                crate::composition::mlx::session::model_session::text_quote::quote_completed_input(
                    session,
                    geometry,
                    config,
                    workspace,
                    storage,
                    self.report.equations(),
                    sampling,
                    &self.storage,
                    state_input,
                    &mut self.recipe,
                    capture,
                    claim,
                    original_table,
                    retained_sources,
                    &self.context,
                    paged_sources.as_ref(),
                )
                .map_err(|cause| match cause {
                    cause @ Error::PrefillControl(
                        WorkingMemoryError::SubmissionTrackingCapacity { .. }
                        | WorkingMemoryError::GraphMetadataCapacity { .. },
                    ) => cause,
                    cause => source_failure("native quote assembly", cause, &self.funding),
                })?;
            if !session.parameter_epoch_matches(self.parameter_epoch) {
                return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch));
            }
            Ok(crate::composition::mlx::session::model_session::text_quote::TextWorkspaceCandidate {
                quote,
                paged_sources,
                output_width: sampling.output_width,
                native_recipe: self.recipe.take(),
                prepared_source: Some(crate::composition::mlx::session::model_session::text_quote::original_prepared::PreparedMediaQuoteSource::new(self.storage.prepared_source().clone(), self.context.clone())),
                planning_metadata: Some(self.funding.clone()),
            })
        })();
        result.map_err(|cause| match cause {
            error @ Error::PrefillControl(
                WorkingMemoryError::SubmissionTrackingCapacity { .. }
                | WorkingMemoryError::GraphMetadataCapacity { .. },
            ) => error,
            cause => retain_planning_error(cause, self.funding.clone()),
        })
    }

    /// Owns an exact missing-receipt diagnostic from the actual source trace.
    /// Complete numerical candidates instead enter the shared quote/control join.
    pub(crate) fn into_pending_admission(self) -> Error {
        let missing = self.recipe.as_ref().and_then(|recipe| {
            recipe
                .records()
                .iter()
                .enumerate()
                .find_map(|(span, record)| {
                    record
                        .first_missing_operation()
                        .map(|operation| (span, operation))
                })
        });
        let (reason, detail) = match missing {
            Some((span, operation)) => (
                PreparedRequestRejection::MissingNumericalWorkspace,
                self.context.metadata_string(format_args!(
                    "original media native recipe is missing span {span} operation {operation}: {:?}",
                    self.report.intervals().get(span).and_then(|interval| interval.operation(operation)).map(|operation| &operation.kind),
                )),
            ),
            None => match self.recipe.as_ref().and_then(|recipe| {
                recipe.sampling_records().iter().find_map(|record| {
                    record.first_missing_operation().map(|operation| (record.phase(), operation))
                })
            }) {
                Some((phase, operation)) => (
                    PreparedRequestRejection::MissingNumericalWorkspace,
                    self.context.metadata_string(format_args!(
                        "original media native recipe is missing sampling {phase:?} operation {operation}"
                    )),
                ),
                None => (
                    PreparedRequestRejection::MissingExecutionControls,
                    self.context.metadata_string(format_args!(
                        "original media native recipe requires its request funding and execution-control join"
                    )),
                ),
            },
        };
        match detail {
            Err(cause) => retain_planning_error(cause, self.funding.clone()),
            Ok(detail) => {
                // The completed diagnostic owns only copied text and a fixed reason.
                // Its source contains no borrowed native data and needs no B alias;
                // self retires the complete source after the paid error is formed.
                let failure = IncompleteMediaRecipe { detail, reason };
                retain_planning_error_with_kind(
                    failure,
                    self.funding.clone(),
                    BackendFailureKind::Unsupported,
                    false,
                )
            }
        }
    }
}

impl CompletedOriginalModelInput {
    /// Projects only this completed B owner. Saved decoder/sampler provenance
    /// remains the paired snapshot's responsibility; no live state is borrowed.
    pub(in crate::composition::mlx::session) fn project_workspace(
        &self,
        context: &WorkspaceContext,
        pool: &eredu_runtime::working_memory::WorkingMemoryPool,
    ) -> Result<OriginalMediaWorkspaceInput, OriginalMediaWorkspaceInputError> {
        self.project_workspace_with_semantics(&self.semantics, context, pool)
    }
    pub(in crate::composition::mlx::session) fn project_workspace_with_semantics(
        &self,
        semantics: &eredu_architectures::media_plan::BoundPreparedMediaSemantics,
        context: &WorkspaceContext,
        pool: &eredu_runtime::working_memory::WorkingMemoryPool,
    ) -> Result<OriginalMediaWorkspaceInput, OriginalMediaWorkspaceInputError> {
        OriginalMediaWorkspaceInput::project_with_metadata(
            self.body.prepared().expect("complete B prepared"),
            semantics.clone(),
            context,
            pool,
        )
    }
}

impl MlxModelInput {
    /// Source-bound diagnostic construction under the actual planning account.
    /// No ordinary lease is created, no source payload is evaluated or cloned,
    /// and no first missing operation is promoted to a complete native bound.
    pub(crate) fn quote_original_media_recipe_funded<'a>(
        &self,
        runtime: &ModelRuntime<MlxBackend<'_>>,
        geometry: InferenceGeometry,
        config: TextGenerationConfig,
        filter: TextFilterWorkspace<'a>,
        capacity: u64,
        capture: Option<BoundCaptureSelection<'_>>,
        interventions: Option<TextInterventionQuote<'_>>,
    ) -> Result<OriginalMediaRecipe, Error> {
        let pool = runtime.backend().memory_pool();
        let session = runtime.session();
        let executable = &session.payload.model;
        let funding = pool
            .prepare_workspace_metadata(
                executable.erased().inference_execution_identity(),
                capacity,
            )
            .map_err(Error::WorkspacePlanning)?;
        let controls = [
            size_of::<OriginalMediaRecipe>(),
            size_of::<Result<OriginalMediaRecipe, Error>>(),
            size_of::<IncompleteMediaRecipe>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<ProjectedResidentState>(),
            size_of::<Option<ProjectedResidentState>>(),
            size_of::<CompletedOriginalModelInput>(),
            size_of::<Option<(PreparedCaptureSelection, InferenceGeometry)>>(),
            size_of::<[Option<TextInterventionQuote<'_>>; 3]>(),
            size_of::<Option<OriginalInterventionSource>>(),
            size_of::<(
                InferenceGeometry,
                TextGenerationConfig,
                TextFilterWorkspace<'_>,
                Option<BoundCaptureSelection<'_>>,
                u64,
                Option<u64>,
            )>(),
        ];
        let bytes = controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .ok_or(Error::WorkspacePlanning(
                WorkspaceMetadataFundingError::Overflow,
            ))?;
        funding
            .reserve_metadata(bytes)
            .map_err(Error::WorkspacePlanning)?;
        let result = (|| {
            if interventions.is_some() && capture.is_none() {
                return Err(source_failure("intervention source association",
                    WorkingMemoryError::IdentityMismatch, &funding));
            }
            let Some(input::OriginalMediaPacket::Original(packet)) = self.original_media.as_ref()
            else {
                return Err(retain_planning_error(
                    WorkingMemoryError::UnknownBound,
                    funding.clone(),
                ));
            };
            packet
                .body
                .source()
                .validate_pool(pool)
                .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
            if session.poison.get()
                || session
                    .authority
                    .try_borrow()
                    .map_or(true, |authority| authority.require_idle().is_err())
            {
                return Err(retain_planning_error(
                    WorkingMemoryError::ExecutionFenced,
                    funding.clone(),
                ));
            }
            if !runtime
                .backend()
                .matches_prepared_target(&session.payload.target)
            {
                return Err(retain_planning_error(
                    WorkingMemoryError::IdentityMismatch,
                    funding.clone(),
                ));
            }
            let current = executable
                .erased()
                .original_request_media_binding()
                .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
            if !packet.semantics.binding().matches(&current)
                || current.frontier() != geometry.cached_positions
            {
                return Err(retain_planning_error(
                    WorkingMemoryError::IdentityMismatch,
                    funding.clone(),
                ));
            }
            let blueprint = executable.inference_blueprint().ok_or_else(|| {
                retain_planning_error(WorkingMemoryError::UnknownBound, funding.clone())
            })?;
            let facts = executable.resident_workspace_mechanisms().ok_or_else(|| {
                retain_planning_error(WorkingMemoryError::UnknownBound, funding.clone())
            })?;
            let mut epoch = None;
            session
                .validate_parameter_epoch(&mut epoch)
                .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
            let parameter_epoch = epoch.ok_or_else(|| {
                retain_planning_error(WorkingMemoryError::IdentityMismatch, funding.clone())
            })?;
            let addressable=executable.prepare_addressable_workspace_sources(facts,pool,&funding)?;
            let context=match &addressable {
                Some(source)=>WorkspaceContext::new_with_metadata_funding(crate::backend::nn::workspace::MlxAddressableWorkspaceMechanisms::new(facts,source.clone()),funding.clone()),
                None=>WorkspaceContext::new_with_metadata_funding(facts,funding.clone()),
            }.map_err(|cause|retain_planning_error(cause,funding.clone()))?;
            executable
                .erased()
                .install_workspace_parameter_representations(&context)
                .map_err(|cause| {
                    source_failure("parameter representation projection", cause, &funding)
                })?;
            let input = packet
                .project_workspace(&context, pool)
                .map_err(|cause| source_failure("input slot projection", cause, &funding))?;
            let batch = std::num::NonZeroU32::new(
                u32::try_from(geometry.batch_size)
                    .map_err(|cause| retain_planning_error(cause, funding.clone()))?,
            )
            .ok_or_else(|| {
                retain_planning_error(WorkingMemoryError::IdentityMismatch, funding.clone())
            })?;
            let projected = executable
                .erased()
                .project_resident_workspace_with_storage(batch, &context)
                .map_err(|cause| source_failure("decoder projection", cause, &funding))?;
            if !projected.storage.is_complete() {
                return Err(source_failure(
                    "decoder backing",
                    WorkingMemoryError::UnknownBound,
                    &funding,
                ));
            }
            let count = projected
                .storage
                .iter()
                .filter(|(_, bytes, _)| *bytes != 0)
                .count();
            let layout =
                RegisteredWorkspaceStorageLayout::<StorageIdentity>::new_with_prepared_source(
                    count,
                    input.source_storage(),
                )
                .map_err(|cause| {
                    source_failure(
                        if input.source_storage().borrowed_storage().is_none() {
                            "completed input backing"
                        } else {
                            "source binding layout"
                        },
                        cause,
                        &funding,
                    )
                })?;
            context
                .charge_metadata(layout.requested_bytes())
                .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
            let storage = layout
                .construct_with_prepared_source(
                    pool,
                    &context,
                    projected
                        .storage
                        .iter()
                        .filter(|(_, bytes, _)| *bytes != 0)
                        .map(|(identity, _, root)| {
                            (StorageIdentity::Native(identity), root.clone())
                        }),
                    input.source_storage().clone(),
                )
                .map_err(|cause| source_failure("source binding", cause, &funding))?;
            // Match original token quotation: this trace owns numerical equations.
            // The common quote composer separately binds the selected layerwise
            // source, unloaded-slot constructors, materialization and transfers.
            let mut recorder = facts.recorder(geometry, &context)
                .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
            if let Some(source)=addressable{recorder.bind_addressable_sources(source).map_err(|cause|retain_planning_error(cause,funding.clone()))?;}
            // Keep the exact plan/path owners that emitted this observed recipe.
            // This clones only existing source aliases; capture admission retains
            // their original storage accounting and grants the later work.
            let capture_selection = capture.map(|bound| (bound.selection().clone(), bound.geometry()));
            let report = if let Some(capture) = capture {
                executable.quote_original_media_capture(
                    input, &current, geometry, &projected.state, &context,
                    config, filter, capture, interventions, &mut recorder,
                ).map_err(|cause| source_failure("observed equation and sampling trace", cause, &funding))?
            } else {
                blueprint.quote_original_media_with_sampling_and_trace(
                    input,
                    &current,
                    geometry,
                    &projected.state,
                    &context,
                    None,
                    config,
                    filter,
                    &mut recorder,
                )
                .map_err(|cause| {
                    source_failure(
                        "equation and sampling trace",
                        cause.into_failure(),
                        &funding,
                    )
                })?
            };
            let recipe = recorder
                .finish(report.equations().span_workspace_plan())
                .map_err(|cause| source_failure("recipe reduction", cause, &funding))?;
            if !session.parameter_epoch_matches(parameter_epoch) {
                return Err(retain_planning_error(
                    WorkingMemoryError::IdentityMismatch,
                    funding.clone(),
                ));
            }
            Ok(OriginalMediaRecipe {
                report,
                recipe: Some(recipe),
                projected: Some(projected),
                storage,
                capture_selection,
                intervention_source: interventions.map(|source| source.source.clone()),
                source: packet.clone(),
                parameter_epoch,
                context,
                funding: funding.clone(),
            })
        })();
        result
    }
}
