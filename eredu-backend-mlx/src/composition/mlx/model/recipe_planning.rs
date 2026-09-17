//! One recipe worker with caller-owned host metadata admission.
use super::{Error, Executable, StorageIdentity};
use crate::backend::nn::workspace::{
    ResidentExecutionMechanisms, MlxParallelWorkspace, MlxParallelWorkspaceMechanisms,
    ProjectedPagedSources, ResidentNativeRecipe, ResidentRecipeRecorder, AddressableSources, MlxAddressableWorkspaceMechanisms,
};
use crate::backend::runtime::distributed::topology::original_source::parallel::OriginalParallelSource;
use eredu_architectures::prepared_execution::{
    PreparedInferenceBlueprint, PreparedTextGenerationWorkspace,
};
use eredu_core::{
    BackendFailure, BackendFailureKind, InferenceGeometry, OutputDemand, SharedBackendFailure,
    TextFilterWorkspace, TextGenerationConfig,
};
use eredu_nn::workspace::{
    WorkspaceContext, WorkspaceMetadataFunding, WorkspaceMetadataFundingError,
};
use eredu_runtime::working_memory::{
    RegisteredWorkspaceStorage, RegisteredWorkspaceStorageLayout, WorkingMemoryError,
    WorkingMemoryPool,
};
use std::{
    error::Error as StdError,
    mem::{size_of, size_of_val},
    num::NonZeroU32,
};

pub(super) mod capture;
mod addressable;
use crate::composition::mlx::session::intervention::TextInterventionQuote;
use eredu_runtime::layered::{BoundCaptureSelection, PreparedCaptureSelection};

type RecipeParts = (
    PreparedTextGenerationWorkspace,
    RegisteredWorkspaceStorage<StorageIdentity>,
    ResidentNativeRecipe,
);

type ScopedRecipeParts = (RecipeParts, Option<ProjectedPagedSources>);

// A private sum type keeps the source/context join intact throughout projection,
// quotation and recipe reduction. Consuming it moves the same Context to the
// accepted planning owner; its mechanism already retains the source.
enum RecipeWorkspace {
    Resident(WorkspaceContext),
    Parallel(MlxParallelWorkspace),
}
impl RecipeWorkspace {
    fn context(&self) -> &WorkspaceContext {
        match self {
            Self::Resident(context) => context,
            Self::Parallel(workspace) => workspace.context(),
        }
    }
    fn parallel(&self) -> Option<&MlxParallelWorkspace> {
        match self {
            Self::Resident(_) => None,
            Self::Parallel(workspace) => Some(workspace),
        }
    }
    fn into_context(self) -> WorkspaceContext {
        match self {
            Self::Resident(context) => context,
            Self::Parallel(workspace) => workspace.into_context(),
        }
    }
}

/// The context and independent funding stay alive through every returned plan.
/// The consuming split is private to composition: its caller installs funding
/// last in the accepted candidate/quote and retains it across report composition.
pub(in crate::composition::mlx) struct FundedResidentRecipePlanning {
    generation: PreparedTextGenerationWorkspace,
    storage: RegisteredWorkspaceStorage<StorageIdentity>,
    recipe: ResidentNativeRecipe,
    paged_sources: Option<ProjectedPagedSources>,
    context: WorkspaceContext,
    funding: WorkspaceMetadataFunding,
}
impl FundedResidentRecipePlanning {
    pub(in crate::composition::mlx) fn parts(
        &self,
    ) -> (
        &PreparedTextGenerationWorkspace,
        &RegisteredWorkspaceStorage<StorageIdentity>,
        &ResidentNativeRecipe,
        &WorkspaceContext,
    ) {
        (&self.generation, &self.storage, &self.recipe, &self.context)
    }

    pub(in crate::composition::mlx) fn into_parts(
        self,
    ) -> (
        PreparedTextGenerationWorkspace,
        RegisteredWorkspaceStorage<StorageIdentity>,
        ResidentNativeRecipe,
        Option<ProjectedPagedSources>,
        WorkspaceContext,
        WorkspaceMetadataFunding,
    ) {
        (
            self.generation,
            self.storage,
            self.recipe,
            self.paged_sources,
            self.context,
            self.funding,
        )
    }
}

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct PlanningFailure<E: StdError + Send + Sync + 'static> {
    #[source]
    cause: E,
    // SharedBackendFailure retires its allocation before this payload; the
    // original cause retires before the final cumulative planning account.
    _funding: WorkspaceMetadataFunding,
}

/// Retains a concrete cause using the existing closed diagnostic owner. The
/// failed producer has already paid its own allocations; this pays only the
/// incremental error shell and its actual constructor/return transports.
/// Refusal destroys the original cause while funding is still live, then
/// returns an inline funding error without allocating a fallback diagnostic.
pub(in crate::composition::mlx) fn retain_planning_error<E>(
    cause: E,
    funding: WorkspaceMetadataFunding,
) -> Error
where
    E: StdError + Send + Sync + 'static,
{
    let (kind, state_preserved) = planning_error_classification(&cause);
    retain_planning_error_with_kind(cause, funding, kind, state_preserved)
}

/// A neutral admission failure already has its public classification. Retain
/// it as the exact cause without reclassifying it as a generic native error.
pub(in crate::composition::mlx) fn retain_planning_failure(
    cause: BackendFailure,
    funding: WorkspaceMetadataFunding,
) -> Error {
    let kind = cause.kind();
    retain_planning_error_with_kind(cause, funding, kind, false)
}

pub(in crate::composition::mlx) fn retain_planning_error_with_kind<E>(
    cause: E,
    funding: WorkspaceMetadataFunding,
    kind: BackendFailureKind,
    state_preserved: bool,
) -> Error
where
    E: StdError + Send + Sync + 'static,
{
    // Build only an inline owner before reservation, so even an unwind from
    // the account callback destroys the cause before its funding token.
    let failure = PlanningFailure {
        cause,
        _funding: funding,
    };
    let bytes = planning_error_bytes::<E>();
    let admission = bytes
        .ok_or(WorkspaceMetadataFundingError::Overflow)
        .and_then(|bytes| failure._funding.reserve_metadata(bytes));
    if let Err(refusal) = admission {
        drop(failure);
        return Error::WorkspacePlanning(refusal);
    }
    Error::retained_original(SharedBackendFailure::new(kind, failure), state_preserved)
}

#[derive(Clone, Copy, Debug, thiserror::Error)]
#[error("{0}")]
struct RecipeInputError(&'static str);
struct RecipeInputs<'a> {
    blueprint: &'a PreparedInferenceBlueprint,
    batch: NonZeroU32,
    facts: ResidentExecutionMechanisms,
}

// Ordinary diagnostics keep their original erasure. Funded diagnostics pay
// before erasure and preserve the same concrete cause under a closed owner.
struct Diagnostics<'a>(Option<&'a WorkspaceMetadataFunding>);

/// The saved-sampler equation worker receives its already funded Context from
/// resume preparation. Reuse the same paid error destination as initial quotes.
pub(super) fn context_source<E: StdError + Send + Sync + 'static>(
    context: &WorkspaceContext,
    cause: E,
) -> Error {
    let funding = context.metadata_funding();
    Diagnostics(funding.as_ref()).source(cause)
}

impl Diagnostics<'_> {
    fn source<E: StdError + Send + Sync + 'static>(&self, cause: E) -> Error {
        match self.0 {
            Some(funding) => retain_planning_error(cause, funding.clone()),
            None => Error::Other(Box::new(cause)),
        }
    }
    fn native(&self, cause: Error) -> Error {
        match self.0 {
            Some(funding) => retain_planning_error(cause, funding.clone()),
            None => cause,
        }
    }
}

impl Executable {
    fn resident_recipe_inputs(
        &self,
        geometry: InferenceGeometry,
    ) -> Result<RecipeInputs<'_>, RecipeInputError> {
        let blueprint = self.inference_blueprint().ok_or(RecipeInputError(
            "executable has no retained inference blueprint",
        ))?;
        let batch = u32::try_from(geometry.batch_size)
            .ok()
            .and_then(NonZeroU32::new)
            .ok_or(RecipeInputError(
                "workspace batch exceeds native tensor extent",
            ))?;
        let facts = self.workspace.ok_or(RecipeInputError(
            "selected native execution has no retained workspace mechanism facts",
        ))?;
        Ok(RecipeInputs {
            blueprint,
            batch,
            facts,
        })
    }

    /// Existing cold diagnostic entry. It issues no planning admission.
    pub(crate) fn quote_registered_resident_text_with_sampling_recipe<'a>(
        &self,
        geometry: InferenceGeometry,
        config: TextGenerationConfig,
        filter: impl Into<TextFilterWorkspace<'a>>,
        pool: &WorkingMemoryPool,
    ) -> Result<RecipeParts, Error> {
        let inputs = self
            .resident_recipe_inputs(geometry)
            .map_err(|cause| Error::ArchitectureModel(cause.0.into()))?;
        let context = WorkspaceContext::new_recording_facts(inputs.facts);
        self.resident_recipe_with_context(
            inputs,
            geometry,
            config,
            filter,
            pool,
            &context,
            None,
            None,
            Diagnostics(None),
            None,
            None,
            None,
            None,
        )
        .map(|(parts, _paged_sources)| parts)
    }

    /// Actual initial planning account, separate from later tensor/execution
    /// admission. Each participating constructor reserves before allocating;
    /// completed spans never refund this cumulative host metadata account.
    pub(in crate::composition::mlx) fn quote_registered_resident_text_with_sampling_recipe_funded<
        'a,
    >(
        &self,
        geometry: InferenceGeometry,
        config: TextGenerationConfig,
        filter: impl Into<TextFilterWorkspace<'a>>,
        pool: &WorkingMemoryPool,
        capacity: u64,
        layerwise: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
    ) -> Result<FundedResidentRecipePlanning, Error> {
        self.resident_recipe_funded(
            geometry,
            config,
            filter.into(),
            pool,
            capacity,
            None,
            None,
            None,
            layerwise,
            |_| Ok(None),
        )
    }

    pub(in crate::composition::mlx) fn quote_registered_resident_text_with_sampling_capture_recipe_funded<
        'a,
    >(
        &self,
        geometry: InferenceGeometry,
        config: TextGenerationConfig,
        filter: impl Into<TextFilterWorkspace<'a>>,
        pool: &WorkingMemoryPool,
        capacity: u64,
        capture: &PreparedCaptureSelection,
        interventions: Option<TextInterventionQuote<'_>>,
        layerwise: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
    ) -> Result<FundedResidentRecipePlanning, Error> {
        self.resident_recipe_funded(
            geometry,
            config,
            filter.into(),
            pool,
            capacity,
            Some(capture),
            interventions,
            None,
            layerwise,
            |_| Ok(None),
        )
    }

    /// Source-aware direct TP quotation. The caller binds its actual selected
    /// native communicator and initialized input runtime under this very same
    /// planning account. This is a quote entry, not a distributed fit grant.
    pub(in crate::composition::mlx) fn quote_registered_direct_parallel_text_with_sampling_recipe_funded<
        'a,
        F,
    >(
        &self,
        geometry: InferenceGeometry,
        config: TextGenerationConfig,
        filter: impl Into<TextFilterWorkspace<'a>>,
        pool: &WorkingMemoryPool,
        capacity: u64,
        capture: Option<&PreparedCaptureSelection>,
        interventions: Option<TextInterventionQuote<'_>>,
        capture_placement: Option<(&eredu_architectures::component_partition::ComponentPartitionLayouts, usize)>,
        layerwise: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
        prepare_source: F,
    ) -> Result<FundedResidentRecipePlanning, Error>
    where
        F: FnOnce(&WorkspaceMetadataFunding) -> Result<OriginalParallelSource, Error>,
    {
        self.resident_recipe_funded(
            geometry,
            config,
            filter.into(),
            pool,
            capacity,
            capture,
            interventions,
            capture_placement,
            layerwise,
            |funding| prepare_source(funding).map(Some),
        )
    }

    fn resident_recipe_funded<F>(
        &self,
        geometry: InferenceGeometry,
        config: TextGenerationConfig,
        filter: TextFilterWorkspace<'_>,
        pool: &WorkingMemoryPool,
        capacity: u64,
        capture: Option<&PreparedCaptureSelection>,
        interventions: Option<TextInterventionQuote<'_>>,
        capture_placement: Option<(&eredu_architectures::component_partition::ComponentPartitionLayouts, usize)>,
        layerwise: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
        prepare_parallel: F,
    ) -> Result<FundedResidentRecipePlanning, Error>
    where
        F: FnOnce(&WorkspaceMetadataFunding) -> Result<Option<OriginalParallelSource>, Error>,
    {
        let funding = pool
            .prepare_workspace_metadata(self.erased().inference_execution_identity(), capacity)
            .map_err(Error::WorkspacePlanning)?;
        let controls = planning_control_bytes()
            .and_then(|bytes| bytes.checked_add(size_of::<F>()))
            .and_then(|bytes| {
                bytes.checked_add(size_of::<Result<Option<OriginalParallelSource>, Error>>())
            })
            .ok_or(Error::WorkspacePlanning(
                WorkspaceMetadataFundingError::Overflow,
            ))?;
        funding
            .reserve_metadata(controls)
            .map_err(Error::WorkspacePlanning)?;
        let inputs = self
            .resident_recipe_inputs(geometry)
            .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        let addressable=self.prepare_addressable_workspace_sources(inputs.facts,pool,&funding)?;
        let workspace = match prepare_parallel(&funding)
            .map_err(|cause| retain_planning_error(cause, funding.clone()))?
        {
            Some(source) => {
                if !source.funding().same_account(&funding) {
                    return Err(retain_planning_error(
                        WorkingMemoryError::IdentityMismatch,
                        funding,
                    ));
                }
                let mechanism=MlxParallelWorkspaceMechanisms::new(inputs.facts,source);
                let workspace=match &addressable {
                    Some(source)=>mechanism.prepare_workspace_with_addressable(source.clone()),
                    None=>mechanism.prepare_workspace(),
                }.map_err(|cause|retain_planning_error(cause,funding.clone()))?;
                RecipeWorkspace::Parallel(workspace)
            }
            None => RecipeWorkspace::Resident(match &addressable {
                Some(source)=>WorkspaceContext::new_with_metadata_funding(MlxAddressableWorkspaceMechanisms::new(inputs.facts,source.clone()),funding.clone()),
                None=>WorkspaceContext::new_with_metadata_funding(inputs.facts,funding.clone()),
            }.map_err(|cause|retain_planning_error(cause,funding.clone()))?),
        };
        let context = workspace.context();
        // The shared architecture entry has this exact validation before its
        // worker. Preserve it here without constructing an uncharged String.
        if capture.is_none()
            && (geometry.batch_size != 1 || geometry.output != OutputDemand::LastPosition)
        {
            return Err(retain_planning_error(
                RecipeInputError(
                    "configured text sampling requires single-sequence final-position generation",
                ),
                funding,
            ));
        }
        let capture = capture
            .map(|selection| selection.bind_geometry(geometry))
            .transpose()
            .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        if let Some(bound) = capture {
            capture::validate(self, geometry, bound, context, capture_placement)
                .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        }
        let ((generation, storage, recipe), paged_sources) = self.resident_recipe_with_context(
            inputs,
            geometry,
            config,
            filter,
            pool,
            context,
            capture,
            interventions,
            Diagnostics(Some(&funding)),
            workspace.parallel(),
            layerwise,
            capture_placement,
            addressable.as_ref(),
        )?;
        Ok(FundedResidentRecipePlanning {
            generation,
            storage,
            recipe,
            paged_sources,
            context: workspace.into_context(),
            funding,
        })
    }

    fn resident_recipe_with_context<'a>(
        &self,
        inputs: RecipeInputs<'_>,
        geometry: InferenceGeometry,
        config: TextGenerationConfig,
        filter: impl Into<TextFilterWorkspace<'a>>,
        pool: &WorkingMemoryPool,
        context: &WorkspaceContext,
        capture: Option<BoundCaptureSelection<'_>>,
        interventions: Option<TextInterventionQuote<'_>>,
        diagnostics: Diagnostics<'_>,
        parallel: Option<&MlxParallelWorkspace>,
        layerwise: Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
        capture_placement: Option<(&eredu_architectures::component_partition::ComponentPartitionLayouts, usize)>,
        addressable:Option<&AddressableSources>,
    ) -> Result<ScopedRecipeParts, Error> {
        // The same immutable executable supplies these descriptive rows and
        // the source projection below. The enclosing text quote captures and
        // revalidates its actual parameter epoch before native execution.
        self.erased()
            .install_workspace_parameter_representations(context)
            .map_err(|cause| diagnostics.source(cause))?;
        // Unobserved bounded equations retain the ordinary resident-shaped trace;
        // their unit parameters still come from this exact prepared copy/read
        // source. Install those facts before any state or module construction.
        // Observed/parallel paths already bind these rows at their checked acquire.
        if capture.is_none() && parallel.is_none() {
            if let Some(source) = layerwise {
                source.extend_workspace_parameter_representations(context)
                    .map_err(|cause| diagnostics.source(cause))?;
            }
        }
        let mut projected = self
            .erased()
            .project_resident_workspace_with_storage(inputs.batch, context)
            .map_err(|cause| diagnostics.native(cause))?;
        if !projected.storage.is_complete() {
            return Err(diagnostics.source(WorkingMemoryError::UnknownBound));
        }
        let roots = || {
            projected
                .storage
                .iter()
                .filter(|(_, bytes, _)| *bytes != 0)
                .map(|(identity, _, root)| (StorageIdentity::Native(identity), root.clone()))
        };
        let storage = if diagnostics.0.is_some() {
            let count = projected
                .storage
                .iter()
                .filter(|(_, bytes, _)| *bytes != 0)
                .count();
            let layout = RegisteredWorkspaceStorageLayout::<StorageIdentity>::new(count)
                .map_err(|cause| diagnostics.source(cause))?;
            context
                .charge_metadata(layout.requested_bytes())
                .map_err(|cause| diagnostics.source(cause))?;
            layout.construct(pool, context, roots())
        } else {
            RegisteredWorkspaceStorage::bind(pool, context, roots())
        }
        .map_err(|cause| diagnostics.source(cause))?;
        let (quote, recipe) = if let Some(parallel) = parallel {
            // Only the private workspace owner can supply this exact pair.
            // A separate scalar-compatible source/context is not sufficient.
            if !std::ptr::eq(context, parallel.context()) {
                return Err(diagnostics.source(WorkingMemoryError::IdentityMismatch));
            }
            let mut recorder = parallel
                .recorder(geometry)
                .map_err(|cause| diagnostics.source(cause))?;
            context.charge_metadata(size_of::<(
                Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
                super::NativeLayerwiseParameters<'_>,
                Option<super::NativeLayerwiseParameters<'_>>,
                Result<PreparedTextGenerationWorkspace,eredu_architectures::prepared_execution::PreparedExecutionError<eredu_nn::Error>>,
            )>()).map_err(|cause|diagnostics.source(cause))?;
            if let Some(source) = layerwise {
                recorder.bind_layerwise_constructor_source(source)
                    .map_err(|cause| diagnostics.source(cause))?;
            }
            let parameters = layerwise.map(super::NativeLayerwiseParameters);
            let quote = match capture {
                Some(bound) => capture::quote_partitioned(inputs.blueprint,geometry,&projected.state,
                    context,config,filter.into(),bound,interventions,&mut recorder,parallel.declaration_source(),
                    capture_placement.ok_or_else(|| diagnostics.source(WorkingMemoryError::UnknownBound))?,
                    parameters.as_ref().map(|source| source as
                        &dyn eredu_architectures::prepared_execution::WorkspaceLayerwiseParameters)),
                None => match layerwise {
                    Some(source) => inputs.blueprint.quote_partitioned_layerwise_text_with_sampling_and_trace(
                        geometry,&projected.state,context,config,filter,&mut recorder,
                        Some(parallel.declaration_source()),&super::NativeLayerwiseParameters(source),
                    ),
                    None => inputs.blueprint.quote_partitioned_text_with_sampling_and_trace(
                        geometry,&projected.state,context,config,filter,&mut recorder,
                        Some(parallel.declaration_source()),
                    ),
                },
            }.map_err(|cause|diagnostics.source(cause))?;
            let recipe = recorder
                .finish(quote.equations.span_workspace_plan())
                .map_err(|cause| diagnostics.source(cause))?;
            (quote, recipe)
        } else {
            let mut recorder =
                inputs.facts.recorder(geometry, context)
                    .map_err(|cause| diagnostics.source(cause))?;
            if let Some(source)=addressable{recorder.bind_addressable_sources(source.clone()).map_err(|cause|diagnostics.source(cause))?;}
            context.charge_metadata(size_of::<(
                Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
                Option<super::NativeLayerwiseParameters<'_>>,
                Option<&dyn eredu_architectures::prepared_execution::WorkspaceLayerwiseParameters>,
            )>()).map_err(|cause| diagnostics.source(cause))?;
            let observed_source = layerwise.filter(|_| capture.is_some());
            if let Some(source) = observed_source {
                recorder.bind_layerwise_constructor_source(source)
                    .map_err(|cause| diagnostics.source(cause))?;
            }
            let parameters = observed_source.map(super::NativeLayerwiseParameters);
            let quote = match capture {
                Some(bound) => capture::quote(
                    inputs.blueprint,
                    geometry,
                    &projected.state,
                    context,
                    config,
                    filter.into(),
                    bound,
                    interventions,
                    &mut recorder,
                    parameters.as_ref().map(|source| source as
                        &dyn eredu_architectures::prepared_execution::WorkspaceLayerwiseParameters),
                ),
                None => inputs
                    .blueprint
                    .quote_replicated_resident_text_with_sampling_and_trace(
                        geometry,
                        &projected.state,
                        context,
                        config,
                        filter,
                        &mut recorder,
                    ),
            }
            .map_err(|cause| diagnostics.source(cause))?;
            let recipe = recorder
                .finish(quote.equations.span_workspace_plan())
                .map_err(|cause| diagnostics.source(cause))?;
            (quote, recipe)
        };
        let mut paged_sources = projected
            .storage
            .take_paged_sources(context)
            .map_err(|cause| diagnostics.source(cause))?;
        // The source now owns canonical pins. Retire this inspection's native
        // aliases before a later metadata failure can retire those pins.
        drop(projected);
        if let Some(source) = &mut paged_sources {
            source
                .prepare_catalogs(quote.equations.span_workspace_plan(), context)
                .map_err(|cause| diagnostics.source(cause))?;
            source
                .prepare_host_program(pool, context)
                .map_err(|cause| diagnostics.source(cause))?;
        }
        Ok(((quote, storage, recipe), paged_sources))
    }
}

fn planning_control_bytes() -> Option<usize> {
    let controls = [
        size_of::<FundedResidentRecipePlanning>(),
        size_of::<RecipeWorkspace>(),
        size_of::<MlxParallelWorkspace>(),
        size_of::<Option<OriginalParallelSource>>(),
        size_of::<bool>(),
        size_of::<Option<&MlxParallelWorkspace>>(),
        size_of::<Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>>(),
        size_of::<Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>>(),
        size_of::<Option<&crate::backend::runtime::execution::generic::LayerwiseWorkspace>>(),
        size_of::<(&WorkspaceContext, &MlxParallelWorkspace)>(),
        size_of::<Result<FundedResidentRecipePlanning, Error>>(),
        size_of::<RecipeParts>(),
        size_of::<ScopedRecipeParts>(),
        size_of::<Option<ProjectedPagedSources>>(),
        size_of::<(
            PreparedTextGenerationWorkspace,
            RegisteredWorkspaceStorage<StorageIdentity>,
            ResidentNativeRecipe,
            Option<ProjectedPagedSources>,
            WorkspaceContext,
            WorkspaceMetadataFunding,
        )>(),
        size_of::<(
            &PreparedTextGenerationWorkspace,
            &RegisteredWorkspaceStorage<StorageIdentity>,
            &ResidentNativeRecipe,
            &WorkspaceContext,
        )>(),
        size_of::<Result<RecipeParts, Error>>(),
        size_of::<RecipeInputs<'_>>(),
        size_of::<RecipeInputError>(),
        // Explicit caller argument, parallel entry argument and shared worker
        // loan may overlap while the actual selection is bound to this context.
        size_of::<[Option<&PreparedCaptureSelection>; 3]>(),
        size_of::<[Option<TextInterventionQuote<'_>>; 5]>(),
        size_of::<[Option<(&eredu_architectures::component_partition::ComponentPartitionLayouts, usize)>; 5]>(),
        size_of::<Option<BoundCaptureSelection<'_>>>(),
        size_of::<BoundCaptureSelection<'_>>(),
        size_of::<Diagnostics<'_>>(),
        size_of::<RegisteredWorkspaceStorageLayout<StorageIdentity>>(),
        size_of::<crate::backend::nn::workspace::ProjectedResidentState>(),
        size_of::<(
            &Executable,
            InferenceGeometry,
            TextGenerationConfig,
            TextFilterWorkspace<'_>,
            &WorkingMemoryPool,
            &WorkspaceContext,
            usize,
            u64,
        )>(),
    ];
    controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)
}

fn planning_error_bytes<E: StdError + Send + Sync + 'static>() -> Option<usize> {
    let controls = [
        size_of::<E>(),
        size_of::<PlanningFailure<E>>(),
        size_of::<WorkspaceMetadataFunding>(),
        size_of::<BackendFailure>(),
        size_of::<SharedBackendFailure>(),
        size_of::<BackendFailureKind>(),
        size_of::<bool>(),
        size_of::<WorkspaceMetadataFundingError>(),
        size_of::<Error>(),
        size_of::<Result<(), WorkspaceMetadataFundingError>>(),
        size_of::<Option<usize>>(),
    ];
    SharedBackendFailure::control_bytes::<PlanningFailure<E>>()
        .and_then(|bytes| bytes.checked_add(size_of_val(&controls)))
        .and_then(|bytes| controls.into_iter().try_fold(bytes, usize::checked_add))
}

fn planning_error_classification<E: StdError + 'static>(cause: &E) -> (BackendFailureKind, bool) {
    let native = (cause as &dyn StdError).downcast_ref::<Error>();
    let kind = native
        .and_then(Error::retained_backend_failure_kind)
        .or_else(|| {
            (cause as &dyn StdError)
                .downcast_ref::<BackendFailure>()
                .map(BackendFailure::kind)
        })
        .unwrap_or(BackendFailureKind::Other);
    let state_preserved = native.is_some_and(Error::model_state_preserved);
    (kind, state_preserved)
}

/// One exact, already debited diagnostic constructor. It contains no E value
/// until failure, and cannot be reused or mint another metadata allowance.
pub(in crate::composition::mlx) struct PreparedPlanningError<E> {
    funding: WorkspaceMetadataFunding,
    marker: std::marker::PhantomData<fn() -> E>,
}
impl<E: StdError + Send + Sync + 'static> PreparedPlanningError<E> {
    pub(in crate::composition::mlx) fn prepare(
        funding: &WorkspaceMetadataFunding,
    ) -> Result<Self, WorkspaceMetadataFundingError> {
        let controls = [
            size_of::<Self>(),
            size_of::<Result<Self, WorkspaceMetadataFundingError>>(),
            size_of::<&WorkspaceMetadataFunding>(),
        ];
        let bytes = planning_error_bytes::<E>()
            .and_then(|bytes| bytes.checked_add(size_of_val(&controls)))
            .and_then(|bytes| controls.into_iter().try_fold(bytes, usize::checked_add))
            .ok_or(WorkspaceMetadataFundingError::Overflow)?;
        funding.reserve_metadata(bytes)?;
        Ok(Self {
            funding: funding.clone(),
            marker: std::marker::PhantomData,
        })
    }
    #[inline(never)]
    pub(in crate::composition::mlx) fn retain(self, cause: E) -> Error {
        let (kind, state_preserved) = planning_error_classification(&cause);
        // The original shared error producer and drop order are unchanged.
        // All its exact controls and allocation were debited by prepare.
        let failure = PlanningFailure {
            cause,
            _funding: self.funding,
        };
        Error::retained_original(SharedBackendFailure::new(kind, failure), state_preserved)
    }
}

/// Recognizes only the actual typed retained owner and exact account in this
/// error's source chain. No global registry or classification grants custody.
pub(in crate::composition::mlx) fn planning_error_has_funding<
    E: StdError + Send + Sync + 'static,
>(
    cause: &Error,
    funding: &WorkspaceMetadataFunding,
) -> bool {
    let mut source: Option<&(dyn StdError + 'static)> = Some(cause);
    while let Some(error) = source {
        if let Some(owned) = error.downcast_ref::<PlanningFailure<E>>() {
            if owned._funding.same_account(funding) {
                return true;
            }
        }
        source = error.source();
    }
    false
}

#[cfg(test)]
mod prepared_failure_tests;
