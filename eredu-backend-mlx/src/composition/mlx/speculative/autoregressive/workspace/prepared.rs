//! Mutable source handoff and actual once-only verification payload storage.
use super::*;
use eredu_runtime::working_memory::{
    OriginalSpeculativeBudgetCustody, OriginalSpeculativeRequest, OriginalSpeculativeRole,
    SpeculativeInvocationRequirements, WorkingMemoryError,
};
use eredu_runtime::{
    replicated_session::ReplicatedTextControlOrigin,
    speculative::autoregressive::{AutoregressiveOccurrenceClaim, AutoregressiveScheduleIdentity},
};
use safemlx::OriginalScopeObserver;
mod native;
pub(crate) use native::{ActiveSpeculativeInvocation, ActiveSpeculativePrefill};

#[derive(Debug, thiserror::Error)]
enum PreparationCause {
    #[error("independent invocation no longer matches its claimed source frontier")]
    Source,
}

/// Exact Target/Draft epochs captured from the actual pair chosen by the
/// caller. The request identity survives moving its schedule into the driver;
/// each invocation revalidates both its assigned role and actual current epoch.
/// No native array, state or work authority is created by this source binding.
pub(crate) struct AutoregressiveSourcePair {
    draft: ReplicatedTextControlOrigin,
    target_media: Option<eredu_runtime::working_memory::MediaSessionBinding>,
    schedule: AutoregressiveScheduleIdentity,
    active: std::cell::RefCell<Option<ActiveSpeculativeInvocation>>,
    prefill: std::cell::RefCell<Option<ActiveSpeculativePrefill>>,
    numerical: crate::composition::mlx::speculative::OriginalSpeculativeNumericalSources,
}
impl AutoregressiveSourcePair {
    pub(crate) fn prepare(
        target: &Executable,
        draft: &Executable,
        schedule: &AutoregressiveSchedulePlan<'_>,
        pool: &WorkingMemoryPool,
        metadata_capacity: u64,
    ) -> Result<Self, Error> {
        let funding = pool
            .prepare_workspace_metadata(
                target.erased().inference_execution_identity(),
                metadata_capacity,
            )
            .map_err(Error::WorkspacePlanning)?;
        Self::prepare_funded(target, draft, schedule, pool, metadata_capacity, funding)
    }
    /// Reuses the actual request account; common numerical ownership preserves
    /// this exact AR request while its model occurrence slots remain here.
    pub(crate) fn prepare_funded(
        target: &Executable,
        draft: &Executable,
        schedule: &AutoregressiveSchedulePlan<'_>,
        pool: &WorkingMemoryPool,
        metadata_capacity: u64,
        funding: HostMetadataFunding,
    ) -> Result<Self, Error> {
        let controls = [
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<ReplicatedTextControlOrigin>(),
            size_of::<Result<eredu_runtime::working_memory::MediaSessionBinding, WorkingMemoryError>>(),
            size_of::<
                Option<
                    Result<
                        ReplicatedTextControlOrigin,
                        eredu_runtime::replicated_session::PreparedControlBindingError,
                    >,
                >,
            >(),
            size_of::<(
                &Executable,
                &Executable,
                &AutoregressiveSchedulePlan<'_>,
                &WorkingMemoryPool,
                u64,
            )>(),
        ];
        let bytes = controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .ok_or(Error::WorkspacePlanning(
                HostMetadataFundingError::Overflow,
            ))?;
        funding
            .reserve_metadata(bytes)
            .map_err(Error::WorkspacePlanning)?;
        let draft_origin = draft
            .erased()
            .resident_control_origin_fixed()
            .ok_or_else(|| retain_planning_error(PreparationCause::Source, funding.clone()))?
            .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        let request = OriginalSpeculativeRequest::prepare(
            pool,
            target.erased().inference_execution_identity(),
            schedule,
            metadata_capacity,
        )
        .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        let numerical =
            crate::composition::mlx::speculative::OriginalSpeculativeNumericalSources::prepare(
                target, request, pool, funding,
            )?;
        Ok(Self {
            draft: draft_origin,
            target_media: target.erased().original_request_media_binding().ok(),
            schedule: schedule.identity(),
            active: std::cell::RefCell::new(None),
            prefill: std::cell::RefCell::new(None),
            numerical,
        })
    }
    pub(crate) fn validate_media_input(
        &self, semantics: &eredu_architectures::media_plan::BoundPreparedMediaSemantics,
    ) -> Result<(), Error> {
        if self.target_media.as_ref().is_some_and(|binding| semantics.binding().matches(binding)) {
            Ok(())
        } else {
            Err(self.retain_startup_error(PreparationCause::Source))
        }
    }
    pub(crate) fn numerical_sources(
        &self,
    ) -> &crate::composition::mlx::speculative::OriginalSpeculativeNumericalSources {
        &self.numerical
    }
    pub(crate) fn with_intervention_declaration(
        mut self,
        declaration: Option<crate::composition::mlx::session::OriginalInterventionDeclaration>,
    ) -> Result<Self, Error> {
        self.numerical = self.numerical.with_intervention_declaration(declaration)?;
        Ok(self)
    }
    pub(crate) fn validate_intervention(
        &self,
        plan: &eredu_core::intervention::AdmittedInterventionPlan,
    ) -> Result<(), Error> {
        self.numerical.validate_intervention(plan)
    }
    pub(crate) fn retain_error(&self, cause: Error) -> Error {
        self.numerical.retain_error(cause)
    }
    pub(crate) fn retain_startup_error<E: std::error::Error + Send + Sync + 'static>(
        &self,
        cause: E,
    ) -> Error {
        self.numerical.retain_startup_error(cause)
    }
    pub(crate) fn numerical_prerequisites(
        &self,
    ) -> (
        &safemlx::PrefillRootsRuntime,
        crate::backend::nn::workspace::MlxMetalWorkspaceMechanisms,
    ) {
        self.numerical.numerical_prerequisites()
    }
    pub(crate) fn metadata_funding(&self) -> &HostMetadataFunding {
        self.numerical.metadata_funding()
    }
    pub(crate) fn schedule_identity(&self) -> AutoregressiveScheduleIdentity {
        self.schedule
    }
    pub(crate) fn request(&self) -> &OriginalSpeculativeRequest {
        self.numerical.request()
    }
    pub(crate) fn validate_environment(
        &self,
        environment: &crate::backend::OriginalCopyEnvironment<'_>,
    ) -> Result<(), Error> {
        self.numerical.validate_environment(environment)
    }
    pub(crate) fn validate_source(
        &self,
        model: &Executable,
        source: AutoregressiveSource,
    ) -> Result<(), Error> {
        model
            .erased()
            .validate_resident_control_origin_fixed(self.origin(source))
            .ok_or_else(|| self.retain_startup_error(PreparationCause::Source))?
            .map_err(|cause| self.retain_startup_error(cause))
    }
    pub(crate) fn matches_state_origin(
        &self,
        source: AutoregressiveSource,
        origin: Option<&ReplicatedTextControlOrigin>,
    ) -> bool {
        origin.is_some_and(|origin| self.origin(source).same_origin(origin))
    }
    fn origin(&self, source: AutoregressiveSource) -> &ReplicatedTextControlOrigin {
        match source {
            AutoregressiveSource::Target => self.numerical.target_origin(),
            AutoregressiveSource::Draft => &self.draft,
        }
    }
}

/// The exact actual source cannot mutate between equation tracing and native
/// preparation. No cached future-state recipe, source address, or logical shape
/// substitutes for these exclusive loans and the cache's parameter epoch.
///
/// Native preparation consumes the complete equation and exact source loans;
/// its private role bank and output completion are constructed after admission.
pub(crate) struct PreparedAutoregressiveInvocation<'source> {
    sources: &'source AutoregressiveSourcePair,
    model: &'source mut Executable,
    state: &'source mut MlxAutoregressiveState,
    input: Option<&'source MlxModelInput>,
    prepared_prefill: Option<super::super::prefill_input::PreparedPrefillInput>,
    claim: AutoregressiveOccurrenceClaim<'source>,
    source_origin: ReplicatedTextControlOrigin,
    group_control_bytes: u64,
    group_source_facts: Option<eredu_runtime::working_memory::HostSourceConstructionFacts>,
    media_source: Option<eredu_runtime::working_memory::RegisteredPreparedWorkspaceStorage<()>>,
    projected: ProjectedResidentState,
    layerwise: Option<crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
    report: InferenceWorkspaceReport,
    recipe: AutoregressiveEquationRecipe,
    context: WorkspaceContext,
    // All actual payload buffers, inspection handles and error prefixes first.
    funding: HostMetadataFunding,
}
impl<'source> PreparedAutoregressiveInvocation<'source> {
    /// Consumes the already-spent claim before any preparation. Refusal or drop
    /// never returns it to the request cursor, including after cache rollback.
    pub(crate) fn prepare(
        sources: &'source AutoregressiveSourcePair,
        model: &'source mut Executable,
        state: &'source mut MlxAutoregressiveState,
        input: Option<&'source MlxModelInput>,
        claim: AutoregressiveOccurrenceClaim<'source>,
        prepared_prefill: Option<super::super::prefill_input::PreparedPrefillInput>,
    ) -> Result<Self, Error> {
        if !claim.belongs_to(sources.schedule) || state.source_role != claim.invocation().source() {
            return Err(retain_planning_error(
                PreparationCause::Source,
                sources.metadata_funding().clone(),
            ));
        }
        sources.validate_source(model, claim.invocation().source())?;
        let mut quote = AutoregressiveWorkspaceRecipe::inspect_funded(
            claim.schedule(),
            model,
            state,
            input,
            claim.invocation(),
            sources.metadata_funding().clone(),
            prepared_prefill.as_ref(),
        )?;
        let controls = [
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<AutoregressiveOccurrenceClaim<'_>>(),
            size_of::<ReplicatedTextControlOrigin>(),
            size_of::<PreparationCause>(),
            size_of::<Option<eredu_runtime::working_memory::HostSourceConstructionFacts>>(),
            size_of::<(u64, Option<eredu_runtime::working_memory::HostSourceConstructionFacts>)>(),
            size_of::<Result<(u64, Option<eredu_runtime::working_memory::HostSourceConstructionFacts>), Error>>(),
            size_of::<(
                &mut Executable,
                &mut MlxAutoregressiveState,
                Option<&MlxModelInput>,
                AutoregressiveOccurrenceClaim<'_>,
                &AutoregressiveSourcePair,
            )>(),
        ];
        let bytes = controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .ok_or(Error::WorkspacePlanning(
                HostMetadataFundingError::Overflow,
            ))?;
        quote
            .funding
            .reserve_metadata(bytes)
            .map_err(Error::WorkspacePlanning)?;
        if quote.frontier != claim.frontier() {
            return Err(retain_planning_error(
                PreparationCause::Source,
                quote.funding.clone(),
            ));
        }
        // The source was validated before projection and remains borrowed here.
        let source_origin = quote
            .state
            .source_origin
            .as_ref()
            .expect("validated source epoch")
            .clone();
        let (group_control_bytes, group_source_facts) = quote
            .model
            .erased()
            .bind_speculative_neural_recipe(quote.layerwise.as_ref(), sources.numerical_sources().pool(), &quote.funding, &mut quote.recipe)
            .map_err(|cause| retain_planning_error(cause, quote.funding.clone()))?;
        let AutoregressiveWorkspaceRecipe {
            media_source,
            projected,
            layerwise,
            report,
            recipe,
            context,
            funding,
            schedule: _,
            model: _,
            state: _,
            input: _,
            frontier: _,
        } = quote;
        // The immutable source loan has ended without any intervening mutation.
        // The same references now become the exclusive native preparation loan.
        Ok(Self {
            sources,
            model,
            state,
            input,
            prepared_prefill,
            claim,
            source_origin,
            group_control_bytes,
            group_source_facts,
            media_source,
            projected,
            layerwise,
            report,
            recipe,
            context,
            funding,
        })
    }

    pub(crate) fn invocation(&self) -> AutoregressiveInvocation {
        self.claim.invocation()
    }
    pub(crate) fn attempted_ordinal(&self) -> usize {
        self.claim.ordinal()
    }
    pub(crate) fn recipe(&self) -> &AutoregressiveEquationRecipe {
        &self.recipe
    }
    pub(crate) fn report(&self) -> &InferenceWorkspaceReport {
        &self.report
    }
}
