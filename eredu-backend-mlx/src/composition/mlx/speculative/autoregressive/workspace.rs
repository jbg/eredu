//! Actual independent lane/source borrowed through finite equation construction.
use super::*;
use crate::backend::nn::workspace::{
    AutoregressiveEquationRecipe, ProjectedResidentState,
};
use crate::composition::mlx::model::{retain_planning_error, Executable};
use eredu_nn::workspace::{
    WorkspaceContext, WorkspaceDtype, WorkspaceMetadataFunding, WorkspaceMetadataFundingError,
};
use eredu_runtime::{
    speculative::autoregressive::{AutoregressiveInvocation, AutoregressiveSchedulePlan},
    working_memory::InferenceWorkspaceReport,
};
use std::{
    mem::{size_of, size_of_val},
    num::NonZeroU64,
};

#[derive(Debug, thiserror::Error)]
#[error("independent equation source is unavailable or differs from its actual invocation")]
struct SourceMismatch;

/// No recipe can escape independently of its actual selected model, independent
/// cache, prompt and schedule loans. This is an equation producer, not a native
/// fit: checkpoint/replay copies, sampling, role-specific completions and
/// actual request admission remain enclosing requirements.
pub(crate) struct AutoregressiveWorkspaceRecipe<'source> {
    schedule: &'source AutoregressiveSchedulePlan<'source>,
    model: &'source Executable,
    state: &'source MlxAutoregressiveState,
    input: Option<&'source MlxModelInput>,
    frontier: u64,
    projected: ProjectedResidentState,
    layerwise: Option<crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
    report: InferenceWorkspaceReport,
    recipe: AutoregressiveEquationRecipe,
    context: WorkspaceContext,
    // Every owned report, native inspection clone and Context retires first.
    funding: WorkspaceMetadataFunding,
}
impl<'source> AutoregressiveWorkspaceRecipe<'source> {
    /// Runs the actual retained architecture and native fact reducer at the
    /// actual lane frontier. Alternative future states are never fabricated.
    /// One new participating Context is tied to this same model source; it may
    /// not be reused to import parameter facts from the other role.
    pub(crate) fn inspect(
        schedule: &'source AutoregressiveSchedulePlan<'source>,
        model: &'source Executable,
        state: &'source MlxAutoregressiveState,
        input: Option<&'source MlxModelInput>,
        invocation: AutoregressiveInvocation,
        metadata_capacity: u64,
    ) -> Result<Self, Error> {
        let funding = state
            .pool
            .prepare_workspace_metadata(
                model.erased().inference_execution_identity(),
                metadata_capacity,
            )
            .map_err(Error::WorkspacePlanning)?;
        Self::inspect_funded(schedule, model, state, input, invocation, funding, None)
    }

    // The request source bank supplies one cumulative funding owner across all
    // attempts. Dropping a failed/discarded invocation cannot reset its account.
    fn inspect_funded(
        schedule: &'source AutoregressiveSchedulePlan<'source>,
        model: &'source Executable,
        state: &'source MlxAutoregressiveState,
        input: Option<&'source MlxModelInput>,
        invocation: AutoregressiveInvocation,
        funding: WorkspaceMetadataFunding,
        prepared_prefill: Option<&super::prefill_input::PreparedPrefillInput>,
    ) -> Result<Self, Error> {
        let controls = [
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<AutoregressiveInvocation>(),
            size_of::<SourceMismatch>(),
            size_of::<eredu_runtime::replicated_session::ReplicatedTextControlOrigin>(),
            size_of::<
                Option<Result<(), eredu_runtime::replicated_session::PreparedControlBindingError>>,
            >(),
            size_of::<ProjectedResidentState>(),
            size_of::<Option<crate::backend::runtime::execution::generic::LayerwiseWorkspace>>(),
            size_of::<
                Result<
                    Option<crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
                    Error,
                >,
            >(),
            size_of::<Result<ProjectedResidentState, Error>>(),
            size_of::<InferenceWorkspaceReport>(),
            size_of::<AutoregressiveEquationRecipe>(),
            size_of::<Result<AutoregressiveEquationRecipe, eredu_nn::Error>>(),
            size_of::<(u64, WorkspaceDtype)>(),
            size_of::<Result<Option<(usize, WorkspaceDtype)>, safemlx::ArrayDescriptorError>>(),
            size_of::<(
                &AutoregressiveSchedulePlan<'_>,
                &Executable,
                &MlxAutoregressiveState,
                Option<&MlxModelInput>,
                AutoregressiveInvocation,
                WorkspaceMetadataFunding,
            )>(),
        ];
        let bytes = controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .and_then(|n| {
                if input.is_some() {
                    n.checked_add(Array::descriptor_control_bytes()?)
                } else {
                    Some(n)
                }
            })
            .ok_or(Error::WorkspacePlanning(
                WorkspaceMetadataFundingError::Overflow,
            ))?;
        funding
            .reserve_metadata(bytes)
            .map_err(Error::WorkspacePlanning)?;
        let mismatch = || retain_planning_error(SourceMismatch, funding.clone());
        if state.source_role != invocation.source() {
            return Err(mismatch());
        }
        let origin = state.source_origin.as_ref().ok_or_else(mismatch)?;
        model
            .erased()
            .validate_resident_control_origin_fixed(origin)
            .ok_or_else(mismatch)?
            .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        let blueprint = model.inference_blueprint().ok_or_else(mismatch)?;
        let mechanism = model.resident_workspace_mechanisms().ok_or_else(mismatch)?;
        let addressable=model.prepare_addressable_workspace_sources(mechanism,&state.pool,&funding)?;
        // Independent banks have their own retained source and child equations.
        // Their exact source adapter below supplies the region quote; they do
        // not claim to execute an ordinary resident expert unit.
        if !model.erased().uses_ordinary_unit_equations() && addressable.is_none() {
            return Err(mismatch());
        }
        let frontier = state.native.generation_fixed().ok_or_else(mismatch)?;
        let (dtype, prefill_chunk_positions) = match invocation.execution_pass() {
            eredu_runtime::ExpertPass::Prefill => {
                let input = input.ok_or_else(mismatch)?;
                let (positions, dtype) = inspect_plain_input(input)
                    .map_err(|cause| retain_planning_error(cause, funding.clone()))?
                    .ok_or_else(mismatch)?;
                if positions != invocation.positions() {
                    return Err(mismatch());
                }
                let dtype = match prepared_prefill {
                    Some(prepared)
                        if usize::try_from(prepared.positions()).ok() == Some(positions) =>
                    {
                        prepared.dtype()
                    }
                    Some(_) => return Err(mismatch()),
                    None => dtype,
                };
                let chunk = input
                    .with_borrowed(|input| input.prefill_chunk_positions())
                    .map(NonZeroU64::get)
                    .unwrap_or(eredu_runtime::prefill::DEFAULT_PREFILL_CHUNK_POSITIONS)
                    .min(u64::try_from(positions).map_err(|_| mismatch())?);
                (dtype, NonZeroU64::new(chunk).ok_or_else(mismatch)?)
            }
            eredu_runtime::ExpertPass::Decode => {
                if input.is_some() {
                    return Err(mismatch());
                }
                // Exactly Array::from_slice::<u32> in the shared native decoder.
                (
                    WorkspaceDtype::Uint32,
                    NonZeroU64::new(u64::try_from(invocation.positions()).map_err(|_| mismatch())?)
                        .ok_or_else(mismatch)?,
                )
            }
            _ => return Err(mismatch()),
        };
        let geometry = schedule
            .workspace_geometry(frontier, invocation, prefill_chunk_positions)
            .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        let context = match &addressable {
            Some(source)=>WorkspaceContext::new_with_metadata_funding(
                crate::backend::nn::workspace::MlxAddressableWorkspaceMechanisms::new(mechanism,source.clone()),funding.clone()),
            None=>WorkspaceContext::new_with_metadata_funding(mechanism, funding.clone()),
        }.map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        let layerwise = model
            .prepared_layerwise_workspace(&context)
            .map_err(|cause| {
                retain_planning_error(
                    cause.at_speculative_stage("layerwise source snapshot"),
                    funding.clone(),
                )
            })?;
        model
            .erased()
            .install_workspace_parameter_representations(&context)
            .map_err(|cause| {
                retain_planning_error(
                    Error::from(cause).at_speculative_stage("parameter representation source"),
                    funding.clone(),
                )
            })?;
        let projected = state
            .native
            .project_workspace(&context)
            .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        if !projected.storage.is_complete() {
            return Err(mismatch());
        }
        let mut recorder = mechanism.recorder(geometry, &context)
            .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        if let Some(source)=addressable {
            recorder.bind_addressable_sources(source)
                .map_err(|cause|retain_planning_error(cause,funding.clone()))?;
        }
        // Match the original text/media composition: record numerical equations
        // here, then bind this retained source's exact unloaded-slot, copy and
        // transfer populations through the selected native operation plan.
        // The layerwise metadata tracer would construct ParameterPlaceholder
        // operations inside each span before replacing them with source rows;
        // those unevaluated constructors are not numerical Eval operations.
        let report = blueprint
            .quote_ordinary_autoregressive_invocation(
                schedule,
                frontier,
                invocation,
                prefill_chunk_positions,
                dtype,
                &projected.state,
                &context,
                &mut recorder,
            )
            .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        let recipe = recorder
            .finish_autoregressive(report.span_workspace_plan(), invocation)
            .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        Ok(Self {
            schedule,
            model,
            state,
            input,
            frontier,
            projected,
            layerwise,
            report,
            recipe,
            context,
            funding,
        })
    }

    pub(crate) fn recipe(&self) -> &AutoregressiveEquationRecipe {
        &self.recipe
    }
    pub(crate) fn report(&self) -> &InferenceWorkspaceReport {
        &self.report
    }
    pub(crate) fn projected_source(&self) -> &ProjectedResidentState {
        &self.projected
    }
    /// Address/frontier check while the immutable loans remain live. This does
    /// not validate parameter epochs or authorize a later mutable source handoff.
    pub(crate) fn matches_source_addresses(
        &self,
        model: &Executable,
        state: &MlxAutoregressiveState,
    ) -> bool {
        std::ptr::eq(self.model, model)
            && std::ptr::eq(self.state, state)
            && state.native.generation_fixed() == Some(self.frontier)
    }
}

mod prepared;
pub(crate) use prepared::{
    ActiveSpeculativeInvocation, ActiveSpeculativePrefill, AutoregressiveSourcePair,
    PreparedAutoregressiveInvocation,
};
