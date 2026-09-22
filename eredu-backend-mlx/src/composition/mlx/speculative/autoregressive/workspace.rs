//! Actual independent lane/source borrowed through finite equation construction.
use super::*;
use crate::backend::nn::workspace::{
    AutoregressiveEquationRecipe, ProjectedPagedSources, ProjectedResidentState,
};
use crate::composition::mlx::model::{Executable, retain_planning_error};
use eredu_nn::workspace::{
    HostMetadataFunding, HostMetadataFundingError, WorkspaceContext, WorkspaceDtype,
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
mod capture;

/// No recipe can escape independently of its actual selected model, independent
/// cache, prompt and schedule loans. This is an equation producer, not a native
/// fit: checkpoint/replay copies, sampling, role-specific completions and
/// actual request admission remain enclosing requirements.
pub(crate) struct AutoregressiveWorkspaceRecipe<'source, 'capture> {
    schedule: &'source AutoregressiveSchedulePlan<'source>,
    model: &'source Executable,
    state: &'source MlxAutoregressiveState,
    input: Option<&'source MlxModelInput>,
    frontier: u64,
    media_source: Option<eredu_runtime::working_memory::RegisteredPreparedWorkspaceStorage<()>>,
    projected: ProjectedResidentState,
    paged: Option<ProjectedPagedSources>,
    layerwise: Option<crate::backend::runtime::execution::generic::LayerwiseWorkspace>,
    report: InferenceWorkspaceReport,
    recipe: AutoregressiveEquationRecipe,
    capture: Option<capture::Quote<'capture>>,
    context: WorkspaceContext,
    // Every owned report, native inspection clone and Context retires first.
    funding: HostMetadataFunding,
}
impl<'source, 'capture> AutoregressiveWorkspaceRecipe<'source, 'capture> {
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
        metadata_capacity: eredu_core::MemoryLimits,
    ) -> Result<Self, Error> {
        let funding = state
            .pool
            .prepare_workspace_metadata(
                model.erased().inference_execution_identity(),
                metadata_capacity,
            )
            .map_err(Error::WorkspacePlanning)?;
        Self::inspect_funded(
            schedule, model, state, input, invocation, funding, None, None, None,
        )
    }

    // The request source bank supplies one cumulative funding owner across all
    // attempts. Dropping a failed/discarded invocation cannot reset its account.
    fn inspect_funded(
        schedule: &'source AutoregressiveSchedulePlan<'source>,
        model: &'source Executable,
        state: &'source MlxAutoregressiveState,
        input: Option<&'source MlxModelInput>,
        invocation: AutoregressiveInvocation,
        funding: HostMetadataFunding,
        prepared_prefill: Option<&super::prefill_input::PreparedPrefillInput>,
        observation: Option<(
            &'capture eredu_runtime::capture::OriginalSpeculativeCaptureProspect,
            &'source AutoregressiveSourcePair,
        )>,
        partition: Option<&crate::composition::mlx::session::OriginalModelPartitionSource>,
    ) -> Result<Self, Error> {
        let controls = [
            size_of::<Self>(),
            size_of::<Result<Self, Error>>(),
            size_of::<AutoregressiveInvocation>(),
            size_of::<SourceMismatch>(),
            size_of::<Option<&crate::composition::mlx::session::OriginalModelPartitionSource>>(),
            size_of::<Option<crate::backend::nn::workspace::ResidentRecipeRecorder>>(),
            size_of::<Option<crate::backend::nn::workspace::ParallelRecipeRecorder>>(),
            size_of::<&mut dyn crate::composition::mlx::model::CaptureRecorder>(),
            size_of::<Option<crate::composition::mlx::model::NativeLayerwiseParameters<'_>>>(),
            size_of::<eredu_runtime::replicated_session::ReplicatedTextControlOrigin>(),
            size_of::<
                Option<Result<(), eredu_runtime::replicated_session::PreparedControlBindingError>>,
            >(),
            size_of::<ProjectedResidentState>(),
            size_of::<Option<ProjectedPagedSources>>(),
            size_of::<
                Result<
                    Option<ProjectedPagedSources>,
                    crate::backend::runtime::cache::residency::CacheSourceFailure,
                >,
            >(),
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
                HostMetadataFunding,
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
            .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?;
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
        let addressable =
            model.prepare_addressable_workspace_sources(mechanism, &state.pool, &funding)?;
        // Independent banks have their own retained source and child equations.
        // Their exact source adapter below supplies the region quote; they do
        // not claim to execute an ordinary resident expert unit.
        if partition.is_some_and(|source| !source.origin().same_origin(origin))
            || (!model.erased().uses_ordinary_unit_equations()
                && addressable.is_none()
                && partition.is_none())
        {
            return Err(mismatch());
        }
        let frontier = state.native.generation_fixed().ok_or_else(mismatch)?;
        let (dtype, prefill_chunk_positions) = match invocation.execution_pass() {
            eredu_runtime::ExpertPass::Prefill => {
                let input = input.ok_or_else(mismatch)?;
                let media_shape = input
                    .with_borrowed(|input| input.original_media().map(|packet| packet.shape()));
                let (positions, dtype) = match media_shape {
                    Some([1, positions]) => (
                        usize::try_from(positions).map_err(|_| mismatch())?,
                        WorkspaceDtype::Uint32,
                    ),
                    Some(_) => return Err(mismatch()),
                    None => inspect_plain_input(input)
                        .map_err(|cause| retain_planning_error(cause, funding.clone()))?
                        .ok_or_else(mismatch)?,
                };
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
        let mut geometry = schedule
            .workspace_geometry(frontier, invocation, prefill_chunk_positions)
            .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        let capture_sequence = observation.is_some_and(|(source, _)| {
            source
                .invocation()
                .requires_sequence_readout()
                .unwrap_or(true)
        });
        if capture_sequence && invocation.execution_pass() == eredu_runtime::ExpertPass::Prefill {
            geometry.output = eredu_core::OutputDemand::Sequence;
        }
        let workspace = crate::composition::mlx::model::RecipeWorkspace::prepare(
            mechanism,
            addressable.as_ref(),
            None,
            partition.map(|source| source.control().workspace_source().clone()),
            &funding,
        )?;
        let context = workspace.context();
        let layerwise = model
            .prepared_layerwise_workspace(&context)
            .map_err(|cause| {
                retain_planning_error(
                    cause.at_speculative_stage("layerwise source snapshot"),
                    funding.clone(),
                )
            })?;
        model
            .install_parameter_source(&context, layerwise.as_ref())
            .map_err(|cause| {
                retain_planning_error(
                    Error::from(cause).at_speculative_stage("parameter representation source"),
                    funding.clone(),
                )
            })?;
        let mut projected = state.native.project_workspace(&context).map_err(|cause| {
            retain_planning_error(
                cause.at_speculative_stage("AR current state projection"),
                funding.clone(),
            )
        })?;
        if !projected.storage.is_complete() {
            return Err(mismatch());
        }
        let media_packet =
            input.and_then(|input| input.with_borrowed(|input| input.original_media().cloned()));
        let media_semantics = media_packet
            .as_ref()
            .map(|packet| {
                let crate::backend::runtime::media::input::OriginalMediaPacket::Original(_) =
                    packet
                else {
                    return Err(mismatch());
                };
                let semantics = prepared_prefill
                    .and_then(|input| input.media_semantics())
                    .cloned()
                    .unwrap_or_else(|| packet.semantics());
                if !semantics.source().same_source(packet.semantics().source())
                    || !state.native.matches_media_binding(
                        model.erased().inference_execution_identity(),
                        semantics.binding(),
                    )
                    || semantics.binding().frontier() != frontier
                {
                    return Err(mismatch());
                }
                Ok(semantics)
            })
            .transpose()?;
        let media=media_packet.as_ref().map(|packet| {
            let source=packet.clone_lowered().into_prepared();
            eredu_architectures::prepared_execution::OriginalMediaWorkspaceInput::project_with_metadata(
                &source,media_semantics.as_ref().ok_or_else(mismatch)?.clone(),&context,&state.pool,
            ).map_err(|cause|retain_planning_error(cause,funding.clone()))
        }).transpose()?;
        let media_source=media.as_ref().map(|input| {
            let source=input.source_storage();
            let layout=eredu_runtime::working_memory::RegisteredWorkspaceStorageLayout::<()>::new_with_prepared_source(0,source)
                .map_err(Error::PrefillControl)?;
            funding.reserve_metadata(layout.requested_bytes()).map_err(Error::WorkspacePlanning)?;
            layout.construct_with_prepared_source(&state.pool,&context,std::iter::empty::<((), eredu_nn::workspace::WorkspaceExistingStorage)>(),source.clone()).map_err(Error::PrefillControl)
        }).transpose()?;
        let mut local_recorder = if workspace.parallel().is_none() {
            let mut recorder = mechanism
                .recorder(geometry, context)
                .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
            if let Some(source) = addressable {
                recorder
                    .bind_addressable_sources(source)
                    .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
            }
            Some(recorder)
        } else {
            None
        };
        let mut parallel_recorder = workspace
            .parallel()
            .map(|source| source.recorder(geometry))
            .transpose()
            .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        let recorder: &mut dyn crate::composition::mlx::model::CaptureRecorder =
            match (local_recorder.as_mut(), parallel_recorder.as_mut()) {
                (Some(recorder), None) => recorder,
                (None, Some(recorder)) => recorder,
                _ => return Err(mismatch()),
            };
        let parameters = workspace
            .parallel()
            .and(layerwise.as_ref())
            .map(crate::composition::mlx::model::NativeLayerwiseParameters);
        let parameters = parameters.as_ref().map(|parameters| {
            parameters as &dyn eredu_architectures::prepared_execution::WorkspaceLayerwiseParameters
        });
        let communication = workspace
            .parallel()
            .map(|source| source.declaration_source());
        // Match the original text/media composition: record numerical equations
        // here, then bind this retained source's exact unloaded-slot, copy and
        // transfer populations through the selected native operation plan.
        // The layerwise metadata tracer would construct ParameterPlaceholder
        // operations inside each span before replacing them with source rows;
        // those unevaluated constructors are not numerical Eval operations.
        let mut capture_quote = None;
        let report = if let Some((source, sources)) = observation {
            let count = geometry
                .input_positions
                .checked_sub(1)
                .and_then(|n| n.checked_div(geometry.prefill_chunk_positions))
                .and_then(|n| n.checked_add(1))
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(mismatch)?;
            let mut edits = context
                .metadata_vec(count)
                .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
            edits.resize_with(count, || std::cell::RefCell::new(None));
            let mut partitions = context
                .metadata_vec(count)
                .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
            partitions.resize_with(count, || std::cell::RefCell::new(None));
            let mut partition_scalars = context
                .metadata_vec(count)
                .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
            let cells = if sources.partition_source(invocation.source()).is_some() {
                source
                    .invocation()
                    .source()
                    .plan()
                    .admission()
                    .plan()
                    .selections
                    .len()
            } else {
                0
            };
            for _ in 0..count {
                let mut values = context
                    .metadata_vec(cells)
                    .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
                values.resize_with(cells, || std::cell::Cell::new(None));
                partition_scalars.push(values);
            }
            let mut readout_operations = context
                .metadata_vec(count)
                .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
            readout_operations.resize_with(count, || std::cell::Cell::new(None));
            let mut readout_construction = context
                .metadata_vec(count)
                .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
            readout_construction.resize_with(count, || std::cell::Cell::new(None));
            let hosts = std::cell::RefCell::new(
                context
                    .metadata_vec(count)
                    .map_err(|cause| retain_planning_error(cause, funding.clone()))?,
            );
            let mut populations = context
                .metadata_vec(count)
                .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
            let transfers = std::cell::Cell::new(
                crate::backend::array_copy::CaptureNativePopulation::default(),
            );
            let paths = model
                .erased()
                .shared_observation_paths()
                .ok_or_else(mismatch)?;
            model.erased().validate_prepared_observation_paths(
                paths,
                eredu_runtime::working_memory::WorkspaceReportMetadata::new(&context),
            )?;
            let mut observer = capture::Observer::new(
                source,
                invocation,
                geometry,
                sources,
                &context,
                &transfers,
                &edits,
                &hosts,
                &partition_scalars,
                &partitions,
                &readout_operations,
            )?;
            let mut trace = capture::Trace {
                recorder,
                transfers: &transfers,
                scalars: &partition_scalars,
                populations: &mut populations,
                context: &context,
                mechanism,
                readout_operations: &readout_operations,
                readout_construction: &readout_construction,
            };
            let report=blueprint.quote_autoregressive_invocation_observed(
                schedule,frontier,invocation,prefill_chunk_positions,dtype,
                media.zip(media_semantics.as_ref().map(|semantics|semantics.binding())),
                &projected.state,context,parameters,
                Some(eredu_architectures::prepared_execution::InvocationWorkspaceObservation::new(paths,&mut observer,source.invocation().origin().prediction as u64)),
                &mut trace,communication).map_err(|cause|retain_planning_error(cause,funding.clone()))?;
            drop(observer);
            let hosts = hosts.into_inner();
            if hosts.len() != count || populations.len() != count {
                return Err(mismatch());
            }
            capture_quote = Some(capture::Quote {
                source,
                hosts,
                populations,
                edits,
                partitions,
                readout_construction,
            });
            report
        } else {
            blueprint
                .quote_autoregressive_invocation_observed(
                    schedule,
                    frontier,
                    invocation,
                    prefill_chunk_positions,
                    dtype,
                    media.zip(
                        media_semantics
                            .as_ref()
                            .map(|semantics| semantics.binding()),
                    ),
                    &projected.state,
                    context,
                    parameters,
                    None,
                    recorder,
                    communication,
                )
                .map_err(|cause| retain_planning_error(cause, funding.clone()))?
        };
        let recipe = match (local_recorder, parallel_recorder) {
            (Some(recorder), None) => recorder.finish_autoregressive_observed(
                report.span_workspace_plan(),
                invocation,
                capture_sequence,
            ),
            (None, Some(recorder)) => recorder.finish_autoregressive_observed(
                report.span_workspace_plan(),
                invocation,
                capture_sequence,
            ),
            _ => return Err(mismatch()),
        }
        .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        let mut paged = projected
            .storage
            .take_paged_sources(context)
            .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        if let Some(source) = &mut paged {
            source
                .prepare_catalogs(report.span_workspace_plan(), context)
                .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
            source
                .prepare_host_program(&state.pool, context)
                .map_err(|cause| retain_planning_error(cause, funding.clone()))?;
        }
        Ok(Self {
            schedule,
            model,
            state,
            input,
            frontier,
            media_source,
            projected,
            paged,
            layerwise,
            report,
            recipe,
            capture: capture_quote,
            context: workspace.into_context(),
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
