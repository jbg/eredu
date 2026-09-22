//! The architecture-declared provider votes use the same selected group sources.
use super::*;
use crate::backend::nn::workspace::LogicalCollectiveQuote;
use crate::backend::runtime::distributed::topology::original_source::parallel::RetainedLogicalCollective;
use eredu_nn::workspace::WorkspaceAllocationPopulation;
pub(crate) struct ExpertProviderVote {
    pub(crate) group: eredu_core::CollectiveGroupId,
    pub(crate) order: usize,
    pub(crate) peers: usize,
    pub(crate) backing: usize,
    pub(crate) logical: Option<RetainedLogicalCollective>,
    pub(crate) native_scratch: Option<WorkspaceAllocationPopulation>,
}
pub(crate) struct ExpertProviderQuote {
    votes: Vec<ExpertProviderVote>,
    pub(crate) ordinary_seed_scratch: Option<WorkspaceAllocationPopulation>,
    pub(crate) scratch: Option<WorkspaceAllocationPopulation>,
    pub(crate) backing: u64,
    pub(crate) births: usize,
    pub(crate) ordinary: Option<OrdinaryParallelControls>,
}
impl ExpertProviderQuote {
    pub(super) fn prepare(local: &ExpertLocalQuote) -> Result<Self, Error> {
        let view = local.declaration.as_view();
        Self::prepare_for(
            &local.source,
            local.mechanism,
            [
                view.provider_tensor_group,
                Some(view.group),
                view.provider_wave_group,
            ],
        )
    }
    pub(super) fn prepare_for(
        source: &OriginalParallelSource,
        mechanism: ResidentExecutionMechanisms,
        groups: [Option<eredu_core::CollectiveGroupId>; 3],
    ) -> Result<Self, Error> {
        let context =
            WorkspaceContext::new_with_metadata_funding(mechanism, source.funding().clone())?;
        context.charge_metadata(size_of::<(
            Self,
            ExpertProviderVote,
            Result<Self, Error>,
            WorkspaceContext,
            [i32; 1],
            [Option<eredu_core::CollectiveGroupId>; 3],
        )>())?;
        let invalid = || {
            context.metadata_error(format_args!(
                "expert provider vote lacks its selected source"
            ))
        };
        let actual = source
            .communication_source()
            .map_err(|cause| source.neural_error(cause))?;
        let mut ordinary_seed_scratch = None;
        let ordinary_seed = if mechanism.allocation().original_storage {
            None
        } else {
            context.charge_metadata(size_of::<(
                WorkspaceTraceReport,
                SpeculativeNumericalRecipe,
                WorkspaceTensor,
                Option<SpeculativeNumericalRecipe>,
                Option<WorkspaceAllocationPopulation>,
            )>())?;
            context.begin_span();
            let seed = crate::backend::nn::workspace::host_array::trace(
                &[1],
                safemlx::Dtype::Int32,
                &context,
            )?;
            ordinary_seed_scratch = context.new_allocation_scratch()?;
            let report = context.finish_report(&[seed])?;
            Some(super::super::numerical(&report, 1, mechanism, &context)?)
        };
        let mut votes = context.metadata_vec(groups.into_iter().flatten().count())?;
        let mut backing = 0u64;
        let mut births = 0usize;
        let mut ordinary = Some(OrdinaryParallelControls::default());
        for id in groups.into_iter().flatten() {
            let selected = actual
                .source()
                .manifest()
                .select_group_operation(id, eredu_runtime::CommunicationOperation::FailureAgreement)
                .map_err(|cause| context.metadata_source(cause))?;
            let order = selected.order();
            let (group, descriptor, _) = actual.group(order).ok_or_else(invalid)?;
            if !selected.requirement().exact_completion()
                || descriptor.local_index() != Some(group.rank())
            {
                return Err(invalid());
            }
            let mut native_scratch = None;
            let (mut bytes, mut created, logical) = if group.is_logical() {
                let layout = context.layout(&[1], WorkspaceDtype::Int32)?;
                let quote = LogicalCollectiveQuote::prepare_agreement(
                    source,
                    order,
                    layout.as_view(),
                    mechanism,
                )?;
                let bytes = quote
                    .output
                    .checked_add(quote.scratch)
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or_else(invalid)?;
                let created = quote.maximum_backing_births().ok_or_else(invalid)?;
                ordinary = ordinary.and_then(|prior| prior.append(quote.ordinary_controls()?));
                let quote = RetainedLogicalCollective::retain_prepared(source, quote)
                    .map_err(|cause| source.neural_error(cause))?;
                (bytes, created, Some(quote))
            } else {
                let persistent = actual
                    .group_persistent(order)
                    .map_err(|cause| source.neural_error(cause))?;
                let native = group.native_group();
                if persistent.native().has_unqualified_storage()
                    || !persistent.native().is_for(native)
                    || !persistent.source().same_source(actual.source())
                {
                    return Err(invalid());
                }
                context.charge_metadata(
                    native
                        .cpu_layout_storage_control_bytes()
                        .ok_or_else(invalid)?,
                )?;
                let layout = native
                    .cpu_layout_storage(
                        &[1],
                        safemlx::Dtype::Int32,
                        safemlx::distributed::GroupWorkerOperation::Sum,
                    )
                    .map_err(|cause| context.metadata_source(cause))?;
                context.charge_metadata(layout.backing_control_bytes().ok_or_else(invalid)?)?;
                let runtime = source.agreement_inputs().ok_or_else(invalid)?.runtime();
                let bytes = layout
                    .backing_capacity(runtime)
                    .map_err(|cause| context.metadata_source(cause))?;
                let created = layout.evaluation().logical_backing_population().0;
                native_scratch = Some(super::super::scratch::native_cpu(
                    &context, mechanism, bytes, created,
                )?);
                context.charge_metadata(
                    layout
                        .completion_layout_control_bytes()
                        .ok_or_else(invalid)?,
                )?;
                let completed = layout
                    .with_completion_layout()
                    .map_err(|cause| context.metadata_source(cause))?;
                ordinary = ordinary.and_then(|prior| {
                    prior.append(OrdinaryParallelControls::group(
                        completed.ordinary_controls()?,
                    )?)
                });
                (bytes, created, None)
            };
            if let Some(seed) = ordinary_seed {
                bytes = bytes
                    .checked_add(
                        usize::try_from(seed.storage.mutable_bytes()).map_err(|_| invalid())?,
                    )
                    .ok_or_else(invalid)?;
                created = created
                    .checked_add(seed.storage.maximum_births())
                    .ok_or_else(invalid)?;
                ordinary=ordinary.and_then(|prior|{
                    use crate::backend::runtime::distributed::completion::MlxCommunicationCompletion;
                    let mut value=prior.append(OrdinaryParallelControls::constructed(seed)?)?;
                    value.calls=value.calls.append(MlxCommunicationCompletion::ordinary_collective_completion_call_controls(group,&[])?)?
                        .metadata(MlxCommunicationCompletion::ordinary_scalar_result_control_bytes()?)?;
                    value.completion_roots=value.completion_roots.max(1);
                    value.completions=value.completions.checked_add(1)?;
                    Some(value)
                });
            }
            backing = backing
                .checked_add(u64::try_from(bytes).map_err(|_| invalid())?)
                .ok_or_else(invalid)?;
            births = births.checked_add(created).ok_or_else(invalid)?;
            votes.push(ExpertProviderVote {
                group: id,
                order,
                peers: group.size(),
                backing: bytes,
                logical,
                native_scratch,
            });
        }
        let mut result = Self {
            votes,
            scratch: None,
            ordinary_seed_scratch,
            backing,
            births,
            ordinary,
        };
        result.scratch = result.allocation_scratch(&context, mechanism)?;
        Ok(result)
    }
    pub(crate) fn allocation_scratch(
        &self,
        context: &WorkspaceContext,
        mechanism: ResidentExecutionMechanisms,
    ) -> Result<Option<WorkspaceAllocationPopulation>, Error> {
        let count = self
            .votes
            .len()
            .checked_mul(2)
            .ok_or(WorkspaceMetadataError::Overflow)?;
        let mut populations = context.metadata_vec(count)?;
        for vote in &self.votes {
            let population = if let Some(logical) = &vote.logical {
                logical.value().allocation_scratch(context, mechanism)?
            } else {
                vote.native_scratch.clone()
            };
            let Some(population) = population else {
                return Ok(None);
            };
            populations.push(population);
            if let Some(seed) = &self.ordinary_seed_scratch {
                populations.push(seed.clone());
            }
        }
        let mut sources = context.metadata_vec(populations.len())?;
        sources.extend(populations.iter().map(|source| (source, 1)));
        context.combine_scratch_populations(&sources).map(Some)
    }
    pub(crate) fn votes(&self) -> impl Iterator<Item = &ExpertProviderVote> {
        self.votes.iter()
    }
    pub(crate) fn len(&self) -> usize {
        self.votes.len()
    }
    pub(crate) fn vote(&self, index: usize) -> Option<&ExpertProviderVote> {
        self.votes.get(index)
    }
}
