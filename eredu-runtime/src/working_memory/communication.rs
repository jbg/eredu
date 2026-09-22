//! Metadata realization of the existing partition communication driver.
//! Immediate completion means only that descriptors were recorded. The selected
//! native source must independently provide every transfer and control bound.
use crate::*;
use eredu_core::{BackendFailure, Submission, checkpoint::TensorDtype};
use eredu_nn::{DistributedNeuralBackend, Error, Tensor, workspace::*};
use std::mem::size_of;

#[cfg(test)]
#[path = "communication_completion_tests.rs"]
mod completion_tests;

/// One actual retained group declaration, lent to the metadata traversal.
#[derive(Debug, Clone)]
pub struct WorkspaceCommunicationGroup {
    source: RetainedCommunicationSource,
    order: usize,
    context: WorkspaceContext,
    terminal: std::rc::Rc<std::cell::Cell<bool>>,
}
/// One actual retained route declaration, lent to the metadata traversal.
#[derive(Debug, Clone)]
pub struct WorkspaceCommunicationRoute {
    source: RetainedCommunicationSource,
    order: usize,
    context: WorkspaceContext,
}
/// Allocation-free scalar/shape inspection for cold partition validation.
#[derive(Debug, Clone, Copy)]
pub struct WorkspaceCommunicationMetadata;

type Communication = PartitionCommunication<
    WorkspaceBackend,
    WorkspaceCommunicationGroup,
    WorkspaceCommunicationRoute,
    WorkspaceCommunicationMetadata,
>;

/// Binds the same checked declaration source used by native realization. All
/// copying is paid from this exact context; no native resources are constructed.
pub fn workspace_partition_communication(
    source: &RetainedCommunicationSource,
    context: &WorkspaceContext,
) -> Result<Communication, Error> {
    context.charge_metadata(size_of::<(
        Communication,
        Result<Communication, Error>,
        &RetainedCommunicationSource,
        &WorkspaceContext,
        PartitionCommunicationAuthority,
    )>())?;
    context.charge_metadata(
        source
            .manifest()
            .retention_copy_bytes()
            .ok_or(WorkspaceMetadataError::Overflow)?,
    )?;
    let manifest = source
        .manifest()
        .try_copy_for_retention()
        .map_err(|cause| context.metadata_source(cause))?;
    let mut groups = context.metadata_vec(manifest.groups().len())?;
    for (order, group) in manifest.groups().iter().enumerate() {
        context.charge_metadata(
            WorkspaceContext::metadata_rc_bytes::<std::cell::Cell<bool>>()
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        groups.push(RealizedCommunicationGroup::new(
            group.id(),
            WorkspaceCommunicationGroup {
                source: source.clone(),
                order,
                context: context.clone(),
                terminal: std::rc::Rc::new(std::cell::Cell::new(false)),
            },
        ));
    }
    let mut routes = context.metadata_vec(manifest.routes().len())?;
    for (order, route) in manifest.routes().iter().enumerate() {
        routes.push(RealizedCommunicationRoute::new(
            route.id(),
            WorkspaceCommunicationRoute {
                source: source.clone(),
                order,
                context: context.clone(),
            },
        ));
    }
    context.charge_metadata(
        PartitionCommunicationAuthority::shared_control_bytes()
            .checked_add(size_of::<Result<Communication, PartitionExecutionError>>())
            .ok_or(WorkspaceMetadataError::Overflow)?,
    )?;
    PartitionCommunication::new(manifest, groups, routes, WorkspaceCommunicationMetadata)
        .map_err(|cause| context.metadata_source(cause))
}
impl WorkspaceCommunicationGroup {
    fn select(
        &self,
        operation: CommunicationOperation,
        context: &WorkspaceContext,
    ) -> Result<&CommunicationGroupDescriptor, Error> {
        context.charge_metadata(
            CommunicationManifest::group_operation_control_bytes()
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        if self.terminal.get() || !context.shares_trace(&self.context) {
            return Err(context.metadata_error(format_args!(
                "cold communication group belongs to another trace"
            )));
        }
        let descriptor = self
            .source
            .group(self.order)
            .ok_or_else(|| {
                context.metadata_error(format_args!("cold communication group source is absent"))
            })?
            .0;
        let selected = self
            .source
            .manifest()
            .select_group_operation(descriptor.id(), operation)
            .map_err(|cause| context.metadata_source(cause))?;
        Ok(selected.descriptor())
    }
}
impl WorkspaceCommunicationRoute {
    fn descriptor(&self) -> &CommunicationRouteDescriptor {
        self.source
            .route(self.order)
            .expect("closed retained route index")
            .0
    }
    fn validate(&self, context: &WorkspaceContext) -> Result<(), Error> {
        if !context.shares_trace(&self.context) {
            return Err(context
                .metadata_error(format_args!("cold boundary route belongs to another trace")));
        }
        let descriptor = self.descriptor();
        let rank = self.source.manifest().rank();
        if descriptor.source() != rank && descriptor.destination() != rank {
            return Err(context.metadata_error(format_args!(
                "cold boundary rank is not a selected endpoint"
            )));
        }
        Ok(())
    }
}
fn completed<T>(
    output: T,
    context: &WorkspaceContext,
) -> Result<Submission<T, super::WorkspaceCompletion>, Error> {
    context.charge_metadata(size_of::<(
        T,
        Submission<T, super::WorkspaceCompletion>,
        Result<Submission<T, super::WorkspaceCompletion>, Error>,
    )>())?;
    Ok(Submission {
        output,
        completion: super::WorkspaceCompletion::recorded(context)?,
    })
}
fn enclosing_submission(context: &WorkspaceContext) -> bool {
    context.completion_strategy() == WorkspaceCompletionStrategy::EnclosingSubmission
}
#[derive(Debug, thiserror::Error)]
enum SourceCause {
    #[error("cold boundary control context differs from its retained world")]
    Identity,
    #[error(transparent)]
    Frame(PreparedBoundaryFrameError),
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct SourceFailure {
    #[source]
    cause: SourceCause,
    source: RetainedCommunicationSource,
    funding: HostMetadataFunding,
}
impl TerminalCommunicationBackend for WorkspaceBackend {
    fn mark_terminal_submission(group: &WorkspaceCommunicationGroup) {
        group.terminal.set(true);
    }
}
impl CommunicationBackend for WorkspaceBackend {
    fn record_model_control_phase(
        group_id: eredu_core::CollectiveGroupId,
        group: &WorkspaceCommunicationGroup,
        phase: DistributedExecutionPhase,
        executor: &WorkspaceContext,
    ) -> Result<(), Error> {
        let selected = group.select(CommunicationOperation::FailureAgreement, executor)?;
        if selected.id() != group_id {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        if let DistributedExecutionPhase::BoundarySourceCompletion(route)
        | DistributedExecutionPhase::BoundarySourceReady(route) = phase
        {
            if !group
                .source
                .manifest()
                .routes()
                .iter()
                .any(|row| row.id() == route)
            {
                return Err(WorkspaceMetadataError::Unqualified.into());
            }
        }
        let phase = match phase {
            DistributedExecutionPhase::Execution => WorkspaceModelControlPhase::Execution,
            DistributedExecutionPhase::BoundarySourceCompletion(route) => {
                WorkspaceModelControlPhase::BoundarySourceCompletion {
                    route: route.value(),
                }
            }
            DistributedExecutionPhase::BoundarySourceReady(route) => {
                WorkspaceModelControlPhase::BoundarySourceReady {
                    route: route.value(),
                }
            }
            _ => return Err(WorkspaceMetadataError::Unqualified.into()),
        };
        executor.record_model_control(WorkspaceModelControl {
            group: selected.id(),
            phase,
        })
    }

    type CommunicationGroup = WorkspaceCommunicationGroup;
    type CommunicationRoute = WorkspaceCommunicationRoute;
    type CommunicationCompletion = super::WorkspaceCompletion;
    type CommunicationError = Error;

    fn with_expert_inactive_wave<E, F>(
        source: Option<eredu_nn::workspace::WorkspaceExpertInactiveWave>,
        _context: Option<&WorkspaceParallelContext>,
        executor: &WorkspaceContext,
        _run: F,
    ) -> Result<Result<(), E>, Error>
    where
        F: FnOnce() -> Result<(), E>,
    {
        eredu_nn::workspace::record_expert_inactive_wave(
            source.ok_or(eredu_nn::workspace::WorkspaceMetadataError::Unqualified)?,
            executor,
        )?;
        Ok(Ok(()))
    }
    fn with_expert_provider_wave<E, F>(
        source: eredu_nn::workspace::WorkspaceExpertProviderWave,
        _context: Option<&WorkspaceParallelContext>,
        executor: &WorkspaceContext,
        _run: F,
    ) -> Result<Result<(), E>, Error>
    where
        F: FnOnce() -> Result<(), E>,
    {
        eredu_nn::workspace::record_expert_provider_wave(source, executor)?;
        Ok(Ok(()))
    }

    fn with_expert_route_region<P, E, F>(
        source: eredu_nn::workspace::WorkspaceExpertRegionView<'_>,
        bank: &mut P,
        input: &WorkspaceTensor,
        routes: &eredu_nn::GroupSelection<WorkspaceTensor>,
        _context: Option<&WorkspaceParallelContext>,
        executor: &WorkspaceContext,
        _run: F,
    ) -> Result<Result<crate::RoutedExpertTensorParallelOutput<WorkspaceTensor>, E>, Error>
    where
        P: eredu_nn::Parameterized<WorkspaceTensor>,
        F: FnOnce(
            &mut P,
            Option<crate::PreparedExpertMovementLoan<'_>>,
        ) -> Result<crate::RoutedExpertTensorParallelOutput<WorkspaceTensor>, E>,
    {
        let output =
            eredu_nn::workspace::record_expert_region(source, bank, input, routes, executor)?;
        Ok(Ok(if source.tensor_partitions.is_some() {
            crate::RoutedExpertTensorParallelOutput::Partial(output)
        } else {
            let (value, bias) = output.into_parts();
            if bias.is_some() {
                return Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into());
            }
            crate::RoutedExpertTensorParallelOutput::Complete(value)
        }))
    }

    fn with_observed_expert_route_region<'observer, P, E, F>(
        source: eredu_nn::workspace::WorkspaceExpertRegionView<'_>,
        bank: &mut P,
        input: &WorkspaceTensor,
        routes: &eredu_nn::GroupSelection<WorkspaceTensor>,
        _context: Option<&WorkspaceParallelContext>,
        executor: &WorkspaceContext,
        mut observer: Option<&'observer mut dyn crate::RoutedUnitObserver<WorkspaceTensor>>,
        run: F,
    ) -> Result<Result<crate::RoutedExpertTensorParallelOutput<WorkspaceTensor>, E>, Error>
    where
        P: eredu_nn::Parameterized<WorkspaceTensor>,
        F: FnOnce(
            &mut P,
            Option<crate::PreparedExpertMovementLoan<'_>>,
            Option<&'observer mut dyn crate::RoutedUnitObserver<WorkspaceTensor>>,
        ) -> Result<crate::RoutedExpertTensorParallelOutput<WorkspaceTensor>, E>,
    {
        let interested = observer.is_some();
        let mut observe = |source: eredu_nn::workspace::WorkspaceExpertObservationView<'_>| {
            observer
                .as_deref_mut()
                .ok_or_else(|| executor.metadata_source(eredu_nn::GroupedUnitError::Unavailable))?
                .observe_region_source(source)
        };
        // The shared constructor gives the exact native forwarding closure
        // type, including its capture padding. No callback is invoked here.
        // The actual observer loan is used solely by the cold source callback.
        let argument_bytes = size_of_val(&run);
        let native_body = crate::backend::observed_expert_region_body(run, None);
        executor.charge_metadata(
            argument_bytes
                .checked_add(size_of_val(&observe))
                .and_then(|n| n.checked_add(size_of_val(&native_body)))
                .and_then(|n| {
                    n.checked_add(size_of::<(
                        bool,
                        Result<eredu_nn::workspace::WorkspaceExpertObservationSource, Error>,
                        Result<
                            Result<crate::RoutedExpertTensorParallelOutput<WorkspaceTensor>, E>,
                            Error,
                        >,
                    )>())
                })
                .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
        )?;
        let output = eredu_nn::workspace::record_expert_region_with_observation(
            source,
            bank,
            input,
            routes,
            executor,
            interested.then_some(&mut observe),
        )?;
        Ok(Ok(if source.tensor_partitions.is_some() {
            crate::RoutedExpertTensorParallelOutput::Partial(output)
        } else {
            let (value, bias) = output.into_parts();
            if bias.is_some() {
                return Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into());
            }
            crate::RoutedExpertTensorParallelOutput::Complete(value)
        }))
    }

    fn with_prepared_publication_group<T, E, F>(
        group: &WorkspaceCommunicationGroup,
        prepared: &WorkspaceParallelContext,
        funding: &HostMetadataFunding,
        executor: &WorkspaceContext,
        run: F,
    ) -> Result<Result<T, E>, Error>
    where
        F: FnOnce(Option<&WorkspaceCommunicationGroup>) -> Result<T, E>,
    {
        executor.charge_metadata(size_of::<(
            F,
            T,
            E,
            Result<T, E>,
            Result<Result<T, E>, Error>,
            &WorkspaceCommunicationGroup,
            &WorkspaceParallelContext,
            &HostMetadataFunding,
        )>())?;
        group.select(CommunicationOperation::Broadcast, executor)?;
        if prepared.rank() != group.source.manifest().rank()
            || prepared.size() != group.source.manifest().world_size()
            || !executor
                .metadata_funding()
                .is_some_and(|actual| actual.same_account(funding))
        {
            return Err(executor.metadata_error(format_args!(
                "cold publication context differs from its selected source"
            )));
        }
        Ok(run(Some(group)))
    }
    fn prepare_boundary_source(
        context: &WorkspaceParallelContext,
        route: &WorkspaceCommunicationRoute,
    ) -> Result<Option<PreparedBoundarySource>, BackendFailure> {
        if !enclosing_submission(&route.context) {
            return Ok(None);
        }
        let funding = route
            .context
            .metadata_funding()
            .ok_or_else(|| BackendFailure::from(HostMetadataFundingError::Unavailable))?;
        let bytes = BackendFailure::source_retention_peak_bytes::<SourceFailure>()
            .ok_or_else(|| BackendFailure::from(HostMetadataFundingError::Overflow))?;
        funding
            .reserve_metadata(bytes)
            .map_err(BackendFailure::from)?;
        let failed = |cause| {
            BackendFailure::from_error(SourceFailure {
                cause,
                source: route.source.clone(),
                funding: funding.clone(),
            })
        };
        if context.rank() != route.source.manifest().rank()
            || context.size() != route.source.manifest().world_size()
        {
            return Err(failed(SourceCause::Identity));
        }
        route
            .source
            .prepare_boundary_source(route.descriptor().id(), &funding)
            .map(Some)
            .map_err(|cause| failed(SourceCause::Frame(cause)))
    }
    fn submit_prepared_boundary_dependencies(
        values: &[ArchitectureBoundaryValue<WorkspaceTensor>],
        source: &PreparedBoundarySource,
        route: &WorkspaceCommunicationRoute,
        _context: &WorkspaceParallelContext,
        executor: &WorkspaceContext,
    ) -> Result<Option<Submission<(), Self::CommunicationCompletion>>, Error> {
        route.validate(executor)?;
        if !source.source().same_source(&route.source)
            || source.descriptor().id() != route.descriptor().id()
        {
            return Err(
                executor.metadata_error(format_args!("cold boundary dependency source differs"))
            );
        }
        let mut roots = executor.metadata_vec(values.len())?;
        roots.extend(values.iter().map(|value| value.tensor()));
        if !roots.is_empty() {
            executor.complete_values(&roots)?;
        }
        completed((), executor).map(Some)
    }
    fn complete_model_dependencies(
        values: &[&WorkspaceTensor],
        _context: &WorkspaceParallelContext,
        executor: &WorkspaceContext,
    ) -> Result<Option<()>, Error> {
        if !enclosing_submission(executor) {
            return Ok(None);
        }
        executor.complete_values(values)?;
        Ok(Some(()))
    }
    fn submit_local_dependencies<'a, I>(
        values: I,
        executor: &WorkspaceContext,
    ) -> Result<Submission<(), Self::CommunicationCompletion>, Error>
    where
        I: IntoIterator<Item = &'a WorkspaceTensor>,
    {
        // A model-scope dependency needs the source-bearing prepared hook.
        // The direct worker owns a separate submission; an enclosing-scope
        // selection does not supply its native constructor/completion facts.
        if enclosing_submission(executor) {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        executor.charge_metadata(size_of::<(
            I,
            I::IntoIter,
            Vec<&'a WorkspaceTensor>,
            &WorkspaceContext,
            Result<(), Error>,
        )>())?;
        let mut roots = executor.metadata_vec(0)?;
        for value in values {
            executor.reserve_metadata_vec(&mut roots, 1)?;
            roots.push(value);
        }
        executor.complete_communication_dependencies(&roots)?;
        completed((), executor)
    }
}
#[derive(Debug, thiserror::Error)]
enum WaveCause<E: std::error::Error + 'static> {
    #[error(transparent)]
    Operation(Error),
    #[error(transparent)]
    Validation(E),
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct WaveFailure<E: std::error::Error + 'static> {
    #[source]
    cause: WaveCause<E>,
    source: RetainedCommunicationSource,
    funding: HostMetadataFunding,
}
impl SumReductionBackend for WorkspaceBackend {
    fn complete_model_sum_wave<E, V>(
        values: &[WorkspaceTensor],
        group: &WorkspaceCommunicationGroup,
        _context: &WorkspaceParallelContext,
        executor: &WorkspaceContext,
        mut validate: V,
    ) -> Result<Option<Vec<WorkspaceTensor>>, BackendFailure>
    where
        E: std::error::Error + Send + Sync + 'static,
        V: FnMut(&[WorkspaceTensor], &HostMetadataFunding, bool) -> Result<(), E>,
    {
        if !enclosing_submission(executor) {
            return Ok(None);
        }
        let funding = executor
            .metadata_funding()
            .ok_or_else(|| HostMetadataFundingError::Unavailable.into_backend_failure())?;
        let frames = std::mem::size_of::<(
            &[WorkspaceTensor],
            &WorkspaceCommunicationGroup,
            &WorkspaceParallelContext,
            &WorkspaceContext,
            Vec<WorkspaceTensor>,
            Vec<&WorkspaceTensor>,
            V,
            std::slice::Iter<'_, WorkspaceTensor>,
            Result<Vec<WorkspaceTensor>, WaveCause<E>>,
            Result<(), E>,
            HostMetadataFunding,
            Option<HostMetadataFunding>,
            WaveFailure<E>,
            WaveCause<E>,
        )>()
        .checked_add(
            BackendFailure::source_retention_peak_bytes::<WaveFailure<E>>()
                .ok_or_else(|| HostMetadataFundingError::Overflow.into_backend_failure())?,
        )
        .ok_or_else(|| HostMetadataFundingError::Overflow.into_backend_failure())?;
        funding
            .reserve_metadata(frames)
            .map_err(HostMetadataFundingError::into_backend_failure)?;
        let run = (|| -> Result<Vec<WorkspaceTensor>, WaveCause<E>> {
            // Validate source identity even for an empty wave.
            group
                .select(CommunicationOperation::AllReduceSum, executor)
                .map_err(WaveCause::Operation)?;
            validate(values, &funding, false).map_err(WaveCause::Validation)?;
            let mut outputs = executor
                .metadata_vec(values.len())
                .map_err(WaveCause::Operation)?;
            for value in values {
                outputs.push(
                    Self::all_reduce_sum(value.clone(), group, executor)
                        .map_err(WaveCause::Operation)?
                        .output,
                );
            }
            let mut roots = executor
                .metadata_vec(outputs.len())
                .map_err(WaveCause::Operation)?;
            roots.extend(outputs.iter());
            if !roots.is_empty() {
                executor
                    .complete_values(&roots)
                    .map_err(WaveCause::Operation)?;
            }
            validate(&outputs, &funding, true).map_err(WaveCause::Validation)?;
            Ok(outputs)
        })();
        run.map(Some).map_err(|cause| {
            BackendFailure::from_error(WaveFailure {
                cause,
                source: group.source.clone(),
                funding: funding.clone(),
            })
        })
    }
    fn complete_model_sum(
        value: &WorkspaceTensor,
        group: &WorkspaceCommunicationGroup,
        _context: &WorkspaceParallelContext,
        executor: &WorkspaceContext,
    ) -> Result<Option<WorkspaceTensor>, Error> {
        if !enclosing_submission(executor) {
            return Ok(None);
        }
        let submission = Self::all_reduce_sum(value.clone(), group, executor)?;
        executor.complete_values(&[&submission.output])?;
        Ok(Some(submission.output))
    }
    fn all_reduce_sum(
        value: WorkspaceTensor,
        group: &WorkspaceCommunicationGroup,
        executor: &WorkspaceContext,
    ) -> Result<Submission<WorkspaceTensor, Self::CommunicationCompletion>, Error> {
        let descriptor = group.select(CommunicationOperation::AllReduceSum, executor)?;
        let parallel = WorkspaceParallelContext::new(
            descriptor.local_index().expect("selected member"),
            descriptor.members().len(),
        )?;
        completed(Self::sum_parallel(value, &parallel, executor)?, executor)
    }
}
// The ordinary routed strategy carries these imperative transport contracts.
// WorkspaceBackend consumes the complete route/inactive source at its existing
// outer hooks above, before exact counts or transport tensors are constructed.
// Entering these inner callbacks would lose that source; reject instead of
// inventing counts or recording a second transport graph.
impl EvenGatherBackend for WorkspaceBackend {
    fn all_gather_even(
        _value: WorkspaceTensor,
        _axis: usize,
        _group: &WorkspaceCommunicationGroup,
        _executor: &WorkspaceContext,
    ) -> Result<Submission<WorkspaceTensor, Self::CommunicationCompletion>, Error> {
        Err(WorkspaceMetadataError::Unqualified.into())
    }
}
impl VariableAllToAllBackend for WorkspaceBackend {
    fn variable_all_to_all(
        _value: WorkspaceTensor,
        _counts: &CommunicationPeerCounts,
        _axis: usize,
        _group: &WorkspaceCommunicationGroup,
        _executor: &WorkspaceContext,
    ) -> Result<Submission<WorkspaceTensor, Self::CommunicationCompletion>, Error> {
        Err(WorkspaceMetadataError::Unqualified.into())
    }
}

impl UnevenGatherBackend for WorkspaceBackend {
    fn complete_model_gather(
        value: &WorkspaceTensor,
        counts: &[usize],
        axis: usize,
        group: &WorkspaceCommunicationGroup,
        _context: &WorkspaceParallelContext,
        executor: &WorkspaceContext,
    ) -> Result<Option<WorkspaceTensor>, Error> {
        if !enclosing_submission(executor) {
            return Ok(None);
        }
        let submission = Self::all_gather_uneven(value.clone(), counts, axis, group, executor)?;
        executor.complete_values(&[&submission.output])?;
        Ok(Some(submission.output))
    }
    fn all_gather_uneven(
        value: WorkspaceTensor,
        counts: &[usize],
        axis: usize,
        group: &WorkspaceCommunicationGroup,
        executor: &WorkspaceContext,
    ) -> Result<Submission<WorkspaceTensor, Self::CommunicationCompletion>, Error> {
        let descriptor = group.select(CommunicationOperation::AllGatherUneven, executor)?;
        if counts.len() != descriptor.members().len() {
            return Err(executor.metadata_error(format_args!(
                "cold gather count differs from selected group"
            )));
        }
        completed(
            value.gather_uneven_axis(
                axis,
                descriptor.local_index().expect("selected member"),
                counts,
                executor,
            )?,
            executor,
        )
    }
}
impl BroadcastBackend for WorkspaceBackend {
    fn broadcast(
        value: WorkspaceTensor,
        root: usize,
        group: &WorkspaceCommunicationGroup,
        executor: &WorkspaceContext,
    ) -> Result<Submission<WorkspaceTensor, Self::CommunicationCompletion>, Error> {
        let descriptor = group.select(CommunicationOperation::Broadcast, executor)?;
        completed(
            value.broadcast_publication(
                descriptor.id(),
                root,
                descriptor.local_index().expect("selected member"),
                descriptor.members().len(),
                executor,
            )?,
            executor,
        )
    }
}
impl BarrierBackend for WorkspaceBackend {
    fn barrier(
        group: &WorkspaceCommunicationGroup,
        executor: &WorkspaceContext,
    ) -> Result<Self::CommunicationCompletion, Error> {
        group.select(CommunicationOperation::Barrier, executor)?;
        // This is descriptive traversal. Reached native control invocations
        // have their own request cursor, source census, role and completion.
        super::WorkspaceCompletion::recorded(executor)
    }
}
impl FailureAgreementBackend for WorkspaceBackend {
    type FailureAgreementOutput = bool;
    fn agree_success(
        local_success: bool,
        group: &WorkspaceCommunicationGroup,
        executor: &WorkspaceContext,
    ) -> Result<Submission<bool, Self::CommunicationCompletion>, Error> {
        group.select(CommunicationOperation::FailureAgreement, executor)?;
        completed(local_success, executor)
    }
    fn resolve_failure_agreement(output: bool) -> Result<bool, Error> {
        Ok(output)
    }
}
impl PointToPointBackend for WorkspaceBackend {
    fn send_receive(
        values: Vec<RoleExactBoundaryValue<WorkspaceTensor>>,
        route: &WorkspaceCommunicationRoute,
        executor: &WorkspaceContext,
    ) -> Result<Submission<Vec<WorkspaceTensor>, Self::CommunicationCompletion>, Error> {
        record_frames(values, route, executor)
    }
    fn send_receive_prepared(
        values: PreparedBoundaryFrames<WorkspaceTensor>,
        route: &WorkspaceCommunicationRoute,
        _context: &WorkspaceParallelContext,
        executor: &WorkspaceContext,
    ) -> Result<Option<Submission<Vec<WorkspaceTensor>, Self::CommunicationCompletion>>, Error>
    {
        route.validate(executor)?;
        let funding = executor
            .metadata_funding()
            .ok_or(WorkspaceMetadataError::Unqualified)?;
        if !values.source().same_source(&route.source)
            || values.route() != route.descriptor().id()
            || !funding.same_account(values.funding())
        {
            return Err(
                executor.metadata_error(format_args!("cold boundary frame source/account differs"))
            );
        }
        let (values, source, funding) = values.into_parts();
        let result = record_frames(values, route, executor).map(Some);
        drop((source, funding));
        result
    }
}
fn record_frames(
    values: Vec<RoleExactBoundaryValue<WorkspaceTensor>>,
    route: &WorkspaceCommunicationRoute,
    context: &WorkspaceContext,
) -> Result<Submission<Vec<WorkspaceTensor>, super::WorkspaceCompletion>, Error> {
    route.validate(context)?;
    context.charge_metadata(size_of::<(
        RoleExactBoundaryValue<WorkspaceTensor>,
        WorkspaceCollective,
        Vec<WorkspaceTensor>,
        Result<Vec<WorkspaceTensor>, Error>,
        usize,
        usize,
    )>())?;
    if enclosing_submission(context)
        && route.descriptor().destination() == route.source.manifest().rank()
    {
        // The enclosing-scope worker settles receiver roots before frame
        // encoding. Ordinary transfer owns them through its final submission.
        let mut roots = context.metadata_vec(values.len())?;
        roots.extend(values.iter().map(|value| value.tensor()));
        if !roots.is_empty() {
            context.complete_values(&roots)?;
        }
    }
    let mut outputs = context.metadata_vec(values.len())?;
    for (ordinal, framed) in values.into_iter().enumerate() {
        let (header, tensor) = framed.into_parts();
        let mut layouts = context.metadata_vec(1)?;
        layouts.push(
            context
                .layout(tensor.shape(), tensor.layout().dtype())?
                .with_representation(tensor.layout().representation()),
        );
        let mut recorded = context.execute(
            WorkspaceOperationKind::Collective(WorkspaceCollective::Boundary {
                route: route.descriptor().id().value(),
                ordinal,
                header_bytes: header.len(),
            }),
            &[&tensor],
            layouts,
        )?;
        outputs.push(recorded.pop().expect("one declared boundary output"));
    }
    completed(outputs, context)
}
impl CommunicationTensorMetadata<WorkspaceBackend> for WorkspaceCommunicationMetadata {
    fn dtype(&self, tensor: &WorkspaceTensor) -> TensorDtype {
        match tensor.layout().dtype() {
            WorkspaceDtype::Float32 => {
                match tensor.layout().representation().map(|value| value.dtype()) {
                    Some(WorkspaceFloatingType::Float16) => TensorDtype::F16,
                    Some(WorkspaceFloatingType::Bfloat16) => TensorDtype::Bf16,
                    _ => TensorDtype::F32,
                }
            }
            WorkspaceDtype::Int32 => TensorDtype::I32,
            WorkspaceDtype::Uint32 => TensorDtype::U32,
            WorkspaceDtype::Uint8 => TensorDtype::U8,
            WorkspaceDtype::Bool => TensorDtype::Bool,
        }
    }
    fn shape(&self, tensor: &WorkspaceTensor) -> Vec<usize> {
        tensor.shape().iter().map(|&value| value as usize).collect()
    }
    fn matches_shape_with_funding(
        &self,
        tensor: &WorkspaceTensor,
        shape: &[i32],
        funding: &HostMetadataFunding,
    ) -> Result<Option<bool>, HostMetadataFundingError> {
        funding.reserve_metadata(size_of::<(&WorkspaceTensor, &[i32], Option<bool>)>())?;
        Ok(Some(tensor.shape() == shape))
    }
    fn fixed_metadata_with_funding(
        &self,
        tensor: &WorkspaceTensor,
        funding: &HostMetadataFunding,
    ) -> Result<Option<(TensorDtype, usize, Option<usize>)>, HostMetadataFundingError> {
        funding.reserve_metadata(size_of::<(
            &WorkspaceTensor,
            TensorDtype,
            usize,
            Option<usize>,
        )>())?;
        Ok(Some((
            self.dtype(tensor),
            tensor.shape().len(),
            tensor.shape().iter().try_fold(1usize, |n, &value| {
                usize::try_from(value)
                    .ok()
                    .and_then(|value| n.checked_mul(value))
            }),
        )))
    }
}
