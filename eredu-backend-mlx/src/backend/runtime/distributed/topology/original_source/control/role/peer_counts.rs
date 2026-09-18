//! Exact integer peer-row gather through the retained request's ordinary role.
use super::*;
use crate::backend::runtime::distributed::completion::{
    CompletedCommunicationWords, OriginalCommunicationWords, PreparedCommunicationWords,
};
use eredu_runtime::{DistributedExecutionPhase, PartitionExecutionError, PreparedPeerCountLoan};
use eredu_core::SharedStorageOwner;
#[path = "peer_counts/table.rs"]
pub(super) mod table;
#[path = "peer_counts/routed.rs"]
mod routed;
use safemlx::{Array, PreparedInputPlan, distributed::GroupWorkerOperation};

#[derive(Debug, thiserror::Error)]
enum PeerCountCause {
    #[error("peer count consensus differs from its exact retained group or matrix")]
    Identity,
    #[error(transparent)]
    Communication(#[from] PartitionExecutionError),
    #[error(transparent)]
    Allocation(#[from] std::collections::TryReserveError),
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct PeerCountFailure {
    #[source]
    cause: PeerCountCause,
    custody: Custody,
}
fn fail(cause: PeerCountCause, custody: &Custody) -> Error {
    Error::with_original_control_source(eredu_core::BackendFailure::new(
        eredu_core::BackendFailureKind::Other,
        PeerCountFailure { cause, custody: custody.clone() },
    ), false)
}
fn controls(custody: &Custody) -> Result<(), Error> {
    reserve(&custody.funding, &[
        size_of::<PeerCountCause>(), size_of::<PeerCountFailure>(), size_of::<Custody>(),
        eredu_core::BackendFailure::source_retention_peak_bytes::<PeerCountFailure>().ok_or_else(overflow)?,
        failure_control_bytes().ok_or_else(overflow)?,
    ])
}
// This is the one wire-schema width used by both the actual encoder and the
// array-free count source. It describes words, never native allocation authority.
impl OriginalParallelSource {
    pub(crate) fn peer_count_wire_words(world: usize) -> Option<usize> {
        world.checked_mul(2)?.checked_add(2)
    }
}

enum Consensus {
    Physical(SharedStorageOwner<table::PeerCountTable>),
    Logical(SharedStorageOwner<table::PeerCountTable>),
}
impl Consensus {
    fn loan<'a>(&'a self, funding: &'a HostMetadataFunding) -> PreparedPeerCountLoan<'a> {
        match self {
            Self::Physical(table) => PreparedPeerCountLoan::new(table.local(), funding, Some(table.clone().erase())),
            Self::Logical(table) => PreparedPeerCountLoan::new(table.local(), funding, Some(table.clone().erase())),
        }
    }
}
impl OriginalParallelControlProjection {
    pub(crate) fn with_peer_count_consensus<T, E, F>(
        &self, local: &[i32], group: &Group, context: &Group, executor: &Stream, run: F,
    ) -> Result<Result<T, E>, Error>
    where F: for<'loan> FnOnce(Option<(&'loan [i32], &'loan HostMetadataFunding)>) -> Result<T, E>,
    {
        self.with_peer_count_source(local, group, context, executor, |loan| {
            let parts = loan.map(PreparedPeerCountLoan::into_parts);
            run(parts.as_ref().map(|(matrix, funding, _)| (*matrix, *funding)))
        })
    }
    pub(crate) fn with_peer_count_source<T, E, F>(
        &self, local: &[i32], group: &Group, context: &Group, executor: &Stream, run: F,
    ) -> Result<Result<T, E>, Error>
    where F: for<'loan> FnOnce(Option<PreparedPeerCountLoan<'loan>>) -> Result<T, E>,
    {
        self.with_context(context, |bound| {
            let Some(_) = bound else { return Err(fail(PeerCountCause::Identity, &self.custody)); };
            let c = &self.custody;
            controls(c)?;
            reserve(&c.funding, &[
                size_of::<F>(), size_of::<T>(), size_of::<E>(), size_of::<Result<T, E>>(),
                size_of::<Result<Result<T, E>, Error>>(), size_of::<Invocation>(),
                size_of::<OriginalParallelControlOwner>(), size_of::<ParallelControlClaim>(),
                size_of::<PreparedInputPlan<'_>>(), size_of::<[usize; 1]>(), size_of::<[Array; 1]>(),
                size_of::<Option<usize>>(), size_of::<(usize, usize)>(),
                size_of::<Result<CompletedCommunicationWords, Error>>(),
                size_of::<Consensus>(), size_of::<Result<Consensus, Error>>(),
                size_of::<PreparedPeerCountLoan<'_>>(),
                size_of::<Result<Result<CompletedCommunicationWords, Error>, eredu_core::BackendFailure>>(),
                size_of::<Result<ParallelControlClaim, eredu_runtime::working_memory::WorkingMemoryError>>(),
            ])?;
            let retained = self.upgrade().map_err(|cause| Error::with_original_control_source(cause, false))?;
            let owner = retained.owner();
            let event = ParallelControlEvent::Phase(DistributedExecutionPhase::Execution);
            let claim = owner.request.cursor.try_borrow_mut()
                .map_err(|_| fail(PeerCountCause::Identity, c))?.claim(event)
                .map_err(|cause| control_error(ControlCause::WorkingMemory(cause), &c.source, &c.funding))?;
            if owner.failed.get() || owner.running.replace(true) {
                owner.failed.set(true);
                return Err(fail(PeerCountCause::Identity, c));
            }
            let _running = Running { running: &owner.running, failed: &owner.failed };
            let model=owner.request.source.model();
            let binding=self.expert_binding(context);
            let count_source=match (binding,model) {
                (Some(binding),Some(model))=>binding.claim_expert_counts(model,group,executor)
                    .map_err(|cause|{owner.failed.set(true);cause})?,
                _=>None,
            };
            let result = (|| {
                let source = owner.request.source.communication_source()?;
                let order = source.source().manifest().groups().iter().enumerate()
                    .find_map(|(order, _)| source.matches_group(order, group).then_some(order))
                    .ok_or_else(|| fail(PeerCountCause::Identity, c))?;
                let (_, descriptor, _) = source.group(order)
                    .ok_or_else(|| fail(PeerCountCause::Identity, c))?;
                source.source().manifest().select_group_operation(descriptor.id(), CommunicationOperation::AllGatherEven)
                    .map_err(|cause| failure(Cause::Rank(cause), &c.source, &c.funding))?;
                let peers = descriptor.members().len();
                if let Some(count)=&count_source {
                    if !count.matches(order,peers,group.rank()) {return Err(fail(PeerCountCause::Identity,c));}
                }
                if local.len()!=peers || local.iter().any(|&count|count<0) {
                    return Err(fail(PeerCountCause::Identity, c));
                }
                let expected = peers.checked_mul(peers).ok_or_else(|| fail(PeerCountCause::Identity, c))?;
                let inputs = owner.request.source.agreement_inputs()
                    .ok_or_else(|| fail(PeerCountCause::Identity, c))?;
                if group.is_logical() {
                    table::PeerCountTable::prepare_controls(c)?;
                    let world_size = source.source().manifest().world_size();
                    let row = table::WireRow::new(local, descriptor.members(), source.world().rank(), world_size, c)?;
                    let shape = [row.values().len()];
                    let plan = inputs.runtime().i32(row.values(), &shape)
                        .map_err(|cause| failure(Cause::Input(cause), &c.source, &c.funding))?;
                    let [input] = super::super::super::inputs::construct_retained(&source, [plan])?;
                    if group.logical_routed_plan().map_err(|_| fail(PeerCountCause::Identity, c))?.is_some() {
                        return routed::execute(&retained, claim, group, order, &input, row.values(), count_source.as_ref(), executor, c)
                            .map(Consensus::Logical);
                    }
                    let model = owner.request.source.model()
                        .ok_or_else(|| fail(PeerCountCause::Identity, c))?;
                    let quote = match &count_source {
                        Some(count)=>count.logical(None).ok_or_else(||fail(PeerCountCause::Identity,c))?,
                        None=>super::super::super::parallel::RetainedLogicalCollective::prepare_count_group(
                            model, order, &input, executor)?,
                    };
                    let identity = claim.identity();
                    let ordinal = claim.ordinal();
                    let world = RefCell::new(None);
                    let mut observe_world = |output: &Array, observer: &OriginalScopeObserver| -> Result<(), Error> {
                        let source = owner.request.source.communication_source()?;
                        let words = PreparedCommunicationWords::read_completed(&source, output, observer)?;
                        let table = table::WorldTable::decode(words.as_slice(), world_size, world_size, c)?;
                        let mut slot = world.try_borrow_mut().map_err(|_| fail(PeerCountCause::Identity, c))?;
                        if slot.is_some() { return Err(fail(PeerCountCause::Identity, c)); }
                        *slot = Some(table);
                        Ok(())
                    };
                    reserve(&c.funding, &[size_of_val(&observe_world),
                        size_of::<std::cell::RefMut<'_, Option<table::WorldTable>>>(),
                        size_of::<std::cell::BorrowMutError>(),
                        size_of::<Result<Consensus, Error>>()])?;
                    // The source loan remains borrowed through both callbacks;
                    // the completed table owns only closed scalar/source custody.
                    return logical::execute_source(&retained, claim, group, &input, executor, quote,
                        Some(&mut observe_world), |output, observer| {
                            let words = PreparedCommunicationWords::prepare(&source, output)?.resolve_completed(observer)?;
                            let world = world.try_borrow_mut().map_err(|_| fail(PeerCountCause::Identity, c))?.take();
                            table::PeerCountTable::new(words.as_slice(), descriptor.members(), world_size,
                                world, order, identity, ordinal, c).map(Consensus::Logical)
                        });
                }
                let shape = [peers];
                let plan = inputs.runtime().i32(local, &shape)
                    .map_err(|cause| failure(Cause::Input(cause), &c.source, &c.funding))?;
                let [input] = super::super::super::inputs::construct(&source, [plan])?;
                let capacity = capacity(&source, Some(order), &input, expected, owner, c)?;
                drop(source);
                let identity = claim.identity();
                let ordinal = claim.ordinal();
                let invocation = Invocation {
                    input, order: Some(order), expected, claim, capacity,
                    owner: OriginalParallelControlOwner(retained.0.clone()),
                };
                let words = run_native_role(invocation, capacity, &owner.bank, &owner.controls, c,
                    |value, observer| Ok(value.run(observer)))
                    .map_err(|cause| Error::with_original_control_source(cause, false))??;
                let source = owner.request.source.communication_source()?;
                let (_, descriptor, _) = source.group(order)
                    .ok_or_else(|| fail(PeerCountCause::Identity, c))?;
                table::PeerCountTable::from_completed_physical(words.as_slice(), descriptor.members(),
                    order, identity, ordinal, c).map(Consensus::Physical)
            })();
            if result.is_err() { owner.failed.set(true); }
            let completed = result?;
            let completed = match completed {
                Consensus::Logical(table) if table.world().is_none() && group.logical_variable_world_plan().is_some() => {
                    let result = complete_world(&retained, local, group, &table, c);
                    if result.is_err() { owner.failed.set(true); }
                    Consensus::Logical(result?)
                }
                completed => completed,
            };
            if let Some(count)=&count_source {
                let model=model.ok_or_else(||fail(PeerCountCause::Identity,c))?;
                let binding=binding.ok_or_else(||fail(PeerCountCause::Identity,c))?;
                let world=matches!(&completed,Consensus::Logical(table) if table.world().is_some());
                binding.complete_expert_counts(model,count,world,executor)
                    .map_err(|cause|{owner.failed.set(true);cause})?;
            }
            Ok(run(Some(completed.loan(&c.funding))))
        })?
    }
}
struct Invocation {
    input: Array,
    order: Option<usize>,
    expected: usize,
    claim: ParallelControlClaim,
    capacity: AgreementCapacity,
    owner: OriginalParallelControlOwner,
}
fn operation<'a>(source: &'a OriginalCommunicationSource<'_>, order: Option<usize>, input: &'a Array)
    -> Result<super::super::super::OriginalCommunicationOperation<'a>, Error> {
    match order {
        Some(order) => source.group_cpu_operation_storage(order, input, GroupWorkerOperation::Gather),
        None => source.world_cpu_operation_storage(input, GroupWorkerOperation::Gather),
    }
}
fn capacity(source: &OriginalCommunicationSource<'_>, order: Option<usize>, input: &Array,
    expected: usize, owner: &Owner, c: &Custody) -> Result<AgreementCapacity, Error>
{
    let operation = operation(source, order, input)?;
    if operation.native().constructor().output_dtype()!=safemlx::Dtype::Int32
        || operation.native().constructor().output_geometry().1!=expected {
        return Err(fail(PeerCountCause::Identity, c));
    }
    let inputs = owner.request.source.agreement_inputs()
        .ok_or_else(|| fail(PeerCountCause::Identity, c))?;
    let backing = operation.backing_storage(inputs.runtime())?.capacity();
    let operation = operation.with_completion()?;
    Ok(AgreementCapacity { graph: operation.graph_capacity(), records: operation.record_capacity(), backing })
}
impl Invocation {
    fn run(&self, observer: &OriginalScopeObserver) -> Result<CompletedCommunicationWords, Error> {
        let owner = self.owner.owner();
        let c = &owner.custody;
        controls(c)?;
        reserve(&c.funding, &[
            size_of::<Self>(), size_of::<PreparedCommunicationWords>(), size_of::<OriginalCommunicationWords>(),
            size_of::<CompletedCommunicationWords>(), size_of::<Result<CompletedCommunicationWords, Error>>(),
            size_of::<PartitionExecutionError>(), size_of::<Option<(&Group, &eredu_runtime::CommunicationGroupDescriptor, bool)>>(),
        ])?;
        let event = ParallelControlEvent::Phase(DistributedExecutionPhase::Execution);
        let source = owner.request.source.communication_source()?;
        if self.claim.identity()!=owner.request.cursor.borrow().identity() || self.claim.event()!=event
            || !self.capacity.covers(capacity(&source, self.order, &self.input, self.expected, owner, c)?) {
            return Err(fail(PeerCountCause::Identity, c));
        }
        let operation = operation(&source, self.order, &self.input)?.with_completion()?;
        let words = PreparedCommunicationWords::prepare_operation(&source, &operation)?;
        let ready = operation.prepare_resources(&source, self.order)?;
        let group = match self.order {
            Some(order) => source.group(order).ok_or_else(|| fail(PeerCountCause::Identity, c))?.0,
            None => source.world(),
        };
        let stream = group.retained_transport_stream().ok_or_else(|| fail(PeerCountCause::Identity, c))?;
        let phase = DistributedExecutionPhase::Execution;
        let submitted = operation.construct_accepted(&source, observer, stream)
            .and_then(|accepted| words.submit_accepted(accepted, ready));
        let (output, completion) = submitted.map_err(|cause| {
            source.authority.fence_protocol_failure(CommunicationOperation::AllGatherEven, phase, None);
            cause
        })?;
        let output = source.authority.wait_with_error(eredu_core::Submission { output, completion },
            CommunicationOperation::AllGatherEven, phase, None,
            |cause| PartitionExecutionError::PreparedCommunication {
                operation: CommunicationOperation::AllGatherEven, phase, completion: true,
                source: failure(Cause::Native(cause), &c.source, &c.funding).into_backend_failure(),
            }).map_err(|cause| fail(cause.into(), c))?;
        let completed = output.resolve()?;
        if completed.as_slice().len()!=self.expected { return Err(fail(PeerCountCause::Identity, c)); }
        Ok(completed)
    }
}

/// Pair/explicit-route gathers produce a local table. If the actual variable
/// plan then selects the world, acquire its complete rows through one additional
/// accepted physical Gather before publishing the count-source loan.
fn complete_world(owner: &OriginalParallelControlOwner, local: &[i32], group: &Group,
    table: &table::PeerCountTable, c: &Custody) -> Result<SharedStorageOwner<table::PeerCountTable>, Error> {
    reserve(&c.funding, &[size_of::<Invocation>(), size_of::<ParallelControlClaim>(),
        size_of::<table::WireRow>(), size_of::<OriginalParallelControlOwner>(),
        size_of::<PreparedInputPlan<'_>>(), size_of::<[PreparedInputPlan<'_>; 1]>(),
        size_of::<[Array; 1]>(), size_of::<[usize; 1]>(), size_of::<Option<usize>>(),
        size_of::<Result<CompletedCommunicationWords, Error>>(),
        size_of::<Result<Result<CompletedCommunicationWords, Error>, eredu_core::BackendFailure>>(),
        size_of::<Result<SharedStorageOwner<table::PeerCountTable>, Error>>()])?;
    let retained = owner.owner();
    let claim = retained.request.cursor.try_borrow_mut().map_err(|_| fail(PeerCountCause::Identity, c))?
        .claim(ParallelControlEvent::Phase(DistributedExecutionPhase::Execution))
        .map_err(|cause| control_error(ControlCause::WorkingMemory(cause), &c.source, &c.funding))?;
    let source = retained.request.source.communication_source()?;
    let (_, descriptor, wave) = source.source().manifest().groups().iter().enumerate()
        .find_map(|(order, _)| source.matches_group(order, group).then_some(order))
        .and_then(|order| source.group(order).map(|(_, descriptor, wave)| (order, descriptor, wave)))
        .ok_or_else(|| fail(PeerCountCause::Identity, c))?;
    let plan = group.logical_variable_world_plan().ok_or_else(|| fail(PeerCountCause::Identity, c))?;
    if !wave || plan.members() != descriptor.members()
        || !group.native_group().shares_native_handle(source.world().native_group()) {
        return Err(fail(PeerCountCause::Identity, c));
    }
    let world = source.source().manifest().world_size();
    let row = table::WireRow::new(local, descriptor.members(), source.world().rank(), world, c)?;
    let inputs = retained.request.source.agreement_inputs().ok_or_else(|| fail(PeerCountCause::Identity, c))?;
    let shape = [row.values().len()];
    let plan = inputs.runtime().i32(row.values(), &shape)
        .map_err(|cause| failure(Cause::Input(cause), &c.source, &c.funding))?;
    let [input] = super::super::super::inputs::construct(&source, [plan])?;
    let expected = world.checked_mul(shape[0]).ok_or_else(overflow)?;
    let capacity = capacity(&source, None, &input, expected, retained, c)?;
    drop(source);
    let invocation = Invocation { input, order: None, expected, claim, capacity,
        owner: OriginalParallelControlOwner(owner.0.clone()) };
    let words = run_native_role(invocation, capacity, &retained.bank, &retained.controls, c,
        |value, observer| Ok(value.run(observer)))
        .map_err(|cause| Error::with_original_control_source(cause, false))??;
    let world = table::WorldTable::decode(words.as_slice(), world, world, c)?;
    table.with_world(world, c)
}
