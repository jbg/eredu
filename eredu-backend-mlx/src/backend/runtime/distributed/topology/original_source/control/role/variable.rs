//! Variable Ring exchange under the retained request's actual child authority.
use super::*;
use eredu_runtime::{CommunicationPeerCounts, CommunicationPeerMatrix,
    DistributedExecutionPhase, PartitionExecutionError};
use safemlx::{Array, PreparedArrayClone, PreparedInputPlan, PreparedInputRuntime};
#[path = "variable/reorder.rs"]
mod reorder;
#[path = "variable/local.rs"]
mod local;
use eredu_core::ErasedSharedStorageOwner;

#[derive(Debug, thiserror::Error)]
enum VariableCause {
    #[error("variable exchange differs from its retained group, counts or source")]
    Identity,
    #[error("variable exchange requires its exact leading-axis transport source")]
    Axis,
    #[error("logical variable exchange requires its retained logical iteration source")]
    LogicalSource,
    #[error(transparent)]
    Allocation(#[from] std::collections::TryReserveError),
    #[error(transparent)]
    Clone(#[from] safemlx::PreparedArrayCloneCause),
    #[error(transparent)]
    Communication(#[from] PartitionExecutionError),
    #[error(transparent)] Neural(#[from] eredu_nn::Error),
    #[error(transparent)] Native(#[from] safemlx::error::Exception),
    #[error(transparent)] Buffer(#[from] safemlx::OriginalBufferCause),
    #[error(transparent)] Stream(#[from] safemlx::StreamCopyCause),
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct VariableFailure {
    #[source] cause: VariableCause,
    custody: Custody,
}
fn fail(cause: VariableCause, c: &Custody) -> Error {
    Error::with_original_control_source(eredu_core::BackendFailure::from_error(
        VariableFailure { cause, custody: c.clone() }), false)
}
fn controls(c: &Custody) -> Result<(), Error> {
    reserve(&c.funding, &[
        size_of::<VariableCause>(), size_of::<VariableFailure>(), size_of::<Custody>(),
        eredu_core::BackendFailure::source_retention_peak_bytes::<VariableFailure>().ok_or_else(overflow)?,
        failure_control_bytes().ok_or_else(overflow)?,
        CommunicationManifest::group_operation_control_bytes().ok_or_else(overflow)?,
    ])
}
fn retain(input: &Array, c: &Custody) -> Result<Array, Error> {
    reserve(&c.funding, &[
        PreparedArrayClone::control_bytes().ok_or_else(overflow)?,
        Array::inspection_clone_handle_bytes(), size_of::<Result<Array, Error>>(),
    ])?;
    let mut slot = PreparedArrayClone::try_prepare_for_inspection()
        .map_err(|cause| fail(cause.into(), c))?;
    slot.fill_for_inspection(input).map_err(|cause| fail(cause.into(), c))
}
fn destination(elements: usize, c: &Custody) -> Result<Vec<usize>, Error> {
    reserve(&c.funding, &[
        elements.checked_mul(size_of::<usize>()).ok_or_else(overflow)?,
        size_of::<Vec<usize>>(), size_of::<Result<Vec<usize>, Error>>(),
        size_of::<std::collections::TryReserveError>(),
    ])?;
    let mut values = Vec::new();
    values.try_reserve_exact(elements).map_err(|cause| fail(cause.into(), c))?;
    Ok(values)
}
fn idle(matrix: &[usize], peers: usize, rank: usize) -> bool {
    (0..peers).all(|peer| matrix[rank * peers + peer] == 0 && matrix[peer * peers + rank] == 0)
}
fn complete_input(region:Option<&crate::backend::runtime::distributed::topology::original_source::parallel::RetainedExpertTransfer>,
    input:&Array,executor:&Stream,c:&Custody)->Result<(),Error>{
    if region.is_none_or(|region|region.requires_input_completion()){
        // Geometry, matrix and source identity are checked before this actual
        // parent completion lends its input into an independent child arena.
        reserve(&c.funding,&[crate::backend::runtime::cache::value_completion_control_bytes(1)
            .ok_or_else(overflow)?,size_of::<Option<&crate::backend::runtime::distributed::topology::original_source::parallel::RetainedExpertTransfer>>(),
            OriginalScopeObserver::control_bytes().ok_or_else(overflow)?,size_of::<OriginalScopeObserver>(),
            safemlx::OperationEvent::traversal_leaf_control_bytes().ok_or_else(overflow)?])?;
        crate::backend::runtime::cache::complete_values([input],executor)
            .map_err(|cause|failure(Cause::Native(cause),&c.source,&c.funding))?;
        // Publish the completed parent descriptor before another arena borrows
        // it. This detaches only its settled Event and performs no new Eval.
        let parent=OriginalScopeObserver::require_current()
            .map_err(|cause|failure(Cause::Native(cause),&c.source,&c.funding))?;
        safemlx::OperationEvent::validate_traversal_leaf(input,&parent)
            .map_err(|cause|failure(Cause::Native(cause),&c.source,&c.funding))?;
    }
    Ok(())
}
impl OriginalParallelControlProjection {
    pub(crate) fn complete_variable_all_to_all(&self, input: &Array,
        counts: &CommunicationPeerCounts, axis: usize, matrix: &CommunicationPeerMatrix<'_>,
        group: &Group, context: &Group, executor: &Stream, funding: &WorkspaceMetadataFunding)
        -> Result<Array, Error> {
        self.with_context(context, |bound| {
            let c = &self.custody;
            controls(c)?;
            reserve(&c.funding, &[
                size_of::<Invocation>(), size_of::<OriginalParallelControlOwner>(),
                size_of::<ParallelControlClaim>(), size_of::<PreparedInputPlan<'_>>(),
                size_of::<[Array; 1]>(), size_of::<Option<Array>>(),
                size_of::<(&Array, &CommunicationPeerCounts, usize, &CommunicationPeerMatrix<'_>, &Group)>(),
                size_of::<crate::MlxTensor>(),
                size_of::<Result<Option<crate::MlxTensor>, Error>>(),
                size_of::<Result<Array, Error>>(), size_of::<Result<Result<Array, Error>, Error>>(),
                size_of::<Result<ParallelControlClaim, eredu_runtime::working_memory::WorkingMemoryError>>(),
                size_of::<Result<Result<Array, Error>, eredu_core::BackendFailure>>(),
                size_of::<[&crate::MlxTensor; 1]>(),
            ])?;
            if bound.is_none() || !funding.same_account(&c.funding) {
                return Err(fail(VariableCause::Identity, c));
            }
            let retained = self.upgrade().map_err(|cause| Error::with_original_control_source(cause, false))?;
            let owner = retained.owner();
            let event = ParallelControlEvent::Phase(DistributedExecutionPhase::Execution);
            let claim = owner.request.cursor.try_borrow_mut()
                .map_err(|_| fail(VariableCause::Identity, c))?.claim(event)
                .map_err(|cause| control_error(ControlCause::WorkingMemory(cause), &c.source, &c.funding))?;
            if owner.failed.get() || owner.running.replace(true) {
                owner.failed.set(true);
                return Err(fail(VariableCause::Identity, c));
            }
            let _running = Running { running: &owner.running, failed: &owner.failed };
            let mut region = None;
            let result = (|| {
                let source = owner.request.source.communication_source()?;
                if let Some(binding)=self.expert_binding(context) {
                    if let Some(model)=owner.request.source.model() {
                        region=binding.claim_expert_transfer(model,executor)?;
                    }
                }
                let order = source.source().manifest().groups().iter().enumerate()
                    .find_map(|(order, _)| source.matches_group(order, group).then_some(order))
                    .ok_or_else(|| fail(VariableCause::Identity, c))?;
                let (_, descriptor, _) = source.group(order)
                    .ok_or_else(|| fail(VariableCause::Identity, c))?;
                source.source().manifest().select_group_operation(descriptor.id(), CommunicationOperation::VariableAllToAll)
                    .map_err(|cause| failure(Cause::Rank(cause), &c.source, &c.funding))?;
                if descriptor != matrix.descriptor() || counts.group_size() != group.size()
                    || !matches!(input.dtype(), safemlx::Dtype::Float16 | safemlx::Dtype::Bfloat16 |
                        safemlx::Dtype::Float32 | safemlx::Dtype::Int32) {
                    return Err(fail(VariableCause::Identity, c));
                }
                if axis >= input.shape().len() || input.shape().is_empty() { return Err(fail(VariableCause::Axis, c)); }
                let peers = descriptor.members().len();
                let rank = descriptor.local_index().ok_or_else(|| fail(VariableCause::Identity, c))?;
                if matrix.forward_counts().len() != peers.checked_mul(peers).ok_or_else(overflow)? {
                    return Err(fail(VariableCause::Identity, c));
                }
                let sent = counts.send().iter().copied().try_fold(0usize, usize::checked_add).ok_or_else(overflow)?;
                if usize::try_from(input.shape()[axis]).ok() != Some(sent)
                    || (0..peers).any(|peer| matrix.count(rank, peer) != Some(counts.send()[peer])
                        || matrix.count(peer, rank) != Some(counts.receive()[peer])) {
                    return Err(fail(VariableCause::Identity, c));
                }
                if region.is_some() {
                    let table=matrix.completed_source().and_then(|source|source.downcast_ref::<super::peer_counts::table::PeerCountTable>())
                        .ok_or_else(||fail(VariableCause::Identity,c))?;
                    if !table.matches(matrix,order,owner,c){return Err(fail(VariableCause::Identity,c));}
                }
                if peers == 1 { return retain(input, c); }
                let plan = group.logical_variable_world_plan();
                let completed_source = matrix.completed_source();
                let table = if group.is_logical() {
                    let table = completed_source.and_then(|source| source.downcast_ref::<super::peer_counts::table::PeerCountTable>())
                        .ok_or_else(|| fail(VariableCause::LogicalSource, c))?;
                    if !table.matches(matrix, order, owner, c) {
                        return Err(fail(VariableCause::Identity, c));
                    }
                    Some(table)
                } else { None };
                if group.is_logical() && plan.is_none() {
                    let selected = group.logical_variable_route_plan()
                        .map_err(|_| fail(VariableCause::LogicalSource, c))?
                        .ok_or_else(|| fail(VariableCause::LogicalSource, c))?;
                    if let Some(region)=&region {
                        if !region.profile.matches(order,false,input,matrix.forward_counts(),peers,rank,matrix.transposed(),axis){
                            return Err(fail(VariableCause::Identity,c));
                        }
                    }
                    complete_input(region.as_ref(),input,executor,c)?;
                    return local::execute(&retained, claim, input, counts, axis, matrix,
                        group, selected, order, executor, c, region.as_ref());
                }
                let world = table.is_some();
                let matrix_values = match table {
                    Some(table) => table.world().ok_or_else(|| fail(VariableCause::LogicalSource, c))?,
                    None => matrix.forward_counts(),
                };
                let (peers, rank) = match plan {
                    Some(plan) => (plan.world_size(), plan.world_rank()),
                    None => (peers, rank),
                };
                if matrix_values.len() != peers.checked_mul(peers).ok_or_else(overflow)? {
                    return Err(fail(VariableCause::Identity, c));
                }
                if let Some(plan) = plan {
                    let local_rank = descriptor.local_index().ok_or_else(|| fail(VariableCause::Identity, c))?;
                    for physical in 0..peers {
                        let logical = plan.members().iter().position(|member| *member == physical);
                        let (sent, received) = match logical {
                            Some(logical) => (matrix.count(local_rank, logical), matrix.count(logical, local_rank)),
                            None => (Some(0), Some(0)),
                        };
                        let (actual_send, actual_receive) = if matrix.transposed() {
                            (matrix_values[physical * peers + rank], matrix_values[rank * peers + physical])
                        } else { (matrix_values[rank * peers + physical], matrix_values[physical * peers + rank]) };
                        if sent != Some(actual_send) || received != Some(actual_receive) {
                            return Err(fail(VariableCause::Identity, c));
                        }
                    }
                }
                if let Some(region)=&region {
                    if !region.profile.matches(order,world,input,matrix_values,peers,rank,matrix.transposed(),axis) {
                        return Err(fail(VariableCause::Identity,c));
                    }
                }
                complete_input(region.as_ref(),input,executor,c)?;
                // Both accepted recovery and the native queued task retain their
                // own paid matrix. The original completed consensus is immutable.
                let mut original = destination(matrix_values.len(), c)?;
                original.extend_from_slice(matrix_values);
                let mut transport = destination(original.len(), c)?;
                transport.extend_from_slice(&original);
                for member in 0..peers {
                    if idle(&original, peers, member) { transport[member * peers + member] = 1; }
                }
                let is_idle = idle(&original, peers, rank);
                let mut first = Some(claim);
                let (input, empty) = if is_idle {
                    reserve(&c.funding, &[PreparedInputRuntime::zeros_plan_control_bytes(),
                        size_of::<PreparedInputPlan<'_>>(), size_of::<[PreparedInputPlan<'_>; 1]>(),
                        size_of::<[Array; 1]>()])?;
                    let mut shape = destination(input.shape().len(), c)?;
                    for (dimension_axis, &dimension) in input.shape().iter().enumerate() {
                        shape.push(if dimension_axis == axis { 1 } else {
                            usize::try_from(dimension).map_err(|_| fail(VariableCause::Identity, c))?
                        });
                    }
                    let inputs = owner.request.source.agreement_inputs()
                        .ok_or_else(|| fail(VariableCause::Identity, c))?;
                    let plan = inputs.runtime().zeros(input.dtype(), &shape)
                        .map_err(|cause| failure(Cause::Input(cause), &c.source, &c.funding))?;
                    let [sentinel] = super::super::super::inputs::construct(&source, [plan])?;
                    (sentinel, Some(retain(input, c)?))
                } else { (retain(input, c)?, None) };
                let input = if axis != 0 {
                    reorder::axis(&retained, first.take().ok_or_else(|| fail(VariableCause::Identity, c))?,
                        input, axis, false, order, completed_source, executor, c)?
                } else { input };
                let input = if let Some(plan) = plan.filter(|plan| !plan.canonical_order() && !is_idle) {
                    reorder::run(&retained, match first.take() { Some(claim) => claim, None => next_claim(owner, c)? },
                        input, counts.send(), plan.input_order(), order, completed_source, executor, c,
                        region.as_ref().and_then(|region|region.profile.input_reorder))?
                } else { input };
                let transposed = matrix.transposed();
                let capacity = capacity(&source, order, world, &input, &transport, transposed, owner, c)?;
                if let Some(region)=&region {
                    if !region.profile.covers(crate::backend::nn::workspace::BoundaryStageCapacity {
                        graph:capacity.graph,records:capacity.records,backing:capacity.backing}) {
                        return Err(fail(VariableCause::Identity,c));
                    }
                }
                let claim = match first.take() { Some(claim) => claim, None => next_claim(owner, c)? };
                reserve(&c.funding, &[size_of::<Option<ErasedSharedStorageOwner>>(),
                    size_of::<ErasedSharedStorageOwner>()])?;
                let invocation = Invocation { input, empty, original, transport, peers, rank, transposed,
                    order, world, completed_source: completed_source.cloned(), claim, capacity,
                    owner: OriginalParallelControlOwner(retained.0.clone()) };
                let output = run_native_role(invocation, capacity, &owner.bank, &owner.controls, c,
                    |value, observer| Ok(value.run(observer)))
                    .map_err(|cause| Error::with_original_control_source(cause, false))??;
                let output = if let Some(plan) = plan.filter(|plan| !plan.canonical_order() && !is_idle) {
                    let mut received = destination(peers, c)?;
                    for peer in 0..peers {
                        received.push(if transposed { matrix_values[rank * peers + peer] }
                            else { matrix_values[peer * peers + rank] });
                    }
                    reorder::run(&retained, next_claim(owner, c)?, output, &received, plan.output_order(),
                        order, completed_source, executor, c,
                        region.as_ref().and_then(|region|region.profile.output_reorder))?
                } else { output };
                if axis != 0 && !is_idle {
                    reorder::axis(&retained, next_claim(owner, c)?, output, axis, true,
                        order, completed_source, executor, c)
                } else { Ok(output) }
            })();
            let result=result.and_then(|output| {
                if let Some(region)=&region {
                    let binding=self.expert_binding(context).ok_or_else(||fail(VariableCause::Identity,c))?;
                    let model=owner.request.source.model().ok_or_else(||fail(VariableCause::Identity,c))?;
                    binding.complete_expert_transfer(model,region,executor)?;
                }
                Ok(output)
            });
            if result.is_err() { owner.failed.set(true); }
            result
        })?
    }
}
// Arrays and the immutable/derived matrices retire before their request source.
struct Invocation {
    input: Array,
    empty: Option<Array>,
    original: Vec<usize>,
    transport: Vec<usize>,
    peers: usize,
    rank: usize,
    transposed: bool,
    order: usize,
    world: bool,
    completed_source: Option<ErasedSharedStorageOwner>,
    claim: ParallelControlClaim,
    capacity: AgreementCapacity,
    owner: OriginalParallelControlOwner,
}
fn next_claim(owner: &Owner, c: &Custody) -> Result<ParallelControlClaim, Error> {
    owner.request.cursor.try_borrow_mut().map_err(|_| fail(VariableCause::Identity, c))?
        .claim(ParallelControlEvent::Phase(DistributedExecutionPhase::Execution))
        .map_err(|cause| control_error(ControlCause::WorkingMemory(cause), &c.source, &c.funding))
}
fn operation<'a>(source: &'a OriginalCommunicationSource<'_>, order: usize, world: bool,
    input: &'a Array, matrix: &'a [usize], transposed: bool)
    -> Result<super::super::super::OriginalCommunicationOperation<'a>, Error> {
    if world { source.world_variable_cpu_operation_storage(order, input, matrix, transposed) }
    else { source.group_variable_cpu_operation_storage(order, input, matrix, transposed) }
}
fn capacity(source: &OriginalCommunicationSource<'_>, order: usize, world: bool, input: &Array,
    matrix: &[usize], transposed: bool, owner: &Owner, c: &Custody) -> Result<AgreementCapacity, Error> {
    let operation = operation(source, order, world, input, matrix, transposed)?;
    let inputs = owner.request.source.agreement_inputs()
        .ok_or_else(|| fail(VariableCause::Identity, c))?;
    let backing = operation.backing_storage(inputs.runtime())?.capacity();
    let operation = operation.with_completion()?;
    Ok(AgreementCapacity { graph: operation.graph_capacity(), records: operation.record_capacity(), backing })
}
impl Invocation {
    fn run(&self, observer: &OriginalScopeObserver) -> Result<Array, Error> {
        let owner = self.owner.owner();
        let c = &owner.custody;
        controls(c)?;
        reserve(&c.funding, &[
            size_of::<Self>(), size_of::<Array>(), size_of::<Result<Array, Error>>(),
            size_of::<PartitionExecutionError>(), size_of::<AgreementCapacity>(),
            size_of::<super::super::super::OriginalCommunicationConstructed>(),
            size_of::<crate::backend::runtime::distributed::completion::OriginalCommunicationCompletion>(),
            size_of::<crate::backend::runtime::distributed::completion::prepared::ReadyCompletionResources>(),
            size_of::<eredu_core::Submission<super::super::super::OriginalCommunicationConstructed,
                crate::backend::runtime::distributed::completion::OriginalCommunicationCompletion>>(),
            size_of::<Result<super::super::super::OriginalCommunicationConstructed, PartitionExecutionError>>(),
            size_of::<(Array, RetainedCommunicationSource, WorkspaceMetadataFunding)>(),
            size_of::<Option<(&Group, &CommunicationGroupDescriptor, bool)>>(),
            size_of::<(&Self, &OriginalScopeObserver)>(),
            // One actual collective output crosses this independently admitted
            // child role, including the sentinel completion for an idle rank.
            safemlx::OperationEvent::traversal_leaf_control_bytes().ok_or_else(overflow)?,
        ])?;
        let source = owner.request.source.communication_source()?;
        if self.claim.identity() != owner.request.cursor.borrow().identity()
            || self.claim.event() != ParallelControlEvent::Phase(DistributedExecutionPhase::Execution)
            || self.original.len() != self.peers.checked_mul(self.peers).ok_or_else(overflow)?
            || self.transport.len() != self.original.len()
            || self.empty.is_some() != idle(&self.original, self.peers, self.rank) {
            return Err(fail(VariableCause::Identity, c));
        }
        for (index, (&original, &actual)) in self.original.iter().zip(&self.transport).enumerate() {
            let expected = if index / self.peers == index % self.peers
                && idle(&self.original, self.peers, index / self.peers) { 1 } else { original };
            if actual != expected { return Err(fail(VariableCause::Identity, c)); }
        }
        if !self.capacity.covers(capacity(&source, self.order, self.world, &self.input, &self.transport,
            self.transposed, owner, c)?) { return Err(fail(VariableCause::Identity, c)); }
        if self.world {
            let table = self.completed_source.as_ref()
                .and_then(|source| source.downcast_ref::<super::peer_counts::table::PeerCountTable>())
                .ok_or_else(|| fail(VariableCause::Identity, c))?;
            if table.world() != Some(self.original.as_slice()) {
                return Err(fail(VariableCause::Identity, c));
            }
        }
        let operation = operation(&source, self.order, self.world, &self.input,
            &self.transport, self.transposed)?.with_completion()?;
        let ready = operation.prepare_resources(&source, if self.world { None } else { Some(self.order) })?;
        let group = if self.world { source.world() } else {
            source.group(self.order).ok_or_else(|| fail(VariableCause::Identity, c))?.0
        };
        let stream = group.retained_transport_stream().ok_or_else(|| fail(VariableCause::Identity, c))?;
        let kind = CommunicationOperation::VariableAllToAll;
        let phase = DistributedExecutionPhase::Execution;
        let (output, completion) = operation.construct_accepted(&source, observer, stream)
            .and_then(|accepted| accepted.submit(ready)).map_err(|cause| {
                source.authority.fence_protocol_failure(kind, phase, None); cause
            })?;
        let output = source.authority.wait_with_error(eredu_core::Submission { output, completion },
            kind, phase, None, |cause| PartitionExecutionError::PreparedCommunication {
                operation: kind, phase, completion: true,
                source: failure(Cause::Native(cause), &c.source, &c.funding).into_backend_failure(),
            }).map_err(|cause| fail(cause.into(), c))?;
        // The accepted wait establishes completion but can leave its Event on
        // the descriptor. Publish it under the actual child observer before
        // the parent reads metadata or starts another numerical traversal.
        // A foreign pending Event remains a strict refusal; this does not Eval.
        safemlx::OperationEvent::validate_traversal_leaf(output.value(), observer)
            .map_err(|cause| fail(cause.into(), c))?;
        if let Some(empty) = &self.empty { drop(output); retain(empty, c) }
        else { let (value, source, funding) = output.into_parts(); drop(source); drop(funding); Ok(value) }
    }
}
