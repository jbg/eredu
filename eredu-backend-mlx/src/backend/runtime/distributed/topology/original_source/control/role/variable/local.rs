//! The shared ordinary local variable itinerary under exact source-bound roles.
use super::*;
use crate::backend::{nn::logical_collective::variable::{self, LocalVariableOperations},
    runtime::distributed::{group::{LogicalExchangePlan, LogicalVariableRoutePlan},
        completion::{PreparedCommunicationWords, OriginalCommunicationCompletion}}};
use super::super::super::super::OwnedOriginalExchangeLayoutRound;
use eredu_core::ErasedSharedStorageOwner;
use crate::backend::nn::workspace::{ExpertLocalStage,ExpertLocalStageBound};
use super::super::super::super::parallel::RetainedExpertTransfer;

fn paid<T>(capacity: usize, c: &Custody) -> Result<Vec<T>, Error> {
    reserve(&c.funding, &[size_of::<Vec<T>>(), size_of::<Result<Vec<T>, Error>>(),
        std::alloc::Layout::array::<T>(capacity).map_err(|_| overflow())?.size(),
        size_of::<std::collections::TryReserveError>()])?;
    let mut values = Vec::new();
    values.try_reserve_exact(capacity).map_err(|cause| fail(cause.into(), c))?;
    Ok(values)
}
pub(super) fn execute(owner: &OriginalParallelControlOwner, first: ParallelControlClaim,
    input: &Array, counts: &CommunicationPeerCounts, axis: usize, matrix: &CommunicationPeerMatrix<'_>,
    group: &Group, plan: LogicalVariableRoutePlan<'_>, order: usize, stream: &Stream, c: &Custody, region:Option<&RetainedExpertTransfer>)
    -> Result<Array, Error> {
    reserve(&c.funding, &[size_of::<Local<'_>>(), size_of::<Array>(), size_of::<Option<ParallelControlClaim>>(),
        size_of::<Result<Array, Error>>(), size_of::<LogicalVariableRoutePlan<'_>>()])?;
    if !std::ptr::eq(plan.group(), group) || matrix.completed_source().is_none() {
        return Err(fail(VariableCause::Identity, c));
    }
    let mut first = Some(first);
    let input = if axis == 0 { retain(input, c)? } else {
        reorder::axis(owner, first.take().ok_or_else(|| fail(VariableCause::Identity, c))?,
            retain(input, c)?, axis, false, order, matrix.completed_source(), stream, c)?
    };
    let mut ops = Local { owner, first, matrix, group, order, stream, custody: c, region, next_value:0 };
    let output = variable::execute(&mut ops, &input, counts.send(), counts.receive(), plan)?;
    if axis == 0 { Ok(output) } else {
        reorder::axis(owner, ops.claim()?, output, axis, true, order, matrix.completed_source(), stream, c)
    }
}
struct Local<'a> {
    owner: &'a OriginalParallelControlOwner,
    first: Option<ParallelControlClaim>,
    matrix: &'a CommunicationPeerMatrix<'a>,
    group: &'a Group,
    order: usize,
    stream: &'a Stream,
    custody: &'a Custody,
    region:Option<&'a RetainedExpertTransfer>,
    next_value:usize,
}
impl Local<'_> {
    fn stage(&self,stage:ExpertLocalStage)->Result<Option<ExpertLocalStageBound>,Error>{
        self.region.map(|region|region.claim_local_stage(stage)).transpose()
    }
    fn claim(&mut self) -> Result<ParallelControlClaim, Error> {
        match self.first.take() { Some(first) => Ok(first), None => next_claim(self.owner.owner(), self.custody) }
    }
    fn peer<T, F>(&mut self, value: usize, step: usize, input: Array, receive_like: Option<Array>, count:bool, finish: F)
        -> Result<T, Error>
    where F: FnOnce(Array, &OriginalScopeObserver) -> Result<T, Error> {
        let c = self.custody;
        let claim = self.claim()?;
        let bound=self.stage(if count{ExpertLocalStage::Count{value,step}}else{ExpertLocalStage::Data{value,step}})?;
        reserve(&c.funding, &[size_of::<Peer>(), size_of::<Result<Peer, Error>>(),
            size_of::<F>(), size_of::<T>(), size_of::<Result<T, Error>>(),
            size_of::<Result<Result<T, Error>, eredu_core::BackendFailure>>(),
            size_of::<Option<ErasedSharedStorageOwner>>(), size_of::<ErasedSharedStorageOwner>()])?;
        let source = self.owner.owner().request.source.communication_source()?;
        let model = self.owner.owner().request.source.model().ok_or_else(|| fail(VariableCause::Identity, c))?;
        let receiver = receive_like.as_ref().unwrap_or(&input);
        let layout = source.group_variable_route_layout(self.order, value, step,
            input.shape(), receiver.shape(), input.dtype())?
            .ok_or_else(|| fail(VariableCause::Identity, c))?.try_into_owned(model)?;
        let capacity = AgreementCapacity { graph: layout.graph_capacity(), records: layout.record_capacity(),
            backing: layout.backing_capacity() };
        if let Some(bound)=bound {
            if !bound.covers_pair(input.shape(),receiver.shape(),input.dtype(),capacity.graph,capacity.records,capacity.backing,
                layout.maximum_backing_births().ok_or_else(overflow)?){return Err(self.invalid());}
        }
        let peer = Peer { input, receive_like, layout, order: self.order, value, step,
            claim, completed_source: self.matrix.completed_source().cloned(),
            owner: OriginalParallelControlOwner(self.owner.0.clone()), custody: c.clone() };
        run_native_role(peer, capacity, &self.owner.owner().native, c,
            |peer, observer| Ok(peer.run(observer).and_then(|value| finish(value, observer))))
            .map_err(|cause| Error::with_original_control_source(cause, false))?
    }
}
impl LocalVariableOperations for Local<'_> {
    type Value = Array;
    type Error = Error;
    fn invalid(&self) -> Error { fail(VariableCause::Identity, self.custody) }
    fn charge(&self, bytes: usize) -> Result<(), Error> {
        self.custody.funding.reserve_metadata(bytes).map_err(Error::WorkspacePlanning)
    }
    fn values(&self, capacity: usize) -> Result<Vec<(usize, Array)>, Error> { paid(capacity, self.custody) }
    fn rows(&self, value: &Array) -> Option<usize> { value.shape().first().and_then(|value| usize::try_from(*value).ok()) }
    fn slice(&mut self, value: &Array, counts: &[usize], destination: usize) -> Result<Array, Error> {
        let claim = self.claim()?;
        let stage=self.stage(ExpertLocalStage::Slice{value:self.next_value})?;
        self.next_value=self.next_value.checked_add(1).ok_or_else(overflow)?;
        reorder::run(self.owner, claim, retain(value, self.custody)?, counts,
            std::iter::once(destination), self.order, self.matrix.completed_source(), self.stream, self.custody,
            stage.and_then(|stage|stage.numerical))
    }
    fn exchange(&mut self, value: usize, step: usize, exchange: LogicalExchangePlan<'_>, input: Array)
        -> Result<Array, Error> {
        let c = self.custody;
        if !std::ptr::eq(exchange.group(), self.group) { return Err(self.invalid()); }
        let plan = self.group.logical_variable_route_plan().map_err(|_| self.invalid())?
            .ok_or_else(|| self.invalid())?;
        let route = plan.values().nth(value).ok_or_else(|| self.invalid())?;
        let selected = route.exchange(step).ok_or_else(|| self.invalid())?;
        if selected.peers() != exchange.peers() || selected.rounds() != 1 { return Err(self.invalid()); }
        let expected = self.matrix.count(route.source_rank(), self.group.rank()).ok_or_else(|| self.invalid())?;
        reserve(&c.funding, &[size_of::<[i32; 1]>(), size_of::<[usize; 1]>(),
            size_of::<[PreparedInputPlan<'_>; 1]>(), size_of::<[Array; 1]>(),
            size_of::<usize>(), size_of::<Result<usize, Error>>(),
            safemlx::PreparedInputRuntime::zeros_plan_control_bytes()])?;
        let source = self.owner.owner().request.source.communication_source()?;
        let runtime = self.owner.owner().request.source.agreement_inputs()
            .ok_or_else(|| self.invalid())?.runtime();
        let row = [input.shape().first().copied().ok_or_else(|| self.invalid())?];
        let shape = [1usize];
        let count = runtime.i32(&row, &shape).map_err(|cause| failure(Cause::Input(cause), &c.source, &c.funding))?;
        let [count] = super::super::super::super::inputs::construct(&source, [count])?;
        let incoming = self.peer(value, step, count, None, true, |output, observer| {
            let words = PreparedCommunicationWords::read_completed(&source, &output, observer)?;
            match words.as_slice() {
                [count] => usize::try_from(*count).map_err(|_| fail(VariableCause::Identity, c)),
                _ => Err(fail(VariableCause::Identity, c)),
            }
        })?;
        if incoming != expected { return Err(self.invalid()); }
        let stage=self.stage(ExpertLocalStage::Receive{value,step})?;
        let receiver=if incoming==0{
            // Initialized host leaves require positive dimensions. The ordinary
            // typed-zero block worker supplies this exact empty output instead.
            let rows=[usize::try_from(input.shape()[0]).map_err(|_|self.invalid())?];
            let claim=self.claim()?;
            reorder::run(self.owner,claim,retain(&input,c)?,&rows,std::iter::empty(),self.order,
                self.matrix.completed_source(),self.stream,c,stage.and_then(|stage|stage.numerical))?
        }else{
            let mut shape = paid(input.shape().len(), c)?;
            for (axis, &dimension) in input.shape().iter().enumerate() {
                shape.push(if axis == 0 { incoming } else { usize::try_from(dimension).map_err(|_| self.invalid())? });
            }
            let plan = runtime.zeros(input.dtype(), &shape)
                .map_err(|cause| failure(Cause::Input(cause), &c.source, &c.funding))?;
            let [receiver] = super::super::super::super::inputs::construct(&source, [plan])?;
            receiver
        };
        self.peer(value, step, input, Some(receiver), false, |output, _| Ok(output))
    }
    fn concatenate(&mut self, values: Vec<(usize, Array)>) -> Result<Array, Error> {
        let mut inputs = paid(values.len(), self.custody)?;
        inputs.extend(values.into_iter().map(|(_, input)| input));
        let claim = self.claim()?;
        let stage=self.stage(ExpertLocalStage::Join)?;
        reorder::concatenate(self.owner, claim, inputs, self.order,
            self.matrix.completed_source(), self.stream, self.custody,stage.and_then(|stage|stage.numerical))
    }
}
// Numerical owners and accepted count evidence retire before the request source.
struct Peer {
    input: Array,
    receive_like: Option<Array>,
    layout: OwnedOriginalExchangeLayoutRound,
    order: usize,
    value: usize,
    step: usize,
    claim: ParallelControlClaim,
    completed_source: Option<ErasedSharedStorageOwner>,
    owner: OriginalParallelControlOwner,
    custody: Custody,
}
impl Peer {
    fn run(&self, observer: &OriginalScopeObserver) -> Result<Array, Error> {
        let c = &self.custody;
        reserve(&c.funding, &[size_of::<Self>(), size_of::<[Array; 2]>(),
            size_of::<OriginalCommunicationCompletion>(), size_of::<Result<Array, Error>>(),
            size_of::<super::super::super::super::ConstructedRouteExchange>(),
            safemlx::OperationEvent::traversal_leaf_control_bytes().and_then(|n| n.checked_mul(2)).ok_or_else(overflow)?])?;
        let owner = self.owner.owner();
        if self.claim.identity() != owner.request.cursor.borrow().identity()
            || self.claim.event() != ParallelControlEvent::Phase(DistributedExecutionPhase::Execution)
            || self.completed_source.as_ref().and_then(|source|
                source.downcast_ref::<super::super::peer_counts::table::PeerCountTable>()).is_none() {
            return Err(fail(VariableCause::Identity, c));
        }
        let source = owner.request.source.communication_source()?;
        let (group, _, _) = source.group(self.order).ok_or_else(|| fail(VariableCause::Identity, c))?;
        let plan = group.logical_variable_route_plan().map_err(|_| fail(VariableCause::Identity, c))?
            .ok_or_else(|| fail(VariableCause::Identity, c))?;
        if plan.values().nth(self.value).and_then(|route| route.exchange(self.step)).is_none() {
            return Err(fail(VariableCause::Identity, c));
        }
        let runtime = owner.request.source.agreement_inputs().ok_or_else(|| fail(VariableCause::Identity, c))?.runtime();
        let stream = group.retained_transport_stream().ok_or_else(|| fail(VariableCause::Identity, c))?;
        let receiver = self.receive_like.as_ref().unwrap_or(&self.input);
        let round = self.layout.bind_actual_pair(&source, &self.input, receiver)?.prepare(&source, runtime)?;
        if round.graph_capacity() > self.layout.graph_capacity() || round.record_capacity() > self.layout.record_capacity()
            || round.backing_capacity() > self.layout.backing_capacity() { return Err(fail(VariableCause::Identity, c)); }
        let kind = CommunicationOperation::VariableAllToAll; let phase = DistributedExecutionPhase::Execution;
        let (output, completion) = round.construct_accepted(&source, observer, stream)?.submit()?;
        let output = source.authority.wait_with_error(eredu_core::Submission { output, completion }, kind, phase, None,
            |cause| PartitionExecutionError::PreparedCommunication { operation: kind, phase, completion: true,
                source: failure(Cause::Native(cause), &c.source, &c.funding).into_backend_failure() })
            .map_err(|cause| fail(cause.into(), c))?;
        for value in output.outputs() {
            safemlx::OperationEvent::validate_traversal_leaf(value, observer).map_err(|cause| fail(cause.into(), c))?;
        }
        let ([sent, received], source, funding) = output.into_parts();
        drop(sent); drop(source); drop(funding);
        Ok(received)
    }
}
