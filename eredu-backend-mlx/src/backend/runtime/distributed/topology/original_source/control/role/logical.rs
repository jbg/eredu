//! Ordinary logical subgroup equations under exact accepted child roles.
//! Initial partial completion borrows model Q; child recovery owns only the
//! actual tensors, array-free source and nonrefunding request event claim.
use super::super::super::parallel::RetainedLogicalCollective;
use super::*;
use crate::backend::{
    nn::{
        logical_collective::{self, Operations},
        workspace::{
            ExistingArrayProjection, LogicalCollectiveKind, LogicalCollectiveQuote, MlxMetalWorkspaceMechanisms, ResidentExecutionMechanisms, selected_parallel_numerical,
            SpeculativeNumericalRecipe,
        },
    },
    runtime::distributed::completion::{
        prepared::{CompletionResourceLayout, PreparedCompletionResources},
        OriginalCommunicationCompletion,
    },
    submission_recovery::prefill::TransientRootsProjection,
};
use eredu_nn::workspace::{WorkspaceContext, WorkspaceTraceReport};
use eredu_runtime::{DistributedExecutionPhase, PartitionExecutionError};
mod packed;
mod routed;
use safemlx::{Array, PreparedNestedRoots, PreparedStreamCopy, StreamCopyPlan};

#[derive(Debug, thiserror::Error)]
enum LogicalCause {
    #[error("logical subgroup source differs from its exact model occurrence")]
    Identity,
    #[error("logical subgroup numerical source is unavailable on this device")]
    Worker,
    #[error(transparent)]
    Neural(#[from] eredu_nn::Error),
    #[error(transparent)]
    Native(#[from] safemlx::error::Exception),
    #[error(transparent)]
    Buffer(#[from] safemlx::OriginalBufferCause),
    #[error(transparent)]
    Stream(#[from] safemlx::StreamCopyCause),
    #[error(transparent)]
    Clone(#[from] safemlx::PreparedArrayCloneCause),
    #[error(transparent)]
    Communication(#[from] PartitionExecutionError),
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct LogicalFailure {
    #[source]
    cause: LogicalCause,
    custody: Custody,
}
fn fail(cause: LogicalCause, c: &Custody) -> Error {
    Error::with_original_control_source(
        eredu_core::BackendFailure::from_error(LogicalFailure {
            cause,
            custody: c.clone(),
        }),
        false,
    )
}
fn failure_bytes() -> Option<usize> {
    [
        size_of::<LogicalFailure>(),
        size_of::<LogicalCause>(),
        size_of::<Custody>(),
        size_of::<Error>(),
        eredu_core::BackendFailure::source_retention_peak_bytes::<LogicalFailure>()?,
    ]
    .into_iter()
    .try_fold(size_of::<[usize; 5]>(), usize::checked_add)
}
fn claim(owner: &Owner, c: &Custody) -> Result<ParallelControlClaim, Error> {
    reserve(
        &c.funding,
        &[
            size_of::<ParallelControlClaim>(),
            size_of::<Result<ParallelControlClaim, Error>>(),
            failure_bytes().ok_or_else(overflow)?,
        ],
    )?;
    owner
        .request
        .cursor
        .try_borrow_mut()
        .map_err(|_| fail(LogicalCause::Identity, c))?
        .claim(ParallelControlEvent::Phase(
            DistributedExecutionPhase::Execution,
        ))
        .map_err(|_| fail(LogicalCause::Identity, c))
}
fn validate_claim(claim: &ParallelControlClaim, owner: &Owner, c: &Custody) -> Result<(), Error> {
    if claim.identity() != owner.request.cursor.borrow().identity()
        || claim.event() != ParallelControlEvent::Phase(DistributedExecutionPhase::Execution)
    {
        return Err(fail(LogicalCause::Identity, c));
    }
    Ok(())
}
fn retained_array(input: &Array, c: &Custody) -> Result<Array, Error> {
    reserve(
        &c.funding,
        &[
            safemlx::PreparedArrayClone::control_bytes().ok_or_else(overflow)?,
            Array::inspection_clone_handle_bytes(),
            size_of::<Result<Array, Error>>(),
            failure_bytes().ok_or_else(overflow)?,
        ],
    )?;
    let mut slot = safemlx::PreparedArrayClone::try_prepare_for_inspection()
        .map_err(|cause| fail(cause.into(), c))?;
    slot.fill_for_inspection(input)
        .map_err(|cause| fail(cause.into(), c))
}
fn wait<T>(
    source: &OriginalCommunicationSource<'_>,
    output: T,
    completion: OriginalCommunicationCompletion,
    operation: CommunicationOperation,
    c: &Custody,
) -> Result<T, Error> {
    reserve(
        &c.funding,
        &[
            size_of::<T>(),
            size_of::<Result<T, Error>>(),
            size_of::<eredu_core::Submission<T, OriginalCommunicationCompletion>>(),
            size_of::<PartitionExecutionError>(),
            failure_bytes().ok_or_else(overflow)?,
        ],
    )?;
    let phase = DistributedExecutionPhase::Execution;
    source
        .authority
        .wait_with_error(
            eredu_core::Submission { output, completion },
            operation,
            phase,
            None,
            |cause| PartitionExecutionError::PreparedCommunication {
                operation,
                phase,
                completion: true,
                source: fail(LogicalCause::Native(cause), c).into_backend_failure(),
            },
        )
        .map_err(|cause| fail(cause.into(), c))
}
impl OriginalParallelControlProjection {
    pub(crate) fn execute_logical_collective(
        &self,
        group: &Group,
        input: &Array,
        stream: &Stream,
        roots: &TransientRootsProjection,
        quote: RetainedLogicalCollective,
    ) -> Result<Array, Error> {
        let c = &self.custody;
        reserve(
            &c.funding,
            &[
                size_of::<RetainedLogicalCollective>(),
                size_of::<Pair>(),
                size_of::<Arithmetic>(),
                size_of::<OriginalParallelControlOwner>(),
                size_of::<[Array; 2]>(),
                size_of::<Array>(),
                size_of::<Result<Array, Error>>(),
                size_of::<Result<[Array; 2], Error>>(),
                failure_bytes().ok_or_else(overflow)?,
            ],
        )?;
        let owner = self
            .upgrade()
            .map_err(|cause| Error::with_original_control_source(cause, false))?;
        let mut first = Some(claim(owner.owner(), c)?);
        if owner.owner().failed.get() || owner.owner().running.replace(true) {
            owner.owner().failed.set(true);
            return Err(fail(LogicalCause::Identity, c));
        }
        let _running = Running {
            running: &owner.owner().running,
            failed: &owner.owner().failed,
        };
        let result = execute_claimed(&owner,
            first.take().ok_or_else(|| fail(LogicalCause::Identity, c))?,
            group, input, stream, quote, InputPreparation::Model(roots), None, |output, _| Ok(output));
        if result.is_err() {
            owner.owner().failed.set(true);
        }
        result
    }
}
pub(super) type WorldObserver<'a> = &'a mut dyn FnMut(&Array, &OriginalScopeObserver) -> Result<(), Error>;
enum InputPreparation<'a> {
    Model(&'a TransientRootsProjection),
    Completed(&'a super::super::super::inputs::CompletedCommunicationInput),
}
/// The caller already owns the actual request event and running guard. Only the
/// closed source constructor can supply this completed numerical input witness.
pub(super) fn execute_source<T, F>(owner: &OriginalParallelControlOwner,
    first: ParallelControlClaim, group: &Group,
    input: &super::super::super::inputs::CompletedCommunicationInput,
    stream: &Stream, quote: RetainedLogicalCollective, world: Option<WorldObserver<'_>>, finish: F) -> Result<T, Error>
where F: FnOnce(Array, &OriginalScopeObserver) -> Result<T, Error> {
    execute_claimed(owner, first, group, input.value(), stream, quote,
        InputPreparation::Completed(input), world, finish)
}
fn execute_claimed<T, F>(owner: &OriginalParallelControlOwner, first: ParallelControlClaim,
    group: &Group, input: &Array, stream: &Stream, quote: RetainedLogicalCollective,
    preparation: InputPreparation<'_>, world: Option<WorldObserver<'_>>, finish: F) -> Result<T, Error>
where F: FnOnce(Array, &OriginalScopeObserver) -> Result<T, Error> {
    let c = &owner.owner().custody;
    reserve(&c.funding, &[size_of::<F>(), size_of::<T>(), size_of::<Result<T, Error>>(),
        size_of::<InputPreparation<'_>>(), size_of::<Option<WorldObserver<'_>>>(), size_of::<Option<ParallelControlClaim>>(),
        size_of::<RetainedLogicalCollective>(), size_of::<Pair>(), size_of::<Arithmetic>(),
        size_of::<OriginalParallelControlOwner>(), size_of::<[Array; 2]>(),
        size_of::<Result<[Array; 2], Error>>(), failure_bytes().ok_or_else(overflow)?])?;
    validate_claim(&first, owner.owner(), c)?;
    let mut first = Some(first);
    (|| {
            let source = owner.owner().request.source.communication_source()?;
            let model = owner
                .owner()
                .request
                .source
                .model()
                .ok_or_else(|| fail(LogicalCause::Identity, c))?;
            let value = quote.value();
            if !value.matches_source(model)
                || !source.matches_group(value.order, group)
                || !group.is_logical()
                || value.rank != group.rank()
                || value.input.shape() != input.shape()
                || value.native_dtype != input.dtype()
            {
                return Err(fail(LogicalCause::Identity, c));
            }
            match preparation {
                InputPreparation::Model(roots) => complete_partial(&source, roots, input, stream, &quote, c)?,
                InputPreparation::Completed(completed) => {
                    if !completed.belongs_to(&source) || !std::ptr::eq(completed.value(), input) {
                        return Err(fail(LogicalCause::Identity, c));
                    }
                }
            }
            if value.routed().is_some() {
                drop(source);
                return routed::execute_finish(owner, quote.clone(), input, stream,
                    first.take().ok_or_else(|| fail(LogicalCause::Identity, c))?, c, finish);
            }
            if value.packed().is_some(){
                drop(source);
                return packed::execute_finish(owner,quote.clone(),input,stream,first.take().ok_or_else(||fail(LogicalCause::Identity,c))?,c,world,finish);
            }
            let exchange=source.group_exchange(value.order)?.ok_or_else(||fail(LogicalCause::Identity,c))?;
            let plan=exchange.plan();
            let pairs=value.exchange_pairs().ok_or_else(||fail(LogicalCause::Identity,c))?;
            if plan.rounds()!=pairs.len(){return Err(fail(LogicalCause::Identity,c));}
            let original = retained_array(input, c)?;
            let next_round=|round:usize, previous:Array| {
                let pair_source=pairs.get(round).ok_or_else(||fail(LogicalCause::Identity,c))?;
                let capacity=AgreementCapacity {graph:pair_source.graph_capacity(),
                    records:pair_source.record_capacity(),backing:pair_source.backing_capacity()};
                let pair=Pair {input:previous,round,quote:quote.clone(),
                    claim:match first.take(){Some(first)=>first,None=>claim(owner.owner(),c)?},
                    owner:OriginalParallelControlOwner(owner.0.clone()),custody:c.clone()};
                let outputs=run_native_role(pair,capacity,&owner.owner().bank,&owner.owner().controls,c,
                    |pair,observer|Ok(pair.run(observer)))
                    .map_err(|cause|Error::with_original_control_source(cause,false))??;
                run_arithmetic(&owner,quote.clone(),ArithmeticStage::Peer,outputs,stream,c)
            };
            reserve(&c.funding,&[plan.iteration_control_bytes::<Array,Error,_>(&next_round)
                .ok_or_else(overflow)?,size_of::<Option<ParallelControlClaim>>()])?;
            let peer=plan.fold_rounds(retained_array(input,c)?,next_round)?;
            drop(exchange);
            drop(source);
            run_arithmetic_finish(owner, quote.clone(), ArithmeticStage::Result,
                [original, peer], stream, c, |output, observer| {
                    let expected = quote.value();
                    let shape = match expected.kind {
                        LogicalCollectiveKind::Sum => output.shape() == input.shape(),
                        LogicalCollectiveKind::Gather if input.shape().is_empty() => output.shape() == [2],
                        LogicalCollectiveKind::Gather => output.shape().len() == input.shape().len()
                            && input.shape()[0].checked_mul(2) == output.shape().first().copied()
                            && output.shape().iter().skip(1).eq(input.shape().iter().skip(1)),
                    };
                    if !shape || output.dtype() != expected.native_dtype {
                        return Err(fail(LogicalCause::Identity, c));
                    }
                    finish(output, observer)
                })
    })()
}
fn complete_partial(
    source: &OriginalCommunicationSource<'_>,
    projection: &TransientRootsProjection,
    input: &Array,
    stream: &Stream,
    quote: &RetainedLogicalCollective,
    c: &Custody,
) -> Result<(), Error> {
    reserve(
        &c.funding,
        &[
            size_of::<PreparedNestedRoots<Custody>>(),
            size_of::<[&Array; 1]>(),
            size_of::<OriginalCommunicationCompletion>(),
            size_of::<Result<(), Error>>(),
            TransientRootsProjection::boundary_traversal_control_bytes().ok_or_else(overflow)?,
            PreparedNestedRoots::<Custody>::control_bytes(1).ok_or_else(overflow)?,
            safemlx::OperationEvent::traversal_leaf_control_bytes().ok_or_else(overflow)?,
            failure_bytes().ok_or_else(overflow)?,
        ],
    )?;
    let (observer, traversal) = projection.boundary_traversal(1, &c.funding)?;
    let mut roots = PreparedNestedRoots::try_new(1, c.clone()).map_err(|failed| {
        let (cause, custody) = failed.into_parts();
        let error = failure(
            Cause::BoundaryRoots(cause),
            &custody.source,
            &custody.funding,
        );
        drop(custody);
        error
    })?;
    let mut ready = PreparedCompletionResources::prepare_original(
        source,
        CompletionResourceLayout {
            arrays: 0,
            counts: &[],
            groups: 1,
            routes: 0,
            streams: 0,
        },
        traversal,
    )?;
    ready.retain_group(quote.value().order)?;
    let completion =
        ready
            .finish()?
            .submit_model_nested(source, &observer, stream, &mut roots, [input], 1)?;
    wait(source, (), completion, quote.value().protocol, c)?;
    safemlx::OperationEvent::validate_traversal_leaf(input, &observer)
        .map_err(|cause| fail(cause.into(), c))
}
struct Pair {
    input: Array,
    round: usize,
    quote: RetainedLogicalCollective,
    claim: ParallelControlClaim,
    owner: OriginalParallelControlOwner,
    custody: Custody,
}
impl Pair {
    fn run(&self, observer: &OriginalScopeObserver) -> Result<[Array; 2], Error> {
        let c = &self.custody;
        let owner = self.owner.owner();
        let value = self.quote.value();
        reserve(
            &c.funding,
            &[
                size_of::<Self>(),
                size_of::<[Array; 2]>(),
                size_of::<Result<[Array; 2], Error>>(),
                safemlx::OperationEvent::traversal_leaf_control_bytes()
                    .and_then(|n| n.checked_mul(2))
                    .ok_or_else(overflow)?,
                failure_bytes().ok_or_else(overflow)?,
            ],
        )?;
        validate_claim(&self.claim, owner, c)?;
        let source = owner.request.source.communication_source()?;
        let group = source.group(value.order).ok_or_else(|| fail(LogicalCause::Identity, c))?.0;
        let rounds = if let Some((route, step)) = value.routed_step() {
            let selected = group.logical_routed_plan().map_err(|_| fail(LogicalCause::Identity, c))?
                .and_then(|plan| plan.value(route)).and_then(|route| route.exchange(step))
                .ok_or_else(|| fail(LogicalCause::Identity, c))?;
            selected.rounds()
        } else { source.group_exchange(value.order)?.ok_or_else(|| fail(LogicalCause::Identity, c))?.rounds() };
        let runtime = owner
            .request
            .source
            .agreement_inputs()
            .ok_or_else(|| fail(LogicalCause::Identity, c))?
            .runtime();
        let stream = group.retained_transport_stream()
            .ok_or_else(|| fail(LogicalCause::Identity, c))?;
        let selected=value.exchange_pairs().and_then(|pairs|pairs.get(self.round)).ok_or_else(||fail(LogicalCause::Identity,c))?;
        if self.round>=rounds{return Err(fail(LogicalCause::Identity,c));}
        let round=selected.bind_actual(&source,&self.input)?.prepare(&source,runtime)?;
        if round.graph_capacity()>selected.graph_capacity()
            || round.record_capacity()>selected.record_capacity()
            || round.backing_capacity()>selected.backing_capacity() {
            return Err(fail(LogicalCause::Identity,c));
        }
        let (outputs, completion) = round
            .construct_accepted(&source, observer, stream)?
            .submit()?;
        let outputs = wait(&source, outputs, completion, value.protocol, c)?;
        for output in outputs.outputs() {
            safemlx::OperationEvent::validate_traversal_leaf(output, observer)
                .map_err(|cause| fail(cause.into(), c))?;
        }
        let (outputs, source, funding) = outputs.into_parts();
        drop((source, funding));
        Ok(outputs)
    }
}
#[derive(Clone, Copy)]
enum ArithmeticStage {
    Peer,
    Result,
    Pack,
    Unpack,
    RoutedResult,
}
enum ArithmeticInputs<T> { Pair([T; 2]), Routed(Vec<(usize, T)>) }
impl<T> ArithmeticInputs<T> {
    fn len(&self) -> usize { match self { Self::Pair(_) => 2, Self::Routed(values) => values.len() } }
    fn iter(&self) -> impl Iterator<Item = &T> {
        let pair = match self { Self::Pair(values) => values.as_slice(), Self::Routed(_) => &[] };
        let routed = match self { Self::Routed(values) => values.as_slice(), Self::Pair(_) => &[] };
        pair.iter().chain(routed.iter().map(|(_, value)| value))
    }
}
struct Arithmetic {
    inputs: ArithmeticInputs<Array>,
    quote: RetainedLogicalCollective,
    stage: ArithmeticStage,
    stream: PreparedStreamCopy<Custody>,
    _report: WorkspaceTraceReport,
    _context: WorkspaceContext,
    recipe: SpeculativeNumericalRecipe,
    capacity: AgreementCapacity,
    claim: ParallelControlClaim,
    owner: OriginalParallelControlOwner,
    custody: Custody,
}
fn arithmetic<O: logical_collective::packed::PackedOperations>(
    ops:&O,inputs:&[O::Value;2],stage:ArithmeticStage,quote:&LogicalCollectiveQuote,
)->Result<O::Value,O::Error>{
    ops.charge(size_of::<(ArithmeticStage,&LogicalCollectiveQuote,&[O::Value;2],Result<O::Value,O::Error>)>())?;
    match stage {
        ArithmeticStage::RoutedResult => Err(ops.invalid()),
        ArithmeticStage::Peer=>{let zero=ops.zero(&inputs[0])?;logical_collective::peer(ops,&inputs[0],&inputs[1],&zero)},
        ArithmeticStage::Result=>match quote.kind {
            LogicalCollectiveKind::Sum=>logical_collective::sum(ops,&inputs[0],&inputs[1]),
            LogicalCollectiveKind::Gather=>logical_collective::gather(ops,&inputs[0],&inputs[1],quote.rank==0),
        },
        ArithmeticStage::Pack=>{
            let source=quote.packed().ok_or_else(||ops.invalid())?;
            let world=usize::try_from(source.world.shape()[0]).map_err(|_|ops.invalid())?;
            logical_collective::packed::pack(ops,&inputs[0],source.slot,world)
        }
        ArithmeticStage::Unpack=>{
            let source=quote.packed().ok_or_else(||ops.invalid())?;
            match quote.kind {
                LogicalCollectiveKind::Sum=>logical_collective::packed::sum_result(ops,&inputs[0],source.slot),
                LogicalCollectiveKind::Gather=>{
                    let stacked=logical_collective::packed::gather_stacked(ops,&inputs[0],&source.member_ranks)?;
                    logical_collective::packed::flatten(ops,&stacked,quote.input.shape(),source.members)
                }
            }
        }
    }
}

fn run_arithmetic(
    owner: &OriginalParallelControlOwner,
    quote: RetainedLogicalCollective,
    stage: ArithmeticStage,
    inputs: [Array; 2],
    stream: &Stream,
    c: &Custody,
) -> Result<Array, Error> {
    let claim = claim(owner.owner(), c)?;
    run_arithmetic_claim(owner,quote,stage,inputs,stream,claim,c)
}
fn run_arithmetic_claim(owner:&OriginalParallelControlOwner,quote:RetainedLogicalCollective,
    stage:ArithmeticStage,inputs:[Array;2],stream:&Stream,claim:ParallelControlClaim,c:&Custody)->Result<Array,Error>{
    run_arithmetic_finish_claim(owner,quote,stage,inputs,stream,claim,c,|value,_|Ok(value))
}
fn run_arithmetic_finish<T,F>(owner:&OriginalParallelControlOwner,quote:RetainedLogicalCollective,
    stage:ArithmeticStage,inputs:[Array;2],stream:&Stream,c:&Custody,finish:F)->Result<T,Error>
where F:FnOnce(Array,&OriginalScopeObserver)->Result<T,Error>{
    let claim=claim(owner.owner(),c)?;
    run_arithmetic_finish_claim(owner,quote,stage,inputs,stream,claim,c,finish)
}
fn run_arithmetic_finish_claim<T,F>(owner:&OriginalParallelControlOwner,quote:RetainedLogicalCollective,
    stage:ArithmeticStage,inputs:[Array;2],stream:&Stream,claim:ParallelControlClaim,c:&Custody,finish:F)->Result<T,Error>
where F:FnOnce(Array,&OriginalScopeObserver)->Result<T,Error>{
    run_arithmetic_finish_inputs(owner, quote, stage, ArithmeticInputs::Pair(inputs), stream, claim, c, finish)
}
fn run_arithmetic_finish_inputs<T,F>(owner:&OriginalParallelControlOwner,quote:RetainedLogicalCollective,
    stage:ArithmeticStage,inputs:ArithmeticInputs<Array>,stream:&Stream,claim:ParallelControlClaim,c:&Custody,finish:F)->Result<T,Error>
where F:FnOnce(Array,&OriginalScopeObserver)->Result<T,Error>{
    reserve(&c.funding,&[size_of::<F>(),size_of::<T>(),size_of::<Result<T,Error>>(),
        size_of::<Result<Result<T,Error>,eredu_core::BackendFailure>>(), size_of::<ArithmeticInputs<Array>>()])?;
    let plan = Arithmetic::prepare(
        inputs,
        quote,
        stage,
        stream,
        claim,
        OriginalParallelControlOwner(owner.0.clone()),
        c.clone(),
    )?;
    let capacity = plan.capacity;
    let kernels = plan.recipe.kernels;
    run_native_role_with_pipeline(
        plan,
        capacity,
        Some(safemlx::PreparedPipelineCachePlan::new(kernels)),
        &owner.owner().bank,
        &owner.owner().controls,
        c,
        |plan, observer| Ok(plan.run(observer).and_then(|value| finish(value, observer))),
    )
    .map_err(|cause| Error::with_original_control_source(cause, false))?
}
impl Arithmetic {
    #[inline(never)]
    fn prepare(
        inputs: ArithmeticInputs<Array>,
        quote: RetainedLogicalCollective,
        stage: ArithmeticStage,
        stream: &Stream,
        claim: ParallelControlClaim,
        owner: OriginalParallelControlOwner,
        custody: Custody,
    ) -> Result<Self, Error> {
        let c = &custody;
        let funding = &c.funding;
        reserve(
            funding,
            &[
                size_of::<Self>(),
                size_of::<Result<Self, Error>>(),
                size_of::<WorkspaceContext>(),
                size_of::<WorkspaceTraceReport>(),
                size_of::<SpeculativeNumericalRecipe>(),
                size_of::<AgreementCapacity>(),
                size_of::<ExistingArrayProjection<'_>>(),
                size_of::<ArithmeticInputs<eredu_nn::workspace::WorkspaceTensor>>(),
                size_of::<Result<ArithmeticInputs<eredu_nn::workspace::WorkspaceTensor>, Error>>(),
                size_of::<StreamCopyPlan<Custody>>(),
                size_of::<PreparedStreamCopy<Custody>>(),
                Stream::device_type_control_bytes().ok_or_else(overflow)?,
                failure_bytes().ok_or_else(overflow)?,
            ],
        )?;
        #[cfg(not(all(target_vendor = "apple", feature = "metal", not(feature = "cuda"))))]
        return Err(fail(LogicalCause::Worker, c));
        #[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
        {
            let mechanism = MlxMetalWorkspaceMechanisms::current_host()
                .map_err(|cause| fail(cause.into(), c))?;
            let mechanism = ResidentExecutionMechanisms::from_stream(mechanism, stream, funding)
                .map_err(|cause| fail(cause.into(), c))?;
            let context = WorkspaceContext::new_with_metadata_funding(mechanism, funding.clone())
                .map_err(|cause| fail(LogicalCause::Neural(cause.into()), c))?;
            let mut projection = ExistingArrayProjection::with_source_count(&context, inputs.len())
                .map_err(|cause| fail(LogicalCause::Neural(context.metadata_source(cause)), c))?;
            let projected = match &inputs {
                ArithmeticInputs::Pair(values) => ArithmeticInputs::Pair([
                    projection.project(&values[0]).map_err(|cause| fail(cause.into(), c))?,
                    projection.project(&values[1]).map_err(|cause| fail(cause.into(), c))?,
                ]),
                ArithmeticInputs::Routed(values) => {
                    let selected = quote.value().routed().ok_or_else(|| fail(LogicalCause::Identity, c))?;
                    if values.len() != selected.values.len() { return Err(fail(LogicalCause::Identity, c)); }
                    let mut projected = context.metadata_vec(values.len())
                        .map_err(|cause| fail(LogicalCause::Neural(cause.into()), c))?;
                    for ((rank, value), expected) in values.iter().zip(&selected.values) {
                        if *rank != expected.source_rank { return Err(fail(LogicalCause::Identity, c)); }
                        projected.push((*rank, projection.project(value).map_err(|cause| fail(cause.into(), c))?));
                    }
                    ArithmeticInputs::Routed(projected)
                }
            };
            if !projection.is_complete() {
                return Err(fail(LogicalCause::Identity, c));
            }
            drop(projection);
            match stage {
                ArithmeticStage::Peer|ArithmeticStage::Result|ArithmeticStage::Pack|ArithmeticStage::RoutedResult=>{
                    for value in inputs.iter() {
                        if value.shape()!=quote.value().input.shape() || value.dtype()!=quote.value().native_dtype {
                            return Err(fail(LogicalCause::Identity,c));
                        }
                    }
                }
                ArithmeticStage::Unpack=>{
                    let ArithmeticInputs::Pair(inputs) = &inputs else { return Err(fail(LogicalCause::Identity, c)); };
                    let selected=quote.value().packed().ok_or_else(||fail(LogicalCause::Identity,c))?;
                    if inputs[0].shape()!=selected.world.shape() || inputs[0].dtype()!=quote.value().native_dtype
                        || inputs[1].shape()!=inputs[0].shape()
                        || inputs[1].dtype()!=inputs[0].dtype() {return Err(fail(LogicalCause::Identity,c));}
                }
            }
            context.begin_span();
            let output = match projected {
                ArithmeticInputs::Pair(projected) => arithmetic(
                    &logical_collective::Workspace(&context), &projected, stage, quote.value()),
                ArithmeticInputs::Routed(projected) => {
                    if !matches!(stage, ArithmeticStage::RoutedResult) { return Err(fail(LogicalCause::Identity, c)); }
                    routed::result(&logical_collective::Workspace(&context), projected, quote.value())
                }
            }.map_err(|cause| fail(cause.into(), c))?;
            let report = context
                .finish_report(&[output])
                .map_err(|cause| fail(cause.into(), c))?;
            let recipe = selected_parallel_numerical(&report, 1, mechanism, &context)
                .map_err(|cause| fail(cause.into(), c))?;
            let runtime = owner
                .owner()
                .request
                .source
                .agreement_inputs()
                .ok_or_else(|| fail(LogicalCause::Identity, c))?
                .runtime();
            let backing = OriginalBufferBudget::population_layout(
                runtime,
                usize::try_from(recipe.storage.mutable_bytes()).map_err(|_| overflow())?,
                recipe.storage.maximum_births(),
            )
            .map_err(|cause| fail(cause.into(), c))?;
            let selected=match stage {
                ArithmeticStage::Peer|ArithmeticStage::Result=>quote.value().arithmetic_capacity(matches!(stage,ArithmeticStage::Peer)),
                ArithmeticStage::Pack=>quote.value().packed().map(|value|value.pack_capacity),
                ArithmeticStage::Unpack=>quote.value().packed().map(|value|value.result_capacity),
                ArithmeticStage::RoutedResult=>quote.value().routed().map(|value|value.result_capacity),
            }.ok_or_else(||fail(LogicalCause::Identity,c))?;
            let capacity = AgreementCapacity {
                graph: selected.graph,
                records: selected.records,
                backing: selected.backing,
            };
            if !capacity.covers(AgreementCapacity {
                graph: recipe.graph_capacity,
                records: recipe.record_capacity,
                backing: backing.capacity(),
            }) {
                return Err(fail(LogicalCause::Identity, c));
            }
            reserve(
                funding,
                &[
                    usize::try_from(recipe.controls).map_err(|_| overflow())?,
                    logical_collective::control_bytes::<logical_collective::Native<'_>>()
                        .and_then(|n| n.checked_mul(3))
                        .ok_or_else(overflow)?,
                    size_of::<(
                        ArithmeticStage,
                        LogicalCollectiveKind,
                        usize,
                        &[Array; 2],
                        Result<Array, safemlx::error::Exception>,
                    )>(),
                    // The ordinary first-axis reshape owns exactly this shape vector.
                    Layout::array::<i32>(quote.value().input.shape().len())
                        .map_err(|_| overflow())?
                        .size(),
                    size_of::<Vec<i32>>(),
                    logical_collective::packed::controls::<logical_collective::Native<'_>>(quote.value().input.shape().len()+1)
                        .and_then(|n|n.checked_mul(2)).ok_or_else(overflow)?,
                    logical_collective::packed::gather_controls::<logical_collective::Native<'_>>(
                        quote.value().packed().map_or(0,|source|source.members)).ok_or_else(overflow)?,
                    Array::static_slice_update_control_bytes().ok_or_else(overflow)?,
                    logical_collective::routed::result_controls::<logical_collective::Native<'_>>(inputs.len())
                        .ok_or_else(overflow)?,
                    safemlx::OperationEvent::traversal_leaf_control_bytes()
                        .and_then(|n| n.checked_mul(inputs.len().checked_add(1)?))
                        .ok_or_else(overflow)?,
                ],
            )?;
            let stream = StreamCopyPlan::<Custody>::capture(stream)
                .map_err(|cause| fail(cause.into(), c))?;
            reserve(
                funding,
                &[
                    stream.control_bytes().ok_or_else(overflow)?,
                    stream.native_wrapper_bytes(),
                    stream.owner_node_layout().size(),
                    Layout::new::<[usize; 2]>()
                        .extend(stream.shared_body_layout())
                        .map_err(|_| overflow())?
                        .0
                        .pad_to_align()
                        .size(),
                ],
            )?;
            let stream = stream.realize(c.clone()).map_err(|error| {
                let (cause, owner) = error.into_parts();
                let error = fail(cause.into(), c);
                drop(owner);
                error
            })?;
            Ok(Self {
                inputs,
                quote,
                stage,
                stream,
                _report: report,
                _context: context,
                recipe,
                capacity,
                claim,
                owner,
                custody,
            })
        }
    }
    fn run(&self, observer: &OriginalScopeObserver) -> Result<Array, Error> {
        let c = &self.custody;
        let stream = self.stream.as_stream();
        let value = self.quote.value();
        reserve(
            &c.funding,
            &[
                size_of::<Array>(),
                size_of::<[Array; 1]>(),
                size_of::<Result<Array, Error>>(),
                size_of::<OriginalCommunicationCompletion>(),
                size_of::<Option<Vec<(usize, Array)>>>(),
                size_of::<Result<Vec<(usize, Array)>, Error>>(),
                size_of::<std::slice::Iter<'_, (usize, Array)>>(),
                size_of::<ArithmeticInputs<Array>>(),
                failure_bytes().ok_or_else(overflow)?,
            ],
        )?;
        validate_claim(&self.claim, self.owner.owner(), c)?;
        let source = self.owner.owner().request.source.communication_source()?;
        let mut ready = PreparedCompletionResources::prepare_original(
            &source,
            CompletionResourceLayout {
                arrays: 0,
                counts: &[],
                groups: 1,
                routes: 0,
                streams: 0,
            },
            self.recipe.completion.traversal,
        )?;
        ready.retain_group(value.order)?;
        let ready = ready.finish()?;
        for input in self.inputs.iter() {
            safemlx::OperationEvent::validate_traversal_leaf(input, observer)
                .map_err(|cause| fail(cause.into(), c))?;
        }
        // Routed result assembly consumes paid aliases; the invocation retains
        // the original roots through every completion or failure outcome.
        let routed_values = match &self.inputs {
            ArithmeticInputs::Pair(_) => None,
            ArithmeticInputs::Routed(values) => {
                let mut aliases = routed::destination(values.len(), c)?;
                for (rank, value) in values { aliases.push((*rank, retained_array(value, c)?)); }
                Some(aliases)
            }
        };
        let bank =
            safemlx::OperationEvent::prepare_resident_graph(self.recipe.completion.graph, observer)
                .map_err(|cause| fail(cause.into(), c))?;
        let ops = logical_collective::Native(stream);
        let output = match (&self.inputs, routed_values) {
            (ArithmeticInputs::Pair(inputs), None) => arithmetic(&ops, inputs, self.stage, value),
            (ArithmeticInputs::Routed(_), Some(inputs)) => routed::result(&ops, inputs, value),
            _ => return Err(fail(LogicalCause::Identity, c)),
        }.map_err(|cause| fail(cause.into(), c))?;
        drop(bank);
        let outputs = [output];
        let completion = ready.submit_original(&source, observer, stream, &outputs)?;
        let [output] = wait(&source, outputs, completion, value.protocol, c)?;
        safemlx::OperationEvent::validate_traversal_leaf(&output, observer)
            .map_err(|cause| fail(cause.into(), c))?;
        Ok(output)
    }
}

/// Same accepted Pair and Peer workers as ordinary logical movement. The final
/// observer also lends its completed row to the existing paid host decoder.
pub(super) fn execute_count_route_step(owner: &OriginalParallelControlOwner,
    first: Option<ParallelControlClaim>, input: Array, quote: RetainedLogicalCollective,
    stream: &Stream, c: &Custody) -> Result<(Array,
        crate::backend::runtime::distributed::completion::CompletedCommunicationWords), Error> {
    use crate::backend::runtime::distributed::completion::{CompletedCommunicationWords, PreparedCommunicationWords};
    reserve(&c.funding, &[size_of::<Pair>(), size_of::<[Array; 2]>(),
        size_of::<(Array, CompletedCommunicationWords)>(),
        size_of::<Result<(Array, CompletedCommunicationWords), Error>>()])?;
    let model = owner.owner().request.source.model().ok_or_else(|| fail(LogicalCause::Identity, c))?;
    if quote.value().routed_step().is_none() || !quote.value().matches_source(model) {
        return Err(fail(LogicalCause::Identity, c));
    }
    let selected = quote.value().exchange_pairs().and_then(|pairs| pairs.first())
        .ok_or_else(|| fail(LogicalCause::Identity, c))?;
    let capacity = AgreementCapacity { graph: selected.graph_capacity(), records: selected.record_capacity(),
        backing: selected.backing_capacity() };
    let pair = Pair { input, round: 0, quote: quote.clone(),
        claim: match first { Some(first) => first, None => claim(owner.owner(), c)? },
        owner: OriginalParallelControlOwner(owner.0.clone()), custody: c.clone() };
    let outputs = run_native_role(pair, capacity, &owner.owner().bank, &owner.owner().controls, c,
        |pair, observer| Ok(pair.run(observer))).map_err(|cause| Error::with_original_control_source(cause, false))??;
    run_arithmetic_finish(owner, quote, ArithmeticStage::Peer, outputs, stream, c, |output, observer| {
        let source = owner.owner().request.source.communication_source()?;
        let words = PreparedCommunicationWords::read_completed(&source, &output, observer)?;
        Ok((output, words))
    })
}
pub(super) fn retain_count_input(input: &Array, c: &Custody) -> Result<Array, Error> {
    retained_array(input, c)
}
