//! Actual block permutation under the same source-bound child role machinery.
use super::*;
use crate::backend::nn::{logical_collective::{self, blocks, axis}, workspace::{
    ExistingArrayProjection, MlxMetalWorkspaceMechanisms, ResidentExecutionMechanisms,
    SpeculativeNumericalRecipe, selected_parallel_numerical,
}};
use crate::backend::runtime::distributed::completion::{OriginalCommunicationCompletion,
    prepared::{PreparedCompletionResources, CompletionResourceLayout}};
use eredu_nn::workspace::{WorkspaceContext, WorkspaceTraceReport, WorkspaceTensor};
use safemlx::{PreparedStreamCopy, StreamCopyPlan};

pub(super) fn run<I: Iterator<Item = usize>>(owner: &OriginalParallelControlOwner,
    claim: ParallelControlClaim, input: Array, counts: &[usize], order: I,
    group_order: usize, completed_source: Option<&ErasedSharedStorageOwner>,
    stream: &Stream, c: &Custody,
    ceiling:Option<crate::backend::nn::workspace::ExpertReorderEnvelope>) -> Result<Array, Error> {
    reserve(&c.funding, &[size_of::<I>(), size_of::<Reorder>(),
        size_of::<Result<Reorder, Error>>(), size_of::<Result<Array, Error>>(),
        size_of::<Result<Result<Array, Error>, eredu_core::BackendFailure>>()])?;
    let mut retained_counts = destination(counts.len(), c)?;
    retained_counts.extend_from_slice(counts);
    let mut retained_order = destination(counts.len(), c)?;
    for member in order {
        if member >= counts.len() || retained_order.len() == counts.len()
            || retained_order.contains(&member) { return Err(fail(VariableCause::Identity, c)); }
        retained_order.push(member);
    }
    let plan = Reorder::prepare(Inputs::Single(input), retained_counts, retained_order, Transform::Blocks, group_order,
        completed_source.cloned(), stream, claim,
        OriginalParallelControlOwner(owner.0.clone()), c.clone())?;
    if let Some(ceiling)=ceiling {
        if !ceiling.covers(plan.input.first(),plan.counts.len(),plan.capacity.graph,plan.capacity.records,
            plan.capacity.backing,&plan.recipe){return Err(fail(VariableCause::Identity,c));}
    }
    let capacity = plan.capacity;
    let kernels = plan.recipe.kernels;
    run_native_role_with_pipeline(plan, capacity, Some(safemlx::PreparedPipelineCachePlan::new(kernels)),
        &owner.owner().bank, &owner.owner().controls, c,
        |plan, observer| Ok(plan.execute(observer)))
        .map_err(|cause| Error::with_original_control_source(cause, false))?
}
#[derive(Clone, Copy)]
enum Transform { Blocks, Axis(axis::AxisPlan, bool), Concatenate }
enum Inputs<T> { Single(T), Group(Vec<T>) }
impl<T> Inputs<T> {
    fn as_slice(&self) -> &[T] { match self { Self::Single(value) => std::slice::from_ref(value), Self::Group(values) => values } }
    fn first(&self) -> &T { &self.as_slice()[0] }
}
pub(super) fn axis(owner: &OriginalParallelControlOwner, claim: ParallelControlClaim,
    input: Array, selected: usize, inverse: bool, group_order: usize,
    completed_source: Option<&ErasedSharedStorageOwner>, stream: &Stream, c: &Custody)
    -> Result<Array, Error> {
    reserve(&c.funding, &[size_of::<Reorder>(), size_of::<Result<Reorder, Error>>(),
        size_of::<Result<Array, Error>>(), size_of::<Transform>(),
        size_of::<Result<Result<Array, Error>, eredu_core::BackendFailure>>()])?;
    let axis = axis::AxisPlan::new(input.shape().len(), selected).ok_or_else(|| fail(VariableCause::Axis, c))?;
    let plan = Reorder::prepare(Inputs::Single(input), Vec::new(), Vec::new(), Transform::Axis(axis, inverse), group_order,
        completed_source.cloned(), stream, claim, OriginalParallelControlOwner(owner.0.clone()), c.clone())?;
    let capacity = plan.capacity; let kernels = plan.recipe.kernels;
    run_native_role_with_pipeline(plan, capacity, Some(safemlx::PreparedPipelineCachePlan::new(kernels)),
        &owner.owner().bank, &owner.owner().controls, c, |plan, observer| Ok(plan.execute(observer)))
        .map_err(|cause| Error::with_original_control_source(cause, false))?
}
pub(super) fn concatenate(owner: &OriginalParallelControlOwner, claim: ParallelControlClaim,
    inputs: Vec<Array>, group_order: usize, completed_source: Option<&ErasedSharedStorageOwner>,
    stream: &Stream, c: &Custody,
    ceiling:Option<crate::backend::nn::workspace::ExpertReorderEnvelope>) -> Result<Array, Error> {
    reserve(&c.funding, &[size_of::<Reorder>(), size_of::<Result<Reorder, Error>>(),
        size_of::<Result<Array, Error>>(), size_of::<Inputs<Array>>(),
        size_of::<Result<Result<Array, Error>, eredu_core::BackendFailure>>()])?;
    if inputs.is_empty() { return Err(fail(VariableCause::Identity, c)); }
    let plan = Reorder::prepare(Inputs::Group(inputs), Vec::new(), Vec::new(), Transform::Concatenate,
        group_order, completed_source.cloned(), stream, claim,
        OriginalParallelControlOwner(owner.0.clone()), c.clone())?;
    if let Some(ceiling)=ceiling {
        if !ceiling.covers_join(plan.input.as_slice(),plan.capacity.graph,plan.capacity.records,plan.capacity.backing,&plan.recipe){
            return Err(fail(VariableCause::Identity,c));
        }
    }
    let capacity = plan.capacity; let kernels = plan.recipe.kernels;
    run_native_role_with_pipeline(plan, capacity, Some(safemlx::PreparedPipelineCachePlan::new(kernels)),
        &owner.owner().bank, &owner.owner().controls, c, |plan, observer| Ok(plan.execute(observer)))
        .map_err(|cause| Error::with_original_control_source(cause, false))?
}
struct Reorder {
    input: Inputs<Array>,
    counts: Vec<usize>,
    order: Vec<usize>,
    transform: Transform,
    stream: PreparedStreamCopy<Custody>,
    _report: WorkspaceTraceReport,
    _context: WorkspaceContext,
    recipe: SpeculativeNumericalRecipe,
    capacity: AgreementCapacity,
    group_order: usize,
    claim: ParallelControlClaim,
    _completed_source: Option<ErasedSharedStorageOwner>,
    owner: OriginalParallelControlOwner,
    custody: Custody,
}
impl Reorder {
    #[inline(never)]
    fn prepare(input: Inputs<Array>, counts: Vec<usize>, order: Vec<usize>, transform: Transform, group_order: usize,
        completed_source: Option<ErasedSharedStorageOwner>, stream: &Stream,
        claim: ParallelControlClaim, owner: OriginalParallelControlOwner, custody: Custody)
        -> Result<Self, Error> {
        let c = &custody;
        reserve(&c.funding, &[size_of::<Self>(), size_of::<WorkspaceContext>(),
            size_of::<WorkspaceTraceReport>(), size_of::<SpeculativeNumericalRecipe>(),
            size_of::<ExistingArrayProjection<'_>>(), size_of::<WorkspaceTensor>(),
            size_of::<Inputs<WorkspaceTensor>>(), size_of::<Result<Inputs<WorkspaceTensor>, Error>>(),
            size_of::<StreamCopyPlan<Custody>>(), size_of::<PreparedStreamCopy<Custody>>(),
            size_of::<Option<ErasedSharedStorageOwner>>(),
            Stream::device_type_control_bytes().ok_or_else(overflow)?])?;
        #[cfg(not(all(target_vendor = "apple", feature = "metal", not(feature = "cuda"))))]
        return Err(fail(VariableCause::LogicalSource, c));
        #[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
        {
            let mechanism = MlxMetalWorkspaceMechanisms::current_host().map_err(|cause| fail(cause.into(), c))?;
            let mechanism = ResidentExecutionMechanisms::from_stream(mechanism, stream, &c.funding)
                .map_err(|cause| fail(cause.into(), c))?;
            let context = WorkspaceContext::new_with_metadata_funding(mechanism, c.funding.clone())
                .map_err(|cause| fail(VariableCause::Neural(cause.into()), c))?;
            let mut projection = ExistingArrayProjection::with_source_count(&context, input.as_slice().len())
                .map_err(|cause| fail(VariableCause::Neural(context.metadata_source(cause)), c))?;
            let projected = match &input {
                Inputs::Single(input) => Inputs::Single(projection.project(input).map_err(|cause| fail(cause.into(), c))?),
                Inputs::Group(inputs) => {
                    let mut values = context.metadata_vec(inputs.len()).map_err(|cause| fail(VariableCause::Neural(cause.into()), c))?;
                    for input in inputs { values.push(projection.project(input).map_err(|cause| fail(cause.into(), c))?); }
                    Inputs::Group(values)
                }
            };
            if !projection.is_complete() { return Err(fail(VariableCause::Identity, c)); }
            drop(projection);
            context.begin_span();
            let ops = logical_collective::Workspace(&context);
            let output = match transform {
                Transform::Blocks => blocks::concatenate(&ops, projected.first(), &counts, order.iter().copied()),
                Transform::Axis(plan, inverse) => axis::transpose(&ops, projected.first(), plan, inverse),
                Transform::Concatenate => blocks::join(&ops, projected.as_slice()),
            }.map_err(|cause| fail(cause.into(), c))?;
            let report = context.finish_report(&[output]).map_err(|cause| fail(cause.into(), c))?;
            let recipe = selected_parallel_numerical(&report, 1, mechanism, &context)
                .map_err(|cause| fail(cause.into(), c))?;
            drop(projected);
            let runtime = owner.owner().request.source.agreement_inputs()
                .ok_or_else(|| fail(VariableCause::Identity, c))?.runtime();
            let backing = OriginalBufferBudget::population_layout(runtime,
                usize::try_from(recipe.storage.mutable_bytes()).map_err(|_| overflow())?,
                recipe.storage.maximum_births()).map_err(|cause| fail(cause.into(), c))?;
            let capacity = AgreementCapacity { graph: recipe.graph_capacity, records: recipe.record_capacity,
                backing: backing.capacity() };
            reserve(&c.funding, &[usize::try_from(recipe.controls).map_err(|_| overflow())?,
                match transform {
                    Transform::Blocks => blocks::controls::<logical_collective::Native<'_>,
                        std::iter::Copied<std::slice::Iter<'_, usize>>>(input.first().shape().len(), counts.len()),
                    Transform::Axis(..) => axis::controls::<logical_collective::Native<'_>>(input.first().shape().len()),
                    Transform::Concatenate => blocks::join_controls::<logical_collective::Native<'_>>(input.as_slice().len()),
                }.ok_or_else(overflow)?,
                safemlx::OperationEvent::traversal_leaf_control_bytes().and_then(|n| n.checked_mul(input.as_slice().len().checked_add(1)?)).ok_or_else(overflow)?,
                size_of::<(Array, [Array; 1], Result<Array, safemlx::error::Exception>)>(),
            ])?;
            let copied = StreamCopyPlan::<Custody>::capture(stream).map_err(|cause| fail(cause.into(), c))?;
            reserve(&c.funding, &[copied.control_bytes().ok_or_else(overflow)?, copied.native_wrapper_bytes(),
                copied.owner_node_layout().size(), Layout::new::<[usize; 2]>().extend(copied.shared_body_layout())
                    .map_err(|_| overflow())?.0.pad_to_align().size()])?;
            let stream = copied.realize(c.clone()).map_err(|failed| {
                let (cause, retained) = failed.into_parts(); let error = fail(cause.into(), c); drop(retained); error
            })?;
            Ok(Self { input, counts, order, transform, stream, _report: report, _context: context,
                recipe, capacity, group_order, claim, _completed_source: completed_source, owner, custody })
        }
    }
    fn execute(&self, observer: &OriginalScopeObserver) -> Result<Array, Error> {
        let c = &self.custody;
        reserve(&c.funding, &[size_of::<Self>(), size_of::<[Array; 1]>(),
            size_of::<OriginalCommunicationCompletion>(), size_of::<PartitionExecutionError>(),
            size_of::<eredu_core::Submission<[Array; 1], OriginalCommunicationCompletion>>(),
            size_of::<Result<[Array; 1], PartitionExecutionError>>(),
        ])?;
        if self.claim.identity() != self.owner.owner().request.cursor.borrow().identity()
            || self.claim.event() != ParallelControlEvent::Phase(DistributedExecutionPhase::Execution) {
            return Err(fail(VariableCause::Identity, c));
        }
        let source = self.owner.owner().request.source.communication_source()?;
        let mut prepared = PreparedCompletionResources::prepare_original(&source,
            CompletionResourceLayout { arrays: 0, counts: &[], groups: 1, routes: 0, streams: 0 },
            self.recipe.completion.traversal)?;
        prepared.retain_group(self.group_order)?;
        let ready = prepared.finish()?;
        for input in self.input.as_slice() {
            safemlx::OperationEvent::validate_traversal_leaf(input, observer)
                .map_err(|cause| fail(cause.into(), c))?;
        }
        let graph = safemlx::OperationEvent::prepare_resident_graph(self.recipe.completion.graph, observer)
            .map_err(|cause| fail(cause.into(), c))?;
        let ops = logical_collective::Native(self.stream.as_stream());
        let output = match self.transform {
            Transform::Blocks => blocks::concatenate(&ops, self.input.first(), &self.counts, self.order.iter().copied()),
            Transform::Axis(plan, inverse) => axis::transpose(&ops, self.input.first(), plan, inverse),
            Transform::Concatenate => blocks::join(&ops, self.input.as_slice()),
        }.map_err(|cause| fail(cause.into(), c))?;
        drop(graph);
        let outputs = [output];
        let completion = ready.submit_original(&source, observer, self.stream.as_stream(), &outputs)?;
        let kind = CommunicationOperation::VariableAllToAll;
        let phase = DistributedExecutionPhase::Execution;
        let [output] = source.authority.wait_with_error(eredu_core::Submission { output: outputs, completion },
            kind, phase, None, |cause| PartitionExecutionError::PreparedCommunication {
                operation: kind, phase, completion: true,
                source: failure(Cause::Native(cause), &c.source, &c.funding).into_backend_failure(),
            }).map_err(|cause| fail(cause.into(), c))?;
        safemlx::OperationEvent::validate_traversal_leaf(&output, observer)
            .map_err(|cause| fail(cause.into(), c))?;
        Ok(output)
    }
}
