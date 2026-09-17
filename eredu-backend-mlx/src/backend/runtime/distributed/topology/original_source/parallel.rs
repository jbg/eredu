//! Explicit one-use model collectives, lent through the existing Group context.
use super::*;
mod boundary;
mod borrowed;
mod logical;
mod expert_region;
pub(crate) use expert_region::{RetainedExpertRegion,RetainedExpertTransfer,RetainedExpertCount,RetainedExpertVote};
pub(crate) use logical::RetainedLogicalCollective;
pub(crate) use boundary::{OriginalBoundaryCall,RetainedPipelineBoundary};
use eredu_nn::workspace::{WorkspaceCollective, WorkspaceOperation, WorkspaceOperationKind};
use safemlx::{
    Array, OriginalScopeObserver, Stream, StreamCopyPlan, distributed::OwnedGroupCpuLayoutStorage,
};
use std::{
    alloc::Layout,
    cell::{Cell, OnceCell, RefCell},
    rc::{Rc, Weak},
};

#[derive(Clone, Copy, Debug, thiserror::Error)]
pub(crate) enum ParallelInvocationCause {
    #[error("parallel invocation has no enclosing original scope")]
    Unbound,
    #[error("parallel invocation construction is closed")]
    Closed,
    #[error("parallel invocation has no remaining quoted collective occurrence")]
    Exhausted,
    #[error("parallel invocation was entered recursively")]
    Recursive,
    #[error("parallel invocation did not consume every quoted collective occurrence")]
    Incomplete,
}
enum InvocationOperation { Broadcast{group:CollectiveGroupId,root:usize}, Sum, Gather{axis:usize,widths:Vec<usize>} }
#[derive(Clone,Copy)]
enum InvocationOperationView<'a> { Sum, Gather{axis:usize,widths:&'a [usize]} }
impl InvocationOperation {
    fn matches(&self,actual:InvocationOperationView<'_>)->bool { match(self,actual) {
        (Self::Sum,InvocationOperationView::Sum)=>true,
        (Self::Gather{axis,widths},InvocationOperationView::Gather{axis:a,widths:w})=>*axis==a&&widths==w,
        _=>false,
    }}
}
struct Occurrence {
    ordinal: usize,
    operation: InvocationOperation,
    native: Option<OwnedGroupCpuLayoutStorage>,
    logical: Option<RetainedLogicalCollective>,
    backing: ParallelBacking,
}
struct BoundaryOccurrence {
    ordinal:usize,
    quote:RetainedPipelineBoundary,
}
struct ExpertOccurrence { ordinal: usize, quote: RetainedExpertRegion }
struct Bound {
    observer: OriginalScopeObserver,
    stream: StreamCopyPlan<()>,
}
struct State {
    group: Group,
    quote_source: OriginalParallelSource,
    occurrences: Vec<Occurrence>,
    boundaries: Vec<BoundaryOccurrence>,
    next_boundary: Cell<usize>,
    expert_regions: Vec<ExpertOccurrence>,
    next_expert: Cell<usize>,
    expert_active: Cell<Option<usize>>,
    expert_local_taken: Cell<bool>,
    expert_local_complete: Cell<bool>,
    expert_transport_next: Cell<usize>,
    expert_transport_pending: Cell<bool>,
    expert_count_taken: Cell<bool>,
    expert_count_complete: Cell<bool>,
    expert_vote_next:Cell<usize>,
    expert_vote_pending:Cell<bool>,
    bound: OnceCell<Bound>,
    next: Cell<usize>,
    calling: Cell<bool>,
    closed: Cell<bool>,
    forward_done:Cell<bool>,
    publication_started:Cell<bool>,
    model_roots:RefCell<Option<crate::backend::submission_recovery::prefill::TransientRootsProjection>>,
    publication_roots:RefCell<Option<crate::backend::submission_recovery::prefill::TransientRootsProjection>>,
    authority: PartitionCommunicationAuthority,
    source: RetainedCommunicationSource,
    funding: WorkspaceMetadataFunding,
}
/// An immutable context clone borrows one explicit invocation. It neither
/// selects a latest source nor creates an observer from thread-local state.
#[derive(Clone)]
pub(crate) struct OriginalParallelBinding {
    state: Weak<State>,
    source: RetainedCommunicationSource,
    fallback: eredu_nn::Error,
    // Every weak allocation alias keeps its original host account alive.
    funding: WorkspaceMetadataFunding,
}
/// The enclosing model Recovery owns this value through completion or
/// quarantine. Closing construction never retires its observer/source owners.
pub(crate) struct OriginalParallelInvocation {
    context: Group,
    state: Option<Rc<State>>,
    funding: WorkspaceMetadataFunding,
}
impl std::fmt::Debug for OriginalParallelInvocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OriginalParallelInvocation")
            .field("occurrences", &self.state().occurrences.len())
            .finish_non_exhaustive()
    }
}
impl Drop for OriginalParallelInvocation {
    fn drop(&mut self) {
        // Deallocate the unique Rc control before dropping its State/H. Weak
        // context aliases retain their own H until that control finally retires.
        if let Some(state) = self.state.take() {
            drop(Rc::into_inner(state));
        }
    }
}
fn overflow() -> Error {
    Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow)
}
fn reserve(funding: &WorkspaceMetadataFunding, parts: &[usize]) -> Result<(), Error> {
    funding
        .reserve_metadata(
            parts
                .iter()
                .copied()
                .try_fold(size_of_val(parts), usize::checked_add)
                .ok_or_else(overflow)?,
        )
        .map_err(Error::WorkspacePlanning)
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct NeuralFailure {
    #[source]
    cause: Error,
    _source: RetainedCommunicationSource,
    funding: WorkspaceMetadataFunding,
}
fn retain_neural(
    cause: Error,
    source: &RetainedCommunicationSource,
    funding: &WorkspaceMetadataFunding,
) -> eredu_nn::Error {
    eredu_nn::Error::backend_retained_source(NeuralFailure {
        cause,
        _source: source.clone(),
        funding: funding.clone(),
    })
}
fn neural_failure(
    cause: Cause,
    source: &RetainedCommunicationSource,
    funding: &WorkspaceMetadataFunding,
) -> eredu_nn::Error {
    retain_neural(failure(cause, source, funding), source, funding)
}
fn neural_failure_controls() -> Option<usize> {
    failure_control_bytes()?
        .checked_add(eredu_nn::Error::retained_source_control_bytes::<
            NeuralFailure,
        >()?)?
        .checked_add(size_of::<NeuralFailure>())?
        .checked_add(size_of::<eredu_nn::Error>())?
        .checked_add(size_of::<Result<Array, eredu_nn::Error>>())
}
impl OriginalCommunicationSource<'_> {
    pub(crate) fn prepare_parallel_invocation(
        &self,
        id: CollectiveGroupId,
        operations: &[WorkspaceOperation],
        runtime: &safemlx::PreparedInputRuntime,
        transport: &Stream,
    ) -> Result<OriginalParallelInvocation, Error> {
        self.prepare_parallel_source(id, runtime, transport)?
            .prepare_invocation(operations)
    }
}

impl OriginalParallelInvocation {
    /// Alias the same one-shot invocation for its actual enclosing Recovery.
    /// Counters, observer binding and source identity remain shared; this cannot
    /// create a second attempt or refund a consumed occurrence.
    pub(crate) fn try_clone_for_retention(&self)->Result<Self,Error> {
        reserve(&self.funding,&[
            size_of::<Self>(),size_of::<Result<Self,Error>>(),size_of::<Option<Rc<State>>>(),
            self.context.retention_copy_bytes().ok_or_else(overflow)?,
            failure_control_bytes().ok_or_else(overflow)?,
        ])?;
        let context=self.context.try_copy_for_retention()
            .map_err(|_|failure(Cause::Resource,&self.state().source,&self.funding))?;
        Ok(Self{context,state:self.state.clone(),funding:self.funding.clone()})
    }
    pub(crate) fn prepare_lending<F,T>(&self)->Result<(),Error> {
        reserve(&self.funding,&[size_of::<F>(),size_of::<T>(),size_of::<Result<T,Error>>(),
            size_of::<Option<&Self>>(),size_of::<Option<(&Self,&OriginalScopeObserver)>>(),
            OriginalScopeObserver::control_bytes().ok_or_else(overflow)?,
            failure_control_bytes().ok_or_else(overflow)?,
        ])
    }
    fn state(&self) -> &State {
        self.state.as_deref().expect("live invocation owner")
    }
    /// Immutable context supplied to the existing architecture executor. Calls
    /// refuse until bind installs the actual admitted enclosing role.
    pub(crate) fn context(&self) -> &Group {
        &self.context
    }
    /// Exact trace ordinal/source pairs for the common model recipe reducer.
    pub(crate) fn occurrences(
        &self,
    ) -> impl Iterator<Item = (usize, &OwnedGroupCpuLayoutStorage)> {
        self.state()
            .occurrences
            .iter()
            .filter_map(|row| row.native.as_ref().map(|native|(row.ordinal,native)))
    }
    /// Lend this context only after the enclosing original role has been
    /// admitted. No native stream is created or selected by this binding.
    pub(crate) fn bind(
        &self,
        observer: &OriginalScopeObserver,
        stream: &Stream,
    ) -> Result<&Group, Error> {
        let state = self.state();
        self.funding.reserve_metadata(Self::bind_control_bytes().ok_or_else(overflow)?)
            .map_err(Error::WorkspacePlanning)?;
        if state.bound.get().is_some()
            || state.closed.get()
            || state.authority.ensure_active().is_err()
            || state.group.native_group().terminal_submission()
            || !crate::backend::runtime::distributed::completion::group_source_available(
                &state.group,
            )
        {
            return Err(failure(Cause::Unavailable, &state.source, &state.funding));
        }
        let stream = StreamCopyPlan::capture(stream)
            .map_err(|_| failure(Cause::Identity, &state.source, &state.funding))?;
        state
            .bound
            .set(Bound {
                observer: observer.clone(),
                stream,
            })
            .map_err(|_| failure(Cause::Identity, &state.source, &state.funding))?;
        Ok(&self.context)
    }
    pub(crate) fn source(&self) -> &OriginalParallelSource {
        &self.state().quote_source
    }
    pub(crate) fn occurrence(
        &self,
        ordinal: usize,
    ) -> Option<(&OwnedGroupCpuLayoutStorage, ParallelBacking)> {
        self.state()
            .occurrences
            .iter()
            .find(|row| row.ordinal == ordinal)
            .and_then(|row|row.native.as_ref().map(|native|(native,row.backing)))
    }
    pub(crate) fn logical_occurrence(&self,ordinal:usize)->Option<&crate::backend::nn::workspace::LogicalCollectiveQuote>{
        self.state().occurrences.iter().find(|row|row.ordinal==ordinal)?.logical.as_ref().map(RetainedLogicalCollective::value)
    }
    pub(crate) fn boundary_occurrence(&self,ordinal:usize)
        ->Option<&crate::backend::nn::workspace::PipelineBoundaryQuote>{
        self.state().boundaries.iter().find(|row|row.ordinal==ordinal).map(|row|row.quote.value())
    }
    /// Invocation input settles before child entry. Five-source publications
    /// Initialized integer sources in this exact retained expert itinerary.
    /// Local routes, metadata columns, and movement indices are all born through
    /// the same source publisher; the optional empty Slice is a separate graph.
    pub(crate) fn expert_input_source_facts(&self)
        ->Result<(u64,usize),Error> {
        let source=self.source();
        reserve(source.funding(),&[size_of::<(&Self,u64,usize)>(),size_of::<[usize;3]>(),
            size_of::<Result<(u64,usize),Error>>()])?;
        let mut bytes=0u64;let mut attempts=0usize;
        for row in &self.state().expert_regions {
            let (maximum_rows,count)=if let Some(local)=row.quote.local() {
                let view=local.declaration.as_view();
                let maximum_rows=local.maximum_rows.max(view.selected_rows().ok_or_else(overflow)?)
                    .max(view.source_rows);
                let count=local.aggregate.as_ref().ok_or_else(||failure(Cause::Identity,
                    &self.state().source,&self.funding))?.empty_slices;
                (maximum_rows,count)
            }else if let Some(inactive)=row.quote.inactive(){
                (inactive.declaration.selected_rows.checked_mul(inactive.declaration.peers)
                    .ok_or_else(overflow)?,inactive.aggregate.empty_slices)
            }else{continue};
            let facts=source.expert_input_source_facts(maximum_rows,count)?;
            bytes=bytes.checked_add(facts.capacity_bytes()).ok_or_else(overflow)?;
            attempts=attempts.checked_add(facts.maximum_attempts()).ok_or_else(overflow)?;
        }
        Ok((bytes,attempts))
    }
    /// stay in Work; their batch frontiers/controls stay in the numerical child.
    pub(crate) fn expert_capture_population(&self)->Option<crate::backend::array_copy::CaptureNativePopulation>{
        self.state().expert_regions.iter().filter_map(|row|row.quote.local())
            .try_fold(Default::default(),|sum:crate::backend::array_copy::CaptureNativePopulation,quote|
                sum.checked_add(quote.parent_capture_population()))
    }
    pub(crate) fn expert_occurrence(&self,ordinal:usize)->Option<&crate::backend::nn::workspace::ExpertLocalQuote>{
        self.state().expert_regions.iter().find(|row|row.ordinal==ordinal).and_then(|row|row.quote.local())
    }
    pub(crate) fn expert_aggregate(&self,ordinal:usize)->Option<&crate::backend::nn::workspace::ExpertRegionAggregate>{
        self.state().expert_regions.iter().find(|row|row.ordinal==ordinal).and_then(|row|row.quote.aggregate())
    }
    pub(crate) fn expert_provider_wave_occurrence(&self,ordinal:usize)
        ->Option<&crate::backend::nn::workspace::ExpertProviderWaveQuote>{
        self.state().expert_regions.iter().find(|row|row.ordinal==ordinal).and_then(|row|row.quote.wave())
    }
    pub(crate) fn boundary_binding(&self)->Result<OriginalParallelBinding,Error>{
        reserve(&self.funding,&[size_of::<OriginalParallelBinding>(),size_of::<Result<OriginalParallelBinding,Error>>(),
            failure_control_bytes().ok_or_else(overflow)?])?;
        self.context.original_parallel_binding().cloned()
            .ok_or_else(||failure(Cause::Identity,&self.state().source,&self.funding))
    }

    /// Stop new construction after successful consumption. This makes no
    /// completion claim and retains the observer/source for enclosing Recovery.
    pub(crate) fn finish_construction(&self) -> Result<(), Error> {
        let state = self.state();
        self.funding.reserve_metadata(Self::finish_control_bytes().ok_or_else(overflow)?)
            .map_err(Error::WorkspacePlanning)?;
        if state.bound.get().is_none()
            || state.forward_done.get()
            || state.calling.get()
            || state.expert_active.get().is_some()
            || state.next_expert.get()!=state.expert_regions.len()
            || state.next_boundary.get()!=state.boundaries.len()
            || state.next.get() != state.occurrences.iter().take_while(|row| !matches!(row.operation,InvocationOperation::Broadcast{..})).count()
        {
            return Err(failure(Cause::Invocation(ParallelInvocationCause::Incomplete),&state.source,&state.funding));
        }
        state.forward_done.set(true);
        if state.next.get()==state.occurrences.len(){state.closed.set(true);}
        Ok(())
    }
}
struct ControlSourceLoan(Option<Rc<State>>);
impl Drop for ControlSourceLoan {
    fn drop(&mut self){if let Some(state)=self.0.take(){drop(Rc::into_inner(state));}}
}
impl OriginalParallelBinding {
    pub(crate) fn validate_control_source(&self,context:&Group,source:&OriginalParallelSource)->Result<(),Error>{
        reserve(&self.funding,&[size_of::<ControlSourceLoan>(),size_of::<Option<Rc<State>>>(),
            size_of::<(&Self,&Group,&OriginalParallelSource)>(),size_of::<Result<(),Error>>(),
            failure_control_bytes().ok_or_else(overflow)?])?;
        let state=self.state.upgrade().ok_or_else(||failure(Cause::Invocation(ParallelInvocationCause::Closed),&self.source,&self.funding))?;
        let loan=ControlSourceLoan(Some(state));
        let state=loan.0.as_deref().expect("live model control source loan");
        if state.closed.get() || state.bound.get().is_none() || state.calling.get()
            || !state.quote_source.same_source(source)
            || !self.funding.same_account(source.funding())
            || !context.native_group().shares_native_handle(state.group.native_group())
            || !context.retained_source().is_some_and(|actual|actual.same_source(&state.source))
            || state.authority.ensure_active().is_err() {
            return Err(failure(Cause::Identity,&self.source,&self.funding));
        }
        Ok(())
    }
    pub(crate) fn funding(&self)->&WorkspaceMetadataFunding {&self.funding}
    pub(crate) fn error(&self,cause:Cause)->eredu_nn::Error {
        match neural_failure_controls().and_then(|n|self.funding.reserve_metadata(n).ok()) {
            Some(())=>neural_failure(cause,&self.source,&self.funding),
            None=>self.fallback.clone(),
        }
    }
    pub(crate) fn sum(&self,context:&Group,input:&Array,stream:&Stream)->Result<Array,eredu_nn::Error> {
        self.execute(context,input,stream,InvocationOperationView::Sum)
    }
    pub(crate) fn gather(&self,context:&Group,input:&Array,stream:&Stream,axis:usize,widths:&[usize])->Result<Array,eredu_nn::Error> {
        self.execute(context,input,stream,InvocationOperationView::Gather{axis,widths})
    }
    fn execute(
        &self,
        context: &Group,
        input: &Array,
        stream: &Stream,
        operation: InvocationOperationView<'_>,
    ) -> Result<Array, eredu_nn::Error> {
        let Some(controls) = neural_failure_controls()
            .and_then(|n| n.checked_add(size_of::<Option<Rc<State>>>()))
            .and_then(|n| n.checked_add(size_of::<(&Self, &Group, &Array, &Stream)>()))
        else {
            return Err(self.fallback.clone());
        };
        if self.funding.reserve_metadata(controls).is_err() {
            return Err(self.fallback.clone());
        }
        let Some(state) = self.state.upgrade() else {
            return Err(neural_failure(
                Cause::Invocation(ParallelInvocationCause::Closed),
                &self.source,
                &self.funding,
            ));
        };
        let loan=ControlSourceLoan(Some(state));
        loan.0.as_deref().expect("live parallel invocation")
            .execute(context, input, stream, operation)
            .map_err(|cause| retain_neural(cause, &self.source, &self.funding))
    }
}
struct Calling<'a>(&'a Cell<bool>);
impl Drop for Calling<'_> {
    fn drop(&mut self) {
        self.0.set(false);
    }
}
impl State {
    fn execute(&self, context: &Group, input: &Array, stream: &Stream,operation:InvocationOperationView<'_>) -> Result<Array, Error> {
        reserve(
            &self.funding,
            &[
                size_of::<Calling<'_>>(),
                size_of::<(&Self, &Group, &Array, &Stream)>(),
                size_of::<Result<Array, Error>>(),
                size_of::<Result<(), eredu_runtime::PartitionExecutionError>>(),
                size_of::<Option<&Occurrence>>(),
                size_of::<InvocationOperationView<'_>>(),size_of::<InvocationOperation>(),
                size_of::<usize>(),
                size_of::<bool>(),
                failure_control_bytes().ok_or_else(overflow)?,
                crate::backend::runtime::distributed::completion::group_source_controls()
                    .ok_or_else(overflow)?,
            ],
        )?;
        let fail = |cause| failure(cause, &self.source, &self.funding);
        let bound = self
            .bound
            .get()
            .ok_or_else(|| fail(Cause::Invocation(ParallelInvocationCause::Unbound)))?;
        if self.closed.get() || self.forward_done.get() {
            return Err(fail(Cause::Invocation(ParallelInvocationCause::Closed)));
        }
        if self.calling.replace(true) {
            return Err(fail(Cause::Invocation(ParallelInvocationCause::Recursive)));
        }
        let _calling = Calling(&self.calling);
        reserve(
            &self.funding,
            &[bound
                .stream
                .source_comparison_control_bytes()
                .ok_or_else(overflow)?],
        )?;
        if context.is_logical()!=self.group.is_logical()
            || !context
                .native_group()
                .shares_native_handle(self.group.native_group())
            || !context
                .retained_source()
                .is_some_and(|source| source.same_source(&self.source))
            || !bound.stream.matches_source(stream)
            || self.authority.ensure_active().is_err()
            || self.group.native_group().terminal_submission()
            || !crate::backend::runtime::distributed::completion::group_source_available(
                &self.group,
            )
        {
            return Err(fail(Cause::Identity));
        }
        let index = self.next.get();
        let row = self
            .occurrences
            .get(index)
            .ok_or_else(|| fail(Cause::Invocation(ParallelInvocationCause::Exhausted)))?;
        if self.expert_active.get().is_some()
            || self.expert_regions.get(self.next_expert.get()).is_some_and(|region| region.ordinal < row.ordinal)
            || self.boundaries.get(self.next_boundary.get()).is_some_and(|boundary|boundary.ordinal<row.ordinal)
            || !row.operation.matches(operation){return Err(fail(Cause::Identity));}
        if let Some(quote)=&row.logical {
            // Spend the exact invocation occurrence before input completion or
            // any child role. Its source alias contains no State/Q backedge.
            self.next.set(index.checked_add(1).ok_or_else(overflow)?);
            return self.execute_logical(context,input,stream,quote);
        }
        if context.is_logical(){return Err(fail(Cause::Identity));}
        let native=row.native.as_ref().ok_or_else(||fail(Cause::Identity))?;
        reserve(
            &self.funding,
            &[
                native.binding_control_bytes().ok_or_else(overflow)?,
                size_of::<
                    Result<
                        safemlx::distributed::GroupCpuOperationStorage<'_>,
                        safemlx::distributed::GroupStorageUnavailable,
                    >,
                >(),
            ],
        )?;
        let actual = native
            .bind_actual(input)
            .map_err(|_| fail(Cause::Resource))?;
        reserve(
            &self.funding,
            &[actual.control_bytes().ok_or_else(overflow)?],
        )?;
        // Spend before entering the native constructor. A failed attempt is
        // never reused and does not retire the role observer or source.
        self.next.set(index + 1);
        let output = actual
            .construct_original(&bound.observer, self.quote_source.transport())
            .map_err(|cause| fail(Cause::Native(cause)))?;
        let shape_matches=match operation {
            InvocationOperationView::Sum=>output.shape()==input.shape(),
            InvocationOperationView::Gather{widths,..}=>output.shape().len()==input.shape().len()
                && output.shape().first().copied()==input.shape().first().copied()
                    .and_then(|n|i32::try_from(widths.len()).ok().and_then(|peers|n.checked_mul(peers)))
                && output.shape().iter().skip(1).eq(input.shape().iter().skip(1)),
        };
        if !shape_matches || output.dtype() != input.dtype() {return Err(fail(Cause::Output));}
        Ok(output)
    }
}

mod session;
mod source;
pub(crate) use source::{OriginalParallelSource, ParallelBacking};

mod publication;
