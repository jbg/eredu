//! Exact connected-member status program with paid original leaf completions.
use super::*;
use crate::backend::nn::workspace::{
    selected_parallel_numerical, ExistingArrayProjection, MlxMetalWorkspaceMechanisms,
    ResidentExecutionMechanisms, SpeculativeNumericalRecipe,
};
use crate::backend::runtime::distributed::completion::{
    prepared::{CompletionResourceLayout, PreparedCompletionResources},
    OriginalCommunicationCompletion,
};
use crate::backend::runtime::distributed::group::{StatusChainOperations, StatusPlan,LogicalPackedWorldPlan};
use crate::backend::nn::logical_collective::{self,packed};
use super::super::packed_world::OwnedPackedWorldSource;
use eredu_core::{BoundedCompletion, BoundedCompletionOutcome, BoundedCompletionWait};
use eredu_nn::{
    workspace::{WorkspaceContext, WorkspaceTraceReport,WorkspaceTensor},
    Tensor,
};
use safemlx::{Array, OperationEvent, OriginalBufferBudget};

#[derive(Clone,Copy)]
enum ArithmeticKind { Alias,Add,Pack{slot:usize,world:usize},Extract{slot:usize,world:usize} }
struct ArithmeticQuote {
    recipe: SpeculativeNumericalRecipe,
    capacity: AgreementCapacity,
    _report: WorkspaceTraceReport,
    _context: WorkspaceContext,
}
impl ArithmeticQuote {
    #[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
    fn prepare(
        inputs: &OriginalAgreementInputs,
        source: &OriginalCommunicationSource<'_>,
        stream: &Stream,
        kind: ArithmeticKind,
    ) -> Result<Self, Error> {
        reserve(
            source.funding(),
            &[
                size_of::<(
                    Self,
                    Result<Self, Error>,
                    WorkspaceContext,
                    WorkspaceTraceReport,
                    SpeculativeNumericalRecipe,
                )>(),
                size_of::<(ResidentExecutionMechanisms, MlxMetalWorkspaceMechanisms)>(),
                size_of::<OriginalBufferBudget>(),
                size_of::<ArithmeticKind>(),
                size_of::<(&OriginalAgreementInputs,&OriginalCommunicationSource<'_>,&Stream,ArithmeticKind)>(),
                size_of::<std::time::Instant>(),
                size_of::<std::time::Duration>(),
                failure_control_bytes().ok_or_else(overflow)?,
            ],
        )?;
        let add=matches!(kind,ArithmeticKind::Add);
        let ordinary = MlxMetalWorkspaceMechanisms::current_host().map_err(|cause| {
            failure(
                Cause::StatusPlanning(cause.into()),
                source.source(),
                source.funding(),
            )
        })?;
        let mechanism =
            ResidentExecutionMechanisms::from_stream(ordinary, stream, source.funding()).map_err(
                |cause| {
                    failure(
                        Cause::StatusPlanning(cause.into()),
                        source.source(),
                        source.funding(),
                    )
                },
            )?;
        if !matches!(mechanism, ResidentExecutionMechanisms::Cpu { .. }) {
            return Err(failure(Cause::Resource, source.source(), source.funding()));
        }
        let context =
            WorkspaceContext::new_with_metadata_funding(mechanism, source.funding().clone())
                .map_err(|cause| {
                    failure(
                        Cause::StatusPlanning(cause.into()),
                        source.source(),
                        source.funding(),
                    )
                })?;
        context
            .charge_metadata(size_of::<(
                Self,
                Result<Self, Error>,
                WorkspaceTraceReport,
                SpeculativeNumericalRecipe,
            )>())
            .map_err(|cause| {
                failure(
                    Cause::StatusPlanning(cause.into()),
                    source.source(),
                    source.funding(),
                )
            })?;
        let mut projection =
            ExistingArrayProjection::with_source_count(&context, if add { 2 } else { 1 }).map_err(
                |cause| {
                    failure(
                        Cause::StatusPlanning(context.metadata_source(cause)),
                        source.source(),
                        source.funding(),
                    )
                },
            )?;
        let left = projection.project(inputs.value(false)).map_err(|cause| {
            failure(
                Cause::StatusPlanning(cause.into()),
                source.source(),
                source.funding(),
            )
        })?;
        let right = if add {
            Some(projection.project(inputs.value(true)).map_err(|cause| {
                failure(
                    Cause::StatusPlanning(cause.into()),
                    source.source(),
                    source.funding(),
                )
            })?)
        } else {
            None
        };
        if !projection.is_complete() {
            return Err(failure(Cause::Resource, source.source(), source.funding()));
        }
        drop(projection);
        // Extraction borrows the exact layout produced by the same packing
        // worker, then starts its own numerical span at that completed source.
        let left=if let ArithmeticKind::Extract{slot,world}=kind {
            let packed=packed::pack(&logical_collective::Workspace(&context),&left,slot,world)
                .map_err(|cause|failure(Cause::StatusPlanning(cause),source.source(),source.funding()))?;
            WorkspaceTensor::existing(packed.layout().clone(),&context)
                .map_err(|cause|failure(Cause::StatusPlanning(cause),source.source(),source.funding()))?
        } else {left};
        context.begin_span();
        let output=match kind {
            ArithmeticKind::Alias=>Ok(left),
            ArithmeticKind::Add=>left.add(right.as_ref().ok_or_else(||failure(Cause::Resource,source.source(),source.funding()))?,&context),
            ArithmeticKind::Pack{slot,world}=>packed::pack(&logical_collective::Workspace(&context),&left,slot,world),
            ArithmeticKind::Extract{slot,..}=>packed::sum_result(&logical_collective::Workspace(&context),&left,slot),
        }.map_err(|cause|failure(Cause::StatusPlanning(cause),source.source(),source.funding()))?;
        let report = context.finish_report(&[output]).map_err(|cause| {
            failure(
                Cause::StatusPlanning(cause.into()),
                source.source(),
                source.funding(),
            )
        })?;
        let recipe =
            selected_parallel_numerical(&report, 1, mechanism, &context).map_err(|cause| {
                failure(
                    Cause::StatusPlanning(cause.into()),
                    source.source(),
                    source.funding(),
                )
            })?;
        let backing = OriginalBufferBudget::population_layout(
            inputs.runtime(),
            usize::try_from(recipe.storage.mutable_bytes()).map_err(|_| overflow())?,
            recipe.storage.maximum_births(),
        )
        .map_err(|cause| failure(Cause::Buffer(cause), source.source(), source.funding()))?;
        reserve(
            source.funding(),
            &[usize::try_from(recipe.controls).map_err(|_| overflow())?],
        )?;
        let capacity = AgreementCapacity {
            graph: recipe.graph_capacity,
            records: recipe.record_capacity,
            backing: backing.capacity(),
        };
        Ok(Self {
            recipe,
            capacity,
            _report: report,
            _context: context,
        })
    }
    #[cfg(not(all(target_vendor = "apple", feature = "metal", not(feature = "cuda"))))]
    fn prepare(
        _inputs: &OriginalAgreementInputs,
        source: &OriginalCommunicationSource<'_>,
        _stream: &Stream,
        _kind: ArithmeticKind,
    ) -> Result<Self, Error> {
        Err(failure(Cause::Resource, source.source(), source.funding()))
    }
}
pub(super) struct PreparedStatusAgreement<'a> {
    input: &'a Array,
    runtime: &'a PreparedInputRuntime,
    order: usize,
    plan: StatusPlan<'a>,
    final_quote: ArithmeticQuote,
    add: Option<ArithmeticQuote>,
    packed:Option<PackedQuote>,
    capacity: AgreementCapacity,
    leaf: AgreementCapacity,
    source: RetainedCommunicationSource,
    funding: HostMetadataFunding,
}
fn sum(a: AgreementCapacity, b: AgreementCapacity) -> Result<AgreementCapacity, Error> {
    Ok(AgreementCapacity {
        graph: a.graph.checked_add(b.graph).ok_or_else(overflow)?,
        records: a.records.checked_add(b.records).ok_or_else(overflow)?,
        backing: a.backing.checked_add(b.backing).ok_or_else(overflow)?,
    })
}
impl<'a> PreparedStatusAgreement<'a> {
    pub(super) fn prepare(
        inputs: &'a OriginalAgreementInputs,
        source: &'a OriginalCommunicationSource<'_>,
        order: usize,
        success: bool,
    ) -> Result<Option<Self>, Error> {
        let group = source
            .group(order)
            .ok_or_else(|| failure(Cause::Resource, source.source(), source.funding()))?
            .0;
        let Some(plan) = group.selected_status_plan()
            .map_err(|_|failure(Cause::LogicalWorldTransport,source.source(),source.funding()))? else {
            return Ok(None);
        };
        reserve(
            source.funding(),
            &[
                size_of::<Self>(),
                size_of::<Option<Self>>(),
                size_of::<Result<Option<Self>, Error>>(),
                size_of::<[Option<usize>; 2]>(),
                size_of::<[AgreementCapacity; 3]>(),size_of::<std::ops::Range<usize>>(),
                size_of::<[GroupWorkerOperation;2]>(),size_of::<usize>(),size_of::<[i32;2]>(),
                size_of_val(&plan.operations()),size_of::<Option<(bool,usize)>>(),
                failure_control_bytes().ok_or_else(overflow)?,
            ],
        )?;
        let stream = group
            .retained_transport_stream()
            .ok_or_else(|| failure(Cause::Resource, source.source(), source.funding()))?;
        reserve(
            source.funding(),
            &[
                plan.control_bytes::<Worker<'_, '_, '_, '_>>()
                    .ok_or_else(overflow)?,
                // Every planned leaf borrows a completed predecessor. Price
                // its exact scoped validation before issuing any native work.
                OperationEvent::traversal_leaf_control_bytes().and_then(|bytes|
                    bytes.checked_mul(plan.sends().checked_add(plan.receives())?
                        .checked_add(plan.additions().checked_mul(2)?)?))
                    .ok_or_else(overflow)?,
                size_of::<ArithmeticQuote>(),
                size_of::<Option<ArithmeticQuote>>(),
                size_of::<safemlx::PreparedArrayClone>(),
                size_of::<safemlx::StreamCopyPlan<()>>(),
                safemlx::PreparedArrayClone::control_bytes().ok_or_else(overflow)?,
                safemlx::Stream::device_type_control_bytes().ok_or_else(overflow)?,
            ],
        )?;
        let final_quote = ArithmeticQuote::prepare(inputs, source, stream, ArithmeticKind::Alias)?;
        let add = if plan.additions()!=0 {
            Some(ArithmeticQuote::prepare(inputs, source, stream, ArithmeticKind::Add)?)
        } else {
            None
        };
        let packed=match plan {StatusPlan::Packed(plan)=>Some(PackedQuote::prepare(inputs,source,order,stream,plan)?),_=>None};
        let mut capacity = final_quote.capacity;
        if let Some(packed)=&packed{capacity=sum(capacity,packed.capacity()?)?;}
        if let Some(add) = &add {
            for _ in 0..plan.additions(){capacity = sum(capacity, add.capacity)?;}
        }
        let mut leaf = AgreementCapacity {
            graph: 0,
            records: 0,
            backing: 0,
        };
        for (sending,peer) in plan.operations() {
                let peer=i32::try_from(peer).map_err(|_|overflow())?;
                let kind=if sending{GroupWorkerOperation::Send{peer}}else{GroupWorkerOperation::Receive{peer}};
                let operation = source.status_cpu_operation_storage(
                    order,
                    plan,
                    inputs.value(success),
                    kind,
                )?;
                if operation.native().constructor().output_geometry() != (1, 1) {
                    return Err(failure(Cause::Output, source.source(), source.funding()));
                }
                let backing = operation.backing_storage(inputs.runtime())?.capacity();
                let completed = operation.with_completion()?;
                let actual = AgreementCapacity {
                    graph: completed.graph_capacity(),
                    records: completed.record_capacity(),
                    backing,
                };
                capacity = sum(capacity, actual)?;
                leaf = leaf.union(actual);
        }
        Ok(Some(Self {
            input: inputs.value(success),
            runtime: inputs.runtime(),
            order,
            plan,
            final_quote,
            add,
            packed,
            capacity,
            leaf,
            source: source.source().clone(),
            funding: source.funding().clone(),
        }))
    }
    pub(super) fn capacity(&self) -> AgreementCapacity {
        self.capacity
    }
    pub(super) fn runtime(&self) -> &PreparedInputRuntime {
        self.runtime
    }
    pub(super) fn submit(
        self,
        source: &OriginalCommunicationSource<'_>,
        observer: &OriginalScopeObserver,
        stream: &Stream,
    ) -> Result<(OriginalCommunicationBool, MlxNeuralCommunicationCompletion), Error> {
        reserve(
            &self.funding,
            &[
                size_of::<Self>(),
                size_of::<Worker<'_, '_, '_, '_>>(),
                size_of::<Array>(),
                size_of::<
                    Result<(OriginalCommunicationBool, MlxNeuralCommunicationCompletion), Error>,
                >(),
                failure_control_bytes().ok_or_else(overflow)?,
            ],
        )?;
        let group = source
            .group(self.order)
            .ok_or_else(|| failure(Cause::Resource, &self.source, &self.funding))?
            .0;
        if !self.source.same_source(source.source())
            || group.selected_status_plan().map_err(|_|failure(Cause::LogicalWorldTransport,&self.source,&self.funding))? != Some(self.plan)
            || stream
                .device_type()
                .map_err(|cause| failure(Cause::Native(cause), &self.source, &self.funding))?
                != safemlx::DeviceType::Cpu
        {
            return Err(failure(Cause::Identity, &self.source, &self.funding));
        }
        let retained = group
            .retained_transport_stream()
            .ok_or_else(|| failure(Cause::Resource, &self.source, &self.funding))?;
        let stream_source = safemlx::StreamCopyPlan::<()>::capture(retained)
            .map_err(|_| failure(Cause::Identity, &self.source, &self.funding))?;
        reserve(
            &self.funding,
            &[stream_source
                .source_comparison_control_bytes()
                .ok_or_else(overflow)?],
        )?;
        if !stream_source.matches_source(stream) {
            return Err(failure(Cause::Identity, &self.source, &self.funding));
        }
        let mut worker = Worker {
            quote: &self,
            source,
            observer,
            stream,
            started: std::time::Instant::now(),
        };
        let result = (|| {
            let output = self.plan.execute(self.input, &mut worker)?;
            let scalar = PreparedCommunicationScalar::prepare_agreement(source, self.order)?;
            let ready = worker.ready(self.final_quote.recipe.completion.traversal)?;
            let graph = OperationEvent::prepare_resident_graph(
                self.final_quote.recipe.completion.graph,
                observer,
            )
            .map_err(|cause| worker.native(cause))?;
            drop(graph);
            worker.check_deadline()?;
            let (value, completion) = scalar.submit(ready, source, observer, stream, output)?;
            let completion = worker.wait_retaining(completion)?;
            Ok((value, completion.into()))
        })();
        if result.is_err() {
            worker.fence();
        }
        result
    }
}
struct Worker<'q, 's, 'input, 'native> {
    quote: &'q PreparedStatusAgreement<'input>,
    source: &'s OriginalCommunicationSource<'native>,
    observer: &'s OriginalScopeObserver,
    stream: &'s Stream,
    started: std::time::Instant,
}
impl Worker<'_, '_, '_, '_> {
    fn native(&self, cause: safemlx::error::Exception) -> Error {
        failure(
            Cause::Native(cause),
            &self.quote.source,
            &self.quote.funding,
        )
    }
    fn fence(&self) {
        self.source.authority.fence_protocol_failure(
            CommunicationOperation::FailureAgreement,
            eredu_runtime::DistributedExecutionPhase::Execution,
            None,
        );
    }
    fn ready(
        &self,
        traversal: safemlx::OperationEvalTraversalLayout,
    ) -> Result<ReadyCompletionResources, Error> {
        let mut ready = PreparedCompletionResources::prepare_original(
            self.source,
            CompletionResourceLayout {
                arrays: 0,
                counts: &[],
                groups: 1,
                routes: 0,
                streams: 0,
            },
            traversal,
        )?;
        ready.retain_group(self.quote.order)?;
        ready.finish()
    }
    fn check_deadline(&self) -> Result<(), Error> {
        let policy = self
            .source
            .source()
            .manifest()
            .completion_policy()
            .ok_or_else(|| failure(Cause::Resource, &self.quote.source, &self.quote.funding))?;
        if self.started.elapsed() >= policy.timeout() {
            self.fence();
            return Err(failure(
                Cause::StatusDeadline,
                &self.quote.source,
                &self.quote.funding,
            ));
        }
        Ok(())
    }
    fn wait_retaining(
        &self,
        completion: OriginalCommunicationCompletion,
    ) -> Result<OriginalCommunicationCompletion, Error> {
        let policy = self
            .source
            .source()
            .manifest()
            .completion_policy()
            .ok_or_else(|| failure(Cause::Resource, &self.quote.source, &self.quote.funding))?;
        let remaining = policy
            .timeout()
            .saturating_sub(self.started.elapsed())
            .max(std::time::Duration::from_nanos(1));
        let wait =
            BoundedCompletionWait::new(remaining, policy.cancellation()).map_err(|_| overflow())?;
        match completion
            .wait_bounded_retaining(wait)
            .map_err(|cause| self.native(cause))?
        {
            eredu_core::BoundedSubmissionOutcome::Completed(completion) => Ok(completion),
            eredu_core::BoundedSubmissionOutcome::DeadlineExceeded { .. } => {
                self.fence();
                Err(failure(
                    Cause::StatusDeadline,
                    &self.quote.source,
                    &self.quote.funding,
                ))
            }
        }
    }
    fn wait(&self, completion: OriginalCommunicationCompletion) -> Result<(), Error> {
        let policy = self
            .source
            .source()
            .manifest()
            .completion_policy()
            .ok_or_else(|| failure(Cause::Resource, &self.quote.source, &self.quote.funding))?;
        let remaining = policy
            .timeout()
            .saturating_sub(self.started.elapsed())
            .max(std::time::Duration::from_nanos(1));
        let wait =
            BoundedCompletionWait::new(remaining, policy.cancellation()).map_err(|_| overflow())?;
        match completion
            .wait_bounded(wait)
            .map_err(|cause| self.native(cause))?
        {
            BoundedCompletionOutcome::Completed => Ok(()),
            BoundedCompletionOutcome::DeadlineExceeded { .. } => {
                self.fence();
                Err(failure(
                    Cause::StatusDeadline,
                    &self.quote.source,
                    &self.quote.funding,
                ))
            }
        }
    }
    fn leaf(&self, input: &Array, operation: GroupWorkerOperation) -> Result<Array, Error> {
        self.check_deadline()?;
        // Predecessor waits establish completion but may leave the completed
        // original event attached. Validate and detach under its owning scope
        // before the read-only next-operation census requires a settled leaf.
        // This does not wait, evaluate, or authorize a pending predecessor.
        OperationEvent::validate_traversal_leaf(input, self.observer)
            .map_err(|cause| self.native(cause))?;
        let value = self.source.status_cpu_operation_storage(
            self.quote.order,
            self.quote.plan,
            input,
            operation,
        )?;
        let backing = value.backing_storage(self.quote.runtime)?.capacity();
        let value = value.with_completion()?;
        if !self.quote.leaf.covers(AgreementCapacity {
            graph: value.graph_capacity(),
            records: value.record_capacity(),
            backing,
        }) {
            return Err(failure(
                Cause::Identity,
                &self.quote.source,
                &self.quote.funding,
            ));
        }
        let ready = value.prepare_resources(self.source, Some(self.quote.order))?;
        let (output, completion) = value
            .construct_accepted(self.source, self.observer, self.stream)?
            .submit(ready)?;
        self.wait(completion)?;
        let (output, source, funding) = output.into_parts();
        drop((source, funding));
        Ok(output)
    }
}
impl StatusChainOperations for Worker<'_, '_, '_, '_> {
    type Value = Array;
    type Error = Error;
    fn packed_sum(&mut self,input:&Array,plan:LogicalPackedWorldPlan<'_>)->Result<Array,Error>{
        let selected=self.quote.packed.as_ref().ok_or_else(||failure(Cause::Identity,&self.quote.source,&self.quote.funding))?;
        if selected.slot!=plan.representative() || selected.world.shape()!=[i32::try_from(plan.world_size()).map_err(|_|overflow())?,1] {
            return Err(failure(Cause::Identity,&self.quote.source,&self.quote.funding));
        }
        let packed=self.packed_arithmetic(input,&selected.pack,true,selected.slot,plan.world_size())?;
        OperationEvent::validate_traversal_leaf(&packed,self.observer).map_err(|cause|self.native(cause))?;
        self.check_deadline()?;
        let operation=selected.world.bind_actual(self.source,&packed)?;
        let ready=operation.prepare_resources(self.source,None)?;
        let stream=self.source.world().retained_transport_stream().ok_or_else(||failure(Cause::Identity,&self.quote.source,&self.quote.funding))?;
        let (output,completion)=operation.construct_accepted(self.source,self.observer,stream)?.submit(ready)?;
        self.wait(completion)?;
        let (output,source,funding)=output.into_parts();
        drop((source,funding));
        self.packed_arithmetic(&output,&selected.extract,false,selected.slot,plan.world_size())
    }
    fn alias(&mut self, input: &Array) -> Result<Array, Error> {
        reserve(
            &self.quote.funding,
            &[
                safemlx::PreparedArrayClone::control_bytes().ok_or_else(overflow)?,
                Array::inspection_clone_handle_bytes(),
                size_of::<Result<Array, Error>>(),
            ],
        )?;
        let mut slot =
            safemlx::PreparedArrayClone::try_prepare_for_inspection().map_err(|cause| {
                failure(
                    Cause::StatusClone(cause),
                    &self.quote.source,
                    &self.quote.funding,
                )
            })?;
        slot.fill_for_inspection(input).map_err(|cause| {
            failure(
                Cause::StatusClone(cause),
                &self.quote.source,
                &self.quote.funding,
            )
        })
    }
    fn receive(&mut self, input: &Array, peer: usize) -> Result<Array, Error> {
        self.leaf(input, GroupWorkerOperation::Receive { peer: i32::try_from(peer).map_err(|_|overflow())? })
    }
    fn send(&mut self, input: &Array, peer: usize) -> Result<(), Error> {
        self.leaf(input, GroupWorkerOperation::Send { peer: i32::try_from(peer).map_err(|_|overflow())? })
            .map(drop)
    }
    fn add(&mut self, left: &Array, right: &Array) -> Result<Array, Error> {
        self.check_deadline()?;
        let quote = self
            .quote
            .add
            .as_ref()
            .ok_or_else(|| failure(Cause::Identity, &self.quote.source, &self.quote.funding))?;
        for value in [left, right] {
            if value.shape() != [1] || value.dtype() != safemlx::Dtype::Int32 {
                return Err(failure(
                    Cause::Identity,
                    &self.quote.source,
                    &self.quote.funding,
                ));
            }
            OperationEvent::validate_traversal_leaf(value, self.observer)
                .map_err(|cause| self.native(cause))?;
        }
        let ready = self.ready(quote.recipe.completion.traversal)?;
        let bank =
            OperationEvent::prepare_resident_graph(quote.recipe.completion.graph, self.observer)
                .map_err(|cause| self.native(cause))?;
        let output = left
            .add(right, self.stream)
            .map_err(|cause| self.native(cause))?;
        drop(bank);
        let completion = ready.submit_original(
            self.source,
            self.observer,
            self.stream,
            std::slice::from_ref(&output),
        )?;
        self.wait(completion)?;
        Ok(output)
    }
}

/// Separate native stages retain one source and use the common packed worker.
struct PackedQuote {pack:ArithmeticQuote,extract:ArithmeticQuote,world:OwnedPackedWorldSource,slot:usize}
impl PackedQuote {
    fn prepare(inputs:&OriginalAgreementInputs,source:&OriginalCommunicationSource<'_>,order:usize,stream:&Stream,plan:LogicalPackedWorldPlan<'_>)->Result<Self,Error>{
        let slot=plan.representative();let world_size=plan.world_size();
        reserve(source.funding(),&[size_of::<Self>(),size_of::<Option<Self>>(),size_of::<Result<Self,Error>>(),
            size_of::<(&OriginalAgreementInputs,&OriginalCommunicationSource<'_>,usize,&Stream,LogicalPackedWorldPlan<'_>)>(),
            size_of::<[i32;2]>(),size_of::<[Array;3]>(),size_of::<OriginalCommunicationCompletion>(),
            size_of::<(bool,usize,usize,&Array,&ArithmeticQuote)>(),
            packed::controls::<logical_collective::Native<'_>>(1).ok_or_else(overflow)?,
            packed::controls::<logical_collective::Native<'_>>(2).ok_or_else(overflow)?,
            OperationEvent::traversal_leaf_control_bytes().and_then(|n|n.checked_mul(3)).ok_or_else(overflow)?])?;
        let pack=ArithmeticQuote::prepare(inputs,source,stream,ArithmeticKind::Pack{slot,world:world_size})?;
        let extract=ArithmeticQuote::prepare(inputs,source,stream,ArithmeticKind::Extract{slot,world:world_size})?;
        let shape=[i32::try_from(world_size).map_err(|_|overflow())?,1];
        let world=source.packed_world_completion_source(order,&shape,safemlx::Dtype::Int32,inputs.runtime())?;
        Ok(Self{pack,extract,world,slot})
    }
    fn capacity(&self)->Result<AgreementCapacity,Error>{sum(sum(self.pack.capacity,self.extract.capacity)?,AgreementCapacity{
        graph:self.world.graph_capacity(),records:self.world.record_capacity(),backing:self.world.backing_capacity()})}
}
impl Worker<'_, '_, '_, '_> {
    fn packed_arithmetic(&self,input:&Array,quote:&ArithmeticQuote,pack:bool,slot:usize,world:usize)->Result<Array,Error>{
        self.check_deadline()?;
        if input.dtype()!=safemlx::Dtype::Int32 || if pack{input.shape()!=[1]}else{input.shape()!=[i32::try_from(world).map_err(|_|overflow())?,1]}{
            return Err(failure(Cause::Identity,&self.quote.source,&self.quote.funding));
        }
        OperationEvent::validate_traversal_leaf(input,self.observer).map_err(|cause|self.native(cause))?;
        let ready=self.ready(quote.recipe.completion.traversal)?;
        let bank=OperationEvent::prepare_resident_graph(quote.recipe.completion.graph,self.observer).map_err(|cause|self.native(cause))?;
        let output=if pack{packed::pack(&logical_collective::Native(self.stream),input,slot,world)}
            else{packed::sum_result(&logical_collective::Native(self.stream),input,slot)}.map_err(|cause|self.native(cause))?;
        drop(bank);
        let completion=ready.submit_original(self.source,self.observer,self.stream,std::slice::from_ref(&output))?;
        self.wait(completion)?;
        Ok(output)
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
