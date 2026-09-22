//! Exact connected-member status program with paid original leaf completions.
use super::super::packed_world::OwnedPackedWorldSource;
use super::*;
use crate::backend::nn::logical_collective::{self, packed};
use crate::backend::nn::workspace::{
    ExistingArrayProjection, MlxMetalWorkspaceMechanisms, ResidentExecutionMechanisms,
    SpeculativeNumericalRecipe, selected_parallel_numerical,
};
use crate::backend::runtime::distributed::completion::{
    OriginalCommunicationCompletion,
    prepared::{CompletionResourceLayout, PreparedCompletionResources},
};
use crate::backend::runtime::distributed::group::{
    LogicalPackedWorldPlan, StatusChainOperations, StatusPlan,
};
use eredu_core::{BoundedCompletion, BoundedCompletionOutcome, BoundedCompletionWait};
use eredu_nn::{
    Tensor,
    workspace::{WorkspaceContext, WorkspaceTensor, WorkspaceTraceReport},
};
use safemlx::{Array, OperationEvent, OriginalBufferBudget};

#[derive(Clone, Copy)]
enum ArithmeticKind {
    Alias,
    Add,
    Pack { slot: usize, world: usize },
    Extract { slot: usize, world: usize },
}
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
                size_of::<(
                    &OriginalAgreementInputs,
                    &OriginalCommunicationSource<'_>,
                    &Stream,
                    ArithmeticKind,
                )>(),
                size_of::<std::time::Instant>(),
                size_of::<std::time::Duration>(),
                failure_control_bytes().ok_or_else(overflow)?,
            ],
        )?;
        let add = matches!(kind, ArithmeticKind::Add);
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
        let left = if let ArithmeticKind::Extract { slot, world } = kind {
            let packed = packed::pack(&logical_collective::Workspace(&context), &left, slot, world)
                .map_err(|cause| {
                    failure(
                        Cause::StatusPlanning(cause),
                        source.source(),
                        source.funding(),
                    )
                })?;
            WorkspaceTensor::existing(packed.layout().clone(), &context).map_err(|cause| {
                failure(
                    Cause::StatusPlanning(cause),
                    source.source(),
                    source.funding(),
                )
            })?
        } else {
            left
        };
        context.begin_span();
        let output = match kind {
            ArithmeticKind::Alias => Ok(left),
            ArithmeticKind::Add => left.add(
                right
                    .as_ref()
                    .ok_or_else(|| failure(Cause::Resource, source.source(), source.funding()))?,
                &context,
            ),
            ArithmeticKind::Pack { slot, world } => {
                packed::pack(&logical_collective::Workspace(&context), &left, slot, world)
            }
            ArithmeticKind::Extract { slot, .. } => {
                packed::sum_result(&logical_collective::Workspace(&context), &left, slot)
            }
        }
        .map_err(|cause| {
            failure(
                Cause::StatusPlanning(cause),
                source.source(),
                source.funding(),
            )
        })?;
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
#[derive(Clone, Copy, PartialEq, Eq)]
enum StatusKind {
    Chain,
    Exchange,
    Routed,
    Packed,
}
impl StatusKind {
    fn of(plan: StatusPlan<'_>) -> Self {
        match plan {
            StatusPlan::Chain(_) => Self::Chain,
            StatusPlan::Exchange { .. } => Self::Exchange,
            StatusPlan::Routed(_) => Self::Routed,
            StatusPlan::Packed(_) => Self::Packed,
        }
    }
}
/// Paid immutable equations and the exact selected itinerary. No input Array,
/// communication owner, callback or scope is retained by these descriptors.
pub(super) struct StatusQuote {
    native: safemlx::distributed::Group,
    kind: StatusKind,
    operations: Vec<(bool, usize)>,
    leaves: Vec<super::quote::LeafQuote>,
    additions: usize,
    packed_axes: Option<(usize, usize)>,
    final_quote: ArithmeticQuote,
    add: Option<ArithmeticQuote>,
    packed: Option<PackedQuote>,
}
impl StatusQuote {
    pub(super) fn prepare(
        inputs: &OriginalAgreementInputs,
        source: &OriginalCommunicationSource<'_>,
        order: usize,
    ) -> Result<Option<Self>, Error> {
        let Some((group, descriptor, _)) = source.group(order) else {
            return Ok(None);
        };
        if descriptor.local_index().is_none()
            || !descriptor.requirements().operations().iter().any(|entry| {
                entry.operation() == CommunicationOperation::FailureAgreement
                    && entry.exact_completion()
            })
        {
            return Ok(None);
        }
        let Some(plan) = group.selected_status_plan().map_err(|_| {
            failure(
                Cause::LogicalWorldTransport,
                source.source(),
                source.funding(),
            )
        })?
        else {
            return Ok(None);
        };
        reserve(
            source.funding(),
            &[
                size_of::<Self>(),
                size_of::<Option<Self>>(),
                size_of::<Result<Option<Self>, Error>>(),
                size_of::<(
                    &OriginalAgreementInputs,
                    &OriginalCommunicationSource<'_>,
                    usize,
                )>(),
                size_of_val(&plan.operations()),
                failure_control_bytes().ok_or_else(overflow)?,
            ],
        )?;
        let stream = group
            .retained_transport_stream()
            .ok_or_else(|| failure(Cause::Resource, source.source(), source.funding()))?;
        let count = plan
            .operations()
            .try_fold(0usize, |n, _| n.checked_add(1))
            .ok_or_else(overflow)?;
        let mut operations = source
            .funding()
            .metadata_vec(count)
            .map_err(Error::Neural)?;
        operations.extend(plan.operations());
        if operations.len() != count {
            return Err(failure(Cause::Resource, source.source(), source.funding()));
        }
        let mut leaves = source
            .funding()
            .metadata_vec(operations.len())
            .map_err(Error::Neural)?;
        for &(sending, peer) in &operations {
            let peer = i32::try_from(peer).map_err(|_| overflow())?;
            let operation = if sending {
                GroupWorkerOperation::Send { peer }
            } else {
                GroupWorkerOperation::Receive { peer }
            };
            leaves.push(super::quote::LeafQuote::prepare(
                source,
                inputs.runtime(),
                order,
                operation,
                true,
            )?);
        }
        let final_quote = ArithmeticQuote::prepare(inputs, source, stream, ArithmeticKind::Alias)?;
        let add = if plan.additions() != 0 {
            Some(ArithmeticQuote::prepare(
                inputs,
                source,
                stream,
                ArithmeticKind::Add,
            )?)
        } else {
            None
        };
        let packed = match plan {
            StatusPlan::Packed(plan) => {
                Some(PackedQuote::prepare(inputs, source, order, stream, plan)?)
            }
            _ => None,
        };
        Ok(Some(Self {
            native: group.native_group().clone(),
            kind: StatusKind::of(plan),
            operations,
            leaves,
            additions: plan.additions(),
            packed_axes: match plan {
                StatusPlan::Packed(plan) => Some((plan.representative(), plan.world_size())),
                _ => None,
            },
            final_quote,
            add,
            packed,
        }))
    }
    fn matches(&self, group: &Group, plan: StatusPlan<'_>) -> bool {
        self.native.shares_native_handle(group.native_group())
            && self.kind == StatusKind::of(plan)
            && self.additions == plan.additions()
            && self.operations.iter().copied().eq(plan.operations())
            && self.packed_axes
                == match plan {
                    StatusPlan::Packed(plan) => Some((plan.representative(), plan.world_size())),
                    _ => None,
                }
    }
    fn execution_control_bytes(&self) -> Option<usize> {
        let mut bytes = usize::try_from(self.final_quote.recipe.controls).ok()?;
        if let Some(add) = &self.add {
            bytes = bytes.checked_add(
                usize::try_from(add.recipe.controls)
                    .ok()?
                    .checked_mul(self.additions)?,
            )?;
        }
        if let Some(packed) = &self.packed {
            bytes = bytes.checked_add(packed.execution_control_bytes()?)?;
        }
        Some(bytes)
    }
}
pub(super) struct PreparedStatusAgreement<'a> {
    input: &'a Array,
    runtime: &'a PreparedInputRuntime,
    order: usize,
    plan: StatusPlan<'a>,
    final_quote: &'a ArithmeticQuote,
    add: Option<&'a ArithmeticQuote>,
    packed: Option<&'a PackedQuote>,
    capacity: AgreementCapacity,
    execution_controls: usize,
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
        let Some(plan) = group.selected_status_plan().map_err(|_| {
            failure(
                Cause::LogicalWorldTransport,
                source.source(),
                source.funding(),
            )
        })?
        else {
            return Ok(None);
        };
        source
            .funding()
            .reserve_metadata(Self::prepare_frame_control_bytes(plan).ok_or_else(overflow)?)
            .map_err(Error::WorkspacePlanning)?;
        let stream = group
            .retained_transport_stream()
            .ok_or_else(|| failure(Cause::Resource, source.source(), source.funding()))?;
        source
            .funding()
            .reserve_metadata(Self::prepare_worker_control_bytes(plan).ok_or_else(overflow)?)
            .map_err(Error::WorkspacePlanning)?;
        let retained = inputs
            .status_quote(order)
            .filter(|quote| quote.matches(group, plan))
            .ok_or_else(|| failure(Cause::Identity, source.source(), source.funding()))?;
        let final_quote = &retained.final_quote;
        let add = retained.add.as_ref();
        let packed = retained.packed.as_ref();
        let mut capacity = final_quote.capacity;
        if let Some(packed) = &packed {
            capacity = sum(capacity, packed.capacity()?)?;
        }
        if let Some(add) = &add {
            for _ in 0..plan.additions() {
                capacity = sum(capacity, add.capacity)?;
            }
        }
        let mut leaf = AgreementCapacity {
            graph: 0,
            records: 0,
            backing: 0,
        };
        for (sending, peer) in plan.operations() {
            let peer = i32::try_from(peer).map_err(|_| overflow())?;
            let kind = if sending {
                GroupWorkerOperation::Send { peer }
            } else {
                GroupWorkerOperation::Receive { peer }
            };
            let operation =
                source.status_cpu_operation_storage(order, plan, inputs.value(success), kind)?;
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
            execution_controls: retained.execution_control_bytes().ok_or_else(overflow)?,
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
        reserve(&self.funding, &[self.execution_controls])?;
        self.funding
            .reserve_metadata(Self::submit_control_bytes().ok_or_else(overflow)?)
            .map_err(Error::WorkspacePlanning)?;
        let group = source
            .group(self.order)
            .ok_or_else(|| failure(Cause::Resource, &self.source, &self.funding))?
            .0;
        if !self.source.same_source(source.source())
            || group
                .selected_status_plan()
                .map_err(|_| failure(Cause::LogicalWorldTransport, &self.source, &self.funding))?
                != Some(self.plan)
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
    fn packed_sum(
        &mut self,
        input: &Array,
        plan: LogicalPackedWorldPlan<'_>,
    ) -> Result<Array, Error> {
        let selected = self
            .quote
            .packed
            .as_ref()
            .ok_or_else(|| failure(Cause::Identity, &self.quote.source, &self.quote.funding))?;
        if selected.slot != plan.representative()
            || selected.world.shape()
                != [i32::try_from(plan.world_size()).map_err(|_| overflow())?, 1]
        {
            return Err(failure(
                Cause::Identity,
                &self.quote.source,
                &self.quote.funding,
            ));
        }
        let packed = self.packed_arithmetic(
            input,
            &selected.pack,
            true,
            selected.slot,
            plan.world_size(),
        )?;
        OperationEvent::validate_traversal_leaf(&packed, self.observer)
            .map_err(|cause| self.native(cause))?;
        self.check_deadline()?;
        let operation = selected.world.bind_actual(self.source, &packed)?;
        let ready = operation.prepare_resources(self.source, None)?;
        let stream = self
            .source
            .world()
            .retained_transport_stream()
            .ok_or_else(|| failure(Cause::Identity, &self.quote.source, &self.quote.funding))?;
        let (output, completion) = operation
            .construct_accepted(self.source, self.observer, stream)?
            .submit(ready)?;
        self.wait(completion)?;
        let (output, source, funding) = output.into_parts();
        drop((source, funding));
        self.packed_arithmetic(
            &output,
            &selected.extract,
            false,
            selected.slot,
            plan.world_size(),
        )
    }
    fn alias(&mut self, input: &Array) -> Result<Array, Error> {
        self.quote
            .funding
            .reserve_metadata(Self::alias_control_bytes().ok_or_else(overflow)?)
            .map_err(Error::WorkspacePlanning)?;
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
        self.leaf(
            input,
            GroupWorkerOperation::Receive {
                peer: i32::try_from(peer).map_err(|_| overflow())?,
            },
        )
    }
    fn send(&mut self, input: &Array, peer: usize) -> Result<(), Error> {
        self.leaf(
            input,
            GroupWorkerOperation::Send {
                peer: i32::try_from(peer).map_err(|_| overflow())?,
            },
        )
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
struct PackedQuote {
    pack: ArithmeticQuote,
    extract: ArithmeticQuote,
    world: OwnedPackedWorldSource,
    slot: usize,
    world_metadata: usize,
}
impl PackedQuote {
    fn prepare(
        inputs: &OriginalAgreementInputs,
        source: &OriginalCommunicationSource<'_>,
        order: usize,
        stream: &Stream,
        plan: LogicalPackedWorldPlan<'_>,
    ) -> Result<Self, Error> {
        let slot = plan.representative();
        let world_size = plan.world_size();
        reserve(
            source.funding(),
            &[
                size_of::<Self>(),
                size_of::<Option<Self>>(),
                size_of::<Result<Self, Error>>(),
                size_of::<(
                    &OriginalAgreementInputs,
                    &OriginalCommunicationSource<'_>,
                    usize,
                    &Stream,
                    LogicalPackedWorldPlan<'_>,
                )>(),
                size_of::<[i32; 2]>(),
            ],
        )?;
        let pack = ArithmeticQuote::prepare(
            inputs,
            source,
            stream,
            ArithmeticKind::Pack {
                slot,
                world: world_size,
            },
        )?;
        let extract = ArithmeticQuote::prepare(
            inputs,
            source,
            stream,
            ArithmeticKind::Extract {
                slot,
                world: world_size,
            },
        )?;
        let shape = [i32::try_from(world_size).map_err(|_| overflow())?, 1];
        let world = source.packed_world_completion_source(
            order,
            &shape,
            safemlx::Dtype::Int32,
            inputs.runtime(),
        )?;
        let world_metadata = world.execution_metadata_bytes(source)?;
        Ok(Self {
            pack,
            extract,
            world,
            slot,
            world_metadata,
        })
    }
    fn execution_control_bytes(&self) -> Option<usize> {
        let parts = [
            usize::try_from(self.pack.recipe.controls).ok()?,
            usize::try_from(self.extract.recipe.controls).ok()?,
            size_of::<[Array; 3]>(),
            size_of::<OriginalCommunicationCompletion>(),
            size_of::<(bool, usize, usize, &Array, &ArithmeticQuote)>(),
            packed::controls::<logical_collective::Native<'_>>(1)?,
            packed::controls::<logical_collective::Native<'_>>(2)?,
            OperationEvent::traversal_leaf_control_bytes()?.checked_mul(3)?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    fn capacity(&self) -> Result<AgreementCapacity, Error> {
        sum(
            sum(self.pack.capacity, self.extract.capacity)?,
            AgreementCapacity {
                graph: self.world.graph_capacity(),
                records: self.world.record_capacity(),
                backing: self.world.backing_capacity(),
            },
        )
    }
}
impl Worker<'_, '_, '_, '_> {
    fn packed_arithmetic(
        &self,
        input: &Array,
        quote: &ArithmeticQuote,
        pack: bool,
        slot: usize,
        world: usize,
    ) -> Result<Array, Error> {
        self.check_deadline()?;
        if input.dtype() != safemlx::Dtype::Int32
            || if pack {
                input.shape() != [1]
            } else {
                input.shape() != [i32::try_from(world).map_err(|_| overflow())?, 1]
            }
        {
            return Err(failure(
                Cause::Identity,
                &self.quote.source,
                &self.quote.funding,
            ));
        }
        OperationEvent::validate_traversal_leaf(input, self.observer)
            .map_err(|cause| self.native(cause))?;
        let ready = self.ready(quote.recipe.completion.traversal)?;
        let bank =
            OperationEvent::prepare_resident_graph(quote.recipe.completion.graph, self.observer)
                .map_err(|cause| self.native(cause))?;
        let output = if pack {
            packed::pack(&logical_collective::Native(self.stream), input, slot, world)
        } else {
            packed::sum_result(&logical_collective::Native(self.stream), input, slot)
        }
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

use eredu_nn::workspace::WorkspaceMetadataAllocation;

impl PreparedStatusAgreement<'_> {
    pub(super) fn prepare_frame_control_bytes(plan: StatusPlan<'_>) -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<Option<Self>>(),
            size_of::<Result<Option<Self>, Error>>(),
            size_of::<[Option<usize>; 2]>(),
            size_of::<[AgreementCapacity; 3]>(),
            size_of::<std::ops::Range<usize>>(),
            size_of::<[GroupWorkerOperation; 2]>(),
            size_of::<usize>(),
            size_of::<[i32; 2]>(),
            size_of_val(&plan.operations()),
            size_of::<Option<(bool, usize)>>(),
            failure_control_bytes()?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}

impl PreparedStatusAgreement<'_> {
    pub(super) fn prepare_worker_control_bytes(plan: StatusPlan<'_>) -> Option<usize> {
        let parts = [
            plan.control_bytes::<Worker<'_, '_, '_, '_>>()?,
            // Every planned leaf borrows a completed predecessor. Price
            // its exact scoped validation before issuing any native work.
            OperationEvent::traversal_leaf_control_bytes().and_then(|bytes| {
                bytes.checked_mul(
                    plan.sends()
                        .checked_add(plan.receives())?
                        .checked_add(plan.additions().checked_mul(2)?)?,
                )
            })?,
            size_of::<ArithmeticQuote>(),
            size_of::<Option<ArithmeticQuote>>(),
            size_of::<safemlx::PreparedArrayClone>(),
            size_of::<safemlx::StreamCopyPlan<()>>(),
            safemlx::PreparedArrayClone::control_bytes()?,
            safemlx::Stream::device_type_control_bytes()?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}

impl PreparedStatusAgreement<'_> {
    pub(super) fn submit_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<Worker<'_, '_, '_, '_>>(),
            size_of::<Array>(),
            size_of::<Result<(OriginalCommunicationBool, MlxNeuralCommunicationCompletion), Error>>(
            ),
            failure_control_bytes()?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}

impl Worker<'_, '_, '_, '_> {
    pub(super) fn alias_control_bytes() -> Option<usize> {
        let parts = [
            safemlx::PreparedArrayClone::control_bytes()?,
            Array::inspection_clone_handle_bytes(),
            size_of::<Result<Array, Error>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}

impl StatusQuote {
    pub(super) fn requirements(
        &self,
        group: &Group,
        plan: StatusPlan<'_>,
    ) -> Option<super::AgreementRequirements> {
        if !self.matches(group, plan) {
            return None;
        }
        let mut capacity = self.final_quote.capacity;
        let mut preparation = PreparedStatusAgreement::prepare_frame_control_bytes(plan)?
            .checked_add(PreparedStatusAgreement::prepare_worker_control_bytes(plan)?)?;
        for leaf in &self.leaves {
            preparation = preparation
                .checked_add(leaf.operation)?
                .checked_add(leaf.backing)?;
            capacity = sum(capacity, leaf.capacity).ok()?;
        }
        let comparison = group
            .retained_transport_stream()
            .and_then(|stream| safemlx::StreamCopyPlan::<()>::capture(stream).ok())?
            .source_comparison_control_bytes()?;
        let aliases = match plan {
            StatusPlan::Chain(plan) => usize::from(plan.left.is_none()),
            StatusPlan::Exchange { .. } => 1,
            StatusPlan::Routed(plan) => plan.len(),
            StatusPlan::Packed(_) => 0,
        };
        let mut submission = PreparedStatusAgreement::submit_control_bytes()?
            .checked_add(super::metadata_bytes(&[comparison])?)?
            .checked_add(super::metadata_bytes(&[self.execution_control_bytes()?])?)?
            .checked_add(Worker::alias_control_bytes()?.checked_mul(aliases)?)?;
        for leaf in &self.leaves {
            submission = submission
                .checked_add(leaf.operation)?
                .checked_add(leaf.backing)?
                .checked_add(leaf.resources)?
                .checked_add(leaf.submit)?;
        }
        let checked_submit =
            ReadyCompletionResources::original_checked_one_root_submit_control_bytes()?;
        let arithmetic_completion = |quote: &ArithmeticQuote| -> Option<usize> {
            super::quote::arithmetic_resources(group, &quote.recipe.completion.traversal)?
                .checked_add(checked_submit)
        };
        submission = submission
            .checked_add(OriginalCommunicationSource::validation_control_bytes()?)?
            .checked_add(PreparedCommunicationScalar::control_bytes()?)?
            .checked_add(arithmetic_completion(&self.final_quote)?)?;
        if let Some(add) = &self.add {
            for _ in 0..self.additions {
                capacity = sum(capacity, add.capacity).ok()?;
            }
            submission =
                submission.checked_add(arithmetic_completion(add)?.checked_mul(self.additions)?)?;
        }
        if let Some(packed) = &self.packed {
            capacity = sum(capacity, packed.capacity().ok()?).ok()?;
            submission = submission
                .checked_add(arithmetic_completion(&packed.pack)?)?
                .checked_add(arithmetic_completion(&packed.extract)?)?
                .checked_add(packed.world_metadata)?;
        }
        Some(super::AgreementRequirements {
            capacity,
            capacity_metadata: preparation,
            execution_metadata: preparation.checked_add(submission)?,
        })
    }
}
