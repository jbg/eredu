//! Real event readiness and durable native parent/child lifetime are distinct.
use super::*;
use crate::backend::runtime::distributed::{
    completion::prepared::{CompletionResourceLayout, PreparedCompletionResources},
    topology::ParallelCommunicators,
};
use eredu_core::{CollectiveGroupId, CompletionCancellationMode};
use eredu_runtime::{
    working_memory::InferenceExecutionIdentity, CommunicationCompletionPolicy,
    CommunicationGroupDescriptor, CommunicationGroupRequirements, CommunicationManifest,
    CommunicationOperationRequirement, PartitionCommunicationAuthority,
};
use safemlx::{
    Device, DeviceType, OperationEvalTraversalLimits, OperationEvent, PreparedInputRuntime,
    PreparedSubmissionScopeOwner, SubmissionScope,
};
use std::time::Duration;

const GRAPH: usize = 4 << 20;
const RECORDS: usize = 1 << 20;
struct FixtureProducer<'a, 'native> {
    source: &'a OriginalCommunicationSource<'native>,
    pool: &'a MemoryLedger,
    runtime: &'a PreparedInputRuntime,
    stream: &'a Stream,
    input: &'a Array,
    backing: usize,
    ready: ReadyCompletionResources,
}
struct FixtureOutput {
    completion: Option<NativeCompletion>,
    child: Option<SubmissionScope>,
    output: Array,
}
impl CommunicationPreparationProducer for FixtureProducer<'_, '_> {
    type Output = FixtureOutput;
    type Error = Error;
    fn source(&self) -> &RetainedCommunicationSource {
        self.source.source()
    }
    fn frame(&self) -> &[u32] {
        &[2, 3, 5]
    }
    fn validate_pool(&self, pool: &MemoryLedger) -> Result<(), WorkingMemoryError> {
        if self.pool.same_ledger(pool) {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        // Actual fixture capacities and canonical typed constructor populations;
        // these are explicit native arena allocations, not a model quote.
        let child = PreparedSubmissionScopeOwner::<Custody>::layout().unwrap();
        let parts = [
            PreparedSubmissionGraphQuota::<Custody>::layout(GRAPH)
                .unwrap()
                .total_bytes()
                .unwrap(),
            PreparedSubmissionRecordQuota::<Custody>::layout(RECORDS)
                .unwrap()
                .total_bytes()
                .unwrap(),
            PreparedPrefillFailure::<Custody>::layout()
                .unwrap()
                .total_bytes()
                .unwrap(),
            PreparedOriginalBufferBudget::<Custody>::layout(self.runtime, self.backing)
                .unwrap()
                .total_owner_bytes()
                .unwrap(),
            self.backing,
            usize::try_from(PreparedRecovery::<Retained, Custody>::control_bytes().unwrap())
                .unwrap(),
            safemlx::PreparedThreadRuntimeHousekeeping::<Custody>::control_bytes().unwrap(),
            safemlx::OriginalNativeControlLayout::inspect()
                .unwrap()
                .fixed_control_bytes,
            OriginalScopeObserver::control_bytes().unwrap(),
            SharedPreparationCustody::allocation_bytes().unwrap(),
            child.allocation_bytes().unwrap(),
            child.native_retirement_control_bytes,
            child.prepared_bytes,
            child.preparation_control_bytes,
            child.begin_control_bytes,
            child.retirement_control_bytes,
            child.preparation_failure_bytes,
            child.begin_failure_bytes,
            size_of::<Self>(),
            size_of::<FixtureOutput>(),
            size_of::<NativeCompletion>(),
            size_of::<Custody>(),
            size_of::<Retained>(),
            size_of::<SubmissionScope>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or(WorkingMemoryError::Overflow)
    }
    fn produce(self, raw: CommunicationPreparationCustody) -> Result<FixtureOutput, Error> {
        let custody = Custody {
            source: self.source.source().clone(),
            native: SharedPreparationCustody::new(raw),
            funding: self.source.funding().clone(),
        };
        let budget =
            PreparedOriginalBufferBudget::try_new(self.runtime, self.backing, custody.clone())
                .unwrap()
                .try_allocate()
                .unwrap();
        let graph = PreparedSubmissionGraphQuota::try_new(GRAPH, custody.clone())
            .unwrap()
            .try_allocate()
            .unwrap();
        let records = PreparedSubmissionRecordQuota::try_new(RECORDS, custody.clone())
            .unwrap()
            .try_allocate()
            .unwrap();
        let failure = PreparedPrefillFailure::try_new(custody.clone())
            .unwrap()
            .try_allocate()
            .unwrap();
        let child = PreparedSubmissionScopeOwner::try_new(custody.clone())
            .unwrap()
            .with_graph_quota(graph.clone())
            .with_record_quota(records.clone());
        let housekeeping = safemlx::PreparedThreadRuntimeHousekeeping::new(
            crate::backend::submission_recovery::reap,
            custody.clone(),
        )
        .try_register()
        .unwrap();
        let retained = Retained {
            budget,
            failure,
            housekeeping,
            custody: custody.clone(),
        };
        let mut recovery = PreparedRecovery::new(retained, custody.clone())
            .unwrap_or_else(|error| panic!("fixture recovery preparation: {:?}", error.cause))
            .with_graph_quota(Some(graph))
            .with_record_quota(Some(records))
            .try_begin()
            .unwrap_or_else(|error| panic!("fixture recovery begin: {:?}", error.cause));
        let (output, completion, observer, child) =
            recovery.configure_scope_with_retention(|scope, retained| {
                scope.enable_scoped_observation().unwrap();
                scope.require_original_native_controls().unwrap();
                retained.failure.bind_original_scope(scope).unwrap();
                scope.enable_original_native_controls().unwrap();
                scope.bind_original_buffer_budget(&retained.budget).unwrap();
                let observer = OriginalScopeObserver::require_current().unwrap();
                OperationEvent::validate_traversal_leaf(self.input, &observer).unwrap();
                let output = self.input.square(self.stream).unwrap();
                let completion = self
                    .ready
                    .submit_original(self.source, &observer, self.stream, [&output])
                    .unwrap();
                // This genuine active descendant keeps the parent structurally
                // pending even after the submitted event and every Record finish.
                let child = SubmissionScope::try_begin_original_child(child, &observer).unwrap();
                Ok::<_, Error>((output, completion, observer, child))
            })?;
        recovery.seal();
        Ok(FixtureOutput {
            output,
            child: Some(child),
            completion: Some(NativeCompletion {
                inner: Some(completion.into()),
                recovery: RefCell::new(Some(recovery)),
                observer,
                custody,
            }),
        })
    }
}

fn run_fixture(timeout: bool) {
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let _runtime = safemlx::PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let world =
        safemlx::distributed::Group::init(false, safemlx::distributed::Backend::Ring).unwrap();
    let (runtime, _) =
        crate::backend::managed_memory::input_allocator::prepare_admitted(&pool).unwrap();
    // Compare ordinary-host retirement boundaries on both sides. Native owner
    // destructors can enqueue another batch; none of these drains waits for work.
    while safemlx::reclaim_allocation_owners() != 0 {}
    let baseline = pool.fixture_host_charge().unwrap();
    let group = CommunicationGroupDescriptor::new(
        CollectiveGroupId::new(59),
        0,
        vec![0],
        Some(0),
        CommunicationGroupRequirements::new([CommunicationOperationRequirement::barrier(true)])
            .unwrap(),
    )
    .unwrap();
    let manifest = CommunicationManifest::new(1, 0, vec![group], vec![])
        .unwrap()
        .with_completion_policy(
            CommunicationCompletionPolicy::new(
                Duration::from_secs(1),
                CompletionCancellationMode::QuarantineUntilComplete,
            )
            .unwrap(),
        );
    let actual = ParallelCommunicators::from_manifest(&manifest, &world, &stream).unwrap();
    let authority = PartitionCommunicationAuthority::from_manifest(&manifest).unwrap();
    // This API compares a shared-domain ceiling, not an incremental allowance.
    // Preserve the actual admitted singleton baseline and add only the same
    // finite headroom reserved for this fixture's source and native arenas.
    let capacity = baseline.checked_add(16 << 20).unwrap();
    let funding = pool
        .prepare_workspace_metadata(
            &InferenceExecutionIdentity::default(),
            crate::memory_fixture::resolved_limits(capacity),
        )
        .unwrap();
    let source = actual
        .bind_original_source(&manifest, &world, &authority, &funding)
        .unwrap();
    let input = Array::from_slice(&[2_f32, -3., 5.], &[3]);
    input.evaluated().unwrap();
    // Square creates one output with the input's byte length. The retained
    // allocator supplies its physical rounding, including the real page size.
    funding
        .reserve_metadata(OriginalBufferBudget::request_layout_control_bytes().unwrap())
        .unwrap();
    let backing = OriginalBufferBudget::request_layout(&runtime, input.nbytes())
        .unwrap()
        .capacity();
    let traversal = OperationEvent::eval_traversal_layout(OperationEvalTraversalLimits {
        roots: 1,
        arrays: 4,
        tape_entries: 3,
        input_edges: 4,
        output_slots: 3,
        streams: 1,
        captures: 8,
    })
    .unwrap();
    let mut ready = PreparedCompletionResources::prepare_original(
        &source,
        CompletionResourceLayout {
            arrays: 1,
            counts: &[],
            groups: 0,
            routes: 0,
            streams: 1,
        },
        traversal,
    )
    .unwrap();
    ready.push_array(input.clone()).unwrap();
    ready.push_stream(stream.clone()).unwrap();
    let mut prepared = pool
        .prepare_communication(FixtureProducer {
            source: &source,
            pool: &pool,
            runtime: &runtime,
            stream: &stream,
            input: &input,
            backing,
            ready: ready.finish().unwrap(),
        })
        .unwrap();
    // Submission adds real cumulative host controls. Release every external
    // source alias so the final comparison includes that source's retirement,
    // instead of incorrectly expecting its live planning account to refund.
    drop(source);
    drop((actual, authority, funding));
    let mut completion = prepared.with_output(|value| value.completion.take().unwrap());
    completion.inner.as_ref().unwrap().wait().unwrap();
    assert!(completion.inner.as_ref().unwrap().is_complete().unwrap());
    // The retained parent event has completed while its child is still current.
    // Authenticate that completed value with the parent observer; evaluating
    // again would ask the child to authorize work owned by the sealed parent.
    assert_eq!(
        prepared
            .output()
            .output
            .completed_in_original_scope(&completion.observer)
            .unwrap()
            .try_as_slice::<f32>()
            .unwrap(),
        &[4., 9., 25.]
    );
    assert!(
        !completion.try_finish().unwrap(),
        "event readiness cannot retire its live descendant"
    );
    assert!(
        completion.recovery.borrow().is_some(),
        "pending retains the same recovery node"
    );
    let mut child = prepared.with_output(|value| value.child.take().unwrap());
    let retirement_deadline = std::time::Instant::now() + Duration::from_secs(10);
    if timeout {
        let wait = BoundedCompletionWait::new(
            Duration::from_millis(1),
            CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap();
        assert!(matches!(
            completion.bounded(wait).unwrap(),
            BoundedCompletionOutcome::DeadlineExceeded {
                cancellation: CompletionCancellationMode::QuarantineUntilComplete
            }
        ));
        drop(completion);
        drop(prepared);
        crate::backend::submission_recovery::reap();
        safemlx::reclaim_allocation_owners();
        assert!(
            pool.fixture_host_charge().unwrap() > baseline,
            "pending native child/recovery retain original source and producer account"
        );
        child.seal();
        drop(child);
    } else {
        child.seal();
        drop(child);
        crate::backend::submission_recovery::wait_for_retirement(|| {
            assert!(
                std::time::Instant::now() < retirement_deadline,
                "sealed child must allow terminal readiness"
            );
            completion.try_finish().unwrap()
        });
        assert!(completion.recovery.borrow().is_none());
        drop(completion);
        drop(prepared);
    }
    crate::backend::submission_recovery::wait_for_retirement(|| {
        // Scope/Graph/Record releases enqueue their Rust custody. Backend
        // recovery retirement alone does not drain this native-owner queue.
        safemlx::reclaim_allocation_owners();
        let used = pool.fixture_host_charge().unwrap();
        assert!(
            std::time::Instant::now() < retirement_deadline,
            "source/producer custody did not retire: {used} vs {baseline}"
        );
        used == baseline
    });
    drop((input, runtime, stream, world));
    safemlx::reclaim_allocation_owners();
}

#[test]
#[ignore = "requires fresh original allocator and native CPU source"]
fn readiness_event_does_not_retire_a_pending_enclosing_scope() {
    run_fixture(false);
}
#[test]
#[ignore = "requires fresh original allocator and native CPU source"]
fn readiness_deadline_keeps_pending_scope_custody_until_terminal() {
    run_fixture(true);
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
