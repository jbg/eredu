use super::super::prepared::{CompletionResourceLayout, PreparedCompletionResources};
use super::*;
use crate::backend::runtime::distributed::topology::ParallelCommunicators;
use eredu_core::{CollectiveGroupId, Completion as _, CompletionCancellationMode};
use eredu_runtime::{
    working_memory::{InferenceExecutionIdentity, MemoryLedger},
    CommunicationCompletionPolicy, CommunicationGroupDescriptor, CommunicationGroupRequirements,
    CommunicationManifest, CommunicationOperationRequirement, PartitionCommunicationAuthority,
};

struct Scope {
    scope: safemlx::SubmissionScope,
    graph: safemlx::SubmissionGraphQuota,
    records: safemlx::SubmissionRecordQuota,
    failure: safemlx::RetainedPrefillFailure,
}
impl Scope {
    fn new() -> Self {
        let graph = safemlx::PreparedSubmissionGraphQuota::try_new(4 << 20, ())
            .unwrap()
            .try_allocate()
            .unwrap();
        let records = safemlx::PreparedSubmissionRecordQuota::try_new(1 << 20, ())
            .unwrap()
            .try_allocate()
            .unwrap();
        let failure = safemlx::PreparedPrefillFailure::try_new(())
            .unwrap()
            .try_allocate()
            .unwrap();
        let mut scope = safemlx::SubmissionScope::try_begin_retaining(
            safemlx::PreparedSubmissionScopeOwner::try_new(())
                .unwrap()
                .with_graph_quota(graph.clone())
                .with_record_quota(records.clone()),
        )
        .unwrap();
        scope.enable_scoped_observation().unwrap();
        scope.require_original_native_controls().unwrap();
        failure.bind_original_scope(&scope).unwrap();
        scope.enable_original_native_controls().unwrap();
        Self {
            scope,
            graph,
            records,
            failure,
        }
    }
    fn finish(mut self, observer: safemlx::OriginalScopeObserver) {
        self.scope.seal();
        crate::backend::submission_recovery::wait_for_retirement(|| {
            let (outcome, status) = observer.progress().unwrap();
            outcome == safemlx::ScopedSubmissionProgress::Observed && status.is_settled()
        });
        observer.retire_completed_records().unwrap();
        drop(observer);
        drop((self.scope, self.graph, self.records, self.failure));
        crate::backend::submission_recovery::reap();
        safemlx::reclaim_allocation_owners();
    }
}
fn prepared_event(
    source: &OriginalCommunicationSource<'_>,
    input: &Array,
    zero: &Array,
    stream: &Stream,
) -> ReadyCompletionResources {
    let traversal =
        safemlx::OperationEvent::eval_traversal_layout(safemlx::OperationEvalTraversalLimits {
            roots: 1,
            arrays: 5,
            tape_entries: 3,
            input_edges: 5,
            output_slots: 3,
            streams: 1,
            captures: 8,
        })
        .expect("current finite native Eval traversal");
    let mut prepared = PreparedCompletionResources::prepare_original(
        source,
        CompletionResourceLayout {
            arrays: 2,
            counts: &[],
            groups: 1,
            routes: 0,
            streams: 1,
        },
        traversal,
    )
    .unwrap();
    prepared.push_array(input.clone()).unwrap();
    prepared.push_array(zero.clone()).unwrap();
    prepared
        .push_group(source.group(0).unwrap().0.clone(), 0)
        .unwrap();
    prepared.push_stream(stream.clone()).unwrap();
    prepared.finish().unwrap()
}
#[test]
fn prepared_words_and_unicode_headers_share_resolver_and_keep_escaped_host_storage_paid() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let _runtime = safemlx::PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let world =
        safemlx::distributed::Group::init(false, safemlx::distributed::Backend::Ring).unwrap();
    let group = CommunicationGroupDescriptor::new(
        CollectiveGroupId::new(47),
        0,
        vec![0],
        Some(0),
        CommunicationGroupRequirements::new([CommunicationOperationRequirement::barrier(true)])
            .unwrap(),
    )
    .unwrap();
    let policy = CommunicationCompletionPolicy::new(
        std::time::Duration::from_millis(1),
        CompletionCancellationMode::QuarantineUntilComplete,
    )
    .unwrap();
    let manifest = CommunicationManifest::new(1, 0, vec![group], vec![])
        .unwrap()
        .with_completion_policy(policy);
    let actual = ParallelCommunicators::from_manifest(&manifest, &world, &stream).unwrap();
    let authority = PartitionCommunicationAuthority::from_manifest(&manifest).unwrap();
    let pool = crate::memory_fixture::ledger(16 << 20, 0).unwrap();
    let funding = pool
        .prepare_workspace_metadata(
            &InferenceExecutionIdentity::default(),
            crate::memory_fixture::resolved_limits(16 << 20),
        )
        .unwrap();
    let source = actual
        .bind_original_source(&manifest, &world, &authority, &funding)
        .unwrap();

    let expected = [7_i32, -11, 0, 13];
    let input = Array::from_slice(&expected, &[4]);
    let zero = Array::from_slice(&[0_i32], &[1]);
    input.evaluated().unwrap();
    zero.evaluated().unwrap();
    let ordinary = MlxCommunicationCompletion::submit(
        [&input],
        vec![input.clone()],
        vec![],
        vec![],
        vec![],
        vec![],
    )
    .unwrap();
    let (ordinary_words, ordinary) = ordinary.with_i32_words(input.clone());
    ordinary.wait().unwrap();
    assert_eq!(ordinary_words.resolve().unwrap(), expected);
    drop(ordinary);
    let ready = prepared_event(&source, &input, &zero, &stream);
    let scope = Scope::new();
    let observer = safemlx::OriginalScopeObserver::require_current().unwrap();
    safemlx::OperationEvent::validate_traversal_leaf(&input, &observer).unwrap();
    safemlx::OperationEvent::validate_traversal_leaf(&zero, &observer).unwrap();
    let output = input.add(&zero, &stream).unwrap();
    let prepared = PreparedCommunicationWords::prepare(&source, output).unwrap();
    let (words, completion) = prepared.submit(ready, &source, &observer, &stream).unwrap();
    assert!(words.result.value.borrow().ready.is_none());
    completion.wait().unwrap();
    assert!(completion.is_complete().unwrap());
    let escaped = words.resolve().unwrap();
    assert_eq!(escaped.as_slice(), expected);
    assert!(
        completion.is_complete().unwrap(),
        "polling consumed result does not allocate or republish its Vec"
    );
    drop(completion);
    drop((input, zero));
    scope.finish(observer);

    // The same resolver must preserve all U32 protocol bits, including words
    // outside I32's positive range; no signed casting or second resolver.
    let unsigned = [0x4552_5052_u32, u32::MAX, 0, 0x8000_0001];
    let input = Array::from_slice(&unsigned, &[4]);
    let zero = Array::from_slice(&[0_u32], &[1]);
    input.evaluated().unwrap();
    zero.evaluated().unwrap();
    let ready = prepared_event(&source, &input, &zero, &stream);
    let scope = Scope::new();
    let observer = safemlx::OriginalScopeObserver::require_current().unwrap();
    safemlx::OperationEvent::validate_traversal_leaf(&input, &observer).unwrap();
    safemlx::OperationEvent::validate_traversal_leaf(&zero, &observer).unwrap();
    let output = input.add(&zero, &stream).unwrap();
    let prepared = PreparedCommunicationU32Words::prepare(&source, output).unwrap();
    let (words, completion) = prepared.submit(ready, &source, &observer, &stream).unwrap();
    completion.wait().unwrap();
    let unsigned_escaped = words.resolve().unwrap();
    assert_eq!(unsigned_escaped.as_slice(), unsigned);
    assert!(completion.is_complete().unwrap());
    drop((completion, input, zero));
    scope.finish(observer);

    for (expectation, valid) in [("rôle→✓", true), ("rôle→✗", false)] {
        let bytes = "rôle→✓".as_bytes();
        let input = Array::from_slice(bytes, &[bytes.len() as i32]);
        let zero = Array::from_slice(&[0_u8], &[1]);
        input.evaluated().unwrap();
        zero.evaluated().unwrap();
        let ordinary = MlxCommunicationCompletion::submit(
            [&input],
            vec![input.clone()],
            vec![],
            vec![],
            vec![],
            vec![],
        )
        .unwrap()
        .with_boundary_headers([(input.clone(), expectation.as_bytes().to_vec())]);
        assert_eq!(ordinary.wait().is_ok(), valid);
        drop(ordinary);
        let ready = prepared_event(&source, &input, &zero, &stream);
        let scope = Scope::new();
        let observer = safemlx::OriginalScopeObserver::require_current().unwrap();
        safemlx::OperationEvent::validate_traversal_leaf(&input, &observer).unwrap();
        safemlx::OperationEvent::validate_traversal_leaf(&zero, &observer).unwrap();
        let output = input.add(&zero, &stream).unwrap();
        let header =
            PreparedCommunicationHeader::prepare(&source, output, expectation.as_bytes()).unwrap();
        let completion = header.submit(ready, &source, &observer, &stream).unwrap();
        let result = completion.wait();
        assert_eq!(result.is_ok(), valid);
        drop((result, completion, input, zero));
        scope.finish(observer);
    }
    drop(source);
    drop((actual, authority, funding, stream, world));
    crate::backend::submission_recovery::reap();
    safemlx::reclaim_allocation_owners();
    assert!(
        pool.fixture_host_charge().unwrap() > 0,
        "extracted word Vec retains its independent H"
    );
    assert_eq!(escaped.as_slice(), expected);
    drop(escaped);
    assert_eq!(unsigned_escaped.as_slice(), unsigned);
    assert!(
        pool.fixture_host_charge().unwrap() > 0,
        "unsigned readout retains its own paid host destination"
    );
    drop(unsigned_escaped);
    assert_eq!(pool.fixture_host_charge().unwrap(), 0);
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
