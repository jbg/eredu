use super::*;
use crate::capture::partition::{PartitionCaptureReceiptLimits, PartitionCaptureReceiptPlan};
use eredu_nn::workspace::{HostMetadataFunding, WorkspaceMetadataAllocation};

fn fragment(
    source: &OriginalCaptureSource,
    metadata: &HostMetadataFunding,
) -> OwnedPartitionFragmentHostPlan {
    let context = PartitionCaptureContext {
        artifact_identity: "actual-artifact".into(),
        execution_identity: "actual-model".into(),
        run_identity: "actual-run".into(),
        overlay_identity: None,
        capture_plan_identity: source.plan().admission().identity().into(),
        selection_index: 0,
        phase: CapturePhase::Decode,
        prediction: 0,
        forward_epoch: 1,
        invocation: Some(CaptureInvocationShape {
            batch: 1,
            sequence: 1,
            context: None,
        }),
        invocation_window: None,
    };
    let mut ledger = CaptureLedger::new(source.plan().admission());
    ledger.begin_step();
    let receipt = PartitionCaptureReceiptPlan::new_complete_shared_funded(
        source.plan(),
        &context,
        0,
        2,
        PartitionCaptureReceiptLimits {
            max_producers: 1,
            max_fragments: 1,
            max_record_bytes: 4096,
        },
        metadata,
        &mut ledger,
    )
    .unwrap();
    OwnedPartitionFragmentHostPlan::prepare(receipt, None).unwrap()
}

#[test]
fn autoregressive_partition_hosts_share_admission_and_outlive_the_consumed_frame() {
    let selected = selected();
    let config = SpeculativeConfig {
        max_tokens: 3,
        max_draft_tokens: 1,
        ..Default::default()
    };
    let schedule = AutoregressiveSchedulePlan::new(
        &selected,
        NonZeroUsize::new(1).unwrap(),
        NonZeroU64::new(2).unwrap(),
        NonZeroU64::new(16).unwrap(),
        &config,
        SpeculativeSchedulerOptions::default(),
    )
    .unwrap();
    let invocation = AutoregressiveInvocation::decode(AutoregressivePass::DraftCommit, 1).unwrap();
    let report = report(
        schedule
            .workspace_geometry(2, invocation, NonZeroU64::new(1).unwrap())
            .unwrap(),
    );
    let capacity = 1 << 26;
    let pool = crate::working_memory::memory_fixture::host_ledger(capacity, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let limits = crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity);
    let metadata = pool
        .prepare_workspace_metadata(&execution, limits.clone())
        .unwrap();
    let source = capture_source(&pool, 2, CaptureTransform::FullTensor, 4);
    let foreign = capture_source(&pool, 2, CaptureTransform::FullTensor, 4);
    let request =
        OriginalSpeculativeRequest::prepare(&pool, &execution, &schedule, limits).unwrap();
    let lineage = request.prepare_model_capture_lineage(&source).unwrap();
    let mask = [true];
    let make_frames = || {
        frames(
            &source,
            &lineage,
            invocation,
            report.span_workspace_plan(),
            &mask,
        )
    };
    let wrong = make_frames()
        .pop()
        .unwrap()
        .with_partition_fragments(vec![fragment(&foreign, &metadata)]);
    assert!(matches!(
        wrong,
        Err(CaptureRunHostError::Memory(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    drop(wrong);
    let frame = make_frames().pop().unwrap();
    let base = frame.initialization_peak_bytes();
    let plan = fragment(&source, &metadata);
    let fragments_peak = plan.initialization_peak_bytes();
    assert!(fragments_peak > 0);
    let mut plans = metadata.metadata_vec(1).unwrap();
    plans.push(plan);
    let frame = frame.with_partition_fragments(plans).unwrap();
    assert!(frame.initialization_peak_bytes() >= base + fragments_peak);
    let plans = [frame];
    let mut cursor = schedule.into_cursor();
    let (role, prepared) = request
        .reserve_role_with_capture(
            cursor.claim(2, invocation).unwrap(),
            requirements(report.span_workspace_plan()),
            AutoregressiveCaptureHostPlan::prepare(&plans).unwrap(),
        )
        .unwrap();
    let admitted = pool.payload_used_bytes().unwrap();
    let mut bank = prepared.construct().unwrap();
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        admitted,
        "construction consumes the same role allowance"
    );
    let mut owner = bank.begin_decode().unwrap();
    let fragments = owner.take_partition_fragments().unwrap();
    assert_eq!(fragments.len(), 1);
    assert!(owner.take_partition_fragments().is_none());
    assert_eq!(pool.payload_used_bytes().unwrap(), admitted);
    drop(owner);
    drop(bank);
    drop(plans);
    drop(role);
    request.close().unwrap();
    drop(request);
    drop(lineage);
    drop(source);
    drop(foreign);
    drop(metadata);
    assert!(
        pool.payload_used_bytes().unwrap() > 0,
        "the actual fragment Host retains its admitted account"
    );
    drop(fragments);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
