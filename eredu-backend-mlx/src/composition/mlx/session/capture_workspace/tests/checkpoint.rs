use super::*;
use eredu_runtime::{
    capture::{FundedCaptureError, ScheduledCaptureBackend},
    working_memory::{
        CaptureRunHostPlan, CaptureTensorClaim, ClaimedCaptureTensor, InferenceExecutionIdentity,
        WorkingMemoryPool,
    },
};

struct MetadataOnly;
impl ScheduledCaptureBackend for MetadataOnly {
    type Tensor = ();
    type Error = std::io::Error;
    fn validate_source(
        &self,
        _: &(),
        _: &CaptureTensorGeometry<'_>,
    ) -> std::result::Result<eredu_core::checkpoint::TensorDtype, Self::Error> {
        unreachable!("the initial metadata-only step selects no tensor")
    }
    fn estimate(
        &self,
        _: &(),
        _: &CaptureTensorGeometry<'_>,
    ) -> std::result::Result<CaptureUsage, CaptureError> {
        unreachable!("the initial metadata-only step selects no tensor")
    }
    fn transform(
        &mut self,
        _: &(),
        _: CaptureTensorClaim<'_, '_>,
    ) -> std::result::Result<ClaimedCaptureTensor, Self::Error> {
        unreachable!("the initial metadata-only step selects no tensor")
    }
}

#[test]
fn saved_decode_origin_prices_first_native_prefill_and_keeps_capture_quota() {
    let source = SharedCapturePlan::new(admitted_limited(
        vec![
            SymbolicDimension::Known(1),
            SymbolicDimension::Known(1),
            SymbolicDimension::Known(2),
        ],
        CaptureTransform::FullTensor,
        vec![],
        1,
        CaptureLimitPolicy::Skip,
    ));
    let plan = CaptureRunHostPlan::prepare(&source).unwrap();
    let bytes = plan.initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let layout = StateMemoryLayout::new(
        LayerSchedule::new(1, vec![cache::LayerCachePolicy::NoState]).unwrap(),
        vec![0],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let bound = |n| WorkspaceBound::bounded(n, "exact neutral capture fixture");
    let state = estimate_runtime_state(
        &layout,
        InputTokenCount::text(3),
        4,
        1,
        std::num::NonZeroU8::new(4).unwrap(),
    )
    .unwrap()
    .with_execution_workspace(ExecutionWorkspaceEstimate {
        geometry: geometry(),
        activations: bound(bytes),
        attention: bound(0),
        vocabulary: bound(0),
        state_update: bound(0),
        materialization: bound(0),
        retained: bound(0),
    })
    .unwrap();
    let (reservation, funding) = pool
        .reserve_with_capacity(
            &InferenceExecutionIdentity::default(),
            &Admission {
                state,
                requested_positions: 7,
                incremental_required_bytes: bytes,
                available_memory_bytes: None,
            },
            bytes,
        )
        .unwrap()
        .into_funding()
        .unwrap();
    let mut parent = funding
        .prepare_capture_run(&reservation, plan)
        .unwrap()
        .into_capture_session()
        .unwrap();
    let epoch = DistributedCommitEpoch::new(1).unwrap();
    parent
        .with_observer(&mut MetadataOnly, 0, &|cause| cause, |observer| {
            observer.prepare_transaction(epoch, eredu_runtime::ExpertPass::Prefill)?;
            observer.complete_transaction(epoch)?;
            observer.finish_transaction(epoch, true);
            Ok::<_, FundedCaptureError<std::io::Error>>(())
        })
        .unwrap()
        .unwrap();
    drop(parent.take_shared_step().unwrap());
    // This native-free fixture uses ordinary host authority for the fixed saved
    // record; the actual claim table above is funded by its exact original H.
    let checkpoint = parent
        .prepare_checkpoint()
        .unwrap()
        .construct(&HostPreparationAuthority::default())
        .unwrap();
    assert_eq!(checkpoint.next_prediction(), 1);
    let saved_usage = checkpoint.inherited_usage();
    let geometry = InferenceGeometry {
        cached_positions: 3,
        input_positions: 1,
        max_output_tokens: 2,
        prefill_chunk_positions: 1,
        ..geometry()
    };
    let context = WorkspaceContext::new(Facts::default());
    let counter = Cell::new(CaptureNativePopulation::default());
    assert!(
        CaptureWorkspaceObserver::with_checkpoint_transfers(
            &checkpoint,
            InferenceGeometry {
                cached_positions: 4,
                ..geometry
            },
            None,
            &context,
            &counter,
        )
        .is_err()
    );
    let (mut observer, _) = CaptureWorkspaceObserver::with_checkpoint_transfers(
        &checkpoint,
        geometry,
        None,
        &context,
        &counter,
    )
    .unwrap();
    let (value, _storage) = imported(&context, &[1, 1, 2], WorkspaceDtype::Float32, Some(8));
    context
        .begin_state_span(std::slice::from_ref(&value))
        .unwrap();
    let first = InferenceWorkspaceSpan::Prefill(eredu_runtime::prefill::PrefillChunk {
        input: 0..1,
        position: 3,
        output: OutputDemand::LastPosition,
    });
    assert!(observer.begin_span(geometry, &first, 0, &context).unwrap());
    observer.observe("block.output", &value).unwrap();
    assert_eq!(counter.get().publications, 1);
    assert_eq!(observer.active, Some(1));
    assert_eq!(observer.ledger.total().captures, saved_usage.captures + 1);
    let report = context.finish_report(std::slice::from_ref(&value)).unwrap();
    assert_eq!(casts(&report), 1);
    observer.end_span(&first, &context).unwrap();

    context
        .begin_state_span(std::slice::from_ref(&value))
        .unwrap();
    let second = InferenceWorkspaceSpan::Decode {
        index: 0,
        position: 4,
        output: OutputDemand::LastPosition,
    };
    assert!(observer.begin_span(geometry, &second, 1, &context).unwrap());
    observer.observe("block.output", &value).unwrap();
    assert_eq!(observer.active, Some(2));
    assert_eq!(
        counter.get().publications,
        0,
        "the first resumed capture spent the remaining quota"
    );
    let report = context.finish_report(std::slice::from_ref(&value)).unwrap();
    assert_eq!(casts(&report), 0);
    observer.end_span(&second, &context).unwrap();
    assert_eq!(checkpoint.inherited_usage(), saved_usage);
    assert_eq!(
        parent.spent_steps(),
        1,
        "cold quotation never mutates the live source"
    );
}
