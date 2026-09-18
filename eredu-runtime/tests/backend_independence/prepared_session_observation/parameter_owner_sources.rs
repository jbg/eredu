use super::*;
use eredu_runtime::{DirectReplicatedTextExecution, ReplicatedTextExecutionStrategy};

// Deliberately provides no static companion. Other required methods are never
// used by this test and cannot fabricate an aggregate through a fallback.
struct CustomStrategy;
impl
    ReplicatedTextExecutionStrategy<
        OrdinaryTextFixture,
        FakeBackend,
        DeviceState<FakeBackend, FakeLayerState>,
        RecordingPolicy,
        RecordingPolicy,
    > for CustomStrategy
{
    fn group_submission_mechanism(_: &()) -> eredu_runtime::GroupSubmissionMechanism {
        eredu_runtime::GroupSubmissionMechanism::PolicyOnly
    }
    type Runtime = ();
    fn mark_terminal_failure(_: &(), _: DistributedExecutionPhase) {}
    fn visit_retained_values(_: &(), _: &mut dyn FnMut(&FakeTensor)) -> bool {
        false
    }
    fn bounded_policy(_: &()) -> Option<&RecordingPolicy> {
        None
    }
    fn execution_residency(
        _: &(),
        _: &eredu_runtime::SelectedReplicatedTextRealization,
    ) -> eredu_runtime::ExecutionResidency {
        panic!("static loan must not ask residency")
    }
    fn forward_with_observer<'a, O>(
        &mut self,
        _: &mut (),
        _: <OrdinaryTextFixture as LayeredArchitecture<
            FakeBackend,
            DeviceState<FakeBackend, FakeLayerState>,
        >>::Input<'a>,
        _: &mut DeviceState<FakeBackend, FakeLayerState>,
        _: eredu_runtime::ExpertPass,
        _: &(),
        _: &mut O,
        _: eredu_core::OutputDemand,
    ) -> Result<
        (
            Option<FakeTensor>,
            <OrdinaryTextFixture as LayeredArchitecture<
                FakeBackend,
                DeviceState<FakeBackend, FakeLayerState>,
            >>::ForwardContext,
        ),
        ReplicatedTextSessionError<Error, &'static str, Infallible>,
    >
    where
        O: ActivationObserver<FakeTensor, Error> + ?Sized,
    {
        panic!("static loan must not execute")
    }
}

#[test]
fn default_custom_strategy_refuses_static_coverage_without_fallback_work() {
    assert!(<CustomStrategy as ReplicatedTextExecutionStrategy<
        OrdinaryTextFixture,
        FakeBackend,
        DeviceState<FakeBackend, FakeLayerState>,
        RecordingPolicy,
        RecordingPolicy,
    >>::static_modules_ref(&())
    .is_none());
}

#[test]
fn borrowed_static_aggregate_uses_actual_resident_and_bounded_runtime_without_work() {
    for residency in [
        LayerWeightResidency::FullyResident,
        LayerWeightResidency::LayerwiseHost(Default::default()),
    ] {
        let (session, counters) = session(residency);
        let before = counters.snapshot();
        session
            .inspect_runtime_execution_fixed(|_, _, runtime| {
                let borrowed = <DirectReplicatedTextExecution as ReplicatedTextExecutionStrategy<
                    OrdinaryTextFixture,
                    FakeBackend,
                    DeviceState<FakeBackend, FakeLayerState>,
                    RecordingPolicy,
                    RecordingPolicy,
                >>::static_modules_ref(runtime)
                .unwrap();
                assert!(std::ptr::eq(borrowed, runtime.static_modules_ref()));
                assert!(std::ptr::eq(
                    borrowed,
                    <DirectReplicatedTextExecution as ReplicatedTextExecutionStrategy<
                        OrdinaryTextFixture,
                        FakeBackend,
                        DeviceState<FakeBackend, FakeLayerState>,
                        RecordingPolicy,
                        RecordingPolicy,
                    >>::static_modules_ref(runtime)
                    .unwrap()
                ));
                Ok::<_, Infallible>(())
            })
            .unwrap()
            .unwrap();
        assert_eq!(counters.snapshot(), before);
    }
}

#[test]
fn custom_partition_owner_unavailable_and_actual_fence_precede_static_access() {
    let (mut session, _, _, calls, _) = partitioned_cache_control_session(
        0,
        DistributedExecutionPhase::ControlCapturePreparation,
        None,
        true,
    );
    session
        .inspect_runtime_execution_fixed(|_, _, runtime| {
            assert!(
                <ReferencePartitionedStrategy as ReplicatedTextExecutionStrategy<
                    OrdinaryTextFixture,
                    FakeBackend,
                    DeviceState<FakeBackend, FakeLayerState>,
                    RecordingPolicy,
                    RecordingPolicy,
                >>::static_modules_ref(runtime)
                .is_none()
            );
            Ok::<_, Infallible>(())
        })
        .unwrap()
        .unwrap();
    assert!(session.capture_control_state(&()).is_err());
    let before = calls.borrow().len();
    let visited = Cell::new(false);
    assert!(session
        .inspect_runtime_execution_fixed(|_, _, _| {
            visited.set(true);
            Ok::<_, Infallible>(())
        })
        .is_err());
    assert!(!visited.get());
    assert_eq!(calls.borrow().len(), before);
}
