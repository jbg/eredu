use super::*;
use eredu_runtime::ReplicatedTextExecutionStrategy;

type State = DeviceState<FakeBackend, FakeLayerState>;
type Strategy = PartitionedTextExecution<
    ReferencePartitionExecutor,
    (),
    (),
    FakeTensorMetadata,
    NoBoundaryTransport,
    NoOutputPublisher,
    NoCommitAgreement,
>;
type Runtime = PartitionedTextRuntime<
    OrdinaryTextFixture,
    FakeBackend,
    State,
    RecordingPolicy,
    ReferencePartitionExecutor,
    (),
    (),
    FakeTensorMetadata,
    NoBoundaryTransport,
    NoOutputPublisher,
    NoCommitAgreement,
>;

#[test]
fn partitioned_strategy_lends_the_actual_resident_policy_without_promoting_bounded_owners() {
    for residency in [
        ExecutionResidency::FullyResident,
        ExecutionResidency::LayerwiseHost,
        ExecutionResidency::DenseDiskStream,
    ] {
        let counters = ReplicatedSessionCounters::default();
        let architecture = OrdinaryTextFixture {
            static_modules: FakeOperator,
            trace: Vec::new(),
            counters: counters.clone(),
            inconsistent_transport: false,
            inconsistent_identity: false,
        };
        let parameters = architecture.parameter_description(&()).unwrap();
        let partition = ArchitecturePartition::from_architecture::<FakeBackend, State, _, _>(
            &architecture,
            [("decoder", 0..1)],
            PartitionOwnership::new(true, true, std::iter::empty::<String>()).unwrap(),
            (),
            NoAuxiliaryBoundarySchema::new(1),
            &parameters,
        )
        .unwrap();
        let driver = LayeredPartitionDriver::new(&partition, 0, 0..1).unwrap();
        let plan = PartitionedExecutionPlan::new(
            architecture.execution_graph().unwrap().into_owned(),
            vec![(ArchitectureGroupKind::Decoder, false)],
            vec![Some(driver)],
            Vec::new(),
            None,
            None,
            PipelineWireContract::new(PipelineActivationDtype::Float32),
        )
        .unwrap();
        let communication = PartitionCommunication::<FakeBackend, (), (), _>::new(
            CommunicationManifest::new(1, 0, Vec::new(), Vec::new()).unwrap(),
            Vec::new(),
            Vec::new(),
            FakeTensorMetadata,
        )
        .unwrap();
        let policy = RecordingPolicy::new(vec![FakeUnit { marker: 37 }]);
        let original_units = policy.units.as_ptr();
        let bounded =
            (residency != ExecutionResidency::FullyResident).then(|| RecordingPolicy::bounded(1));
        let original_bounded = bounded.as_ref().map(|value| value.units.as_ptr());
        let runtime = Runtime::new(
            plan,
            ReferencePartitionExecutor {
                architecture,
                policy,
                expose_policy: true,
                fail_after_state: Rc::new(Cell::new(false)),
            },
            communication,
            (),
            NoBoundaryTransport,
            NoOutputPublisher,
            NoCommitAgreement,
            residency,
            bounded,
        )
        .unwrap();
        let before = counters.snapshot();
        for _ in 0..2 {
            assert_eq!(
                <Strategy as ReplicatedTextExecutionStrategy<
                    OrdinaryTextFixture, FakeBackend, State, RecordingPolicy, RecordingPolicy,
                >>::group_submission_mechanism(&runtime),
                eredu_runtime::GroupSubmissionMechanism::PolicyOnly,
                "the manual executor's completion worker is independent of residency",
            );
            let resident = <Strategy as ReplicatedTextExecutionStrategy<
                OrdinaryTextFixture,
                FakeBackend,
                State,
                RecordingPolicy,
                RecordingPolicy,
            >>::resident_policy(&runtime);
            if residency == ExecutionResidency::FullyResident {
                let resident =
                    resident.expect("the selected resident executor must lend its policy");
                assert_eq!(
                    resident.units.as_ptr(),
                    original_units,
                    "the policy was cloned"
                );
                assert_eq!(resident.units[0].as_ref().unwrap().marker, 37);
            } else {
                assert!(
                    resident.is_none(),
                    "bounded ownership is not resident authority"
                );
            }
            let bounded = <Strategy as ReplicatedTextExecutionStrategy<
                OrdinaryTextFixture,
                FakeBackend,
                State,
                RecordingPolicy,
                RecordingPolicy,
            >>::bounded_policy(&runtime);
            assert_eq!(bounded.map(|value| value.units.as_ptr()), original_bounded);
        }
        assert_eq!(
            counters.snapshot(),
            before,
            "inspection performed execution or construction"
        );
    }
}
