use super::*;
use eredu_core::{HostMetadataAccount, HostMetadataFunding, HostMetadataFundingError};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
#[derive(Debug)]
struct Account {
    calls: Arc<AtomicUsize>,
    stop: usize,
}
impl HostMetadataAccount for Account {
    fn reserve_metadata(&self, _: usize) -> Result<(), HostMetadataFundingError> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        assert!(call <= self.stop, "producer after refusal");
        if call == self.stop {
            Err(HostMetadataFundingError::Unavailable)
        } else {
            Ok(())
        }
    }
}
fn account(stop: usize) -> (HostMetadataFunding, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let funding = HostMetadataFunding::new(Account {
        calls: calls.clone(),
        stop: stop + 1,
    })
    .unwrap();
    (funding, calls)
}
fn fixture(duplicate: bool) -> eredu_runtime::ReplicatedTextRequirements {
    use eredu_runtime::*;
    let graph = ExecutionGraph::new(
        vec![
            ExecutionGroupSpec::root("encoder"),
            ExecutionGroupSpec::with_dependencies("decoder", ["encoder"]),
        ],
        "decoder",
    )
    .unwrap();
    let units = ExecutionUnitLayout::new(&graph, [5, 2]).unwrap();
    let transport = |kind, first: Vec<String>, last: Vec<String>| ArchitectureGroupTransport {
        placement: ArchitectureGroupPlacement::Pipeline,
        kind,
        first_owner_static_roles: first,
        last_owner_static_roles: last,
        merge_destination: ArchitectureMergeDestination::LastOwner,
        parallel_subgroup: None,
        request_optional: false,
    };
    let mut first = vec!["embed".into(), "shared".into()];
    if duplicate {
        first.push("embed".into());
    }
    ReplicatedTextRequirements::new(
        "fixture.partition-source",
        eredu_nn::NeuralOperatorCapabilities::EXP,
        graph,
        units,
        vec![
            transport(
                ArchitectureGroupKind::VisionEncoder,
                first,
                vec!["shared".into(), "project".into()],
            ),
            transport(
                ArchitectureGroupKind::Decoder,
                vec!["decoder-input".into()],
                vec!["readout".into()],
            ),
        ],
        StateLayout::new(
            eredu_core::LayerSchedule::new(
                2,
                vec![
                    eredu_core::cache::LayerCachePolicy::key_value(
                        eredu_core::AttentionPolicy::Full,
                        1,
                        8,
                    )
                    .unwrap(); 2
                ],
            )
            .unwrap(),
        )
        .unwrap(),
        ReplicatedTextStateAccess::KeyValue,
        vec![],
    )
    .unwrap()
}
#[test]
fn rank_source_keeps_pipeline_ownership_and_same_state_equations() {
    let execution = fixture(false);
    for pp in [1, 2, 4, 8] {
        let topology = eredu_core::ParallelTopology::new(2, pp, 1, 1).unwrap();
        for global in 0..topology.world_size() {
            let topology = ParallelRankTopology::new(topology, global).unwrap();
            let expected = rank_requirements_inner(&execution, None, topology, false).unwrap();
            let (funding, _) = account(usize::MAX - 1);
            let actual = source(&execution, topology, Allocation(Some(&funding))).unwrap();
            assert_eq!(actual.groups, expected.groups);
            assert_eq!(actual.ownership, expected.ownership);
            assert_eq!(actual.publication_owner, expected.publication_owner);
            assert!(actual.state.is_none());
            assert!(actual.parameter_targets.is_empty());
            let with_state =
                rank_requirements(&execution, Some(execution.state_layout()), topology).unwrap();
            assert_eq!(with_state.groups, actual.groups);
            if pp == 1 {
                assert_eq!(
                    actual.ownership.static_roles(),
                    ["embed", "shared", "project", "decoder-input", "readout"]
                );
                assert!(actual.ownership.owns_input());
                assert!(actual.ownership.owns_output());
                assert_eq!(
                    actual.groups.iter().map(|g| g.units()).collect::<Vec<_>>(),
                    [0..5, 0..2]
                );
                assert!(with_state.state.is_some());
            }
            if pp == 2 && topology.pipeline_parallel_rank() == 0 {
                assert_eq!(
                    actual.groups.iter().map(|g| g.units()).collect::<Vec<_>>(),
                    [0..3, 0..1]
                );
                assert!(actual.ownership.owns_input());
                assert!(!actual.ownership.owns_output());
            }
        }
    }
}
#[test]
fn every_rank_source_destination_refuses_before_the_next_producer() {
    let topology =
        ParallelRankTopology::new(eredu_core::ParallelTopology::new(1, 1, 1, 1).unwrap(), 0)
            .unwrap();
    for duplicate in [false, true] {
        let execution = fixture(duplicate);
        let (funding, calls) = account(usize::MAX - 1);
        let result = source(&execution, topology, Allocation(Some(&funding)));
        if duplicate {
            assert_eq!(
                result.err().unwrap().to_string(),
                rank_requirements(&execution, None, topology).err().unwrap()
            );
        } else {
            assert!(result.is_ok());
        }
        let count = calls.load(Ordering::SeqCst) - 1;
        assert!(count > 30);
        for stop in 0..count {
            let (funding, calls) = account(stop);
            assert!(
                matches!(
                    source(&execution, topology, Allocation(Some(&funding))),
                    Err(Cause::Funding(HostMetadataFundingError::Unavailable))
                ),
                "cut {stop}"
            );
            assert_eq!(calls.load(Ordering::SeqCst), stop + 2);
        }
    }
}
