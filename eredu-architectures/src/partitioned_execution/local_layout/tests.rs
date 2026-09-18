use super::*;
use eredu_runtime::{
    ArchitectureParameterDescription, ExecutionGraph, ExecutionGroupSpec, ExecutionUnitLayout,
    MemberSharding, OwnedParameterGroupSpec, ParameterGroupOwner, ParameterGroupSpec,
    ParameterMemberSpec, ParameterRole,
};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};
#[derive(Debug)]
struct State {
    calls: AtomicUsize,
    stop: AtomicUsize,
    retired: AtomicBool,
}
#[derive(Debug)]
struct Account(Arc<State>);
impl eredu_core::HostMetadataAccount for Account {
    fn reserve_metadata(&self, _: usize) -> Result<(), HostMetadataFundingError> {
        let call = self.0.calls.fetch_add(1, Ordering::SeqCst);
        let stop = self.0.stop.load(Ordering::SeqCst);
        assert!(call <= stop, "producer reached after refusal");
        if call == stop {
            Err(HostMetadataFundingError::Unavailable)
        } else {
            Ok(())
        }
    }
}
impl Drop for Account {
    fn drop(&mut self) {
        self.0.retired.store(true, Ordering::SeqCst);
    }
}
fn account(stop: usize) -> (HostMetadataFunding, Arc<State>) {
    let state = Arc::new(State {
        calls: AtomicUsize::new(0),
        stop: AtomicUsize::new(usize::MAX),
        retired: AtomicBool::new(false),
    });
    let funding = HostMetadataFunding::new(Account(state.clone())).unwrap();
    state.calls.store(0, Ordering::SeqCst);
    state.stop.store(stop, Ordering::SeqCst);
    (funding, state)
}
fn fixture() -> ArchitectureParameterDescription {
    let graph = ExecutionGraph::new(vec![ExecutionGroupSpec::root("decoder")], "decoder").unwrap();
    let units = ExecutionUnitLayout::new(&graph, [1]).unwrap();
    let kinds = [
        (vec![8, 12], MemberSharding::Replicated),
        (vec![8, 12], MemberSharding::Equal { axis: 1 }),
        (vec![9, 12], MemberSharding::Balanced { axis: 0 }),
        (vec![8, 12], MemberSharding::Partitioned { axis: 1 }),
        (
            vec![8, 11],
            MemberSharding::PartitionedChunks {
                axis: 1,
                chunk_size: 4,
            },
        ),
        (
            vec![8, 22],
            MemberSharding::PartitionedChunkSegments {
                axis: 1,
                chunk_size: 4,
                segments: vec![0..11, 11..22],
            },
        ),
        (
            vec![8, 24],
            MemberSharding::PartitionedSegments {
                axis: 1,
                segments: vec![0..12, 12..24],
            },
        ),
        (
            vec![8, 13],
            MemberSharding::Segmented {
                axis: 1,
                segments: vec![0..5, 5..13],
            },
        ),
    ];
    let mut groups: Vec<_> = kinds
        .into_iter()
        .enumerate()
        .map(|(index, (shape, sharding))| {
            let partitioned = matches!(
                &sharding,
                MemberSharding::Partitioned { .. }
                    | MemberSharding::PartitionedChunks { .. }
                    | MemberSharding::PartitionedSegments { .. }
                    | MemberSharding::PartitionedChunkSegments { .. }
            );
            let member = ParameterMemberSpec::new(format!("weight-{index}"), shape, sharding);
            if partitioned {
                ParameterGroupSpec::partitioned(
                    format!("group-{index}"),
                    ParameterRole::RowProjection,
                    3,
                    [member],
                )
                .unwrap()
            } else {
                ParameterGroupSpec::new(
                    format!("group-{index}"),
                    ParameterRole::RowProjection,
                    [member],
                )
                .unwrap()
            }
        })
        .collect();
    groups.push(
        ParameterGroupSpec::partitioned(
            "experts",
            ParameterRole::ExpertIntermediate,
            3,
            [ParameterMemberSpec::new(
                "expert.weight",
                vec![4, 8, 12],
                MemberSharding::Partitioned { axis: 2 },
            )],
        )
        .unwrap(),
    );
    let owned = groups
        .iter()
        .cloned()
        .map(|group| {
            OwnedParameterGroupSpec::new(ParameterGroupOwner::static_role("weights"), group)
        })
        .collect::<Vec<_>>();
    ArchitectureParameterDescription::new(&graph, &units, groups, owned).unwrap()
}
#[test]
fn funded_layout_preserves_every_sharding_worker_and_retains_actual_account() {
    let source = fixture();
    let topology = eredu_core::ParallelTopology::new(2, 1, 2, 1).unwrap();
    for rank in 0..topology.world_size() {
        let topology = eredu_core::ParallelRankTopology::new(topology, rank).unwrap();
        let expected = super::super::derive_partitioned_local_layout(&source, topology).unwrap();
        let (funding, state) = account(usize::MAX);
        let actual =
            derive_partitioned_local_layout_with_funding(&source, topology, funding).unwrap();
        assert_eq!(actual.layout(), &expected);
        let expert = actual.layout().tensor("expert.weight").unwrap();
        assert_eq!(expert.local_shape()[0], 2);
        assert_eq!(expert.additional_placements().len(), 1);
        let mut names = actual.layout().tensors().map(|(name, _)| name);
        let mut prior = names.next().unwrap();
        for next in names {
            assert!(prior < next);
            prior = next;
        }
        assert!(!state.retired.load(Ordering::SeqCst));
        drop(actual);
        assert!(state.retired.load(Ordering::SeqCst));
    }
}
#[test]
fn every_layout_reservation_refuses_before_later_producers_and_keeps_error_custody() {
    let source = fixture();
    let topology = eredu_core::ParallelRankTopology::new(
        eredu_core::ParallelTopology::new(2, 1, 2, 1).unwrap(),
        1,
    )
    .unwrap();
    let (funding, state) = account(usize::MAX);
    drop(derive_partitioned_local_layout_with_funding(&source, topology, funding).unwrap());
    let calls = state.calls.load(Ordering::SeqCst);
    assert!(calls > 80);
    for stop in 0..calls {
        let (funding, state) = account(stop);
        let error =
            derive_partitioned_local_layout_with_funding(&source, topology, funding).unwrap_err();
        assert!(
            matches!(
                error.cause,
                Cause::Funding(HostMetadataFundingError::Unavailable)
            ),
            "cut {stop}"
        );
        assert_eq!(state.calls.load(Ordering::SeqCst), stop + 1);
        let cause = std::error::Error::source(&error).unwrap();
        assert!(cause.source().unwrap().is::<HostMetadataFundingError>());
        assert!(!state.retired.load(Ordering::SeqCst));
        drop(error);
        assert!(state.retired.load(Ordering::SeqCst));
    }
}
#[test]
fn malformed_layout_uses_same_diagnostic_and_refuses_before_its_text() {
    let source = fixture();
    let expected = local_layout(&source, 0, 1, 0, 0).unwrap_err();
    let (funding, state) = account(usize::MAX);
    let actual = local_layout_worker(&source, 0, 1, 0, 0, Allocation(Some(&funding))).unwrap_err();
    assert_eq!(actual.to_string(), expected);
    for stop in 0..state.calls.load(Ordering::SeqCst) {
        let (funding, state) = account(stop);
        assert!(matches!(
            local_layout_worker(&source, 0, 1, 0, 0, Allocation(Some(&funding))),
            Err(Cause::Funding(_))
        ));
        assert_eq!(state.calls.load(Ordering::SeqCst), stop + 1);
    }
}
