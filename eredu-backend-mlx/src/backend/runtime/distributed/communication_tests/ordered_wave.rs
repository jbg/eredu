use super::*;

pub(super) const WORKER_RANK: &str = "EREDU_ORDERED_WORLD_WAVE_WORKER";

#[test]
fn worker() {
    let Some(rank) = std::env::var_os(WORKER_RANK) else {
        return;
    };
    let rank = rank.to_string_lossy().parse::<usize>().unwrap();
    let native = distributed::init(true, Backend::Ring).unwrap();
    assert_eq!((native.rank(), native.size()), (rank, 8));
    // Neither adjacent nor opposite Ring peers: these pairs use the packed
    // world-collective fallback, as TP=2/PP=2/EP=2 does.
    let first = (rank / 4) * 4 + rank % 2;
    let members = vec![first, first + 2];
    let id = CollectiveGroupId::new(200 + first as u32);
    let requirement = |operation| {
        CommunicationOperationRequirement::tensors(
            operation,
            [TensorDtype::F32],
            CommunicationTensorLimits::new(1, 1, 8, None).unwrap(),
            true,
        )
        .unwrap()
    };
    let manifest = CommunicationManifest::new(
        8,
        rank,
        vec![
            CommunicationGroupDescriptor::new(
                id,
                0,
                members.clone(),
                Some((rank % 4) / 2),
                CommunicationGroupRequirements::new([
                    requirement(CommunicationOperation::AllReduceSum),
                    requirement(CommunicationOperation::AllGatherEven),
                ])
                .unwrap(),
            )
            .unwrap(),
        ],
        vec![],
    )
    .unwrap()
    .with_completion_policy(completion_policy());
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let communication = ParallelCommunicators::from_manifest(&manifest, &native, &stream).unwrap();
    let group = communication.communication_group(id).unwrap();
    assert!(group.is_logical());
    let left = Array::from_slice(&[rank as f32 + 1.0, -(rank as f32) - 2.0], &[2]);
    let right = Array::from_slice(&[100.0 + rank as f32, 50.0 - rank as f32], &[2]);
    let sums = [
        super::super::group::all_sum(&left, group, &stream).unwrap(),
        super::super::group::all_sum(&right, group, &stream).unwrap(),
    ];
    // Consumers legitimately traverse independent branches in different orders.
    // Native matching must retain the common submission order on every rank.
    for index in if group.rank() == 0 { [1, 0] } else { [0, 1] } {
        sums[index].evaluated().unwrap();
    }
    let rank_sum = (2 * first + 2) as f32;
    assert_eq!(
        sums[0].evaluated().unwrap().as_slice::<f32>(),
        &[rank_sum + 2.0, -rank_sum - 4.0]
    );
    assert_eq!(
        sums[1].evaluated().unwrap().as_slice::<f32>(),
        &[200.0 + rank_sum, 100.0 - rank_sum]
    );

    let gathers = [
        super::super::group::all_gather(&left, group, &stream).unwrap(),
        super::super::group::all_gather(&right, group, &stream).unwrap(),
    ];
    for index in if group.rank() == 0 { [1, 0] } else { [0, 1] } {
        gathers[index].evaluated().unwrap();
    }
    let expected_left = members
        .iter()
        .flat_map(|rank| [*rank as f32 + 1.0, -(*rank as f32) - 2.0])
        .collect::<Vec<_>>();
    let expected_right = members
        .iter()
        .flat_map(|rank| [100.0 + *rank as f32, 50.0 - *rank as f32])
        .collect::<Vec<_>>();
    assert_eq!(
        gathers[0].evaluated().unwrap().as_slice::<f32>(),
        expected_left
    );
    assert_eq!(
        gathers[1].evaluated().unwrap().as_slice::<f32>(),
        expected_right
    );
}
