use eredu_backend_mlx::backend::{
    nn::shared::MlxNeuralBackend, runtime::checkpoint::store::MlxParameterMaterializationContext,
};
use eredu_checkpoint::store::SafetensorsWeightStore;
use eredu_runtime::{placement::load_rank_partition, PlacementPlan, PlacementRank};

use std::{path::Path, time::Duration};

use eredu_backend_mlx::backend::{DeviceAssignment, MlxRankContext};
use eredu_core::{
    consensus::BoundedConsensusTransport as _, BoundedCompletionWait, BoundedSubmissionOutcome,
    CompletionCancellationMode,
};
use eredu_runtime::TensorPlacement;
use safemlx::{
    distributed::{self, Backend},
    DeviceType, Stream,
};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

const WORKER_RANK: &str = "EREDU_PARTITION_RING_WORKER";
const BOUNDED_CONSENSUS_WORKER_RANK: &str = "EREDU_BOUNDED_CONSENSUS_RING_WORKER";
const CHECKPOINT_DIR: &str = "EREDU_PARTITION_CHECKPOINT";

#[test]
fn bounded_consensus_ring_worker() {
    let Some(rank) = std::env::var_os(BOUNDED_CONSENSUS_WORKER_RANK) else {
        return;
    };
    let rank: usize = rank.to_string_lossy().parse().unwrap();
    let world = distributed::init(true, Backend::Ring).unwrap();
    let stream = Stream::new_with_device(&safemlx::Device::new(DeviceType::Cpu, 0));
    let completion = eredu_core::communication::CommunicationCompletionPolicy::new(
        Duration::from_secs(2),
        CompletionCancellationMode::QuarantineUntilComplete,
    )
    .unwrap();
    let manifest = eredu_core::communication::CommunicationManifest::new(2, rank, vec![], vec![])
        .unwrap()
        .with_completion_policy(completion);
    let transport = eredu_backend_mlx::native::NativeWorldManifestTransport::new(&world, &stream);
    let prepared =
        eredu_runtime::communication::prepare_world_communication(transport, &manifest).unwrap();
    let session =
        eredu_backend_mlx::backend::distributed::MlxDistributedSession::from_prepared(prepared)
            .unwrap();
    let submission = session
        .submit_all_gather_words(&[rank as u32, 0x8000_0000 | rank as u32])
        .unwrap();
    let wait = BoundedCompletionWait::new(
        Duration::from_secs(2),
        CompletionCancellationMode::QuarantineUntilComplete,
    )
    .unwrap();
    let output = match submission.wait_bounded(wait).unwrap() {
        BoundedSubmissionOutcome::Completed(output) => output,
        BoundedSubmissionOutcome::DeadlineExceeded { cancellation } => {
            panic!("bounded Ring consensus timed out with {cancellation:?}")
        }
    };
    assert_eq!(
        session.resolve_all_gather_words(output).unwrap(),
        [0, 0x8000_0000, 1, 0x8000_0001]
    );
}

#[test]
fn partition_ring_worker() {
    let Some(rank) = std::env::var_os(WORKER_RANK) else {
        return;
    };
    let expected_rank: usize = rank.to_string_lossy().parse().unwrap();
    let checkpoint = std::env::var_os(CHECKPOINT_DIR).unwrap();
    let group = distributed::init(true, Backend::Ring).unwrap();
    let topology = eredu_core::ParallelRankTopology::new(
        eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap(),
        expected_rank,
    )
    .unwrap();
    assert_eq!(topology.global_rank(), expected_rank);

    let device = DeviceAssignment::new(DeviceType::Cpu, 0);
    let stream = Stream::new_with_device(&device.device().unwrap());
    let rank_context = MlxRankContext::new(topology.world_size(), expected_rank, device).unwrap();
    let mut plan = PlacementPlan::new(
        PlacementRank::new(rank_context.world_size(), rank_context.global_rank()).unwrap(),
    );
    plan.insert_expected(
        "model.projection.weight",
        vec![2, 4],
        TensorPlacement::Shard {
            axis: 1,
            index: topology.tensor_parallel_rank(),
            parts: topology.tensor_parallel_size(),
        },
    )
    .unwrap();
    plan.insert("model.remote.weight", TensorPlacement::Omit);
    let source = SafetensorsWeightStore::open(Path::new(&checkpoint)).unwrap();
    let context = MlxParameterMaterializationContext::new(&stream, &stream);
    let partition = load_rank_partition::<MlxNeuralBackend, _>(&source, &plan, |lease| {
        context
            .weight_lease(lease)
            .and_then(|value| value.materialize(&stream, &stream))
            .and_then(|value| value.synchronize())
            .map_err(eredu_core::BackendFailure::from_error)
    })
    .unwrap();
    assert_eq!(partition.len(), 1);
    assert_eq!(partition.opened_shards().len(), 1);
    assert_eq!(
        partition.opened_shards()[0].file_name().unwrap(),
        "local.safetensors"
    );

    let local = partition.get("model.projection.weight").unwrap();
    assert_eq!(local.shape(), &[2, 2]);
    let gathered = distributed::all_gather(local, &group, &stream).unwrap();
    let gathered = gathered.evaluated().unwrap();
    assert_eq!(gathered.as_array().shape(), &[4, 2]);
    assert_eq!(gathered.as_slice::<i32>(), &[0, 1, 10, 11, 2, 3, 12, 13]);
}

#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_partition_load() {
    let checkpoint = tempfile::tempdir().unwrap();
    let bytes = [0i32, 1, 2, 3, 10, 11, 12, 13]
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect::<Vec<_>>();
    let view = TensorView::new(Dtype::I32, vec![2, 4], &bytes).unwrap();
    serialize_to_file(
        [("model.projection.weight", view)],
        None,
        &checkpoint.path().join("local.safetensors"),
    )
    .unwrap();
    std::fs::write(
        checkpoint.path().join("remote.safetensors"),
        b"this remote-only shard must not be opened",
    )
    .unwrap();
    std::fs::write(
        checkpoint.path().join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({
            "metadata": {},
            "weight_map": {
                "model.projection.weight": "local.safetensors",
                "model.remote.weight": "remote.safetensors"
            }
        }))
        .unwrap(),
    )
    .unwrap();

    super::workers::run(
        "partition::partition_ring_worker",
        Duration::from_secs(30),
        |command, rank| {
            command
                .env(WORKER_RANK, rank.to_string())
                .env(CHECKPOINT_DIR, checkpoint.path());
        },
    );
}

#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_two_process_bounded_consensus_transport() {
    super::workers::run(
        "partition::bounded_consensus_ring_worker",
        Duration::from_secs(30),
        |command, rank| {
            command.env(BOUNDED_CONSENSUS_WORKER_RANK, rank.to_string());
        },
    );
}
