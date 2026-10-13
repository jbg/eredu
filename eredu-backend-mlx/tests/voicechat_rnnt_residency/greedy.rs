//! Shared greedy policy over a transported predictor and independent joint providers.
use super::*;
use eredu_architectures::voicechat::{
    rnnt::RnntProjectionExecution,
    rnnt_pipeline::{decode_frame, RnntPipelineCursor},
    rnnt_projection_preparation::PreparedRnntProjectionBindings,
};
use eredu_backend_mlx::{
    backend::nn::shared::MlxNeuralBackend as Backend,
    native::{
        materialize_voicechat_rnnt_projection, materialize_voicechat_rnnt_projection_parallel,
        MlxReplicatedCoordinator, MlxRnntPredictorPipeline, MlxRnntProjection,
    },
};
use eredu_core::{
    checkpoint::TensorDtype, CollectiveGroupId, CompletionCancellationMode, DistributedCommitEpoch,
};
use eredu_runtime::{
    CommunicationCompletionPolicy, CommunicationGroupDescriptor, CommunicationGroupRequirements,
    CommunicationManifest, CommunicationOperation, CommunicationOperationRequirement,
    CommunicationTensorLimits,
};

pub(super) fn coordinator(
    world: &safemlx::distributed::Group,
    stream: &Stream,
    cohort: bool,
) -> MlxReplicatedCoordinator {
    let (manifest, id, owner) = coordination_manifest(world, cohort);
    MlxReplicatedCoordinator::from_manifest(manifest, id, owner, world, stream).unwrap()
}

pub(super) fn coordination_manifest(
    world: &safemlx::distributed::Group,
    cohort: bool,
) -> (CommunicationManifest, CollectiveGroupId, usize) {
    let degree = world.size() / 2;
    let stage = world.rank() / degree;
    let members: Vec<_> = if cohort {
        (stage * degree..(stage + 1) * degree).collect()
    } else {
        (0..world.size()).collect()
    };
    let owner = *members.last().unwrap();
    let id = CollectiveGroupId::new(if cohort { 80 + stage as u32 } else { 79 });
    let mut requirements = vec![
        CommunicationOperationRequirement::failure_agreement(true),
        CommunicationOperationRequirement::tensors(
            CommunicationOperation::Broadcast,
            [TensorDtype::I32],
            CommunicationTensorLimits::new(1, 1, 1, None).unwrap(),
            true,
        )
        .unwrap(),
    ];
    if cohort {
        for op in [
            CommunicationOperation::AllReduceSum,
            CommunicationOperation::AllGatherUneven,
        ] {
            requirements.push(
                CommunicationOperationRequirement::tensors(
                    op,
                    [TensorDtype::F32],
                    CommunicationTensorLimits::new(1, 2, 16, None).unwrap(),
                    true,
                )
                .unwrap(),
            );
        }
    }
    let local = members
        .iter()
        .position(|rank| *rank == world.rank())
        .unwrap();
    let group = CommunicationGroupDescriptor::new(
        id,
        0,
        members,
        Some(local),
        CommunicationGroupRequirements::new(requirements).unwrap(),
    )
    .unwrap();
    let manifest = CommunicationManifest::new(world.size(), world.rank(), vec![group], vec![])
        .unwrap()
        .with_completion_policy(
            CommunicationCompletionPolicy::new(
                std::time::Duration::from_secs(30),
                CompletionCancellationMode::QuarantineUntilComplete,
            )
            .unwrap(),
        );
    (manifest, id, owner)
}
pub(super) fn load(
    source: SharedCheckpointSource,
    fixture: &Value,
    mode: u64,
    authority: Option<&MlxReplicatedCoordinator>,
    weights: &Stream,
    stream: &Stream,
) -> MlxRnntProjection {
    let dim = dimensions(fixture);
    let prepared = PreparedRnntProjectionBindings::from_rnnt(prepared(dim, source)).unwrap();
    assert!(
        prepared
            .source()
            .source_keys()
            .iter()
            .all(|key| !key.contains("dec_rnn")),
        "joint providers never retain recurrent checkpoint authority"
    );
    let residency = match mode {
        0 => LayerWeightResidency::FullyResident,
        1 => LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
            OffloadConfig::new(Some(64), Some(1 << 20), 1).unwrap(),
        )),
        _ => LayerWeightResidency::DenseDiskStream(
            DenseDiskStreamLoadOptions::new(64, 0, 0, 0).unwrap(),
        ),
    };
    let (model, metadata) = match authority {
        Some(authority) => materialize_voicechat_rnnt_projection_parallel(
            prepared, authority, residency, weights, stream,
        ),
        None => materialize_voicechat_rnnt_projection(prepared, residency, weights, stream),
    }
    .unwrap();
    // Vocabulary rows shard; both joint input affines remain replicated.
    let rows = (dim.vocab_size + 1) as u64 / if authority.is_some() { 2 } else { 1 };
    let vocabulary_bytes = rows * (dim.predictor_width + dim.joint_width + 1) as u64 * 4;
    let affine_bytes =
        dim.joint_width as u64 * (dim.encoder_width + dim.predictor_width + 2) as u64 * 4;
    assert_eq!(
        metadata.layer_parameter_bytes(),
        vocabulary_bytes + affine_bytes
    );
    assert_eq!(metadata.static_device_bytes(), 0);
    model
}

#[allow(clippy::too_many_arguments)]
pub(super) fn exercise(
    predictor: &mut MlxRnntPredictorPipeline,
    directory: &std::path::Path,
    source: SharedCheckpointSource,
    fixture: &Value,
    mode: u64,
    world: &safemlx::distributed::Group,
    weights: &Stream,
    stream: &Stream,
) {
    let cohort = (world.size() > 2).then(|| coordinator(world, stream, true));
    let mut decision = coordinator(world, stream, false);
    let mut projections = load(
        source.clone(),
        fixture,
        mode,
        cohort.as_ref(),
        weights,
        stream,
    );
    let mut cursor = RnntPipelineCursor::new(0, DistributedCommitEpoch::new(100).unwrap(), 1000);
    let limits = RnntLimits {
        max_symbols_per_frame: 3,
        hypothesis_capacity: 32,
    };
    let mut state = RnntState::default();
    let mut hypothesis = Vec::new();
    for (index, frame) in fixture["streaming_frames"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let input = tensor(&frame["encoder"], dimensions(fixture).encoder_width);
        let before = retained(&state, stream);
        let candidate = decode_frame::<Backend, _, _, _>(
            &mut projections,
            predictor,
            &mut decision,
            [301, mode, index as u64],
            &mut cursor,
            &input,
            &state,
            limits,
            stream,
        )
        .unwrap();
        assert_eq!(retained(&state, stream), before);
        let expected: Vec<_> = frame["emitted"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(candidate.emitted, expected);
        assert_eq!(
            candidate.first_blank,
            frame["first_blank"].as_bool().unwrap()
        );
        assert_eq!(
            candidate.ended_blank,
            frame["ended_blank"].as_bool().unwrap()
        );
        assert_eq!(
            candidate.hit_symbol_limit,
            frame["hit_symbol_limit"].as_bool().unwrap()
        );
        pipeline::compare_prediction(candidate.state.predictor().unwrap(), frame, fixture, stream);
        hypothesis.extend_from_slice(&candidate.emitted);
        assert_eq!(candidate.state.hypothesis(), hypothesis);
        assert_eq!(candidate.state.frames(), index as u64 + 1);
        let prior_cursor = cursor.next_step();
        let replay = decode_frame::<Backend, _, _, _>(
            &mut projections,
            predictor,
            &mut decision,
            [302, mode, index as u64],
            &mut cursor,
            &input,
            &state,
            limits,
            stream,
        )
        .unwrap();
        assert!(
            cursor.next_step() > prior_cursor,
            "replay uses fresh identities"
        );
        assert_eq!(candidate.emitted, replay.emitted);
        assert_eq!(
            retained(&candidate.state, stream),
            retained(&replay.state, stream)
        );
        state = candidate.state;
    }
    // Capacity failure after an emitted symbol must preserve local parent carry;
    // accepted reservations remain consumed before a successful retry.
    let input = tensor(
        &fixture["streaming_frames"][1]["encoder"],
        dimensions(fixture).encoder_width,
    );
    let before = retained(&state, stream);
    let before_cursor = cursor.next_step();
    assert!(decode_frame::<Backend, _, _, _>(
        &mut projections,
        predictor,
        &mut decision,
        [303, mode, 0],
        &mut cursor,
        &input,
        &state,
        RnntLimits {
            hypothesis_capacity: state.hypothesis().len() + 1,
            ..limits
        },
        stream
    )
    .is_err());
    assert!(cursor.next_step() > before_cursor);
    assert_eq!(retained(&state, stream), before);
    decode_frame::<Backend, _, _, _>(
        &mut projections,
        predictor,
        &mut decision,
        [304, mode, 0],
        &mut cursor,
        &input,
        &state,
        limits,
        stream,
    )
    .unwrap();
    let reads = source.source_diagnostics().unwrap().physical_reads;
    let mut exhausted = RnntPipelineCursor::new(cursor.next_step(), cursor.next_epoch(), 0);
    assert!(decode_frame::<Backend, _, _, _>(
        &mut projections,
        predictor,
        &mut decision,
        [305, mode, 0],
        &mut exhausted,
        &input,
        &state,
        limits,
        stream
    )
    .is_err());
    assert_eq!(source.source_diagnostics().unwrap().physical_reads, reads);
    assert_eq!(retained(&state, stream), before);

    if cohort.is_some() {
        // One peer fails architecture ingress before any lease. Every cohort
        // participant exits the same traversal phase and remains usable.
        let reads = source.source_diagnostics().unwrap().physical_reads;
        let malformed = MlxTensor::from_array(Array::from_slice(&[1f32], &[1, 1]));
        let valid = MlxTensor::from_array(Array::from_slice(&[1f32, 2., 3.], &[1, 3]));
        assert!(projections
            .project_output(
                if world.rank().is_multiple_of(2) {
                    &malformed
                } else {
                    &valid
                },
                stream
            )
            .is_err());
        assert_eq!(source.source_diagnostics().unwrap().physical_reads, reads);
        projections.project_output(&valid, stream).unwrap();
    }
    if mode == 2 && cohort.is_some() {
        // Mutate only rank zero's already-admitted file after constructing a
        // fresh disk provider. Its first learned-unit acquisition must fail
        // before a peer can enter the vocabulary collective.
        let path = directory.join("greedy-acquisition-failure.safetensors");
        let fault_source = checkpoint(fixture, false, &path);
        let mut fault_projection = load(
            fault_source,
            fixture,
            mode,
            cohort.as_ref(),
            weights,
            stream,
        );
        if world.rank() == 0 {
            std::fs::write(&path, b"invalidated admitted source").unwrap();
        }
        let predictor_reads = predictor
            .residency_report()
            .unwrap()
            .weight_store()
            .physical_reads;
        let empty = RnntState::default();
        let previous_step = cursor.next_step();
        assert!(decode_frame::<Backend, _, _, _>(
            &mut fault_projection,
            predictor,
            &mut decision,
            [306, mode, 0],
            &mut cursor,
            &input,
            &empty,
            limits,
            stream,
        )
        .is_err());
        assert!(cursor.next_step() > previous_step);
        assert!(retained(&empty, stream).is_empty());
        assert_eq!(
            predictor
                .residency_report()
                .unwrap()
                .weight_store()
                .physical_reads,
            predictor_reads
        );
        // Ordinary local failure does not poison the communication authority.
        // A healthy exact-source provider can retry the same immutable parent
        // under fresh caller-reserved identities.
        decode_frame::<Backend, _, _, _>(
            &mut projections,
            predictor,
            &mut decision,
            [307, mode, 0],
            &mut cursor,
            &input,
            &empty,
            limits,
            stream,
        )
        .unwrap();
    }
    if mode > 0 {
        assert!(
            projections
                .residency_report()
                .unwrap()
                .offload()
                .peak_resident_bytes()
                .get(MemoryTier::Device)
                <= 64
        );
    }
}

#[test]
#[ignore = "spawns two/four local MLX Ring ranks and opens loopback sockets; run explicitly"]
fn native_rnnt_greedy_pp2_and_tp2_pp2_match_pinned_frames_and_replay() {
    pipeline::launch(2, "pipeline::native_rnnt_greedy_pipeline_worker");
    pipeline::launch(4, "pipeline::native_rnnt_greedy_pipeline_worker");
}
