//! Codes, local neural carry and replicated spectral overlap share one gated transition.
use super::*;
use eredu_backend_mlx::native::{
    materialize_prvq_parallel, MlxEarDecoderPipeline, MlxEarDecoderPipelineState, MlxPrvq,
    MlxReplicatedCoordinator,
};
use eredu_codec::{ear::AudioDecoderKernel, ear_pipeline::PipelineAudioDecoder};
use eredu_core::{
    checkpoint::TensorDtype, CollectiveGroupId, CompletionCancellationMode, DistributedCommitEpoch,
};
use eredu_runtime::{
    CommunicationCompletionPolicy, CommunicationGroupDescriptor, CommunicationGroupRequirements,
    CommunicationManifest, CommunicationOperation, CommunicationOperationRequirement,
    CommunicationTensorLimits,
};

type State = AudioDecoderState<MlxTensor, MlxEarDecoderPipelineState>;
fn retained_local(state: &State, context: &Stream) -> Vec<(Vec<i32>, Vec<f32>)> {
    let mut tensors = Vec::new();
    state.visit_retained_tensors(&mut |tensor| {
        tensors.push((tensor.shape().to_vec(), tensor.to_f32_vec(context).unwrap()))
    });
    tensors
}
fn snapshot(state: &State, context: &Stream) -> State {
    use eredu_core::execution_control::{SnapshotLimits, SnapshotResourceKind};
    use eredu_runtime::execution_control::{SnapshotBudget, SnapshotValue};
    let budget = SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 1,
        max_branches: 0,
        retained_bytes: 1 << 20,
        cumulative_copy_bytes: 1 << 20,
    });
    let reservation = budget
        .reserve(
            SnapshotResourceKind::Snapshot,
            Some(state.estimate_snapshot_value().unwrap()),
        )
        .unwrap();
    let copied = state.copy_snapshot_value(&reservation).unwrap();
    assert_eq!(
        retained_local(&copied, context),
        retained_local(state, context)
    );
    assert_eq!(
        copied.neural().global_units(),
        state.neural().global_units()
    );
    assert_eq!(copied.frames_seen(), state.frames_seen());
    assert_eq!(copied.samples_emitted(), state.samples_emitted());
    copied
}
fn coordinator(
    world: &safemlx::distributed::Group,
    stream: &Stream,
    cohort: bool,
) -> MlxReplicatedCoordinator {
    let degree = world.size() / 2;
    let stage = world.rank() / degree;
    let members: Vec<_> = if cohort {
        (stage * degree..(stage + 1) * degree).collect()
    } else {
        (0..world.size()).collect()
    };
    let owner = *members.last().unwrap();
    let group = CollectiveGroupId::new(if cohort { 92 + stage as u32 } else { 91 });
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
        requirements.push(
            CommunicationOperationRequirement::tensors(
                CommunicationOperation::AllReduceSum,
                [TensorDtype::F32],
                CommunicationTensorLimits::new(1, 3, 8192, None).unwrap(),
                true,
            )
            .unwrap(),
        );
    }
    let local = members
        .iter()
        .position(|rank| *rank == world.rank())
        .unwrap();
    let descriptor = CommunicationGroupDescriptor::new(
        group,
        0,
        members,
        Some(local),
        CommunicationGroupRequirements::new(requirements).unwrap(),
    )
    .unwrap();
    let manifest = CommunicationManifest::new(world.size(), world.rank(), vec![descriptor], vec![])
        .unwrap()
        .with_completion_policy(
            CommunicationCompletionPolicy::new(
                std::time::Duration::from_secs(30),
                CompletionCancellationMode::QuarantineUntilComplete,
            )
            .unwrap(),
        );
    MlxReplicatedCoordinator::from_manifest(manifest, group, owner, world, stream).unwrap()
}
fn quantizer_bindings(
    config: Geometry,
    source: SharedCheckpointSource,
) -> PreparedQuantizerBindings {
    let recipes = source
        .source_keys()
        .into_iter()
        .map(|name| {
            (
                name.clone(),
                DerivedWeightRecipe::Source {
                    key: name,
                    selection: TensorSelection::Full,
                },
            )
        })
        .collect();
    PreparedQuantizerBindings::from_recipes(config, "tts_model.audio_codec.prvq", source, &recipes)
        .unwrap()
}
fn step(
    kernel: &AudioDecoderKernel<Backend>,
    neural: &mut MlxEarDecoderPipeline,
    quantizer: &mut MlxPrvq,
    coordinator: &mut MlxReplicatedCoordinator,
    codes: &MlxTensor,
    parent: &State,
    mode: u64,
    epoch: u64,
    stream: &Stream,
) -> Result<(MlxTensor, State), eredu_codec::Error> {
    PipelineAudioDecoder::new(
        kernel,
        neural,
        quantizer,
        coordinator,
        [704, mode, epoch],
        DistributedCommitEpoch::new(epoch).unwrap(),
    )
    .step(codes, parent, true, stream)
}
fn finish(
    kernel: &AudioDecoderKernel<Backend>,
    neural: &mut MlxEarDecoderPipeline,
    quantizer: &mut MlxPrvq,
    coordinator: &mut MlxReplicatedCoordinator,
    parent: &State,
    mode: u64,
    epoch: u64,
    stream: &Stream,
) -> Result<(Option<MlxTensor>, State), eredu_codec::Error> {
    PipelineAudioDecoder::new(
        kernel,
        neural,
        quantizer,
        coordinator,
        [704, mode, epoch],
        DistributedCommitEpoch::new(epoch).unwrap(),
    )
    .finish(parent, true, stream)
}
fn compare_tensor(actual: &MlxTensor, expected: &MlxTensor, fixture: &Value, stream: &Stream) {
    assert_eq!(actual.shape(), expected.shape());
    let absolute = fixture["comparison"]["absolute_tolerance"]
        .as_f64()
        .unwrap() as f32;
    let relative = fixture["comparison"]["relative_tolerance"]
        .as_f64()
        .unwrap() as f32;
    for (actual, expected) in actual
        .to_f32_vec(stream)
        .unwrap()
        .into_iter()
        .zip(expected.to_f32_vec(stream).unwrap())
    {
        assert!(
            (actual - expected).abs() <= absolute + relative * expected.abs(),
            "pipeline {actual} differs from ordinary same-checkpoint {expected}"
        );
    }
}

pub(super) fn exercise(
    neural: &mut MlxEarDecoderPipeline,
    source: SharedCheckpointSource,
    case: &Value,
    fixture: &Value,
    mode: u64,
    world: &safemlx::distributed::Group,
    weights: &Stream,
    stream: &Stream,
    neural_budget: u64,
) {
    let config = AudioDecoderConfig {
        quantizer: Geometry {
            depth: 3,
            cardinality: 5,
            channels: 4,
        },
        neural: geometry(&case["geometry"]),
        spectral: spectral(&case["geometry"]),
    };
    let quantizer_prepared = quantizer_bindings(config.quantizer, source.clone());
    assert_eq!(quantizer_prepared.source().source_keys().len(), 6);
    assert!(quantizer_prepared
        .source()
        .source_keys()
        .iter()
        .all(|name| name.starts_with("tts_model.audio_codec.prvq.")));
    let sharded = world.size() == 4 && mode == 2;
    let cohort = sharded.then(|| coordinator(world, stream, true));
    // Channel shards each retain their scalar variance replica.
    let quantizer_budget = if sharded {
        5 * 2 * 4 + 4
    } else {
        5 * 4 * 4 + 4
    };
    let quantizer_residency = match mode {
        0 => LayerWeightResidency::FullyResident,
        1 => LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
            OffloadConfig::new(Some(quantizer_budget), Some(1 << 20), 1).unwrap(),
        )),
        _ => LayerWeightResidency::DenseDiskStream(
            DenseDiskStreamLoadOptions::new(quantizer_budget, 0, 0, 0).unwrap(),
        ),
    };
    let (mut quantizer, metadata) = match cohort.as_ref() {
        Some(cohort) => materialize_prvq_parallel(
            quantizer_prepared,
            cohort,
            quantizer_residency,
            weights,
            stream,
        ),
        None => materialize_prvq(quantizer_prepared, quantizer_residency, weights, stream),
    }
    .unwrap();
    assert_eq!(metadata.layer_parameter_bytes(), 3 * quantizer_budget);
    assert_eq!(metadata.static_device_bytes(), 0);
    let (ordinary_neural, _) = materialize_ear_decoder(
        prepared(config.neural.clone(), source.clone()),
        LayerWeightResidency::FullyResident,
        weights,
        stream,
    )
    .unwrap();
    let (ordinary_quantizer, _) = materialize_prvq(
        quantizer_bindings(config.quantizer, source.clone()),
        LayerWeightResidency::FullyResident,
        weights,
        stream,
    )
    .unwrap();
    let mut ordinary = AudioDecoder::<Backend, _, _>::from_parts(
        config.clone(),
        ordinary_quantizer,
        ordinary_neural,
        stream,
    )
    .unwrap();
    let kernel = AudioDecoderKernel::<Backend>::new(config, stream).unwrap();
    let mut coordinator = coordinator(world, stream, false);
    let mut state = PipelineAudioDecoder::new(
        &kernel,
        neural,
        &mut quantizer,
        &mut coordinator,
        [704, mode, 1],
        DistributedCommitEpoch::new(1).unwrap(),
    )
    .initial_state(1)
    .unwrap();
    let initial = state.clone();
    let mut reference_state = ordinary.initial_state(1).unwrap();
    let codes = code_tensor(&case["codes"], stream);
    let mut offset = 0;
    let mut epoch = 1;
    for (index, reference) in case["streaming_unflushed"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let count = reference["input"]["shape"][1].as_i64().unwrap() as i32;
        let input = codes
            .index(
                &[
                    Index::Full,
                    Index::Range(offset, offset + count),
                    Index::Full,
                ],
                stream,
            )
            .unwrap();
        let parent_roots = retained_local(&state, stream);
        if index == 1 && mode == 0 {
            let mismatched = PipelineAudioDecoder::new(
                &kernel,
                neural,
                &mut quantizer,
                &mut coordinator,
                [704, mode, epoch],
                DistributedCommitEpoch::new(epoch).unwrap(),
            )
            .initial_state(2)
            .unwrap()
            .map_neural(|_| state.neural().clone());
            let batched =
                MlxTensor::concatenate(&[input.clone(), input.clone()], 0, stream).unwrap();
            let reads = source.source_diagnostics().unwrap().physical_reads;
            epoch += 1;
            assert!(step(
                &kernel,
                neural,
                &mut quantizer,
                &mut coordinator,
                &batched,
                &mismatched,
                mode,
                epoch,
                stream
            )
            .is_err());
            assert_eq!(
                source.source_diagnostics().unwrap().physical_reads,
                reads,
                "restored neural batch is checked before quantizer acquisition"
            );
            assert_eq!(retained_local(&state, stream), parent_roots);
        }
        if index == 0 && mode == 2 {
            let reads = source.source_diagnostics().unwrap().physical_reads;
            let invalid = if world.rank() == 0 {
                MlxTensor::from_i32_slice(&[0; 4], &[1, 1, 4], stream).unwrap()
            } else {
                input.clone()
            };
            epoch += 1;
            assert!(step(
                &kernel,
                neural,
                &mut quantizer,
                &mut coordinator,
                &invalid,
                &state,
                mode,
                epoch,
                stream
            )
            .is_err());
            assert_eq!(
                source.source_diagnostics().unwrap().physical_reads,
                reads,
                "participant input rejection precedes all learned acquisition"
            );
            assert_eq!(retained_local(&state, stream), parent_roots);
        }
        epoch += 1;
        let (pcm, candidate) = step(
            &kernel,
            neural,
            &mut quantizer,
            &mut coordinator,
            &input,
            &state,
            mode,
            epoch,
            stream,
        )
        .unwrap();
        let (ordinary_pcm, ordinary_candidate) = ordinary
            .step(&input, &reference_state, true, stream)
            .unwrap();
        compare(
            &pcm.to_f32_vec(stream).unwrap(),
            &reference["waveform"],
            fixture,
        );
        compare_tensor(&pcm, &ordinary_pcm, fixture, stream);
        pipeline::compare_cache(
            candidate.neural(),
            reference,
            candidate.neural().global_units(),
            fixture,
            stream,
        );
        assert_eq!(candidate.frames_seen(), ordinary_candidate.frames_seen());
        assert_eq!(
            candidate.samples_emitted(),
            ordinary_candidate.samples_emitted()
        );
        assert_eq!(
            candidate.spectral().retained_frames(),
            ordinary_candidate.spectral().retained_frames()
        );
        if let (Some((real, imaginary)), Some((expected_real, expected_imaginary))) = (
            candidate.spectral().history(),
            ordinary_candidate.spectral().history(),
        ) {
            compare_tensor(real, expected_real, fixture, stream);
            compare_tensor(imaginary, expected_imaginary, fixture, stream);
        } else {
            assert!(
                candidate.spectral().history().is_none()
                    && ordinary_candidate.spectral().history().is_none()
            );
        }
        assert_eq!(retained_local(&state, stream), parent_roots);
        if index == 1 {
            let restored = snapshot(&state, stream);
            epoch += 1;
            let (replayed, replay_state) = step(
                &kernel,
                neural,
                &mut quantizer,
                &mut coordinator,
                &input,
                &restored,
                mode,
                epoch,
                stream,
            )
            .unwrap();
            assert_eq!(
                replayed.to_f32_vec(stream).unwrap(),
                pcm.to_f32_vec(stream).unwrap()
            );
            assert_eq!(
                retained_local(&replay_state, stream),
                retained_local(&candidate, stream)
            );
        }
        offset += count;
        state = candidate;
        reference_state = ordinary_candidate;
    }
    let parent_roots = retained_local(&state, stream);
    let reads = source.source_diagnostics().unwrap().physical_reads;
    epoch += 1;
    let (tail, finished) = finish(
        &kernel,
        neural,
        &mut quantizer,
        &mut coordinator,
        &state,
        mode,
        epoch,
        stream,
    )
    .unwrap();
    assert_eq!(
        source.source_diagnostics().unwrap().physical_reads,
        reads,
        "drain acquires neither codebooks nor neural units"
    );
    let (expected_tail, expected_finished) =
        ordinary.finish(&reference_state, true, stream).unwrap();
    let tail = tail.unwrap();
    compare(
        &tail.to_f32_vec(stream).unwrap(),
        &case["finish_tail"],
        fixture,
    );
    compare_tensor(&tail, &expected_tail.unwrap(), fixture, stream);
    assert!(finished.is_finished());
    assert_eq!(
        finished.samples_emitted(),
        expected_finished.samples_emitted()
    );
    assert_eq!(finished.frames_seen(), offset as u64);
    assert!(pipeline::retained_pipeline(finished.neural(), stream).is_empty());
    assert_eq!(retained_local(&state, stream), parent_roots);
    assert_eq!(initial.frames_seen(), 0);
    assert!(retained_local(&initial, stream).is_empty());
    epoch += 1;
    assert!(finish(
        &kernel,
        neural,
        &mut quantizer,
        &mut coordinator,
        &finished,
        mode,
        epoch,
        stream
    )
    .unwrap()
    .0
    .is_none());
    assert_eq!(source.source_diagnostics().unwrap().physical_reads, reads);
    if mode > 0 {
        assert!(
            neural
                .residency_report()
                .unwrap()
                .offload()
                .peak_resident_bytes()
                .get(MemoryTier::Device)
                <= neural_budget
        );
        assert!(
            quantizer
                .residency_report()
                .unwrap()
                .offload()
                .peak_resident_bytes()
                .get(MemoryTier::Device)
                <= quantizer_budget
        );
    }
}

#[test]
#[ignore = "spawns two/four local MLX Ring ranks and opens loopback sockets; run explicitly"]
fn native_ear_waveform_pp2_tp2pp2_preserves_startup_replay_and_drain() {
    pipeline::launch(2, "pipeline::native_ear_waveform_pipeline_worker");
    pipeline::launch(4, "pipeline::native_ear_waveform_pipeline_worker");
}
