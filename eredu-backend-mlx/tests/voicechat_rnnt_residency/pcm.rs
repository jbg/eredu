//! PCM preprocessing and both neural frontiers use the same retained pipeline bundle.
use super::perception::perception_fixture;
use super::*;
use eredu_architectures::voicechat::{
    config::VoiceChatConfig,
    frontend::{FrontendOptions, PcmPrecision, VoiceChatFrontend},
    input::{InputExecution, InputFrame, InputState},
    input_pipeline::PipelineInput,
    perception_rnnt::{PerceptionRnnt, PerceptionRnntLimits, PerceptionRnntState},
    perception_rnnt_preparation::PreparedPerceptionRnntPipelineBindings,
    rnnt_pipeline::RnntPipelineCursor,
    rnnt_predictor_distribution::PreparedRnntPredictorPipelineBindings,
    rnnt_projection_preparation::PreparedRnntProjectionBindings,
};
use eredu_backend_mlx::{
    backend::nn::shared::MlxNeuralBackend as Backend,
    native::{
        materialize_voicechat_perception, materialize_voicechat_perception_rnnt_pipeline,
        MlxPerceptionPipelineState, MlxPerceptionRnntPipeline,
    },
};
use eredu_core::DistributedCommitEpoch;
use eredu_media::audio::streaming::{FinalPcmFrame, PcmFinishPolicy};
use eredu_runtime::CommunicationManifest;

type State = InputState<MlxTensor, MlxPerceptionPipelineState>;
fn roots(state: &State, stream: &Stream) -> Retained {
    let mut values = Vec::new();
    state.visit_retained_tensors(&mut |tensor| {
        values.push((
            tensor.shape().to_vec(),
            tensor.element_type().unwrap(),
            tensor.to_f32_vec(stream).unwrap(),
        ))
    });
    values
}
fn frontend(fixture: &Value) -> VoiceChatFrontend {
    let config = VoiceChatConfig::from_json(include_str!(
        "../../../eredu-architectures/src/voicechat/fixtures/official-inference-config.json"
    ))
    .unwrap();
    VoiceChatFrontend::with_coefficients(
        &config.perception.preprocessor,
        FrontendOptions {
            pcm_precision: PcmPrecision::F32,
            buffer_frames: 3,
            queue_capacity_samples: 4096,
            pad_and_drop_preencoded: false,
        },
        perception_fixture::floats(&fixture["preprocessing"]["window"]),
        perception_fixture::floats(&fixture["preprocessing"]["filters"]),
    )
    .unwrap()
}
fn compare_tensor(actual: &MlxTensor, expected: &MlxTensor, tolerance: &Value, stream: &Stream) {
    assert_eq!(actual.shape(), expected.shape());
    let absolute = tolerance["tolerance"]["absolute"].as_f64().unwrap() as f32;
    let relative = tolerance["tolerance"]["relative"].as_f64().unwrap() as f32;
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
fn compare_frame(
    actual: &InputFrame<MlxTensor, MlxPerceptionPipelineState>,
    expected: &InputFrame<MlxTensor>,
    perception_tolerance: &Value,
    rnnt_tolerance: &Value,
    stream: &Stream,
) {
    assert_eq!(
        (
            actual.frame_index,
            actual.start_sample,
            actual.valid_samples,
            actual.source
        ),
        (
            expected.frame_index,
            expected.start_sample,
            expected.valid_samples,
            expected.source
        )
    );
    assert_eq!(
        (
            &actual.emitted,
            actual.first_blank,
            actual.ended_blank,
            actual.hit_symbol_limit
        ),
        (
            &expected.emitted,
            expected.first_blank,
            expected.ended_blank,
            expected.hit_symbol_limit
        )
    );
    compare_tensor(
        &actual.encoder,
        &expected.encoder,
        perception_tolerance,
        stream,
    );
    compare_tensor(
        &actual.projected,
        &expected.projected,
        perception_tolerance,
        stream,
    );
    assert_eq!(
        actual.state.frontend().frames(),
        expected.state.frontend().frames()
    );
    assert_eq!(
        actual.state.neural().frames(),
        expected.state.neural().frames()
    );
    assert_eq!(
        actual.state.neural().rnnt().hypothesis(),
        expected.state.neural().rnnt().hypothesis()
    );
    for local in actual.state.neural().perception().blocks() {
        let global = &expected.state.neural().perception().blocks()[0];
        assert_eq!(local.valid_channel(), global.valid_channel());
        compare_tensor(
            local.channel().unwrap(),
            global.channel().unwrap(),
            perception_tolerance,
            stream,
        );
        compare_tensor(
            local.convolution().unwrap(),
            global.convolution().unwrap(),
            perception_tolerance,
            stream,
        );
    }
    let prediction = actual.state.neural().rnnt().predictor().unwrap();
    let reference = expected.state.neural().rnnt().predictor().unwrap();
    compare_tensor(
        prediction.output(),
        reference.output(),
        rnnt_tolerance,
        stream,
    );
    for (global, local) in prediction
        .global_layers()
        .zip(&prediction.local_state().layers)
    {
        compare_tensor(
            &local.hidden,
            &reference.local_state().layers[global].hidden,
            rnnt_tolerance,
            stream,
        );
        compare_tensor(
            &local.cell,
            &reference.local_state().layers[global].cell,
            rnnt_tolerance,
            stream,
        );
    }
}
fn advance(
    bundle: &mut MlxPerceptionRnntPipeline,
    state: &State,
    cursor: &mut RnntPipelineCursor,
    epoch: u64,
    identity: u64,
    limits: PerceptionRnntLimits,
    stream: &Stream,
) -> Result<
    Option<InputFrame<MlxTensor, MlxPerceptionPipelineState>>,
    eredu_architectures::voicechat::input::InputError,
> {
    let (perception, projections, predictor, coordinator) = bundle.parts_mut();
    let mut input = PipelineInput::<Backend, _, _, _, _>::new(
        perception,
        projections,
        predictor,
        coordinator,
        [501, identity, epoch],
        DistributedCommitEpoch::new(epoch).unwrap(),
        cursor,
    );
    let result = input.advance_pcm(state, limits, stream)?;
    assert_eq!(input.is_consumed(), result.is_some());
    if result.is_none() {
        assert!(input.advance_pcm(state, limits, stream)?.is_none());
        assert!(!input.is_consumed());
    }
    Ok(result)
}
fn snapshot(state: &State, stream: &Stream) -> State {
    use eredu_core::execution_control::{SnapshotLimits, SnapshotResourceKind};
    use eredu_runtime::execution_control::{SnapshotBudget, SnapshotValue};
    let budget = SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 1,
        max_branches: 0,
        retained_bytes: 1 << 20,
        cumulative_copy_bytes: 1 << 20,
    });
    let estimate = state.estimate_snapshot_value().unwrap();
    let reservation = budget
        .reserve(SnapshotResourceKind::Snapshot, Some(estimate))
        .unwrap();
    let copy = state.copy_snapshot_value(&reservation).unwrap();
    assert_eq!(roots(&copy, stream), roots(state, stream));
    assert_eq!(
        copy.frontend().queued_samples(),
        state.frontend().queued_samples()
    );
    assert_eq!(copy.frontend().frames(), state.frontend().frames());
    assert_eq!(copy.neural().frames(), state.neural().frames());
    copy
}

#[allow(clippy::too_many_arguments)]
pub(super) fn exercise(
    predictor: PreparedRnntPredictorPipelineBindings,
    predictor_residency: LayerWeightResidency,
    directory: &std::path::Path,
    rnnt_source: SharedCheckpointSource,
    rnnt_reference: &Value,
    mode: u64,
    world: &safemlx::distributed::Group,
    weights: &Stream,
    stream: &Stream,
) {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::TOOL_SILENCE_REFERENCE_JSON).unwrap();
    let dim = perception_fixture::dimensions(&fixture);
    let source = perception_fixture::checkpoint(
        &fixture,
        false,
        &directory.join(format!("pcm-perception-{mode}.safetensors")),
    );
    let selected =
        perception_fixture::prepare_pipeline(dim, source.clone(), mode as usize % 2, mode, world);
    let (decision_manifest, decision_group, owner) = greedy::coordination_manifest(world, false);
    let mut groups = decision_manifest.groups().to_vec();
    let projection_group = if world.size() > 2 {
        let (cohort, id, _) = greedy::coordination_manifest(world, true);
        for group in cohort.groups() {
            groups.push(
                eredu_runtime::CommunicationGroupDescriptor::new(
                    group.id(),
                    groups.len(),
                    group.members().to_vec(),
                    group.local_index(),
                    group.requirements().clone(),
                )
                .unwrap(),
            );
        }
        Some(id)
    } else {
        None
    };
    let coordination = CommunicationManifest::new(world.size(), world.rank(), groups, vec![])
        .unwrap()
        .with_completion_policy(decision_manifest.completion_policy().unwrap());
    let reads = (
        source.source_diagnostics().unwrap().physical_reads,
        rnnt_source.source_diagnostics().unwrap().physical_reads,
    );
    let prepared = PreparedPerceptionRnntPipelineBindings::new(
        selected.prepared,
        predictor,
        PreparedRnntProjectionBindings::from_rnnt(prepared(
            dimensions(rnnt_reference),
            rnnt_source.clone(),
        ))
        .unwrap(),
        coordination,
        decision_group,
        owner,
        projection_group,
    )
    .unwrap();
    let resources = prepared.resources();
    assert_eq!(
        (
            source.source_diagnostics().unwrap().physical_reads,
            rnnt_source.source_diagnostics().unwrap().physical_reads
        ),
        reads
    );
    let projection_residency = match mode {
        0 => LayerWeightResidency::FullyResident,
        1 => LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
            OffloadConfig::new(Some(64), Some(1 << 20), 1).unwrap(),
        )),
        _ => LayerWeightResidency::DenseDiskStream(
            DenseDiskStreamLoadOptions::new(64, 0, 0, 0).unwrap(),
        ),
    };
    let (mut bundle, metadata) = materialize_voicechat_perception_rnnt_pipeline(
        prepared,
        world,
        selected.residency,
        predictor_residency,
        projection_residency,
        weights,
        stream,
    )
    .unwrap();
    assert_eq!(bundle.resources(), resources);
    assert_eq!(
        resources.perception_parameter_bytes,
        metadata.perception.layer_parameter_bytes()
    );
    assert_eq!(
        resources.predictor_parameter_bytes,
        metadata.predictor.layer_parameter_bytes()
    );
    assert_eq!(
        resources.projection_parameter_bytes,
        metadata.projections.layer_parameter_bytes()
    );
    assert_eq!(
        resources.perception_cache_bytes,
        if selected.owns_block { 80 } else { 0 }
    );
    assert_eq!(
        resources.prediction_cache_bytes,
        if world.rank() < world.size() / 2 {
            24
        } else {
            16
        }
    );
    let (ordinary_perception, _) = materialize_voicechat_perception(
        perception_fixture::prepared(dim, source.clone()),
        LayerWeightResidency::FullyResident,
        weights,
        stream,
    )
    .unwrap();
    let (ordinary_rnnt, _) = materialize_voicechat_rnnt(
        prepared_rnnt(dimensions(rnnt_reference), rnnt_source.clone()),
        LayerWeightResidency::FullyResident,
        weights,
        stream,
    )
    .unwrap();
    let mut ordinary =
        PerceptionRnnt::<Backend, _, _>::from_parts(ordinary_perception, ordinary_rnnt).unwrap();
    let pcm: Vec<_> = (0..2877i64)
        .map(|i| (((i * 73 + i * i * 19) % 2048) - 1024) as f32 / 8192.)
        .collect();
    let finish = PcmFinishPolicy {
        final_frame: FinalPcmFrame::ZeroPad,
        trailing_silence_frames: 1,
    };
    let limits = PerceptionRnntLimits {
        frame_capacity: 16,
        rnnt: RnntLimits {
            max_symbols_per_frame: 3,
            hypothesis_capacity: 64,
        },
    };
    let mut whole =
        InputState::from_parts(frontend(&fixture), PerceptionRnntState::default()).unwrap();
    whole.submit(&pcm, 16000).unwrap();
    whole.finish(finish).unwrap();
    let mut expected = Vec::new();
    while let Some(frame) = ordinary.advance_pcm(&whole, limits, stream).unwrap() {
        whole = frame.state.clone();
        expected.push(frame);
    }
    assert_eq!(expected.len(), 4);
    assert!(whole.frontend().is_drained());
    let mut state =
        InputState::from_parts(frontend(&fixture), bundle.initial_state().unwrap()).unwrap();
    let mut cursor =
        RnntPipelineCursor::new(0, DistributedCommitEpoch::new(100_000).unwrap(), 1000);
    let mut epoch = 20_000;
    let mut delivered = 0;
    let mut offset = 0;
    for chunk in [31, 1, 1000, 247, 3, 1595] {
        state.submit(&pcm[offset..offset + chunk], 16000).unwrap();
        offset += chunk;
        loop {
            let parent = roots(&state, stream);
            let queued = state.frontend().queued_samples();
            let before_cursor = cursor.next_step();
            epoch += 1;
            let output = advance(
                &mut bundle,
                &state,
                &mut cursor,
                epoch,
                mode,
                limits,
                stream,
            )
            .unwrap();
            assert_eq!(roots(&state, stream), parent);
            assert_eq!(state.frontend().queued_samples(), queued);
            let Some(frame) = output else {
                assert_eq!(cursor.next_step(), before_cursor);
                break;
            };
            compare_frame(
                &frame,
                &expected[delivered],
                &fixture,
                rnnt_reference,
                stream,
            );
            if delivered == 1 {
                let restored = snapshot(&state, stream);
                epoch += 1;
                let replay = advance(
                    &mut bundle,
                    &restored,
                    &mut cursor,
                    epoch,
                    mode,
                    limits,
                    stream,
                )
                .unwrap()
                .unwrap();
                assert_eq!(roots(&replay.state, stream), roots(&frame.state, stream));
                assert_eq!(replay.emitted, frame.emitted);
            }
            state = frame.state;
            delivered += 1;
        }
    }
    state.finish(finish).unwrap();
    let before = roots(&state, stream);
    let queued = state.frontend().queued_samples();
    let reads = (
        source.source_diagnostics().unwrap().physical_reads,
        rnnt_source.source_diagnostics().unwrap().physical_reads,
    );
    epoch += 1;
    assert!(advance(
        &mut bundle,
        &state,
        &mut cursor,
        epoch,
        mode,
        PerceptionRnntLimits {
            frame_capacity: state.neural().frames(),
            ..limits
        },
        stream
    )
    .is_err());
    assert_eq!(roots(&state, stream), before);
    assert_eq!(state.frontend().queued_samples(), queued);
    assert_eq!(
        (
            source.source_diagnostics().unwrap().physical_reads,
            rnnt_source.source_diagnostics().unwrap().physical_reads
        ),
        reads
    );
    loop {
        epoch += 1;
        let Some(frame) = advance(
            &mut bundle,
            &state,
            &mut cursor,
            epoch,
            mode,
            limits,
            stream,
        )
        .unwrap() else {
            break;
        };
        compare_frame(
            &frame,
            &expected[delivered],
            &fixture,
            rnnt_reference,
            stream,
        );
        state = frame.state;
        delivered += 1;
    }
    assert_eq!(delivered, expected.len());
    assert!(state.frontend().is_drained());
    assert_eq!(state.frontend().frames(), 4);
    assert_eq!(state.neural().rnnt().frames(), 4);
    let reports = bundle.residency_reports().unwrap();
    if mode > 0 {
        assert!(
            reports
                .perception
                .offload()
                .peak_resident_bytes()
                .get(MemoryTier::Device)
                <= if mode == 1 {
                    selected.largest
                } else {
                    selected.largest * 2
                }
        );
        assert!(
            reports
                .predictor
                .offload()
                .peak_resident_bytes()
                .get(MemoryTier::Device)
                <= 192
        );
        assert!(
            reports
                .projections
                .offload()
                .peak_resident_bytes()
                .get(MemoryTier::Device)
                <= 64
        );
    }
}
fn prepared_rnnt(dim: RnntDimensions, source: SharedCheckpointSource) -> PreparedRnntBindings {
    super::prepared(dim, source)
}

#[test]
#[ignore = "spawns two/four local MLX Ring ranks and opens loopback sockets; run explicitly"]
fn native_pcm_aggregate_pp2_tp2pp2_matches_ordinary_chunks_and_drain() {
    pipeline::launch(2, "pipeline::native_pcm_pipeline_worker");
    pipeline::launch(4, "pipeline::native_pcm_pipeline_worker");
}
