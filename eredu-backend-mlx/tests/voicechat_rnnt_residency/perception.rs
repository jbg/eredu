//! One transported perception frame supplies both language features and RNNT.
use super::*;
use eredu_architectures::voicechat::{
    perception_rnnt::{PerceptionRnntLimits, PerceptionRnntState},
    perception_rnnt_pipeline::advance,
    rnnt_pipeline::RnntPipelineCursor,
};
use eredu_backend_mlx::{
    backend::nn::shared::MlxNeuralBackend as Backend,
    native::{MlxPerceptionPipelineState, MlxRnntPredictorPipeline},
};
use eredu_core::DistributedCommitEpoch;

#[path = "../support/voicechat_perception.rs"]
pub(super) mod perception_fixture;

type State = PerceptionRnntState<MlxTensor, MlxPerceptionPipelineState>;
fn roots(state: &State, stream: &Stream) -> Retained {
    let mut values = Vec::new();
    state.visit_retained_tensors(&mut |tensor| {
        values.push((
            tensor.shape().to_vec(),
            tensor.element_type().unwrap(),
            tensor.to_f32_vec(stream).unwrap(),
        ));
    });
    values
}
fn compare_perception(value: &MlxTensor, expected: &Value, fixture: &Value, stream: &Stream) {
    let shape: Vec<_> = expected["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    assert_eq!(value.shape(), shape);
    compare(value, &expected["values"], fixture, stream);
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
    let retained = roots(state, stream);
    let estimate = state.estimate_snapshot_value().unwrap();
    let reservation = budget
        .reserve(SnapshotResourceKind::Snapshot, Some(estimate))
        .unwrap();
    let copied = state.copy_snapshot_value(&reservation).unwrap();
    assert_eq!(roots(&copied, stream), retained);
    assert_eq!(copied.frames(), state.frames());
    assert_eq!(
        copied.perception().global_blocks(),
        state.perception().global_blocks()
    );
    assert_eq!(
        copied.rnnt().predictor().unwrap().global_layers(),
        state.rnnt().predictor().unwrap().global_layers()
    );
    copied
}

pub(super) fn exercise(
    predictor: &mut MlxRnntPredictorPipeline,
    directory: &std::path::Path,
    rnnt_source: SharedCheckpointSource,
    reference: &Value,
    mode: u64,
    world: &safemlx::distributed::Group,
    weights: &Stream,
    stream: &Stream,
) {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::PERCEPTION_REFERENCE_JSON).unwrap();
    let dim = perception_fixture::dimensions(&fixture);
    let perception_source = perception_fixture::checkpoint(
        &fixture,
        false,
        &directory.join(format!("paired-perception-{mode}.safetensors")),
    );
    // Both semantic cuts are used without multiplying the residency matrix.
    let cut = mode as usize % 2;
    let perception_fixture::PipelineFixture {
        mut model,
        state: initial,
        largest,
        owns_block,
    } = perception_fixture::pipeline(
        dim,
        perception_source.clone(),
        cut,
        mode,
        world,
        weights,
        stream,
    );
    assert_eq!(
        model.dimensions().block.width,
        dimensions(reference).encoder_width
    );
    let cohort = (world.size() > 2).then(|| greedy::coordinator(world, stream, true));
    let mut coordinator = greedy::coordinator(world, stream, false);
    let mut projections = greedy::load(
        rnnt_source.clone(),
        reference,
        mode,
        cohort.as_ref(),
        weights,
        stream,
    );
    let mut cursor = RnntPipelineCursor::new(0, DistributedCommitEpoch::new(10_000).unwrap(), 1000);
    let mut state = PerceptionRnntState::from_perception(initial).unwrap();
    let limits = PerceptionRnntLimits {
        frame_capacity: 32,
        rnnt: RnntLimits {
            max_symbols_per_frame: 3,
            hypothesis_capacity: 64,
        },
    };
    let mut epoch = 1000;
    for (index, window) in fixture["subsampling"]["steps"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let input = perception_fixture::tensor(&window["input"]);
        let drop = window["drop"].as_u64().unwrap() as usize;
        let expected = &fixture["perception"]["steps"][index];
        let rnnt_expected = &reference["streaming_frames"][index];
        let before = roots(&state, stream);
        let dense_before = model
            .dense_stream_report()
            .unwrap()
            .map(|report| report.prefill_forwards() + report.decode_forwards());
        epoch += 1;
        let candidate = advance::<Backend, _, _, _, _>(
            &mut model,
            &mut projections,
            predictor,
            &mut coordinator,
            [401, mode, index as u64],
            DistributedCommitEpoch::new(epoch).unwrap(),
            &mut cursor,
            &input,
            drop,
            &state,
            limits,
            stream,
        )
        .unwrap();
        assert_eq!(roots(&state, stream), before);
        assert_eq!(state.frames(), index as u64);
        assert_eq!(candidate.state.frames(), index as u64 + 1);
        assert_eq!(candidate.state.rnnt().frames(), candidate.state.frames());
        compare_perception(
            &candidate.encoder,
            &expected["encoder_output"],
            &fixture,
            stream,
        );
        compare_perception(
            &candidate.projected,
            &expected["projected_output"],
            &fixture,
            stream,
        );
        let emitted: Vec<_> = rnnt_expected["emitted"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(candidate.emitted, emitted);
        assert_eq!(
            candidate.first_blank,
            rnnt_expected["first_blank"].as_bool().unwrap()
        );
        assert_eq!(
            candidate.ended_blank,
            rnnt_expected["ended_blank"].as_bool().unwrap()
        );
        assert_eq!(
            candidate.hit_symbol_limit,
            rnnt_expected["hit_symbol_limit"].as_bool().unwrap()
        );
        pipeline::compare_prediction(
            candidate.state.rnnt().predictor().unwrap(),
            rnnt_expected,
            reference,
            stream,
        );
        assert_eq!(
            candidate.state.perception().blocks().len(),
            usize::from(owns_block)
        );
        if let Some(cache) = candidate.state.perception().blocks().first() {
            assert_eq!(
                cache.valid_channel(),
                expected["valid_cache"].as_u64().unwrap() as usize
            );
            compare_perception(
                cache.channel().unwrap(),
                &expected["channel_cache"],
                &fixture,
                stream,
            );
            compare_perception(
                &cache
                    .convolution()
                    .unwrap()
                    .transpose_axes(&[0, 2, 1], stream)
                    .unwrap(),
                &expected["time_cache"],
                &fixture,
                stream,
            );
        }
        if let Some(before) = dense_before {
            let report = model.dense_stream_report().unwrap().unwrap();
            assert_eq!(
                report.prefill_forwards() + report.decode_forwards(),
                before + 1 + u64::from(owns_block),
                "one traversal of each owned perception group supplies both consumers"
            );
        }
        if index == 1 {
            epoch += 1;
            let replay = advance::<Backend, _, _, _, _>(
                &mut model,
                &mut projections,
                predictor,
                &mut coordinator,
                [402, mode, index as u64],
                DistributedCommitEpoch::new(epoch).unwrap(),
                &mut cursor,
                &input,
                drop,
                &state,
                limits,
                stream,
            )
            .unwrap();
            assert_eq!(replay.emitted, candidate.emitted);
            assert_eq!(
                roots(&replay.state, stream),
                roots(&candidate.state, stream)
            );
            let expected_roots = usize::from(owns_block) * 2
                + if world.rank() < world.size() / 2 {
                    3
                } else {
                    2
                };
            assert_eq!(
                roots(&candidate.state, stream).len(),
                expected_roots,
                "snapshot owns only local perception and predictor carry plus required readout"
            );
            state = snapshot(&candidate.state, stream);
        } else {
            state = candidate.state;
        }
    }
    let input = perception_fixture::tensor(&fixture["subsampling"]["steps"][5]["input"]);
    let drop = fixture["subsampling"]["steps"][5]["drop"].as_u64().unwrap() as usize;
    let before = roots(&state, stream);
    let reads = (
        perception_source
            .source_diagnostics()
            .unwrap()
            .physical_reads,
        rnnt_source.source_diagnostics().unwrap().physical_reads,
    );
    epoch += 1;
    assert!(advance::<Backend, _, _, _, _>(
        &mut model,
        &mut projections,
        predictor,
        &mut coordinator,
        [403, mode, 0],
        DistributedCommitEpoch::new(epoch).unwrap(),
        &mut cursor,
        &input,
        drop,
        &state,
        PerceptionRnntLimits {
            frame_capacity: state.frames(),
            ..limits
        },
        stream,
    )
    .is_err());
    assert_eq!(roots(&state, stream), before);
    assert_eq!(
        (
            perception_source
                .source_diagnostics()
                .unwrap()
                .physical_reads,
            rnnt_source.source_diagnostics().unwrap().physical_reads
        ),
        reads
    );
    if mode == 2 && cohort.is_some() {
        let perception_fault_path =
            directory.join("paired-perception-acquisition-failure.safetensors");
        let fault_source = perception_fixture::checkpoint(&fixture, false, &perception_fault_path);
        let mut faulty =
            perception_fixture::pipeline(dim, fault_source, cut, mode, world, weights, stream)
                .model;
        // With cut zero, stage one owns Conformer then projection. Invalidate
        // one of its two replicas before its first Conformer acquisition, while
        // its peer would otherwise enter the block's neural all-reduce.
        assert_eq!(cut, 0);
        if world.rank() == world.size() / 2 {
            std::fs::write(
                &perception_fault_path,
                b"invalidated admitted perception source",
            )
            .unwrap();
        }
        let before_rnnt_reads = rnnt_source.source_diagnostics().unwrap().physical_reads;
        let previous_step = cursor.next_step();
        epoch += 1;
        assert!(advance::<Backend, _, _, _, _>(
            &mut faulty,
            &mut projections,
            predictor,
            &mut coordinator,
            [406, mode, 0],
            DistributedCommitEpoch::new(epoch).unwrap(),
            &mut cursor,
            &input,
            drop,
            &state,
            limits,
            stream,
        )
        .is_err());
        assert!(cursor.next_step() > previous_step);
        assert_eq!(
            rnnt_source.source_diagnostics().unwrap().physical_reads,
            before_rnnt_reads,
            "perception cohort failure fences every RNNT phase"
        );
        assert_eq!(roots(&state, stream), before);
        epoch += 1;
        let retry = advance::<Backend, _, _, _, _>(
            &mut model,
            &mut projections,
            predictor,
            &mut coordinator,
            [407, mode, 0],
            DistributedCommitEpoch::new(epoch).unwrap(),
            &mut cursor,
            &input,
            drop,
            &state,
            limits,
            stream,
        )
        .unwrap();
        assert_eq!(retry.state.frames(), state.frames() + 1);
        assert_eq!(roots(&state, stream), before);
        let fault_path = directory.join("paired-rnnt-acquisition-failure.safetensors");
        let fault_source = checkpoint(reference, false, &fault_path);
        let mut fault_projection = greedy::load(
            fault_source,
            reference,
            mode,
            cohort.as_ref(),
            weights,
            stream,
        );
        if world.rank() == 0 {
            std::fs::write(&fault_path, b"invalidated admitted projection source").unwrap();
        }
        let previous = cursor.next_step();
        let before_forwards = model.dense_stream_report().unwrap().unwrap();
        let before_forwards =
            before_forwards.prefill_forwards() + before_forwards.decode_forwards();
        epoch += 1;
        assert!(advance::<Backend, _, _, _, _>(
            &mut model,
            &mut fault_projection,
            predictor,
            &mut coordinator,
            [404, mode, 0],
            DistributedCommitEpoch::new(epoch).unwrap(),
            &mut cursor,
            &input,
            drop,
            &state,
            limits,
            stream,
        )
        .is_err());
        let after = model.dense_stream_report().unwrap().unwrap();
        assert_eq!(
            after.prefill_forwards() + after.decode_forwards(),
            before_forwards + 1 + u64::from(owns_block),
            "RNNT failed after one candidate traversal of each owned perception group"
        );
        assert!(cursor.next_step() > previous);
        assert_eq!(
            roots(&state, stream),
            before,
            "RNNT failure discards both candidate states"
        );
        epoch += 1;
        let retry = advance::<Backend, _, _, _, _>(
            &mut model,
            &mut projections,
            predictor,
            &mut coordinator,
            [405, mode, 0],
            DistributedCommitEpoch::new(epoch).unwrap(),
            &mut cursor,
            &input,
            drop,
            &state,
            limits,
            stream,
        )
        .unwrap();
        assert_eq!(retry.state.frames(), state.frames() + 1);
        assert_eq!(roots(&state, stream), before);
    }
    if mode > 0 {
        let budget = if mode == 1 { largest } else { largest * 2 };
        assert!(
            model
                .residency_report()
                .unwrap()
                .offload()
                .peak_resident_bytes()
                .get(MemoryTier::Device)
                <= budget
        );
    }
}

#[test]
#[ignore = "spawns two/four local MLX Ring ranks and opens loopback sockets; run explicitly"]
fn native_perception_and_rnnt_pp2_tp2pp2_share_frames_and_replay() {
    pipeline::launch(2, "pipeline::native_perception_rnnt_pipeline_worker");
    pipeline::launch(4, "pipeline::native_perception_rnnt_pipeline_worker");
}
