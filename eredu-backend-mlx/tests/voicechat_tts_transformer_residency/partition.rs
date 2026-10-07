//! Native parameter providers and kernels across canonical TTS stage boundaries.
use super::*;
use eredu_architectures::voicechat::tts_transformer_layered::{
    tts_transformer_parameter_description, TransformerArchitecture, TransformerBoundary,
    TransformerBoundarySchema,
};
use eredu_backend_mlx::native::{
    materialize_voicechat_tts_transformer_partition, MlxTtsTransformerPartition,
};
use eredu_core::{ParallelRankTopology, ParallelTopology};
use eredu_runtime::{
    ArchitectureBoundary, ArchitectureParameters, ArchitecturePartition,
    ArchitectureStatePartitionPlan, ArchitectureStatePartitionRule, LayeredPartitionInput,
    LayeredPartitionOutput, PartitionOwnership, RuntimeLayerState,
};
use std::ops::Range;

struct Stage {
    model: MlxTtsTransformerPartition,
    state: eredu_backend_mlx::native::MlxTtsTransformerPipelineState,
    blocks: Range<usize>,
    budget: u64,
}

fn run(
    stages: &mut [Stage],
    input: &MlxTensor,
    mask: Option<&MlxTensor>,
    schema: TransformerBoundarySchema,
    stream: &Stream,
) -> MlxTensor {
    let mut incoming: Option<(MlxTensor, TransformerBoundary<MlxTensor>)> = None;
    let count = stages.len();
    for (index, stage) in stages.iter_mut().enumerate() {
        let input = match incoming.take() {
            Some((hidden, auxiliary)) => LayeredPartitionInput::Hidden { hidden, auxiliary },
            None => LayeredPartitionInput::Tokens(input),
        };
        let output = stage
            .model
            .forward_stage(
                input,
                if index == 0 { mask } else { None },
                &mut stage.state,
                stream,
            )
            .unwrap();
        match output {
            LayeredPartitionOutput::Boundary { hidden, auxiliary } => {
                assert!(index + 1 < count);
                let wire = schema.encode(auxiliary).unwrap();
                let auxiliary = schema
                    .decode(wire.into_iter().map(|value| value.into_parts().1).collect())
                    .unwrap();
                incoming = Some((hidden, auxiliary));
            }
            LayeredPartitionOutput::Final { output, .. } => {
                assert_eq!(index + 1, count);
                return output;
            }
        }
    }
    panic!("pipeline did not produce normalized output")
}

fn compare_caches(stages: &[Stage], ordinary: &MlxKeyValueState, fixture: &Value, stream: &Stream) {
    let ordinary: &[MlxKeyValueLayerState] = ordinary.as_ref();
    for stage in stages {
        let local = stage.state.caches();
        assert_eq!(stage.state.causal_position().unwrap(), ordinary[0].offset());
        for (actual, expected) in local.iter().zip(&ordinary[stage.blocks.clone()]) {
            assert_eq!(actual.offset(), expected.offset());
            let values =
                RuntimeLayerState::<MlxNeuralBackend>::retained_values(actual).collect::<Vec<_>>();
            let expected = RuntimeLayerState::<MlxNeuralBackend>::retained_values(expected)
                .collect::<Vec<_>>();
            assert_eq!(values.len(), expected.len());
            for (actual, expected) in values.iter().zip(&expected) {
                assert_eq!(actual.shape(), expected.shape());
                compare(
                    &actual.to_f32_vec(stream).unwrap(),
                    &serde_json::json!({"values":expected.to_f32_vec(stream).unwrap()}),
                    fixture,
                );
            }
        }
    }
}

pub(super) fn exercise(
    fixture: &Value,
    selected_fixture: &Value,
    ordinary_source: SharedCheckpointSource,
    source: SharedCheckpointSource,
    parallel: Option<&safemlx::distributed::Group>,
    weights: &Stream,
    stream: &Stream,
) {
    for causal in [true, false] {
        let family = if causal { "gemma3" } else { "t5gemma" };
        let data = &fixture[family];
        let configuration = |value: &Value| {
            TransformerConfig::new(&geometry(
                if causal {
                    &value["config"]
                } else {
                    &value["config"]["encoder"]
                },
                causal,
            ))
            .unwrap()
        };
        let config = configuration(&selected_fixture[family]);
        let ordinary_config = configuration(data);
        let description = tts_transformer_parameter_description(&config).unwrap();
        let state_plan =
            ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::group_units(
                0,
                0..2,
            )]);
        // A middle-block address followed by norm in one stage, then a full
        // block range followed by a standalone stateless normalization stage.
        let cuts = [
            vec![vec![(0, 0..1)], vec![(0, 1..2), (1, 0..1)]],
            vec![vec![(0, 0..2)], vec![(1, 0..1)]],
        ];
        for cut in &cuts {
            for policy in 0..3 {
                let mut stages = Vec::new();
                for (index, groups) in cut.iter().enumerate() {
                    let global = prepared(config.clone(), source.clone());
                    let (global, architecture) = if let Some(parallel) = parallel {
                        let topology = ParallelRankTopology::new(
                            ParallelTopology::new(parallel.size(), 1, 1, 1).unwrap(),
                            parallel.rank(),
                        )
                        .unwrap();
                        let (global, local) = global.select_parallel(topology).unwrap();
                        (
                            global,
                            TransformerArchitecture::<MlxNeuralBackend>::new_parallel(
                                config.clone(),
                                &local,
                            )
                            .unwrap(),
                        )
                    } else {
                        (
                            global,
                            TransformerArchitecture::<MlxNeuralBackend>::new(config.clone())
                                .unwrap(),
                        )
                    };
                    let partition = ArchitecturePartition::from_description(
                        &description,
                        groups.iter().map(|(group, range)| {
                            (description.graph().groups()[*group].id(), range.clone())
                        }),
                        PartitionOwnership::new(
                            index == 0,
                            index + 1 == cut.len(),
                            std::iter::empty::<String>(),
                        )
                        .unwrap(),
                        &architecture.state_layout().unwrap(),
                        &state_plan,
                        (),
                        TransformerBoundarySchema::new(&config),
                    )
                    .unwrap();
                    let selected = global.into_partition(&partition).unwrap();
                    let largest = selected
                        .unit_bindings()
                        .iter()
                        .map(|(_, unit)| binding_bytes(unit).unwrap())
                        .max()
                        .unwrap();
                    let total: u64 = selected
                        .unit_bindings()
                        .iter()
                        .map(|(_, unit)| binding_bytes(unit).unwrap())
                        .sum();
                    let budget = if policy == 2 { largest * 2 } else { largest };
                    let residency = match policy {
                        0 => LayerWeightResidency::FullyResident,
                        1 => LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
                            OffloadConfig::new(Some(budget), Some(1 << 20), 1).unwrap(),
                        )),
                        _ => LayerWeightResidency::DenseDiskStream(
                            DenseDiskStreamLoadOptions::new(budget, 0, 0, 0).unwrap(),
                        ),
                    };
                    let (model, metadata) = materialize_voicechat_tts_transformer_partition(
                        selected, &partition, parallel, residency, weights, stream,
                    )
                    .unwrap();
                    assert_eq!(metadata.static_device_bytes(), 0);
                    assert_eq!(metadata.layer_parameter_bytes(), total);
                    let blocks = groups
                        .iter()
                        .find(|(group, _)| *group == 0)
                        .map_or(2..2, |(_, range)| range.clone());
                    let state = model.initial_state().unwrap();
                    if blocks.is_empty() {
                        assert!(model.state_layout().is_none());
                    }
                    stages.push(Stage {
                        model,
                        state,
                        blocks,
                        budget,
                    });
                }
                let (mut ordinary, _) = materialize_voicechat_tts_transformer(
                    prepared(ordinary_config.clone(), ordinary_source.clone()),
                    LayerWeightResidency::FullyResident,
                    weights,
                    stream,
                )
                .unwrap();
                if causal {
                    let mut ordinary_cache = MlxKeyValueState::device(
                        eredu_architectures::decoder::state_layout(&ordinary_config).unwrap(),
                    )
                    .unwrap();
                    for step in data["chunks"].as_array().unwrap() {
                        let input = tensor(&step["input"]);
                        let expected = ordinary
                            .forward_causal(&input, ordinary_cache.as_mut(), stream)
                            .unwrap();
                        compare(
                            &expected.to_f32_vec(stream).unwrap(),
                            &step["output"],
                            fixture,
                        );
                        let output = run(
                            &mut stages,
                            &input,
                            None,
                            TransformerBoundarySchema::new(&config),
                            stream,
                        );
                        compare(
                            &output.to_f32_vec(stream).unwrap(),
                            &step["output"],
                            fixture,
                        );
                        compare_caches(&stages, &ordinary_cache, fixture, stream);
                    }
                } else {
                    let mask = tensor(&data["attention_mask"])
                        .equal_i32(1, stream)
                        .unwrap();
                    for batch in data["batches"].as_array().unwrap() {
                        let output = run(
                            &mut stages,
                            &tensor(&batch["input"]),
                            Some(&mask),
                            TransformerBoundarySchema::new(&config),
                            stream,
                        );
                        compare(
                            &output.to_f32_vec(stream).unwrap(),
                            &batch["output"],
                            fixture,
                        );
                    }
                }
                if policy != 0 {
                    for stage in &stages {
                        assert!(
                            stage
                                .model
                                .residency_report()
                                .unwrap()
                                .offload()
                                .peak_resident_bytes()
                                .get(MemoryTier::Device)
                                <= stage.budget
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn native_tts_partition_stages_preserve_hf_outputs_caches_and_residency() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::TRANSFORMER_REFERENCE_JSON).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let source = checkpoint(
        &fixture,
        false,
        &directory.path().join("weights.safetensors"),
    );
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let weights = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    exercise(
        &fixture,
        &fixture,
        source.clone(),
        source,
        None,
        &weights,
        &stream,
    );
}
