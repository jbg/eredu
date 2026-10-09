//! Cached VoiceChat execution preserves the complete hybrid state across weight tiers.
use std::{collections::BTreeMap, sync::Arc};

use eredu_architectures::{
    artifact_preparation::PhysicalArtifactId,
    nemotron_h,
    voicechat::{
        artifact::ParameterAdmission,
        backbone::{BackbonePreparationError, PreparedBackboneParameters},
        backbone_execution::{execute, BackboneRequest, OrdinaryBackbone},
        channels::{ChannelFusion, FunctionHead, PromptPolicy},
        checkpoint::Component,
        config::VoiceChatConfig,
    },
};
use eredu_backend_mlx::{
    backend::{nn::shared::MlxNeuralBackend as Backend, runtime::cache::state::MlxHybridState},
    native::materialize_voicechat_backbone,
    MlxTensor,
};
use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    store::{SafetensorsWeightStore, SharedCheckpointSource, StoreError, TensorSelection},
    LinearFormat,
};
use eredu_core::residency::{MemoryTier, OffloadConfig};
use eredu_nn::{LinearFormatSpec, ParameterMetadata, ParameterVisitorMut, Parameterized, Tensor};
use eredu_runtime::{
    DenseDiskStreamLoadOptions, LayerRuntimeState, LayerWeightResidency, LayerwiseLoadOptions,
    ResidentRuntime, RuntimeStateComponents,
};
use safemlx::{Device, DeviceType, Stream};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use serde_json::Value;

type Reference = ResidentRuntime<nemotron_h::LayeredModel<Backend>, Backend, MlxHybridState>;

fn args() -> nemotron_h::ModelArgs {
    nemotron_h::model_args_from_config_value(&serde_json::json!({
        "model_type":"nemotron_h", "vocab_size":24, "hidden_size":4,
        "intermediate_size":8, "num_hidden_layers":3,
        "hybrid_override_pattern":"M*-", "num_attention_heads":2,
        "num_key_value_heads":2, "head_dim":2, "mamba_num_heads":2,
        "n_groups":2, "mamba_head_dim":2, "ssm_state_size":3,
        "conv_kernel":3, "chunk_size":2, "num_nextn_predict_layers":0,
        "tie_word_embeddings":false, "residual_in_fp32":true,
        "layer_norm_epsilon":1e-5
    }))
    .unwrap()
}

fn canonical(name: &str) -> String {
    match name {
        "model.embeddings.weight" => "stt_model.embed_tokens.weight".into(),
        "model.norm_f.weight" => "stt_model.llm.norm_f.weight".into(),
        "lm_head.weight" => "stt_model.lm_head.weight".into(),
        _ => name
            .replacen("model.layers.", "stt_model.llm.layers.", 1)
            .replace(".mamba.", ".mixer.")
            .replace(".attention.", ".mixer.")
            .replace(".mlp.", ".mixer."),
    }
}

// Match the existing nonzero native coupled fixture; recurrent parameters must
// not retain their unloaded zero defaults, which would hide state divergence.
fn values(name: &str, length: usize) -> Vec<f32> {
    if name.ends_with(".mamba.dt_bias") {
        return vec![-1.; length];
    }
    if name.ends_with(".mamba.A_log") {
        return vec![-0.3; length];
    }
    if name.ends_with(".mamba.D") {
        return vec![1.; length];
    }
    let mut hash = 2_166_136_261_u32;
    for byte in name.bytes() {
        hash = (hash ^ u32::from(byte)).wrapping_mul(16_777_619);
    }
    (0..length)
        .map(|index| {
            let mixed = hash
                .wrapping_add((index as u32).wrapping_mul(747_796_405))
                .rotate_left((index % 23) as u32);
            let centered = (mixed % 2001) as f32 - 1000.;
            if name.ends_with(".norm.weight") {
                1. + centered / 50_000.
            } else {
                centered / 25_000.
            }
        })
        .collect()
}

struct Bind<'a> {
    fixture: &'a Value,
    stream: &'a Stream,
    weights: BTreeMap<String, (Vec<usize>, Vec<u8>)>,
}
impl<'a> ParameterVisitorMut<'a, MlxTensor> for Bind<'_> {
    fn visit_mut(&mut self, metadata: ParameterMetadata, slot: &'a mut MlxTensor) {
        let name = metadata.id.as_str();
        let key = canonical(name);
        let shape = slot.shape().to_vec();
        let data = if let Some(value) = self.fixture["parameters"].get(&key) {
            value["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| n.as_f64().unwrap() as f32)
                .collect()
        } else {
            values(name, shape.iter().map(|n| *n as usize).product())
        };
        *slot = MlxTensor::from_f32_slice(&data, &shape, self.stream).unwrap();
        assert!(self
            .weights
            .insert(
                key,
                (
                    shape.into_iter().map(|n| n as usize).collect(),
                    data.into_iter().flat_map(f32::to_le_bytes).collect(),
                )
            )
            .is_none());
    }
}

fn state() -> MlxHybridState {
    MlxHybridState::device(nemotron_h::state_layout(&args()).unwrap(), &[]).unwrap()
}

fn snapshot(state: &mut MlxHybridState) -> (Vec<i32>, Vec<(Vec<i32>, Vec<f32>)>) {
    let positions = (0..3)
        .map(|index| state.layer(index).unwrap().position())
        .collect();
    let tensors = state
        .retained_arrays()
        .into_iter()
        .map(|array| {
            let evaluated = array.evaluated().unwrap();
            (array.shape().to_vec(), evaluated.as_slice::<f32>().to_vec())
        })
        .collect();
    (positions, tensors)
}

#[test]
fn native_backbone_residency_preserves_prefill_cached_dual_logits_and_hybrid_state() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::BACKBONE_REFERENCE_JSON).unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let weights_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut reference = Reference::new(
        nemotron_h::LayeredModel::new(args(), &stream).unwrap(),
        &stream,
    )
    .unwrap();
    let mut bind = Bind {
        fixture: &fixture,
        stream: &stream,
        weights: BTreeMap::new(),
    };
    reference
        .architecture_mut()
        .static_modules_mut()
        .visit_parameters_mut(&mut bind);
    for unit in reference.units_mut().iter_mut().flatten() {
        unit.visit_parameters_mut(&mut bind);
    }
    let total_bytes = bind
        .weights
        .values()
        .map(|(_, bytes)| bytes.len() as u64)
        .sum::<u64>();
    let static_bytes = 2 * 24 * 4 * 4 + 4 * 4;
    let layer_bytes = (0..3)
        .map(|index| {
            bind.weights
                .iter()
                .filter(|(name, _)| name.starts_with(&format!("stt_model.llm.layers.{index}.")))
                .map(|(_, (_, bytes))| bytes.len() as u64)
                .sum::<u64>()
        })
        .collect::<Vec<_>>();
    let host_device_bytes = static_bytes + layer_bytes.iter().max().unwrap();
    // The shared disk transfer mechanism retains a two-block device window.
    let disk_device_bytes = static_bytes
        + layer_bytes
            .windows(2)
            .map(|window| window.iter().sum::<u64>())
            .max()
            .unwrap();
    bind.weights
        .insert("unrelated.weight".into(), (vec![1], vec![0; 4]));
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("backbone.safetensors");
    serialize_to_file(
        bind.weights.iter().map(|(name, (shape, bytes))| {
            (
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            )
        }),
        None,
        &path,
    )
    .unwrap();
    let source: SharedCheckpointSource = Arc::new(SafetensorsWeightStore::open(&path).unwrap());
    let admissions = bind
        .weights
        .iter()
        .filter(|(name, _)| name.as_str() != "unrelated.weight")
        .map(|(name, (_, bytes))| {
            let recipe = DerivedWeightRecipe::source(name, TensorSelection::Full);
            (
                name.clone(),
                ParameterAdmission {
                    name: name.clone(),
                    component: Component::Backbone,
                    artifact: PhysicalArtifactId::Primary,
                    output: recipe.infer(source.as_ref()).unwrap(),
                    stored_bytes: bytes.len() as u64,
                    conversion_workspace_bytes: 0,
                    materialization_peak_bytes: recipe
                        .peak_materialization_bytes(source.as_ref())
                        .unwrap()
                        + bytes.len() as u64,
                    recipe,
                },
            )
        })
        .collect();
    let prepared =
        PreparedBackboneParameters::from_parameters(args(), source.clone(), &admissions).unwrap();
    let mut malformed = admissions.clone();
    malformed.remove("stt_model.lm_head.weight").unwrap();
    assert!(matches!(
        PreparedBackboneParameters::from_parameters(args(), source.clone(), &malformed),
        Err(BackbonePreparationError::Catalog(_))
    ));
    let mut malformed = admissions.clone();
    let wrong = malformed.get_mut("stt_model.lm_head.weight").unwrap();
    wrong.recipe = DerivedWeightRecipe::source("unrelated.weight", TensorSelection::Full);
    wrong.output = wrong.recipe.infer(source.as_ref()).unwrap();
    assert!(matches!(
        PreparedBackboneParameters::from_parameters(args(), source.clone(), &malformed),
        Err(BackbonePreparationError::Catalog(_))
    ));
    assert_eq!(source.source_diagnostics().unwrap().physical_reads, 0);
    let config = VoiceChatConfig::from_json(include_str!(
        "../../eredu-architectures/src/voicechat/fixtures/official-inference-config.json"
    ))
    .unwrap();
    let fusion = ChannelFusion::from_config(&config.channels).unwrap();
    let function = || {
        let mut head = FunctionHead::<Backend>::new(
            4,
            24,
            LinearFormatSpec::unscaled(LinearFormat::Dense).unwrap(),
            &stream,
        )
        .unwrap();
        head.visit_parameters_mut(&mut Bind {
            fixture: &fixture,
            stream: &stream,
            weights: BTreeMap::new(),
        });
        head
    };
    let run = |runtime: &mut Reference| {
        let mut state = state();
        let mut head = function();
        (0..3)
            .map(|step| {
                let audio =
                    MlxTensor::from_f32_slice(&[0.12, -0.31, 0.08, 0.42], &[1, 1, 4], &stream)
                        .unwrap();
                let agent = MlxTensor::from_i32_slice(&[3 + step], &[1, 1], &stream).unwrap();
                let function = MlxTensor::from_i32_slice(&[9 - step], &[1, 1], &stream).unwrap();
                let request = if step == 0 {
                    BackboneRequest::Prompt {
                        tokens: &[5, 7],
                        policy: PromptPolicy {
                            repeat: 1,
                            max_tokens: 4,
                        },
                    }
                } else {
                    BackboneRequest::Audio {
                        projected_audio: &audio,
                        agent_tokens: &agent,
                        function_tokens: &function,
                    }
                };
                let output = execute(
                    &mut OrdinaryBackbone::new(runtime),
                    &mut head,
                    &fusion,
                    request,
                    &mut state,
                    &stream,
                )
                .unwrap();
                (
                    [
                        output.normalized.to_f32_vec(&stream).unwrap(),
                        output.logits.agent.to_f32_vec(&stream).unwrap(),
                        output.logits.function.to_f32_vec(&stream).unwrap(),
                    ],
                    snapshot(&mut state),
                )
            })
            .collect::<Vec<_>>()
    };
    let expected = run(&mut reference);
    assert!(expected[2]
        .0
        .iter()
        .flatten()
        .any(|value| value.abs() > 1e-4));
    for (policy, device_budget) in [
        (LayerWeightResidency::FullyResident, total_bytes),
        (
            LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
                OffloadConfig::new(Some(host_device_bytes), Some(1 << 20), 1).unwrap(),
            )),
            host_device_bytes,
        ),
        (
            LayerWeightResidency::DenseDiskStream(
                DenseDiskStreamLoadOptions::new(disk_device_bytes, 0, 0, 0).unwrap(),
            ),
            disk_device_bytes,
        ),
    ] {
        let mut runtime =
            materialize_voicechat_backbone(&prepared, &policy, &weights_stream, &stream).unwrap();
        assert_eq!(runtime.static_lease_count(), 1);
        assert!(matches!(
            runtime
                .checkpoint_store()
                .source_metadata("unrelated.weight"),
            Err(StoreError::UnauthorizedTensor { .. })
        ));
        let mut state = state();
        let mut head = function();
        for (step, (expected_output, expected_state)) in expected.iter().enumerate() {
            let audio =
                MlxTensor::from_f32_slice(&[0.12, -0.31, 0.08, 0.42], &[1, 1, 4], &stream).unwrap();
            let agent = MlxTensor::from_i32_slice(&[3 + step as i32], &[1, 1], &stream).unwrap();
            let function = MlxTensor::from_i32_slice(&[9 - step as i32], &[1, 1], &stream).unwrap();
            let request = if step == 0 {
                BackboneRequest::Prompt {
                    tokens: &[5, 7],
                    policy: PromptPolicy {
                        repeat: 1,
                        max_tokens: 4,
                    },
                }
            } else {
                BackboneRequest::Audio {
                    projected_audio: &audio,
                    agent_tokens: &agent,
                    function_tokens: &function,
                }
            };
            let output = execute(
                &mut OrdinaryBackbone::new(&mut runtime),
                &mut head,
                &fusion,
                request,
                &mut state,
                &stream,
            )
            .unwrap();
            let actual = [
                output.normalized.to_f32_vec(&stream).unwrap(),
                output.logits.agent.to_f32_vec(&stream).unwrap(),
                output.logits.function.to_f32_vec(&stream).unwrap(),
            ];
            assert_eq!(
                &actual, expected_output,
                "step {step}: shared hidden and both heads"
            );
            assert_eq!(
                &snapshot(&mut state),
                expected_state,
                "step {step}: recurrent and attention state"
            );
            assert_eq!(state.offset(), 4 + step as i32);
        }
        let report = runtime.residency_report().unwrap();
        assert_eq!(
            report
                .units()
                .iter()
                .map(|unit| unit.expected_bytes())
                .sum::<u64>(),
            total_bytes,
            "all physical parameters are charged once",
        );
        let pinned = report
            .units()
            .iter()
            .filter(|unit| unit.device_pins() > 0)
            .collect::<Vec<_>>();
        if policy.is_fully_resident() {
            assert_eq!(
                pinned.iter().map(|unit| unit.expected_bytes()).sum::<u64>(),
                total_bytes,
                "resident execution retains every unit's native ownership",
            );
        } else {
            assert_eq!(
                pinned.len(),
                1,
                "only the static vocabulary unit remains leased"
            );
            assert_eq!(pinned[0].expected_bytes(), static_bytes);
        }
        assert!(report.weight_store().physical_reads > 0);
        assert!(
            report
                .offload()
                .peak_resident_bytes()
                .get(MemoryTier::Device)
                <= device_budget
        );
        if let Some(disk) = runtime.dense_stream_report().unwrap() {
            assert_eq!(disk.prefill_forwards(), 1);
            assert_eq!(disk.decode_forwards(), 2);
        }
    }
}
