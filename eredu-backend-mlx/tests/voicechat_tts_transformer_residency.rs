//! Causal and character EAR-TTS transformers use exact bounded native providers.
#[path = "voicechat_tts_transformer_residency/partition.rs"]
mod partition;
#[path = "voicechat_tts_transformer_residency/pipeline.rs"]
mod pipeline;
use std::{collections::BTreeMap, sync::Arc};

use eredu_architectures::{
    artifact_preparation::PhysicalArtifactId,
    voicechat::{
        artifact::ParameterAdmission,
        checkpoint::Component,
        config::GemmaConfig,
        tts_transformer::TransformerConfig,
        tts_transformer_layered::{
            CausalTransformerExecution, TransformerError, TransformerExecution,
        },
        tts_transformer_preparation::PreparedTtsTransformerBindings,
    },
};
use eredu_backend_mlx::{
    backend::nn::shared::MlxNeuralBackend,
    backend::runtime::{
        cache::state::{MlxKeyValueLayerState, MlxKeyValueState},
        checkpoint::binding::binding_bytes,
    },
    native::materialize_voicechat_tts_transformer,
    MlxTensor,
};
use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    store::{SafetensorsWeightStore, SharedCheckpointSource, TensorSelection},
};
use eredu_core::residency::{MemoryTier, OffloadConfig};
use eredu_nn::{AttentionCache, AttentionRequest, Tensor};
use eredu_runtime::{DenseDiskStreamLoadOptions, LayerWeightResidency, LayerwiseLoadOptions};
use safemlx::{Array, Device, DeviceType, Stream};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use serde_json::Value;

fn floats(value: &Value) -> Vec<f32> {
    value["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}
fn tensor(value: &Value) -> MlxTensor {
    let shape: Vec<i32> = value["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_i64().unwrap() as i32)
        .collect();
    MlxTensor::from_array(Array::from_slice(&floats(value), &shape))
}
fn geometry(value: &Value, causal: bool) -> GemmaConfig {
    let n = |key: &str| value[key].as_u64().unwrap() as usize;
    GemmaConfig {
        hidden_size: n("hidden_size"),
        intermediate_size: n("intermediate_size"),
        num_hidden_layers: n("num_hidden_layers"),
        num_attention_heads: n("num_attention_heads"),
        num_key_value_heads: n("num_key_value_heads"),
        head_dim: n("head_dim"),
        hidden_activation: value["hidden_activation"].as_str().unwrap().into(),
        max_position_embeddings: n("max_position_embeddings"),
        rms_norm_eps: value["rms_norm_eps"].as_f64().unwrap(),
        rope_theta: value["rope_theta"].as_f64().unwrap(),
        rope_local_base_freq: value["rope_local_base_freq"].as_f64(),
        attention_bias: value["attention_bias"].as_bool().unwrap(),
        query_pre_attn_scalar: n("query_pre_attn_scalar"),
        sliding_window: n("sliding_window"),
        layer_types: serde_json::from_value(value["layer_types"].clone()).unwrap(),
        final_logit_softcapping: value["final_logit_softcapping"].as_f64(),
        attn_logit_softcapping: value["attn_logit_softcapping"].as_f64(),
        rope_scaling: None,
        causal,
    }
}
fn checkpoint(fixture: &Value, f16: bool, path: &std::path::Path) -> SharedCheckpointSource {
    let mut values: Vec<_> = ["gemma3", "t5gemma"]
        .into_iter()
        .flat_map(|part| {
            fixture[part]["parameters"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(name, value)| {
                    let shape: Vec<usize> = value["shape"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_u64().unwrap() as usize)
                        .collect();
                    let bytes: Vec<u8> = if f16 {
                        floats(value)
                            .into_iter()
                            .flat_map(|v| half::f16::from_f32(v).to_bits().to_le_bytes())
                            .collect()
                    } else {
                        floats(value)
                            .into_iter()
                            .flat_map(f32::to_le_bytes)
                            .collect()
                    };
                    (name.clone(), shape, bytes)
                })
        })
        .collect();
    values.push((
        "unrelated.weight".into(),
        vec![1],
        vec![0; if f16 { 2 } else { 4 }],
    ));
    let dtype = if f16 { Dtype::F16 } else { Dtype::F32 };
    serialize_to_file(
        values.iter().map(|(name, shape, bytes)| {
            (
                name.as_str(),
                TensorView::new(dtype, shape.clone(), bytes).unwrap(),
            )
        }),
        None,
        path,
    )
    .unwrap();
    Arc::new(SafetensorsWeightStore::open(path).unwrap())
}
fn prepared(
    config: TransformerConfig,
    source: SharedCheckpointSource,
) -> PreparedTtsTransformerBindings {
    let parameters: BTreeMap<_, _> = source
        .source_keys()
        .into_iter()
        .map(|name| {
            let metadata = source.source_metadata(&name).unwrap();
            let recipe = DerivedWeightRecipe::Source {
                key: name.clone(),
                selection: TensorSelection::Full,
            };
            let output = recipe.infer(source.as_ref()).unwrap();
            let peak = recipe.peak_materialization_bytes(source.as_ref()).unwrap()
                + metadata.encoded_byte_len;
            (
                name.clone(),
                ParameterAdmission {
                    linear_format: eredu_checkpoint::LinearFormat::Dense,
                    name,
                    component: Component::Tts,
                    artifact: PhysicalArtifactId::Primary,
                    output,
                    stored_bytes: metadata.encoded_byte_len,
                    conversion_workspace_bytes: 0,
                    materialization_peak_bytes: peak,
                    recipe,
                },
            )
        })
        .collect();
    let reads = source.source_diagnostics().unwrap().physical_reads;
    let bindings =
        PreparedTtsTransformerBindings::from_parameters(config, source.clone(), &parameters)
            .unwrap();
    assert_eq!(
        source.source_diagnostics().unwrap().physical_reads,
        reads,
        "cold binding projection reads no payload"
    );
    bindings
}
fn state_values(state: &MlxKeyValueState, stream: &Stream) -> Vec<(Vec<i32>, Vec<f32>)> {
    state
        .retained_arrays()
        .into_iter()
        .map(|array| {
            let tensor = MlxTensor::from_array(array.clone());
            (tensor.shape().to_vec(), tensor.to_f32_vec(stream).unwrap())
        })
        .collect()
}

// Fail after the first ordinary decoder unit completed. The runtime must retire
// its consumer event and abort the next lease before this model can be reused.
struct RejectAppend {
    cache: MlxKeyValueLayerState,
    reject: bool,
}
impl eredu_runtime::RuntimeLayerState<MlxNeuralBackend> for RejectAppend {
    type RetainedValues<'a> = <MlxKeyValueLayerState as eredu_runtime::RuntimeLayerState<
        MlxNeuralBackend,
    >>::RetainedValues<'a>;
    fn retained_values(&self) -> Self::RetainedValues<'_> {
        eredu_runtime::RuntimeLayerState::<MlxNeuralBackend>::retained_values(&self.cache)
    }
}

impl AttentionCache<MlxTensor> for RejectAppend {
    fn offset(&self) -> i32 {
        self.cache.offset()
    }
    fn max_size(&self) -> Option<i32> {
        self.cache.max_size()
    }
    fn update_for_attention(
        &mut self,
        keys: MlxTensor,
        values: MlxTensor,
        stream: &Stream,
    ) -> Result<(MlxTensor, MlxTensor), eredu_nn::Error> {
        if self.reject {
            return Err(eredu_nn::Error::backend(
                "injected second-layer cache append failure",
            ));
        }
        self.cache.update_for_attention(keys, values, stream)
    }
    fn attention(
        &mut self,
        request: AttentionRequest<'_, MlxTensor>,
        stream: &Stream,
    ) -> Result<MlxTensor, eredu_nn::Error> {
        self.cache.attention(request, stream)
    }
}

fn compare(values: &[f32], expected: &Value, fixture: &Value) {
    let expected = floats(expected);
    assert_eq!(values.len(), expected.len());
    let absolute = fixture["comparison"]["absolute_tolerance"]
        .as_f64()
        .unwrap() as f32;
    let relative = fixture["comparison"]["relative_tolerance"]
        .as_f64()
        .unwrap() as f32;
    for (index, (actual, expected)) in values.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= absolute + relative * expected.abs(),
            "{index}: {actual} versus pinned HF {expected}"
        );
    }
}

#[test]
fn native_tts_transformer_residency_preserves_masks_caches_and_casts_after_abort() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::TRANSFORMER_REFERENCE_JSON).unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let weights = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let directory = tempfile::tempdir().unwrap();
    for f16 in [false, true] {
        let source = checkpoint(
            &fixture,
            f16,
            &directory.path().join(if f16 {
                "f16.safetensors"
            } else {
                "f32.safetensors"
            }),
        );
        for causal in [true, false] {
            let data = &fixture[if causal { "gemma3" } else { "t5gemma" }];
            let dimensions = geometry(
                if causal {
                    &data["config"]
                } else {
                    &data["config"]["encoder"]
                },
                causal,
            );
            let config = TransformerConfig::new(&dimensions).unwrap();
            let (_, layout, restricted, units) =
                prepared(config.clone(), source.clone()).into_parts();
            let expected_keys = data["parameters"].as_object().unwrap();
            assert_eq!(restricted.source_keys().len(), expected_keys.len());
            assert!(
                restricted
                    .source_keys()
                    .iter()
                    .all(|key| expected_keys.contains_key(key)),
                "other transformer and unrelated tensors must be inaccessible"
            );
            let learned_elements: u64 = expected_keys
                .values()
                .map(|v| v["values"].as_array().unwrap().len() as u64)
                .sum();
            let largest = units
                .iter()
                .map(|unit| binding_bytes(unit).unwrap())
                .max()
                .unwrap();
            // The shared disk streamer retains a two-unit transfer window.
            // Price that window within each group, excluding the separate norm.
            let disk_depth = LayerWeightResidency::DenseDiskStream(
                DenseDiskStreamLoadOptions::new(u64::MAX, 0, 0, 0).unwrap(),
            )
            .device_depth(units.len());
            let disk_budget = (0..layout.group_count())
                .flat_map(|group| {
                    let bytes = units[layout.group_range(group).unwrap()]
                        .iter()
                        .map(|unit| binding_bytes(unit).unwrap())
                        .collect::<Vec<_>>();
                    bytes
                        .windows(disk_depth.min(bytes.len()))
                        .map(|window| window.iter().sum::<u64>())
                        .collect::<Vec<_>>()
                })
                .max()
                .unwrap();
            let policies = [
                LayerWeightResidency::FullyResident,
                LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
                    OffloadConfig::new(Some(largest), Some(1 << 20), 1).unwrap(),
                )),
                LayerWeightResidency::DenseDiskStream(
                    DenseDiskStreamLoadOptions::new(disk_budget, 0, 0, 0).unwrap(),
                ),
            ];
            let mut expected = None;
            for residency in policies {
                let (mut model, metadata) = materialize_voicechat_tts_transformer(
                    prepared(config.clone(), source.clone()),
                    residency,
                    &weights,
                    &stream,
                )
                .unwrap();
                assert_eq!(metadata.static_device_bytes(), 0);
                assert_eq!(model.static_lease_count(), 0);
                assert_eq!(
                    metadata.layer_parameter_bytes(),
                    learned_elements * 4,
                    "execution storage includes F16 to F32 conversion"
                );
                let mut actual = Vec::new();
                if causal {
                    let layout = eredu_architectures::decoder::state_layout(&config).unwrap();
                    let mut caches = MlxKeyValueState::device(layout.clone()).unwrap();
                    let layers: &[MlxKeyValueLayerState] = caches.as_ref();
                    let mut rejected: Vec<_> = layers
                        .iter()
                        .enumerate()
                        .map(|(index, cache)| RejectAppend {
                            cache: cache.clone(),
                            reject: index == 1,
                        })
                        .collect();
                    assert!(matches!(
                        model.forward_causal(
                            &tensor(&data["chunks"][0]["input"]),
                            &mut rejected,
                            &stream
                        ),
                        Err(TransformerError::Compute(_))
                    ));
                    assert!(
                        rejected[0].offset() > 0,
                        "failure follows a completed decoder unit"
                    );
                    assert_eq!(
                        caches.offset(),
                        0,
                        "unpublished failed branches never change canonical state"
                    );
                    drop(rejected);
                    for execution in ["full", "chunks"] {
                        caches.clear().unwrap();
                        for (index, step) in data[execution].as_array().unwrap().iter().enumerate()
                        {
                            let output = model
                                .forward_causal(&tensor(&step["input"]), caches.as_mut(), &stream)
                                .unwrap_or_else(|error| {
                                    panic!(
                                        "{residency:?} F16={f16} {execution} step{index}: {error}"
                                    )
                                });
                            let values = output.to_f32_vec(&stream).unwrap();
                            if !f16 {
                                compare(&values, &step["output"], &fixture);
                            }
                            actual.push((values, state_values(&caches, &stream), caches.offset()));
                        }
                    }
                } else {
                    let mask = tensor(&data["attention_mask"])
                        .equal_i32(1, &stream)
                        .unwrap();
                    for batch in data["batches"].as_array().unwrap() {
                        let output = model
                            .forward_encoder(&tensor(&batch["input"]), &mask, &stream)
                            .unwrap_or_else(|error| {
                                panic!("{residency:?} F16={f16} character encoder: {error}")
                            });
                        let values = output.to_f32_vec(&stream).unwrap();
                        if !f16 {
                            compare(&values, &batch["output"], &fixture);
                        }
                        actual.push((values, Vec::new(), 0));
                    }
                }
                if let Some(expected) = &expected {
                    assert_eq!(
                        &actual, expected,
                        "residency parity causal={causal} F16={f16}"
                    );
                } else {
                    expected = Some(actual);
                }
                let report = model.residency_report().unwrap();
                assert!(report.weight_store().physical_reads > 0);
                if !residency.is_fully_resident() {
                    let budget = if matches!(residency, LayerWeightResidency::DenseDiskStream(_)) {
                        disk_budget
                    } else {
                        largest
                    };
                    assert!(
                        report
                            .offload()
                            .peak_resident_bytes()
                            .get(MemoryTier::Device)
                            <= budget
                    );
                }
                if matches!(residency, LayerWeightResidency::DenseDiskStream(_)) {
                    let streaming = model.dense_stream_report().unwrap().unwrap();
                    assert!(streaming.prefill_forwards() + streaming.decode_forwards() > 0);
                }
            }
        }
    }
}

const TTS_TP_WORKER: &str = "EREDU_TTS_TRANSFORMER_RING_RANK";
const TTS_TP_DIRECTORY: &str = "EREDU_TTS_TRANSFORMER_RING_DIRECTORY";

// The pinned tiny oracle has Q2/KV1. Duplicating its K/V projection preserves
// its GQA equation while making the shared KV-group planner select TP2.
fn duplicated_kv_fixture(fixture: &Value) -> Value {
    let mut expanded = fixture.clone();
    for family in ["gemma3", "t5gemma"] {
        let part = &mut expanded[family];
        let config = if family == "gemma3" {
            &mut part["config"]
        } else {
            &mut part["config"]["encoder"]
        };
        assert_eq!(config["num_key_value_heads"], 1);
        config["num_key_value_heads"] = 2.into();
        for (name, parameter) in part["parameters"].as_object_mut().unwrap() {
            if name.ends_with(".k_proj.weight") || name.ends_with(".v_proj.weight") {
                let rows = parameter["shape"][0].as_u64().unwrap();
                parameter["shape"][0] = (rows * 2).into();
                let values = parameter["values"].as_array_mut().unwrap();
                values.extend(values.clone());
            }
        }
    }
    expanded
}

fn compare_native_cache(
    actual: &MlxKeyValueState,
    expected: &MlxKeyValueState,
    fixture: &Value,
    stream: &Stream,
) {
    assert_eq!(actual.offset(), expected.offset());
    let actual = state_values(actual, stream);
    let expected = state_values(expected, stream);
    assert_eq!(actual.len(), expected.len());
    let absolute = fixture["comparison"]["absolute_tolerance"]
        .as_f64()
        .unwrap() as f32;
    let relative = fixture["comparison"]["relative_tolerance"]
        .as_f64()
        .unwrap() as f32;
    for ((shape, values), (expected_shape, expected_values)) in actual.iter().zip(&expected) {
        assert_eq!(shape, expected_shape);
        for (value, expected) in values.iter().zip(expected_values) {
            assert!(
                (value - expected).abs() <= absolute + relative * expected.abs(),
                "rank-local cached value {value} differs from ordinary {expected}"
            );
        }
    }
}

#[test]
fn native_tts_transformer_tp2_worker() {
    let Ok(rank) = std::env::var(TTS_TP_WORKER) else {
        return;
    };
    let rank: usize = rank.parse().unwrap();
    let parallel = safemlx::distributed::init(true, safemlx::distributed::Backend::Ring).unwrap();
    assert_eq!(parallel.size(), 2);
    assert_eq!(parallel.rank() as usize, rank);
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let weights = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::TRANSFORMER_REFERENCE_JSON).unwrap();
    let expanded = duplicated_kv_fixture(&fixture);
    // The parent owns the containing directory even if a worker is killed.
    let directory = tempfile::tempdir_in(std::env::var_os(TTS_TP_DIRECTORY).unwrap()).unwrap();
    let ordinary_source = checkpoint(
        &fixture,
        false,
        &directory.path().join("ordinary.safetensors"),
    );
    let selected_source = checkpoint(&expanded, false, &directory.path().join("tp2.safetensors"));
    for causal in [true, false] {
        let family = if causal { "gemma3" } else { "t5gemma" };
        let data = &fixture[family];
        let configuration = |part: &Value| {
            TransformerConfig::new(&geometry(
                if causal {
                    &part["config"]
                } else {
                    &part["config"]["encoder"]
                },
                causal,
            ))
            .unwrap()
        };
        let ordinary_config = configuration(data);
        let selected_config = configuration(&expanded[family]);
        let largest = prepared(selected_config.clone(), selected_source.clone())
            .unit_bindings()
            .iter()
            .map(|unit| binding_bytes(unit).unwrap())
            .max()
            .unwrap();
        for residency in [
            LayerWeightResidency::FullyResident,
            LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
                OffloadConfig::new(Some(largest), Some(1 << 20), 1).unwrap(),
            )),
            LayerWeightResidency::DenseDiskStream(
                DenseDiskStreamLoadOptions::new(largest * 2, 0, 0, 0).unwrap(),
            ),
        ] {
            let (mut ordinary, _) = materialize_voicechat_tts_transformer(
                prepared(ordinary_config.clone(), ordinary_source.clone()),
                LayerWeightResidency::FullyResident,
                &weights,
                &stream,
            )
            .unwrap();
            let (mut selected, metadata) =
                eredu_backend_mlx::native::materialize_voicechat_tts_transformer_parallel(
                    prepared(selected_config.clone(), selected_source.clone()),
                    &parallel,
                    residency,
                    &weights,
                    &stream,
                )
                .unwrap();
            assert_eq!(metadata.static_device_bytes(), 0);
            if causal {
                // Every rank owns one of two identical KV heads, matching the
                // original oracle's sole head and retained local/window state.
                let layout = selected.state_layout().clone();
                assert_eq!(
                    &layout,
                    &eredu_architectures::decoder::state_layout(&ordinary_config).unwrap(),
                    "local KV geometry preserves the original one-head oracle"
                );
                for execution in ["full", "chunks"] {
                    let mut ordinary_cache = MlxKeyValueState::device(layout.clone()).unwrap();
                    let mut selected_cache = MlxKeyValueState::device(layout.clone()).unwrap();
                    for step in data[execution].as_array().unwrap() {
                        let input = tensor(&step["input"]);
                        let expected = ordinary
                            .forward_causal(&input, ordinary_cache.as_mut(), &stream)
                            .unwrap();
                        compare(
                            &expected.to_f32_vec(&stream).unwrap(),
                            &step["output"],
                            &fixture,
                        );
                        let actual = selected
                            .forward_causal(&input, selected_cache.as_mut(), &stream)
                            .unwrap();
                        compare(
                            &actual.to_f32_vec(&stream).unwrap(),
                            &step["output"],
                            &fixture,
                        );
                        compare_native_cache(&selected_cache, &ordinary_cache, &fixture, &stream);
                    }
                }
            } else {
                let mask = tensor(&data["attention_mask"])
                    .equal_i32(1, &stream)
                    .unwrap();
                for batch in data["batches"].as_array().unwrap() {
                    let actual = selected
                        .forward_encoder(&tensor(&batch["input"]), &mask, &stream)
                        .unwrap();
                    compare(
                        &actual.to_f32_vec(&stream).unwrap(),
                        &batch["output"],
                        &fixture,
                    );
                }
            }
            if !residency.is_fully_resident() {
                let report = selected.residency_report().unwrap();
                let limit = if matches!(residency, LayerWeightResidency::DenseDiskStream(_)) {
                    largest * 2
                } else {
                    largest
                };
                assert!(
                    report
                        .offload()
                        .peak_resident_bytes()
                        .get(MemoryTier::Device)
                        <= limit
                );
            }
        }
    }
    partition::exercise(
        &fixture,
        &expanded,
        ordinary_source,
        selected_source,
        Some(&parallel),
        &weights,
        &stream,
    );
}

#[test]
#[ignore = "spawns two local MLX Ring ranks and opens loopback sockets; run explicitly"]
fn native_tts_transformer_tp2_preserves_hf_outputs_masks_and_cached_state() {
    use std::{
        net::TcpListener,
        process::{Child, Command, Stdio},
        time::{Duration, Instant},
    };
    struct Children(Vec<Child>);
    impl Drop for Children {
        fn drop(&mut self) {
            for child in &mut self.0 {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
    let sockets = (0..2)
        .map(|_| TcpListener::bind(("127.0.0.1", 0)).unwrap())
        .collect::<Vec<_>>();
    let hosts = sockets
        .iter()
        .map(|socket| vec![format!("127.0.0.1:{}", socket.local_addr().unwrap().port())])
        .collect::<Vec<_>>();
    let directory = tempfile::tempdir().unwrap();
    let hostfile = directory.path().join("hosts.json");
    std::fs::write(&hostfile, serde_json::to_vec(&hosts).unwrap()).unwrap();
    drop(sockets);
    let mut children = Children(Vec::new());
    for rank in 0..2 {
        children.0.push(
            Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "native_tts_transformer_tp2_worker",
                    "--nocapture",
                ])
                .env(TTS_TP_WORKER, rank.to_string())
                .env(TTS_TP_DIRECTORY, directory.path())
                .env("MLX_RANK", rank.to_string())
                .env("MLX_HOSTFILE", &hostfile)
                .env_remove("MLX_RING_VERBOSE")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        );
    }
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let statuses = children
            .0
            .iter_mut()
            .map(|child| child.try_wait().unwrap())
            .collect::<Vec<_>>();
        if statuses.iter().all(Option::is_some) {
            break;
        }
        if Instant::now() >= deadline || statuses.iter().flatten().any(|status| !status.success()) {
            for child in &mut children.0 {
                let _ = child.kill();
            }
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    for (rank, child) in children.0.drain(..).enumerate() {
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "TP2 rank {rank}: {}\n{}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
