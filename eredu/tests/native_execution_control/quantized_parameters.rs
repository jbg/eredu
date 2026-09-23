use super::*;
use eredu_core::{intervention::InterventionDtype, parameters::*};
use std::collections::BTreeMap;

type TensorBytes = (String, Vec<usize>, Vec<u8>);
pub(super) fn write_tensors(root: &Path, tensors: &BTreeMap<String, TensorBytes>) {
    let mut data: Vec<u8> = vec![];
    let mut header = serde_json::Map::new();
    for (name, (dtype, shape, bytes)) in tensors {
        let start = data.len();
        data.extend(bytes);
        header.insert(
            name.clone(),
            serde_json::json!({"dtype":dtype,"shape":shape,"data_offsets":[start,data.len()]}),
        );
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut output = (header.len() as u64).to_le_bytes().to_vec();
    output.extend(header);
    output.extend(data);
    std::fs::write(root.join("model.safetensors"), output).unwrap();
}

pub(super) fn packed_fixture(tied: bool) -> (Fixture, Fixture, BTreeMap<String, Vec<f32>>) {
    packed_fixture_format(tied, false)
}

fn packed_fixture_format(
    tied: bool,
    mxfp4: bool,
) -> (Fixture, Fixture, BTreeMap<String, Vec<f32>>) {
    let packed = fixture(false);
    let reference = fixture(false);
    let mut config = serde_json::json!({"model_type":"qwen2", "hidden_size":32,
        "num_hidden_layers":2, "intermediate_size":64, "num_attention_heads":4,
        "num_key_value_heads":2, "rms_norm_eps":0.00001, "vocab_size":64,
        "eos_token_id":[], "max_position_embeddings":1024, "tie_word_embeddings":tied,
        "quantization_config":{"group_size":32,"bits":4,
            "mode": if mxfp4 { "mxfp4" } else { "affine" }}});
    std::fs::write(
        packed.0.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let resolved = eredu_architectures::configuration::resolve_model_config(&config).unwrap();
    let plan = resolved.architecture.checkpoint();
    let mut stored = BTreeMap::new();
    let mut dense = BTreeMap::new();
    let mut values = BTreeMap::new();
    let scale = |row: usize, group: usize| {
        if mxfp4 {
            2.0_f32.powi(((row + group) % 3) as i32 - 6)
        } else {
            ((row + group) % 3 + 1) as f32 / 64.0
        }
    };
    for tensor in plan.common_tensors.iter().chain(
        plan.layout_groups
            .iter()
            .filter_map(|group| group.variants.first())
            .flat_map(|variant| &variant.tensors),
    ) {
        if tensor.requirement != eredu_checkpoint::schema::TensorRequirement::Required {
            continue;
        }
        let count = tensor.shape.iter().product::<usize>();
        if matches!(
            tensor.dtype,
            eredu_checkpoint::schema::StoredDtypeConstraint::Exact(
                eredu_checkpoint::StoredDtype::U32
            )
        ) {
            let rows = tensor.shape[0];
            let cols = tensor.shape[1] * 8;
            let mut codes = vec![0_u32; count];
            let mut decoded = Vec::with_capacity(rows * cols);
            for row in 0..rows {
                for col in 0..cols {
                    let code = ((row * 7 + col * 3 + 5) % 16) as u32;
                    codes[row * tensor.shape[1] + col / 8] |= code << (col % 8 * 4);
                    let value = if mxfp4 {
                        let table = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
                        table[(code % 8) as usize] * if code < 8 { 1.0 } else { -1.0 }
                    } else {
                        code as f32 - 8.0
                    };
                    decoded.push(value * scale(row, col / 32));
                }
            }
            stored.insert(
                tensor.key.clone(),
                (
                    "U32".into(),
                    tensor.shape.clone(),
                    codes.iter().flat_map(|code| code.to_le_bytes()).collect(),
                ),
            );
            dense.insert(
                tensor.key.clone(),
                (
                    if mxfp4 { "BF16" } else { "F32" }.into(),
                    vec![rows, cols],
                    if mxfp4 {
                        // Native MXFP4 embedding/dequantization defaults to BF16.
                        // Every fixture value is exact in that dtype. Retain it
                        // in the independent model until an edit replaces it.
                        decoded
                            .iter()
                            .flat_map(|v| {
                                assert_eq!(v.to_bits() & 65535, 0);
                                ((v.to_bits() >> 16) as u16).to_le_bytes()
                            })
                            .collect()
                    } else {
                        decoded.iter().flat_map(|v| v.to_le_bytes()).collect()
                    },
                ),
            );
            values.insert(tensor.key.clone(), decoded);
        } else if mxfp4 && tensor.key.ends_with(".scales") {
            stored.insert(
                tensor.key.clone(),
                (
                    "U8".into(),
                    tensor.shape.clone(),
                    (0..count)
                        .map(|i| 121 + ((i / tensor.shape[1] + i % tensor.shape[1]) % 3) as u8)
                        .collect(),
                ),
            );
        } else {
            let data: Vec<f32> = (0..count)
                .map(|i| {
                    if tensor.key.ends_with(".scales") {
                        scale(i / tensor.shape[1], i % tensor.shape[1])
                    } else if tensor.key.ends_with(".biases") {
                        -8.0 * scale(i / tensor.shape[1], i % tensor.shape[1])
                    } else if tensor.key.contains("norm") {
                        1.0
                    } else {
                        (((i * 17 + tensor.key.len() * 7) % 101) as f32 - 50.0) * 0.003
                    }
                })
                .collect();
            let bytes = (
                "F32".into(),
                tensor.shape.clone(),
                data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            );
            stored.insert(tensor.key.clone(), bytes.clone());
            if tensor.role != eredu_checkpoint::schema::TensorRole::Companion {
                dense.insert(tensor.key.clone(), bytes);
                values.insert(tensor.key.clone(), data);
            }
        }
    }
    assert!(stored.values().any(|(dtype, _, _)| dtype == "U32"));
    write_tensors(&packed.0, &stored);
    config
        .as_object_mut()
        .unwrap()
        .remove("quantization_config");
    std::fs::write(
        reference.0.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    write_tensors(&reference.0, &dense);
    (packed, reference, values)
}

fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!((a - b).abs() <= 3e-5 + 3e-5 * b.abs(), "{a} != {b}");
    }
}

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run with --no-default-features --features mlx"
)]
fn native_quantized_component_masks_match_independent_dense_reference() {
    for tied in [false, true] {
        verify_packed_component_masks(tied, false, eredu_core::ResidencyPlan::FullyResident);
    }
}

fn verify_packed_component_masks(tied: bool, mxfp4: bool, residency: eredu_core::ResidencyPlan) {
    let (root, reference, _) = packed_fixture_format(tied, mxfp4);
    let run = |fixture| {
        super::components::component_masks_with_residency(
            fixture,
            "model.layers.0.attention.channels",
            "model.layers.0.feed_forward.units",
            residency.clone(),
        )
    };
    let (baseline, masked) = run(root);
    let (expected_baseline, expected_masked) = run(reference);
    for (actual, expected) in [(baseline, expected_baseline), (masked, expected_masked)] {
        assert_eq!(
            actual.keys().collect::<Vec<_>>(),
            expected.keys().collect::<Vec<_>>()
        );
        for (path, values) in actual {
            close(&values, &expected[&path]);
        }
    }
}

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run with --no-default-features --features mlx"
)]
fn native_quantized_effective_queries_and_edits_preserve_packed_originals() {
    for tied in [false, true] {
        verify_quantized_parameter_edits(tied, eredu_core::ResidencyPlan::FullyResident);
    }
}

pub(super) fn verify_quantized_parameter_edits(tied: bool, residency: eredu_core::ResidencyPlan) {
    verify_packed_parameter_edits(tied, residency, false, LocalDevice::Cpu);
}

fn verify_packed_parameter_edits(
    tied: bool,
    residency: eredu_core::ResidencyPlan,
    mxfp4: bool,
    device: LocalDevice,
) {
    let (root, reference, expected_values) = packed_fixture_format(tied, mxfp4);
    verify_parameter_fixture(
        ParameterFixture {
            root,
            reference,
            expected_values,
            source_file: "model.safetensors",
            promote_reference: mxfp4,
        },
        residency,
        device,
    );
}

pub(super) struct ParameterFixture {
    pub root: Fixture,
    pub reference: Fixture,
    pub expected_values: BTreeMap<String, Vec<f32>>,
    pub source_file: &'static str,
    pub promote_reference: bool,
}

pub(super) fn verify_parameter_fixture(
    fixture: ParameterFixture,
    residency: eredu_core::ResidencyPlan,
    device: LocalDevice,
) {
    let ParameterFixture {
        root,
        reference,
        expected_values,
        source_file,
        promote_reference,
    } = fixture;
    let source_path = root.0.join(source_file);
    let checkpoint = if source_file.ends_with(".gguf") {
        &source_path
    } else {
        &root.0
    };
    let source_bytes = std::fs::read(&source_path).unwrap();
    let execution =
        ExecutionPlan::fully_resident(local_device_plan(device).unwrap()).with_residency(residency);
    let (mut model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), checkpoint, &execution)
            .unwrap()
            .into_parts();
    let (mut expected, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &reference.0, &execution)
            .unwrap()
            .into_parts();
    let limits = CaptureUsage {
        captures: 256,
        retained_bytes: 512 << 20,
        host_bytes: 32 << 20,
        encoded_bytes: 32 << 20,
    };
    let facts = model.parameter_discovery().unwrap();
    for p in &facts.parameters {
        if let Some(expected) = expected_values.get(&p.id) {
            assert!(p.supported, "{}: {}", p.id, p.condition);
            let full = model
                .query_parameter(
                    &facts.identity,
                    &p.id,
                    ParameterRegion {
                        starts: vec![0; p.shape.len()],
                        shape: p.shape.clone(),
                    },
                    limits,
                )
                .unwrap();
            assert_eq!(&full.values, expected, "effective {}", p.id);
        } else {
            assert!(
                !p.supported,
                "companions require coordinated weight semantics"
            );
        }
    }
    super::parameters::verify_native_projections(&mut model, limits);
    let prefix = [1, 2, 5, 7];
    let baseline = super::parameters::parameter_logits(&mut model, &prefix).0;
    close(
        &baseline,
        &super::parameters::parameter_logits(&mut expected, &prefix).0,
    );
    let baseline_decode = super::parameters::parameter_decode_logits(&mut model, &prefix, false);
    super::parameters::compare_parameter_decodes(
        &baseline_decode,
        &super::parameters::parameter_decode_logits(&mut expected, &prefix, false),
        None,
    );
    drop(expected);
    let mut specs = vec![
        ("model.layers.0.self_attn.q_proj.weight", false),
        ("model.layers.0.self_attn.k_proj.weight", false),
        ("model.layers.0.self_attn.v_proj.weight", false),
        ("model.layers.0.self_attn.o_proj.weight", true),
        ("model.layers.0.mlp.gate_proj.weight", false),
        ("model.layers.0.mlp.up_proj.weight", false),
        ("model.layers.0.mlp.down_proj.weight", true),
        ("model.layers.1.self_attn.v_proj.weight", false),
        ("model.layers.1.mlp.gate_proj.weight", false),
        ("model.layers.1.mlp.down_proj.weight", true),
    ];
    specs.push(("model.embed_tokens.weight", false));
    let edits: Vec<_> = specs
        .into_iter()
        .enumerate()
        .map(|(i, (id, column))| {
            let p = facts.parameters.iter().find(|p| p.id == id).unwrap();
            let region = if column {
                ParameterRegion {
                    starts: vec![0, 1],
                    shape: vec![p.shape[0], 1],
                }
            } else {
                ParameterRegion {
                    starts: vec![1, 0],
                    shape: vec![1, p.shape[1]],
                }
            };
            ParameterEdit {
                id: format!("e{i}"),
                parameter: id.into(),
                parameter_shape: p.shape.clone(),
                dtype: InterventionDtype::Float32,
                update: ParameterUpdate::Add {
                    values: (0..region.shape.iter().product::<u64>())
                        .map(|j| ((j + i as u64) % 5) as f32 * 0.12 - 0.19)
                        .collect(),
                },
                region,
            }
        })
        .collect();
    if promote_reference {
        promote_reference_matrices(&reference.0, &edits);
    }
    super::parameters::edit_reference(&reference.0, &edits);
    let overlay = model
        .admit_parameter_overlay(ParameterOverlayPlan {
            schema_version: PARAMETER_SCHEMA_VERSION,
            base_identity: facts.identity,
            provenance: "independent packed decode and F32 parameter edits".into(),
            edits,
        })
        .unwrap();
    let usage = model.parameter_discovery().unwrap().usage;
    assert!(matches!(
        model.activate_parameter_overlay(&overlay, usage),
        Err(ParameterError::Budget(_))
    ));
    assert_eq!(model.parameter_discovery().unwrap().usage, usage);
    model.activate_parameter_overlay(&overlay, limits).unwrap();
    let (mut expected, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &reference.0, &execution)
            .unwrap()
            .into_parts();
    let changed = super::parameters::parameter_logits_mode(&mut model, &prefix, true);
    assert_eq!(changed.1.as_deref(), Some(overlay.identity()));
    assert_ne!(changed.0, baseline);
    close(
        &changed.0,
        &super::parameters::parameter_logits(&mut expected, &prefix).0,
    );
    let other = [7, 5, 2, 1];
    close(
        &super::parameters::parameter_logits(&mut model, &other).0,
        &super::parameters::parameter_logits(&mut expected, &other).0,
    );
    super::parameters::verify_native_projections(&mut model, limits);
    super::parameters::compare_parameter_decodes(
        &super::parameters::parameter_decode_logits(&mut model, &prefix, true),
        &super::parameters::parameter_decode_logits(&mut expected, &prefix, false),
        Some(overlay.identity()),
    );
    let active = model.parameter_discovery().unwrap();
    model.remove_parameter_overlay(&active.identity).unwrap();
    assert_eq!(
        baseline_decode,
        super::parameters::parameter_decode_logits(&mut model, &prefix, false)
    );
    assert_eq!(
        super::parameters::parameter_logits(&mut model, &prefix).0,
        baseline
    );
    assert_eq!(std::fs::read(&source_path).unwrap(), source_bytes);
}

// Independent checkpoint preparation: only edited BF16 matrices become F32.
fn promote_reference_matrices(root: &Path, edits: &[ParameterEdit]) {
    let bytes = std::fs::read(root.join("model.safetensors")).unwrap();
    let length = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&bytes[8..8 + length]).unwrap();
    let tensors = header
        .into_iter()
        .map(|(name, tensor)| {
            let start = 8 + length + tensor["data_offsets"][0].as_u64().unwrap() as usize;
            let end = 8 + length + tensor["data_offsets"][1].as_u64().unwrap() as usize;
            let mut dtype = tensor["dtype"].as_str().unwrap().to_owned();
            let mut values = bytes[start..end].to_vec();
            if edits.iter().any(|edit| edit.parameter == name) && dtype == "BF16" {
                values = values
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .flat_map(|word| {
                        let bits = u16::from_le_bytes(*word) as u32;
                        (bits << 16).to_le_bytes()
                    })
                    .collect();
                dtype = "F32".into();
            }
            (
                name,
                (
                    dtype,
                    serde_json::from_value(tensor["shape"].clone()).unwrap(),
                    values,
                ),
            )
        })
        .collect();
    write_tensors(root, &tensors);
}

pub(super) fn packed_residencies() -> [eredu_core::ResidencyPlan; 3] {
    [
        eredu_core::ResidencyPlan::FullyResident,
        eredu_core::ResidencyPlan::LayerwiseHost {
            device_layer_window: 1,
            device_budget_bytes: Some(8 << 20),
            host_budget_bytes: Some(8 << 20),
        },
        eredu_core::ResidencyPlan::DenseDiskStream {
            device_budget_bytes: 8 << 20,
            host_budget_bytes: 8 << 20,
            host_lookahead: 1,
            background_queue: 1,
        },
    ]
}

fn verify_mxfp4_parameter_lifecycle(device: LocalDevice) {
    for tied in [false, true] {
        for residency in packed_residencies() {
            verify_packed_parameter_edits(tied, residency, true, device);
        }
    }
}

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run with --no-default-features --features mlx"
)]
fn native_mxfp4_queries_and_overlays_preserve_packed_originals() {
    verify_mxfp4_parameter_lifecycle(LocalDevice::Cpu);
}

#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
#[ignore = "requires an accessible Metal device; run explicitly on a GPU worker"]
fn native_mxfp4_queries_and_overlays_metal() {
    verify_mxfp4_parameter_lifecycle(LocalDevice::Accelerator(0));
}

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run with --no-default-features --features mlx"
)]
fn native_mxfp4_component_masks_match_independent_reference() {
    for tied in [false, true] {
        for residency in packed_residencies() {
            verify_packed_component_masks(tied, true, residency);
        }
    }
}
