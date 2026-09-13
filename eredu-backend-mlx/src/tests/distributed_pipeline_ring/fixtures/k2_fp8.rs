// Encoded expert banks with independent scalar bytes and block-scale equations.
fn k2_fp8_phase(name: &str) -> usize {
    name.bytes()
        .fold(0u32, |h, b| h.wrapping_mul(31).wrapping_add(b.into())) as usize
}

fn k2_fp8_scale(name: &str, block: usize) -> f32 {
    (1 + (k2_fp8_phase(name) + block * 3) % 5) as f32 / 128.
}

fn k2_fp8_code(name: &str, index: usize) -> u8 {
    let phase = k2_fp8_phase(name);
    (0x28 + (index * 13 + phase) % 32) as u8 | if (index + phase) % 3 == 0 { 128 } else { 0 }
}

fn k2_fp8_scalar(name: &str, shape: &[usize], index: usize) -> f32 {
    let code = k2_fp8_code(name, index);
    let exponent = ((code >> 3) & 15) as i32;
    let mantissa = (code & 7) as f32;
    let value = if exponent == 0 {
        mantissa * 2_f32.powi(-9)
    } else {
        (1. + mantissa / 8.) * 2_f32.powi(exponent - 7)
    };
    let block = (index / shape[1] / 128) * shape[1].div_ceil(128) + index % shape[1] / 128;
    value * if code & 128 == 0 { 1. } else { -1. } * k2_fp8_scale(name, block)
}

fn write_k2_fp8_fixture(directory: &Path, partial: bool) {
    write_k2_fp8_fixture_with_dense_tail(directory, partial, false);
}

fn write_k2_fp8_fixture_with_dense_tail(directory: &Path, partial: bool, dense_tail: bool) {
    write_k2_fp8_fixture_with_tails(directory, partial, dense_tail, false);
}

fn write_k2_fp8_fixture_with_tails(
    directory: &Path,
    partial: bool,
    dense_tail: bool,
    fused_tail: bool,
) {
    write_k2_fp8_fixture_geometry(directory, partial, dense_tail, fused_tail, false);
}

fn write_k2_fp8_fixture_with_attention_tails(directory: &Path) {
    write_k2_fp8_fixture_geometry(directory, true, true, true, true);
}

fn write_k2_fp8_fixture_with_non_f32_scales(directory: &Path) {
    write_k2_fp8_fixture_with_attention_tails(directory);
    let path = directory.join("model.safetensors");
    let bytes = std::fs::read(&path).unwrap();
    let source = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let tensors = source
        .tensors()
        .into_iter()
        .map(|(name, view)| {
            let (dtype, bytes) = if name.ends_with("_scale_inv") {
                assert_eq!(view.dtype(), Dtype::F32);
                // Keep every fused bank's companions homogeneous, while PP
                // crosses between both admitted floating scale encodings.
                let bfloat = name.starts_with("model.layers.1.");
                let bytes = view
                    .data()
                    .chunks_exact(4)
                    .flat_map(|bytes| {
                        let value = f32::from_le_bytes(bytes.try_into().unwrap());
                        let bits = if bfloat {
                            let encoded = half::bf16::from_f32(value);
                            assert_eq!(encoded.to_f32(), value);
                            encoded.to_bits()
                        } else {
                            let encoded = half::f16::from_f32(value);
                            assert_eq!(encoded.to_f32(), value);
                            encoded.to_bits()
                        };
                        bits.to_le_bytes()
                    })
                    .collect();
                (if bfloat { Dtype::BF16 } else { Dtype::F16 }, bytes)
            } else {
                (view.dtype(), view.data().to_vec())
            };
            (name, view.shape().to_vec(), dtype, bytes)
        })
        .collect::<Vec<_>>();
    serialize_to_file(
        tensors.iter().map(|(name, shape, dtype, bytes)| {
            (
                name.as_str(),
                TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            )
        }),
        None,
        &path,
    )
    .unwrap();
}

fn write_k2_fp8_fixture_geometry(
    directory: &Path,
    partial: bool,
    dense_tail: bool,
    fused_tail: bool,
    attention_tail: bool,
) {
    let source: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../eredu-architectures/tests/fixtures/k2_horizon/reference.json"
    )))
    .unwrap();
    let mut config = source["mova"]["config"].clone();
    config["hidden_size"] = if partial { 130 } else { 128 }.into();
    config["intermediate_size"] = if dense_tail { 259 } else { 128 }.into();
    config["moe_intermediate_size"] = if fused_tail {
        259
    } else if partial {
        384
    } else {
        256
    }
    .into();
    if attention_tail {
        config["num_attention_heads"] = 66.into();
        config["num_key_value_heads"] = 33.into();
    }
    config["num_experts"] = 3.into();
    config["num_experts_per_tok"] = 3.into();
    config["num_shared_experts"] = 1.into();
    config["query_key_norm"] = attention_tail.into();
    config["vocab_size"] = 64.into();
    config["eos_token_id"] = serde_json::json!([]);
    let args = eredu_architectures::k2_horizon::model_args_from_config_value(&config).unwrap();
    let ignored: Vec<_> = eredu_architectures::k2_horizon::parameter_shapes(&args, false)
        .unwrap()
        .into_iter()
        .filter(|(name, _)| {
            !name.contains(".mlp.experts.")
            && !(dense_tail && name.starts_with("model.layers.0.mlp.") && name.ends_with(".weight"))
            // Keep the last attention layer dense so PP crosses a change in
            // local head count, while both edited attention layers remain FP8.
            && !(attention_tail && name.contains(".self_attn.")
                && !name.starts_with("model.layers.2.self_attn.")
                && (["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"]
                    .iter().any(|field| name.ends_with(field))
                    || name.contains(".v_experts.") && name.ends_with(".weight")))
        })
        .map(|(name, _)| name)
        .collect();
    config["quantization_config"] = serde_json::json!({"quant_method":"fp8",
        "activation_scheme":"dynamic", "weight_block_size":[128,128], "ignored_layers":ignored});
    let resolved = eredu_architectures::configuration::resolve_model_config(&config).unwrap();
    let tensors = resolved
        .architecture
        .checkpoint()
        .common_tensors
        .iter()
        .map(|tensor| {
            let name = &tensor.key;
            let shape = tensor.shape.clone();
            let count = shape.iter().product::<usize>();
            let (dtype, bytes): (Dtype, Vec<u8>) =
                if tensor.role == eredu_checkpoint::schema::TensorRole::Companion {
                    let owner = name.strip_suffix("_scale_inv").unwrap();
                    (
                        Dtype::F32,
                        (0..count)
                            .flat_map(|i| k2_fp8_scale(owner, i).to_le_bytes())
                            .collect(),
                    )
                } else if matches!(
                    tensor.dtype,
                    eredu_checkpoint::schema::StoredDtypeConstraint::Exact(
                        eredu_checkpoint::StoredDtype::F8E4M3
                    )
                ) {
                    (
                        Dtype::F8_E4M3,
                        (0..count).map(|i| k2_fp8_code(name, i)).collect(),
                    )
                } else {
                    let phase = k2_fp8_phase(name);
                    (
                        Dtype::F32,
                        (0..count)
                            .flat_map(|i| {
                                (if name.contains("norm") {
                                    0.9 + (i % 7) as f32 * 0.03
                                } else {
                                    (((i * 17 + phase) % 101) as f32 - 50.) * 0.003
                                })
                                .to_le_bytes()
                            })
                            .collect(),
                    )
                };
            (name.clone(), shape, dtype, bytes)
        })
        .collect::<Vec<_>>();
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    serialize_to_file(
        tensors.iter().map(|(name, shape, dtype, bytes)| {
            (
                name.as_str(),
                TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            )
        }),
        None,
        &directory.join("model.safetensors"),
    )
    .unwrap();
}

fn k2_fp8_parameter_phase_limits(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    used: eredu_core::capture::CaptureUsage,
) -> eredu_core::capture::CaptureUsage {
    use eredu_core::{capture::CaptureUsage, DistributedBackend, DistributedSession};
    let participants = MlxBackend::distributed_session(runtime.session())
        .unwrap()
        .descriptor()
        .world_size() as u64;
    // Global catalog delivery reserves every participant's metadata. This is
    // cumulative work, not simultaneous storage. Give each test phase a finite
    // per-participant allowance on top of all earlier queries and edits.
    used.checked_add(
        CaptureUsage {
            captures: 100_000,
            retained_bytes: 512 << 20,
            host_bytes: 1 << 30,
            encoded_bytes: 512 << 20,
        }
        .checked_mul(participants)
        .unwrap(),
    )
    .unwrap()
}

fn verify_k2_fp8_bank_query_replay(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    reference: &mut ModelRuntime<MlxBackend<'_>>,
    owned_entries: usize,
) {
    use eredu_core::parameters::*;
    let facts = MlxBackend::parameter_discovery(runtime).unwrap();
    let competing = facts
        .parameters
        .iter()
        .filter(|p| p.id.ends_with(".v_experts.weight") && p.access().query)
        .collect::<Vec<_>>();
    assert_eq!(competing.len(), 2);
    let limits = k2_fp8_parameter_phase_limits(runtime, facts.usage);
    let mut previous_usage = facts.usage;
    for _ in 0..2 {
        runtime.reset().unwrap();
        reference.reset().unwrap();
        let before = runtime.session().parameter_bank_report().unwrap().unwrap();
        // Query every expert and output row so all EP/TP owners load a competing
        // MoVA member, even when the fixture never routes tokens to that member.
        // The cache is bounded by one layer's local FFN members or one token's
        // local value-member batch, whichever is larger. Competing value sources
        // force eviction; the subsequent forward must reload FFN members.
        for parameter in &competing {
            let queried = MlxBackend::query_parameter(
                runtime,
                &facts.identity,
                &parameter.id,
                ParameterRegion {
                    starts: vec![0, 0, 0],
                    shape: vec![parameter.shape[0], parameter.shape[1], 1],
                },
                limits,
            )
            .unwrap();
            assert!(
                previous_usage.exceeded(queried.usage).is_none(),
                "reset and bank eviction must preserve cumulative parameter charges"
            );
            assert!(queried.usage.captures > previous_usage.captures);
            previous_usage = queried.usage;
        }
        for prefill in [true, false, false] {
            let expected = parameter_fixture_forward(reference, prefill);
            assert_parameter_predictions(
                parameter_fixture_forward(runtime, prefill),
                &expected,
                1e-4,
            );
        }
        let after = runtime.session().parameter_bank_report().unwrap().unwrap();
        assert_eq!(after.owned_entries(), owned_entries);
        assert_eq!(after.banks().len(), 2);
        let ffn = eredu_architectures::k2_horizon::ExpertBank::FeedForward.id();
        let before_ffn = before.banks().get(&ffn).unwrap();
        let after_ffn = after.banks().get(&ffn).unwrap();
        assert!(
            after_ffn.bulk().device().misses() + after_ffn.incremental().device().misses()
                > before_ffn.bulk().device().misses() + before_ffn.incremental().device().misses(),
            "forward must reload FFN members after competing bank queries"
        );
        assert!(
            after.bulk().device().evictions() + after.incremental().device().evictions()
                > before.bulk().device().evictions() + before.incremental().device().evictions(),
            "query/forward replay must evict competing bank members"
        );
    }
    runtime.reset().unwrap();
    reference.reset().unwrap();
}

fn verify_k2_fp8_source_parameters(runtime: &mut ModelRuntime<MlxBackend<'_>>) {
    use eredu_core::parameters::*;
    let facts = MlxBackend::parameter_discovery(runtime).unwrap();
    let limits = k2_fp8_parameter_phase_limits(runtime, facts.usage);
    // Dense and attention tails cross scale blocks and TP ownership boundaries.
    // Decode directly from the fixture equation, independently of native queries.
    for parameter in facts.parameters.iter().filter(|p| {
        (p.id.starts_with("model.layers.0.mlp.") || p.id.contains(".self_attn."))
            && p.shape.len() == 2
            && p.access().query
            && matches!(
                p.input_transform,
                ProjectionInputTransform::BlockFp8E4m3 { .. }
            )
    }) {
        let shape = [parameter.shape[0] as usize, parameter.shape[1] as usize];
        let rows = [0, 127, 128, 255, 256, shape[0] - 1]
            .into_iter()
            .filter(|row| *row < shape[0])
            .collect::<std::collections::BTreeSet<_>>();
        for row in rows {
            let expected = (0..shape[1])
                .map(|column| k2_fp8_scalar(&parameter.id, &shape, row * shape[1] + column))
                .collect::<Vec<_>>();
            let region = ParameterRegion {
                starts: vec![row as u64, 0],
                shape: vec![1, shape[1] as u64],
            };
            let actual = MlxBackend::query_parameter(
                runtime,
                &facts.identity,
                &parameter.id,
                region.clone(),
                limits,
            )
            .unwrap();
            assert_eq!(
                actual.values, expected,
                "matrix FP8 tail {}, row={row}",
                parameter.id
            );
            let coefficients = (0..shape[1])
                .map(|i| (i % 7) as f32 * 0.3 - 0.7)
                .collect::<Vec<_>>();
            let projection = MlxBackend::project_parameter(
                runtime,
                &facts.identity,
                &parameter.id,
                ParameterProjection {
                    region,
                    axis: 1,
                    directions: 1,
                    coefficients: coefficients.clone(),
                },
                limits,
            )
            .unwrap();
            let expected = expected
                .iter()
                .zip(&coefficients)
                .map(|(x, y)| *x as f64 * *y as f64)
                .sum::<f64>();
            assert!((projection.values[0] as f64 - expected).abs() <= 2e-6 + 2e-6 * expected.abs());
        }
    }
    let banks = facts
        .parameters
        .iter()
        .filter(|p| {
            p.shape.len() == 3
                && p.access().query
                && (p.id.contains(".mlp.experts.")
                    || p.id.ends_with(".v_experts.weight")
                        && matches!(
                            p.input_transform,
                            ProjectionInputTransform::BlockFp8E4m3 { .. }
                        ))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        banks
            .iter()
            .filter(|p| p.id.contains(".mlp.experts."))
            .count(),
        4
    );
    for bank in banks {
        assert!(matches!(
            bank.input_transform,
            ProjectionInputTransform::BlockFp8E4m3 { .. }
        ));
        let read = bank.id.ends_with("gate_up_proj");
        let value_bank = bank.id.ends_with(".v_experts.weight");
        let root = bank.id.rsplit_once('.').unwrap().0;
        let rows = bank.shape[1] as usize;
        let columns = bank.shape[2] as usize;
        let selected_rows = if value_bank {
            [0, 127, 128, rows - 1]
                .into_iter()
                .filter(|row| *row < rows)
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect()
        } else if read {
            vec![0, rows / 2 - 1, rows / 2, rows - 1]
        } else {
            vec![0, rows - 1]
        };
        for expert in [0, bank.shape[0] as usize - 1] {
            for row in &selected_rows {
                let (field, source_row, source_rows) = if read {
                    (
                        if *row < rows / 2 {
                            "gate_proj"
                        } else {
                            "up_proj"
                        },
                        row % (rows / 2),
                        rows / 2,
                    )
                } else {
                    ("down_proj", *row, rows)
                };
                let source = if value_bank {
                    format!("{root}.{expert}.weight")
                } else {
                    format!("{root}.{expert}.{field}.weight")
                };
                let expected = (0..columns)
                    .map(|col| {
                        k2_fp8_scalar(&source, &[source_rows, columns], source_row * columns + col)
                    })
                    .collect::<Vec<_>>();
                let region = ParameterRegion {
                    starts: vec![expert as u64, *row as u64, 0],
                    shape: vec![1, 1, columns as u64],
                };
                let actual = MlxBackend::query_parameter(
                    runtime,
                    &facts.identity,
                    &bank.id,
                    region.clone(),
                    limits,
                )
                .unwrap();
                assert_eq!(
                    actual.values, expected,
                    "independent FP8 decode {}, expert={expert}, row={row}",
                    bank.id
                );
                let coefficients = (0..2 * columns)
                    .map(|i| (i % 7) as f32 * 0.3 - 0.7)
                    .collect::<Vec<_>>();
                let projection = MlxBackend::project_parameter(
                    runtime,
                    &facts.identity,
                    &bank.id,
                    ParameterProjection {
                        region,
                        axis: 2,
                        directions: 2,
                        coefficients: coefficients.clone(),
                    },
                    limits,
                )
                .unwrap();
                for direction in 0..2 {
                    let expected = expected
                        .iter()
                        .zip(&coefficients[direction * columns..(direction + 1) * columns])
                        .map(|(x, y)| *x as f64 * *y as f64)
                        .sum::<f64>();
                    assert!(
                        (projection.values[direction] as f64 - expected).abs()
                            <= 2e-6 + 2e-6 * expected.abs()
                    );
                }
            }
        }
    }
}

// Applies public logical edits to independent source tensors, without using the
// runtime overlay implementation, its decoder, or its coordinate projection.
fn verify_k2_fp8_independent_overlay(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    checkpoint: &Path,
    stream: &Stream,
    edits: &[eredu_core::parameters::ParameterEdit],
) {
    use eredu_core::parameters::{
        ParameterBackend as _, ParameterUpdate, ProjectionInputTransform,
    };
    let bytes = std::fs::read(checkpoint.join("model.safetensors")).unwrap();
    let source = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let mut tensors = source
        .tensors()
        .into_iter()
        .map(|(name, view)| {
            (
                name,
                (view.dtype(), view.shape().to_vec(), view.data().to_vec()),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(checkpoint.join("config.json")).unwrap()).unwrap();
    let mut promoted = std::collections::BTreeSet::new();
    for edit in edits {
        if !promoted.insert(&edit.parameter) {
            continue;
        }
        if edit.parameter_shape.len() == 2 {
            let (dtype, shape, data) = tensors.get_mut(&edit.parameter).unwrap();
            if *dtype == Dtype::F8_E4M3 {
                *data = (0..shape.iter().product())
                    .flat_map(|i| k2_fp8_scalar(&edit.parameter, shape, i).to_le_bytes())
                    .collect();
                *dtype = Dtype::F32;
                tensors.remove(&format!("{}_scale_inv", edit.parameter));
                config["quantization_config"]["ignored_layers"]
                    .as_array_mut()
                    .unwrap()
                    .push(edit.parameter.clone().into());
            }
            continue;
        }
        let fields: &[&str] = if edit.parameter.ends_with(".gate_up_proj") {
            &["gate_proj", "up_proj"]
        } else if edit.parameter.ends_with(".down_proj") {
            &["down_proj"]
        } else if edit.parameter.ends_with(".v_experts.weight") {
            &[""]
        } else {
            continue;
        };
        let root = edit.parameter.rsplit_once('.').unwrap().0;
        for expert in 0..edit.parameter_shape[0] {
            for field in fields {
                let name = if field.is_empty() {
                    format!("{root}.{expert}.weight")
                } else {
                    format!("{root}.{expert}.{field}.weight")
                };
                let (dtype, shape, data) = tensors.get_mut(&name).unwrap();
                if *dtype == Dtype::F32 {
                    continue;
                }
                assert_eq!(*dtype, Dtype::F8_E4M3);
                *data = (0..shape.iter().product())
                    .flat_map(|i| k2_fp8_scalar(&name, shape, i).to_le_bytes())
                    .collect();
                *dtype = Dtype::F32;
                tensors.remove(&format!("{name}_scale_inv"));
                config["quantization_config"]["ignored_layers"]
                    .as_array_mut()
                    .unwrap()
                    .push(name.into());
            }
        }
    }
    for edit in edits {
        let ParameterUpdate::Add { values } = &edit.update else {
            panic!("fixture additive update")
        };
        for (linear, delta) in values.iter().enumerate() {
            let mut remainder = linear;
            let mut coordinates = vec![0; edit.region.shape.len()];
            for axis in (0..coordinates.len()).rev() {
                coordinates[axis] = edit.region.starts[axis] as usize
                    + remainder % edit.region.shape[axis] as usize;
                remainder /= edit.region.shape[axis] as usize;
            }
            let (name, coordinates) = if coordinates.len() == 2 {
                (edit.parameter.clone(), coordinates)
            } else {
                assert_eq!(coordinates.len(), 3);
                let (expert, row, col) = (coordinates[0], coordinates[1], coordinates[2]);
                if let Some(root) = edit.parameter.strip_suffix(".gate_up_proj") {
                    let width = edit.parameter_shape[1] as usize / 2;
                    (
                        format!(
                            "{root}.{expert}.{}.weight",
                            if row < width { "gate_proj" } else { "up_proj" }
                        ),
                        vec![row % width, col],
                    )
                } else if let Some(root) = edit.parameter.strip_suffix(".down_proj") {
                    (format!("{root}.{expert}.down_proj.weight"), vec![row, col])
                } else if let Some(root) = edit.parameter.strip_suffix(".v_experts.weight") {
                    (format!("{root}.v_experts.{expert}.weight"), vec![row, col])
                } else {
                    panic!("source mapping {}", edit.parameter)
                }
            };
            let (dtype, shape, data) = tensors.get_mut(&name).unwrap();
            assert_eq!(*dtype, Dtype::F32);
            let offset = coordinates
                .iter()
                .zip(shape.iter())
                .fold(0, |offset, (coordinate, extent)| {
                    offset * extent + coordinate
                })
                * 4;
            let before = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
            data[offset..offset + 4].copy_from_slice(&(before + delta).to_le_bytes());
        }
    }
    let independent = tempfile::tempdir().unwrap();
    std::fs::write(
        independent.path().join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    serialize_to_file(
        tensors.iter().map(|(name, (dtype, shape, bytes))| {
            (
                name.as_str(),
                TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            )
        }),
        None,
        &independent.path().join("model.safetensors"),
    )
    .unwrap();
    let weights_stream = fixture_weights_stream(stream);
    let backend = MlxBackend::new(stream, &weights_stream);
    let prepared = load_model(&backend, independent.path(), MlxLoadRequest::default()).unwrap();
    let mut reference = ModelRuntime::from_prepared(backend, prepared).unwrap();
    let current = MlxBackend::parameter_discovery(runtime).unwrap();
    for bank in current
        .parameters
        .iter()
        .filter(|p| p.id.contains(".mlp.experts.") && p.shape.len() == 3 && p.access().query)
    {
        assert_eq!(
            matches!(
                bank.input_transform,
                ProjectionInputTransform::BlockFp8E4m3 { .. }
            ),
            bank.id.starts_with("model.layers.2."),
            "global input transform {}",
            bank.id
        );
    }
    runtime.reset().unwrap();
    for prefill in [true, false, false] {
        let expected = parameter_fixture_forward(&mut reference, prefill);
        assert_parameter_predictions(parameter_fixture_forward(runtime, prefill), &expected, 1e-4);
    }
    runtime.reset().unwrap();
    assert_eq!(
        bytes,
        std::fs::read(checkpoint.join("model.safetensors")).unwrap()
    );
}
