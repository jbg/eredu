use super::*;
use eredu_core::parameters::*;

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run with --no-default-features --features mlx"
)]
fn native_streamed_parameter_queries_follow_selected_bindings_and_preserve_forward_order() {
    let (packed, dense, values) = super::quantized_parameters::packed_fixture(false);
    let resident = ExecutionPlan::fully_resident(local_device_plan(LocalDevice::Cpu).unwrap());
    let (mut reference, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &dense.0, &resident)
            .unwrap()
            .into_parts();
    let expected = super::parameters::parameter_logits(&mut reference, &[0, 1, 2]);
    let limits = CaptureUsage {
        captures: 1024,
        retained_bytes: 1 << 30,
        host_bytes: 64 << 20,
        encoded_bytes: 64 << 20,
    };
    for root in [&dense, &packed] {
        for residency in [
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
        ] {
            let execution = resident.clone().with_residency(residency);
            let (mut model, _) = LoadedModel::load_execution_plan(
                &MlxBackendFactory::default(),
                &root.0,
                &execution,
            )
            .unwrap()
            .into_parts();
            let facts = model.parameter_discovery().unwrap();
            assert!(!facts.parameters.is_empty());
            assert_eq!(model.parameter_discovery().unwrap(), facts);
            // Reverse order enters the last layer before the first layer; a query
            // must not consume the disk stream's next-forward cursor.
            for parameter in facts
                .parameters
                .iter()
                .rev()
                .filter(|parameter| parameter.supported)
            {
                let region = ParameterRegion {
                    starts: vec![0; parameter.shape.len()],
                    shape: parameter.shape.clone(),
                };
                let before = model.parameter_discovery().unwrap().usage;
                assert!(matches!(
                    model.query_parameter(&facts.identity, &parameter.id, region.clone(), before),
                    Err(ParameterError::Budget(_))
                ));
                assert_eq!(model.parameter_discovery().unwrap().usage, before);
                let actual = model
                    .query_parameter(&facts.identity, &parameter.id, region, limits)
                    .unwrap();
                assert_eq!(actual.values, values[&parameter.id], "{}", parameter.id);
            }
            super::parameters::verify_native_projections(&mut model, limits);
            let actual = super::parameters::parameter_logits_mode(&mut model, &[0, 1, 2], true);
            assert_eq!(actual.1, None);
            for (actual, expected) in actual.0.iter().zip(&expected.0) {
                assert!((actual - expected).abs() <= 3e-5 + 3e-5 * expected.abs());
            }
            // Query after a completed forward, then start a different prefix.
            super::parameters::verify_native_projections(&mut model, limits);
            let actual = super::parameters::parameter_logits(&mut model, &[2, 0, 1]);
            let expected = super::parameters::parameter_logits(&mut reference, &[2, 0, 1]);
            for (actual, expected) in actual.0.iter().zip(&expected.0) {
                assert!((actual - expected).abs() <= 3e-5 + 3e-5 * expected.abs());
            }
        }
    }
}

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run with --no-default-features --features mlx"
)]
fn native_streamed_parameter_overlays_survive_reload_and_restore_packed_sources() {
    for tied in [false, true] {
        for residency in [
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
        ] {
            super::quantized_parameters::verify_quantized_parameter_edits(tied, residency);
        }
    }
}

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run with --no-default-features --features mlx"
)]
fn native_streamed_shared_parameter_overlays_cover_each_invocation() {
    for residency in [
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
    ] {
        super::parameters::verify_shared_parameter_edits(residency);
    }
}

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run with --no-default-features --features mlx"
)]
fn native_streamed_parameter_failure_does_not_publish_an_earlier_unit() {
    use eredu_core::intervention::InterventionDtype;
    for residency in [
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
    ] {
        let root = fixture(false);
        let execution = ExecutionPlan::fully_resident(local_device_plan(LocalDevice::Cpu).unwrap())
            .with_residency(residency);
        let limits = CaptureUsage {
            captures: 128,
            retained_bytes: 64 << 20,
            host_bytes: 8 << 20,
            encoded_bytes: 8 << 20,
        };
        let (mut original, _) =
            LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
                .unwrap()
                .into_parts();
        let facts = original.parameter_discovery().unwrap();
        let make_edit = |id: &str, value: f32| {
            let parameter = facts
                .parameters
                .iter()
                .find(|parameter| parameter.id == id)
                .unwrap();
            ParameterEdit {
                id: id.into(),
                parameter: id.into(),
                parameter_shape: parameter.shape.clone(),
                dtype: InterventionDtype::Float32,
                region: ParameterRegion {
                    starts: vec![0, 0],
                    shape: vec![1, parameter.shape[1]],
                },
                update: ParameterUpdate::Replace {
                    values: vec![value; parameter.shape[1] as usize],
                },
            }
        };
        let first = make_edit("model.layers.0.mlp.down_proj.weight", 0.2);
        let mut overflow = make_edit("model.layers.1.mlp.down_proj.weight", f32::MAX);
        drop(original);
        super::parameters::edit_reference(&root.0, &[overflow.clone()]);
        let bytes = std::fs::read(root.0.join("model.safetensors")).unwrap();
        let (mut model, _) =
            LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
                .unwrap()
                .into_parts();
        let facts = model.parameter_discovery().unwrap();
        let before = model
            .query_parameter(
                &facts.identity,
                &first.parameter,
                first.region.clone(),
                limits,
            )
            .unwrap();
        overflow.update = ParameterUpdate::Add {
            values: vec![f32::MAX; overflow.region.shape[1] as usize],
        };
        let authority = model
            .admit_parameter_overlay(ParameterOverlayPlan {
                schema_version: PARAMETER_SCHEMA_VERSION,
                base_identity: facts.identity.clone(),
                provenance: "cross-unit atomic rejection".into(),
                edits: vec![first.clone(), overflow],
            })
            .unwrap();
        let usage = model.parameter_discovery().unwrap().usage;
        let failure = model.activate_parameter_overlay(&authority, limits);
        assert!(
            matches!(&failure, Err(ParameterError::Invalid(_))),
            "{failure:?}"
        );
        let rejected = model.parameter_discovery().unwrap();
        assert_eq!(rejected.identity, facts.identity);
        assert_eq!(rejected.overlay_identity, None);
        assert!(rejected.usage.retained_bytes > usage.retained_bytes);
        assert_eq!(
            model
                .query_parameter(
                    &facts.identity,
                    &first.parameter,
                    first.region.clone(),
                    limits
                )
                .unwrap()
                .values,
            before.values
        );
        // A completed value rejection must leave the owner usable for another loan/publication.
        let valid = model
            .admit_parameter_overlay(ParameterOverlayPlan {
                schema_version: PARAMETER_SCHEMA_VERSION,
                base_identity: facts.identity.clone(),
                provenance: "retry after rejected result".into(),
                edits: vec![first.clone()],
            })
            .unwrap();
        let active = model.activate_parameter_overlay(&valid, limits).unwrap();
        assert_eq!(
            model
                .query_parameter(
                    &active.identity,
                    &first.parameter,
                    first.region.clone(),
                    limits
                )
                .unwrap()
                .values,
            first.update.values()
        );
        let restored = model.remove_parameter_overlay(&active.identity).unwrap();
        assert_eq!(
            model
                .query_parameter(&restored.identity, &first.parameter, first.region, limits)
                .unwrap()
                .values,
            before.values
        );
        assert_eq!(
            std::fs::read(root.0.join("model.safetensors")).unwrap(),
            bytes
        );
    }
}

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run with --no-default-features --features mlx"
)]
fn native_streamed_parameter_metadata_uses_prepared_bfloat16_dtype() {
    use eredu_core::intervention::InterventionDtype;
    let root = fixture(false);
    let bytes = std::fs::read(root.0.join("model.safetensors")).unwrap();
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
    let mut encoded = std::collections::BTreeMap::new();
    let mut widened = std::collections::BTreeMap::new();
    let bf16_bits =
        |value: f32| ((value.to_bits() + 0x7fff + ((value.to_bits() >> 16) & 1)) >> 16) as u16;
    for (id, tensor) in header.as_object().unwrap() {
        assert_eq!(tensor["dtype"], "F32");
        let start = 8 + header_len + tensor["data_offsets"][0].as_u64().unwrap() as usize;
        let end = 8 + header_len + tensor["data_offsets"][1].as_u64().unwrap() as usize;
        let bits: Vec<u16> = bytes[start..end]
            .chunks_exact(4)
            .map(|bytes| bf16_bits(f32::from_le_bytes(bytes.try_into().unwrap())))
            .collect();
        widened.insert(
            id.clone(),
            bits.iter()
                .map(|bits| f32::from_bits((*bits as u32) << 16))
                .collect::<Vec<_>>(),
        );
        encoded.insert(
            id.clone(),
            (
                "BF16".into(),
                tensor["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|n| n.as_u64().unwrap() as usize)
                    .collect(),
                bits.iter().flat_map(|bits| bits.to_le_bytes()).collect(),
            ),
        );
    }
    super::quantized_parameters::write_tensors(&root.0, &encoded);
    let limits = CaptureUsage {
        captures: 128,
        retained_bytes: 64 << 20,
        host_bytes: 8 << 20,
        encoded_bytes: 8 << 20,
    };
    for residency in [
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
    ] {
        let execution = ExecutionPlan::fully_resident(local_device_plan(LocalDevice::Cpu).unwrap())
            .with_residency(residency);
        let (mut model, _) =
            LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
                .unwrap()
                .into_parts();
        let facts = model.parameter_discovery().unwrap();
        for parameter in &facts.parameters {
            assert!(parameter.supported, "{}", parameter.id);
            assert_eq!(
                parameter.dtype,
                Some(InterventionDtype::Bfloat16),
                "{}",
                parameter.id
            );
            let actual = model
                .query_parameter(
                    &facts.identity,
                    &parameter.id,
                    ParameterRegion {
                        starts: vec![0; parameter.shape.len()],
                        shape: parameter.shape.clone(),
                    },
                    limits,
                )
                .unwrap();
            assert_eq!(actual.values, widened[&parameter.id]);
        }
        let parameter = facts
            .parameters
            .iter()
            .find(|parameter| parameter.id == "model.layers.1.mlp.down_proj.weight")
            .unwrap();
        let region = ParameterRegion {
            starts: vec![0, 0],
            shape: vec![1, parameter.shape[1]],
        };
        let edit = ParameterEdit {
            id: "bf16-update".into(),
            parameter: parameter.id.clone(),
            parameter_shape: parameter.shape.clone(),
            dtype: InterventionDtype::Bfloat16,
            region: region.clone(),
            update: ParameterUpdate::Add {
                values: vec![0.1; parameter.shape[1] as usize],
            },
        };
        let overlay = model
            .admit_parameter_overlay(ParameterOverlayPlan {
                schema_version: PARAMETER_SCHEMA_VERSION,
                base_identity: facts.identity.clone(),
                provenance: "independent BF16 bits and rounding".into(),
                edits: vec![edit],
            })
            .unwrap();
        let active = model.activate_parameter_overlay(&overlay, limits).unwrap();
        let actual = model
            .query_parameter(&active.identity, &parameter.id, region.clone(), limits)
            .unwrap();
        let expected: Vec<_> = widened[&parameter.id][..parameter.shape[1] as usize]
            .iter()
            .map(|value| f32::from_bits((bf16_bits(value + 0.1) as u32) << 16))
            .collect();
        assert_eq!(actual.values, expected);
        assert_eq!(actual.dtype, InterventionDtype::Bfloat16);
        let restored = model.remove_parameter_overlay(&active.identity).unwrap();
        assert_eq!(
            model
                .query_parameter(&restored.identity, &parameter.id, region, limits)
                .unwrap()
                .values,
            widened[&parameter.id][..parameter.shape[1] as usize]
        );
    }
}
