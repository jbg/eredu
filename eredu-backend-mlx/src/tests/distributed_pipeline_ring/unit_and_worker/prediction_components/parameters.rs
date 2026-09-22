fn assert_prediction_modules_idle(runtime: &ModelRuntime<MlxBackend<'_>>) {
    let report = runtime.session().residency_report().unwrap().unwrap();
    assert!(!report.unit_sources().is_empty());
    for id in report.unit_sources().keys() {
        let unit = report.units().iter().find(|unit| unit.id() == id).unwrap();
        assert_eq!(unit.device_pins(), 0, "prediction loan released for {id:?}");
        if unit.planned_tier() != eredu_core::residency::MemoryTier::Device {
            assert!(
                !unit.device_resident(),
                "prediction module evicted while idle: {id:?}"
            );
        }
    }
}

// Loaded global queries and coordinated target/prediction edits across replicas.
fn verify_partitioned_prediction_parameters(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    reference: &mut ModelRuntime<MlxBackend<'_>>,
    rank: usize,
    graph: &eredu_core::ArchitectureDescriptor,
    integer_buffers: &std::collections::BTreeSet<String>,
    run: impl Fn(&mut ModelRuntime<MlxBackend<'_>>, u8, [u32; 3], bool) -> PredictionResult,
) {
    use eredu_core::{capture::CaptureUsage, intervention::InterventionDtype, parameters::*};
    assert_prediction_modules_idle(runtime);
    let facts = MlxBackend::parameter_discovery(runtime).unwrap();
    let ordinary = MlxBackend::parameter_discovery(reference).unwrap();
    // Every public query charges the complete catalog exchange. Calibrate that
    // logical reservation on this actual topology, then allow the five queries
    // per editable matrix, overlay attempts, and discovery calls below. This is
    // cumulative work, not a simultaneous host allocation. Explicit rejection
    // trials still use the already consumed usage as their limit.
    let measured = MlxBackend::parameter_discovery(runtime).unwrap();
    let catalog_cost = CaptureUsage {
        captures: measured
            .usage
            .captures
            .checked_sub(facts.usage.captures)
            .unwrap(),
        retained_bytes: measured
            .usage
            .retained_bytes
            .checked_sub(facts.usage.retained_bytes)
            .unwrap(),
        host_bytes: measured
            .usage
            .host_bytes
            .checked_sub(facts.usage.host_bytes)
            .unwrap(),
        encoded_bytes: measured
            .usage
            .encoded_bytes
            .checked_sub(facts.usage.encoded_bytes)
            .unwrap(),
    };
    let exchanges = u64::try_from(facts.parameters.len())
        .unwrap()
        .checked_mul(8)
        .unwrap()
        .checked_add(32)
        .unwrap();
    let limits = CaptureUsage {
        captures: 100_000,
        retained_bytes: 8 << 30,
        host_bytes: 16 << 30,
        encoded_bytes: 8 << 30,
    }
    .checked_add(catalog_cost.checked_mul(exchanges).unwrap())
    .unwrap();
    eprintln!("prediction parameter catalog rank={rank} cost={catalog_cost:?} exchanges={exchanges} limits={limits:?}");
    let embedding = &graph.component_readout.as_ref().unwrap().embedding_weight;
    let mut edits = Vec::new();
    let mut originals = Vec::new();
    let mut shared = std::collections::BTreeSet::new();
    let mut fp8_rows_verified = 0;
    let v4_source_bytes = std::env::var_os("EREDU_RING_DEEPSEEK_FP8").map(|_| {
        std::fs::read(
            Path::new(&std::env::var_os(CHECKPOINT_DIR).unwrap()).join("model.safetensors"),
        )
        .unwrap()
    });
    let v4_source = v4_source_bytes
        .as_ref()
        .map(|bytes| safetensors::SafeTensors::deserialize(bytes).unwrap());
    let mut v4_fp8_rows_verified = 0;
    let mut mixed_rows_verified = 0;
    // Use effective identities, not checkpoint-name reconstruction. Every slot
    // includes fusion, norm, prediction heads, MLA, routed and shared gated FFNs.
    assert_eq!(facts.parameters.len(), ordinary.parameters.len());
    for parameter in &ordinary.parameters {
        let loaded = facts
            .parameters
            .iter()
            .find(|p| p.id == parameter.id)
            .unwrap();
        assert_eq!(loaded.access(), parameter.access());
        if !parameter.access().query {
            // Canonical packed companions are accessed through the effective
            // matrix and cannot be independently edited in this fixture. Exact
            // I32 checkpoint buffers are routing indices, outside floating edits.
            assert!(
                integer_buffers.contains(&parameter.id)
                    || [".scales", ".biases", "_scales", "_biases", "_scale_inv"]
                        .iter()
                        .any(|suffix| parameter.id.ends_with(suffix)),
                "unsupported effective parameter {}: {}",
                parameter.id,
                parameter.condition
            );
            assert!(!parameter.access().replacement);
            continue;
        }
        assert!(loaded.supported, "{}: {}", loaded.id, loaded.condition);
        assert_eq!(loaded.shape, parameter.shape);
        if !shared.insert(loaded.shared_id.clone()) {
            continue;
        }
        let mut region = ParameterRegion {
            starts: vec![0; parameter.shape.len()],
            shape: vec![1; parameter.shape.len()],
        };
        *region.shape.last_mut().unwrap() = *parameter.shape.last().unwrap();
        if &parameter.id == embedding {
            region.starts[0] = 1;
        }
        let actual = MlxBackend::query_parameter(
            runtime,
            &facts.identity,
            &parameter.id,
            region.clone(),
            limits,
        )
        .unwrap();
        let expected = MlxBackend::query_parameter(
            reference,
            &ordinary.identity,
            &parameter.id,
            region.clone(),
            limits,
        )
        .unwrap();
        assert_eq!(
            actual.values, expected.values,
            "global query {}",
            parameter.id
        );
        if std::env::var_os("EREDU_RING_PREDICTION_FP8").is_some() {
            fp8_rows_verified += usize::from(verify_qwen_fp8_source_row(
                &parameter.id,
                &parameter.shape,
                &region,
                &actual.values,
            ));
        }
        if let Some(source) = &v4_source {
            if matches!(
                parameter.input_transform,
                ProjectionInputTransform::BlockFp8E4m3 { .. }
            ) || std::env::var_os("EREDU_RING_DEEPSEEK_MIXED_FP8").is_some()
                && parameter.id.contains(".switch_mlp.")
            {
                verify_v4_fp8_source_row(
                    source,
                    &parameter.id,
                    &parameter.shape,
                    &region,
                    &actual.values,
                );
                v4_fp8_rows_verified += 1;
                mixed_rows_verified += usize::from(parameter.id.contains(".switch_mlp."));
            }
        }
        if parameter.shape.len() == 2 {
            let width = region.shape[1] as usize;
            let coefficients: Vec<_> = (0..2 * width)
                .map(|i| (i % 7) as f32 * 0.125 - 0.25)
                .collect();
            let projection = MlxBackend::project_parameter(
                runtime,
                &facts.identity,
                &parameter.id,
                ParameterProjection {
                    region: region.clone(),
                    axis: 1,
                    directions: 2,
                    coefficients: coefficients.clone(),
                },
                limits,
            )
            .unwrap();
            for direction in 0..2 {
                let expected = expected
                    .values
                    .iter()
                    .zip(&coefficients[direction * width..(direction + 1) * width])
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum::<f64>();
                assert!((f64::from(projection.values[direction]) - expected).abs() < 2e-6);
            }
            let column = ParameterRegion {
                starts: vec![0, 0],
                shape: vec![parameter.shape[0], 1],
            };
            let actual = MlxBackend::query_parameter(
                runtime,
                &facts.identity,
                &parameter.id,
                column.clone(),
                limits,
            )
            .unwrap();
            let expected = MlxBackend::query_parameter(
                reference,
                &ordinary.identity,
                &parameter.id,
                column,
                limits,
            )
            .unwrap();
            assert_eq!(
                actual.values, expected.values,
                "global column {}",
                parameter.id
            );
        }
        let values = (0..actual.values.len())
            .map(|i| 0.01 + (i % 5) as f32 * 0.003)
            .collect();
        originals.push(actual);
        edits.push(ParameterEdit {
            id: format!("coordinated-{}", edits.len()),
            parameter: parameter.id.clone(),
            parameter_shape: parameter.shape.clone(),
            dtype: InterventionDtype::Float32,
            region,
            update: ParameterUpdate::Add { values },
        });
    }
    if std::env::var_os("EREDU_RING_PREDICTION_FP8").is_some() {
        // Two target banks plus one prediction bank (gate/up and down each),
        // and the target recurrent QKV projection use actual block-FP8 sources.
        assert_eq!(fp8_rows_verified, 7);
    }
    if v4_source.is_some() {
        assert!(
            v4_fp8_rows_verified > 0,
            "FP8 fixture must verify actual effective matrices"
        );
    }
    if std::env::var_os("EREDU_RING_DEEPSEEK_MIXED_FP8").is_some() {
        assert!(
            mixed_rows_verified > 0,
            "mixed fixture verifies encoded expert rows"
        );
    }
    let readout = &graph.component_scopes[0].readout;
    let stream_readout =
        (v4_source.is_some() && !readout.score_writes.is_empty()).then_some(readout);
    let edited_readout = v4_source.as_ref().map(|source| {
        assert!(readout.bias.is_none());
        assert_eq!(
            readout.output_transform,
            eredu_core::component::ComponentOutputTransform::Identity
        );
        let parameter = ordinary
            .parameters
            .iter()
            .find(|parameter| parameter.id == readout.weight)
            .unwrap();
        assert_eq!(parameter.shape.len(), 2);
        let width = parameter.shape[1] as usize;
        let weights = independently_edited_fixture_matrix(
            reference,
            source,
            &ordinary,
            &parameter.id,
            &edits,
            limits,
        );
        let mut gated = Vec::new();
        for group in &graph.component_scopes[0].components {
            use eredu_core::component::{
                ComponentActivation, ComponentReadRole, ComponentRowMapping,
            };
            if !matches!(group.activation_equation, ComponentActivation::Gated { .. }) {
                continue;
            }
            let read = |role| group.reads.iter().find(|read| read.role == role).unwrap();
            for read in &group.reads {
                assert!(read.bias.is_none() && read.input_projections.is_empty());
                assert_eq!(read.rows, ComponentRowMapping::Direct { offset: 0 });
            }
            let gate = independently_edited_fixture_matrix(
                reference,
                source,
                &ordinary,
                &read(ComponentReadRole::Gate).weight,
                &edits,
                limits,
            );
            let value = independently_edited_fixture_matrix(
                reference,
                source,
                &ordinary,
                &read(ComponentReadRole::Value).weight,
                &edits,
                limits,
            );
            assert_eq!(gate.len(), value.len());
            gated.push(EditedGatedEvidence {
                group: group.clone(),
                width: gate.len() / group.count,
                gate,
                value,
            });
        }
        let mut routed = Vec::new();
        for group in &graph.component_scopes[0].routed_components {
            use eredu_core::component::RoutedComponentParameter;
            let ids = group
                .reads
                .iter()
                .map(|read| {
                    assert!(read.bias.is_none());
                    let RoutedComponentParameter::Packed { name } = &read.weight else {
                        panic!("packed fixture expert read")
                    };
                    name.parameter.as_str()
                })
                .collect::<Vec<_>>();
            assert_eq!(ids.len(), 2);
            assert_eq!(ids[0], ids[1], "fused fixture gate/up bank");
            let parameter = ordinary
                .parameters
                .iter()
                .find(|parameter| parameter.id == ids[0])
                .unwrap();
            let weights = independently_edited_fixture_matrix(
                reference, source, &ordinary, ids[0], &edits, limits,
            );
            routed.push(EditedRoutedEvidence {
                group: group.clone(),
                parameter: ids[0].to_owned(),
                shape: parameter.shape.clone(),
                weights,
            });
        }
        let mut ffn_writes = Vec::new();
        if !readout.score_writes.is_empty() {
            for bank in &routed {
                let shared = gated
                    .iter()
                    .find(|shared| shared.group.layer_index == bank.group.layer_index)
                    .unwrap();
                let cycle = readout
                    .stream_residual
                    .as_ref()
                    .unwrap()
                    .cycles
                    .iter()
                    .find(|cycle| {
                        cycle.layer_index == bank.group.layer_index
                            && graph.node(&bank.group.node_id).unwrap().parent.as_deref()
                                == Some(cycle.node_id.as_str())
                    })
                    .unwrap();
                use eredu_core::component::RoutedComponentParameter;
                let RoutedComponentParameter::Packed { name } = &bank.group.write_weight else {
                    panic!("packed fixture expert write")
                };
                assert!(bank.group.write_bias.is_none());
                assert!(shared.group.write_bias.is_none());
                assert!(shared.group.output_normalization.is_none());
                assert!(shared.group.output_gate.is_none());
                let shared_weights = independently_edited_fixture_matrix(
                    reference,
                    source,
                    &ordinary,
                    &shared.group.write_weight,
                    &edits,
                    limits,
                );
                let routed_weights = independently_edited_fixture_matrix(
                    reference,
                    source,
                    &ordinary,
                    &name.parameter,
                    &edits,
                    limits,
                );
                ffn_writes.push(EditedFfnWriteEvidence {
                    path: cycle.write.clone(),
                    shared: shared.group.clone(),
                    routed: bank.group.clone(),
                    shared_weights,
                    routed_weights,
                });
            }
        }
        EditedReadoutEvidence {
            equation: readout.clone(),
            width,
            weights,
            gated,
            routed,
            ffn_writes,
        }
    });
    let plan = |base_identity: &str| ParameterOverlayPlan {
        schema_version: PARAMETER_SCHEMA_VERSION,
        base_identity: base_identity.into(),
        provenance: "coordinated target/prediction edit with shared pipeline embedding".into(),
        edits: edits.clone(),
    };
    let overlay = AdmittedParameterOverlay::admit(plan(&facts.identity), &facts).unwrap();
    let local_overlay =
        AdmittedParameterOverlay::admit(plan(&ordinary.identity), &ordinary).unwrap();
    let baseline = run(reference, 0, [1, 2, 3], false);
    // One peer rejects the reservation; no owner may publish any part of the edit.
    let used = MlxBackend::parameter_discovery(runtime).unwrap().usage;
    assert!(MlxBackend::activate_parameter_overlay(
        runtime,
        &overlay,
        if rank == 1 { used } else { limits }
    )
    .is_err());
    assert_eq!(
        MlxBackend::parameter_discovery(runtime).unwrap().identity,
        facts.identity
    );
    eprintln!("prediction parameter comparison rank={rank}: rejected reservation");
    assert_partitioned_prediction_result(&run(runtime, 0, [1, 2, 3], false), &baseline);
    // A late peer rejection must roll back both target and prediction weights.
    if rank == 1 {
        runtime
            .session_mut()
            .reject_next_parameter_publication_for_test();
    }
    assert!(MlxBackend::activate_parameter_overlay(runtime, &overlay, limits).is_err());
    assert_eq!(
        MlxBackend::parameter_discovery(runtime).unwrap().identity,
        facts.identity
    );
    eprintln!("prediction parameter comparison rank={rank}: rejected publication");
    assert_partitioned_prediction_result(&run(runtime, 0, [1, 2, 3], false), &baseline);
    let active = MlxBackend::activate_parameter_overlay(runtime, &overlay, limits).unwrap();
    let local_active =
        MlxBackend::activate_parameter_overlay(reference, &local_overlay, limits).unwrap();
    if let Some(readout) = &edited_readout {
        let mut matrices = vec![(
            readout.equation.weight.as_str(),
            readout.weights.as_slice(),
            readout.width,
        )];
        for gated in &readout.gated {
            use eredu_core::component::ComponentReadRole;
            for (role, weights) in [
                (ComponentReadRole::Gate, gated.gate.as_slice()),
                (ComponentReadRole::Value, gated.value.as_slice()),
            ] {
                let read = gated
                    .group
                    .reads
                    .iter()
                    .find(|read| read.role == role)
                    .unwrap();
                matrices.push((read.weight.as_str(), weights, gated.width));
            }
        }
        for (id, weights, width) in matrices {
            let region = ParameterRegion {
                starts: vec![0, 0],
                shape: vec![(weights.len() / width) as u64, width as u64],
            };
            let actual =
                MlxBackend::query_parameter(runtime, &active.identity, id, region.clone(), limits)
                    .unwrap();
            let expected =
                MlxBackend::query_parameter(reference, &local_active.identity, id, region, limits)
                    .unwrap();
            for loaded in [actual, expected] {
                assert_eq!(loaded.values, weights, "complete edited matrix {id} agrees with independently decoded source plus edits");
            }
        }
        for routed in &readout.routed {
            let region = ParameterRegion {
                starts: vec![0; routed.shape.len()],
                shape: routed.shape.clone(),
            };
            let actual = MlxBackend::query_parameter(
                runtime,
                &active.identity,
                &routed.parameter,
                region.clone(),
                limits,
            )
            .unwrap();
            let expected = MlxBackend::query_parameter(
                reference,
                &local_active.identity,
                &routed.parameter,
                region,
                limits,
            )
            .unwrap();
            for loaded in [actual, expected] {
                assert_eq!(loaded.values, routed.weights, "complete edited routed matrix agrees with independently decoded source plus edits");
            }
        }
    }
    assert_eq!(
        active.overlay_identity.as_deref(),
        Some(overlay.intent_identity())
    );
    for (edit, original) in edits.iter().zip(&originals) {
        let actual = MlxBackend::query_parameter(
            runtime,
            &active.identity,
            &edit.parameter,
            edit.region.clone(),
            limits,
        )
        .unwrap();
        for ((actual, original), added) in
            actual.values.iter().zip(&original.values).zip(edit.update.values())
        {
            assert_eq!(*actual, *original + *added, "edited {}", edit.parameter);
        }
    }
    for prefix in [[1, 2, 3], [3, 2, 1]] {
        eprintln!("prediction parameter comparison rank={rank}: active {prefix:?}");
        let expected = run(reference, 0, prefix, false);
        let actual = run(runtime, 0, prefix, true);
        assert_partitioned_prediction_result_with_readout(
            &actual,
            &expected,
            edited_readout.as_ref(),
            stream_readout,
        );
        for phase in &actual.1 {
            for partition in &phase.captures.as_step().partitions {
                assert_eq!(
                    partition.context.overlay_identity.as_deref(),
                    Some(overlay.intent_identity())
                );
            }
        }
        if prefix == [1, 2, 3] {
            for path in [
                eredu_core::MODEL_LOGITS_OBSERVATION_PATH,
                graph.component_scopes[0].readout.logits.as_str(),
            ] {
                let payloads = |result: &PredictionResult| {
                    result
                        .1
                        .iter()
                        .flat_map(|phase| &phase.captures.as_step().records)
                        .filter(|record| record.path == path)
                        .filter_map(|record| record.payload.clone())
                        .collect::<Vec<_>>()
                };
                assert_ne!(
                    payloads(&expected),
                    payloads(&baseline),
                    "edit must change {path}"
                );
            }
        }
    }
    if rank == 1 {
        runtime
            .session_mut()
            .reject_next_parameter_publication_for_test();
    }
    assert!(MlxBackend::remove_parameter_overlay(runtime, &active.identity).is_err());
    assert_eq!(
        MlxBackend::parameter_discovery(runtime).unwrap().identity,
        active.identity
    );
    eprintln!("prediction parameter comparison rank={rank}: rejected removal");
    assert_partitioned_prediction_result_with_readout(
        &run(runtime, 0, [1, 2, 3], false),
        &run(reference, 0, [1, 2, 3], false),
        edited_readout.as_ref(),
        stream_readout,
    );
    let restored = MlxBackend::remove_parameter_overlay(runtime, &active.identity).unwrap();
    MlxBackend::remove_parameter_overlay(reference, &local_active.identity).unwrap();
    assert!(restored.overlay_identity.is_none());
    for (edit, original) in edits.iter().zip(&originals) {
        let actual = MlxBackend::query_parameter(
            runtime,
            &restored.identity,
            &edit.parameter,
            edit.region.clone(),
            limits,
        )
        .unwrap();
        assert_eq!(&actual.values, &original.values, "restored {}", edit.parameter);
    }
    eprintln!("prediction parameter comparison rank={rank}: restored");
    assert_partitioned_prediction_result(&run(runtime, 0, [1, 2, 3], true), &baseline);
    let final_usage = MlxBackend::parameter_discovery(runtime).unwrap().usage;
    assert!(final_usage.host_bytes > measured.usage.host_bytes);
    assert!(final_usage.captures <= limits.captures);
    assert!(final_usage.retained_bytes <= limits.retained_bytes);
    assert!(final_usage.host_bytes <= limits.host_bytes);
    assert!(final_usage.encoded_bytes <= limits.encoded_bytes);
    assert_prediction_modules_idle(runtime);
    assert_prediction_modules_idle(reference);
    eprintln!("prediction parameter cumulative usage rank={rank}: {final_usage:?}");
}

#[test]
fn native_v3_prediction_quantization_reports_auxiliary_work() {
    let root = tempfile::tempdir().unwrap();
    write_deepseek_transform_fixture_with_prediction(root.path(), 1, 64, 1);
    let original = std::fs::read(root.path().join("model-00001-of-00001.safetensors")).unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let backend = MlxBackend::new(&stream, &stream);
    let options = eredu_runtime::NormalizedLoadRequest::with_quantization(
        eredu_core::QuantizationRequest::Affine {
            group_size: 32,
            bits: 4,
        },
    );
    let load = |options| {
        load_model(
            &backend,
            root.path(),
            MlxLoadRequest::from_normalized(options),
        )
        .unwrap()
    };
    let target = load(
        options
            .clone()
            .with_drafting(eredu_runtime::DraftingLoadRequest::Disabled),
    );
    let complete = load(options);
    let target = target.materialization_report().unwrap();
    let complete = complete.materialization_report().unwrap();
    assert!(complete.transformed_weights > target.transformed_weights);
    assert!(complete.source_bytes_read > target.source_bytes_read);
    assert!(complete.output_bytes > target.output_bytes);
    assert!(complete.peak_planned_working_set_bytes <= complete.admitted_working_set_bytes);
    assert_eq!(
        std::fs::read(root.path().join("model-00001-of-00001.safetensors")).unwrap(),
        original
    );
}

// Independent E4M3 bytes and block scales, not another native materialization.
fn verify_qwen_fp8_source_row(
    parameter: &str,
    shape: &[u64],
    region: &eredu_core::parameters::ParameterRegion,
    values: &[f32],
) -> bool {
    let (source, rows, columns, row) = if parameter.ends_with(".linear_attn.in_proj_qkv.weight") {
        assert_eq!(shape.len(), 2);
        (
            parameter.to_owned(),
            shape[0] as usize,
            shape[1] as usize,
            region.starts[0] as usize,
        )
    } else if let Some((root, projection)) = parameter.split_once(".mlp.experts.") {
        assert_eq!(shape.len(), 3);
        let expert = region.starts[0];
        let row = region.starts[1] as usize;
        let (field, rows, row) = match projection {
            "gate_up_proj" => {
                let rows = shape[1] as usize / 2;
                (
                    if row < rows { "gate_proj" } else { "up_proj" },
                    rows,
                    row % rows,
                )
            }
            "down_proj" => ("down_proj", shape[1] as usize, row),
            _ => return false,
        };
        (
            format!("{root}.mlp.experts.{expert}.{field}.weight"),
            rows,
            shape[2] as usize,
            row,
        )
    } else {
        return false;
    };
    let column_start = *region.starts.last().unwrap() as usize;
    for (column, &actual) in values.iter().enumerate() {
        let expected = k2_fp8_scalar(
            &source,
            &[rows, columns],
            row * columns + column_start + column,
        );
        assert_eq!(
            actual, expected,
            "independent FP8 source {source} row {row} column {column}"
        );
    }
    true
}
