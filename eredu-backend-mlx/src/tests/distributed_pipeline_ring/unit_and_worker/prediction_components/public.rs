// Public backend contracts plus the same controlled driver used by the facade.
fn prove_public_bounded(device: DeviceType) {
    prove_public_bounded_fixture(device, |path| {
        write_deepseek_fixture_with_values(path, 2, 2, true)
    });
}

fn prove_public_bounded_fixture(device: DeviceType, write: impl FnOnce(&Path)) {
    use eredu_core::{capture::*, intervention::*, speculative::*};
    use eredu_runtime::speculative::{
        ControlledSpeculativeOptions, ControlledSpeculativeSession, DriveControlledSpeculation,
    };
    let checkpoint = tempfile::tempdir().unwrap();
    write(checkpoint.path());
    let config =
        serde_json::from_slice(&std::fs::read(checkpoint.path().join("config.json")).unwrap())
            .unwrap();
    let resolved = eredu_architectures::configuration::MODEL_CONFIGURATIONS
        .resolve_safetensors(&config)
        .unwrap();
    let graph = resolved.architecture_plan().architecture_descriptor();
    let scope = &graph.component_scopes[0];
    let channel = scope
        .components
        .iter()
        .find(|group| {
            matches!(
                group.activation_equation,
                eredu_core::component::ComponentActivation::Attention { .. }
            )
        })
        .unwrap();
    let paths = [
        channel.activation.clone(),
        channel.effective_activation.clone(),
        scope.readout.normalized.clone(),
        scope.readout.logits.clone(),
        eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
    ];
    let run = |enabled: bool, mask: bool, maximum: u64| {
        let stream = Stream::new_with_device(&Device::new(device, 0));
        let weights = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let backend = crate::native::backend(&stream, &weights);
        let model = load_model(&backend, checkpoint.path(), MlxLoadRequest::default()).unwrap();
        let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        let discovery =
            <MlxBackend<'_> as SpeculativeGenerationBackend>::speculative_activation_discovery(
                &runtime,
            )
            .unwrap();
        let ordinary =
            <MlxBackend<'_> as eredu_core::TextGenerationBackend>::capture_discovery(&runtime)
                .unwrap();
        for path in paths.iter().take(4) {
            let point = discovery.captures.catalog.get(path).unwrap();
            assert!(point
                .requirements
                .contains(&eredu_core::ObservationRequirement::PredictionExecution));
            let support = discovery
                .captures
                .support
                .points
                .iter()
                .find(|p| &p.path == path)
                .unwrap();
            assert_eq!(
                support.prefill,
                eredu_core::ObservationSupportStatus::Supported
            );
            assert!(!ordinary
                .support
                .points
                .iter()
                .find(|p| &p.path == path)
                .is_some_and(|p| p.prefill == eredu_core::ObservationSupportStatus::Supported));
        }
        let per_step = CaptureUsage {
            captures: 128,
            retained_bytes: 8 << 20,
            host_bytes: 8 << 20,
            encoded_bytes: 8 << 20,
        };
        let admitted = SpeculativeActivationPlan {
            schema_version: SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
            bounds: CaptureInvocationBounds {
                batch: 1,
                max_sequence: 3,
                max_context: None,
                max_predictions: 8,
            },
            captures: CapturePlan {
                schema_version: CAPTURE_SCHEMA_VERSION,
                selections: paths
                    .iter()
                    .enumerate()
                    .map(|(index, path)| CaptureSelection {
                        id: index.to_string(),
                        path: path.clone(),
                        schedule: Default::default(),
                        slices: vec![],
                        transform: CaptureTransform::Preview { max_elements: 2048 },
                    })
                    .collect(),
                limits: CaptureLimits {
                    per_step,
                    cumulative: CaptureUsage {
                        captures: maximum,
                        ..per_step.checked_mul(8).unwrap()
                    },
                    physical_native_bytes: None,
                    on_limit: CaptureLimitPolicy::Fail,
                },
            },
            interventions: InterventionPlan {
                schema_version: INTERVENTION_SCHEMA_VERSION,
                operations: if mask {
                    vec![InterventionOperation {
                        id: "public-channel-mask".into(),
                        target: channel.activation.clone(),
                        schedule: Default::default(),
                        slices: vec![],
                        action: InterventionAction::MaskComponents {
                            dtype: InterventionDtype::Float32,
                            indices: vec![0],
                            keep_selected: false,
                        },
                        evidence: InterventionEvidence::Preview { max_elements: 2048 },
                    }]
                } else {
                    vec![]
                },
            },
        }
        .admit(&discovery)
        .unwrap();
        let identity = admitted.identity().to_owned();
        let tokens = [1_u32, 2, 3];
        let prompt = Array::from_slice(&tokens, &[1, 3]);
        let parts = [text_input_part(&prompt)];
        let config = SpeculativeConfig {
            max_tokens: 5,
            max_draft_tokens: 2,
            temperature: 0.0,
            eos_token_ids: vec![],
        };
        let mut records = Vec::new();
        let mut failure = None;
        let options = ControlledSpeculativeOptions {
            activations: enabled.then_some(admitted),
            ..Default::default()
        };
        let visitor = DriveControlledSpeculation::new(
            Default::default(),
            options,
            |session: &mut dyn ControlledSpeculativeSession| {
                let mut sequence = 0;
                loop {
                    match session.step() {
                        Ok(Some(step)) => {
                            assert_eq!(step.sequence, sequence);
                            sequence += 1;
                            assert!(step.captures.is_empty());
                            records.extend(step.activations);
                        }
                        Ok(None) => break,
                        Err(error) => {
                            while let Some(record) = session.take_activation_evidence()? {
                                assert_eq!(record.sequence, sequence);
                                sequence += 1;
                                records.push(record.activation);
                            }
                            return Err(error);
                        }
                    }
                }
                assert!(session.take_activation_evidence()?.is_none());
                Ok(())
            },
            &mut failure,
        );
        let (output, publications) = execute_neutral_embedded_mtp_with(
            &mut runtime,
            synthetic_prediction_input(&parts, &tokens),
            config.clone(),
            visitor,
        );
        assert!(runtime
            .session_mut()
            .take_speculative_activation_capture()
            .unwrap()
            .is_none());
        if output.is_ok() {
            let (fresh, _) = execute_neutral_embedded_mtp(
                &mut runtime,
                synthetic_prediction_input(&parts, &tokens),
                config,
            );
            fresh.unwrap();
            assert!(runtime
                .session_mut()
                .take_speculative_activation_capture()
                .unwrap()
                .is_none());
        }
        for record in &records {
            assert_eq!(
                record.admission_identity.as_deref(),
                Some(identity.as_str())
            );
            assert!(
                serde_json::to_vec(record).unwrap().len() as u64
                    <= record.captures.step_usage.encoded_bytes
            );
        }
        (output, publications, records, failure)
    };
    let (baseline, _, absent, _) = run(false, false, 512);
    assert!(absent.is_empty());
    let (observed, _, records, failure) = run(true, false, 512);
    assert!(failure.is_none());
    assert_eq!(baseline.unwrap().token_ids(), observed.unwrap().token_ids());
    assert_eq!(records[0].phase, SpeculativeActivationPhase::TargetPrefill);
    assert_eq!(records[0].captures.invocation.unwrap().sequence, 3);
    assert_eq!(
        records[1].phase,
        SpeculativeActivationPhase::PredictionPrefill
    );
    assert_eq!(records[1].captures.invocation.unwrap().sequence, 2);
    assert!(records.iter().all(|r| r.completed));
    let (masked, _, edited, failure) = run(true, true, 512);
    masked.unwrap();
    assert!(failure.is_none());
    let logits = |records: &[SpeculativeActivationCapture]| -> Vec<_> {
        records
            .iter()
            .flat_map(|r| &r.captures.records)
            .filter(|r| r.path == scope.readout.logits)
            .filter_map(|r| r.payload.clone())
            .collect()
    };
    assert_ne!(logits(&records), logits(&edited));
    assert!(edited
        .iter()
        .flat_map(|r| &r.captures.interventions)
        .any(|edit| !edit.evidence.is_empty()));
    let (failed, _, evidence, failure) = run(true, false, 5);
    assert!(failed.is_err());
    assert!(matches!(
        failure,
        Some(SpeculativeControlError::Capture(CaptureError::Limit {
            budget: CaptureBudget::Captures,
            cumulative: true
        }))
    ));
    assert!(evidence.iter().any(|record| !record.completed));
    let (failed, publications, evidence, failure) = run(true, false, 1);
    assert!(failed.is_err());
    assert_eq!(publications, 0);
    assert!(evidence.is_empty());
    assert!(matches!(
        failure,
        Some(SpeculativeControlError::Capture(CaptureError::Limit { .. }))
    ));
}

#[test]
fn native_v3_public_speculative_activation_admission_cpu() {
    prove_public_bounded(DeviceType::Cpu);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires a local MLX Metal device"]
fn native_v3_public_speculative_activation_admission_metal() {
    prove_public_bounded(DeviceType::Gpu);
}

#[test]
fn native_qwen_public_speculative_activation_admission_cpu() {
    for (model_type, routed) in [
        ("qwen3_next", false),
        ("qwen3_next", true),
        ("qwen3_5_text", false),
        ("qwen3_5_moe_text", true),
    ] {
        for tied in [false, true] {
            eprintln!("Qwen public prediction: {model_type}, routed={routed}, tied={tied}");
            prove_public_bounded_fixture(DeviceType::Cpu, |path| {
                let mut config = if routed {
                    qwen_hybrid_moe_config(model_type)
                } else {
                    qwen_hybrid_config(model_type)
                };
                config["mtp_num_hidden_layers"] = 2.into();
                config["tie_word_embeddings"] = tied.into();
                write_qwen_hybrid_config_fixture(path, config);
            });
        }
    }
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires a local MLX Metal device"]
fn native_qwen_public_speculative_activation_admission_metal() {
    for model_type in ["qwen3_next", "qwen3_5_text"] {
        prove_public_bounded_fixture(DeviceType::Gpu, |path| {
            let mut config = qwen_hybrid_config(model_type);
            config["mtp_num_hidden_layers"] = 2.into();
            write_qwen_hybrid_config_fixture(path, config);
        });
    }
}

pub(super) fn inkling_component_prediction_fixture(path: &Path, routed: bool, chain_norm: bool) {
    inkling_component_fixture(path, routed, Some(chain_norm));
}

pub(super) fn inkling_component_fixture(path: &Path, routed: bool, prediction: Option<bool>) {
    inkling_component_fixture_config(path, routed, prediction, false);
}

pub(super) fn inkling_component_fixture_config(
    path: &Path,
    routed: bool,
    prediction: Option<bool>,
    quantized: bool,
) {
    let mut config = inkling_quantizable_config();
    if quantized {
        // TP2 keeps complete 32-value input groups for every column-sharded
        // quantized projection. Preserve the established F32 fixture separately.
        for field in [
            "hidden_size",
            "intermediate_size",
            "dense_intermediate_size",
            "moe_intermediate_size",
        ] {
            config["text_config"][field] = 64.into();
        }
        config["text_config"]["head_dim"] = 16.into();
        config["text_config"]["swa_head_dim"] = 16.into();
    }
    config.as_object_mut().unwrap().remove("vision_config");
    config["text_config"]["num_hidden_layers"] = 2.into();
    config["text_config"]["layer_types"] =
        serde_json::json!(["full_attention", "sliding_attention"]);
    config["text_config"]["dense_mlp_idx"] = if routed { 0 } else { 2 }.into();
    config["text_config"]["model_max_length"] = 64.into();
    if let Some(chain_norm) = prediction {
        config["mtp_config"] = serde_json::json!({
            "num_nextn_predict_layers": 2, "local_layer_ids": [1],
            "chain_hidden_post_norm": chain_norm
        });
    } else {
        config.as_object_mut().unwrap().remove("mtp_config");
        if routed {
            config["text_config"]["n_shared_experts"] = 4.into();
        }
    }
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let (_, arrays) = initialized_inkling_parameters(&config, stream);
    let arrays = arrays
        .into_iter()
        .map(|(name, array)| {
            let seed = name
                .bytes()
                .fold(0_u32, |a, b| a.wrapping_mul(31).wrapping_add(u32::from(b)));
            let norm = name.contains("norm") && name.ends_with("weight");
            let scale = name.ends_with("global_scale");
            let values = (0..array.size())
                .map(|index| {
                    let delta = ((index * 7 + (seed % 97) as usize) % 41) as f32 - 20.0;
                    if scale {
                        1.3
                    } else if norm {
                        1.0 + delta * 0.002
                    } else {
                        delta * 0.008
                    }
                })
                .collect::<Vec<_>>();
            let value = Array::from_slice(&values, array.shape())
                .as_dtype(array.dtype(), stream)
                .unwrap();
            (name, value)
        })
        .collect::<Vec<_>>();
    std::fs::write(
        path.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
    Array::save_safetensors(
        arrays.iter().map(|(name, array)| (name.as_str(), array)),
        None,
        path.join("model.safetensors"),
    )
    .unwrap();
}

#[test]
fn native_inkling_public_speculative_activation_admission_cpu() {
    for routed in [false, true] {
        for chain_norm in [false, true] {
            eprintln!("Inkling public prediction: routed={routed}, chain_norm={chain_norm}");
            prove_public_bounded_fixture(DeviceType::Cpu, |path| {
                inkling_component_prediction_fixture(path, routed, chain_norm)
            });
        }
    }
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires a local MLX Metal device"]
fn native_inkling_public_speculative_activation_admission_metal() {
    for chain_norm in [false, true] {
        prove_public_bounded_fixture(DeviceType::Gpu, |path| {
            inkling_component_prediction_fixture(path, true, chain_norm)
        });
    }
}
