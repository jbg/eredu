// Every bounded query has a finite allowance above all prior charged work.
fn parameter_fixture_limits(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    allowance: eredu_core::capture::CaptureUsage,
) -> eredu_core::capture::CaptureUsage {
    use eredu_core::parameters::ParameterBackend as _;
    MlxBackend::parameter_discovery(runtime)
        .unwrap()
        .usage
        .checked_add(allowance)
        .unwrap()
}

// Public distributed edits, real cached predictions, rollback and restoration.
fn parameter_fixture_forward(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    prefill: bool,
) -> Option<Vec<f32>> {
    use crate::backend::runtime::media::input::ModelInput;
    let output = if prefill && std::env::var_os(COMPONENT_CAPTURE_MEDIA).is_some() {
        let prompt = component_capture_prompt(runtime);
        runtime.prefill(prompt).unwrap().wait().unwrap()
    } else if prefill {
        let prompt = Array::from_slice(&[1u32, 2, 3], &[1, 3]);
        let parts = [text_input_part(&prompt)];
        runtime
            .prefill(ModelInput::new(&parts).into())
            .unwrap()
            .wait()
            .unwrap()
    } else {
        runtime
            .decode(Array::from_slice(&[4u32], &[1, 1]))
            .unwrap()
            .wait()
            .unwrap()
    };
    output.logits().map(|value| {
        value
            .as_array()
            .evaluated()
            .unwrap()
            .as_slice::<f32>()
            .to_vec()
    })
}

#[track_caller]
fn assert_parameter_predictions(
    actual: Option<Vec<f32>>,
    expected: &Option<Vec<f32>>,
    tolerance: f32,
) {
    if let Some(actual) = actual {
        let expected = expected.as_ref().unwrap();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= tolerance,
                "edited prediction {actual} vs {expected}"
            );
        }
    }
}

fn verify_public_partition_parameter_overlays(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    reference: &mut ModelRuntime<MlxBackend<'_>>,
    rank: usize,
    family: FixtureFamily,
    checkpoint: &Path,
    stream: &Stream,
) {
    use eredu_core::{
        capture::CaptureUsage, execution_control::NativeTextStateBackend, parameters::*,
    };
    let facts = MlxBackend::parameter_discovery(runtime).unwrap();
    let ordinary = MlxBackend::parameter_discovery(reference).unwrap();
    let weights = facts
        .parameters
        .iter()
        .filter(|p| p.access().replacement && component_fixture_weight(family, &p.id))
        .collect::<Vec<_>>();
    let qwen_hybrid = matches!(
        family,
        FixtureFamily::Qwen3Next
            | FixtureFamily::Qwen3NextMoe
            | FixtureFamily::Qwen35
            | FixtureFamily::Qwen35Moe
            | FixtureFamily::Qwen35Multimodal
            | FixtureFamily::Qwen35MoeMultimodal
    );
    let mut candidates: Vec<_> = if family.is_v3()
        || family == FixtureFamily::DeepSeekV4
        || qwen_hybrid
        || matches!(
            family,
            FixtureFamily::KimiLinear
                | FixtureFamily::KimiLinearGguf
                | FixtureFamily::MuseGlimmer
                | FixtureFamily::MuseGlimmerMoe
                | FixtureFamily::MuseGlimmerGguf(_)
                | FixtureFamily::Qwen3Vl
                | FixtureFamily::Qwen3VlMoe
                | FixtureFamily::Gemma
        ) {
        // Include both MLA bottlenecks, all head projections, every gated read
        // and write, normalization, embedding and output in one atomic edit.
        weights.clone()
    } else if family == FixtureFamily::Qwen3 {
        [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.1.input_layernorm.weight",
            "lm_head.weight",
        ]
        .into_iter()
        .map(|id| facts.parameters.iter().find(|p| p.id == id).unwrap())
        .collect()
    } else {
        vec![
            weights[0],
            weights[weights.len() / 2],
            *weights.last().unwrap(),
        ]
    };
    if matches!(
        family,
        FixtureFamily::NemotronH | FixtureFamily::NemotronHGguf
    ) {
        candidates = [
            "model.embeddings.weight",
            "model.layers.0.mamba.in_proj.weight",
            "model.layers.0.mamba.conv1d.weight",
            "model.layers.0.mamba.out_proj.weight",
            if family == FixtureFamily::NemotronH {
                "model.layers.1.mlp.up_proj.weight"
            } else {
                "model.layers.1.moe.shared_experts.up_proj.weight"
            },
            if family == FixtureFamily::NemotronH {
                "model.layers.1.mlp.down_proj.weight"
            } else {
                "model.layers.1.moe.shared_experts.down_proj.weight"
            },
            "model.layers.2.moe.shared_experts.up_proj.weight",
            "model.layers.2.moe.shared_experts.down_proj.weight",
            "model.layers.3.attention.q_proj.weight",
            "model.layers.3.attention.k_proj.weight",
            "model.layers.3.attention.v_proj.weight",
            "model.layers.3.attention.o_proj.weight",
            "model.norm_f.weight",
            "lm_head.weight",
        ]
        .into_iter()
        .map(|id| {
            facts
                .parameters
                .iter()
                .find(|p| p.id == id)
                .unwrap_or_else(|| panic!("missing effective {id}"))
        })
        .collect();
    }
    if matches!(
        family,
        FixtureFamily::K2Dense | FixtureFamily::K2Mova | FixtureFamily::K2Fp8(_)
    ) {
        candidates = [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
        ]
        .into_iter()
        .map(|id| facts.parameters.iter().find(|p| p.id == id).unwrap())
        .collect();
        if matches!(family, FixtureFamily::K2Mova | FixtureFamily::K2Fp8(_)) {
            candidates.extend(
                [
                    "model.layers.1.self_attn.q_proj.weight",
                    "model.layers.1.self_attn.k_proj.weight",
                    "model.layers.1.self_attn.v_experts.weight",
                    "model.layers.1.self_attn.o_proj.weight",
                    "model.layers.1.mlp.shared_experts.gate_proj.weight",
                    "model.layers.1.mlp.shared_experts.up_proj.weight",
                    "model.layers.1.mlp.shared_experts.down_proj.weight",
                ]
                .into_iter()
                .map(|id| facts.parameters.iter().find(|p| p.id == id).unwrap()),
            );
        }
    }
    candidates.extend(
        facts
            .parameters
            .iter()
            .filter(|p| p.access().replacement && p.shape.len() == 3)
            .filter(|p| !family.is_k2_fp8() || p.id.starts_with("model.layers.1."))
            .take(if family == FixtureFamily::Qwen3MoeGguf {
                4
            } else if family.is_k2_fp8() {
                usize::MAX
            } else if matches!(
                family,
                FixtureFamily::NemotronH | FixtureFamily::NemotronHGguf
            ) {
                // The convolution is also rank three. Include both expert
                // matrices as well as its distinct segmented channel layout.
                usize::MAX
            } else {
                2
            }),
    );
    if matches!(
        family,
        FixtureFamily::GptOss
            | FixtureFamily::Qwen3Moe
            | FixtureFamily::Qwen3MoeGguf
            | FixtureFamily::Lfm2Moe
            | FixtureFamily::K2Mova
            | FixtureFamily::K2Fp8(_)
            | FixtureFamily::NemotronH
            | FixtureFamily::NemotronHGguf
            | FixtureFamily::DeepSeek
            | FixtureFamily::DeepSeekGguf
            | FixtureFamily::MuseGlimmerMoe
            | FixtureFamily::MuseGlimmerGguf(true)
            | FixtureFamily::Qwen3VlMoe
    ) {
        assert!(
            candidates.iter().filter(|p| p.shape.len() == 3).count() >= 2,
            "public coordinated edit must include both grouped read and write parameters"
        );
    }
    if family == FixtureFamily::Qwen3MoeGguf {
        assert_eq!(
            candidates.iter().filter(|p| p.shape.len() == 3).count(),
            4,
            "both Q8_0 and IQ4_NL layers must participate in the coordinated edit"
        );
    }
    let mut shared = std::collections::BTreeSet::new();
    let mut edits: Vec<_> = candidates
        .into_iter()
        .filter(|target| shared.insert(target.shared_id.clone()))
        .enumerate()
        .map(|(index, target)| {
            let mut region = ParameterRegion {
                starts: target.shape.iter().map(|n| u64::from(*n > 2)).collect(),
                shape: target
                    .shape
                    .iter()
                    .enumerate()
                    .map(|(axis, n)| {
                        if axis == 0 && target.shape.len() == 3 {
                            1
                        } else if axis + 1 == target.shape.len() {
                            if *n > 2 {
                                n - 2
                            } else {
                                *n
                            }
                        } else {
                            (*n).min(2)
                        }
                    })
                    .collect(),
            };
            if family.is_k2_fp8()
                && target.id.starts_with("model.layers.0.mlp.")
                && target.shape.len() == 2
                && target.shape[0] == 259
            {
                // Change read rows owned exclusively by the rank containing
                // the three-element tail, alongside the cross-rank write edit.
                region.starts[0] = 257;
            }
            if family.is_k2_fp8()
                && target.id.ends_with(".gate_up_proj")
                && target.shape.len() == 3
                && target.shape[1] == 518
            {
                region.starts[1] = 257;
            }
            if family.is_k2_fp8() && target.id.contains(".self_attn.") {
                if target.shape.len() == 2 && [132, 264].contains(&target.shape[0]) {
                    region.starts[0] = target.shape[0] - 2;
                } else if target.id.ends_with(".o_proj.weight") && target.shape[1] == 264 {
                    region.starts[1] = 255;
                    region.shape[1] = 9;
                } else if target.id.ends_with(".v_experts.weight") && target.shape[1] == 132 {
                    region.starts[1] = 130;
                }
            }
            let count = region.shape.iter().product::<u64>() as usize;
            ParameterEdit {
                id: format!("coordinated-{index}"),
                parameter: target.id.clone(),
                parameter_shape: target.shape.clone(),
                dtype: target.dtype.unwrap(),
                region,
                update: ParameterUpdate::Add {
                    values: (0..count).map(|i| 0.013 * ((i % 5) as f32 - 1.5)).collect(),
                },
            }
        })
        .collect();
    let up_tail_edits: Vec<_> = edits
        .iter()
        .filter(|edit| {
            family.is_k2_fp8()
                && edit.parameter.ends_with(".gate_up_proj")
                && edit.parameter_shape[1] == 518
        })
        .cloned()
        .map(|mut edit| {
            edit.id.push_str("-up-tail");
            edit.region.starts[1] = 516;
            edit
        })
        .collect();
    edits.extend(up_tail_edits);
    // Exercise both packed read halves and another EP owner in the same
    // transaction, including the independently partitioned expert write bank.
    let additional_expert_edits = edits
        .iter()
        .filter(|edit| {
            (family.is_v3() || family == FixtureFamily::DeepSeekV4 || qwen_hybrid)
                && edit.parameter_shape.len() == 3
                && (edit.parameter.ends_with(".gate_up_proj")
                    || edit.parameter.ends_with(".down_proj"))
        })
        .cloned()
        .map(|mut edit| {
            edit.id.push_str("-last-expert");
            edit.region.starts[0] = edit.parameter_shape[0] - 1;
            if edit.parameter.ends_with(".gate_up_proj") {
                edit.region.starts[1] += edit.parameter_shape[1] / 2;
            }
            edit
        })
        .collect::<Vec<_>>();
    edits.extend(additional_expert_edits);
    assert!(edits.len() >= 2);
    let plan = |identity: &str| ParameterOverlayPlan {
        schema_version: PARAMETER_SCHEMA_VERSION,
        base_identity: identity.into(),
        provenance: "public Ring coordinated attention/FFN fixture".into(),
        edits: edits.clone(),
    };
    let overlay = AdmittedParameterOverlay::admit(plan(&facts.identity), &facts).unwrap();
    let reference_overlay =
        AdmittedParameterOverlay::admit(plan(&ordinary.identity), &ordinary).unwrap();
    // Earlier effective-parameter checks consumed the same cumulative ledger.
    // Add each operation's allowance without resetting those charges.
    // This transaction edits every selected matrix, including published-size
    // projectors. Budget promotion and rollback/copy work from those
    // full affected geometries, not just the small uploaded rectangles.
    let affected_bytes = edits
        .iter()
        .try_fold(0u64, |total, edit| {
            let elements = edit
                .parameter_shape
                .iter()
                .try_fold(1u64, |n, dim| n.checked_mul(*dim))?;
            total.checked_add(elements.checked_mul(4)?)
        })
        .unwrap();
    let phase_allowance = CaptureUsage {
        captures: 100_000,
        retained_bytes: (2u64 << 30)
            .checked_add(affected_bytes.checked_mul(32).unwrap())
            .unwrap(),
        host_bytes: (4u64 << 30)
            .checked_add(affected_bytes.checked_mul(8).unwrap())
            .unwrap(),
        encoded_bytes: 2 << 30,
    };
    let limits = facts.usage.checked_add(phase_allowance).unwrap();
    let baseline = parameter_fixture_forward(reference, true);
    assert_parameter_predictions(
        parameter_fixture_forward(runtime, true),
        &baseline,
        family.comparison_tolerance(),
    );
    // Establish ordinary cached parity independently of overlay rollback. The
    // fresh prefix below remains the shared starting point for failure trials.
    let ordinary_decode = parameter_fixture_forward(reference, false);
    assert_parameter_predictions(
        parameter_fixture_forward(runtime, false),
        &ordinary_decode,
        family.comparison_tolerance(),
    );
    runtime.reset().unwrap();
    reference.reset().unwrap();
    let baseline = parameter_fixture_forward(reference, true);
    assert_parameter_predictions(
        parameter_fixture_forward(runtime, true),
        &baseline,
        family.comparison_tolerance(),
    );
    let saved = MlxBackend::capture_native_text_state(runtime).unwrap();
    // Completed budget rejection precedes all native edit work on every rank.
    assert!(MlxBackend::activate_parameter_overlay(
        runtime,
        &overlay,
        if rank == 1 { facts.usage } else { limits }
    )
    .is_err());
    MlxBackend::validate_native_text_state(runtime, &saved).unwrap();
    // This failure occurs after native publication. Every owner restores its
    // weights, cached prefix and snapshot compatibility before the retry.
    if rank == 1 {
        runtime
            .session_mut()
            .reject_next_parameter_publication_for_test();
    }
    // A rejected attempt still charges prepared owners. Each retry gets a new
    // finite allowance above that cumulative work, just like bounded queries.
    let limits = parameter_fixture_limits(runtime, phase_allowance);
    let publication_error = MlxBackend::activate_parameter_overlay(runtime, &overlay, limits)
        .expect_err("injected post-publication rejection must roll back");
    if rank == 1 {
        assert!(
            publication_error
                .to_string()
                .contains("injected rejection after completed parameter publication"),
            "{publication_error}"
        );
    }
    MlxBackend::validate_native_text_state(runtime, &saved).unwrap();
    let rejected = MlxBackend::parameter_discovery(runtime).unwrap();
    assert_eq!(rejected.identity, facts.identity);
    assert_eq!(rejected.overlay_identity, None);
    assert_eq!(rejected.parameters, facts.parameters);
    let expected = parameter_fixture_forward(reference, false);
    assert_parameter_predictions(
        parameter_fixture_forward(runtime, false),
        &expected,
        family.comparison_tolerance(),
    );
    let limits = parameter_fixture_limits(runtime, phase_allowance);
    let active = MlxBackend::activate_parameter_overlay(runtime, &overlay, limits).unwrap();
    let limits = parameter_fixture_limits(reference, phase_allowance);
    let reference_active =
        MlxBackend::activate_parameter_overlay(reference, &reference_overlay, limits).unwrap();
    assert_ne!(active.identity, facts.identity);
    assert_eq!(
        active.overlay_identity.as_deref(),
        Some(overlay.intent_identity())
    );
    assert!(MlxBackend::validate_native_text_state(runtime, &saved).is_err());
    for edit in &edits {
        // Each bounded query receives an explicit allowance on top of its
        // ledger. Larger rank counts charge more coordination for this pass.
        let limits = parameter_fixture_limits(runtime, phase_allowance);
        let actual = MlxBackend::query_parameter(
            runtime,
            &active.identity,
            &edit.parameter,
            edit.region.clone(),
            limits,
        )
        .unwrap();
        let limits = parameter_fixture_limits(reference, phase_allowance);
        let expected = MlxBackend::query_parameter(
            reference,
            &reference_active.identity,
            &edit.parameter,
            edit.region.clone(),
            limits,
        )
        .unwrap();
        assert_eq!(
            actual.values, expected.values,
            "public active parameter {}",
            edit.parameter
        );
    }
    if family == FixtureFamily::DeepSeekV4 && std::env::var_os(REQUANTIZE).is_none() {
        v4_components::verify_independent_overlay(runtime, checkpoint, stream, &edits);
    }
    if family.is_k2_fp8() {
        verify_k2_fp8_independent_overlay(runtime, checkpoint, stream, &edits);
        if std::env::var_os(EXPERT_CACHE).is_some() {
            let report = runtime.session().parameter_bank_report().unwrap().unwrap();
            let bank = report
                .banks()
                .get(&eredu_architectures::k2_horizon::ExpertBank::FeedForward.id())
                .filter(|bank| {
                    bank.placements()
                        .iter()
                        .any(|(_, placement)| placement.owner_unit() == 1)
                });
            if let Some(bank) = bank {
                let compact_peak = bank
                    .bulk()
                    .peak_compact_bank_bytes()
                    .max(bank.incremental().peak_compact_bank_bytes());
                let encoded_member = bank.owned_bytes() / bank.owned_entries() as u64;
                // EP may construct a single expert at a time. One promoted
                // FFN member exceeds three encoded FFN members in this fixture;
                // a short FFN tail can be smaller than an unrelated value bank.
                assert!(
                    compact_peak > 3 * encoded_member,
                    "promoted FP32 FFN compact peak {compact_peak} must exceed three encoded FFN members of {encoded_member} bytes"
                );
            }
        }
    }
    let expected = parameter_fixture_forward(reference, true);
    assert!(
        expected
            .as_ref()
            .unwrap()
            .iter()
            .zip(baseline.as_ref().unwrap())
            .any(|(x, y)| (x - y).abs() > 1e-6),
        "nonzero parameter edit must affect inference"
    );
    assert_parameter_predictions(
        parameter_fixture_forward(runtime, true),
        &expected,
        family.comparison_tolerance(),
    );
    for _ in 0..2 {
        let expected = parameter_fixture_forward(reference, false);
        assert_parameter_predictions(
            parameter_fixture_forward(runtime, false),
            &expected,
            family.comparison_tolerance(),
        );
    }
    verify_active_parameter_capture_and_branches(runtime, reference, overlay.intent_identity());
    runtime.reset().unwrap();
    reference.reset().unwrap();
    let expected = parameter_fixture_forward(reference, true);
    assert_parameter_predictions(
        parameter_fixture_forward(runtime, true),
        &expected,
        family.comparison_tolerance(),
    );
    let saved_active = MlxBackend::capture_native_text_state(runtime).unwrap();
    assert!(MlxBackend::remove_parameter_overlay(
        runtime,
        if rank == 1 {
            "stale-version"
        } else {
            &active.identity
        }
    )
    .is_err());
    if rank == 1 {
        runtime
            .session_mut()
            .reject_next_parameter_publication_for_test();
    }
    let publication_error = MlxBackend::remove_parameter_overlay(runtime, &active.identity)
        .expect_err("injected removal rejection must restore active parameters");
    if rank == 1 {
        assert!(
            publication_error
                .to_string()
                .contains("injected rejection after completed parameter publication"),
            "{publication_error}"
        );
    }
    MlxBackend::validate_native_text_state(runtime, &saved_active).unwrap();
    let rejected = MlxBackend::parameter_discovery(runtime).unwrap();
    assert_eq!(rejected.identity, active.identity);
    assert_eq!(rejected.overlay_identity, active.overlay_identity);
    let expected = parameter_fixture_forward(reference, false);
    assert_parameter_predictions(
        parameter_fixture_forward(runtime, false),
        &expected,
        family.comparison_tolerance(),
    );
    let restored = MlxBackend::remove_parameter_overlay(runtime, &active.identity).unwrap();
    let reference_restored =
        MlxBackend::remove_parameter_overlay(reference, &reference_active.identity).unwrap();
    assert_ne!(restored.identity, active.identity);
    assert!(restored.overlay_identity.is_none());
    assert_eq!(restored.parameters, facts.parameters);
    assert!(MlxBackend::validate_native_text_state(runtime, &saved_active).is_err());
    // Restoration preserves all charged edit, query and coordination work.
    for edit in &edits {
        let limits = parameter_fixture_limits(runtime, phase_allowance);
        let actual = MlxBackend::query_parameter(
            runtime,
            &restored.identity,
            &edit.parameter,
            edit.region.clone(),
            limits,
        )
        .unwrap();
        let limits = parameter_fixture_limits(reference, phase_allowance);
        let expected = MlxBackend::query_parameter(
            reference,
            &reference_restored.identity,
            &edit.parameter,
            edit.region.clone(),
            limits,
        )
        .unwrap();
        assert_eq!(
            actual.values, expected.values,
            "public restored parameter {}",
            edit.parameter
        );
    }
    let expected = parameter_fixture_forward(reference, true);
    assert_eq!(expected, baseline);
    assert_parameter_predictions(
        parameter_fixture_forward(runtime, true),
        &expected,
        family.comparison_tolerance(),
    );
    for _ in 0..2 {
        let expected = parameter_fixture_forward(reference, false);
        assert_parameter_predictions(
            parameter_fixture_forward(runtime, false),
            &expected,
            family.comparison_tolerance(),
        );
    }
    runtime.session_mut().reset().unwrap();
    reference.session_mut().reset().unwrap();
}

fn verify_active_parameter_capture_and_branches(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    reference: &mut ModelRuntime<MlxBackend<'_>>,
    overlay: &str,
) {
    use eredu_core::{capture::*, TextGenerationBackend as _};
    let discovery = MlxBackend::capture_discovery(runtime).unwrap();
    let usage = CaptureUsage {
        captures: 1000,
        retained_bytes: 64 << 20,
        host_bytes: 128 << 20,
        encoded_bytes: 64 << 20,
    };
    let plan = CapturePlan {
        schema_version: 1,
        selections: vec![CaptureSelection {
            id: "edited-logits".into(),
            path: "model.logits".into(),
            schedule: CaptureSchedule::default(),
            slices: vec![],
            transform: CaptureTransform::FullTensor,
        }],
        limits: CaptureLimits {
            per_step: usage,
            cumulative: usage.checked_mul(8).unwrap(),
            physical_native_bytes: None,
            on_limit: CaptureLimitPolicy::Fail,
        },
    }
    .admit(
        &discovery.catalog,
        &discovery.support,
        &discovery.support.capture,
        CaptureRequestShape {
            batch: 1,
            prompt_tokens: component_capture_prompt_tokens().len() as u64,
            max_predictions: 3,
        },
    )
    .unwrap();
    let sampling = eredu_core::resolve_generation_config(
        None,
        eredu_core::GenerationConfigOverrides {
            max_new_tokens: Some(3),
            temperature: Some(0.0),
            ..Default::default()
        },
    )
    .unwrap();
    let run = |runtime: &mut ModelRuntime<MlxBackend<'_>>| {
        runtime.reset().unwrap();
        let mut generation = component_capture_generation(runtime, sampling);
        generation.enable_capture(plan.clone()).unwrap();
        (0..3)
            .map(|_| {
                let token = generation.next().unwrap().unwrap().token_id();
                (token, generation.take_captured_step().unwrap().unwrap())
            })
            .collect::<Vec<_>>()
    };
    let expected = run(reference);
    let actual = run(runtime);
    for ((token, step), (expected_token, expected_step)) in actual.iter().zip(&expected) {
        assert_eq!(token, expected_token);
        assert_eq!(step.partitions.len(), 1);
        assert_eq!(
            step.partitions[0].context.overlay_identity.as_deref(),
            Some(overlay)
        );
        let (Some(CapturePayload::Tensor(actual)), Some(CapturePayload::Tensor(expected))) =
            (&step.records[0].payload, &expected_step.records[0].payload)
        else {
            panic!("edited logits capture")
        };
        assert_eq!(actual.shape(), expected.shape());
        let (
            eredu_core::TensorObservationData::F32(actual),
            eredu_core::TensorObservationData::F32(expected),
        ) = (actual.data(), expected.data())
        else {
            panic!("F32 fixture")
        };
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= 2e-4 + 2e-4 * expected.abs());
        }
    }
    // Reuse the ordinary controlled driver and existing exact replay/sibling
    // conformance under the active parameter version, including nonrefunded
    // copy/observation budgets and a skipped-capture sibling.
    verify_component_capture_branches(runtime, &plan, sampling);
}
