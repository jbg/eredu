use super::*;
use crate::composition::mlx::MlxBackend;
use eredu_core::refinement::{GenerationPhase, PredictionScoreKind};
use eredu_core::{ControlledTextGeneration, ModelRuntime, TextGenerationBackend, TokenOutput};
use std::{cell::Cell, rc::Rc};

fn checkpoint() -> tempfile::TempDir {
    checkpoint_with_sparse(true)
}

fn checkpoint_with_vision() -> tempfile::TempDir {
    let text = checkpoint();
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(text.path().join("config.json")).unwrap()).unwrap();
    config["image_token_id"] = 10.into();
    config["vision_config"] = serde_json::json!({
        "hidden_size":4, "intermediate_size":6, "num_hidden_layers":2,
        "num_attention_heads":1, "num_key_value_heads":1, "head_dim":4,
        "patch_size":2, "pooling_kernel_size":1, "position_embedding_size":4,
        "rms_norm_eps":0.00001, "standardize":true, "use_clipped_linears":false
    });
    tiny_heterogeneous_artifact(config)
}

fn checkpoint_with_sparse(sparse: bool) -> tempfile::TempDir {
    let mut config = serde_json::json!({
        "model_type": "diffusion_gemma", "canvas_length": 3,
        "text_config": {
            "model_type":"diffusion_gemma_text", "hidden_size":4, "num_hidden_layers":1,
            "intermediate_size":6, "num_attention_heads":1, "num_key_value_heads":1,
            "head_dim":4, "rms_norm_eps":0.00001, "vocab_size":11,
            "max_position_embeddings":64, "num_experts":3, "top_k_experts":2,
            "moe_intermediate_size":3, "layer_types":["full_attention"]
        }
    });
    if !sparse {
        let text = config["text_config"].as_object_mut().unwrap();
        text.insert("enable_moe_block".into(), false.into());
        for key in ["num_experts", "top_k_experts", "moe_intermediate_size"] {
            text.remove(key);
        }
    }
    tiny_heterogeneous_artifact(config)
}

#[test]
fn loaded_refinement_queries_constraints_only_for_finalization_and_encodes_authoritative_blocks() {
    struct Constraints {
        queries: Rc<Cell<usize>>,
        commits: Rc<Cell<usize>>,
    }
    impl eredu_core::TokenFilterController for Constraints {
        type Error = std::convert::Infallible;
        fn current_filter(&mut self) -> Result<eredu_core::TokenFilter, Self::Error> {
            assert_eq!(self.queries.get(), self.commits.get());
            let token = [5, 8, 2, 9][self.commits.get()];
            self.queries.set(self.queries.get() + 1);
            Ok(eredu_core::TokenFilter::allowed((0..11).map(|id| id == token).collect()).unwrap())
        }
        fn commit_token(&mut self, token: u32) -> Result<(), Self::Error> {
            assert_eq!(token, [5, 8, 2, 9][self.commits.get()]);
            self.commits.set(self.commits.get() + 1);
            Ok(())
        }
        fn is_complete(&mut self) -> Result<bool, Self::Error> {
            Ok(false)
        }
    }
    let policies = [
        eredu_runtime::OrdinaryWeightResidency::FullyResident,
        eredu_runtime::OrdinaryWeightResidency::LayerwiseHost(Default::default()),
        eredu_runtime::OrdinaryWeightResidency::DenseDiskStream(Default::default()),
    ];
    for root in [
        checkpoint_with_sparse(true),
        checkpoint_with_sparse(false),
        checkpoint_with_vision(),
    ] {
        for policy in policies {
            let (stream, weights) = execution_streams();
            let backend = crate::native::backend(&stream, &weights);
            let residency = eredu_runtime::WeightResidency::with_independent_parameter_banks(
                policy,
                Default::default(),
            );
            let request = crate::MlxLoadRequest::from_normalized(
                eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(residency),
            );
            let model = eredu_core::load_model(&backend, root.path(), request)
                .unwrap_or_else(|error| panic!("{policy:?}: {error}"));
            let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
            let queries = Rc::new(Cell::new(0));
            let commits = Rc::new(Cell::new(0));
            let sampling = eredu_core::resolve_generation_config(
                None,
                eredu_core::GenerationConfigOverrides {
                    max_new_tokens: Some(4),
                    temperature: Some(0.0),
                    ..Default::default()
                },
            )
            .unwrap();
            let mut generation = ControlledTextGeneration::new(
                &mut runtime,
                vec![1, 2],
                eredu_core::TextGenerationConfig::new(sampling).with_seed(13),
                Constraints {
                    queries: queries.clone(),
                    commits: commits.clone(),
                },
            )
            .unwrap();
            let mut contexts = 0;
            let mut refinements = 0;
            let mut tokens = Vec::new();
            while let Some(step) = generation.advance_operation().unwrap() {
                match step {
                    eredu_core::TextAdvancement::Progress(progress) => {
                        assert_eq!(queries.get(), commits.get());
                        match progress.phase {
                            GenerationPhase::Context => contexts += 1,
                            GenerationPhase::Refinement => refinements += 1,
                            GenerationPhase::Finalization => panic!("finalization is a commitment"),
                        }
                    }
                    eredu_core::TextAdvancement::Token(token) => {
                        assert_eq!(
                            token.score_kind(),
                            PredictionScoreKind::ConstrainedFinalization
                        );
                        tokens.push(token.output().token_id().unwrap());
                        assert_eq!(token.position().committed_tokens as usize, tokens.len());
                    }
                }
            }
            assert_eq!(tokens, [5, 8, 2, 9]);
            assert_eq!(contexts, 2, "a completed block re-enters context once");
            assert!(refinements >= 2);
            assert_eq!(queries.get(), 4);
            assert_eq!(commits.get(), 4);
        }
    }
}

#[test]
fn loaded_padded_media_context_and_refinement_match_across_residency_policies() {
    use crate::backend::runtime::media::input;
    use crate::composition::mlx::MlxModelInput;
    use eredu_core::{InputExtent, InputMetadataKey as Key, InputModality as Modality};
    fn media(modality: Modality, height: i32, width: i32) -> input::InputPart {
        let patches = height * width;
        let values: Vec<f32> = (0..patches * 12)
            .map(|i| (i as f32 * 0.031).sin() * 0.2)
            .collect();
        let positions: Vec<i32> = (0..patches).flat_map(|i| [i % width, i / width]).collect();
        input::input_part(
            modality,
            input::InputPayload::Tensor(Array::from_slice(&values, &[1, patches, 12])),
            [
                (
                    Key::PatchGrid,
                    Array::from_slice(&[1, height, width], &[1, 3]),
                ),
                (
                    Key::PatchPositions,
                    Array::from_slice(&positions, &[1, patches, 2]),
                ),
            ],
            [InputExtent::PatchGrid {
                time: 1,
                height: height as usize,
                width: width as usize,
            }],
        )
        .unwrap()
    }
    let root = checkpoint_with_vision();
    let parts = vec![
        input::token_ids_part(&Array::from_slice(&[1u32, 10], &[1, 2])).unwrap(),
        media(Modality::Image, 2, 2),
        input::token_ids_part(&Array::from_slice(&[3u32, 10], &[1, 2])).unwrap(),
        media(Modality::Video, 2, 3),
    ];
    let mut expected: Option<Vec<Vec<f32>>> = None;
    for policy in [
        eredu_runtime::OrdinaryWeightResidency::FullyResident,
        eredu_runtime::OrdinaryWeightResidency::LayerwiseHost(Default::default()),
        eredu_runtime::OrdinaryWeightResidency::DenseDiskStream(Default::default()),
    ] {
        let (stream, weights) = execution_streams();
        let backend = crate::native::backend(&stream, &weights);
        let request = crate::MlxLoadRequest::from_normalized(
            eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(
                eredu_runtime::WeightResidency::with_independent_parameter_banks(
                    policy,
                    Default::default(),
                ),
            ),
        );
        let model = eredu_core::load_model(&backend, root.path(), request).unwrap();
        let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
        let mut actual = Vec::new();
        let mut feedback = None;
        for step in 0..3 {
            let prompt = if step == 0 {
                MlxModelInput::from(input::ModelInput::new(&parts))
            } else {
                MlxBackend::prepare_text_prompt(runtime.backend(), vec![step as u32 + 4]).unwrap()
            };
            let encoded = runtime
                .prefill(prompt)
                .unwrap()
                .wait()
                .unwrap()
                .into_logits()
                .unwrap();
            actual.push(
                encoded
                    .as_array()
                    .as_dtype(Dtype::Float32, &stream)
                    .unwrap()
                    .into_evaluated()
                    .unwrap()
                    .try_to_vec::<f32>()
                    .unwrap(),
            );
            let before = runtime
                .session_mut()
                .with_model_operation(|model| Ok(model.erased().forecast_state_offset()?))
                .unwrap();
            for _ in 0..2 {
                let canvas = Array::from_slice(&[2i32, 4, 7], &[1, 3]);
                let scores = runtime
                    .session_mut()
                    .with_model_operation(|model| {
                        let output = model.erased_mut().refine_block_with_observer(
                            &canvas,
                            feedback.as_ref(),
                            &stream,
                            &mut eredu_runtime::NoopObserver,
                        )?;
                        output.evaluated()?;
                        Ok(output)
                    })
                    .unwrap();
                actual.push(
                    scores
                        .as_dtype(Dtype::Float32, &stream)
                        .unwrap()
                        .into_evaluated()
                        .unwrap()
                        .try_to_vec::<f32>()
                        .unwrap(),
                );
                feedback = Some(scores);
            }
            assert_eq!(
                runtime
                    .session_mut()
                    .with_model_operation(|model| Ok(model.erased().forecast_state_offset()?))
                    .unwrap(),
                before
            );
        }
        // The released encoder's text visibility is causal: changing text-only
        // chunk boundaries preserves its final encoding and subsequent canvas.
        let mut chunk_reference: Option<Vec<Vec<f32>>> = None;
        for chunks in [
            vec![vec![1, 2, 3, 4, 5]],
            vec![vec![1, 2], vec![3], vec![4, 5]],
        ] {
            runtime.reset().unwrap();
            let mut encoded = None;
            for tokens in chunks {
                let prompt = MlxBackend::prepare_text_prompt(runtime.backend(), tokens).unwrap();
                encoded = Some(
                    runtime
                        .prefill(prompt)
                        .unwrap()
                        .wait()
                        .unwrap()
                        .into_logits()
                        .unwrap(),
                );
            }
            let canvas = Array::from_slice(&[2i32, 4, 7], &[1, 3]);
            let scores = runtime
                .session_mut()
                .with_model_operation(|model| {
                    let output = model.erased_mut().refine_block_with_observer(
                        &canvas,
                        None,
                        &stream,
                        &mut eredu_runtime::NoopObserver,
                    )?;
                    output.evaluated()?;
                    Ok(output)
                })
                .unwrap();
            let values = [encoded.unwrap().as_array().clone(), scores]
                .into_iter()
                .enumerate()
                .map(|(index, value)| {
                    let data = value
                        .as_dtype(Dtype::Float32, &stream)
                        .unwrap()
                        .into_evaluated()
                        .unwrap()
                        .try_to_vec::<f32>()
                        .unwrap();
                    if index == 0 {
                        data[data.len() - 4..].to_vec()
                    } else {
                        data
                    }
                })
                .collect::<Vec<_>>();
            if let Some(expected) = &chunk_reference {
                assert_eq!(
                    values.iter().map(Vec::len).collect::<Vec<_>>(),
                    expected.iter().map(Vec::len).collect::<Vec<_>>()
                );
                for (a, b) in values.iter().flatten().zip(expected.iter().flatten()) {
                    assert!(
                        (a - b).abs() <= 2e-5,
                        "chunk boundary changed encoder/refinement: {a} != {b}"
                    );
                }
            } else {
                chunk_reference = Some(values);
            }
        }
        for row in &actual {
            assert!(row.iter().all(|x| x.is_finite()));
            assert!(row.iter().any(|x| x.abs() > 1e-8));
        }
        if let Some(expected) = &expected {
            for (actual, expected) in actual.iter().zip(expected) {
                assert_eq!(actual.len(), expected.len());
                for (a, b) in actual.iter().zip(expected) {
                    assert!((a - b).abs() <= 2e-5, "{policy:?}: {a} != {b}");
                }
            }
        } else {
            expected = Some(actual);
        }
    }
}

#[derive(Clone, Default)]
struct SnapshotController(Vec<u32>);

impl eredu_core::TokenFilterController for SnapshotController {
    type Error = std::convert::Infallible;
    fn current_filter(&mut self) -> Result<eredu_core::TokenFilter, Self::Error> {
        Ok(eredu_core::TokenFilter::All)
    }
    fn commit_token(&mut self, token: u32) -> Result<(), Self::Error> {
        self.0.push(token);
        Ok(())
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }
}

impl eredu_runtime::execution_control::SnapshotTokenController for SnapshotController {
    fn snapshot_storage_bytes(&self) -> Option<u64> {
        (std::mem::size_of::<Self>() as u64).checked_add((self.0.len() as u64).checked_mul(4)?)
    }
    fn fork_snapshot(&self) -> Result<Self, String> {
        Ok(self.clone())
    }
}

#[test]
fn loaded_block_snapshots_replay_and_fork_context_refinement_and_partial_finalization() {
    use eredu_core::{execution_control::SnapshotLimits, TextGenerationDriver};
    use eredu_runtime::execution_control::{
        ManagedTextContinuation, SnapshotBudget, TextBranchRequest, TextContinuationSnapshot,
    };
    let root = checkpoint();
    let (stream, weights) = execution_streams();
    let backend = crate::native::backend(&stream, &weights);
    let request = crate::MlxLoadRequest::from_normalized(
        eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(
            eredu_runtime::WeightResidency::with_independent_parameter_banks(
                eredu_runtime::OrdinaryWeightResidency::FullyResident,
                Default::default(),
            ),
        ),
    );
    let model = eredu_core::load_model(&backend, root.path(), request).unwrap();
    let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    let sampling = eredu_core::resolve_generation_config(
        None,
        eredu_core::GenerationConfigOverrides {
            max_new_tokens: Some(7),
            ..Default::default()
        },
    )
    .unwrap();
    let policy = eredu_core::refinement::RefinementOptions {
        max_denoising_steps: 2,
        stability_threshold: 10,
        ..Default::default()
    }
    .try_into()
    .unwrap();
    let budget = SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 1,
        max_branches: 1,
        retained_bytes: 128 << 20,
        cumulative_copy_bytes: 512 << 20,
    });
    // Boundaries after context, the first refinement, and the first committed
    // position preserve different pending inputs and mutable block state.
    for boundary_operations in [1, 2, 4] {
        runtime.reset().unwrap();
        use eredu_core::capture::*;
        let discovery = MlxBackend::capture_discovery(&runtime).unwrap();
        let point = discovery
            .catalog
            .points
            .iter()
            .find(|point| point.path == "model.decoder.layers.0.residual.scaled")
            .expect("shared stack output hook");
        let allowance = CaptureUsage {
            captures: 1000,
            retained_bytes: 16 << 20,
            host_bytes: 16 << 20,
            encoded_bytes: 16 << 20,
        };
        let request = CaptureRequestShape {
            batch: 1,
            prompt_tokens: 2,
            max_predictions: 7,
        };
        let capture = CapturePlan {
            schema_version: CAPTURE_SCHEMA_VERSION,
            provenance: eredu_core::capture::CaptureProvenance::ContentExact,
            selections: vec![CaptureSelection {
                id: "shared-output".into(),
                path: point.path.clone(),
                schedule: Default::default(),
                transform: CaptureTransform::FullTensor,
                slices: vec![],
            }],
            limits: CaptureLimits {
                per_step: allowance,
                cumulative: allowance,
                physical_native_bytes: None,
                on_limit: CaptureLimitPolicy::Fail,
            },
        }
        .admit_invocations(
            &discovery.catalog,
            &discovery.support,
            &discovery.support.capture,
            MlxBackend::text_capture_invocation_bounds(&runtime, request)
                .unwrap()
                .unwrap(),
        )
        .unwrap();
        let prompt = MlxBackend::prepare_text_prompt(runtime.backend(), vec![1, 2]).unwrap();
        let mut driver = TextGenerationDriver::new(&mut runtime);
        let mut continuation = driver
            .start(
                prompt,
                eredu_core::TextGenerationConfig::new(sampling)
                    .with_seed(13)
                    .with_refinement_policy(Some(policy))
                    .with_draft_events(true)
                    .with_refinement_canvas(Some(std::sync::Arc::new(
                        eredu_core::refinement::RefinementCanvas {
                            seed_tokens: vec![Some(5), None, Some(7)],
                            pinned_positions: vec![0, 2],
                        },
                    ))),
                SnapshotController::default(),
            )
            .unwrap();
        driver.enable_capture(&mut continuation, capture).unwrap();
        let mut state = ManagedTextContinuation::root(continuation);
        for _ in 0..boundary_operations {
            state.advance_operation(&mut driver).unwrap().unwrap();
            state.take_completed_step(&mut driver).unwrap();
        }
        let saved_position = state.position();
        let saved = TextContinuationSnapshot::capture(
            &mut state.boundary(&mut driver).unwrap(),
            &budget,
            Some(4096),
        )
        .unwrap();
        let growth = saved
            .native_continuation_growth(driver.runtime(), 7)
            .unwrap();
        let mut child = saved
            .fork(
                &mut state.boundary(&mut driver).unwrap(),
                &budget,
                TextBranchRequest {
                    session_id: "block-fork",
                    max_predictions: 7,
                    capture_limits: Some(CaptureLimits {
                        per_step: allowance,
                        cumulative: allowance,
                        physical_native_bytes: None,
                        on_limit: CaptureLimitPolicy::Fail,
                    }),
                    intervention: None,
                    host_bytes: Some(4096),
                    continuation_growth_bytes: Some(growth + (1 << 20)),
                },
            )
            .unwrap();
        fn collect<'a>(
            driver: &mut TextGenerationDriver<'_, MlxBackend<'a>>,
            state: &mut ManagedTextContinuation<MlxBackend<'a>, SnapshotController>,
        ) -> Vec<(
            eredu_core::refinement::GenerationPosition,
            (Option<GenerationPhase>, Option<u32>),
        )> {
            let mut trajectory = Vec::new();
            while let Some(step) = state.advance_operation(driver).unwrap() {
                let captured = state.take_completed_step(driver).unwrap();
                let value = match step {
                    eredu_core::TextAdvancement::Progress(progress) => {
                        let captured = captured.expect("one physical invocation capture");
                        let shape = captured.invocation.unwrap();
                        assert!(captured
                            .records
                            .iter()
                            .all(|record| record.outcome == CaptureOutcome::Captured));
                        assert_eq!(
                            shape.sequence,
                            if progress.phase == GenerationPhase::Context && progress.block == 0 {
                                2
                            } else {
                                3
                            }
                        );
                        assert_eq!(captured.prediction_index, state.position().committed_tokens);
                        if progress.phase == GenerationPhase::Refinement {
                            let ids = progress
                                .tentative_token_ids
                                .as_ref()
                                .expect("opt-in tentative IDs");
                            assert_eq!(ids.len(), 3);
                            assert_eq!(
                                (ids[0], ids[2]),
                                (5, 7),
                                "snapshot/fork lost pinned positions"
                            );
                            assert!(ids.iter().all(|id| *id < 11));
                        }
                        (Some(progress.phase), None)
                    }
                    eredu_core::TextAdvancement::Token(token) => {
                        assert!(captured.is_none(), "finalization has no model forward");
                        assert_eq!(
                            token.score_kind(),
                            PredictionScoreKind::ConstrainedFinalization
                        );
                        (Some(GenerationPhase::Finalization), Some(token.token_id()))
                    }
                };
                trajectory.push((state.position(), value));
            }
            trajectory
        }
        let expected = collect(&mut driver, &mut state);
        let copy_before = budget.usage().cumulative_copy_bytes;
        saved
            .restore(&mut state.boundary(&mut driver).unwrap(), &budget)
            .unwrap();
        assert_eq!(state.position(), saved_position);
        assert_eq!(collect(&mut driver, &mut state), expected);
        assert!(budget.usage().cumulative_copy_bytes > copy_before);
        child.exchange(&mut driver, &mut state).unwrap();
        assert_eq!(collect(&mut driver, &mut state), expected);
        child.exchange(&mut driver, &mut state).unwrap();
    }
}

/// Runs only with the pinned, externally stored release and independent fixture.
#[test]
#[ignore = "requires the pinned 50 GB released checkpoint and independent reference"]
fn released_diffusion_public_loader_matches_cached_reference() {
    released_public_loader_reference(false);
}

#[test]
#[ignore = "requires the pinned published GGUF and independent dequantized reference"]
fn released_diffusion_gguf_public_loader_matches_cached_reference() {
    released_public_loader_reference(true);
}

fn released_public_loader_reference(gguf: bool) {
    use eredu_nn::Tensor;
    let root = std::path::PathBuf::from(
        std::env::var_os("EREDU_DIFFUSION_CHECKPOINT").expect("checkpoint path"),
    );
    let reference_path = std::path::PathBuf::from(
        std::env::var_os("EREDU_DIFFUSION_REFERENCE").expect("reference path"),
    );
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("source_manifest.json")).unwrap()).unwrap();
    let reference_manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(reference_path.with_extension("json")).unwrap())
            .unwrap();
    assert_eq!(manifest["id"], "google/diffusiongemma-26B-A4B-it");
    assert_eq!(manifest["sha"], "f7f5b7f5fa82ffc52addd066915886d497f5517b");
    assert_eq!(reference_manifest["model_revision"], manifest["sha"]);
    assert_eq!(
        reference_manifest["reference_revision"],
        "55c3cc9f03cd4fbf2d27d382dcc8fcc544631f8e"
    );
    let (stream, weights) = execution_streams();
    let reference = Array::load_safetensors(reference_path, &stream).unwrap();
    let backend = crate::native::backend(&stream, &weights);
    let request = crate::MlxLoadRequest::from_normalized(
        eredu_runtime::NormalizedLoadRequest::default().with_weight_residency(
            eredu_runtime::WeightResidency::with_independent_parameter_banks(
                eredu_runtime::OrdinaryWeightResidency::FullyResident,
                Default::default(),
            ),
        ),
    );
    let path = if gguf {
        std::path::PathBuf::from(std::env::var_os("EREDU_DIFFUSION_GGUF").expect("GGUF path"))
    } else {
        root
    };
    if gguf {
        assert_eq!(
            reference_manifest["quantized_source"]["sha256"],
            "24523b6c833c9ce9f5f34f9b333ab1517d73d6f1e76a103645353114c8028bc5"
        );
    }
    let model = eredu_core::load_model(&backend, &path, request).unwrap();
    let mut runtime = ModelRuntime::from_prepared(backend, model).unwrap();
    println!("Loaded pinned released checkpoint through public admission and native construction");
    fn compare(
        label: &str,
        actual: &Array,
        expected: &Array,
        stream: &Stream,
        tolerance: f32,
        relative_rms_limit: Option<f64>,
    ) {
        assert_eq!(actual.shape(), expected.shape(), "{label}");
        let actual = actual
            .as_dtype(Dtype::Float32, stream)
            .unwrap()
            .into_evaluated()
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap();
        let expected = expected
            .as_dtype(Dtype::Float32, stream)
            .unwrap()
            .into_evaluated()
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap();
        let maximum = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let rms = (actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| f64::from(a - b).powi(2))
            .sum::<f64>()
            / actual.len() as f64)
            .sqrt();
        let reference_rms = (expected.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>()
            / expected.len() as f64)
            .sqrt();
        let relative_rms = rms / reference_rms.max(f64::MIN_POSITIVE);
        println!("{label}: max_absolute_error={maximum} rms_error={rms} relative_rms_error={relative_rms}");
        assert!(actual
            .iter()
            .chain(&expected)
            .all(|value| value.is_finite()));
        assert!(maximum <= tolerance, "{label}: {maximum}");
        if label.starts_with("logits.") || label.starts_with("shared_feedback.") {
            let width = expected.len() / 2;
            let argmax = |row: &[f32]| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .unwrap()
                    .0
            };
            for (actual, expected) in actual.chunks_exact(width).zip(expected.chunks_exact(width)) {
                assert_eq!(
                    argmax(actual),
                    argmax(expected),
                    "{label}: changed prediction"
                );
            }
        }
        if let Some(limit) = relative_rms_limit {
            assert!(
                relative_rms <= limit,
                "{label}: relative rms={relative_rms}"
            );
        }
    }
    let cases: [(&[u32], &[i32]); 3] = [
        (&[2, 100, 200], &[101, 211]),
        (&[17], &[421, 653]),
        (&[31, 29], &[1019, 2047]),
    ];
    let mut feedback = None;
    for (step, (tokens, canvas)) in cases.into_iter().enumerate() {
        let prompt = MlxBackend::prepare_text_prompt(runtime.backend(), tokens.to_vec()).unwrap();
        let encoded = runtime
            .prefill(prompt)
            .unwrap()
            .wait()
            .unwrap()
            .into_logits()
            .unwrap();
        compare(
            &format!("encode.{step}"),
            encoded.as_array(),
            &reference[&format!("encode.{step}")],
            &stream,
            if gguf { 1e-3 } else { 2e-5 },
            None,
        );
        let canvas = Array::from_slice(canvas, &[1, canvas.len() as i32]);
        if gguf && step == 1 {
            let shared_feedback = &reference["prior.0"];
            let probe = runtime
                .session_mut()
                .with_model_operation(|model| {
                    let output = model.erased_mut().refine_block_with_observer(
                        &canvas,
                        Some(shared_feedback),
                        &stream,
                        &mut eredu_runtime::NoopObserver,
                    )?;
                    output.evaluated()?;
                    Ok(output)
                })
                .unwrap();
            compare(
                "shared_feedback.1",
                &probe,
                &reference["logits.1"],
                &stream,
                1e-2,
                Some(2e-4),
            );
        }
        let decoded = runtime
            .session_mut()
            .with_model_operation(|model| {
                let output = model.erased_mut().refine_block_with_observer(
                    &canvas,
                    feedback.as_ref(),
                    &stream,
                    &mut eredu_runtime::NoopObserver,
                )?;
                output.evaluated()?;
                Ok(output)
            })
            .unwrap();
        compare(
            &format!("logits.{step}"),
            &decoded,
            &reference[&format!("logits.{step}")],
            &stream,
            if gguf { 1e-2 } else { 2e-5 },
            gguf.then_some(2e-4),
        );
        feedback = Some(
            MlxTensor::from_array(decoded)
                .divide(
                    &MlxTensor::from_f32_slice(&[0.8], &[1], &stream).unwrap(),
                    &stream,
                )
                .unwrap()
                .cast_float(
                    if gguf {
                        eredu_nn::TensorElementType::F32
                    } else {
                        eredu_nn::TensorElementType::Bf16
                    },
                    &stream,
                )
                .unwrap()
                .into_array(),
        );
        assert_eq!(
            runtime
                .session_mut()
                .with_model_operation(|model| Ok(model.erased().forecast_state_offset()?))
                .unwrap(),
            Some([3, 4, 6][step])
        );
    }
}
