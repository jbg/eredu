//! Persisted hybrid state resumes through the admitted shared text driver.
use super::*;
use eredu_core::cache::{PromptCacheDescriptor, PromptCacheOptions};

fn run(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    input: &[u32],
    predictions: usize,
    controlled: bool,
) -> Vec<u32> {
    let config = TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                temperature: Some(0.0),
                max_new_tokens: Some(predictions),
                ..Default::default()
            },
        )
        .unwrap(),
    )
    .with_inference_policy(eredu_core::TextInferencePolicy {
        prefill_chunk_positions: NonZeroU64::new(2),
        memory_limits: eredu_core::MemoryLimitDeclarations::new([(
            "host".into(),
            eredu_core::MemoryLimit::Finite(64 << 30),
        )]),
        submission_tracking_capacity_bytes: None,
        graph_metadata_capacity_bytes: None,
    });
    let mut output = Vec::new();
    if controlled {
        let mut run = ControlledTextGeneration::from_token_ids_with_sequence(
            runtime,
            eredu_core::TokenIdsInputPlan::new(input).unwrap(),
            config,
            disk::Controller::default(),
            None,
            GenerationSequenceRequest::new(predictions, &[]),
        )
        .unwrap();
        let mut sequence = run
            .take_prepared_sequence()
            .unwrap()
            .prepare_storage()
            .unwrap();
        for token in &mut run {
            let token = token.unwrap().token_id();
            sequence
                .commit(token, eredu_core::TokenTerminalSignals::default())
                .unwrap();
            output.push(token);
        }
        assert_eq!(sequence.tokens(), output.as_slice());
    } else {
        let mut run = TextGeneration::from_token_ids_with_sequence(
            runtime,
            eredu_core::TokenIdsInputPlan::new(input).unwrap(),
            config,
            TokenFilter::All,
            None,
            GenerationSequenceRequest::new(predictions, &[]),
        )
        .unwrap();
        let mut sequence = run
            .take_prepared_sequence()
            .unwrap()
            .prepare_storage()
            .unwrap();
        for token in &mut run {
            let token = token.unwrap().token_id().unwrap();
            sequence
                .commit(token, eredu_core::TokenTerminalSignals::default())
                .unwrap();
            output.push(token);
        }
        assert_eq!(sequence.tokens(), output.as_slice());
    }
    assert_eq!(output.len(), predictions);
    runtime.synchronize().unwrap();
    output
}

#[test]
fn admitted_hybrid_prompt_cache_restores_nonzero_state_and_controlled_continuation() {
    if !crate::tests::support::native_process::enter("hybrid-prompt-cache") {
        return;
    }
    crate::tests::support::test_utils::initialize_original_sources();
    use crate::composition::mlx::replicated_text::tests::{
        qwen_hybrid_config, tiny_heterogeneous_artifact,
    };
    let artifact = tiny_heterogeneous_artifact(qwen_hybrid_config());
    let inspection =
        eredu_architectures::configuration::inspect_artifact_with_prepared_gguf_headers(
            artifact.path(),
        )
        .unwrap();
    let factory = crate::MlxBackendFactory::default().with_state_residency(
        eredu_runtime::CacheResidencyPolicy::Paged(
            eredu_runtime::PagedCacheOptions::new(2, 1 << 20, 1 << 20, 1)
                .unwrap()
                .with_full_attention(true),
        ),
    );
    let plan = eredu_core::ExecutionPlan::fully_resident(
        eredu_core::DevicePlan::new("mlx", "metal:0").unwrap(),
    );
    let selected = eredu_core::select_execution_plan_target(&factory, &plan, inspection).unwrap();
    let target = eredu_core::realize_execution_plan_target(&factory, &plan, selected).unwrap();
    let mut runtime = target.into_runtime().unwrap();
    let prefix = [2, 5, 7];
    let next = run(&mut runtime, &prefix, 1, false)[0];
    let saved_numeric = runtime
        .session()
        .payload
        .model
        .erased()
        .fixed_numeric_state_snapshot()
        .unwrap();
    assert!(!saved_numeric.is_empty());
    assert!(
        saved_numeric
            .iter()
            .flat_map(|(_, _, _, values)| values)
            .all(|v| v.is_finite())
    );
    assert!(
        saved_numeric
            .iter()
            .flat_map(|(_, _, _, values)| values)
            .any(|v| *v != 0.0)
    );
    let descriptor = PromptCacheDescriptor::from_model_identity(
        runtime.session().prompt_cache_model_identity().unwrap(),
        "hybrid-admitted-fixture",
        "tokens:2,5,7",
        1,
    )
    .unwrap();
    let directory = tempfile::tempdir().unwrap();
    let destination = directory.path().join("cache");
    let saved = {
        let (backend, session) = runtime.parts_mut();
        session
            .save_prompt_cache(
                backend,
                &destination,
                descriptor.clone(),
                &prefix,
                &PromptCacheOptions::default(),
            )
            .unwrap()
    };
    assert!(!saved.blocks.is_empty());
    assert!(!saved.state_tensors.is_empty());
    let expected = run(&mut runtime, &[next], 3, false);
    let expected_numeric = runtime
        .session()
        .payload
        .model
        .erased()
        .fixed_numeric_state_snapshot()
        .unwrap();
    {
        let (backend, session) = runtime.parts_mut();
        let restored = session
            .load_prompt_cache(backend, &destination, &descriptor, &prefix)
            .unwrap();
        assert_eq!(restored.state_tensors, saved.state_tensors);
        assert_eq!(restored.blocks, saved.blocks);
    }
    let model = runtime.session().payload.model.erased();
    assert!(
        model
            .retained_inference_authority()
            .unwrap()
            .admission()
            .is_none()
    );
    assert_eq!(
        model.original_text_frontier().unwrap(),
        Some(prefix.len() as u64)
    );
    assert_eq!(model.fixed_numeric_state_snapshot().unwrap(), saved_numeric);
    assert!(
        !runtime
            .session()
            .payload
            .retained_idle_storage()
            .unwrap()
            .has_empty_decoder_storage()
            .unwrap()
    );
    let actual = run(&mut runtime, &[next], 3, true);
    assert_eq!(actual, expected);
    let actual_numeric = runtime
        .session()
        .payload
        .model
        .erased()
        .fixed_numeric_state_snapshot()
        .unwrap();
    assert_eq!(actual_numeric.len(), expected_numeric.len());
    for ((layer, role, shape, values), (expected_layer, expected_role, expected_shape, expected)) in
        actual_numeric.iter().zip(&expected_numeric)
    {
        assert_eq!(
            (layer, role, shape),
            (expected_layer, expected_role, expected_shape)
        );
        assert_eq!(values.len(), expected.len());
        assert!(
            values
                .iter()
                .zip(expected)
                .all(|(actual, expected)| actual.is_finite()
                    && expected.is_finite()
                    && (actual - expected).abs() <= 2e-5)
        );
    }
    assert_eq!(
        runtime
            .session()
            .payload
            .model
            .erased()
            .original_text_frontier()
            .unwrap(),
        Some(6)
    );
}
