//! Nonzero prepared bank reads compared with the same resident parameter.
use crate::backend::MlxBackend;
use eredu_core::capture::CaptureUsage;
use eredu_core::parameters::{ParameterBackend, ParameterProjection, ParameterRegion};
use eredu_core::residency::CacheEvictionPolicy;
use eredu_core::{
    DevicePlan, ExecutionPlan, ExpertCachePlan, TextGeneration, TextGenerationConfig, TokenOutput,
};
use eredu_runtime::parameter_operations::PreparedParameterLocation;

#[test]
fn prepared_bank_query_keeps_member_values_result_custody_and_future_generation() {
    if !crate::tests::support::native_process::enter("prepared parameter bank query") {
        return;
    }
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let initial = pool.snapshot().unwrap();
    let artifact =
        crate::composition::mlx::replicated_text::tests::tiny_artifact("qwen3_moe", false);
    let limits = CaptureUsage {
        captures: u64::MAX,
        retained_bytes: u64::MAX,
        host_bytes: u64::MAX,
        encoded_bytes: u64::MAX,
    };
    let mut parameter = None;
    let mut reference = None;
    let mut expected_tokens = None;
    // Inspect the independently selected source first so its actual bank slot
    // chooses the declaration. The resident run uses that exact same identity.
    for independent in [true, false] {
        let mut plan = ExecutionPlan::fully_resident(DevicePlan::new("mlx", "cpu:0").unwrap());
        if independent {
            plan = plan.with_expert_cache(Some(ExpertCachePlan::new(
                Some(1 << 20),
                Some(1 << 20),
                1 << 20,
                1 << 20,
                CacheEvictionPolicy::LeastRecentlyUsed,
            )));
        }
        let inspection =
            eredu_architectures::configuration::inspect_artifact_with_prepared_gguf_headers(
                artifact.path(),
            )
            .unwrap();
        let factory = crate::MlxBackendFactory::default();
        let selected =
            eredu_core::select_execution_plan_target(&factory, &plan, inspection).unwrap();
        let target = eredu_core::realize_execution_plan_target(&factory, &plan, selected).unwrap();
        let mut runtime = target.into_runtime().unwrap();
        if independent {
            parameter = Some(
                runtime
                    .session()
                    .original_model_source()
                    .unwrap()
                    .erased()
                    .prepared_parameter_slots()
                    .iter()
                    .find(|slot| {
                        matches!(slot.location, PreparedParameterLocation::Bank { .. })
                            && slot.materialized.shape.len() == 3
                            && slot.materialized.shape.iter().all(|&n| n >= 2)
                    })
                    .expect("actual selected independent-bank matrix")
                    .parameter
                    .id
                    .as_str()
                    .to_owned(),
            );
        }
        let id = parameter.as_ref().unwrap();
        let discovery = MlxBackend::parameter_discovery(&mut runtime).unwrap();
        let region = ParameterRegion {
            starts: vec![0, 0, 0],
            shape: vec![2, 2, 2],
        };
        let result = MlxBackend::query_parameter(
            &mut runtime,
            &discovery.identity,
            id,
            region.clone(),
            limits,
        )
        .unwrap();
        assert!(result.values.iter().any(|&value| value != 0.0));
        if let Some(expected) = &reference {
            assert_eq!(&result.values, expected);
        } else {
            reference = Some(result.values.clone());
        }
        let alias = result.clone();
        let projected = MlxBackend::project_parameter(
            &mut runtime,
            &discovery.identity,
            id,
            ParameterProjection {
                region,
                axis: 2,
                directions: 1,
                coefficients: vec![0.5, -0.25],
            },
            limits,
        )
        .unwrap();
        for (row, actual) in projected.values.iter().enumerate() {
            let expected = result.values[row * 2] * 0.5 - result.values[row * 2 + 1] * 0.25;
            assert!((actual - expected).abs() < 1e-5);
        }
        drop(result);
        assert_eq!(&alias.values, reference.as_ref().unwrap());
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
        let config = TextGenerationConfig::new(
            eredu_core::resolve_generation_config(
                None,
                eredu_core::GenerationConfigOverrides {
                    temperature: Some(0.0),
                    max_new_tokens: Some(2),
                    top_k: Some(0),
                    top_p: Some(1.0),
                    min_p: Some(0.0),
                    repetition_penalty: Some(1.0),
                    ..Default::default()
                },
            )
            .unwrap(),
        )
        .with_seed(31);
        let generated = TextGeneration::new(&mut runtime, vec![1, 2], config)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(generated.len(), 2);
        let token_ids = generated
            .iter()
            .map(|value| value.token_id().unwrap())
            .collect::<Vec<_>>();
        if let Some(expected) = &expected_tokens {
            assert_eq!(&token_ids, expected);
        } else {
            expected_tokens = Some(token_ids);
        }
        assert_eq!(&alias.values, reference.as_ref().unwrap());
        assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
        // Each comparison loads a new independently admitted model. Retire the
        // completed request and its retained result owners before that load.
        drop(generated);
        drop(projected);
        drop(alias);
        drop(discovery);
        drop(runtime);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            safemlx::memory::clear_cache().unwrap();
            crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
            safemlx::reclaim_allocation_owners();
            crate::backend::ordinary_retirement::reclaim_all();
            let retired = pool.snapshot().unwrap();
            if retired.funding_accounts == initial.funding_accounts
                && retired.reservations == initial.reservations
            {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "completed bank comparison owners did not retire: {retired:?}"
            );
            std::thread::yield_now();
        }
    }
}
