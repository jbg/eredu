//! Exact prepared B source, CPU pending tail, separate copy account and custody.
use super::*;
use crate::backend::array_copy::PreparedPendingTokenInput;
use crate::composition::mlx::MlxPreparedInputMaterializer;
#[cfg(test)]
use crate::memory_fixture::LedgerFixture as _;
use eredu_runtime::input::host::{
    HostInputPart, HostTensorValues, HostTensorView, PreparedHostInputPlan,
};
use std::cell::RefCell;

fn prepared(array: &safemlx::Array, elements: usize) -> PreparedPendingTokenInput<'_> {
    if elements == 1 {
        PreparedPendingTokenInput::new_fixed(array)
    } else {
        PreparedPendingTokenInput::new_prefill_fixed(
            array,
            NonZeroU64::new(elements as u64).unwrap(),
        )
    }
    .unwrap()
}

#[test]
#[ignore = "requires managed native Metal allocator and CPU execution"]
fn original_cpu_pending_input_preserves_signed_scalar_prefill_and_escaped_custody() {
    if !crate::tests::support::native_process::enter("original-speculative-source") {
        return;
    }
    let artifact = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_fixture(artifact.path());
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let backend = admitted_backend(&pool);
    let cpu = MlxBackend::for_prepared_execution_plan(
        PreparedExecutionStreams::for_cpu_factory(&pool)
            .unwrap()
            .unwrap(),
        MlxDeviceIdentity::from_realized_device(
            &safemlx::Device::new(safemlx::DeviceType::Cpu, 0),
            None,
        )
        .unwrap(),
    );
    let initial = pool.fixture_host_charge().unwrap();
    let (target_config, draft_config, selected) = source_configs(&backend, artifact.path());
    let target = load(&backend, &target_config);
    let draft = load(&backend, &draft_config);
    let loaded = pool.fixture_host_charge().unwrap();
    let config = SpeculativeConfig {
        max_tokens: 2,
        max_draft_tokens: 1,
        temperature: 0.0,
        eos_token_ids: Vec::new(),
    };
    let schedule = AutoregressiveSchedulePlan::new(
        &selected,
        NonZeroUsize::new(1).unwrap(),
        NonZeroU64::new(2).unwrap(),
        NonZeroU64::new(32).unwrap(),
        &config,
        SpeculativeSchedulerOptions::default(),
    )
    .unwrap();
    let signed = [2i32, -7, i32::MIN, 19, 5];
    let unsigned = [1u32, u32::MAX, 17, (i32::MAX as u32) + 1, 9];
    let signed_expected = signed.map(|v| v as u32);
    let scalar = [-7i32];
    let scalar_expected = [(-7i32) as u32];
    let scalar_unsigned = [u32::MAX];
    let escaped = {
        let pair = AutoregressiveSourcePair::prepare(
            target.original_model_source().unwrap(),
            draft.original_model_source().unwrap(),
            &schedule,
            &pool,
            crate::memory_fixture::resolved_limits(REQUEST_CEILING),
        )
        .unwrap();
        let environment = cpu.original_copy_environment().unwrap();
        let sources = pair.numerical_sources();
        let (roots, mechanisms) = sources.numerical_prerequisites();
        let materializer = MlxPreparedInputMaterializer::prepare_admitted(&pool).unwrap();
        [
            (
                &[1usize][..],
                HostTensorValues::I32(&scalar),
                &scalar_expected[..],
            ),
            (
                &[1usize, 1, 1][..],
                HostTensorValues::U32(&scalar_unsigned),
                &scalar_unsigned[..],
            ),
            (
                &[1usize, 5][..],
                HostTensorValues::I32(&signed),
                &signed_expected[..],
            ),
            (
                &[1usize, 5][..],
                HostTensorValues::U32(&unsigned),
                &unsigned[..],
            ),
        ]
        .map(|(shape, values, expected)| {
            let parts = [HostInputPart {
                modality: eredu_core::InputModality::Text,
                kind: eredu_core::InputPayloadKind::TokenIds,
                payload: HostTensorView { shape, values },
                metadata: &[],
                extents: &[],
            }];
            let input = pool
                .compile_prepared_host_input(PreparedHostInputPlan::prepare(&parts).unwrap())
                .unwrap();
            let (copy, refused) =
                materializer.with_test_original_copy_source(&input, &pool, |original| {
                    let source = original
                        .array()
                        .try_allocation_info()
                        .unwrap()
                        .unwrap()
                        .identity();
                    // Ordinary and admitted execution invoke the identical worker.
                    let ordinary_roots = RefCell::new(Vec::new());
                    let ordinary = prepared(original.array(), expected.len())
                        .copy_retained(environment.stream(), &ordinary_roots)
                        .unwrap();
                    let ordinary_values = ordinary
                        .array()
                        .evaluated()
                        .unwrap()
                        .try_to_vec::<u32>()
                        .unwrap();
                    assert_eq!(ordinary_values, expected);
                    drop((ordinary, ordinary_roots));
                    // Same backing is insufficient: the prepared loan names this
                    // exact array object. Refusal preserves its real source owner.
                    let alias = original.array().try_clone_handle().unwrap();
                    let refused = prepared(&alias, expected.len())
                        .copy_registered_with_prepared(
                            Some(&original),
                            &environment,
                            roots,
                            mechanisms,
                            sources.metadata_funding(),
                            &crate::memory_fixture::resolved_limits(REQUEST_CEILING),
                        )
                        .err()
                        .unwrap();
                    drop(alias);
                    let copy = prepared(original.array(), expected.len())
                        .copy_registered_with_prepared(
                            Some(&original),
                            &environment,
                            roots,
                            mechanisms,
                            sources.metadata_funding(),
                            &crate::memory_fixture::resolved_limits(REQUEST_CEILING),
                        )
                        .unwrap();
                    copy.validate_completed_stream(
                        environment.stream(),
                        sources.metadata_funding(),
                    )
                    .unwrap();
                    assert_ne!(
                        copy.array()
                            .try_allocation_info()
                            .unwrap()
                            .unwrap()
                            .identity(),
                        source
                    );
                    assert_eq!(copy.array().shape(), &[1, expected.len() as i32]);
                    assert_eq!(copy.array().dtype(), safemlx::Dtype::Uint32);
                    assert_eq!(
                        copy.array().evaluated().unwrap().as_slice::<u32>(),
                        ordinary_values
                    );
                    (copy, refused)
                });
            drop(input);
            (copy, refused, expected)
        })
    };
    assert!(pool.fixture_host_charge().unwrap() > loaded);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    drop((target, draft));
    for (copy, _, expected) in &escaped {
        assert_eq!(
            copy.array().evaluated().unwrap().as_slice::<u32>(),
            *expected
        );
    }
    drop(escaped);
    drop(schedule);
    drop((target_config, draft_config, selected));
    settle(&pool, initial);
}
