//! Actual shifted carry/prefix join with registered source, views and retirement.
use super::*;
use crate::backend::{
    array_copy::IsolatedArrayCopy, managed_memory::gpu_stream::PreparedExecutionStreams,
    MlxBackend, MlxDeviceIdentity,
};
use crate::composition::mlx::speculative::autoregressive::AutoregressiveSourcePair;
use crate::composition::mlx::speculative::sampling::numerical::native_tests::{
    admitted_backend, load, settle, source_configs, REQUEST_CEILING,
};
use crate::composition::mlx::MlxPreparedInputMaterializer;
use eredu_core::{SpeculativeConfig, SpeculativeSchedulerOptions};
use eredu_runtime::input::host::{
    HostInputPart, HostTensorValues, HostTensorView, PreparedHostInputPlan,
};
use eredu_runtime::speculative::autoregressive::AutoregressiveSchedulePlan;
use std::num::{NonZeroU64, NonZeroUsize};

#[test]
#[ignore = "requires managed native allocator and source-qualified CPU execution"]
fn original_cpu_capture_concatenation_joins_shifted_views_and_retires_escaped_sources() {
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
    let initial = pool.used_bytes().unwrap();
    let (target_config, draft_config, selected) = source_configs(&backend, artifact.path());
    let target = load(&backend, &target_config);
    let draft = load(&backend, &draft_config);
    let loaded = pool.used_bytes().unwrap();
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
    let data = (0..30)
        .map(|i| (i as f32 - 9.0) * 0.125)
        .collect::<Vec<_>>();
    let expected = data[24..27]
        .iter()
        .chain(&data[..3])
        .chain(&data[6..9])
        .copied()
        .collect::<Vec<_>>();
    let mut escaped = Vec::new();
    for shape in [&[1, 5, 6][..], &[1, 5, 2, 3][..]] {
        let pair = AutoregressiveSourcePair::prepare(
            target.original_model_source().unwrap(),
            draft.original_model_source().unwrap(),
            &schedule,
            &pool,
            REQUEST_CEILING,
        )
        .unwrap();
        let environment = cpu.original_copy_environment().unwrap();
        let sources = pair.numerical_sources();
        let (roots, mechanisms) = sources.numerical_prerequisites();
        let parts = [HostInputPart {
            modality: eredu_core::InputModality::Text,
            kind: eredu_core::InputPayloadKind::Embeddings,
            payload: HostTensorView {
                shape,
                values: HostTensorValues::F32(&data),
            },
            metadata: &[],
            extents: &[],
        }];
        let input = pool
            .compile_prepared_host_input(PreparedHostInputPlan::prepare(&parts).unwrap())
            .unwrap();
        let materializer = MlxPreparedInputMaterializer::prepare_admitted(&pool).unwrap();
        let copy = materializer.with_test_original_copy_source(&input, &pool, |original| {
            IsolatedArrayCopy::new(original.array())
                .copy_prepared(
                    &original,
                    &environment,
                    roots,
                    mechanisms,
                    sources.metadata_funding(),
                    REQUEST_CEILING,
                )
                .unwrap()
        });
        let proof = RegisteredTensorSource::from_copy(
            &copy,
            sources.request().source_identity(),
            environment.stream(),
            sources.metadata_funding(),
        )
        .unwrap();
        let (array, copy_owner) = copy.into_parts();
        let native = MlxTensor::from_array(array);
        let source = borrow_tensor(
            &native,
            None,
            Some(&proof),
            Meaning::Capture,
            sources,
            &environment,
        )
        .unwrap();
        drop((native, copy_owner, proof, input));
        let invoke =
            |kind, left: &OriginalNumericalValue, right: Option<&OriginalNumericalValue>| {
                let result = NumericalProducer::execute(
                    sources,
                    &environment,
                    roots,
                    mechanisms,
                    kind,
                    left,
                    right,
                )
                .unwrap();
                let NumericalOutput::Tensor(value) = result else {
                    panic!("capture operation lost tensor meaning")
                };
                value
            };
        // Narrow each source row while preserving an outer gap. A flags-only
        // unconditional row-major declaration would misdescribe this source.
        let narrow = invoke(
            program::SpeculativeNumericalKind::TensorAxisRange {
                axis: 2,
                start: 0,
                end: if shape.len() == 3 { 3 } else { 1 },
            },
            &source,
            None,
        );
        let carry = invoke(
            program::SpeculativeNumericalKind::TensorRange { start: 4, end: 5 },
            &narrow,
            None,
        );
        let prefix = invoke(
            program::SpeculativeNumericalKind::TensorRange { start: 0, end: 2 },
            &narrow,
            None,
        );
        assert_eq!(
            prefix
                .value()
                .array
                .try_descriptor()
                .unwrap()
                .row_contiguous(),
            Some(false)
        );
        let ordinary = safemlx::ops::concatenate_axis(
            &[&carry.value().array, &prefix.value().array],
            1,
            environment.stream(),
        )
        .unwrap();
        let oracle = ordinary.evaluated().unwrap().try_to_vec::<f32>().unwrap();
        drop(ordinary);
        assert_eq!(oracle, expected);
        let error = match NumericalProducer::execute(
            sources,
            &environment,
            roots,
            mechanisms,
            program::SpeculativeNumericalKind::TensorConcatenate { right_positions: 1 },
            &carry,
            Some(&prefix),
        ) {
            Err(error) => error,
            Ok(_) => panic!("a mismatched second-source position count must be refused"),
        };
        assert!(super::super::native::is_program_source_failure(&error),
            "the exact second-source refusal survives: {error:?}");
        let output = invoke(
            program::SpeculativeNumericalKind::TensorConcatenate { right_positions: 2 },
            &carry,
            Some(&prefix),
        );
        assert_eq!(
            output
                .value()
                .array
                .evaluated()
                .unwrap()
                .try_to_vec::<f32>()
                .unwrap(),
            oracle
        );
        assert_eq!(
            output
                .value()
                .array
                .try_descriptor()
                .unwrap()
                .row_contiguous(),
            Some(true)
        );
        assert_ne!(
            output
                .value()
                .array
                .try_allocation_info()
                .unwrap()
                .unwrap()
                .identity(),
            source
                .value()
                .array
                .try_allocation_info()
                .unwrap()
                .unwrap()
                .identity()
        );
        assert!(output
            .value()
            .provenance
            .source()
            .belongs_to_request(pair.request()));
        drop((carry, prefix, narrow, source));
        escaped.push((output, error));
    }
    assert!(pool.used_bytes().unwrap() > loaded);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    drop((target, draft));
    for (output, _) in &escaped {
        assert_eq!(
            output
                .value()
                .array
                .evaluated()
                .unwrap()
                .try_to_vec::<f32>()
                .unwrap(),
            expected
        );
    }
    drop(escaped);
    drop(schedule);
    drop((target_config, draft_config, selected));
    settle(&pool, initial);
}
