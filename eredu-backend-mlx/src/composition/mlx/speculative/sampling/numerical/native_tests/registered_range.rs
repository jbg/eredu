//! Real I/full-B, independent copy, numerical range and escaped source custody.
use super::*;
use crate::backend::array_copy::IsolatedArrayCopy;
use crate::composition::mlx::MlxPreparedInputMaterializer;
use eredu_runtime::input::host::{
    HostInputPart, HostTensorValues, HostTensorView, PreparedHostInputPlan,
};

#[test]
#[ignore = "requires native Metal execution"]
fn original_registered_token_ranges_preserve_signed_bits_backing_and_escaped_custody() {
    let artifact = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_fixture(artifact.path());
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let backend = admitted_backend(&pool);
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
    let unsigned = [1u32, u32::MAX, 17, (i32::MAX as u32) + 1, 9];
    let signed = [2i32, -7, i32::MIN, 19, 5];
    let escaped = {
        let pair = AutoregressiveSourcePair::prepare(
            target.original_model_source().unwrap(),
            draft.original_model_source().unwrap(),
            &schedule,
            &pool,
            REQUEST_CEILING,
        )
        .unwrap();
        let environment = backend.original_copy_environment().unwrap();
        let sources = pair.numerical_sources();
        let (roots, mechanisms) = sources.numerical_prerequisites();
        let materializer = MlxPreparedInputMaterializer::prepare_admitted(&pool).unwrap();
        [
            HostTensorValues::U32(&unsigned),
            HostTensorValues::I32(&signed),
        ]
        .map(|values| {
            let parts = [HostInputPart {
                modality: eredu_core::InputModality::Text,
                kind: eredu_core::InputPayloadKind::TokenIds,
                payload: HostTensorView {
                    shape: &[1, 5],
                    values,
                },
                metadata: &[],
                extents: &[],
            }];
            let input = pool
                .compile_prepared_host_input(PreparedHostInputPlan::prepare(&parts).unwrap())
                .unwrap();
            let source = materializer.with_test_original_copy_source(&input, &pool, |original| {
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
            let source = registered_copy_input(source, sources, &environment).unwrap();
            let proof = RegisteredTensorSource::from_value(&source, sources, &environment).unwrap();
            let kind = program::SpeculativeNumericalKind::TokenRange { start: 1, end: 4 };
            let result=if source.value().array.dtype()==safemlx::Dtype::Uint32 {
                NumericalProducer::execute(sources,&environment,roots,mechanisms,kind,&source,None)
            } else {
                let context=SpeculativeExecutionStreams::single(environment.stream())
                    .with_original_sources(&pair,&environment).unwrap();
                // Selected Draft is the same actual stream in this assignment;
                // signed bits and escaped backing must follow the shared worker.
                NumericalProducer::execute_at(context,eredu_core::speculative::SamplingPlacement::Draft,
                    kind,&source,None)
            };
            let NumericalOutput::Tensor(output) = result.unwrap() else {
                panic!("range lost tensor meaning")
            };
            assert_eq!(output.value().array.shape(), &[1, 3]);
            assert_eq!(output.value().array.dtype(), source.value().array.dtype());
            proof
                .validate_array(&output.value().array, sources.metadata_funding())
                .unwrap();
            let backing = source
                .value()
                .array
                .try_allocation_info()
                .unwrap()
                .unwrap()
                .identity();
            assert_eq!(
                output
                    .value()
                    .array
                    .try_allocation_info()
                    .unwrap()
                    .unwrap()
                    .identity(),
                backing
            );
            drop(source);
            drop(input);
            (output, proof)
        })
    };
    assert!(pool.used_bytes().unwrap() > loaded);
    assert_eq!(pool.unquoted_owner_count().unwrap(), 0);
    drop((target, draft));
    assert_eq!(
        escaped[0]
            .0
            .value()
            .array
            .evaluated()
            .unwrap()
            .as_slice::<u32>(),
        &unsigned[1..4]
    );
    assert_eq!(
        escaped[1]
            .0
            .value()
            .array
            .evaluated()
            .unwrap()
            .as_slice::<i32>(),
        &signed[1..4]
    );
    assert_eq!(escaped[0].0.value().array.dtype(), safemlx::Dtype::Uint32);
    assert_eq!(escaped[1].0.value().array.dtype(), safemlx::Dtype::Int32);
    drop(escaped);
    drop(schedule);
    drop((target_config, draft_config, selected));
    settle(&pool, initial);
}
