//! Global IDs yield exact local row contributions before any collective.
use super::*;
use eredu_nn::DistributedNeuralBackend as _;

#[test]
fn mlx_vocabulary_local_contribution_preserves_global_ids_and_sentinel() {
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let width = 32;
    let vocabulary = 7;
    let source = (0..vocabulary * width)
        .map(|i| ((i * 17 % 113) as f32 - 56.) * 0.0317)
        .collect::<Vec<_>>();
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        let rounded = source
            .iter()
            .map(|v| {
                if dtype == Dtype::Bfloat16 {
                    half::bf16::from_f32(*v).to_f32()
                } else {
                    *v
                }
            })
            .collect::<Vec<_>>();
        let ids = [-1i32, 0, 3, 6, 3];
        let tokens = MlxTensor::from_array(Array::from_slice(&ids, &[1, ids.len() as i32]));
        let mut contributions = Vec::new();
        for rows in [0..3, 3..7] {
            let mut table = MlxNeuralBackend::vocabulary_parallel_embedding(
                EmbeddingSpec {
                    vocabulary,
                    dimensions: width,
                    weight: parameter("table.weight"),
                    format: test_format("table.weight", LinearFormat::Dense),
                },
                eredu_nn::VocabularyParallelRange {
                    global_vocabulary: vocabulary as usize,
                    local: rows.clone(),
                },
                stream,
            )
            .unwrap();
            bind_embedding(
                &mut table,
                Array::from_slice(
                    &source[rows.start * width as usize..rows.end * width as usize],
                    &[rows.len() as i32, width],
                )
                .as_dtype(dtype, stream)
                .unwrap(),
                stream,
            );
            let scope = TokenValidationScope::begin().unwrap();
            let value = table
                .lookup(&tokens, EmbeddingLookupPolicy::ZeroSentinel(-1), stream)
                .unwrap();
            assert_eq!(value.as_array().dtype(), dtype);
            let exported = value.as_array().as_dtype(Dtype::Float32, stream).unwrap();
            let validations = scope.finish();
            async_eval_with_event(std::iter::once(&exported).chain(validations.arrays()))
                .unwrap()
                .synchronize()
                .unwrap();
            validations.validate_completed().unwrap();
            let expected = ids
                .iter()
                .flat_map(|id| {
                    let rounded = &rounded;
                    let rows = &rows;
                    (0..width as usize).map(move |column| {
                        if *id >= 0 && rows.contains(&(*id as usize)) {
                            rounded[*id as usize * width as usize + column]
                        } else {
                            0.
                        }
                    })
                })
                .collect::<Vec<_>>();
            close(&MlxTensor::from_array(exported), &expected, 0.);
            contributions.push(value);
            for invalid in [-2i32, vocabulary] {
                let scope = TokenValidationScope::begin().unwrap();
                let invalid = MlxTensor::from_array(Array::from_slice(&[invalid], &[1, 1]));
                let value = table
                    .lookup(&invalid, EmbeddingLookupPolicy::ZeroSentinel(-1), stream)
                    .unwrap();
                let validations = scope.finish();
                async_eval_with_event(
                    std::iter::once(value.as_array()).chain(validations.arrays()),
                )
                .unwrap()
                .synchronize()
                .unwrap();
                assert!(
                    validations.validate_completed().is_err(),
                    "global bounds must survive shard masking"
                );
            }
        }
        let combined = contributions[0].add(&contributions[1], stream).unwrap();
        let expected = ids
            .iter()
            .flat_map(|id| {
                let rounded = &rounded;
                (0..width as usize).map(move |column| {
                    if *id >= 0 {
                        rounded[*id as usize * width as usize + column]
                    } else {
                        0.
                    }
                })
            })
            .collect::<Vec<_>>();
        close(
            &MlxTensor::from_array(
                combined
                    .as_array()
                    .as_dtype(Dtype::Float32, stream)
                    .unwrap(),
            ),
            &expected,
            0.,
        );
    }
}
