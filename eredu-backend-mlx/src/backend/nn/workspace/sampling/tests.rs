use super::*;
use eredu_core::TokenFilter;
use eredu_nn::Tensor;
use eredu_runtime::{
    working_memory::{
        quote_sampling_workspace, WorkspaceSamplingBackend, WorkspaceSamplingRandomState,
    },
    ConfiguredTextSampler, GenerationSampler, MirostatV2Sampler, PenaltyConfig, Sampler,
    SamplingBackend,
};

fn selected() -> MlxMetalWorkspaceMechanisms {
    MlxMetalWorkspaceMechanisms {
        allocation: MetalAllocationFacts { page_size: 16384 },
        sdpa_blocks: None,
    }
}
fn penalty() -> PenaltyConfig {
    PenaltyConfig {
        repeat_penalty: 1.2,
        repeat_last_n: -1,
        frequency_penalty: 0.1,
        presence_penalty: 0.3,
    }
}
fn apply<B: SamplingBackend>(
    case: usize,
    logits: &B::Logits,
    context: &B::Context,
) -> Result<B::Logits, B::Error>
where
    B::Logits: Clone,
{
    match case {
        0 => B::apply_penalties(logits, &[2, 2, 5, u32::MAX, 1, 2, 5], penalty(), context),
        1 => B::apply_top_k(logits.clone(), 5, context),
        2 => B::apply_top_p(logits.clone(), 0.81, context),
        3 => B::apply_min_p(logits.clone(), 0.2, context),
        4 => B::apply_token_filter(
            logits,
            &TokenFilter::Allowed(vec![true, false, true]),
            context,
        ),
        5 => B::apply_mirostat(logits, &[2, 2, 5], penalty(), 0.7, 1.5, context),
        _ => unreachable!(),
    }
}

#[test]
fn sampling_native_facts_cover_sort_growth_host_masks_and_retained_key_replacement() {
    for width in [1, 37, 2048, 2049, 4097] {
        let context = WorkspaceContext::new(selected());
        let layout = WorkspaceLayout::new(&[1, width], WorkspaceDtype::Float32).unwrap();
        for case in 0..6 {
            context.begin_span();
            let logits = WorkspaceTensor::existing(layout.clone(), &context).unwrap();
            let output = apply::<WorkspaceSamplingBackend>(case, &logits, &context).unwrap();
            assert_eq!(output.layout(), &layout);
            let report = context.report(&[]).unwrap();
            assert!(report.total_bytes.is_some(), "case={case} width={width}");
            if case == 0 {
                assert_eq!(report.host_workspace_bytes, Some(width as u64 * 5 + 7 * 4));
            }
            if case == 4 {
                assert_eq!(report.host_workspace_bytes, Some(width as u64 * 2));
            }
        }
        let sampler =
            ConfiguredTextSampler::Standard(GenerationSampler::new().penalties(1.2, -1, 0.1, 0.3));
        let key = WorkspaceSamplingRandomState::from_seed(&context).unwrap();
        let report = quote_sampling_workspace(
            &sampler,
            0.7,
            Some(&key),
            &layout,
            &TokenFilter::All,
            9,
            &context,
        )
        .unwrap();
        assert!(report.peak.bytes().is_some());
        assert_eq!(report.final_history_bytes, 64);
    }
    let mut op = WorkspaceOperation {
        kind: WorkspaceOperationKind::Sampling(WorkspaceSamplingOperation::SplitRandomKey),
        inputs: vec![WorkspaceLayout::new(&[2], WorkspaceDtype::Uint32).unwrap()],
        outputs: vec![WorkspaceLayout::new(&[2, 2], WorkspaceDtype::Uint32).unwrap()],
    };
    assert!(matches!(
        selected().operation_bound(&op).unwrap().unwrap().outputs[0],
        WorkspaceOutputStorage::Allocate(_)
    ));
    for count in [1, 2, 128, 2048, 2049, 8193] {
        op.outputs[0] = WorkspaceLayout::new(&[count, 2], WorkspaceDtype::Uint32).unwrap();
        let bound = selected().operation_bound(&op).unwrap().unwrap();
        let expected = selected().allocation.buffer_capacity(count as u64 * 8).unwrap();
        assert!(matches!(bound.outputs[0], WorkspaceOutputStorage::Allocate(actual) if actual == expected));
        assert_eq!(bound.scratch_bytes, 0);
    }
    op.outputs[0] = WorkspaceLayout::new(&[3], WorkspaceDtype::Uint32).unwrap();
    assert!(selected().operation_bound(&op).is_err());
    op.kind = WorkspaceOperationKind::Sampling(WorkspaceSamplingOperation::ValidateToken {
        cardinality: 37,
    });
    op.outputs[0] = WorkspaceLayout::new(&[2], WorkspaceDtype::Int32).unwrap();
    assert!(selected().operation_bound(&op).unwrap().is_some());
    op.outputs[0] = op.inputs[0].clone();
    assert!(selected().operation_bound(&op).is_err());
}

#[test]
fn optional_filter_facts_cover_masked_work_and_unfiltered_input_aliases() {
    for width in [1, 37, 2049] {
        for rows in [1, 3] {
            let layout = WorkspaceLayout::new(&[rows, width], WorkspaceDtype::Float32).unwrap();
            let mut operation = WorkspaceOperation {
                kind: WorkspaceOperationKind::Sampling(WorkspaceSamplingOperation::TokenFilter),
                inputs: vec![layout.clone()],
                outputs: vec![layout.clone()],
            };
            let masked = selected().operation_bound(&operation).unwrap().unwrap();
            let masked_host = selected()
                .host_workspace_bound(&operation)
                .unwrap()
                .unwrap();
            operation.kind =
                WorkspaceOperationKind::Sampling(WorkspaceSamplingOperation::OptionalTokenFilter);
            let optional = selected().operation_bound(&operation).unwrap().unwrap();
            let optional_host = selected()
                .host_workspace_bound(&operation)
                .unwrap()
                .unwrap();
            assert_eq!(optional.scratch_bytes, masked.scratch_bytes);
            assert_eq!(optional_host.bytes, masked_host.bytes);
            assert_eq!(optional_host.bytes, (rows as u64 + 1) * width as u64);
            let output_bytes = match &optional.outputs[0] {
                WorkspaceOutputStorage::AllocateOrAliasInputs { bytes, inputs } => {
                    assert_eq!(inputs, &[0]);
                    *bytes
                }
                other => panic!("optional filtering lost an output possibility: {other:?}"),
            };
            assert!(matches!(
                &masked.outputs[0],
                WorkspaceOutputStorage::AllocateOrAliasInputs { bytes, inputs }
                    if *bytes == output_bytes && inputs == &[0]
            ));

            let context = WorkspaceContext::new(selected());
            // The native input may be a small logical view of a much larger
            // backing. Optional All must preserve that complete owner.
            let existing_bytes = output_bytes.checked_add(8192).unwrap();
            let storage = WorkspaceExistingStorage::new(Some(existing_bytes), &context);
            let input =
                WorkspaceTensor::existing_with_storage(layout.clone(), &storage, &context).unwrap();
            context.begin_state_span([&input]).unwrap();
            let output = context
                .execute(operation.kind, &[&input], vec![layout])
                .unwrap();
            let report = context.report(&output).unwrap();
            assert_eq!(
                report.state.as_ref().unwrap().retained_bytes,
                Some(existing_bytes + output_bytes)
            );
            assert_eq!(report.state.as_ref().unwrap().displaced_bytes, Some(0));
            assert!(report.total_bytes.is_some());
            assert_eq!(report.host_workspace_bytes, Some(masked_host.bytes));
        }
    }
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[ignore = "requires exclusive Metal allocator measurement; run with --test-threads=1"]
fn metal_sampling_primitive_peaks_fit_cold_bounds_across_sort_thresholds() {
    use crate::{backend::runtime::generation::MlxSamplingBackend, MlxTensor};
    use safemlx::{Array, Device, DeviceType, Dtype, Stream};
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let selected = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
        for width in [1, 37, 2048, 2049, 4097] {
            for rows in [1, 3] {
                for transposed in [false, true] {
                    let shape = [rows, width];
                    let count = (rows * width) as usize;
                    let values = (0..count)
                        .map(|i| ((i * 13 % 53) as f32 - 26.0) * 0.25)
                        .collect::<Vec<_>>();
                    let mut storage = values.clone();
                    if transposed {
                        for row in 0..rows as usize {
                            for col in 0..width as usize {
                                storage[col * rows as usize + row] =
                                    values[row * width as usize + col];
                            }
                        }
                    }
                    let storage_shape = if transposed { [width, rows] } else { shape };
                    let mut input = Array::from_slice(&storage, &storage_shape)
                        .as_dtype(dtype, &stream)
                        .unwrap();
                    if transposed {
                        input = input.transpose(&stream).unwrap();
                    }
                    let input = MlxTensor::from_array(input);
                    safemlx::transforms::eval([input.as_array()]).unwrap();
                    for case in 0..6 {
                        if case == 5 && rows != 1 {
                            continue;
                        }
                        let context = WorkspaceContext::new(selected);
                        let meta = WorkspaceTensor::existing(
                            WorkspaceLayout::new(&shape, WorkspaceDtype::Float32).unwrap(),
                            &context,
                        )
                        .unwrap();
                        let _output =
                            apply::<WorkspaceSamplingBackend>(case, &meta, &context).unwrap();
                        let allowed = context
                            .report(&[])
                            .unwrap()
                            .tensor_buffers
                            .total_bytes
                            .unwrap();
                        stream.synchronize().unwrap();
                        let before = safemlx::memory::active_memory().unwrap();
                        safemlx::memory::reset_peak_memory().unwrap();
                        let output = apply::<MlxSamplingBackend>(case, &input, &stream).unwrap();
                        safemlx::transforms::eval([output.as_array()]).unwrap();
                        stream.synchronize().unwrap();
                        let observed = safemlx::memory::peak_memory()
                            .unwrap()
                            .saturating_sub(before) as u64;
                        assert!(observed <= allowed, "sampling case={case} dtype={dtype:?} shape={shape:?} strided={transposed}: {observed} > {allowed}");
                        let actual = output.to_f32_vec(&stream).unwrap();
                        assert_eq!(actual.len(), count);
                        assert!(actual.iter().all(|value| !value.is_nan()));
                        if case == 0 {
                            for (index, (&raw, &actual)) in values.iter().zip(&actual).enumerate() {
                                let id = index % width as usize;
                                let occurrences = [2, 2, 5, u32::MAX, 1, 2, 5]
                                    .iter()
                                    .filter(|&&token| token as usize == id)
                                    .count();
                                let expected = if occurrences == 0 {
                                    raw
                                } else {
                                    (if raw > 0.0 { raw / 1.2 } else { raw * 1.2 })
                                        - 0.1 * occurrences as f32
                                        - 0.3
                                };
                                assert!(
                                    (actual - expected).abs() <= 0.08 + expected.abs() * 0.01,
                                    "penalty {actual} != {expected}"
                                );
                            }
                        }
                        eprintln!("sampling case={case} dtype={dtype:?} shape={shape:?} strided={transposed} observed={observed} bound={allowed}");
                    }
                }
            }
        }
    }
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[ignore = "requires exclusive Metal allocator measurement; run with --test-threads=1"]
fn metal_configured_sampling_request_peaks_fit_shared_policy_quotes() {
    use crate::{
        backend::{random::RandomState, runtime::generation::MlxSamplingBackend},
        MlxTensor,
    };
    use safemlx::{Array, Device, DeviceType, Stream};
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let facts = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    for width in [37, 2049, 4097] {
        for adaptive in [false, true] {
            for temperature in [0.0, 0.7] {
                if adaptive && temperature == 0.0 {
                    continue;
                }
                let mut sampler = if adaptive {
                    ConfiguredTextSampler::MirostatV2(
                        MirostatV2Sampler::new(5.0, 0.1)
                            .unwrap()
                            .penalties(1.2, -1, 0.1, 0.3),
                    )
                } else {
                    ConfiguredTextSampler::Standard(
                        GenerationSampler::new().penalties(1.2, -1, 0.1, 0.3),
                    )
                };
                let values = (0..width)
                    .map(|i| ((i * 13 % 53) as f32 - 26.0) * 0.25)
                    .collect::<Vec<_>>();
                let logits = MlxTensor::from_array(Array::from_slice(&values, &[1, 1, width]));
                safemlx::transforms::eval([logits.as_array()]).unwrap();
                let mut random = (temperature > 0.0).then(|| RandomState::with_seed(31).unwrap());
                if let Some(random) = &random {
                    safemlx::transforms::eval([random.as_array()]).unwrap();
                }
                let context = WorkspaceContext::new(facts);
                let meta_random = random.as_ref().map(|random| {
                    WorkspaceSamplingRandomState::from_key(
                        super::super::project_existing_arrays([random.as_array()], &context)
                            .unwrap()
                            .remove(0),
                    )
                    .unwrap()
                });
                let filter = TokenFilter::Allowed(vec![true, false, true, true, false, true]);
                let quote = quote_sampling_workspace(
                    &sampler,
                    temperature,
                    meta_random.as_ref(),
                    &WorkspaceLayout::new(logits.as_array().shape(), WorkspaceDtype::Float32)
                        .unwrap(),
                    &filter,
                    9,
                    &context,
                )
                .unwrap();
                stream.synchronize().unwrap();
                let before = safemlx::memory::active_memory().unwrap();
                safemlx::memory::reset_peak_memory().unwrap();
                for _ in 0..9 {
                    let masked =
                        MlxSamplingBackend::apply_token_filter(&logits, &filter, &stream).unwrap();
                    let token = Sampler::<MlxSamplingBackend>::sample(
                        &mut sampler,
                        &masked,
                        temperature,
                        random.as_mut(),
                        &stream,
                    )
                    .unwrap();
                    let id = MlxSamplingBackend::token_id(&token, &stream).unwrap();
                    assert!(filter.allows(id));
                }
                stream.synchronize().unwrap();
                let observed = safemlx::memory::peak_memory()
                    .unwrap()
                    .saturating_sub(before) as u64;
                assert!(
                    observed <= quote.tensor_peak_bytes.unwrap(),
                    "sampling request {observed} exceeds {:?}",
                    quote.tensor_peak_bytes
                );
                assert_eq!(sampler.history_len(), 9);
                assert_eq!(
                    sampler.history_capacity() as u64 * 4,
                    quote.final_history_bytes
                );
                eprintln!("configured sampling width={width} adaptive={adaptive} temperature={temperature} observed={observed} tensor_bound={:?} managed_bound={:?}", quote.tensor_peak_bytes, quote.peak.bytes());
            }
        }
    }
}

#[test]
fn uniform_facts_require_real_key_and_fixed_f32_draw_geometry() {
    let mut op = WorkspaceOperation {
        kind: WorkspaceOperationKind::Sampling(WorkspaceSamplingOperation::UniformUnitInterval),
        inputs: vec![WorkspaceLayout::new(&[2], WorkspaceDtype::Uint32).unwrap()],
        outputs: vec![WorkspaceLayout::new(&[1], WorkspaceDtype::Float32).unwrap()],
    };
    let bound = selected().operation_bound(&op).unwrap().unwrap();
    let scalar = selected().allocation.buffer_capacity(4).unwrap();
    assert!(matches!(bound.outputs[0], WorkspaceOutputStorage::Allocate(bytes) if bytes == scalar));
    assert!(bound.scratch_bytes >= 4 * scalar, "eager uniform sources must remain priced");
    for (shape, dtype) in [(&[2][..], WorkspaceDtype::Float32),
        (&[1][..], WorkspaceDtype::Uint32), (&[][..], WorkspaceDtype::Float32)] {
        op.outputs[0] = WorkspaceLayout::new(shape, dtype).unwrap();
        assert!(selected().operation_bound(&op).is_err());
    }
    op.outputs[0] = WorkspaceLayout::new(&[1], WorkspaceDtype::Float32).unwrap();
    op.inputs[0] = WorkspaceLayout::new(&[2], WorkspaceDtype::Float32).unwrap();
    assert!(selected().operation_bound(&op).is_err());
}
