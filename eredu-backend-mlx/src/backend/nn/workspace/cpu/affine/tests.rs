use super::*;
use eredu_checkpoint::AffineQuantization;
use eredu_nn::ParameterSpec;

#[test]
#[ignore = "requires qualified native allocator and selected CPU execution"]
fn original_cpu_affine_matches_independent_packed_weight_equations() {
    use crate::backend::{
        managed_memory::gpu_stream::PreparedExecutionStreams, MlxBackend, MlxDeviceIdentity,
    };
    use safemlx::{
        Array, Device, DeviceType, OriginalBufferBudget, OriginalScopeObserver, PrefillRoots,
        PrefillRootsRuntime, PreparedOriginalBufferBudget, PreparedPrefillFailure,
        PreparedSubmissionGraphQuota, PreparedSubmissionRecordQuota, PreparedSubmissionScopeOwner,
        SubmissionScope,
    };
    use std::sync::Arc;

    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let (ordinary, cpu) = mechanisms();
    let streams = PreparedExecutionStreams::for_cpu_factory_with_matmul(&pool, cpu.matmul)
        .unwrap()
        .unwrap();
    let backend = MlxBackend::for_prepared_execution_plan(
        streams,
        MlxDeviceIdentity::from_realized_device(&Device::new(DeviceType::Cpu, 0), None).unwrap(),
    );
    let environment = backend.original_copy_environment().unwrap();
    let stream = environment.stream();
    let runtime = PrefillRootsRuntime::prepare_for_stream(stream, stream).unwrap();
    let allocator = environment.input_runtime().unwrap();
    let width = 128usize;
    let columns = 7usize;
    for shape in [&[3, 128][..], &[2, 3, 128][..], &[2, 1, 3, 128][..]] {
        let rows = shape[..shape.len() - 1]
            .iter()
            .map(|&n| n as usize)
            .product::<usize>();
        let data = (0..rows * width)
            .map(|i| (i as i32 * 7 % 23 - 11) as f32 / 32.0)
            .collect::<Vec<_>>();
        for group in [16, 32, 64, 128] {
            for bits in [2, 3, 4, 5, 6, 8] {
                let mask = (1u32 << bits) - 1;
                let codes = (0..columns * width)
                    .map(|i| (i as u32 * 13 + 3) & mask)
                    .collect::<Vec<_>>();
                let mut packed = vec![0u32; columns * width * bits as usize / 32];
                // Independent little-endian bit packing, including word-crossing codes.
                for (i, &code) in codes.iter().enumerate() {
                    let bit = i * bits as usize;
                    packed[bit / 32] |= code << (bit % 32);
                    if bit % 32 + bits as usize > 32 {
                        packed[bit / 32 + 1] |= code >> (32 - bit % 32);
                    }
                }
                let groups = width / group as usize;
                let scales = (0..columns * groups)
                    .map(|i| (i % 5 + 1) as f32 / 64.0)
                    .collect::<Vec<_>>();
                let biases = (0..columns * groups)
                    .map(|i| (i as i32 % 7 - 3) as f32 / 16.0)
                    .collect::<Vec<_>>();
                let expected = (0..rows * columns)
                    .map(|index| {
                        let row = index / columns;
                        let column = index % columns;
                        (0..width)
                            .map(|k| {
                                let companion = column * groups + k / group as usize;
                                data[row * width + k]
                                    * (scales[companion] * codes[column * width + k] as f32
                                        + biases[companion])
                            })
                            .sum::<f32>()
                    })
                    .collect::<Vec<_>>();
                let sources = [
                    Array::from_slice(&data, shape),
                    Array::from_slice(
                        &packed,
                        &[columns as i32, (width * bits as usize / 32) as i32],
                    ),
                    Array::from_slice(&scales, &[columns as i32, groups as i32]),
                    Array::from_slice(&biases, &[columns as i32, groups as i32]),
                ];
                for source in &sources {
                    source.evaluated().unwrap();
                }
                let context = WorkspaceContext::new(cpu);
                let op = operation(shape, group, bits, WorkspaceFloatingType::Float32);
                let inputs = op
                    .inputs
                    .iter()
                    .map(|layout| WorkspaceTensor::existing(layout.clone(), &context).unwrap())
                    .collect::<Vec<_>>();
                let outputs = context
                    .execute(op.kind, &inputs.iter().collect::<Vec<_>>(), op.outputs)
                    .unwrap();
                let report = context.finish_report(&outputs).unwrap();
                let recipe = SpeculativeNumericalRecipe::inspect_cpu_equations(
                    &report, ordinary, cpu, &context,
                )
                .unwrap();
                let completion = recipe.completion;
                let physical = OriginalBufferBudget::population_layout(
                    &allocator,
                    recipe.storage.mutable_bytes() as usize,
                    recipe.storage.maximum_births(),
                )
                .unwrap()
                .capacity();
                let graph =
                    PreparedSubmissionGraphQuota::try_new(recipe.graph_capacity, Arc::new(()))
                        .unwrap()
                        .try_allocate()
                        .unwrap();
                let records =
                    PreparedSubmissionRecordQuota::try_new(recipe.record_capacity, Arc::new(()))
                        .unwrap()
                        .try_allocate()
                        .unwrap();
                let budget =
                    PreparedOriginalBufferBudget::try_new(&allocator, physical, Arc::new(()))
                        .unwrap()
                        .try_allocate()
                        .unwrap();
                let failure = PreparedPrefillFailure::try_new(Arc::new(()))
                    .unwrap()
                    .try_allocate()
                    .unwrap();
                let mut roots = PrefillRoots::new_retained(&runtime, 1, &graph, &failure).unwrap();
                let mut scope = SubmissionScope::try_begin_retaining(
                    PreparedSubmissionScopeOwner::try_new(Arc::new(()))
                        .unwrap()
                        .with_graph_quota(graph.clone())
                        .with_record_quota(records.clone()),
                )
                .unwrap();
                scope.enable_scoped_observation().unwrap();
                scope.require_original_native_controls().unwrap();
                roots.bind_scope(&scope).unwrap();
                scope.enable_original_native_controls().unwrap();
                scope.bind_original_buffer_budget(&budget).unwrap();
                let observer = OriginalScopeObserver::require_current().unwrap();
                for source in &sources {
                    OperationEvent::validate_traversal_leaf(source, &observer).unwrap();
                }
                let bank =
                    OperationEvent::prepare_resident_graph(completion.graph, &observer).unwrap();
                let actual = safemlx::ops::quantized_matmul_with_mode(
                    &sources[0],
                    &sources[1],
                    &sources[2],
                    Some(&sources[3]),
                    true,
                    group,
                    bits,
                    safemlx::ops::QuantizationMode::Affine,
                    stream,
                )
                .unwrap();
                drop(bank);
                roots.append(&actual).unwrap();
                roots
                    .complete_current_scope_on_stream_prepared(stream, &completion.traversal)
                    .unwrap_or_else(|e| panic!("shape={shape:?}, group={group}, bits={bits}: {e}"));
                assert!(!observer.status().failed());
                assert!(budget.occupied_bytes() > 0 && budget.occupied_bytes() <= physical);
                let values = actual.evaluated().unwrap().try_to_vec::<f32>().unwrap();
                for (i, (value, expected)) in values.iter().zip(&expected).enumerate() {
                    assert!((value - expected).abs() <= 1e-5,
                        "shape={shape:?}, group={group}, bits={bits}, index={i}: {value} != {expected}");
                }
                scope.seal();
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
                crate::backend::submission_recovery::wait_for_retirement(|| {
                    let (_, status) = observer.progress().unwrap();
                    assert!(!status.failed() && !status.blocked());
                    if status.is_settled() {
                        return true;
                    }
                    assert!(std::time::Instant::now() < deadline);
                    false
                });
                assert_eq!(
                    observer.retire_completed_records().unwrap(),
                    safemlx::SubmissionRetirement::CompleteSnapshot
                );
                safemlx::try_with_submission_retirement(|| {
                    drop((
                        actual, roots, scope, observer, failure, records, graph, budget,
                    ))
                })
                .unwrap();
                safemlx::reclaim_allocation_owners();
            }
        }
    }
}

fn floating(shape: &[i32], dtype: WorkspaceFloatingType) -> WorkspaceLayout {
    WorkspaceLayout::new(shape, WorkspaceDtype::Float32)
        .unwrap()
        .with_representation(Some(WorkspaceRepresentation::new(dtype, true)))
}

fn operation(
    shape: &[i32],
    group: i32,
    bits: i32,
    dtype: WorkspaceFloatingType,
) -> WorkspaceOperation {
    let width = shape[shape.len() - 1];
    let columns = 7;
    let mut output = shape.to_vec();
    *output.last_mut().unwrap() = columns;
    WorkspaceOperation {
        kind: WorkspaceOperationKind::Projection(
            LinearFormatSpec::affine(
                LinearFormat::Affine(AffineQuantization::new(group, bits).unwrap()),
                ParameterSpec::trainable("matrix.scales").unwrap(),
                ParameterSpec::trainable("matrix.biases").unwrap(),
            )
            .unwrap(),
        ),
        inputs: vec![
            floating(shape, dtype),
            WorkspaceLayout::new(&[columns, width * bits / 32], WorkspaceDtype::Uint32).unwrap(),
            floating(&[columns, width / group], dtype),
            floating(&[columns, width / group], dtype),
        ],
        outputs: vec![floating(&output, dtype)],
    }
}

fn mechanisms() -> (MlxMetalWorkspaceMechanisms, MlxCpuWorkspaceMechanisms) {
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let matmul =
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    (
        ordinary,
        MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), matmul),
    )
}

#[test]
fn cpu_affine_quotes_packed_worker_casts_and_output_bias() {
    let (_, cpu) = mechanisms();
    for shape in [&[3, 128][..], &[2, 3, 128][..], &[2, 1, 3, 128][..]] {
        for group in [16, 32, 64, 128] {
            for bits in [2, 3, 4, 5, 6, 8] {
                for dtype in [
                    WorkspaceFloatingType::Float32,
                    WorkspaceFloatingType::Float16,
                    WorkspaceFloatingType::Bfloat16,
                ] {
                    let op = operation(shape, group, bits, dtype);
                    let plan = cpu
                        .plan(op.as_view())
                        .unwrap()
                        .expect("packed affine worker");
                    assert_eq!(plan.dtype, dtype);
                    assert_eq!(plan.population.primitives, 1);
                    assert_eq!(plan.population.input_edges, 4);
                    assert_eq!(plan.population.births, 1);
                    assert_eq!(plan.population.maximum_captures, 6);
                    assert_eq!(plan.scratch_bytes, 0);
                    assert_eq!(plan.parameter_shells, 1);
                    assert_eq!(plan.seeds, 0);
                    assert_eq!(
                        plan.output_bytes,
                        cpu.allocation
                            .fixed_buffer_capacity(op.outputs[0].elements().unwrap() * 4)
                            .unwrap()
                    );
                }
            }
        }
    }
    let mut mixed = operation(&[1, 4, 128], 32, 4, WorkspaceFloatingType::Float16);
    mixed.inputs[2] = floating(&[7, 4], WorkspaceFloatingType::Float32);
    let plan = cpu.plan(mixed.as_view()).unwrap().unwrap();
    assert_eq!(plan.dtype, WorkspaceFloatingType::Float32);
    assert_eq!(plan.population.primitives, 3); // Cast x and affine biases, then QMM.
    assert_eq!(plan.population.births, 3);
    assert_eq!(plan.population.input_edges, 6);
    mixed
        .inputs
        .push(floating(&[7], WorkspaceFloatingType::Float32));
    let plan = cpu.plan(mixed.as_view()).unwrap().unwrap();
    assert_eq!(plan.population.primitives, 5); // Broadcast output bias and add.
    assert_eq!(plan.population.births, 4);
    assert_eq!(plan.population.input_edges, 9);
}

#[test]
fn cpu_affine_requires_geometry_and_actual_floating_sources() {
    let (_, cpu) = mechanisms();
    let op = operation(&[1, 4, 128], 32, 4, WorkspaceFloatingType::Float32);
    for ordinal in [0, 2, 3] {
        let mut unknown = op.clone();
        unknown.inputs[ordinal] = unknown.inputs[ordinal].clone().with_representation(None);
        assert!(cpu.plan(unknown.as_view()).unwrap().is_none());
        unknown.inputs[ordinal] =
            op.inputs[ordinal]
                .clone()
                .with_representation(Some(WorkspaceRepresentation::new(
                    WorkspaceFloatingType::Float32,
                    false,
                )));
        assert!(cpu.plan(unknown.as_view()).unwrap().is_none());
    }
    let mut wrong = op.clone();
    wrong.inputs[1] = WorkspaceLayout::new(&[7, 17], WorkspaceDtype::Uint32).unwrap();
    assert!(cpu.plan(wrong.as_view()).unwrap().is_none());
    let mut replacement = op;
    replacement.inputs[1] = floating(&[7, 128], WorkspaceFloatingType::Float32);
    replacement.inputs[2] = replacement.inputs[2].clone().with_representation(None);
    replacement.inputs[3] = replacement.inputs[3].clone().with_representation(None);
    assert!(
        cpu.plan(replacement.as_view()).unwrap().is_some(),
        "unused companions do not qualify a floating replacement"
    );
    for (rank, rows, columns, width, group, bits) in [
        (1, 1, 7, 128, 32, 4),
        (5, 1, 7, 128, 32, 4),
        (2, 0, 7, 128, 32, 4),
        (2, 1, 7, 127, 32, 4),
        (2, 1, 7, 128, 8, 4),
        (2, 1, 7, 128, 32, 7),
        (2, usize::MAX, 7, 128, 32, 4),
    ] {
        assert!(OperationEvent::cpu_affine_quantized_layout(
            Dtype::Float32,
            rank,
            rows,
            columns,
            width,
            group,
            bits,
            false
        )
        .is_none());
    }
}
