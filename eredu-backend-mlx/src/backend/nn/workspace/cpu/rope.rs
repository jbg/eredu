//! Existing full-width CPU RoPE fallback and its actual supplied-frequency wrapper.
use super::*;
use eredu_nn::{RotaryAlgorithm, RotaryArithmetic};
use safemlx::{CpuUnaryOperation, Dtype};
mod embeddings;
mod input_products;
fn cast(p: &mut CpuPopulation, from: Dtype, rank: usize, count: usize) -> Option<()> {
    p.copy(
        OperationEvent::cpu_cast_layout(from, Dtype::Float32, rank, count, false)?,
        1,
    )
}
fn unary(p: &mut CpuPopulation, kind: CpuUnaryOperation, rank: usize, count: usize) -> Option<()> {
    cast(p, Dtype::Float32, rank, count)?;
    p.unary(OperationEvent::cpu_unary_layout(
        kind,
        Dtype::Float32,
        rank,
        false,
    )?)
}
fn binary(
    p: &mut CpuPopulation,
    kind: CpuBinaryOperation,
    rank: usize,
    count: usize,
    left: (usize, usize, Dtype),
    right: (usize, usize, Dtype),
) -> Option<()> {
    for (r, n, dtype) in [left, right] {
        cast(p, dtype, r, n)?;
        p.copy(
            OperationEvent::cpu_broadcast_alias_layout(r, rank, false)?,
            1,
        )?;
    }
    p.binary(OperationEvent::cpu_binary_layout(
        kind,
        Dtype::Float32,
        rank,
        count,
        false,
    )?)
}
pub(super) fn inspect(
    operation: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    if matches!(operation.kind, WorkspaceOperationKindView::Rotary(_, None)) {
        return embeddings::inspect(operation, mechanism);
    }
    let (spec, offset) = match operation.kind {
        WorkspaceOperationKindView::Rotary(spec, Some(offset)) => (spec, offset),
        _ => return Ok(None),
    };
    if spec.arithmetic == RotaryArithmetic::InputProducts {
        return input_products::inspect(operation, mechanism, spec, offset);
    }
    if spec.arithmetic != RotaryArithmetic::Native
        || !matches!(
            spec.algorithm,
            RotaryAlgorithm::Default | RotaryAlgorithm::Yarn { .. }
        )
        || spec.traditional
    {
        return Ok(None);
    }
    spec.algorithm.validate_fixed()?;
    if spec.dimensions <= 0
        || spec.dimensions % 2 != 0
        || !spec.base.is_finite()
        || spec.base <= 0.0
    {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU RoPE scalar geometry differs",
        ));
    }
    if offset < 0 || operation.inputs.len() != 1 || operation.outputs.len() != 1 {
        return Ok(None);
    }
    let input = operation.inputs.get(0).expect("RoPE source");
    let output = operation.outputs.get(0).expect("RoPE output");
    if input.shape().len() != 4
        || input.shape().iter().any(|&n| n <= 0)
        || spec.dimensions != input.shape()[3]
        || input.dtype() != WorkspaceDtype::Float32
        || output.dtype() != WorkspaceDtype::Float32
        || !input.representation().is_some_and(|r| {
            r.dtype() == WorkspaceFloatingType::Float32 && r.last_axis_contiguous()
        })
    {
        return Ok(None);
    }
    if input.shape() != output.shape() {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU RoPE output geometry differs",
        ));
    }
    let count = usize::try_from(input.elements()?)?;
    if count > i32::MAX as usize {
        return Ok(None);
    }
    let positions = usize::try_from(input.shape()[2])?;
    let dimensions = usize::try_from(spec.dimensions)?;
    let yarn = matches!(spec.algorithm, RotaryAlgorithm::Yarn { .. });
    let rank = if yarn { 3 } else { 4 };
    let batches = usize::try_from(input.shape()[0])?
        .checked_mul(if yarn {
            usize::try_from(input.shape()[1])?
        } else {
            1
        })
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    let batch_elements = count / batches;
    let seeds = batches
        .checked_mul(3)
        .and_then(|n| n.checked_add(usize::from(yarn)))
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    let source = (|| {
        let mut p = CpuPopulation::default();
        if yarn {
            // The existing module multiplies by its concentration, making a
            // dense result before flattening batch and head axes. Its frequency
            // denominator is a retained original parameter, not a new seed.
            binary(
                &mut p,
                CpuBinaryOperation::Multiply,
                4,
                count,
                (4, count, Dtype::Float32),
                (0, 1, Dtype::Float32),
            )?;
            p.copy(OperationEvent::cpu_reshape_alias_layout(4, 3, false)?, 1)?;
            p.hidden_leaves = 1;
        }
        let mut batch = CpuPopulation::default();
        if batches != 1 {
            batch.copy(OperationEvent::cpu_slice_layout(rank, false, false)?, 1)?;
        }
        fallback(
            &mut batch,
            rank,
            dimensions,
            batch_elements,
            positions,
            yarn,
        )?;
        p.add(CpuPopulation {
            construction_entries: batch.construction_entries.checked_mul(batches)?,
            primitives: batch.primitives.checked_mul(batches)?,
            input_edges: batch.input_edges.checked_mul(batches)?,
            hidden_leaves: 0,
            maximum_operands: batch.maximum_operands,
            maximum_captures: batch.maximum_captures,
            births: batch.births.checked_mul(batches)?,
            extents: batch.extents.checked_mul(batches)?,
            controls: batch.controls.checked_mul(batches)?,
        })?;
        if batches != 1 {
            // The actual Rust wrapper joins all batch results at once. Its
            // native source prices each destination slice/job and input edge.
            for _ in 0..batches {
                cast(&mut p, Dtype::Float32, rank, batch_elements)?;
            }
            p.concatenate(
                OperationEvent::cpu_concatenate_many_layout(
                    Dtype::Float32,
                    rank,
                    batches,
                    count,
                    false,
                )?,
                batches,
            )?;
        }
        if yarn {
            p.copy(OperationEvent::cpu_reshape_alias_layout(3, 4, false)?, 1)?;
        }
        p.controls = p
            .controls
            .checked_add(crate::backend::nn::rope::rotary_control_bytes(
                false, batches,
            )?)?;
        Some(p)
    })();
    let Some(mut population) = source else {
        return Ok(None);
    };
    let full = mechanism
        .allocation
        .fixed_buffer_capacity(facts::mul(input.elements()?, 4)?)?;
    // Every frequency/position/half/full intermediate is bounded by the actual
    // input extent. No temporary or alias-backed output receives early credit.
    let scratch = facts::mul(
        full,
        u64::try_from(
            population
                .births
                .checked_sub(1)
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
        )?,
    )?;
    let scratch_bytes = facts::add(
        scratch,
        facts::mul(
            mechanism.allocation.fixed_buffer_capacity(4)?,
            u64::try_from(seeds)?,
        )?,
    )?;
    let frames = [
        size_of::<CpuPopulation>() * 3,
        size_of::<Option<CpuPopulation>>(),
        size_of::<OperationPlan>(),
        size_of::<Option<OperationPlan>>(),
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<WorkspaceLayoutView<'_>>() * 2,
        size_of::<MlxCpuWorkspaceMechanisms>(),
        size_of::<eredu_nn::RotarySpec>(),
        size_of::<i32>(),
        size_of::<usize>() * 12,
        size_of::<bool>(),
        size_of::<(&mut CpuPopulation, usize, usize, usize, usize, bool)>(),
        size_of::<std::ops::Range<usize>>(),
        size_of::<(
            &mut CpuPopulation,
            CpuBinaryOperation,
            usize,
            usize,
            (usize, usize, Dtype),
            (usize, usize, Dtype),
        )>(),
        size_of::<(&mut CpuPopulation, CpuUnaryOperation, usize, usize)>(),
        size_of::<(&mut CpuPopulation, Dtype, usize, usize)>(),
        size_of::<[(usize, usize, Dtype); 2]>(),
        size_of::<std::array::IntoIter<(usize, usize, Dtype), 2>>(),
        size_of::<[CpuUnaryOperation; 2]>(),
        size_of::<std::array::IntoIter<CpuUnaryOperation, 2>>(),
        size_of::<[CpuBinaryOperation; 2]>(),
        size_of::<std::array::IntoIter<CpuBinaryOperation, 2>>(),
        size_of::<std::ops::Range<i32>>(),
        size_of::<CpuCopyEvalLayout>(),
        size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<safemlx::CpuUnaryEvalLayout>(),
        size_of::<Option<safemlx::CpuUnaryEvalLayout>>(),
        size_of::<safemlx::CpuBinaryEvalLayout>(),
        size_of::<Option<safemlx::CpuBinaryEvalLayout>>(),
        size_of::<Option<()>>(),
        size_of::<std::slice::Iter<i32>>(),
        size_of::<u64>() * 3,
        size_of::<(&safemlx::Array, i32, bool, f32, f32, i32, &safemlx::Stream)>(),
        size_of::<Result<safemlx::Array, safemlx::error::Exception>>(),
    ];
    population.controls = frames.into_iter().try_fold(
        population
            .controls
            .checked_add(size_of_val(&frames))
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
        |n, b| {
            n.checked_add(b)
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)
        },
    )?;
    Ok(Some(OperationPlan {
        dtype: WorkspaceFloatingType::Float32,
        population,
        output_bytes: full,
        scratch_bytes,
        rank: 4,
        parameter_shells: 0,
        alias_input: None,
        seeds,
        validations: 0,
    }))
}

/// One unchanged native fallback invocation. Rank three expands its singleton
/// head axis; rank four keeps it. The supplied denominator takes the actual
/// reciprocal worker (scalar Divide), while Default constructs its frequencies.
fn fallback(
    p: &mut CpuPopulation,
    rank: usize,
    dimensions: usize,
    count: usize,
    positions: usize,
    yarn: bool,
) -> Option<()> {
    let half = dimensions / 2;
    let theta = positions.checked_mul(half)?;
    if rank == 3 {
        p.copy(
            OperationEvent::cpu_expand_dims_alias_layout(3, 4, false)?,
            1,
        )?;
    }
    p.copy(
        OperationEvent::cpu_arange_float_layout(positions, false)?,
        0,
    )?;
    binary(
        p,
        CpuBinaryOperation::Add,
        1,
        positions,
        (1, positions, Dtype::Float32),
        (0, 1, Dtype::Int32),
    )?;
    binary(
        p,
        CpuBinaryOperation::Multiply,
        1,
        positions,
        (1, positions, Dtype::Float32),
        (0, 1, Dtype::Float32),
    )?;
    if yarn {
        binary(
            p,
            CpuBinaryOperation::Divide,
            1,
            half,
            (0, 1, Dtype::Float32),
            (1, half, Dtype::Float32),
        )?;
    } else {
        p.copy(OperationEvent::cpu_arange_float_layout(half, false)?, 0)?;
        binary(
            p,
            CpuBinaryOperation::Multiply,
            1,
            half,
            (1, half, Dtype::Float32),
            (0, 1, Dtype::Float32),
        )?;
        unary(p, CpuUnaryOperation::Exponential, 1, half)?;
    }
    p.copy(
        OperationEvent::cpu_expand_dims_alias_layout(1, 2, false)?,
        1,
    )?;
    binary(
        p,
        CpuBinaryOperation::Multiply,
        2,
        theta,
        (2, positions, Dtype::Float32),
        (1, half, Dtype::Float32),
    )?;
    for kind in [CpuUnaryOperation::Cosine, CpuUnaryOperation::Sine] {
        unary(p, kind, 2, theta)?;
        cast(p, Dtype::Float32, 2, theta)?;
    }
    p.copy(OperationEvent::cpu_slice_layout(4, false, false)?, 1)?;
    p.copy(OperationEvent::cpu_slice_layout(4, false, false)?, 1)?;
    for combine in [CpuBinaryOperation::Subtract, CpuBinaryOperation::Add] {
        for _ in 0..2 {
            binary(
                p,
                CpuBinaryOperation::Multiply,
                4,
                count / 2,
                (4, count / 2, Dtype::Float32),
                (2, theta, Dtype::Float32),
            )?;
        }
        binary(
            p,
            combine,
            4,
            count / 2,
            (4, count / 2, Dtype::Float32),
            (4, count / 2, Dtype::Float32),
        )?;
    }
    cast(p, Dtype::Float32, 4, count / 2)?;
    cast(p, Dtype::Float32, 4, count / 2)?;
    p.concatenate(
        OperationEvent::cpu_concatenate_layout(Dtype::Float32, 4, count / 2, count / 2, false)?,
        2,
    )?;
    p.copy(OperationEvent::cpu_reshape_alias_layout(4, rank, false)?, 1)?;
    p.controls = p
        .controls
        .checked_add(OperationEvent::cpu_rope_fallback_control_bytes(
            rank, dimensions, count,
        )?)?;
    Some(())
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests {
    use super::*;
    use eredu_nn::{NeuralBackend, RotaryOperator, RotaryPosition, RotarySpec, Tensor};
    #[test]
    fn cpu_transposed_head_rotary_preserves_exact_sources_and_dense_output() {
        let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let selected =
            MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected);
        for (batch, positions, heads, width) in [(1, 2, 2, 4), (2, 3, 2, 8)] {
            let context = WorkspaceContext::new(cpu);
            let base = WorkspaceTensor::existing(
                context
                    .layout(&[batch, positions, heads, width], WorkspaceDtype::Float32)
                    .unwrap()
                    .with_representation(Some(WorkspaceRepresentation::new(
                        WorkspaceFloatingType::Float32,
                        true,
                    ))),
                &context,
            )
            .unwrap();
            let input = base.transpose_axes(&[0, 2, 1, 3], &context).unwrap();
            let representation = input.layout().representation().unwrap();
            assert!(!representation.row_contiguous());
            assert!(representation.last_axis_contiguous());
            let spec = RotarySpec {
                arithmetic: RotaryArithmetic::Native,
                algorithm: RotaryAlgorithm::Default,
                dimensions: width,
                traditional: false,
                base: 1_000_000.0,
            };
            let mut rope = WorkspaceBackend::rotary(spec, &context).unwrap();
            context.begin_span();
            let output = rope
                .forward(&input, RotaryPosition::Offset(3), &context)
                .unwrap();
            assert_eq!(
                output.layout().representation(),
                Some(WorkspaceRepresentation::new(
                    WorkspaceFloatingType::Float32,
                    true
                ))
            );
            let report = context.report(&[output]).unwrap();
            let operation = report.operations[0].as_view();
            let plan = cpu.plan(operation).unwrap().unwrap();
            assert_eq!(plan.seeds, 3 * batch as usize);
            assert_eq!(plan.alias_input, None);
            SpeculativeNumericalRecipe::inspect_cpu_equations(&report, ordinary, cpu, &context)
                .unwrap();
            let source = operation.inputs.get(0).unwrap();
            let contiguous = [
                source.with_representation(Some(WorkspaceRepresentation::new(
                    WorkspaceFloatingType::Float32,
                    true,
                ))),
            ];
            let unchanged = cpu
                .plan(WorkspaceOperationView {
                    inputs: WorkspaceLayoutList::Views(&contiguous),
                    ..operation
                })
                .unwrap()
                .unwrap();
            assert_eq!(plan.population.primitives, unchanged.population.primitives);
            assert_eq!(plan.population.births, unchanged.population.births);
            assert_eq!(plan.population.controls, unchanged.population.controls);
            assert_eq!(plan.scratch_bytes, unchanged.scratch_bytes);
            for representation in [
                None,
                Some(WorkspaceRepresentation::new(
                    WorkspaceFloatingType::Float32,
                    false,
                )),
            ] {
                let unknown = [source.with_representation(representation)];
                assert!(cpu
                    .plan(WorkspaceOperationView {
                        inputs: WorkspaceLayoutList::Views(&unknown),
                        ..operation
                    })
                    .unwrap()
                    .is_none());
            }
        }
    }
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod native_tests;
