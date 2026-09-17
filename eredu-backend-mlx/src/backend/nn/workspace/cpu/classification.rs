//! Existing floating classification and Bool-to-U32 operators used by capture.
use super::*;
use safemlx::{CpuUnaryOperation, Dtype};

#[derive(Clone, Copy)]
enum Kind { Finite, Nan, PositiveInfinity, NegativeInfinity, MaskU32 }

pub(super) fn inspect(operation: WorkspaceOperationView<'_>, mechanism: MlxCpuWorkspaceMechanisms)
    -> facts::FactResult<Option<OperationPlan>> {
    let kind = match operation.kind {
        WorkspaceOperationKindView::Elementwise("is_finite") => Kind::Finite,
        WorkspaceOperationKindView::Elementwise("is_nan") => Kind::Nan,
        WorkspaceOperationKindView::Elementwise("is_positive_infinity") => Kind::PositiveInfinity,
        WorkspaceOperationKindView::Elementwise("is_negative_infinity") => Kind::NegativeInfinity,
        WorkspaceOperationKindView::Elementwise("bool_to_u32") => Kind::MaskU32,
        _ => return Ok(None),
    };
    let Some([input]) = operation.inputs.array() else {
        return Err(MlxWorkspaceFactError::descriptor("CPU classification input population differs"));
    };
    let Some([output]) = operation.outputs.array() else {
        return Err(MlxWorkspaceFactError::descriptor("CPU classification output population differs"));
    };
    let cast = matches!(kind, Kind::MaskU32);
    let (from, to) = if cast { (WorkspaceDtype::Bool, WorkspaceDtype::Uint32) }
        else { (WorkspaceDtype::Float32, WorkspaceDtype::Bool) };
    if input.dtype() != from || output.dtype() != to || input.shape() != output.shape() {
        return Err(MlxWorkspaceFactError::descriptor("CPU classification changes declared source geometry"));
    }
    let rank = input.shape().len();
    if rank > 4 || input.shape().iter().any(|&n| n <= 0) {
        return Ok(None);
    }
    if !cast && !strided_capture::f32_source(input) {
        return Ok(None);
    }
    let elements = usize::try_from(input.elements()?)?;
    if elements > i32::MAX as usize { return Ok(None); }
    let mut population = CpuPopulation::default();
    // These are the actual floating branches of mlx::isfinite/isnan/isposinf/
    // isneginf. Same-dtype AsType and same-shape broadcast return aliases.
    // The two infinity scalars broadcast only when the source is not scalar.
    let (equal, not_equal, logical_or, logical_not, seeds) = match kind {
        Kind::Finite => (2usize, 1usize, 2usize, true, 2usize),
        Kind::Nan => (0, 1, 0, false, 0),
        Kind::PositiveInfinity | Kind::NegativeInfinity => (1, 0, 0, false, 1),
        Kind::MaskU32 => (0, 0, 0, false, 0),
    };
    let source = (|| {
        if cast {
            let source = OperationEvent::cpu_cast_layout(Dtype::Bool, Dtype::Uint32, rank, elements, false)?;
            if source.backing_births() != 1 { return None; }
            population.copy(source, 1)?;
        }
        for _ in 0..if rank == 0 { 0 } else { seeds } {
            population.copy(OperationEvent::cpu_broadcast_alias_layout(0, rank, false)?, 1)?;
        }
        for (kind, dtype, count) in [(CpuBinaryOperation::Equal, Dtype::Float32, equal),
            (CpuBinaryOperation::NotEqual, Dtype::Float32, not_equal),
            (CpuBinaryOperation::LogicalOr, Dtype::Bool, logical_or)] {
            for _ in 0..count {
                population.binary(OperationEvent::cpu_binary_layout(kind, dtype, rank, elements, false)?)?;
            }
        }
        if logical_not {
            population.unary(OperationEvent::cpu_unary_layout(CpuUnaryOperation::LogicalNot, Dtype::Bool, rank, false)?)?;
        }
        Some(())
    })();
    if source.is_none() || population.births == 0 { return Ok(None); }
    // All predicate outputs are Bool; the conversion's single output is U32.
    // The only scalar sources in this worker are its actual F32 infinities.
    let output_bytes = mechanism.allocation.fixed_buffer_capacity(output.bytes()?)?;
    let scalar_bytes = mechanism.allocation.fixed_buffer_capacity(4)?;
    let scratch_bytes = facts::add(
        facts::mul(output_bytes, u64::try_from(population.births - 1)?)?,
        facts::mul(scalar_bytes, u64::try_from(seeds)?)?)?;
    let frames = [strided_capture::control_bytes().ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?, size_of::<WorkspaceOperationView<'_>>(), size_of::<WorkspaceLayoutView<'_>>() * 2,
        size_of::<[WorkspaceLayoutView<'_>; 1]>() * 2,
        size_of::<Option<[WorkspaceLayoutView<'_>; 1]>>() * 2,
        size_of::<MlxCpuWorkspaceMechanisms>(), size_of::<Kind>(), size_of::<WorkspaceDtype>() * 2,
        size_of::<WorkspaceRepresentation>(), size_of::<Option<WorkspaceRepresentation>>(),
        size_of::<CpuPopulation>(), size_of::<Option<()>>(), size_of::<CpuCopyEvalLayout>(),
        size_of::<Option<CpuCopyEvalLayout>>(), size_of::<safemlx::CpuBinaryEvalLayout>(),
        size_of::<Option<safemlx::CpuBinaryEvalLayout>>(), size_of::<safemlx::CpuUnaryEvalLayout>(),
        size_of::<Option<safemlx::CpuUnaryEvalLayout>>(), size_of::<OperationPlan>(),
        size_of::<facts::FactResult<Option<OperationPlan>>>(), size_of::<usize>() * 9,
        size_of::<u64>() * 3, size_of::<bool>() * 2, size_of::<std::slice::Iter<'_, i32>>(),
        size_of::<std::ops::Range<usize>>() * 2,
        size_of::<[(CpuBinaryOperation, Dtype, usize); 3]>(),
        size_of::<std::array::IntoIter<(CpuBinaryOperation, Dtype, usize), 3>>(),
        size_of::<(CpuBinaryOperation, Dtype, usize)>(),
        size_of::<(&mut CpuPopulation, bool, usize, usize, usize, usize, usize, bool)>(),
        // Actual ordinary scalar/Array/C-wrapper transports, rather than a
        // dynamically owned predicate program or a new numerical dispatcher.
        safemlx::Array::as_dtype_control_bytes().ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
        size_of::<safemlx::Array>() * 8,
        size_of::<Result<safemlx::Array, safemlx::error::Exception>>(),
        size_of::<(&safemlx::Array, &safemlx::Stream)>()];
    population.controls = frames.into_iter().try_fold(
        population.controls.checked_add(size_of_val(&frames)).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
        |n, b| n.checked_add(b).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW))?;
    Ok(Some(OperationPlan { dtype: WorkspaceFloatingType::Float32, population, alias_input: None,
        output_bytes, scratch_bytes, rank, parameter_shells: 0, seeds, validations: 0 }))
}

#[cfg(all(test, target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
mod tests {
    use super::*;
    use eredu_nn::Tensor;
    #[test]
    fn cpu_capture_classification_keeps_actual_mask_and_conversion_sources() {
        let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let selected = MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected);
        for shape in [&[][..], &[19][..]] {
            for name in ["is_finite", "is_nan", "is_positive_infinity", "is_negative_infinity"] {
                let context = WorkspaceContext::new(cpu);
                let input = WorkspaceTensor::existing(
                    context.layout(shape, WorkspaceDtype::Float32).unwrap()
                        .with_representation(Some(WorkspaceRepresentation::new(
                            WorkspaceFloatingType::Float32, true))), &context).unwrap();
                context.begin_state_span([&input]).unwrap();
                let mask = context.execute(WorkspaceOperationKind::Elementwise(name), &[&input],
                    vec![context.layout(shape, WorkspaceDtype::Bool).unwrap()]).unwrap().remove(0);
                let count = context.execute(WorkspaceOperationKind::Elementwise("bool_to_u32"), &[&mask],
                    vec![context.layout(shape, WorkspaceDtype::Uint32).unwrap()]).unwrap().remove(0);
                let report = context.finish_report(std::slice::from_ref(&count)).unwrap();
                assert!(report.unpriced_operations.is_empty() && report.unpriced_host_operations.is_empty());
                let recipe = SpeculativeNumericalRecipe::inspect_cpu_equations(&report, ordinary, cpu, &context).unwrap();
                assert_eq!(recipe.completion.nested_completions, 0);
                assert_eq!(recipe.completion.traversal.limits().roots, 1);
                assert_eq!(recipe.kernels, 0);
                assert!(recipe.graph_capacity > 0 && recipe.record_capacity > 0);
                assert!(report.tensor_buffers.retained_bytes.is_some_and(|n| n >= count.layout().bytes().unwrap()));
                assert_eq!(mask.layout().representation(), None);
                assert_eq!(count.layout().representation(), None);
            }
        }
        let context = WorkspaceContext::new(cpu);
        let input = WorkspaceLayout::new(&[19], WorkspaceDtype::Float32).unwrap();
        let output = WorkspaceLayout::new(&[19], WorkspaceDtype::Bool).unwrap();
        let inputs = [input.as_view()];
        let outputs = [output.as_view()];
        let operation = WorkspaceOperationView { kind: WorkspaceOperationKindView::Elementwise("is_finite"),
            inputs: WorkspaceLayoutList::Views(&inputs),
            outputs: WorkspaceLayoutList::Views(&outputs) };
        assert!(cpu.plan(operation).unwrap().is_none());
        let source = WorkspaceTensor::existing(
            context.layout(&[19], WorkspaceDtype::Float32).unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(
                    WorkspaceFloatingType::Float32, true))), &context).unwrap();
        let wrong = context.layout(&[18], WorkspaceDtype::Bool).unwrap();
        assert!(context.execute(WorkspaceOperationKind::Elementwise("is_nan"), &[&source], vec![wrong]).is_err());
    }
}
