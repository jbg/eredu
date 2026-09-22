//! Positive F32 layouts used by the existing capture predicate workers.
use super::*;

pub(super) fn f32_source(input: WorkspaceLayoutView<'_>) -> bool {
    let Some(representation) = input.representation() else {
        return false;
    };
    if input.dtype() != WorkspaceDtype::Float32
        || representation.dtype() != WorkspaceFloatingType::Float32
        || input.shape().len() > 4
        || input.shape().iter().any(|&n| n <= 0)
    {
        return false;
    }
    let Some(strides) = views::physical_strides(input, representation) else {
        return false;
    };
    let mut span = 1u64;
    for (&extent, &stride) in input.shape().iter().zip(&strides) {
        let Ok(stride) = u64::try_from(stride) else {
            return false;
        };
        let Some(next) = u64::try_from(extent - 1)
            .ok()
            .and_then(|n| n.checked_mul(stride))
            .and_then(|n| span.checked_add(n))
        else {
            return false;
        };
        span = next;
    }
    // The actual Binary worker's vector/recursive geometry uses int lengths.
    // Native source inspection separately checks the real retained data_size.
    span <= i32::MAX as u64
}
pub(super) fn control_bytes() -> Option<usize> {
    let frames = [
        views::physical_stride_control_bytes()?,
        size_of::<WorkspaceLayoutView<'_>>(),
        size_of::<WorkspaceRepresentation>(),
        size_of::<Option<WorkspaceRepresentation>>(),
        size_of::<[i64; 4]>(),
        size_of::<Option<[i64; 4]>>(),
        size_of::<u64>() * 3,
        size_of::<Option<u64>>() * 3,
        size_of::<i32>(),
        size_of::<i64>(),
        size_of::<bool>(),
        size_of::<std::slice::Iter<'_, i32>>(),
        size_of::<std::iter::Zip<std::slice::Iter<'_, i32>, std::slice::Iter<'_, i64>>>(),
        size_of::<(&i32, &i64)>(),
        size_of::<Result<u64, std::num::TryFromIntError>>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}

#[cfg(all(test, target_vendor = "apple", not(feature = "cuda")))]
mod tests {
    use super::*;
    use crate::backend::array_copy::{HistogramProgram, SummaryProgram};
    #[test]
    fn projected_capture_predicates_use_actual_positive_strides_without_dense_source_inference() {
        let native = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let selected =
            MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu = MlxCpuWorkspaceMechanisms::new(native.allocation(), selected);
        for histogram in [false, true] {
            let context = WorkspaceContext::new(cpu);
            let layout = context
                .layout(&[19], WorkspaceDtype::Float32)
                .unwrap()
                .with_representation(Some(
                    WorkspaceRepresentation::new(WorkspaceFloatingType::Float32, false)
                        .with_element_strides(&[2])
                        .unwrap(),
                ));
            let storage = WorkspaceExistingStorage::try_new(Some(37 * 4), &context).unwrap();
            let input = WorkspaceTensor::existing_with_storage(layout, &storage, &context).unwrap();
            context.begin_state_span([&input]).unwrap();
            let mut retained = Vec::new();
            let (count, frontiers) = if histogram {
                let program = HistogramProgram::new(19, &[-1.0, 0.0, 1.0]).unwrap();
                let population = program.population().unwrap();
                program.trace(&input, &context, &mut retained).unwrap();
                (population.retained_outputs, population.scalar_completions)
            } else {
                let program = SummaryProgram::new(19).unwrap();
                let population = program.population().unwrap();
                program.trace(&input, &context, &mut retained).unwrap();
                (population.retained_outputs, population.scalar_completions)
            };
            assert_eq!(retained.len(), count);
            let report = context.finish_report(std::slice::from_ref(&input)).unwrap();
            assert!(
                report.unpriced_operations.is_empty(),
                "{:#?}",
                report.unpriced_operations
            );
            assert!(report.unpriced_host_operations.is_empty());
            let recipe = SpeculativeNumericalRecipe::inspect_cpu_capture(
                &report, count, frontiers, native, cpu, &context,
            )
            .unwrap();
            assert_eq!(recipe.completion.nested_completions, frontiers);
            assert_eq!(report.state.unwrap().retained_bytes, Some(37 * 4));
        }
        for representation in [
            None,
            Some(WorkspaceRepresentation::new(
                WorkspaceFloatingType::Float32,
                false,
            )),
            Some(WorkspaceRepresentation::new(
                WorkspaceFloatingType::Float16,
                true,
            )),
        ] {
            let input = WorkspaceLayout::new(&[19], WorkspaceDtype::Float32)
                .unwrap()
                .with_representation(representation);
            assert!(!f32_source(input.as_view()));
        }
    }
}
