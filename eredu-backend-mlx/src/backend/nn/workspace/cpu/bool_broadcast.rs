//! The selected nonempty Bool Broadcast worker aliases its genuine source.
use super::*;

pub(super) fn inspect(
    operation: WorkspaceOperationView<'_>,
) -> facts::FactResult<Option<OperationPlan>> {
    if !matches!(
        operation.kind,
        WorkspaceOperationKindView::View("broadcast")
    ) {
        return Ok(None);
    }
    let (Some([input]), Some([output])) = (operation.inputs.array(), operation.outputs.array())
    else {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU predicate broadcast population differs",
        ));
    };
    if input.dtype() != WorkspaceDtype::Bool
        || output.dtype() != WorkspaceDtype::Bool
        || input.representation().is_some()
        || output.representation().is_some()
    {
        return Ok(None);
    }
    let (rank, target_rank) = (input.shape().len(), output.shape().len());
    if rank > 4
        || target_rank > 4
        || input.shape().iter().chain(output.shape()).any(|&n| n <= 0)
        || input.elements()? > i32::MAX as u64
        || output.elements()? > i32::MAX as u64
    {
        return Ok(None);
    }
    if rank > target_rank
        || input
            .shape()
            .iter()
            .rev()
            .zip(output.shape().iter().rev())
            .any(|(&source, &target)| source != 1 && source != target)
    {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU predicate broadcast axes differ",
        ));
    }
    let same = input.shape() == output.shape();
    let mut population = CpuPopulation::default();
    if !same {
        let Some(native) = OperationEvent::cpu_broadcast_alias_layout(rank, target_rank, false)
        else {
            return Ok(None);
        };
        if native.backing_births() != 0 || population.copy(native, 1).is_none() {
            return Ok(None);
        }
    }
    let frames = [
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<WorkspaceLayoutView<'_>>() * 2,
        size_of::<Option<[WorkspaceLayoutView<'_>; 1]>>() * 2,
        size_of::<CpuPopulation>(),
        size_of::<OperationPlan>(),
        size_of::<Option<OperationPlan>>(),
        size_of::<facts::FactResult<Option<OperationPlan>>>(),
        size_of::<CpuCopyEvalLayout>(),
        size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<usize>() * 2,
        size_of::<bool>(),
        size_of::<u64>() * 2,
        size_of::<(&i32, &i32)>(),
        size_of::<std::iter::Chain<std::slice::Iter<'_, i32>, std::slice::Iter<'_, i32>>>(),
        size_of::<
            std::iter::Zip<
                std::iter::Rev<std::slice::Iter<'_, i32>>,
                std::iter::Rev<std::slice::Iter<'_, i32>>,
            >,
        >(),
    ];
    population.controls = frames
        .into_iter()
        .try_fold(
            population
                .controls
                .checked_add(size_of_val(&frames))
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
            usize::checked_add,
        )
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    Ok(Some(OperationPlan {
        dtype: WorkspaceFloatingType::Float32,
        population,
        alias_input: Some(0),
        output_bytes: 0,
        scratch_bytes: 0,
        rank: target_rank,
        parameter_shells: usize::from(same),
        seeds: 0,
        validations: 0,
    }))
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests {
    use super::*;
    use crate::{
        backend::{
            managed_memory::gpu_stream::PreparedExecutionStreams, MlxBackend, MlxDeviceIdentity,
        },
        MlxTensor,
    };
    use eredu_nn::Tensor;
    use safemlx::{Device, DeviceType};

    #[test]
    fn cpu_predicate_broadcast_retains_bool_source_and_nonzero_alias_values() {
        if !crate::tests::support::native_process::enter("cpu-predicate-broadcast") {
            return;
        }
        let ledger = crate::tests::support::test_utils::initialize_original_sources();
        let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let choice =
            MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), choice);
        let streams = PreparedExecutionStreams::for_cpu_factory_with_matmul(&ledger, choice)
            .unwrap()
            .unwrap();
        let backend = MlxBackend::for_prepared_execution_plan(
            streams,
            MlxDeviceIdentity::from_realized_device(&Device::new(DeviceType::Cpu, 0), None)
                .unwrap(),
        );
        let values: [bool; 32] = std::array::from_fn(|index| index % 3 == 1);
        let actual_source = MlxTensor::from_array(safemlx::Array::from_slice(&values, &[32]));
        drop(actual_source.as_array().evaluated().unwrap());
        for shape in [&[32][..], &[1, 1, 32][..], &[2, 1, 32][..]] {
            let context = WorkspaceContext::new(cpu);
            let source = WorkspaceTensor::existing(
                context.layout(&[32], WorkspaceDtype::Bool).unwrap(),
                &context,
            )
            .unwrap();
            context.begin_state_span([&source]).unwrap();
            let output = source.broadcast_to(shape, &context).unwrap();
            assert_eq!(output.layout().dtype(), WorkspaceDtype::Bool);
            assert_eq!(output.layout().representation(), None);
            let report = context.finish_report(&[output]).unwrap();
            assert!(report.unpriced_operations.is_empty());
            assert_eq!(report.tensor_buffers.total_bytes, Some(0));
            let recipe = SpeculativeNumericalRecipe::inspect_cpu_outputs(
                &report, 1, ordinary, cpu, &context,
            )
            .unwrap();
            assert_eq!(recipe.storage.maximum_births(), 0);
            super::super::test_execution::run(
                recipe,
                &backend,
                &[&actual_source],
                |stream| actual_source.broadcast_to(shape, stream).unwrap(),
                |result| {
                    assert_eq!(result.shape(), shape);
                    let actual = result.as_array().evaluated().unwrap();
                    assert!(actual
                        .try_iter::<bool>()
                        .unwrap()
                        .enumerate()
                        .all(|(i, v)| v == values[i % 32]));
                    assert_eq!(
                        result.as_array().try_allocation_info().unwrap(),
                        actual_source.as_array().try_allocation_info().unwrap()
                    );
                },
            );
        }
        let source = [WorkspaceLayoutView::new(&[32], WorkspaceDtype::Bool).unwrap()];
        let malformed = [WorkspaceLayoutView::new(&[1, 1, 31], WorkspaceDtype::Bool).unwrap()];
        assert!(inspect(WorkspaceOperationView {
            kind: WorkspaceOperationKindView::View("broadcast"),
            inputs: WorkspaceLayoutList::Views(&source),
            outputs: WorkspaceLayoutList::Views(&malformed)
        })
        .is_err());
    }
}
