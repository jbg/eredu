//! The source-visible scalar depthwise 1D convolution worker.
use super::*;

pub(super) fn inspect(
    operation: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    let WorkspaceOperationKindView::Convolution {
        stride,
        padding,
        dilation,
        groups,
        transposed,
    } = operation.kind
    else {
        return Ok(None);
    };
    if transposed.is_some() {
        return Ok(None);
    }
    let Some([input, weight]) = operation.inputs.array() else {
        return Ok(None);
    };
    let Some([output]) = operation.outputs.array() else {
        return Ok(None);
    };
    let Some(shape) = crate::backend::nn::convolution::original::cpu_depthwise_geometry(
        input.shape(),
        weight.shape(),
        stride,
        padding,
        dilation,
        groups,
    ) else {
        return Ok(None);
    };
    if output.shape() != shape
        || [input, weight, output]
            .iter()
            .any(|v| v.dtype() != WorkspaceDtype::Float32)
    {
        return Ok(None);
    }
    let Some(left) = input.representation().filter(|r| r.row_contiguous()) else {
        return Ok(None);
    };
    let Some(right) = weight.representation().filter(|r| r.row_contiguous()) else {
        return Ok(None);
    };
    let dtype = super::super::representation::promote(left.dtype(), right.dtype());
    let native = |kind| match kind {
        WorkspaceFloatingType::Float32 => safemlx::Dtype::Float32,
        WorkspaceFloatingType::Float16 => safemlx::Dtype::Float16,
        WorkspaceFloatingType::Bfloat16 => safemlx::Dtype::Bfloat16,
    };
    let mut population = CpuPopulation::default();
    let mut scratch_bytes = 0;
    for (source, representation) in [(input, left), (weight, right)] {
        // Native astype returns the same array immediately for this dtype.
        if representation.dtype() == dtype {
            continue;
        }
        let Some(cast) = OperationEvent::cpu_cast_layout(
            native(representation.dtype()),
            native(dtype),
            3,
            usize::try_from(source.elements()?)?,
            false,
        ) else {
            return Ok(None);
        };
        if population.copy(cast, 1).is_none() {
            return Ok(None);
        }
        if cast.backing_births() != 0 {
            scratch_bytes = facts::add(
                scratch_bytes,
                mechanism
                    .allocation
                    .fixed_buffer_capacity(facts::mul(source.elements()?, 4)?)?,
            )?;
        }
    }
    let Some(worker) = OperationEvent::cpu_depthwise_convolution_layout(native(dtype), false)
    else {
        return Ok(None);
    };
    if population.copy(worker, 2).is_none() {
        return Ok(None);
    }
    population.maximum_captures = population.maximum_captures.max(1);
    let frames = [
        crate::backend::nn::convolution::original::cpu_control_bytes()
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<MlxCpuWorkspaceMechanisms>(),
        size_of::<[WorkspaceLayoutView<'_>; 3]>(),
        size_of::<[WorkspaceRepresentation; 2]>(),
        size_of::<[Option<WorkspaceRepresentation>; 2]>(),
        size_of::<WorkspaceFloatingType>(),
        size_of::<[&[i32]; 3]>(),
        size_of::<Option<&[i32]>>(),
        size_of::<[i32; 3]>(),
        size_of::<Option<[i32; 3]>>(),
        size_of::<i32>(),
        size_of::<CpuPopulation>(),
        size_of::<OperationPlan>(),
        size_of::<Option<OperationPlan>>(),
        size_of::<CpuCopyEvalLayout>() * 2,
        size_of::<Option<CpuCopyEvalLayout>>() * 2,
        size_of::<std::array::IntoIter<(WorkspaceLayoutView<'_>, WorkspaceRepresentation), 2>>(),
        size_of::<u64>() * 2,
        size_of::<usize>(),
        size_of::<bool>(),
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
        dtype,
        population,
        alias_input: None,
        output_bytes: mechanism
            .allocation
            .fixed_buffer_capacity(facts::mul(output.elements()?, 4)?)?,
        scratch_bytes,
        rank: 3,
        parameter_shells: 0,
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
        backend::managed_memory::gpu_stream::PreparedExecutionStreams,
        backend::{MlxBackend, MlxDeviceIdentity},
        MlxTensor,
    };
    use eredu_nn::Tensor;
    use safemlx::{Array, Device, DeviceType};

    #[test]
    fn scalar_depthwise_convolution_matches_nonzero_reference_under_real_native_custody() {
        let pool = crate::tests::support::test_utils::initialize_original_sources();
        let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let selected =
            MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected);
        let streams = PreparedExecutionStreams::for_cpu_factory_with_matmul(&pool, selected)
            .unwrap()
            .unwrap();
        let backend = MlxBackend::for_prepared_execution_plan(
            streams,
            MlxDeviceIdentity::from_realized_device(&Device::new(DeviceType::Cpu, 0), None)
                .unwrap(),
        );
        for (batch, length, channels, kernel) in [(2, 5, 4, 3), (1, 5, 64, 4)] {
            let data = (0..batch * length * channels)
                .map(|n| (n % 13) as f32 * 0.25 - 1.5)
                .collect::<Vec<_>>();
            let weights = (0..channels * kernel)
                .map(|n| (n % 7) as f32 * 0.125 - 0.25)
                .collect::<Vec<_>>();
            let input = MlxTensor::from_array(Array::from_slice(&data, &[batch, length, channels]));
            let weight = MlxTensor::from_array(Array::from_slice(&weights, &[channels, kernel, 1]));
            input.as_array().evaluated().unwrap();
            weight.as_array().evaluated().unwrap();
            let context = WorkspaceContext::new(cpu);
            let make = |shape: &[i32]| {
                WorkspaceTensor::existing(
                    context
                        .layout(shape, WorkspaceDtype::Float32)
                        .unwrap()
                        .with_representation(Some(WorkspaceRepresentation::new(
                            WorkspaceFloatingType::Float32,
                            true,
                        ))),
                    &context,
                )
                .unwrap()
            };
            let x = make(&[batch, length, channels]);
            let w = make(&[channels, kernel, 1]);
            context.begin_span();
            let output = WorkspaceTensor::conv1d(&x, &w, 1, 0, 1, channels, &context).unwrap();
            let report = context.finish_report(&[output]).unwrap();
            let plan = cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
            assert_eq!(plan.population.births, 1);
            assert_eq!(plan.scratch_bytes, 0);
            let recipe =
                SpeculativeNumericalRecipe::inspect_cpu_equations(&report, ordinary, cpu, &context)
                    .unwrap();
            super::super::test_execution::run(
                recipe,
                &backend,
                &[&input, &weight],
                |stream| MlxTensor::conv1d(&input, &weight, 1, 0, 1, channels, stream).unwrap(),
                |actual| {
                    let values = actual.as_array().evaluated().unwrap();
                    let values = values.as_slice::<f32>();
                    let mut offset = 0;
                    for n in 0..batch {
                        for position in 0..length - kernel + 1 {
                            for channel in 0..channels {
                                let expected = (0..kernel)
                                    .map(|k| {
                                        data[((n * length + position + k) * channels + channel)
                                            as usize]
                                            * weights[(channel * kernel + k) as usize]
                                    })
                                    .sum::<f32>();
                                assert!((values[offset] - expected).abs() < 1e-6);
                                offset += 1;
                            }
                        }
                    }
                    assert_eq!(offset, values.len());
                },
            );
        }
    }
}
