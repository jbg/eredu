//! Exact eager typed host-array constructor used by native source producers.
use super::*;
use std::mem::{size_of, size_of_val};

pub(super) fn dtype(
    operation: WorkspaceOperationView<'_>,
) -> Option<(safemlx::Dtype, Option<WorkspaceFloatingType>)> {
    use WorkspaceFloatingType as F;
    use safemlx::Dtype as N;
    let WorkspaceOperationKindView::Elementwise(name) = operation.kind else {
        return None;
    };
    let (native, floating, logical) = match name {
        "host_array_f32" => (N::Float32, Some(F::Float32), WorkspaceDtype::Float32),
        "host_array_f16" => (N::Float16, Some(F::Float16), WorkspaceDtype::Float32),
        "host_array_bf16" => (N::Bfloat16, Some(F::Bfloat16), WorkspaceDtype::Float32),
        "host_array_i32" => (N::Int32, None, WorkspaceDtype::Int32),
        "host_array_u8" => (N::Uint8, None, WorkspaceDtype::Uint8),
        "host_array_u32" | "text_prompt_u32" => (N::Uint32, None, WorkspaceDtype::Uint32),
        "host_array_bool" => (N::Bool, None, WorkspaceDtype::Bool),
        _ => return None,
    };
    let [output] = operation.outputs.array()?;
    if name == "text_prompt_u32"
        && !matches!(output.shape(), [batch, positions] if *batch > 0 && *positions > 0)
    {
        return None;
    }
    if !operation.inputs.is_empty()
        || output.dtype() != logical
        || output.shape().len() > 32
        || output.shape().iter().any(|&n| n < 0)
        || output
            .representation()
            .is_some_and(|r| Some(r.dtype()) != floating)
    {
        return None;
    }
    Some((native, floating))
}

/// Tensor's borrowed-slice constructor owns an eager seed followed by Copy.
/// Copy retains the same backing while binding its lazy value to the stream.
pub(super) fn slice_dtype(
    operation: WorkspaceOperationView<'_>,
) -> Option<(safemlx::Dtype, Option<WorkspaceFloatingType>)> {
    let name = match operation.kind {
        WorkspaceOperationKindView::Elementwise("from_f32_slice")
        | WorkspaceOperationKindView::GeneratedF32Initialization => "host_array_f32",
        WorkspaceOperationKindView::Elementwise("from_i32_slice") => "host_array_i32",
        _ => return None,
    };
    dtype(WorkspaceOperationView {
        kind: WorkspaceOperationKindView::Elementwise(name),
        ..operation
    })
}

/// Names the same eager constructor; it creates no scalar-fill or lazy Copy.
pub(crate) fn trace(
    shape: &[i32],
    dtype: safemlx::Dtype,
    context: &WorkspaceContext,
) -> Result<WorkspaceTensor, Error> {
    let (name, logical) = match dtype {
        safemlx::Dtype::Float32 => ("host_array_f32", WorkspaceDtype::Float32),
        safemlx::Dtype::Float16 => ("host_array_f16", WorkspaceDtype::Float32),
        safemlx::Dtype::Bfloat16 => ("host_array_bf16", WorkspaceDtype::Float32),
        safemlx::Dtype::Int32 => ("host_array_i32", WorkspaceDtype::Int32),
        safemlx::Dtype::Uint8 => ("host_array_u8", WorkspaceDtype::Uint8),
        safemlx::Dtype::Uint32 => ("host_array_u32", WorkspaceDtype::Uint32),
        safemlx::Dtype::Bool => ("host_array_bool", WorkspaceDtype::Bool),
        _ => return Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into()),
    };
    let mut outputs = context.metadata_vec(1)?;
    outputs.push(context.layout(shape, logical)?);
    let mut values = context.execute(WorkspaceOperationKind::Elementwise(name), &[], outputs)?;
    Ok(values.pop().expect("one declared eager host array"))
}

/// Borrowed typed slice, checked dimensions, C call and error transports. The
/// resident seed layout separately accounts for native Array/Data/C wrappers.
/// Source payload storage remains with the producer which owns that slice.
pub(in crate::backend::nn) fn control_bytes() -> Option<usize> {
    let frames = [
        size_of::<&[u8]>() * 2,
        size_of::<&[i32]>() * 2,
        size_of::<safemlx::Array>() * 2,
        size_of::<Result<safemlx::Array, safemlx::error::Exception>>() * 2,
        size_of::<(&[u8], &[i32], i32)>(),
        size_of::<Option<usize>>() * 2,
        size_of::<usize>() * 2,
        size_of::<i32>(),
        size_of::<std::slice::Iter<'_, i32>>(),
        size_of::<Result<i32, std::num::TryFromIntError>>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}

pub(super) fn slice_control_bytes() -> Option<usize> {
    let frames = [
        size_of::<(&[u8], &[i32], &safemlx::Stream)>(),
        size_of::<safemlx::Array>(),
        size_of::<Result<safemlx::Array, safemlx::error::Exception>>(),
        size_of::<Result<crate::MlxTensor, Error>>(),
        size_of::<crate::MlxTensor>(),
    ];
    frames.into_iter().try_fold(
        control_bytes()?.checked_add(size_of_val(&frames))?,
        usize::checked_add,
    )
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
        MlxTensor,
        backend::{
            MlxBackend, MlxDeviceIdentity, managed_memory::gpu_stream::PreparedExecutionStreams,
        },
    };
    use eredu_nn::Tensor;
    use safemlx::{Array, Device, DeviceType, Dtype};

    #[test]
    fn tensor_host_slices_preserve_values_and_seed_custody_through_copy_alias() {
        if !crate::tests::support::native_process::enter("cpu-tensor-host-slices") {
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
        for shape in [
            &[][..],
            &[32][..],
            &[2, 3][..],
            &[1, 1, 1, 2, 3][..],
            &[0][..],
        ] {
            let count = shape.iter().map(|&n| n as usize).product::<usize>();
            let integers = (0..count).map(|n| n as i32 * 3 - 11).collect::<Vec<_>>();
            let floats = integers
                .iter()
                .map(|&n| n as f32 * 0.25)
                .collect::<Vec<_>>();
            for integer in [true, false] {
                let context = WorkspaceContext::new(cpu);
                context.begin_span();
                let value = if integer {
                    WorkspaceTensor::from_i32_slice(&integers, shape, &context).unwrap()
                } else {
                    WorkspaceTensor::from_f32_slice(&floats, shape, &context).unwrap()
                };
                let report = context.finish_report(&[value]).unwrap();
                let plan = cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
                assert_eq!(
                    (
                        plan.seeds,
                        plan.population.primitives,
                        plan.population.births
                    ),
                    (1, 1, 0)
                );
                assert_eq!(plan.scratch_bytes, 0);
                let recipe = SpeculativeNumericalRecipe::inspect_cpu_equations(
                    &report, ordinary, cpu, &context,
                )
                .unwrap();
                cpu::test_execution::run(
                    recipe,
                    &backend,
                    &[],
                    |stream| {
                        if integer {
                            MlxTensor::from_i32_slice(&integers, shape, stream).unwrap()
                        } else {
                            MlxTensor::from_f32_slice(&floats, shape, stream).unwrap()
                        }
                    },
                    |actual| {
                        assert_eq!(actual.shape(), shape);
                        let completed = actual.as_array().evaluated().unwrap();
                        if integer {
                            assert_eq!(completed.as_slice::<i32>(), integers);
                        } else {
                            assert_eq!(completed.as_slice::<f32>(), floats);
                        }
                    },
                );
            }
        }
    }

    #[test]
    fn eager_typed_arrays_use_one_seed_without_scalar_fill_or_lazy_copy() {
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
        for dtype in [
            Dtype::Float32,
            Dtype::Float16,
            Dtype::Bfloat16,
            Dtype::Int32,
            Dtype::Uint32,
            Dtype::Bool,
        ] {
            let context = WorkspaceContext::new(cpu);
            let output = trace(&[2, 2], dtype, &context).unwrap();
            let report = context.finish_report(&[output]).unwrap();
            let plan = cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
            assert_eq!(
                (
                    plan.seeds,
                    plan.population.primitives,
                    plan.population.births,
                    plan.scratch_bytes
                ),
                (1, 0, 0, 0)
            );
            let recipe =
                SpeculativeNumericalRecipe::inspect_cpu_equations(&report, ordinary, cpu, &context)
                    .unwrap();
            cpu::test_execution::run(
                recipe,
                &backend,
                &[],
                |_| {
                    MlxTensor::from_array(match dtype {
                        Dtype::Float32 => {
                            Array::try_from_slice(&[2.0f32, -3.0, 5.0, 7.0], &[2, 2]).unwrap()
                        }
                        Dtype::Float16 => Array::try_from_slice(
                            &[2.0f32, -3.0, 5.0, 7.0].map(half::f16::from_f32),
                            &[2, 2],
                        )
                        .unwrap(),
                        Dtype::Bfloat16 => Array::try_from_slice(
                            &[2.0f32, -3.0, 5.0, 7.0].map(half::bf16::from_f32),
                            &[2, 2],
                        )
                        .unwrap(),
                        Dtype::Int32 => Array::try_from_slice(&[2i32, -3, 5, 7], &[2, 2]).unwrap(),
                        Dtype::Uint32 => {
                            Array::try_from_slice(&[2u32, 17, 16777217, u32::MAX], &[2, 2]).unwrap()
                        }
                        Dtype::Bool => {
                            Array::try_from_slice(&[true, false, true, false], &[2, 2]).unwrap()
                        }
                        _ => unreachable!(),
                    })
                },
                |actual| {
                    let values = actual.as_array().evaluated().unwrap();
                    match dtype {
                        Dtype::Float32 => {
                            assert_eq!(values.as_slice::<f32>(), &[2.0, -3.0, 5.0, 7.0])
                        }
                        Dtype::Float16 => assert_eq!(
                            values
                                .as_slice::<half::f16>()
                                .iter()
                                .map(|v| v.to_f32())
                                .collect::<Vec<_>>(),
                            [2.0, -3.0, 5.0, 7.0]
                        ),
                        Dtype::Bfloat16 => assert_eq!(
                            values
                                .as_slice::<half::bf16>()
                                .iter()
                                .map(|v| v.to_f32())
                                .collect::<Vec<_>>(),
                            [2.0, -3.0, 5.0, 7.0]
                        ),
                        Dtype::Int32 => assert_eq!(values.as_slice::<i32>(), &[2, -3, 5, 7]),
                        Dtype::Uint32 => {
                            assert_eq!(values.as_slice::<u32>(), &[2, 17, 16777217, u32::MAX])
                        }
                        Dtype::Bool => {
                            assert_eq!(values.as_slice::<bool>(), &[true, false, true, false])
                        }
                        _ => unreachable!(),
                    }
                },
            );
        }
    }
}
