//! One actual dense frontend and the selected existing CPU SIMD worker.
use super::*;
use eredu_checkpoint::LinearFormat;

pub(super) fn inspect(
    operation: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    let (linear, constructed) = match operation.kind {
        WorkspaceOperationKindView::Matmul => (false, false),
        WorkspaceOperationKindView::DenseLinear => (true, false),
        WorkspaceOperationKindView::Projection(format)
            if format.encoding() == LinearFormat::Dense =>
        {
            format.validate_fixed()?;
            (true, true)
        }
        _ => return Ok(None),
    };
    if operation.outputs.len() != 1
        || !(operation.inputs.len() == 2 || linear && operation.inputs.len() == 3)
    {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU dense source input/output population differs",
        ));
    }
    let addmm = linear && !constructed && operation.inputs.len() == 3;
    // Tensor::linear with bias is AddMM, a separate native Eval source. The
    // constructed PhysicalLinear instead performs Matmul then Broadcast/Add.
    let left = operation.inputs.get(0).expect("checked dense inputs");
    let weight = operation.inputs.get(1).expect("checked dense inputs");
    let output = operation.outputs.get(0).expect("checked dense output");
    let rank = left.shape().len();
    if !(2..=4).contains(&rank) || weight.shape().len() != 2 || output.shape().len() != rank {
        return Ok(None);
    }
    let Some(left_dtype) = left.representation().map(|r| r.dtype()) else {
        return Ok(None);
    };
    let Some(weight_dtype) = weight.representation().map(|r| r.dtype()) else {
        return Ok(None);
    };
    let matrix_dtype = super::super::representation::promote(left_dtype, weight_dtype);
    // Bias promotion occurs after the matrix product. It cannot qualify a
    // unqualified same-F16 Matmul implementation merely because the final output is F32.
    let bias_dtype = operation
        .inputs
        .get(2)
        .and_then(|bias| bias.representation().map(|r| r.dtype()));
    let dtype = bias_dtype.map_or(matrix_dtype, |bias| {
        super::super::representation::promote(matrix_dtype, bias)
    });
    for input in operation.inputs.iter() {
        if input.dtype() != WorkspaceDtype::Float32
            || !input.representation().is_some_and(|r| r.row_contiguous())
        {
            return Ok(None);
        }
        if input.shape().iter().any(|&d| d <= 0) {
            return Ok(None);
        }
    }
    if output.dtype() != WorkspaceDtype::Float32 || output.shape().iter().any(|&d| d <= 0) {
        return Ok(None);
    }
    let k = left.shape()[rank - 1];
    let (right_k, n) = if linear {
        (weight.shape()[1], weight.shape()[0])
    } else {
        (weight.shape()[0], weight.shape()[1])
    };
    if k != right_k
        || output.shape()[..rank - 1] != left.shape()[..rank - 1]
        || output.shape()[rank - 1] != n
        || operation.inputs.get(2).is_some_and(|b| b.shape() != [n])
    {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU dense source geometry differs",
        ));
    }
    let Some(m) = left.shape()[..rank - 1]
        .iter()
        .try_fold(1u32, |v, &d| v.checked_mul(d as u32))
    else {
        return Ok(None);
    };
    let selected = mechanism.matmul.selected();
    let geometry = if matrix_dtype == WorkspaceFloatingType::Float16 {
        selected.float16_geometry(2, m, n as u32, k as u32, 1)
    } else {
        selected.geometry(2, m, n as u32, k as u32, 1)
    };
    let Ok(geometry) = geometry else {
        return Ok(None);
    };
    let native_dtype = |dtype| match dtype {
        WorkspaceFloatingType::Float32 => safemlx::Dtype::Float32,
        WorkspaceFloatingType::Bfloat16 => safemlx::Dtype::Bfloat16,
        WorkspaceFloatingType::Float16 => safemlx::Dtype::Float16,
    };
    // AddMM promotes all three operands before multiplication. The selected
    // source currently qualifies the actual all-F32 frontend and tiled worker.
    if addmm
        && [left_dtype, weight_dtype, bias_dtype.unwrap()]
            .into_iter()
            .any(|dtype| dtype != WorkspaceFloatingType::Float32)
    {
        return Ok(None);
    }
    let native = if addmm {
        OperationEvent::cpu_tiled_addmm_layout(2, m as usize, n as usize, k as usize, 1, false)
    } else {
        match matrix_dtype {
            WorkspaceFloatingType::Float32 => mechanism.matmul.eval_layout(geometry, false),
            WorkspaceFloatingType::Bfloat16 => OperationEvent::cpu_bf16_matmul_layout(
                2, m as usize, n as usize, k as usize, 1, false,
            ),
            WorkspaceFloatingType::Float16 => {
                mechanism.matmul.float16_eval_layout(geometry, 0, false)
            }
        }
    };
    let Some(native) = native else {
        return Ok(None);
    };
    if native.backing_births() != 1 {
        return Ok(None);
    }
    let mut population = CpuPopulation::default();
    let mut converted_bytes = 0u64;
    let mut cast =
        |population: &mut CpuPopulation, from, to, rank, elements: usize| -> Option<()> {
            if from == to {
                return Some(());
            }
            let source = OperationEvent::cpu_cast_layout(
                native_dtype(from),
                native_dtype(to),
                rank,
                elements,
                false,
            )?;
            // Every reached mixed floating promotion is F32. Charge its actual
            // input/weight/bias extent, never the possibly smaller output extent.
            if to != WorkspaceFloatingType::Float32 || source.backing_births() != 1 {
                return None;
            }
            let bytes = mechanism
                .allocation
                .fixed_buffer_capacity(u64::try_from(elements).ok()?.checked_mul(4)?)
                .ok()?;
            converted_bytes = converted_bytes.checked_add(bytes)?;
            population.copy(source, 1)
        };
    let source = (|| {
        if linear {
            population.copy(OperationEvent::cpu_transpose_alias_layout(2, false)?, 1)?;
        }
        // mlx::matmul performs promotion after the projection's transpose
        // and before flattening. The cast of a transposed weight has its own
        // complete backing and retains its original parameter source.
        cast(
            &mut population,
            left_dtype,
            matrix_dtype,
            rank,
            usize::try_from(left.elements().ok()?).ok()?,
        )?;
        cast(
            &mut population,
            weight_dtype,
            matrix_dtype,
            2,
            usize::try_from(weight.elements().ok()?).ok()?,
        )?;
        if rank > 2 {
            population.copy(OperationEvent::cpu_reshape_alias_layout(rank, 2, false)?, 1)?;
        }
        if addmm {
            population.copy(OperationEvent::cpu_broadcast_alias_layout(1, 2, false)?, 1)?;
        }
        population.copy(native, if addmm { 3 } else { 2 })?;
        if rank > 2 {
            population.copy(OperationEvent::cpu_reshape_alias_layout(2, rank, false)?, 1)?;
        }
        if operation.inputs.len() == 3 && !addmm {
            cast(
                &mut population,
                matrix_dtype,
                dtype,
                rank,
                usize::try_from(output.elements().ok()?).ok()?,
            )?;
            cast(
                &mut population,
                bias_dtype?,
                dtype,
                1,
                usize::try_from(n).ok()?,
            )?;
            population.copy(
                OperationEvent::cpu_broadcast_alias_layout(1, rank, false)?,
                1,
            )?;
            population.binary(OperationEvent::cpu_binary_layout(
                CpuBinaryOperation::Add,
                native_dtype(dtype),
                rank,
                usize::try_from(output.elements().ok()?).ok()?,
                false,
            )?)?;
        }
        Some(())
    })();
    let cast_controls = size_of_val(&cast);
    drop(cast);
    if source.is_none() {
        return Ok(None);
    }
    let output_bytes = mechanism
        .allocation
        .fixed_buffer_capacity(facts::mul(output.elements()?, 4)?)?;
    let scratch_bytes = facts::add(
        converted_bytes,
        if operation.inputs.len() == 3 && !addmm {
            output_bytes
        } else {
            0
        },
    )?;
    if constructed {
        let Some(probe) = super::super::super::matrix::row_projection_probe_control_bytes() else {
            return Ok(None);
        };
        population.controls = population
            .controls
            .checked_add(probe)
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    }
    let frames = [
        size_of::<bool>(),
        size_of::<OperationPlan>(),
        size_of::<Option<OperationPlan>>(),
        size_of::<CpuPopulation>() * 2,
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<WorkspaceLayoutView<'_>>() * 3,
        size_of::<MlxCpuWorkspaceMechanisms>(),
        size_of::<WorkspaceFloatingType>() * 5,
        size_of::<Option<WorkspaceFloatingType>>(),
        size_of::<safemlx::Dtype>() * 2,
        size_of::<(
            &mut CpuPopulation,
            WorkspaceFloatingType,
            WorkspaceFloatingType,
            usize,
            usize,
        )>(),
        size_of::<u64>() * 2,
        size_of::<Option<u64>>(),
        size_of::<Option<()>>(),
        size_of::<Result<u64, MlxWorkspaceFactError>>(),
        // Actual closure capture, not an assumed callable layout.
        cast_controls,
        size_of::<usize>(),
        size_of::<[u32; 5]>(),
        size_of::<(i32, i32)>(),
        size_of::<eredu_nn::SelectedCpuMatmul>(),
        size_of::<eredu_nn::CpuMatmulGeometry>(),
        size_of::<Result<eredu_nn::CpuMatmulGeometry, eredu_nn::CpuMatmulError>>(),
        size_of::<CpuCopyEvalLayout>(),
        size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<safemlx::CpuBinaryEvalLayout>(),
        size_of::<Option<safemlx::CpuBinaryEvalLayout>>(),
        size_of::<Option<()>>(),
        size_of::<usize>() * 6,
        size_of::<u64>() * 2,
        size_of::<std::slice::Iter<i32>>(),
        size_of::<Option<u32>>(),
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
        alias_input: None,
        seeds: 0,
        validations: 0,
        dtype,
        population,
        output_bytes,
        scratch_bytes,
        rank,
        parameter_shells: usize::from(constructed),
    }))
}

#[cfg(all(test, target_vendor = "apple", not(feature = "cuda")))]
mod tests;
