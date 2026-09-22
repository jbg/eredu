//! The configured grouped branch of the shared MlxRmsNorm worker.
use super::*;
use eredu_nn::NormalizationConstructionSpec;

pub(super) fn inspect(
    operation: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
    spec: &NormalizationConstructionSpec,
    groups: i32,
) -> facts::FactResult<Option<OperationPlan>> {
    spec.validate_fixed()?;
    let (learned, offset) = match spec.scale {
        NormalizationScale::Unit => (false, false),
        NormalizationScale::Learned(_) => (true, false),
        NormalizationScale::LearnedOffset { .. } => (true, true),
    };
    if operation.inputs.len() != 1 + usize::from(learned) || operation.outputs.len() != 1 {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU grouped RMS operand population differs",
        ));
    }
    let input = operation.inputs.get(0).expect("grouped RMS input");
    let output = operation.outputs.get(0).expect("grouped RMS output");
    let rank = input.shape().len();
    // The existing source-visible row reduction/control query supports rank 1..4.
    // Grouping adds one actual axis before that same reduction.
    if !(1..=3).contains(&rank) || input.shape().iter().any(|&n| n <= 0) {
        return Ok(None);
    }
    if input.shape() != output.shape() || input.shape().last() != Some(&spec.dimensions) {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU grouped RMS construction geometry differs",
        ));
    }
    if input.dtype() != WorkspaceDtype::Float32 || output.dtype() != WorkspaceDtype::Float32 {
        return Ok(None);
    }
    let Some(representation) = input.representation().filter(|r| r.last_axis_contiguous()) else {
        return Ok(None);
    };
    let native_dtype = |dtype| match dtype {
        WorkspaceFloatingType::Float32 => Dtype::Float32,
        WorkspaceFloatingType::Float16 => Dtype::Float16,
        WorkspaceFloatingType::Bfloat16 => Dtype::Bfloat16,
    };
    let dtype = representation.dtype();
    let input_native = native_dtype(dtype);
    let gain_native = if learned {
        let gain = operation.inputs.get(1).expect("grouped RMS gain");
        if gain.shape() != [spec.dimensions] || gain.dtype() != WorkspaceDtype::Float32 {
            return Err(MlxWorkspaceFactError::descriptor(
                "CPU grouped RMS gain geometry differs",
            ));
        }
        let Some(gain) = gain.representation().filter(|r| r.row_contiguous()) else {
            return Ok(None);
        };
        native_dtype(gain.dtype())
    } else {
        Dtype::Float32
    };
    let width = usize::try_from(spec.dimensions / groups)?;
    let count = usize::try_from(input.elements()?)?;
    if count > i32::MAX as usize {
        return Ok(None);
    }
    let rows = count / width;
    let dimensions = usize::try_from(spec.dimensions)?;
    let mut shape = [0; 4];
    shape[..rank - 1].copy_from_slice(&input.shape()[..rank - 1]);
    shape[rank - 1] = groups;
    shape[rank] = spec.dimensions / groups;
    let grouped_shape = &shape[..rank + 1];
    let source = (|| {
        let mut p = Population::default();
        p.cast(input_native, Dtype::Float32, rank, count, Buffer::Full)?;
        if input_native == Dtype::Float32 {
            let strides = super::super::views::physical_strides(input, representation)?;
            p.copy(
                OperationEvent::cpu_reshape_layout(
                    input.shape(),
                    &strides[..rank],
                    grouped_shape,
                    false,
                )?,
                Some(Buffer::Full),
            )?;
        } else {
            // A real widening cast produces dense storage before the reshape.
            p.copy(
                OperationEvent::cpu_reshape_alias_layout(rank, rank + 1, false)?,
                None,
            )?;
        }
        p.unary(CpuUnaryOperation::Square, rank + 1, Buffer::Full)?;
        if width == 1 {
            // sum_owned returns its same-shape cast for an empty reduction axis set.
            p.cast(
                Dtype::Float32,
                Dtype::Float32,
                rank + 1,
                count,
                Buffer::Full,
            )?;
        } else {
            p.copy(
                OperationEvent::cpu_row_sum_layout(rank + 1, width, rows, false)?,
                Some(Buffer::Row),
            )?;
        }
        p.binary(
            CpuBinaryOperation::Divide,
            Dtype::Float32,
            rank + 1,
            rows,
            Buffer::Row,
            Buffer::Scalar,
            0,
            1,
            Buffer::Row,
        )?;
        p.binary(
            CpuBinaryOperation::Add,
            Dtype::Float32,
            rank + 1,
            rows,
            Buffer::Row,
            Buffer::Scalar,
            0,
            1,
            Buffer::Row,
        )?;
        p.unary(CpuUnaryOperation::Rsqrt, rank + 1, Buffer::Row)?;
        p.binary(
            CpuBinaryOperation::Multiply,
            Dtype::Float32,
            rank + 1,
            count,
            Buffer::Full,
            Buffer::Row,
            rank + 1,
            rows,
            Buffer::Full,
        )?;
        p.cast(
            Dtype::Float32,
            Dtype::Float32,
            rank + 1,
            count,
            Buffer::Full,
        )?;
        p.copy(
            OperationEvent::cpu_reshape_alias_layout(rank + 1, rank, false)?,
            None,
        )?;
        if learned {
            p.cast(gain_native, Dtype::Float32, 1, dimensions, Buffer::Vector)?;
            if offset {
                p.binary(
                    CpuBinaryOperation::Add,
                    Dtype::Float32,
                    1,
                    dimensions,
                    Buffer::Vector,
                    Buffer::Scalar,
                    0,
                    1,
                    Buffer::Vector,
                )?;
            }
            p.binary(
                CpuBinaryOperation::Multiply,
                Dtype::Float32,
                rank,
                count,
                Buffer::Full,
                Buffer::Vector,
                1,
                dimensions,
                Buffer::Full,
            )?;
        }
        p.cast(Dtype::Float32, input_native, rank, count, Buffer::Full)?;
        p.native.controls = p
            .native
            .controls
            .checked_add(OperationEvent::cpu_rms_fallback_control_bytes(
                Dtype::Float32,
                rank + 1,
                width,
                rows,
            )?)?
            .checked_add(safemlx::Stream::device_type_control_bytes()?)?
            .checked_add(super::super::views::physical_stride_control_bytes()?)?;
        Some(p)
    })();
    let Some(mut source) = source else {
        return Ok(None);
    };
    if source
        .births
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        != Some(source.native.births)
    {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU grouped RMS source population differs",
        ));
    }
    let seeds = 2 + usize::from(offset); // actual mean width, epsilon and optional scale offset.
    let full = mechanism
        .allocation
        .fixed_buffer_capacity(facts::mul(input.elements()?, 4)?)?;
    let row = mechanism
        .allocation
        .fixed_buffer_capacity(facts::mul(u64::try_from(rows)?, 4)?)?;
    let vector = mechanism
        .allocation
        .fixed_buffer_capacity(facts::mul(u64::try_from(dimensions)?, 4)?)?;
    let scalar = mechanism.allocation.fixed_buffer_capacity(4)?;
    let total = source
        .births
        .into_iter()
        .zip([full, row, vector, scalar])
        .try_fold(
            facts::mul(scalar, u64::try_from(seeds)?)?,
            |n, (births, bytes)| facts::add(n, facts::mul(bytes, u64::try_from(births)?)?),
        )?;
    let scratch_bytes = total
        .checked_sub(full)
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    let frames = [
        size_of::<Population>() * 3,
        size_of::<Option<Population>>(),
        size_of::<OperationPlan>(),
        size_of::<Option<OperationPlan>>(),
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<WorkspaceLayoutView<'_>>() * 3,
        size_of::<MlxCpuWorkspaceMechanisms>(),
        size_of::<WorkspaceFloatingType>(),
        size_of::<Dtype>() * 3,
        size_of::<WorkspaceRepresentation>(),
        size_of::<Option<WorkspaceRepresentation>>(),
        size_of::<(
            CpuBinaryOperation,
            Dtype,
            usize,
            usize,
            Buffer,
            Buffer,
            usize,
            usize,
            Buffer,
        )>(),
        size_of::<(Dtype, Dtype, usize, usize, Buffer)>(),
        size_of::<(CpuUnaryOperation, usize, Buffer)>(),
        size_of::<(&mut Population, CpuCopyEvalLayout, Option<Buffer>)>(),
        size_of::<(&mut Population, Buffer, usize)>(),
        size_of::<(&mut Population, usize, usize)>(),
        size_of::<Option<Buffer>>(),
        size_of::<Option<()>>(),
        size_of::<CpuCopyEvalLayout>(),
        size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<safemlx::CpuBinaryEvalLayout>(),
        size_of::<Option<safemlx::CpuBinaryEvalLayout>>(),
        size_of::<safemlx::CpuUnaryEvalLayout>(),
        size_of::<Option<safemlx::CpuUnaryEvalLayout>>(),
        size_of::<(
            &mut crate::backend::nn::shared::MlxRmsNorm,
            &crate::MlxTensor,
            &safemlx::Stream,
        )>(),
        size_of::<(&safemlx::Array, f32, &safemlx::Stream)>(),
        size_of::<safemlx::Array>() * 7,
        size_of::<Result<safemlx::Array, eredu_nn::Error>>(),
        size_of::<Result<crate::MlxTensor, eredu_nn::Error>>(),
        size_of::<Result<safemlx::Array, safemlx::error::Exception>>(),
        size_of::<Result<Option<safemlx::Array>, safemlx::error::Exception>>(),
        // MlxRmsNorm creates this one shape Vec with exactly rank+1 capacity.
        size_of::<Vec<i32>>(),
        (rank + 1) * size_of::<i32>(),
        size_of::<[i32; 2]>(),
        size_of::<[i32; 4]>(),
        size_of::<[i64; 4]>(),
        size_of::<usize>() * 10,
        size_of::<u64>() * 5,
        size_of::<[usize; 4]>(),
        size_of::<[u64; 4]>(),
        size_of::<std::iter::Zip<std::array::IntoIter<usize, 4>, std::array::IntoIter<u64, 4>>>(),
        size_of::<std::slice::Iter<i32>>(),
        size_of::<bool>() * 2,
        size_of::<(
            WorkspaceOperationView<'_>,
            MlxCpuWorkspaceMechanisms,
            &NormalizationConstructionSpec,
            i32,
        )>(),
    ];
    source.native.controls = frames.into_iter().try_fold(
        source
            .native
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
        dtype,
        population: source.native,
        output_bytes: full,
        scratch_bytes,
        rank: rank + 1,
        parameter_shells: usize::from(learned),
        seeds,
        validations: 0,
    }))
}
