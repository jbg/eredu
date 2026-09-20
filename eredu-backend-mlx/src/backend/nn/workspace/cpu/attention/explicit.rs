//! CPU census of the shared explicit score transform and probability product.
use super::super::program::Program;
use super::*;
use safemlx::CpuUnaryOperation;

#[derive(Clone, Copy)]
struct Matrix {
    shape: [i32; 4],
    strides: [i64; 4],
    dtype: WorkspaceFloatingType,
}
impl Matrix {
    fn input(value: WorkspaceLayoutView<'_>) -> Option<Self> {
        let representation = value.representation()?;
        if value.dtype() != WorkspaceDtype::Float32 {
            return None;
        }
        Some(Self {
            shape: value.shape().try_into().ok()?,
            strides: super::super::views::physical_strides(value, representation)?,
            dtype: representation.dtype(),
        })
    }
    fn dense(shape: [i32; 4], dtype: WorkspaceFloatingType) -> Option<Self> {
        let mut strides = [1i64; 4];
        for axis in (0..3).rev() {
            strides[axis] = strides[axis + 1].checked_mul(i64::from(shape[axis + 1]))?;
        }
        Some(Self {
            shape,
            strides,
            dtype,
        })
    }
    fn elements(self) -> Option<usize> {
        self.shape
            .into_iter()
            .try_fold(1usize, |n, d| n.checked_mul(usize::try_from(d).ok()?))
            .filter(|&n| n > 0 && n <= i32::MAX as usize)
    }
    fn cast(self, p: &mut Program, dtype: WorkspaceFloatingType) -> Option<Self> {
        p.cast(
            native_dtype(self.dtype),
            native_dtype(dtype),
            4,
            self.elements()?,
        )?;
        if self.dtype == dtype {
            Some(self)
        } else {
            Self::dense(self.shape, dtype)
        }
    }
    fn expanded(self, p: &mut Program, heads: i32) -> Option<Self> {
        let [b, kv, tokens, width] = self.shape;
        let repeats = heads.checked_div(kv)?;
        p.reshape(4, 5)?;
        p.broadcast(5, 5)?;
        let shape = [b, kv, repeats, tokens, width];
        let strides = [
            self.strides[0],
            self.strides[1],
            0,
            self.strides[2],
            self.strides[3],
        ];
        let target = [b, heads, tokens, width];
        let source = OperationEvent::cpu_reshape_layout(&shape, &strides, &target, false)?;
        let dense = Self::dense(target, self.dtype)?;
        p.copy(source, 1, dense.elements()?, native_dtype(self.dtype))?;
        if source.backing_births() != 0 {
            return Some(dense);
        }
        Some(Self {
            shape: target,
            strides: [
                self.strides[0],
                if repeats == 1 { self.strides[1] } else { 0 },
                self.strides[2],
                self.strides[3],
            ],
            dtype: self.dtype,
        })
    }
    fn transpose(mut self, p: &mut Program) -> Option<Self> {
        p.copy(
            OperationEvent::cpu_transpose_alias_layout(4, false)?,
            1,
            0,
            native_dtype(self.dtype),
        )?;
        self.shape.swap(2, 3);
        self.strides.swap(2, 3);
        Some(self)
    }
    fn copies(self) -> usize {
        // Unit axes retain unobservable raw native strides; allow the actual
        // check_transpose compaction even when canonical logical strides fit.
        usize::from(
            self.shape[2] == 1
                || self.shape[3] == 1
                || !((self.strides[2] == i64::from(self.shape[3]) && self.strides[3] == 1)
                    || (self.strides[2] == 1 && self.strides[3] == i64::from(self.shape[2]))),
        )
    }
}
fn product(p: &mut Program, left: Matrix, right: Matrix) -> Option<Matrix> {
    let dtype = super::super::super::representation::promote(left.dtype, right.dtype);
    let left = left.cast(p, dtype)?;
    let right = right.cast(p, dtype)?;
    let [b, h, m, k] = left.shape;
    if right.shape[..2] != [b, h] || right.shape[2] != k {
        return None;
    }
    let n = right.shape[3];
    let output = Matrix::dense([b, h, m, n], dtype)?;
    let births = p.native.births;
    matmul(
        &mut p.native,
        p.mechanism,
        native_dtype(dtype),
        4,
        m as usize,
        n as usize,
        k as usize,
        (b as usize).checked_mul(h as usize)?,
        left.copies() + right.copies(),
    )?;
    let maximum = left
        .elements()?
        .max(right.elements()?)
        .max(output.elements()?);
    p.bytes = p.bytes.checked_add(
        p.capacity(maximum, native_dtype(dtype))?
            .checked_mul((p.native.births - births) as u64)?,
    )?;
    Some(output)
}

pub(super) fn inspect(
    operation: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    let WorkspaceOperationKindView::Attention {
        causal: false,
        window: None,
        sinks,
        softcap,
        arithmetic,
    } = operation.kind
    else {
        return Ok(None);
    };
    if !softcap && arithmetic != AttentionArithmetic::InputScores {
        return Ok(None);
    }
    let source = (|| {
        let base = 3 + usize::from(sinks);
        if operation.outputs.len() != 1 || !(base..=base + 1).contains(&operation.inputs.len()) {
            return None;
        }
        let q = Matrix::input(operation.inputs.get(0)?)?;
        let k = Matrix::input(operation.inputs.get(1)?)?;
        let v = Matrix::input(operation.inputs.get(2)?)?;
        let [b, h, queries, width] = q.shape;
        let [_, kv, keys, _] = k.shape;
        let dv = v.shape[3];
        for matrix in [q, k, v] {
            matrix.elements()?;
        }
        if b != k.shape[0]
            || k.shape[..3] != v.shape[..3]
            || width != k.shape[3]
            || h % kv != 0
            || operation.outputs.get(0)?.shape() != [b, h, queries, dv]
            || operation.outputs.get(0)?.dtype() != WorkspaceDtype::Float32
        {
            return None;
        }
        if arithmetic == AttentionArithmetic::InputScores
            && i64::from(queries) * i64::from(keys)
                > i64::from(crate::backend::nn::attention::INPUT_SCORE_ROW_BUDGET)
        {
            return None;
        }
        let mask = if operation.inputs.len() > base {
            Some(operation.inputs.get(3)?)
        } else {
            None
        };
        if let Some(mask) = mask {
            if !matches!(mask.dtype(), WorkspaceDtype::Bool | WorkspaceDtype::Float32)
                || mask.shape().len() > 4
                || mask.shape().iter().any(|&n| n <= 0)
                || mask
                    .shape()
                    .iter()
                    .rev()
                    .zip([b, h, queries, keys].iter().rev())
                    .any(|(&n, &m)| n != 1 && n != m)
                || (mask.dtype() == WorkspaceDtype::Float32 && mask.representation().is_none())
            {
                return None;
            }
        }
        let sink = if sinks {
            Some(operation.inputs.last()?)
        } else {
            None
        };
        if let Some(sink) = sink {
            if sink.shape() != [h]
                || sink.dtype() != WorkspaceDtype::Float32
                || sink.representation().is_none()
            {
                return None;
            }
        }
        let mut p = Program::new(mechanism);
        let score_dtype = if arithmetic == AttentionArithmetic::InputScores {
            q.dtype
        } else {
            WorkspaceFloatingType::Float32
        };
        let key = k.expanded(&mut p, h)?.cast(&mut p, score_dtype)?;
        let values = v.expanded(&mut p, h)?;
        let query = q.cast(&mut p, score_dtype)?;
        let key = key.transpose(&mut p)?;
        let mut scores =
            product(&mut p, query, key)?.cast(&mut p, WorkspaceFloatingType::Float32)?;
        let count = scores.elements()?;
        p.scalar_binary(CpuBinaryOperation::Multiply, Dtype::Float32, 4, count)?;
        scores = scores.cast(&mut p, score_dtype)?;
        if softcap {
            // Real eager F32 scalars promote half scores before tanh. The
            // explicit worker casts back only after both scalar products.
            scores = scores.cast(&mut p, WorkspaceFloatingType::Float32)?;
            p.scalar_binary(CpuBinaryOperation::Multiply, Dtype::Float32, 4, count)?;
            p.cast(Dtype::Float32, Dtype::Float32, 4, count)?;
            p.unary(CpuUnaryOperation::Tanh, Dtype::Float32, 4, count)?;
            p.scalar_binary(CpuBinaryOperation::Multiply, Dtype::Float32, 4, count)?;
            scores = scores.cast(&mut p, score_dtype)?;
        }
        if let Some(mask) = mask {
            let rank = mask.shape().len();
            let elements = usize::try_from(mask.elements().ok()?).ok()?;
            if mask.dtype() == WorkspaceDtype::Bool {
                p.scalar()?;
                p.cast(Dtype::Bool, Dtype::Bool, rank, elements)?;
                scores = scores.cast(&mut p, WorkspaceFloatingType::Float32)?;
                p.cast(Dtype::Float32, Dtype::Float32, 0, 1)?;
                p.broadcast(rank, 4)?;
                p.broadcast(4, 4)?;
                p.broadcast(0, 4)?;
                p.copy(
                    OperationEvent::cpu_typed_select_broadcast_layout(
                        Dtype::Float32,
                        4,
                        count,
                        false,
                    )?,
                    3,
                    count,
                    Dtype::Float32,
                )?;
            } else {
                p.cast(
                    native_dtype(mask.representation()?.dtype()),
                    native_dtype(score_dtype),
                    rank,
                    elements,
                )?;
                p.binary(
                    CpuBinaryOperation::Add,
                    native_dtype(score_dtype),
                    4,
                    count,
                    4,
                    rank,
                    count,
                    elements,
                )?;
                scores = Matrix::dense(scores.shape, score_dtype)?.cast(&mut p, score_dtype)?;
            }
        }
        if let Some(sink) = sink {
            let sink_native = native_dtype(score_dtype);
            p.cast(
                native_dtype(sink.representation()?.dtype()),
                sink_native,
                1,
                h as usize,
            )?;
            p.reshape(1, 4)?;
            p.broadcast(4, 4)?;
            let promoted = super::super::super::representation::promote(scores.dtype, score_dtype);
            let sink_count = (b as usize)
                .checked_mul(h as usize)?
                .checked_mul(queries as usize)?;
            p.cast(native_dtype(scores.dtype), native_dtype(promoted), 4, count)?;
            p.cast(sink_native, native_dtype(promoted), 4, sink_count)?;
            let joined = count.checked_add(sink_count)?;
            let native = OperationEvent::cpu_concatenate_many_layout(
                native_dtype(promoted),
                4,
                2,
                joined,
                false,
            )?;
            p.native.concatenate(native, 2)?;
            p.bytes = p.bytes.checked_add(
                p.capacity(joined, native_dtype(promoted))?
                    .checked_mul(native.backing_births() as u64)?,
            )?;
            p.controls(safemlx::ops::concatenate_axis_control_bytes()?)?;
            scores = Matrix::dense([b, h, queries, keys.checked_add(1)?], promoted)?;
        }
        scores = scores.cast(&mut p, WorkspaceFloatingType::Float32)?;
        let columns = scores.shape[3] as usize;
        let rows = scores.elements()?.checked_div(columns)?;
        p.cast(Dtype::Float32, Dtype::Float32, 4, scores.elements()?)?;
        p.copy(
            OperationEvent::cpu_typed_softmax_layout(
                Dtype::Float32,
                true,
                4,
                columns,
                rows,
                false,
            )?,
            1,
            scores.elements()?,
            Dtype::Float32,
        )?;
        p.slice(4)?;
        // Removing a sink column leaves the softmax row stride intact.
        scores.shape[3] = keys;
        let probabilities = scores.cast(&mut p, q.dtype)?;
        let output = product(&mut p, probabilities, values)?;
        let output_bytes = p.capacity(output.elements()?, native_dtype(output.dtype))?;
        let frames = [
            size_of::<Program>() * 2,
            size_of::<Matrix>() * 12,
            size_of::<Option<Matrix>>(),
            size_of::<OperationPlan>(),
            size_of::<Option<OperationPlan>>(),
            size_of::<WorkspaceOperationView<'_>>(),
            size_of::<WorkspaceLayoutView<'_>>() * 6,
            size_of::<MlxCpuWorkspaceMechanisms>(),
            size_of::<[i32; 5]>(),
            size_of::<[i64; 5]>(),
            size_of::<[i32; 4]>() * 4,
            size_of::<[i64; 4]>() * 3,
            size_of::<usize>() * 20,
            size_of::<Dtype>() * 4,
            size_of::<WorkspaceFloatingType>() * 4,
            size_of::<CpuCopyEvalLayout>(),
            size_of::<Option<CpuCopyEvalLayout>>(),
            size_of::<(&mut Program, Matrix, Matrix)>(),
            size_of::<(&mut Matrix, &mut Program, i32)>(),
            size_of::<safemlx::Array>() * 16,
            size_of::<Option<safemlx::Array>>() * 2,
            size_of::<Result<safemlx::Array, safemlx::error::Exception>>(),
            size_of::<(
                &safemlx::Array,
                &safemlx::Array,
                &safemlx::Array,
                f32,
                Option<&safemlx::Array>,
                Option<&safemlx::Array>,
                Option<f32>,
                AttentionArithmetic,
                &safemlx::Stream,
            )>(),
            super::super::views::physical_stride_control_bytes()?.checked_mul(3)?,
            safemlx::Stream::device_type_control_bytes()?.checked_mul(3)?,
        ];
        p.controls(
            frames
                .into_iter()
                .try_fold(size_of_val(&frames), usize::checked_add)?,
        )?;
        Some(OperationPlan {
            dtype: output.dtype,
            population: p.native,
            alias_input: None,
            output_bytes,
            scratch_bytes: p.bytes.checked_sub(output_bytes)?,
            rank: 5,
            parameter_shells: p.shells,
            seeds: p.seeds,
            validations: 0,
        })
    })();
    Ok(source)
}
