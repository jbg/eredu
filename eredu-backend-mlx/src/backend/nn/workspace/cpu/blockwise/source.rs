//! CPU primitive census of the existing blockwise accumulator equation.
use super::*;
use safemlx::CpuUnaryOperation;

#[derive(Clone, Copy, Default)]
pub(super) struct Source {
    pub native: CpuPopulation,
    pub seeds: usize,
    pub maximum: usize,
}
impl Source {
    fn extent(&mut self, elements: usize) -> Option<()> {
        if elements == 0 || elements > i32::MAX as usize {
            return None;
        }
        self.maximum = self.maximum.max(elements);
        Some(())
    }
    fn seed(&mut self) -> Option<()> {
        self.seeds = self.seeds.checked_add(1)?;
        self.extent(1)
    }
    fn copy(&mut self, source: CpuCopyEvalLayout, inputs: usize, elements: usize) -> Option<()> {
        self.extent(elements)?;
        self.native.copy(source, inputs)
    }
    pub(super) fn cast(&mut self, from: Dtype, to: Dtype, rank: usize, count: usize) -> Option<()> {
        self.copy(
            OperationEvent::cpu_cast_layout(from, to, rank, count, false)?,
            1,
            count,
        )
    }
    fn alias(&mut self, from: usize, to: usize, count: usize) -> Option<()> {
        self.copy(
            OperationEvent::cpu_broadcast_alias_layout(from, to, false)?,
            1,
            count,
        )
    }
    fn unary(
        &mut self,
        kind: CpuUnaryOperation,
        dtype: Dtype,
        rank: usize,
        count: usize,
    ) -> Option<()> {
        self.cast(dtype, dtype, rank, count)?;
        self.extent(count)?;
        self.native
            .unary(OperationEvent::cpu_unary_layout(kind, dtype, rank, false)?)
    }
    fn binary(
        &mut self,
        kind: CpuBinaryOperation,
        dtype: Dtype,
        rank: usize,
        count: usize,
        left: (usize, usize),
        right: (usize, usize),
    ) -> Option<()> {
        for (source_rank, elements) in [left, right] {
            self.cast(dtype, dtype, source_rank, elements)?;
            self.alias(source_rank, rank, count)?;
        }
        self.extent(count)?;
        self.native.binary(OperationEvent::cpu_binary_layout(
            kind, dtype, rank, count, false,
        )?)
    }
    fn f32_binary(
        &mut self,
        kind: CpuBinaryOperation,
        count: usize,
        left: usize,
        right: usize,
    ) -> Option<()> {
        self.binary(kind, Dtype::Float32, 4, count, (4, left), (4, right))
    }
    fn scalar(
        &mut self,
        kind: CpuBinaryOperation,
        dtype: Dtype,
        rank: usize,
        count: usize,
    ) -> Option<()> {
        self.seed()?;
        self.binary(kind, dtype, rank, count, (rank, count), (0, 1))
    }
    fn select(
        &mut self,
        rank: usize,
        count: usize,
        condition: (usize, usize),
        left: (usize, usize),
        right: (usize, usize),
    ) -> Option<()> {
        self.cast(Dtype::Bool, Dtype::Bool, condition.0, condition.1)?;
        for value in [left, right] {
            self.cast(Dtype::Float32, Dtype::Float32, value.0, value.1)?;
        }
        for (source_rank, _) in [condition, left, right] {
            self.alias(source_rank, rank, count)?;
        }
        self.copy(
            OperationEvent::cpu_select_broadcast_layout(rank, count, false)?,
            3,
            count,
        )
    }
    fn matmul(
        &mut self,
        mechanism: MlxCpuWorkspaceMechanisms,
        m: usize,
        n: usize,
        k: usize,
        batches: usize,
        copies: usize,
    ) -> Option<()> {
        let output = batches.checked_mul(m)?.checked_mul(n)?;
        self.extent(output)?;
        self.cast(
            Dtype::Float32,
            Dtype::Float32,
            4,
            batches.checked_mul(m)?.checked_mul(k)?,
        )?;
        self.cast(
            Dtype::Float32,
            Dtype::Float32,
            4,
            batches.checked_mul(k)?.checked_mul(n)?,
        )?;
        super::super::attention::matmul(
            &mut self.native,
            mechanism,
            Dtype::Float32,
            4,
            m,
            n,
            k,
            batches,
            copies,
        )
    }
    fn replicate(
        &mut self,
        b: usize,
        heads: usize,
        repeats: usize,
        length: usize,
        width: usize,
    ) -> Option<()> {
        let input = b
            .checked_mul(heads)?
            .checked_mul(length)?
            .checked_mul(width)?;
        if repeats == 1 {
            return self.extent(input);
        }
        self.copy(
            OperationEvent::cpu_reshape_alias_layout(4, 5, false)?,
            1,
            input,
        )?;
        let count = input.checked_mul(repeats)?;
        self.alias(5, 5, count)?;
        let shape = [
            i32::try_from(b).ok()?,
            i32::try_from(heads).ok()?,
            i32::try_from(repeats).ok()?,
            i32::try_from(length).ok()?,
            i32::try_from(width).ok()?,
        ];
        let stride = [
            i64::try_from(heads.checked_mul(length)?.checked_mul(width)?).ok()?,
            i64::try_from(length.checked_mul(width)?).ok()?,
            0,
            i64::try_from(width).ok()?,
            1,
        ];
        let target = [
            shape[0],
            shape[1].checked_mul(shape[2])?,
            shape[3],
            shape[4],
        ];
        self.copy(
            OperationEvent::cpu_reshape_layout(&shape, &stride, &target, false)?,
            1,
            count,
        )
    }
    fn reduction(&mut self, maximum: bool, width: usize, rows: usize) -> Option<()> {
        if width == 1 {
            return self.cast(Dtype::Float32, Dtype::Float32, 4, rows);
        }
        let source = if maximum {
            OperationEvent::cpu_row_max_layout(4, width, rows, false)?
        } else {
            OperationEvent::cpu_row_sum_layout(4, width, rows, false)?
        };
        self.copy(source, 1, rows)
    }
    fn mask(
        &mut self,
        g: eredu_nn::operation_geometry::AbsoluteAttentionMaskGeometry,
    ) -> Option<()> {
        let q = usize::try_from(g.queries()).ok()?;
        let k = usize::try_from(g.keys()).ok()?;
        let count = q.checked_mul(k)?;
        for length in [q, k] {
            self.copy(
                OperationEvent::cpu_arange_int_layout(safemlx::Dtype::Int32, length, false)?,
                0,
                length,
            )?;
            self.copy(
                OperationEvent::cpu_reshape_alias_layout(1, 2, false)?,
                1,
                length,
            )?;
        }
        self.binary(
            CpuBinaryOperation::GreaterEqual,
            Dtype::Int32,
            2,
            count,
            (2, q),
            (2, k),
        )?;
        if g.window().is_some() {
            self.scalar(CpuBinaryOperation::Subtract, Dtype::Int32, 2, q)?;
            self.binary(
                CpuBinaryOperation::GreaterEqual,
                Dtype::Int32,
                2,
                count,
                (2, k),
                (2, q),
            )?;
            if g.prefix() > 0 {
                self.scalar(CpuBinaryOperation::Less, Dtype::Int32, 2, k)?;
                self.binary(
                    CpuBinaryOperation::LogicalOr,
                    Dtype::Bool,
                    2,
                    count,
                    (2, count),
                    (2, k),
                )?;
            }
            self.binary(
                CpuBinaryOperation::LogicalAnd,
                Dtype::Bool,
                2,
                count,
                (2, count),
                (2, count),
            )?;
        }
        Some(())
    }
}

pub(super) fn accumulate(
    mechanism: MlxCpuWorkspaceMechanisms,
    policy: eredu_nn::workspace::WorkspaceBlockwisePolicy,
    stage: Stage<'_>,
    query_copies: usize,
) -> Option<Source> {
    let Stage::Accumulate {
        q,
        k,
        v,
        mask,
        sink,
        bias,
        previous,
        value_pass,
        absolute,
    } = stage
    else {
        return None;
    };
    let [b, h, queries, d] = q.map(|n| n as usize);
    let [_, kv, keys, _] = k.map(|n| n as usize);
    let dv = v[3] as usize;
    let batches = b.checked_mul(h)?;
    let rows = batches.checked_mul(queries)?;
    let scores = rows.checked_mul(keys)?;
    let output = rows.checked_mul(dv)?;
    let mut p = Source::default();
    p.replicate(b, kv, h / kv, keys, d)?;
    p.replicate(b, kv, h / kv, keys, dv)?;
    p.cast(
        Dtype::Float32,
        Dtype::Float32,
        4,
        batches.checked_mul(keys)?.checked_mul(d)?,
    )?;
    p.cast(
        Dtype::Float32,
        Dtype::Float32,
        4,
        batches.checked_mul(keys)?.checked_mul(dv)?,
    )?;
    p.copy(
        OperationEvent::cpu_transpose_alias_layout(4, false)?,
        1,
        batches.checked_mul(keys)?.checked_mul(d)?,
    )?;
    let input_scores = policy.options.arithmetic == eredu_nn::AttentionArithmetic::InputScores;
    if !input_scores {
        p.scalar(
            CpuBinaryOperation::Multiply,
            Dtype::Float32,
            4,
            rows.checked_mul(d)?,
        )?;
    }
    p.matmul(
        mechanism,
        queries,
        keys,
        d,
        batches,
        if input_scores { query_copies } else { 0 },
    )?;
    if input_scores {
        p.cast(Dtype::Float32, Dtype::Float32, 4, scores)?;
        p.cast(Dtype::Float32, Dtype::Float32, 4, scores)?;
        p.scalar(CpuBinaryOperation::Multiply, Dtype::Float32, 4, scores)?;
        p.cast(Dtype::Float32, Dtype::Float32, 4, scores)?;
    }
    if policy.options.softcap.is_some() {
        p.scalar(CpuBinaryOperation::Multiply, Dtype::Float32, 4, scores)?;
        p.unary(CpuUnaryOperation::Tanh, Dtype::Float32, 4, scores)?;
        p.scalar(CpuBinaryOperation::Multiply, Dtype::Float32, 4, scores)?;
    }
    if let Some(bias) = bias {
        let count = usize::try_from(bias.elements().ok()?).ok()?;
        p.cast(Dtype::Float32, Dtype::Float32, bias.shape().len(), count)?;
        p.binary(
            CpuBinaryOperation::Add,
            Dtype::Float32,
            4,
            scores,
            (4, scores),
            (bias.shape().len(), count),
        )?;
    }
    p.mask(absolute)?;
    let mask_page = queries.checked_mul(keys)?;
    if let Some(mask) = mask {
        p.copy(
            OperationEvent::cpu_slice_layout(4, false, false)?,
            1,
            scores,
        )?;
        if mask.dtype() == WorkspaceDtype::Bool {
            p.binary(
                CpuBinaryOperation::LogicalAnd,
                Dtype::Bool,
                4,
                scores,
                (2, mask_page),
                (4, scores),
            )?;
        } else {
            p.scalar(CpuBinaryOperation::Equal, Dtype::Float32, 4, scores)?;
            p.unary(CpuUnaryOperation::LogicalNot, Dtype::Bool, 4, scores)?;
            p.binary(
                CpuBinaryOperation::LogicalAnd,
                Dtype::Bool,
                4,
                scores,
                (2, mask_page),
                (4, scores),
            )?;
            p.cast(Dtype::Float32, Dtype::Float32, 4, scores)?;
            p.f32_binary(CpuBinaryOperation::Add, scores, scores, scores)?;
        }
    }
    let effective = if mask.is_some() {
        (4, scores)
    } else {
        (2, mask_page)
    };
    p.seed()?;
    p.select(4, scores, effective, (4, scores), (0, 1))?;
    p.cast(Dtype::Float32, Dtype::Float32, 4, scores)?;
    if value_pass {
        p.scalar(CpuBinaryOperation::Greater, Dtype::Float32, 4, rows)?;
        p.seed()?;
        p.select(4, rows, (4, rows), (4, rows), (0, 1))?;
        p.f32_binary(CpuBinaryOperation::Subtract, scores, scores, rows)?;
        p.unary(CpuUnaryOperation::Exponential, Dtype::Float32, 4, scores)?;
        p.cast(Dtype::Bool, Dtype::Float32, effective.0, effective.1)?;
        p.binary(
            CpuBinaryOperation::Multiply,
            Dtype::Float32,
            4,
            scores,
            (4, scores),
            effective,
        )?;
        p.f32_binary(CpuBinaryOperation::Divide, scores, scores, rows)?;
        p.cast(Dtype::Float32, Dtype::Float32, 4, scores)?;
        p.cast(Dtype::Float32, Dtype::Float32, 4, scores)?;
        p.matmul(mechanism, queries, dv, keys, batches, 0)?;
        if previous {
            p.f32_binary(CpuBinaryOperation::Add, output, output, output)?;
        }
    } else {
        p.reduction(true, keys, rows)?;
        p.f32_binary(CpuBinaryOperation::Subtract, scores, scores, rows)?;
        p.unary(CpuUnaryOperation::Exponential, Dtype::Float32, 4, scores)?;
        p.cast(Dtype::Bool, Dtype::Float32, effective.0, effective.1)?;
        p.binary(
            CpuBinaryOperation::Multiply,
            Dtype::Float32,
            4,
            scores,
            (4, scores),
            effective,
        )?;
        p.reduction(false, keys, rows)?;
        if input_scores {
            p.seed()?;
            p.cast(Dtype::Float32, Dtype::Float32, 0, 1)?;
            p.alias(0, 4, output)?;
            p.copy(
                OperationEvent::cpu_scalar_full_layout(Dtype::Float32, 4, output, false)?,
                1,
                output,
            )?;
        } else {
            p.matmul(mechanism, queries, dv, keys, batches, 0)?;
        }
        if previous || sink.is_some() {
            if !previous {
                p.cast(Dtype::Float32, Dtype::Float32, 1, h)?;
                p.copy(OperationEvent::cpu_reshape_alias_layout(1, 4, false)?, 1, h)?;
                p.alias(4, 4, rows)?;
            }
            p.f32_binary(CpuBinaryOperation::Maximum, rows, rows, rows)?;
            for _ in 0..2 {
                p.f32_binary(CpuBinaryOperation::Subtract, rows, rows, rows)?;
                p.unary(CpuUnaryOperation::Exponential, Dtype::Float32, 4, rows)?;
            }
            if previous {
                for extent in [rows, output] {
                    for _ in 0..2 {
                        p.f32_binary(CpuBinaryOperation::Multiply, extent, extent, rows)?;
                    }
                    p.f32_binary(CpuBinaryOperation::Add, extent, extent, extent)?;
                }
            } else {
                p.f32_binary(CpuBinaryOperation::Multiply, rows, rows, rows)?;
                p.f32_binary(CpuBinaryOperation::Add, rows, rows, rows)?;
                p.f32_binary(CpuBinaryOperation::Multiply, output, output, rows)?;
            }
        }
    }
    Some(p)
}

pub(super) fn begin(q: [i32; 4], mask: Option<WorkspaceLayoutView<'_>>) -> Option<Source> {
    let count = q
        .into_iter()
        .try_fold(1usize, |n, d| n.checked_mul(d as usize))?;
    let mut p = Source::default();
    p.cast(Dtype::Float32, Dtype::Float32, 4, count)?;
    if let Some(mask) = mask {
        let elements = (q[0] as usize)
            .checked_mul(q[1] as usize)?
            .checked_mul(q[2] as usize)?
            .checked_mul(*mask.shape().last()? as usize)?;
        p.alias(mask.shape().len(), 4, elements)?;
    }
    Some(p)
}
pub(super) fn finish(shape: [i32; 4], input_scores: bool) -> Option<Source> {
    let count = shape
        .into_iter()
        .try_fold(1usize, |n, d| n.checked_mul(d as usize))?;
    let mut p = Source::default();
    if !input_scores {
        let rows = count / shape[3] as usize;
        p.scalar(CpuBinaryOperation::Greater, Dtype::Float32, 4, rows)?;
        p.seed()?;
        p.select(4, rows, (4, rows), (4, rows), (0, 1))?;
        p.f32_binary(CpuBinaryOperation::Divide, count, count, rows)?;
    }
    p.cast(Dtype::Float32, Dtype::Float32, 4, count)?;
    Some(p)
}
