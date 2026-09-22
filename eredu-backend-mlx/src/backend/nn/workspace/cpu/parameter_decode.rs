//! Exact CPU AffineDequantizeFallback and FloatingDequantizeFallback programs.
use super::super::parameter_decode::{self as geometry, Geometry};
use super::*;
use safemlx::Dtype;
mod fp8;
mod gguf;

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests;

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod fp8_tests;

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod gguf_tests;

struct Census {
    native: CpuPopulation,
    bytes: u64,
    seeds: usize,
    mechanism: MlxCpuWorkspaceMechanisms,
}
impl Census {
    fn capacity(&self, n: usize, dtype: Dtype) -> Option<u64> {
        let width = match dtype {
            Dtype::Int8 | Dtype::Uint8 => 1,
            Dtype::Float16 | Dtype::Bfloat16 => 2,
            Dtype::Uint32 | Dtype::Float32 => 4,
            _ => return None,
        };
        self.mechanism
            .allocation
            .fixed_buffer_capacity(u64::try_from(n).ok()?.checked_mul(width)?)
            .ok()
    }
    fn copy(
        &mut self,
        layout: CpuCopyEvalLayout,
        inputs: usize,
        n: usize,
        dtype: Dtype,
    ) -> Option<()> {
        self.bytes = self.bytes.checked_add(
            self.capacity(n, dtype)?
                .checked_mul(layout.backing_births() as u64)?,
        )?;
        self.native.copy(layout, inputs)
    }
    fn seed(&mut self, n: usize, dtype: Dtype) -> Option<()> {
        self.seeds = self.seeds.checked_add(1)?;
        self.bytes = self.bytes.checked_add(self.capacity(n, dtype)?)?;
        Some(())
    }
    fn reshape(&mut self, before: usize, after: usize) -> Option<()> {
        self.copy(
            OperationEvent::cpu_reshape_alias_layout(before, after, false)?,
            1,
            0,
            Dtype::Float32,
        )
    }
    fn reshape_source(
        &mut self,
        before: usize,
        after: usize,
        n: usize,
        dtype: Dtype,
    ) -> Option<()> {
        self.copy(
            OperationEvent::cpu_reshape_copy_layout(before, after, false)?,
            1,
            n,
            dtype,
        )
    }
    fn cast(&mut self, from: Dtype, to: Dtype, rank: usize, n: usize) -> Option<()> {
        self.copy(
            OperationEvent::cpu_cast_layout(from, to, rank, n, false)?,
            1,
            n,
            to,
        )
    }
    fn binary(
        &mut self,
        kind: CpuBinaryOperation,
        dtype: Dtype,
        rank: usize,
        n: usize,
        left: (usize, usize),
        right: (usize, usize),
    ) -> Option<()> {
        self.binary_mixed(
            kind,
            dtype,
            rank,
            n,
            (dtype, left.0, left.1),
            (dtype, right.0, right.1),
        )
    }
    fn binary_mixed(
        &mut self,
        kind: CpuBinaryOperation,
        dtype: Dtype,
        rank: usize,
        n: usize,
        left: (Dtype, usize, usize),
        right: (Dtype, usize, usize),
    ) -> Option<()> {
        self.cast(left.0, dtype, left.1, left.2)?;
        self.cast(right.0, dtype, right.1, right.2)?;
        for before in [left.1, right.1] {
            self.copy(
                OperationEvent::cpu_broadcast_alias_layout(before, rank, false)?,
                1,
                0,
                dtype,
            )?;
        }
        let source = OperationEvent::cpu_binary_layout(kind, dtype, rank, n, false)?;
        self.bytes = self.bytes.checked_add(
            self.capacity(n, dtype)?
                .checked_mul(source.backing_births() as u64)?,
        )?;
        self.native.binary(source)
    }
    fn concatenate(&mut self, dtype: Dtype, rank: usize, inputs: usize, each: usize) -> Option<()> {
        for _ in 0..inputs {
            self.cast(dtype, dtype, rank, each)?;
        }
        let n = inputs.checked_mul(each)?;
        let source = OperationEvent::cpu_concatenate_many_layout(dtype, rank, inputs, n, false)?;
        self.bytes = self.bytes.checked_add(
            self.capacity(n, dtype)?
                .checked_mul(source.backing_births() as u64)?,
        )?;
        self.native.concatenate(source, inputs)
    }
    fn affine(&mut self, g: Geometry) -> Option<()> {
        use CpuBinaryOperation as B;
        let r = g.rank;
        if g.bits.is_power_of_two() {
            let parts = 32 / g.bits;
            for _ in 0..parts {
                for kind in [B::LeftShift, B::RightShift] {
                    self.seed(1, Dtype::Uint32)?;
                    self.binary(kind, Dtype::Uint32, r, g.words, (r, g.words), (0, 1))?;
                }
                self.reshape(r, r + 1)?;
            }
            self.concatenate(Dtype::Uint32, r + 1, parts, g.words)?;
        } else {
            let expanded = g.words.checked_mul(32)?;
            self.reshape(r, r + 1)?;
            self.copy(
                OperationEvent::cpu_arange_int_layout(Dtype::Uint32, 32, false)?,
                0,
                32,
                Dtype::Uint32,
            )?;
            self.binary(
                B::RightShift,
                Dtype::Uint32,
                r + 1,
                expanded,
                (r + 1, g.words),
                (1, 32),
            )?;
            self.seed(1, Dtype::Uint32)?;
            self.binary(
                B::BitwiseAnd,
                Dtype::Uint32,
                r + 1,
                expanded,
                (r + 1, expanded),
                (1, 1),
            )?;
            self.reshape(r + 1, r + 1)?;
            self.copy(
                OperationEvent::cpu_arange_int_layout(Dtype::Uint32, g.bits, false)?,
                0,
                g.bits,
                Dtype::Uint32,
            )?;
            self.binary(
                B::LeftShift,
                Dtype::Uint32,
                r + 1,
                expanded,
                (r + 1, expanded),
                (1, g.bits),
            )?;
            self.copy(
                OperationEvent::cpu_u32_row_sum_layout(r + 1, g.bits, g.values, false)?,
                1,
                g.values,
                Dtype::Uint32,
            )?;
            self.copy(
                OperationEvent::cpu_squeeze_layout(r + 1, false)?,
                1,
                0,
                Dtype::Uint32,
            )?;
        }
        self.reshape(if g.bits.is_power_of_two() { r + 1 } else { r }, r + 1)?;
        self.reshape(r, r + 1)?;
        self.binary_mixed(
            B::Multiply,
            Dtype::Float32,
            r + 1,
            g.values,
            (Dtype::Uint32, r + 1, g.values),
            (Dtype::Float32, r + 1, g.scales),
        )?;
        self.reshape(r, r + 1)?;
        self.binary(
            B::Add,
            Dtype::Float32,
            r + 1,
            g.values,
            (r + 1, g.values),
            (r + 1, g.scales),
        )?;
        self.reshape(r + 1, r)?;
        self.cast(Dtype::Float32, Dtype::Float32, r, g.values)
    }
    fn mx_fp4(&mut self, g: Geometry) -> Option<()> {
        use CpuBinaryOperation as B;
        let bytes = g.words.checked_mul(4)?;
        self.seed(16, Dtype::Bfloat16)?;
        self.reshape_source(g.rank, 2, g.words, Dtype::Uint32)?;
        // Integer source views carry no invented stride evidence. The actual
        // View can use one General-copy temporary before reinterpreting bytes.
        self.copy(
            OperationEvent::cpu_byte_view_layout(
                Dtype::Uint32,
                Dtype::Int8,
                2,
                bytes,
                true,
                false,
            )?,
            1,
            bytes,
            Dtype::Int8,
        )?;
        for kind in [B::BitwiseAnd, B::RightShift] {
            self.seed(1, Dtype::Int8)?;
            self.binary(kind, Dtype::Int8, 2, bytes, (2, bytes), (0, 1))?;
        }
        for _ in 0..2 {
            self.cast(Dtype::Int8, Dtype::Int8, 2, bytes)?;
            self.copy(
                OperationEvent::cpu_gather_layout(
                    Dtype::Bfloat16,
                    Dtype::Int8,
                    1,
                    2,
                    16,
                    bytes,
                    1,
                    false,
                )?,
                2,
                bytes,
                Dtype::Bfloat16,
            )?;
        }
        self.concatenate(Dtype::Bfloat16, 3, 2, bytes)?;
        self.reshape(3, 2)?;
        self.reshape_source(g.rank, 2, g.scales, Dtype::Uint8)?;
        self.cast(Dtype::Uint8, Dtype::Bfloat16, 2, g.scales)?;
        self.seed(1, Dtype::Bfloat16)?;
        self.binary(
            B::Subtract,
            Dtype::Bfloat16,
            2,
            g.scales,
            (2, g.scales),
            (0, 1),
        )?;
        self.seed(1, Dtype::Bfloat16)?;
        self.binary(
            B::Power,
            Dtype::Bfloat16,
            2,
            g.scales,
            (0, 1),
            (2, g.scales),
        )?;
        self.binary(
            B::Multiply,
            Dtype::Bfloat16,
            2,
            g.values,
            (2, g.values),
            (2, g.scales),
        )?;
        self.reshape(2, g.rank)
    }
}

pub(super) fn inspect(
    op: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    if let Some(plan) = gguf::inspect(op, mechanism)? {
        return Ok(Some(plan));
    }
    if let Some(plan) = fp8::inspect(op, mechanism)? {
        return Ok(Some(plan));
    }
    let Some(g) = geometry::inspect(op)? else {
        return Ok(None);
    };
    let mut census = Census {
        native: CpuPopulation::default(),
        bytes: 0,
        seeds: 0,
        mechanism,
    };
    if if g.affine {
        census.affine(g)
    } else {
        census.mx_fp4(g)
    }
    .is_none()
    {
        return Ok(None);
    }
    let dtype = g.dtype();
    let physical = if g.affine {
        Dtype::Float32
    } else {
        Dtype::Bfloat16
    };
    let actual_output = census
        .capacity(g.values, physical)
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    let output_bytes = mechanism
        .allocation
        .fixed_buffer_capacity(facts::mul(g.values as u64, 4)?)?;
    let frames = [
        size_of::<Census>(),
        size_of::<Geometry>(),
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<OperationPlan>(),
        size_of::<Option<OperationPlan>>(),
        size_of::<CpuCopyEvalLayout>(),
        size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<safemlx::CpuBinaryEvalLayout>(),
        size_of::<Option<safemlx::CpuBinaryEvalLayout>>(),
        size_of::<usize>() * 20,
        size_of::<Dtype>() * 4,
        size_of::<CpuBinaryOperation>(),
        size_of::<Option<()>>(),
        size_of::<std::ops::Range<usize>>(),
        size_of::<std::array::IntoIter<CpuBinaryOperation, 2>>(),
        size_of::<(&mut Census, Geometry)>(),
        size_of::<(
            &mut Census,
            CpuBinaryOperation,
            Dtype,
            usize,
            usize,
            (usize, usize),
            (usize, usize),
        )>(),
        size_of::<(
            &mut Census,
            CpuBinaryOperation,
            Dtype,
            usize,
            usize,
            (Dtype, usize, usize),
            (Dtype, usize, usize),
        )>(),
    ];
    census.native.controls = frames
        .into_iter()
        .try_fold(
            census
                .native
                .controls
                .checked_add(size_of_val(&frames))
                .and_then(|n| n.checked_add(geometry::control_bytes()?))
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
            usize::checked_add,
        )
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    Ok(Some(OperationPlan {
        dtype,
        population: census.native,
        alias_input: None,
        output_bytes,
        scratch_bytes: census
            .bytes
            .checked_sub(actual_output)
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
        rank: g.rank + 1,
        parameter_shells: 0,
        seeds: census.seeds,
        validations: 0,
    }))
}
