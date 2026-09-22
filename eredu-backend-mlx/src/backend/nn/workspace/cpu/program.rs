//! Shared additive census of actual CPU constructor and Eval sources.
use super::*;
use safemlx::{CpuUnaryOperation, Dtype};

pub(super) struct Program {
    pub(super) native: CpuPopulation,
    pub(super) bytes: u64,
    pub(super) seeds: usize,
    pub(super) shells: usize,
    pub(super) validations: usize,
    pub(super) mechanism: MlxCpuWorkspaceMechanisms,
}
impl Program {
    pub(super) fn new(mechanism: MlxCpuWorkspaceMechanisms) -> Self {
        Self {
            native: CpuPopulation::default(),
            bytes: 0,
            seeds: 0,
            shells: 0,
            validations: 0,
            mechanism,
        }
    }
    pub(super) fn controls(&mut self, bytes: usize) -> Option<()> {
        self.native.controls = self.native.controls.checked_add(bytes)?;
        Some(())
    }
    pub(super) fn capacity(&self, elements: usize, dtype: Dtype) -> Option<u64> {
        let item = match dtype {
            Dtype::Bool => 1,
            Dtype::Int64 | Dtype::Uint64 | Dtype::Float64 | Dtype::Complex64 => 8,
            // The existing narrower sources retain their four-byte envelope.
            _ => 4,
        };
        self.mechanism
            .allocation
            .fixed_buffer_capacity((elements as u64).checked_mul(item)?)
            .ok()
    }
    pub(super) fn child(&mut self, child: OperationPlan) -> Option<()> {
        self.native.add(child.population)?;
        self.bytes = self
            .bytes
            .checked_add(child.output_bytes)?
            .checked_add(child.scratch_bytes)?;
        self.seeds = self.seeds.checked_add(child.seeds)?;
        self.shells = self.shells.checked_add(child.parameter_shells)?;
        self.validations = self.validations.checked_add(child.validations)?;
        self.controls(size_of::<(&mut Self, OperationPlan, Option<()>)>())
    }
    pub(super) fn repeat(&mut self, other: &Self, count: usize) -> Option<()> {
        let p = other.native;
        self.native.add(CpuPopulation {
            construction_entries: p.construction_entries.checked_mul(count)?,
            primitives: p.primitives.checked_mul(count)?,
            input_edges: p.input_edges.checked_mul(count)?,
            hidden_leaves: p.hidden_leaves.checked_mul(count)?,
            maximum_operands: p.maximum_operands,
            maximum_captures: p.maximum_captures,
            births: p.births.checked_mul(count)?,
            extents: p.extents.checked_mul(count)?,
            controls: p.controls.checked_mul(count)?,
        })?;
        self.bytes = self
            .bytes
            .checked_add(other.bytes.checked_mul(count as u64)?)?;
        self.seeds = self.seeds.checked_add(other.seeds.checked_mul(count)?)?;
        self.shells = self.shells.checked_add(other.shells.checked_mul(count)?)?;
        self.validations = self
            .validations
            .checked_add(other.validations.checked_mul(count)?)?;
        self.controls(size_of::<(
            &mut Self,
            &Self,
            usize,
            CpuPopulation,
            Option<()>,
        )>())
    }
    pub(super) fn copy(
        &mut self,
        source: CpuCopyEvalLayout,
        inputs: usize,
        elements: usize,
        dtype: Dtype,
    ) -> Option<()> {
        self.bytes = self.bytes.checked_add(
            self.capacity(elements, dtype)?
                .checked_mul(source.backing_births() as u64)?,
        )?;
        self.native.copy(source, inputs)?;
        self.controls(size_of::<(
            &mut Self,
            CpuCopyEvalLayout,
            usize,
            usize,
            Dtype,
            Option<()>,
        )>())
    }
    pub(super) fn scalar(&mut self) -> Option<()> {
        self.seeds = self.seeds.checked_add(1)?;
        self.bytes = self.bytes.checked_add(self.capacity(1, Dtype::Int32)?)?;
        self.controls(size_of::<(&mut Self, Option<()>)>())
    }
    pub(super) fn cast(
        &mut self,
        from: Dtype,
        to: Dtype,
        rank: usize,
        elements: usize,
    ) -> Option<()> {
        self.copy(
            OperationEvent::cpu_cast_layout(from, to, rank, elements, false)?,
            1,
            elements,
            to,
        )
    }
    pub(super) fn broadcast(&mut self, before: usize, after: usize) -> Option<()> {
        self.copy(
            OperationEvent::cpu_broadcast_alias_layout(before, after, false)?,
            1,
            0,
            Dtype::Float32,
        )
    }
    pub(super) fn reshape(&mut self, before: usize, after: usize) -> Option<()> {
        self.copy(
            OperationEvent::cpu_reshape_alias_layout(before, after, false)?,
            1,
            0,
            Dtype::Float32,
        )
    }
    pub(super) fn slice(&mut self, rank: usize) -> Option<()> {
        self.copy(
            OperationEvent::cpu_slice_layout(rank, false, false)?,
            1,
            0,
            Dtype::Float32,
        )
    }
    pub(super) fn unary(
        &mut self,
        kind: CpuUnaryOperation,
        dtype: Dtype,
        rank: usize,
        elements: usize,
    ) -> Option<()> {
        let source = OperationEvent::cpu_unary_layout(kind, dtype, rank, false)?;
        self.bytes = self.bytes.checked_add(
            self.capacity(elements, dtype)?
                .checked_mul(source.backing_births() as u64)?,
        )?;
        self.native.unary(source)?;
        self.controls(size_of::<(
            &mut Self,
            CpuUnaryOperation,
            Dtype,
            usize,
            usize,
            safemlx::CpuUnaryEvalLayout,
            Option<()>,
        )>())
    }
    pub(super) fn binary(
        &mut self,
        kind: CpuBinaryOperation,
        dtype: Dtype,
        rank: usize,
        elements: usize,
        left_rank: usize,
        right_rank: usize,
        left_count: usize,
        right_count: usize,
    ) -> Option<()> {
        self.cast(dtype, dtype, left_rank, left_count)?;
        self.cast(dtype, dtype, right_rank, right_count)?;
        self.broadcast(left_rank, rank)?;
        self.broadcast(right_rank, rank)?;
        let source = OperationEvent::cpu_binary_layout(kind, dtype, rank, elements, false)?;
        self.bytes = self.bytes.checked_add(
            self.capacity(elements, dtype)?
                .checked_mul(source.backing_births() as u64)?,
        )?;
        self.native.binary(source)?;
        self.controls(size_of::<(
            &mut Self,
            CpuBinaryOperation,
            Dtype,
            usize,
            usize,
            usize,
            usize,
            usize,
            usize,
            safemlx::CpuBinaryEvalLayout,
            Option<()>,
        )>())
    }
    pub(super) fn scalar_binary(
        &mut self,
        kind: CpuBinaryOperation,
        dtype: Dtype,
        rank: usize,
        elements: usize,
    ) -> Option<()> {
        self.scalar()?;
        self.binary(kind, dtype, rank, elements, rank, 0, elements, 1)
    }
    pub(super) fn full(&mut self, dtype: Dtype, rank: usize, elements: usize) -> Option<()> {
        self.scalar()?;
        self.broadcast(0, rank)?;
        self.copy(
            OperationEvent::cpu_scalar_full_layout(dtype, rank, elements, false)?,
            1,
            elements,
            dtype,
        )?;
        self.controls(super::super::zero_fill::control_bytes()?)
    }
    pub(super) fn gather(
        &mut self,
        dtype: Dtype,
        index: Dtype,
        source_rank: usize,
        source_elements: usize,
        selections: usize,
        width: usize,
    ) -> Option<()> {
        // One destination plus source/index/output weak Data publications.
        self.native.maximum_captures = self.native.maximum_captures.max(4);
        self.broadcast(1, 1)?;
        self.cast(index, index, 1, selections)?;
        self.copy(
            OperationEvent::cpu_gather_layout(
                dtype,
                index,
                source_rank,
                1,
                source_elements,
                selections,
                width,
                false,
            )?,
            2,
            selections.checked_mul(width)?,
            dtype,
        )?;
        self.copy(
            OperationEvent::cpu_squeeze_layout(source_rank + 1, false)?,
            1,
            0,
            dtype,
        )
    }
}
