//! Dense lookup uses the ordinary domain validation, safe indices, and masks.
use super::*;
use eredu_checkpoint::LinearFormat;
use eredu_nn::EmbeddingLookupPolicy;
use safemlx::{CpuUnaryOperation, Dtype};

#[derive(Clone, Copy, Default)]
struct Population {
    native: CpuPopulation,
    index_births: usize,
    mask_births: usize,
    scalar_births: usize,
    value_births: usize,
}
#[derive(Clone, Copy)]
enum Buffer {
    Index,
    Mask,
    Scalar,
    Value,
}
impl Population {
    fn births(&mut self, buffer: Option<Buffer>, count: usize) -> Option<()> {
        let slot = match buffer {
            Some(Buffer::Index) => &mut self.index_births,
            Some(Buffer::Mask) => &mut self.mask_births,
            Some(Buffer::Scalar) => &mut self.scalar_births,
            Some(Buffer::Value) => &mut self.value_births,
            None => return Some(()),
        };
        *slot = slot.checked_add(count)?;
        Some(())
    }
    fn copy(
        &mut self,
        source: CpuCopyEvalLayout,
        inputs: usize,
        buffer: Option<Buffer>,
    ) -> Option<()> {
        self.births(buffer, source.backing_births())?;
        self.native.copy(source, inputs)
    }
    fn cast(
        &mut self,
        source: Dtype,
        destination: Dtype,
        rank: usize,
        count: usize,
        buffer: Buffer,
    ) -> Option<()> {
        self.copy(
            OperationEvent::cpu_cast_layout(source, destination, rank, count, false)?,
            1,
            Some(buffer),
        )
    }
    fn alias(&mut self, rank: usize, output_rank: usize) -> Option<()> {
        self.copy(
            OperationEvent::cpu_broadcast_alias_layout(rank, output_rank, false)?,
            1,
            None,
        )
    }
    fn binary(
        &mut self,
        kind: CpuBinaryOperation,
        dtype: Dtype,
        rank: usize,
        count: usize,
        scalar: bool,
    ) -> Option<()> {
        self.binary_output(kind, dtype, rank, count, scalar, Buffer::Mask)
    }
    fn binary_output(
        &mut self,
        kind: CpuBinaryOperation,
        dtype: Dtype,
        rank: usize,
        count: usize,
        scalar: bool,
        output: Buffer,
    ) -> Option<()> {
        let input = if dtype == Dtype::Bool {
            Buffer::Mask
        } else {
            Buffer::Index
        };
        // Preserve the ordinary cast/broadcast constructor envelope. A native
        // identity cast/view can use less; it does not mint another source.
        self.cast(dtype, dtype, rank, count, input)?;
        self.cast(
            dtype,
            dtype,
            if scalar { 0 } else { rank },
            if scalar { 1 } else { count },
            if scalar { Buffer::Scalar } else { input },
        )?;
        self.alias(rank, rank)?;
        self.alias(if scalar { 0 } else { rank }, rank)?;
        let source = OperationEvent::cpu_binary_layout(kind, dtype, rank, count, false)?;
        self.births(Some(output), source.backing_births())?;
        self.native.binary(source)
    }
}

// Shared prefix of the ordinary embedding and sampled/forced-token workers.
// Every leaf is queried from the same CPU native source as its actual Eval.
fn token_validation_population(
    index_dtype: Dtype,
    rank: usize,
    count: usize,
    sentinel: bool,
) -> Option<Population> {
    let mut source = Population::default();
    source.cast(index_dtype, Dtype::Int32, rank, count, Buffer::Index)?;
    if count != 0 {
        source.binary(
            CpuBinaryOperation::GreaterEqual,
            Dtype::Int32,
            rank,
            count,
            true,
        )?;
        source.binary(CpuBinaryOperation::Less, Dtype::Int32, rank, count, true)?;
        source.binary(
            CpuBinaryOperation::LogicalAnd,
            Dtype::Bool,
            rank,
            count,
            false,
        )?;
        if sentinel {
            source.binary(CpuBinaryOperation::Equal, Dtype::Int32, rank, count, true)?;
            source.binary(
                CpuBinaryOperation::LogicalOr,
                Dtype::Bool,
                rank,
                count,
                false,
            )?;
        }
        source.cast(Dtype::Bool, Dtype::Bool, rank, count, Buffer::Mask)?;
        let not = OperationEvent::cpu_unary_layout(
            CpuUnaryOperation::LogicalNot,
            Dtype::Bool,
            rank,
            false,
        )?;
        source.births(Some(Buffer::Mask), not.backing_births())?;
        source.native.unary(not)?;
        if rank != 0 {
            source.copy(
                OperationEvent::cpu_boolean_reduce_layout(false, rank, count, false)?,
                1,
                Some(Buffer::Scalar),
            )?;
            source.copy(OperationEvent::cpu_squeeze_layout(rank, false)?, 1, None)?;
        }
        source.cast(Dtype::Int32, Dtype::Int32, rank, count, Buffer::Index)?;
    }
    let frames = [
        size_of::<(Dtype, usize, usize, bool)>(),
        size_of::<Population>(),
        size_of::<Option<Population>>(),
    ];
    source.native.controls = frames.into_iter().try_fold(
        source.native.controls.checked_add(size_of_val(&frames))?,
        usize::checked_add,
    )?;
    Some(source)
}

/// The existing tensor take-axis predicate differs from token normalization:
/// it preserves signed negative indices and does not cast the borrowed input
/// before its two comparisons. Its safe index output is retained by Gather.
pub(super) fn checked_take_indices(
    rank: usize,
    count: usize,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    if rank != 1 || count == 0 || count > i32::MAX as usize {
        return Ok(None);
    }
    let source = (|| {
        let mut source = Population::default();
        for (kind, dtype, scalar) in [
            (CpuBinaryOperation::GreaterEqual, Dtype::Int32, true),
            (CpuBinaryOperation::Less, Dtype::Int32, true),
            (CpuBinaryOperation::LogicalAnd, Dtype::Bool, false),
        ] {
            // Same-dtype casts and the indices' same-shape broadcast are
            // frontend candidates only. Each comparison really broadcasts its
            // scalar before the binary Eval; the mask conjunction does not.
            if scalar {
                source.alias(0, rank)?;
            }
            let binary = OperationEvent::cpu_binary_layout(kind, dtype, rank, count, false)?;
            source.births(Some(Buffer::Mask), binary.backing_births())?;
            source.native.binary(binary)?;
        }
        let inverse = OperationEvent::cpu_unary_layout(
            CpuUnaryOperation::LogicalNot,
            Dtype::Bool,
            rank,
            false,
        )?;
        source.births(Some(Buffer::Mask), inverse.backing_births())?;
        source.native.unary(inverse)?;
        // any_owned elides its Reduce for a singleton axis, but always
        // squeezes the selected axis when keepdims is false.
        if count != 1 {
            source.copy(
                OperationEvent::cpu_boolean_reduce_layout(false, rank, count, false)?,
                1,
                Some(Buffer::Scalar),
            )?;
        }
        source.copy(OperationEvent::cpu_squeeze_layout(rank, false)?, 1, None)?;
        // zeros_like is one eager I32 seed followed by Broadcast and Full;
        // where's same-dtype casts/broadcasts leave only its Select Eval.
        source.alias(0, rank)?;
        source.copy(
            OperationEvent::cpu_scalar_full_layout(Dtype::Int32, rank, count, false)?,
            1,
            Some(Buffer::Index),
        )?;
        source.copy(
            OperationEvent::cpu_select_layout(Dtype::Int32, rank, count, false)?,
            3,
            Some(Buffer::Index),
        )?;
        // Three binary frontends each have two casts, two broadcasts and the
        // binary constructor. LogicalNot has cast+unary, any has Reduce (or
        // identity cast)+Squeeze, zeros has Broadcast+cast+Full, and Select has
        // three casts+three broadcasts+Select. Graph storage counts candidates;
        // CPU tape, edges and backing births above count only actual Evals.
        source.native.construction_entries = 3 * 5 + 2 + 2 + 3 + 7;
        Some(source)
    })();
    let Some(mut source) = source else {
        return Ok(None);
    };
    let index = mechanism.allocation.fixed_buffer_capacity(
        (count as u64)
            .checked_mul(4)
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
    )?;
    let mask = mechanism.allocation.fixed_buffer_capacity(count as u64)?;
    let scalar = mechanism.allocation.fixed_buffer_capacity(4)?;
    let scratch_bytes = facts::add(
        facts::add(
            facts::mul(
                index,
                u64::try_from(
                    source
                        .index_births
                        .checked_sub(1)
                        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
                )?,
            )?,
            facts::mul(mask, u64::try_from(source.mask_births)?)?,
        )?,
        facts::mul(
            scalar,
            u64::try_from(
                source
                    .scalar_births
                    .checked_add(3)
                    .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
            )?,
        )?,
    )?;
    let frames = [
        size_of::<(usize, usize, MlxCpuWorkspaceMechanisms)>(),
        size_of::<Population>(),
        size_of::<Option<Population>>(),
        size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<CpuCopyEvalLayout>(),
        size_of::<OperationPlan>(),
        size_of::<facts::FactResult<Option<OperationPlan>>>(),
        size_of::<[u64; 4]>(),
        size_of::<Dtype>(),
        size_of::<bool>(),
        size_of::<CpuBinaryOperation>(),
        size_of::<safemlx::CpuBinaryEvalLayout>(),
        size_of::<Option<safemlx::CpuBinaryEvalLayout>>(),
        size_of::<safemlx::CpuUnaryEvalLayout>(),
        size_of::<Option<safemlx::CpuUnaryEvalLayout>>(),
        size_of::<[(CpuBinaryOperation, Dtype, bool); 3]>(),
        size_of::<std::array::IntoIter<(CpuBinaryOperation, Dtype, bool), 3>>(),
        size_of::<(&mut Population, CpuCopyEvalLayout, usize, Option<Buffer>)>(),
        size_of::<(&mut Population, Option<Buffer>, usize)>(),
        size_of::<(&mut Population, usize, usize)>(),
        size_of::<&mut usize>(),
        size_of::<Option<Buffer>>(),
        size_of::<Option<()>>(),
        super::super::zero_fill::control_bytes()
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
    ];
    source.native.controls = frames
        .into_iter()
        .try_fold(
            source
                .native
                .controls
                .checked_add(size_of_val(&frames))
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
            usize::checked_add,
        )
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    Ok(Some(OperationPlan {
        dtype: WorkspaceFloatingType::Float32,
        population: source.native,
        alias_input: None,
        output_bytes: index,
        scratch_bytes,
        rank,
        parameter_shells: 0,
        seeds: 3,
        validations: 1,
    }))
}

pub(super) fn inspect_token_validation(
    operation: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    if !matches!(
        operation.kind,
        WorkspaceOperationKindView::Sampling(WorkspaceSamplingOperation::ValidateToken { .. })
    ) {
        return Ok(None);
    }
    let [input, output] =
        super::super::sampling::token_validation_layouts(operation).ok_or_else(|| {
            MlxWorkspaceFactError::descriptor("CPU token validation domain or geometry differs")
        })?;
    let rank = input.shape().len();
    let count = usize::try_from(input.elements()?)?;
    if rank > 4 || count > i32::MAX as usize {
        return Ok(None);
    }
    let dtype = if input.dtype() == WorkspaceDtype::Int32 {
        Dtype::Int32
    } else {
        Dtype::Uint32
    };
    let Some(mut source) = token_validation_population(dtype, rank, count, false) else {
        return Ok(None);
    };
    let output_birth = usize::from(dtype != Dtype::Int32);
    let index_births = source
        .index_births
        .checked_sub(output_birth)
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    let index_capacity = mechanism
        .allocation
        .fixed_buffer_capacity(output.bytes()?)?;
    let mask_capacity = mechanism
        .allocation
        .fixed_buffer_capacity(input.elements()?)?;
    let scalar_capacity = mechanism.allocation.fixed_buffer_capacity(4)?;
    let seeds = if count == 0 { 0usize } else { 2 };
    let scratch_bytes = facts::add(
        facts::add(
            facts::mul(index_capacity, u64::try_from(index_births)?)?,
            facts::mul(mask_capacity, u64::try_from(source.mask_births)?)?,
        )?,
        facts::mul(
            scalar_capacity,
            u64::try_from(
                source
                    .scalar_births
                    .checked_add(seeds)
                    .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
            )?,
        )?,
    )?;
    let frames = [
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<MlxCpuWorkspaceMechanisms>(),
        size_of::<[WorkspaceLayoutView<'_>; 2]>(),
        size_of::<Option<[WorkspaceLayoutView<'_>; 2]>>(),
        size_of::<Population>(),
        size_of::<Option<Population>>(),
        size_of::<OperationPlan>(),
        size_of::<facts::FactResult<Option<OperationPlan>>>(),
        size_of::<(Dtype, usize, usize, bool)>(),
        size_of::<usize>() * 5,
        size_of::<u64>() * 4,
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
        dtype: WorkspaceFloatingType::Float32,
        population: source.native,
        alias_input: if output_birth == 0 { Some(0) } else { None },
        output_bytes: if output_birth == 0 { 0 } else { index_capacity },
        scratch_bytes,
        rank,
        parameter_shells: 0,
        seeds,
        validations: usize::from(count != 0),
    }))
}

/// The actual original grouped-ID prefix shares the validation predicate with
/// lookup, but reuses that predicate for its zeros_like/Select safe output.
pub(super) fn group_indices(
    index_dtype: Dtype,
    rank: usize,
    count: usize,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    if rank > 2 || count == 0 || count > i32::MAX as usize {
        return Ok(None);
    }
    let source = (|| {
        let mut source = token_validation_population(index_dtype, rank, count, false)?;
        source.alias(rank, rank)?;
        source.alias(0, rank)?;
        source.cast(Dtype::Int32, Dtype::Int32, rank, count, Buffer::Index)?;
        source.copy(
            OperationEvent::cpu_scalar_full_layout(Dtype::Int32, rank, count, false)?,
            1,
            Some(Buffer::Index),
        )?;
        source.cast(Dtype::Bool, Dtype::Bool, rank, count, Buffer::Mask)?;
        source.cast(Dtype::Int32, Dtype::Int32, rank, count, Buffer::Index)?;
        source.cast(Dtype::Int32, Dtype::Int32, rank, count, Buffer::Index)?;
        for _ in 0..3 {
            source.alias(rank, rank)?;
        }
        source.copy(
            OperationEvent::cpu_select_layout(Dtype::Int32, rank, count, false)?,
            3,
            Some(Buffer::Index),
        )?;
        Some(source)
    })();
    let Some(mut source) = source else {
        return Ok(None);
    };
    let index = mechanism.allocation.fixed_buffer_capacity(
        (count as u64)
            .checked_mul(4)
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
    )?;
    let mask = mechanism.allocation.fixed_buffer_capacity(count as u64)?;
    let scalar = mechanism.allocation.fixed_buffer_capacity(4)?;
    let scratch_bytes = facts::add(
        facts::add(
            facts::mul(
                index,
                u64::try_from(
                    source
                        .index_births
                        .checked_sub(1)
                        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
                )?,
            )?,
            facts::mul(mask, u64::try_from(source.mask_births)?)?,
        )?,
        facts::mul(
            scalar,
            u64::try_from(
                source
                    .scalar_births
                    .checked_add(3)
                    .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
            )?,
        )?,
    )?;
    let frames = [
        size_of::<(Dtype, usize, usize, MlxCpuWorkspaceMechanisms)>(),
        size_of::<Population>(),
        size_of::<Option<Population>>(),
        size_of::<OperationPlan>(),
        size_of::<facts::FactResult<Option<OperationPlan>>>(),
        size_of::<[u64; 4]>(),
    ];
    source.native.controls = frames
        .into_iter()
        .try_fold(
            source
                .native
                .controls
                .checked_add(size_of_val(&frames))
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
            usize::checked_add,
        )
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    Ok(Some(OperationPlan {
        dtype: WorkspaceFloatingType::Float32,
        population: source.native,
        alias_input: None,
        output_bytes: index,
        scratch_bytes,
        rank,
        parameter_shells: 0,
        seeds: 3,
        validations: 1,
    }))
}

pub(super) fn inspect(
    operation: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    let (operation, parallel_policy) = match operation.kind {
        WorkspaceOperationKindView::VocabularyParallelLookup { policy, .. } => {
            let Some(child) = super::super::parallel_lookup::embedding(operation)? else {
                return Ok(None);
            };
            (child, Some(policy))
        }
        _ => (operation, None),
    };
    let WorkspaceOperationKindView::Embedding(format, policy) = operation.kind else {
        return Ok(None);
    };
    let policy = parallel_policy.unwrap_or(policy);
    policy.validate_fixed()?;
    let parallel = parallel_policy.is_some();
    let sentinel = matches!(policy, EmbeddingLookupPolicy::ZeroSentinel(_));
    if format.encoding() != LinearFormat::Dense {
        return Ok(None);
    }
    format.validate_fixed()?;
    if operation.inputs.len() != 2 || operation.outputs.len() != 1 {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU strict lookup population differs",
        ));
    }
    let ids = operation.inputs.get(0).expect("two lookup inputs");
    let weight = operation.inputs.get(1).expect("two lookup inputs");
    let output = operation.outputs.get(0).expect("one lookup output");
    let rank = ids.shape().len();
    let index_dtype = match ids.dtype() {
        WorkspaceDtype::Int32 => Dtype::Int32,
        WorkspaceDtype::Uint32 => Dtype::Uint32,
        _ => return Ok(None),
    };
    if rank > 2
        || ids.shape().iter().any(|&n| n <= 0)
        || weight.shape().len() != 2
        || weight.dtype() != WorkspaceDtype::Float32
        || weight.shape().iter().any(|&n| n <= 0)
        || output.dtype() != WorkspaceDtype::Float32
    {
        return Ok(None);
    }
    let Some(representation) = weight.representation() else {
        return Ok(None);
    };
    let dtype = representation.dtype();
    let native_dtype = match dtype {
        WorkspaceFloatingType::Float32 => Dtype::Float32,
        WorkspaceFloatingType::Bfloat16 => Dtype::Bfloat16,
        WorkspaceFloatingType::Float16 => Dtype::Float16,
    };
    if !output.shape().iter().copied().eq(ids
        .shape()
        .iter()
        .copied()
        .chain(std::iter::once(weight.shape()[1])))
    {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU strict lookup output differs",
        ));
    }
    let count = usize::try_from(ids.elements()?)?;
    let source_elements = usize::try_from(weight.elements()?)?;
    let columns = usize::try_from(weight.shape()[1])?;
    let output_elements = usize::try_from(output.elements()?)?;
    if count > i32::MAX as usize
        || source_elements > i32::MAX as usize
        || output_elements > i32::MAX as usize
    {
        return Ok(None);
    }
    let source = (|| {
        let mut source = token_validation_population(index_dtype, rank, count, sentinel)?;
        // Actual safe-index mask repeats ge/lt/and before zeros_like/where.
        source.binary(
            CpuBinaryOperation::GreaterEqual,
            Dtype::Int32,
            rank,
            count,
            true,
        )?;
        source.binary(CpuBinaryOperation::Less, Dtype::Int32, rank, count, true)?;
        source.binary(
            CpuBinaryOperation::LogicalAnd,
            Dtype::Bool,
            rank,
            count,
            false,
        )?;
        if parallel {
            // The actual local-ID subtraction is followed by a scalar-zero
            // Select, not a materialized zeros_like index tensor.
            source.binary_output(
                CpuBinaryOperation::Subtract,
                Dtype::Int32,
                rank,
                count,
                true,
                Buffer::Index,
            )?;
            source.cast(Dtype::Bool, Dtype::Bool, rank, count, Buffer::Mask)?;
            source.cast(Dtype::Int32, Dtype::Int32, rank, count, Buffer::Index)?;
            source.cast(Dtype::Int32, Dtype::Int32, 0, 1, Buffer::Scalar)?;
            source.alias(rank, rank)?;
            source.alias(rank, rank)?;
            source.alias(0, rank)?;
            source.copy(
                OperationEvent::cpu_typed_select_broadcast_layout(
                    Dtype::Int32,
                    rank,
                    count,
                    false,
                )?,
                3,
                Some(Buffer::Index),
            )?;
        } else {
            source.alias(rank, rank)?;
            source.alias(0, rank)?;
            source.cast(Dtype::Int32, Dtype::Int32, rank, count, Buffer::Index)?;
            source.copy(
                OperationEvent::cpu_scalar_full_layout(Dtype::Int32, rank, count, false)?,
                1,
                Some(Buffer::Index),
            )?;
            source.cast(Dtype::Bool, Dtype::Bool, rank, count, Buffer::Mask)?;
            source.cast(Dtype::Int32, Dtype::Int32, rank, count, Buffer::Index)?;
            source.cast(Dtype::Int32, Dtype::Int32, rank, count, Buffer::Index)?;
            for _ in 0..3 {
                source.alias(rank, rank)?;
            }
            source.copy(
                OperationEvent::cpu_select_layout(Dtype::Int32, rank, count, false)?,
                3,
                Some(Buffer::Index),
            )?;
        }
        // Shared dense table lookup retains the I32 index and floating table;
        // neither this path nor its quote widens/copies the complete table.
        source.cast(Dtype::Int32, Dtype::Int32, rank, count, Buffer::Index)?;
        source.copy(
            OperationEvent::cpu_gather_layout(
                native_dtype,
                Dtype::Int32,
                2,
                rank,
                source_elements,
                count,
                columns,
                false,
            )?,
            2,
            None,
        )?;
        source.copy(
            OperationEvent::cpu_squeeze_layout(rank + 2, false)?,
            1,
            None,
        )?;
        if sentinel && !parallel {
            // Serial lookup recomputes the sentinel predicate after the row
            // kernel. Parallel lookup already has its exact ownership mask.
            source.binary(CpuBinaryOperation::Equal, Dtype::Int32, rank, count, true)?;
        }
        if parallel || sentinel {
            // Expand the actual ownership or sentinel mask; zeros_like and
            // final Select preserve the gathered parameter's physical dtype.
            source.alias(rank, rank + 1)?;
            source.alias(rank + 1, rank + 1)?;
            source.alias(0, rank + 1)?;
            // zeros_like's eager scalar is I32, broadcast before Full's
            // actual cast to the gathered table dtype (including F16/BF16).
            source.cast(
                Dtype::Int32,
                native_dtype,
                rank + 1,
                output_elements,
                Buffer::Value,
            )?;
            source.copy(
                OperationEvent::cpu_scalar_full_layout(
                    native_dtype,
                    rank + 1,
                    output_elements,
                    false,
                )?,
                1,
                Some(Buffer::Value),
            )?;
            source.cast(Dtype::Bool, Dtype::Bool, rank + 1, count, Buffer::Mask)?;
            source.cast(
                native_dtype,
                native_dtype,
                rank + 1,
                output_elements,
                Buffer::Value,
            )?;
            source.cast(
                native_dtype,
                native_dtype,
                rank + 1,
                output_elements,
                Buffer::Value,
            )?;
            for _ in 0..3 {
                source.alias(rank + 1, rank + 1)?;
            }
            source.copy(
                OperationEvent::cpu_typed_select_broadcast_layout(
                    native_dtype,
                    rank + 1,
                    output_elements,
                    false,
                )?,
                3,
                Some(Buffer::Value),
            )?;
        }
        Some(source)
    })();
    let Some(mut source) = source else {
        return Ok(None);
    };
    let classified = source
        .index_births
        .checked_add(source.mask_births)
        .and_then(|n| n.checked_add(source.scalar_births))
        .and_then(|n| n.checked_add(source.value_births))
        .and_then(|n| n.checked_add(1))
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    if classified != source.native.births {
        return Err(MlxWorkspaceFactError::descriptor(
            "CPU lookup physical source count differs",
        ));
    }
    // Validation bounds and ownership bounds are independent eager values.
    // Parallel adds its output zeros_like scalar and optionally the global
    // sentinel ID. Serial sentinel adds two separately constructed IDs and
    // the output zeros_like scalar, leaving the ordinary safe-index source.
    let seeds =
        5usize + usize::from(parallel) + usize::from(sentinel) * if parallel { 1 } else { 3 };
    let index_capacity = mechanism
        .allocation
        .fixed_buffer_capacity(facts::mul(ids.elements()?, 4)?)?;
    let mask_capacity = mechanism
        .allocation
        .fixed_buffer_capacity(ids.elements()?)?;
    let scalar_capacity = mechanism.allocation.fixed_buffer_capacity(4)?;
    let scratch_bytes = facts::add(
        facts::add(
            facts::mul(index_capacity, u64::try_from(source.index_births)?)?,
            facts::mul(mask_capacity, u64::try_from(source.mask_births)?)?,
        )?,
        facts::mul(
            scalar_capacity,
            u64::try_from(
                source
                    .scalar_births
                    .checked_add(seeds)
                    .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
            )?,
        )?,
    )?;
    // Gather owns one value buffer. For a masked lookup the final selected
    // value is output and the gathered/zero buffers remain scratch.
    let output_bytes = mechanism
        .allocation
        .fixed_buffer_capacity(facts::mul(output.elements()?, 4)?)?;
    let scratch_bytes = facts::add(
        scratch_bytes,
        facts::mul(output_bytes, u64::try_from(source.value_births)?)?,
    )?;
    let frames = [
        size_of::<Population>() * 3,
        size_of::<Option<Population>>(),
        size_of::<OperationPlan>(),
        size_of::<Option<OperationPlan>>(),
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<WorkspaceLayoutView<'_>>() * 3,
        size_of::<MlxCpuWorkspaceMechanisms>(),
        size_of::<CpuPopulation>() * 2,
        size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<CpuCopyEvalLayout>(),
        size_of::<safemlx::CpuBinaryEvalLayout>(),
        size_of::<Option<safemlx::CpuBinaryEvalLayout>>(),
        size_of::<safemlx::CpuUnaryEvalLayout>(),
        size_of::<Option<safemlx::CpuUnaryEvalLayout>>(),
        size_of::<(Dtype, Dtype, usize, usize, Buffer)>(),
        size_of::<(CpuBinaryOperation, Dtype, usize, usize, bool)>(),
        size_of::<(&mut Population, CpuCopyEvalLayout, usize, Option<Buffer>)>(),
        size_of::<(&mut Population, Option<Buffer>, usize)>(),
        size_of::<(&mut Population, usize, usize)>(),
        size_of::<&mut usize>(),
        size_of::<Option<Buffer>>(),
        size_of::<Option<()>>(),
        size_of::<WorkspaceFloatingType>(),
        size_of::<usize>() * 12,
        size_of::<u64>() * 5,
        size_of::<std::ops::Range<usize>>(),
        size_of::<Option<EmbeddingLookupPolicy>>(),
        size_of::<EmbeddingLookupPolicy>(),
        size_of::<bool>() * 2,
        size_of::<(CpuBinaryOperation, Dtype, usize, usize, bool, Buffer)>(),
        if parallel {
            crate::backend::nn::shared::parallel_vocabulary_lookup_control_bytes()
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?
        } else {
            0
        },
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
        output_bytes,
        scratch_bytes,
        rank: rank + 2,
        parameter_shells: if parallel { 6 } else { 1 },
        seeds,
        validations: 1,
    }))
}
