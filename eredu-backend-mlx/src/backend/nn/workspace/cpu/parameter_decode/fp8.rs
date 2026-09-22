//! CPU task census of the unchanged standalone block-FP8 graph.
use super::geometry::fp8 as source;
use super::*;
use safemlx::CpuUnaryOperation;

fn populate(c: &mut Census, g: source::Geometry) -> Option<()> {
    let r = g.work_rank;
    if g.partitioned {
        c.reshape_source(g.rank, r, g.values, Dtype::Uint8)?;
        c.reshape_source(g.rank, r, g.scales, g.scale_dtype)?;
    }
    if g.encoded_scale {
        c.seed(256, Dtype::Float32)?;
        c.cast(Dtype::Uint8, Dtype::Uint32, r, g.scales)?;
        c.reshape(1, 1)?; // take without an axis flattens the table
        c.cast(Dtype::Uint32, Dtype::Uint32, r, g.scales)?;
        c.copy(
            OperationEvent::cpu_gather_layout(
                Dtype::Float32,
                Dtype::Uint32,
                1,
                r,
                256,
                g.scales,
                1,
                false,
            )?,
            2,
            g.scales,
            Dtype::Float32,
        )?;
        c.copy(
            OperationEvent::cpu_squeeze_layout(r + 1, false)?,
            1,
            0,
            Dtype::Float32,
        )?;
    }
    for count in [g.first_repeat()?, g.second_repeat()?] {
        // ops.cpp::repeat: expand_dims, Broadcast, then the actual reshape
        // compaction. The repeated dimension carries a zero source stride.
        c.reshape(r, r + 1)?;
        c.copy(
            OperationEvent::cpu_broadcast_alias_layout(r + 1, r + 1, false)?,
            1,
            0,
            g.decoded_scale(),
        )?;
        c.reshape_source(r + 1, r, count, g.decoded_scale())?;
    }
    let unary =
        OperationEvent::cpu_unary_layout(CpuUnaryOperation::FromFp8, Dtype::Float32, r, false)?;
    c.bytes = c.bytes.checked_add(
        c.capacity(g.values, Dtype::Float32)?
            .checked_mul(unary.backing_births() as u64)?,
    )?;
    c.native.unary(unary)?;
    c.copy(
        OperationEvent::cpu_slice_layout(r, false, false)?,
        1,
        0,
        g.decoded_scale(),
    )?;
    c.binary_mixed(
        CpuBinaryOperation::Multiply,
        Dtype::Float32,
        r,
        g.values,
        (Dtype::Float32, r, g.values),
        (g.decoded_scale(), r, g.second_repeat()?),
    )?;
    if g.partitioned {
        c.reshape(r, g.rank)?;
    }
    Some(())
}
pub(super) fn inspect(
    op: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    let Some(g) = source::inspect(op)? else {
        return Ok(None);
    };
    let mut census = Census {
        native: CpuPopulation::default(),
        bytes: 0,
        seeds: 0,
        mechanism,
    };
    if populate(&mut census, g).is_none() {
        return Ok(None);
    }
    let output = census
        .capacity(g.values, Dtype::Float32)
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    let frames = [
        size_of::<Census>(),
        size_of::<source::Geometry>(),
        size_of::<OperationPlan>(),
        size_of::<Option<OperationPlan>>(),
        size_of::<CpuCopyEvalLayout>(),
        size_of::<Option<CpuCopyEvalLayout>>(),
        size_of::<safemlx::CpuUnaryEvalLayout>(),
        size_of::<Option<safemlx::CpuUnaryEvalLayout>>(),
        size_of::<CpuUnaryOperation>(),
        size_of::<safemlx::CpuBinaryEvalLayout>(),
        size_of::<usize>() * 16,
        size_of::<Dtype>() * 3,
        size_of::<(&mut Census, source::Geometry)>(),
        size_of::<std::array::IntoIter<usize, 2>>(),
        source::control_bytes(g).ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
    ];
    census.native.controls = frames
        .into_iter()
        .try_fold(
            census
                .native
                .controls
                .checked_add(size_of_val(&frames))
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
            usize::checked_add,
        )
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    Ok(Some(OperationPlan {
        dtype: WorkspaceFloatingType::Float32,
        population: census.native,
        alias_input: None,
        output_bytes: output,
        scratch_bytes: census
            .bytes
            .checked_sub(output)
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
        rank: g.work_rank + 1,
        parameter_shells: 0,
        seeds: census.seeds,
        validations: 0,
    }))
}
