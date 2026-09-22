//! Exact ordinary relative-profile constructor, including its value-only branches.
use super::*;

pub(super) fn lowering(op: WorkspaceOperationView<'_>) -> Option<Lowering> {
    let g = super::super::positional::relative_geometry(op).ok()??;
    // Outer coordinates: Arange2, indexing reshapes2, Subtract5/6,
    // causal GreaterEqual5/6 and its eager I32 zero.
    let mut value = Lowering::plain(14, 14, 1);
    if g.windowed {
        // Window Less and LogicalAnd; one actual integer endpoint.
        value.primitives += 10;
        value.edges += 12;
        value.seeds += 1;
    }
    if g.repeated {
        // Each K/V performs reshape, broadcast and joining reshape.
        value.primitives += 6;
        value.edges += 6;
    }
    if g.scaled {
        // Arange, Add, cast, Divide, Maximum, Log, Multiply, Add,
        // scale reshape and query Multiply. Binary calls include both
        // cast/broadcast candidates even when the exact dtype aliases.
        value.primitives += 35;
        value.edges += 40;
        value.seeds += 5;
    }
    // Bias: a second coordinate pair/subtract (9/8); two clip binaries
    // (10/12); cast/index/broadcast (3/3); native GatherAxis (3/4);
    // ge/lt/and (15/18); Where (7/9). Two clip bounds, two validity
    // bounds and the zero bias are genuine eager scalar sources.
    value.primitives += 47;
    value.edges += 54;
    value.seeds += 5;
    if g.scaled {
        value.primitives += 5;
        value.edges += 6;
    }
    // Query scale, key transpose, Matmul8/9, bias Add, causal Where,
    // precise Softmax2/2 and Matmul8/9; scale and -infinity scalars.
    value.primitives += 36;
    value.edges += 42;
    value.seeds += 2;
    // Four compactions/two partials per actual Matmul and one softmax
    // compaction. Reshape/cast outputs are already primitive births.
    value.maximum_births = value.primitives.checked_add(value.seeds)?.checked_add(13)?;
    value.intermediate_rank = if g.repeated { 5 } else { 4 };
    value.backend_shells =
        crate::backend::nn::relative_attention::returned_handles(g.repeated, g.scaled, g.windowed)?;
    value.unqualified_kernel_owner = grouped_indexed_source_requirement();
    Some(value)
}

pub(super) fn copy_profile(op: WorkspaceOperationView<'_>) -> Option<copy_rank::Profile> {
    let g = super::super::positional::relative_geometry(op).ok()??;
    // Four coordinate expansions, one gather expansion, optional tau reshape
    // and the two reshapes per repeated K/V. Only these copies reach rank5;
    // the actual pointwise, GatherAxis and Matmul workers have rank <= 4.
    let copies = safemlx::ops::OriginalCopyWorkerLayout::inspect(
        if g.repeated { 5 } else { 4 },
        5 + usize::from(g.scaled) + 4 * usize::from(g.repeated),
        0,
        0,
    )?;
    let geometry_controls = std::mem::size_of::<super::super::positional::RelativeGeometry>() * 2
        + std::mem::size_of::<
            Result<Option<super::super::positional::RelativeGeometry>, MlxWorkspaceFactError>,
        >();
    Some(copy_rank::Profile {
        worker_rank: 4,
        extra_extents: copies.allocation_extents(),
        controls: copies
            .control_bytes()?
            .checked_add(geometry_controls)?
            .checked_add(crate::backend::nn::relative_attention::control_bytes(
                g.repeated, g.scaled, g.windowed,
            )?)?,
    })
}
