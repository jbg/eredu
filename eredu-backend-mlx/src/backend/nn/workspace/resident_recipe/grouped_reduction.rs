//! Constructor population of the existing stable group-ID reduction worker.
use super::*;

/// Replace the Sum tail already included by a grouped lowering. Tensor bytes
/// are independently emitted by `grouped::Cost::weighted_sum` using this same
/// architecture-owned reduction. No numerical operation changes here.
pub(super) fn extend(
    value: &mut Lowering,
    reduction: eredu_nn::GroupReduction,
    top_k: usize,
    calls: usize,
) -> Option<()> {
    match reduction {
        eredu_nn::GroupReduction::Sum => return Some(()),
        eredu_nn::GroupReduction::SequentialGroupOrder => {}
    }
    if calls == 0 {
        return Some(());
    }
    // selection::weighted_group_sum restores group IDs by the same zero,
    // reshape, one-index Scatter, reshape sequence (10/12/1); ArgSort (1/1),
    // ExpandDims/Broadcast (2/2), GatherAxis (3/4), and the zero accumulator
    // (4/4/1). Each original top-k slot then executes Slice/Squeeze (2/2),
    // Add with its two casts/two broadcasts (5/6), and the actual output-dtype
    // restoration (1/1). These replace the prior Sum/Squeeze (2/2).
    let primitives = 18usize
        .checked_add(top_k.checked_mul(8)?)?
        .checked_mul(calls)?;
    let edges = 21usize
        .checked_add(top_k.checked_mul(9)?)?
        .checked_mul(calls)?;
    let seeds = 2usize.checked_mul(calls)?;
    value.primitives = value.primitives.checked_add(primitives)?;
    value.edges = value.edges.checked_add(edges)?;
    value.seeds = value.seeds.checked_add(seeds)?;
    // The existing two reduction scratch births become five sort temporaries.
    // All actual constructors and scalar seeds retain their ordinary births.
    value.maximum_births = value
        .maximum_births
        .checked_add(primitives)?
        .checked_add(seeds)?
        .checked_add(calls.checked_mul(3)?)?;
    // This sort is over I32 group IDs in each source row, not the flattened
    // routing permutation. Its pass count depends on top-k; source rows are
    // batched by that same native sort dispatch without extra submissions.
    value.additional_sort_kernels = Some(
        value
            .additional_sort_kernels?
            .checked_add(grouped_sort_kernels(top_k)?.checked_mul(calls)?)?,
    );
    value.intermediate_rank = value.intermediate_rank.max(3);
    Some(())
}
