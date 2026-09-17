//! Existing token-domain validation, selected rows, and optional sentinel mask.
use super::*;
use eredu_checkpoint::LinearFormat;

fn dense() -> Lowering {
    // Actual strict MlxEmbedding::lookup: normalization casts(2),
    // six ge/lt/and calls(6*5), not(2), any+squeeze(2), zeros_like
    // (two Broadcasts+cast+Full), where(7), gather/cast+squeeze(3).
    // The five eager scalar descriptors survive the construction prefix.
    // Reduce may additionally allocate one compaction and one intermediate.
    let primitives = 2 + 6 * 5 + 2 + 2 + 4 + 7 + 3;
    Lowering {
        primitives,
        edges: 2 + 6 * 6 + 2 + 2 + 4 + 9 + 4,
        seeds: 5,
        validations: 1,
        maximum_births: primitives + 5 + 2,
        streams: 1,
        hidden_leaves: 0,
        maximum_operands: 4,
        intermediate_rank: 0,
        grouped_output_chunks: 0,
        grouped_output_calls: 1,
        grouped_unit_observers: 0,
        bf16_projection_calls: 0,
        pointwise_calls: 0,
        row_rms_calls: 0,
        row_sum_calls: 0,
        recurrent_calls: 0,
        router_cpu_partitions: 0,
        additional_sort_kernels: Some(0),
        nested_completions: 0,
        backend_shells: 0,
        helper_controls: 0,
        unqualified_kernel_owner: None,
    }
}

pub(super) fn lowering(operation: WorkspaceOperationView<'_>) -> Option<Lowering> {
    if let WorkspaceOperationKindView::VocabularyParallelLookup{policy,..}=operation.kind {
        let child=super::super::parallel_lookup::embedding(operation).ok()??;
        let mut value=lowering(child)?;
        // Same global validation and selected row kernel as strict lookup.
        // Local subtraction replaces the index zeros; final expand/zero/where
        // masks every nonlocal row. Global negative sentinel validation adds
        // only eq/or: ownership masking already zeros its output.
        let sentinel=usize::from(matches!(policy,eredu_nn::EmbeddingLookupPolicy::ZeroSentinel(_)));
        let primitives=13usize.checked_add(sentinel.checked_mul(10)?)?;
        let edges=16usize.checked_add(sentinel.checked_mul(12)?)?;
        let seeds=1usize.checked_add(sentinel)?;
        value.primitives=value.primitives.checked_add(primitives)?;
        value.edges=value.edges.checked_add(edges)?;
        value.seeds=value.seeds.checked_add(seeds)?;
        value.maximum_births=value.maximum_births.checked_add(primitives)?.checked_add(seeds)?;
        return Some(value);
    }

    let WorkspaceOperationKindView::Embedding(format, policy) = operation.kind
    else {
        return None;
    };
    policy.validate_fixed().ok()?;
    let mut value = dense();
    if format.encoding() != LinearFormat::Dense {
        let input = operation.inputs.first()?;
        crate::nn::QuantizedEmbedding::forward_control_bytes(input.shape().len())?;
        packed_rows(&mut value, format.encoding())?;
        // Selected rows are rank two; gather adds one axis and packed
        // dequantization retains its existing four-axis intermediate.
        value.intermediate_rank = 4;
    }
    if matches!(policy, eredu_nn::EmbeddingLookupPolicy::ZeroSentinel(_)) {
        // validate_token_domain adds Equal + LogicalOr (two casts, two
        // broadcasts, one binary each). The same registered assertion still
        // validates the full domain; it does not create a second boundary.
        // After the unchanged selected-row worker, MlxEmbedding::lookup adds
        // Equal, ExpandDims, zeros_like (two broadcasts, cast, Full), and
        // where (three casts, three broadcasts, Select). Both sentinel IDs
        // and the floating zeros_like constructor have their own eager scalar.
        let primitives = 2 * 5 + 5 + 1 + 4 + 7;
        let edges = 2 * 6 + 6 + 1 + 4 + 9;
        let seeds = 3;
        value.primitives = value.primitives.checked_add(primitives)?;
        value.edges = value.edges.checked_add(edges)?;
        value.seeds = value.seeds.checked_add(seeds)?;
        value.maximum_births = value.maximum_births.checked_add(primitives)?.checked_add(seeds)?;
    }
    Some(value)
}

/// Exactly validate_token_domain(..., None): normalization and result casts,
/// three cast/broadcast/binary calls, Bool cast/Not and all-axis Any/Squeeze.
/// The assertion is a hidden completion root independent of returned IDs.
pub(super) fn token_validation_lowering(operation: WorkspaceOperationView<'_>) -> Option<Lowering> {
    let [input, _] = super::super::sampling::token_validation_layouts(operation)?;
    if input.elements().ok()? == 0 {
        // Empty worker returns its I32 cast before reserving an assertion.
        return Some(Lowering::plain(1, 1, 0));
    }
    let reduction = if input.shape().is_empty() { 0 } else { 2 };
    let mut value = reduction_lowering(2 + 3 * 5 + 2 + reduction,
        2 + 3 * 6 + 2 + reduction, 2, reduction);
    value.validations = 1;
    Some(value)
}

fn packed_rows(value: &mut Lowering, format: LinearFormat) -> Option<()> {
    match format {
        LinearFormat::Affine(_) => {
            // Replace dense Gather/cast/Squeeze (3 nodes, 4 edges) with:
            // flatten; three row gathers (3*3/4); two companion casts;
            // Quantize(3 inputs) + its affine outer cast; F32 cast; reshape.
            value.primitives = value.primitives.checked_sub(3)?.checked_add(16)?;
            value.edges = value.edges.checked_sub(4)?.checked_add(21)?;
            // One additional primitive-sized shared-control slot is consumed
            // by the concrete fallback's Graph-owned callable. The native
            // constructor census includes its exact actual shared-owner type.
            value.primitives = value.primitives.checked_add(1)?;
            value.maximum_births = value.primitives.checked_add(value.seeds)?.checked_add(5)?;
        }
        LinearFormat::MxFp4 => {
            // Flatten; two row gathers; Quantize(2 inputs); F32 cast; reshape.
            value.primitives = value.primitives.checked_sub(3)?.checked_add(10)?;
            value.edges = value.edges.checked_sub(4)?.checked_add(13)?;
            value.primitives = value.primitives.checked_add(1)?; // same callable owner
            value.maximum_births = value.primitives.checked_add(value.seeds)?.checked_add(4)?;
        }
        // GGUF custom source owners and block-FP8 have distinct actual workers.
        _ => return None,
    }
    Some(())
}
