//! Centroid-selected readout includes actual gathered weight replicas for every
//! position. It does not assume a direct indexed product against a shared bank.

use super::{
    matrix::matmul_cost_fixed as matmul_cost,
    reduction::{capacity_fixed as capacity, sum_cost_fixed as sum_cost},
    *,
};

use super::facts::{self, add, mul, Emitter, FactResult, Output};

fn invalid() -> MlxWorkspaceFactError {
    MlxWorkspaceFactError::descriptor("invalid Metal masked-output workspace descriptor")
}

pub(super) fn operation_bound(
    op: &WorkspaceOperation,
    a: NativeAllocationFacts,
) -> Result<Option<WorkspaceOperationBound>, Error> {
    facts::ordinary_with(
        |sink| emit(op.as_view(), a, sink),
        |error| ordinary_error(op, error),
    )
}

pub(super) fn ordinary_error(op: &WorkspaceOperation, error: MlxWorkspaceFactError) -> Error {
    if let MlxWorkspaceFactCause::MaskedOutput(_) = error.cause() {
        if let WorkspaceOperationKind::MaskedOutputProjection {
            top_centroids,
            mask_margin,
        } = op.kind
        {
            if let [hidden, weight, centroids, ordering] = op.inputs.as_slice() {
                // Only the ordinary failure adapter formats the actual source
                // shapes. Fixed count/fill never invokes this adapter.
                if let Err(original) = eredu_nn::operation_geometry::validate_masked_output_geometry(
                    hidden.shape(),
                    weight.shape(),
                    centroids.shape(),
                    ordering.shape(),
                    top_centroids,
                    mask_margin,
                ) {
                    return original;
                }
            }
        }
    }
    error.ordinary()
}

#[derive(Clone, Copy)]
pub(super) struct ReadoutGeometry {
    pub batch: i32,
    pub sequence: i32,
    pub hidden: i32,
    pub vocabulary: i32,
    pub centroids: i32,
    pub selected: i32,
}
pub(super) fn geometry(op: WorkspaceOperationView<'_>) -> FactResult<Option<ReadoutGeometry>> {
    let WorkspaceOperationKindView::MaskedOutputProjection {
        top_centroids,
        mask_margin,
    } = op.kind
    else {
        return Ok(None);
    };
    let Some([hidden, weight, centroids, ordering]) = op.inputs.array() else {
        return Err(invalid());
    };
    let Some([output]) = op.outputs.array() else {
        return Err(invalid());
    };
    eredu_nn::operation_geometry::validate_masked_output_geometry_fixed(
        hidden.shape(),
        weight.shape(),
        centroids.shape(),
        ordering.shape(),
        top_centroids,
        mask_margin,
    )?;
    let [b, s, h] = <[i32; 3]>::try_from(hidden.shape()).map_err(|_| invalid())?;
    let v = weight.shape()[0];
    let c = centroids.shape()[2];
    let selected = top_centroids * (v / c);
    if output.shape() != [b, s, v] || output.dtype() != WorkspaceDtype::Float32 {
        return Err(invalid());
    }
    if [hidden, weight, centroids]
        .iter()
        .any(|x| x.dtype() != WorkspaceDtype::Float32)
        || !matches!(
            ordering.dtype(),
            WorkspaceDtype::Int32 | WorkspaceDtype::Uint32
        )
    {
        return Ok(None);
    }
    Ok(Some(ReadoutGeometry {
        batch: b,
        sequence: s,
        hidden: h,
        vocabulary: v,
        centroids: c,
        selected,
    }))
}

pub(super) fn emit(
    op: WorkspaceOperationView<'_>,
    a: NativeAllocationFacts,
    sink: &mut Emitter<'_>,
) -> FactResult<Option<WorkspaceOperationFacts>> {
    let Some(ReadoutGeometry {
        batch: b,
        sequence: s,
        hidden: h,
        vocabulary: v,
        centroids: c,
        selected,
    }) = geometry(op)?
    else {
        return Ok(None);
    };
    let centroids = op.inputs.get(2).unwrap();
    let output = op.outputs.get(0).unwrap();
    let r = |n| capacity(a, n);
    let out = r(output.elements()?)?;
    let rows = mul(b as u64, s as u64)?;
    let total = if rows == 0 {
        // Explicit empty F32 zeros bypass native sort, gather and reductions.
        add(out, r(1)?)?
    } else {
        let ids = mul(rows, selected as u64)?;
        let replicas = mul(ids, h as u64)?;
        // Partition retains its complete centroid axis. A noncontiguous sort
        // can copy scores; slicing the selected indices retains that backing.
        let mut bytes = add(
            sampling::sort_fixed(a, centroids.elements()?, rows, c as u64)?,
            r(centroids.elements()?)?,
        )?;
        // Ordering reshape, token gather, flatten/reshape copies for indices,
        // and the actual weight gather plus its possible reshape copy.
        bytes = add(bytes, add(r(v as u64)?, mul(4, r(ids)?)?)?)?;
        bytes = add(bytes, mul(2, r(replicas)?)?)?;
        bytes = add(bytes, matmul_cost(&[b, s, 1, h], &[b, s, h, selected], a)?)?;
        // Min uses the same native reduction dispatch and <=F32 partial
        // buffers as sum. Keep a separate result for every batch/position.
        bytes = add(bytes, sum_cost(a, ids, rows, selected as u64)?)?;
        // Margin scalar, possible row/scalar promotion and subtraction, F32
        // fill conversion, dense vocabulary fill and scatter output. Include
        // selected-score promotion before scattering into the F32 output.
        bytes = add(bytes, add(mul(4, r(rows)?)?, mul(2, r(1)?)?)?)?;
        add(bytes, add(mul(2, out)?, r(ids)?)?)?
    };
    // Empty zeros and the nonempty margin each construct one eager F32
    // source; row/scalar casts remain allocations on the execution stream.
    sink.default_scratch(r(1)?, 1)?;
    sink.output(Output::Allocate(out))?;
    sink.finish(total - out, format_args!("MLX Metal masked readout: full centroid partition backing, strided ordering/indices, selected weight replicas per position, dense products and copies/casts, per-position minimum, F32 fill/scatter; all child buffers retained through completion; page={} with bounded oversized reuse; no disjoint host payload", a.page_size())).map(Some)
}

#[cfg(test)]
mod tests;
