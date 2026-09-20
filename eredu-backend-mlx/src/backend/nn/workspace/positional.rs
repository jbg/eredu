//! Selected relative-profile attention and multi-axis position construction.
//! Bounds include every child allocation through completion and preserve the
//! distinction between shared native storage and Rust frequency payloads.

use super::facts::{self, add, buffer_capacity, mul, Emitter, FactResult, Output};
use super::{matrix::matmul_cost_fixed as matmul_cost, reduction::capacity_fixed as capacity, *};
use eredu_nn::multimodal::{MultiAxisRotaryLayout, MultiAxisRotarySpec, MultiAxisRotarySpecRef};

fn invalid() -> MlxWorkspaceFactError {
    MlxWorkspaceFactError::descriptor("invalid Metal positional workspace descriptor")
}
fn count(shape: &[i32]) -> FactResult<u64> {
    shape
        .iter()
        .try_fold(1, |n, d| mul(n, u64::try_from(*d).map_err(|_| invalid())?))
}
fn wide(a: NativeAllocationFacts, n: u64) -> FactResult<u64> {
    buffer_capacity(a, mul(n.max(1), 8)?)
}

pub(super) fn operation_bound(
    op: &WorkspaceOperation,
    a: NativeAllocationFacts,
) -> Result<Option<WorkspaceOperationBound>, Error> {
    facts::ordinary_with(
        |sink| emit(op.as_view(), a, sink),
        |error| match &op.kind {
            WorkspaceOperationKind::MultiAxisRotary(spec)
            | WorkspaceOperationKind::PreparedMultiAxisRotary(spec) => {
                ordinary_rotary_error(spec, error)
            }
            _ => error.ordinary(),
        },
    )
}

pub(super) fn emit(
    op: WorkspaceOperationView<'_>,
    a: NativeAllocationFacts,
    sink: &mut Emitter<'_>,
) -> FactResult<Option<WorkspaceOperationFacts>> {
    match &op.kind {
        WorkspaceOperationKindView::RelativeAttention {
            ..
        } => relative(
            op,
            a,
            sink,
        ),
        WorkspaceOperationKindView::MultiAxisRotary(spec)
        | WorkspaceOperationKindView::PreparedMultiAxisRotary(spec) => rotary(op, a, *spec, sink),
        _ => Ok(None),
    }
}

#[derive(Clone, Copy)]
pub(super) struct RelativeGeometry {
    pub q: [i32; 4],
    pub k: [i32; 4],
    pub repeated: bool,
    pub scaled: bool,
    pub windowed: bool,
}

/// Same fixed descriptor validation feeds physical buffers and native graph
/// construction. A valid but unqualified dtype remains an explicit absence.
pub(super) fn relative_geometry(op: WorkspaceOperationView<'_>) -> FactResult<Option<RelativeGeometry>> {
    let WorkspaceOperationKindView::RelativeAttention {
        query_offset: qo, key_offset: ko, window,
        log_scaling_floor: floor, log_scaling_alpha: alpha,
    } = op.kind else { return Ok(None); };
    if op.inputs.len() != 4 || op.outputs.len() != 1 {
        return Err(invalid());
    }
    let q = op.inputs.get(0).unwrap().shape();
    let k = op.inputs.get(1).unwrap().shape();
    let v = op.inputs.get(2).unwrap().shape();
    let p = op.inputs.get(3).unwrap().shape();
    if q.len() != 4
        || k.len() != 4
        || v != k
        || p.len() != 4
        || q.iter().chain(k).chain(p).any(|n| *n <= 0)
        || q[0] != k[0]
        || q[3] != k[3]
        || q[1] % k[1] != 0
        || q[..3] != p[..3]
        || qo < 0
        || ko < 0
        || qo.checked_add(q[2]).is_none()
        || ko.checked_add(k[2]).is_none()
        || window.is_some_and(|n| n <= 0)
        || floor.is_some_and(|n| n <= 0)
        || !alpha.is_finite()
        || op.outputs.get(0).unwrap().shape() != q
        || op.outputs.get(0).unwrap().dtype() != WorkspaceDtype::Float32
    {
        return Err(invalid());
    }
    if op
        .inputs
        .iter()
        .any(|x| x.dtype() != WorkspaceDtype::Float32)
    {
        return Ok(None);
    }
    Ok(Some(RelativeGeometry {
        q: q.try_into().map_err(|_| invalid())?,
        k: k.try_into().map_err(|_| invalid())?,
        repeated: q[1] != k[1],
        scaled: window.is_none() && floor.is_some(),
        windowed: window.is_some(),
    }))
}

fn relative(
    op: WorkspaceOperationView<'_>,
    a: NativeAllocationFacts,
    sink: &mut Emitter<'_>,
) -> FactResult<Option<WorkspaceOperationFacts>> {
    let Some(g) = relative_geometry(op)? else { return Ok(None); };
    let q = &g.q;
    let k = &g.k;
    let [b, h, n, d] = g.q;
    let keys = k[2];
    let grid = mul(n as u64, keys as u64)?;
    let score_count = mul(mul(b as u64, h as u64)?, grid)?;
    let r = |n| capacity(a, n);
    let boolean = |n: u64| buffer_capacity(a, n.max(1));
    let query = r(count(q)?)?;
    let score = r(score_count)?;
    let scalar = r(1)?;
    let coordinates = add(add(r(n as u64)?, r(keys as u64)?)?, r(grid)?)?;
    // The outer causal/window mask and RelativeAttentionKernel::bias each
    // construct their own exact I32 coordinates and distances.
    let mut total = add(mul(2, coordinates)?, add(boolean(grid)?, scalar)?)?;
    if g.windowed {
        total = add(total, add(mul(2, boolean(grid)?)?, scalar)?)?;
    }
    if h != k[1] {
        // Per K/V: rank-inserting reshape can copy, followed by a broadcast
        // view and a head-joining reshape that can materialize repeated heads.
        total = add(
            total,
            add(
                mul(2, r(count(k)?)?)?,
                mul(2, r(count(&[b, h, keys, d])?)?)?,
            )?,
        )?;
    }
    let scaled = g.scaled;
    if scaled {
        // I32 positions/add, F32 cast/divide/max/log/multiply/add, reshape, five
        // scalars, and query multiplication with possible input promotion.
        total = add(
            total,
            add(add(mul(9, r(n as u64)?)?, mul(5, scalar)?)?, mul(2, query)?)?,
        )?;
    }
    // Clip distances with two integer endpoints. GatherAxis reads strided
    // profile rows directly; the broadcast indices do not replicate profiles.
    total = add(total, add(mul(2, r(grid)?)?, mul(2, scalar)?)?)?;
    total = add(total, score)?;
    // Bias validity (ge/lt/and), two integer bounds, and zeroing invalid
    // distances with a possible profile-result promotion and scalar cast.
    total = add(
        total,
        add(
            add(mul(3, boolean(grid)?)?, mul(4, scalar)?)?,
            mul(2, score)?,
        )?,
    )?;
    if scaled {
        total = add(total, mul(2, score)?)?;
    }
    // Query 1/D scaling, QK product, bias addition and final causal masking.
    total = add(total, add(mul(2, query)?, mul(2, scalar)?)?)?;
    total = add(total, matmul_cost(q, &[b, h, d, keys], a)?)?;
    total = add(total, add(mul(5, score)?, mul(2, scalar)?)?)?;
    total = add(total, score)?; // precise last-axis softmax output/contiguous copy
    total = add(total, matmul_cost(&[b, h, n, keys], &[b, h, keys, d], a)?)?;
    sink.output(Output::Allocate(query))?;
    sink.finish(total.checked_sub(query).ok_or_else(invalid)?, format_args!("MLX Metal relative-profile attention: exact integer masks, optional logarithmic query/bias scaling, direct profile gather, GQA reshape copies, two selected dense products and precise softmax; all casts and child buffers retained through completion; page={} with bounded oversized reuse; no disjoint host payload",a.page_size())).map(Some)
}

/// The native worker casts positions to F32, multiplies stored F32
/// frequencies, and applies cosine/sine before reshaping. Reuse its exact
/// geometry validation; this scalar fact grants no allocation or submission.
pub(super) fn rotary_representation(op: WorkspaceOperationView<'_>, output: usize)
    -> Option<WorkspaceRepresentation> {
    let spec = match op.kind {
        WorkspaceOperationKindView::MultiAxisRotary(spec)
        | WorkspaceOperationKindView::PreparedMultiAxisRotary(spec) => spec,
        _ => return None,
    };
    if output >= 2 { return None; }
    rotary_geometry(op, spec).ok()??;
    Some(WorkspaceRepresentation::new(WorkspaceFloatingType::Float32, false))
}

fn rotary_geometry(op: WorkspaceOperationView<'_>, spec: MultiAxisRotarySpecRef<'_>)
    -> FactResult<Option<(u64, i32)>> {
    let dimensions = spec.dimensions()?;
    if op.inputs.len() != 1 || op.outputs.len() != 2 {
        return Err(invalid());
    }
    let input = op.inputs.get(0).unwrap();
    let shape = input.shape();
    if shape.len() < 2 || shape.last().copied() != Some(spec.axes.len() as i32) {
        return Err(invalid());
    }
    let rows = count(&shape[..shape.len() - 1])?;
    i32::try_from(rows).map_err(|_| invalid())?;
    let prefix = &shape[..shape.len() - 1];
    if op.outputs.iter().any(|x| {
        !x.shape()
            .iter()
            .copied()
            .eq(prefix.iter().copied().chain(std::iter::once(dimensions)))
            || x.dtype() != WorkspaceDtype::Float32
    }) {
        return Err(invalid());
    }
    if !matches!(
        input.dtype(),
        WorkspaceDtype::Int32 | WorkspaceDtype::Uint32
    ) {
        return Ok(None);
    }
    Ok(Some((rows, dimensions)))
}

fn rotary(
    op: WorkspaceOperationView<'_>,
    a: NativeAllocationFacts,
    spec: MultiAxisRotarySpecRef<'_>,
    sink: &mut Emitter<'_>,
) -> FactResult<Option<WorkspaceOperationFacts>> {
    let Some((rows, dimensions)) = rotary_geometry(op, spec)? else { return Ok(None); };
    let input = op.inputs.get(0).expect("validated rotary geometry");
    let full = capacity(a, mul(rows, dimensions as u64)?)?;
    let half = dimensions as u64 / 2;
    // Input flattening can copy, then all integer positions widen once to I64.
    let mut total = add(capacity(a, input.elements()?)?, wide(a, input.elements()?)?)?;
    // Index-axis removal can reshape-copy a strided column. Signed I32 offsets
    // saturate via I64 add/min/max; U32 needs fewer operations and is bounded by
    // the same envelope. Include all native integer endpoint scalars.
    let column = add(mul(5, wide(a, rows)?)?, mul(4, wide(a, 1)?)?)?;
    match spec.layout {
        MultiAxisRotaryLayout::IndependentAxes | MultiAxisRotaryLayout::SplitHalves => {
            for axis in spec.axes {
                let frequencies = axis.dimensions as u64 / 2;
                total = add(total, add(column, capacity(a, rows)?)?)?;
                // From-slice host/shared native storage and its stream copy;
                // the Rust frequency vector is a distinct host bound below.
                total = add(total, mul(2, capacity(a, frequencies)?)?)?;
                total = add(total, capacity(a, mul(rows, frequencies)?)?)?;
                if spec.layout == MultiAxisRotaryLayout::IndependentAxes {
                    total = add(total, capacity(a, mul(rows, axis.dimensions as u64)?)?)?;
                }
            }
            if spec.layout == MultiAxisRotaryLayout::SplitHalves {
                total = add(total, capacity(a, mul(rows, half)?)?)?;
            }
        }
        MultiAxisRotaryLayout::RoundRobinSections => {
            // Only the selected integer column graph is constructed per global
            // frequency; unused independent-axis frequency graphs are omitted.
            total = add(total, mul(half, column)?)?;
            total = add(total, wide(a, mul(rows, half)?)?)?; // concatenate selected I64 columns
            total = add(total, mul(2, capacity(a, half)?)?)?; // frequency source and stream copy
            total = add(total, mul(2, capacity(a, mul(rows, half)?)?)?)?; // cast and product
        }
    }
    // Joined/repeated angles, cos and sin, and two potential output reshapes.
    total = add(total, mul(5, full)?)?;
    sink.output(Output::Allocate(full))?;
    sink.output(Output::Allocate(full))?;
    sink.finish(total.checked_sub(mul(2,full)?).ok_or_else(invalid)?, format_args!("MLX Metal multi-axis rotary {:?}: signed saturation in widened integer coordinates, frequency construction and native stream copies, selected layout concatenations, F32 angles/cos/sin and possible reshape copies; all child buffers retained through completion; page={} with bounded oversized reuse; Rust frequency payload priced separately",spec.layout,a.page_size())).map(Some)
}

pub(super) fn ordinary_rotary_error(
    spec: &MultiAxisRotarySpec,
    error: MlxWorkspaceFactError,
) -> Error {
    if matches!(error.cause(), MlxWorkspaceFactCause::RotaryTable(_)) {
        // Only on ordinary failure: the same closed validator formats the
        // original owned policy's existing diagnostic, with no source clone.
        // The fixed companion never enters this ordinary diagnostic bridge.
        if let Err(original) = spec.dimensions() {
            return original;
        }
    }
    error.ordinary()
}

pub(super) fn rotary_host_bytes(spec: &MultiAxisRotarySpec) -> Result<u64, Error> {
    rotary_host_bytes_fixed(spec.as_ref()).map_err(|error| ordinary_rotary_error(spec, error))
}

pub(super) fn rotary_host_bytes_fixed(spec: MultiAxisRotarySpecRef<'_>) -> FactResult<u64> {
    let dimensions = spec.dimensions()?;
    // An exact-sized frequency vector lives until the end of its iteration;
    // from_slice owns a native copy before the next vector is constructed.
    let frequencies = match spec.layout {
        MultiAxisRotaryLayout::RoundRobinSections => dimensions / 2,
        _ => spec
            .axes
            .iter()
            .map(|a| a.dimensions / 2)
            .max()
            .ok_or_else(invalid)?,
    };
    mul(frequencies as u64, 4)
}

#[cfg(test)]
mod tests;
