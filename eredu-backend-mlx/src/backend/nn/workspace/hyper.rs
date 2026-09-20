//! Native multi-stream residual mixing. Coefficients crossing a sublayer or
//! observer boundary retain their own backing; all child buffers are charged
//! through completion. Sinkhorn work scales with the selected pass count.

use super::{
    matrix::{einsum_matmul_cost_fixed as einsum_matmul_cost, matmul_cost_fixed as matmul_cost},
    reduction::{capacity_fixed as capacity, sum_cost_fixed as sum_cost},
    *,
};

use super::facts::{self, Emitter, FactResult, Output, add, mul};

fn invalid() -> MlxWorkspaceFactError {
    MlxWorkspaceFactError::descriptor("invalid Metal multi-stream workspace descriptor")
}
fn product(shape: &[i32]) -> FactResult<u64> {
    shape
        .iter()
        .try_fold(1, |n, &d| mul(n, u64::try_from(d).map_err(|_| invalid())?))
}
fn times(a: i32, b: i32) -> FactResult<i32> {
    a.checked_mul(b).ok_or_else(invalid)
}
fn positive(epsilon: f32) -> FactResult<()> {
    if epsilon.is_finite() && epsilon > 0. {
        Ok(())
    } else {
        Err(invalid())
    }
}
fn input(op: &WorkspaceOperationView<'_>, at: usize, shape: &[i32]) -> FactResult<()> {
    if op.inputs.get(at).is_some_and(|l| l.shape() == shape) {
        Ok(())
    } else {
        Err(invalid())
    }
}
fn output(op: &WorkspaceOperationView<'_>, at: usize, shape: &[i32]) -> FactResult<()> {
    if op
        .outputs
        .get(at)
        .is_some_and(|l| l.shape() == shape && l.dtype() == WorkspaceDtype::Float32)
    {
        Ok(())
    } else {
        Err(invalid())
    }
}
fn residual(op: &WorkspaceOperationView<'_>, at: usize) -> FactResult<[i32; 4]> {
    let shape: [i32; 4] = op
        .inputs
        .get(at)
        .ok_or_else(invalid)?
        .shape()
        .try_into()
        .map_err(|_| invalid())?;
    if shape[0] < 0 || shape[1] < 0 || shape[2] <= 0 || shape[3] <= 0 {
        return Err(invalid());
    }
    Ok(shape)
}
fn parameters(
    op: &WorkspaceOperationView<'_>,
    shape: [i32; 4],
    rows: i32,
    scales: i32,
) -> FactResult<()> {
    if op.inputs.len() != 4 {
        return Err(invalid());
    }
    let width = times(shape[2], shape[3])?;
    input(op, 1, &[rows, width])?;
    input(op, 2, &[rows])?;
    input(op, 3, &[scales])
}
fn floating(op: &WorkspaceOperationView<'_>) -> bool {
    op.inputs
        .iter()
        .all(|l| l.dtype() == WorkspaceDtype::Float32)
}
fn finish(
    op: &WorkspaceOperationView<'_>,
    a: NativeAllocationFacts,
    total: u64,
    detail: &'static str,
    sink: &mut Emitter<'_>,
) -> FactResult<Option<WorkspaceOperationFacts>> {
    // The old collection computed EVERY output capacity before summing them.
    // Preserve that error order without a dynamic metadata destination.
    for output in op.outputs.iter() {
        capacity(a, output.elements()?)?;
    }
    let retained = op
        .outputs
        .iter()
        .try_fold(0, |n, output| add(n, capacity(a, output.elements()?)?))?;
    let scratch = total.checked_sub(retained).ok_or_else(invalid)?;
    for output in op.outputs.iter() {
        sink.output(Output::Allocate(capacity(a, output.elements()?)?))?;
    }
    sink.finish(scratch, format_args!("MLX Metal multi-stream {detail}: selected F32 reductions/products and coefficient arithmetic, all child buffers through completion including casts and stride-dependent reshape copies; page={} with bounded oversized reuse; native payload only, no disjoint host numerical vectors", a.page_size())).map(Some)
}

/// Hyper preparation always widens before its unfused weightless RMS equation.
fn prepare(a: NativeAllocationFacts, shape: [i32; 4]) -> FactResult<u64> {
    let [b, t, s, h] = shape;
    // Dense frontend flattening uses an I32 leading extent.
    let rows = times(b, t)? as u64;
    let width = times(s, h)? as u64;
    let n = mul(rows, width)?;
    // Input widening/flattening, square and final product; mean sum, division,
    // epsilon add and rsqrt. Mean divisor and epsilon are native scalar arrays.
    add(
        add(mul(4, capacity(a, n)?)?, mul(3, capacity(a, rows)?)?)?,
        add(sum_cost(a, n, rows, width)?, mul(2, capacity(a, 1)?)?)?,
    )
}

pub(super) fn operation_bound(
    op: &WorkspaceOperation,
    a: NativeAllocationFacts,
) -> Result<Option<WorkspaceOperationBound>, Error> {
    facts::ordinary(|sink| emit(op.as_view(), a, sink))
}

pub(super) enum Geometry<'a> {
    Collapse {
        shape: [i32; 4],
        width: i32,
        spec: &'a eredu_nn::HyperConnectionSpec,
        epsilon: f32,
    },
    Expand([i32; 4]),
    HeadCoefficients {
        shape: [i32; 4],
        spec: &'a eredu_nn::HyperHeadSpec,
    },
    HeadSum([i32; 4]),
}

// One allocation-free validator is shared by tensor facts and native receipts.
// It borrows the actual neutral declaration; no synthetic allocation facts or
// reconstructed parameter/source owner are used to obtain geometry.
pub(super) fn geometry(op: WorkspaceOperationView<'_>) -> FactResult<Option<Geometry<'_>>> {
    let op = &op;
    use WorkspaceOperationKindView as K;
    let result = match op.kind {
        K::HyperCollapse(spec, epsilon) => {
            spec.validate_fixed()?;
            positive(epsilon)?;
            let shape = residual(op, 0)?;
            let [b, t, s, h] = shape;
            if s != spec.streams || h != spec.hidden_size || op.outputs.len() != 4 {
                return Err(invalid());
            }
            let width = times(s.checked_add(2).ok_or_else(invalid)?, s)?;
            times(b, t)?;
            parameters(op, shape, width, 3)?;
            output(op, 0, &[b, t, h])?;
            output(op, 1, &[b, t, s])?;
            output(op, 2, &[b, t, s])?;
            output(op, 3, &[b, t, s, s])?;
            Geometry::Collapse {
                shape,
                width,
                spec,
                epsilon,
            }
        }
        K::HyperExpand => {
            let shape = residual(op, 1)?;
            let [b, t, s, h] = shape;
            if op.inputs.len() != 4 || op.outputs.len() != 1 {
                return Err(invalid());
            }
            input(op, 0, &[b, t, h])?;
            input(op, 2, &[b, t, s])?;
            input(op, 3, &[b, t, s, s])?;
            output(op, 0, &shape)?;
            Geometry::Expand(shape)
        }
        K::HyperHeadCoefficients(spec) => {
            spec.validate_fixed()?;
            let shape = residual(op, 0)?;
            let [b, t, s, h] = shape;
            if s != spec.streams || h != spec.hidden_size || op.outputs.len() != 1 {
                return Err(invalid());
            }
            parameters(op, shape, s, 1)?;
            output(op, 0, &[b, t, s])?;
            Geometry::HeadCoefficients { shape, spec }
        }
        K::HyperHeadSum => {
            let shape = residual(op, 0)?;
            let [b, t, s, h] = shape;
            if op.inputs.len() != 2 || op.outputs.len() != 1 {
                return Err(invalid());
            }
            input(op, 1, &[b, t, s])?;
            output(op, 0, &[b, t, h])?;
            Geometry::HeadSum(shape)
        }
        _ => return Ok(None),
    };
    if !floating(op) {
        return Ok(None);
    }
    Ok(Some(result))
}

pub(super) fn emit(
    op: WorkspaceOperationView<'_>,
    a: NativeAllocationFacts,
    sink: &mut Emitter<'_>,
) -> FactResult<Option<WorkspaceOperationFacts>> {
    let Some(geometry) = geometry(op)? else {
        return Ok(None);
    };
    let op = &op;
    match geometry {
        Geometry::Collapse {
            shape, width, spec, ..
        } => {
            let [b, t, s, h] = shape;
            let rows = mul(b as u64, t as u64)?;
            let vectors = mul(rows, s as u64)?;
            let matrices = mul(vectors, s as u64)?;
            let vector = capacity(a, vectors)?;
            let matrix = capacity(a, matrices)?;
            let scalar = capacity(a, 1)?;
            let mut total = prepare(a, shape)?;
            total = add(
                total,
                matmul_cost(&[b, t, times(s, h)?], &[times(s, h)?, width], a)?,
            )?;
            // Split's explicit mix cast/flatten, scale/base casts, and three
            // scalar index-axis removals (which can reshape-copy strided data).
            total = add(
                total,
                add(
                    mul(2, capacity(a, mul(rows, width as u64)?)?)?,
                    add(
                        add(capacity(a, width as u64)?, capacity(a, 3)?)?,
                        mul(3, scalar)?,
                    )?,
                )?,
            )?;
            // Each pre/post vector: multiply, base add, sigmoid, epsilon add
            // or factor-two multiply, and final reshape. Two scalar constants.
            total = add(total, add(mul(10, vector)?, mul(2, scalar)?)?)?;
            // Matrix logits multiply/add/reshape, precise final-axis softmax,
            // epsilon add and final output reshape.
            total = add(total, add(mul(6, matrix)?, scalar)?)?;
            let pass = add(
                sum_cost(a, matrices, vectors, s as u64)?,
                add(add(vector, scalar)?, matrix)?,
            )?;
            let passes = mul(
                2,
                u64::try_from(spec.sinkhorn_iterations).map_err(|_| invalid())?,
            )?
            .checked_sub(1)
            .ok_or_else(invalid)?;
            total = add(total, mul(passes, pass)?)?;
            total = add(
                total,
                einsum_matmul_cost(&[b, t, 1, s], &[b, t, s, h], &[b, t, h], a)?,
            )?;
            total = add(total, capacity(a, product(&[b, t, h])?)?)?; // restore residual dtype
            finish(op, a, total, "collapse and Sinkhorn", sink)
        }
        Geometry::Expand(shape) => {
            let [b, t, s, h] = shape;
            // Coefficient expansion/promotion; sublayer cast/index reshape;
            // residual cast, injected product, addition and final dtype cast.
            let mut total = add(
                mul(2, capacity(a, product(&[b, t, s])?)?)?,
                add(
                    mul(2, capacity(a, product(&[b, t, h])?)?)?,
                    mul(4, capacity(a, product(&shape)?)?)?,
                )?,
            )?;
            total = add(
                total,
                einsum_matmul_cost(&[b, t, s, s], &[b, t, s, h], &shape, a)?,
            )?;
            finish(op, a, total, "residual expansion", sink)
        }
        Geometry::HeadCoefficients { shape, .. } => {
            let [b, t, s, h] = shape;
            let vector = capacity(a, product(&[b, t, s])?)?;
            let mut total = prepare(a, shape)?;
            total = add(
                total,
                matmul_cost(&[b, t, times(s, h)?], &[times(s, h)?, s], a)?,
            )?;
            // Scale/base promotions, multiply/add/sigmoid/epsilon-add outputs,
            // and the separately constructed epsilon scalar.
            total = add(
                total,
                add(
                    mul(4, vector)?,
                    add(capacity(a, s as u64)?, mul(2, capacity(a, 1)?)?)?,
                )?,
            )?;
            finish(op, a, total, "final stream coefficients", sink)
        }
        Geometry::HeadSum(shape) => {
            let [b, t, s, h] = shape;
            let n = product(&shape)?;
            let out = product(&[b, t, h])?;
            let total = add(
                add(
                    mul(2, capacity(a, n)?)?,
                    mul(2, capacity(a, product(&[b, t, s])?)?)?,
                )?,
                add(sum_cost(a, n, out, s as u64)?, capacity(a, out)?)?,
            )?;
            finish(op, a, total, "final weighted stream sum", sink)
        }
    }
}

mod structure;
pub(super) use structure::inspect as structure;

#[cfg(test)]
mod tests;
