//! Dense products in the vendored Metal implementation. Optional actual source
//! representation refines the selected projection; unknown inputs cover every
//! compatible copy, cast and GEMM dispatch, including NAX.

use super::facts::{self, Aliases, Emitter, FactResult, Output, add, mul};
use super::{reduction::capacity_fixed as capacity, *};
use eredu_checkpoint::LinearFormat;

/// The same native product cost used inside composite attention mechanisms.
pub(super) fn matmul_cost(
    left: &[i32],
    right: &[i32],
    allocation: NativeAllocationFacts,
) -> Result<u64, Error> {
    matmul_cost_fixed(left, right, allocation).map_err(MlxWorkspaceFactError::ordinary)
}

pub(super) fn matmul_cost_fixed(
    left: &[i32],
    right: &[i32],
    allocation: NativeAllocationFacts,
) -> FactResult<u64> {
    Geometry::new(left, right)?.matmul_cost(allocation)
}

/// Two-operand einsum lowering without diagonals or lone-axis reductions.
/// Callers supply the already lowered batched matrix shapes. MLX transposes
/// and reshapes both inputs and the result around the selected dense product.
pub(super) fn einsum_matmul_cost(
    left: &[i32],
    right: &[i32],
    result: &[i32],
    allocation: NativeAllocationFacts,
) -> Result<u64, Error> {
    einsum_matmul_cost_fixed(left, right, result, allocation)
        .map_err(MlxWorkspaceFactError::ordinary)
}

pub(super) fn einsum_matmul_cost_fixed(
    left: &[i32],
    right: &[i32],
    result: &[i32],
    allocation: NativeAllocationFacts,
) -> FactResult<u64> {
    add(
        matmul_cost_fixed(left, right, allocation)?,
        add(
            add(
                capacity(allocation, elements(left)?)?,
                capacity(allocation, elements(right)?)?,
            )?,
            capacity(allocation, elements(result)?)?,
        )?,
    )
}

pub(super) fn operation_bound(
    operation: &WorkspaceOperation,
    allocation: NativeAllocationFacts,
) -> Result<Option<WorkspaceOperationBound>, Error> {
    facts::ordinary(|sink| emit(operation.as_view(), allocation, sink))
}

// The kind excludes general matmul, groups and column projection. Other or
// unknown sources keep the full generic union, even at the same logical shape.
fn selected_bf16_row(operation: WorkspaceOperationView<'_>) -> bool {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        let Some(input) = operation.inputs.get(0) else {
            return false;
        };
        let Some(weight) = operation.inputs.get(1) else {
            return false;
        };
        let (Some(input_rep), Some(weight_rep)) = (input.representation(), weight.representation())
        else {
            return false;
        };
        let Some(&width) = input.shape().last() else {
            return false;
        };
        let Some(elements) = input.elements().ok() else {
            return false;
        };
        input_rep.dtype() == WorkspaceFloatingType::Bfloat16
            && weight_rep.dtype() == WorkspaceFloatingType::Bfloat16
            && weight_rep.row_contiguous()
            && super::super::matrix::bf16_row_width_supported(width)
            && input.shape().len()
                <= crate::backend::managed_memory::bf16_projection_kernel::OUTPUT_DIMENSIONS
            && elements > 0
            && elements / width as u64 <= i32::MAX as u64
            && weight.shape().len() == 2
            && weight.shape()[0] > 0
            && weight.shape()[1] == width
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = operation;
        false
    }
}

// The mixed source still needs the full BF16-to-F32 weight cast. For a
// nondegenerate row-contiguous [N,K] matrix, transpose produces strides [1,K].
// GPU AsType uses CopyType::Vector and preserves those strides; Metal matmul
// check_transpose accepts them without a second weight compaction. Singleton
// dimensions stay unknown because their unused strides are not constrained by
// row contiguity. Other operation kinds cannot select this proof.
fn selected_mixed_row_weight(operation: WorkspaceOperationView<'_>) -> bool {
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        let (Some(input), Some(weight)) = (operation.inputs.get(0), operation.inputs.get(1)) else {
            return false;
        };
        let (Some(input_rep), Some(weight_rep)) = (input.representation(), weight.representation())
        else {
            return false;
        };
        input_rep.dtype() == WorkspaceFloatingType::Float32
            && weight_rep.dtype() == WorkspaceFloatingType::Bfloat16
            && weight_rep.row_contiguous()
            && weight.shape().len() == 2
            && weight.shape()[0] > 1
            && weight.shape()[1] > 1
    }
    #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
    {
        let _ = operation;
        false
    }
}

pub(super) fn emit(
    operation: WorkspaceOperationView<'_>,
    allocation: NativeAllocationFacts,
    sink: &mut Emitter<'_>,
) -> FactResult<Option<WorkspaceOperationFacts>> {
    let (linear, constructed) = match &operation.kind {
        WorkspaceOperationKindView::Matmul => (false, false),
        WorkspaceOperationKindView::DenseLinear => (true, false),
        WorkspaceOperationKindView::Projection(format)
            if format.encoding() == LinearFormat::Dense =>
        {
            format.validate_fixed()?;
            (true, true)
        }
        _ => return Ok(None),
    };
    if operation.outputs.len() != 1
        || !(operation.inputs.len() == 2 || linear && operation.inputs.len() == 3)
    {
        return Err(MlxWorkspaceFactError::descriptor(
            "invalid Metal dense product descriptor",
        ));
    }
    if operation
        .inputs
        .iter()
        .any(|layout| layout.dtype() != WorkspaceDtype::Float32)
    {
        return Ok(None);
    }
    let left = operation.inputs.get(0).expect("checked two inputs").shape();
    let weight = operation.inputs.get(1).expect("checked two inputs").shape();
    let transposed_weight;
    let right = if linear {
        if weight.len() != 2
            || operation
                .inputs
                .get(2)
                .is_some_and(|bias| bias.shape() != [weight[0]])
        {
            return Err(MlxWorkspaceFactError::descriptor(
                "invalid Metal dense projection weight/bias",
            ));
        }
        transposed_weight = [weight[1], weight[0]];
        &transposed_weight[..]
    } else {
        weight
    };
    let geometry = Geometry::new(left, right)?;
    let output = operation.outputs.get(0).expect("checked one output");
    if !geometry
        .output
        .dimensions()
        .eq(output.shape().iter().copied())
        || output.dtype() != WorkspaceDtype::Float32
    {
        return Err(MlxWorkspaceFactError::descriptor(
            "Metal dense product output geometry differs",
        ));
    }
    let output_capacity = capacity(allocation, output.elements()?)?;
    let selected_row = constructed && selected_bf16_row(operation);
    let mixed_row_weight = constructed && selected_mixed_row_weight(operation);
    let mut total = if selected_row {
        // The completed source proves BF16 row-major weight, and propagation
        // proves BF16 input. The ordinary worker therefore selects its custom
        // row kernel: no dtype cast or weight compaction occurs. Keep the
        // possible input compaction, result/reshape bound and actual I32 group
        // seed. F32-sized capacities remain conservative for these BF16 values.
        add(
            capacity(allocation, geometry.left_elements)?,
            add(mul(2, output_capacity)?, capacity(allocation, 1)?)?,
        )?
    } else {
        geometry.matmul_cost_with_compaction(allocation, !mixed_row_weight)?
    };
    let biased = operation.inputs.len() == 3;
    if constructed {
        // PhysicalLinear may select a complete-F32-accumulation BF16 row kernel.
        // Dense tied embedding readout selects this same ordinary worker.
        // The custom kernel has one contiguous copy per input, one scalar group
        // index, one result, and a possible final reshape copy. Cover both paths.
        if !selected_row && geometry.k > 0 && geometry.k % 32 == 0 && output.elements()? > 0 {
            let custom = add(
                add(
                    capacity(allocation, geometry.left_elements)?,
                    capacity(allocation, geometry.right_elements)?,
                )?,
                add(mul(2, output_capacity)?, capacity(allocation, 1)?)?,
            )?;
            total = total.max(custom);
        }
        if biased {
            // Constructed projections add bias after matmul/custom projection;
            // possible casts of result and bias precede the binary result.
            total = add(
                total,
                add(mul(2, output_capacity)?, capacity(allocation, geometry.n)?)?,
            )?;
        }
    } else if biased {
        // Tensor::linear uses addmm: the bias is read directly with its strides,
        // after a possible dtype cast. A vector input can reshape the bias too.
        total = add(total, capacity(allocation, geometry.n)?)?;
        if left.len() == 1 {
            total = add(total, output_capacity)?;
        }
    }
    // Matmul results never retain the input allocation. Addmm with zero K and
    // the separate bias addition can retain bias, so preserve that possible root.
    let storage = if biased {
        Output::AllocateOrAliasInputs {
            bytes: output_capacity,
            inputs: Aliases::Slice(&[2]),
        }
    } else {
        Output::Allocate(output_capacity)
    };
    sink.output(storage)?;
    if selected_row {
        return sink.finish(total - output_capacity, format_args!("actual BF16 input and completed row-contiguous BF16 weight select the shared Metal row-projection worker; possible input compaction, scalar group seed, result/reshape and separate bias retained; no weight cast or compaction; page={} with bounded oversized reuse", allocation.page_size())).map(Some);
    }
    if mixed_row_weight {
        return sink.finish(total - output_capacity, format_args!("actual F32 input and completed row-contiguous BF16 matrix weight: transpose and vector AsType preserve a Metal-accepted matrix layout; full F32 casts, input/reshape/result/split-K and existing custom/bias ceiling retained, with no second weight compaction; page={} with bounded oversized reuse", allocation.page_size())).map(Some);
    }
    sink.finish(total - output_capacity, format_args!("vendored MLX Metal dense product: at-most-F32 casts, stride-dependent input/reshape copies, result and one geometry-selected split-K accumulator; maximum of SIMD and NAX dispatches, plus custom BF16 projection/bias when selected; unbatched weights are not replicated by frontend flattening; page={} with bounded oversized reuse; active tensor buffers only", allocation.page_size())).map(Some)
}

struct Geometry<'a> {
    m: u64,
    n: u64,
    k: u64,
    batches: u64,
    left_elements: u64,
    right_elements: u64,
    flatten_left: bool,
    batched_right: bool,
    output: WorkspaceMatmulShape<'a>,
}
fn elements(shape: &[i32]) -> FactResult<u64> {
    shape.iter().try_fold(1, |count, &n| mul(count, n as u64))
}
impl<'a> Geometry<'a> {
    fn new(left: &'a [i32], right: &'a [i32]) -> FactResult<Self> {
        let output = WorkspaceMatmulShape::new(left, right).map_err(|cause| {
            MlxWorkspaceFactError::descriptor(match cause {
                WorkspaceShapeError::MatmulRank => "Metal matmul requires ranked inputs",
                WorkspaceShapeError::MatmulReduction => "Metal matmul reduction extents differ",
                WorkspaceShapeError::IncompatibleBroadcast => "Metal matmul batch extents differ",
                WorkspaceShapeError::DestinationRank { .. } => {
                    "workspace shape destination rank mismatch"
                }
            })
        })?;
        let m = if left.len() == 1 {
            1
        } else {
            left[left.len() - 2] as u64
        };
        let k = left[left.len() - 1] as u64;
        let n = if right.len() == 1 {
            1
        } else {
            right[right.len() - 1] as u64
        };
        let batch_rank = left
            .len()
            .saturating_sub(2)
            .max(right.len().saturating_sub(2));
        // Preserve forward multiplication/error order after the shared trailing
        // compatibility pass; zero does not bypass an earlier multiplication.
        let batches = output
            .dimensions()
            .take(batch_rank)
            .try_fold(1, |count, extent| mul(count, extent as u64))?;
        Ok(Self {
            m,
            n,
            k,
            batches,
            left_elements: elements(left)?,
            right_elements: elements(right)?,
            flatten_left: left.len() > 2 && right.len() <= 2,
            batched_right: right.len() > 2,
            output,
        })
    }

    fn output_elements(&self) -> FactResult<u64> {
        self.output
            .dimensions()
            .try_fold(1, |count, extent| mul(count, extent as u64))
    }

    fn matmul_cost(&self, allocation: NativeAllocationFacts) -> FactResult<u64> {
        self.matmul_cost_with_compaction(allocation, true)
    }

    fn matmul_cost_with_compaction(
        &self,
        allocation: NativeAllocationFacts,
        copy_right: bool,
    ) -> FactResult<u64> {
        let result = capacity(allocation, self.output_elements()?)?;
        // Promotion precedes broadcasting. Each original operand can be cast
        // once; each GPU operand can then need one contiguous copy. The exact
        // selected mixed projection may exclude only the right copy below.
        let mut total = add(
            result,
            add(
                capacity(allocation, self.left_elements)?,
                capacity(allocation, self.right_elements)?,
            )?,
        )?;
        let (a_copy, b_copy) = if self.batched_right {
            (
                mul(mul(self.batches, self.m)?, self.k)?,
                mul(mul(self.batches, self.k)?, self.n)?,
            )
        } else {
            (self.left_elements, self.right_elements)
        };
        total = add(
            total,
            add(
                capacity(allocation, a_copy)?,
                if copy_right {
                    capacity(allocation, b_copy)?
                } else {
                    0
                },
            )?,
        )?;
        if self.flatten_left {
            // Flattening a strided leading batch may copy A; B remains a single
            // matrix. Include the final unflatten result's possible copy too.
            total = add(
                total,
                add(capacity(allocation, self.left_elements)?, result)?,
            )?;
        }
        if self.k == 0 || self.output_elements()? == 0 {
            return add(total, capacity(allocation, 1)?);
        }
        let mut partitions = if self.batches == 1 {
            split_partitions(self.m, self.n, self.k)?
        } else {
            0
        };
        if self.batches > 1 {
            // Native stride collapse can turn broadcast B and contiguous A into
            // one taller GEMM even when frontend B had explicit batch axes.
            partitions = partitions.max(split_partitions(
                mul(self.batches, self.m)?,
                self.n,
                self.k,
            )?);
        }
        if partitions > 0 {
            total = add(
                total,
                capacity(allocation, mul(partitions, self.output_elements()?)?)?,
            )?;
        }
        Ok(total)
    }
}

/// Number of F32 partials per result element. SIMD and NAX paths are mutually
/// exclusive, and neither GEMV nor ordinary batched GEMM allocates partials.
fn split_partitions(m: u64, n: u64, k: u64) -> FactResult<u64> {
    if m.min(n) <= 1 {
        return Ok(0);
    }
    steel_split_partitions_fixed(m, n, k)
}

/// Direct Steel GEMM callers (including unfolded convolution) do not use the
/// tensor matmul frontend's GEMV shortcut for a one-row or one-column result.
pub(super) fn steel_split_partitions(m: u64, n: u64, k: u64) -> Result<u64, Error> {
    steel_split_partitions_fixed(m, n, k).map_err(MlxWorkspaceFactError::ordinary)
}

pub(super) fn steel_split_partitions_fixed(m: u64, n: u64, k: u64) -> FactResult<u64> {
    if m == 0 || n == 0 || k == 0 {
        return Ok(0);
    }
    let mut partitions = 0;
    let largest = m.max(n);
    if mul(m.div_ceil(16), n.div_ceil(16))? <= 2048 && k >= 128 && k >= largest {
        let ratio = (k / 16) / mul(m.div_ceil(32), n.div_ceil(32))?;
        partitions = ratio.clamp(2, 32).next_power_of_two();
    }
    if k >= mul(3, largest)? || (largest <= 1024 && k > mul(2, largest)?) {
        let size = match k {
            0..=1024 => k / 2,
            1025..=2048 => 1024,
            2049..=4096 => 2048,
            _ => 4096,
        };
        partitions = partitions.max(k.div_ceil(size));
    }
    Ok(partitions)
}

#[cfg(test)]
mod tests;
