//! Selected packed projections and row lookups. Packed storage is retained;
//! Metal kernels decode in registers rather than materializing a dense weight.

use super::facts::{self, add, buffer_capacity, mul, Emitter, FactResult, Output};
use super::{reduction::capacity_fixed as capacity, *};
use eredu_checkpoint::{BlockFp8ScaleEncoding, LinearFormat};
use eredu_nn::{LinearFormatSpec, LinearRowLayout};

pub(super) fn operation_bound(
    op: &WorkspaceOperation,
    allocation: MetalAllocationFacts,
) -> Result<Option<WorkspaceOperationBound>, Error> {
    facts::ordinary(|sink| emit(op.as_view(), allocation, sink))
}

pub(super) fn emit(
    op: WorkspaceOperationView<'_>,
    allocation: MetalAllocationFacts,
    sink: &mut Emitter<'_>,
) -> FactResult<Option<WorkspaceOperationFacts>> {
    if matches!(
        op.kind,
        WorkspaceOperationKindView::ProjectionPrepare(_)
            | WorkspaceOperationKindView::ProjectionFinish(_)
    ) {
        return observed_operation_bound(op, allocation, sink);
    }
    let (format, policy) = match &op.kind {
        WorkspaceOperationKindView::Projection(format) => (format, None),
        WorkspaceOperationKindView::Embedding(format, policy) => (format, Some(*policy)),
        _ => return Ok(None),
    };
    if format.encoding() == LinearFormat::Dense {
        return Ok(None);
    }
    format.validate_fixed()?;
    if let Some(policy) = policy {
        policy.validate_fixed()?;
    }
    // The ordinary embedding constructor does not select block-FP8. Grouped
    // and independently based row blocks have distinct operation descriptors.
    if matches!(format.encoding(), LinearFormat::E4M3BlockFp8(_))
        && (policy.is_some() || format.row_layout() != LinearRowLayout::Contiguous)
    {
        return Ok(None);
    }
    #[cfg(feature = "cuda")]
    if matches!(
        format.encoding(),
        LinearFormat::GgufIQuant { .. } | LinearFormat::E4M3BlockFp8(_)
    ) {
        return Ok(None);
    }
    let geometry = Geometry::new(op, format, policy.is_some())?;
    let Some(g) = geometry else {
        return Ok(None);
    };
    let r = |n| capacity(allocation, n);
    let output = r(op.outputs.get(0).unwrap().elements()?)?;
    let mut scratch = match (format.encoding(), policy) {
        (LinearFormat::Affine(config), None) => {
            let mut copies = add(
                mul(2, r(g.input.elements()?)?)?,
                g.packed_capacity(allocation)?,
            )?;
            // Frontend promotion and backend row compaction are distinct.
            for companion in g.companions.iter() {
                copies = add(
                    copies,
                    mul(2, buffer_capacity(allocation, companion.bytes()?)?)?,
                )?;
            }
            add(
                copies,
                quantized_split_scratch(
                    g.positions,
                    g.rows,
                    g.columns,
                    config.group_size as u64,
                    allocation,
                )?,
            )?
        }
        (LinearFormat::MxFp4, None) => {
            let copies = add(
                r(g.input.elements()?)?,
                add(
                    g.packed_capacity(allocation)?,
                    buffer_capacity(allocation, g.companions.get(0).unwrap().bytes()?)?,
                )?,
            )?;
            add(
                copies,
                quantized_split_scratch(g.positions, g.rows, g.columns, 32, allocation)?,
            )?
        }
        (LinearFormat::Affine(_) | LinearFormat::MxFp4, Some(policy)) => {
            let mut selected =
                buffer_capacity(allocation, mul(g.positions, g.weight.bytes()? / g.rows)?)?;
            for companion in g.companions.iter() {
                selected = add(
                    selected,
                    buffer_capacity(allocation, mul(g.positions, companion.bytes()? / g.rows)?)?,
                )?;
            }
            // Flatten IDs, gather packed rows/companions, possible contiguous
            // copies in the dequantizer, F32 restoration and final reshape.
            // Differently typed affine companions promote selected rows to
            // F32. Gathered rows and conversion outputs are contiguous, so
            // promotion replaces the possible compaction rather than adding
            // a third selected-row buffer for either companion.
            add(
                indexing::embedding_validation_cost_fixed(
                    g.positions,
                    op.outputs.get(0).unwrap().elements()?,
                    policy,
                    allocation,
                )?,
                add(r(g.positions)?, add(mul(2, selected)?, mul(2, output)?)?)?,
            )?
        }
        (LinearFormat::GgufIQuant { .. }, None) => {
            // Input flattening and one custom-kernel contiguous copy; direct
            // packed-weight compaction; one possible output reshape copy.
            add(
                mul(2, r(g.input.elements()?)?)?,
                add(g.packed_capacity(allocation)?, output)?,
            )?
        }
        (LinearFormat::GgufIQuant { .. }, Some(policy)) => add(
            indexing::embedding_validation_cost_fixed(
                g.positions,
                op.outputs.get(0).unwrap().elements()?,
                policy,
                allocation,
            )?,
            add(
                r(g.positions)?,
                add(g.packed_capacity(allocation)?, output)?,
            )?,
        )?,
        (LinearFormat::E4M3BlockFp8(config), None) => {
            let costs = fp8_costs(&g, config, allocation, output)?;
            add(
                add(costs.prepare_scratch, costs.finish_scratch)?,
                add(costs.values, costs.scales)?,
            )?
        }
        _ => return Ok(None),
    };
    if g.bias.is_some() {
        // PhysicalLinear adds output bias after the packed kernel: possible
        // result/bias casts and the final pointwise result.
        scratch = add(scratch, add(mul(2, output)?, r(g.rows)?)?)?;
    }
    sink.output(Output::Allocate(output))?;
    sink.finish(scratch, format_args!("selected MLX Metal packed {:?}: exact physical companions; direct packed kernels or selected-row dequantization; all compatible casts, compaction, shape copies and geometry-selected quantized split-K/reduction storage retained through completion; FP8 activation quantization and scale lookup included; no dense weight expansion; page={} with bounded oversized reuse; tensor buffers only, disjoint host facts and observation factories separately required",format.encoding(),allocation.page_size())).map(Some)
}

/// This partitions the existing full envelope before tracing. The two compact
/// activation roots are outputs of prepare and inputs of finish, never counted
/// again as unnamed prepare scratch. Weight-scale decoding stays in prepare
/// scratch through the enclosing span, matching the native retained local.
struct Fp8Costs {
    values: u64,
    scales: u64,
    prepare_scratch: u64,
    finish_scratch: u64,
}
fn fp8_costs(
    g: &Geometry<'_>,
    config: eredu_checkpoint::BlockFp8Format,
    a: MetalAllocationFacts,
    output: u64,
) -> FactResult<Fp8Costs> {
    let r = |n| capacity(a, n);
    let values = buffer_capacity(a, mul(g.positions, g.columns)?)?;
    let scales = r(mul(g.positions, g.columns.div_ceil(128))?)?;
    let weight_scales = g.companions.get(0).unwrap().elements()?;
    let mut prepare_scratch = add(mul(2, r(g.input.elements()?)?)?, r(weight_scales)?)?;
    if config.scale_encoding == BlockFp8ScaleEncoding::Ue8m0 {
        prepare_scratch = add(prepare_scratch, add(r(256)?, mul(2, r(weight_scales)?)?)?)?;
    }
    let finish_scratch = add(
        add(values, scales)?,
        add(g.packed_capacity(a)?, mul(2, output)?)?,
    )?;
    Ok(Fp8Costs {
        values,
        scales,
        prepare_scratch,
        finish_scratch,
    })
}

fn observed_operation_bound(
    op: WorkspaceOperationView<'_>,
    allocation: MetalAllocationFacts,
    sink: &mut Emitter<'_>,
) -> FactResult<Option<WorkspaceOperationFacts>> {
    let (format, prepare) = match &op.kind {
        WorkspaceOperationKindView::ProjectionPrepare(format) => (format, true),
        WorkspaceOperationKindView::ProjectionFinish(format) => (format, false),
        _ => unreachable!(),
    };
    let LinearFormat::E4M3BlockFp8(config) = format.encoding() else {
        return Ok(None);
    };
    format.validate_fixed()?;
    if format.row_layout() != LinearRowLayout::Contiguous {
        return Ok(None);
    }
    #[cfg(feature = "cuda")]
    {
        return Ok(None);
    }
    #[cfg(not(feature = "cuda"))]
    {
        let original_count = if prepare {
            op.inputs.len()
        } else {
            op.inputs.len().checked_sub(2).ok_or_else(|| {
                MlxWorkspaceFactError::descriptor("FP8 finish has no prepared roots")
            })?
        };
        if original_count < 3 {
            return invalid();
        }
        let shape = op.inputs.get(0).unwrap().shape();
        let weight_shape = op.inputs.get(1).unwrap().shape();
        if shape.is_empty() || weight_shape.len() != 2 {
            return invalid();
        }
        let score_prefix = &shape[..shape.len() - 1];
        let score_tail = [weight_shape[0]];
        // Validate score layout before packed geometry to preserve error precedence.
        // Keep that ordering through the shared two-slice byte-count kernel.
        let score_bytes =
            WorkspaceLayoutView::joined_bytes(score_prefix, &score_tail, WorkspaceDtype::Float32)?;
        let Some(g) = Geometry::from_parts(
            op.inputs.slice(0..original_count).unwrap(),
            None,
            format,
            false,
        )?
        else {
            return Ok(None);
        };
        let plan = eredu_nn::BlockFp8InputReconstructionPlan::new(g.input.shape())
            .map_err(MlxWorkspaceFactError::from)?;
        let values_shape = plan.values_shape();
        let scales_shape = plan.scales_shape();
        let prepared = [
            WorkspaceLayoutView::new(&values_shape, WorkspaceDtype::Uint8)?,
            WorkspaceLayoutView::new(&scales_shape, WorkspaceDtype::Float32)?,
        ];
        if (prepare && op.outputs != WorkspaceLayoutList::Views(&prepared))
            || (!prepare
                && (op.inputs.slice(original_count..op.inputs.len()).unwrap()
                    != WorkspaceLayoutList::Views(&prepared)
                    || op.outputs.len() != 1
                    || op.outputs.first().unwrap().dtype() != WorkspaceDtype::Float32
                    || !op
                        .outputs
                        .first()
                        .unwrap()
                        .shape()
                        .iter()
                        .eq(score_prefix.iter().chain(&score_tail))))
        {
            return invalid();
        }
        let output = buffer_capacity(allocation, score_bytes)?;
        let costs = fp8_costs(&g, config, allocation, output)?;
        let mut scratch_bytes = if prepare {
            costs.prepare_scratch
        } else {
            costs.finish_scratch
        };
        if !prepare && g.bias.is_some() {
            scratch_bytes = add(
                scratch_bytes,
                add(mul(2, output)?, capacity(allocation, g.rows)?)?,
            )?;
        }
        if prepare {
            sink.output(Output::Allocate(costs.values))?;
            sink.output(Output::Allocate(costs.scales))?;
        } else {
            sink.output(Output::Allocate(output))?;
        }
        sink.finish(scratch_bytes, format_args!("selected Metal block-FP8 {}: exact whole-projection envelope partition; compact activation roots counted once, decoded weight scales remain live through callback and finish, generated reconstruction separately traced; page={}", if prepare { "prepare" } else { "finish" }, allocation.page_size())).map(Some)
    }
}

struct Geometry<'a> {
    input: WorkspaceLayoutView<'a>,
    weight: WorkspaceLayoutView<'a>,
    companions: WorkspaceLayoutList<'a>,
    bias: Option<WorkspaceLayoutView<'a>>,
    positions: u64,
    rows: u64,
    columns: u64,
}
impl<'a> Geometry<'a> {
    fn new(
        op: WorkspaceOperationView<'a>,
        format: &LinearFormatSpec,
        embedding: bool,
    ) -> FactResult<Option<Self>> {
        if op.inputs.len() < 2 || op.outputs.len() != 1 {
            return invalid();
        }
        Self::from_parts(op.inputs, op.outputs.first(), format, embedding)
    }
    // None is the private observed-projection score already validated from
    // the original source prefix and the selected weight row count.
    fn from_parts(
        inputs: WorkspaceLayoutList<'a>,
        output: Option<WorkspaceLayoutView<'a>>,
        format: &LinearFormatSpec,
        embedding: bool,
    ) -> FactResult<Option<Self>> {
        if inputs.len() < 2 || (embedding && output.is_none()) {
            return invalid();
        }
        let input = inputs.get(0).unwrap();
        let weight = inputs.get(1).unwrap();
        if weight.shape().len() != 2
            || weight.shape().iter().any(|n| *n <= 0)
            || output.map_or(input.shape().is_empty(), |o| o.shape().is_empty())
        {
            return invalid();
        }
        if output.is_some_and(|o| o.dtype() != WorkspaceDtype::Float32)
            || (!embedding && input.dtype() != WorkspaceDtype::Float32)
            || (embedding
                && !matches!(
                    input.dtype(),
                    WorkspaceDtype::Int32 | WorkspaceDtype::Uint32
                ))
        {
            return Ok(None);
        }
        let columns = if embedding {
            *output.unwrap().shape().last().unwrap()
        } else {
            *input.shape().last().ok_or_else(|| {
                MlxWorkspaceFactError::descriptor("packed projection requires ranked input")
            })?
        };
        if columns <= 0 {
            return invalid();
        }
        let columns = columns as u64;
        let rows = weight.shape()[0] as u64;
        if let Some(output) = output {
            let (prefix, last) = if embedding {
                (input.shape(), columns as i32)
            } else {
                (&input.shape()[..input.shape().len() - 1], rows as i32)
            };
            if !output
                .shape()
                .iter()
                .copied()
                .eq(prefix.iter().copied().chain(std::iter::once(last)))
            {
                return invalid();
            }
        }
        // The selected encodings have at most two distinct physical companion
        // tensors. This equation-local array does not constrain rank or rows.
        let mut expected_companions = [(0, WorkspaceDtype::Uint8); 2];
        let companion_count;
        let (packed_columns, dtype) = match format.encoding() {
            LinearFormat::Affine(config) => {
                if columns % config.group_size as u64 != 0
                    || mul(columns, config.bits as u64)? % 32 != 0
                {
                    return invalid();
                }
                expected_companions =
                    [(columns / config.group_size as u64, WorkspaceDtype::Float32); 2];
                companion_count = 2;
                (
                    mul(columns, config.bits as u64)? / 32,
                    WorkspaceDtype::Uint32,
                )
            }
            LinearFormat::MxFp4 => {
                if columns % 32 != 0 {
                    return invalid();
                }
                expected_companions[0] = (columns / 32, WorkspaceDtype::Uint8);
                companion_count = 1;
                (columns / 8, WorkspaceDtype::Uint32)
            }
            LinearFormat::GgufIQuant { ggml_type, .. } => {
                let Some(native) = crate::backend::nn::native_quantization::NativeQuantizationFormat::from_ggml_type(ggml_type) else { return Ok(None); };
                let (block, bytes) = native.block_geometry();
                companion_count = 0;
                if columns % block as u64 != 0 {
                    return invalid();
                }
                (
                    mul(columns / block as u64, bytes as u64)?,
                    WorkspaceDtype::Uint8,
                )
            }
            LinearFormat::E4M3BlockFp8(config) => {
                if config.block_rows != 128 || config.block_columns != 128 {
                    return Ok(None);
                }
                companion_count = 1;
                expected_companions[0] = (
                    columns.div_ceil(128),
                    match config.scale_encoding {
                        BlockFp8ScaleEncoding::FloatingPoint => WorkspaceDtype::Float32,
                        BlockFp8ScaleEncoding::Ue8m0 => WorkspaceDtype::Uint8,
                    },
                );
                (columns, WorkspaceDtype::Uint8)
            }
            LinearFormat::Dense => return Ok(None),
        };
        if weight.shape()[1] as u64 != packed_columns || weight.dtype() != dtype {
            return invalid();
        }
        let parameter_end = 2 + companion_count;
        if inputs.len() != parameter_end && (embedding || inputs.len() != parameter_end + 1) {
            return invalid();
        }
        let companions = inputs.slice(2..parameter_end).unwrap();
        let scale_rows = if matches!(format.encoding(), LinearFormat::E4M3BlockFp8(_)) {
            rows.div_ceil(128)
        } else {
            rows
        };
        for (actual, (width, dtype)) in companions
            .iter()
            .zip(expected_companions[..companion_count].iter().copied())
        {
            if actual.shape() != [scale_rows as i32, width as i32] || actual.dtype() != dtype {
                return invalid();
            }
        }
        let bias = inputs.get(parameter_end);
        if bias.is_some_and(|b| b.shape() != [rows as i32] || b.dtype() != WorkspaceDtype::Float32)
        {
            return invalid();
        }
        let positions = if embedding {
            input.elements()?
        } else {
            input.elements()? / columns
        };
        // Native custom and packed dispatch computes flattened extents in i32.
        for n in [positions, mul(positions, columns)?, mul(positions, rows)?] {
            i32::try_from(n)
                .map_err(|_| MlxWorkspaceFactError::descriptor("packed native grid exceeds i32"))?;
        }
        Ok(Some(Self {
            input,
            weight,
            companions,
            bias,
            positions,
            rows,
            columns,
        }))
    }
    fn packed_capacity(&self, allocation: MetalAllocationFacts) -> FactResult<u64> {
        buffer_capacity(allocation, self.weight.bytes()?)
    }
}

fn quantized_split_scratch(
    m: u64,
    n: u64,
    k: u64,
    group: u64,
    a: MetalAllocationFacts,
) -> FactResult<u64> {
    // Minimum qmv->qmm threshold across supported Metal generations and GPU
    // sizes; a cold selection need not initialize a device to bound the union.
    let vector_limit = if k <= 2048 && n <= 2048 {
        14
    } else if k <= 4096 && n <= 4096 {
        10
    } else {
        6
    };
    if group == 16 || m < vector_limit {
        return Ok(0);
    }
    let tiles = mul(m.div_ceil(32), n.div_ceil(32))?;
    let mut partitions = (512 / tiles).max(1).min(k / group);
    while partitions > 1 && k % mul(partitions, group)? != 0 {
        partitions -= 1;
    }
    if partitions <= 1 {
        return Ok(0);
    }
    let output = mul(m, n)?;
    let mut bytes = capacity(a, mul(partitions, output)?)?;
    // The direct strided reduction can itself use a 32-partial second pass.
    // Split-K never exceeds 512, so the >=1024 long-column path is impossible.
    if partitions > 256 && output / 32 < 1024 {
        bytes = add(bytes, capacity(a, mul(32, output)?)?)?;
    }
    Ok(bytes)
}
fn invalid<T>() -> FactResult<T> {
    Err(MlxWorkspaceFactError::descriptor(
        "invalid Metal packed projection/embedding descriptor",
    ))
}

#[cfg(test)]
mod tests;
