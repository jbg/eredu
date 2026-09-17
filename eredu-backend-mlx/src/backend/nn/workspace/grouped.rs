//! Allocation facts for native packed expert equations, including the actual
//! selection-sized weight replicas of the affine group-16 fallback.
use super::{
    reduction::{capacity_fixed as capacity, sum_cost_fixed as sum_cost},
    sampling::sort_fixed as sort,
    *,
};
use eredu_checkpoint::{BlockFp8ScaleEncoding, LinearFormat};
use eredu_nn::{
    GatedProductGroupLayout, GroupReduction, GroupedLinearActivation, GroupedProjectionSpec,
    LinearFormatSpec,
};

mod projection;
use projection::Projection;

use super::facts::{self, add, buffer_capacity, mul, Emitter, FactResult, Output};

fn invalid() -> MlxWorkspaceFactError {
    MlxWorkspaceFactError::descriptor("invalid Metal grouped workspace descriptor")
}

#[derive(Clone, Copy)]
struct Cost {
    a: MetalAllocationFacts,
    tensor: u64,
    host: u64,
}
impl Cost {
    fn new(a: MetalAllocationFacts) -> Self {
        Self {
            a,
            tensor: 0,
            host: 0,
        }
    }
    fn charge(&mut self, bytes: u64) -> FactResult<()> {
        self.tensor = add(self.tensor, bytes)?;
        Ok(())
    }
    fn buffers(&mut self, n: u64, count: u64) -> FactResult<()> {
        self.charge(mul(capacity(self.a, n)?, count)?)
    }
    fn bytes(&mut self, bytes: u64, count: u64) -> FactResult<()> {
        self.charge(mul(buffer_capacity(self.a, bytes.max(4))?, count)?)
    }
    fn pointwise(&mut self, n: u64, operands: u64) -> FactResult<()> {
        self.buffers(n, add(operands, 1)?)?;
        self.buffers(1, operands)
    }
    fn sum(&mut self, n: u64, outputs: u64, extent: u64) -> FactResult<()> {
        self.charge(sum_cost(self.a, n, outputs, extent)?)
    }
    fn append(&mut self, other: Self, count: u64) -> FactResult<()> {
        self.charge(mul(other.tensor, count)?)?;
        self.host = self.host.max(other.host);
        Ok(())
    }
    fn plan(&mut self, routes: u64) -> FactResult<()> {
        self.buffers(routes, 2)?; // flatten and I32 group IDs
        self.charge(sort(self.a, routes, 1, routes)?)?;
        self.buffers(routes, 2)?; // sorted IDs and signed original-slot indices
        self.pointwise(routes, 1) // token indices: floor division by top-k
    }
    fn weighted_sum(
        &mut self,
        tokens: u64,
        k: u64,
        width: u64,
        reduction: GroupReduction,
    ) -> FactResult<()> {
        let routes = mul(tokens, k)?;
        let full = mul(routes, width)?;
        self.buffers(routes, 2)?; // coefficient flatten and gather
        self.pointwise(full, 2)?; // weighted source rows
        if tokens == 0 {
            return self.buffers(1, 2);
        }
        self.buffers(full, 4)?; // zero destination, source reshape, scatter result, final reshape
        self.buffers(routes, 1)?; // possible normalized scatter indices
        self.buffers(1, 2)?; // zero fill and dtype cast
        if reduction == GroupReduction::Sum {
            return self.sum(full, mul(tokens, width)?, k);
        }
        self.buffers(routes, 4)?; // zero IDs, update reshape, scatter, [tokens,k] reshape
        self.buffers(1, 2)?;
        self.charge(sort(self.a, routes, tokens, k)?)?;
        self.buffers(full, 1)?; // direct axis gather; broadcast order is a view
        self.buffers(mul(tokens, width)?, 1)?; // initial sequential accumulator
        self.buffers(1, 2)?;
        let mut step = Cost::new(self.a);
        step.pointwise(mul(tokens, width)?, 2)?;
        step.buffers(mul(tokens, width)?, 1)?; // restoration to ordered input dtype
        self.append(step, k)
    }
}

#[derive(Clone, Copy)]
enum Activation {
    Linear(GroupedLinearActivation),
    Gated(eredu_nn::GatedProductPolicy),
    Relu2,
}
struct Bank<'a> {
    groups: i32,
    input: i32,
    units: i32,
    output: i32,
    first: &'a GroupedProjectionSpec,
    down: Option<&'a GroupedProjectionSpec>,
    activation: Activation,
    reduction: GroupReduction,
    chunks: bool,
}
impl<'a> Bank<'a> {
    fn new(bank: &'a WorkspaceGroupedBank) -> FactResult<Option<Self>> {
        Ok(Some(match bank {
            WorkspaceGroupedBank::Linear(s) => {
                s.validate_fixed()?;
                Self {
                    groups: s.group_count(),
                    input: s.input_dimensions(),
                    units: s.output_dimensions(),
                    output: s.output_dimensions(),
                    first: s.projection(),
                    down: None,
                    activation: Activation::Linear(s.activation()),
                    reduction: s.reduction(),
                    chunks: false,
                }
            }
            WorkspaceGroupedBank::GatedProduct(s) => {
                s.validate_fixed()?;
                let GatedProductGroupLayout::Packed { gate_up, down } = s.layout() else {
                    // Independent members execute through the runtime provider,
                    // not through this native packed-bank implementation.
                    return Ok(None);
                };
                if s.input_dimensions() != s.output_dimensions() {
                    return Ok(None);
                }
                match (gate_up.format().encoding(), down.format().encoding()) {
                    (LinearFormat::E4M3BlockFp8(a), LinearFormat::E4M3BlockFp8(b)) if a == b => {}
                    (LinearFormat::E4M3BlockFp8(_), _) | (_, LinearFormat::E4M3BlockFp8(_)) => {
                        return Ok(None)
                    }
                    _ => {}
                }
                Self {
                    groups: s.group_count(),
                    input: s.input_dimensions(),
                    units: s.intermediate_dimensions(),
                    output: s.output_dimensions(),
                    first: gate_up,
                    down: Some(down),
                    activation: Activation::Gated(s.policy()),
                    reduction: s.reduction(),
                    chunks: true,
                }
            }
            WorkspaceGroupedBank::Relu2(s) => {
                s.validate_fixed()?;
                if s.up().bias().is_some()
                    || s.down().bias().is_some()
                    || matches!(s.up().format().encoding(), LinearFormat::E4M3BlockFp8(_))
                    || matches!(s.down().format().encoding(), LinearFormat::E4M3BlockFp8(_))
                {
                    return Ok(None);
                }
                Self {
                    groups: s.group_count(),
                    input: s.hidden_dimensions(),
                    units: s.intermediate_dimensions(),
                    output: s.hidden_dimensions(),
                    first: s.up(),
                    down: Some(s.down()),
                    activation: Activation::Relu2,
                    reduction: GroupReduction::Sum,
                    chunks: false,
                }
            }
        }))
    }
    fn activate(&self, routes: u64, a: MetalAllocationFacts) -> FactResult<Option<Cost>> {
        let mut cost = Cost::new(a);
        let n = mul(routes, self.units as u64)?;
        match self.activation {
            Activation::Linear(GroupedLinearActivation::Identity) => {}
            Activation::Linear(GroupedLinearActivation::Silu) => {
                // Shared SiLU may use a custom F32 kernel or the scalar/native
                // fallback: sigmoid (negate/exp/add/reciprocal), multiply and
                // all possible input promotions/broadcast materializations.
                cost.buffers(n, 11)?;
                cost.buffers(1, 1)?;
            }
            Activation::Relu2 => {
                cost.pointwise(n, 1)?;
                cost.pointwise(n, 1)?;
            }
            Activation::Gated(policy) => {
                let shape = [i32::try_from(routes).map_err(|_| invalid())?, self.units];
                let layout = WorkspaceLayoutView::new(&shape, WorkspaceDtype::Float32)?;
                let inputs = [layout, layout];
                let outputs = [layout];
                let op = WorkspaceOperationView {
                    kind: WorkspaceOperationKindView::GatedProduct(policy),
                    inputs: WorkspaceLayoutList::Views(&inputs),
                    outputs: WorkspaceLayoutList::Views(&outputs),
                };
                let mut sink = Emitter::count();
                let Some(bound) = basic::emit(op, a, &mut sink)? else {
                    return Ok(None);
                };
                cost.charge(bound.scratch_bytes)?;
                match sink.first_output() {
                    Some(
                        WorkspaceOutputEffect::Allocate(bytes)
                        | WorkspaceOutputEffect::AllocateOrAliasInputs { bytes, .. },
                    ) => cost.charge(bytes)?,
                    _ => return Err(invalid()),
                }
            }
        }
        Ok(Some(cost))
    }
}

pub(super) fn bounds(
    op: &WorkspaceOperation,
    a: MetalAllocationFacts,
) -> Result<Option<(WorkspaceOperationBound, u64)>, Error> {
    let mut host = 0;
    let tensor = facts::ordinary_with(
        |sink| {
            Ok(emit(op.as_view(), a, sink)?.map(|(tensor, bytes)| {
                host = bytes;
                tensor
            }))
        },
        |error| ordinary_error(op, error),
    )?;
    Ok(tensor.map(|tensor| (tensor, host)))
}

pub(super) fn ordinary_error(op: &WorkspaceOperation, error: MlxWorkspaceFactError) -> Error {
    if matches!(
        error.cause(),
        MlxWorkspaceFactCause::GroupedLinear(_)
            | MlxWorkspaceFactCause::GroupedBank(_)
            | MlxWorkspaceFactCause::GroupedRelu2(_)
    ) {
        if let WorkspaceOperationKind::Grouped { bank, .. } = &op.kind {
            let original = match bank.as_ref() {
                WorkspaceGroupedBank::Linear(spec) => spec.validate(),
                WorkspaceGroupedBank::GatedProduct(spec) => spec.validate(),
                WorkspaceGroupedBank::Relu2(spec) => spec.validate(),
            };
            if let Err(original) = original {
                return original;
            }
        }
    }
    error.ordinary()
}

pub(super) fn emit(
    op: WorkspaceOperationView<'_>,
    a: MetalAllocationFacts,
    sink: &mut Emitter<'_>,
) -> FactResult<Option<(WorkspaceOperationFacts, u64)>> {
    let WorkspaceOperationKindView::Grouped {
        bank,
        phase,
        partitions,
    } = &op.kind
    else {
        return Ok(None);
    };
    let Some(bank) = Bank::new(bank)? else {
        return Ok(None);
    };
    if bank.groups == 0 {
        return Ok(None);
    }
    if op.inputs.len() < 4 || *partitions == Some(0) {
        return Err(invalid());
    }
    let input = op.inputs.get(0).unwrap();
    if input.shape().last() != Some(&bank.input) {
        return Err(invalid());
    }
    if matches!(bank.activation, Activation::Linear(_))
        && (input.shape().len() != 2
            || partitions.is_some()
            || *phase != WorkspaceGroupedPhase::Whole)
    {
        return Err(invalid());
    }
    let tokens = input.elements()? / bank.input as u64;
    let ids = op.inputs.get(1).unwrap();
    let k = *ids.shape().last().ok_or_else(invalid)?;
    if ids.shape().len() != 2
        || k <= 0
        || ids.shape()[0] as u64 != tokens
        || !matches!(ids.dtype(), WorkspaceDtype::Int32 | WorkspaceDtype::Uint32)
        || op.inputs.get(2).unwrap().shape() != ids.shape()
        || op.inputs.get(3).unwrap().shape() != ids.shape()
    {
        return Err(invalid());
    }
    if [input, op.inputs.get(2).unwrap(), op.inputs.get(3).unwrap()]
        .iter()
        .any(|v| v.dtype() != WorkspaceDtype::Float32)
    {
        return Ok(None);
    }
    let routes = mul(tokens, k as u64)?;
    let routes_i32 = i32::try_from(routes).map_err(|_| invalid())?;
    let read = if matches!(bank.activation, Activation::Gated(_)) {
        bank.units.checked_mul(2).ok_or_else(invalid)?
    } else {
        bank.units
    };
    let mut slot = 4;
    let Some(first) = Projection::new(
        bank.first,
        bank.groups,
        bank.input,
        read,
        op.inputs,
        &mut slot,
        a,
    )?
    else {
        return Ok(None);
    };
    let down = if let Some(spec) = bank.down {
        let Some(down) = Projection::new(
            spec,
            bank.groups,
            bank.units,
            bank.output,
            op.inputs,
            &mut slot,
            a,
        )?
        else {
            return Ok(None);
        };
        Some(down)
    } else {
        None
    };
    let units = *phase == WorkspaceGroupedPhase::Units;
    let finish = *phase == WorkspaceGroupedPhase::Finish;
    if op.inputs.len() != slot + if finish { 4 } else { 0 } {
        return Err(invalid());
    }
    let units_shape = [routes_i32, bank.units];
    let ids_shape = [routes_i32];
    let phase_shapes: [&[i32]; 4] = [&units_shape, &ids_shape, &ids_shape, &ids_shape];
    if finish {
        for (i, shape) in phase_shapes.iter().enumerate() {
            let value = op.inputs.get(slot + i).unwrap();
            if value.shape() != *shape {
                return Err(invalid());
            }
            if i == 0 && value.dtype() != WorkspaceDtype::Float32 {
                return Ok(None);
            }
            if i > 0
                && !matches!(
                    value.dtype(),
                    WorkspaceDtype::Int32 | WorkspaceDtype::Uint32
                )
            {
                return Err(invalid());
            }
        }
    }
    let separate_bias = partitions.is_some() && down.as_ref().is_some_and(|p| p.bias.is_some());
    let output_count = if units {
        4
    } else {
        1 + usize::from(separate_bias)
    };
    if output_count != op.outputs.len() {
        return Err(invalid());
    }
    for (i, output) in op.outputs.iter().enumerate() {
        let correct_shape = if units {
            output.shape() == phase_shapes[i]
        } else {
            output
                .shape()
                .iter()
                .copied()
                .eq(input.shape()[..input.shape().len() - 1]
                    .iter()
                    .copied()
                    .chain(std::iter::once(bank.output)))
        };
        if !correct_shape
            || output.dtype()
                != if units && i > 0 {
                    WorkspaceDtype::Uint32
                } else {
                    WorkspaceDtype::Float32
                }
        {
            return Err(invalid());
        }
    }
    let mut total = Cost::new(a);
    if !finish {
        total.buffers(input.elements()?, 1)?; // native adapter flatten
        if matches!(
            bank.activation,
            Activation::Linear(_) | Activation::Gated(_)
        ) {
            total.charge(indexing::embedding_validation_cost_fixed(
                routes,
                0,
                eredu_nn::EmbeddingLookupPolicy::Strict,
                a,
            )?)?;
        }
    }
    let mut retained_storage = [0; 4];
    let retained = &mut retained_storage[..output_count];
    let chunk = |tokens: u64| -> FactResult<Option<(Cost, Option<[u64; 4]>)>> {
        let n = mul(tokens, k as u64)?;
        let mut cost = Cost::new(a);
        if !finish {
            cost.plan(n)?;
            cost.buffers(mul(n, bank.input as u64)?, 1)?; // selected hidden rows
            cost.append(first.cost(n, a)?, 1)?;
            let Some(activation) = bank.activate(n, a)? else {
                return Ok(None);
            };
            cost.append(activation, 1)?;
        }
        if units {
            // The metadata result aggregates the actual per-chunk allocations,
            // including each buffer's separate allocator capacity rounding.
            return Ok(Some((
                cost,
                Some([
                    // A donated gate/up slice may retain the complete fused
                    // projection backing even when only activated units escape.
                    capacity(a, mul(n, read as u64)?)?,
                    capacity(a, n)?,
                    capacity(a, n)?,
                    capacity(a, n)?,
                ]),
            )));
        }
        if let Some(down) = &down {
            cost.append(down.cost(n, a)?, 1)?;
        }
        cost.weighted_sum(tokens, k as u64, bank.output as u64, bank.reduction)?;
        Ok(Some((cost, None)))
    };
    use super::super::grouped::{
        GROUPED_PROJECTION_CHUNK_THRESHOLD as THRESHOLD, GROUPED_PROJECTION_CHUNK_TOKENS as CHUNK,
    };
    let chunks = if bank.chunks && tokens > THRESHOLD as u64 {
        [
            (CHUNK as u64, tokens / CHUNK as u64),
            (tokens % CHUNK as u64, u64::from(tokens % CHUNK as u64 != 0)),
        ]
    } else {
        [(tokens, 1), (0, 0)]
    };
    let chunked = chunks.iter().map(|(_, count)| *count).sum::<u64>() > 1;
    for (rows, count) in chunks {
        if count == 0 {
            continue;
        }
        let Some((cost, outputs)) = chunk(rows)? else {
            return Ok(None);
        };
        total.append(cost, count)?;
        if let Some(outputs) = outputs {
            for (sum, bytes) in retained.iter_mut().zip(outputs) {
                *sum = add(*sum, mul(bytes, count)?)?;
            }
        }
    }
    if !units {
        if chunked {
            total.buffers(mul(tokens, bank.output as u64)?, 1)?;
        } // concatenated chunk results
        if separate_bias {
            total.plan(routes)?;
            total.buffers(mul(routes, bank.output as u64)?, 1)?;
            total.weighted_sum(tokens, k as u64, bank.output as u64, bank.reduction)?;
            total.pointwise(mul(tokens, bank.output as u64)?, 2)?; // subtract bias from ordinary output
        }
        for retained in retained.iter_mut() {
            *retained = capacity(a, mul(tokens, bank.output as u64)?)?;
            total.charge(*retained)?; // final adapter reshape may copy each result
        }
    }
    let output_bytes = retained.iter().try_fold(0, |sum, &bytes| add(sum, bytes))?;
    let scratch = total.tensor.checked_sub(output_bytes).ok_or_else(invalid)?;
    for &bytes in retained.iter() {
        sink.output(Output::Allocate(bytes))?;
    }
    let tensor = sink.finish(scratch, format_args!("MLX Metal packed expert equations: exact native grouped projection and physical companions, full selection sorting/gather, activated units, native weighted reduction and optional TP bias correction; actual gated-bank chunk thresholds {THRESHOLD}/{CHUNK} with every lazy child retained through completion; affine group-16 selected packed-weight replicas included; separately rounded per-chunk output ownership; page={} with bounded oversized reuse",a.page_size()))?;
    Ok(Some((tensor, total.host)))
}

#[cfg(test)]
mod tests;

/// Finite branch census for a local expert child (one selected expert per row).
///
/// The positive nonchunked grouped census is componentwise nondecreasing:
/// projection/reduction capacities are rounded nonnegative multiples of rows;
/// ID validation reduces to one scalar; sorting adds merge buffers/kernels at
/// increasing lengths. Its zero/singleton alias cases are inspected separately.
/// The actual Gated worker alone has a threshold/chunk loop. For each remainder
/// modulo CHUNK, adding a full chunk only adds retained child buffers, nodes,
/// validations and kernels; output/concatenation capacities also grow. Inspect
/// the largest admitted member of every remainder class and the last unchunked
/// member. This includes full tails and singleton tails without O(maximum) work.
/// No parameter format/worker qualification follows from these candidates:
/// every candidate still traverses the same exact native recipe inspector.
pub(super) fn expert_row_candidates(
    kernel: eredu_nn::workspace::WorkspaceExpertKernel<'_>, maximum: usize,
) -> impl Iterator<Item = usize> {
    use crate::backend::nn::grouped::{
        GROUPED_PROJECTION_CHUNK_THRESHOLD as THRESHOLD,
        GROUPED_PROJECTION_CHUNK_TOKENS as CHUNK,
    };
    let chunked = matches!(kernel, eredu_nn::workspace::WorkspaceExpertKernel::Gated(_));
    let threshold = THRESHOLD as usize;
    let chunk = CHUNK as usize;
    // Zero belongs to the separately retained actual empty-provider recipe.
    let unchunked = if chunked { maximum.min(threshold) } else { maximum };
    [1, unchunked].into_iter()
        .filter(move |rows| *rows != 0 && *rows <= maximum)
        .chain((0..if chunked && maximum > threshold { chunk } else { 0 })
            .filter_map(move |remainder| {
                let distance = (maximum % chunk + chunk - remainder) % chunk;
                maximum.checked_sub(distance).filter(|rows| *rows > threshold)
            }))
}
