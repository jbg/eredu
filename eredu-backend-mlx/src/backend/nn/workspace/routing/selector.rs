//! Cold execution of the authoritative intervention recipe. Native primitive
//! facts below own allocation costs; the neutral driver owns action sequencing.
use super::super::facts::{self, add, mul, Aliases, Emitter, FactResult, Output};
use super::super::{reduction::capacity_fixed as capacity, sampling::sort_fixed as sort};
use super::*;
use eredu_checkpoint::LinearFormat;
use eredu_nn::{
    routing_intervention::{
        execute_routing_intervention_fixed, FixedRoutingExecutionError, RoutingMechanism,
        RoutingRows,
    },
    GroupScoring, GroupSelection, RoutingPrecision, TopKGroupSelectionSpec, TopKGroupSelectorSpec,
};
use std::cell::Cell;

#[derive(Clone)]
struct Value {
    id: usize,
    elements: u64,
    storage: Output<'static>,
}
struct Counter<'a> {
    a: NativeAllocationFacts,
    spec: &'a TopKGroupSelectorSpec,
    rows: u64,
    projection: WorkspaceOperationFacts,
    projection_output: u64,
    tensor: Cell<u64>,
    host: Cell<u64>,
    next: Cell<usize>,
}
impl Counter<'_> {
    fn bytes(&self, n: u64) -> FactResult<u64> {
        capacity(self.a, n)
    }
    fn charge(&self, bytes: u64) -> FactResult<()> {
        self.tensor.set(add(self.tensor.get(), bytes)?);
        Ok(())
    }
    fn buffers(&self, n: u64, copies: u64) -> FactResult<()> {
        self.charge(mul(copies, self.bytes(n)?)?)
    }
    fn host(&self, bytes: u64) {
        self.host.set(self.host.get().max(bytes));
    }
    fn value(&self, elements: u64, storage: Output<'static>) -> Value {
        let id = self.next.get();
        self.next.set(id + 1);
        Value {
            id,
            elements,
            storage,
        }
    }
    fn allocation(&self, n: u64) -> FactResult<Value> {
        self.buffers(n, 1)?;
        Ok(self.value(n, Output::Allocate(self.bytes(n)?)))
    }
    fn pointwise(&self, n: u64, operands: u64) -> FactResult<Value> {
        // Possible promotion of each operand, one result and scalar operands.
        self.buffers(n, operands)?;
        self.buffers(1, operands)?;
        self.allocation(n)
    }
    fn reduction(&self, n: u64, rows: u64, extent: u64) -> FactResult<Value> {
        self.charge(super::super::reduction::sum_cost_fixed(
            self.a, n, rows, extent,
        )?)?;
        Ok(self.value(rows, Output::Allocate(self.bytes(rows)?)))
    }
    fn finish(&self, decision: GroupSelection<Value>) -> FactResult<GroupSelection<Value>> {
        let (ids, scores, weights) = decision.into_parts();
        let weights = if self.spec.arithmetic().coefficients == RoutingPrecision::Preserve {
            weights
        } else {
            // Metadata intentionally covers all <=F32 input/parameter dtypes.
            // The requested final dtype can require an independent allocation.
            self.allocation(weights.elements)?
        };
        Ok(GroupSelection::new(ids, scores, weights))
    }
    fn largest(&self, rows: u64, width: u64, count: u64) -> FactResult<Value> {
        let n = mul(rows, width)?;
        let selected = mul(rows, count)?;
        self.pointwise(n, 1)?; // negation by scalar, including possible cast
        self.charge(sort(self.a, n, rows, width)?)?;
        if count < width && rows != 0 {
            self.allocation(selected)?; // gather chosen scores
            self.reduction(selected, rows, count)?; // cutoff minimum
            for (size, extent) in [(n, width), (selected, count)] {
                self.pointwise(size, 2)?; // equality with cutoff
                self.allocation(size)?; // bool -> I32
                self.reduction(size, rows, extent)?;
            }
            self.pointwise(rows, 2)?; // more total ties than selected ties
            self.reduction(rows, 1, rows)?; // any, directly read scalar
                                            // Data-dependent CPU fallback has another full native index output.
                                            // std::nth_element mutates it in place: no host payload vector.
            self.allocation(n)?;
            // Original routing preserves the same global crossing-tie decision
            // lazily. The three-operand Select may promote each operand and
            // owns its selected-sized result in addition to both index arrays.
            self.pointwise(selected, 3)?;
            // Both actual stream frontiers may use the fast-fence U32 backing.
            // Slow Event mode has no such payload; retain the larger alternative.
            self.buffers(1, 2)?;
        }
        Ok(self.value(selected, Output::Allocate(self.bytes(n)?)))
    }
    fn all(&self, n: u64) -> FactResult<bool> {
        self.reduction(n, 1, n)?;
        Ok(true) // successful path includes every later predicate and operation
    }
    fn gathered_keep(&self, n: u64) -> FactResult<()> {
        let groups = self.spec.selection().group_count() as u64;
        self.host(groups); // exact Vec<bool> copied to native storage
        self.allocation(groups)?;
        self.allocation(n)?; // direct gather
        self.pointwise(self.rows, 1)?; // complement of selected-row mask
        self.pointwise(n, 2)?;
        Ok(())
    }
}

impl RoutingMechanism for Counter<'_> {
    type Value = Value;
    type Rows = RoutingRows;
    type Error = MlxWorkspaceFactError;
    fn policy(&self) -> FactResult<(TopKGroupSelectionSpec, bool)> {
        Ok((
            self.spec.selection(),
            self.spec.coefficient_scale().is_some(),
        ))
    }
    fn token_rows(&self, _: &Value) -> FactResult<u64> {
        Ok(self.rows)
    }
    fn rows(&self, selection: RoutingRows, _: u64) -> FactResult<RoutingRows> {
        for n in [selection.first, selection.end, selection.stride] {
            i32::try_from(n).map_err(|_| invalid())?;
        }
        self.allocation(self.rows)?; // native arange
                                     // ge, lt, subtract, remainder, eq, two logical_and operations.
        for operands in [1, 1, 1, 1, 1, 2, 2] {
            self.pointwise(self.rows, operands)?;
        }
        Ok(selection)
    }
    fn project(&mut self, input: &Value) -> FactResult<Value> {
        self.allocation(input.elements)?; // possible flatten copy
        if self.spec.input_transform().is_some() && self.rows != 0 {
            self.pointwise(input.elements, 1)?; // square
            self.reduction(
                input.elements,
                self.rows,
                self.spec.input_dimensions() as u64,
            )?;
            self.pointwise(self.rows, 1)?; // mean division
            self.pointwise(self.rows, 1)?; // epsilon
            self.pointwise(self.rows, 1)?; // rsqrt
            self.pointwise(input.elements, 2)?; // normalization
        }
        if self.spec.input_transform().is_some() {
            self.pointwise(input.elements, 2)?; // learned scale, also for empty input
            if self
                .spec
                .input_transform()
                .unwrap()
                .inverse_sqrt_dimensions()
            {
                self.pointwise(input.elements, 1)?;
            }
        }
        self.charge(self.projection.scratch_bytes)?;
        self.charge(self.projection_output)?;
        // Projection-specific promotion and final precision boundary, in
        // addition to the shared dense/packed mechanism's native casts/copies.
        self.allocation(input.elements)?;
        self.allocation(mul(self.rows, self.spec.selection().group_count() as u64)?)
    }
    fn transform(&self, raw: &Value) -> FactResult<Value> {
        let n = raw.elements;
        let work = if self.spec.arithmetic().scores == RoutingPrecision::Preserve {
            raw.clone()
        } else {
            self.allocation(n)?
        };
        match self.spec.selection().scoring() {
            GroupScoring::SelectedSoftmax => Ok(work),
            GroupScoring::Softmax => self.allocation(n), // precise last-axis kernel/copy
            GroupScoring::Sigmoid => {
                self.allocation(n)?; // F32 input
                self.pointwise(n, 1)?; // custom/ordinary sigmoid
                self.allocation(n) // restore score dtype
            }
            GroupScoring::SqrtSoftplus => {
                self.pointwise(n, 2)?; // logaddexp(x, 0)
                self.pointwise(n, 1) // sqrt
            }
            _ => Err(invalid()),
        }
    }
    fn ranking(&self, scores: &Value) -> FactResult<Value> {
        if self.spec.correction_bias().is_some() {
            self.pointwise(scores.elements, 2)
        } else {
            Ok(scores.clone())
        }
    }
    fn select(&self, _: &Value) -> FactResult<Value> {
        let p = self.spec.selection();
        let (groups, k, partitions, chosen) = (
            p.group_count() as u64,
            p.top_k() as u64,
            p.selection_partitions() as u64,
            p.selected_groups() as u64,
        );
        if self.rows == 0 {
            return self.allocation(0);
        }
        if partitions != 1 {
            let width = groups / partitions;
            let partition_rows = mul(self.rows, partitions)?;
            let n = mul(self.rows, groups)?;
            self.allocation(n)?; // grouped reshape can copy
            if width > 2 {
                self.charge(sort(self.a, n, partition_rows, width)?)?;
            }
            self.reduction(
                mul(partition_rows, width.min(2))?,
                partition_rows,
                width.min(2),
            )?;
            self.largest(self.rows, partitions, chosen)?;
            self.host(mul(groups, 4)?);
            self.allocation(groups)?; // partition-ID vector native copy
            let comparisons = mul(mul(self.rows, chosen)?, groups)?;
            self.pointwise(comparisons, 2)?;
            self.allocation(comparisons)?; // I32 mask
            self.reduction(comparisons, n, chosen)?;
            self.pointwise(n, 1)?; // > zero
            self.pointwise(n, 3)?; // where(mask, scores, -infinity)
        }
        self.largest(self.rows, groups, k)
    }
    fn weights(&self, _: &Value, ids: Value) -> FactResult<GroupSelection<Value>> {
        let p = self.spec.selection();
        let n = ids.elements;
        let mut weights = self.allocation(n)?; // direct GatherAxis
        if p.scoring() == GroupScoring::SelectedSoftmax {
            weights = self.allocation(n)?;
        }
        let scores = weights.clone();
        if p.normalize_selected() {
            self.allocation(n)?; // widened reduction input
            self.reduction(n, self.rows, p.top_k() as u64)?;
            self.allocation(self.rows)?; // reduction result dtype restore
            if p.normalization_epsilon() != 0. {
                self.pointwise(self.rows, 1)?;
            }
            weights = self.pointwise(n, 2)?;
        }
        if p.coefficient_scale() != 1. {
            weights = self.pointwise(n, 1)?;
        }
        if self.spec.coefficient_scale().is_some() {
            self.allocation(n)?; // direct learned-scale gather
            weights = self.pointwise(n, 2)?;
        }
        Ok(GroupSelection::new(ids, scores, weights))
    }
    fn add_columns(
        &self,
        value: &Value,
        _: &[u32],
        _: &[f32],
        _: &RoutingRows,
    ) -> FactResult<Value> {
        let groups = self.spec.selection().group_count() as u64;
        self.host(mul(groups, 4)?);
        self.buffers(groups, 2)?; // copied F32 host correction, then score dtype
        self.pointwise(value.elements, 2)?;
        self.pointwise(value.elements, 3) // row-conditioned selection
    }
    fn fill_columns(&self, value: &Value, _: &[u32], _: f32, _: &RoutingRows) -> FactResult<Value> {
        let groups = self.spec.selection().group_count() as u64;
        self.host(groups);
        self.allocation(groups)?;
        self.pointwise(self.rows, 1)?;
        self.pointwise(value.elements, 2)?;
        self.buffers(1, 2)?; // fill and dtype cast
        self.pointwise(value.elements, 3)
    }
    fn replace_rows(&self, indices: &Value, ids: &[u32], _: &RoutingRows) -> FactResult<Value> {
        self.buffers(ids.len() as u64, 2)?; // borrowed forced IDs copied/cast
                                            // Slice update can copy the logical index array or donate its full
                                            // partition backing; preserve the larger native capacity either way.
        self.charge(allocated(&indices.storage))?;
        Ok(self.value(indices.elements, indices.storage.clone()))
    }
    fn fill_gathered(
        &self,
        value: &Value,
        _: &Value,
        _: &[u32],
        _: f32,
        _: &RoutingRows,
    ) -> FactResult<Value> {
        self.gathered_keep(value.elements)?;
        self.buffers(1, 2)?;
        self.pointwise(value.elements, 3)
    }
    fn excludes(&self, indices: &Value, _: &[u32], _: &RoutingRows) -> FactResult<bool> {
        self.gathered_keep(indices.elements)?;
        self.all(indices.elements)
    }
    fn finite(&self, value: &Value) -> FactResult<bool> {
        self.pointwise(value.elements, 1)?;
        self.all(value.elements)
    }
    fn nonnegative(&self, value: &Value) -> FactResult<bool> {
        self.pointwise(value.elements, 1)?;
        self.all(value.elements)
    }
    fn positive_row_sums(&self, value: &Value) -> FactResult<bool> {
        self.reduction(
            value.elements,
            self.rows,
            self.spec.selection().top_k() as u64,
        )?;
        self.pointwise(self.rows, 1)?;
        self.all(self.rows)
    }
}

fn allocated(storage: &Output<'_>) -> u64 {
    match storage {
        Output::Allocate(bytes) | Output::AllocateOrAliasInputs { bytes, .. } => *bytes,
        _ => 0,
    }
}

pub(super) fn bounds(
    op: &WorkspaceOperation,
    a: NativeAllocationFacts,
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

fn intervention_error(
    error: FixedRoutingExecutionError<MlxWorkspaceFactError>,
) -> MlxWorkspaceFactError {
    match error {
        FixedRoutingExecutionError::Invalid(cause) => MlxWorkspaceFactError::routing_invalid(cause),
        FixedRoutingExecutionError::Native(cause) => cause.in_routing(),
    }
}

pub(super) fn ordinary_error(op: &WorkspaceOperation, error: MlxWorkspaceFactError) -> Error {
    if matches!(error.cause(), MlxWorkspaceFactCause::Selector(_)) {
        if let WorkspaceOperationKind::GroupSelection { spec, .. } = &op.kind {
            if let Err(original) = spec.validate() {
                return original;
            }
        }
    }
    error.ordinary()
}

/// Borrow the exact flattened selector projection for the existing numerical
/// and native producers. Floating replacement follows the ordinary worker's
/// branch before packed companions; no substitute tensor or storage is created.
pub(in super::super) fn with_projection<R>(
    op: WorkspaceOperationView<'_>,
    run: impl FnOnce(WorkspaceOperationView<'_>) -> R,
) -> FactResult<Option<R>> {
    let WorkspaceOperationKindView::GroupSelection { spec, supplied_indices, .. } = op.kind else {
        return Ok(None);
    };
    spec.validate_fixed()?;
    let hidden = op.inputs.get(0).ok_or_else(invalid)?;
    let width = spec.input_dimensions();
    if width <= 0 || hidden.shape().last() != Some(&width) { return Err(invalid()); }
    let rows = i32::try_from(hidden.elements()? / width as u64).map_err(|_| invalid())?;
    let start = 1 + usize::from(supplied_indices);
    let parameters = 1 + usize::from(spec.format().scale().is_some())
        + usize::from(spec.format().affine_bias().is_some()) + usize::from(spec.bias().is_some());
    let extra = usize::from(spec.correction_bias().is_some())
        + usize::from(spec.input_transform().is_some()) + usize::from(spec.coefficient_scale().is_some());
    if op.inputs.len() != start + parameters + extra { return Err(invalid()); }
    let input_shape = [rows, width];
    // Flatten alone preserves scalar precision. An optional input transform
    // has its own promotion and cannot borrow the pre-transform scalar fact.
    // Potential copies are paid outside; original strides do not describe this view.
    let input = WorkspaceLayoutView::new(&input_shape, hidden.dtype())?.with_representation(
        spec.input_transform().is_none().then(|| hidden.representation()).flatten()
            .map(|actual| WorkspaceRepresentation::new(actual.dtype(), false)));
    let mut inputs = [input; 5];
    let weight = op.inputs.get(start).ok_or_else(invalid)?;
    let floating = weight.dtype() == WorkspaceDtype::Float32;
    let dense_format = eredu_nn::LinearFormatSpec::unscaled(LinearFormat::Dense)
        .map_err(|_| invalid())?;
    let (format, count) = if floating {
        inputs[1] = weight;
        if spec.bias().is_some() {
            inputs[2] = op.inputs.get(start + parameters - 1).ok_or_else(invalid)?;
        }
        (&dense_format, 2 + usize::from(spec.bias().is_some()))
    } else {
        for (index, parameter) in op.inputs.slice(start..start + parameters).unwrap().iter().enumerate() {
            inputs[index + 1] = parameter;
        }
        (spec.format(), parameters + 1)
    };
    let output_shape = [rows, spec.selection().group_count()];
    let outputs = [WorkspaceLayoutView::new(&output_shape, WorkspaceDtype::Float32)?];
    Ok(Some(run(WorkspaceOperationView {
        kind: WorkspaceOperationKindView::Projection(format),
        inputs: WorkspaceLayoutList::Views(&inputs[..count]),
        outputs: WorkspaceLayoutList::Views(&outputs),
    })))
}

pub(super) fn emit(
    op: WorkspaceOperationView<'_>,
    a: NativeAllocationFacts,
    sink: &mut Emitter<'_>,
) -> FactResult<Option<(WorkspaceOperationFacts, u64)>> {
    let WorkspaceOperationKindView::GroupSelection {
        spec,
        supplied_indices,
        control,
    } = &op.kind
    else {
        return Ok(None);
    };
    spec.validate_fixed()?;
    if *supplied_indices && control.is_some() {
        return Err(invalid());
    }
    let hidden = op.inputs.first().ok_or_else(invalid)?;
    let width = spec.input_dimensions();
    if hidden.shape().last() != Some(&width) {
        return Err(invalid());
    }
    let rows = hidden.elements()? / width as u64;
    let rows_i32 = i32::try_from(rows).map_err(|_| invalid())?;
    let k = spec.selection().top_k();
    let start = 1 + usize::from(*supplied_indices);
    let parameters = 1
        + usize::from(spec.format().scale().is_some())
        + usize::from(spec.format().affine_bias().is_some())
        + usize::from(spec.bias().is_some());
    let extra = usize::from(spec.correction_bias().is_some())
        + usize::from(spec.input_transform().is_some())
        + usize::from(spec.coefficient_scale().is_some());
    if op.inputs.len() != start + parameters + extra {
        return Err(invalid());
    }
    let id_dtype = if *supplied_indices {
        let ids = op.inputs.get(1).unwrap();
        if !matches!(ids.dtype(), WorkspaceDtype::Int32 | WorkspaceDtype::Uint32)
            || ids.elements()? != mul(rows, k as u64)?
        {
            return Err(invalid());
        }
        ids.dtype()
    } else {
        WorkspaceDtype::Uint32
    };
    let decisions = 1 + usize::from(control.as_ref().is_some_and(|c| c.capture_original));
    if op.outputs.len() != decisions * 3 {
        return Err(invalid());
    }
    for (i, output) in op.outputs.iter().enumerate() {
        if output.shape() != [rows_i32, k]
            || output.dtype()
                != if i % 3 == 0 {
                    id_dtype
                } else {
                    WorkspaceDtype::Float32
                }
        {
            return Err(invalid());
        }
    }
    let mut slot = start + parameters;
    for (present, size) in [
        (
            spec.correction_bias().is_some(),
            spec.selection().group_count(),
        ),
        (spec.input_transform().is_some(), width),
        (
            spec.coefficient_scale().is_some(),
            spec.selection().group_count(),
        ),
    ] {
        if present {
            if op.inputs.get(slot).unwrap().shape() != [size] {
                return Err(invalid());
            }
            if op.inputs.get(slot).unwrap().dtype() != WorkspaceDtype::Float32 {
                return Ok(None);
            }
            slot += 1;
        }
    }
    // Selector construction currently realizes dense, affine, MXFP4 and native
    // GGML matrices; do not inherit ordinary projection's block-FP8 support.
    if !matches!(
        spec.format().encoding(),
        LinearFormat::Dense
            | LinearFormat::Affine(_)
            | LinearFormat::MxFp4
            | LinearFormat::GgufIQuant { .. }
    ) {
        return Ok(None);
    }
    let mut projection_sink = Emitter::count();
    let projection = with_projection(op, |projection_op| {
        let WorkspaceOperationKindView::Projection(format) = projection_op.kind else { unreachable!() };
        if format.encoding() == LinearFormat::Dense {
            super::super::matrix::emit(projection_op, a, &mut projection_sink)
        } else {
            super::super::packed::emit(projection_op, a, &mut projection_sink)
        }
    })?;
    let Some(projection) = projection else { return Ok(None); };
    let projection = projection?;
    let Some(projection) = projection else {
        return Ok(None);
    };
    let projection_output = match projection_sink.first_output() {
        Some(
            WorkspaceOutputEffect::Allocate(bytes)
            | WorkspaceOutputEffect::AllocateOrAliasInputs { bytes, .. },
        ) => bytes,
        Some(_) => 0,
        None => return Err(invalid()),
    };
    let mut counter = Counter {
        a,
        spec,
        rows,
        projection,
        projection_output,
        tensor: Cell::new(0),
        host: Cell::new(0),
        next: Cell::new(0),
    };
    let input = counter.value(hidden.elements()?, Output::AliasInput(0));
    let mut decisions = [None, None];
    if let Some(control) = control {
        // Price the actual selected native driver's ordinary validation
        // workspace. This cold companion validates the same source with the
        // borrowed worker, while native execution still owns its partition Vec.
        counter.host(mul(spec.selection().selection_partitions() as u64, 4)?);
        let result = execute_routing_intervention_fixed(&mut counter, &input, control)
            .map_err(intervention_error)?;
        if let Some(original) = result.original {
            decisions[0] = Some(counter.finish(original)?);
        }
        decisions[usize::from(decisions[0].is_some())] = Some(counter.finish(result.effective)?);
    } else {
        let raw = counter.project(&input)?;
        let scores = counter.transform(&raw)?;
        let ids = if *supplied_indices {
            counter.buffers(mul(rows, k as u64)?, 1)?;
            counter.value(
                mul(rows, k as u64)?,
                Output::AllocateOrAliasInputs {
                    bytes: counter.bytes(mul(rows, k as u64)?)?,
                    inputs: Aliases::Slice(&[1]),
                },
            )
        } else {
            counter.select(&counter.ranking(&scores)?)?
        };
        decisions[0] = Some(counter.finish(counter.weights(&scores, ids)?)?);
    }
    let mut roots = [0; 6];
    let mut root_count = 0;
    let mut retained = 0;
    for decision in decisions.into_iter().flatten() {
        let (ids, scores, weights) = decision.into_parts();
        for value in [ids, scores, weights] {
            let storage =
                if let Some(prior) = roots[..root_count].iter().position(|id| *id == value.id) {
                    Output::AliasOutput(prior)
                } else {
                    retained = add(retained, allocated(&value.storage))?;
                    value.storage
                };
            roots[root_count] = value.id;
            root_count += 1;
            sink.output(storage)?;
        }
    }
    let tensor = sink.finish(counter.tensor.get().checked_sub(retained).ok_or_else(invalid)?, format_args!("MLX Metal top-k router: shared dense/packed projection, input RMS, staged dtype/scoring policy, complete partition backing, all tie detection and possible CPU partition, grouped eligibility, gathered/normalized/scaled coefficients; intervention costs execute the authoritative neutral recipe including original capture and predicates; all child buffers retained through completion; page={} with bounded oversized reuse", a.page_size()))?;
    Ok(Some((tensor, counter.host.get())))
}

#[cfg(test)]
mod tests;
