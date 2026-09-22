//! Census of the actual ordinary bank discovery/remapping traversal.
use super::super::resident_recipe::{OrdinaryCpuPopulation, OrdinaryNativeControls};
use super::*;
use crate::backend::error::Error as NativeError;
use crate::backend::runtime::residency::parameter_bank::indexed_numerical::{
    self, Operation, Visitor,
};
use safemlx::{CpuUnaryOperation, Dtype};

#[derive(Clone, Copy)]
pub(super) struct Value {
    pub(super) dtype: Dtype,
    pub(super) rank: usize,
    pub(super) elements: usize,
}
struct Census {
    program: program::Program,
    calls: OrdinaryCallControls,
}
fn unknown() -> NativeError {
    NativeError::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::UnknownBound)
}
fn overflow() -> NativeError {
    NativeError::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::Overflow)
}
fn source<T>(value: Option<T>) -> Result<T, NativeError> {
    value.ok_or_else(unknown)
}
fn checked<T>(value: Option<T>) -> Result<T, NativeError> {
    value.ok_or_else(overflow)
}

/// Fixed numerical workers only. Ordinary graph ownership, evaluation and
/// dispatch metadata are separate contributors; no original arena is implied.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct OrdinaryIndexedNumericalFacts {
    pub(crate) backing_bytes: u64,
    pub(crate) backing_births: usize,
    /// Actual safe caller metadata from the shared semantic visitor.
    pub(crate) fixed_host_controls: u64,
    /// Caller-created root-vector backing, excluded from the raw primitive DAG.
    pub(crate) caller_native_controls: OrdinaryNativeControls,
    pub(crate) primitives: usize,
    pub(crate) construction_entries: usize,
    pub(crate) input_edges: usize,
    pub(crate) seeds: usize,
    pub(crate) parameter_shells: usize,
    /// Discovery completes its histogram and invalid count together once per
    /// chunk. Bank output completion is already in the shared equation source.
    pub(crate) discovery_completions: usize,
    /// Reachable native discovery/remap graph counts. Discovery inputs may be
    /// lazy ancestors of the enclosing equation, so its actual two-root Eval
    /// must use the combined reachable graph, not this population in isolation.
    pub(crate) raw_population: Option<OrdinaryCpuPopulation>,
    /// Only frontend and selected worker allocations. A caller that derives
    /// these again from the combined raw population must not add them twice.
    pub(crate) local_native_controls: OrdinaryNativeControls,
}
impl MlxCpuWorkspaceMechanisms {
    pub(crate) fn ordinary_indexed_numerical_facts(
        self,
        first: eredu_runtime::expert::AddressableChunkCensus,
        dtype: WorkspaceDtype,
    ) -> Result<OrdinaryIndexedNumericalFacts, NativeError> {
        if first.index() != 0 {
            return Err(unknown());
        }
        let dtype = match dtype {
            WorkspaceDtype::Int32 => Dtype::Int32,
            WorkspaceDtype::Uint32 => Dtype::Uint32,
            _ => return Err(unknown()),
        };
        let count = first.plan().len();
        if count == 0 {
            return Err(unknown());
        }
        let mut facts = OrdinaryIndexedNumericalFacts::default();
        // The driver repeats full chunks and executes one exact final tail.
        // Lookup span is bounded by the actual immutable local member domain.
        for (index, occurrences) in [(0, count - 1), (count - 1, 1)] {
            if occurrences == 0 {
                continue;
            }
            let rows = first.plan().range(index).ok_or_else(unknown)?.len();
            let input = Value {
                dtype,
                rank: 2,
                elements: checked(rows.checked_mul(first.routes()))?,
            };
            let discovery = discovery_with_calls(self, input, first.members())?;
            let remap = remap_with_calls(self, input, first.members())?;
            let repetitions = u64::try_from(occurrences).map_err(|_| overflow())?;
            for (plan, calls) in [discovery, remap] {
                let population = source(OrdinaryCpuPopulation::from_operation(plan, 1))?;
                facts.caller_native_controls = checked(
                    facts
                        .caller_native_controls
                        .append(checked(calls.observed.repeat(occurrences))?),
                )?;
                let mut controls = OrdinaryNativeControls::default();
                checked(controls.include(source(
                    OperationEvent::ordinary_frontend_control_layout(
                        population.construction_entries,
                        population.seeds,
                        population.maximum_rank,
                        population.maximum_operands,
                    ),
                )?))?;
                checked(controls.include(source(
                    OperationEvent::ordinary_cpu_dispatch_envelope(
                        population.dispatch_graph_extents,
                    ),
                )?))?;
                facts.local_native_controls = checked(
                    facts
                        .local_native_controls
                        .append(checked(controls.repeat(occurrences))?),
                )?;
                let repeated = checked(population.repeat(occurrences))?;
                facts.raw_population = Some(match facts.raw_population {
                    Some(previous) => checked(previous.append(repeated))?,
                    None => repeated,
                });
                facts.backing_bytes = checked(
                    facts.backing_bytes.checked_add(checked(
                        checked(plan.output_bytes.checked_add(plan.scratch_bytes))?
                            .checked_mul(repetitions),
                    )?),
                )?;
                let births = checked(plan.population.births.checked_add(plan.seeds))?;
                facts.backing_births = checked(
                    facts
                        .backing_births
                        .checked_add(checked(births.checked_mul(occurrences))?),
                )?;
                facts.fixed_host_controls = checked(
                    facts
                        .fixed_host_controls
                        .checked_add(checked(calls.metadata_bytes.checked_mul(repetitions))?),
                )?;
                macro_rules! add_count {
                    ($field:ident, $value:expr) => {
                        facts.$field = checked(
                            facts
                                .$field
                                .checked_add(checked(($value).checked_mul(occurrences))?),
                        )?;
                    };
                }
                add_count!(primitives, plan.population.primitives);
                add_count!(construction_entries, plan.population.construction_entries);
                add_count!(input_edges, plan.population.input_edges);
                add_count!(seeds, plan.seeds);
                add_count!(parameter_shells, plan.parameter_shells);
            }
            // The real ordinary chunk worker constructs fixed root vectors for
            // discovery and output completion. Their C shells are already paid
            // by OrdinaryIndexedChunkSource; only observed vector backing joins
            // this numerical source. The actual Eval sees the enclosing DAG.
            for roots in [2, 1] {
                let mut root_vector = OrdinaryNativeControls::default();
                checked(root_vector.include(source(
                    OperationEvent::ordinary_array_vector_control_layout(roots),
                )?))?;
                facts.caller_native_controls = checked(
                    facts
                        .caller_native_controls
                        .append(checked(root_vector.repeat(occurrences))?),
                )?;
            }
            facts.discovery_completions =
                checked(facts.discovery_completions.checked_add(occurrences))?;
        }
        Ok(facts)
    }
}
impl Census {
    fn candidate(&mut self) -> Result<(), NativeError> {
        self.program.native.construction_entries =
            checked(self.program.native.construction_entries.checked_add(1))?;
        Ok(())
    }
    fn copy(
        &mut self,
        layout: CpuCopyEvalLayout,
        inputs: usize,
        elements: usize,
        dtype: Dtype,
    ) -> Result<(), NativeError> {
        checked(self.program.copy(layout, inputs, elements, dtype))
    }
    fn cast(&mut self, input: Value, dtype: Dtype) -> Result<(), NativeError> {
        if input.dtype == dtype {
            return self.candidate();
        }
        self.copy(
            source(OperationEvent::cpu_cast_layout(
                input.dtype,
                dtype,
                input.rank,
                input.elements,
                false,
            ))?,
            1,
            input.elements,
            dtype,
        )
    }
    fn broadcast(&mut self, input: Value, rank: usize, elements: usize) -> Result<(), NativeError> {
        if input.rank == rank && input.elements == elements {
            return self.candidate();
        }
        self.copy(
            source(OperationEvent::cpu_broadcast_alias_layout(
                input.rank, rank, false,
            ))?,
            1,
            0,
            input.dtype,
        )
    }
    fn full(&mut self, elements: usize) -> Result<(), NativeError> {
        checked(self.program.scalar())?;
        self.broadcast(
            Value {
                dtype: Dtype::Int32,
                rank: 0,
                elements: 1,
            },
            1,
            elements,
        )?;
        self.candidate()?; // full_impl's identity I32 cast.
        self.copy(
            source(OperationEvent::cpu_scalar_full_layout(
                Dtype::Int32,
                1,
                elements,
                false,
            ))?,
            1,
            elements,
            Dtype::Int32,
        )?;
        checked(
            self.program
                .controls(source(super::super::zero_fill::control_bytes())?),
        )
    }
    fn binary(
        &mut self,
        kind: CpuBinaryOperation,
        left: Value,
        right: Value,
    ) -> Result<Value, NativeError> {
        let dtype = Dtype::from_promoting_types(left.dtype, right.dtype);
        let rank = left.rank.max(right.rank);
        let elements = left.elements.max(right.elements);
        self.cast(left, dtype)?;
        self.cast(right, dtype)?;
        self.broadcast(left, rank, elements)?;
        self.broadcast(right, rank, elements)?;
        let native = source(OperationEvent::cpu_binary_layout(
            kind, dtype, rank, elements, false,
        ))?;
        let capacity = source(self.program.capacity(elements, Dtype::Bool))?;
        self.program.bytes = checked(self.program.bytes.checked_add(checked(
            capacity.checked_mul(native.backing_births() as u64),
        )?))?;
        checked(self.program.native.binary(native))?;
        Ok(Value {
            dtype: Dtype::Bool,
            rank,
            elements,
        })
    }
}
impl Visitor for Census {
    type Value = Value;
    // A cold producer supplies the actual lookup span, without manufacturing a
    // payload or treating a descriptive constructor as execution authority.
    type HostI32 = usize;
    fn dtype(&self, value: &Value) -> Dtype {
        value.dtype
    }
    fn elements(&self, value: &Value) -> usize {
        value.elements
    }
    fn host_i32(&mut self, &elements: &usize) -> Result<Value, NativeError> {
        self.calls = checked(self.calls.append(source(OrdinaryCallControls::call(
            safemlx::ops::OrdinaryRecipeCall::IndicesI32 { elements },
        ))?))?;
        self.calls = checked(
            self.calls
                .metadata(source(indexed_numerical::operation_control_bytes())?),
        )?;
        if elements == 0 || elements > i32::MAX as usize {
            return Err(unknown());
        }
        self.program.seeds = checked(self.program.seeds.checked_add(1))?;
        self.program.bytes = checked(
            self.program
                .bytes
                .checked_add(source(self.program.capacity(elements, Dtype::Int32))?),
        )?;
        checked(
            self.program
                .controls(source(super::super::host_array::control_bytes())?),
        )?;
        checked(
            self.program
                .controls(source(indexed_numerical::operation_control_bytes())?),
        )?;
        Ok(Value {
            dtype: Dtype::Int32,
            rank: 1,
            elements,
        })
    }
    fn apply(&mut self, operation: Operation, inputs: &[&Value]) -> Result<Value, NativeError> {
        self.calls = checked(
            self.calls
                .append(source(ordinary_operation_calls(operation))?),
        )?;
        self.calls = checked(
            self.calls
                .metadata(source(indexed_numerical::operation_control_bytes())?),
        )?;
        let value = match operation {
            Operation::Flatten => {
                let input = *inputs[0];
                // The worker passes [-1], so native reshape's literal-shape
                // equality check cannot elide even a rank-one input. Integer
                // descriptors carry no stride proof: retain the same General-
                // copy envelope, whose actual worker may choose an alias.
                self.copy(
                    source(OperationEvent::cpu_reshape_copy_layout(
                        input.rank, 1, false,
                    ))?,
                    1,
                    input.elements,
                    input.dtype,
                )?;
                Value { rank: 1, ..input }
            }
            Operation::Scalar(_) => {
                checked(self.program.scalar())?;
                checked(
                    self.program
                        .controls(source(super::super::basic::scalar_i32_control_bytes())?),
                )?;
                Value {
                    dtype: Dtype::Int32,
                    rank: 0,
                    elements: 1,
                }
            }
            Operation::Less => self.binary(CpuBinaryOperation::Less, *inputs[0], *inputs[1])?,
            Operation::GreaterEqual => {
                self.binary(CpuBinaryOperation::GreaterEqual, *inputs[0], *inputs[1])?
            }
            Operation::LogicalAnd => {
                self.binary(CpuBinaryOperation::LogicalAnd, *inputs[0], *inputs[1])?
            }
            Operation::LogicalNot => {
                let input = *inputs[0];
                self.cast(input, Dtype::Bool)?;
                checked(self.program.unary(
                    CpuUnaryOperation::LogicalNot,
                    Dtype::Bool,
                    input.rank,
                    input.elements,
                ))?;
                Value {
                    dtype: Dtype::Bool,
                    ..input
                }
            }
            Operation::CastI32 => {
                let input = *inputs[0];
                self.cast(input, Dtype::Int32)?;
                Value {
                    dtype: Dtype::Int32,
                    ..input
                }
            }
            Operation::CountNonzero => {
                let input = *inputs[0];
                // bool_condition's borrowed Bool source is cloned once.
                self.program.shells = checked(self.program.shells.checked_add(1))?;
                self.cast(input, Dtype::Int32)?;
                if input.elements != 1 {
                    self.copy(
                        source(OperationEvent::cpu_flat_i32_sum_layout(
                            input.elements,
                            false,
                        ))?,
                        1,
                        1,
                        Dtype::Int32,
                    )?;
                } else {
                    self.candidate()?;
                }
                self.copy(
                    source(OperationEvent::cpu_squeeze_layout(1, false))?,
                    1,
                    0,
                    Dtype::Int32,
                )?;
                Value {
                    dtype: Dtype::Int32,
                    rank: 0,
                    elements: 1,
                }
            }
            Operation::ZerosI32(elements) | Operation::OnesI32(elements) => {
                let elements = usize::try_from(elements).map_err(|_| overflow())?;
                self.full(elements)?;
                Value {
                    dtype: Dtype::Int32,
                    rank: 1,
                    elements,
                }
            }
            Operation::Select => {
                let elements = inputs[0].elements;
                for input in inputs {
                    self.cast(**input, input.dtype)?;
                    self.broadcast(**input, 1, elements)?;
                }
                self.copy(
                    source(OperationEvent::cpu_select_layout(
                        Dtype::Int32,
                        1,
                        elements,
                        false,
                    ))?,
                    3,
                    elements,
                    Dtype::Int32,
                )?;
                Value {
                    dtype: Dtype::Int32,
                    rank: 1,
                    elements,
                }
            }
            Operation::Histogram(bins) => {
                let bins = usize::try_from(bins).map_err(|_| overflow())?;
                self.full(bins)?;
                // ScatterAxis casts its values, broadcasts indices and values,
                // then broadcasts all three while ignoring the take axis.
                // These exact rank-one shapes leave six frontend candidates.
                self.cast(*inputs[0], Dtype::Int32)?;
                for _ in 0..5 {
                    self.candidate()?;
                }
                self.copy(
                    source(OperationEvent::cpu_int32_histogram_layout(
                        Dtype::Int32,
                        bins,
                        inputs[0].elements,
                        false,
                    ))?,
                    3,
                    bins,
                    Dtype::Int32,
                )?;
                self.program.native.maximum_captures = self.program.native.maximum_captures.max(6);
                checked(
                    self.program.controls(
                        size_of::<Vec<i32>>()
                            .checked_add(size_of::<i32>())
                            .ok_or_else(overflow)?,
                    ),
                )?;
                Value {
                    dtype: Dtype::Int32,
                    rank: 1,
                    elements: bins,
                }
            }
            Operation::Copy => {
                let input = *inputs[0];
                self.copy(
                    source(OperationEvent::cpu_copy_alias_layout(input.rank, false))?,
                    1,
                    0,
                    input.dtype,
                )?;
                input
            }
            Operation::Take => {
                let lookup = *inputs[0];
                let indices = *inputs[1];
                self.candidate()?; // flatten of the rank-one lookup.
                self.broadcast(indices, indices.rank, indices.elements)?;
                self.cast(indices, indices.dtype)?;
                self.copy(
                    source(OperationEvent::cpu_gather_layout(
                        Dtype::Int32,
                        indices.dtype,
                        1,
                        indices.rank,
                        lookup.elements,
                        indices.elements,
                        1,
                        false,
                    ))?,
                    2,
                    indices.elements,
                    Dtype::Int32,
                )?;
                self.program.native.maximum_captures = self.program.native.maximum_captures.max(4);
                self.copy(
                    source(OperationEvent::cpu_squeeze_layout(
                        indices.rank.checked_add(1).ok_or_else(overflow)?,
                        false,
                    ))?,
                    1,
                    0,
                    Dtype::Int32,
                )?;
                Value {
                    dtype: Dtype::Int32,
                    ..indices
                }
            }
        };
        checked(
            self.program
                .controls(source(indexed_numerical::operation_control_bytes())?),
        )?;
        Ok(value)
    }
}

fn finish(
    mut census: Census,
    rank: usize,
    outputs: &[Value],
) -> Result<(OperationPlan, OrdinaryCallControls), NativeError> {
    let calls = checked(
        census
            .calls
            .metadata(source(indexed_numerical::traversal_control_bytes())?),
    )?;
    let output_bytes = outputs.iter().try_fold(0u64, |total, output| {
        checked(total.checked_add(source(
            census.program.capacity(output.elements, output.dtype),
        )?))
    })?;
    checked(
        census
            .program
            .controls(source(indexed_numerical::traversal_control_bytes())?),
    )?;
    let p = census.program;
    Ok((
        OperationPlan {
            dtype: WorkspaceFloatingType::Float32,
            population: p.native,
            alias_input: None,
            output_bytes,
            scratch_bytes: checked(p.bytes.checked_sub(output_bytes))?,
            rank,
            parameter_shells: p.shells,
            seeds: p.seeds,
            validations: p.validations,
        },
        calls,
    ))
}
pub(super) fn discovery(
    mechanism: MlxCpuWorkspaceMechanisms,
    input: Value,
    upper: usize,
) -> Result<OperationPlan, NativeError> {
    discovery_with_calls(mechanism, input, upper).map(|(plan, _)| plan)
}
fn discovery_with_calls(
    mechanism: MlxCpuWorkspaceMechanisms,
    input: Value,
    upper: usize,
) -> Result<(OperationPlan, OrdinaryCallControls), NativeError> {
    if input.rank > 4 || input.elements == 0 || input.elements > i32::MAX as usize {
        return Err(unknown());
    }
    let mut census = Census {
        program: program::Program::new(mechanism),
        calls: OrdinaryCallControls::default(),
    };
    let result = indexed_numerical::discover(&mut census, &input, upper)?;
    finish(census, input.rank, &[result.histogram, result.invalid])
}
pub(super) fn remap(
    mechanism: MlxCpuWorkspaceMechanisms,
    input: Value,
    lookup: usize,
) -> Result<OperationPlan, NativeError> {
    remap_with_calls(mechanism, input, lookup).map(|(plan, _)| plan)
}
fn remap_with_calls(
    mechanism: MlxCpuWorkspaceMechanisms,
    input: Value,
    lookup: usize,
) -> Result<(OperationPlan, OrdinaryCallControls), NativeError> {
    if input.rank > 4 || input.elements == 0 || input.elements > i32::MAX as usize {
        return Err(unknown());
    }
    let mut census = Census {
        program: program::Program::new(mechanism),
        calls: OrdinaryCallControls::default(),
    };
    let result = indexed_numerical::remap(&mut census, &input, &lookup)?;
    finish(census, checked(input.rank.checked_add(1))?, &[result])
}

pub(super) fn inspect(
    operation: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    if !matches!(
        operation.kind,
        WorkspaceOperationKindView::Elementwise("indexed_bank_discovery")
    ) {
        return Ok(None);
    }
    let Some([input]) = operation.inputs.array() else {
        return Ok(None);
    };
    let Some([histogram, invalid]) = operation.outputs.array() else {
        return Ok(None);
    };
    let dtype = match input.dtype() {
        WorkspaceDtype::Int32 => Dtype::Int32,
        WorkspaceDtype::Uint32 => Dtype::Uint32,
        _ => return Ok(None),
    };
    if input.representation().is_some()
        || histogram.dtype() != WorkspaceDtype::Int32
        || invalid.dtype() != WorkspaceDtype::Int32
        || histogram.shape().len() != 1
        || histogram.shape()[0] <= 0
        || !invalid.shape().is_empty()
    {
        return Ok(None);
    }
    match discovery(
        mechanism,
        Value {
            dtype,
            rank: input.shape().len(),
            elements: usize::try_from(input.elements()?)?,
        },
        histogram.shape()[0] as usize,
    ) {
        Ok(plan) => Ok(Some(plan)),
        Err(NativeError::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
        )) => Ok(None),
        Err(NativeError::PrefillControl(
            eredu_runtime::working_memory::WorkingMemoryError::Overflow,
        )) => Err(MlxWorkspaceFactError::POPULATION_OVERFLOW),
        Err(_) => Err(MlxWorkspaceFactError::descriptor(
            "ordinary indexed discovery source geometry differs",
        )),
    }
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
#[path = "indexed_bank_tests.rs"]
mod tests;

// The same Visitor observes actual safe calls before native lowering. Implicit
// native casts/reshapes are already in the Program and add no C wrappers.
fn ordinary_operation_calls(operation: Operation) -> Option<OrdinaryCallControls> {
    use safemlx::ops::OrdinaryRecipeCall as C;
    let call = OrdinaryCallControls::call;
    match operation {
        Operation::Flatten => call(C::Reshape { rank: 1 }),
        Operation::Scalar(_) => call(C::ScalarI32),
        Operation::Less | Operation::GreaterEqual | Operation::LogicalAnd | Operation::Take => {
            call(C::Binary)
        }
        Operation::LogicalNot | Operation::Copy => call(C::Unary),
        Operation::CastI32 => call(C::Cast),
        Operation::ZerosI32(_) | Operation::OnesI32(_) => call(C::Fill { rank: 1 }),
        Operation::Select => call(C::Select),
        Operation::CountNonzero => call(C::Cast)?.append(call(C::SumAll)?)?.metadata(
            safemlx::Array::ordinary_clone_control_bytes()?
                .checked_add(size_of::<(&safemlx::Array, &safemlx::Stream)>().checked_mul(2)?)?
                .checked_add(
                    size_of::<Result<safemlx::Array, safemlx::error::Exception>>()
                        .checked_mul(2)?,
                )?,
        ),
        Operation::Histogram(_) => {
            // SegmentSum's rank-one path copies its actual shape, creates zeros
            // then calls ScatterAxis. The Cow borrows the unchanged IDs.
            let frames = [
                size_of::<Vec<i32>>(),
                size_of::<i32>(),
                size_of::<std::borrow::Cow<'_, safemlx::Array>>(),
                size_of::<(&safemlx::Array, &safemlx::Array, i32, i32, &safemlx::Stream)>() * 2,
                size_of::<usize>(),
                size_of::<i32>(),
                size_of::<safemlx::Array>(),
                size_of::<Result<safemlx::Array, safemlx::error::Exception>>(),
            ];
            call(C::Fill { rank: 1 })?
                .append(call(C::ScatterAddAxis)?)?
                .metadata(
                    frames
                        .into_iter()
                        .try_fold(size_of_val(&frames), usize::checked_add)?,
                )
        }
    }
}
