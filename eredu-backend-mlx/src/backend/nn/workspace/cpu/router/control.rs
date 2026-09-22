//! Source census follows the shared intervention driver and ordinary router stages.
use super::*;
use eredu_nn::routing_intervention::{
    execute_routing_intervention_fixed, FixedRoutingExecutionError, GroupSelectionAction,
    GroupSelectionControl, RoutingMechanism, RoutingRows,
};
use eredu_nn::{GroupSelection, TopKGroupSelectionSpec};
use std::cell::{Cell, RefCell};

#[derive(Clone, Copy)]
struct Value(usize);
#[derive(Debug, thiserror::Error)]
#[error("the selected CPU routing primitive has no qualified source")]
struct Missing;
struct Counter<'a> {
    program: RefCell<Program>,
    completions: Cell<usize>,
    op: WorkspaceOperationView<'a>,
    geometry: Geometry<'a>,
}
impl Counter<'_> {
    fn step(&self, f: impl FnOnce(&mut Program) -> Option<()>) -> Result<(), Missing> {
        f(&mut self.program.borrow_mut()).ok_or(Missing)
    }
    fn predicate(&self, elements: usize) -> Result<bool, Missing> {
        self.step(|p| {
            p.cast(Dtype::Bool, Dtype::Bool, 2, elements)?;
            p.copy(
                OperationEvent::cpu_boolean_reduce_layout(true, 2, elements, false)?,
                1,
                1,
                Dtype::Bool,
            )?;
            p.copy(
                OperationEvent::cpu_squeeze_layout(2, false)?,
                1,
                0,
                Dtype::Bool,
            )
        })?;
        self.completions
            .set(self.completions.get().checked_add(1).ok_or(Missing)?);
        Ok(true)
    }
    fn keep(&self) -> Result<(), Missing> {
        self.step(|p| eager(p, self.geometry.groups, Dtype::Bool))
    }
    fn inverted_rows(&self) -> Result<(), Missing> {
        self.step(|p| {
            p.cast(Dtype::Bool, Dtype::Bool, 2, self.geometry.rows)?;
            p.unary(
                CpuUnaryOperation::LogicalNot,
                Dtype::Bool,
                2,
                self.geometry.rows,
            )
        })
    }
    fn gathered_keep(&self, n: usize) -> Result<(), Missing> {
        self.keep()?;
        self.step(|p| {
            // take_axis's exact one-index Gather and final squeeze.
            p.broadcast(2, 2)?;
            p.cast(Dtype::Uint32, Dtype::Uint32, 2, n)?;
            p.copy(
                OperationEvent::cpu_gather_layout(
                    Dtype::Bool,
                    Dtype::Uint32,
                    1,
                    2,
                    self.geometry.groups,
                    n,
                    1,
                    false,
                )?,
                2,
                n,
                Dtype::Bool,
            )?;
            p.copy(
                OperationEvent::cpu_squeeze_layout(3, false)?,
                1,
                0,
                Dtype::Bool,
            )?;
            p.native.maximum_captures = p.native.maximum_captures.max(4);
            Some(())
        })?;
        self.inverted_rows()?;
        self.step(|p| {
            p.binary(
                CpuBinaryOperation::LogicalOr,
                Dtype::Bool,
                2,
                n,
                2,
                2,
                n,
                self.geometry.rows,
            )
        })
    }
}
fn eager(p: &mut Program, elements: usize, dtype: Dtype) -> Option<()> {
    p.controls(super::super::super::host_array::control_bytes()?)?;
    p.seeds = p.seeds.checked_add(1)?;
    p.bytes = p.bytes.checked_add(p.capacity(elements, dtype)?)?;
    Some(())
}
fn select(
    p: &mut Program,
    elements: usize,
    condition: usize,
    other_rank: usize,
    other_elements: usize,
) -> Option<()> {
    p.cast(Dtype::Bool, Dtype::Bool, 2, condition)?;
    p.cast(Dtype::Float32, Dtype::Float32, 2, elements)?;
    p.cast(Dtype::Float32, Dtype::Float32, other_rank, other_elements)?;
    p.broadcast(2, 2)?;
    p.broadcast(2, 2)?;
    p.broadcast(other_rank, 2)?;
    p.copy(
        OperationEvent::cpu_typed_select_broadcast_layout(Dtype::Float32, 2, elements, false)?,
        3,
        elements,
        Dtype::Float32,
    )
}
fn scalar_fill(p: &mut Program) -> Option<()> {
    p.scalar()?;
    p.controls(basic::scalar_f32_control_bytes()?)?;
    p.cast(Dtype::Float32, Dtype::Float32, 0, 1)
}
fn compare(
    p: &mut Program,
    kind: CpuBinaryOperation,
    dtype: Dtype,
    n: usize,
    left_rank: usize,
    left_n: usize,
    right_rank: usize,
    right_n: usize,
) -> Option<()> {
    p.cast(dtype, dtype, left_rank, left_n)?;
    p.cast(dtype, dtype, right_rank, right_n)?;
    p.broadcast(left_rank, 2)?;
    p.broadcast(right_rank, 2)?;
    let source = OperationEvent::cpu_binary_layout(kind, dtype, 2, n, false)?;
    p.bytes = p.bytes.checked_add(
        p.capacity(n, Dtype::Bool)?
            .checked_mul(source.backing_births() as u64)?,
    )?;
    p.native.binary(source)
}
impl RoutingMechanism for Counter<'_> {
    type Value = Value;
    type Rows = RoutingRows;
    type Error = Missing;
    fn policy(&self) -> Result<(TopKGroupSelectionSpec, bool), Missing> {
        let spec = self.geometry.spec.ok_or(Missing)?;
        Ok((spec.selection(), spec.coefficient_scale().is_some()))
    }
    fn token_rows(&self, _: &Value) -> Result<u64, Missing> {
        Ok(self.geometry.rows as u64)
    }
    fn rows(&self, rows: RoutingRows, _: u64) -> Result<RoutingRows, Missing> {
        self.step(|p| {
            let n = self.geometry.rows;
            p.copy(
                OperationEvent::cpu_arange_int_layout(Dtype::Int32, n, false)?,
                0,
                n,
                Dtype::Int32,
            )?;
            p.reshape(1, 2)?;
            for kind in [CpuBinaryOperation::GreaterEqual, CpuBinaryOperation::Less] {
                p.scalar()?;
                compare(p, kind, Dtype::Int32, n, 2, n, 0, 1)?;
            }
            p.binary(
                CpuBinaryOperation::LogicalAnd,
                Dtype::Bool,
                2,
                n,
                2,
                2,
                n,
                n,
            )?;
            p.scalar_binary(CpuBinaryOperation::Subtract, Dtype::Int32, 2, n)?;
            p.scalar_binary(CpuBinaryOperation::Remainder, Dtype::Int32, 2, n)?;
            p.scalar()?;
            compare(p, CpuBinaryOperation::Equal, Dtype::Int32, n, 2, n, 0, 1)?;
            p.binary(
                CpuBinaryOperation::LogicalAnd,
                Dtype::Bool,
                2,
                n,
                2,
                2,
                n,
                n,
            )
        })?;
        Ok(rows)
    }
    fn project(&mut self, _: &Value) -> Result<Value, Missing> {
        self.step(|p| selector_projection(p, self.geometry))?;
        Ok(Value(
            self.geometry.count(self.geometry.groups).ok_or(Missing)?,
        ))
    }
    fn transform(&self, value: &Value) -> Result<Value, Missing> {
        self.step(|p| selector_transform(p, self.geometry))?;
        Ok(*value)
    }
    fn ranking(&self, value: &Value) -> Result<Value, Missing> {
        self.step(|p| selector_ranking(p, self.geometry))?;
        Ok(*value)
    }
    fn select(&self, _: &Value) -> Result<Value, Missing> {
        self.step(|p| selector_indices(p, self.geometry))?;
        Ok(Value(
            self.geometry.count(self.geometry.selected).ok_or(Missing)?,
        ))
    }
    fn weights(&self, _: &Value, ids: Value) -> Result<GroupSelection<Value>, Missing> {
        self.step(|p| selector_weights(p, self.op, self.geometry))?;
        Ok(GroupSelection::new(ids, ids, ids))
    }
    fn replace_rows(
        &self,
        indices: &Value,
        ids: &[u32],
        rows: &RoutingRows,
    ) -> Result<Value, Missing> {
        self.step(|p| {
            // Borrowed exact IDs are one eager U32 source, never a new host copy.
            p.controls(super::super::super::host_array::control_bytes()?)?;
            p.seeds = p.seeds.checked_add(1)?;
            p.bytes = p.bytes.checked_add(p.capacity(ids.len(), Dtype::Uint32)?)?;
            p.cast(Dtype::Uint32, Dtype::Uint32, 2, ids.len())?;
            if rows.first == 0 && rows.end == self.geometry.rows as u64 && rows.stride == 1 {
                return Some(());
            }
            // The selector's partition slice can retain the wider ranking row.
            // Explicit compaction makes the existing static overwrite source exact.
            p.copy(
                OperationEvent::cpu_contiguous_layout(2, false)?,
                1,
                indices.0,
                Dtype::Uint32,
            )?;
            p.copy(
                OperationEvent::cpu_static_update_layout(2, indices.0, ids.len(), false)?,
                2,
                indices.0,
                Dtype::Uint32,
            )
        })?;
        Ok(*indices)
    }
    fn finite(&self, value: &Value) -> Result<bool, Missing> {
        self.step(|p| {
            let shape = [
                self.geometry.rows as i32,
                (value.0 / self.geometry.rows) as i32,
            ];
            let input = WorkspaceLayoutView::new(&shape, WorkspaceDtype::Float32)
                .ok()?
                .with_representation(Some(WorkspaceRepresentation::new(
                    WorkspaceFloatingType::Float32,
                    true,
                )));
            let output = WorkspaceLayoutView::new(&shape, WorkspaceDtype::Bool).ok()?;
            let inputs = [input];
            let outputs = [output];
            p.child(
                classification::inspect(
                    WorkspaceOperationView {
                        kind: WorkspaceOperationKindView::Elementwise("is_finite"),
                        inputs: WorkspaceLayoutList::Views(&inputs),
                        outputs: WorkspaceLayoutList::Views(&outputs),
                    },
                    p.mechanism,
                )
                .ok()??,
            )
        })?;
        self.predicate(value.0)
    }
    fn nonnegative(&self, value: &Value) -> Result<bool, Missing> {
        self.step(|p| {
            p.scalar()?;
            compare(
                p,
                CpuBinaryOperation::GreaterEqual,
                Dtype::Float32,
                value.0,
                2,
                value.0,
                0,
                1,
            )
        })?;
        self.predicate(value.0)
    }
    fn positive_row_sums(&self, value: &Value) -> Result<bool, Missing> {
        self.step(|p| {
            p.router_sum(self.geometry.rows, self.geometry.selected)?;
            // Native sum_axis(false) removes the selected axis before comparison.
            p.copy(
                OperationEvent::cpu_squeeze_layout(2, false)?,
                1,
                0,
                Dtype::Float32,
            )?;
            p.scalar()?;
            p.broadcast(0, 1)?;
            let source = OperationEvent::cpu_binary_layout(
                CpuBinaryOperation::Greater,
                Dtype::Float32,
                1,
                self.geometry.rows,
                false,
            )?;
            p.bytes = p
                .bytes
                .checked_add(p.capacity(self.geometry.rows, Dtype::Bool)?)?;
            p.native.binary(source)?;
            p.copy(
                OperationEvent::cpu_boolean_reduce_layout(true, 1, self.geometry.rows, false)?,
                1,
                1,
                Dtype::Bool,
            )?;
            p.copy(
                OperationEvent::cpu_squeeze_layout(1, false)?,
                1,
                0,
                Dtype::Bool,
            )
        })?;
        let _ = value;
        self.completions
            .set(self.completions.get().checked_add(1).ok_or(Missing)?);
        Ok(true)
    }
    fn add_columns(
        &self,
        value: &Value,
        _: &[u32],
        _: &[f32],
        _: &RoutingRows,
    ) -> Result<Value, Missing> {
        self.step(|p| {
            eager(p, self.geometry.groups, Dtype::Float32)?;
            p.cast(Dtype::Float32, Dtype::Float32, 2, self.geometry.groups)?;
            p.binary(
                CpuBinaryOperation::Add,
                Dtype::Float32,
                2,
                value.0,
                2,
                2,
                value.0,
                self.geometry.groups,
            )?;
            select(p, value.0, self.geometry.rows, 2, value.0)
        })?;
        Ok(*value)
    }
    fn fill_columns(
        &self,
        value: &Value,
        _: &[u32],
        _: f32,
        _: &RoutingRows,
    ) -> Result<Value, Missing> {
        self.keep()?;
        self.step(|p| p.reshape(1, 2))?;
        self.inverted_rows()?;
        self.step(|p| {
            p.binary(
                CpuBinaryOperation::LogicalOr,
                Dtype::Bool,
                2,
                value.0,
                2,
                2,
                self.geometry.groups,
                self.geometry.rows,
            )?;
            scalar_fill(p)?;
            select(p, value.0, value.0, 0, 1)
        })?;
        Ok(*value)
    }
    fn fill_gathered(
        &self,
        value: &Value,
        _: &Value,
        _: &[u32],
        _: f32,
        _: &RoutingRows,
    ) -> Result<Value, Missing> {
        self.gathered_keep(value.0)?;
        self.step(|p| {
            scalar_fill(p)?;
            select(p, value.0, value.0, 0, 1)
        })?;
        Ok(*value)
    }
    fn excludes(&self, value: &Value, _: &[u32], _: &RoutingRows) -> Result<bool, Missing> {
        self.gathered_keep(value.0)?;
        self.predicate(value.0)
    }
}
fn selected_control(op: WorkspaceOperationView<'_>) -> Option<&GroupSelectionControl> {
    match op.kind {
        WorkspaceOperationKindView::GroupSelection { control, .. } => control,
        _ => None,
    }
}
fn outputs(
    g: Geometry<'_>,
    control: &GroupSelectionControl,
    a: NativeAllocationFacts,
) -> facts::FactResult<([facts::Output<'static>; 6], usize, u64)> {
    let (original, original_bytes) = storage(g, a)?;
    let selected = a.fixed_buffer_capacity(
        (g.rows as u64)
            .checked_mul(g.selected as u64)
            .and_then(|n| n.checked_mul(4))
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
    )?;
    let mut result = [facts::Output::Allocate(0); 6];
    let offset = if control.capture_original {
        result[..3].copy_from_slice(&original);
        3
    } else {
        0
    };
    let ids = if matches!(control.action, GroupSelectionAction::Force(_)) {
        selected
    } else {
        a.fixed_buffer_capacity(facts::mul(g.rows as u64, g.groups as u64 * 4)?)?
    };
    let independent = g.independent_weights()
        || matches!(control.action, GroupSelectionAction::ZeroContribution(_));
    result[offset] = facts::Output::Allocate(ids);
    result[offset + 1] = facts::Output::Allocate(selected);
    result[offset + 2] = if independent {
        facts::Output::Allocate(selected)
    } else {
        facts::Output::AliasOutput(offset + 1)
    };
    let retained = selected
        .checked_mul(1 + u64::from(independent))
        .and_then(|n| n.checked_add(ids))
        .and_then(|n| {
            n.checked_add(if control.capture_original {
                original_bytes
            } else {
                0
            })
        })
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    Ok((result, offset + 3, retained))
}
fn inspect_complete(
    op: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
    g: Geometry<'_>,
) -> facts::FactResult<Option<(OperationPlan, usize)>> {
    let control = selected_control(op).ok_or_else(invalid)?;
    let mut counter = Counter {
        program: RefCell::new(Program::new(mechanism)),
        completions: Cell::new(0),
        op,
        geometry: g,
    };
    match execute_routing_intervention_fixed(
        &mut counter,
        &Value(g.count(g.width).ok_or_else(invalid)?),
        control,
    ) {
        Ok(_) => (),
        Err(FixedRoutingExecutionError::Native(_)) => return Ok(None),
        Err(FixedRoutingExecutionError::Invalid(cause)) => {
            return Err(MlxWorkspaceFactError::routing_invalid(cause));
        }
    }
    let completions = counter.completions.get();
    let mut p = counter.program.into_inner();
    let decisions = 1 + usize::from(control.capture_original);
    // select_intervened applies its concrete final precision boundary once per output.
    for _ in 0..decisions {
        p.cast(
            Dtype::Float32,
            Dtype::Float32,
            2,
            g.count(g.selected).ok_or_else(invalid)?,
        )
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    }
    let (_, _, retained) = outputs(g, control, mechanism.allocation)?;
    p.controls(
        crate::backend::nn::grouped::TopKGroupSelector::intervention_control_bytes()
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
    )
    .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    p.controls(
        size_of::<Counter<'_>>()
            + size_of::<Value>() * 8
            + size_of::<RoutingRows>() * 2
            + size_of::<[facts::Output<'_>; 6]>()
            + size_of::<OperationPlan>()
            + size_of::<RefCell<Program>>()
            + size_of::<Result<(), Missing>>()
            + size_of::<
                Result<
                    eredu_nn::routing_intervention::IntervenedGroupSelection<Value>,
                    FixedRoutingExecutionError<Missing>,
                >,
            >(),
    )
    .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    Ok(Some((
        OperationPlan {
            dtype: WorkspaceFloatingType::Float32,
            population: p.native,
            alias_input: None,
            output_bytes: retained,
            scratch_bytes: p.bytes.checked_sub(retained).ok_or_else(invalid)?,
            rank: g.rank.max(3),
            parameter_shells: p.shells,
            seeds: p.seeds,
            validations: 0,
        },
        completions,
    )))
}
pub(super) fn inspect(
    op: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
    g: Geometry<'_>,
) -> facts::FactResult<Option<OperationPlan>> {
    Ok(inspect_complete(op, mechanism, g)?.map(|(plan, _)| plan))
}
pub(super) fn nested_completions(
    op: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
    g: Geometry<'_>,
) -> facts::FactResult<usize> {
    inspect_complete(op, mechanism, g)?
        .map(|(_, count)| count)
        .ok_or_else(invalid)
}
pub(super) fn emit_outputs(
    op: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
    g: Geometry<'_>,
    sink: &mut facts::Emitter<'_>,
) -> facts::FactResult<()> {
    let (outputs, count, _) = outputs(
        g,
        selected_control(op).ok_or_else(invalid)?,
        mechanism.allocation,
    )?;
    for output in outputs.into_iter().take(count) {
        sink.output(output)?;
    }
    Ok(())
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests {
    use super::*;
    use eredu_nn::workspace::WorkspaceParameterRepresentation;
    use eredu_nn::{
        GroupSelectionOperator, GroupedNeuralBackend, LinearFormatSpec, ParameterSpec,
        RoutingArithmetic, TopKGroupSelectionSpec,
    };

    #[test]
    fn force_recipe_quotes_partial_rows_original_decision_and_each_validation() {
        if !crate::tests::support::native_process::enter("cpu-controlled-router-source") {
            return;
        }
        let _startup = crate::tests::support::test_utils::initialize_original_sources();
        let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let choice =
            MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), choice);
        for (first, end, stride) in [(0, 3, 1), (1, 3, 1), (0, 3, 2)] {
            for capture_original in [false, true] {
                let context = WorkspaceContext::new(cpu);
                let repr = Some(WorkspaceRepresentation::new(
                    WorkspaceFloatingType::Float32,
                    true,
                ));
                context
                    .install_parameter_representations(vec![WorkspaceParameterRepresentation::new(
                        eredu_nn::ParameterId::new("router.weight").unwrap(),
                        context
                            .layout(&[4, 8], WorkspaceDtype::Float32)
                            .unwrap()
                            .with_representation(repr),
                    )])
                    .unwrap();
                let policy =
                    TopKGroupSelectionSpec::new(4, 2, GroupScoring::Softmax, true).unwrap();
                let spec = TopKGroupSelectorSpec::new(
                    8,
                    ParameterSpec::trainable("router.weight").unwrap(),
                    LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense).unwrap(),
                    policy,
                )
                .unwrap()
                .with_arithmetic(RoutingArithmetic::uniform(RoutingPrecision::Float32));
                let mut selector = WorkspaceBackend::top_k_group_selector(spec, &context).unwrap();
                let input = WorkspaceTensor::existing(
                    context
                        .layout(&[1, 3, 8], WorkspaceDtype::Float32)
                        .unwrap()
                        .with_representation(repr),
                    &context,
                )
                .unwrap();
                let rows = (end - first + stride - 1) / stride;
                let control = GroupSelectionControl {
                    expected: policy,
                    learned_coefficient_scale: false,
                    first_row: first,
                    end_row: end,
                    row_stride: stride,
                    action: GroupSelectionAction::Force([3, 1].repeat(rows as usize)),
                    capture_original,
                };
                context.begin_state_span([&input]).unwrap();
                let decisions = selector
                    .select_intervened(&input, &control, &context)
                    .unwrap();
                let mut outputs = Vec::new();
                if let Some(original) = decisions.original {
                    let (a, b, c) = original.into_parts();
                    outputs.extend([a, b, c]);
                }
                let (a, b, c) = decisions.effective.into_parts();
                outputs.extend([a, b, c]);
                let report = context.finish_report(&outputs).unwrap();
                assert!(report.unpriced_operations.is_empty(), "{report:?}");
                assert!(report.unpriced_host_operations.is_empty(), "{report:?}");
                let recipe = SpeculativeNumericalRecipe::inspect_cpu_outputs(
                    &report,
                    outputs.len(),
                    ordinary,
                    cpu,
                    &context,
                )
                .unwrap();
                assert_eq!(recipe.completion.nested_completions, 5);
                assert!(recipe.storage.maximum_births() > outputs.len());
            }
        }
    }
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod native_tests {
    use super::*;
    use crate::{
        backend::{
            managed_memory::gpu_stream::PreparedExecutionStreams,
            nn::grouped::{TopKGroupScoring, TopKGroupSelector, TopKGroupSelectorConfig},
            MlxBackend, MlxDeviceIdentity,
        },
        MlxTensor,
    };
    use eredu_nn::workspace::WorkspaceParameterRepresentation;
    use eredu_nn::{
        GroupSelectionOperator, GroupedNeuralBackend, LinearFormatSpec, ParameterSpec,
        RoutingArithmetic, TopKGroupSelectionSpec,
    };
    use safemlx::{Array, Device, DeviceType};

    #[test]
    fn original_force_rows_preserve_unselected_values_and_backing_custody() {
        if !crate::tests::support::native_process::enter("cpu-force-original-values") {
            return;
        }
        routing_rows(false, true);
    }

    #[test]
    fn metal_original_force_rows_preserve_unselected_values_and_backing_custody() {
        if !crate::tests::support::native_process::enter("metal-force-original-values") {
            return;
        }
        routing_rows(true, true);
    }

    #[test]
    fn cpu_mask_and_score_edits_preserve_nonzero_values_and_custody() {
        if !crate::tests::support::native_process::enter("cpu-routing-actions") {
            return;
        }
        routing_rows(false, false);
    }
    #[test]
    fn metal_mask_and_score_edits_preserve_nonzero_values_and_custody() {
        if !crate::tests::support::native_process::enter("metal-routing-actions") {
            return;
        }
        routing_rows(true, false);
    }
    fn action(kind: usize, count: usize) -> GroupSelectionAction {
        use eredu_nn::routing_intervention::GroupScoreStage as Stage;
        match kind {
            0 => GroupSelectionAction::Force([1, 0].repeat(count)),
            1 => GroupSelectionAction::Exclude(vec![3]),
            2 => GroupSelectionAction::ZeroContribution(vec![3]),
            3 | 4 | 5 => GroupSelectionAction::Bias {
                stage: match kind {
                    3 => Stage::RawLogits,
                    4 => Stage::TransformedScores,
                    _ => Stage::RankingScores,
                },
                ids: vec![0],
                values: vec![if kind == 3 { 10.0 } else { 2.0 }],
            },
            _ => unreachable!(),
        }
    }
    fn routing_rows(metal: bool, force_only: bool) {
        let startup = crate::tests::support::test_utils::initialize_original_sources();
        let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let choice =
            MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
        let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), choice);
        let streams = if metal {
            PreparedExecutionStreams::for_factory(&startup)
        } else {
            PreparedExecutionStreams::for_cpu_factory_with_matmul(&startup, choice)
        }
        .unwrap()
        .unwrap();
        let backend = MlxBackend::for_prepared_execution_plan(
            streams,
            MlxDeviceIdentity::from_realized_device(
                &Device::new(
                    if metal {
                        DeviceType::Gpu
                    } else {
                        DeviceType::Cpu
                    },
                    0,
                ),
                metal.then_some(crate::backend::MlxAcceleratorFamily::Metal),
            )
            .unwrap(),
        );
        let environment = backend.original_copy_environment().unwrap();
        let stream = environment.stream();
        let input_data =
            std::array::from_fn::<_, 24, _>(|n| (n % 8 + 1) as f32 / 16.0 + (n / 8) as f32 / 8.0);
        let weight_data =
            std::array::from_fn::<_, 32, _>(|n| ((n / 8 + 1) * (n % 8 + 1)) as f32 / 64.0);
        let input = MlxTensor::from_array(Array::from_slice(&input_data, &[1, 3, 8]));
        let weight = MlxTensor::from_array(Array::from_slice(&weight_data, &[4, 8]));
        input.as_array().evaluated().unwrap();
        weight.as_array().evaluated().unwrap();
        let arithmetic = RoutingArithmetic::uniform(RoutingPrecision::Float32);
        let policy = TopKGroupSelectionSpec::new(4, 2, GroupScoring::Softmax, true).unwrap();
        for kind in if force_only { 0..1 } else { 0..6 } {
            for capture_original in [false, true] {
                for (first, end, stride) in [(0, 3, 1), (1, 3, 1), (0, 3, 2)] {
                    let count = (end - first + stride - 1) / stride;
                    let control = GroupSelectionControl {
                        expected: policy,
                        learned_coefficient_scale: false,
                        first_row: first,
                        end_row: end,
                        row_stride: stride,
                        action: action(kind, count as usize),
                        capture_original,
                    };
                    let context = if metal {
                        WorkspaceContext::new(ordinary)
                    } else {
                        WorkspaceContext::new(cpu)
                    };
                    let repr = Some(WorkspaceRepresentation::new(
                        WorkspaceFloatingType::Float32,
                        true,
                    ));
                    context
                        .install_parameter_representations(vec![
                            WorkspaceParameterRepresentation::new(
                                eredu_nn::ParameterId::new("router.weight").unwrap(),
                                context
                                    .layout(&[4, 8], WorkspaceDtype::Float32)
                                    .unwrap()
                                    .with_representation(repr),
                            ),
                        ])
                        .unwrap();
                    let spec = TopKGroupSelectorSpec::new(
                        8,
                        ParameterSpec::trainable("router.weight").unwrap(),
                        LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense).unwrap(),
                        policy,
                    )
                    .unwrap()
                    .with_arithmetic(arithmetic);
                    let mut symbolic =
                        WorkspaceBackend::top_k_group_selector(spec, &context).unwrap();
                    let source = WorkspaceTensor::existing(
                        context
                            .layout(&[1, 3, 8], WorkspaceDtype::Float32)
                            .unwrap()
                            .with_representation(repr),
                        &context,
                    )
                    .unwrap();
                    context.begin_state_span([&source]).unwrap();
                    let decisions = symbolic
                        .select_intervened(&source, &control, &context)
                        .unwrap();
                    let mut outputs = Vec::new();
                    if let Some(original) = decisions.original {
                        let (a, b, c) = original.into_parts();
                        outputs.extend([a, b, c]);
                    }
                    let (a, b, c) = decisions.effective.into_parts();
                    outputs.extend([a, b, c]);
                    let report = context.finish_report(&outputs).unwrap();
                    assert!(report.unpriced_operations.is_empty(), "{report:?}");
                    assert_eq!(report.host_workspace_bytes,
                        Some(match kind { 0 | 1 | 2 => 4, _ => 16 }),
                        "actual maximum of partition-ID, keep-mask and correction Host payloads, action {kind}");
                    let recipe = if metal {
                        SpeculativeNumericalRecipe::inspect_completed_outputs(
                            &report,
                            outputs.len(),
                            crate::backend::nn::workspace::ResidentExecutionMechanisms::Metal(
                                ordinary,
                            ),
                            &context,
                        )
                    } else {
                        SpeculativeNumericalRecipe::inspect_cpu_outputs(
                            &report,
                            outputs.len(),
                            ordinary,
                            cpu,
                            &context,
                        )
                    }
                    .unwrap();
                    let config = TopKGroupSelectorConfig::new(
                        2,
                        4,
                        8,
                        TopKGroupScoring::Softmax,
                        true,
                        0.0,
                        1.0,
                        1,
                        1,
                        false,
                        false,
                        None,
                        false,
                        false,
                    )
                    .unwrap()
                    .with_arithmetic(arithmetic);
                    let mut native =
                        TopKGroupSelector::new_with_quantization(config, None, stream).unwrap();
                    native.weight.value = weight.as_array().clone();
                    assert_eq!(
                        recipe.completion.nested_completions,
                        match kind {
                            0 => 5,
                            2 => 4,
                            _ => 6,
                        },
                        "action {kind}, Metal {metal}"
                    );
                    if capture_original {
                        run::<6>(
                            recipe,
                            &backend,
                            &input,
                            &weight,
                            &mut native,
                            &control,
                            &input_data,
                            &weight_data,
                            kind,
                        );
                    } else {
                        run::<3>(
                            recipe,
                            &backend,
                            &input,
                            &weight,
                            &mut native,
                            &control,
                            &input_data,
                            &weight_data,
                            kind,
                        );
                    }
                }
            }
        }
    }
    fn run<const N: usize>(
        recipe: SpeculativeNumericalRecipe,
        backend: &MlxBackend<'_>,
        input: &MlxTensor,
        weight: &MlxTensor,
        native: &mut TopKGroupSelector,
        control: &GroupSelectionControl,
        input_data: &[f32; 24],
        weight_data: &[f32; 32],
        kind: usize,
    ) {
        super::super::super::test_execution::run_many(
            recipe,
            backend,
            &[input, weight],
            |stream| {
                let (original, effective) = native
                    .select_intervened(input.as_array(), control, stream)
                    .unwrap();
                let mut values = original
                    .into_iter()
                    .chain(std::iter::once(effective))
                    .flat_map(|decision| [decision.indices, decision.scores, decision.weights]);
                std::array::from_fn::<_, N, _>(|_| MlxTensor::from_array(values.next().unwrap()))
            },
            |actual| {
                for decision in 0..N / 3 {
                    let ids = actual[decision * 3]
                        .as_array()
                        .evaluated()
                        .unwrap()
                        .try_to_vec::<u32>()
                        .unwrap();
                    let scores = actual[decision * 3 + 1]
                        .as_array()
                        .evaluated()
                        .unwrap()
                        .try_to_vec::<f32>()
                        .unwrap();
                    let weights = actual[decision * 3 + 2]
                        .as_array()
                        .evaluated()
                        .unwrap()
                        .try_to_vec::<f32>()
                        .unwrap();
                    for row in 0..3 {
                        let changed = (!control.capture_original || decision == 1)
                            && row as u64 >= control.first_row
                            && (row as u64) < control.end_row
                            && (row as u64 - control.first_row) % control.row_stride == 0;
                        let mut logits = std::array::from_fn::<_, 4, _>(|group| {
                            (0..8)
                                .map(|column| {
                                    input_data[row * 8 + column] as f64
                                        * weight_data[group * 8 + column] as f64
                                })
                                .sum::<f64>()
                        });
                        if changed && kind == 3 {
                            logits[0] += 10.0;
                        }
                        let sum = logits.iter().map(|value| value.exp()).sum::<f64>();
                        let mut probabilities = logits.map(|value| value.exp() / sum);
                        if changed && kind == 4 {
                            probabilities[0] += 2.0;
                        }
                        let mut ranking = probabilities;
                        if changed && kind == 1 {
                            ranking[3] = f64::NEG_INFINITY;
                        }
                        if changed && kind == 5 {
                            ranking[0] += 2.0;
                        }
                        if changed && kind == 0 {
                            assert_eq!(&ids[row * 2..row * 2 + 2], &[1, 0]);
                        } else {
                            let mut ranked = [0u32, 1, 2, 3];
                            ranked.sort_by(|&a, &b| {
                                ranking[b as usize].total_cmp(&ranking[a as usize])
                            });
                            let mut expected = [ranked[0], ranked[1]];
                            expected.sort();
                            let mut found = [ids[row * 2], ids[row * 2 + 1]];
                            found.sort();
                            assert_eq!(found, expected, "action {kind}, row {row}");
                        }
                        let expected = ids[row * 2..row * 2 + 2]
                            .iter()
                            .map(|&id| probabilities[id as usize])
                            .collect::<Vec<_>>();
                        let chosen = expected.iter().sum::<f64>();
                        for slot in 0..2 {
                            let expected_weight =
                                if changed && kind == 2 && ids[row * 2 + slot] == 3 {
                                    0.0
                                } else {
                                    expected[slot] / chosen
                                };
                            assert!(
                                (scores[row * 2 + slot] as f64 - expected[slot]).abs() < 3e-6,
                                "action {kind}, row {row}, score {scores:?} vs {expected:?}"
                            );
                            assert!(
                                (weights[row * 2 + slot] as f64 - expected_weight).abs() < 3e-6,
                                "action {kind}, row {row}, weights {weights:?} vs {expected_weight}"
                            );
                        }
                    }
                }
            },
        );
    }
}
