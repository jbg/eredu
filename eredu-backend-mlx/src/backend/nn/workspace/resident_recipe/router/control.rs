//! The shared intervention driver composes the same selector stage populations.
use super::*;
use eredu_nn::routing_intervention::{
    execute_routing_intervention_fixed, RoutingMechanism, RoutingRows,
};
use eredu_nn::{GroupSelection, TopKGroupSelectionSpec};
use std::cell::RefCell;

#[derive(Clone, Copy)]
struct Value;
#[derive(Debug, thiserror::Error)]
#[error("controlled router native population overflows")]
struct Missing;
struct Counter {
    phases: Phases,
    policy: TopKGroupSelectionSpec,
    learned: bool,
    rows: u64,
    total: RefCell<Lowering>,
}
impl Counter {
    fn append(&self, value: Lowering) -> Result<(), Missing> {
        let mut sum = self.total.borrow_mut();
        macro_rules! add { ($($name:ident),* $(,)?) => { $(sum.$name = sum.$name.checked_add(value.$name).ok_or(Missing)?;)* }; }
        add!(
            primitives,
            edges,
            seeds,
            validations,
            maximum_births,
            hidden_leaves,
            grouped_output_chunks,
            grouped_unit_observers,
            bf16_projection_calls,
            pointwise_calls,
            row_rms_calls,
            row_sum_calls,
            recurrent_calls,
            router_cpu_partitions,
            nested_completions,
            backend_shells,
            helper_controls
        );
        sum.additional_sort_kernels = Some(
            sum.additional_sort_kernels
                .ok_or(Missing)?
                .checked_add(value.additional_sort_kernels.ok_or(Missing)?)
                .ok_or(Missing)?,
        );
        sum.streams = sum.streams.max(value.streams);
        sum.maximum_operands = sum.maximum_operands.max(value.maximum_operands);
        sum.intermediate_rank = sum.intermediate_rank.max(value.intermediate_rank);
        sum.unqualified_kernel_owner = sum
            .unqualified_kernel_owner
            .or(value.unqualified_kernel_owner);
        Ok(())
    }
    fn primitive(&self, primitives: usize, edges: usize, seeds: usize) -> Result<(), Missing> {
        self.append(Lowering::plain(primitives, edges, seeds))
    }
    fn gathered_keep(&self) -> Result<(), Missing> {
        // Eager Bool keep vector; broadcast/cast/Gather/squeeze; mask Not and Or.
        self.primitive(4 + 2 + 5, 5 + 2 + 6, 1)
    }
    fn predicate(&self) -> Result<bool, Missing> {
        // all(false): Bool conversion, full Reduce, and squeeze. The ordinary
        // reduction source permits its two-pass/compaction backing population.
        let mut value = reduction_lowering(3, 3, 0, 2);
        value.nested_completions = 1;
        self.append(value)?;
        Ok(true)
    }
}
impl RoutingMechanism for Counter {
    type Value = Value;
    type Rows = RoutingRows;
    type Error = Missing;
    fn policy(&self) -> Result<(TopKGroupSelectionSpec, bool), Missing> {
        Ok((self.policy, self.learned))
    }
    fn token_rows(&self, _: &Value) -> Result<u64, Missing> {
        Ok(self.rows)
    }
    fn rows(&self, rows: RoutingRows, _: u64) -> Result<RoutingRows, Missing> {
        // Exact arange/reshape plus ge,lt,sub,remainder,eq and two logical_and
        // calls. Binary promotion/broadcast candidates are the shared 5/6 source.
        self.primitive(2 + 7 * 5, 1 + 7 * 6, 5)?;
        Ok(rows)
    }
    fn project(&mut self, _: &Value) -> Result<Value, Missing> {
        self.append(self.phases.projection)?;
        Ok(Value)
    }
    fn transform(&self, _: &Value) -> Result<Value, Missing> {
        self.append(self.phases.transform)?;
        Ok(Value)
    }
    fn ranking(&self, _: &Value) -> Result<Value, Missing> {
        self.append(self.phases.ranking)?;
        Ok(Value)
    }
    fn select(&self, _: &Value) -> Result<Value, Missing> {
        self.append(self.phases.selection)?;
        Ok(Value)
    }
    fn weights(&self, _: &Value, _: Value) -> Result<GroupSelection<Value>, Missing> {
        self.append(self.phases.weights)?;
        Ok(GroupSelection::new(Value, Value, Value))
    }
    fn replace_rows(&self, _: &Value, _: &[u32], rows: &RoutingRows) -> Result<Value, Missing> {
        self.primitive(1, 1, 1)?; // exact eager forced IDs and dtype boundary
        if rows.first != 0 || rows.end != self.rows || rows.stride != 1 {
            self.primitive(1, 1, 0)?; // same explicit Contiguous source as CPU
            self.primitive(3, 4, 0)?; // same static update cast/broadcast/overwrite
        }
        Ok(Value)
    }
    fn finite(&self, _: &Value) -> Result<bool, Missing> {
        // Existing classification lowering: two scalar comparisons, isnan,
        // two Or operators and Not, sharing exact ordinary primitive bodies.
        self.primitive(27, 32, 2)?;
        self.predicate()
    }
    fn nonnegative(&self, _: &Value) -> Result<bool, Missing> {
        self.primitive(5, 6, 1)?;
        self.predicate()
    }
    fn positive_row_sums(&self, _: &Value) -> Result<bool, Missing> {
        self.append(reduction_lowering(3, 3, 0, 2))?;
        self.primitive(5, 6, 1)?;
        self.predicate()
    }
    fn add_columns(
        &self,
        _: &Value,
        _: &[u32],
        _: &[f32],
        _: &RoutingRows,
    ) -> Result<Value, Missing> {
        // Eager F32 correction, dtype boundary, Add and row-conditioned Select.
        self.primitive(1 + 5 + 7, 1 + 6 + 9, 1)?;
        Ok(Value)
    }
    fn fill_columns(
        &self,
        _: &Value,
        _: &[u32],
        _: f32,
        _: &RoutingRows,
    ) -> Result<Value, Missing> {
        // Eager Bool keep; reshape, Not, Or, typed fill and Select.
        self.primitive(1 + 2 + 5 + 1 + 7, 1 + 2 + 6 + 1 + 9, 2)?;
        Ok(Value)
    }
    fn fill_gathered(
        &self,
        _: &Value,
        _: &Value,
        _: &[u32],
        _: f32,
        _: &RoutingRows,
    ) -> Result<Value, Missing> {
        self.gathered_keep()?;
        self.primitive(1 + 7, 1 + 9, 1)?;
        Ok(Value)
    }
    fn excludes(&self, _: &Value, _: &[u32], _: &RoutingRows) -> Result<bool, Missing> {
        self.gathered_keep()?;
        self.predicate()
    }
}
pub(super) fn population(
    operation: WorkspaceOperationView<'_>,
    require_original_source: bool,
) -> Option<Lowering> {
    let WorkspaceOperationKindView::GroupSelection {
        spec,
        supplied_indices: false,
        control: Some(control),
    } = operation.kind
    else {
        return None;
    };
    let outputs = operation.outputs.slice(0..3)?;
    let phases = super::phases(
        WorkspaceOperationView {
            kind: WorkspaceOperationKindView::GroupSelection {
                spec,
                supplied_indices: false,
                control: None,
            },
            inputs: operation.inputs,
            outputs,
        },
        require_original_source,
    )?;
    let rows = operation
        .inputs
        .get(0)?
        .elements()
        .ok()?
        .checked_div(spec.input_dimensions() as u64)?;
    let mut counter = Counter {
        phases,
        policy: spec.selection(),
        learned: spec.coefficient_scale().is_some(),
        rows,
        total: RefCell::new(Lowering::plain(0, 0, 0)),
    };
    execute_routing_intervention_fixed(&mut counter, &Value, control).ok()?;
    let decisions = 1 + usize::from(control.capture_original);
    counter.primitive(decisions, decisions, 0).ok()?; // final output precision boundary
    let mut total = counter.total.into_inner();
    total.helper_controls = total.helper_controls.checked_add(
        crate::backend::nn::grouped::TopKGroupSelector::intervention_control_bytes()?,
    )?;
    // NativeRouting::transform's raw alias, the driver's shared score/ranking
    // pair, selected-score alias per decision, and its retained original result.
    total.backend_shells = total.backend_shells.checked_add(3 + decisions)?;
    total.helper_controls = total
        .helper_controls
        .checked_add(std::mem::size_of::<Counter>())?
        .checked_add(std::mem::size_of::<Phases>())?
        .checked_add(std::mem::size_of::<(Value, RoutingRows, Option<Lowering>)>())?;
    Some(total)
}
