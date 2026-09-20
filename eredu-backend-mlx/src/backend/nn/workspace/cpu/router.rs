//! CPU sources for the ordinary selected and joint routing equations.
//! Rank selection retains its complete U32 partition backing; score and
//! coefficient aliases remain distinct from their logical selected shapes.
use super::program::Program;
use super::*;
use eredu_nn::{GroupScoring, RoutingPrecision, TopKGroupSelectorSpec};
use safemlx::{CpuUnaryOperation, Dtype};

fn invalid() -> MlxWorkspaceFactError {
    MlxWorkspaceFactError::descriptor("CPU selector source geometry differs")
}
#[derive(Clone, Copy)]
struct Geometry<'a> {
    spec: Option<&'a TopKGroupSelectorSpec>,
    rows: usize,
    width: usize,
    groups: usize,
    selected: usize,
    shared: usize,
    rank: usize,
    supplied: bool,
}
impl Geometry<'_> {
    fn count(self, width: usize) -> Option<usize> {
        self.rows.checked_mul(width)
    }
    fn independent_weights(self) -> bool {
        self.spec.is_none_or(|s| {
            s.selection().normalize_selected()
                || s.selection().coefficient_scale() != 1.0
                || s.coefficient_scale().is_some()
        })
    }
}
fn geometry(op: WorkspaceOperationView<'_>) -> facts::FactResult<Option<Geometry<'_>>> {
    let (spec, supplied, groups, selected, shared) = match op.kind {
        WorkspaceOperationKindView::GroupSelection {
            spec,
            supplied_indices,
            control,
        } => {
            spec.validate_fixed()?;
            if control.is_some()
                || spec.format().encoding() != eredu_checkpoint::LinearFormat::Dense
                || spec.selection().selection_partitions() != 1
                || spec.selection().selected_groups() != 1
            {
                return Ok(None);
            }
            (
                Some(spec),
                supplied_indices,
                spec.selection().group_count(),
                spec.selection().top_k(),
                0,
            )
        }
        WorkspaceOperationKindView::JointGroupSelection(spec) => {
            let Some(g) = super::super::routing::joint_geometry(op)? else {
                return Ok(None);
            };
            (None, false, g.groups, g.selected, spec.always_on_groups())
        }
        _ => return Ok(None),
    };
    if op.outputs.len() != 3 {
        return Err(invalid());
    }
    let input = op.inputs.get(0).ok_or_else(invalid)?;
    let rank = input.shape().len();
    let width = *input.shape().last().ok_or_else(invalid)?;
    if !(1..=4).contains(&rank) || width <= 0 || input.shape().iter().any(|&n| n <= 0) {
        return Ok(None);
    }
    let rows = usize::try_from(input.elements()?)? / usize::try_from(width)?;
    let g = Geometry {
        spec,
        rows,
        width: width as usize,
        groups: groups as usize,
        selected: selected as usize,
        shared: shared as usize,
        rank,
        supplied,
    };
    if let Some(spec) = spec {
        let first = 1 + usize::from(supplied);
        let count = first
            + 1
            + usize::from(spec.bias().is_some())
            + usize::from(spec.correction_bias().is_some())
            + usize::from(spec.input_transform().is_some())
            + usize::from(spec.coefficient_scale().is_some());
        if spec.input_dimensions() != width
            || op.inputs.len() != count
            || op.inputs.get(first).unwrap().shape() != [groups, width]
        {
            return Err(invalid());
        }
        let mut slot = first + 1;
        for present in [spec.bias().is_some(), spec.correction_bias().is_some()] {
            if present {
                if op.inputs.get(slot).unwrap().shape() != [groups] {
                    return Err(invalid());
                }
                slot += 1;
            }
        }
        if spec.input_transform().is_some() {
            if op.inputs.get(slot).unwrap().shape() != [width] {
                return Err(invalid());
            }
            slot += 1;
        }
        if spec.coefficient_scale().is_some() && op.inputs.get(slot).unwrap().shape() != [groups] {
            return Err(invalid());
        }
    }
    for (index, value) in op.inputs.iter().enumerate() {
        if supplied && index == 1 {
            if !matches!(
                value.dtype(),
                WorkspaceDtype::Int32 | WorkspaceDtype::Uint32
            ) || value.shape() != [rows as i32, selected]
            {
                return Err(invalid());
            }
        } else if value.dtype() != WorkspaceDtype::Float32
            || !value
                .representation()
                .is_some_and(|r| r.dtype() == WorkspaceFloatingType::Float32 && r.row_contiguous())
        {
            return Ok(None);
        }
        if value.elements()? > i32::MAX as u64 {
            return Ok(None);
        }
    }
    let ids = if supplied {
        op.inputs.get(1).unwrap().dtype()
    } else {
        WorkspaceDtype::Uint32
    };
    for (index, output) in op.outputs.iter().enumerate() {
        let expected_width = if index == 2 && spec.is_none() {
            shared
        } else {
            selected
        };
        if output.shape() != [rows as i32, expected_width]
            || output.dtype()
                != if index == 0 {
                    ids
                } else {
                    WorkspaceDtype::Float32
                }
        {
            return Err(invalid());
        }
    }
    if rows == 0
        || [
            g.count(g.groups),
            g.count(g.selected + g.shared),
            g.groups.checked_mul(g.width),
        ]
        .into_iter()
        .any(|n| n.is_none_or(|n| n > i32::MAX as usize))
    {
        return Ok(None);
    }
    Ok(Some(g))
}

fn storage(
    g: Geometry<'_>,
    a: NativeAllocationFacts,
) -> facts::FactResult<([facts::Output<'static>; 3], u64)> {
    use facts::Output;
    let bytes = |width| a.fixed_buffer_capacity(facts::mul(g.rows as u64, width as u64 * 4)?);
    if g.spec.is_none() {
        let ids = bytes(g.groups - g.shared)?;
        let coefficients = bytes(g.selected + g.shared)?;
        return Ok((
            [
                Output::Allocate(ids),
                Output::Allocate(coefficients),
                Output::AliasOutput(1),
            ],
            facts::add(ids, coefficients)?,
        ));
    }
    let selected = bytes(g.selected)?;
    let ids = if g.supplied { 0 } else { bytes(g.groups)? };
    let weights = g.independent_weights();
    Ok((
        [
            if g.supplied {
                Output::AliasInput(1)
            } else {
                Output::Allocate(ids)
            },
            Output::Allocate(selected),
            if weights {
                Output::Allocate(selected)
            } else {
                Output::AliasOutput(1)
            },
        ],
        facts::add(ids, facts::mul(selected, 1 + u64::from(weights))?)?,
    ))
}

impl Program {
    fn router_softmax(&mut self, rows: usize, columns: usize) -> Option<()> {
        let count = rows.checked_mul(columns)?;
        self.cast(Dtype::Float32, Dtype::Float32, 2, count)?;
        self.native.maximum_captures = self.native.maximum_captures.max(3);
        self.copy(
            OperationEvent::cpu_softmax_layout(2, columns, rows, false)?,
            1,
            count,
            Dtype::Float32,
        )?;
        self.controls(size_of::<(
            &mut Self,
            usize,
            usize,
            usize,
            Option<CpuCopyEvalLayout>,
            Option<()>,
        )>())
    }
    fn router_sigmoid(&mut self, count: usize, wrapper: bool) -> Option<()> {
        if wrapper {
            self.controls(crate::backend::nn::arithmetic::cpu_pointwise_probe_control_bytes()?)?;
        }
        self.cast(Dtype::Float32, Dtype::Float32, 2, count)?;
        self.unary(CpuUnaryOperation::Sigmoid, Dtype::Float32, 2, count)?;
        if wrapper {
            self.cast(Dtype::Float32, Dtype::Float32, 2, count)?;
        }
        self.controls(size_of::<(&mut Self, usize, bool, Option<usize>, Option<()>)>())
    }
    fn router_softplus(&mut self, count: usize) -> Option<()> {
        self.scalar()?; // layers::softplus's actual I32 zero
        self.cast(Dtype::Int32, Dtype::Float32, 0, 1)?;
        self.binary(
            CpuBinaryOperation::LogAddExp,
            Dtype::Float32,
            2,
            count,
            2,
            0,
            count,
            1,
        )?;
        self.controls(size_of::<(&mut Self, usize, Option<()>)>())
    }
    fn router_partition(&mut self, count: usize) -> Option<()> {
        let source = OperationEvent::cpu_argpartition_source_layout(2, count, false)?;
        self.native.add(CpuPopulation {
            construction_entries: 1,
            primitives: 1,
            input_edges: 1,
            hidden_leaves: 0,
            maximum_operands: 1,
            maximum_captures: 3,
            births: 1,
            extents: source.allocation_extents(),
            controls: source.control_bytes()?,
        })?;
        self.bytes = self
            .bytes
            .checked_add(self.capacity(count, Dtype::Uint32)?)?;
        self.controls(size_of::<(
            &mut Self,
            usize,
            safemlx::CpuArgPartitionLayout,
            Option<()>,
        )>())
    }
    fn router_gather_axis(&mut self, count: usize) -> Option<()> {
        self.broadcast(2, 2)?;
        self.broadcast(2, 2)?;
        self.native.maximum_captures = self.native.maximum_captures.max(4);
        self.copy(
            OperationEvent::cpu_gather_axis_row_layout(2, count, false)?,
            2,
            count,
            Dtype::Float32,
        )?;
        self.controls(size_of::<(
            &mut Self,
            usize,
            Option<CpuCopyEvalLayout>,
            Option<()>,
        )>())
    }
    fn router_sum(&mut self, rows: usize, width: usize) -> Option<()> {
        self.cast(Dtype::Float32, Dtype::Float32, 2, rows.checked_mul(width)?)?;
        if width > 1 {
            self.native.maximum_captures = self.native.maximum_captures.max(3);
            self.copy(
                OperationEvent::cpu_row_sum_layout(2, width, rows, false)?,
                1,
                rows,
                Dtype::Float32,
            )?;
        }
        self.cast(Dtype::Float32, Dtype::Float32, 2, rows)?;
        self.controls(size_of::<(
            &mut Self,
            usize,
            usize,
            Option<usize>,
            Option<CpuCopyEvalLayout>,
            Option<()>,
        )>())
    }
    fn router_projection(&mut self, g: Geometry<'_>) -> Option<()> {
        let input_shape = [g.rows as i32, g.width as i32];
        let weight_shape = [g.groups as i32, g.width as i32];
        let output_shape = [g.rows as i32, g.groups as i32];
        let layout = |shape| {
            WorkspaceLayoutView::new(shape, WorkspaceDtype::Float32)
                .ok()
                .map(|l| {
                    l.with_representation(Some(WorkspaceRepresentation::new(
                        WorkspaceFloatingType::Float32,
                        true,
                    )))
                })
        };
        let inputs = [layout(&input_shape)?, layout(&weight_shape)?];
        let outputs = [layout(&output_shape)?];
        let operation = WorkspaceOperationView {
            kind: WorkspaceOperationKindView::DenseLinear,
            inputs: WorkspaceLayoutList::Views(&inputs),
            outputs: WorkspaceLayoutList::Views(&outputs),
        };
        self.child(dense::inspect(operation, self.mechanism).ok()??)?;
        self.controls(
            size_of::<(&mut Self, Geometry<'_>, Option<()>)>()
                + size_of::<[i32; 2]>() * 3
                + size_of::<[WorkspaceLayoutView<'_>; 2]>()
                + size_of::<[WorkspaceLayoutView<'_>; 1]>()
                + size_of::<WorkspaceOperationView<'_>>()
                + size_of_val(&layout)
                + size_of::<Option<OperationPlan>>()
                + size_of::<facts::FactResult<Option<OperationPlan>>>(),
        )
    }
}

fn selector(p: &mut Program, op: WorkspaceOperationView<'_>, g: Geometry<'_>) -> Option<()> {
    let spec = g.spec?;
    let policy = spec.selection();
    let input = g.count(g.width)?;
    let logits = g.count(g.groups)?;
    let selected = g.count(g.selected)?;
    p.reshape(g.rank, 2)?;
    if let Some(transform) = spec.input_transform() {
        p.unary(CpuUnaryOperation::Square, Dtype::Float32, 2, input)?;
        p.router_sum(g.rows, g.width)?;
        p.scalar()?; // mean_owned constructs its divisor directly in Float32
        p.binary(
            CpuBinaryOperation::Divide,
            Dtype::Float32,
            2,
            g.rows,
            2,
            0,
            g.rows,
            1,
        )?;
        p.scalar_binary(CpuBinaryOperation::Add, Dtype::Float32, 2, g.rows)?;
        p.cast(Dtype::Float32, Dtype::Float32, 2, g.rows)?;
        p.unary(CpuUnaryOperation::Rsqrt, Dtype::Float32, 2, g.rows)?;
        p.binary(
            CpuBinaryOperation::Multiply,
            Dtype::Float32,
            2,
            input,
            2,
            2,
            input,
            g.rows,
        )?;
        p.binary(
            CpuBinaryOperation::Multiply,
            Dtype::Float32,
            2,
            input,
            2,
            1,
            input,
            g.width,
        )?;
        if transform.inverse_sqrt_dimensions() {
            p.scalar_binary(CpuBinaryOperation::Multiply, Dtype::Float32, 2, input)?;
        }
    }
    if spec.arithmetic().projection == RoutingPrecision::Float32 {
        p.cast(Dtype::Float32, Dtype::Float32, 2, input)?;
        p.cast(
            Dtype::Float32,
            Dtype::Float32,
            2,
            g.groups.checked_mul(g.width)?,
        )?;
    } else {
        p.controls(crate::backend::nn::matrix::row_projection_probe_control_bytes()?)?;
    }
    p.router_projection(g)?;
    if spec.bias().is_some() {
        p.binary(
            CpuBinaryOperation::Add,
            Dtype::Float32,
            2,
            logits,
            2,
            1,
            logits,
            g.groups,
        )?;
    }
    p.cast(Dtype::Float32, Dtype::Float32, 2, logits)?; // projection precision
    p.cast(Dtype::Float32, Dtype::Float32, 2, logits)?; // scores precision
    match policy.scoring() {
        GroupScoring::Softmax => p.router_softmax(g.rows, g.groups)?,
        GroupScoring::SelectedSoftmax => {}
        GroupScoring::Sigmoid => p.router_sigmoid(logits, true)?,
        GroupScoring::SqrtSoftplus => {
            p.router_softplus(logits)?;
            p.cast(Dtype::Float32, Dtype::Float32, 2, logits)?;
            p.unary(CpuUnaryOperation::Sqrt, Dtype::Float32, 2, logits)?;
        }
        _ => return None,
    }
    if g.supplied {
        // [-1,k] resolves to the existing [rows,k] shape. The native
        // reshape planner preserves every source stride and its entire Data.
        p.reshape(2, 2)?;
    } else {
        if spec.correction_bias().is_some() {
            p.binary(
                CpuBinaryOperation::Add,
                Dtype::Float32,
                2,
                logits,
                2,
                1,
                logits,
                g.groups,
            )?;
        }
        // CPU uses the unchanged value-only ArgPartition directly. The GPU
        // crossing-tie branch and its second stream are not part of this call.
        p.controls(safemlx::OriginalScopeObserver::control_bytes()?)?;
        p.controls(safemlx::Stream::device_type_control_bytes()?)?;
        p.scalar_binary(CpuBinaryOperation::Multiply, Dtype::Float32, 2, logits)?;
        p.router_partition(logits)?;
        p.slice(2)?;
        p.reshape(2, 2)?;
    }
    p.router_gather_axis(selected)?;
    if policy.scoring() == GroupScoring::SelectedSoftmax {
        p.router_softmax(g.rows, g.selected)?;
    }
    if policy.normalize_selected() {
        p.controls(safemlx::Stream::device_type_control_bytes()?)?;
        p.router_sum(g.rows, g.selected)?;
        if policy.normalization_epsilon() != 0.0 {
            p.scalar_binary(CpuBinaryOperation::Add, Dtype::Float32, 2, g.rows)?;
        }
        p.binary(
            CpuBinaryOperation::Divide,
            Dtype::Float32,
            2,
            selected,
            2,
            2,
            selected,
            g.rows,
        )?;
    }
    if policy.coefficient_scale() != 1.0 {
        p.scalar_binary(CpuBinaryOperation::Multiply, Dtype::Float32, 2, selected)?;
    }
    if spec.coefficient_scale().is_some() {
        let native = if g.supplied && op.inputs.get(1)?.dtype() == WorkspaceDtype::Int32 {
            Dtype::Int32
        } else {
            Dtype::Uint32
        };
        // take_axis receives rank-two IDs; Gather produces [rows,k,1], then
        // squeezes the selected source axis while preserving ID shape.
        p.broadcast(2, 2)?;
        p.cast(native, native, 2, selected)?;
        p.copy(
            OperationEvent::cpu_gather_layout(
                Dtype::Float32,
                native,
                1,
                2,
                g.groups,
                selected,
                1,
                false,
            )?,
            2,
            selected,
            Dtype::Float32,
        )?;
        p.copy(
            OperationEvent::cpu_squeeze_layout(3, false)?,
            1,
            0,
            Dtype::Float32,
        )?;
        p.native.maximum_captures = p.native.maximum_captures.max(4);
        p.binary(
            CpuBinaryOperation::Multiply,
            Dtype::Float32,
            2,
            selected,
            2,
            2,
            selected,
            selected,
        )?;
    }
    p.cast(Dtype::Float32, Dtype::Float32, 2, selected)?;
    p.controls(
        size_of::<(
            &mut Program,
            WorkspaceOperationView<'_>,
            Geometry<'_>,
            Option<()>,
        )>() + size_of::<usize>() * 3
            + size_of::<&TopKGroupSelectorSpec>()
            + size_of::<eredu_nn::TopKGroupSelectionSpec>()
            + size_of::<Dtype>() * 2
            + size_of::<&eredu_nn::SelectorInputTransformSpec>(),
    )
}
fn joint(p: &mut Program, op: WorkspaceOperationView<'_>, g: Geometry<'_>) -> Option<()> {
    let primary = g.groups.checked_sub(g.shared)?;
    let chosen = g.count(g.selected)?;
    let primary_count = g.count(primary)?;
    let joined = g.count(g.selected + g.shared)?;
    p.reshape(g.rank, 2)?;
    p.router_projection(g)?;
    for _ in 0..2 {
        p.slice(2)?;
        p.reshape(2, 2)?;
    }
    p.router_sigmoid(primary_count, false)?;
    p.binary(
        CpuBinaryOperation::Add,
        Dtype::Float32,
        2,
        primary_count,
        2,
        1,
        primary_count,
        primary,
    )?;
    p.router_partition(primary_count)?;
    p.slice(2)?;
    p.reshape(2, 2)?;
    p.router_gather_axis(chosen)?;
    p.cast(Dtype::Float32, Dtype::Float32, 2, chosen)?;
    p.cast(Dtype::Float32, Dtype::Float32, 2, g.count(g.shared)?)?;
    let concat = OperationEvent::cpu_concatenate_many_layout(Dtype::Float32, 2, 2, joined, false)?;
    p.bytes = p.bytes.checked_add(
        p.capacity(joined, Dtype::Float32)?
            .checked_mul(concat.backing_births() as u64)?,
    )?;
    p.native.concatenate(concat, 2)?;
    p.controls(safemlx::ops::concatenate_axis_control_bytes()?)?;
    p.unary(CpuUnaryOperation::Negative, Dtype::Float32, 2, joined)?;
    p.router_softplus(joined)?;
    p.unary(CpuUnaryOperation::Negative, Dtype::Float32, 2, joined)?;
    p.router_softmax(g.rows, g.selected + g.shared)?;
    p.scalar_binary(CpuBinaryOperation::Multiply, Dtype::Float32, 2, joined)?;
    p.binary(
        CpuBinaryOperation::Multiply,
        Dtype::Float32,
        2,
        joined,
        2,
        1,
        joined,
        1,
    )?;
    for _ in 0..2 {
        p.slice(2)?;
        p.reshape(2, 2)?;
    }
    p.controls(crate::backend::nn::grouped::joint_selection_control_bytes()?)?;
    p.controls(
        size_of::<(
            &mut Program,
            WorkspaceOperationView<'_>,
            Geometry<'_>,
            Option<()>,
        )>() + size_of::<usize>() * 4
            + size_of::<CpuCopyEvalLayout>(),
    )?;
    let _ = op;
    Some(())
}

pub(super) fn inspect(
    op: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
) -> facts::FactResult<Option<OperationPlan>> {
    let Some(g) = geometry(op)? else {
        return Ok(None);
    };
    let (_, retained) = storage(g, mechanism.allocation)?;
    let mut p = Program::new(mechanism);
    if (if g.spec.is_some() {
        selector(&mut p, op, g)
    } else {
        joint(&mut p, op, g)
    })
    .is_none()
    {
        return Ok(None);
    }
    let frames = [
        size_of::<WorkspaceOperationView<'_>>(),
        size_of::<MlxCpuWorkspaceMechanisms>(),
        size_of::<Geometry<'_>>() * 2,
        size_of::<Option<Geometry<'_>>>(),
        size_of::<Program>(),
        size_of::<[facts::Output<'_>; 3]>(),
        size_of::<([facts::Output<'_>; 3], u64)>(),
        size_of::<Option<()>>(),
        size_of::<u64>() * 2,
        size_of::<OperationPlan>(),
        size_of::<facts::FactResult<Option<OperationPlan>>>(),
        size_of::<WorkspaceLayoutView<'_>>() * 3,
        size_of::<Option<WorkspaceLayoutView<'_>>>(),
        size_of::<usize>() * 12,
        size_of::<i32>() * 5,
        size_of::<bool>() * 5,
        size_of::<[Option<usize>; 3]>(),
        size_of::<std::array::IntoIter<Option<usize>, 3>>(),
        size_of_val(&op.inputs.iter().enumerate()),
    ];
    p.controls(
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
    )
    .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    if g.spec.is_some() {
        p.controls(
            crate::backend::nn::grouped::TopKGroupSelector::cpu_selection_control_bytes(g.supplied)
                .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?,
        )
        .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
    }
    Ok(Some(OperationPlan {
        dtype: WorkspaceFloatingType::Float32,
        population: p.native,
        alias_input: None,
        output_bytes: retained,
        scratch_bytes: p.bytes.checked_sub(retained).ok_or_else(invalid)?,
        rank: g
            .rank
            .max(if g.spec.is_some_and(|s| s.coefficient_scale().is_some()) {
                3
            } else {
                2
            }),
        parameter_shells: p.shells,
        seeds: p.seeds,
        validations: p.validations,
    }))
}
pub(super) fn emit_outputs(
    op: WorkspaceOperationView<'_>,
    mechanism: MlxCpuWorkspaceMechanisms,
    sink: &mut facts::Emitter<'_>,
) -> facts::FactResult<()> {
    let g = geometry(op)?.ok_or_else(invalid)?;
    for output in storage(g, mechanism.allocation)?.0 {
        sink.output(output)?;
    }
    Ok(())
}
pub(super) fn representation(
    op: WorkspaceOperationView<'_>,
    index: usize,
) -> Option<WorkspaceRepresentation> {
    let g = geometry(op).ok()??;
    if index == 0 || index > 2 {
        return None;
    }
    if g.spec.is_some() {
        return Some(WorkspaceRepresentation::new(
            WorkspaceFloatingType::Float32,
            true,
        ));
    }
    let width = if index == 1 { g.selected } else { g.shared };
    let stride = (g.selected + g.shared) as u64;
    Some(
        WorkspaceRepresentation::new(
            WorkspaceFloatingType::Float32,
            g.rows == 1 || width == g.selected + g.shared,
        )
        .with_last_axis_contiguous(true)
        .with_element_strides(&[stride, 1])?,
    )
}

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod tests;
