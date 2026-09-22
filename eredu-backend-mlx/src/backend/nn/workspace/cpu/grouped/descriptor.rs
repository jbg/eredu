//! Borrowed geometry of the actual dense F32 packed grouped worker.
use super::*;
use eredu_nn::{GatedProductGroupLayout, GroupedProjectionSpec};

#[derive(Clone, Copy)]
pub(super) enum Activation {
    Linear(eredu_nn::GroupedLinearActivation),
    Gated(eredu_nn::GatedProductPolicy),
    Relu2,
}
#[derive(Clone, Copy)]
pub(super) struct Projection {
    pub(super) input: usize,
    pub(super) output: usize,
    pub(super) bias: bool,
}
#[derive(Clone, Copy)]
pub(super) struct Descriptor {
    pub(super) groups: usize,
    pub(super) tokens: usize,
    pub(super) routes: usize,
    pub(super) input: usize,
    pub(super) units: usize,
    pub(super) output: usize,
    pub(super) first: Projection,
    pub(super) down: Option<Projection>,
    pub(super) activation: Activation,
    pub(super) reduction: eredu_nn::GroupReduction,
    pub(super) phase: WorkspaceGroupedPhase,
    pub(super) separate_bias: bool,
    pub(super) input_rank: usize,
    pub(super) index: Dtype,
}
fn floating(value: WorkspaceLayoutView<'_>, contiguous: bool) -> bool {
    value.dtype() == WorkspaceDtype::Float32
        && value.representation().is_some_and(|r| {
            r.dtype() == WorkspaceFloatingType::Float32 && (!contiguous || r.row_contiguous())
        })
}
fn operand_floating(value: WorkspaceLayoutView<'_>, contiguous: bool) -> bool {
    if contiguous {
        return floating(value, true);
    }
    // The Metal caller accepts any supported floating operand and strides.
    // Each selected parameter bank below must independently prove F32: this
    // excludes the BF16 row kernel and makes GatherMM's promoted result F32.
    // This does not establish a new representation on the source layout.
    value.dtype() == WorkspaceDtype::Float32
}
fn projection(
    spec: &GroupedProjectionSpec,
    groups: usize,
    input: usize,
    output: usize,
    operation: WorkspaceOperationView<'_>,
    slot: &mut usize,
    contiguous: bool,
) -> facts::FactResult<Option<Projection>> {
    if spec.format().encoding() != eredu_checkpoint::LinearFormat::Dense
        || spec.format().scale().is_some()
        || spec.format().affine_bias().is_some()
    {
        return Ok(None);
    }
    let weight = operation.inputs.get(*slot).ok_or_else(invalid)?;
    *slot += 1;
    if weight.shape() != [groups as i32, output as i32, input as i32] {
        return Err(invalid());
    }
    if !floating(weight, contiguous) {
        return Ok(None);
    }
    let bias = spec.bias().is_some();
    if bias {
        let value = operation.inputs.get(*slot).ok_or_else(invalid)?;
        *slot += 1;
        if value.shape() != [groups as i32, output as i32] {
            return Err(invalid());
        }
        if !floating(value, contiguous) {
            return Ok(None);
        }
    }
    Ok(Some(Projection {
        input,
        output,
        bias,
    }))
}
impl Descriptor {
    pub(super) fn inspect(
        operation: WorkspaceOperationView<'_>,
    ) -> facts::FactResult<Option<Self>> {
        Self::inspect_layout(operation, true)
    }
    /// Geometry for the same positive F32 callers after independent Metal
    /// allocation/dispatch qualification. Their safe signatures accept strides;
    /// native internal copies stay in that selected Metal allocation envelope.
    pub(super) fn inspect_metal_callers(
        operation: WorkspaceOperationView<'_>,
    ) -> facts::FactResult<Option<Self>> {
        Self::inspect_layout(operation, false)
    }
    fn inspect_layout(
        operation: WorkspaceOperationView<'_>,
        contiguous: bool,
    ) -> facts::FactResult<Option<Self>> {
        let WorkspaceOperationKindView::Grouped {
            bank,
            phase,
            partitions,
        } = operation.kind
        else {
            return Ok(None);
        };
        if partitions == Some(0) || operation.inputs.len() < 4 {
            return Err(invalid());
        }
        let (groups, input, units, output, first, down, activation, reduction) = match bank {
            WorkspaceGroupedBank::Linear(spec) => {
                spec.validate_fixed()?;
                if phase != WorkspaceGroupedPhase::Whole || partitions.is_some() {
                    return Ok(None);
                }
                (
                    spec.group_count(),
                    spec.input_dimensions(),
                    spec.output_dimensions(),
                    spec.output_dimensions(),
                    spec.projection(),
                    None,
                    Activation::Linear(spec.activation()),
                    spec.reduction(),
                )
            }
            WorkspaceGroupedBank::GatedProduct(spec) => {
                spec.validate_fixed()?;
                if spec.input_dimensions() != spec.output_dimensions() {
                    return Ok(None);
                }
                let GatedProductGroupLayout::Packed { gate_up, down } = spec.layout() else {
                    return Ok(None);
                };
                (
                    spec.group_count(),
                    spec.input_dimensions(),
                    spec.intermediate_dimensions(),
                    spec.output_dimensions(),
                    gate_up,
                    Some(down),
                    Activation::Gated(spec.policy()),
                    spec.reduction(),
                )
            }
            WorkspaceGroupedBank::Relu2(spec) => {
                spec.validate_fixed()?;
                if spec.up().bias().is_some() || spec.down().bias().is_some() {
                    return Ok(None);
                }
                (
                    spec.group_count(),
                    spec.hidden_dimensions(),
                    spec.intermediate_dimensions(),
                    spec.hidden_dimensions(),
                    spec.up(),
                    Some(spec.down()),
                    Activation::Relu2,
                    eredu_nn::GroupReduction::Sum,
                )
            }
        };
        if [groups, input, units, output].iter().any(|n| *n <= 0) {
            return Ok(None);
        }
        let (groups, input, units, output) = (
            groups as usize,
            input as usize,
            units as usize,
            output as usize,
        );
        let source = operation.inputs.get(0).ok_or_else(invalid)?;
        if matches!(activation, Activation::Linear(_)) && source.shape().len() != 2 {
            return Ok(None);
        }
        let ids = operation.inputs.get(1).ok_or_else(invalid)?;
        if !(2..=4).contains(&source.shape().len())
            || source.shape().last() != Some(&(input as i32))
            || source.shape().iter().any(|n| *n <= 0)
            || ids.shape().len() != 2
            || ids.shape()[1] <= 0
        {
            return Err(invalid());
        }
        let tokens = usize::try_from(source.elements()?)? / input;
        let routes = ids.shape()[1] as usize;
        let selections = tokens
            .checked_mul(routes)
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
        if selections > i32::MAX as usize
            || ids.shape()[0] as usize != tokens
            || operation.inputs.get(2).ok_or_else(invalid)?.shape() != ids.shape()
            || operation.inputs.get(3).ok_or_else(invalid)?.shape() != ids.shape()
        {
            return Err(invalid());
        }
        if ![
            source,
            operation.inputs.get(2).unwrap(),
            operation.inputs.get(3).unwrap(),
        ]
        .into_iter()
        .all(|value| operand_floating(value, contiguous))
        {
            return Ok(None);
        }
        let index = match ids.dtype() {
            WorkspaceDtype::Int32 => Dtype::Int32,
            WorkspaceDtype::Uint32 => Dtype::Uint32,
            _ => return Ok(None),
        };
        let mut slot = 4;
        let read = units
            .checked_mul(if matches!(activation, Activation::Gated(_)) {
                2
            } else {
                1
            })
            .ok_or(MlxWorkspaceFactError::POPULATION_OVERFLOW)?;
        if read > i32::MAX as usize {
            return Ok(None);
        }
        let Some(first) = projection(first, groups, input, read, operation, &mut slot, contiguous)?
        else {
            return Ok(None);
        };
        let down = match down {
            Some(spec) => match projection(
                spec, groups, units, output, operation, &mut slot, contiguous,
            )? {
                Some(value) => Some(value),
                None => return Ok(None),
            },
            None => None,
        };
        let finish = phase == WorkspaceGroupedPhase::Finish;
        if operation.inputs.len() != slot + if finish { 4 } else { 0 } {
            return Err(invalid());
        }
        let separate_bias = partitions.is_some() && down.is_some_and(|p| p.bias);
        let units_shape = [selections as i32, units as i32];
        let ids_shape = [selections as i32];
        if finish {
            for index in 0..4 {
                let value = operation.inputs.get(slot + index).ok_or_else(invalid)?;
                if value.shape()
                    != if index == 0 {
                        &units_shape[..]
                    } else {
                        &ids_shape[..]
                    }
                {
                    return Err(invalid());
                }
                if index == 0 && !operand_floating(value, contiguous) {
                    return Ok(None);
                }
                if index != 0
                    && !matches!(
                        value.dtype(),
                        WorkspaceDtype::Int32 | WorkspaceDtype::Uint32
                    )
                {
                    return Err(invalid());
                }
            }
        }
        let units_phase = phase == WorkspaceGroupedPhase::Units;
        if operation.outputs.len()
            != if units_phase {
                4
            } else {
                1 + usize::from(separate_bias)
            }
        {
            return Err(invalid());
        }
        for (index, value) in operation.outputs.iter().enumerate() {
            let shape = if units_phase {
                value.shape()
                    == if index == 0 {
                        &units_shape[..]
                    } else {
                        &ids_shape[..]
                    }
            } else {
                value
                    .shape()
                    .iter()
                    .copied()
                    .eq(source.shape()[..source.shape().len() - 1]
                        .iter()
                        .copied()
                        .chain(std::iter::once(output as i32)))
            };
            if !shape
                || value.dtype()
                    != if units_phase && index != 0 {
                        WorkspaceDtype::Uint32
                    } else {
                        WorkspaceDtype::Float32
                    }
            {
                return Err(invalid());
            }
        }
        Ok(Some(Self {
            groups,
            tokens,
            routes,
            input,
            units,
            output,
            first,
            down,
            activation,
            reduction,
            phase,
            separate_bias,
            input_rank: source.shape().len(),
            index,
        }))
    }
    pub(super) fn chunks(self) -> [(usize, usize); 2] {
        use crate::backend::nn::grouped::{
            GROUPED_PROJECTION_CHUNK_THRESHOLD as LIMIT, GROUPED_PROJECTION_CHUNK_TOKENS as CHUNK,
        };
        if matches!(self.activation, Activation::Gated(_)) && self.tokens > LIMIT as usize {
            [
                (CHUNK as usize, self.tokens / CHUNK as usize),
                (
                    self.tokens % CHUNK as usize,
                    usize::from(self.tokens % CHUNK as usize != 0),
                ),
            ]
        } else {
            [(self.tokens, 1), (0, 0)]
        }
    }
}
