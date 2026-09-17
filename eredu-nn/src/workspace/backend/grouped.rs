use super::*;
use crate::{
    routing_intervention::{GroupSelectionControl, IntervenedGroupSelection},
    GatedProductGroupLayout, GroupSelection, GroupSelectionOperator, GroupedGatedProductOperator,
    GroupedGatedProductSpec, GroupedLinearOperator, GroupedLinearSpec, GroupedNeuralBackend,
    GroupedProjectionSpec, GroupedRelu2Operator, GroupedRelu2Spec, GroupedUnitBatch,
    GroupedUnitError, GroupedUnitObserver, Index, JointGroupSelection, JointGroupSelectionInput,
    JointGroupSelectionSpec, ProjectionInputObserver, TensorParallelGroupedGatedProductOperator,
    TensorParallelGroupedOutput, TensorParallelGroupedRelu2Operator, TopKGroupSelectorSpec,
};

mod observation;

/// Complete grouped equation retained by a metadata invocation. No data-dependent
/// route distribution is assumed; a native bound must cover every possible one.
#[derive(Clone, Debug)]
pub enum WorkspaceGroupedBank {
    /// Selected complete-input projection, possibly owning an output shard.
    Linear(GroupedLinearSpec),
    /// Packed or independently materialized gated-product groups.
    GatedProduct(GroupedGatedProductSpec),
    /// ReLU-squared groups.
    Relu2(GroupedRelu2Spec),
}

/// Boundary of one logical selected-group equation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkspaceGroupedPhase {
    /// Routing, activated projections, down projection and weighted reduction.
    Whole,
    /// Sorting and activation, exposing all logical selected rows in callback
    /// order. A tiled mechanism prices the sum of its separate backing stores,
    /// including every tile's overhead; no physical concatenation is implied.
    Units,
    /// Down projection and reduction consuming the effective observed units.
    Finish,
}

/// Grouped metadata module retaining exact physical parameter slots.
#[derive(Clone, Debug)]
pub struct WorkspaceGroups<S> {
    parameters: Vec<Parameter<WorkspaceTensor>>,
    spec: S,
}

// These three construction specs own only geometry, arithmetic choices and
// parameter descriptions. Do not extend this completeness claim to arbitrary S.
macro_rules! parameterized_groups {
    ($($spec:ty),+ $(,)?) => {$(
        impl crate::Parameterized<WorkspaceTensor> for WorkspaceGroups<$spec> {
            fn visit_parameter_sources<'a, V>(&'a self, visitor: &mut V) -> Result<(), crate::ParameterSourceError>
            where V: crate::ParameterSourceVisitor<'a, WorkspaceTensor> {
                crate::Parameterized::visit_parameter_sources(&self.parameters, visitor)
            }

            fn retained_value_slot_bound(&self) -> Option<usize> {
                crate::Parameterized::retained_value_slot_bound(&self.parameters)
            }
            fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
            where V: crate::ParameterVisitor<'a, WorkspaceTensor> {
                crate::Parameterized::visit_parameters(&self.parameters, visitor);
            }
            fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
            where V: crate::ParameterVisitorMut<'a, WorkspaceTensor> {
                crate::Parameterized::visit_parameters_mut(&mut self.parameters, visitor);
            }
            fn set_trainable(&mut self, trainable: bool) {
                crate::Parameterized::set_trainable(&mut self.parameters, trainable);
            }
            fn visit_retained_values(&self, visitor: &mut dyn FnMut(&WorkspaceTensor)) -> bool {
                crate::Parameterized::visit_retained_values(&self.parameters, visitor)
            }
        }
    )+};
}
parameterized_groups!(GroupedLinearSpec, GroupedGatedProductSpec, GroupedRelu2Spec);

/// Selector metadata retaining all projection, correction and scale parameters.
#[derive(Clone, Debug, crate::Parameterized)]
#[parameterized(tensor = "WorkspaceTensor")]
pub struct WorkspaceGroupSelector {
    parameters: Vec<Parameter<WorkspaceTensor>>,
    #[parameter(skip, metadata)]
    spec: TopKGroupSelectorSpec,
}

fn rows(input: &WorkspaceTensor, width: i32, context: &WorkspaceContext) -> Result<i32, Error> {
    if input.shape().is_empty() || input.shape().last() != Some(&width) || width <= 0 {
        return Err(context.metadata_error(format_args!(
            "workspace grouped input width differs from construction"
        )));
    }
    let count = input
        .layout
        .as_view()
        .elements()
        .map_err(|cause| context.metadata_error(format_args!("{cause}")))?
        / width as u64;
    i32::try_from(count)
        .map_err(|_| context.metadata_error(format_args!("workspace grouped row count overflow")))
}

fn integer_indices(input: &WorkspaceTensor) -> bool {
    matches!(
        input.layout.dtype,
        WorkspaceDtype::Int32 | WorkspaceDtype::Uint32
    )
}

fn projection_parameters(
    spec: &GroupedProjectionSpec,
    input: i32,
    output: i32,
    groups: Option<i32>,
    context: &WorkspaceContext,
) -> Result<Vec<Parameter<WorkspaceTensor>>, Error> {
    let mut parameters = physical_parameters(
        &LinearSpec {
            input,
            output,
            weight: context.clone_metadata(spec.weight())?,
            bias: spec
                .bias()
                .map(|value| context.clone_metadata(value))
                .transpose()?,
            format: context.clone_metadata(spec.format())?,
        },
        context,
    )?;
    if let Some(groups) = groups {
        for parameter in &mut parameters {
            let mut shape = context.metadata_vec(
                parameter
                    .as_ref()
                    .shape()
                    .len()
                    .checked_add(1)
                    .ok_or(WorkspaceMetadataError::Overflow)?,
            )?;
            shape.push(groups);
            shape.extend_from_slice(parameter.as_ref().shape());
            let dtype = parameter.as_ref().layout.dtype;
            // The stored parameter includes the group axis. Re-query its exact
            // retained identity and complete geometry after constructing that
            // axis; the temporary ungrouped projection cannot establish its
            // native scalar or stride facts.
            parameter.replace(WorkspaceTensor::existing(
                context.parameter_layout(&parameter.spec, &shape, dtype)?,
                context,
            )?);
        }
    }
    Ok(parameters)
}

impl WorkspaceGroupSelector {
    fn trace(
        &self,
        input: &WorkspaceTensor,
        supplied: Option<&WorkspaceTensor>,
        control: Option<&GroupSelectionControl>,
        context: &WorkspaceContext,
    ) -> Result<IntervenedGroupSelection<WorkspaceTensor>, Error> {
        let count = rows(input, self.spec.input_dimensions(), context)?;
        let k = self.spec.selection().top_k();
        if let Some(indices) = supplied {
            if !integer_indices(indices)
                || indices
                    .layout
                    .as_view()
                    .elements()
                    .map_err(|cause| context.metadata_error(format_args!("{cause}")))?
                    != count as u64 * k as u64
            {
                return Err(context.metadata_error(format_args!(
                    "workspace selected IDs differ from token/top-k geometry"
                )));
            }
        }
        if let Some(control) = control {
            control.validate(
                self.spec.selection(),
                self.spec.coefficient_scale().is_some(),
                count as u64,
            )?;
        }
        let mut inputs = context.metadata_vec(
            self.parameters
                .len()
                .checked_add(1 + usize::from(supplied.is_some()))
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        inputs.push(input);
        inputs.extend(supplied);
        inputs.extend(self.parameters.iter().map(Parameter::as_ref));
        let decision_layouts = |outputs: &mut Vec<WorkspaceLayout>| -> Result<(), Error> {
            outputs.extend([
                context.layout(
                    &[count, k],
                    supplied.map_or(WorkspaceDtype::Uint32, |ids| ids.layout.dtype),
                )?,
                context.layout(&[count, k], WorkspaceDtype::Float32)?,
                context.layout(&[count, k], WorkspaceDtype::Float32)?,
            ]);
            Ok(())
        };
        let capture_original = control.is_some_and(|c| c.capture_original);
        let mut outputs = context.metadata_vec(if capture_original { 6 } else { 3 })?;
        decision_layouts(&mut outputs)?;
        if capture_original {
            decision_layouts(&mut outputs)?;
        }
        let mut values = context
            .execute(
                WorkspaceOperationKind::GroupSelection {
                    spec: context.box_metadata(context.clone_metadata(&self.spec)?)?,
                    supplied_indices: supplied.is_some(),
                    control: control
                        .map(|value| context.box_metadata(context.clone_metadata(value)?))
                        .transpose()?,
                },
                &inputs,
                outputs,
            )?
            .into_iter();
        let mut decision = || {
            GroupSelection::new(
                values.next().unwrap(),
                values.next().unwrap(),
                values.next().unwrap(),
            )
        };
        let original = capture_original.then(&mut decision);
        let effective = decision();
        Ok(IntervenedGroupSelection {
            original,
            effective,
        })
    }
}
impl GroupSelectionOperator<WorkspaceTensor> for WorkspaceGroupSelector {
    fn select(
        &mut self,
        input: &WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<GroupSelection<WorkspaceTensor>, Error> {
        Ok(self.trace(input, None, None, context)?.effective)
    }
    fn select_indices(
        &mut self,
        input: &WorkspaceTensor,
        indices: &WorkspaceTensor,
        context: &WorkspaceContext,
    ) -> Result<GroupSelection<WorkspaceTensor>, Error> {
        Ok(self.trace(input, Some(indices), None, context)?.effective)
    }
    fn select_intervened(
        &mut self,
        input: &WorkspaceTensor,
        control: &GroupSelectionControl,
        context: &WorkspaceContext,
    ) -> Result<IntervenedGroupSelection<WorkspaceTensor>, Error> {
        self.trace(input, None, Some(control), context)
    }
}

impl WorkspaceGroupedBank {
    fn geometry(&self) -> (i32, i32, i32, i32) {
        match self {
            Self::Linear(s) => (
                s.group_count(),
                s.input_dimensions(),
                s.output_dimensions(),
                s.output_dimensions(),
            ),
            Self::GatedProduct(s) => (
                s.group_count(),
                s.input_dimensions(),
                s.intermediate_dimensions(),
                s.output_dimensions(),
            ),
            Self::Relu2(s) => (
                s.group_count(),
                s.hidden_dimensions(),
                s.intermediate_dimensions(),
                s.hidden_dimensions(),
            ),
        }
    }
    fn down_bias(&self) -> bool {
        match self {
            Self::Linear(_) => false,
            Self::GatedProduct(s) => match s.layout() {
                GatedProductGroupLayout::Packed { down, .. } => down.bias().is_some(),
                GatedProductGroupLayout::Independent(groups) => {
                    groups.iter().any(|g| g.down().bias().is_some())
                }
            },
            Self::Relu2(s) => s.down().bias().is_some(),
        }
    }
}
impl<S: super::super::policy_clone::MetadataClone + Into<WorkspaceGroupedBank>> WorkspaceGroups<S> {
    fn bank(&self, context: &WorkspaceContext) -> Result<WorkspaceGroupedBank, Error> {
        context.charge_metadata(
            super::super::policy_clone::controls::<WorkspaceGroupedBank>()
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        Ok(context.clone_metadata(&self.spec)?.into())
    }
    fn trace(
        &self, input: &WorkspaceTensor, selection: &GroupSelection<WorkspaceTensor>,
        partitions: Option<usize>, context: &WorkspaceContext,
        observer: Option<&mut dyn GroupedUnitObserver<WorkspaceTensor>>,
    ) -> Result<TensorParallelGroupedOutput<WorkspaceTensor>, Error> {
        let bank = self.bank(context)?;
        let mut parameters = context.metadata_vec(self.parameters.len())?;
        parameters.extend(self.parameters.iter().map(Parameter::as_ref));
        bank.trace_with_parameters(&parameters, input, selection, partitions, context, observer)
    }
}
impl WorkspaceGroupedBank {
    fn kind(&self, phase: WorkspaceGroupedPhase, partitions: Option<usize>, context: &WorkspaceContext)
        -> Result<WorkspaceOperationKind, Error> {
        context.charge_metadata(super::super::policy_clone::controls::<WorkspaceGroupedBank>()
            .ok_or(WorkspaceMetadataError::Overflow)?)?;
        let bank = match self {
            Self::Linear(spec) => Self::Linear(context.clone_metadata(spec)?),
            Self::GatedProduct(spec) => Self::GatedProduct(context.clone_metadata(spec)?),
            Self::Relu2(spec) => Self::Relu2(context.clone_metadata(spec)?),
        };
        Ok(WorkspaceOperationKind::Grouped { bank: context.box_metadata(bank)?, phase, partitions })
    }
    /// The ordinary grouped recorder with an exact borrowed parameter projection.
    /// Dynamic region sources use this same equation after binding actual rows.
    pub fn trace_with_parameters(
        &self,
        parameters: &[&WorkspaceTensor],
        input: &WorkspaceTensor,
        selection: &GroupSelection<WorkspaceTensor>,
        partitions: Option<usize>,
        context: &WorkspaceContext,
        observer: Option<&mut dyn GroupedUnitObserver<WorkspaceTensor>>,
    ) -> Result<TensorParallelGroupedOutput<WorkspaceTensor>, Error> {
        let (groups, width, units, output) = self.geometry();
        let count = rows(input, width, context)?;
        let ids = selection.group_indices();
        let k = ids.shape().last().copied().unwrap_or(0);
        if partitions == Some(0)
            || !integer_indices(ids)
            || ids.shape().len() < 2
            || k <= 0
            || ids
                .layout
                .as_view()
                .elements()
                .map_err(|cause| context.metadata_error(format_args!("{cause}")))?
                != count as u64 * k as u64
            || selection.coefficients().shape() != ids.shape()
            || selection.selected_scores().shape() != ids.shape()
        {
            return Err(context.metadata_error(format_args!(
                "workspace grouped selection geometry is invalid"
            )));
        }
        let mut inputs = context.metadata_vec(
            parameters
                .len()
                .checked_add(if observer.is_some() { 8 } else { 4 })
                .ok_or(WorkspaceMetadataError::Overflow)?,
        )?;
        inputs.extend([
            input,
            ids,
            selection.selected_scores(),
            selection.coefficients(),
        ]);
        inputs.extend_from_slice(parameters);
        context.validate_values(inputs.iter().copied())?;
        let mut shape = context.metadata_vec(input.shape().len())?;
        shape.extend_from_slice(input.shape());
        *shape.last_mut().unwrap() = output;
        let layout = context.layout(&shape, WorkspaceDtype::Float32)?;
        let separate_bias = partitions.is_some() && self.down_bias();
        let mut outputs = context.metadata_vec(1 + usize::from(separate_bias))?;
        outputs.push(layout.clone());
        if separate_bias {
            outputs.push(layout);
        }
        let values = if let Some(observer) = observer {
            let schedule = context
                .mechanisms
                .grouped_observation_schedule(self, count as u32)?
                .ok_or_else(|| {
                    context.metadata_source(WorkspaceGroupedObservationScheduleUnavailable)
                })?;
            let routes = count.checked_mul(k).ok_or_else(|| {
                context.metadata_error(format_args!("workspace selected route count overflow"))
            })?;
            let mut layouts = context.metadata_vec(4)?;
            layouts.extend([
                context.layout(&[routes, units], WorkspaceDtype::Float32)?,
                context.layout(&[routes], WorkspaceDtype::Uint32)?,
                context.layout(&[routes], WorkspaceDtype::Uint32)?,
                context.layout(&[routes], WorkspaceDtype::Uint32)?,
            ]);
            let mut boundary = context
                .execute(
                    self.kind(WorkspaceGroupedPhase::Units, partitions, context)?,
                    &inputs,
                    layouts,
                )?
                .into_iter();
            let original = boundary.next().unwrap();
            let group_indices = boundary.next().unwrap();
            let selection_indices = boundary.next().unwrap();
            let token_indices = boundary.next().unwrap();
            let coefficients = selection.coefficients().reshape(&[count, k], context)?;
            let effective = observation::trace(
                GroupedUnitBatch {
                    values: &original,
                    group_indices: &group_indices,
                    selection_indices: &selection_indices,
                    token_indices: &token_indices,
                    coefficients: &coefficients,
                    token_offset: 0,
                    total_token_count: count as usize,
                    group_count: groups as usize,
                },
                schedule,
                observer,
                context,
            )?;
            inputs.extend([
                &effective,
                &group_indices,
                &selection_indices,
                &token_indices,
            ]);
            context.execute(
                self.kind(WorkspaceGroupedPhase::Finish, partitions, context)?,
                &inputs,
                outputs,
            )?
        } else {
            context.execute(
                self.kind(WorkspaceGroupedPhase::Whole, partitions, context)?,
                &inputs,
                outputs,
            )?
        };
        let mut values = values.into_iter();
        Ok(TensorParallelGroupedOutput::new(
            values.next().unwrap(),
            values.next(),
        ))
    }
}

impl GroupedLinearOperator<WorkspaceTensor> for WorkspaceGroups<GroupedLinearSpec> {
    fn spec(&self) -> &GroupedLinearSpec {
        &self.spec
    }
    fn forward_grouped(
        &mut self,
        input: &WorkspaceTensor,
        selection: &GroupSelection<WorkspaceTensor>,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        Ok(self
            .trace(input, selection, None, context, None)?
            .into_parts()
            .0)
    }
}

macro_rules! group_operator {
    ($ordinary:ident, $parallel:ident, $spec:ty, $variant:ident) => {
        impl $ordinary<WorkspaceTensor> for WorkspaceGroups<$spec> {
            fn spec(&self) -> &$spec {
                &self.spec
            }
            fn forward_grouped(
                &mut self,
                input: &WorkspaceTensor,
                selection: &GroupSelection<WorkspaceTensor>,
                context: &WorkspaceContext,
            ) -> Result<WorkspaceTensor, Error> {
                Ok(self
                    .trace(input, selection, None, context, None)?
                    .into_parts()
                    .0)
            }
            fn forward_grouped_with_unit_observer(
                &mut self,
                input: &WorkspaceTensor,
                selection: &GroupSelection<WorkspaceTensor>,
                context: &WorkspaceContext,
                observer: Option<&mut dyn GroupedUnitObserver<WorkspaceTensor>>,
            ) -> Result<WorkspaceTensor, Error> {
                Ok(self
                    .trace(input, selection, None, context, observer)?
                    .into_parts()
                    .0)
            }
        }
        impl $parallel<WorkspaceTensor> for WorkspaceGroups<$spec> {
            fn forward_grouped_tensor_parallel(
                &mut self,
                input: &WorkspaceTensor,
                selection: &GroupSelection<WorkspaceTensor>,
                partitions: usize,
                context: &WorkspaceContext,
            ) -> Result<TensorParallelGroupedOutput<WorkspaceTensor>, Error> {
                self.trace(input, selection, Some(partitions), context, None)
            }
            fn forward_grouped_tensor_parallel_with_unit_observer(
                &mut self,
                input: &WorkspaceTensor,
                selection: &GroupSelection<WorkspaceTensor>,
                partitions: usize,
                context: &WorkspaceContext,
                observer: Option<&mut dyn GroupedUnitObserver<WorkspaceTensor>>,
            ) -> Result<TensorParallelGroupedOutput<WorkspaceTensor>, Error> {
                self.trace(input, selection, Some(partitions), context, observer)
            }
        }
    };
}
group_operator!(
    GroupedGatedProductOperator,
    TensorParallelGroupedGatedProductOperator,
    GroupedGatedProductSpec,
    GatedProduct
);
group_operator!(
    GroupedRelu2Operator,
    TensorParallelGroupedRelu2Operator,
    GroupedRelu2Spec,
    Relu2
);

impl GroupedNeuralBackend for WorkspaceBackend {
    fn record_addressable_region_source(source:WorkspaceAddressableRegionView<'_>,input:&WorkspaceTensor,
        routes:&GroupSelection<WorkspaceTensor>,context:&WorkspaceContext,
        observe:Option<&mut dyn FnMut(WorkspaceAddressableObservationView<'_>)->Result<WorkspaceAddressableObservationSource,Error>>)
        ->Result<Option<TensorParallelGroupedOutput<WorkspaceTensor>>,Error>{
        record_addressable_region_with_observation(source,input,routes,context,observe).map(Some)
    }

    fn with_addressable_region<P, E, F>(
        source: WorkspaceAddressableRegionView<'_>, owner: &mut P,
        input: &WorkspaceTensor, routes: &GroupSelection<WorkspaceTensor>,
        context: &WorkspaceContext, _run: F,
    ) -> Result<Result<TensorParallelGroupedOutput<WorkspaceTensor>,E>,Error>
    where F: FnOnce(&mut P, Option<crate::PreparedIndexedInvocationLoan<'_>>)
        -> Result<TensorParallelGroupedOutput<WorkspaceTensor>,E> {
        let _ = owner;
        record_addressable_region(source,input,routes,context).map(Ok)
    }

    type LinearGroups = WorkspaceGroups<GroupedLinearSpec>;
    type Selector = WorkspaceGroupSelector;
    type GatedProductGroups = WorkspaceGroups<GroupedGatedProductSpec>;
    type Relu2Groups = WorkspaceGroups<GroupedRelu2Spec>;

    fn grouped_linear_bank(
        spec: GroupedLinearSpec,
        context: &WorkspaceContext,
    ) -> Result<Self::LinearGroups, Error> {
        spec.validate()?;
        let parameters = projection_parameters(
            spec.projection(),
            spec.input_dimensions(),
            spec.output_dimensions(),
            Some(spec.group_count()),
            context,
        )?;
        Ok(WorkspaceGroups { parameters, spec })
    }
    fn top_k_group_selector(
        spec: TopKGroupSelectorSpec,
        context: &WorkspaceContext,
    ) -> Result<Self::Selector, Error> {
        spec.validate()?;
        let mut parameters = physical_parameters(
            &LinearSpec {
                input: spec.input_dimensions(),
                output: spec.selection().group_count(),
                weight: context.clone_metadata(spec.weight())?,
                bias: spec
                    .bias()
                    .map(|value| context.clone_metadata(value))
                    .transpose()?,
                format: context.clone_metadata(spec.format())?,
            },
            context,
        )?;
        for (extra, width) in [
            (spec.correction_bias(), spec.selection().group_count()),
            (
                spec.input_transform().map(|t| t.scale()),
                spec.input_dimensions(),
            ),
            (spec.coefficient_scale(), spec.selection().group_count()),
        ] {
            if let Some(spec) = extra {
                context.reserve_metadata_vec(&mut parameters, 1)?;
                parameters.push(parameter(
                    context.clone_metadata(spec)?,
                    &[width],
                    WorkspaceDtype::Float32,
                    context,
                )?);
            }
        }
        Ok(WorkspaceGroupSelector { parameters, spec })
    }
    fn grouped_gated_product(
        spec: GroupedGatedProductSpec,
        context: &WorkspaceContext,
    ) -> Result<Self::GatedProductGroups, Error> {
        spec.validate()?;
        let mut parameters = Vec::new();
        let (input, units, output) = (
            spec.input_dimensions(),
            spec.intermediate_dimensions(),
            spec.output_dimensions(),
        );
        match spec.layout() {
            GatedProductGroupLayout::Packed { gate_up, down } => {
                let fused = units.checked_mul(2).ok_or_else(|| {
                    context.metadata_error(format_args!("workspace fused gate/up width overflow"))
                })?;
                {
                    let additional = projection_parameters(
                        gate_up,
                        input,
                        fused,
                        Some(spec.group_count()),
                        context,
                    )?;
                    context.reserve_metadata_vec(&mut parameters, additional.len())?;
                    parameters.extend(additional);
                }
                {
                    let additional = projection_parameters(
                        down,
                        units,
                        output,
                        Some(spec.group_count()),
                        context,
                    )?;
                    context.reserve_metadata_vec(&mut parameters, additional.len())?;
                    parameters.extend(additional);
                }
            }
            GatedProductGroupLayout::Independent(groups) => {
                for group in groups {
                    {
                        let additional =
                            projection_parameters(group.gate(), input, units, None, context)?;
                        context.reserve_metadata_vec(&mut parameters, additional.len())?;
                        parameters.extend(additional);
                    }
                    {
                        let additional =
                            projection_parameters(group.up(), input, units, None, context)?;
                        context.reserve_metadata_vec(&mut parameters, additional.len())?;
                        parameters.extend(additional);
                    }
                    {
                        let additional =
                            projection_parameters(group.down(), units, output, None, context)?;
                        context.reserve_metadata_vec(&mut parameters, additional.len())?;
                        parameters.extend(additional);
                    }
                }
            }
        }
        Ok(WorkspaceGroups { parameters, spec })
    }
    fn grouped_relu2(
        spec: GroupedRelu2Spec,
        context: &WorkspaceContext,
    ) -> Result<Self::Relu2Groups, Error> {
        spec.validate()?;
        let mut parameters = projection_parameters(
            spec.up(),
            spec.hidden_dimensions(),
            spec.intermediate_dimensions(),
            Some(spec.group_count()),
            context,
        )?;
        {
            let additional = projection_parameters(
                spec.down(),
                spec.intermediate_dimensions(),
                spec.hidden_dimensions(),
                Some(spec.group_count()),
                context,
            )?;
            context.reserve_metadata_vec(&mut parameters, additional.len())?;
            parameters.extend(additional);
        }
        Ok(WorkspaceGroups { parameters, spec })
    }
    fn grouped_linear(
        linear: &mut WorkspaceLinear,
        input: &WorkspaceTensor,
        groups: i32,
        output_per_group: i32,
        context: &WorkspaceContext,
    ) -> Result<WorkspaceTensor, Error> {
        Self::grouped_linear_with_input_observer(
            linear,
            input,
            groups,
            output_per_group,
            context,
            None,
        )
    }
    fn grouped_linear_with_input_observer(
        linear: &mut WorkspaceLinear,
        input: &WorkspaceTensor,
        groups: i32,
        output_per_group: i32,
        context: &WorkspaceContext,
        observer: Option<&mut dyn ProjectionInputObserver<WorkspaceTensor>>,
    ) -> Result<WorkspaceTensor, Error> {
        if groups <= 0
            || output_per_group <= 0
            || input.shape().len() != 4
            || input.dim(1) != groups
            || groups.checked_mul(output_per_group) != Some(linear.spec.output)
            || input.dim(3) != linear.spec.input
        {
            return Err(context.metadata_error(format_args!(
                "invalid workspace grouped projection geometry"
            )));
        }
        context.validate_values(
            std::iter::once(input).chain(linear.parameters.iter().map(Parameter::as_ref)),
        )?;
        let projected = linear.forward_with_input_observer(input, context, observer)?;
        let mut pieces = context.metadata_vec(groups as usize)?;
        for group in 0..groups {
            pieces.push(
                projected
                    .index(
                        &[
                            Index::Full,
                            Index::At(group),
                            Index::Full,
                            Index::Range(group * output_per_group, (group + 1) * output_per_group),
                        ],
                        context,
                    )?
                    .expand_dims(1, context)?,
            );
        }
        WorkspaceTensor::concatenate(&pieces, 1, context)
    }
    fn joint_group_selection(
        input: JointGroupSelectionInput<'_, WorkspaceTensor>,
        context: &WorkspaceContext,
    ) -> Result<JointGroupSelection<WorkspaceTensor>, Error> {
        input.validate()?;
        let count = rows(
            input.hidden(),
            *input.hidden().shape().last().unwrap(),
            context,
        )?;
        let spec = JointGroupSelectionSpec::new(
            input.selectable_groups(),
            input.always_on_groups(),
            input.top_k(),
            input.coefficient_scale(),
        )?;
        let mut layouts = context.metadata_vec(3)?;
        layouts.extend([
            context.layout(&[count, input.top_k()], WorkspaceDtype::Uint32)?,
            context.layout(&[count, input.top_k()], WorkspaceDtype::Float32)?,
            context.layout(&[count, input.always_on_groups()], WorkspaceDtype::Float32)?,
        ]);
        let mut values = context
            .execute(
                WorkspaceOperationKind::JointGroupSelection(spec),
                &[
                    input.hidden(),
                    input.weight(),
                    input.correction_bias(),
                    input.global_scale(),
                ],
                layouts,
            )?
            .into_iter();
        Ok(JointGroupSelection::new(
            values.next().unwrap(),
            values.next().unwrap(),
            values.next().unwrap(),
        ))
    }
}

impl From<GroupedLinearSpec> for WorkspaceGroupedBank {
    fn from(spec: GroupedLinearSpec) -> Self {
        Self::Linear(spec)
    }
}

impl From<GroupedGatedProductSpec> for WorkspaceGroupedBank {
    fn from(spec: GroupedGatedProductSpec) -> Self {
        Self::GatedProduct(spec)
    }
}

impl From<GroupedRelu2Spec> for WorkspaceGroupedBank {
    fn from(spec: GroupedRelu2Spec) -> Self {
        Self::Relu2(spec)
    }
}
