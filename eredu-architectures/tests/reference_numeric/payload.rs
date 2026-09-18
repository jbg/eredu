//! Scalar payload binding used by the independent prepared-execution adapter.
//!
//! Kernel reference fixtures may deliberately generate parameters; prepared
//! execution proof uses this binder and therefore consumes the selected bytes.

use super::*;
use eredu_checkpoint::{recipe::RecipeDtype, store::CheckpointSource, StoredDtype};

include!("../support/numeric/payload.rs");

struct Bind<'a> {
    values: &'a BTreeMap<String, NumericTensor>,
    omitted: &'a BTreeSet<String>,
    bound: BTreeSet<String>,
    error: Option<String>,
}

pub(super) fn poison_unselected_static(
    module: &mut impl Parameterized<NumericTensor>,
    tasks: &[ReplicatedTextMaterializationTask],
) -> Vec<String> {
    struct Poison<'a> {
        selected: BTreeSet<&'a str>,
        omitted: Vec<String>,
    }
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Poison<'_> {
        fn visit_mut(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, value: &'a mut NumericTensor) {
            if !self.selected.contains(metadata.id().as_str()) {
                value.data.fill(f32::NAN);
                self.omitted.push(metadata.id().to_string());
            }
        }
    }
    let mut visitor = Poison {
        selected: tasks
            .iter()
            .flat_map(|task| {
                std::iter::once(task.name())
                    .chain(task.aliases().iter().map(String::as_str))
                    .chain(
                        task.output_companions()
                            .iter()
                            .map(|companion| companion.name()),
                    )
            })
            .collect(),
        omitted: Vec::new(),
    };
    module.visit_parameters_mut(&mut visitor);
    visitor.omitted
}

impl<'a> ParameterVisitorMut<'a, NumericTensor> for Bind<'_> {
    fn visit_mut(&mut self, metadata: eredu_nn::ParameterMetadataView<'_>, value: &'a mut NumericTensor) {
        let name = metadata.id().as_str();
        if self.omitted.contains(name) || name == "numeric.unit_norm.weight" {
            return;
        }
        let Some(payload) = self.values.get(name) else {
            self.error
                .get_or_insert_with(|| format!("numeric selected tasks omit parameter {name:?}"));
            return;
        };
        if payload.shape != value.shape {
            self.error.get_or_insert_with(|| {
                format!(
                    "numeric payload {name:?} shape {:?} differs from module {:?}",
                    payload.shape, value.shape
                )
            });
            return;
        }
        *value = payload.clone();
        self.bound.insert(name.to_owned());
        REFERENCE_STAGE_EVIDENCE.with(|evidence| {
            evidence.borrow_mut().bound_parameters.insert(
                name.to_owned(),
                (
                    value.shape.clone(),
                    value.data.iter().map(|value| value.to_bits()).collect(),
                ),
            );
        });
    }
}

pub(super) fn bind_module<M: Parameterized<NumericTensor>>(
    module: &mut M,
    tasks: &[ReplicatedTextMaterializationTask],
    checkpoint: &dyn CheckpointSource,
    context: &NumericContext,
) -> Result<usize, Error> {
    let values = values(tasks, checkpoint, context)?;
    let omitted = BTreeSet::new();
    let mut visitor = Bind {
        values: &values,
        omitted: &omitted,
        bound: BTreeSet::new(),
        error: None,
    };
    module.visit_parameters_mut(&mut visitor);
    if let Some(error) = visitor.error {
        return Err(Error::backend(error));
    }
    Ok(visitor.bound.len())
}

pub(super) fn bind<A>(
    architecture: &mut A,
    units: &mut [A::Unit],
    tasks: &[ReplicatedTextMaterializationTask],
    omitted: &[String],
    checkpoint: &dyn CheckpointSource,
    context: &NumericContext,
) -> Result<usize, Error>
where
    A: LayeredArchitecture<
        NumericBackend,
        DeviceState<NumericBackend, NumericHybridLayerState>,
        Error = Error,
    >,
{
    let values = values(tasks, checkpoint, context)?;
    let omitted = omitted.iter().cloned().collect();
    let mut visitor = Bind {
        values: &values,
        omitted: &omitted,
        bound: BTreeSet::new(),
        error: None,
    };
    architecture
        .static_modules_mut()
        .visit_parameters_mut(&mut visitor);
    for unit in units {
        unit.visit_parameters_mut(&mut visitor);
    }
    if let Some(error) = visitor.error {
        return Err(Error::backend(error));
    }
    Ok(visitor.bound.len())
}

fn values(
    tasks: &[ReplicatedTextMaterializationTask],
    checkpoint: &dyn CheckpointSource,
    context: &NumericContext,
) -> Result<BTreeMap<String, NumericTensor>, Error> {
    let mut values = BTreeMap::new();
    // Fused physical inputs may be admitted aliases of several distinct derived
    // outputs. Such a source identity cannot name any one executable output.
    let mut alias_claims = BTreeMap::<&str, usize>::new();
    for task in tasks {
        for name in std::iter::once(task.name())
            .chain(task.aliases().iter().map(String::as_str))
            .collect::<BTreeSet<_>>()
        {
            *alias_claims.entry(name).or_default() += 1;
        }
    }
    for task in tasks {
        let mut value = recipe_value(
            &task.source_recipe().map_err(Error::backend)?,
            checkpoint,
            context,
        )?;
        if let Some(layout) = context.tensor_layout(task.name()) {
            if value.shape == shape(layout.global_shape())? {
                for placement in layout.additional_placements() {
                    value = select_parameter(&value, placement)?;
                }
                value = select_parameter(&value, layout.placement())?;
            }
        }
        match task.executable() {
            eredu_checkpoint::LinearFormat::Dense => {}
            eredu_checkpoint::LinearFormat::Affine(format)
                if matches!(
                    task.lowering(),
                    eredu_runtime::WeightLoweringKind::Transform
                        | eredu_runtime::WeightLoweringKind::DerivedTransform
                ) =>
            {
                // This scalar mechanism stores an expanded affine execution weight.
                // It is a numerical oracle, not packed-storage/performance evidence.
                let (expanded, scale, bias) = affine_expansion(value, format)?;
                value = expanded;
                for companion in task.output_companions() {
                    let payload = match companion.role() {
                        eredu_nn::LinearCompanionRole::Scale => &scale,
                        eredu_nn::LinearCompanionRole::AffineBias => &bias,
                    };
                    let expected = context
                        .tensor_layout(companion.name())
                        .map_or(companion.logical_shape(), |layout| layout.local_shape());
                    if payload.shape != shape(expected)? {
                        return Err(Error::backend(format!(
                            "scalar affine companion {} geometry {:?} differs from local admission {expected:?}",
                            companion.name(), payload.shape,
                        )));
                    }
                    if values
                        .insert(companion.name().to_owned(), payload.clone())
                        .is_some()
                    {
                        return Err(Error::backend("scalar affine companion is duplicated"));
                    }
                }
            }
            _ => {
                return Err(Error::backend(
                    "numeric payload binder does not admit this executable lowering",
                ))
            }
        }
        for name in std::iter::once(task.name()).chain(
            task.aliases()
                .iter()
                .map(String::as_str)
                .filter(|name| *name != task.name() && alias_claims.get(name) == Some(&1)),
        ) {
            if values.insert(name.to_owned(), value.clone()).is_some() {
                return Err(Error::backend(format!(
                    "numeric selected payload duplicates {name:?}"
                )));
            }
        }
    }
    Ok(values)
}

fn affine_expansion(
    mut input: NumericTensor,
    format: eredu_checkpoint::AffineQuantization,
) -> Result<(NumericTensor, NumericTensor, NumericTensor), Error> {
    format.validate().map_err(Error::backend)?;
    let width = *input
        .shape
        .last()
        .ok_or_else(|| Error::backend("affine input has no axis"))?;
    if width <= 0
        || width % format.group_size != 0
        || input.data.iter().any(|value| !value.is_finite())
    {
        return Err(Error::backend(
            "invalid scalar affine input geometry or values",
        ));
    }
    let mut shape = input.shape.clone();
    *shape.last_mut().unwrap() = width / format.group_size;
    let mut scales = Vec::with_capacity(elements(&shape));
    let mut biases = Vec::with_capacity(elements(&shape));
    let max_code = ((1_u32 << format.bits) - 1) as f32;
    for group in input.data.chunks_exact_mut(format.group_size as usize) {
        let minimum = group.iter().copied().fold(f32::INFINITY, f32::min);
        let maximum = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = (maximum - minimum) / max_code;
        scales.push(scale);
        biases.push(minimum);
        for value in group {
            let code = if scale == 0.0 {
                0.0
            } else {
                ((*value - minimum) / scale).round().clamp(0.0, max_code)
            };
            *value = code * scale + minimum;
        }
    }
    Ok((
        input,
        NumericTensor::new(shape.clone(), scales),
        NumericTensor::new(shape, biases),
    ))
}

#[derive(Clone)]
pub(super) struct BoundedBinding {
    checkpoint: RetainedCheckpointSource,
    units: Vec<Vec<ReplicatedTextMaterializationTask>>,
    omitted: BTreeSet<String>,
    live: std::rc::Rc<Cell<usize>>,
}

pub(super) struct PayloadLease {
    live: std::rc::Rc<Cell<usize>>,
}
impl Drop for PayloadLease {
    fn drop(&mut self) {
        self.live.set(self.live.get() - 1);
    }
}

impl BoundedBinding {
    pub(super) fn restrict_units(&mut self, ordinals: &[usize]) -> Result<(), Error> {
        self.units = ordinals
            .iter()
            .map(|ordinal| {
                self.units.get(*ordinal).cloned().ok_or_else(|| {
                    Error::backend("bounded partition ordinal is outside selected tasks")
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(())
    }

    pub(super) fn prepare<A>(
        architecture: &mut A,
        layout: &eredu_runtime::ExecutionUnitLayout,
        tasks: &[ReplicatedTextMaterializationTask],
        omitted: &[String],
        checkpoint: RetainedCheckpointSource,
        context: &NumericContext,
    ) -> Result<Self, Error>
    where
        A: LayeredArchitecture<
            NumericBackend,
            DeviceState<NumericBackend, NumericHybridLayerState>,
            Error = Error,
        >,
    {
        let plan = eredu_runtime::plan_replicated_text_materialization_tasks(tasks, layout)
            .map_err(Error::backend)?;
        let static_tasks = plan
            .static_task_indices()
            .iter()
            .map(|&index| tasks[index].clone())
            .collect::<Vec<_>>();
        let units = plan
            .unit_task_indices()
            .iter()
            .map(|indices| indices.iter().map(|&index| tasks[index].clone()).collect())
            .collect();
        for task in &static_tasks {
            record_reference_payload_reads(verify_materialization_task_payloads(
                task,
                checkpoint.as_ref(),
            )?);
        }
        let values = values(&static_tasks, checkpoint.as_ref(), context)?;
        let omitted = omitted.iter().cloned().collect();
        let mut visitor = Bind {
            values: &values,
            omitted: &omitted,
            bound: BTreeSet::new(),
            error: None,
        };
        architecture
            .static_modules_mut()
            .visit_parameters_mut(&mut visitor);
        if let Some(error) = visitor.error {
            return Err(Error::backend(error));
        }
        Ok(Self {
            checkpoint,
            units,
            omitted,
            live: std::rc::Rc::new(Cell::new(0)),
        })
    }

    pub(super) fn acquire<U: Parameterized<NumericTensor>>(
        &self,
        ordinal: usize,
        unit: &mut U,
        context: &NumericContext,
    ) -> Result<PayloadLease, Error> {
        let tasks = self
            .units
            .get(ordinal)
            .ok_or_else(|| Error::backend("bounded scalar unit is outside selected tasks"))?;
        for task in tasks {
            record_reference_payload_reads(verify_materialization_task_payloads(
                task,
                self.checkpoint.as_ref(),
            )?);
        }
        let values = values(tasks, self.checkpoint.as_ref(), context)?;
        let mut visitor = Bind {
            values: &values,
            omitted: &self.omitted,
            bound: BTreeSet::new(),
            error: None,
        };
        unit.visit_parameters_mut(&mut visitor);
        if let Some(error) = visitor.error {
            return Err(Error::backend(error));
        }
        self.live.set(self.live.get() + 1);
        REFERENCE_STAGE_EVIDENCE.with(|evidence| {
            let mut evidence = evidence.borrow_mut();
            evidence.bounded_unit_acquisitions.push(ordinal);
            evidence.peak_bound_units = evidence.peak_bound_units.max(self.live.get());
        });
        Ok(PayloadLease {
            live: std::rc::Rc::clone(&self.live),
        })
    }
}

// Apply the same neutral member placement contract used by native binding.
pub(super) fn addressable_values(
    member: &eredu_runtime::AddressableBankMember,
    checkpoint: &dyn CheckpointSource,
    context: &NumericContext,
) -> Result<(BTreeMap<String, NumericTensor>, u64), Error> {
    let bindings = member
        .parameters()
        .iter()
        .map(|parameter| {
            eredu_runtime::WeightBinding::from_recipe(
                parameter.binding_name(),
                parameter.recipe().clone(),
                parameter.source_bytes(),
            )
            .and_then(|binding| binding.with_logical_target(parameter.task().name()))
            .map_err(Error::backend)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let bindings = eredu_runtime::place_addressable_member_bindings(
        bindings,
        checkpoint,
        context
            .local_layout
            .as_ref()
            .ok_or_else(|| Error::backend("addressable partition has no local layout"))?,
    )
    .map_err(Error::backend)?;
    let mut values = BTreeMap::new();
    let mut bytes = 0;
    for (parameter, binding) in member.parameters().iter().zip(bindings) {
        let recipe = binding.source_recipe();
        let metadata = recipe.infer(checkpoint).map_err(Error::backend)?;
        bytes += eredu_runtime::selected_addressable_parameter_bytes(parameter.task(), &metadata)
            .map_err(Error::backend)?;
        let mut value = recipe_value(&recipe, checkpoint, context)?;
        match parameter.task().executable() {
            eredu_checkpoint::LinearFormat::Dense => {}
            eredu_checkpoint::LinearFormat::Affine(format)
                if matches!(
                    parameter.task().lowering(),
                    eredu_runtime::WeightLoweringKind::Transform
                        | eredu_runtime::WeightLoweringKind::DerivedTransform
                ) =>
            {
                // The same scalar affine oracle serves ordinary and independent
                // banks. Selected byte charges still describe packed storage.
                let (expanded, scale, bias) = affine_expansion(value, format)?;
                value = expanded;
                let companions = parameter.quantization_companions().ok_or_else(|| {
                    Error::backend("scalar affine member has no companion bindings")
                })?;
                values.insert(companions.scale().to_owned(), scale);
                let bias_name = companions
                    .affine_bias()
                    .ok_or_else(|| Error::backend("scalar affine member has no bias binding"))?;
                values.insert(bias_name.to_owned(), bias);
            }
            _ => {
                return Err(Error::backend(
                    "scalar member payload does not admit this executable lowering",
                ));
            }
        }
        values.insert(parameter.binding_name().to_owned(), value);
    }
    Ok((values, bytes))
}
