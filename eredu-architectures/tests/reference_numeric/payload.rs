//! Scalar payload binding used by the independent prepared-execution adapter.
//!
//! Kernel reference fixtures may deliberately generate parameters; prepared
//! execution proof uses this binder and therefore consumes the selected bytes.

use super::*;
use eredu_checkpoint::{recipe::RecipeDtype, store::CheckpointSource, StoredDtype};

fn shape(values: &[usize]) -> Result<Vec<i32>, Error> {
    values
        .iter()
        .map(|&value| i32::try_from(value).map_err(Error::backend))
        .collect()
}

fn decode(bytes: &[u8], dtype: &StoredDtype, dimensions: &[usize]) -> Result<NumericTensor, Error> {
    let (width, values): (usize, Vec<f32>) = match dtype {
        StoredDtype::F32 => (
            4,
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_le_bytes(*chunk))
                .collect(),
        ),
        StoredDtype::F16 => (
            2,
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|chunk| half::f16::from_bits(u16::from_le_bytes(*chunk)).to_f32())
                .collect(),
        ),
        StoredDtype::BF16 => (
            2,
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|chunk| half::bf16::from_bits(u16::from_le_bytes(*chunk)).to_f32())
                .collect(),
        ),
        StoredDtype::I32 => (
            4,
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| i32::from_le_bytes(*chunk) as f32)
                .collect(),
        ),
        StoredDtype::U8 => (1, bytes.iter().map(|&value| f32::from(value)).collect()),
        dtype => {
            return Err(Error::backend(format!(
                "numeric dense payload cannot decode {dtype:?}"
            )))
        }
    };
    let dimensions = shape(dimensions)?;
    if !bytes.len().is_multiple_of(width) || elements(&dimensions) != values.len() {
        return Err(Error::backend(
            "numeric payload length does not match the admitted shape",
        ));
    }
    Ok(NumericTensor::new(dimensions, values))
}

fn select(value: NumericTensor, selection: &TensorSelection) -> Result<NumericTensor, Error> {
    match selection {
        TensorSelection::Full => Ok(value),
        TensorSelection::Range { axis, start, end } => select_parameter(
            &value,
            &TensorPlacement::Range {
                axis: *axis,
                start: *start,
                end: *end,
            },
        ),
        TensorSelection::Indices { axis, indices } => select_parameter(
            &value,
            &TensorPlacement::Indices {
                axis: *axis,
                indices: indices.clone(),
            },
        ),
        TensorSelection::Contiguous {
            offset_elements,
            shape: dimensions,
        } => {
            let dimensions = shape(dimensions)?;
            let end = offset_elements
                .checked_add(elements(&dimensions))
                .ok_or_else(|| Error::backend("numeric payload selection overflowed"))?;
            let values = value
                .data
                .get(*offset_elements..end)
                .ok_or_else(|| Error::backend("numeric payload selection exceeds source"))?;
            Ok(NumericTensor::new(dimensions, values.to_vec()))
        }
    }
}

fn recipe_value(
    recipe: &DerivedWeightRecipe,
    checkpoint: &dyn CheckpointSource,
    context: &NumericContext,
) -> Result<NumericTensor, Error> {
    match recipe {
        DerivedWeightRecipe::Source { key, selection } => {
            let lease = checkpoint
                .acquire_lease(TensorReadRequest {
                    key: key.clone(),
                    selection: selection.clone(),
                    policy: ReadPolicy::RequireBounded,
                })
                .map_err(Error::backend)?;
            record_reference_payload_reads(vec![ReferencePayloadRead {
                task: key.clone(),
                source: key.clone(),
                selection: selection.clone(),
                output_shape: lease.output_shape().to_vec(),
                encoded_bytes: lease.bounded_read_proof().length_bytes,
                physically_bounded: lease.bounded_read_proof().physically_bounded,
            }]);
            match &lease {
                eredu_checkpoint::store::CheckpointLease::Gguf(gguf) => {
                    let converted = gguf.materialize_portable().map_err(Error::backend)?;
                    match converted.converted() {
                        eredu_gguf::ConvertedTensor::Dense(tensor) => {
                            let dtype = match tensor.dtype {
                                eredu_gguf::DenseDtype::F32 => StoredDtype::F32,
                                eredu_gguf::DenseDtype::F16 => StoredDtype::F16,
                                eredu_gguf::DenseDtype::Bf16 => StoredDtype::BF16,
                                _ => {
                                    return Err(Error::backend(
                                        "numeric GGUF payload dtype is not admitted",
                                    ))
                                }
                            };
                            decode(&tensor.data, &dtype, lease.output_shape())
                        }
                        _ => Err(Error::backend(
                            "numeric dense payload binder does not admit packed GGUF",
                        )),
                    }
                }
                _ => decode(
                    lease
                        .encoded_bytes()
                        .ok_or_else(|| Error::backend("numeric lease has no scalar bytes"))?,
                    &lease.metadata().stored_dtype,
                    lease.output_shape(),
                ),
            }
        }
        DerivedWeightRecipe::Select { input, selection } => {
            select(recipe_value(input, checkpoint, context)?, selection)
        }
        DerivedWeightRecipe::Concatenate { axis, inputs } => {
            let inputs = inputs
                .iter()
                .map(|input| recipe_value(input, checkpoint, context))
                .collect::<Result<Vec<_>, _>>()?;
            NumericTensor::concatenate(
                &inputs,
                i32::try_from(*axis).map_err(Error::backend)?,
                context,
            )
        }
        DerivedWeightRecipe::Stack { axis, inputs } => {
            let inputs = inputs
                .iter()
                .map(|input| {
                    let value = recipe_value(input, checkpoint, context)?;
                    let mut dimensions = value.shape.clone();
                    if *axis > dimensions.len() {
                        return Err(Error::backend("numeric stack axis exceeds rank"));
                    }
                    dimensions.insert(*axis, 1);
                    value.reshape(&dimensions, context)
                })
                .collect::<Result<Vec<_>, _>>()?;
            NumericTensor::concatenate(
                &inputs,
                i32::try_from(*axis).map_err(Error::backend)?,
                context,
            )
        }
        DerivedWeightRecipe::Reshape {
            input,
            shape: dimensions,
        } => recipe_value(input, checkpoint, context)?.reshape(&shape(dimensions)?, context),
        DerivedWeightRecipe::Transpose { input, axes } => {
            recipe_value(input, checkpoint, context)?.transpose_axes(&shape(axes)?, context)
        }
        DerivedWeightRecipe::Cast { input, dtype } => {
            let value = recipe_value(input, checkpoint, context)?;
            match dtype {
                RecipeDtype::F32 => Ok(value),
                RecipeDtype::F16 => Ok(value.map(|x| half::f16::from_f32(x).to_f32())),
                RecipeDtype::BF16 => Ok(value.map(|x| half::bf16::from_f32(x).to_f32())),
                _ => Err(Error::backend("numeric dense payload cast is not admitted")),
            }
        }
        DerivedWeightRecipe::NegLog { input } => {
            Ok(recipe_value(input, checkpoint, context)?.map(|x| -x.ln()))
        }
        DerivedWeightRecipe::SubtractOne { input } => {
            Ok(recipe_value(input, checkpoint, context)?.map(|x| x - 1.0))
        }
        DerivedWeightRecipe::View { .. } => Err(Error::backend(
            "numeric dense payload reinterpretation is not admitted",
        )),
    }
}

struct Bind<'a> {
    values: &'a BTreeMap<String, NumericTensor>,
    omitted: &'a BTreeSet<String>,
    bound: BTreeSet<String>,
    error: Option<String>,
}

impl<'a> ParameterVisitorMut<'a, NumericTensor> for Bind<'_> {
    fn visit_mut(&mut self, metadata: ParameterMetadata, value: &'a mut NumericTensor) {
        let name = metadata.id.as_str();
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
        for name in std::iter::once(task.name()).chain(task.aliases().iter().map(String::as_str)) {
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

pub(super) struct BoundedBinding {
    checkpoint: SharedCheckpointSource,
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
    pub(super) fn prepare<A>(
        architecture: &mut A,
        layout: &eredu_runtime::ExecutionUnitLayout,
        tasks: &[ReplicatedTextMaterializationTask],
        omitted: &[String],
        checkpoint: SharedCheckpointSource,
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
