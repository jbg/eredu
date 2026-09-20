use super::*;
use eredu_nn::Tensor;
use eredu_nn::workspace::{WorkspaceLayoutView, WorkspaceRepresentation};

pub(super) fn floating(dtype: InterventionDtype) -> F {
    match dtype {
        InterventionDtype::Float32 => F::Float32,
        InterventionDtype::Float16 => F::Float16,
        InterventionDtype::Bfloat16 => F::Bfloat16,
    }
}

// Missing physical evidence is not inferred from logical F32. The actual
// prepared source supplies the scalar type and native execution rechecks it.
// Any retained physical evidence must agree before the trace changes.
pub(super) fn validate_precision(
    value: &WorkspaceTensor,
    dtype: InterventionDtype,
) -> Result<(), Failure> {
    if value.layout().dtype() != D::Float32
        || value
            .layout()
            .representation()
            .is_some_and(|representation| representation.dtype() != floating(dtype))
    {
        return Err(Failure::ShapeMismatch);
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub(super) struct Value {
    pub(super) shape: Shape,
    // None denotes Boolean or integer auxiliaries, never unknown floating data.
    pub(super) dtype: Option<InterventionDtype>,
}
impl Value {
    fn new(shape: &[i32], dtype: Option<InterventionDtype>) -> Result<Self, Failure> {
        if shape.len() > 32 {
            return Err(Failure::ShapeMismatch);
        }
        let mut axes = [0; 32];
        axes[..shape.len()].copy_from_slice(shape);
        Ok(Self {
            shape: Shape {
                axes,
                rank: shape.len(),
            },
            dtype,
        })
    }
}
#[derive(Default)]
pub(super) struct Count {
    pub(super) roots: usize,
    pub(super) host_bytes: usize,
}
impl Kernel for Count {
    type Value = Value;
    fn shape<'a>(&self, value: &'a Value) -> &'a [i32] {
        value.shape.slice()
    }
    fn dtype(&self, value: &Value) -> Result<InterventionDtype, Failure> {
        value.dtype.ok_or(Failure::ShapeMismatch)
    }
    fn emit(&mut self, op: Op<'_, Value>) -> Result<Value, Failure> {
        self.roots = self.roots.checked_add(1).ok_or(Failure::GeometryOverflow)?;
        Ok(match op {
            Op::Indices(indices) => Value::new(&[i32::try_from(indices.len()).map_err(|_| Failure::GeometryOverflow)?], None)?,
            Op::Reshape(source, shape) => Value::new(shape, source.dtype)?,
            Op::IndexedSelect(source, indices) => Value { shape: indices.shape, dtype: source.dtype },
            Op::IndexedUpdate(source, _, _) => *source,
            Op::Select(source, slice) => Value {
                shape: Shape::new(&slice.shape)?,
                dtype: source.dtype,
            },
            Op::Update(source, _, _) => *source,
            Op::Zero(shape, dtype) => Value {
                shape: Shape::new(shape)?,
                dtype: Some(dtype),
            },
            Op::Scalar(_) => Value::new(&[], Some(InterventionDtype::Float32))?,
            Op::Cast(value, dtype) => Value {
                shape: value.shape,
                dtype: Some(dtype),
            },
            Op::Broadcast(value, shape) => Value {
                shape: shape.shape,
                dtype: value.dtype,
            },
            Op::Mask(_, value) => Value {
                shape: value.shape,
                dtype: None,
            },
            Op::Columns(_, _, width) => {
                self.host_bytes = self
                    .host_bytes
                    .checked_add(usize::try_from(width).map_err(|_| Failure::GeometryOverflow)?)
                    .ok_or(Failure::GeometryOverflow)?;
                Value::new(&[width], None)?
            }
            Op::Tensor(tensor) => Value {
                shape: Shape::new(&tensor.shape)?,
                dtype: Some(tensor.values.dtype()),
            },
            Op::Binary(_, a, _) => *a,
            Op::Where(_, value, _) => *value,
        })
    }
}

// This exact worker's precision is independent of the general workspace add
// descriptor, which also represents residual paths with optional F32 promotion.
// Keep that global descriptor conservative; do not invent a cast or view to
// repair evidence lost by another primitive.
pub(super) struct TracedValue {
    pub(super) tensor: WorkspaceTensor,
    pub(super) dtype: Option<InterventionDtype>,
}
pub(super) struct Trace<'a> {
    pub(super) context: &'a WorkspaceContext,
    pub(super) retained: &'a mut Vec<WorkspaceTensor>,
}
pub(super) fn control_bytes() -> Option<usize> {
    let frames = [
        size_of::<Adapter<Trace<'static>>>(),
        size_of::<[TracedValue; 4]>(),
        size_of::<Op<'static, TracedValue>>(),
        size_of::<Result<TracedValue, Failure>>(),
        size_of::<Result<TracedValue, CaptureExecutionError<Failure>>>(),
        size_of::<F>(),
        size_of::<WorkspaceLayoutView<'static>>(),
        size_of::<WorkspaceRepresentation>(),
        size_of::<Option<InterventionDtype>>(),
        size_of::<Result<InterventionDtype, Failure>>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
impl Trace<'_> {
    fn operation(
        &self,
        kind: K,
        inputs: &[&WorkspaceTensor],
        shape: &[i32],
        dtype: D,
    ) -> Result<WorkspaceTensor, Failure> {
        let mut outputs = self.context.metadata_vec(1)?;
        outputs.push(self.context.layout(shape, dtype)?);
        Ok(self.context.execute(kind, inputs, outputs)?.remove(0))
    }
}
impl Kernel for Trace<'_> {
    type Value = TracedValue;
    fn shape<'a>(&self, value: &'a TracedValue) -> &'a [i32] {
        value.tensor.shape()
    }
    fn dtype(&self, value: &TracedValue) -> Result<InterventionDtype, Failure> {
        value.dtype.ok_or(Failure::ShapeMismatch)
    }
    fn emit(&mut self, op: Op<'_, TracedValue>) -> Result<TracedValue, Failure> {
        let (tensor, dtype) = match op {
            Op::Indices(indices) => (WorkspaceTensor::initialized(&[i32::try_from(indices.len()).map_err(|_| Failure::GeometryOverflow)?], D::Int32, self.context)?, None),
            Op::Reshape(source, shape) => (source.tensor.reshape(shape, self.context)?, source.dtype),
            Op::IndexedSelect(source, indices) => (
                source.tensor.select_elements_with_indices(&indices.tensor, self.context)?, source.dtype),
            Op::IndexedUpdate(source, indices, update) => (
                source.tensor.update_elements_with_indices(&indices.tensor, &update.tensor, self.context)?, source.dtype),
            Op::Select(source, slice) => (
                source.tensor.static_slice(
                    Shape::new(&slice.starts)?.slice(),
                    Shape::new(&slice.ends)?.slice(),
                    Shape::new(&slice.strides)?.slice(),
                    self.context,
                )?,
                source.dtype,
            ),
            Op::Update(source, slice, replacement) => (
                source.tensor.update_static_slice(
                    &replacement.tensor,
                    Shape::new(&slice.starts)?.slice(),
                    Shape::new(&slice.ends)?.slice(),
                    Shape::new(&slice.strides)?.slice(),
                    self.context,
                )?,
                source.dtype,
            ),
            Op::Zero(shape, dtype) => (
                WorkspaceTensor::zeros_from_prototype(
                    Shape::new(shape)?.slice(),
                    WorkspaceLayoutView::new(&[], D::Float32).map_err(eredu_nn::Error::from)?
                        .with_representation(Some(WorkspaceRepresentation::new(floating(dtype), true))),
                    self.context,
                )?,
                Some(dtype),
            ),
            Op::Scalar(_) => (
                self.operation(K::Elementwise("scalar_f32"), &[], &[], D::Float32)?,
                Some(InterventionDtype::Float32),
            ),
            Op::Cast(value, dtype) => (
                value.tensor.cast_floating(floating(dtype), self.context)?,
                Some(dtype),
            ),
            Op::Broadcast(value, shape) => (
                self.operation(
                    K::View("broadcast"),
                    &[&value.tensor],
                    shape.tensor.shape(),
                    value.tensor.layout().dtype(),
                )?,
                value.dtype,
            ),
            Op::Mask(_, value) => (
                WorkspaceTensor::initialized(value.tensor.shape(), D::Bool, self.context)?,
                None,
            ),
            Op::Columns(_, _, width) => (
                WorkspaceTensor::initialized(&[width], D::Bool, self.context)?,
                None,
            ),
            Op::Tensor(tensor) => (
                WorkspaceTensor::initialized_floating(
                    Shape::new(&tensor.shape)?.slice(),
                    floating(tensor.values.dtype()),
                    self.context,
                )?,
                Some(tensor.values.dtype()),
            ),
            Op::Binary(kind, a, b) => (
                self.operation(
                    K::Elementwise(match kind {
                        Binary::Add => "add",
                        Binary::Multiply => "multiply",
                    }),
                    &[&a.tensor, &b.tensor],
                    a.tensor.shape(),
                    D::Float32,
                )?,
                a.dtype,
            ),
            Op::Where(mask, value, fill) => (
                self.operation(
                    K::Elementwise("where"),
                    &[&mask.tensor, &value.tensor, &fill.tensor],
                    value.tensor.shape(),
                    D::Float32,
                )?,
                value.dtype,
            ),
        };
        if let Some(dtype) = dtype {
            validate_precision(&tensor, dtype)?;
        }
        self.context.reserve_metadata_vec(self.retained, 1)?;
        self.retained.push(tensor.clone());
        Ok(TracedValue { tensor, dtype })
    }
}
