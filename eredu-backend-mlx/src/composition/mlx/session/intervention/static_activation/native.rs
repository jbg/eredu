use super::*;
pub(in crate::composition::mlx::session::intervention) struct Native<'a, 'o> {
    pub(super) stream: &'a Stream,
    pub(super) completion: CaptureCompletion<'o>,
    pub(super) roots: Option<&'a RefCell<Vec<Array>>>,
}
impl Kernel for Native<'_, '_> {
    type Value = Array;
    fn shape<'a>(&self, value: &'a Array) -> &'a [i32] {
        value.shape()
    }
    fn dtype(&self, value: &Array) -> Result<InterventionDtype, Failure> {
        match value.dtype() {
            Dtype::Float32 => Ok(InterventionDtype::Float32),
            Dtype::Float16 => Ok(InterventionDtype::Float16),
            Dtype::Bfloat16 => Ok(InterventionDtype::Bfloat16),
            dtype => Err(Failure::UnsupportedDtype(dtype)),
        }
    }
    fn emit(&mut self, op: Op<'_, Array>) -> Result<Array, Failure> {
        let value = match op {
            Op::Indices(IndexSource::Actual(indices)) => Array::try_from_slice(indices, &[i32::try_from(indices.len()).map_err(|_| Failure::GeometryOverflow)?])?,
            Op::Indices(IndexSource::Bound(_)) => return Err(Failure::ClaimMismatch),
            Op::Reshape(source, shape) => source.reshape(shape, self.stream)?,
            Op::IndexedSelect(source, indices) => source.take(indices, self.stream)?,
            Op::IndexedUpdate(source, indices, update) => source.try_flat_index_update(indices, update, self.stream)?,
            Op::Select(source, slice) => source.try_slice(
                Shape::new(&slice.starts)?.slice(),
                Shape::new(&slice.ends)?.slice(),
                Shape::new(&slice.strides)?.slice(),
                self.stream,
            )?,
            Op::Update(source, slice, replacement) => source.try_slice_update(
                replacement,
                Shape::new(&slice.starts)?.slice(),
                Shape::new(&slice.ends)?.slice(),
                Shape::new(&slice.strides)?.slice(),
                self.stream,
            )?,
            Op::Zero(shape, dtype) => safemlx::ops::zeros_dtype(
                Shape::new(shape)?.slice(),
                native_dtype(dtype),
                self.stream,
            )?,
            Op::Scalar(value) => Array::try_from_f32(value)?,
            Op::Cast(value, dtype) => value.as_dtype(native_dtype(dtype), self.stream)?,
            Op::Broadcast(value, shape) => {
                safemlx::ops::broadcast_to(value, shape.shape(), self.stream)?
            }
            Op::Mask(keep, value) => Array::try_from_slice(keep, value.shape())?,
            Op::Columns(ids, keep_selected, width) => {
                let width = usize::try_from(width).map_err(|_| Failure::ShapeMismatch)?;
                if ids.iter().any(|id| *id as usize >= width) {
                    return Err(Failure::ShapeMismatch);
                }
                // This exact width is part of population.host_bytes before Q.
                // No growth, widening or source-plan clone occurs here.
                let mut keep = Vec::new();
                keep.try_reserve_exact(width)?;
                if keep.capacity() != width {
                    return Err(Failure::GeometryOverflow);
                }
                keep.resize(width, !keep_selected);
                for id in ids {
                    keep[*id as usize] = keep_selected;
                }
                Array::try_from_slice(&keep, &[width as i32])?
            }
            Op::Tensor(tensor) => {
                let shape = Shape::new(&tensor.shape)?;
                match &tensor.values {
                    InterventionValues::Float32(values) => {
                        Array::try_from_slice(values, shape.slice())?
                    }
                    InterventionValues::Float16(bits) => {
                        Array::try_from_slice(bits.reinterpret_cast::<half::f16>(), shape.slice())?
                    }
                    InterventionValues::Bfloat16(bits) => {
                        Array::try_from_slice(bits.reinterpret_cast::<half::bf16>(), shape.slice())?
                    }
                }
            }
            Op::Binary(Binary::Add, a, b) => a.add(b, self.stream)?,
            Op::Binary(Binary::Multiply, a, b) => a.multiply(b, self.stream)?,
            Op::Where(mask, value, fill) => safemlx::ops::r#where(mask, value, fill, self.stream)?,
        };
        if let Some(roots) = self.roots {
            let mut roots = roots.try_borrow_mut().map_err(|_| Failure::CollectorBusy)?;
            if roots.len() == roots.capacity() {
                return Err(Failure::GeometryOverflow);
            }
            // The accepted numerical Recovery owns every prefix on failure.
            roots.push(self.completion.clone_array(&value)?);
            drop(roots);
            // Full replacement can disconnect earlier emitted values from the
            // final output DAG. Each retained prefix therefore owns a completed
            // frontier before publication, even when later arithmetic ignores it.
            // Retention precedes settlement so failure preserves the real graph.
            self.completion.settle(&value, self.stream)?;
        }
        Ok(value)
    }
}
