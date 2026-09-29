//! Family-owned fixed geometry for ragged QSA tails and their rotary provenance.
use super::*;
use eredu_core::cache::{
    MutableStateResidency, StateTensorDimension as Dim, StateTensorDtype, StateTensorPolicy,
    StateTensorRole,
};
use eredu_runtime::RuntimeStateComponents;

/// Exact storage declaration for all incomplete QSA blocks in an input batch.
/// Controls are I32: `[last_seen, length, first_rotary_offset_or_mode, positions…]`.
/// Mode -2 denotes an empty block; -1 denotes retained media rotary embeddings.
/// Raw keys and media rotary values retain independently declared floating types.
#[derive(Debug, Clone, Copy)]
pub struct QsaPartialStateSpec {
    ratio: i32,
    dimensions: i32,
    rotary_dimensions: i32,
    raw_element: TensorElementType,
    rotary_element: TensorElementType,
    control_slot: u32,
    buffer_slot: u32,
}
impl QsaPartialStateSpec {
    /// Reserves one integer slot and three consecutive generic buffer slots.
    pub fn new(
        ratio: i32,
        dimensions: i32,
        rotary_dimensions: i32,
        raw_element: TensorElementType,
        rotary_element: TensorElementType,
        control_slot: u32,
        buffer_slot: u32,
    ) -> Result<Self, QsaError> {
        let floating = |t| {
            matches!(
                t,
                TensorElementType::F16 | TensorElementType::Bf16 | TensorElementType::F32
            )
        };
        if ratio <= 0
            || ratio.checked_add(2).is_none()
            || dimensions <= 0
            || rotary_dimensions <= 0
            || rotary_dimensions > dimensions
            || buffer_slot.checked_add(2).is_none()
            || !floating(raw_element)
            || !floating(rotary_element)
            || (rotary_element != raw_element && rotary_element != TensorElementType::F32)
        {
            return Err(QsaError::Geometry);
        }
        Ok(Self {
            ratio,
            dimensions,
            rotary_dimensions,
            raw_element,
            rotary_element,
            control_slot,
            buffer_slot,
        })
    }
    fn capacity(self) -> i32 {
        (self.ratio - 1).max(1)
    }
    fn control_width(self) -> i32 {
        self.ratio + 2
    }
    /// Stable roles, in controls/raw-keys/cosine/sine order.
    pub fn roles(self) -> [StateTensorRole; 4] {
        [
            StateTensorRole::IntegerHistory {
                slot: self.control_slot,
            },
            StateTensorRole::Auxiliary {
                slot: self.buffer_slot,
            },
            StateTensorRole::Auxiliary {
                slot: self.buffer_slot + 1,
            },
            StateTensorRole::Auxiliary {
                slot: self.buffer_slot + 2,
            },
        ]
    }
    /// Declares fixed capacities; padding determines per-lane lengths in controls.
    pub fn policies(self) -> Vec<StateTensorPolicy> {
        let d = |n| Dim::fixed(n).expect("validated QSA dimension");
        let dtype = |element| {
            if element == TensorElementType::F32 {
                StateTensorDtype::Float32
            } else {
                StateTensorDtype::Floating
            }
        };
        let shapes = [
            vec![Dim::Batch, d(self.control_width())],
            vec![Dim::Batch, d(self.capacity()), d(self.dimensions)],
            vec![Dim::Batch, d(self.rotary_dimensions)],
            vec![Dim::Batch, d(self.rotary_dimensions)],
        ];
        let types = [
            StateTensorDtype::Int32,
            dtype(self.raw_element),
            dtype(self.rotary_element),
            dtype(self.rotary_element),
        ];
        self.roles()
            .into_iter()
            .zip(shapes)
            .zip(types)
            .map(|((role, shape), dtype)| {
                StateTensorPolicy::new(
                    role,
                    shape,
                    dtype,
                    MutableStateResidency::AlwaysDeviceMutable,
                )
                .expect("validated bounded QSA state")
            })
            .collect()
    }
    /// Conservative tensor, host-control and handle workspace for packing/restoring.
    /// Persistent state itself is described independently by `policies`.
    pub fn required_workspace<T: Tensor>(self, batch: usize) -> Result<u64, QsaError> {
        if batch == 0 || batch > i32::MAX as usize {
            return Err(QsaError::Geometry);
        }
        let raw = (self.capacity() as u64)
            .checked_mul(self.dimensions as u64)
            .ok_or(QsaError::Geometry)?;
        let values = raw
            .checked_add(2 * self.rotary_dimensions as u64)
            .and_then(|v| v.checked_add(self.control_width() as u64))
            .ok_or(QsaError::Geometry)?;
        values
            .checked_mul(32)
            .and_then(|v| v.checked_add(16 * std::mem::size_of::<T>() as u64))
            .and_then(|v| v.checked_mul(batch as u64))
            .ok_or(QsaError::Geometry)
    }
    fn admit<T: Tensor>(self, batch: usize, bytes: u64) -> Result<(), QsaError> {
        let required = self.required_workspace::<T>(batch)?;
        if required > bytes {
            return Err(QsaError::Workspace {
                required,
                limit: bytes,
            });
        }
        Ok(())
    }
    fn validate_lane<T: Tensor>(self, lane: &QsaPartialBlock<T>) -> Result<(), QsaError> {
        let count = lane.positions.len();
        if lane.ratio != self.ratio
            || lane.dimensions != self.dimensions
            || count >= self.ratio as usize
            || lane.last_seen.is_some_and(|p| p < 0)
            || lane
                .positions
                .iter()
                .any(|p| *p < 0 || lane.last_seen.is_none_or(|last| *p > last))
            || !lane.positions.windows(2).all(|p| p[0] < p[1])
        {
            return Err(QsaError::Geometry);
        }
        match (&lane.raw, &lane.first_position) {
            (None, None) if count == 0 => Ok(()),
            (Some(raw), Some(first)) if count > 0 => {
                if raw.shape() != [count as i32, self.dimensions]
                    || raw.element_type() != Some(self.raw_element)
                {
                    return Err(QsaError::Geometry);
                }
                match first {
                    QsaBlockPosition::Offset(offset) if *offset >= 0 => Ok(()),
                    QsaBlockPosition::Embeddings { cosine, sine }
                        if cosine.shape() == sine.shape()
                            && cosine.element_type() == Some(self.rotary_element)
                            && sine.element_type() == Some(self.rotary_element)
                            && cosine.shape().last() == Some(&self.rotary_dimensions)
                            && cosine.shape().iter().rev().skip(1).all(|n| *n == 1) =>
                    {
                        Ok(())
                    }
                    _ => Err(QsaError::Geometry),
                }
            }
            _ => Err(QsaError::Geometry),
        }
    }
    /// Publishes every tail component together after validating and constructing
    /// all replacement tensors. Stream appends participate in the caller's unit transaction.
    pub fn write<B: NeuralBackend, S: RuntimeStateComponents<B>>(
        self,
        state: &mut S,
        lanes: &[QsaPartialBlock<B::Tensor>],
        workspace_bytes: u64,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), QsaError> {
        self.admit::<B::Tensor>(lanes.len(), workspace_bytes)?;
        for role in self.roles() {
            state.fixed_component(role)?;
        }
        for lane in lanes {
            self.validate_lane(lane)?;
        }
        let zero = |shape: &[i32], element| -> Result<B::Tensor, QsaError> {
            let count = shape
                .iter()
                .try_fold(1usize, |n, d| n.checked_mul(*d as usize))
                .ok_or(QsaError::Geometry)?;
            Ok(B::Tensor::from_f32_slice(&vec![0.; count], shape, context)?
                .cast_float(element, context)?)
        };
        let raw_zero = zero(&[self.capacity(), self.dimensions], self.raw_element)?;
        let rotary_zero = zero(&[1, self.rotary_dimensions], self.rotary_element)?;
        let mut controls = Vec::with_capacity(lanes.len() * self.control_width() as usize);
        let mut raws = Vec::with_capacity(lanes.len());
        let mut cosines = Vec::with_capacity(lanes.len());
        let mut sines = Vec::with_capacity(lanes.len());
        for lane in lanes {
            let mode = match &lane.first_position {
                None => -2,
                Some(QsaBlockPosition::Offset(p)) => *p,
                Some(QsaBlockPosition::Embeddings { .. }) => -1,
            };
            controls.extend([
                lane.last_seen.unwrap_or(-1),
                lane.positions.len() as i32,
                mode,
            ]);
            controls.extend(&lane.positions);
            controls.extend(std::iter::repeat_n(
                -1,
                (self.ratio - 1) as usize - lane.positions.len(),
            ));
            let raw = match &lane.raw {
                Some(raw) => B::Tensor::pad(
                    raw,
                    &[(0, self.capacity() - raw.dim(0)), (0, 0)],
                    eredu_nn::PadMode::Constant,
                    context,
                )?,
                None => raw_zero.clone(),
            };
            raws.push(raw.reshape(&[1, self.capacity(), self.dimensions], context)?);
            let (cosine, sine) = match &lane.first_position {
                Some(QsaBlockPosition::Embeddings { cosine, sine }) => (
                    cosine.reshape(&[1, self.rotary_dimensions], context)?,
                    sine.reshape(&[1, self.rotary_dimensions], context)?,
                ),
                _ => (rotary_zero.clone(), rotary_zero.clone()),
            };
            cosines.push(cosine);
            sines.push(sine);
        }
        let values = [
            B::Tensor::from_i32_slice(
                &controls,
                &[lanes.len() as i32, self.control_width()],
                context,
            )?,
            B::Tensor::concatenate(&raws, 0, context)?.compact(context)?,
            B::Tensor::concatenate(&cosines, 0, context)?.compact(context)?,
            B::Tensor::concatenate(&sines, 0, context)?.compact(context)?,
        ];
        state.replace_fixed_components(
            self.roles()
                .into_iter()
                .zip(values.into_iter().map(Some))
                .collect(),
        )?;
        Ok(())
    }
    /// Restores exact lane-local state. A wholly absent tuple denotes new/reset
    /// state; a partially missing tuple or malformed control vector is rejected.
    pub fn read<B: NeuralBackend, S: RuntimeStateComponents<B>>(
        self,
        state: &mut S,
        batch: usize,
        workspace_bytes: u64,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Vec<QsaPartialBlock<B::Tensor>>, QsaError> {
        self.admit::<B::Tensor>(batch, workspace_bytes)?;
        let mut values = Vec::with_capacity(4);
        for role in self.roles() {
            values.push(state.fixed_component(role)?.clone());
        }
        if values.iter().all(Option::is_none) {
            return (0..batch)
                .map(|_| QsaPartialBlock::new(self.ratio, self.dimensions))
                .collect();
        }
        let values = values
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or(QsaError::Geometry)?;
        for ((value, policy), element) in values.iter().zip(self.policies()).zip([
            TensorElementType::I32,
            self.raw_element,
            self.rotary_element,
            self.rotary_element,
        ]) {
            if value.shape()
                != policy
                    .resolved_shape(batch, 0)
                    .map_err(|_| QsaError::Geometry)?
                || value.element_type() != Some(element)
            {
                return Err(QsaError::Geometry);
            }
        }
        let controls = values[0].to_i32_vec(context)?;
        if controls.len() != batch * self.control_width() as usize {
            return Err(QsaError::Geometry);
        }
        let mut result = Vec::with_capacity(batch);
        for (lane, control) in controls
            .chunks_exact(self.control_width() as usize)
            .enumerate()
        {
            let (last, count, mode) = (control[0], control[1], control[2]);
            if last < -1
                || count < 0
                || count >= self.ratio
                || mode < -2
                || (count == 0) != (mode == -2)
                || control[3 + count as usize..].iter().any(|v| *v != -1)
            {
                return Err(QsaError::Geometry);
            }
            let raw = if count == 0 {
                None
            } else {
                Some(
                    values[1]
                        .index(
                            &[Index::At(lane as i32), Index::Range(0, count), Index::Full],
                            context,
                        )?
                        .compact(context)?,
                )
            };
            let first_position = match mode {
                -2 => None,
                -1 => {
                    let take = |i: usize| -> Result<B::Tensor, QsaError> {
                        Ok(values[i]
                            .index(
                                &[Index::Range(lane as i32, lane as i32 + 1), Index::Full],
                                context,
                            )?
                            .reshape(&[1, 1, 1, self.rotary_dimensions], context)?
                            .compact(context)?)
                    };
                    Some(QsaBlockPosition::Embeddings {
                        cosine: take(2)?,
                        sine: take(3)?,
                    })
                }
                offset => Some(QsaBlockPosition::Offset(offset)),
            };
            let partial = QsaPartialBlock {
                ratio: self.ratio,
                dimensions: self.dimensions,
                raw,
                positions: control[3..3 + count as usize].to_vec(),
                first_position,
                last_seen: (last >= 0).then_some(last),
            };
            self.validate_lane(&partial)?;
            result.push(partial);
        }
        Ok(result)
    }
}
