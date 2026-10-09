//! Bounded, parameter-owned relative-position constants for single-frame attention.

use eredu_nn::{
    Error, LinearOperator, NeuralBackend, ParameterVisitor, ParameterVisitorMut, Parameterized,
    Tensor, TensorContextKey, TensorElementType,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PositionKey {
    context: TensorContextKey,
    width: i32,
    element: Option<TensorElementType>,
}

#[derive(Debug, Clone)]
struct PositionValues<T> {
    key: PositionKey,
    positions: T,
    projected: Option<T>,
}

/// The layer owns one constant geometry, independent of candidate causal state.
/// Large prefill chunks bypass the cache rather than increasing retained memory.
#[derive(Debug)]
pub(super) struct RelativePositionProjection<B: NeuralBackend> {
    linear: B::Linear,
    single_frame_total: i32,
    cached: Option<PositionValues<B::Tensor>>,
}

impl<B: NeuralBackend> Clone for RelativePositionProjection<B> {
    fn clone(&self) -> Self {
        // A new parameter owner starts cold; snapshots copy causal state only.
        Self::new(self.linear.clone(), self.single_frame_total)
    }
}

impl<B: NeuralBackend> Parameterized<B::Tensor> for RelativePositionProjection<B> {
    fn visit_parameters<'a, V>(&'a self, visitor: &mut V)
    where
        V: ParameterVisitor<'a, B::Tensor>,
    {
        self.linear.visit_parameters(visitor);
    }

    fn visit_parameters_mut<'a, V>(&'a mut self, visitor: &mut V)
    where
        V: ParameterVisitorMut<'a, B::Tensor>,
    {
        // Visitors can publish arbitrary replacements. Position data is
        // parameter independent, but its projection must be rebuilt.
        self.invalidate_projection();
        self.linear.visit_parameters_mut(visitor);
    }

    fn set_trainable(&mut self, trainable: bool) {
        self.invalidate_projection();
        self.linear.set_trainable(trainable);
    }
}

impl<B: NeuralBackend> RelativePositionProjection<B> {
    pub(super) fn new(linear: B::Linear, single_frame_total: i32) -> Self {
        Self {
            linear,
            single_frame_total,
            cached: None,
        }
    }

    fn invalidate_projection(&mut self) {
        if let Some(cached) = &mut self.cached {
            cached.projected = None;
        }
    }

    pub(super) fn set_compute_element(
        &mut self,
        element: Option<TensorElementType>,
    ) -> Result<(), Error> {
        self.invalidate_projection();
        self.linear.set_compute_element(element)
    }

    pub(super) fn forward(
        &mut self,
        total: i32,
        width: i32,
        prototype: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let identity = if total == self.single_frame_total {
            B::Tensor::immutable_context_key(context)?
        } else {
            None
        };
        let Some(identity) = identity else {
            return self.linear.forward(
                &relative_positions(total, width, prototype, context)?,
                context,
            );
        };
        let key = PositionKey {
            context: identity,
            width,
            element: prototype.element_type(),
        };
        if self.cached.as_ref().is_none_or(|cached| cached.key != key) {
            let positions = relative_positions(total, width, prototype, context)?;
            self.cached = Some(PositionValues {
                key,
                positions,
                projected: None,
            });
        }
        let cached = self
            .cached
            .as_mut()
            .expect("position constants initialized");
        if let Some(projected) = &cached.projected {
            return Ok(projected.clone());
        }
        let projected = self.linear.forward(&cached.positions, context)?;
        cached.projected = Some(projected.clone());
        Ok(projected)
    }
}

fn relative_positions<T: Tensor>(
    total: i32,
    width: i32,
    prototype: &T,
    context: &T::Context,
) -> Result<T, Error> {
    let coefficient = -(10000.0_f64.ln() / width as f64) as f32;
    let mut data = Vec::with_capacity((2 * total - 1) as usize * width as usize);
    for position in (1 - total..total).rev() {
        for feature in (0..width).step_by(2) {
            let phase = position as f32 * (feature as f32 * coefficient).exp();
            data.push(phase.sin());
            data.push(phase.cos());
        }
    }
    let tensor = T::from_f32_slice(&data, &[1, 2 * total - 1, width], context)?;
    match prototype.element_type() {
        Some(dtype) if dtype != eredu_nn::TensorElementType::F32 => {
            tensor.cast_float(dtype, context)
        }
        _ => Ok(tensor),
    }
}
