//! Bounded exact integer state using the shared fixed-component lifecycle.
use crate::{RuntimeStateComponents, StateError};
use eredu_core::cache::{
    MutableStateResidency, StateTensorDimension, StateTensorDtype, StateTensorPolicy,
    StateTensorRole,
};
use eredu_nn::{NeuralBackend, Tensor};

/// Architecture-declared integer history; reset values are equation policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntegerHistorySpec {
    slot: u32,
    capacity: i32,
    initial: i32,
}
/// Geometry, component binding or exact integer transfer failure.
#[derive(Debug, thiserror::Error)]
pub enum IntegerHistoryError {
    /// State and request geometry differ.
    #[error("invalid bounded integer history geometry")]
    Geometry,
    /// The owning layer did not declare the requested component.
    #[error(transparent)]
    State(#[from] StateError),
    /// Native integer transfer failed; no floating fallback is attempted.
    #[error(transparent)]
    Tensor(#[from] eredu_nn::Error),
}
impl IntegerHistorySpec {
    /// Declares a nonzero history capacity and literal initial integer.
    pub fn new(slot: u32, capacity: i32, initial: i32) -> Result<Self, IntegerHistoryError> {
        if capacity <= 0 {
            return Err(IntegerHistoryError::Geometry);
        }
        Ok(Self {
            slot,
            capacity,
            initial,
        })
    }
    /// Exact fixed-state role shared by discovery, snapshots and persistence.
    pub fn role(self) -> StateTensorRole {
        StateTensorRole::IntegerHistory { slot: self.slot }
    }
    /// Declares bounded integer geometry through the ordinary state layout.
    pub fn policy(self) -> StateTensorPolicy {
        StateTensorPolicy::new(
            self.role(),
            vec![
                StateTensorDimension::Batch,
                StateTensorDimension::fixed(self.capacity).expect("validated capacity"),
            ],
            StateTensorDtype::Int32,
            MutableStateResidency::AlwaysDeviceMutable,
        )
        .expect("valid integer history policy")
    }
    fn count(self, batch: usize) -> Result<usize, IntegerHistoryError> {
        if batch == 0 || batch > i32::MAX as usize {
            return Err(IntegerHistoryError::Geometry);
        }
        batch
            .checked_mul(self.capacity as usize)
            .ok_or(IntegerHistoryError::Geometry)
    }
    /// Reads the exact retained host integers, supplying declared initialization
    /// only when the component is empty. No inference from embeddings is allowed.
    pub fn read<B: NeuralBackend, S: RuntimeStateComponents<B>>(
        self,
        state: &mut S,
        batch: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Vec<i32>, IntegerHistoryError> {
        let count = self.count(batch)?;
        let Some(value) = state.fixed_component(self.role())? else {
            return Ok(vec![self.initial; count]);
        };
        if value.element_type() != Some(eredu_nn::TensorElementType::I32)
            || value.shape() != [batch as i32, self.capacity]
        {
            return Err(IntegerHistoryError::Geometry);
        }
        let result = value.to_i32_vec(context)?;
        if result.len() != count {
            return Err(IntegerHistoryError::Geometry);
        }
        Ok(result)
    }
    /// Proposes only the bounded suffix; the caller retains transaction authority
    /// and publishes the resulting integers after its unit succeeds.
    pub fn next(
        self,
        previous: &[i32],
        input: &[i32],
        batch: usize,
        tokens: usize,
    ) -> Result<Vec<i32>, IntegerHistoryError> {
        let count = self.count(batch)?;
        if tokens == 0 || previous.len() != count || batch.checked_mul(tokens) != Some(input.len())
        {
            return Err(IntegerHistoryError::Geometry);
        }
        let width = self.capacity as usize;
        let mut output = Vec::with_capacity(count);
        for b in 0..batch {
            if tokens < width {
                output.extend_from_slice(&previous[b * width + tokens..(b + 1) * width]);
            }
            output.extend_from_slice(
                &input[b * tokens + tokens.saturating_sub(width)..(b + 1) * tokens],
            );
        }
        Ok(output)
    }
    /// Replaces the ordinary fixed component. Its owning driver completes and
    /// checkpoints this tensor together with recurrent/convolution/attention state.
    pub fn write<B: NeuralBackend, S: RuntimeStateComponents<B>>(
        self,
        state: &mut S,
        values: &[i32],
        batch: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), IntegerHistoryError> {
        if values.len() != self.count(batch)? {
            return Err(IntegerHistoryError::Geometry);
        }
        let destination = state.fixed_component(self.role())?;
        let next = B::Tensor::from_i32_slice(values, &[batch as i32, self.capacity], context)?;
        if next.element_type() != Some(eredu_nn::TensorElementType::I32) {
            return Err(IntegerHistoryError::Geometry);
        }
        *destination = Some(next);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_suffix_preserves_exact_integers_and_reset_literal() {
        let spec = IntegerHistorySpec::new(3, 3, 7).unwrap();
        let prior = [i32::MAX, 16777217, -16777217, 1, 2, 3];
        assert_eq!(
            spec.next(&prior, &[i32::MIN, 99], 2, 1).unwrap(),
            [16777217, -16777217, i32::MIN, 2, 3, 99]
        );
        assert_eq!(
            spec.next(&prior, &[0, 1, 2, 3, 4, 5, 6, 7], 2, 4).unwrap(),
            [1, 2, 3, 5, 6, 7]
        );
        assert_eq!(spec.policy().resolved_shape(2, 100000).unwrap(), [2, 3]);
        assert!(spec.next(&prior, &[1], 2, 1).is_err());
        assert!(IntegerHistorySpec::new(0, 0, 0).is_err());
    }
}
