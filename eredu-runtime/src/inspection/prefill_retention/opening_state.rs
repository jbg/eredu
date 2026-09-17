//! Lexical access to the actual state at the canonical pre-prepare boundary.
use crate::{RuntimeState, StateError};
use eredu_nn::NeuralBackend;
use std::marker::PhantomData;

trait OpeningValues<T> {
    fn visit(&self, visitor: &mut dyn FnMut(&T)) -> Result<(), StateError>;
}

/// Borrowed state values from the exact upcoming SessionPrefill boundary.
///
/// This is source access, not evidence of registered storage, native settlement,
/// complete model/residency inventory, available bytes or execution authority.
/// Concrete state iteration may allocate; the enclosing original preparation
/// must account for that iteration and any native inventory it constructs.
pub struct PrefillOpeningState<'a, T> {
    source: &'a dyn OpeningValues<T>,
}

impl<T> PrefillOpeningState<'_, T> {
    /// Visits all tensor values directly retained by the actual local state. No tensor
    /// handles are cloned by this view. A state error leaves prior visits intact;
    /// consumers must reject incomplete inventories rather than infer emptiness.
    pub fn visit(&self, visitor: &mut dyn FnMut(&T)) -> Result<(), StateError> {
        self.source.visit(visitor)
    }

    /// Lends the same traversal through a borrowed tensor representation adapter.
    /// The adapter must preserve each tensor's underlying storage identity. It
    /// creates no source, account, registration or completion authority. Native
    /// composition uses its fixed tensor wrapper conversion, not application
    /// supplied mapping, when building an original opening inventory.
    pub fn with_tensor_adapter<U, R>(
        &self,
        map: for<'v> fn(&'v T) -> &'v U,
        operation: impl FnOnce(&PrefillOpeningState<'_, U>) -> R,
    ) -> R {
        let mapped = MappedOpeningValues { source: self, map };
        operation(&PrefillOpeningState { source: &mapped })
    }

    /// Fixed view/source/two-representation-adapter construction overlap only.
    /// State iterators, native inventory tables, model/residency-manager storage
    /// and compiler/allocator overhead are separate obligations.
    pub fn control_peak_bytes() -> Option<u64> {
        std::mem::size_of::<Self>()
            .checked_mul(3)?
            .checked_add(std::mem::size_of::<RuntimeOpeningState<'_, (), ()>>())?
            .checked_add(std::mem::size_of::<MappedOpeningValues<'_, (), ()>>().checked_mul(2)?)?
            .checked_mul(3)?
            .try_into()
            .ok()
    }
}

struct MappedOpeningValues<'a, T, U> {
    source: &'a PrefillOpeningState<'a, T>,
    map: for<'v> fn(&'v T) -> &'v U,
}
impl<T, U> OpeningValues<U> for MappedOpeningValues<'_, T, U> {
    fn visit(&self, visitor: &mut dyn FnMut(&U)) -> Result<(), StateError> {
        self.source.visit(&mut |value| visitor((self.map)(value)))
    }
}

// This concrete adapter is created only while SessionPrefill holds its exclusive
// session loan. The state owner supplies the complete local storage traversal.
pub(crate) struct RuntimeOpeningState<'a, B, S> {
    state: &'a S,
    backend: PhantomData<B>,
}
impl<'a, B, S> RuntimeOpeningState<'a, B, S>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
{
    pub(crate) fn new(state: &'a S) -> Self {
        Self {
            state,
            backend: PhantomData,
        }
    }

    pub(crate) fn borrow(&self) -> PrefillOpeningState<'_, B::Tensor> {
        PrefillOpeningState { source: self }
    }
}
impl<B, S> OpeningValues<B::Tensor> for RuntimeOpeningState<'_, B, S>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
{
    fn visit(&self, visitor: &mut dyn FnMut(&B::Tensor)) -> Result<(), StateError> {
        self.state.visit_all_retained_values(visitor)
    }
}

#[cfg(test)]
mod tests;
