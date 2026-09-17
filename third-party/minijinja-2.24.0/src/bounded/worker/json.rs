//! One registered JSON formatter over the actual fixed temporary destinations.
use super::*;
#[cfg(feature = "json")]
pub(in crate::bounded) type Scratch<'a> = crate::bounded::json_format::Scratch<'a,Slot>;
use crate::bounded::render::{JsonCapacity, RenderPlanError};
#[cfg(feature = "json")]
mod consumer;
#[cfg(feature = "json")]
mod view;
#[cfg(not(feature = "json"))]
pub(in crate::bounded) struct Scratch<'a>(std::marker::PhantomData<&'a ()>);
pub(in crate::bounded) fn buffer_bytes(capacity: JsonCapacity) -> Option<usize> {
    #[cfg(feature = "json")]
    {
        capacity.bytes_for::<Slot>()
    }
    #[cfg(not(feature = "json"))]
    {
        (capacity == JsonCapacity::default()).then_some(0)
    }
}
pub(in crate::bounded) fn prepare<'a>(
    capacity: JsonCapacity,
) -> Result<Scratch<'a>, RenderPlanError> {
    #[cfg(feature = "json")]
    {
        Scratch::fixed(capacity).map_err(|error| match error {
            crate::bounded::json_format::Error::Reserve(cause) => RenderPlanError::Reserve(cause),
            _ => RenderPlanError::Overflow,
        })
    }
    #[cfg(not(feature = "json"))]
    {
        if capacity != JsonCapacity::default() {
            return Err(RenderPlanError::Geometry);
        }
        Ok(Scratch(std::marker::PhantomData))
    }
}
#[cfg(not(feature = "json"))]
impl Engine<'_, '_, '_> {
    pub(super) fn apply_json(&mut self, _: Option<u16>) -> Result<(), Error> {
        Err(Error::Geometry)
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    #[cfg(feature = "json")]
    {
        consumer::control_bytes()
    }
    #[cfg(not(feature = "json"))]
    {
        Some(std::mem::size_of::<JsonCapacity>())
    }
}
