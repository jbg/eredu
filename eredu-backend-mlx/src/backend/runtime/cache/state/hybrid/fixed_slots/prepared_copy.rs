//! Borrowed fixed-role copy leaf, sharing the native/metadata array program.

use super::{FixedStateSlots, MlxTensor, Slot};
use crate::backend::array_copy::IsolatedArrayCopy;
use safemlx::{error::Exception, Array, Stream};
use std::cell::RefCell;

/// Binds the exact slot table without allocation, array cloning, or inspection.
/// The caller keeps the source settled and exclusive through tracing/copying;
/// a Rust borrow alone cannot prevent mutation through external native aliases.
/// This plan carries neither allocation authority nor a complete snapshot bound.
pub(in crate::backend::runtime::cache::state::hybrid) struct PreparedFixedStateCopy<'a> {
    source: &'a FixedStateSlots,
}

impl<'a> PreparedFixedStateCopy<'a> {
    pub(super) const fn new(source: &'a FixedStateSlots) -> Self {
        Self { source }
    }

    /// Visits present slots in sorted role order, including aliased operands
    /// separately. The visitor receives the actual original descriptors; absent
    /// slots have no numerical operand and are preserved by the copy worker.
    pub(in crate::backend::runtime::cache::state::hybrid) fn visit_operands(
        &self,
        visitor: &mut dyn FnMut(&'a Array),
    ) {
        for (_, value) in self.source.iter() {
            if let Some(value) = value {
                visitor(value.as_array());
            }
        }
    }

    /// Exact destination slot-box payload, including absent roles. Excludes
    /// tensor backing, table identity/custody metadata, enclosing layer/state
    /// values, plan construction scratch, recovery collector and bookkeeping.
    pub(in crate::backend::runtime::cache::state::hybrid) fn payload_bytes(&self) -> Option<u64> {
        self.source.payload_bytes()
    }

    pub(in crate::backend::runtime::cache::state::hybrid) fn copy(
        self,
        stream: &Stream,
    ) -> Result<FixedStateSlots, Exception> {
        self.copy_with_roots(stream, None)
    }

    /// Retains every contiguous temporary and final destination before the next
    /// fallible operation. The caller must retain this collector through exact
    /// completion or failure recovery; only returned slots are final outputs.
    pub(in crate::backend::runtime::cache::state::hybrid) fn copy_retained(
        self,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
    ) -> Result<FixedStateSlots, Exception> {
        self.copy_with_roots(stream, Some(roots))
    }

    fn copy_with_roots(
        self,
        stream: &Stream,
        roots: Option<&RefCell<Vec<Array>>>,
    ) -> Result<FixedStateSlots, Exception> {
        self.source
            .try_map_values(|value| copy_value(value, stream, roots))
    }
}

// Both unquoted snapshot copying and admitted grouped construction use this
// exact numerical leaf. The caller separately funds/owns the actual role slot.
fn copy_value(
    value: &MlxTensor,
    stream: &Stream,
    roots: Option<&RefCell<Vec<Array>>>,
) -> Result<MlxTensor, Exception> {
    let prepared = IsolatedArrayCopy::new(value.as_array());
    let copied = match roots {
        Some(roots) => prepared.copy_retained(stream, roots)?,
        None => prepared.copy(stream)?,
    };
    #[cfg(test)]
    tests::after_slot_copy()?;
    Ok(MlxTensor::from_array(copied))
}

pub(in crate::backend::runtime::cache::state::hybrid) fn copy_slot_retained(
    source: &Slot,
    stream: &Stream,
    roots: &RefCell<Vec<Array>>,
) -> Result<Slot, Exception> {
    Ok((
        source.0,
        source
            .1
            .as_ref()
            .map(|value| copy_value(value, stream, Some(roots)))
            .transpose()?,
    ))
}

#[cfg(test)]
mod tests;
