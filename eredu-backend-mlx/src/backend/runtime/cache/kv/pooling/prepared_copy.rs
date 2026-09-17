use super::PoolingCache;
use crate::backend::array_copy::IsolatedArrayCopy;
use safemlx::{Array, Stream, error::Exception};
use std::cell::RefCell;

/// Exact optional operands in pending-values, pending-gates, pooled,
/// overlap-values, overlap-gates order. The source borrow preserves scalar
/// metadata and slot presence until the destination is complete. The caller
/// retains source custody, settled exclusive access and recovery authority;
/// this leaf grants no allocation or execution permission.
pub(crate) struct PreparedPoolingCopy<'a> {
    source: &'a PoolingCache,
    slots: [Option<&'a Array>; 5],
}

impl<'a> PreparedPoolingCopy<'a> {
    pub(super) fn new(source: &'a PoolingCache) -> Self {
        Self {
            source,
            slots: source.state_arrays(),
        }
    }

    /// Visits the same ordered operands the worker consumes, without cloning
    /// descriptors, constructing views or allocating an intermediate slot list.
    /// Aliased slots remain separate requested destinations.
    pub(crate) fn visit_operands(&self, visitor: &mut dyn FnMut(&'a Array)) {
        for source in self.slots.into_iter().flatten() {
            visitor(source);
        }
    }

    /// Fixed controls for an operand-free invocation of the same copy worker.
    pub(crate) fn empty_copy_control_bytes(&self) -> Option<usize> {
        if self.slots.iter().any(Option::is_some) {
            return None;
        }
        self.control_bytes::<Exception>()
    }

    pub(crate) fn control_bytes<E>(&self) -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let frames = [
            size_of::<Self>(),
            size_of::<&Self>(),
            size_of::<&Stream>(),
            size_of::<PoolingCache>(),
            size_of::<Result<PoolingCache, E>>(),
            size_of::<Option<Result<Array, E>>>(),
            size_of::<Result<Option<Array>, E>>(),
            size_of::<&mut dyn FnMut(IsolatedArrayCopy<'a>) -> Result<Array, E>>(),
            size_of::<&mut dyn FnMut(&'a Array) -> Result<Array, E>>(),
            size_of::<Option<usize>>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }

    pub(crate) fn copy(self, stream: &Stream) -> Result<PoolingCache, Exception> {
        self.copy_with(&mut |copy| copy.copy(stream))
    }

    /// Keeps each produced contiguous intermediate and independent copy in the
    /// caller's recovery collector before the next fallible native operation.
    /// The collector includes temporary roots, not just final destinations.
    pub(crate) fn copy_retained(
        self,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
    ) -> Result<PoolingCache, Exception> {
        self.copy_with(&mut |copy| copy.copy_retained(stream, roots))
    }

    pub(crate) fn copy_with<E>(
        self,
        copy: &mut dyn FnMut(IsolatedArrayCopy<'a>) -> Result<Array, E>,
    ) -> Result<PoolingCache, E> {
        let mut copy = |source| copy(IsolatedArrayCopy::new(source));
        let [
            pending_values,
            pending_gates,
            pooled,
            overlap_values,
            overlap_gates,
        ] = self.slots;
        Ok(PoolingCache {
            ratio: self.source.ratio,
            pending_values: pending_values.map(&mut copy).transpose()?,
            pending_gates: pending_gates.map(&mut copy).transpose()?,
            pooled: pooled.map(&mut copy).transpose()?,
            overlap_values: overlap_values.map(&mut copy).transpose()?,
            overlap_gates: overlap_gates.map(copy).transpose()?,
            processed_tokens: self.source.processed_tokens,
        })
    }
}

#[cfg(test)]
mod tests;
