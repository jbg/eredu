use super::ConcatKeyValueCache;
use crate::backend::array_copy::IsolatedArrayCopy;
use safemlx::{Array, Stream, error::Exception};
use std::cell::RefCell;

/// Exact stored slots, including padding. The source borrow preserves every
/// cache field until the destination is complete; no logical slice is created.
pub(crate) struct PreparedConcatCopy<'a> {
    source: &'a ConcatKeyValueCache,
    keys: Option<IsolatedArrayCopy<'a>>,
    values: Option<IsolatedArrayCopy<'a>>,
}

impl<'a> PreparedConcatCopy<'a> {
    pub(super) fn new(source: &'a ConcatKeyValueCache) -> Self {
        Self {
            source,
            keys: source.keys.as_ref().map(IsolatedArrayCopy::new),
            values: source.values.as_ref().map(IsolatedArrayCopy::new),
        }
    }

    /// Visits the stored padded operands in key/value order. The immutable
    /// source borrow is the same one retained by both closed copy primitives.
    pub(crate) fn visit_operands(&self, visitor: &mut dyn FnMut(&'a Array)) {
        for source in self.source.keys.iter().chain(self.source.values.iter()) {
            visitor(source);
        }
    }

    /// Controls for the existing leaf worker only when it has no native operand.
    /// A populated source needs numerical copy admission and is not certified here.
    pub(crate) fn empty_copy_control_bytes(&self) -> Option<usize> {
        if self.keys.is_some() || self.values.is_some() {
            return None;
        }
        self.control_bytes::<Exception>()
    }

    /// Named representation-worker controls; each supplied numerical copy
    /// callback must independently supply its own admission and completion.
    pub(crate) fn control_bytes<E>(&self) -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let frames = [
            size_of::<Self>(),
            size_of::<&Self>(),
            size_of::<&Stream>(),
            size_of::<ConcatKeyValueCache>(),
            size_of::<Result<ConcatKeyValueCache, E>>(),
            size_of::<Option<Result<Array, E>>>(),
            size_of::<Result<Option<Array>, E>>(),
            size_of::<&mut dyn FnMut(IsolatedArrayCopy<'a>) -> Result<Array, E>>(),
            size_of::<Option<usize>>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }

    pub(crate) fn copy(self, stream: &Stream) -> Result<ConcatKeyValueCache, Exception> {
        self.copy_with(&mut |copy| copy.copy(stream))
    }

    /// Preserves compaction intermediates and final outputs in the caller's
    /// existing recovery collector before any later fallible copy. This is not
    /// a list of final publication roots and grants no allocation authority.
    pub(crate) fn copy_retained(
        self,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
    ) -> Result<ConcatKeyValueCache, Exception> {
        self.copy_with(&mut |copy| copy.copy_retained(stream, roots))
    }

    pub(crate) fn copy_with<E>(
        self,
        copy: &mut dyn FnMut(IsolatedArrayCopy<'a>) -> Result<Array, E>,
    ) -> Result<ConcatKeyValueCache, E> {
        let keys = self.keys.map(&mut *copy).transpose()?;
        let values = self.values.map(copy).transpose()?;
        // Even aliased source slots have distinct requested destinations. Keep
        // partial-copy cleanup within the caller's existing recovery scope.
        Ok(ConcatKeyValueCache {
            keys,
            values,
            key_only: self.source.key_only,
            offset: self.source.offset,
            length: self.source.length,
            capacity: self.source.capacity,
            step: self.source.step,
            max_size: self.source.max_size,
            attention_window: self.source.attention_window,
        })
    }

    /// Diagnostic for this numerical component, not whole-snapshot admission.
    /// Source and destination remain retained together; unknown source backing
    /// must be checked separately from the report's new-allocation total.
    #[cfg(test)]
    fn component_report(
        &self,
        context: &'a eredu_nn::workspace::WorkspaceContext,
    ) -> Result<(bool, eredu_nn::workspace::WorkspaceTraceReport), eredu_nn::Error> {
        use crate::backend::nn::workspace::ExistingArrayProjection;

        let mut projection = ExistingArrayProjection::new(context);
        let mut retained = self
            .source
            .arrays()
            .map(|array| projection.project(array))
            .collect::<Result<Vec<_>, _>>()?;
        context.begin_state_span(&retained)?;
        for copy in self.keys.iter().chain(self.values.iter()) {
            retained.push(copy.trace(&mut projection)?);
        }
        Ok((projection.is_complete(), context.report(&retained)?))
    }
}

#[cfg(test)]
mod tests;
