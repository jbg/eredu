use super::CompressedLatentCache;
use crate::backend::array_copy::IsolatedArrayCopy;
use safemlx::{Array, Stream, error::Exception};
use std::cell::RefCell;

/// The two logical resident views in latent-then-rotary order. Storage padding
/// remains part of the source backing but is not a requested destination slot.
/// The caller keeps source custody and exclusive settled access through copy
/// completion/recovery. This borrow grants no allocation or execution authority.
pub(crate) struct PreparedCompressedCopy<'a> {
    source: &'a CompressedLatentCache,
    latent: Option<&'a Array>,
    rotary: Option<&'a Array>,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum InvalidResidentCopy {
    #[error("paged compressed copy requires an independent manager-copy mechanism")]
    Paged,
    #[error("invalid resident compressed state frontier or capacity step")]
    Frontier,
    #[error("incomplete resident compressed state inventory")]
    Inventory,
}

impl<'a> PreparedCompressedCopy<'a> {
    pub(crate) fn preparation_control_bytes() -> Option<usize> {
        use std::mem::size_of;
        size_of::<Self>()
            .checked_add(size_of::<Result<Self, InvalidResidentCopy>>())?
            .checked_add(size_of::<&CompressedLatentCache>())?
            .checked_add(size_of::<[&Option<Array>; 4]>())
    }

    pub(super) fn new(source: &'a CompressedLatentCache) -> Result<Self, Exception> {
        Self::new_fixed(source).map_err(Exception::from_source)
    }

    pub(super) fn new_fixed(
        source: &'a CompressedLatentCache,
    ) -> Result<Self, InvalidResidentCopy> {
        if source.paged.is_some() {
            return Err(InvalidResidentCopy::Paged);
        }
        if source.offset != source.length
            || source.length < 0
            || source.capacity < source.length
            || source.step <= 0
        {
            return Err(InvalidResidentCopy::Frontier);
        }
        match (
            &source.latent_storage,
            &source.rotary_key_storage,
            &source.latent,
            &source.rotary_key,
        ) {
            (None, None, None, None) if source.length == 0 && source.capacity == 0 => {}
            (Some(_), Some(_), Some(_), Some(_)) => {}
            _ => return Err(InvalidResidentCopy::Inventory),
        }
        Ok(Self {
            source,
            latent: source.latent.as_ref(),
            rotary: source.rotary_key.as_ref(),
        })
    }

    /// Visits exact borrowed operands without cloning handles, querying native
    /// metadata, constructing views or allocating a slot list. Aliased inputs
    /// are visited twice because they require separate destination copies.
    pub(crate) fn visit_operands(&self, visitor: &mut dyn FnMut(&'a Array)) {
        for source in self.latent.into_iter().chain(self.rotary) {
            visitor(source);
        }
    }

    pub(crate) fn copy(self, stream: &Stream) -> Result<CompressedLatentCache, Exception> {
        self.copy_with(&mut |copy| copy.copy(stream), &mut Array::try_clone_handle)
    }

    /// Retains every compaction intermediate and final array before the next
    /// fallible native operation. The caller retains this collector through
    /// exact native recovery; it is not a list of publishable final allocations.
    pub(crate) fn copy_retained(
        self,
        stream: &Stream,
        roots: &RefCell<Vec<Array>>,
    ) -> Result<CompressedLatentCache, Exception> {
        self.copy_with(
            &mut |copy| copy.copy_retained(stream, roots),
            &mut Array::try_clone_handle,
        )
    }

    pub(crate) fn copy_control_bytes<E>(&self) -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let frames = [
            size_of::<Self>(),
            size_of::<&Self>(),
            size_of::<&Stream>(),
            size_of::<CompressedLatentCache>(),
            size_of::<Result<CompressedLatentCache, E>>(),
            size_of::<Option<Result<Array, E>>>(),
            size_of::<Result<Option<Array>, E>>(),
            size_of::<&mut dyn FnMut(IsolatedArrayCopy<'a>) -> Result<Array, E>>(),
            size_of::<&mut dyn FnMut(&'a Array) -> Result<Array, E>>(),
            size_of::<&mut dyn FnMut(&Array) -> Result<Array, E>>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }

    /// Copies logical views once, then retains their exact returned backing as
    /// destination stores through the caller's separately funded alias worker.
    pub(crate) fn copy_with<E>(
        self,
        copy: &mut dyn FnMut(IsolatedArrayCopy<'a>) -> Result<Array, E>,
        alias: &mut dyn FnMut(&Array) -> Result<Array, E>,
    ) -> Result<CompressedLatentCache, E> {
        let mut copy = |source| copy(IsolatedArrayCopy::new(source));
        let latent = self.latent.map(&mut copy).transpose()?;
        let rotary = self.rotary.map(copy).transpose()?;
        Ok(CompressedLatentCache {
            latent_storage: latent.as_ref().map(&mut *alias).transpose()?,
            rotary_key_storage: rotary.as_ref().map(alias).transpose()?,
            latent,
            rotary_key: rotary,
            offset: self.source.offset,
            length: self.source.length,
            capacity: self.source.length,
            step: self.source.step,
            paged: None,
        })
    }
}

mod copied_workspace;

#[cfg(test)]
mod tests;
