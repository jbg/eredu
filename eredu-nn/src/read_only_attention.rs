//! Attention over retained context and a transient, uncommitted token span.

use crate::{AttentionCache, AttentionRequest, Error, Tensor};

/// A cache mechanism able to attend without appending projected state.
///
/// Implementations may update scratch storage and residency telemetry, but must
/// preserve retained values, logical frontier, and eviction eligibility. Paged
/// implementations scan leased blocks and transient keys as one softmax domain.
pub trait ReadOnlyAttentionCache<T: Tensor>: AttentionCache<T> {
    /// Attends to retained context followed by the request's transient keys and
    /// values. The caller supplies the complete positional visibility mask.
    fn attention_with_transient(
        &mut self,
        request: AttentionRequest<'_, T>,
        context: &T::Context,
    ) -> Result<T, Error>;
}

/// A non-appending cache view for ordinary projected-attention components.
///
/// Projected keys/values returned by `update_for_attention` are transient. The
/// following attention invocation reads the owner's context without advancing
/// it. This view owns no tensors, does not clone a cache, and cannot publish KV.
pub struct ReadOnlyCache<'a, C> {
    owner: &'a mut C,
}

impl<'a, C> ReadOnlyCache<'a, C> {
    /// Borrows a cache exclusively for one read-only invocation.
    pub fn new(owner: &'a mut C) -> Self {
        Self { owner }
    }
}

impl<T: Tensor, C: ReadOnlyAttentionCache<T>> AttentionCache<T> for ReadOnlyCache<'_, C> {
    fn offset(&self) -> i32 {
        self.owner.offset()
    }

    fn max_size(&self) -> Option<i32> {
        self.owner.max_size()
    }

    fn memory_retained_positions(&self) -> Option<u64> {
        self.owner.memory_retained_positions()
    }

    fn uses_blockwise_attention(&self) -> bool {
        self.owner.uses_blockwise_attention()
    }

    fn update_for_attention(
        &mut self,
        keys: T,
        values: T,
        _: &T::Context,
    ) -> Result<(T, T), Error> {
        Ok((keys, values))
    }

    fn attention(
        &mut self,
        request: AttentionRequest<'_, T>,
        context: &T::Context,
    ) -> Result<T, Error> {
        self.owner.attention_with_transient(request, context)
    }
}
