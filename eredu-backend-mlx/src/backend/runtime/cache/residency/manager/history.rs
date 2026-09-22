//! Prepaid checkpoint history entries sharing the ordinary discard predicate.
use super::*;
use eredu_nn::workspace::{HostMetadataFunding, WorkspaceContext, WorkspaceMetadataAllocation};
use std::mem::{size_of, size_of_val};

/// Funding stays outside the strong allocation, including its last alias.
#[derive(Clone, Debug)]
pub(crate) struct CacheHistoryOwner {
    inner: Arc<CacheHistoryRetention>,
    _funding: Option<HostMetadataFunding>,
}
struct Node {
    history: Weak<CacheHistoryRetention>,
    next: Option<CacheHistoryNode>,
}
/// A weak entry also retains the payer because its Arc header survives the
/// checkpoint's last strong alias. No raw/weak owner escapes this wrapper.
pub(super) struct CacheHistoryNode {
    inner: Option<Box<Node>>,
    _funding: Option<HostMetadataFunding>,
}
impl std::fmt::Debug for CacheHistoryNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CacheHistoryNode").finish_non_exhaustive()
    }
}
impl Drop for CacheHistoryNode {
    fn drop(&mut self) {
        // Move out and release each Box allocation before its payer. Iteration
        // avoids recursive stack growth for a long request history.
        let mut next = self.inner.take().and_then(|boxed| {
            let Node { history, next } = *boxed;
            drop(history);
            next
        });
        while let Some(mut owner) = next {
            let Node {
                history,
                next: following,
            } = *owner.inner.take().expect("live history entry");
            drop(history);
            next = following;
            // This node's Box and weak Arc header have retired already.
            drop(owner);
        }
    }
}
impl CacheHistoryNode {
    pub(super) fn inspection_control_bytes() -> Option<usize> {
        let frames = [
            size_of::<(&Self, &CacheBlockId)>(),
            size_of::<Option<&Self>>(),
            size_of::<&Node>(),
            size_of::<Option<Arc<CacheHistoryRetention>>>(),
            size_of::<bool>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
    pub(super) fn retains(&self, id: &CacheBlockId) -> bool {
        let mut node = Some(self);
        while let Some(current) = node {
            let value = current.inner.as_ref().expect("live history node");
            if value
                .history
                .upgrade()
                .is_some_and(|history| history.contains(id))
            {
                return true;
            }
            node = value.next.as_ref();
        }
        false
    }
}
/// One actual future checkpoint interval; no manager mutation at preparation.
pub(crate) struct PreparedCacheHistory {
    manager: CacheResidencyManager,
    owner: CacheHistoryOwner,
    node: CacheHistoryNode,
}
impl PreparedCacheHistory {
    pub(crate) fn prepare(
        manager: &CacheResidencyManager,
        layer: usize,
        start: i64,
        end: i64,
        context: &WorkspaceContext,
    ) -> Result<Self, CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, context);
        if start < 0 || start >= end {
            return Err(fail(CacheSourceError::Geometry));
        }
        let arc = std::alloc::Layout::new::<[usize; 2]>()
            .extend(std::alloc::Layout::new::<CacheHistoryRetention>())
            .ok()
            .map(|(layout, _)| layout.pad_to_align().size())
            .ok_or_else(|| fail(CacheSourceError::Overflow))?;
        let bytes = Self::control_bytes()
            .and_then(|n| n.checked_add(arc))
            .and_then(|n| n.checked_add(size_of::<Node>()))
            .ok_or_else(|| fail(CacheSourceError::Overflow))?;
        context
            .charge_metadata(bytes)
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let owner = CacheHistoryOwner {
            inner: Arc::new(CacheHistoryRetention {
                global_layer: layer,
                representation: CacheRepresentation::KeyValue,
                start,
                end,
            }),
            _funding: context.metadata_funding(),
        };
        let node = CacheHistoryNode {
            inner: Some(Box::new(Node {
                history: Arc::downgrade(&owner.inner),
                next: None,
            })),
            _funding: context.metadata_funding(),
        };
        Ok(Self {
            manager: manager.clone(),
            owner,
            node,
        })
    }
    pub(crate) fn install(
        mut self,
        generation: u64,
    ) -> Result<CacheHistoryOwner, CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::retained_source(cause, self.owner._funding.clone());
        let mut state = self.manager.inner.state.try_lock().map_err(|cause| {
            fail(match cause {
                std::sync::TryLockError::WouldBlock => CacheSourceError::Busy,
                std::sync::TryLockError::Poisoned(_) => CacheSourceError::Poisoned,
            })
        })?;
        if state.generation != generation || !self.manager.borrowed_storage_complete(&state) {
            return Err(fail(CacheSourceError::Identity));
        }
        // Unlink dead entries without allocation. Retire their metadata and
        // payer only after releasing the manager guard.
        let mut old = state.prepared_history.take();
        let mut retained = None;
        let mut retired = None;
        while let Some(mut entry) = old {
            let node = entry.inner.as_mut().expect("live history entry");
            old = node.next.take();
            let target = if node.history.strong_count() == 0 {
                &mut retired
            } else {
                &mut retained
            };
            node.next = target.take();
            *target = Some(entry);
        }
        self.node
            .inner
            .as_mut()
            .expect("prepared history entry")
            .next = retained;
        state.prepared_history = Some(self.node);
        drop(state);
        drop(retired);
        Ok(self.owner)
    }
    pub(crate) fn control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            CacheHistoryNode::inspection_control_bytes()?,
            size_of::<CacheHistoryOwner>(),
            size_of::<CacheHistoryNode>(),
            size_of::<Option<CacheHistoryNode>>().checked_mul(4)?,
            size_of::<Node>(),
            size_of::<Result<CacheHistoryOwner, CacheSourceFailure>>(),
            size_of::<CacheSourceFailure>(),
            size_of::<MutexGuard<'_, CacheManagerState>>(),
            size_of::<
                Result<
                    MutexGuard<'_, CacheManagerState>,
                    std::sync::TryLockError<MutexGuard<'_, CacheManagerState>>,
                >,
            >(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
}
