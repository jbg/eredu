//! Read-only agreement between an actual saved pager and its completed copy.
use super::*;
use crate::backend::runtime::cache::residency::{CacheSourceError, CacheSourceFailure};
use std::mem::size_of_val;

impl ProjectedNativeStorage {
    /// Rebind only the already quoted metadata itinerary after the complete
    /// independent-copy geometry, grouping and native backing checks succeeded.
    /// Actual manager/pins remain those of this destination; no old role moves.
    pub(crate) fn inherit_copied_paged_host_traces(
        &mut self,
        source: &Self,
        context: &WorkspaceContext,
    ) -> Result<(), CacheSourceFailure> {
        self.validate_independent_paged_copy(source, context)?;
        context
            .charge_metadata(size_of::<(
                &mut Self,
                &Self,
                &WorkspaceContext,
                Result<(), CacheSourceFailure>,
                std::iter::Zip<
                    std::slice::IterMut<
                        '_,
                        crate::backend::runtime::cache::kv::ProjectedPagedSource,
                    >,
                    std::slice::Iter<'_, crate::backend::runtime::cache::kv::ProjectedPagedSource>,
                >,
            )>())
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        // Validate every destination before changing any descriptor. The source
        // was produced in this same quote; each destination is freshly projected.
        for (copied, saved) in self.paged_sources.iter().zip(&source.paged_sources) {
            match (copied.host_trace(), saved.host_trace()) {
                (Some(empty), Some(_)) => empty
                    .validate_unstarted(context)
                    .map_err(|cause| CacheSourceFailure::metadata(cause, context))?,
                (None, None) => {}
                _ => {
                    return Err(CacheSourceFailure::source(
                        CacheSourceError::Identity,
                        context,
                    ));
                }
            }
        }
        for (copied, saved) in self.paged_sources.iter_mut().zip(&source.paged_sources) {
            copied.inherit_host_trace(saved);
        }
        Ok(())
    }
    /// Geometry and independence check after the closed copy worker has
    /// published its destination. This is not a copy receipt or execution grant;
    /// current source pins and the accepted role still authorize native work.
    pub(crate) fn validate_independent_paged_copy(
        &self,
        source: &Self,
        context: &WorkspaceContext,
    ) -> Result<(), CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, context);
        let pairs = self.paged_sources.iter().zip(&source.paged_sources);
        let controls = [
            size_of::<(&Self, &Self, &WorkspaceContext)>(),
            size_of_val(&pairs),
            size_of::<(
                usize,
                &crate::backend::runtime::cache::kv::ProjectedPagedSource,
                &crate::backend::runtime::cache::kv::ProjectedPagedSource,
            )>(),
            size_of::<Result<(), CacheSourceFailure>>(),
            Self::iteration_control_bytes()
                .and_then(|n| n.checked_mul(3))
                .ok_or_else(|| fail(CacheSourceError::Overflow))?,
            size_of::<Option<&std::sync::Arc<safemlx::ImmutableHostTransferBuffer>>>(),
            size_of::<(safemlx::AllocationIdentity, u64)>(),
        ];
        context
            .charge_metadata(
                controls
                    .into_iter()
                    .try_fold(size_of_val(&controls), usize::checked_add)
                    .ok_or_else(|| fail(CacheSourceError::Overflow))?,
            )
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        if !self.is_complete()
            || !source.is_complete()
            || source.paged_sources.is_empty()
            || self.paged_sources.len() != source.paged_sources.len()
        {
            return Err(fail(CacheSourceError::Identity));
        }
        for (index, (copied, saved)) in pairs.enumerate() {
            let actual = copied.geometry();
            let original = saved.geometry();
            if copied.manager().same_catalog(saved.manager())
                || copied.append_geometry() != saved.append_geometry()
                || actual.global_layer != original.global_layer
                || actual.rank != original.rank
                || actual.tail != original.tail
                || actual.blocks.len() != original.blocks.len()
            {
                return Err(fail(CacheSourceError::Identity));
            }
            // Dense layer order does not replace physical module identity or
            // sharing. The independent copy must keep the same manager groups.
            for previous in 0..index {
                if copied
                    .manager()
                    .same_catalog(self.paged_sources[previous].manager())
                    != saved
                        .manager()
                        .same_catalog(source.paged_sources[previous].manager())
                {
                    return Err(fail(CacheSourceError::Identity));
                }
            }
            for (block, old) in actual.blocks.iter().zip(&original.blocks) {
                if block.id.global_layer != old.id.global_layer
                    || block.id.representation != old.id.representation
                    || block.id.rank != old.id.rank
                    || block.id.start != old.id.start
                    || block.id.end != old.id.end
                    || block.phase != old.phase
                    || block.arrays != old.arrays
                {
                    return Err(fail(CacheSourceError::Identity));
                }
            }
        }
        // Immutable Host pages preserve the actual saved owner and its B
        // provenance; each distinct manager still owns its own canonical rows.
        // Reject unrelated equal-sized Host buffers and omitted source owners.
        for (identity, bytes, _) in source.iter() {
            if let Some(original) = source.native_host(identity) {
                let copied = self
                    .native_host(identity)
                    .ok_or_else(|| fail(CacheSourceError::Identity))?;
                if !std::sync::Arc::ptr_eq(copied, original)
                    || self
                        .iter()
                        .find(|(id, _, _)| *id == identity)
                        .map(|(_, n, _)| n)
                        != Some(bytes)
                {
                    return Err(fail(CacheSourceError::Identity));
                }
            }
        }
        if self.iter().any(|(identity, _, _)| {
            self.native_host(identity).is_some() && source.native_host(identity).is_none()
        }) {
            return Err(fail(CacheSourceError::Identity));
        }
        // Actual Device/tail copies must have independent allocation identities;
        // logical byte equality cannot substitute for that native observation.
        if self
            .iter()
            .any(|(identity, bytes, _)| bytes != 0 && source.native_array(identity).is_some())
        {
            return Err(fail(CacheSourceError::Identity));
        }
        Ok(())
    }
}
