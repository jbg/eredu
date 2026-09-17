//! Exact accepted model role uses the ordinary append partition and kernels.
use super::*;
use crate::backend::nn::workspace::{
    OriginalPagedAppendClaim, PagedAppendInput, ProjectedPagedSources,
};
use crate::backend::runtime::cache::residency::{CacheBlockMetadata, CacheSourceError};
use eredu_runtime::{cache::PagedAppendPlan, working_memory::WorkspacePagedGeometry};
use safemlx::OriginalScopeObserver;
use std::mem::size_of;

impl PagedKeyValueCache {
    pub(super) fn append_original(
        &mut self,
        keys: Array,
        values: Array,
        retain_for_attention: bool,
        stream: &Stream,
    ) -> Result<(), Exception> {
        self.with_original_append(keys, values, stream, |cache, claim, keys, values| {
            cache.append_claimed(claim, keys, values, retain_for_attention, stream)
        })
    }

    pub(super) fn with_original_append<R>(
        &mut self,
        keys: Array,
        values: Array,
        stream: &Stream,
        run: impl FnOnce(
            &mut Self,
            &mut OriginalPagedAppendClaim<'_>,
            Array,
            Array,
        ) -> Result<R, Exception>,
    ) -> Result<R, Exception> {
        let observer = OriginalScopeObserver::require_current()?;
        // These fixed descriptors only select the retained source program.
        // They cannot create a manager/source identity or native grant.
        if keys.ndim() != 4
            || values.ndim() != 4
            || keys.dim(-2) <= 0
            || CacheBlockMetadata::floating_dtype_bytes(keys.dtype()).is_none()
            || CacheBlockMetadata::floating_dtype_bytes(values.dtype()).is_none()
            || keys.shape()[..3] != values.shape()[..3]
            || values.dim(3) != if self.key_only { 1 } else { keys.dim(3) }
        {
            return Err(OriginalPagedAppendClaim::input_error(
                &observer,
                CacheSourceError::Geometry,
            ));
        }
        let plan = PagedAppendPlan::new(
            self.manager.options().block_size_tokens(),
            keys.dim(-2),
            self.tail_len(),
            self.tail_start,
            self.offset,
        )
        .map_err(|cause| {
            OriginalPagedAppendClaim::input_error(&observer, CacheSourceError::Append(cause))
        })?;
        let geometry = WorkspacePagedGeometry {
            block_size: self.manager.options().block_size_tokens(),
            dimensions: [keys.dim(0), keys.dim(1), keys.dim(3)],
            offset: self.offset,
            tail_start: self.tail_start,
            window: self.sliding_window,
            prefix_tokens: self.prefix_tokens,
            key_only: self.key_only,
            retain_discarded: self.manager.options().retains_discarded_for_persistence(),
        };
        let manager = self.manager.clone();
        ProjectedPagedSources::with_current_append(
            PagedAppendInput {
                manager: &manager,
                global_layer: self.global_layer,
                rank: self.rank,
                geometry,
                plan,
                dtypes: [keys.dtype(), values.dtype()],
                tail_dtypes: self
                    .tail_keys
                    .as_ref()
                    .zip(self.tail_values.as_ref())
                    .map(|(keys, values)| [keys.dtype(), values.dtype()]),
            },
            |claim| {
                if self.tail_keys.is_some() != self.tail_values.is_some() {
                    return Err(claim.error(CacheSourceError::Geometry));
                }
                if let (Some(previous_keys), Some(previous_values)) =
                    (&self.tail_keys, &self.tail_values)
                {
                    let length = plan.initial_frontier().1;
                    let expected_keys = [keys.dim(0), keys.dim(1), length, keys.dim(3)];
                    let expected_values = [values.dim(0), values.dim(1), length, values.dim(3)];
                    if previous_keys.shape() != expected_keys
                        || previous_values.shape() != expected_values
                        || CacheBlockMetadata::floating_dtype_bytes(previous_keys.dtype()).is_none()
                        || CacheBlockMetadata::floating_dtype_bytes(previous_values.dtype())
                            .is_none()
                    {
                        return Err(claim.error(CacheSourceError::Geometry));
                    }
                }
                manager.begin_original_append(claim, self.tail_bytes(), stream)?;
                run(self, claim, keys, values)
            },
        )
    }

    pub(super) fn append_claimed(
        &mut self,
        claim: &mut OriginalPagedAppendClaim<'_>,
        keys: Array,
        values: Array,
        retain_for_attention: bool,
        stream: &Stream,
    ) -> Result<(), Exception> {
        if self.sliding_window.is_some() && !retain_for_attention && !claim.visible_completed() {
            return Err(claim.error(CacheSourceError::AppendWindow));
        }
        let plan = claim.plan();
        let manager = self.manager.clone();
        let previous_keys = self
            .tail_keys
            .as_ref()
            .map(Array::try_clone_handle)
            .transpose()?;
        let previous_values = self
            .tail_values
            .as_ref()
            .map(Array::try_clone_handle)
            .transpose()?;
        let previous_start = self.tail_start;
        let previous_offset = self.offset;
        let result = plan.run(
            &mut append::NativeAppend {
                cache: &mut *self,
                keys,
                values,
                stream,
                original: Some(&mut *claim),
            },
            retain_for_attention,
        );
        if let Err(cause) = result {
            self.tail_keys = previous_keys;
            self.tail_values = previous_values;
            self.tail_start = previous_start;
            self.offset = previous_offset;
            return Err(
                match manager.rollback_original_append(claim, self.tail_bytes(), previous_offset) {
                    Ok(()) => cause,
                    Err(rollback) => {
                        CacheResidencyManager::original_append_rollback_error(cause, rollback)
                    }
                },
            );
        }
        Ok(())
    }

    pub(super) fn seal_tail_original(
        &mut self,
        claim: &mut OriginalPagedAppendClaim<'_>,
        stream: &Stream,
    ) -> Result<(), Exception> {
        let id = claim
            .publication_id(claim.published_count())
            .ok_or_else(|| claim.error(CacheSourceError::Identity))?
            .clone();
        if id.start != self.tail_start
            || id.rank != self.rank
            || id.global_layer != self.global_layer
        {
            return Err(claim.error(CacheSourceError::Identity));
        }
        let keys = self
            .tail_keys
            .take()
            .ok_or_else(|| claim.error(CacheSourceError::Geometry))?;
        let Some(values) = self.tail_values.take() else {
            self.tail_keys = Some(keys);
            return Err(claim.error(CacheSourceError::Geometry));
        };
        let end = id.end;
        if let Err(cause) = self
            .manager
            .publish_original_tail(claim, 0, end, true, false)
        {
            self.tail_keys = Some(keys);
            self.tail_values = Some(values);
            return Err(cause);
        }
        let result = (|| {
            let arrays = CacheBlockArrays::KeyValue {
                keys: keys.try_clone_handle()?,
                values: values.try_clone_handle()?,
            };
            self.manager.seal_original_block(claim, id, arrays, stream)
        })();
        if let Err(cause) = result {
            self.tail_keys = Some(keys);
            self.tail_values = Some(values);
            return Err(
                match self
                    .manager
                    .publish_original_tail(claim, self.tail_bytes(), end, false, true)
                {
                    Ok(()) => cause,
                    Err(rollback) => {
                        CacheResidencyManager::original_append_rollback_error(cause, rollback)
                    }
                },
            );
        }
        self.tail_start = end;
        Ok(())
    }
    pub(crate) fn original_append_control_bytes() -> Option<usize> {
        let frames = [
            size_of::<(&mut Self, Array, Array, bool, &Stream)>(),
            size_of::<(&mut Self, Array, Array, &Stream)>(),
            size_of::<(
                &mut Self,
                &mut OriginalPagedAppendClaim<'_>,
                Array,
                Array,
                bool,
                &Stream,
            )>(),
            size_of::<(&mut Self, &mut OriginalPagedAppendClaim<'_>, &Stream)>(),
            size_of::<append::NativeAppend<'_, '_>>(),
            size_of::<OriginalScopeObserver>(),
            size_of::<CacheResidencyManager>(),
            size_of::<WorkspacePagedGeometry>(),
            size_of::<PagedAppendInput<'_>>(),
            size_of::<PagedAppendPlan>(),
            size_of::<eredu_runtime::cache::PagedAppendSteps>(),
            size_of::<eredu_runtime::cache::PagedAppendStep>(),
            size_of::<(Option<Array>, Option<Array>, i64, i64)>(),
            size_of::<[i32; 4]>(),
            size_of::<[i32; 4]>(),
            size_of::<CacheBlockId>(),
            size_of::<CacheBlockArrays>(),
            size_of::<Result<(), Exception>>(),
            size_of::<Result<(), Exception>>(),
            size_of::<Result<PagedAppendPlan, eredu_runtime::cache::PagedAppendError>>(),
            size_of::<(&CacheResidencyManager, &mut Self, &Array, &Array, &Stream)>(),
            safemlx::Stream::device_type_control_bytes()?,
            PagedAppendPlan::control_bytes::<append::NativeAppend<'_, '_>>()?,
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
}
