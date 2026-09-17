//! Exact implemented native paged scope, before request or source mutation.
use super::*;
use crate::backend::runtime::cache::residency::CacheBlockMetadata;
use eredu_core::{AttentionPolicy, LayerSchedule, cache::LayerCachePolicy};
use eredu_runtime::{CacheResidencyPolicy, PagedCacheOptions};

impl PagedKeyValueCache {
    /// Capability predicate only. Catalog/source/role checks still authenticate
    /// every actual manager, page, invocation and native completion. The byte
    /// class only screens unsupported storage widths; it never selects F16
    /// versus BF16 for an empty cache or overrides any actual native dtype.
    /// A declared fixed companion does not change the paged KV mechanism:
    /// complete hybrid projection separately authenticates every actual fixed
    /// role, dtype, extent and backing through the shared fixed-state worker.
    /// Stable Host and own-writer Disk sources use the closed saved-copy and
    /// retained-resume workers. Actual source phases, finite I/O destinations,
    /// source-backed roles and copy/completion custody remain mandatory.
    pub(crate) fn original_residency_supported(
        policy: &CacheResidencyPolicy,
        layout: &LayerSchedule<LayerCachePolicy>,
        floating_bytes: std::num::NonZeroU8,
    ) -> bool {
        Self::original_residency_for_operation(policy, layout, floating_bytes, true)
    }
    /// Actual generation has the complete Host append/scan source itinerary.
    /// Saved state uses its separate closed copy/resume source and destination.
    pub(crate) fn original_generation_residency_supported(
        policy: &CacheResidencyPolicy,
        layout: &LayerSchedule<LayerCachePolicy>,
        floating_bytes: std::num::NonZeroU8,
    ) -> bool {
        Self::original_residency_for_operation(policy, layout, floating_bytes, true)
    }
    /// Reset constructs an independent empty namespace from the same stable
    /// source configuration. It performs no page copy, promotion or execution.
    /// Subsequent generation retains its own source and role admission.
    pub(crate) fn original_reset_residency_supported(
        policy: &CacheResidencyPolicy,
        layout: &LayerSchedule<LayerCachePolicy>,
        floating_bytes: std::num::NonZeroU8,
    ) -> bool {
        Self::original_residency_for_operation(policy, layout, floating_bytes, true)
    }
    fn original_residency_for_operation(
        policy: &CacheResidencyPolicy,
        layout: &LayerSchedule<LayerCachePolicy>,
        floating_bytes: std::num::NonZeroU8,
        host: bool,
    ) -> bool {
        match policy {
            CacheResidencyPolicy::Device => true,
            CacheResidencyPolicy::Paged(options) => {
                matches!(floating_bytes.get(), 2 | 4)
                    && original_options_supported(options, host)
                    && layout
                        .iter()
                        .any(|layer| matches!(layer, LayerCachePolicy::KeyValue { .. }
                            | LayerCachePolicy::KeyValueWithFixedState { .. }))
                    && layout.iter().all(|layer| {
                        matches!(
                            layer,
                            LayerCachePolicy::NoState
                                | LayerCachePolicy::KeyValue {
                                    attention: AttentionPolicy::Full | AttentionPolicy::Sliding { .. },
                                    ..
                                }
                                | LayerCachePolicy::KeyValueWithFixedState {
                                    attention: AttentionPolicy::Full | AttentionPolicy::Sliding { .. },
                                    ..
                                }
                        )
                    })
            }
        }
    }
}
fn original_options_supported(options: &PagedCacheOptions, host: bool) -> bool {
    options.full_attention_enabled()
        && (host || options.host_budget_bytes() == 0)
        && !options.process_sampling_enabled()
}
impl ProjectedPagedSource {
    /// Validate actual retained page/tail representation before constructing any
    /// native program; policy geometry alone grants no source authority.
    pub(crate) fn original_program_validation_control_bytes() -> usize {
        size_of::<(
            &Self,
            &PagedCacheSourceGeometry,
            Result<(), CacheSourceError>,
            std::slice::Iter<'static, PagedCacheBlockGeometry>,
            std::slice::Iter<'static, PagedCacheArrayGeometry>,
            Dtype,
            Option<u64>,
        )>()
    }
    pub(crate) fn validate_original_program(&self) -> Result<(), CacheSourceError> {
        let geometry = self.geometry();
        if !original_options_supported(self.manager().options(), true)
            || geometry.key_only
            || geometry
                .tail
                .iter()
                .flatten()
                .any(|array| CacheBlockMetadata::floating_dtype_bytes(array.dtype).is_none())
            || geometry.blocks.iter().any(|block| {
                block
                    .arrays
                    .iter()
                    .any(|array| CacheBlockMetadata::floating_dtype_bytes(array.dtype).is_none())
            })
        {
            return Err(CacheSourceError::Geometry);
        }
        Ok(())
    }
}
