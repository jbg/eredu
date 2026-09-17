//! The same logical block window for cache scans and retained-source inspection.
use eredu_core::cache::{CacheBlockId, CacheRepresentation};

/// Allocation-free selection of the existing catalog's ordered block entries.
/// Session/rank authorization remains with the actual catalog owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheBlockSelection {
    layer: usize,
    representation: CacheRepresentation,
    start: i64,
    end: i64,
    prefix: i64,
}
impl CacheBlockSelection {
    /// Preserves the ordinary scan predicate, including retained prefix blocks.
    /// This descriptive range neither creates a cache block nor grants a lease.
    pub const fn new(
        layer: usize,
        representation: CacheRepresentation,
        start: i64,
        end: i64,
        prefix: i64,
    ) -> Self {
        Self {
            layer,
            representation,
            start,
            end,
            prefix,
        }
    }
    /// Whether an actual catalog identity overlaps this selected range/prefix.
    pub fn includes(self, id: &CacheBlockId) -> bool {
        id.global_layer == self.layer
            && id.representation == self.representation
            && interval_overlaps(id.start, id.end, self.start, self.end, self.prefix)
    }
    /// Whether an actual block is wholly outside a retained prefix and the
    /// current sliding window. A prefix ending inside a block retains that
    /// whole immutable block. Native lease/import/history pins are additional
    /// independent retention conditions owned by the actual cache manager.
    pub const fn outside_retained_window(
        block_start: i64, block_end: i64, visible_start: i64, prefix: i64,
    ) -> bool {
        block_end <= visible_start && block_start >= prefix
    }
    /// Architecture-global layer selected from the same catalog.
    pub const fn layer(self) -> usize {
        self.layer
    }
}

/// Shared geometric predicate only; catalog/session identity is checked by the
/// source owner before an interval reaches a scan.
pub(super) const fn interval_overlaps(
    block_start: i64,
    block_end: i64,
    start: i64,
    end: i64,
    prefix: i64,
) -> bool {
    block_start < end && (block_end > start || block_start < prefix)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partial_prefix_retains_its_whole_block_at_discard_boundary() {
        assert!(!CacheBlockSelection::outside_retained_window(0,3,5,1));
        assert!(CacheBlockSelection::outside_retained_window(3,6,6,1));
        assert!(!CacheBlockSelection::outside_retained_window(3,6,5,1));
        assert!(!CacheBlockSelection::outside_retained_window(0,3,5,3));
        assert!(CacheBlockSelection::outside_retained_window(0,3,5,0));
    }
    #[test]
    fn block_selection_preserves_prefix_overlap_endpoints_and_representation() {
        let id = CacheBlockId {
            session_id: 9,
            global_layer: 3,
            representation: CacheRepresentation::KeyValue,
            start: 0,
            end: 2,
            rank: None,
        };
        let selection = CacheBlockSelection::new(3, CacheRepresentation::KeyValue, 6, 10, 2);
        assert!(selection.includes(&id));
        assert!(!selection.includes(&CacheBlockId {
            start: 2,
            end: 6,
            ..id.clone()
        }));
        assert!(selection.includes(&CacheBlockId {
            start: 5,
            end: 7,
            ..id.clone()
        }));
        assert!(!selection.includes(&CacheBlockId {
            start: 10,
            end: 12,
            ..id.clone()
        }));
        assert!(!selection.includes(&CacheBlockId {
            global_layer: 4,
            ..id.clone()
        }));
        assert!(!selection.includes(&CacheBlockId {
            representation: CacheRepresentation::CompressedLatentRotary,
            ..id
        }));
    }
}
