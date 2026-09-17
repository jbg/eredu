//! Borrowed diagnostic association, not a prepared collector or admission proof.
use super::*;
use crate::backend::runtime::cache::state::NativeStateSlotCounts;

pub(crate) struct PreparedNativeStateSlotBounds<'a, S: MlxStateMechanisms> {
    source: &'a S,
    counts: NativeStateSlotCounts,
}
impl<'a, S: MlxStateMechanisms> PreparedNativeStateSlotBounds<'a, S> {
    pub(crate) fn prepare(source: &'a S) -> Option<Self> {
        Some(Self {
            source,
            counts: source.retained_owner_slot_counts()?,
        })
    }
    pub(crate) fn source(&self) -> &'a S {
        self.source
    }
    pub(crate) fn counts(&self) -> NativeStateSlotCounts {
        self.counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::cache::LayerCachePolicy;
    use eredu_core::{AttentionPolicy, LayerSchedule};
    #[test]
    fn borrowed_state_fact_keeps_exact_instance_and_actual_shared_host_identity() {
        let layout = eredu_runtime::StateLayout::new(
            LayerSchedule::new(
                1,
                vec![LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 2).unwrap()],
            )
            .unwrap(),
        )
        .unwrap();
        let source = MlxKeyValueState::device(layout).unwrap();
        let other = source.clone();
        let bound = PreparedNativeStateSlotBounds::prepare(&source).unwrap();
        assert!(std::ptr::eq(bound.source(), &source));
        assert!(!std::ptr::eq(bound.source(), &other));
        assert_eq!(
            bound.source().layer_slot_metadata().identity(),
            source.layer_slot_metadata().identity()
        );
        assert_ne!(
            bound.source().layer_slot_metadata().identity(),
            other.layer_slot_metadata().identity()
        );
        assert_eq!(bound.counts().arrays, 2);
        assert_eq!(bound.counts().slot_tables, 1);
    }
}
