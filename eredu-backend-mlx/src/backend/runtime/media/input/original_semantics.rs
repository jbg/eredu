//! Private completed-upload carrier. All aliases use consuming final retirement.
use crate::{backend::managed_memory::NativeMemoryOwner, MlxTensor};
use eredu_architectures::{
    media_plan::BoundPreparedMediaSemantics, processor_execution::OriginalHostLowering,
};
use std::sync::Arc;
struct Payload {
    lowered: OriginalHostLowering<MlxTensor>,
    semantics: BoundPreparedMediaSemantics,
    native_input: Option<crate::composition::mlx::MlxOriginalPreparedNativeInput>,
    // Ordinary native metadata/control custody outlives the packet shell/payload.
    memory: NativeMemoryOwner,
}
pub(crate) enum OriginalMediaPacket {
    Ordinary(OrdinaryMediaPacket),
    Original(crate::composition::mlx::CompletedOriginalModelInput),
}
pub(crate) struct OrdinaryMediaPacket(Option<Arc<Payload>>);
impl OriginalMediaPacket {
    pub(crate) fn original(packet: crate::composition::mlx::CompletedOriginalModelInput) -> Self {
        Self::Original(packet)
    }
    /// Called only by the ordinary upload worker after all real roots complete.
    pub(crate) fn completed(
        lowered: OriginalHostLowering<MlxTensor>,
        semantics: BoundPreparedMediaSemantics,
        memory: NativeMemoryOwner,
        native_input: Option<crate::composition::mlx::MlxOriginalPreparedNativeInput>,
    ) -> Self {
        assert!(lowered.source().same_source(semantics.source()));
        assert!(native_input
            .as_ref()
            .is_none_or(|n| n.source().same_source(lowered.source())));
        Self::Ordinary(OrdinaryMediaPacket(Some(Arc::new(Payload {
            lowered,
            semantics,
            native_input,
            memory,
        }))))
    }
    pub(crate) fn shape(&self) -> [u64; 2] {
        match self {
            Self::Ordinary(v) => [
                1,
                v.0.as_deref()
                    .expect("live ordinary packet")
                    .semantics
                    .decoder_positions() as u64,
            ],
            Self::Original(v) => v.shape(),
        }
    }
    pub(crate) fn clone_lowered(&self) -> OriginalHostLowering<MlxTensor> {
        match self {
            Self::Ordinary(v) => {
                v.0.as_deref()
                    .expect("live ordinary packet")
                    .lowered
                    .clone()
            }
            Self::Original(v) => v.clone_lowered(),
        }
    }
    pub(crate) fn semantics(&self) -> BoundPreparedMediaSemantics {
        match self {
            Self::Ordinary(v) => {
                v.0.as_deref()
                    .expect("live ordinary packet")
                    .semantics
                    .clone()
            }
            Self::Original(v) => v.semantics(),
        }
    }
}
impl Clone for OriginalMediaPacket {
    fn clone(&self) -> Self {
        match self {
            Self::Ordinary(v) => Self::Ordinary(OrdinaryMediaPacket(Some(Arc::clone(
                v.0.as_ref().expect("live ordinary packet"),
            )))),
            Self::Original(v) => Self::Original(v.clone()),
        }
    }
}
impl Drop for OrdinaryMediaPacket {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
impl std::fmt::Debug for OriginalMediaPacket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OriginalMediaPacket")
            .field("decoder_shape", &self.shape())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
thread_local! { static PREPARATIONS:std::cell::Cell<usize>=const {std::cell::Cell::new(0)}; }
#[cfg(test)]
pub(crate) fn record_original_semantic_preparation() {
    PREPARATIONS.set(PREPARATIONS.get() + 1);
}
#[cfg(test)]
pub(crate) fn reset_original_semantic_preparations() {
    PREPARATIONS.set(0);
}
#[cfg(test)]
pub(crate) fn original_semantic_preparations() -> usize {
    PREPARATIONS.get()
}
