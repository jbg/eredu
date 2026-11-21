//! MLX storage and immutable-identity mechanisms for the portable selected policy.
use super::*;
use eredu_core::checkpoint_mechanisms::unit_binding::ImmutableUnitStorage;

/// Native certificate for the immutable parameter storage of one execution context.
#[doc(hidden)]
pub struct MlxImmutableParameters {
    stream_index: i32,
    identity: eredu_core::resources::ResourceIdentity,
}

impl MlxImmutableParameters {
    pub(crate) fn new(stream: &Stream) -> safemlx::error::Result<Self> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Ok(Self {
            stream_index: stream.get_index()?,
            identity: eredu_core::resources::ResourceIdentity {
                scope: "mlx.immutable.parameters".into(),
                key: NEXT.fetch_add(1, Ordering::Relaxed).to_string(),
            },
        })
    }
}

impl ImmutableUnitStorage<Stream> for MlxImmutableParameters {
    fn identity(&self, stream: &Stream) -> Option<&eredu_core::resources::ResourceIdentity> {
        (stream.get_index().ok() == Some(self.stream_index)).then_some(&self.identity)
    }
}
