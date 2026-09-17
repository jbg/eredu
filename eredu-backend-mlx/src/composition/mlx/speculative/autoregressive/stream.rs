//! Ordinary or originally prepared immutable stream custody.
use safemlx::{PreparedStreamCopy, Stream};
use eredu_core::HostPreparationAuthority;

/// The original variant shares its already paid C wrapper. Its closed native
/// owner retires the wrapper, Arc and retirement shell before host custody.
#[derive(Clone)]
pub(in crate::composition::mlx::speculative) enum StateStream {
    Ordinary(Stream),
    Original(PreparedStreamCopy<HostPreparationAuthority>),
}
impl From<Stream> for StateStream {
    fn from(value: Stream) -> Self {
        Self::Ordinary(value)
    }
}
impl std::ops::Deref for StateStream {
    type Target = Stream;
    fn deref(&self) -> &Stream {
        match self {
            Self::Ordinary(stream) => stream,
            Self::Original(stream) => stream.as_stream(),
        }
    }
}
