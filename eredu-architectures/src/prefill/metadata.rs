//! Host-only destinations for retained input geometry, independent of native scope.
use eredu_nn::{
    Error,
    workspace::{WorkspaceContext, WorkspaceMetadataError, HostMetadataFunding},
};
#[derive(Clone, Copy)]
pub(crate) enum Metadata<'a> {
    Context(crate::decoder::identity::Metadata<'a>),
    Funding(&'a HostMetadataFunding),
}
impl<'a> Metadata<'a> {
    pub(crate) fn new(context: Option<&'a WorkspaceContext>) -> Self {
        Self::Context(crate::decoder::identity::Metadata::new(context))
    }
    pub(crate) fn funded(funding: Option<&'a HostMetadataFunding>) -> Self {
        funding.map_or_else(|| Self::new(None), Self::Funding)
    }
    pub(crate) fn checked(self) -> bool {
        match self {
            Self::Context(m) => m.context().is_some(),
            Self::Funding(_) => true,
        }
    }
    pub(crate) fn context(self) -> Option<&'a WorkspaceContext> {
        match self {
            Self::Context(m) => m.context(),
            Self::Funding(_) => None,
        }
    }
    pub(crate) fn controls<T>(self) -> Result<(), Error> {
        if !self.checked() {
            return Ok(());
        }
        let parts = [
            std::mem::size_of::<T>(),
            std::mem::size_of::<Result<T, Error>>(),
            std::mem::size_of::<Self>(),
        ];
        let bytes = parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
            .ok_or(WorkspaceMetadataError::Overflow)?;
        match self {
            Self::Context(m) => m
                .context()
                .expect("checked context")
                .charge_metadata(bytes)
                .map_err(Error::from),
            Self::Funding(f) => f
                .reserve_metadata(bytes)
                .map_err(|cause| Error::from(WorkspaceMetadataError::from(cause))),
        }
    }
    pub(crate) fn vector<T>(self, n: usize) -> Result<Vec<T>, Error> {
        match self {
            Self::Context(m) => m.vector(n),
            Self::Funding(f) => f.metadata_vec(n),
        }
    }
    pub(crate) fn error(self, message: std::fmt::Arguments<'_>) -> Error {
        match self {
            Self::Context(m) => m.error(message),
            Self::Funding(f) => f.metadata_error(message),
        }
    }
    pub(crate) fn source<E: std::error::Error + Send + Sync + 'static>(self, cause: E) -> Error {
        match self {
            Self::Context(m) => m.source(cause),
            Self::Funding(f) => f.metadata_source(cause),
        }
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
