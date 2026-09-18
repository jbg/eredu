//! Borrowed native/source entries for a caller-owned, prepriced collector.

use super::{ResidencyError, RetainedHostBuffer};
use eredu_checkpoint::store::{CheckpointSource, SourceStorageRef, StoreError};
use safemlx::{Array, ImmutableHostTransferBuffer};
use std::sync::Arc;

/// Actual source fields borrowed during a manager's state loan. These facts do
/// not own payloads, deduplicate aliases, inspect native metadata or grant work.
#[derive(Clone, Copy)]
pub(crate) enum RetainedStorageRef<'a> {
    Array(&'a Array),
    CanonicalArray(&'a super::super::manager::CanonicalArrayOwner),
    Host(&'a Arc<ImmutableHostTransferBuffer>),
    RetainedHost(&'a RetainedHostBuffer),
    Bytes(&'a Arc<[u8]>),
    Source(SourceStorageRef<'a>),
}

/// Cold manager inspection never waits for a held state mutex.
#[derive(Debug, thiserror::Error)]
pub(crate) enum RetainedStorageInspectionError {
    #[error("native manager state is busy during borrowed storage inspection")]
    Busy,
    #[error(transparent)]
    Residency(#[from] ResidencyError),
}

// Error conversion and any secondary cause destruction occur only after the
// manager state loan ends. Neither variant allocates an extra error wrapper.
pub(crate) enum RetainedStorageVisitFailure<E> {
    Inspection(RetainedStorageInspectionError),
    Callback {
        error: E,
        later_source_error: Option<StoreError>,
    },
}

impl<E> RetainedStorageVisitFailure<E> {
    pub(crate) fn callback(error: E) -> Self {
        Self::Callback {
            error,
            later_source_error: None,
        }
    }

    pub(crate) fn inspection(error: impl Into<ResidencyError>) -> Self {
        Self::Inspection(RetainedStorageInspectionError::Residency(error.into()))
    }

    /// Call only after releasing the manager's state mutex.
    pub(crate) fn into_error(self) -> E
    where
        E: From<RetainedStorageInspectionError>,
    {
        match self {
            Self::Inspection(error) => error.into(),
            Self::Callback {
                error,
                later_source_error,
            } => {
                drop(later_source_error);
                error
            }
        }
    }
}

pub(crate) fn visit_checkpoint_storage<E>(
    source: &dyn CheckpointSource,
    visitor: &mut dyn FnMut(RetainedStorageRef<'_>) -> Result<(), E>,
    failure: &mut Option<E>,
) -> Result<bool, RetainedStorageVisitFailure<E>> {
    // The slot is owned outside the manager state-loan closure, so a source
    // panic after callback failure cannot destroy the caller error under lock.
    debug_assert!(failure.is_none());
    let result = source.visit_source_storage(&mut |source| {
        if failure.is_none() {
            *failure = visitor(RetainedStorageRef::Source(source)).err();
        }
    });
    if let Some(error) = failure.take() {
        return Err(RetainedStorageVisitFailure::Callback {
            error,
            later_source_error: result.err(),
        });
    }
    result.map_err(RetainedStorageVisitFailure::inspection)
}
