//! One prospective shared error destination for a consumed ordinary cursor.
use super::*;
use std::{fmt, sync::Arc};

struct FailureOwner {
    error: Option<PreparedChatSessionError>,
    // The source error and shared allocation retire before their original payer.
    funding: HostMetadataFunding,
}
/// Shared terminal cause and committed prefix, retaining the original cursor.
/// A clone aliases its one paid owner; it does not copy token history.
pub struct ControlledSessionFailure(Option<Arc<FailureOwner>>);
impl ControlledSessionFailure {
    fn owner(&self) -> &Arc<FailureOwner> {
        self.0.as_ref().expect("live failure owner")
    }
    pub(super) fn prepare(funding: &HostMetadataFunding) -> Result<Self, RecordConstructionError> {
        let result = (|| {
            records::construction::controls(
                funding,
                &[
                    size_of::<Self>(),
                    size_of::<FailureOwner>(),
                    size_of::<Option<FailureOwner>>(),
                    size_of::<PreparedChatSessionError>(),
                    size_of::<Result<Self, RecordConstructionError>>(),
                ],
            )?;
            records::construction::shell::<FailureOwner>(funding)?;
            Ok(Self(Some(Arc::new(FailureOwner {
                error: None,
                funding: funding.clone(),
            }))))
        })();
        result.map_err(|cause| RecordConstructionError::retain(cause, funding))
    }
    pub(super) fn publish(&mut self, error: PreparedChatSessionError) -> Self {
        let owner = Arc::get_mut(self.0.as_mut().expect("live failure slot"))
            .expect("only published failures can be aliased");
        assert!(owner.error.is_none(), "one terminal transition");
        owner.error = Some(error);
        Self(Some(self.owner().clone()))
    }
    /// Borrows the original typed cause without separating it from storage.
    pub fn error(&self) -> &PreparedChatSessionError {
        self.owner().error.as_ref().expect("published failure")
    }
    pub(super) fn prefix(&self) -> &[u32] {
        self.owner()
            .error
            .as_ref()
            .and_then(PreparedChatSessionError::committed_token_ids)
            .unwrap_or(&[])
    }
}
impl Clone for ControlledSessionFailure {
    fn clone(&self) -> Self {
        Self(Some(self.owner().clone()))
    }
}
impl Drop for ControlledSessionFailure {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            if let Some(owner) = Arc::into_inner(owner) {
                drop(owner);
            }
        }
    }
}
impl fmt::Debug for ControlledSessionFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.owner().error.fmt(f)
    }
}
impl fmt::Display for ControlledSessionFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error().fmt(f)
    }
}
impl std::error::Error for ControlledSessionFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.error())
    }
}
