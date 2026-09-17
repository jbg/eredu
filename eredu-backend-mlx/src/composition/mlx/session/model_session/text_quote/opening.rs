//! Single-assignment opening identity; numerical provenance stays with its owner.

use eredu_runtime::working_memory::{
    InferenceRequest, InferenceRetention, InferenceStateRevision, WorkingMemoryError,
};
use std::cell::OnceCell;

#[derive(Debug)]
struct Opening {
    revision: InferenceStateRevision,
    predecessor: Option<InferenceRequest>,
}

/// Private stable owner for a quote that may precede installation. There is no
/// production constructor of an unsealed TextExecutionQuote in this slice.
/// This cell cannot prove copy provenance, publish state or issue work itself.
#[derive(Debug)]
pub(super) struct OpeningSeal {
    opening: OnceCell<Opening>,
    before_install: Option<InferenceStateRevision>,
}

impl OpeningSeal {
    /// Ordinary preparation already inspected this exact live opening.
    pub(super) fn ordinary(
        revision: InferenceStateRevision,
        predecessor: Option<InferenceRequest>,
    ) -> Self {
        Self {
            opening: OnceCell::from(Opening {
                revision,
                predecessor,
            }),
            before_install: None,
        }
    }

    /// Used only by the later closed resume draft constructor. Capturing the
    /// current revision does not establish that any replacement was installed.
    pub(super) fn pending(retained: &InferenceRetention) -> Self {
        Self {
            opening: OnceCell::new(),
            before_install: Some(retained.revision().clone()),
        }
    }

    fn get(&self) -> Result<&Opening, WorkingMemoryError> {
        self.opening
            .get()
            .ok_or(WorkingMemoryError::ExecutionFenced)
    }

    pub(super) fn require_sealed(&self) -> Result<(), WorkingMemoryError> {
        self.get().map(|_| ())
    }

    pub(super) fn require_pending(&self) -> Result<(), WorkingMemoryError> {
        if self.opening.get().is_some() || self.before_install.is_none() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }

    pub(super) fn predecessor(&self) -> Result<Option<&InferenceRequest>, WorkingMemoryError> {
        Ok(self.get()?.predecessor.as_ref())
    }

    pub(super) fn validate(
        &self,
        retained: &InferenceRetention,
        cached_positions: u64,
    ) -> Result<(), WorkingMemoryError> {
        let opening = self.get()?;
        retained.validate_revision(&opening.revision)?;
        match (&opening.predecessor, retained.admission()) {
            (None, None) => Ok(()),
            (Some(expected), Some(actual)) if actual.position() == cached_positions => {
                actual.request().validate_same_request(expected)
            }
            _ => Err(WorkingMemoryError::IdentityMismatch),
        }
    }

    /// Only TextExecutionQuote's runtime-reading endpoint calls this after its
    /// exact target/frontier checks. The installer must independently bind and
    /// exchange the actual completed copy. No raw revision can be published.
    pub(super) fn publish_installed(
        &self,
        actual: &InferenceRetention,
    ) -> Result<(), WorkingMemoryError> {
        self.require_pending()?;
        if !actual.is_empty() || self.before_install.as_ref() == Some(actual.revision()) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.opening
            .set(Opening {
                revision: actual.revision().clone(),
                predecessor: None,
            })
            .map_err(|_| WorkingMemoryError::IdentityMismatch)
    }
}

#[cfg(test)]
mod tests;
