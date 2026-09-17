//! Exact absent-table representation; no empty table or zero-byte owner is created.

use super::*;

mod prompt;
use eredu_runtime::RuntimeState;
pub(super) use prompt::{InitializedStatelessPrompt, PreparedStatelessPrompt};

#[derive(Clone, Copy)]
enum Source<'a> {
    Live(&'a MlxPoolingAttentionState),
    Saved(&'a SavedStatelessPoolingCopy),
}

/// Borrows the actual absence. The enclosing paired worker separately retains
/// its real source owner and all registered source origins through completion.
#[derive(Clone, Copy)]
pub(super) struct PreparedStatelessPoolingCopy<'a> {
    source: Source<'a>,
}

impl<'a> PreparedStatelessPoolingCopy<'a> {
    pub(super) fn prepare(source: &'a MlxPoolingAttentionState) -> Result<Self, Error> {
        Self::prepare_fixed(source).map_err(ResidentDecoderPreparationError::into_error)
    }
    pub(super) fn prepare_fixed(
        source: &'a MlxPoolingAttentionState,
    ) -> Result<Self, ResidentDecoderPreparationError> {
        if source.prepare_layer_copy_slots()?.is_some() {
            return Err(WorkingMemoryError::IdentityMismatch.into());
        }
        Ok(Self {
            source: Source::Live(source),
        })
    }

    pub(super) fn shared_layout(&self) -> Option<&'a SharedStateLayout> {
        match self.source {
            Source::Live(source) => source.shared_layout(),
            Source::Saved(source) => source.layout.as_ref(),
        }
    }

    pub(super) fn copy(self, admitted: Self) -> Result<SavedStatelessPoolingCopy, Error> {
        let same = self.same_source(&admitted);
        if !same {
            return Err(mismatch());
        }
        // Only immutable metadata already held by the real source is shared.
        // The paired sampling/native account owns all copied payload custody;
        // this absence carries no table, initializer or runnable state grant.
        Ok(SavedStatelessPoolingCopy {
            layout: self.shared_layout().cloned(),
        })
    }

    fn same_source(&self, other: &Self) -> bool {
        match (self.source, other.source) {
            (Source::Live(left), Source::Live(right)) => std::ptr::eq(left, right),
            (Source::Saved(left), Source::Saved(right)) => std::ptr::eq(left, right),
            _ => false,
        }
    }
}

pub(super) struct SavedStatelessPoolingCopy {
    layout: Option<SharedStateLayout>,
}

impl SavedStatelessPoolingCopy {
    pub(super) fn prepare_copy(&self) -> PreparedStatelessPoolingCopy<'_> {
        PreparedStatelessPoolingCopy {
            source: Source::Saved(self),
        }
    }

    pub(super) fn shared_layout(&self) -> Option<&SharedStateLayout> {
        self.layout.as_ref()
    }
}
