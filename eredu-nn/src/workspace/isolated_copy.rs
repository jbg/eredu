//! Sealed metadata preparation for the fixed isolated-copy program.

use super::*;
use crate::{IsolatedCopyMechanism, isolated_copy};
mod finite;
pub use finite::{
    WorkspaceCopyPreparationError, WorkspaceCopyPreparationLayout,
    WorkspaceCopyPreparationLayoutBuilder, WorkspaceIsolatedCopyPreparation,
};

/// Structural or arithmetic rejection while preparing an isolated copy.
#[derive(Debug, thiserror::Error)]
pub enum WorkspaceCopyError {
    /// A source or its borrowed selection belongs to another metadata context.
    #[error("isolated-copy source belongs to another workspace context")]
    ContextMismatch,
    /// An operation result cannot stand in for an imported physical source.
    #[error("isolated-copy source {index} was not imported from existing storage")]
    SourceNotImported {
        /// Ordered source slot that failed validation.
        index: usize,
    },
    /// A source does not directly name one of the selected physical roots.
    #[error("isolated-copy source {index} does not name a selected existing root")]
    SourceStorageMismatch {
        /// Ordered source slot that failed validation.
        index: usize,
    },
    /// Some selected root is absent from the actual ordered source slots.
    #[error("isolated-copy sources do not exactly cover the borrowed storage selection")]
    SourceSetMismatch,
    /// The provider could not certify the source backing's capacity.
    #[error("isolated-copy source {index} has unknown backing capacity")]
    UnknownSourceCapacity {
        /// Ordered source slot lacking a complete capacity fact.
        index: usize,
    },
    /// A known deep-copy effect permits aliasing instead of independent storage.
    #[error("isolated-copy destination {index} is not independent")]
    NonIndependentDestination {
        /// Ordered destination slot with an invalid storage declaration.
        index: usize,
    },
    /// A checked byte or extent calculation exceeded its representation.
    #[error("{operation}")]
    Overflow {
        /// The arithmetic operation whose result cannot be represented.
        operation: &'static str,
    },
    /// The selected mechanism rejected the operation, preserving its cause.
    #[error("{0}")]
    Mechanism(#[source] Error),
}

fn map_error(error: Error) -> WorkspaceCopyError {
    use std::error::Error as _;
    match error
        .source()
        .and_then(|source| source.downcast_ref::<WorkspaceOverflow>())
    {
        Some(overflow) => WorkspaceCopyError::Overflow {
            operation: overflow.0,
        },
        None => WorkspaceCopyError::Mechanism(error),
    }
}

fn validate_sources(
    context: &WorkspaceContext,
    borrowed: &WorkspaceBorrowedStorage,
    sources: &[WorkspaceTensor],
) -> Result<usize, WorkspaceCopyError> {
    if !borrowed.belongs_to(context) {
        return Err(WorkspaceCopyError::ContextMismatch);
    }
    // The borrowed selection already owns unique physical roots. Compare
    // those and the supplied slots directly rather than constructing three
    // additional owning identity sets for this fixed copy program.
    let selected = borrowed.roots();
    for (index, source) in sources.iter().enumerate() {
        if !Rc::ptr_eq(&source.context, &context.identity) {
            return Err(WorkspaceCopyError::ContextMismatch);
        }
        if !source.imported_existing {
            return Err(WorkspaceCopyError::SourceNotImported { index });
        }
        if source.storage.bytes.is_none() {
            return Err(WorkspaceCopyError::UnknownSourceCapacity { index });
        }
        if !source.storage.possible_aliases.is_empty()
            || !selected
                .iter()
                .any(|root| Rc::ptr_eq(&root.storage, &source.storage))
        {
            return Err(WorkspaceCopyError::SourceStorageMismatch { index });
        }
    }
    if selected.iter().any(|root| {
        !sources
            .iter()
            .any(|source| Rc::ptr_eq(&root.storage, &source.storage))
    }) {
        return Err(WorkspaceCopyError::SourceSetMismatch);
    }
    let closing_count = sources
        .len()
        .checked_mul(2)
        .ok_or(WorkspaceCopyError::Overflow {
            operation: "isolated-copy closing root count overflow",
        })?;

    Ok(closing_count)
}

/// Immutable preparation of one independent copy per ordered source slot.
///
/// Only the fixed isolated-copy program produces this plan. It retains metadata,
/// not physical storage or allocation permission. A runtime must separately bind
/// its original source selection to registered physical charges. Missing native
/// or host facts remain an incomplete diagnostic with no incremental bound.
#[derive(Debug)]
pub struct WorkspaceIsolatedCopyPlan {
    source_storage: WorkspaceBorrowedStorage,
    source_layouts: Vec<WorkspaceLayout>,
    report: WorkspaceTraceReport,
    incremental_bytes: Option<u64>,
}

impl WorkspaceIsolatedCopyPlan {
    /// Validates exact imported sources, then traces copies in a private ledger.
    ///
    /// The private context shares only mechanism facts with `context`; neither
    /// caller resets nor callback access to the caller's trace can alter this
    /// program. Aliased source slots remain distinct destination copy requests.
    pub fn prepare(
        context: &WorkspaceContext,
        borrowed: &WorkspaceBorrowedStorage,
        sources: &[WorkspaceTensor],
    ) -> Result<Self, WorkspaceCopyError> {
        let closing_count = validate_sources(context, borrowed, sources)?;
        // This legacy producer allocates its private context and diagnostic
        // clones ordinarily. Finite callers use prepare_finite_with_layout,
        // which supplies the complete private-program constructor layout.
        // Never detach a new identity from a caller's metadata allowance.
        if context.facts.is_some() {
            return Err(WorkspaceCopyError::Mechanism(
                WorkspaceMetadataError::Unqualified.into(),
            ));
        }
        let selected = borrowed.roots();

        let private = WorkspaceContext {
            identity: Rc::new(WorkspaceIdentity::new()),
            mechanisms: context.mechanisms.clone(),
            facts: context.facts.clone(),
            trace: Rc::new(RefCell::new(Trace::default())),
            borrowed: Rc::new(RefCell::new(None)),
            parameter_representations: Rc::new(RefCell::new(None)),
            tracing_started: Rc::new(report::Lifecycle::imported(selected.len())),
            funding: None,
        };
        let roots: Vec<_> = borrowed
            .roots()
            .iter()
            .map(|root| WorkspaceExistingStorage {
                storage: root.storage.clone(),
                context: private.identity.clone(),
            })
            .collect();
        let private_borrowed =
            WorkspaceBorrowedStorage::new(&private, &roots).map_err(map_error)?;
        private
            .set_borrowed_storage(private_borrowed)
            .map_err(map_error)?;
        let mut closing = Vec::with_capacity(closing_count);
        closing.extend(sources.iter().map(|source| WorkspaceTensor {
            layout: source.layout.clone(),
            storage: source.storage.clone(),
            context: private.identity.clone(),
            imported_existing: true,
        }));
        private.begin_state_span(&closing).map_err(map_error)?;
        // Sources occupy the first slots; every prior destination remains a
        // closing root while subsequent copies add their overlap and scratch.
        for index in 0..sources.len() {
            let output = isolated_copy(CopyTrace {
                source: &closing[index],
                context: &private,
                forbidden: &closing,
                index,
            })?;
            closing.push(output);
        }
        let report = private.report(&closing).map_err(map_error)?;
        let incremental_bytes = report
            .residual
            .as_ref()
            .expect("the private copy ledger has an installed borrowed selection")
            .total_bytes;
        Ok(Self {
            source_storage: borrowed.clone(),
            source_layouts: sources.iter().map(|source| source.layout.clone()).collect(),
            report,
            incremental_bytes,
        })
    }

    /// Original selection token to associate with registered physical roots.
    /// Ordinary preparation rebases the diagnostic token privately; finite
    /// preparation retains this same immutable token without exporting graph indices.
    pub fn source_storage(&self) -> &WorkspaceBorrowedStorage {
        &self.source_storage
    }

    /// Exact ordered layouts; shared backing never removes a requested copy.
    pub fn source_layouts(&self) -> &[WorkspaceLayout] {
        &self.source_layouts
    }

    /// Immutable diagnostics produced by the private fixed-program trace.
    pub fn report(&self) -> &WorkspaceTraceReport {
        &self.report
    }

    /// Root-aware demand excluding only the exact original source identities.
    /// Missing tensor or disjoint host coverage remains `None`.
    pub fn incremental_bytes(&self) -> Option<u64> {
        self.incremental_bytes
    }
}

struct CopyTrace<'a> {
    source: &'a WorkspaceTensor,
    context: &'a WorkspaceContext,
    forbidden: &'a [WorkspaceTensor],
    index: usize,
}

impl IsolatedCopyMechanism for CopyTrace<'_> {
    type Value = WorkspaceTensor;
    type Error = WorkspaceCopyError;

    fn contiguous(&self) -> Result<Self::Value, Self::Error> {
        self.source.contiguous(self.context).map_err(map_error)
    }

    fn deep_copy(&self, contiguous: Self::Value) -> Result<Self::Value, Self::Error> {
        let output = contiguous.deep_copy(self.context).map_err(map_error)?;
        if output.storage.bytes.is_some()
            && (Rc::ptr_eq(&output.storage, &contiguous.storage)
                || self
                    .forbidden
                    .iter()
                    .any(|root| Rc::ptr_eq(&root.storage, &output.storage))
                || !output.storage.possible_aliases.is_empty())
        {
            return Err(WorkspaceCopyError::NonIndependentDestination { index: self.index });
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests;
