//! Original controller compilation and immutable-source provenance.
use super::OriginalChatProfilePreparation;
use crate::working_memory::{MemoryLedger, WorkingMemoryError};
use eredu_core::{
    HostMetadataFunding, HostMetadataFundingError, SharedControllerBytes,
    SharedControllerDeclaration, SharedStorageIdentity,
};
use std::{
    alloc::Layout,
    error::Error,
    fmt,
    mem::{size_of, size_of_val},
    sync::{Arc, atomic::AtomicUsize},
};

/// Borrowed immutable outputs of one controller compiler invocation.
/// These references describe storage, not its funding or execution permission.
#[derive(Clone, Copy, Debug, Default)]
pub struct ControllerCompilationSources<'a> {
    /// Canonical serialized controller recipe, when one is needed.
    pub recipe: Option<&'a SharedControllerBytes>,
    /// Immutable compiled grammar, distinct from mutable recognition state.
    pub grammar: Option<&'a SharedControllerDeclaration>,
    /// Independent completion validation declarations, when applicable.
    pub validation: Option<&'a SharedControllerDeclaration>,
}
impl ControllerCompilationSources<'_> {
    fn retains_funding(self, funding: &HostMetadataFunding) -> bool {
        self.recipe
            .is_none_or(|source| source.retains_funding(funding))
            && self
                .grammar
                .is_none_or(|source| source.retains_funding(funding))
            && self
                .validation
                .is_none_or(|source| source.retains_funding(funding))
    }
    fn identities(self) -> Identities {
        Identities {
            recipe: self.recipe.map(|source| *source.identity()),
            grammar: self.grammar.map(|source| *source.identity()),
            validation: self.validation.map(|source| *source.identity()),
        }
    }
}

/// Output inspection must only borrow the immutable owners constructed by the
/// compiler. It must neither allocate nor reconstruct another representation.
pub trait ControllerCompilationOutput {
    /// The exact owners retained by this result and all execution consumers.
    fn controller_sources(&self) -> ControllerCompilationSources<'_>;
}

/// Explicit allocation contract for a facade-owned controller compiler.
///
/// Implementations must construct their output through prospectively funded
/// producers using the supplied account. Every escaping output, alias and error
/// must retain that account through its actual allocation retirement. Borrowed
/// inputs remain caller-owned; existing output storage cannot be adopted or
/// relabeled as newly funded. A required unknown producer must refuse before
/// running. Implementations must not retry through an unenforced compiler.
pub trait OriginalControllerCompiler {
    /// Immutable compiled policy and declarations.
    type Output: ControllerCompilationOutput;
    /// Typed compiler failure, retaining any partial output.
    type Error: Error + Send + Sync + 'static;
    /// Compile once against the original source preparation account.
    fn compile(self, funding: &HostMetadataFunding) -> Result<Self::Output, Self::Error>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Identities {
    recipe: Option<SharedStorageIdentity>,
    grammar: Option<SharedStorageIdentity>,
    validation: Option<SharedStorageIdentity>,
}
#[derive(Debug)]
struct Receipt {
    identities: Identities,
    preparation: OriginalChatProfilePreparation,
}

/// Closed provenance from one prospectively funded compilation. This records
/// exact source identities and retains their actual construction account. It
/// grants no storage discount, native allocation or submission authority.
#[derive(Debug, Clone)]
pub struct OriginalControllerCompilation(Option<Arc<Receipt>>);
impl Drop for OriginalControllerCompilation {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
impl OriginalControllerCompilation {
    fn receipt(&self) -> &Receipt {
        self.0.as_deref().expect("live controller compilation")
    }
    /// Exact compilation identity, independent of identical recipe contents.
    pub fn same_compilation(&self, other: &Self) -> bool {
        Arc::ptr_eq(
            self.0.as_ref().expect("live compilation"),
            other.0.as_ref().expect("live compilation"),
        )
    }
    /// Authenticate all immutable outputs against the original source pool.
    /// This read-only operation never adopts, copies or registers their storage.
    pub fn validate_sources(
        &self,
        sources: ControllerCompilationSources<'_>,
        pool: &MemoryLedger,
    ) -> Result<(), WorkingMemoryError> {
        self.receipt()
            .preparation
            .validate_pool(pool)
            .map_err(|_| WorkingMemoryError::IdentityMismatch)?;
        if sources.identities() != self.receipt().identities {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }
    /// Authenticate the exact grammar outputs and the originally derived trie.
    /// This accepts no registered replacement or byte-equal reconstruction.
    pub fn validate_grammar_sources(
        &self,
        recipe: &SharedControllerBytes,
        declaration: &SharedControllerDeclaration,
        trie: &crate::working_memory::OriginalTokenTrieSource,
    ) -> Result<(), WorkingMemoryError> {
        let receipt = self.receipt();
        if receipt.identities.recipe != Some(*recipe.identity())
            || receipt.identities.grammar != Some(*declaration.identity())
            || !trie.matches_semantic_tokenizer(receipt.preparation.tokenizer())
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }
    /// Authenticate the full-schema declaration consumed after a completed call.
    /// Grammar acceptance and equal schema contents confer no source identity.
    pub fn validate_validation_sources(
        &self,
        recipe: &SharedControllerBytes,
        validation: &SharedControllerDeclaration,
    ) -> Result<(), WorkingMemoryError> {
        let identities = &self.receipt().identities;
        if identities.recipe != Some(*recipe.identity())
            || identities.validation != Some(*validation.identity())
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }
    pub(in crate::working_memory) fn validate_consumer(
        &self,
        preparation: &crate::working_memory::PreparedSemanticSource,
    ) -> Result<(), WorkingMemoryError> {
        self.receipt()
            .preparation
            .validate_semantic_preparation(preparation)
    }
    /// Fixed borrowed receipt/source comparisons, including role projections.
    pub fn validation_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<(&Self, &SharedControllerBytes, &SharedControllerDeclaration)>(),
            size_of::<(
                &Self,
                &SharedControllerBytes,
                &SharedControllerDeclaration,
                &crate::working_memory::OriginalTokenTrieSource,
            )>(),
            size_of::<(&Self, &crate::working_memory::PreparedSemanticSource)>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<Option<SharedStorageIdentity>>(),
            size_of::<bool>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Actual original construction payer; comparison grants no replacement allowance.
    pub fn metadata_funding(&self) -> &HostMetadataFunding {
        self.receipt().preparation.metadata_funding()
    }
    /// Join this immutable compilation to the exact tokenizer and selected
    /// execution of the eventual semantic consumer. Equal source contents or a
    /// shared pool cannot substitute for either identity.
    pub fn validate_preparation(
        &self,
        sources: ControllerCompilationSources<'_>,
        preparation: &crate::working_memory::PreparedSemanticSource,
    ) -> Result<(), WorkingMemoryError> {
        self.validate_sources(sources, self.receipt().preparation.tokenizer().pool())?;
        self.validate_consumer(preparation)
    }
}

#[derive(Debug)]
enum Cause<E> {
    Funding(HostMetadataFundingError),
    Compiler(E),
    Source(WorkingMemoryError),
}
/// A rejected compilation retains its original preparation after its compiler
/// diagnostic and partial owners retire.
#[derive(Debug)]
pub struct OriginalControllerCompilationError<E> {
    cause: Cause<E>,
    preparation: OriginalChatProfilePreparation,
}
impl<E: Error + 'static> fmt::Display for OriginalControllerCompilationError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Funding(cause) => fmt::Display::fmt(cause, f),
            Cause::Compiler(cause) => fmt::Display::fmt(cause, f),
            Cause::Source(cause) => fmt::Display::fmt(cause, f),
        }
    }
}
impl<E: Error + 'static> Error for OriginalControllerCompilationError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(match &self.cause {
            Cause::Funding(cause) => cause,
            Cause::Compiler(cause) => cause,
            Cause::Source(cause) => cause,
        })
    }
}

impl OriginalChatProfilePreparation {
    /// Invoke the explicit bounded compiler contract exactly once. Constructor,
    /// output-inspection and receipt controls are funded before the invocation;
    /// the compiler pays its actual destinations through the same account.
    pub fn compile_controller<C: OriginalControllerCompiler>(
        &self,
        compiler: C,
    ) -> Result<
        (C::Output, OriginalControllerCompilation),
        OriginalControllerCompilationError<C::Error>,
    > {
        let result = (|| {
            let shell = Layout::new::<[AtomicUsize; 2]>()
                .extend(Layout::new::<Receipt>())
                .map_err(|_| Cause::Funding(HostMetadataFundingError::Overflow))?
                .0
                .pad_to_align()
                .size();
            let parts = [
                shell,
                size_of::<C>(),
                size_of::<C::Output>(),
                size_of::<C::Error>(),
                size_of::<Receipt>(),
                size_of::<Option<Receipt>>(),
                size_of::<Identities>(),
                size_of::<Arc<Receipt>>(),
                size_of::<Option<Arc<Receipt>>>(),
                size_of::<ControllerCompilationSources<'_>>(),
                size_of::<(
                    &OriginalControllerCompilation,
                    ControllerCompilationSources<'_>,
                    &crate::working_memory::PreparedSemanticSource,
                )>(),
                size_of::<Result<(), WorkingMemoryError>>(),
                size_of::<OriginalControllerCompilation>(),
                size_of::<OriginalControllerCompilationError<C::Error>>(),
                size_of::<Cause<C::Error>>(),
                size_of::<(&Self, C)>(),
                size_of::<Result<C::Output, C::Error>>(),
                size_of::<Result<(C::Output, OriginalControllerCompilation), Cause<C::Error>>>(),
                size_of::<
                    Result<
                        (C::Output, OriginalControllerCompilation),
                        OriginalControllerCompilationError<C::Error>,
                    >,
                >(),
                HostMetadataFunding::reservation_control_bytes(),
            ];
            let bytes = parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or(Cause::Funding(HostMetadataFundingError::Overflow))?;
            self.metadata_funding()
                .reserve_metadata(bytes)
                .map_err(Cause::Funding)?;
            let output = compiler
                .compile(self.metadata_funding())
                .map_err(Cause::Compiler)?;
            let sources = output.controller_sources();
            if !sources.retains_funding(self.metadata_funding()) {
                return Err(Cause::Source(WorkingMemoryError::IdentityMismatch));
            }
            let identities = sources.identities();
            let receipt = OriginalControllerCompilation(Some(Arc::new(Receipt {
                identities,
                preparation: self.clone(),
            })));
            Ok((output, receipt))
        })();
        result.map_err(|cause| OriginalControllerCompilationError {
            cause,
            preparation: self.clone(),
        })
    }
}

#[cfg(test)]
mod tests;
