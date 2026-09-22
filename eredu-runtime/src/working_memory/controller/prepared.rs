//! Closed association of independently paid semantic state with its execution.
use super::*;
use crate::working_memory::{
    InferenceExecutionIdentity, OriginalControllerCompilation, OriginalForbiddenSource,
    OriginalTokenTrieSource, OriginalTokenizer, PreparedSemanticSource,
};
use eredu_core::{
    OriginalSourceWitness, PreparedControllerSource, SpeculativeTokenFilterController,
    speculative::PreparedGrammarController,
};
use std::mem::{size_of, size_of_val};

/// Fixed constructor refusal preserving the original metadata funding cause.
#[derive(Debug, thiserror::Error)]
pub enum PreparedControllerBindingError {
    /// The existing cumulative account refused concrete control storage.
    #[error(transparent)]
    Funding(#[from] eredu_core::HostMetadataFundingError),
    /// Actual controller inputs, tokenizer or mutable payer did not match.
    #[error(transparent)]
    Source(#[from] WorkingMemoryError),
}

#[cfg(test)]
mod tests;

#[derive(Debug, Clone)]
enum Inputs {
    Plain {
        validity: SharedTokenFilter,
    },
    Grammar {
        trie: OriginalTokenTrieSource,
        validity: SharedTokenFilter,
        recipe: SharedControllerBytes,
        declaration: SharedControllerDeclaration,
        compilation: OriginalControllerCompilation,
    },
    Forbidden {
        source: OriginalForbiddenSource,
        validity: SharedTokenFilter,
    },
}

/// Exact semantic preparation and state inputs. Clones retain the same paid
/// account and immutable sources; they create no parser or native permission.
#[derive(Debug, Clone)]
pub struct PreparedControllerBinding {
    inputs: Inputs,
    preparation: PreparedSemanticSource,
}
impl PartialEq for PreparedControllerBinding {
    fn eq(&self, other: &Self) -> bool {
        self.preparation.same_preparation(&other.preparation)
            && match (&self.inputs, &other.inputs) {
                (Inputs::Plain { validity: a }, Inputs::Plain { validity: b }) => a.same_storage(b),
                (
                    Inputs::Grammar {
                        trie: a,
                        validity: av,
                        recipe: ar,
                        declaration: ad,
                        compilation: ac,
                    },
                    Inputs::Grammar {
                        trie: b,
                        validity: bv,
                        recipe: br,
                        declaration: bd,
                        compilation: bc,
                    },
                ) => {
                    ac.same_compilation(bc)
                        && a.same_source(b)
                        && av.same_storage(bv)
                        && ar.same_storage(br)
                        && ad.same_storage(bd)
                }
                (
                    Inputs::Forbidden {
                        source: a,
                        validity: av,
                    },
                    Inputs::Forbidden {
                        source: b,
                        validity: bv,
                    },
                ) => a.inputs().same_source(b.inputs()) && av.same_storage(bv),
                _ => false,
            }
    }
}
impl Eq for PreparedControllerBinding {}
impl PreparedControllerBinding {
    fn source<C: SpeculativeTokenFilterController>(
        controller: &C,
    ) -> Option<PreparedControllerSource<'_>> {
        match (
            controller.prepared_grammar(),
            controller.prepared_forbidden_source(),
            controller.prepared_plain_source(),
        ) {
            (None, None, Some(source)) => Some(source.preparation_source()),
            (Some(grammar), None, None) => Some(PreparedControllerSource::Grammar(
                grammar.prepared_grammar_source(),
            )),
            (None, Some(source), None) => Some(PreparedControllerSource::Forbidden(source)),
            _ => None,
        }
    }
    /// Pays concrete binding controls, then authenticates actual source owners
    /// and mutable history funding. No ordinary source is adopted or registered.
    pub fn new<C: SpeculativeTokenFilterController>(
        preparation: &PreparedSemanticSource,
        controller: &C,
    ) -> Result<Self, PreparedControllerBindingError> {
        let parts = [
            Self::validation_control_bytes().ok_or(WorkingMemoryError::Overflow)?,
            size_of::<Self>(),
            size_of::<Inputs>(),
            size_of::<PreparedControllerBindingError>(),
            size_of::<Result<Self, PreparedControllerBindingError>>(),
            size_of::<(&PreparedSemanticSource, &C)>(),
            eredu_core::HostMetadataFunding::reservation_control_bytes(),
        ];
        preparation.metadata_funding().reserve_metadata(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or(WorkingMemoryError::Overflow)?,
        )?;
        let source = Self::source(controller).ok_or(WorkingMemoryError::IdentityMismatch)?;
        let inputs = match source {
            PreparedControllerSource::Plain { validity, .. } => Inputs::Plain {
                validity: validity.clone(),
            },
            PreparedControllerSource::Grammar(source) => Inputs::Grammar {
                trie: source
                    .tokenizer()
                    .downcast_ref::<OriginalTokenTrieSource>()
                    .ok_or(WorkingMemoryError::IdentityMismatch)?
                    .clone(),
                validity: source.validity().clone(),
                recipe: source.recipe().clone(),
                declaration: source.declaration().clone(),
                compilation: source
                    .compilation()
                    .downcast_ref::<OriginalControllerCompilation>()
                    .ok_or(WorkingMemoryError::IdentityMismatch)?
                    .clone(),
            },
            PreparedControllerSource::Forbidden(source) => Inputs::Forbidden {
                source: source
                    .original_storage()
                    .and_then(|w| w.downcast_ref::<OriginalForbiddenSource>())
                    .ok_or(WorkingMemoryError::IdentityMismatch)?
                    .clone(),
                validity: source.validity().clone(),
            },
        };
        let binding = Self {
            inputs,
            preparation: preparation.clone(),
        };
        binding.validate_source(source)?;
        Ok(binding)
    }
    pub(in crate::working_memory) fn validation_control_bytes() -> Option<usize> {
        let parts = [
            OriginalTokenTrieSource::grammar_validation_control_bytes()?,
            OriginalForbiddenSource::validation_control_bytes()?,
            size_of::<PreparedControllerSource<'_>>(),
            size_of::<Option<PreparedControllerSource<'_>>>(),
            size_of::<TextControllerStorage<'_>>(),
            size_of::<OriginalSourceWitness<'_>>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<Option<&Self>>(),
            size_of::<(&Self, PreparedControllerSource<'_>)>(),
            size_of::<(&Self, &MemoryLedger, &InferenceExecutionIdentity)>(),
            size_of::<(&Self, &Self)>(),
            size_of::<bool>(),
            size_of::<(
                &Self,
                &crate::working_memory::PreparedSemanticState,
                PreparedControllerSource<'_>,
                usize,
            )>(),
            size_of::<Option<&eredu_core::SemanticStateOwner>>(),
            size_of::<Option<&crate::working_memory::PreparedSemanticState>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Borrows the exact input tokenizer, including its original validity owner.
    pub fn tokenizer(&self) -> &OriginalTokenizer {
        self.preparation.tokenizer()
    }
    /// The cumulative payer authenticated for this controller's mutable state.
    /// Retaining it never refreshes spent storage or grants native authority.
    pub fn metadata_funding(&self) -> &eredu_core::HostMetadataFunding {
        self.preparation.metadata_funding()
    }
    /// Exact emitted mask extent. Parser/history storage remains with its own
    /// authenticated account; it is not a numerical allowance in this run.
    pub fn workspace(&self) -> Option<TextControllerWorkspace<'_>> {
        Some(TextControllerWorkspace {
            filter: self.tokenizer().generation_domain()?.into(),
            additional_host_bytes: 0,
        })
    }
    /// Declaration of the actual current source. A foreign successor or funding
    /// account returns Unknown rather than borrowing this binding's identity.
    pub fn storage<'a, C: SpeculativeTokenFilterController>(
        &'a self,
        controller: &'a C,
    ) -> TextControllerStorage<'a> {
        match Self::source(controller).filter(|source| self.validate_source(*source).is_ok()) {
            Some(source) => TextControllerStorage::RunOwnedWithPreparedSemantic {
                binding: OriginalSourceWitness::new(self),
                source,
            },
            None => TextControllerStorage::Unknown,
        }
    }
    pub(super) fn validate_semantic_state(
        &self,
        state: &crate::working_memory::PreparedSemanticState,
        source: PreparedControllerSource<'_>,
        maximum: usize,
    ) -> Result<(), WorkingMemoryError> {
        if !self.preparation.same_preparation(state.preparation())
            || state.token_capacity() != maximum
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.validate_source(source)?;
        state.validate_controller_source(source)
    }
    pub(super) fn validate_execution(
        &self,
        pool: &MemoryLedger,
        execution: &InferenceExecutionIdentity,
    ) -> Result<(), WorkingMemoryError> {
        self.preparation.validate(pool, execution)
    }
    pub(super) fn validate_source(
        &self,
        actual: PreparedControllerSource<'_>,
    ) -> Result<(), WorkingMemoryError> {
        let reject = || WorkingMemoryError::IdentityMismatch;
        let pool = self.tokenizer().pool();
        let funding = self.preparation.metadata_funding();
        let domain = self.tokenizer().generation_domain().ok_or_else(reject)?;
        let validity = match (&self.inputs, actual) {
            (
                Inputs::Plain { validity },
                PreparedControllerSource::Plain {
                    history,
                    validity: actual,
                },
            ) => {
                pool.validate_shared_controller_source(
                    eredu_core::SharedControllerSource::Filter(actual),
                )?;
                if !validity.same_storage(actual) || !history.is_funded_by(funding) {
                    return Err(reject());
                }
                validity
            }
            (
                Inputs::Grammar {
                    trie,
                    validity,
                    recipe,
                    declaration,
                    compilation,
                },
                PreparedControllerSource::Grammar(source),
            ) => {
                trie.validate_grammar_source(source, pool)?;
                compilation.validate_consumer(&self.preparation)?;
                if !source
                    .compilation()
                    .downcast_ref::<OriginalControllerCompilation>()
                    .is_some_and(|actual| compilation.same_compilation(actual))
                {
                    return Err(reject());
                }
                if !trie.matches_semantic_tokenizer(self.tokenizer())
                    || !validity.same_storage(source.validity())
                    || !recipe.same_storage(source.recipe())
                    || !declaration.same_storage(source.declaration())
                    || !funding.same_account(source.funding())
                    || !source.history_is_funded_by(funding)
                {
                    return Err(reject());
                }
                validity
            }
            (
                Inputs::Forbidden { source, validity },
                PreparedControllerSource::Forbidden(actual),
            ) => {
                source.validate_controller(actual, pool)?;
                if !source.matches_semantic_tokenizer(self.tokenizer())
                    || !validity.same_storage(actual.validity())
                    || !actual.history_is_funded_by(funding)
                {
                    return Err(reject());
                }
                validity
            }
            _ => return Err(reject()),
        };
        // Source identities above provide authority. Values additionally ensure
        // the exact input token domain agrees with the parser's validity mask.
        if validity.as_ref() != domain {
            return Err(reject());
        }
        Ok(())
    }
}
