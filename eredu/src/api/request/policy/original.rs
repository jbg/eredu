//! Source-authenticated compilation through the same policy producer.
use super::{CompilationFailure, CompiledChatPolicy};
use crate::runtime::chat::preparation_memory::{
    PreparationFailure, PreparationFunding, StorageFailure,
};
use crate::{
    api::request::probes::original::Tokens,
    runtime::chat::{
        ChatTemplateRequest, PreparedFormatProfile,
        constraints::{ConstraintCompiler, ConstraintCompilerSourceError},
    },
};
use eredu_core::{HostMetadataFunding, HostMetadataFundingError, ModelRuntime};
use eredu_runtime::working_memory::{
    OriginalChatBackend, OriginalChatProfilePreparation, OriginalControllerCompilation,
    OriginalControllerCompilationError, OriginalControllerCompiler, OriginalEncodedTokenIds,
    OriginalTextSourceError, OriginalTokenizer,
};
use eredu_text::tokenizer::structural::{
    StructuralTokenFailure, resolve_structural_with, structural_control_bytes,
};
use std::{
    error::Error,
    fmt,
    mem::{size_of, size_of_val},
};

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Metadata(#[from] HostMetadataFundingError),
    #[error(transparent)]
    Compiler(#[from] ConstraintCompilerSourceError),
    #[error(transparent)]
    Policy(#[from] CompilationFailure<StructuralFailure>),
}
/// Compiler/encoding errors retire before the exact profile account used here.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(crate) struct Failure {
    #[source]
    cause: Cause,
    _funding: HostMetadataFunding,
}

pub(crate) fn compile_original<B: OriginalChatBackend>(
    runtime: &ModelRuntime<B>,
    profile: &PreparedFormatProfile,
    request: &ChatTemplateRequest,
    eos: &[u32],
    preparation: &OriginalChatProfilePreparation,
    memory: crate::runtime::chat::DependencyMemoryPolicy,
) -> Result<
    (CompiledChatPolicy, OriginalControllerCompilation),
    OriginalControllerCompilationError<Failure>,
> {
    preparation.compile_controller(Compiler {
        runtime,
        tokenizer: preparation.tokenizer(),
        profile,
        request,
        eos,
        memory,
    })
}

struct Compiler<'a, B: OriginalChatBackend> {
    runtime: &'a ModelRuntime<B>,
    tokenizer: &'a OriginalTokenizer,
    profile: &'a PreparedFormatProfile,
    request: &'a ChatTemplateRequest,
    eos: &'a [u32],
    memory: crate::runtime::chat::DependencyMemoryPolicy,
}
impl<B: OriginalChatBackend> OriginalControllerCompiler for Compiler<'_, B> {
    type Output = CompiledChatPolicy;
    type Error = Failure;
    fn compile(self, funding: &HostMetadataFunding) -> Result<Self::Output, Self::Error> {
        compile_policy(
            self.runtime,
            self.tokenizer,
            self.profile,
            self.request,
            self.eos,
            funding,
            self.memory,
        )
    }
}

fn compile_policy<B: OriginalChatBackend>(
    runtime: &ModelRuntime<B>,
    tokenizer: &OriginalTokenizer,
    profile: &PreparedFormatProfile,
    request: &ChatTemplateRequest,
    eos: &[u32],
    funding: &HostMetadataFunding,
    memory: crate::runtime::chat::DependencyMemoryPolicy,
) -> Result<CompiledChatPolicy, Failure> {
    let result = (|| -> Result<_, Cause> {
        let parts = [
            size_of::<(
                &ModelRuntime<B>,
                &OriginalTokenizer,
                &PreparedFormatProfile,
                &ChatTemplateRequest,
                &[u32],
                &HostMetadataFunding,
            )>(),
            size_of::<HostMetadataFunding>(),
            size_of::<ConstraintCompiler>(),
            size_of::<Option<ConstraintCompiler>>(),
            size_of::<Option<PreparationFunding>>(),
            size_of::<Result<ConstraintCompiler, ConstraintCompilerSourceError>>(),
            size_of::<Cause>(),
            size_of::<Failure>(),
            size_of::<Result<CompiledChatPolicy, Cause>>(),
            size_of::<Result<CompiledChatPolicy, Failure>>(),
            size_of::<Result<CompiledChatPolicy, CompilationFailure<StructuralFailure>>>(),
        ];
        let bytes = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or(HostMetadataFundingError::Overflow)?;
        funding.reserve_metadata(bytes)?;
        let compiler = if profile.dialect.is_some() || profile.tool_dialect.is_some() {
            Some(
                ConstraintCompiler::from_original_tokenizer_with_memory_policy(
                    tokenizer.clone(),
                    eos,
                    funding,
                    memory,
                )?,
            )
        } else {
            None
        };
        // Literal text for an unrecognized protocol requires no grammar trie.
        // Its immutable policy metadata still uses the exact source account.
        let metadata_funding = if compiler.is_none() {
            Some(PreparationFunding::from_metadata(funding).with_memory_policy(memory))
        } else {
            None
        };
        let allocation = compiler
            .as_ref()
            .map(ConstraintCompiler::allocation_funding)
            .or(metadata_funding.as_ref())
            .expect("compiler or metadata funding");
        Ok(CompiledChatPolicy::compile(
            profile,
            request,
            eos,
            Ok(compiler.as_ref()),
            allocation,
            |spellings, payer| resolve(runtime, tokenizer, spellings, payer),
        )?)
    })();
    result.map_err(|cause| Failure {
        cause,
        _funding: funding.clone(),
    })
}

#[derive(Debug)]
enum StructuralCause {
    Funding(PreparationFailure),
    Storage(StorageFailure),
    Validation(StructuralTokenFailure<OriginalTextSourceError, OriginalEncodedTokenIds>),
}
#[derive(Debug)]
struct StructuralFailure {
    cause: StructuralCause,
    _funding: PreparationFunding,
}
fn resolve<B: OriginalChatBackend>(
    runtime: &ModelRuntime<B>,
    tokenizer: &OriginalTokenizer,
    spellings: &[String],
    funding: &PreparationFunding,
) -> Result<Vec<u32>, StructuralFailure> {
    let result = (|| -> Result<_, StructuralCause> {
        let parts = [
            size_of::<Tokens<'_, B>>(),
            size_of::<Vec<u32>>(),
            size_of::<StructuralFailure>(),
            size_of::<StructuralCause>(),
            size_of::<Result<Vec<u32>, StructuralFailure>>(),
            size_of::<Result<Vec<u32>, StructuralCause>>(),
            size_of::<(
                &ModelRuntime<B>,
                &OriginalTokenizer,
                &[String],
                &PreparationFunding,
            )>(),
        ];
        let controls = structural_control_bytes::<Tokens<'_, B>, String>()
            .and_then(|n| n.checked_add(size_of_val(&parts)))
            .and_then(|n| parts.into_iter().try_fold(n, usize::checked_add))
            .ok_or_else(|| StructuralCause::Storage(funding.storage_overflow()))?;
        funding
            .reserve(controls)
            .map_err(StructuralCause::Funding)?;
        let mut ids = Vec::new();
        funding
            .try_grow_vec(&mut ids, spellings.len())
            .map_err(StructuralCause::Storage)?;
        ids.resize(spellings.len(), 0);
        resolve_structural_with(&mut Tokens::new(runtime, tokenizer), spellings, &mut ids)
            .map_err(StructuralCause::Validation)?;
        Ok(ids)
    })();
    result.map_err(|cause| StructuralFailure {
        cause,
        _funding: funding.clone(),
    })
}
impl fmt::Display for StructuralFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use StructuralTokenFailure as V;
        match &self.cause {
            StructuralCause::Funding(e) => e.fmt(f),
            StructuralCause::Storage(e) => e.fmt(f),
            StructuralCause::Validation(error) => match error {
                V::Destination => f.write_str("structural token destination population changed"),
                V::Empty { index } => write!(f, "structural token {index} has an empty spelling"),
                V::Repeated { index } => {
                    write!(f, "structural token {index} repeats a prior spelling")
                }
                V::MissingAdded { index } => {
                    write!(f, "structural token {index} is not an added token")
                }
                V::Forward { index, id } => write!(
                    f,
                    "structural token {index} does not resolve to added ID {id}"
                ),
                V::Reverse { index, id } => write!(
                    f,
                    "structural token {index} does not round-trip through ID {id}"
                ),
                V::Encoding { index, cause } => {
                    write!(f, "failed to encode structural token {index}: {cause}")
                }
                V::NonAtomic { index, id, encoded } => write!(
                    f,
                    "structural token {index} is not atomic with ID {id}; encoded as {:?}",
                    encoded.ids()
                ),
                V::Ambiguous {
                    previous,
                    index,
                    id,
                } => write!(
                    f,
                    "structural tokens {previous} and {index} both resolve to ID {id}"
                ),
            },
        }
    }
}
impl Error for StructuralFailure {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.cause {
            StructuralCause::Funding(e) => Some(e),
            StructuralCause::Storage(e) => Some(e),
            StructuralCause::Validation(StructuralTokenFailure::Encoding { cause, .. }) => {
                Some(cause)
            }
            StructuralCause::Validation(_) => None,
        }
    }
}
