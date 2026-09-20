//! Original full-schema declarations, retained source custody and paid callbacks.
use super::original::{self, Source};
use crate::runtime::chat::constraints::recipe::ConstraintRecipe;
use crate::runtime::chat::preparation_memory::{
    PreparationFailure as FundingFailure, PreparationFunding, StorageFailure,
};
use eredu_core::{
    BackendFailure, ControllerDeclarationData, HostPreparationAuthority,
    SharedControllerDeclaration, SharedStorageIdentity,
};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError, WorkspaceContext};
use eredu_runtime::working_memory::{
    OriginalSemanticControllerSource, OriginalToolValidation, WorkingMemoryError, WorkingMemoryPool,
};
use std::{
    alloc::Layout,
    fmt,
    mem::{size_of, size_of_val},
    sync::Arc,
};

mod tagged;

struct CompiledSchemas {
    rows: Vec<(String, Source, Option<tagged::Tagged>)>,
    bytes: u64,
    funding: PreparationFunding,
}
struct Data {
    schemas: CompiledSchemas,
    recipe: SharedStorageIdentity,
}
impl ControllerDeclarationData for Data {
    fn owned_capacity_bytes(&self) -> Option<u64> {
        None
    }
    fn admission_bytes(&self) -> Option<u64> {
        Some(self.schemas.bytes)
    }
}
#[derive(Debug, thiserror::Error)]
enum CompilationCause {
    #[error(transparent)]
    Parameter(original::CompilationFailure),
    #[error(transparent)]
    TaggedGrammar(#[from] crate::runtime::chat::grammar_text::Error),
    #[error(transparent)]
    Funding(#[from] FundingFailure),
    #[error(transparent)]
    Storage(#[from] StorageFailure),
    #[error("tools[{index}].function.parameters: {cause}")]
    Schema {
        index: usize,
        #[source]
        cause: original::CompilationFailure,
    },
    #[error("tool schema source extent overflow")]
    Overflow,
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(crate) struct CompilationFailure {
    #[source]
    cause: CompilationCause,
    funding: PreparationFunding,
    authority: HostPreparationAuthority,
}
fn compile_data(
    tools: &[super::ToolDefinition<'_>],
    funding: &PreparationFunding,
    authority: &HostPreparationAuthority,
    tagged: bool,
) -> Result<CompiledSchemas, CompilationFailure> {
    let result = (|| -> Result<_, CompilationCause> {
        // These are the actual facade producers. The dependency's compiler is
        // responsible for the validator graph and its transient construction.
        let shell = SharedControllerDeclaration::source_shell_bytes::<Data>()
            .ok_or(CompilationCause::Overflow)?;
        let controls = [
            SharedControllerDeclaration::source_constructor_bytes::<Data>()
                .ok_or(CompilationCause::Overflow)?,
            size_of::<CompiledSchemas>(),
            size_of::<PendingSchemas>(),
            size_of::<Historical>(),
            size_of::<Data>(),
            size_of::<CompilationFailure>(),
            size_of::<CompilationCause>(),
            size_of::<Result<CompiledSchemas, CompilationFailure>>(),
            size_of::<Result<CompiledSchemas, CompilationCause>>(),
            size_of::<Result<Source, original::CompilationFailure>>(),
            size_of::<Result<usize, original::CompilationFailure>>(),
            size_of::<Layout>(),
            size_of::<Option<original::CompilationFailure>>(),
            size_of::<std::iter::Enumerate<std::slice::Iter<'_, super::ToolDefinition<'_>>>>(),
            size_of::<(
                &[super::ToolDefinition<'_>],
                &PreparationFunding,
                &HostPreparationAuthority,
            )>(),
        ];
        funding.reserve(
            controls
                .into_iter()
                .try_fold(size_of_val(&controls), usize::checked_add)
                .ok_or(CompilationCause::Overflow)?,
        )?;
        let mut rows = Vec::new();
        funding.try_grow_vec(&mut rows, tools.len())?;
        let mut bytes = shell;
        for (index, definition) in tools.iter().enumerate() {
            let validator = Source::compile(definition.parameters, authority, funding)
                .map_err(|cause| CompilationCause::Schema { index, cause })?;
            let name = funding.try_copy_str(definition.name)?;
            bytes = bytes
                .checked_add(validator.admission_bytes())
                .and_then(|n| n.checked_add(name.capacity()))
                .ok_or(CompilationCause::Overflow)?;
            let tagged = tagged
                .then(|| tagged::Tagged::compile(definition.parameters, authority, funding))
                .transpose()?;
            bytes = bytes
                .checked_add(tagged.as_ref().map_or(Ok(0), tagged::Tagged::bytes)?)
                .ok_or(CompilationCause::Overflow)?;
            rows.push((name, validator, tagged));
        }
        bytes = bytes
            .checked_add(
                Layout::array::<(String, Source, Option<tagged::Tagged>)>(rows.capacity())
                    .map_err(|_| CompilationCause::Overflow)?
                    .size(),
            )
            .ok_or(CompilationCause::Overflow)?;
        Ok(CompiledSchemas {
            rows,
            bytes: u64::try_from(bytes).map_err(|_| CompilationCause::Overflow)?,
            funding: funding.clone(),
        })
    })();
    result.map_err(|cause| CompilationFailure {
        cause,
        funding: funding.clone(),
        authority: authority.clone(),
    })
}
/// Compiles completion validators before grammar lowering. Borrowed grammar
/// projections never rebuild these graphs or clone the application schemas.
pub(crate) struct PendingSchemas {
    schemas: CompiledSchemas,
    authority: HostPreparationAuthority,
}
impl PendingSchemas {
    pub(crate) fn compile(
        tools: &[super::ToolDefinition<'_>],
        funding: &PreparationFunding,
        authority: &HostPreparationAuthority,
        tagged: bool,
    ) -> Result<Self, CompilationFailure> {
        Ok(Self {
            schemas: compile_data(tools, funding, authority, tagged)?,
            authority: authority.clone(),
        })
    }
    pub(crate) fn bind(self, recipe: &ConstraintRecipe) -> Historical {
        Historical {
            source: SharedControllerDeclaration::new(
                Data {
                    schemas: self.schemas,
                    recipe: *recipe.source().identity(),
                },
                self.authority.clone(),
            ),
        }
    }
}
/// Owns the original compiled immutable schemas. Every alias retains their
/// prospective construction funding through the shared declaration owner.
#[derive(Clone, Debug)]
pub(crate) struct Historical {
    source: SharedControllerDeclaration,
}
impl Historical {
    pub(crate) fn source(&self) -> &SharedControllerDeclaration {
        &self.source
    }
    fn data(&self, recipe: &ConstraintRecipe) -> Result<&Data, PreparationCause> {
        self.source
            .declaration::<Data>()
            .filter(|data| &data.recipe == recipe.source().identity())
            .ok_or(PreparationCause::Source)
    }
    pub(crate) fn prepare(
        &self,
        recipe: &ConstraintRecipe,
        funding: &HostMetadataFunding,
    ) -> Result<Arc<dyn OriginalToolValidation>, PreparationFailure> {
        let result = (|| {
            let parts = [
                SharedControllerDeclaration::inspection_control_bytes::<Data>()
                    .ok_or(PreparationCause::Overflow)?,
                eredu_runtime::working_memory::OriginalTokenTrieSource::grammar_validation_control_bytes()
                    .ok_or(PreparationCause::Overflow)?,
                WorkspaceContext::metadata_arc_bytes::<Callback>()
                    .ok_or(PreparationCause::Overflow)?,
                size_of::<Self>(),
                size_of::<Callback>(),
                size_of::<PreparationCause>(),
                size_of::<PreparationFailure>(),
                size_of::<Result<Arc<dyn OriginalToolValidation>, PreparationFailure>>(),
                size_of::<Result<&Data, PreparationCause>>(),
                size_of::<Option<&Data>>(),
                size_of::<(&Self, &ConstraintRecipe, &HostMetadataFunding)>(),
                size_of::<(OriginalSemanticControllerSource<'_>, &WorkingMemoryPool)>(),
                size_of::<Result<(), WorkingMemoryError>>(),
            ];
            funding.reserve_metadata(
                parts
                    .into_iter()
                    .try_fold(size_of_val(&parts), usize::checked_add)
                    .ok_or(PreparationCause::Overflow)?,
            )?;
            self.data(recipe)?;
            Ok(Arc::new(Callback {
                source: self.clone(),
                recipe: recipe.clone(),
                funding: funding.clone(),
            }) as Arc<dyn OriginalToolValidation>)
        })();
        result.map_err(|cause| PreparationFailure {
            cause,
            source: self.clone(),
            recipe: recipe.clone(),
            funding: funding.clone(),
        })
    }
}
#[derive(Debug, thiserror::Error)]
enum PreparationCause {
    #[error("tool schema declaration does not match its exact compiled recipe")]
    Source,
    #[error("original tool schema callback extent overflow")]
    Overflow,
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
}
#[derive(Debug)]
pub(crate) struct PreparationFailure {
    cause: PreparationCause,
    source: Historical,
    recipe: ConstraintRecipe,
    funding: HostMetadataFunding,
}
impl fmt::Display for PreparationFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for PreparationFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            PreparationCause::Funding(error) => Some(error),
            cause => Some(cause),
        }
    }
}
#[derive(Debug)]
struct Callback {
    source: Historical,
    recipe: ConstraintRecipe,
    funding: HostMetadataFunding,
}
#[derive(Debug, thiserror::Error)]
enum CallbackCause {
    #[error(transparent)]
    Tagged(#[from] tagged::ParseFailure),
    #[error("tool function is absent from the exact compiled source")]
    UnknownTool,
    #[error("original tool source identity changed")]
    Source,
    #[error(transparent)]
    Validation(#[from] original::Failure),
}
#[derive(Debug)]
struct CallbackFailure {
    // Actual argument and parse-prefix buffers retire before source and funding.
    cause: CallbackCause,
    source: Historical,
    recipe: ConstraintRecipe,
    funding: HostMetadataFunding,
}
impl fmt::Display for CallbackFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for CallbackFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            CallbackCause::Validation(error) => Some(error),
            cause => Some(cause),
        }
    }
}
impl OriginalToolValidation for Callback {
    fn contains_tagged_tool(&self, name: &str) -> bool {
        self.source.data(&self.recipe).ok().is_some_and(|data| {
            data.schemas
                .rows
                .iter()
                .any(|(actual, _, tagged)| actual == name && tagged.is_some())
        })
    }
    fn tagged_missing_required(
        &self,
        name: &str,
        parameters: &eredu_text::semantic_channels::tagged::TaggedParameters,
    ) -> bool {
        self.source
            .data(&self.recipe)
            .ok()
            .and_then(|data| {
                data.schemas
                    .rows
                    .iter()
                    .find(|(actual, _, _)| actual == name)
            })
            .and_then(|row| row.2.as_ref())
            .is_none_or(|tagged| tagged.missing(parameters))
    }
    fn parse_tagged_parameter(
        &self,
        name: &str,
        parameter: &str,
        declared: Option<&str>,
        raw: &str,
        funding: &HostMetadataFunding,
    ) -> Result<serde_json::Value, BackendFailure> {
        let result = (|| {
            let data = self
                .source
                .data(&self.recipe)
                .map_err(|_| CallbackCause::Source)?;
            let tagged = data
                .schemas
                .rows
                .iter()
                .find(|(actual, _, _)| actual == name)
                .and_then(|row| row.2.as_ref())
                .ok_or(CallbackCause::UnknownTool)?;
            Ok(tagged.parse(parameter, declared, raw, funding)?)
        })();
        result.map_err(|cause| {
            BackendFailure::from_error(CallbackFailure {
                cause,
                source: self.source.clone(),
                recipe: self.recipe.clone(),
                funding: funding.clone(),
            })
        })
    }
    fn validate_source(
        &self,
        controller: OriginalSemanticControllerSource<'_>,
        pool: &WorkingMemoryPool,
    ) -> Result<(), WorkingMemoryError> {
        let OriginalSemanticControllerSource::Grammar(grammar) = controller else {
            return Err(WorkingMemoryError::IdentityMismatch);
        };
        if !self.recipe.source().same_storage(grammar.recipe())
            || self.source.data(&self.recipe).is_err()
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let trie = grammar
            .tokenizer()
            .downcast_ref::<eredu_runtime::working_memory::OriginalTokenTrieSource>()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        trie.validate_grammar_source(grammar, pool)?;
        grammar
            .compilation()
            .downcast_ref::<eredu_runtime::working_memory::OriginalControllerCompilation>()
            .ok_or(WorkingMemoryError::IdentityMismatch)?
            .validate_validation_sources(self.recipe.source(), &self.source.source)
    }
    fn failure_control_bytes(&self) -> Option<usize> {
        let parts = [
            BackendFailure::source_retention_peak_bytes::<CallbackFailure>()?,
            size_of::<CallbackFailure>(),
            size_of::<CallbackCause>(),
            size_of::<Result<(), CallbackCause>>(),
            size_of::<Result<(), BackendFailure>>(),
            size_of::<Option<&(String, Source, Option<tagged::Tagged>)>>(),
            size_of::<std::slice::Iter<'_, (String, Source, Option<tagged::Tagged>)>>(),
            size_of::<(&Self, &str, &str, &HostMetadataFunding)>(),
            SharedControllerDeclaration::inspection_control_bytes::<Data>()?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    fn validate(
        &self,
        name: &str,
        arguments: &str,
        funding: &HostMetadataFunding,
    ) -> Result<(), BackendFailure> {
        // The shared channel caller reserves failure_control_bytes first. Even
        // the Source worker's first reserve failure can therefore retain its
        // actual source/input without any new unfunded error transport.
        let result = (|| {
            let data = self
                .source
                .data(&self.recipe)
                .map_err(|_| CallbackCause::Source)?;
            let (_, source, _) = data
                .schemas
                .rows
                .iter()
                .find(|(actual, _, _)| actual == name)
                .ok_or(CallbackCause::UnknownTool)?;
            source.validate(arguments, funding)?;
            Ok::<_, CallbackCause>(())
        })();
        result.map_err(|cause| {
            BackendFailure::from_error(CallbackFailure {
                cause,
                source: self.source.clone(),
                recipe: self.recipe.clone(),
                funding: funding.clone(),
            })
        })
    }
}

#[cfg(test)]
mod tests;
