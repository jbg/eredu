//! Cold full-schema declarations, registered source custody and paid callbacks.
use super::original::{self, Source};
use crate::runtime::chat::constraints::recipe::ConstraintRecipe;
use eredu_core::{
    BackendFailure, ControllerDeclarationData, HostPreparationAuthority, ModelRuntime,
    SharedControllerDeclaration, SharedControllerSource, SharedStorageIdentity,
    TextGenerationBackend,
};
use eredu_nn::workspace::{
    WorkspaceContext, WorkspaceMetadataFunding, WorkspaceMetadataFundingError,
};
use eredu_runtime::working_memory::{
    OriginalSemanticControllerSource, OriginalToolValidation, WorkingMemoryError, WorkingMemoryPool,
};
use jsonschema::OriginalValidationError;
use std::{
    alloc::Layout,
    fmt,
    mem::{size_of, size_of_val},
    sync::Arc,
};

struct Data {
    rows: Vec<(String, Source)>,
    // An unsupported source is released while the cold exclusion remains live.
    // Its fixed typed cause is retained; ordinary preparation keeps its behavior.
    refusal: Option<OriginalValidationError>,
    recipe: SharedStorageIdentity,
    bytes: u64,
}
impl ControllerDeclarationData for Data {
    fn owned_capacity_bytes(&self) -> Option<u64> {
        Some(self.bytes)
    }
}
fn compile_data(tools: &[serde_json::Value], recipe: &ConstraintRecipe) -> Result<Data, String> {
    // Both ordinary and original source construction consume the same validation
    // and tool declaration worker. No prepared invocation calls this function.
    let definitions = super::parse_tools_with(tools, |schema| {
        Source::compile(schema, &HostPreparationAuthority::unmanaged())
    })?;
    let mut rows = Vec::with_capacity(definitions.len());
    let mut refusal = None;
    let shell = SharedControllerDeclaration::source_shell_bytes::<Data>()
        .ok_or("tool schema source extent overflow")?;
    let mut bytes = shell;
    for definition in definitions {
        match definition.validator.capacity_bytes() {
            Ok(capacity) => {
                bytes = bytes
                    .checked_add(capacity)
                    .and_then(|n| n.checked_add(definition.name.capacity()))
                    .ok_or("tool schema source extent overflow")?;
                rows.push((definition.name, definition.validator));
            }
            Err(cause) => {
                refusal = Some(cause);
                break;
            }
        }
    }
    if refusal.is_some() {
        // Drop the actual partial graph, row allocation and all names before
        // returning the fixed refusal. No unknown allocation is priced as zero.
        drop(rows);
        rows = Vec::new();
        bytes = shell;
    } else {
        bytes = bytes
            .checked_add(
                Layout::array::<(String, Source)>(rows.capacity())
                    .map_err(|_| "tool schema row extent overflow")?
                    .size(),
            )
            .ok_or("tool schema source extent overflow")?;
    }
    Ok(Data {
        rows,
        refusal,
        recipe: recipe.source().identity().clone(),
        bytes: u64::try_from(bytes).map_err(|_| "tool schema source extent overflow")?,
    })
}
/// Owns only independently compiled immutable schema graphs. Cold aliases retain
/// exclusion; registration constructs fresh graphs and replaces it with storage.
#[derive(Clone, Debug)]
pub(crate) struct Historical {
    source: SharedControllerDeclaration,
    authority: HostPreparationAuthority,
}
impl Historical {
    pub(crate) fn compile(
        tools: &[serde_json::Value],
        recipe: &ConstraintRecipe,
        authority: &HostPreparationAuthority,
    ) -> Result<Self, String> {
        Ok(Self {
            source: SharedControllerDeclaration::new(compile_data(tools, recipe)?),
            authority: authority.clone(),
        })
    }
    fn data(&self, recipe: &ConstraintRecipe) -> Result<&Data, PreparationCause> {
        self.source
            .declaration::<Data>()
            .filter(|data| &data.recipe == recipe.source().identity())
            .ok_or(PreparationCause::Source)
    }
    pub(crate) fn register<B: TextGenerationBackend>(
        &self,
        runtime: &ModelRuntime<B>,
        old: &ConstraintRecipe,
        new: &ConstraintRecipe,
    ) -> Result<Self, BackendFailure> {
        let source =
            B::prepare_shared_controller_declaration(runtime, || self.registered_data(old, new))?;
        Ok(Self {
            source,
            authority: HostPreparationAuthority::unmanaged(),
        })
    }
    fn registered_data(
        &self,
        old: &ConstraintRecipe,
        new: &ConstraintRecipe,
    ) -> Result<Data, BackendFailure> {
        let result = (|| {
            self.data(old).map_err(|error| error.to_string())?;
            if old.source().as_ref() != new.source().as_ref() {
                return Err("tool schema recipe changed".into());
            }
            compile_data(&new.tools()?, new)
        })();
        result.map_err(|cause| {
            BackendFailure::from_error(RegistrationFailure {
                cause,
                _schema_source: self.clone(),
                recipe: old.clone(),
            })
        })
    }
    #[cfg(test)]
    pub(crate) fn register_in_pool(
        &self,
        pool: &WorkingMemoryPool,
        old: &ConstraintRecipe,
        new: &ConstraintRecipe,
    ) -> Result<Self, BackendFailure> {
        let source = pool
            .prepare_shared_controller_declaration(|| self.registered_data(old, new))
            .map_err(BackendFailure::from_error)?;
        Ok(Self {
            source,
            authority: HostPreparationAuthority::unmanaged(),
        })
    }
    pub(crate) fn prepare(
        &self,
        recipe: &ConstraintRecipe,
        funding: &WorkspaceMetadataFunding,
    ) -> Result<Arc<dyn OriginalToolValidation>, PreparationFailure> {
        let result = (|| {
            let parts = [
                SharedControllerDeclaration::inspection_control_bytes::<Data>()
                    .ok_or(PreparationCause::Overflow)?,
                WorkingMemoryPool::shared_controller_source_validation_control_bytes()
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
                size_of::<(&Self, &ConstraintRecipe, &WorkspaceMetadataFunding)>(),
                size_of::<(OriginalSemanticControllerSource<'_>, &WorkingMemoryPool)>(),
                size_of::<Result<(), WorkingMemoryError>>(),
            ];
            funding.reserve_metadata(
                parts
                    .into_iter()
                    .try_fold(size_of_val(&parts), usize::checked_add)
                    .ok_or(PreparationCause::Overflow)?,
            )?;
            if self.data(recipe)?.refusal.is_some() {
                return Err(PreparationCause::Refused);
            }
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
#[error("{cause}")]
struct RegistrationFailure {
    cause: String,
    // Retained declaration custody is not an error source.
    _schema_source: Historical,
    recipe: ConstraintRecipe,
}
#[derive(Debug, thiserror::Error)]
enum PreparationCause {
    #[error("tool schema declaration does not match its exact registered recipe")]
    Source,
    #[error("original tool schema source requires an unqualified constructor")]
    Refused,
    #[error("original tool schema callback extent overflow")]
    Overflow,
    #[error(transparent)]
    Funding(#[from] WorkspaceMetadataFundingError),
}
#[derive(Debug)]
pub(crate) struct PreparationFailure {
    cause: PreparationCause,
    source: Historical,
    recipe: ConstraintRecipe,
    funding: WorkspaceMetadataFunding,
}
impl fmt::Display for PreparationFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let PreparationCause::Refused = self.cause {
            if let Ok(data) = self.source.data(&self.recipe) {
                if let Some(error) = &data.refusal {
                    return fmt::Display::fmt(error, f);
                }
            }
        }
        fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for PreparationFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            PreparationCause::Refused => self
                .source
                .data(&self.recipe)
                .ok()?
                .refusal
                .as_ref()
                .map(|e| e as _),
            PreparationCause::Funding(error) => Some(error),
            cause => Some(cause),
        }
    }
}
#[derive(Debug)]
struct Callback {
    source: Historical,
    recipe: ConstraintRecipe,
    funding: WorkspaceMetadataFunding,
}
#[derive(Debug, thiserror::Error)]
enum CallbackCause {
    #[error("tool function is absent from the exact registered source")]
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
    funding: WorkspaceMetadataFunding,
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
    fn validate_source(
        &self,
        controller: OriginalSemanticControllerSource<'_>,
        pool: &WorkingMemoryPool,
    ) -> Result<(), WorkingMemoryError> {
        let OriginalSemanticControllerSource::Grammar(grammar) = controller else {
            return Err(WorkingMemoryError::IdentityMismatch);
        };
        if !self.recipe.source().same_storage(grammar.recipe())
            || self
                .source
                .data(&self.recipe)
                .map_or(true, |data| data.refusal.is_some())
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        pool.validate_shared_controller_source(SharedControllerSource::Bytes(
            self.recipe.source(),
        ))?;
        pool.validate_shared_controller_source(SharedControllerSource::Declaration(
            &self.source.source,
        ))
    }
    fn failure_control_bytes(&self) -> Option<usize> {
        let parts = [
            BackendFailure::source_retention_peak_bytes::<CallbackFailure>()?,
            size_of::<CallbackFailure>(),
            size_of::<CallbackCause>(),
            size_of::<Result<(), CallbackCause>>(),
            size_of::<Result<(), BackendFailure>>(),
            size_of::<Option<&(String, Source)>>(),
            size_of::<std::slice::Iter<'_, (String, Source)>>(),
            size_of::<(&Self, &str, &str, &WorkspaceMetadataFunding)>(),
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
        funding: &WorkspaceMetadataFunding,
    ) -> Result<(), BackendFailure> {
        // The shared channel caller reserves failure_control_bytes first. Even
        // the Source worker's first reserve failure can therefore retain its
        // actual source/input without any new unfunded error transport.
        let result = (|| {
            let data = self
                .source
                .data(&self.recipe)
                .map_err(|_| CallbackCause::Source)?;
            let (_, source) = data
                .rows
                .iter()
                .find(|(actual, _)| actual == name)
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
