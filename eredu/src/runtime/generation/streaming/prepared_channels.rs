//! Facade declaration projection; runtime owns the paid mutable channel worker.
use crate::runtime::chat::{SemanticRuntimePlan, dialect};
use eredu_core::{HostPreparationAuthority, SpeculativeBuffer, generation::SemanticEvent};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use eredu_runtime::working_memory::{
    OriginalForbiddenSource, OriginalSemanticControllerSource, OriginalSemanticChannelParser, OriginalSemanticChannelParserError,
    OriginalSemanticChannelSource, OriginalSemanticChannelSourceError, OriginalTokenizer,
};
use std::mem::{size_of, size_of_val};
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    ToolSchema(#[from] crate::runtime::chat::tool_schema::registered::PreparationFailure),
    #[error(transparent)]
    Declaration(#[from] dialect::DeclarationError),
    #[error(transparent)]
    Source(#[from] OriginalSemanticChannelSourceError),
    #[error(transparent)]
    Parser(#[from] OriginalSemanticChannelParserError),
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
    #[error(transparent)]
    Buffer(#[from] eredu_core::SpeculativeBufferAllocationError),
    #[error("prepared semantic declaration extent overflow")]
    Overflow,
    #[error("prepared semantic declaration population changed")]
    Population,
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub(crate) struct Failure {
    #[source]
    cause: Cause,
    funding: HostMetadataFunding,
}
/// Borrow the actual recipe under paid reference storage, then copy into the
/// existing original source compiler. No ordinary parser or tool map is built.
pub(crate) fn compile_source(
    plan: &SemanticRuntimePlan,
    tokenizer: &OriginalTokenizer,
    source: &OriginalForbiddenSource,
    funding: &HostMetadataFunding,
) -> Result<OriginalSemanticChannelSource, Failure> {
    compile_source_for_controller(plan, tokenizer, OriginalSemanticControllerSource::Forbidden(source), funding)
}
/// Both prepared controller kinds consume the same structural/literal compiler.
pub(crate) fn compile_source_for_controller(
    plan: &SemanticRuntimePlan,
    tokenizer: &OriginalTokenizer,
    source: OriginalSemanticControllerSource<'_>,
    funding: &HostMetadataFunding,
) -> Result<OriginalSemanticChannelSource, Failure> {
    let retain = |cause| Failure {
        cause,
        funding: funding.clone(),
    };
    let result = (|| -> Result<OriginalSemanticChannelSource, Cause> {
        let count = plan.original_structural_tokens().len();
        let parts = [
            dialect::channels::control_bytes().ok_or(Cause::Overflow)?,
            dialect::ProfileDeclaration::control_bytes().ok_or(Cause::Overflow)?,
            size_of::<(
                &SemanticRuntimePlan,
                &OriginalTokenizer,
                OriginalSemanticControllerSource<'_>,
                &HostMetadataFunding,
            )>(),
            size_of::<Result<OriginalSemanticChannelSource, Failure>>(),
            size_of::<Result<OriginalSemanticChannelSource, OriginalSemanticChannelSourceError>>(),
            size_of::<Result<&'static dialect::DeclarativeDialectSpec, dialect::DeclarationError>>(
            ),
            size_of::<
                Result<
                    SpeculativeBuffer<(u32, &str, bool)>,
                    eredu_core::SpeculativeBufferAllocationError,
                >,
            >(),
            size_of::<Result<(), eredu_core::generation::GenerationError>>(),
            size_of::<Failure>(),
            size_of::<Cause>(),
            size_of::<PreparedChannelParser>(),
            size_of::<Result<PreparedChannelParser, Failure>>(),
            SpeculativeBuffer::<(u32, &str, bool)>::retained_control_bytes(count)
                .ok_or(Cause::Overflow)?,
            HostPreparationAuthority::retention_bytes::<HostMetadataFunding>()
                .ok_or(Cause::Overflow)?,
        ];
        funding.reserve_metadata(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or(Cause::Overflow)?,
        )?;
        let spec = match source {
            OriginalSemanticControllerSource::Forbidden(source) => plan.original_forbidden_channel_program(source)?,
            OriginalSemanticControllerSource::Grammar(source) => plan.original_grammar_channel_program(source)?,
        };
        let mut rows = SpeculativeBuffer::try_new_retained(
            count,
            HostPreparationAuthority::retain(funding.clone()),
        )?;
        rows.try_extend(plan.original_structural_tokens())
            .map_err(|_| Cause::Population)?;
        let validation = if matches!(source, OriginalSemanticControllerSource::Grammar(_)) && dialect::channels::tools(spec).is_some() {
            plan.original_tool_validation(funding)?
        } else { None };
        Ok(tokenizer.compile_semantic_channel_source_with_validation(
            source,
            dialect::channels::program(spec),
            &rows,
            dialect::channels::tools(spec),
            validation,
        )?)
    })();
    result.map_err(retain)
}
pub(crate) struct PreparedChannelParser {
    inner: OriginalSemanticChannelParser,
    funding: HostMetadataFunding,
}
impl PreparedChannelParser {
    pub(crate) fn prepare(
        plan: &SemanticRuntimePlan,
        tokenizer: &OriginalTokenizer,
        source: &OriginalForbiddenSource,
        input_bytes: usize,
        funding: &HostMetadataFunding,
    ) -> Result<Self, Failure> {
        let source = compile_source(plan, tokenizer, source, funding)?;
        let inner = OriginalSemanticChannelParser::prepare(&source, input_bytes, funding).map_err(
            |cause| Failure {
                cause: cause.into(),
                funding: funding.clone(),
            },
        )?;
        Ok(Self {
            inner,
            funding: funding.clone(),
        })
    }
    fn failure(&self, cause: OriginalSemanticChannelParserError) -> Failure {
        Failure {
            cause: cause.into(),
            funding: self.funding.clone(),
        }
    }
    pub(crate) fn push(&mut self, text: &str) -> Result<(), Failure> {
        self.inner.push(text).map_err(|cause| self.failure(cause))
    }
    pub(crate) fn finish(&mut self) -> Result<(), Failure> {
        self.inner.finish().map_err(|cause| self.failure(cause))
    }
    pub(crate) fn cancel(&mut self) {
        self.inner.cancel();
    }
    pub(crate) fn copy(&self, funding: &HostMetadataFunding) -> Result<Self, Failure> {
        funding
            .reserve_metadata(size_of::<Self>() + size_of::<Result<Self, Failure>>())
            .map_err(|cause| Failure {
                cause: cause.into(),
                funding: funding.clone(),
            })?;
        let inner = self
            .inner
            .copy(funding)
            .map_err(|cause| self.failure(cause))?;
        Ok(Self {
            inner,
            funding: funding.clone(),
        })
    }
    pub(crate) fn take_events(&mut self) -> Result<SpeculativeBuffer<SemanticEvent>, Failure> {
        self.inner
            .take_events()
            .map_err(|cause| self.failure(cause))
    }
}
