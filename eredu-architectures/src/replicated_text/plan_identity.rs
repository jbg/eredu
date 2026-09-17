
//! Validation against actual normalized declarations, with caller-owned metadata.
use super::*;
use crate::decoder::identity::Metadata;
use eredu_runtime::PreparedTextContractError;

pub(super) fn message(
    metadata: Metadata<'_>,
    text: std::fmt::Arguments<'_>,
) -> PreparedTextContractError {
    match metadata.context() {
        Some(context) => PreparedTextContractError::Metadata(context.metadata_error(text)),
        None => PreparedTextContractError::Contract(text.to_string()),
    }
}
fn failure(metadata: Metadata<'_>, cause: eredu_nn::Error) -> PreparedTextContractError {
    match metadata.context() {
        Some(_) => PreparedTextContractError::Metadata(cause),
        None => PreparedTextContractError::Contract(cause.to_string()),
    }
}

pub(super) fn validate(
    requirements: &ReplicatedTextRequirements,
    config: &EligibleConfig<'_>,
    metadata: Metadata<'_>,
) -> Result<(), PreparedTextContractError> {
    metadata
        .controls::<(
            eredu_runtime::ArchitectureExecutionGraph<'_>,
            eredu_runtime::StateLayout,
            usize,
            ReplicatedTextStateAccess,
        )>()
        .map_err(|cause| failure(metadata, cause))?;
    let identity = config
        .identity_for_construction(metadata)
        .map_err(|cause| failure(metadata, cause))?;
    if requirements.architecture_identity() != identity {
        return Err(message(
            metadata,
            format_args!("selected realization belongs to a different normalized architecture"),
        ));
    }
    let graph = eredu_runtime::ArchitectureExecutionGraph::single(config.execution_group())
        .map_err(|cause| message(metadata, format_args!("{cause}")))?;
    let count = config.unit_count_with_diagnostic(|text| message(metadata, text))?;
    let layout = config
        .layout_for_construction(metadata)
        .map_err(|cause| failure(metadata, cause))?;
    if requirements.operators() != config.operators()
        || !graph.matches(requirements.execution_graph())
        || !requirements
            .execution_units()
            .matches_single_group(config.execution_group(), count)
        || requirements.group_transports().len() != 1
        || !config.transport_matches(&requirements.group_transports()[0])
        || requirements.state_layout() != &layout
        || requirements.state_access() != EligibleConfig::state_access_from_layout(&layout)
    {
        return Err(message(
            metadata,
            format_args!("selected realization structure differs from the normalized architecture"),
        ));
    }
    Ok(())
}

impl EligibleConfig<'_> {
    fn identity_for_construction(&self, metadata: Metadata<'_>) -> Result<String, eredu_nn::Error> {
        if metadata.context().is_none() {
            return Ok(self.architecture_identity());
        }
        match self {
            Self::Nanbeige(args) => metadata.configured(*args),
            Self::Gemma2(args) => metadata.configured(*args),
            Self::Llama(args) => metadata.configured(*args),
            Self::K2Horizon(args) => metadata.configured(*args),
            Self::Qwen(args) => metadata.configured(*args),
            Self::Lfm2(args) => {
                crate::lfm2::config::prompt_cache_architecture_fingerprint_with_metadata(
                    args, metadata,
                )
            }
            Self::KimiLinear(args) => {
                crate::kimi_linear::config::prompt_cache_architecture_fingerprint_with_metadata(
                    args, metadata,
                )
            }
            Self::NemotronH(args) => {
                crate::nemotron_h::config::prompt_cache_architecture_fingerprint_with_metadata(
                    args, metadata,
                )
            }
            Self::QwenHybrid(args) => {
                crate::qwen::hybrid::prompt_cache_architecture_fingerprint_with_metadata(
                    args, metadata,
                )
            }
            Self::DeepSeekV3(args) => {
                crate::deepseek::config::v3_architecture_fingerprint_with_metadata(args, metadata)
            }
            // These configurations are excluded by both normalized replicated
            // eligibility selectors before this validator. Preserve their owning
            // diagnostic API for other internal compatibility callers.
            _ => Ok(self.architecture_identity()),
        }
    }

    fn layout_for_construction(
        &self,
        metadata: Metadata<'_>,
    ) -> Result<eredu_runtime::StateLayout, eredu_nn::Error> {
        let Some(context) = metadata.context() else {
            return self
                .state_layout()
                .map_err(eredu_nn::Error::backend_message);
        };
        match self {
            Self::Nanbeige(args) => crate::decoder::state_layout_with_metadata(*args, context),
            Self::Gemma2(args) => crate::decoder::state_layout_with_metadata(*args, context),
            Self::Llama(args) => crate::decoder::state_layout_with_metadata(*args, context),
            Self::K2Horizon(args) => crate::decoder::state_layout_with_metadata(*args, context),
            Self::Qwen(args) => crate::decoder::state_layout_with_metadata(*args, context),
            Self::Lfm2(args) => crate::lfm2::state_layout_with_metadata(args, context),
            Self::KimiLinear(args) => crate::kimi_linear::state_layout_with_metadata(args, context),
            Self::NemotronH(args) => crate::nemotron_h::state_layout_with_metadata(args, context),
            Self::QwenHybrid(args) => {
                crate::qwen::hybrid::state_layout_with_metadata(args, context)
            }
            Self::DeepSeekV3(args) => {
                crate::deepseek::v3::state_layout_with_metadata(args, context)
            }
            _ => self
                .state_layout()
                .map_err(eredu_nn::Error::backend_message),
        }
    }

    fn transport_matches(&self, expected: &eredu_runtime::ArchitectureGroupTransport) -> bool {
        match self {
            Self::DeepSeekV4(_) => self.group_transport() == *expected,
            _ => crate::transport::decoder_declaration().matches(expected),
        }
    }
}
