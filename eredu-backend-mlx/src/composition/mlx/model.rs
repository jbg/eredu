//! Architecture-erased executable and generation dispatch.

use std::path::Path;

use eredu_core::cache::{PromptCacheDescriptor, PromptCacheManifest, PromptCacheOptions};
use eredu_core::{SpeculativeCapability, SpeculativeDraftSource};
use eredu_runtime::{CacheResidencyPolicy, CacheResidencyReport, PagedCacheOptions};
use safemlx::{error::Exception, Array, Stream};

use crate::backend::error::Error;
use crate::backend::runtime::media::input;

/// The single architecture-erased outer boundary for complete MLX execution.
pub(crate) struct Executable {
    inner: Box<dyn super::replicated_text::ErasedReplicatedTextExecutable>,
}

impl Executable {
    pub(super) fn new(
        inner: Box<dyn super::replicated_text::ErasedReplicatedTextExecutable>,
    ) -> Self {
        Self { inner }
    }

    pub(crate) fn erased(
        &self,
    ) -> &(dyn super::replicated_text::ErasedReplicatedTextExecutable + '_) {
        self.inner.as_ref()
    }

    pub(crate) fn erased_mut(
        &mut self,
    ) -> &mut (dyn super::replicated_text::ErasedReplicatedTextExecutable + '_) {
        self.inner.as_mut()
    }

    pub(crate) fn install_embedded_prediction_observers(
        &mut self,
        observers: eredu_architectures::speculative_execution::EmbeddedPredictionObservers<
            crate::MlxTensor,
            Array,
            Exception,
        >,
    ) -> bool {
        self.erased_mut()
            .install_embedded_prediction_observers(observers)
    }

    pub(crate) fn has_neutral_partitioned_control(&self) -> bool {
        self.erased().has_partition_control()
    }

    pub(crate) fn reset_cache_distributed(&mut self) -> Result<(), Exception> {
        self.erased_mut()
            .reset_cache_distributed()
            .map_err(|error| Exception::custom(error.to_string()))
    }

    pub(crate) fn load_prompt_cache_distributed(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
    ) -> Result<Option<PromptCacheManifest>, Exception> {
        self.erased_mut()
            .load_prompt_cache_distributed(directory, expected, prefix_token_ids)
            .map_err(|error| Exception::custom(error.to_string()))
    }

    pub(crate) fn load_prompt_cache_for_input_distributed(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        input_identity: eredu_runtime::PreparedInputCacheIdentity,
    ) -> Result<Option<PromptCacheManifest>, Exception> {
        self.erased_mut()
            .load_prompt_cache_for_input_distributed(
                directory,
                expected,
                prefix_token_ids,
                input_identity,
            )
            .map_err(|error| Exception::custom(error.to_string()))
    }

    pub(crate) fn save_prompt_cache_distributed(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<Option<PromptCacheManifest>, Exception> {
        self.erased_mut()
            .save_prompt_cache_distributed(destination, descriptor, prefix_token_ids, options)
            .map_err(|error| Exception::custom(error.to_string()))
    }

    pub fn speculative_capability(&self) -> SpeculativeCapability {
        if self.erased().has_embedded_prediction() {
            return SpeculativeCapability::Ready {
                draft_source: SpeculativeDraftSource::Embedded,
            };
        }
        match self
            .erased()
            .capability_estimate()
            .speculative_draft_source()
        {
            Some(draft_source @ SpeculativeDraftSource::Separate) => {
                SpeculativeCapability::Declared { draft_source }
            }
            Some(draft_source) => SpeculativeCapability::Unsupported {
                draft_source,
                architecture: self.effective_model_type().to_owned(),
            },
            None => SpeculativeCapability::Unavailable,
        }
    }

    pub fn residency_report(&self) -> Result<Option<eredu_runtime::ResidencyReport>, Error> {
        self.erased().residency_report()
    }

    pub fn dense_stream_report(
        &self,
    ) -> Result<Option<eredu_runtime::DenseDiskStreamReport>, Error> {
        self.erased().dense_stream_report()
    }

    pub fn materialization_report(&self) -> Option<&eredu_runtime::WeightMaterializationReport> {
        self.erased().materialization_report()
    }

    pub fn parameter_bank_report(
        &self,
    ) -> Result<
        Option<crate::backend::runtime::residency::parameter_bank::ParameterBankResidencyReport>,
        Error,
    > {
        self.erased().parameter_bank_report()
    }

    pub fn effective_model_type(&self) -> &str {
        self.erased().effective_model_type()
    }

    pub fn prompt_cache_model_identity(
        &self,
    ) -> Result<eredu_core::cache::PromptCacheModelIdentity, Exception> {
        Ok(self.erased().prompt_cache_model_identity().clone())
    }

    pub fn reset_cache_with_options(
        &mut self,
        _policy: CacheResidencyPolicy,
    ) -> Result<(), Exception> {
        self.reset_cache()
    }

    pub fn reset_cache(&mut self) -> Result<(), Exception> {
        self.erased_mut().reset_cache()
    }

    pub fn load_prompt_cache(
        &mut self,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        _options: PagedCacheOptions,
        _stream: &Stream,
    ) -> Result<PromptCacheManifest, Exception> {
        self.erased_mut()
            .load_prompt_cache(directory.as_ref(), expected, prefix_token_ids)
            .map_err(|error| Exception::custom(error.to_string()))
    }

    pub fn load_prompt_cache_for_input(
        &mut self,
        directory: impl AsRef<Path>,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        input_identity: eredu_runtime::PreparedInputCacheIdentity,
        _options: PagedCacheOptions,
        _stream: &Stream,
    ) -> Result<PromptCacheManifest, Exception> {
        self.erased_mut()
            .load_prompt_cache_for_input(
                directory.as_ref(),
                expected,
                prefix_token_ids,
                input_identity,
            )
            .map_err(|error| Exception::custom(error.to_string()))
    }

    pub fn save_prompt_cache(
        &mut self,
        destination: impl AsRef<Path>,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        _stream: &Stream,
    ) -> Result<PromptCacheManifest, Exception> {
        self.erased_mut()
            .save_prompt_cache(destination.as_ref(), descriptor, prefix_token_ids, options)
            .map_err(|error| Exception::custom(error.to_string()))
    }

    pub fn cache_residency_report(&self) -> Result<Option<CacheResidencyReport>, Exception> {
        self.erased().cache_residency_report()
    }

    pub(crate) fn prefill(
        &mut self,
        input: input::ModelInput<'_>,
        stream: &Stream,
    ) -> Result<Array, Error> {
        self.erased_mut().prefill(input, stream)
    }

    pub(crate) fn decode(&mut self, tokens: &Array, stream: &Stream) -> Result<Array, Error> {
        self.erased_mut().decode(tokens, stream)
    }
}
