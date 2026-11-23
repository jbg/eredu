//! Prompt-cache transactions on the installed text executable.

use super::*;
use crate::PreparedInputCacheIdentity;
use eredu_core::cache::{
    PromptCacheDescriptor, PromptCacheError, PromptCacheManifest, PromptCacheModelIdentity,
    PromptCacheOptions, PromptCacheRead, PromptCacheStore, PromptCacheStoreError,
};

impl<R: 'static, N, G> TextSessionDriver<R, N, G> {
    /// The identity of the cache state owned by this rank.
    pub fn prompt_cache_model_identity<B>(
        &self,
        backend: &B,
    ) -> Result<PromptCacheModelIdentity, B::Error>
    where
        B: BackendProvider<Error: From<BackendFailure>>,
        N: TextSessionMechanisms<B, Resources = R, Generation = G>,
        R: DerefMut<Target = TextSessionResourcesFor<B, N>>,
    {
        self.ensure_parameter_cache_compatible()
            .map_err(BackendFailure::from_error)?;
        self.native
            .validate_native_target(&self.payload.retained, backend)?;
        self.payload
            .model
            .as_ref()
            .prompt_cache_model_identity()
            .cloned()
            .ok_or_else(|| BackendFailure::from_error(PromptCacheError::StatelessPartition).into())
    }

    /// Restores a pinned persisted prefix in one settled model transaction.
    pub fn restore_prompt_cache<B>(
        &mut self,
        backend: &B,
        read: &dyn PromptCacheRead,
        expected: &PromptCacheDescriptor,
        prefix: &[u32],
        input: Option<PreparedInputCacheIdentity>,
    ) -> Result<PromptCacheManifest, B::Error>
    where
        B: BackendProvider<Error: From<BackendFailure>>,
        N: TextSessionMechanisms<B, Resources = R, Generation = G>
            + TextTransactionMechanisms<Error = B::Error>,
        R: DerefMut<Target = TextSessionResourcesFor<B, N>>,
    {
        self.ensure_parameter_cache_compatible()
            .map_err(BackendFailure::from_error)?;
        self.native
            .validate_native_target(&self.payload.retained, backend)?;
        self.with_model_transaction(|model| {
            cache_transaction_outcome(
                model
                    .as_mut()
                    .prompt_cache(N::context(backend))
                    .ok_or_else(unavailable)
                    .and_then(|cache| cache.restore(read, expected, prefix, input)),
            )
        })?
        .map_err(Into::into)
    }

    /// Persists the committed prefix in one settled model transaction.
    pub fn save_prompt_cache<B>(
        &mut self,
        backend: &B,
        store: &dyn PromptCacheStore,
        descriptor: PromptCacheDescriptor,
        prefix: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, B::Error>
    where
        B: BackendProvider<Error: From<BackendFailure>>,
        N: TextSessionMechanisms<B, Resources = R, Generation = G>
            + TextTransactionMechanisms<Error = B::Error>,
        R: DerefMut<Target = TextSessionResourcesFor<B, N>>,
    {
        self.ensure_parameter_cache_compatible()
            .map_err(BackendFailure::from_error)?;
        self.native
            .validate_native_target(&self.payload.retained, backend)?;
        self.with_model_transaction(|model| {
            cache_transaction_outcome(
                model
                    .as_mut()
                    .prompt_cache(N::context(backend))
                    .ok_or_else(unavailable)
                    .and_then(|cache| cache.save(store, descriptor, prefix, options)),
            )
        })?
        .map_err(Into::into)
    }
}

// A rejected cache publication or provisional restore can preserve the live
// executable. Settle the native scope before exposing that rejection.
fn cache_transaction_outcome<T, E: From<BackendFailure>>(
    result: Result<T, BackendFailure>,
) -> Result<Result<T, BackendFailure>, E> {
    match result {
        Err(error) if PromptCacheStoreError::publication_declined(&error) => {
            Ok(Err(error.with_model_state_preserved(true)))
        }
        Err(error) if !error.model_state_preserved() => Err(error.into()),
        outcome => Ok(outcome),
    }
}

fn unavailable() -> BackendFailure {
    BackendFailure::from_error(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "this session does not support prompt-cache persistence",
    ))
    .with_model_state_preserved(true)
}

impl<B, R, N, M, G> eredu_core::AutomaticPromptCacheSession<B> for TextSessionDriver<R, N, G>
where
    B: eredu_core::TextGenerationBackend,
    B::Error: From<BackendFailure>,
    crate::text_session_driver::TextPromptFor<B>: TextPrefillInput,
    N: TextSessionMechanisms<B, Resources = R, Model = M, Generation = G>
        + TextTransactionMechanisms<Error = B::Error>,
    R: DerefMut<Target = TextSessionResourcesFor<B, N>> + 'static,
    M: InstalledTextState + AsRef<N::Executable>,
    Self: eredu_core::backend::TextOperationDriver<
        B,
        Prompt = crate::text_session_driver::TextPromptFor<B>,
    >,
{
    fn prompt_cache_model_identity(
        runtime: &eredu_core::ModelRuntime<B, Self>,
    ) -> Result<PromptCacheModelIdentity, BackendFailure> {
        let session = runtime.session();
        if N::control_transport(&session.payload.retained).is_some() {
            return Err(BackendFailure::from_error(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "automatic prompt-cache reuse requires process-local cache-hit and prefill admission",
            )));
        }
        session
            .prompt_cache_model_identity(runtime.backend())
            .map_err(BackendFailure::from_error)
    }

    fn prompt_cache_prefix_fingerprint(
        runtime: &eredu_core::ModelRuntime<B, Self>,
        prefix: &[u32],
    ) -> Result<String, BackendFailure> {
        let input = <Self as eredu_core::backend::TextOperationDriver<B>>::prepare_text_prompt(
            runtime.backend(),
            prefix.to_vec(),
        )
        .map_err(BackendFailure::from_error)?;
        Ok(runtime
            .session()
            .payload
            .model
            .as_ref()
            .prompt_cache_prefix_identity(prefix, input.prompt_cache_identity()))
    }

    fn estimate_prompt_cache_state(
        runtime: &eredu_core::ModelRuntime<B, Self>,
    ) -> Result<Option<u64>, BackendFailure> {
        let session = runtime.session();
        session
            .native
            .validate_native_target(&session.payload.retained, runtime.backend())
            .map_err(BackendFailure::from_error)?;
        session
            .submission_idle()
            .map_err(N::submission_authority_error)
            .map_err(BackendFailure::from_error)?;
        let model = &session.payload.model;
        if let Some(bytes) = model.as_ref().prompt_cache_storage_bytes()? {
            return Ok(Some(bytes));
        }
        if let eredu_core::execution_control::ControlSupport::Supported =
            super::erased_state::state_support(model)
        {
            return super::erased_state::supported_state(model)?
                .estimate_copy(None)
                .map(|estimate| {
                    estimate
                        .and_then(super::erased_state::ErasedTextState::include_slot_cost)
                        .map(|e| e.retained_bytes.max(e.copy_bytes))
                });
        }
        Ok(None)
    }

    fn save_prompt_cache(
        runtime: &mut eredu_core::ModelRuntime<B, Self>,
        store: &dyn PromptCacheStore,
        descriptor: PromptCacheDescriptor,
        prefix: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, BackendFailure> {
        let (backend, session) = runtime.parts_mut();
        session
            .save_prompt_cache(backend, store, descriptor, prefix, options)
            .map_err(BackendFailure::from_error)
    }

    fn restore_prompt_cache(
        runtime: &mut eredu_core::ModelRuntime<B, Self>,
        read: &dyn PromptCacheRead,
        descriptor: &PromptCacheDescriptor,
        prefix: &[u32],
    ) -> Result<PromptCacheManifest, BackendFailure> {
        let input = <Self as eredu_core::backend::TextOperationDriver<B>>::prepare_text_prompt(
            runtime.backend(),
            prefix.to_vec(),
        )
        .map_err(BackendFailure::from_error)?;
        let (backend, session) = runtime.parts_mut();
        session
            .restore_prompt_cache(
                backend,
                read,
                descriptor,
                prefix,
                input.prompt_cache_identity().cloned(),
            )
            .map_err(BackendFailure::from_error)
    }
}
