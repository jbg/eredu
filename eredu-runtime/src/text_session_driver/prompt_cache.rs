//! Prompt-cache transactions on the installed text executable.

use super::*;
use crate::PreparedInputCacheIdentity;
use eredu_core::cache::{
    PromptCacheDescriptor, PromptCacheError, PromptCacheManifest, PromptCacheModelIdentity,
    PromptCacheOptions, PromptCacheStore,
};

impl<R: 'static, N> TextSessionDriver<R, N> {
    /// The identity of the cache state owned by this rank.
    pub fn prompt_cache_model_identity<B>(
        &self,
        backend: &B,
    ) -> Result<PromptCacheModelIdentity, B::Error>
    where
        B: BackendProvider<Error: From<BackendFailure>>,
        N: TextSessionMechanisms<B, Resources = R>,
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

    /// Opens a compatible persisted prefix in one settled model transaction.
    pub fn load_prompt_cache<B>(
        &mut self,
        backend: &B,
        store: &dyn PromptCacheStore,
        expected: &PromptCacheDescriptor,
        prefix: &[u32],
        input: Option<PreparedInputCacheIdentity>,
    ) -> Result<PromptCacheManifest, B::Error>
    where
        B: BackendProvider<Error: From<BackendFailure>>,
        N: TextSessionMechanisms<B, Resources = R> + TextTransactionMechanisms<Error = B::Error>,
        R: DerefMut<Target = TextSessionResourcesFor<B, N>>,
    {
        self.ensure_parameter_cache_compatible()
            .map_err(BackendFailure::from_error)?;
        self.native
            .validate_native_target(&self.payload.retained, backend)?;
        self.with_model_transaction(|model| {
            model
                .as_mut()
                .prompt_cache()
                .ok_or_else(unavailable)?
                .load(store, expected, prefix, input)
                .map_err(Into::into)
        })
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
        N: TextSessionMechanisms<B, Resources = R> + TextTransactionMechanisms<Error = B::Error>,
        R: DerefMut<Target = TextSessionResourcesFor<B, N>>,
    {
        self.ensure_parameter_cache_compatible()
            .map_err(BackendFailure::from_error)?;
        self.native
            .validate_native_target(&self.payload.retained, backend)?;
        self.with_model_transaction(|model| {
            model
                .as_mut()
                .prompt_cache()
                .ok_or_else(unavailable)?
                .save(store, descriptor, prefix, options)
                .map_err(Into::into)
        })
    }
}

fn unavailable() -> BackendFailure {
    BackendFailure::from_error(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "this session does not support prompt-cache persistence",
    ))
}

impl<B, R, N, M> eredu_core::AutomaticPromptCacheSession<B> for TextSessionDriver<R, N>
where
    B: eredu_core::TextGenerationBackend,
    B::Error: From<eredu_core::BackendError> + From<BackendFailure>,
    crate::text_session_driver::TextPromptFor<B>: TextPrefillInput,
    N: TextSessionMechanisms<B, Resources = R, Model = M>
        + TextTransactionMechanisms<Error = B::Error>,
    R: DerefMut<Target = TextSessionResourcesFor<B, N>> + 'static,
    M: InstalledTextState,
    Self: eredu_core::backend::TextOperationDriver<
        B,
        Prompt = crate::text_session_driver::TextPromptFor<B>,
    >,
{
    fn prompt_cache_model_identity(
        runtime: &eredu_core::ModelRuntime<B, Self>,
    ) -> Result<PromptCacheModelIdentity, BackendFailure> {
        let session = runtime.session();
        if !matches!(
            session.state_layout().residency(),
            crate::CacheResidencyPolicy::Paged(_)
        ) {
            return Err(BackendFailure::from_error(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "prompt-cache import/export requires a plan with paged execution state",
            )));
        }
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
        Ok(input.prompt_cache_identity().map_or_else(
            || eredu_core::cache::prompt_cache_token_fingerprint(prefix),
            |identity| identity.prefix_content_fingerprint().to_owned(),
        ))
    }

    fn estimate_prompt_cache_state(
        runtime: &eredu_core::ModelRuntime<B, Self>,
    ) -> Result<Option<u64>, BackendFailure> {
        use eredu_core::execution_control::NativeTextStateSession as _;
        Self::estimate_native_text_state(runtime, None)
            .map(|estimate| estimate.map(|e| e.retained_bytes.max(e.copy_bytes)))
            .map_err(BackendFailure::from_error)
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

    fn load_prompt_cache(
        runtime: &mut eredu_core::ModelRuntime<B, Self>,
        store: &dyn PromptCacheStore,
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
            .load_prompt_cache(
                backend,
                store,
                descriptor,
                prefix,
                input.prompt_cache_identity().cloned(),
            )
            .map_err(BackendFailure::from_error)
    }
}
