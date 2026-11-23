//! Store and input-identity policy for persisted prefixes of a bound text session.

use super::*;
use crate::cache::live::RestoreCache;
use eredu_core::{
    cache::{PromptCacheRead, PromptCacheStore},
    BackendFailure,
};

pub(super) fn empty_store() -> PromptCacheError {
    PromptCacheError::Incompatible("prompt cache store is empty".into())
}

/// The input identity required to persist or restore a selected prefix.
#[derive(Clone, Copy)]
pub enum PromptCacheInput {
    /// Token IDs identify a local text-only prefix.
    Tokens,
    /// Prepared-input identity also binds the semantic contents of composite input.
    PreparedInput,
}

/// Borrowed access to the portable persistence driver for an erased text session.
pub struct TextPromptCache<'a, C: ?Sized> {
    session: &'a mut dyn PromptCacheSession<C>,
    context: &'a C,
    input: PromptCacheInput,
}

impl<C: ?Sized> TextPromptCache<'_, C> {
    /// Restores the prefix, validating the prepared-input identity when supplied.
    pub fn restore(
        self,
        read: &dyn PromptCacheRead,
        expected: &PromptCacheDescriptor,
        prefix: &[u32],
        input: Option<PreparedInputCacheIdentity>,
    ) -> Result<PromptCacheManifest, BackendFailure> {
        self.session
            .restore_prefix(read, expected, prefix, self.input, input, self.context)
    }

    /// Persists the prefix under the store budget and the session input contract.
    pub fn save(
        self,
        store: &dyn PromptCacheStore,
        descriptor: PromptCacheDescriptor,
        prefix: &[u32],
        options: &PromptCacheOptions,
    ) -> Result<PromptCacheManifest, BackendFailure> {
        self.session
            .persist_prefix(store, descriptor, prefix, options, self.input, self.context)
    }
}

pub(crate) trait PromptCacheSession<C: ?Sized> {
    fn restore_prefix(
        &mut self,
        read: &dyn PromptCacheRead,
        expected: &PromptCacheDescriptor,
        prefix: &[u32],
        requirement: PromptCacheInput,
        input: Option<PreparedInputCacheIdentity>,
        context: &C,
    ) -> Result<PromptCacheManifest, BackendFailure>;

    fn persist_prefix(
        &mut self,
        store: &dyn PromptCacheStore,
        descriptor: PromptCacheDescriptor,
        prefix: &[u32],
        options: &PromptCacheOptions,
        requirement: PromptCacheInput,
        context: &C,
    ) -> Result<PromptCacheManifest, BackendFailure>;
}

#[derive(Debug, thiserror::Error)]
enum PersistenceError {
    #[error("process-local memory stores do not provide all-rank atomic prompt-cache control")]
    LocalStore,
    #[error("prompt-cache prefix differs from committed token input")]
    TokenPrefix,
    #[error("composite prompt-cache loading requires the prepared-input identity")]
    MissingInput,
    #[error("composite prompt cache requires a committed prepared-input identity")]
    MissingCommittedInput,
}

impl<A, B, M, D> ReplicatedTextSession<A, B, M, D>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    A::Error: std::error::Error + Send + Sync + 'static,
    M::PolicyError: std::error::Error + Send + Sync + 'static,
    M::Error: std::error::Error + Send + Sync + 'static,
{
    /// Uses the selected model's input contract and the retained session topology.
    pub fn prompt_cache<'a>(
        &'a mut self,
        context: &'a B::Executor,
        input: PromptCacheInput,
    ) -> TextPromptCache<'a, B::Executor> {
        TextPromptCache {
            session: self,
            context,
            input,
        }
    }
}

impl<A, B, M, D> PromptCacheSession<B::Executor> for ReplicatedTextSession<A, B, M, D>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    A::Error: std::error::Error + Send + Sync + 'static,
    M::PolicyError: std::error::Error + Send + Sync + 'static,
    M::Error: std::error::Error + Send + Sync + 'static,
{
    fn restore_prefix(
        &mut self,
        read: &dyn PromptCacheRead,
        expected: &PromptCacheDescriptor,
        prefix: &[u32],
        requirement: PromptCacheInput,
        input: Option<PreparedInputCacheIdentity>,
        context: &B::Executor,
    ) -> Result<PromptCacheManifest, BackendFailure> {
        let preserved = |error: PersistenceError| {
            BackendFailure::new(eredu_core::BackendFailureKind::InvalidInput, error)
                .with_model_state_preserved(true)
        };
        if D::PARTITIONED_SESSION {
            return Err(preserved(PersistenceError::LocalStore));
        }
        match requirement {
            PromptCacheInput::Tokens => {
                let fingerprint = eredu_core::cache::prompt_cache_token_fingerprint(prefix);
                if expected.prefix_content_fingerprint() != fingerprint
                    || input
                        .as_ref()
                        .is_some_and(|value| value.semantic_content_fingerprint() != fingerprint)
                {
                    return Err(preserved(PersistenceError::TokenPrefix));
                }
            }
            PromptCacheInput::PreparedInput if input.is_none() => {
                return Err(preserved(PersistenceError::MissingInput));
            }
            PromptCacheInput::PreparedInput => {}
        }
        self.restore_prompt_cache(
            read,
            expected,
            prefix,
            input.filter(|_| matches!(requirement, PromptCacheInput::PreparedInput)),
            context,
        )
        .map_err(|error| BackendFailure::from_error(error).with_model_state_preserved(true))
    }

    fn persist_prefix(
        &mut self,
        store: &dyn PromptCacheStore,
        descriptor: PromptCacheDescriptor,
        prefix: &[u32],
        options: &PromptCacheOptions,
        requirement: PromptCacheInput,
        context: &B::Executor,
    ) -> Result<PromptCacheManifest, BackendFailure> {
        if matches!(requirement, PromptCacheInput::Tokens) && !D::PARTITIONED_SESSION {
            let fingerprint = eredu_core::cache::prompt_cache_token_fingerprint(prefix);
            if descriptor.prefix_content_fingerprint() != fingerprint
                || self
                    .local_token_prefix
                    .as_ref()
                    .is_none_or(|committed| committed.fingerprint() != fingerprint)
            {
                return Err(BackendFailure::new(
                    eredu_core::BackendFailureKind::InvalidInput,
                    PersistenceError::TokenPrefix,
                )
                .with_model_state_preserved(true));
            }
        }
        let options = options.clone().with_capture_limit(store.max_bytes());
        let input = match requirement {
            PromptCacheInput::Tokens => None,
            PromptCacheInput::PreparedInput => Some(
                self.committed_prompt_input_identity()
                    .cloned()
                    .ok_or_else(|| {
                        BackendFailure::from_error(PersistenceError::MissingCommittedInput)
                    })?,
            ),
        };
        if D::PARTITIONED_SESSION {
            let directory = store
                .directory()
                .ok_or_else(|| BackendFailure::from_error(PersistenceError::LocalStore))?;
            return self
                .save_prompt_cache_distributed_inner(
                    directory,
                    descriptor,
                    prefix,
                    &options,
                    input.as_ref(),
                    context,
                )
                .map_err(BackendFailure::from_error)?
                .ok_or_else(|| BackendFailure::from_error(PromptCacheError::StatelessPartition));
        }
        match input.as_ref() {
            Some(identity) => self.save_prompt_cache_for_input(
                store, descriptor, prefix, &options, identity, context,
            ),
            None => self.save_prompt_cache(store, descriptor, prefix, &options, context),
        }
        .map_err(BackendFailure::from_error)
    }
}

pub(super) fn restore_cached_state<A, B, M>(
    mechanisms: &mut M,
    read: &dyn PromptCacheRead,
    expected: &PromptCacheDescriptor,
    identity: &PromptCacheModelIdentity,
    prefix: &[u32],
    selected: &SelectedStateRealization,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<
    (M::State, PromptCacheManifest),
    ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    A::Error: std::fmt::Display,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    let paged = selected
        .paging_options()
        .map_err(|error| ReplicatedTextSessionError::Contract(error.to_string()))?;
    let allocation = selected
        .allocation()
        .map_err(|error| ReplicatedTextSessionError::Contract(error.to_string()))?;
    let restore = crate::PromptCacheRestore::new(
        read,
        expected,
        identity,
        prefix,
        paged.map_or(
            read.manifest().block_size_tokens,
            crate::PagedCacheOptions::block_size_tokens,
        ),
    )?;
    let manifest = restore.manifest();
    let prefix_len = i32::try_from(manifest.total_prefix_tokens)
        .map_err(|_| PromptCacheError::Malformed("prompt-cache prefix exceeds i32".into()))?;
    let layer_frontiers = expected.layer_frontiers(manifest.total_prefix_tokens)?;
    let Some(options) = paged else {
        if expected.topology().cache_rank_identity().is_some() {
            return Err(ReplicatedTextSessionError::Contract(
                "resident prompt cache requires local state".into(),
            ));
        }
        let mut state = mechanisms
            .allocate_state(&allocation, None, None, 0, context)
            .map_err(ReplicatedTextSessionError::Mechanism)?;
        mechanisms
            .install_resident_cache(&mut state, &restore, &layer_frontiers, context)
            .map_err(ReplicatedTextSessionError::Mechanism)?;
        return Ok((state, manifest.clone()));
    };
    let mut target =
        M::CacheRestore::begin(options, context).map_err(ReplicatedTextSessionError::LiveCache)?;
    restore.restore_blocks(&mut target)?;
    restore.restore_states(&mut target)?;
    let restored = target
        .complete(manifest)
        .map_err(ReplicatedTextSessionError::LiveCache)?;
    let state = mechanisms
        .install_restored_cache(
            restored,
            manifest.topology.cache_rank_identity(),
            manifest.global_layer_start,
            prefix_len,
            &layer_frontiers,
            &allocation,
        )
        .map_err(ReplicatedTextSessionError::Mechanism)?;
    Ok((state, manifest.clone()))
}

pub(super) fn publish_cached_state<A, B, M>(
    mechanisms: &mut M,
    state: &mut M::State,
    store: &dyn PromptCacheStore,
    descriptor: PromptCacheDescriptor,
    prefix: &[u32],
    options: &PromptCacheOptions,
) -> Result<PromptCacheManifest, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    A::Error: std::fmt::Display,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    let layer_frontiers = descriptor.layer_frontiers(prefix.len())?;
    let inventory = mechanisms
        .cache_export(state, &descriptor, &layer_frontiers)
        .map_err(ReplicatedTextSessionError::Mechanism)?;
    Ok(crate::publish_prompt_cache(
        &inventory, store, descriptor, prefix, options,
    )?)
}
