//! Portable cache/state policy and prompt-cache schemas.
//!
//! Runtime ownership, admission, storage transitions, and backing-store
//! scheduling live in `eredu-runtime` so this crate remains declarative.

mod policy;
mod prompt;

pub use policy::{
    CacheBlockId, CachePolicyError, CacheRankIdentity, CacheRepresentation, CacheTier,
    LayerCachePolicy, MutableStateResidency, PoolingStateComponent, StateComponentPolicy,
    StateComponentRole, StateResidencyClass, StateTensorDimension, StateTensorDtype,
    StateTensorOwner, StateTensorPolicy, StateTensorPresence, StateTensorRole,
};
pub use prompt::{
    derive_prompt_cache_architecture_fingerprint, prompt_cache_token_fingerprint,
    validate_prompt_cache_model_identity, PromptCacheBlock, PromptCacheDescriptor,
    PromptCacheError, PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
    PromptCacheStateSegment, PromptCacheStateTensor, PromptCacheTopology,
    PROMPT_CACHE_SCHEMA_VERSION,
};
