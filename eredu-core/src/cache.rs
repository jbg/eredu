//! Portable cache/state policy and prompt-cache schemas.
//!
//! Runtime ownership, admission, storage transitions, and backing-store
//! scheduling live in `eredu-runtime` so this crate remains declarative.

mod policy;
mod prompt;
mod shared_manifest;
pub use shared_manifest::{PreparedPromptCacheManifest, SharedPromptCacheManifest};

pub use policy::{
    CacheBlockId, CachePolicyError, CacheRankIdentity, CacheRepresentation, CacheTier,
    LayerCachePolicy, MutableStateResidency, PoolingStateComponent, StateComponentPolicy,
    StateComponentRole, StateResidencyClass, StateTensorDimension, StateTensorDimensionError,
    StateTensorDtype, StateTensorOwner, StateTensorPolicy, StateTensorPresence, StateTensorRole,
};
pub use prompt::{
    PROMPT_CACHE_SCHEMA_VERSION, PromptCacheArchitectureFingerprint, PromptCacheBlock,
    PromptCacheDescriptor, PromptCacheDiagnosticKind, PromptCacheError, PromptCacheManifest,
    PromptCacheModelIdentity, PromptCacheOptions, PromptCacheStateSegment, PromptCacheStateTensor,
    PromptCacheTopology, derive_prompt_cache_architecture_fingerprint,
    prompt_cache_token_fingerprint, prompt_cache_token_fingerprint_control_bytes,
    prompt_cache_token_fingerprint_into, validate_prompt_cache_model_identity,
};
