//! Complete HF tokenizer recipe inputs, independent of compiled grammar state.
//!
//! These helpers require existing host preparation authority. They do not
//! register the returned bytes or establish a finite parser/cache bound.

use eredu_core::HostPreparationAuthority;
use eredu_text::tokenizer::{ModelCachePolicy, Tokenizer as ChatTokenizer, token_id_vocabulary};
use llguidance::toktrie::TokEnv;
use serde::{Deserialize, Serialize, de::IgnoredAny};
use tokenizers::Tokenizer;

const RECIPE_VERSION: u32 = 3;

#[derive(Serialize)]
struct RecipeRef<'a> {
    version: u32,
    tokenizer: &'a Tokenizer,
    cache_policy: ModelCachePolicy,
    // Upstream Tokenizer serde omits this behavior-changing option.
    encode_special_tokens: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RecipeHeader {
    version: u32,
    tokenizer: IgnoredAny,
    cache_policy: ModelCachePolicy,
    encode_special_tokens: bool,
}

/// Temporary preparation products. The caller packs and registers `bytes`
/// while retaining its outer authority; only the verified restored environment
/// should be used for eager grammar validation.
pub(crate) struct FrozenTokenizer {
    pub(crate) bytes: Vec<u8>,
    pub(crate) grammar: FrozenGrammarTokenizer,
    pub(crate) environment: TokEnv,
}

/// Source of the actual tokenizer after the shared grammar prefix policy. Only
/// `freeze` can create it; consumers cannot substitute a normalized JSON object.
pub(crate) struct FrozenGrammarTokenizer {
    bytes: Vec<u8>,
    object: std::ops::Range<usize>,
    encode_special_tokens: bool,
    canonical: bool,
}
impl FrozenGrammarTokenizer {
    pub(crate) fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    pub(crate) fn object_range(&self) -> std::ops::Range<usize> {
        self.object.clone()
    }
    pub(crate) fn canonical(&self) -> bool {
        self.canonical
    }
    pub(crate) fn encode_special_tokens(&self) -> bool {
        self.encode_special_tokens
    }
}

pub(crate) fn freeze(
    tokenizer: &ChatTokenizer,
    eos: &[u32],
    authority: &HostPreparationAuthority,
) -> Result<FrozenTokenizer, String> {
    // Raw imports and mutable access expose no upstream cache-capacity getter.
    // Unknown cache configuration is restored with caching disabled.
    let policy = tokenizer
        .model_cache_policy()
        .unwrap_or_else(ModelCachePolicy::disabled);
    let bytes = encode(&**tokenizer, policy)?;
    let mut restored = decode(&bytes)?;
    validate_configuration(&**tokenizer, &restored, &bytes)?;

    let original_environment = super::from_tokenizer_with_authority(tokenizer, eos, authority)?;
    eredu_text::tokenizer_storage::remove_input_prefixes(&mut restored)
        .map_err(|error| error.to_string())?;
    let grammar_bytes = encode(&restored, policy)?;
    let object = tokenizer_span(&grammar_bytes)
        .ok_or("normalized tokenizer has no validated source object")?;
    let encode_special_tokens = restored.get_encode_special_tokens();
    let environment = super::from_prepared_raw_with_info(restored, eos, None, authority)?;
    let grammar = FrozenGrammarTokenizer {
        bytes: grammar_bytes,
        object,
        encode_special_tokens,
        canonical: environment.tokenize_is_canonical(),
    };
    let original = original_environment.tok_trie();
    let copied = environment.tok_trie();
    if original.info() != copied.info()
        || original.eos_tokens() != copied.eos_tokens()
        || (0..original.info().vocab_size).any(|id| original.token(id) != copied.token(id))
    {
        return Err(
            "frozen tokenizer roundtrip changed token bytes or special-token metadata".into(),
        );
    }
    Ok(FrozenTokenizer {
        bytes,
        grammar,
        environment,
    })
}

/// Reconstructs the frozen environment without consulting a current model's
/// tokenizer or retaining a chat-template environment. `bytes` are the original
/// private recipe section validated by `freeze` before publication.
pub(crate) fn restore(
    bytes: &[u8],
    eos: &[u32],
    authority: &HostPreparationAuthority,
) -> Result<TokEnv, String> {
    restore_with_info(bytes, eos, None, authority)
}

/// Restores the same frozen tokenizer with the exact metadata of its retained
/// prepared environment; no token-name role selection replaces those facts.
pub(crate) fn restore_with_info(
    bytes: &[u8],
    eos: &[u32],
    info: Option<&llguidance::toktrie::TokRxInfo>,
    authority: &HostPreparationAuthority,
) -> Result<TokEnv, String> {
    super::from_raw_with_info(decode(bytes)?, eos, info, authority)
}

/// Reconstructs the recorded result of the shared prefix worker, without
/// reapplying policy to a different or current tokenizer. Only a closed recipe
/// or `freeze` supplies these bytes; ordinary HF construction remains authorized
/// by the same explicit host authority as the existing restore path.
pub(crate) fn restore_prepared(
    bytes: &[u8],
    eos: &[u32],
    info: Option<&llguidance::toktrie::TokRxInfo>,
    authority: &HostPreparationAuthority,
) -> Result<TokEnv, String> {
    super::from_prepared_raw_with_info(decode(bytes)?, eos, info, authority)
}

// Ordinary closed construction records this range before original admission.
// The retained range permits lexical compilation from the historical tokenizer
// without a second original-path header parser or owned Value.
pub(crate) fn tokenizer_span(bytes: &[u8]) -> Option<std::ops::Range<usize>> {
    #[derive(Deserialize)]
    struct Borrowed<'a> {
        #[serde(borrow)]
        tokenizer: &'a serde_json::value::RawValue,
    }
    let header: RecipeHeader = serde_json::from_slice(bytes).ok()?;
    header.policy().ok()?;
    let parsed: Borrowed<'_> = serde_json::from_slice(bytes).ok()?;
    let raw = parsed.tokenizer.get().as_bytes();
    let start = (raw.as_ptr() as usize).checked_sub(bytes.as_ptr() as usize)?;
    let end = start.checked_add(raw.len())?;
    (bytes.get(start..end)? == raw).then_some(start..end)
}
impl RecipeHeader {
    fn policy(&self) -> Result<ModelCachePolicy, String> {
        if self.version != RECIPE_VERSION {
            return Err(format!(
                "unsupported frozen tokenizer recipe version {}",
                self.version
            ));
        }
        Ok(self.cache_policy)
    }
}

fn decode(bytes: &[u8]) -> Result<Tokenizer, String> {
    // Validate the entire fixed recipe header before constructing nested HF.
    // IgnoredAny visits the tokenizer JSON without building a second graph;
    // policy may appear after tokenizer and JSON key order is immaterial.
    let header: RecipeHeader = serde_json::from_slice(bytes)
        .map_err(|error| format!("failed to restore frozen tokenizer: {error}"))?;
    let policy = header.policy()?;
    let _ = header.tokenizer;
    #[derive(Deserialize)]
    struct RecipeBody {
        tokenizer: Tokenizer,
    }
    let RecipeBody { mut tokenizer } = serde_json::from_slice(bytes)
        .map_err(|error| format!("failed to restore frozen tokenizer: {error}"))?;
    policy.apply(&mut tokenizer);
    tokenizer.set_encode_special_tokens(header.encode_special_tokens);
    Ok(tokenizer)
}

fn encode(tokenizer: &Tokenizer, policy: ModelCachePolicy) -> Result<Vec<u8>, String> {
    serde_json::to_vec(&RecipeRef {
        version: RECIPE_VERSION,
        tokenizer,
        cache_policy: policy,
        encode_special_tokens: tokenizer.get_encode_special_tokens(),
    })
    .map_err(|error| format!("failed to serialize frozen tokenizer: {error}"))
}

fn validate_configuration(
    original: &Tokenizer,
    restored: &Tokenizer,
    bytes: &[u8],
) -> Result<(), String> {
    // Serialize both sides before parsing: direct to_value can give f32 model
    // options a different JSON-number representation from the text serializer.
    let original_value: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|error| format!("failed to inspect frozen tokenizer: {error}"))?;
    let header: RecipeHeader = serde_json::from_slice(bytes)
        .map_err(|error| format!("failed to inspect frozen tokenizer: {error}"))?;
    let restored_bytes = encode(restored, header.policy()?)?;
    let restored_value: serde_json::Value = serde_json::from_slice(&restored_bytes)
        .map_err(|error| format!("failed to inspect restored tokenizer: {error}"))?;
    if original_value != restored_value {
        return Err("frozen tokenizer roundtrip changed serialized configuration".into());
    }
    if token_id_vocabulary(original) != token_id_vocabulary(restored) {
        return Err("frozen tokenizer roundtrip changed canonical token IDs or spellings".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests;
