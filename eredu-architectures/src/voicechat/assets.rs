//! Exact auxiliary assets retained before VoiceChat construction.
//!
//! Resolution is supplied by the caller. Neither these contracts nor a prepared
//! asset perform network access, interpret NeMo targets, or reopen files.

use std::{collections::BTreeMap, error::Error, path::Path, sync::Arc};

use serde_json::Value;
use sha2::{Digest, Sha256};

/// Official VoiceChat release providing its configuration and RNNT tokenizer.
pub const VOICECHAT_ASSET_MODEL: &str = "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B";
/// Exact official release revision.
pub const VOICECHAT_ASSET_REVISION: &str = "443794ea956ef0065f001967ffd00e77f519cb39";
/// Base language model providing the backbone configuration and tokenizer.
pub const VOICECHAT_BASE_MODEL: &str = "nvidia/NVIDIA-Nemotron-Nano-9B-v2";
/// Exact base model release revision.
pub const VOICECHAT_BASE_REVISION: &str = "6533e8de2c68e4536bf7c411d7a3ce5734111476";

/// Semantic ownership of an auxiliary artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum VoiceChatAssetRole {
    /// Nested NeMo model and preprocessing configuration.
    Configuration,
    /// Nemotron-H backbone configuration.
    BackboneConfiguration,
    /// Base tokenizer's vocabulary, merges and normalization.
    TextTokenizer,
    /// Base tokenizer's auxiliary declarations.
    TextTokenizerConfiguration,
    /// Exact SentencePiece model used for RNNT transcription.
    RnntTokenizer,
    /// RNNT tokenizer declarations; embedded training paths are not resolved.
    RnntTokenizerConfiguration,
    /// Ordered RNNT vocabulary, checked against the joint network configuration.
    RnntVocabulary,
}

/// Immutable request identifying one exact release asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoiceChatAssetRequest {
    /// Architecture-declared purpose.
    pub role: VoiceChatAssetRole,
    /// Repository identifier.
    pub model: &'static str,
    /// Exact repository revision.
    pub revision: &'static str,
    /// Relative filename within that revision.
    pub filename: &'static str,
    /// Expected byte length.
    pub byte_len: usize,
    /// SHA-256 of the published bytes, encoded as lowercase hexadecimal.
    pub sha256: &'static str,
}

use VoiceChatAssetRole as Role;

const fn official(
    role: Role,
    filename: &'static str,
    byte_len: usize,
    sha256: &'static str,
) -> VoiceChatAssetRequest {
    VoiceChatAssetRequest {
        role,
        model: VOICECHAT_ASSET_MODEL,
        revision: VOICECHAT_ASSET_REVISION,
        filename,
        byte_len,
        sha256,
    }
}
const fn base(
    role: Role,
    filename: &'static str,
    byte_len: usize,
    sha256: &'static str,
) -> VoiceChatAssetRequest {
    VoiceChatAssetRequest {
        role,
        model: VOICECHAT_BASE_MODEL,
        revision: VOICECHAT_BASE_REVISION,
        filename,
        byte_len,
        sha256,
    }
}

/// Complete auxiliary asset contract for the pinned VoiceChat release.
pub const VOICECHAT_ASSET_REQUESTS: [VoiceChatAssetRequest; 7] = [
    official(
        Role::Configuration,
        "config.json",
        31984,
        "2e0b67b56fefeb4815e436d66ad84f2f94be192b1d543054c810ed239a28cff5",
    ),
    base(
        Role::BackboneConfiguration,
        "config.json",
        1507,
        "642dc554a725ea216be55c3018d0a6f3dc9963b24779c3cc97f5a0ea21e9dcc3",
    ),
    base(
        Role::TextTokenizer,
        "tokenizer.json",
        17078330,
        "3277c00fe5fb3963b3cb7c07b7f183722d2af4d775a4aea7cfb3684d7cccbc2f",
    ),
    base(
        Role::TextTokenizerConfiguration,
        "tokenizer_config.json",
        181326,
        "a3e4d48a0f8b4ce6fd746464199299cbdefe6ee202e34048940da2d9838a9aa6",
    ),
    official(
        Role::RnntTokenizer,
        "rnnt_tokenizer/tokenizer.model",
        251056,
        "07d4e5a63840a53ab2d4d106d2874768143fb3fbdd47938b3910d2da05bfb0a9",
    ),
    official(
        Role::RnntTokenizerConfiguration,
        "rnnt_tokenizer/tokenizer_config.json",
        268,
        "10e393422195708d9d131f66dda09430dc472d3f7d36c3e4fd0a5135bf4809b0",
    ),
    official(
        Role::RnntVocabulary,
        "rnnt_tokenizer/vocab.json",
        9242,
        "35051f795cec22352180d3a3d2a1383ffbe0d968cd14049ad6da254492f3678e",
    ),
];

/// Immutable validated bytes and their exact release provenance.
#[derive(Clone)]
pub struct RetainedVoiceChatAsset {
    request: VoiceChatAssetRequest,
    bytes: Arc<[u8]>,
}

impl std::fmt::Debug for RetainedVoiceChatAsset {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RetainedVoiceChatAsset")
            .field("request", &self.request)
            .finish_non_exhaustive()
    }
}

impl RetainedVoiceChatAsset {
    /// The exact request whose length and digest were verified.
    pub const fn provenance(&self) -> &VoiceChatAssetRequest {
        &self.request
    }
    /// Shares retained bytes without reopening the source.
    pub fn bytes(&self) -> Arc<[u8]> {
        self.bytes.clone()
    }
}

/// Portable asset preparation failure, retaining resolver and JSON error sources.
#[derive(Debug, thiserror::Error)]
pub enum VoiceChatAssetError {
    /// The application could not resolve the exact request.
    #[error("could not resolve VoiceChat asset {role:?}: {source}")]
    Resolve {
        /// Failed role.
        role: VoiceChatAssetRole,
        /// Original resolver failure.
        #[source]
        source: Box<dyn Error + Send + Sync>,
    },
    /// Bytes differ from the pinned published asset.
    #[error("VoiceChat asset {role:?} differs from the pinned release (expected {expected_len} bytes, received {actual_len}; SHA-256 {actual_sha256})")]
    ContentMismatch {
        /// Failed role.
        role: VoiceChatAssetRole,
        /// Published byte length.
        expected_len: usize,
        /// Resolved byte length.
        actual_len: usize,
        /// Digest of the resolved bytes.
        actual_sha256: String,
    },
    /// A retained JSON declaration is invalid.
    #[error("invalid VoiceChat asset JSON for {role:?}: {source}")]
    Json {
        /// Failed role.
        role: VoiceChatAssetRole,
        /// JSON failure.
        #[source]
        source: serde_json::Error,
    },
    /// Cross-asset semantic declarations disagree.
    #[error("inconsistent VoiceChat assets: {0}")]
    Inconsistent(String),
}

/// All auxiliary assets consumed by architecture construction and text policy.
#[derive(Debug, Clone)]
pub struct PreparedVoiceChatAssets {
    assets: BTreeMap<VoiceChatAssetRole, RetainedVoiceChatAsset>,
    configuration: Value,
    backbone_configuration: Value,
    rnnt_vocabulary: Vec<String>,
}

impl PreparedVoiceChatAssets {
    /// Resolves each exact request once, checks published bytes, and retains them.
    ///
    /// The caller controls local caching or downloading. Successful preparation
    /// removes that resolver from every construction and execution path.
    pub fn resolve<E: Error + Send + Sync + 'static>(
        mut resolver: impl FnMut(&VoiceChatAssetRequest) -> Result<Vec<u8>, E>,
    ) -> Result<Self, VoiceChatAssetError> {
        let mut assets = BTreeMap::new();
        for request in VOICECHAT_ASSET_REQUESTS {
            let bytes = resolver(&request).map_err(|source| VoiceChatAssetError::Resolve {
                role: request.role,
                source: Box::new(source),
            })?;
            let asset = retain(request, bytes)?;
            assets.insert(request.role, asset);
        }
        let read_json = |role| {
            serde_json::from_slice::<Value>(&assets[&role].bytes)
                .map_err(|source| VoiceChatAssetError::Json { role, source })
        };
        let configuration = read_json(Role::Configuration)?;
        let backbone_configuration = read_json(Role::BackboneConfiguration)?;
        let tokenizer = read_json(Role::TextTokenizer)?;
        let rnnt_vocabulary: Vec<String> =
            serde_json::from_slice(&assets[&Role::RnntVocabulary].bytes).map_err(|source| {
                VoiceChatAssetError::Json {
                    role: Role::RnntVocabulary,
                    source,
                }
            })?;
        validate_semantics(
            &configuration,
            &backbone_configuration,
            &tokenizer,
            &rnnt_vocabulary,
        )?;
        Ok(Self {
            assets,
            configuration,
            backbone_configuration,
            rnnt_vocabulary,
        })
    }

    /// Reads the declared files from separate official and base snapshot roots.
    pub fn from_local_directories(
        official_root: &Path,
        base_root: &Path,
    ) -> Result<Self, VoiceChatAssetError> {
        Self::resolve(|request| {
            let root = if request.model == VOICECHAT_ASSET_MODEL {
                official_root
            } else {
                base_root
            };
            std::fs::read(root.join(request.filename))
        })
    }

    /// One exact retained asset; every declared role is present.
    pub fn asset(&self, role: VoiceChatAssetRole) -> &RetainedVoiceChatAsset {
        &self.assets[&role]
    }
    /// Declarative nested official configuration; `_target_` strings are data.
    pub const fn configuration(&self) -> &Value {
        &self.configuration
    }
    /// Exact base Nemotron-H configuration, before VoiceChat token-policy overrides.
    pub const fn backbone_configuration(&self) -> &Value {
        &self.backbone_configuration
    }
    /// Ordered RNNT vocabulary, with the blank at `len()` rather than in the list.
    pub fn rnnt_vocabulary(&self) -> &[String] {
        &self.rnnt_vocabulary
    }

    /// Checks the extracted backbone's embedded tokenizer against the exact
    /// retained base assets with VoiceChat's BOS/EOS/padding overrides.
    pub fn validate_split_backbone_tokenizer(
        &self,
        metadata: &BTreeMap<String, eredu_gguf::MetadataValue>,
    ) -> Result<(), VoiceChatAssetError> {
        validate_split_backbone_tokenizer(metadata)
    }
}

fn validate_split_backbone_tokenizer(
    metadata: &BTreeMap<String, eredu_gguf::MetadataValue>,
) -> Result<(), VoiceChatAssetError> {
    let invalid = |key: &str| {
        VoiceChatAssetError::Inconsistent(format!(
            "split backbone {key} differs from the retained tokenizer contract"
        ))
    };
    // These ordered-sequence digests are independently reconstructed from the
    // pinned tokenizer, then checked against the published split GGUF header.
    // Each string sequence hashes its u64 count followed by u64 UTF-8 lengths
    // and bytes. Integer sequences hash their count followed by I64 values.
    for (key, count, expected) in [
        (
            "tokenizer.ggml.tokens",
            131072,
            "c25f4ade7da3cc72fa02c36f4a645cb0f2f372a42e5673c6b93a6fc32541eff1",
        ),
        (
            "tokenizer.ggml.merges",
            269443,
            "9cfdc4f21fe6479d1711e073aaff3a11985141776e5033492f79ad3a5ccb2372",
        ),
    ] {
        let values = metadata
            .get(key)
            .and_then(eredu_gguf::MetadataValue::as_strings)
            .ok_or_else(|| invalid(key))?;
        if values.len() != count {
            return Err(invalid(key));
        }
        let mut hash = Sha256::new();
        hash.update((count as u64).to_le_bytes());
        for value in values {
            hash.update((value.len() as u64).to_le_bytes());
            hash.update(value.as_bytes());
        }
        if digest_hex(hash) != expected {
            return Err(invalid(key));
        }
    }
    let key = "tokenizer.ggml.token_type";
    let types = metadata
        .get(key)
        .and_then(eredu_gguf::MetadataValue::to_i64_vec)
        .ok_or_else(|| invalid(key))?;
    if types.len() != 131072 {
        return Err(invalid(key));
    }
    let mut hash = Sha256::new();
    hash.update((types.len() as u64).to_le_bytes());
    for value in types {
        hash.update(value.to_le_bytes());
    }
    if digest_hex(hash) != "5ee8a9260e5514bf9471c95b9d11c8d2099b8b1385a6507061685df5a53aead4" {
        return Err(invalid(key));
    }
    let key = "tokenizer.chat_template";
    let template = metadata
        .get(key)
        .and_then(eredu_gguf::MetadataValue::as_str)
        .ok_or_else(|| invalid(key))?;
    let mut hash = Sha256::new();
    hash.update(template.as_bytes());
    if digest_hex(hash) != "b7a3a520a4bc1beae6a25d615f9215580df8eea926f488256c2bef74039433c0" {
        return Err(invalid(key));
    }
    for (key, expected) in [
        ("tokenizer.ggml.model", "gpt2"),
        ("tokenizer.ggml.pre", "tekken"),
    ] {
        if metadata
            .get(key)
            .and_then(eredu_gguf::MetadataValue::as_str)
            != Some(expected)
        {
            return Err(invalid(key));
        }
    }
    for (key, expected) in [
        ("tokenizer.ggml.bos_token_id", 1),
        ("tokenizer.ggml.eos_token_id", 2),
        ("tokenizer.ggml.padding_token_id", 12),
    ] {
        if metadata
            .get(key)
            .and_then(eredu_gguf::MetadataValue::as_i64)
            != Some(expected)
        {
            return Err(invalid(key));
        }
    }
    for key in [
        "tokenizer.ggml.add_bos_token",
        "tokenizer.ggml.add_eos_token",
    ] {
        if metadata
            .get(key)
            .and_then(eredu_gguf::MetadataValue::as_bool)
            != Some(false)
        {
            return Err(invalid(key));
        }
    }
    Ok(())
}

fn digest_hex(hash: Sha256) -> String {
    hash.finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn retain(
    request: VoiceChatAssetRequest,
    bytes: Vec<u8>,
) -> Result<RetainedVoiceChatAsset, VoiceChatAssetError> {
    let actual_sha256 = Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    if bytes.len() != request.byte_len || actual_sha256 != request.sha256 {
        return Err(VoiceChatAssetError::ContentMismatch {
            role: request.role,
            expected_len: request.byte_len,
            actual_len: bytes.len(),
            actual_sha256,
        });
    }
    Ok(RetainedVoiceChatAsset {
        request,
        bytes: bytes.into(),
    })
}

fn validate_semantics(
    configuration: &Value,
    backbone: &Value,
    tokenizer: &Value,
    rnnt_vocabulary: &[String],
) -> Result<(), VoiceChatAssetError> {
    let invalid = |reason: &str| VoiceChatAssetError::Inconsistent(reason.into());
    // VoiceChat overrides base eos=12 with </s>=2 and uses 12 for padding.
    if backbone.get("eos_token_id").and_then(Value::as_u64) != Some(12) {
        return Err(invalid(
            "base EOS declaration differs from the pinned Nemotron release",
        ));
    }
    for (token, expected) in [
        ("<s>", 1),
        ("</s>", 2),
        ("<SPECIAL_11>", 11),
        ("<SPECIAL_12>", 12),
        ("<SPECIAL_20>", 20),
        ("<SPECIAL_21>", 21),
        ("<SPECIAL_22>", 22),
        ("^", 1094),
        ("$", 1036),
    ] {
        let id = tokenizer
            .pointer("/model/vocab")
            .and_then(|v| v.get(token))
            .and_then(Value::as_u64)
            .or_else(|| {
                tokenizer
                    .get("added_tokens")?
                    .as_array()?
                    .iter()
                    .find(|entry| entry.get("content").and_then(Value::as_str) == Some(token))?
                    .get("id")?
                    .as_u64()
            });
        if id != Some(expected) {
            return Err(invalid(
                "text tokenizer control tokens differ from VoiceChat policy",
            ));
        }
    }
    let vocabulary = configuration
        .pointer("/_rnnt_merge_info/joint_config/vocabulary")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("RNNT joint vocabulary is absent"))?;
    if vocabulary.len() != rnnt_vocabulary.len()
        || vocabulary
            .iter()
            .zip(rnnt_vocabulary)
            .any(|(actual, expected)| actual.as_str() != Some(expected.as_str()))
        || configuration
            .pointer("/_rnnt_merge_info/decoder_config/vocab_size")
            .and_then(Value::as_u64)
            != Some(rnnt_vocabulary.len() as u64)
    {
        return Err(invalid(
            "RNNT tokenizer vocabulary differs from decoder/joint geometry",
        ));
    }
    Ok(())
}
