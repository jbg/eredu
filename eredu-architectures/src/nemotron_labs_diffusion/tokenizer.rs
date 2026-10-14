//! Released tokenizer vocabulary and reserved-identity policy.
use super::FamilyConfig;
use crate::decoder::Config;

/// Tokenizer assets disagree with the admitted neural vocabulary or reserved IDs.
#[derive(Debug, thiserror::Error)]
pub enum TokenizerPolicyError {
    /// Highest canonical ID does not match the released readout extent.
    #[error("tokenizer ID extent {actual} differs from neural vocabulary {expected}")]
    Vocabulary {
        /// Canonical tokenizer ID extent.
        actual: u64,
        /// Admitted readout row count.
        expected: u64,
    },
    /// A required reserved spelling maps to a different identity.
    #[error("required token {token} does not map to canonical ID {id}")]
    Reserved {
        /// Published canonical spelling.
        token: String,
        /// Required neural identity.
        id: u32,
    },
}

/// Checks the full fast-tokenizer mapping, including the VLM's extra placeholder
/// row. This policy does not select a backend or allocate a native tensor.
pub fn validate(
    config: &FamilyConfig,
    tokenizer: &eredu_text::tokenizer::Tokenizer,
) -> Result<(), TokenizerPolicyError> {
    let vocabulary = eredu_text::tokenizer::token_id_vocabulary(tokenizer);
    let extent = vocabulary
        .last_key_value()
        .map_or(0, |(id, _)| u64::from(*id) + 1);
    let expected = config.vocabulary_size() as u64;
    if extent != expected {
        return Err(TokenizerPolicyError::Vocabulary {
            actual: extent,
            expected,
        });
    }
    let mut reserved = vec![
        ("<s>".to_owned(), 1),
        ("<|im_end|>".to_owned(), 11),
        (
            format!("<SPECIAL_{}>", config.mask_token_id),
            config.mask_token_id,
        ),
    ];
    if config.vision.is_some() {
        reserved.extend(
            super::processing::IMAGE_TOKENS
                .into_iter()
                .map(|(token, id)| (token.to_owned(), id)),
        );
    }
    for (token, id) in reserved {
        if vocabulary.get(&id) != Some(&token) || tokenizer.token_to_id(&token) != Some(id) {
            return Err(TokenizerPolicyError::Reserved { token, id });
        }
    }
    Ok(())
}
