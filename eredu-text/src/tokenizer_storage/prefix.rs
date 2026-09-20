//! Input-prefix normalization through public component replacement APIs.
use super::*;
use tokenizers::{NormalizerWrapper, PreTokenizerWrapper, normalizers, pre_tokenizers};

/// Removes Prepend normalizers and Metaspace pre-tokenizer prefixes recursively.
/// Decoder semantics and every other component are preserved. Replacing a
/// normalizer refreshes upstream's normalized added-token matcher.
pub fn remove_input_prefixes(tokenizer: &mut tokenizers::Tokenizer) -> tokenizers::Result<()> {
    fn without_prepend(normalizer: NormalizerWrapper) -> Option<NormalizerWrapper> {
        match normalizer {
            NormalizerWrapper::Prepend(_) => None,
            NormalizerWrapper::Sequence(sequence) => {
                let members = sequence
                    .as_ref()
                    .iter()
                    .cloned()
                    .filter_map(without_prepend)
                    .collect::<Vec<_>>();
                (!members.is_empty())
                    .then(|| NormalizerWrapper::Sequence(normalizers::Sequence::new(members)))
            }
            other => Some(other),
        }
    }

    fn without_metaspace_prefix(pre_tokenizer: PreTokenizerWrapper) -> PreTokenizerWrapper {
        match pre_tokenizer {
            PreTokenizerWrapper::Metaspace(mut metaspace) => {
                metaspace.prepend_scheme = pre_tokenizers::metaspace::PrependScheme::Never;
                PreTokenizerWrapper::Metaspace(metaspace)
            }
            PreTokenizerWrapper::Sequence(sequence) => {
                PreTokenizerWrapper::Sequence(pre_tokenizers::sequence::Sequence::new(
                    sequence
                        .as_ref()
                        .iter()
                        .cloned()
                        .map(without_metaspace_prefix)
                        .collect(),
                ))
            }
            other => other,
        }
    }

    if let Some(normalizer) = tokenizer.get_normalizer().cloned() {
        tokenizer.with_normalizer(without_prepend(normalizer))?;
    }
    if let Some(pre_tokenizer) = tokenizer.get_pre_tokenizer().cloned() {
        tokenizer.with_pre_tokenizer(Some(without_metaspace_prefix(pre_tokenizer)));
    }
    Ok(())
}

pub(super) fn removal_is_identity(tokenizer: &tokenizers::Tokenizer) -> bool {
    fn normalizer(n: &NormalizerWrapper) -> bool {
        match n {
            NormalizerWrapper::Prepend(_) => false,
            NormalizerWrapper::Sequence(s) => s.as_ref().iter().all(normalizer),
            _ => true,
        }
    }
    fn pretokenizer(p: &PreTokenizerWrapper) -> bool {
        match p {
            PreTokenizerWrapper::Metaspace(m) => {
                m.get_prepend_scheme() == pre_tokenizers::metaspace::PrependScheme::Never
            }
            PreTokenizerWrapper::Sequence(s) => s.as_ref().iter().all(pretokenizer),
            _ => true,
        }
    }
    tokenizer.get_normalizer().is_none_or(normalizer)
        && tokenizer.get_pre_tokenizer().is_none_or(pretokenizer)
}

/// A prefix-normalized source requires an independent model copy. The estimate
/// includes a complete construction footprint; its origin remains retained.
#[derive(Debug)]
pub struct InputPrefixPlan<'a> {
    source: &'a PreparedTokenizer,
    requirements: TokenizerRequirements,
}
impl PreparedTokenizer {
    /// Returns no plan when removal is the identity; otherwise estimates one copy.
    pub fn input_prefix_plan(&self) -> Result<Option<InputPrefixPlan<'_>>, TokenizerSourceError> {
        if self.input_prefix_removal_is_identity() {
            return Ok(None);
        }
        let estimated_bytes = self
            .root()
            .estimate
            .construction(self.root().source_bytes)
            .ok_or_else(TokenizerSourceError::overflow)?;
        Ok(Some(InputPrefixPlan {
            source: self,
            requirements: TokenizerRequirements { estimated_bytes },
        }))
    }
}
impl InputPrefixPlan<'_> {
    /// Estimate for the independently resident model, matcher and decode program.
    pub fn requirements(&self) -> TokenizerRequirements {
        self.requirements
    }
    /// Copies and normalizes once after admission, retaining the original identity.
    pub fn compile(self) -> Result<PreparedTokenizer, InputPrefixFailure> {
        let root = self.source.root();
        let mut tokenizer = root.tokenizer.clone();
        if let Err(cause) = remove_input_prefixes(&mut tokenizer) {
            return Err(InputPrefixFailure {
                failure: TokenizerConstructionFailure {
                    cause: Cause::Root(cause),
                    tokenizer: Some(tokenizer),
                    _metadata: None,
                },
                _origin: Arc::clone(&self.source.root),
            });
        }
        build(
            tokenizer,
            root.source_bytes,
            root.estimate,
            Some(Arc::clone(&self.source.root)),
            #[cfg(feature = "tokenizer-compiler-test-support")]
            None,
        )
        .map_err(|failure| InputPrefixFailure {
            failure,
            _origin: Arc::clone(&self.source.root),
        })
    }
}
/// Actual construction failure and the retained origin identity.
#[derive(Debug)]
pub struct InputPrefixFailure {
    failure: TokenizerConstructionFailure,
    _origin: Arc<Root>,
}

impl fmt::Display for InputPrefixFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.failure, f)
    }
}
impl std::error::Error for InputPrefixFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.failure)
    }
}
