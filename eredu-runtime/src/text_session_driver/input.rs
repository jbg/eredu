//! Portable ownership of ordered text prompts and retained prefill cursors.

use eredu_core::{
    BackendFailure, BackendFailureKind, InputModality, InputTensorIdentity, PreparedInputError,
};

use super::{PlainTextPrefill, TextPrefillInput};
use eredu_core::input::PlainTextTokens;

use crate::{prefill::ErasedPrefillCursor, PreparedInputCacheIdentity};
use eredu_core::{PreparedInputPart, PreparedInputPayload};

/// Ordered runtime input for model prefill.
#[derive(Debug)]
pub struct TextPromptView<'a, T> {
    /// Ordered input parts consumed by the model.
    pub parts: &'a [PreparedInputPart<T>],
    cache_identity: Option<&'a PreparedInputCacheIdentity>,
    prefill_fragment: bool,
}

impl<T> Copy for TextPromptView<'_, T> {}

impl<T> Clone for TextPromptView<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> TextPromptView<'a, T> {
    /// Creates a typed input from ordered parts.
    pub fn new(parts: &'a [PreparedInputPart<T>]) -> Self {
        Self {
            parts,
            cache_identity: None,
            prefill_fragment: false,
        }
    }

    /// Creates a typed input carrying its exact cache-relevant semantic identity.
    pub fn with_cache_identity(
        parts: &'a [PreparedInputPart<T>],
        cache_identity: &'a PreparedInputCacheIdentity,
    ) -> Self {
        Self {
            parts,
            cache_identity: Some(cache_identity),
            prefill_fragment: false,
        }
    }

    /// Whether this view is one fragment of an admitted plain prompt.
    pub const fn is_prefill_fragment(self) -> bool {
        self.prefill_fragment
    }

    /// Returns the identity coupled to these exact prepared values, when supplied.
    pub const fn cache_identity(self) -> Option<&'a PreparedInputCacheIdentity> {
        self.cache_identity
    }
}

/// Ordered prepared tensors and the portable state of a prefill invocation.
#[derive(Clone)]
pub struct PreparedTextPrompt<T> {
    parts: Vec<PreparedInputPart<T>>,
    cache_identity: Option<PreparedInputCacheIdentity>,
    fragment: bool,
    cursor: Option<ErasedPrefillCursor>,
}

impl<T: std::fmt::Debug> std::fmt::Debug for PreparedTextPrompt<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedTextPrompt")
            .field("parts", &self.parts)
            .field("cache_identity", &self.cache_identity())
            .field(
                "prefill_step",
                &self.prefill_cursor().map(ErasedPrefillCursor::next_step),
            )
            .finish()
    }
}

impl<T: Clone> From<TextPromptView<'_, T>> for PreparedTextPrompt<T> {
    fn from(input: TextPromptView<'_, T>) -> Self {
        Self {
            parts: input.parts.to_vec(),
            cache_identity: input.cache_identity.cloned(),
            fragment: input.prefill_fragment,
            cursor: None,
        }
    }
}

impl<T> PreparedTextPrompt<T> {
    /// Wraps an already admitted token tensor as a new invocation input.
    pub(super) fn from_committed_tokens(tokens: T) -> Result<Self, BackendFailure> {
        let part = PreparedInputPart::new_with_extents(
            InputModality::Text,
            PreparedInputPayload::TokenIds(tokens),
            [],
            [],
        )
        .map_err(BackendFailure::from_error)?;
        Ok(Self {
            parts: vec![part],
            cache_identity: None,
            fragment: false,
            cursor: None,
        })
    }

    pub(super) fn from_prepared_media(
        input: eredu_core::PreparedModelInput<T>,
        identity: PreparedInputCacheIdentity,
    ) -> Self {
        Self {
            parts: input.into_parts(),
            cache_identity: Some(identity),
            fragment: false,
            cursor: None,
        }
    }

    /// Borrows the admitted prepared payload.
    pub fn with_borrowed<R>(&self, use_prompt: impl FnOnce(TextPromptView<'_, T>) -> R) -> R {
        use_prompt(self.borrowed())
    }

    /// Returns the exact semantic identity of this prepared prefix.
    pub const fn cache_identity(&self) -> Option<&PreparedInputCacheIdentity> {
        self.cache_identity.as_ref()
    }

    /// Returns the cursor that owns the remaining prepared prefill, if started.
    pub fn prefill_cursor(&self) -> Option<&ErasedPrefillCursor> {
        self.cursor.as_ref()
    }

    /// Couples the prepared tensors with caller-supplied semantic content.
    pub fn with_semantic_content_fingerprint(
        mut self,
        fingerprint: impl Into<String>,
        identify: impl Fn(&T) -> Result<InputTensorIdentity, PreparedInputError>,
    ) -> Result<Self, BackendFailure>
    where
        T: Clone,
    {
        self.cache_identity = Some(prepared_prompt_identity(
            &self.parts,
            fingerprint,
            identify,
        )?);
        self.fragment = false;
        Ok(self)
    }

    /// Returns the unadorned token tensor of a plain text prompt.
    pub fn plain_token_tensor(&self) -> Option<&T> {
        let [part] = self.parts.as_slice() else {
            return None;
        };
        if part.modality() != InputModality::Text
            || !part.metadata().is_empty()
            || !part.extents().is_empty()
        {
            return None;
        }
        match part.payload() {
            PreparedInputPayload::TokenIds(tokens) => Some(tokens),
            _ => None,
        }
    }
}

impl<T> TextPrefillInput for PreparedTextPrompt<T> {
    type Borrowed<'a>
        = TextPromptView<'a, T>
    where
        T: 'a;

    fn borrowed(&self) -> Self::Borrowed<'_> {
        TextPromptView {
            parts: &self.parts,
            cache_identity: self.cache_identity.as_ref(),
            prefill_fragment: self.fragment,
        }
    }

    fn prompt_cache_identity(&self) -> Option<&PreparedInputCacheIdentity> {
        self.cache_identity()
    }

    fn prefill_cursor(&self) -> Option<&ErasedPrefillCursor> {
        self.cursor.as_ref()
    }

    fn prefill_cursor_mut(&mut self) -> Option<&mut ErasedPrefillCursor> {
        self.cursor.as_mut()
    }

    fn install_prefill_cursor(&mut self, cursor: ErasedPrefillCursor) {
        self.parts.clear();
        self.cursor = Some(cursor);
    }
}

/// Admits a plain prompt and attaches its exact portable cache identity.
pub fn prepare_plain_text_prompt<B, T>(
    backend: &B,
    ids: Vec<u32>,
) -> Result<PreparedTextPrompt<T>, B::Error>
where
    B: PlainTextTokens<T>,
    B::Error: From<BackendFailure>,
    T: Clone,
{
    if ids.is_empty() {
        return Err(BackendFailure::from_error(PreparedInputError::EmptyInput).into());
    }
    let tokens = backend.materialize_tokens(&ids)?;
    PreparedTextPrompt::from_committed_tokens(tokens)?
        .with_semantic_content_fingerprint(
            eredu_core::cache::prompt_cache_token_fingerprint(&ids),
            B::token_identity,
        )
        .map_err(Into::into)
}

impl<B, T> PlainTextPrefill<B> for PreparedTextPrompt<T>
where
    B: PlainTextTokens<T>,
    B::Error: From<BackendFailure>,
{
    fn plain_token_count(&self) -> Option<usize> {
        B::token_count(self.plain_token_tensor()?)
    }

    fn split_plain_prompt(
        &self,
        backend: &B,
        at: std::num::NonZeroUsize,
    ) -> Result<(Self, Self), B::Error> {
        let tokens = self.plain_token_tensor().ok_or_else(|| {
            BackendFailure::new(
                BackendFailureKind::InvalidInput,
                std::io::Error::other("expected a plain text prompt"),
            )
        })?;
        let (prefix, suffix) = backend.split_tokens(tokens, at)?;
        let part = |tokens| {
            PreparedInputPart::new_with_extents(
                InputModality::Text,
                PreparedInputPayload::TokenIds(tokens),
                [],
                [],
            )
            .map_err(BackendFailure::from_error)
        };
        Ok((
            Self {
                parts: vec![part(prefix)?],
                cache_identity: None,
                fragment: true,
                cursor: None,
            },
            Self {
                parts: vec![part(suffix)?],
                cache_identity: self.cache_identity.clone(),
                fragment: true,
                cursor: None,
            },
        ))
    }
}

/// Couples prepared tensors to their semantic content and exact tensor identities.
fn prepared_prompt_identity<T: Clone>(
    parts: &[PreparedInputPart<T>],
    fingerprint: impl Into<String>,
    identify: impl Fn(&T) -> Result<InputTensorIdentity, PreparedInputError>,
) -> Result<PreparedInputCacheIdentity, BackendFailure> {
    let descriptors = parts
        .iter()
        .map(|part| part.descriptor(&identify))
        .collect::<Result<Vec<_>, _>>()
        .map_err(BackendFailure::from_error)?;
    let identity =
        eredu_core::PreparedInputIdentity::new(descriptors).map_err(BackendFailure::from_error)?;
    PreparedInputCacheIdentity::new(identity, fingerprint).map_err(BackendFailure::from_error)
}
