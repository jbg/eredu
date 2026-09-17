//! Borrowed authentication of the aggregate compiler's supported source profile.
use super::Tokenizer;
use crate::{
    decoders::DecoderWrapper as D, models::ModelWrapper as M, normalizers::NormalizerWrapper as N,
    pre_tokenizers::PreTokenizerWrapper as P, processors::PostProcessorWrapper as Q,
};

// Return-type queries never call the supplied function or construct a source.
fn added_return<T>(_: impl FnOnce(&'static super::AddedVocabulary) -> T) -> usize {
    std::mem::size_of::<T>()
}
impl Tokenizer {
    /// Named borrowed comparison frames for the supported aggregate profile.
    /// No source, iterator, cache, or allocation is constructed by this query.
    pub fn configuration_comparison_control_bytes() -> Option<usize> {
        use std::mem::size_of;
        let controls = [
            size_of::<[&Self; 2]>(), size_of::<[&super::AddedVocabulary; 2]>(),
            size_of::<[Option<&N>; 2]>(), size_of::<[Option<&P>; 2]>(),
            size_of::<[Option<&Q>; 2]>(), size_of::<[Option<&D>; 2]>(),
            size_of::<[usize; 2]>(), size_of::<bool>(),
            added_return(super::AddedVocabulary::tokens),
            added_return(super::AddedVocabulary::tokens),
            added_return(super::AddedVocabulary::vocabulary),
            size_of::<(u32, super::added_vocabulary::AddedTokenRef<'_>)>(),
            size_of::<Option<super::added_vocabulary::AddedTokenRef<'_>>>(),
            size_of::<(&str, u32)>(),
            size_of::<([&[P]; 2], std::iter::Zip<std::slice::Iter<'_, P>, std::slice::Iter<'_, P>>)>(),
            size_of::<([&[Q]; 2], std::iter::Zip<std::slice::Iter<'_, Q>, std::slice::Iter<'_, Q>>)>(),
            size_of::<[&D; 2]>(), size_of::<[&[D]; 2]>(),
            crate::models::bpe::BPE::configuration_comparison_control_bytes()?,
            crate::processors::template::compiled::CompiledTemplate::configuration_comparison_control_bytes()?,
        ];
        controls.iter().copied().try_fold(0usize, usize::checked_add)
    }

    /// Tests the complete supported compiled configuration against a selected tokenizer.
    /// This borrows both sources, ignores execution caches, and allocates nothing.
    /// Unsupported source components return false; vocabulary equality alone is insufficient.
    pub fn matches_compiled_configuration(&self, selected: &Self) -> bool {
        if !matches!((self.get_model(), selected.get_model()),
            (M::BPE(a), M::BPE(b)) if a.same_configuration(b))
            || self.get_truncation().is_some()
            || selected.get_truncation().is_some()
            || self.get_padding().is_some()
            || selected.get_padding().is_some()
            || !matches!(
                (self.get_normalizer(), selected.get_normalizer()),
                (None, None) | (Some(N::NFC(_)), Some(N::NFC(_)))
            )
            || !optional(self.get_pre_tokenizer(), selected.get_pre_tokenizer(), pre)
            || !optional(
                self.get_post_processor(),
                selected.get_post_processor(),
                post,
            )
            || !optional(self.get_decoder(), selected.get_decoder(), decoder)
        {
            return false;
        }
        let a = self.get_added_vocabulary();
        let b = selected.get_added_vocabulary();
        a.get_encode_special_tokens() == b.get_encode_special_tokens()
            && a.len() == b.len()
            && a.tokens().count() == b.tokens().count()
            && a.vocabulary().all(|(word, id)| {
                b.token_to_id(word, selected.get_model()) == Some(id)
                    && a.is_special_token(word) == b.is_special_token(word)
            })
            && a.tokens().all(|(id, x)| {
                b.token_ref(id).is_some_and(|y| {
                    x.content == y.content
                        && x.single_word == y.single_word
                        && x.lstrip == y.lstrip
                        && x.rstrip == y.rstrip
                        && x.normalized == y.normalized
                        && x.special == y.special
                })
            })
    }
}
fn optional<T>(a: Option<&T>, b: Option<&T>, same: fn(&T, &T) -> bool) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => same(a, b),
        _ => false,
    }
}
fn pre(a: &P, b: &P) -> bool {
    match (a, b) {
        (P::Sequence(a), P::Sequence(b)) => {
            let (a, b) = (a.as_ref(), b.as_ref());
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| pre_leaf(a, b))
        }
        _ => pre_leaf(a, b),
    }
}
fn pre_leaf(a: &P, b: &P) -> bool {
    match (a, b) {
        (P::ByteLevel(a), P::ByteLevel(b)) => a == b,
        (P::Digits(a), P::Digits(b)) => a == b,
        #[cfg(all(feature = "fancy-regex", not(feature = "onig")))]
        (P::CompiledByteLevel(a), P::ByteLevel(b)) => a.settings() == *b,
        #[cfg(feature = "fancy-regex")]
        (P::CompiledByteLevel(a), P::CompiledByteLevel(b)) => a == b,
        #[cfg(all(feature = "fancy-regex", not(feature = "onig")))]
        (P::CompiledRegexSplit(a), P::Split(b)) => {
            matches!(&b.pattern, crate::pre_tokenizers::split::SplitPattern::Regex(pattern)
                if a.pattern() == pattern && b.regex.matches_pattern(pattern))
                && b.behavior == crate::SplitDelimiterBehavior::Isolated
                && !b.invert
        }
        #[cfg(feature = "fancy-regex")]
        (P::CompiledRegexSplit(a), P::CompiledRegexSplit(b)) => a == b,
        _ => false,
    }
}
fn post(a: &Q, b: &Q) -> bool {
    match (a, b) {
        (Q::Sequence(a), Q::Sequence(b)) => {
            let (a, b) = (a.as_ref(), b.as_ref());
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| post_leaf(a, b))
        }
        _ => post_leaf(a, b),
    }
}
fn post_leaf(a: &Q, b: &Q) -> bool {
    match (a, b) {
        (Q::ByteLevel(a), Q::ByteLevel(b)) => a == b,
        (Q::CompiledTemplate(a), Q::Template(b)) => a.matches_template(b),
        (Q::CompiledTemplate(a), Q::CompiledTemplate(b)) => a == b,
        _ => false,
    }
}
fn decoder(a: &D, b: &D) -> bool {
    match (a, b) {
        (D::ByteLevel(a), D::ByteLevel(b)) => a == b,
        (D::Sequence(a), D::Sequence(b)) => {
            let (a, b) = (a.get_decoders(), b.get_decoders());
            a.len() == 1
                && b.len() == 1
                && matches!((&a[0], &b[0]),
                (D::ByteLevel(a), D::ByteLevel(b)) if a == b)
        }
        _ => false,
    }
}
