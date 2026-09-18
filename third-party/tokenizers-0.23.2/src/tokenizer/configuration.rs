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
            crate::models::wordlevel::WordLevel::configuration_comparison_control_bytes()?,
            crate::models::unigram::Unigram::configuration_comparison_control_bytes()?,
            crate::decoders::fixed_profile::Profile::control_bytes()?,
            size_of::<std::iter::Zip<std::slice::Iter<'_,D>,std::slice::Iter<'_,D>>>(),
            super::normalization_source::comparison_control_bytes()?,
            std::mem::size_of::<[PreLeaves<'_>; 2]>(),
            std::mem::size_of::<[Option<Result<&P, ()>>; 2]>(),
            crate::processors::template::compiled::TemplateProcessing::configuration_comparison_control_bytes()?,
        ];
        controls
            .iter()
            .copied()
            .try_fold(0usize, usize::checked_add)
    }

    /// Tests the complete supported compiled configuration against a selected tokenizer.
    /// This borrows both sources, ignores execution caches, and allocates nothing.
    /// Unsupported source components return false; vocabulary equality alone is insufficient.
    pub fn matches_compiled_configuration(&self, selected: &Self) -> bool {
        super::TokenizerInput::from(self).matches_compiled_configuration(selected)
    }
}
impl super::TokenizerInput<'_> {
    /// Authenticate the complete projected input configuration by borrow.
    pub fn matches_compiled_configuration(self, selected: &Tokenizer) -> bool {
        let source = self.root;
        if !match (source.get_model(), selected.get_model()) {
            (M::BPE(a), M::BPE(b)) => a.same_configuration(b),
            (M::WordLevel(a), M::WordLevel(b)) => a.same_configuration(b),
            (M::Unigram(a), M::Unigram(b)) => a.same_configuration(b),
            _ => false,
        } || source.get_truncation().is_some()
            || selected.get_truncation().is_some()
            || source.get_padding().is_some()
            || selected.get_padding().is_some()
            || !super::normalization_source::same_input(self, selected.get_normalizer())
            || !pre(
                source.get_pre_tokenizer(),
                selected.get_pre_tokenizer(),
                self.remove_input_prefixes,
            )
            || !optional(
                source.get_post_processor(),
                selected.get_post_processor(),
                post,
            )
            || !optional(source.get_decoder(), selected.get_decoder(), decoder)
        {
            return false;
        }
        let a = self.added;
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
struct PreLeaves<'a> {
    stack: [std::slice::Iter<'a, P>; crate::utils::borrowed_json::DEPTH],
    depth: usize,
}
impl<'a> PreLeaves<'a> {
    fn new(value: Option<&'a P>) -> Self {
        let mut stack = std::array::from_fn(|_| [].iter());
        stack[0] = value.map_or(&[][..], std::slice::from_ref).iter();
        Self { stack, depth: 1 }
    }
}
impl<'a> Iterator for PreLeaves<'a> {
    type Item = Result<&'a P, ()>;
    fn next(&mut self) -> Option<Self::Item> {
        while self.depth != 0 {
            match self.stack[self.depth - 1].next() {
                None => self.depth -= 1,
                Some(P::Sequence(value)) => {
                    if self.depth == self.stack.len() {
                        self.depth = 0;
                        return Some(Err(()));
                    }
                    self.stack[self.depth] = value.as_ref().iter();
                    self.depth += 1;
                }
                Some(value) => return Some(Ok(value)),
            }
        }
        None
    }
}
fn pre(a: Option<&P>, b: Option<&P>, remove_prefixes: bool) -> bool {
    let (mut a, mut b) = (PreLeaves::new(a), PreLeaves::new(b));
    loop {
        match (a.next(), b.next()) {
            (None, None) => return true,
            (Some(Ok(a)), Some(Ok(b))) if pre_leaf(a, b, remove_prefixes) => {}
            _ => return false,
        }
    }
}
fn pre_leaf(a: &P, b: &P, remove_prefixes: bool) -> bool {
    match (a, b) {
        (P::Metaspace(a), P::Metaspace(b)) => {
            a.get_replacement() == b.get_replacement()
                && a.get_split() == b.get_split()
                && b.get_prepend_scheme()
                    == if remove_prefixes {
                        crate::pre_tokenizers::metaspace::PrependScheme::Never
                    } else {
                        a.get_prepend_scheme()
                    }
        }
        (P::ByteLevel(a), P::ByteLevel(b)) => {
            a == b
                && (!a.use_regex || {
                    #[cfg(feature = "fancy-regex")]
                    {
                        true
                    }
                    #[cfg(not(feature = "fancy-regex"))]
                    {
                        false
                    }
                })
        }
        (P::Digits(a), P::Digits(b)) => a == b,
        (P::Whitespace(a), P::Whitespace(b)) => a == b,
        #[cfg(feature = "fancy-regex")]
        (P::Split(a), P::Split(b)) => {
            a == b && a.workspace_plan().is_some() && b.workspace_plan().is_some()
        }
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
        (Q::Template(a), Q::Template(b)) => a == b,
        _ => false,
    }
}
fn decoder(a: &D, b: &D) -> bool {
    match (a, b) {
        (D::Sequence(a), D::Sequence(b)) => {
            let (a, b) = (a.get_decoders(), b.get_decoders());
            a.len() == b.len() && a.len() <= 5 && a.iter().zip(b).all(|(a, b)| decoder_leaf(a, b))
        }
        _ => decoder_leaf(a, b),
    }
}
fn decoder_leaf(a: &D, b: &D) -> bool {
    use crate::decoders::fixed_profile::Component;
    match (a, b) {
        (D::ByteLevel(a), D::ByteLevel(b)) => a == b,
        (D::Metaspace(a), D::Metaspace(b)) => a == b,
        _ => {
            let (a, b) = (Component::from_decoder(a), Component::from_decoder(b));
            a != Component::Other && a == b
        }
    }
}
