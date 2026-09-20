//! Fallible encoding through stock toktrie's byte, special and marker workers.
//! Callers admit dependency headroom before entering these infallible Vec APIs.
use llguidance::toktrie::{TokTrie, TokenizerEnv};
use std::cell::RefCell;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TokenizationMode {
    Plain,
    Special,
    Marker,
}

#[derive(Debug)]
pub(crate) struct TokenizationFailure<E> {
    pub(crate) cause: E,
    pub(crate) partial: Vec<u32>,
}

/// Preserve the first encoding error while the upstream worker retires its own
/// temporary buffers. No further encoder call occurs after that error. The
/// returned partial IDs retain their caller's admission through its error owner.
pub(crate) fn tokenize<E: Send, F>(
    trie: &TokTrie,
    bytes: &[u8],
    mode: TokenizationMode,
    encode: F,
) -> Result<(Vec<u32>, usize), TokenizationFailure<E>>
where
    F: FnMut(&str) -> Result<Vec<u32>, E> + Send,
{
    struct Environment<'a, F, E> {
        trie: &'a TokTrie,
        encode: RefCell<F>,
        failure: RefCell<Option<E>>,
    }
    impl<F, E> Environment<'_, F, E>
    where
        F: FnMut(&str) -> Result<Vec<u32>, E>,
    {
        fn encode(&self, text: &str) -> Vec<u32> {
            if self.failure.borrow().is_some() {
                return Vec::new();
            }
            match (self.encode.borrow_mut())(text) {
                Ok(ids) => ids,
                Err(error) => {
                    *self.failure.borrow_mut() = Some(error);
                    Vec::new()
                }
            }
        }
    }
    impl<F, E: Send> TokenizerEnv for Environment<'_, F, E>
    where
        F: FnMut(&str) -> Result<Vec<u32>, E> + Send,
    {
        fn tok_trie(&self) -> &TokTrie {
            self.trie
        }
        fn tokenize_bytes(&self, bytes: &[u8]) -> Vec<u32> {
            self.trie
                .tokenize_with_greedy_fallback(bytes, |text| self.encode(text))
        }
        fn tokenize_bytes_special(&self, bytes: &[u8]) -> Vec<u32> {
            self.trie.tokenize_with_greedy_fallback(bytes, |text| {
                self.trie
                    .tokenize_with_special(text, |text| self.encode(text))
            })
        }
    }
    let environment = Environment {
        trie,
        encode: RefCell::new(encode),
        failure: RefCell::new(None),
    };
    let (ids, fixed) = match mode {
        TokenizationMode::Plain => (environment.tokenize_bytes(bytes), 0),
        TokenizationMode::Special => (environment.tokenize_bytes_special(bytes), 0),
        TokenizationMode::Marker => environment.tokenize_bytes_marker(bytes),
    };
    match environment.failure.into_inner() {
        Some(cause) => Err(TokenizationFailure {
            cause,
            partial: ids,
        }),
        None => Ok((ids, fixed)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llguidance::toktrie::TokRxInfo;
    use std::convert::Infallible;
    fn trie() -> TokTrie {
        let mut tokens: Vec<Vec<u8>> = (0..=255).map(|byte| vec![byte]).collect();
        tokens.push(b"\xff<end>".to_vec());
        TokTrie::from(&TokRxInfo::new(257, 256), &tokens)
    }
    #[test]
    fn bytes_special_tokens_and_numeric_markers_keep_distinct_semantics() {
        let trie = trie();
        let encode = |text: &str| Ok::<_, Infallible>(text.bytes().map(u32::from).collect());
        let plain = tokenize(&trie, "é<end>".as_bytes(), TokenizationMode::Plain, encode).unwrap();
        assert_eq!(plain, (vec![195, 169, 60, 101, 110, 100, 62], 0));
        let special = tokenize(
            &trie,
            "é<end>".as_bytes(),
            TokenizationMode::Special,
            encode,
        )
        .unwrap();
        assert_eq!(special, (vec![195, 169, 256], 0));
        assert_eq!(
            tokenize(&trie, b"a\xfe<end>", TokenizationMode::Special, encode).unwrap(),
            (vec![97, 254, 256], 0)
        );
        assert_eq!(
            tokenize(
                &trie,
                b"a\xff[17]b\xff<end>c",
                TokenizationMode::Marker,
                encode
            )
            .unwrap(),
            (vec![97, 17, 98, 256, 99], 4)
        );
        assert_eq!(
            tokenize(&trie, b"\xff[9999]", TokenizationMode::Marker, encode).unwrap(),
            (b"[9999]".iter().map(|&b| u32::from(b)).collect(), 0)
        );
    }
    #[test]
    fn first_encoding_failure_stops_callbacks_and_returns_partial_ids() {
        let trie = trie();
        let mut calls = 0;
        let failure = tokenize(
            &trie,
            b"a\xff[17]b\xff[19]c",
            TokenizationMode::Marker,
            |text| {
                calls += 1;
                if calls == 2 {
                    Err("second chunk")
                } else {
                    Ok(text.bytes().map(u32::from).collect())
                }
            },
        )
        .unwrap_err();
        assert_eq!(calls, 2);
        assert_eq!(failure.cause, "second chunk");
        // The upstream marker worker can still append literal IDs after the
        // error; the whole returned prefix belongs to the failure, never output.
        assert_eq!(failure.partial, [97, 17, 19]);
    }
}
