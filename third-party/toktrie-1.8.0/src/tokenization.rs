//! Borrowed parts of the ordinary tokenizer callbacks, without result allocation.
use crate::{TokTrie, TokenId, TrieNode, parse_numeric_token};
use std::mem::{size_of, size_of_val};

/// The same longest-prefix greedy walk; missing bytes are skipped, as in the
/// ordinary tokenizer fallback. It neither allocates nor grants destination space.
pub struct GreedyTokens<'a> {
    trie: &'a TokTrie,
    bytes: &'a [u8],
    offset: usize,
}
impl Iterator for GreedyTokens<'_> {
    type Item = TokenId;
    fn next(&mut self) -> Option<TokenId> {
        while self.offset < self.bytes.len() {
            let mut node = self.trie.root();
            let mut last_token = None;
            let mut last = self.offset;
            for j in self.offset..self.bytes.len() {
                if let Some(child) = self.trie.child_at_byte(node, self.bytes[j]) {
                    node = child;
                    if let Some(token) = node.token_id() {
                        last_token = Some(token);
                        last = j;
                    }
                } else {
                    break;
                }
            }
            self.offset = last + 1;
            if last_token.is_some() {
                return last_token;
            }
        }
        None
    }
}
/// Exact ordinary valid UTF-8 callback or invalid-byte greedy fallback span.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Utf8TokenizationPart<'a> {
    /// Supplied to the actual tokenizer, including an entirely empty input.
    Text(&'a str),
    /// Invalid UTF-8 fragment supplied to the same trie greedy walk.
    Greedy(&'a [u8]),
}
/// String callback boundaries after ordinary special-token recognition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpecialTokenizationPart<'a> {
    /// A literal segment, including an unrecognized `<name>`.
    Text(&'a str),
    /// An exact special-token lookup in this trie.
    Token(TokenId),
}
/// Ordinary marker callback boundaries. Raw byte spans are deliberately not
/// split: a TokenizerEnv implementation still sees exactly its original calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarkerTokenizationPart<'a> {
    /// One unchanged normal span before the next special marker.
    Bytes(&'a [u8]),
    /// A recognized marked spelling or in-domain numeric token.
    Token(TokenId),
}
impl TokTrie {
    /// Borrows the ordinary greedy walk without allocating an intermediate Vec.
    pub fn greedy_tokens<'a>(&'a self, bytes: &'a [u8]) -> GreedyTokens<'a> {
        GreedyTokens {
            trie: self,
            bytes,
            offset: 0,
        }
    }
    /// Visits exact ordinary valid/invalid spans. Error stops before any later
    /// callback; completed output custody belongs to the visitor.
    pub fn visit_utf8_tokenization<'a, E>(
        bytes: &'a [u8],
        mut visit: impl FnMut(Utf8TokenizationPart<'a>) -> Result<(), E>,
    ) -> Result<(), E> {
        match std::str::from_utf8(bytes) {
            Ok(text) => visit(Utf8TokenizationPart::Text(text)),
            Err(_) => {
                for chunk in bytes.utf8_chunks() {
                    if !chunk.valid().is_empty() {
                        visit(Utf8TokenizationPart::Text(chunk.valid()))?;
                    }
                    if !chunk.invalid().is_empty() {
                        visit(Utf8TokenizationPart::Greedy(chunk.invalid()))?;
                    }
                }
                Ok(())
            }
        }
    }
    /// Visits the existing `<name>` recognition worker in its original order.
    pub fn visit_special_tokenization<'a, E>(
        &self,
        text: &'a str,
        mut visit: impl FnMut(SpecialTokenizationPart<'a>) -> Result<(), E>,
    ) -> Result<(), E> {
        let max_len = 100;
        let bytes = text.as_bytes();
        let mut last = 0;
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] != b'<' {
                i += 1;
                continue;
            }
            let mut valid = true;
            let mut j = i + 1;
            let mut len_inside = 0;
            while j < bytes.len() && len_inside < max_len {
                match bytes[j] {
                    b'<' => {
                        valid = false;
                        break;
                    }
                    b'>' => break,
                    _ => {
                        len_inside += 1;
                        j += 1;
                    }
                }
            }
            if !valid || j >= bytes.len() || bytes[j] != b'>' || len_inside == 0 {
                i += 1;
                continue;
            }
            let name = &text[i..=j];
            if let Some(token) = self.get_special_token(name) {
                if last < i {
                    visit(SpecialTokenizationPart::Text(&text[last..i]))?;
                }
                visit(SpecialTokenizationPart::Token(token))?;
            } else {
                visit(SpecialTokenizationPart::Text(&text[last..=j]))?;
            }
            i = j + 1;
            last = i;
        }
        if last < bytes.len() {
            visit(SpecialTokenizationPart::Text(&text[last..]))?;
        }
        Ok(())
    }
    /// Visits the existing marker worker, preserving malformed-marker fallback,
    /// the 100-byte spelling scan, 20-byte numeric scan and vocabulary check.
    pub fn visit_marker_tokenization<'a, E>(
        &self,
        bytes: &'a [u8],
        mut visit: impl FnMut(MarkerTokenizationPart<'a>) -> Result<(), E>,
    ) -> Result<(), E> {
        let mut idx = 0;
        let ff = Self::SPECIAL_TOKEN_MARKER;
        while idx < bytes.len() {
            let normal_len = bytes[idx..]
                .iter()
                .position(|&x| x == ff)
                .unwrap_or(bytes.len() - idx);
            if normal_len != 0 {
                visit(MarkerTokenizationPart::Bytes(&bytes[idx..idx + normal_len]))?;
                idx += normal_len;
            }
            idx += 1;
            if idx + 2 < bytes.len() && bytes[idx] == b'<' {
                let spec_len = bytes[idx..std::cmp::min(bytes.len(), idx + 100)]
                    .iter()
                    .position(|&x| x == b'>');
                if let Some(mut spec_len) = spec_len {
                    spec_len += 1;
                    let spec_token = &bytes[idx - 1..idx + spec_len];
                    if let Some(id) = self.token_id_at_bytes(spec_token) {
                        visit(MarkerTokenizationPart::Token(id))?;
                        idx += spec_len;
                    }
                }
            } else if idx < bytes.len() {
                if let Some((n_bytes, id)) = parse_numeric_token(&bytes[idx..]) {
                    if id < self.vocab_size() as u32 {
                        visit(MarkerTokenizationPart::Token(id))?;
                        idx += n_bytes;
                    }
                }
            }
        }
        Ok(())
    }
    /// Fixed borrowed worker frames for an actual visitor/result type. This is
    /// inspection only; callback encoding and output storage need their own bounds.
    pub fn tokenization_control_bytes<V, E>() -> Option<usize> {
        let parts = [
            size_of::<V>(),
            size_of::<E>(),
            size_of::<Result<(), E>>(),
            size_of::<GreedyTokens<'_>>(),
            size_of::<Utf8TokenizationPart<'_>>(),
            size_of::<SpecialTokenizationPart<'_>>(),
            size_of::<MarkerTokenizationPart<'_>>(),
            size_of::<std::str::Utf8Chunks<'_>>(),
            size_of::<std::str::Utf8Chunk<'_>>(),
            size_of::<Result<&str, std::str::Utf8Error>>(),
            size_of::<Option<(usize, TokenId)>>(),
            size_of::<(&TokTrie, &[u8], &mut V)>(),
            size_of::<(&TokTrie, &str, &mut V)>(),
            size_of::<(&TrieNode, usize, Option<TokenId>, usize)>(),
            size_of::<(usize, usize, usize, usize, bool)>(),
            size_of::<std::slice::Iter<'_, u8>>(),
            size_of::<std::ops::Range<usize>>(),
            size_of::<Result<u32, std::num::ParseIntError>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ApproximateTokEnv, TokEnv, TokRxInfo, TokenizerEnv};
    use std::{convert::Infallible, sync::Mutex};

    struct Probe {
        env: TokEnv,
        calls: Mutex<Vec<Vec<u8>>>,
    }
    impl TokenizerEnv for Probe {
        fn tok_trie(&self) -> &TokTrie {
            self.env.tok_trie()
        }
        fn tokenize_bytes(&self, bytes: &[u8]) -> Vec<TokenId> {
            self.calls.lock().unwrap().push(bytes.to_vec());
            self.env.tokenize_bytes(bytes)
        }
    }
    #[test]
    fn shared_tokenization_parts_keep_callback_boundaries_markers_and_failed_prefix() {
        let probe = Probe {
            env: ApproximateTokEnv::single_byte_env(),
            calls: Mutex::new(Vec::new()),
        };
        let bytes = b"a\xff<|tool|>b\xff[65]c\xff[9999]d\xff<bad>\xe2\x82e\xff";
        let (ids, fixed) = probe.tokenize_bytes_marker(bytes);
        let mut expected = vec![u32::from(b'a'), 256, u32::from(b'b'), 65];
        expected.extend(b"c[9999]d<bad>\xe2\x82e".iter().map(|&x| u32::from(x)));
        assert_eq!(ids, expected);
        assert_eq!(fixed, 4);
        assert_eq!(
            *probe.calls.lock().unwrap(),
            vec![
                b"a".to_vec(),
                b"b".to_vec(),
                b"c".to_vec(),
                b"[9999]d".to_vec(),
                b"<bad>\xe2\x82e".to_vec()
            ]
        );

        let text = "é<|tool|>🙂<bad><><x<|user|>>tail";
        let mut parts = Vec::new();
        probe
            .tok_trie()
            .visit_special_tokenization(text, |part| -> Result<(), Infallible> {
                parts.push(part);
                Ok(())
            })
            .unwrap();
        assert_eq!(
            parts,
            vec![
                SpecialTokenizationPart::Text("é"),
                SpecialTokenizationPart::Token(256),
                SpecialTokenizationPart::Text("🙂<bad>"),
                SpecialTokenizationPart::Text("<><x"),
                SpecialTokenizationPart::Token(258),
                SpecialTokenizationPart::Text(">tail")
            ]
        );
        let actual = probe
            .tok_trie()
            .tokenize_with_special(text, |s| s.bytes().map(u32::from).collect());
        let mut expected = Vec::new();
        for part in parts {
            match part {
                SpecialTokenizationPart::Text(s) => expected.extend(s.bytes().map(u32::from)),
                SpecialTokenizationPart::Token(t) => expected.push(t),
            }
        }
        assert_eq!(actual, expected);

        let mut utf8 = Vec::new();
        TokTrie::visit_utf8_tokenization(b"a\xe2\x82b\xff", |part| -> Result<(), Infallible> {
            utf8.push(part);
            Ok(())
        })
        .unwrap();
        assert_eq!(
            utf8,
            vec![
                Utf8TokenizationPart::Text("a"),
                Utf8TokenizationPart::Greedy(b"\xe2\x82"),
                Utf8TokenizationPart::Text("b"),
                Utf8TokenizationPart::Greedy(b"\xff")
            ]
        );
        let mut empty_calls = 0;
        TokTrie::visit_utf8_tokenization(b"", |part| -> Result<(), Infallible> {
            assert_eq!(part, Utf8TokenizationPart::Text(""));
            empty_calls += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(empty_calls, 1);
        let mut accepted = Vec::new();
        let failure = probe.tok_trie().visit_marker_tokenization(bytes, |part| {
            if accepted.len() == 2 {
                return Err(17);
            }
            accepted.push(part);
            Ok(())
        });
        assert_eq!(failure, Err(17));
        assert_eq!(
            accepted,
            vec![
                MarkerTokenizationPart::Bytes(b"a"),
                MarkerTokenizationPart::Token(256)
            ]
        );
        let sparse = TokTrie::from(
            &TokRxInfo::new(3, 2),
            &[b"ab".to_vec(), b"a".to_vec(), b"bc".to_vec()],
        );
        assert_eq!(
            sparse.greedy_tokens(b"abxabc").collect::<Vec<_>>(),
            vec![0, 0]
        );
        assert_eq!(sparse.greedy_tokenize(b"abxabc"), vec![0, 0]);
        assert!(
            TokTrie::tokenization_control_bytes::<&mut Vec<TokenId>, Infallible>().unwrap() > 0
        );
    }
}
