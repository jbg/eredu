mod raw_tokens;
pub use raw_tokens::{MarkedTokenBytes, RawTokenBytes};
mod decode;
pub use decode::{RawTokenDecodeError, RawTokenDecodeFailure, RawTokenDecodePlan};
pub(crate) mod prepared;

// use 8:24 encoding - num_ch:tok_id (ch_byte:ch_off)* - 8 bytes per tree node
// special case num_ch=0xff -> num_ch=0x100

use core::str;

use bytemuck_derive::{Pod, Zeroable};

use crate::{bytes::to_hex_string, tokenv::parse_numeric_token, SimpleVob};

/// Numeric identifier for a single token in a tokenizer's vocabulary.
pub type TokenId = u32;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Zeroable, Pod)]
#[repr(C)]
pub struct BinTokRxInfo {
    pub vocab_size: u32,
    pub tok_eos: TokenId,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TokRxInfo {
    pub vocab_size: u32,
    pub tok_eos: TokenId,
    pub tok_bos: Option<TokenId>,
    pub tok_pad: Option<TokenId>,
    pub tok_unk: Option<TokenId>,
    pub tok_end_of_turn: Option<TokenId>,
}

impl TokRxInfo {
    pub fn new(vocab_size: u32, tok_eos: TokenId) -> Self {
        TokRxInfo {
            vocab_size,
            tok_eos,
            tok_bos: None,
            tok_pad: None,
            tok_unk: None,
            tok_end_of_turn: None,
        }
    }

    pub fn from_bin(info: &BinTokRxInfo) -> Self {
        TokRxInfo {
            vocab_size: info.vocab_size,
            tok_eos: info.tok_eos,
            tok_bos: None,
            tok_pad: None,
            tok_unk: None,
            tok_end_of_turn: None,
        }
    }

    pub fn to_bin(&self) -> BinTokRxInfo {
        BinTokRxInfo {
            vocab_size: self.vocab_size,
            tok_eos: self.tok_eos,
        }
    }
}

/// Byte-level constraint interface used for trie-based token filtering.
///
/// Implementations maintain a stack of states. The trie walker pushes bytes
/// onto the stack as it descends, queries [`Recognizer::byte_allowed`] or
/// [`Recognizer::try_push_byte`] to test transitions, and pops bytes when
/// backtracking. This lets [`TokTrie`] efficiently compute the set of
/// tokens that satisfy the constraint.
pub trait Recognizer {
    /// for _ in 0..num { stack.pop() }
    fn pop_bytes(&mut self, num: usize);
    /// "Collapse" the stack so that it consists only of its former
    /// top element.
    /// X = stack.top(); stack.empty(); stack.push(X)
    fn collapse(&mut self);
    /// check if stack.top() transitions via byte to a viable state
    fn byte_allowed(&mut self, byte: u8) -> bool {
        if self.try_push_byte(byte) {
            self.pop_bytes(1);
            true
        } else {
            false
        }
    }
    /// Called when iteration over the trie is finished
    /// Stack has exactly one element then, except when iteration started from non-root node.
    /// In that case, the stack may have more than one element, and trie_finished() needs to pop the excessive elements.
    fn trie_finished(&mut self);
    /// Called when iteration over the trie is started
    fn trie_started(&mut self, _dbg_lbl: &str) {}
    /// This combines `push_byte` and `byte_allowed` into one function for performance.
    fn try_push_byte(&mut self, byte: u8) -> bool;
    /// Check if there are any errors to be reported to the user.
    fn get_error(&mut self) -> Option<String> {
        None
    }
    fn save_stats(&mut self, _nodes_walked: usize) {}
}

#[derive(Clone, Copy)]
struct TokDesc {
    len: u32,
    off: u32,
}

/// A prefix tree (trie) of every token in a tokenizer's vocabulary.
///
/// The trie maps byte sequences to [`TokenId`]s and supports efficient
/// constrained-decoding queries: given a [`Recognizer`] that accepts or
/// rejects byte sequences, [`TokTrie::add_bias`] walks the trie and
/// returns the set of tokens whose byte representations are accepted.
#[derive(Clone)]
pub struct TokTrie {
    info: TokRxInfo,
    token_offsets: Vec<TokDesc>,
    token_data: Vec<u8>,
    nodes: Vec<TrieNode>,
    max_token_len: usize,
    eos_tokens: Vec<TokenId>,
    sorted_vocab: Vec<u32>,
}

#[derive(Clone, Copy, Zeroable, Pod)]
#[repr(C)]
pub struct TrieNode {
    // byte:token
    bits: u32,
    bits2: u32,
}

pub const INVALID_TOKEN: TokenId = 0xffff_ffff;

const NO_TOKEN: u32 = 0xffffff;

// PARENT_BITS=10 allows for up to 1024 parents, which is likely enough for tokens up to 2k bytes
// this leaves 32-10 = 22 bits for subtree size, which allows for up to ~2M tokens
// (4M trie nodes)
// GLM4 tokenizer has a token with 1024 spaces - it requires PARENT_BITS >= 9
// Note that because of the ~2M limit, we have ~3 bits left free in 'bits' field
const PARENT_BITS: u32 = 10;
const PARENT_MASK: u32 = (1 << PARENT_BITS) - 1;

impl TrieNode {
    fn new(byte: u8, token_id: u32, num_parents: usize) -> TrieNode {
        assert!(num_parents > 0);
        assert!(num_parents <= (1 << PARENT_BITS) as usize);
        TrieNode {
            bits: (token_id << 8) | byte as u32,
            bits2: (num_parents - 1) as u32,
        }
    }

    #[inline(always)]
    pub fn byte(&self) -> u8 {
        (self.bits & 0xff) as u8
    }

    #[inline(always)]
    pub fn subtree_size(&self) -> usize {
        (self.bits2 >> PARENT_BITS) as usize
    }

    fn set_subtree_size(&mut self, size: usize) {
        assert!(size < (1 << (32 - PARENT_BITS)));
        self.bits2 = (self.bits2 & PARENT_MASK) | ((size as u32) << PARENT_BITS);
    }

    #[inline(always)]
    pub fn num_parents(&self) -> usize {
        ((self.bits2 & PARENT_MASK) + 1) as usize
    }

    #[inline(always)]
    pub fn token_id(&self) -> Option<u32> {
        let r = self.bits >> 8;
        if r == NO_TOKEN {
            None
        } else {
            Some(r)
        }
    }
}

impl TokTrie {
    // see https://github.com/microsoft/llguidance/blob/main/docs/special_tokens.md
    pub const SPECIAL_TOKEN_MARKER: u8 = 0xff;

    pub fn from(info: &TokRxInfo, words: &[Vec<u8>]) -> Self {
        let eos = [info.tok_eos];
        prepared::TokTrieConstructionPlan::from_vecs(info, words, &eos)
            .expect("valid token trie source")
            .compile()
            .expect("token trie construction")
    }

    pub fn filter(&self, filter: &SimpleVob) -> Self {
        prepared::TokTrieConstructionPlan::prepare_filter(self, filter)
            .expect("valid token trie filter source")
            .compile()
            .expect("token trie filter construction")
    }

    pub fn with_eos_token(&self, eos_token: TokenId) -> Self {
        self.with_eos_tokens(&[eos_token])
    }

    pub fn with_eos_tokens(&self, eos_tokens: &[TokenId]) -> Self {
        assert!(!eos_tokens.is_empty(), "eos_tokens must not be empty");
        let vocab = self.vocab_size() as u32;
        for &tok in eos_tokens {
            assert!(
                tok < vocab,
                "EOS token ID {tok} is out of range (vocab_size={vocab})"
            );
        }
        let mut r = self.clone();
        r.info.tok_eos = eos_tokens[0];
        r.eos_tokens = eos_tokens.to_vec();
        r
    }

    pub fn with_info(&self, info: TokRxInfo) -> Self {
        let mut r = self.clone();
        r.info = info;
        r.eos_tokens = vec![info.tok_eos];
        r
    }

    pub fn build_chat_mode_trie(&self) -> Self {
        self.with_eos_token(self.info.tok_end_of_turn.unwrap_or(self.info.tok_eos))
    }

    fn node_offset(&self, n: &TrieNode) -> usize {
        let off = (n as *const _ as usize - self.root() as *const _ as usize)
            / std::mem::size_of::<TrieNode>();
        assert!(off < self.nodes.len());
        off
    }

    fn next_node(&self, n: &TrieNode) -> usize {
        self.node_offset(n) + n.subtree_size()
    }

    pub fn info(&self) -> &TokRxInfo {
        &self.info
    }

    pub fn eos_token(&self) -> TokenId {
        self.info.tok_eos
    }

    pub fn eos_tokens(&self) -> &[TokenId] {
        &self.eos_tokens
    }

    pub fn vocab_size(&self) -> usize {
        self.info.vocab_size as usize
    }

    pub fn alloc_token_set(&self) -> SimpleVob {
        crate::TokenMaskConstructionPlan::for_trie(self)
            .expect("valid trie token mask source")
            .compile()
            .expect("token mask construction")
    }

    pub fn singleton_token_set(&self, tok: TokenId) -> SimpleVob {
        let mut r = self.alloc_token_set();
        r.allow_token(tok);
        r
    }

    /// Returns a token set containing all EOS tokens.
    pub fn eos_token_set(&self) -> SimpleVob {
        let mut r = self.alloc_token_set();
        let vocab = self.vocab_size() as u32;
        for &eos in self.eos_tokens() {
            if eos != INVALID_TOKEN && eos < vocab {
                r.allow_token(eos);
            }
        }
        r
    }

    pub fn token_set_dbg(&self, ts: &SimpleVob) -> String {
        let max_examples = 50;

        let ts_neg = ts.negated();
        let use_neg = ts_neg.num_set() * 10 < ts.num_set();
        let ts1 = if use_neg { &ts_neg } else { ts };
        let num_set = ts1.num_set();
        let max_tok = std::cmp::min(max_examples, num_set);
        let mut token_names = Vec::new();
        // make sure we include EOS first if it's allowed
        if self.info.tok_eos != INVALID_TOKEN && ts1.is_allowed(self.info.tok_eos) {
            token_names.push("EOS".to_string());
        }
        for idx in 0..self.vocab_size() {
            if idx as TokenId != self.info.tok_eos && ts1.is_allowed(idx as TokenId) {
                token_names.push(self.token_dbg(idx as TokenId));
                if token_names.len() >= max_tok {
                    break;
                }
            }
        }
        if token_names.len() < num_set {
            token_names.push("...".to_string());
        }
        format!(
            "TokenSet: {}/{}; {}{}",
            ts.num_set(),
            self.vocab_size(),
            if use_neg { "ALL EXCEPT " } else { "" },
            token_names.join(" ")
        )
    }

    pub fn alloc_logits(&self) -> Vec<f32> {
        vec![0.0; self.vocab_size() + 1]
    }

    pub fn test_trace_tokens(&self, toks: &[u32]) -> String {
        self.tokens_dbg_ext(toks, false)
    }

    pub const MAX_DBG_TOKENS: usize = 200;

    pub fn tokens_dbg(&self, toks: &[u32]) -> String {
        self.tokens_dbg_ext(toks, true)
    }

    fn tokens_dbg_ext(&self, toks: &[u32], quote: bool) -> String {
        // if the token list is too long, we are typically interested in the most recent ones
        let (limited, toks) = if toks.len() > Self::MAX_DBG_TOKENS {
            ("…", &toks[toks.len() - Self::MAX_DBG_TOKENS..])
        } else {
            ("", toks)
        };

        let joined = toks
            .iter()
            .map(|t| self.token_dbg_ext(*t, false))
            .collect::<Vec<_>>()
            .join("‧");

        if quote {
            format!("⟦{limited}{joined}⟧")
        } else if limited.is_empty() {
            joined
        } else {
            format!("{limited}{joined}")
        }
    }

    pub fn token_dbg(&self, idx: u32) -> String {
        self.token_dbg_ext(idx, true)
    }

    fn token_dbg_ext(&self, idx: u32, quote: bool) -> String {
        if idx == self.info.tok_eos {
            "≺EOS≻".to_string()
        } else if idx as usize >= self.vocab_size() {
            format!("≺OOB[{idx}]≻")
        } else {
            // format!("{:?}[{}]", self.token_str(idx), idx)
            let bytes = self.token(idx);
            if bytes.len() > 1 && bytes[0] == TokTrie::SPECIAL_TOKEN_MARKER {
                String::from_utf8_lossy(&bytes[1..]).to_string()
            } else {
                let s = String::from_utf8_lossy(bytes);
                if s.is_empty() {
                    format!("≺EMPTY[{idx}]≻")
                } else if !s.contains('\u{fffd}') {
                    let mut s = format!("{s:?}").replace("\\\"", "\"");
                    s.remove(0);
                    s.pop();
                    if quote {
                        format!("⟨{s}⟩")
                    } else {
                        s
                    }
                } else {
                    let bytes = self.token(idx);
                    format!("≺HEX[{}]≻", to_hex_string(bytes))
                }
            }
        }
    }

    pub fn token_str(&self, idx: u32) -> String {
        String::from_utf8_lossy(self.token(idx)).to_string()
    }

    pub fn token_len(&self, idx: u32) -> usize {
        let t = self.token(idx);
        if t.is_empty() || t[0] == TokTrie::SPECIAL_TOKEN_MARKER {
            let mut idx = idx;
            let mut len = 1;
            while idx >= 10 {
                idx /= 10;
                len += 1;
            }
            // token 1234 -> \xff [ 1234 ]
            len + 3
        } else {
            t.len()
        }
    }

    pub fn token(&self, idx: u32) -> &[u8] {
        if idx >= self.token_offsets.len() as u32 {
            return &[];
        }
        let desc = self.token_offsets[idx as usize];
        let len = desc.len as usize;
        let off = desc.off as usize;
        &self.token_data[off..(off + len)]
    }

    pub fn decode(&self, tokens: &[TokenId]) -> Vec<u8> {
        self.decode_ext(tokens, true)
    }

    pub fn decode_ext(&self, tokens: &[TokenId], include_special: bool) -> Vec<u8> {
        let mut res = Vec::with_capacity(tokens.len() * 6 + 32); // approximately
        for &tok in tokens {
            self.decode_token_with(tok, include_special, &mut |part| {
                res.extend_from_slice(part);
                Ok::<(), std::convert::Infallible>(())
            })
            .unwrap();
        }
        res
    }

    pub fn decode_as_special(&self, tok: TokenId) -> Vec<u8> {
        MarkedTokenBytes::new(tok).bytes().to_vec()
    }

    pub fn decode_raw(&self, tokens: &[TokenId]) -> Vec<u8> {
        let mut result = Vec::with_capacity(tokens.len() * 6 + 32);
        result.extend(self.raw_token_bytes(tokens));
        result
    }

    pub fn decode_str(&self, tokens: &[TokenId]) -> String {
        String::from_utf8_lossy(&self.decode(tokens)).to_string()
    }

    pub fn decode_raw_to_decode(&self, bytes: &[u8]) -> Vec<u8> {
        let mut res = Vec::new();
        self.decode_raw_with(bytes, &mut |part| {
            res.extend_from_slice(part);
            Ok::<(), std::convert::Infallible>(())
        })
        .unwrap();
        res
    }

    pub fn is_special_token(&self, tok: TokenId) -> bool {
        let bytes = self.token(tok);
        !bytes.is_empty() && bytes[0] == TokTrie::SPECIAL_TOKEN_MARKER
    }

    pub fn get_special_token(&self, name: &str) -> Option<TokenId> {
        self.child_at_byte(self.root(), TokTrie::SPECIAL_TOKEN_MARKER)
            .and_then(|n| {
                self.child_at_bytes(n, name.as_bytes())
                    .and_then(|n| n.token_id())
            })
    }

    pub fn get_special_tokens(&self) -> Vec<TokenId> {
        let mut res = Vec::new();
        let pref_node = self
            .child_at_byte(self.root(), TokTrie::SPECIAL_TOKEN_MARKER)
            .expect("missing special token prefix");
        let mut stack = vec![pref_node];
        while let Some(n) = stack.pop() {
            for c in self.node_children(n) {
                if let Some(tok) = c.token_id() {
                    res.push(tok);
                    if res.len() > Self::MAX_DBG_TOKENS + 1 {
                        break;
                    }
                }
                stack.push(c);
            }
        }
        res.remove(0);
        res
    }

    pub fn greedy_tokenize(&self, bytes: &[u8]) -> Vec<TokenId> {
        self.greedy_tokens(bytes).collect()
    }

    /// Tokenize a string, interpreting `<name>` as special tokens.
    pub fn tokenize_with_special<F>(&self, s: &str, str_tokenize: F) -> Vec<TokenId>
    where
        F: Fn(&str) -> Vec<TokenId>,
    {
        let mut out = Vec::new();
        self.visit_special_tokenization(s, |part| -> Result<(), std::convert::Infallible> {
            match part {
                crate::SpecialTokenizationPart::Text(text) => out.extend(str_tokenize(text)),
                crate::SpecialTokenizationPart::Token(token) => out.push(token),
            }
            Ok(())
        })
        .expect("infallible ordinary special-token collector");
        out
    }

    pub fn tokenize_with_greedy_fallback(
        &self,
        bytes: &[u8],
        str_tokenize: impl Fn(&str) -> Vec<TokenId>,
    ) -> Vec<TokenId> {
        match str::from_utf8(bytes) {
            Ok(s) => {
                // fast path
                str_tokenize(s)
            }
            Err(_) => {
                let mut res = vec![];
                Self::visit_utf8_tokenization(
                    bytes,
                    |part| -> Result<(), std::convert::Infallible> {
                        match part {
                            crate::Utf8TokenizationPart::Text(text) => {
                                res.extend(str_tokenize(text))
                            }
                            crate::Utf8TokenizationPart::Greedy(bytes) => {
                                res.extend(self.greedy_tokens(bytes))
                            }
                        }
                        Ok(())
                    },
                )
                .expect("infallible ordinary UTF-8 collector");
                res
            }
        }
    }

    pub fn has_extensions(&self, bytes: &[u8]) -> bool {
        match self.child_at_bytes(self.root(), bytes) {
            None => false,
            Some(n) => n.subtree_size() > 1,
        }
    }

    pub fn token_id(&self, bytes: &[u8]) -> Option<TokenId> {
        let (tok, len) = self.prefix_token_id(bytes);
        // println!("tok_id {:?} {:?} {:?} ", bytes, tok, len);
        if len == bytes.len() {
            Some(tok)
        } else {
            None
        }
    }

    pub fn prefix_token_id(&self, bytes: &[u8]) -> (TokenId, usize) {
        assert!(!bytes.is_empty());
        let mut last = (0, 0);
        let mut n = self.root();
        for (idx, byte) in bytes.iter().enumerate() {
            n = match self.child_at_byte(n, *byte) {
                Some(n) => n,
                None => break,
            };
            if let Some(tok) = n.token_id() {
                last = (tok, idx + 1);
            }
        }
        last
    }

    pub fn max_token_len(&self) -> usize {
        self.max_token_len
    }

    fn validate_with(&self, used: &mut [bool]) {
        assert_eq!(used.len(), self.info.vocab_size as usize);
        assert_eq!(self.root().subtree_size(), self.nodes.len());
        // Check every extent before child traversal so each iterator advances.
        for n in &self.nodes {
            assert!(n.subtree_size() > 0);
            assert!(self.next_node(n) <= self.nodes.len());
        }
        for n in &self.nodes {
            if let Some(tok) = n.token_id() {
                assert!(tok < self.info.vocab_size);
                assert!(!used[tok as usize]);
                used[tok as usize] = true;
            }
            let endp = self.next_node(n);
            for child in self.node_children(n) {
                assert!(self.next_node(child) <= endp);
            }
        }
        for idx in 0..self.info.vocab_size {
            let _ = self.token(idx);
        }
    }

    fn validate(&self) {
        self.validate_with(&mut vec![false; self.info.vocab_size as usize]);
    }

    pub fn root(&self) -> &TrieNode {
        &self.nodes[0]
    }

    pub fn check_against(&self, tokens: &[Vec<u8>]) {
        for (idx, bytes) in tokens.iter().enumerate() {
            let tid = idx as TokenId;
            assert!(bytes == self.token(tid));
            let root = self.root();
            if !bytes.is_empty() {
                let tid2 = self
                    .child_at_bytes(root, bytes)
                    .unwrap()
                    .token_id()
                    .unwrap();
                if tid != tid2 {
                    let par = self
                        .child_at_bytes(root, &bytes[0..bytes.len() - 1])
                        .unwrap();
                    let has_it = self.node_children(par).any(|n| {
                        n.subtree_size() == 1
                            && n.byte() == bytes[bytes.len() - 1]
                            && n.token_id() == Some(tid)
                    });
                    assert!(has_it);
                }
            }
        }
    }

    pub fn child_at_byte<'a>(&'a self, n: &'a TrieNode, byte: u8) -> Option<&'a TrieNode> {
        self.node_children(n).find(|&child| child.byte() == byte)
    }

    pub fn all_subtokens(&self, bytes: &[u8]) -> Vec<TokenId> {
        let mut r = Vec::new();
        for i in 0..bytes.len() {
            let mut n = self.root();
            for &b in &bytes[i..] {
                n = match self.child_at_byte(n, b) {
                    Some(n) => n,
                    None => break,
                };
                if let Some(tok) = n.token_id() {
                    r.push(tok);
                }
            }
        }
        r
    }

    pub fn node_children(&self, n: &TrieNode) -> NodeChildren<'_> {
        let off = self.node_offset(n);
        NodeChildren {
            trie: self,
            current_offset: off + 1,
            end_offset: off + n.subtree_size(),
        }
    }

    pub fn child_at_bytes<'a>(&'a self, mut n: &'a TrieNode, bytes: &[u8]) -> Option<&'a TrieNode> {
        for &byte in bytes {
            n = self.child_at_byte(n, byte)?
        }
        Some(n)
    }

    pub fn token_id_at_bytes(&self, bytes: &[u8]) -> Option<TokenId> {
        self.child_at_bytes(self.root(), bytes)
            .and_then(|n| n.token_id())
    }

    /// Return how many tokens and bytes need to chopped off tokens,
    /// so that we do not limit all possible future tokenizations matching the recognizer.
    pub fn chop_tokens(&self, r: &mut impl Recognizer, tokens: &[TokenId]) -> (usize, usize) {
        let suffix_tokens = &tokens[tokens.len().saturating_sub(4)..];
        let length = self
            .raw_token_bytes_len(suffix_tokens)
            .expect("raw token suffix extent");
        let suffix = self.raw_token_bytes(suffix_tokens);
        for offset in length.saturating_sub(self.max_token_len())..length {
            let mut node = Some(self.root());
            for byte in suffix.clone().skip(offset) {
                node = node.and_then(|node| self.child_at_byte(node, byte));
                if node.is_none() {
                    break;
                }
            }
            if node.is_some_and(|node| self.has_valid_extensions_from_node(r, node)) {
                let chop_bytes = length - offset;
                let mut total = 0;
                for count in 1..=tokens.len() {
                    total += self.token_len(tokens[tokens.len() - count]);
                    if total >= chop_bytes {
                        return (count, total);
                    }
                }
                unreachable!();
            }
        }
        (0, 0)
    }

    /// Check if add_bias() would have returned any tokens.
    #[inline(never)]
    pub fn has_valid_extensions(&self, r: &mut impl Recognizer, start: &[u8]) -> bool {
        self.child_at_bytes(self.root(), start)
            .is_some_and(|node| self.has_valid_extensions_from_node(r, node))
    }

    fn has_valid_extensions_from_node(&self, r: &mut impl Recognizer, n: &TrieNode) -> bool {
        r.trie_started("has_valid_extensions");
        let off = self.node_offset(n);
        let mut p = off + 1;
        let endp = off + n.subtree_size();
        let mut ok = false;
        let mut next_pop = 0;
        while p < endp {
            r.pop_bytes(next_pop);
            let n = &self.nodes[p];
            let b = n.byte();
            if r.try_push_byte(b) {
                if n.token_id().is_some() {
                    ok = true;
                    break;
                }
                next_pop = if n.subtree_size() == 1 {
                    n.num_parents()
                } else {
                    0
                };
                p += 1;
            } else {
                p += n.subtree_size();
                next_pop = n.num_parents() - 1;
            }
        }
        r.trie_finished();
        ok
    }

    pub fn all_prefixes(&self, bytes: &[u8]) -> Vec<TokenId> {
        let mut r = Vec::new();
        let mut n = self.root();
        for &b in bytes {
            if let Some(c) = self.child_at_byte(n, b) {
                n = c;
                if let Some(tok) = n.token_id() {
                    r.push(tok);
                }
            } else {
                break;
            }
        }
        r
    }

    /// Exact host control shells for the existing trie walk. Nonempty prefixes
    /// use the same borrowed fixed recognizer before the caller recognizer. This
    /// excludes the caller's recognizer and mask destinations, which stay owned
    /// by their enclosing source-qualified operation.
    pub fn add_bias_control_bytes<R: Recognizer>(start: &[u8]) -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        fn frame<R>() -> Option<usize> {
            let parts = [
                size_of::<(&TokTrie, &mut R, &mut SimpleVob, &[u8])>(),
                size_of::<Option<&TrieNode>>(),
                size_of::<&TrieNode>(),
                size_of::<(&TokTrie, &mut R, &mut SimpleVob, &TrieNode)>(),
                size_of::<&[TrieNode]>(),
                size_of::<(usize, usize, usize, usize, usize, usize, usize)>(),
                size_of::<(u32, u8, u32, Option<u32>, usize)>(),
                size_of::<(usize, usize)>(),
                size_of::<(&mut R, usize)>(),
                size_of::<(&mut R, u8)>(),
                size_of::<(&TokTrie, &TrieNode, &[u8])>(),
                size_of::<std::slice::Iter<'_, u8>>(),
                size_of::<(&TokTrie, &TrieNode, u8, Option<&TrieNode>)>(),
                size_of::<NodeChildren<'_>>(),
                size_of::<(&TrieNode, u8)>(),
            ];
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
        }
        let base = frame::<R>()?;
        if start.is_empty() {
            Some(base)
        } else {
            base.checked_add(size_of::<FixedRecognizer<'_>>())?
                .checked_add(frame::<FixedRecognizer<'_>>()?)
        }
    }

    pub fn add_bias(&self, r: &mut impl Recognizer, toks: &mut SimpleVob, start: &[u8]) {
        // all prefixes of 'start' are also allowed
        if !start.is_empty() {
            let mut fixed = FixedRecognizer::new(start);
            self.add_bias(&mut fixed, toks, &[]);
        }

        let n = self.child_at_bytes(self.root(), start);
        if n.is_none() {
            return;
        }
        let n = n.unwrap();
        r.trie_started("add_bias");
        let (next_pop, nodes_walked) = self.add_bias_inner(r, toks, n);
        if start.is_empty() {
            // if start was non-empty, trie_finished() is supposed to clean this up
            r.pop_bytes(next_pop);
        }
        r.trie_finished();
        r.save_stats(nodes_walked);
        // Clean up the fake token set by add_bias_inner for nodes without token_id.
        // Note: If add_bias_inner panics, this cleanup won't run, leaving the fake token set.
        // This is acceptable since panics indicate unrecoverable errors and the program state
        // is likely already corrupted.
        let defl_tok = self.vocab_size() as u32;
        toks.disallow_token(defl_tok);
    }

    #[inline(never)]
    fn add_bias_inner(
        &self,
        r: &mut impl Recognizer,
        toks: &mut SimpleVob,
        n: &TrieNode,
    ) -> (usize, usize) {
        // Use a fake token at vocab_size to avoid branching in the hot loop.
        // This is safe because alloc_token_set() allocates capacity for vocab_size + 1 tokens.
        // The fake token is cleaned up in add_bias() after the walk completes.
        let defl_tok = self.vocab_size() as u32;
        let off = self.node_offset(n);
        let total_nodes = n.subtree_size();
        let mut p = off + 1;
        let endp = off + total_nodes;
        let nodes = &self.nodes[..endp];
        let mut next_pop = 0;
        let mut num_skip = 0;
        while p < endp {
            r.pop_bytes(next_pop);
            // The loop bound and this slice share endp; checked indexing retains
            // the exact same node access and lets the optimizer prove the range.
            let n = &nodes[p];
            let b = n.byte();
            if r.try_push_byte(b) {
                // Avoid branching: always set a token (either real or fake at defl_tok)
                let tok = n.token_id().unwrap_or(defl_tok);
                debug_assert!(
                    tok <= self.vocab_size() as u32,
                    "token {} out of valid range (vocab_size: {})",
                    tok,
                    self.vocab_size()
                );
                toks.allow_token(tok);
                next_pop = if n.subtree_size() == 1 {
                    n.num_parents()
                } else {
                    0
                };
                p += 1;
            } else {
                let subtree_size = n.subtree_size();
                p += subtree_size;
                // it's slightly faster to count skipped nodes, than walked nodes
                num_skip += subtree_size - 1;
                next_pop = n.num_parents() - 1;
            }
        }
        (next_pop, total_nodes - num_skip)
    }

    pub fn all_tokens(&self) -> Vec<Vec<u8>> {
        (0..self.vocab_size())
            .map(|idx| self.token(idx as u32).to_vec())
            .collect()
    }

    pub fn sorted_tokens(&self) -> Vec<(u32, Vec<u8>)> {
        let mut res = vec![];
        let n = self.root();
        let off = self.node_offset(n);
        let mut p = off + 1;
        let endp = off + n.subtree_size();
        let mut next_pop = 0;
        let mut bytes = vec![];
        while p < endp {
            bytes.drain(bytes.len() - next_pop..);
            let n = &self.nodes[p];
            let b = n.byte();
            bytes.push(b);
            if let Some(t) = n.token_id() {
                res.push((t, bytes.clone()));
            }
            next_pop = if n.subtree_size() == 1 {
                n.num_parents()
            } else {
                0
            };
            p += 1;
        }
        res
    }

    fn count_until_depth(&self, depth: usize) -> (usize, usize) {
        let mut count = 0;
        let mut num_tokens = 0;
        let mut stack = vec![(self.root(), 0)];
        while let Some((n, d)) = stack.pop() {
            if d == depth {
                continue;
            } else {
                for c in self.node_children(n) {
                    count += 1;
                    if c.token_id().is_some() {
                        num_tokens += 1;
                    }
                    stack.push((c, d + 1));
                }
            }
        }
        (count, num_tokens)
    }

    pub fn trie_stats(&self) -> String {
        let mut nodes_histogram = vec![0; 256];

        let mut token_nodes = 0;

        let n = self.root();
        let off = self.node_offset(n);
        let mut p = off + 1;
        let endp = off + n.subtree_size();
        while p < endp {
            let n = &self.nodes[p];

            if n.token_id().is_some() {
                token_nodes += 1;
            }

            let last_ch = self.next_node(n);
            let mut ch_p = p + 1;
            let mut num_children = 0;

            while ch_p < last_ch {
                let ch = &self.nodes[ch_p];
                ch_p += ch.subtree_size();
                num_children += 1;
            }

            nodes_histogram[std::cmp::min(9, num_children)] += 1;

            p += 1;
        }

        let mut histogram = String::new();

        if false {
            for (idx, num) in nodes_histogram.iter().enumerate() {
                if *num > 0 {
                    if !histogram.is_empty() {
                        histogram.push_str(", ");
                    }
                    histogram.push_str(&format!("{idx}:{num}"));
                }
            }
        }

        if false {
            for n in self.node_children(self.root()) {
                histogram.push_str(&format!(
                    "\n{} => {} {}",
                    n.byte(),
                    self.node_children(n).count(),
                    n.subtree_size()
                ));
            }
        }

        if false {
            for depth in 0..30 {
                let (count, num_tokens) = self.count_until_depth(depth);
                histogram.push_str(&format!(
                    "\ndepth {depth}: {count} nodes {num_tokens} tokens"
                ));
            }
        }

        if !histogram.is_empty() {
            histogram = format!("\n{histogram}");
        }

        format!(
            "{}{} nodes, {} token nodes, {} token bytes, {} max len",
            histogram,
            self.nodes.len(),
            token_nodes,
            self.token_data.len(),
            self.max_token_len,
        )
    }
}

pub struct NodeChildren<'a> {
    trie: &'a TokTrie,
    current_offset: usize,
    end_offset: usize,
}

impl<'a> Iterator for NodeChildren<'a> {
    type Item = &'a TrieNode;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_offset < self.end_offset {
            let node = &self.trie.nodes[self.current_offset];
            self.current_offset += node.subtree_size();
            Some(node)
        } else {
            None
        }
    }
}

const NO_NODE: u32 = 0xffff_ffff;

/// A temporary node used to construct the Trie before it is serialized
/// into the highly-compact `TrieNode` format.
/// Children are stored as a linked list to avoid nested Vec allocations.
struct BuilderNode {
    /// The vocabulary token ID. `NO_TOKEN` if this node doesn't complete a token.
    token_id: u32,
    /// The byte value of this specific node.
    byte: u8,
    /// Index of the first child in the `nodes` arena.
    first_child: u32,
    /// Index of the next sibling in the `nodes` arena.
    next_sibling: u32,
    /// Index of the last child in the `nodes` arena. Caching this makes appending O(1).
    last_child: u32,
}

/// An arena-allocated builder for the tokenizer Trie.
/// It uses a flat array `Vec<BuilderNode>` instead of nested `Vec`s to drastically
/// reduce memory allocations and avoid thread-contention in global allocators.
#[derive(Clone, Copy)]
struct SerializeFrame {
    output: usize,
    child: u32,
    parents: usize,
}

struct TrieBuilder {
    nodes: Vec<BuilderNode>,
    /// Fast O(1) lookup array for the root node's children.
    root_children: [u32; 256],
}

impl TrieBuilder {
    fn empty() -> Self {
        Self {
            nodes: Vec::new(),
            root_children: [NO_NODE; 256],
        }
    }

    fn push_root(&mut self, byte: u8) {
        assert!(self.nodes.is_empty());
        self.nodes.push(BuilderNode {
            token_id: NO_TOKEN,
            byte,
            first_child: NO_NODE,
            next_sibling: NO_NODE,
            last_child: NO_NODE,
        });
    }

    /// Inserts a word into the Trie.
    /// Words *must* be inserted in their original ID order to correctly handle
    /// duplicate tokens (multiple IDs mapping to the exact same byte string).
    fn insert(&mut self, word: &[u8], token_id: u32) {
        if word.is_empty() {
            assert!(self.nodes[0].token_id == NO_TOKEN);
            self.nodes[0].token_id = token_id;
            return;
        }

        let mut curr_node_idx = 0;

        for (i, &byte) in word.iter().enumerate() {
            let is_last_byte = i == word.len() - 1;
            let mut found_existing_path = false;

            // Step 1: Find if this byte already exists among the current node's children.
            if curr_node_idx == 0 {
                // Fast O(1) path for the root node
                let root_child_idx = self.root_children[byte as usize];
                if root_child_idx != NO_NODE {
                    let child_node = &self.nodes[root_child_idx as usize];
                    // If we reach the end of a word, and that node already has a token ID,
                    // we must not overwrite it. We bypass the `found` flag to force the
                    // creation of a new, duplicate sibling node at the end of the linked list.
                    if is_last_byte && child_node.token_id != NO_TOKEN {
                        // This is a duplicate token; fall through to append a new node.
                    } else {
                        curr_node_idx = root_child_idx as usize;
                        found_existing_path = true;
                    }
                }
            } else {
                // Slower linked-list traversal for deeper nodes
                let mut child_idx = self.nodes[curr_node_idx].first_child;
                while child_idx != NO_NODE {
                    let child_node = &self.nodes[child_idx as usize];
                    if child_node.byte == byte {
                        if is_last_byte && child_node.token_id != NO_TOKEN {
                            // Duplicate token; fall through to append.
                        } else {
                            curr_node_idx = child_idx as usize;
                            found_existing_path = true;
                            break;
                        }
                    }
                    child_idx = child_node.next_sibling;
                }
            }

            // Step 2: If the byte doesn't exist (or is a duplicate), create a new node.
            if !found_existing_path {
                let new_node_idx = self.nodes.len() as u32;
                self.nodes.push(BuilderNode {
                    token_id: NO_TOKEN,
                    byte,
                    first_child: NO_NODE,
                    next_sibling: NO_NODE,
                    last_child: NO_NODE,
                });

                // If it's a new unique root child, cache it in the fast lookup array.
                if curr_node_idx == 0 && self.root_children[byte as usize] == NO_NODE {
                    self.root_children[byte as usize] = new_node_idx;
                }

                // Append the new node to the end of the current node's child linked list.
                // Using `last_child` makes this an O(1) operation.
                let last_child_idx = self.nodes[curr_node_idx].last_child;
                if last_child_idx == NO_NODE {
                    self.nodes[curr_node_idx].first_child = new_node_idx;
                } else {
                    self.nodes[last_child_idx as usize].next_sibling = new_node_idx;
                }
                self.nodes[curr_node_idx].last_child = new_node_idx;

                curr_node_idx = new_node_idx as usize;
            }
        }

        // Assign the token ID to the final leaf node.
        self.nodes[curr_node_idx].token_id = token_id;
    }

    // The same depth-first encoding as the former recursive serializer. Its
    // explicit continuation stack can be reserved before original construction.
    fn serialize_into(
        &self,
        data: &mut Vec<TrieNode>,
        stack: &mut Vec<SerializeFrame>,
        num_parents: usize,
    ) -> Result<(), prepared::TokTrieSourceError> {
        let push = |node_idx: usize,
                    parents: usize,
                    data: &mut Vec<TrieNode>,
                    stack: &mut Vec<SerializeFrame>| {
            let node = &self.nodes[node_idx];
            let encoded_parents = parents.max(1);
            if encoded_parents > (1 << PARENT_BITS) {
                return Err(prepared::TokTrieSourceError::NodeEncoding);
            }
            let output = data.len();
            data.push(TrieNode::new(node.byte, node.token_id, encoded_parents));
            stack.push(SerializeFrame {
                output,
                child: node.first_child,
                parents,
            });
            Ok(())
        };
        push(0, num_parents, data, stack)?;
        while let Some(frame) = stack.last_mut() {
            if frame.child == NO_NODE {
                let output = frame.output;
                let size = data.len() - output;
                if size >= (1 << (32 - PARENT_BITS)) {
                    return Err(prepared::TokTrieSourceError::NodeEncoding);
                }
                data[output].set_subtree_size(size);
                stack.pop();
            } else {
                let child = frame.child as usize;
                frame.child = self.nodes[child].next_sibling;
                let parents = if frame.child == NO_NODE {
                    frame.parents + 1
                } else {
                    1
                };
                push(child, parents, data, stack)?;
            }
        }
        Ok(())
    }

    fn serialize(&mut self, data: &mut Vec<TrieNode>, num_parents: usize) {
        self.serialize_into(data, &mut Vec::new(), num_parents)
            .expect("representable token trie");
    }
}
struct FixedRecognizer<'a> {
    bytes: &'a [u8],
    bytes_ptr: usize,
}

impl<'a> FixedRecognizer<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        FixedRecognizer {
            bytes,
            bytes_ptr: 0,
        }
    }
}

impl Recognizer for FixedRecognizer<'_> {
    fn collapse(&mut self) {}
    fn trie_finished(&mut self) {}

    fn pop_bytes(&mut self, num: usize) {
        self.bytes_ptr -= num;
    }

    fn try_push_byte(&mut self, byte: u8) -> bool {
        if self.bytes_ptr < self.bytes.len() && self.bytes[self.bytes_ptr] == byte {
            self.bytes_ptr += 1;
            true
        } else {
            false
        }
    }
}

pub struct AnythingGoes;

impl Recognizer for AnythingGoes {
    fn collapse(&mut self) {}
    fn trie_finished(&mut self) {}
    fn pop_bytes(&mut self, _num: usize) {}
    fn try_push_byte(&mut self, _byte: u8) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepared_filtered_trie_preserves_source_order_mask_backing_and_failure_prefix() {
        use prepared::{TokTrieConstructionPlan, TokTrieSourceError};
        let words = [
            b"ab".as_slice(),
            b"a",
            b"ab",
            b"\xff<stop>",
            b"b",
            b"abc",
            b"a\xff",
        ]
        .iter()
        .map(|word| word.to_vec())
        .collect::<Vec<_>>();
        let mut info = TokRxInfo::new(words.len() as u32, 3);
        info.tok_pad = Some(4);
        let source = TokTrie::from(&info, &words).with_eos_tokens(&[3, 4, 3]);
        let mut mask = source.alloc_token_set();
        for id in [0, 2, 3, 6] {
            mask.allow_token(id);
        }
        let plan = TokTrieConstructionPlan::prepare_filter(&source, &mask).unwrap();
        let required = plan.requirements();
        let filtered = plan.compile().unwrap();
        assert_eq!(filtered.info(), source.info());
        assert_eq!(filtered.eos_tokens(), source.eos_tokens());
        assert_eq!(filtered.sorted_vocab, source.sorted_vocab);
        assert!(required.buffer_bytes() > filtered.token_data.len());
        for id in 0..info.vocab_size {
            assert_eq!(
                filtered.token(id),
                if mask.is_allowed(id) {
                    source.token(id)
                } else {
                    &[]
                }
            );
        }
        assert_eq!(filtered.token_id(b"ab"), Some(0));
        let mut recognizer_mask = source.alloc_token_set();
        filtered.add_bias(&mut FixedRecognizer::new(b"ab"), &mut recognizer_mask, &[]);
        assert!(recognizer_mask.is_allowed(0) && recognizer_mask.is_allowed(2));
        assert!(!recognizer_mask.is_allowed(1));
        // A second filter must preserve the same original ordering and aliases.
        let twice = TokTrieConstructionPlan::prepare_filter(&filtered, &mask)
            .unwrap()
            .compile()
            .unwrap();
        assert_eq!(twice.sorted_vocab, source.sorted_vocab);
        assert_eq!(twice.token_data, filtered.token_data);
        assert_eq!(twice.eos_tokens(), &[3, 4, 3]);
        let mut failed = TokTrieConstructionPlan::prepare_filter(&source, &mask).unwrap();
        failed.failure = Some(7);
        let error = failed.compile().err().unwrap();
        assert!(error.allocation_error().is_some());
        assert!(error.buffer_capacities()[..7].iter().all(|&n| n > 0));
        assert_eq!(error.buffer_capacities()[7], 0);
        assert_eq!(
            TokTrieConstructionPlan::prepare_filter(&source, &SimpleVob::new()).unwrap_err(),
            TokTrieSourceError::Mask
        );
        let empty = source.alloc_token_set();
        let none = TokTrieConstructionPlan::prepare_filter(&source, &empty)
            .unwrap()
            .compile()
            .unwrap();
        assert!(none.token_data.is_empty());
        assert_eq!(none.eos_tokens(), source.eos_tokens());
        assert_eq!(none.sorted_vocab, source.sorted_vocab);
        assert_eq!(none.root().subtree_size(), 1);
        drop((source, mask, words, filtered));
        assert_eq!(twice.token(3), b"\xff<stop>");
        drop(error);
    }

    #[test]
    fn source_prepared_trie_preserves_duplicate_ids_special_bytes_eos_and_real_failure_prefixes() {
        use prepared::{TokTrieConstructionPlan, TokTrieSourceError};
        let words: Vec<Vec<u8>> = [
            b"".as_slice(),
            b"ab",
            b"a",
            b"ab",
            b"\xff<stop>",
            b"b",
            b"abc",
            b"a\xff",
        ]
        .iter()
        .map(|bytes| bytes.to_vec())
        .collect();
        let refs: Vec<_> = words.iter().map(Vec::as_slice).collect();
        let info = TokRxInfo::new(words.len() as u32, 4);
        let eos = [4, 5, 4];
        let plan = TokTrieConstructionPlan::prepare(&info, &refs, &eos).unwrap();
        let requirements = plan.requirements();
        assert_eq!(
            requirements.required_bytes(),
            requirements.buffer_bytes() + requirements.control_bytes()
        );
        let trie = plan.compile().unwrap();
        assert_eq!(trie.info(), &info);
        assert_eq!(trie.eos_tokens(), eos);
        assert_eq!(trie.sorted_vocab, [0, 2, 1, 3, 6, 7, 5, 4]);
        for (id, word) in words.iter().enumerate() {
            assert_eq!(trie.token(id as u32), word);
        }
        assert_eq!(trie.token_id(b"ab"), Some(1));
        assert_eq!(trie.token_id(b"abc"), Some(6));
        let mut mask = trie.alloc_token_set();
        trie.add_bias(&mut FixedRecognizer::new(b"ab"), &mut mask, &[]);
        let selected: Vec<_> = (0..info.vocab_size)
            .filter(|&id| mask.is_allowed(id))
            .collect();
        assert_eq!(selected, [1, 2, 3]);
        let filtered = trie.filter(&mask);
        assert_eq!(filtered.eos_tokens(), eos);
        assert_eq!(filtered.token(1), b"ab");
        assert_eq!(filtered.token(3), b"ab");
        assert_eq!(filtered.token(6), b"");
        let ordinary = TokTrie::from(&info, &words).with_eos_tokens(&eos);
        assert_eq!(trie.nodes.len(), ordinary.nodes.len());
        assert!(trie
            .nodes
            .iter()
            .zip(&ordinary.nodes)
            .all(|(a, b)| (a.bits, a.bits2) == (b.bits, b.bits2)));
        assert_eq!(trie.sorted_vocab, ordinary.sorted_vocab);
        for failure in 0..8 {
            let mut plan = TokTrieConstructionPlan::prepare(&info, &refs, &eos).unwrap();
            plan.failure = Some(failure);
            let error = plan.compile().err().expect("actual selected reserve fails");
            assert!(error.allocation_error().is_some());
            let capacities = error.buffer_capacities();
            assert!(capacities[..failure].iter().all(|&count| count > 0));
            assert!(capacities[failure..].iter().all(|&count| count == 0));
        }
        assert_eq!(
            TokTrieConstructionPlan::prepare(&info, &refs, &[]).unwrap_err(),
            TokTrieSourceError::Eos
        );
        assert_eq!(
            TokTrieConstructionPlan::prepare(&info, &refs, &[99]).unwrap_err(),
            TokTrieSourceError::Eos
        );
        // Actual parent-pop encoding refuses a chain too long only after its
        // paid destinations exist; the partial serialized nodes remain owned.
        let deep_word = vec![b'x'; (1 << PARENT_BITS) + 1];
        let deep = [deep_word.as_slice()];
        let error = TokTrieConstructionPlan::prepare(&TokRxInfo::new(1, 0), &deep, &[0])
            .unwrap()
            .compile()
            .err()
            .expect("unrepresentable actual pop count");
        drop(deep_word);
        assert_eq!(error.source_error(), Some(TokTrieSourceError::NodeEncoding));
        assert!(error.buffer_capacities().iter().all(|&count| count > 0));
        assert!(error.partial.nodes.len() > 0);
        drop((trie, ordinary, words));
        assert_eq!(filtered.token(1), b"ab");
    }

    fn make_test_trie(eos: TokenId) -> TokTrie {
        let info = TokRxInfo::new(4, eos);
        let words = vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec(), b"d".to_vec()];
        TokTrie::from(&info, &words)
    }

    #[test]
    fn test_default_single_eos() {
        let trie = make_test_trie(2);
        assert_eq!(trie.eos_token(), 2);
        assert_eq!(trie.eos_tokens(), &[2]);
    }

    #[test]
    fn test_with_eos_tokens_multiple() {
        let trie = make_test_trie(0).with_eos_tokens(&[1, 3]);
        assert_eq!(trie.eos_token(), 1);
        assert_eq!(trie.eos_tokens(), &[1, 3]);
        assert_eq!(trie.info().tok_eos, 1);
    }

    #[test]
    fn test_with_eos_token_backwards_compat() {
        let trie = make_test_trie(0).with_eos_token(2);
        assert_eq!(trie.eos_token(), 2);
        assert_eq!(trie.eos_tokens(), &[2]);
    }

    #[test]
    fn test_with_info_resets_eos_tokens() {
        let trie = make_test_trie(0).with_eos_tokens(&[1, 2]);
        let trie2 = trie.with_info(TokRxInfo::new(4, 3));
        assert_eq!(trie2.eos_token(), 3);
        assert_eq!(trie2.eos_tokens(), &[3]);
    }

    #[test]
    fn test_filter_preserves_eos_tokens() {
        let trie = make_test_trie(0).with_eos_tokens(&[1, 2]);
        let mut filter = trie.alloc_token_set();
        for i in 0..4 {
            filter.allow_token(i);
        }
        let filtered = trie.filter(&filter);
        assert_eq!(filtered.eos_tokens(), &[1, 2]);
    }

    #[test]
    #[should_panic(expected = "eos_tokens must not be empty")]
    fn test_with_eos_tokens_empty_panics() {
        make_test_trie(0).with_eos_tokens(&[]);
    }

    #[test]
    fn test_eos_token_set_single() {
        let trie = make_test_trie(2);
        let set = trie.eos_token_set();
        assert!(set.is_allowed(2));
        assert!(!set.is_allowed(0));
        assert!(!set.is_allowed(1));
        assert_eq!(set.num_set(), 1);
    }

    #[test]
    fn test_eos_token_set_multiple() {
        let trie = make_test_trie(0).with_eos_tokens(&[1, 3]);
        let set = trie.eos_token_set();
        assert!(set.is_allowed(1));
        assert!(set.is_allowed(3));
        assert!(!set.is_allowed(0));
        assert!(!set.is_allowed(2));
        assert_eq!(set.num_set(), 2);
    }
    #[test]
    fn raw_capture_decode_uses_shared_numeric_special_spelling_and_exact_output() {
        let words = vec![
            b"plain".to_vec(),
            b"\xff<tool>".to_vec(),
            Vec::new(),
            b"\xc3\xa9".to_vec(),
        ];
        let trie = TokTrie::from(&TokRxInfo::new(words.len() as u32, 2), &words);
        assert_eq!(
            trie.decode_ext(&[0, 1, 2, 3, u32::MAX], true),
            b"plain<tool><[2]>\xc3\xa9<[4294967295]>"
        );
        assert_eq!(
            trie.decode_ext(&[0, 1, 2, 3, u32::MAX], false),
            b"plain\xc3\xa9"
        );
        let raw = b"prefix\xff[1]/\xff[2]/\xff[3]/\xff[4294967295]/\xff[bad]/\xff[4294967296]/\xff";
        let expected =
            b"prefix<tool>/<[2]>/\xc3\xa9/<[4294967295]>/\xff[bad]/\xff[4294967296]/\xff";
        assert_eq!(trie.decode_raw_to_decode(raw), expected);
        let plan = trie.raw_decode_plan(raw).unwrap();
        assert_eq!(plan.output_bytes(), expected.len());
        assert!(plan.required_bytes() > plan.output_bytes());
        let decoded = plan.compile().unwrap();
        assert_eq!(decoded, expected);
        assert_eq!(decoded.capacity(), expected.len());
        let empty = trie.raw_decode_plan(b"").unwrap().compile().unwrap();
        assert!(empty.is_empty());
        assert_eq!(empty.capacity(), 0);
        let mut calls = 0;
        let mut prefix = Vec::new();
        let error = trie
            .decode_raw_with(raw, &mut |part| {
                calls += 1;
                if calls == 8 {
                    return Err("capture sink");
                }
                prefix.extend_from_slice(part);
                Ok(())
            })
            .unwrap_err();
        assert_eq!(error, "capture sink");
        assert_eq!(calls, 8);
        assert_eq!(prefix, b"prefix<tool>");
        drop(trie);
        assert_eq!(decoded, expected);
    }
}
