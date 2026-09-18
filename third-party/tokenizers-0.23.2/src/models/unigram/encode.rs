use super::model::{Unigram, UnigramError, K_UNK_PENALTY};
use std::{alloc::Layout, collections::TryReserveError, mem::size_of};

#[derive(Debug, Clone, Default)]
struct BestPathNode {
    id: usize,
    best_path_score: f64,
    starts_at: Option<usize>,
}
/// Reusable storage of the original deterministic Unigram Viterbi worker.
#[derive(Debug, Default)]
pub(crate) struct UnigramScratch {
    nodes: Vec<BestPathNode>,
    spans: Vec<(usize, usize)>,
}
impl UnigramScratch {
    pub(crate) fn buffer_bytes(bytes: usize) -> Option<usize> {
        if bytes == 0 {
            return Some(0);
        }
        Layout::array::<BestPathNode>(bytes.checked_add(1)?)
            .ok()?
            .size()
            .checked_add(Layout::array::<(usize, usize)>(bytes).ok()?.size())
    }
    pub(crate) fn control_bytes() -> usize {
        size_of::<Self>()
            + size_of::<BestPathNode>()
            + size_of::<(usize, usize, usize, usize, bool, f64, f64, f64)>()
            + size_of::<Option<(usize, usize)>>()
            + size_of::<super::trie::TrieIterator<'static, u8, std::str::Bytes<'static>>>()
            + size_of::<std::slice::Iter<'static, (usize, usize)>>()
            + size_of::<std::iter::Rev<std::slice::Iter<'static, (usize, usize)>>>()
            + size_of::<Result<(), UnigramError>>()
            + size_of::<[u8; 6]>()
            + size_of::<std::str::Bytes<'static>>()
            + size_of::<std::str::Chars<'static>>()
            + size_of::<Option<&'static u32>>()
            + size_of::<Result<&'static str, std::str::Utf8Error>>()
    }
    pub(crate) fn reserve_nodes(&mut self, count: usize) -> Result<(), TryReserveError> {
        self.nodes
            .try_reserve_exact(count.saturating_sub(self.nodes.len()))
    }
    pub(crate) fn reserve_spans(&mut self, count: usize) -> Result<(), TryReserveError> {
        self.spans
            .try_reserve_exact(count.saturating_sub(self.spans.len()))
    }
    pub(crate) fn capacities(&self) -> [usize; 2] {
        [self.nodes.capacity(), self.spans.capacity()]
    }
    pub(crate) fn spans(&self) -> &[(usize, usize)] {
        &self.spans
    }
    pub(crate) fn encode(&mut self, model: &Unigram, sentence: &str) -> Result<(), UnigramError> {
        self.nodes.clear();
        self.spans.clear();
        if sentence.is_empty() {
            return Ok(());
        }
        let size = sentence.len();
        assert!(self.nodes.capacity() > size && self.spans.capacity() >= size);
        self.nodes.resize(size + 1, BestPathNode::default());
        let unk_score = model.min_score - K_UNK_PENALTY;
        let mut starts_at = 0;
        while starts_at < size {
            let best_path_score_till_here = self.nodes[starts_at].best_path_score;
            let mut has_single_node = false;
            let mblen = sentence[starts_at..].chars().next().unwrap().len_utf8();
            for length in model
                .trie
                .common_prefix_lengths(sentence[starts_at..].bytes())
            {
                let key_pos = starts_at + length;
                let token = &sentence[starts_at..key_pos];
                let target_node = &mut self.nodes[key_pos];
                let id = model.token_to_ids.get(token).unwrap();
                let score = model.vocab.get(*id as usize).unwrap().1;
                let candidate_best_path_score = score + best_path_score_till_here;
                if target_node.starts_at.is_none()
                    || candidate_best_path_score > target_node.best_path_score
                {
                    target_node.best_path_score = candidate_best_path_score;
                    target_node.starts_at = Some(starts_at);
                    target_node.id = *id as usize;
                }
                if !has_single_node && length == mblen {
                    has_single_node = true;
                }
            }
            if !has_single_node {
                let target_node = &mut self.nodes[starts_at + mblen];
                let candidate_best_path_score = unk_score + best_path_score_till_here;
                if target_node.starts_at.is_none()
                    || candidate_best_path_score > target_node.best_path_score
                {
                    target_node.best_path_score = candidate_best_path_score;
                    target_node.starts_at = Some(starts_at);
                    target_node.id = model.unk_id.ok_or(UnigramError::MissingUnkId)?;
                }
            }
            starts_at += mblen;
        }
        let mut ends_at = size;
        let mut unknown_end = None;
        while ends_at > 0 {
            let node = &self.nodes[ends_at];
            let starts_at = node.starts_at.unwrap();
            if model.fuse_unk && Some(node.id) == model.unk_id {
                unknown_end.get_or_insert(ends_at);
            } else {
                if let Some(end) = unknown_end.take() {
                    self.spans.push((ends_at, end));
                }
                self.spans.push((starts_at, ends_at));
            }
            ends_at = starts_at;
        }
        if let Some(end) = unknown_end {
            self.spans.push((0, end));
        }
        self.spans.reverse();
        Ok(())
    }
}
impl Unigram {
    pub(crate) fn has_identity_profile(&self) -> bool {
        self.cache.is_none()
            && self.is_optimized
            && (self.alpha.is_none() || self.alpha == Some(0.0))
    }
    /// Publish each selected span through the original whole-span fallback rule.
    /// Checking all bytes first preserves the ordinary all-or-unknown decision.
    pub(crate) fn visit_piece_ids(
        &self,
        piece: &str,
        mut emit: impl FnMut(u32, &str),
    ) -> Result<(), UnigramError> {
        if let Some(&id) = self.token_to_ids.get(piece) {
            emit(id, piece);
            return Ok(());
        }
        if self.byte_fallback
            && piece
                .bytes()
                .all(|b| self.token_to_ids.contains_key(byte_name(b).as_str()))
        {
            for byte in piece.bytes() {
                let name = byte_name(byte);
                let text = name.as_str();
                emit(*self.token_to_ids.get(text).unwrap(), text);
            }
        } else {
            emit(self.unk_id.ok_or(UnigramError::MissingUnkId)? as u32, piece);
        }
        Ok(())
    }
}
struct ByteName([u8; 6]);
impl ByteName {
    fn as_str(&self) -> &str {
        std::str::from_utf8(&self.0).expect("ASCII byte spelling")
    }
}
fn byte_name(byte: u8) -> ByteName {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    ByteName([
        b'<',
        b'0',
        b'x',
        HEX[(byte >> 4) as usize],
        HEX[(byte & 15) as usize],
        b'>',
    ])
}
