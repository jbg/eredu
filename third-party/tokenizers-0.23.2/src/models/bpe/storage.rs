//! Read-only storage dispatch for the existing BPE algorithm.
use super::{
    super::OrderedVocabIter,
    model::{MergeMap, Vocab},
    Pair,
};
use ahash::AHashMap;
use serde::{Serialize, Serializer};

#[derive(Clone, Debug)]
pub(super) struct VocabEntry {
    pub id: u32,
    pub start: usize,
    pub len: usize,
    pub ordinal: usize,
}
#[derive(Clone, Debug)]
pub(super) struct MergeEntry {
    pub pair: Pair,
    pub value: (u32, u32),
}
#[derive(Clone, Debug, Default)]
pub(super) struct Packed {
    pub entries: Vec<VocabEntry>,
    pub bytes: Vec<u8>,
    pub ids: Vec<usize>,
    pub merges: Vec<MergeEntry>,
}
impl Packed {
    pub fn spelling(&self, entry: &VocabEntry) -> &str {
        std::str::from_utf8(&self.bytes[entry.start..entry.start + entry.len])
            .expect("compiled BPE spelling is UTF-8")
    }
}

#[derive(Clone)]
pub(super) enum Storage {
    Legacy {
        vocab: Vocab,
        reverse: AHashMap<u32, String>,
        merges: MergeMap,
    },
    Packed(Packed),
}
impl Storage {
    pub fn id(&self, token: &str) -> Option<&u32> {
        match self {
            Self::Legacy { vocab, .. } => vocab.get(token),
            Self::Packed(p) => p
                .entries
                .binary_search_by(|e| p.spelling(e).cmp(token))
                .ok()
                .map(|i| &p.entries[i].id),
        }
    }
    pub fn token(&self, id: u32) -> Option<&str> {
        match self {
            Self::Legacy { reverse, .. } => reverse.get(&id).map(String::as_str),
            Self::Packed(p) => p
                .ids
                .binary_search_by_key(&id, |&i| p.entries[i].id)
                .ok()
                .map(|i| p.spelling(&p.entries[p.ids[i]])),
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Self::Legacy { vocab, .. } => vocab.len(),
            Self::Packed(p) => p.entries.len(),
        }
    }
    pub fn reverse_len(&self) -> usize {
        match self {
            Self::Legacy { reverse, .. } => reverse.len(),
            Self::Packed(p) => p.ids.len(),
        }
    }
    pub fn vocab(&self) -> VocabIter<'_> {
        match self {
            Self::Legacy { vocab, .. } => VocabIter::Legacy(vocab.iter()),
            Self::Packed(p) => VocabIter::Packed(p, p.entries.iter()),
        }
    }
    pub fn merges(&self) -> MergeIter<'_> {
        match self {
            Self::Legacy { merges, .. } => MergeIter::Legacy(merges.iter()),
            Self::Packed(p) => MergeIter::Packed(p.merges.iter()),
        }
    }
    pub fn merge_len(&self) -> usize {
        match self {
            Self::Legacy { merges, .. } => merges.len(),
            Self::Packed(p) => p.merges.len(),
        }
    }
    #[cfg(test)]
    pub fn legacy_vocab(&self) -> &Vocab {
        match self {
            Self::Legacy { vocab, .. } => vocab,
            _ => panic!("expected Legacy storage"),
        }
    }
    #[cfg(test)]
    pub fn legacy_merges(&self) -> &MergeMap {
        match self {
            Self::Legacy { merges, .. } => merges,
            _ => panic!("expected Legacy storage"),
        }
    }
}

pub(super) trait MergeLookup {
    fn get(&self, pair: &Pair) -> Option<&(u32, u32)>;
}
impl MergeLookup for MergeMap {
    fn get(&self, pair: &Pair) -> Option<&(u32, u32)> {
        AHashMap::get(self, pair)
    }
}
impl MergeLookup for Storage {
    fn get(&self, pair: &Pair) -> Option<&(u32, u32)> {
        match self {
            Self::Legacy { merges, .. } => merges.get(pair),
            Self::Packed(p) => p
                .merges
                .binary_search_by_key(pair, |e| e.pair)
                .ok()
                .map(|i| &p.merges[i].value),
        }
    }
}
impl PartialEq for Storage {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len()
            && self.reverse_len() == other.reverse_len()
            && self.merge_len() == other.merge_len()
            && self.vocab().all(|(token, id)| {
                other.id(token) == Some(id) && self.token(*id) == other.token(*id)
            })
            && self
                .merges()
                .all(|(pair, value)| other.get(pair) == Some(value))
    }
}

pub(super) enum VocabIter<'a> {
    Legacy(std::collections::hash_map::Iter<'a, String, u32>),
    Packed(&'a Packed, std::slice::Iter<'a, VocabEntry>),
}
impl<'a> Iterator for VocabIter<'a> {
    type Item = (&'a str, &'a u32);
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Legacy(iter) => iter.next().map(|(s, id)| (s.as_str(), id)),
            Self::Packed(p, iter) => iter.next().map(|e| ((*p).spelling(e), &e.id)),
        }
    }
}
pub(super) enum MergeIter<'a> {
    Legacy(std::collections::hash_map::Iter<'a, Pair, (u32, u32)>),
    Packed(std::slice::Iter<'a, MergeEntry>),
}
impl<'a> Iterator for MergeIter<'a> {
    type Item = (&'a Pair, &'a (u32, u32));
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Legacy(iter) => iter.next(),
            Self::Packed(iter) => iter.next().map(|e| (&e.pair, &e.value)),
        }
    }
}
pub(crate) struct BpeIds<'a>(pub(super) VocabIter<'a>);
impl Iterator for BpeIds<'_> {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        self.0.next().map(|(_, id)| *id)
    }
}

pub(super) struct OrderedStorage<'a>(pub &'a Storage);
impl Serialize for OrderedStorage<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            Storage::Legacy { reverse, .. } => OrderedVocabIter::new(reverse).serialize(serializer),
            Storage::Packed(p) => serializer.collect_map(
                p.ids
                    .iter()
                    .map(|&i| (p.spelling(&p.entries[i]), p.entries[i].id)),
            ),
        }
    }
}
