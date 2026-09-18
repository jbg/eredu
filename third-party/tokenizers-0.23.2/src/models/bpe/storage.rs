//! Read-only storage dispatch for the existing BPE algorithm.
use super::{
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
/// Immutable execution tables shared by serde, source compilation and trainers.
#[derive(Clone, Debug, Default)]
pub(super) struct Storage {
    pub entries: Vec<VocabEntry>,
    pub bytes: Vec<u8>,
    pub ids: Vec<usize>,
    pub merges: Vec<MergeEntry>,
}
impl Storage {
    pub fn spelling(&self, entry: &VocabEntry) -> &str {
        std::str::from_utf8(&self.bytes[entry.start..entry.start + entry.len])
            .expect("BPE spelling is UTF-8")
    }
    pub fn from_maps(vocab: Vocab, reverse: AHashMap<u32, String>, merges: MergeMap) -> Self {
        let mut storage = Self::default();
        for (ordinal, (token, id)) in vocab.into_iter().enumerate() {
            let start = storage.bytes.len();
            storage.bytes.extend_from_slice(token.as_bytes());
            storage.entries.push(VocabEntry {
                id,
                start,
                len: token.len(),
                ordinal,
            });
        }
        let bytes = &storage.bytes;
        storage.entries.sort_unstable_by(|a, b| {
            bytes[a.start..a.start + a.len].cmp(&bytes[b.start..b.start + b.len])
        });
        storage.ids = storage
            .entries
            .iter()
            .enumerate()
            .filter_map(|(i, entry)| {
                (reverse.get(&entry.id).map(String::as_str) == Some(storage.spelling(entry)))
                    .then_some(i)
            })
            .collect();
        let entries = &storage.entries;
        storage.ids.sort_unstable_by_key(|&i| entries[i].id);
        storage.merges = merges
            .into_iter()
            .map(|(pair, value)| MergeEntry { pair, value })
            .collect();
        storage.merges.sort_unstable_by_key(|e| e.pair);
        storage
    }
    pub fn id(&self, token: &str) -> Option<&u32> {
        self.entries
            .binary_search_by(|e| self.spelling(e).cmp(token))
            .ok()
            .map(|i| &self.entries[i].id)
    }
    pub fn token(&self, id: u32) -> Option<&str> {
        // The retained ID index is already sorted. Dense IDs can name their
        // slot directly; verify the actual ID before using it so holes and
        // canonical aliases retain the same binary-search behavior.
        if let Some(&index) = self.ids.get(id as usize) {
            let entry = &self.entries[index];
            if entry.id == id {
                return Some(self.spelling(entry));
            }
        }
        self.ids
            .binary_search_by_key(&id, |&i| self.entries[i].id)
            .ok()
            .map(|i| self.spelling(&self.entries[self.ids[i]]))
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn reverse_len(&self) -> usize {
        self.ids.len()
    }
    pub fn vocab(&self) -> VocabIter<'_> {
        VocabIter(self, self.entries.iter())
    }
    pub fn merges(&self) -> MergeIter<'_> {
        MergeIter(self.merges.iter())
    }
    pub fn merge_len(&self) -> usize {
        self.merges.len()
    }
    #[cfg(test)]
    pub fn owned_vocab(&self) -> Vocab {
        self.vocab().map(|(s, id)| (s.to_owned(), *id)).collect()
    }
    #[cfg(test)]
    pub fn owned_merges(&self) -> MergeMap {
        self.merges().map(|(pair, value)| (*pair, *value)).collect()
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
        self.merges
            .binary_search_by_key(pair, |e| e.pair)
            .ok()
            .map(|i| &self.merges[i].value)
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

pub(super) struct VocabIter<'a>(&'a Storage, std::slice::Iter<'a, VocabEntry>);
impl<'a> Iterator for VocabIter<'a> {
    type Item = (&'a str, &'a u32);
    fn next(&mut self) -> Option<Self::Item> {
        self.1.next().map(|e| (self.0.spelling(e), &e.id))
    }
}
pub(super) struct MergeIter<'a>(std::slice::Iter<'a, MergeEntry>);
impl<'a> Iterator for MergeIter<'a> {
    type Item = (&'a Pair, &'a (u32, u32));
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|e| (&e.pair, &e.value))
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
        serializer.collect_map(
            self.0
                .ids
                .iter()
                .map(|&i| (self.0.spelling(&self.0.entries[i]), self.0.entries[i].id)),
        )
    }
}
