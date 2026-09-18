use std::hash::Hash;

/// Indexed prefix nodes share one edge table. Retirement is iterative and a
/// traversal borrows its input: matching a prefix creates no spelling buffer.
#[derive(Clone, Debug)]
pub struct Trie<Label> {
    leaves: Vec<bool>,
    edges: hashbrown::HashMap<(usize, Label), usize>,
}
impl<Label> Default for Trie<Label> {
    fn default() -> Self {
        Self {
            leaves: Vec::new(),
            edges: hashbrown::HashMap::new(),
        }
    }
}
impl<Label: Eq + Hash + Copy> Trie<Label> {
    pub(crate) fn buffer_bytes(edges: usize) -> Option<usize> {
        let nodes = edges.checked_add(1)?;
        let table = Self::default()
            .edges
            .try_reserve_layout(edges)
            .ok()?
            .map_or(0, |l| l.size());
        std::alloc::Layout::array::<bool>(nodes)
            .ok()?
            .size()
            .checked_add(table)
    }
    pub(crate) fn reserve_nodes(
        &mut self,
        nodes: usize,
    ) -> Result<(), std::collections::TryReserveError> {
        self.leaves
            .try_reserve_exact(nodes.saturating_sub(self.leaves.len()))
    }
    pub(crate) fn reserve_edges(&mut self, edges: usize) -> Result<(), hashbrown::TryReserveError> {
        self.edges
            .try_reserve(edges.saturating_sub(self.edges.len()))
    }
    pub(crate) fn allocated_bytes(&self) -> Option<usize> {
        self.leaves
            .capacity()
            .checked_add(self.edges.allocation_size())
    }
    pub fn push(&mut self, element: &[Label]) {
        // Ordinary callers use the same indexed insertion; the source compiler
        // already reserves its complete destinations before entering this loop.
        let mut at = 0;
        if self.leaves.is_empty() {
            self.leaves.push(false);
        }
        for &label in element {
            at = if let Some(&next) = self.edges.get(&(at, label)) {
                next
            } else {
                let next = self.leaves.len();
                self.leaves.push(false);
                self.edges.insert((at, label), next);
                next
            };
        }
        self.leaves[at] = true;
    }
    pub fn common_prefix_lengths<T: Iterator<Item = Label>>(
        &self,
        iterator: T,
    ) -> TrieIterator<'_, Label, T> {
        TrieIterator {
            trie: self,
            at: 0,
            length: 0,
            iterator,
            done: self.leaves.is_empty(),
        }
    }
}
pub struct TrieIterator<'a, Label, T> {
    trie: &'a Trie<Label>,
    at: usize,
    length: usize,
    iterator: T,
    done: bool,
}
impl<Label: Eq + Hash + Copy, T: Iterator<Item = Label>> Iterator for TrieIterator<'_, Label, T> {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        if self.done {
            return None;
        }
        loop {
            let Some(label) = self.iterator.next() else {
                self.done = true;
                return None;
            };
            let Some(&next) = self.trie.edges.get(&(self.at, label)) else {
                self.done = true;
                return None;
            };
            self.at = next;
            self.length += 1;
            if self.trie.leaves[self.at] {
                return Some(self.length);
            }
        }
    }
}
