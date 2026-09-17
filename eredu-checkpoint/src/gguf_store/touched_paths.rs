use super::*;
use std::alloc::Layout;

#[derive(Debug, Clone, Copy)]
pub(super) struct ShardCoordinate {
    pub checkpoint: usize,
    pub shard: usize,
}
impl ShardCoordinate {
    fn path(self, materializers: &[TensorMaterializer]) -> &Path {
        materializers[self.checkpoint].shards()[self.shard].path()
    }
}

#[derive(Debug)]
pub(super) struct TouchedEntry {
    representative: ShardCoordinate,
    first_success: Option<ShardCoordinate>,
}

#[derive(Debug)]
pub(super) struct TouchedPaths {
    entries: Vec<TouchedEntry>,
    requested: Layout,
    touched: usize,
}
impl TouchedPaths {
    pub fn new(materializers: &[TensorMaterializer]) -> Result<Self, StoreError> {
        let count = materializers
            .iter()
            .try_fold(0usize, |n, m| n.checked_add(m.shards().len()))
            .ok_or_else(overflow)?;
        let requested = Layout::array::<TouchedEntry>(count).map_err(|_| overflow())?;
        let mut entries = Vec::with_capacity(count);
        Self::fill(&mut entries, materializers);
        Ok(Self {
            entries,
            requested,
            touched: 0,
        })
    }

    pub(super) fn from_prepared(
        mut entries: Vec<TouchedEntry>,
        materializers: &[TensorMaterializer],
    ) -> Self {
        let requested =
            Layout::array::<TouchedEntry>(entries.capacity()).expect("prechecked touched layout");
        Self::fill(&mut entries, materializers);
        Self {
            entries,
            requested,
            touched: 0,
        }
    }
    fn fill(entries: &mut Vec<TouchedEntry>, materializers: &[TensorMaterializer]) {
        for (checkpoint, materializer) in materializers.iter().enumerate() {
            for shard in 0..materializer.shards().len() {
                entries.push(TouchedEntry {
                    representative: ShardCoordinate { checkpoint, shard },
                    first_success: None,
                });
            }
        }
        order_touched(entries, materializers);
        entries.dedup_by(|a, b| {
            a.representative.path(materializers) == b.representative.path(materializers)
        });
    }

    pub(super) fn construction_controls() -> Option<usize> {
        use std::mem::size_of;
        [
            size_of::<TouchedSift>(),
            size_of::<std::iter::Rev<std::ops::Range<usize>>>(),
            size_of::<&mut [TouchedEntry]>(),
            size_of::<&[TensorMaterializer]>(),
            size_of::<std::iter::Enumerate<std::slice::Iter<'static, TensorMaterializer>>>(),
            size_of::<std::ops::Range<usize>>(),
            size_of::<&Path>(),
            size_of::<&Path>(),
            size_of::<std::cmp::Ordering>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }
    pub fn capacity_layout(&self) -> Result<Layout, StoreError> {
        let actual =
            Layout::array::<TouchedEntry>(self.entries.capacity()).map_err(|_| overflow())?;
        debug_assert!(actual.size() >= self.requested.size());
        Ok(actual)
    }

    pub fn find(&self, materializers: &[TensorMaterializer], path: &Path) -> usize {
        self.entries
            .binary_search_by(|entry| entry.representative.path(materializers).cmp(path))
            .expect("every immutable catalog shard has a prepared coordinate class")
    }

    pub fn mark(&mut self, index: usize, source: ShardCoordinate) {
        let entry = &mut self.entries[index];
        if entry.first_success.is_none() {
            entry.first_success = Some(source);
            self.touched += 1;
        }
    }

    pub fn paths<'a>(
        &'a self,
        materializers: &'a [TensorMaterializer],
    ) -> impl ExactSizeIterator<Item = &'a Path> {
        TouchedIter {
            entries: self.entries.iter(),
            materializers,
            remaining: self.touched,
        }
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.touched == 0
    }
}

fn overflow() -> StoreError {
    StoreError::Overflow {
        context: "GGUF touched coordinate layout".into(),
    }
}

struct TouchedIter<'a> {
    entries: std::slice::Iter<'a, TouchedEntry>,
    materializers: &'a [TensorMaterializer],
    remaining: usize,
}
impl<'a> Iterator for TouchedIter<'a> {
    type Item = &'a Path;
    fn next(&mut self) -> Option<Self::Item> {
        for entry in self.entries.by_ref() {
            if let Some(source) = entry.first_success {
                self.remaining -= 1;
                return Some(source.path(self.materializers));
            }
        }
        None
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}
impl ExactSizeIterator for TouchedIter<'_> {}

#[cfg(test)]
mod tests;

struct TouchedSift {
    root: usize,
    end: usize,
    child: usize,
}
fn order_touched(entries: &mut [TouchedEntry], materializers: &[TensorMaterializer]) {
    fn sift(values: &mut [TouchedEntry], root: usize, end: usize, m: &[TensorMaterializer]) {
        let mut frame = TouchedSift {
            root,
            end,
            child: 0,
        };
        while frame.root < frame.end / 2 {
            frame.child = frame.root * 2 + 1;
            if frame.child + 1 < frame.end
                && values[frame.child].representative.path(m)
                    < values[frame.child + 1].representative.path(m)
            {
                frame.child += 1;
            }
            if values[frame.root].representative.path(m)
                >= values[frame.child].representative.path(m)
            {
                break;
            }
            values.swap(frame.root, frame.child);
            frame.root = frame.child;
        }
    }
    for root in (0..entries.len() / 2).rev() {
        sift(entries, root, entries.len(), materializers);
    }
    for end in (1..entries.len()).rev() {
        entries.swap(0, end);
        sift(entries, 0, end, materializers);
    }
}
