//! Fixed completion storage and typed admission for the common tile driver.
use super::{SubmittedQuantizationTile, BOUNDED_QUANTIZATION_TILE_BUFFERS};
use eredu_runtime::working_memory::WorkingMemoryError;

#[derive(Debug, thiserror::Error)]
pub(in super::super) enum PipelineAdmissionError<E> {
    #[error("tile pipeline controls: {0}")]
    Policy(#[source] WorkingMemoryError),
    #[error("tile pipeline admission: {0}")]
    Admission(#[source] super::super::layout::CacheAdmissionError),
    #[error("tile pipeline: {0}")]
    Producer(#[source] E),
}

/// Fixed window plus insertion/removal and writeback transport. The producer's
/// native completion storage has its own admission; source/recipe/overlay
/// metadata and the caller's larger preparation frames are separate owners.
pub(in super::super) fn required_bytes<C>() -> Result<usize, WorkingMemoryError> {
    [
        size_of::<TileWindow<C>>(),
        size_of::<SubmittedQuantizationTile<C>>(),
        size_of::<Option<SubmittedQuantizationTile<C>>>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
    .ok_or(WorkingMemoryError::Overflow)
}

pub(super) struct TileWindow<C> {
    slots: [Option<SubmittedQuantizationTile<C>>; BOUNDED_QUANTIZATION_TILE_BUFFERS],
    head: usize,
    len: usize,
}

impl<C> TileWindow<C> {
    pub(super) fn new() -> Self {
        Self {
            slots: std::array::from_fn(|_| None),
            head: 0,
            len: 0,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }

    pub(super) fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(super) fn push_back(&mut self, tile: SubmittedQuantizationTile<C>) {
        assert!(self.len < self.slots.len(), "bounded tile window is full");
        let index = (self.head + self.len) % self.slots.len();
        self.slots[index] = Some(tile);
        self.len += 1;
    }

    pub(super) fn pop_front(&mut self) -> Option<SubmittedQuantizationTile<C>> {
        if self.is_empty() {
            return None;
        }
        let tile = self.slots[self.head].take();
        self.head = (self.head + 1) % self.slots.len();
        self.len -= 1;
        tile
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = &SubmittedQuantizationTile<C>> {
        (0..self.len).map(|offset| {
            self.slots[(self.head + offset) % self.slots.len()]
                .as_ref()
                .expect("occupied tile window slot")
        })
    }
}

impl<C> Drop for TileWindow<C> {
    fn drop(&mut self) {
        // Preserve logical FIFO retirement when the ring has wrapped. Each
        // completion's own Drop retains unfinished native work independently.
        while let Some(tile) = self.pop_front() {
            drop(tile);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::RefCell, rc::Rc};

    struct Completion(usize, Rc<RefCell<Vec<usize>>>);
    impl Drop for Completion {
        fn drop(&mut self) {
            self.1.borrow_mut().push(self.0);
        }
    }

    #[test]
    fn wrapped_window_abandons_completions_in_submission_order() {
        let retired = Rc::new(RefCell::new(Vec::new()));
        let tile = |index| SubmittedQuantizationTile {
            completion: Completion(index, retired.clone()),
            output_start: index,
            rows: 1,
            planned_working_set_bytes: 320,
            output_shard: 0,
        };
        let mut window = TileWindow::new();
        window.push_back(tile(0));
        window.push_back(tile(1));
        drop(window.pop_front().unwrap());
        window.push_back(tile(2));
        assert_eq!(
            window
                .iter()
                .map(|tile| tile.output_start)
                .collect::<Vec<_>>(),
            [1, 2]
        );
        drop(window);
        assert_eq!(*retired.borrow(), [0, 1, 2]);
    }
}
