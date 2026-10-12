//! Facade-owned reconciliation of replaceable tentative output.

use eredu_core::{
    refinement::{DraftEvent, DraftRevision, GenerationPosition},
    TextProgress,
};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_BRANCH: AtomicU64 = AtomicU64::new(1);

pub(super) struct DraftJournal {
    enabled: bool,
    branch: u64,
    active: Option<(u64, u64)>,
}

impl DraftJournal {
    pub(super) fn new(enabled: bool) -> Self {
        Self {
            enabled,
            branch: NEXT_BRANCH
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                    value.checked_add(1)
                })
                .expect("draft branch identity exhausted"),
            active: None,
        }
    }

    pub(super) fn fork(&self) -> Self {
        Self::new(self.enabled)
    }

    pub(super) fn progress(
        &mut self,
        progress: &mut TextProgress,
        position: GenerationPosition,
        epoch: u64,
    ) -> Option<DraftEvent> {
        let token_ids = progress.tentative_token_ids.take()?;
        if !self.enabled {
            return None;
        }
        self.active = Some((epoch, progress.block));
        Some(DraftEvent::Updated {
            draft: DraftRevision {
                branch: self.branch,
                epoch,
                block: progress.block,
                revision: progress.iteration,
                committed_start: position.committed_tokens,
                token_ids,
            },
        })
    }

    // A finalizer can replace any tentative position under evolving constraints.
    // Clear the whole block before publishing its first authoritative commitment.
    pub(super) fn clear(&mut self) -> Option<DraftEvent> {
        self.active
            .take()
            .map(|(epoch, block)| DraftEvent::Cleared {
                branch: self.branch,
                epoch,
                block,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::refinement::GenerationPhase;

    #[test]
    fn restore_and_fork_reconcile_drafts_without_reusing_output_identity() {
        let mut parent = DraftJournal::new(true);
        let mut child = parent.fork();
        let progress = || TextProgress {
            phase: GenerationPhase::Refinement,
            block: 3,
            iteration: 2,
            tentative_token_ids: Some(vec![7, 4]),
        };
        let position = GenerationPosition {
            committed_tokens: 9,
            ..Default::default()
        };
        let first = parent.progress(&mut progress(), position, 0).unwrap();
        let DraftEvent::Updated { draft: original } = first else {
            panic!()
        };
        assert_eq!(original.committed_start, 9);
        assert_eq!(
            parent.clear(),
            Some(DraftEvent::Cleared {
                branch: original.branch,
                epoch: 0,
                block: 3
            })
        );
        assert_eq!(parent.clear(), None);
        let DraftEvent::Updated { draft: restored } =
            parent.progress(&mut progress(), position, 1).unwrap()
        else {
            panic!()
        };
        let DraftEvent::Updated { draft: forked } =
            child.progress(&mut progress(), position, 0).unwrap()
        else {
            panic!()
        };
        assert_eq!(restored.branch, original.branch);
        assert!(restored.epoch > original.epoch);
        assert_ne!(forked.branch, original.branch);
        let mut disabled = DraftJournal::new(false);
        let mut raw = progress();
        assert!(disabled.progress(&mut raw, position, 0).is_none());
        assert!(raw.tentative_token_ids.is_none());
        assert!(disabled.clear().is_none());
    }
}
