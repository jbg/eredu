//! Concrete controlled-session row and immutable-state owners.
use super::*;
use eredu_core::{HostPreparationAuthority, SpeculativeBuffer};
use std::{
    alloc::Layout,
    cell::Cell,
    mem::size_of,
    ops::{Deref, Index},
    rc::Rc,
};

/// Stable increasing identifiers, compact live rows and independently paid
/// replacement storage. Releasing a row never rewinds an identifier or account.
pub(super) struct ControlTable<T> {
    rows: SpeculativeBuffer<(u64, T)>,
}
impl<T> Default for ControlTable<T> {
    fn default() -> Self {
        Self {
            rows: SpeculativeBuffer::default(),
        }
    }
}
impl<T> ControlTable<T> {
    fn position(&self, id: &u64) -> Result<usize, usize> {
        self.rows.binary_search_by_key(id, |(id, _)| *id)
    }
    pub(super) fn contains_key(&self, id: &u64) -> bool {
        self.position(id).is_ok()
    }
    pub(super) fn get_mut(&mut self, id: &u64) -> Option<&mut T> {
        self.position(id).ok().map(|index| &mut self.rows[index].1)
    }
    pub(super) fn remove(&mut self, id: &u64) -> Option<T> {
        self.position(id)
            .ok()
            .and_then(|index| self.rows.remove(index))
            .map(|(_, value)| value)
    }
    fn next_capacity(&self) -> Result<Option<usize>, SpeculativeControlError> {
        if self.rows.len() < self.rows.capacity() {
            return Ok(None);
        }
        self.rows
            .capacity()
            .checked_mul(2)
            .map(|n| Some(n.max(1)))
            .ok_or(ExecutionControlError::Overflow.into())
    }
    pub(super) fn growth_bytes<E: SpeculativeExecutor>(
        &self,
        executor: &E,
    ) -> Result<usize, SpeculativeControlError> {
        match self.next_capacity()? {
            None => Ok(0),
            Some(count) => executor
                .driver_buffer_bytes::<(u64, T)>(count)
                .ok_or(ExecutionControlError::UnknownEstimate.into()),
        }
    }
    pub(super) fn prepare_insert<E: SpeculativeExecutor>(
        &mut self,
        executor: &E,
        context: E::Context<'_>,
    ) -> Result<(), SpeculativeControlError> {
        self.prepare_insert_with(|count| {
            executor.driver_buffer(count, context).map_err(|e| {
                SpeculativeControlError::backend_with_retained(e, E::take_retained_failure)
            })
        })
    }
    fn prepare_insert_with(
        &mut self,
        construct: impl FnOnce(usize) -> Result<SpeculativeBuffer<(u64, T)>, SpeculativeControlError>,
    ) -> Result<(), SpeculativeControlError> {
        let Some(count) = self.next_capacity()? else {
            return Ok(());
        };
        let mut rows = construct(count)?;
        // Admission and destination allocation precede moving any live row.
        rows.try_extend(std::mem::take(&mut self.rows))
            .map_err(|_| SpeculativeControlError::Invalid("snapshot table destination capacity"))?;
        self.rows = rows;
        Ok(())
    }
    pub(super) fn insert(&mut self, id: u64, value: T) -> Result<(), SpeculativeControlError> {
        if self.rows.last().is_some_and(|(last, _)| *last >= id) {
            return Err(SpeculativeControlError::Invalid(
                "snapshot identifier order",
            ));
        }
        self.rows
            .try_push((id, value))
            .map_err(|_| SpeculativeControlError::Invalid("snapshot table destination capacity"))
    }
}
impl<T> Index<&u64> for ControlTable<T> {
    type Output = T;
    fn index(&self, id: &u64) -> &T {
        &self.rows[self
            .position(id)
            .expect("validated controlled state handle")]
        .1
    }
}

/// No Weak/raw Rc escapes. The final alias frees the shared shell before state,
/// reservation and paying host authority are destroyed in that order.
pub(super) struct SavedOwner<E: SpeculativeExecutor, S: SpeculativeSampling, C>(
    Option<Rc<Saved<E, S, C>>>,
);
impl<E: SpeculativeExecutor, S: SpeculativeSampling, C> SavedOwner<E, S, C> {
    pub(super) fn new(value: Saved<E, S, C>) -> Self {
        Self(Some(Rc::new(value)))
    }
    pub(super) fn control_bytes() -> Option<usize> {
        let shared = Layout::new::<[Cell<usize>; 2]>()
            .extend(Layout::new::<Saved<E, S, C>>())
            .ok()?
            .0
            .pad_to_align()
            .size();
        [
            shared,
            size_of::<Self>(),
            size_of::<Saved<E, S, C>>(),
            size_of::<Option<Saved<E, S, C>>>(),
            size_of::<Rc<Saved<E, S, C>>>(),
            size_of::<Result<Self, SpeculativeControlError>>(),
            size_of::<PreparedSave>(),
            size_of::<Result<PreparedSave, SpeculativeControlError>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }
}
impl<E: SpeculativeExecutor, S: SpeculativeSampling, C> Clone for SavedOwner<E, S, C> {
    fn clone(&self) -> Self {
        Self(Some(Rc::clone(self.0.as_ref().expect("live saved owner"))))
    }
}
impl<E: SpeculativeExecutor, S: SpeculativeSampling, C> Deref for SavedOwner<E, S, C> {
    type Target = Saved<E, S, C>;
    fn deref(&self) -> &Self::Target {
        self.0.as_deref().expect("live saved owner")
    }
}
impl<E: SpeculativeExecutor, S: SpeculativeSampling, C> Drop for SavedOwner<E, S, C> {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Rc::into_inner(owner));
        }
    }
}

pub(super) struct PreparedSave {
    pub(super) estimate: SnapshotEstimate,
    pub(super) reservation: crate::execution_control::PendingSnapshotReservation,
    pub(super) host: HostPreparationAuthority,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    #[derive(Debug)]
    struct Custody(Arc<AtomicUsize>);
    impl Drop for Custody {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn controlled_table_refusal_preserves_rows_copy_debits_and_final_custody() {
        let retired = Arc::new(AtomicUsize::new(0));
        let host = HostPreparationAuthority::retain(Custody(retired.clone()));
        let budget = SnapshotBudget::new_with_host(
            SnapshotLimits {
                max_snapshots: 2,
                max_branches: 1,
                retained_bytes: 20,
                cumulative_copy_bytes: 30,
            },
            host.clone(),
        );
        let mut table = ControlTable::default();
        let first = budget
            .reserve_pending(
                SnapshotResourceKind::Snapshot,
                Some(SnapshotEstimate {
                    retained_bytes: 4,
                    copy_bytes: 7,
                }),
            )
            .unwrap();
        table
            .prepare_insert_with(|count| {
                Ok(SpeculativeBuffer::try_new_retained(count, host.clone()).unwrap())
            })
            .unwrap();
        table
            .insert(4, first.publish_with_host(host.clone()))
            .unwrap();
        let second = budget
            .reserve_pending(
                SnapshotResourceKind::Snapshot,
                Some(SnapshotEstimate {
                    retained_bytes: 5,
                    copy_bytes: 9,
                }),
            )
            .unwrap();
        assert!(table
            .prepare_insert_with(|_| Err(SpeculativeControlError::Invalid(
                "injected metadata refusal"
            )))
            .is_err());
        drop(second);
        assert!(table.contains_key(&4));
        assert_eq!(table[&4].retained_bytes(), 4);
        assert_eq!(
            budget.usage(),
            SnapshotUsage {
                snapshots: 1,
                branches: 0,
                retained_bytes: 4,
                cumulative_copy_bytes: 16,
            }
        );
        // A fresh attempt keeps the earlier consumed copy amount, uses a real
        // new destination, and leaves identity lookup/removal unchanged.
        let retry = budget
            .reserve_pending(
                SnapshotResourceKind::Snapshot,
                Some(SnapshotEstimate {
                    retained_bytes: 5,
                    copy_bytes: 9,
                }),
            )
            .unwrap();
        table
            .prepare_insert_with(|count| {
                Ok(SpeculativeBuffer::try_new_retained(count, host.clone()).unwrap())
            })
            .unwrap();
        table
            .insert(9, retry.publish_with_host(host.clone()))
            .unwrap();
        let alias = table[&4].clone();
        drop(table.remove(&4));
        drop(table.remove(&9));
        assert!(!table.contains_key(&4));
        assert_eq!(
            budget.usage(),
            SnapshotUsage {
                snapshots: 1,
                branches: 0,
                retained_bytes: 4,
                cumulative_copy_bytes: 25,
            }
        );
        drop(table);
        drop(budget);
        drop(host);
        assert_eq!(retired.load(Ordering::SeqCst), 0);
        // The last lease alias retires its shell, refunds against the still-live
        // initialized budget, then releases both host-custody aliases.
        drop(alias);
        assert_eq!(retired.load(Ordering::SeqCst), 1);
    }
}
