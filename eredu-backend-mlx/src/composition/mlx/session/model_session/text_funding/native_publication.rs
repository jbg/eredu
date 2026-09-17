//! Finite terminal transactions retained by the same actual Work recovery.
use super::*;
use crate::backend::runtime::residency::storage::{
    native_storage::BankOwner, PendingNativePublication,
};

struct Node {
    pending: PendingNativePublication,
    next: Option<Box<Node>>,
}
#[derive(Default)]
pub(super) struct NativePublications {
    refused_inventory: RefCell<Option<RetainedStorage>>,
    head: RefCell<Option<Box<Node>>>,
    failed: Cell<bool>,
}
struct Retain<'a> {
    node: Option<Box<Node>>,
    owner: &'a NativePublications,
}
impl Drop for Retain<'_> {
    fn drop(&mut self) {
        let mut node = self.node.take().expect("owned publication transaction");
        node.next = self.owner.head.borrow_mut().take();
        *self.owner.head.borrow_mut() = Some(node);
    }
}
impl Drop for NativePublications {
    fn drop(&mut self) {
        // No recursive list teardown. Every node and its allocation retire
        // before the enclosing Work's accepted original control guard.
        drop(self.refused_inventory.get_mut().take());
        let mut head = self.head.get_mut().take();
        while let Some(mut node) = head {
            head = node.next.take();
            drop(node);
        }
    }
}
impl NativePublications {
    pub(super) fn publish(
        &self,
        inventory: RetainedStorage,
        bank: &BankOwner,
        scope: &mut WorkingMemoryFundingScope,
        controls: &OriginalTextControlGuard,
        fixed: Option<&OriginalResidentResetSource>,
        prepared: Option<&eredu_runtime::input::OriginalPreparedWorkspaceSource>,
    ) -> Result<RetainedStoragePublication, Error> {
        if self.failed.replace(true) {
            return Err(Error::PrefillControl(WorkingMemoryError::ExecutionFenced));
        }
        let attempt = {
            let mut bank = match bank.try_borrow_mut() {
                Ok(bank) => bank,
                Err(_) => {
                    // No attempt or node exists yet. This terminal refusal has
                    // the same actual inventory custody as a claim refusal.
                    *self.refused_inventory.borrow_mut() = Some(inventory);
                    return Err(Error::PrefillScopeReentrant);
                }
            };
            bank.claim_publication(scope)
        };
        let attempt = match attempt {
            Ok(attempt) => attempt,
            Err(cause) => {
                // Exhaustion/health refusal creates no new node or native
                // preparation. Preserve the first actual inventory in place.
                *self.refused_inventory.borrow_mut() = Some(inventory);
                return Err(Error::PrefillControl(cause));
            }
        };
        // The global selected count is spent BEFORE this one fixed node. Its
        // managed node layout is included in the selected collector contribution.
        let mut retained = Retain {
            node: Some(Box::new(Node {
                pending: PendingNativePublication::new(inventory, attempt),
                next: None,
            })),
            owner: self,
        };
        let result = retained
            .node
            .as_mut()
            .expect("retained transaction")
            .pending
            .publish(scope, controls, fixed, prepared);
        if result.is_ok() {
            self.failed.set(false);
        }
        result
    }
}

/// One actual pending Box and its construction/return/unwind transports.
pub(super) fn control_bytes(rows: usize) -> Option<u64> {
    use std::mem::size_of;
    let controls = [
        size_of::<Node>(),
        size_of::<Node>(),
        size_of::<Box<Node>>(),
        size_of::<Option<Box<Node>>>(),
        size_of::<Retain<'static>>(),
        size_of::<Result<RetainedStoragePublication, Error>>(),
        size_of::<std::cell::RefMut<'static, Option<Box<Node>>>>(),
    ];
    let own = controls
        .into_iter()
        .try_fold(std::mem::size_of_val(&controls), usize::checked_add)?;
    u64::try_from(own)
        .ok()?
        .checked_add(PendingNativePublication::control_bytes(rows)?)
}
