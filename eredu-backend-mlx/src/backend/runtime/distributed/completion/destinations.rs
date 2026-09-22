//! The same registry/quarantine policy with a destination paid before submission.
use super::prepared::ResourceCustody;
use std::mem::size_of;

#[derive(Debug)]
struct Node<T> {
    value: Option<T>,
    next: Option<Box<Node<T>>>,
    custody: Option<ResourceCustody>,
    ordinary: Option<eredu_core::HostPreparationAuthority>,
}
/// An empty, one-use slot. Its allocation is released before value and funding.
#[derive(Debug)]
pub(super) struct Destination<T>(Option<Box<Node<T>>>);
impl<T> Destination<T> {
    pub(super) fn control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Node<T>>(), // actual Box allocation
            size_of::<Node<T>>(), // construction/unboxed retirement transport
            size_of::<Self>(),
            size_of::<Option<T>>(),
            size_of::<Option<Box<Node<T>>>>(),
            size_of::<(&mut Destinations<T>, &mut Option<Box<Node<T>>>)>(),
            size_of::<Iter<'_, T>>(),
            size_of::<IterMut<'_, T>>(),
            size_of::<(usize, bool)>(),
        ];
        parts
            .into_iter()
            .try_fold(std::mem::size_of_val(&parts), usize::checked_add)
    }
    pub(super) fn new(custody: Option<ResourceCustody>) -> Self {
        Self(Some(Box::new(Node {
            value: None,
            next: None,
            custody,
            ordinary: None,
        })))
    }
    pub(super) fn ordinary(host: eredu_core::HostPreparationAuthority) -> Self {
        Self(Some(Box::new(Node {
            value: None,
            next: None,
            custody: None,
            ordinary: Some(host),
        })))
    }
}
impl<T> Drop for Destination<T> {
    fn drop(&mut self) {
        if let Some(node) = self.0.take() {
            let Node {
                value,
                next,
                custody,
                ordinary,
            } = *node;
            debug_assert!(next.is_none());
            drop(value);
            drop(next);
            drop(custody);
            drop(ordinary);
        }
    }
}
#[derive(Debug)]
pub(super) struct Destinations<T> {
    head: Option<Box<Node<T>>>,
    len: usize,
}
impl<T> Default for Destinations<T> {
    fn default() -> Self {
        Self::new()
    }
}
impl<T> Destinations<T> {
    pub(super) const fn new() -> Self {
        Self { head: None, len: 0 }
    }
    pub(super) fn len(&self) -> usize {
        self.len
    }
    pub(super) fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Ordinary callers retain their existing allocation contract.
    pub(super) fn push(&mut self, value: T) {
        self.push_prepared(value, Destination::new(None));
    }
    /// Moves into the already allocated node; never grows a container.
    pub(super) fn push_prepared(&mut self, value: T, mut destination: Destination<T>) {
        let mut node = destination.0.take().expect("unused completion destination");
        node.value = Some(value);
        let mut link = &mut self.head;
        while let Some(existing) = link {
            link = &mut existing.next;
        }
        *link = Some(node);
        self.len = self
            .len
            .checked_add(1)
            .expect("completion destination count overflow");
    }
    pub(super) fn iter(&self) -> Iter<'_, T> {
        Iter(self.head.as_deref())
    }
    pub(super) fn iter_mut(&mut self) -> IterMut<'_, T> {
        IterMut(self.head.as_deref_mut())
    }
    pub(super) fn retain(&mut self, mut retain: impl FnMut(&T) -> bool) {
        let mut link = &mut self.head;
        while let Some(node) = link.as_ref() {
            // Evaluate before unlinking: a panicking observer leaves the entire
            // remaining queue under its iterative destructor and paid owners.
            if retain(
                node.value
                    .as_ref()
                    .expect("occupied completion destination"),
            ) {
                link = &mut link.as_mut().expect("retained node").next;
            } else {
                let mut node = link.take().expect("removed node");
                *link = node.next.take();
                self.len -= 1;
                // Unbox first; then retire actual values before their funding.
                let Node {
                    value,
                    next,
                    custody,
                    ordinary,
                } = *node;
                drop(value);
                drop(next);
                drop(custody);
                drop(ordinary);
            }
        }
    }
    pub(super) fn clear(&mut self) {
        self.retain(|_| false);
    }
}
impl<T> Drop for Destinations<T> {
    fn drop(&mut self) {
        self.clear();
    }
}
pub(super) struct Iter<'a, T>(Option<&'a Node<T>>);
impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        let node = self.0.take()?;
        self.0 = node.next.as_deref();
        node.value.as_ref()
    }
}
pub(super) struct IterMut<'a, T>(Option<&'a mut Node<T>>);
impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;
    fn next(&mut self) -> Option<Self::Item> {
        let node = self.0.take()?;
        self.0 = node.next.as_deref_mut();
        node.value.as_mut()
    }
}
impl<'a, T> IntoIterator for &'a mut Destinations<T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}
#[cfg(test)]
impl<T> std::ops::Index<usize> for Destinations<T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        self.iter().nth(index).expect("completion index")
    }
}
#[cfg(test)]
impl<T> std::ops::IndexMut<usize> for Destinations<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        self.iter_mut().nth(index).expect("completion index")
    }
}
