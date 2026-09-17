//! Balanced indexing of already owned nodes, without allocation during mutation.
//!
//! Node construction is an ordinary allocating prerequisite. Callers using a
//! bounded allocator policy must protect its actual layout and nested key/value
//! storage before construction. This collection supplies no budget or authority.
use std::{alloc::Layout, cmp::Ordering};

const MAX_DEPTH: usize = usize::BITS as usize * 2;
type Link<K, V, C> = Option<PreparedIndexNode<K, V, C>>;

#[derive(Debug)]
struct Node<K, V, C> {
    key: K,
    value: V,
    left: Link<K, V, C>,
    right: Link<K, V, C>,
    height: usize,
    keep: bool,
}

// These are the actual owning/borrowed recursion frames used below. Their
// bounded simultaneous population is separate from every node's heap layout.
struct RelinkFrame<K, V, C> {
    node: Link<K, V, C>,
    left: Link<K, V, C>,
    right: Link<K, V, C>,
    len: usize,
}
struct ObserveFrame<'a, K, V, C> {
    node: &'a mut PreparedIndexNode<K, V, C>,
    left: usize,
}

/// One independently allocated node. Its custody retires after its Box, including
/// the Box allocation itself. Nested key/value storage has the same outer custody.
#[derive(Debug)]
pub struct PreparedIndexNode<K, V, C> {
    node: Box<Node<K, V, C>>,
    _custody: C,
}
impl<K, V, C> PreparedIndexNode<K, V, C> {
    /// Allocate one ordinary Box after the caller's construction prerequisites.
    pub fn new(key: K, value: V, custody: C) -> Self {
        Self {
            node: Box::new(Node {
                key,
                value,
                left: None,
                right: None,
                height: 1,
                keep: true,
            }),
            _custody: custody,
        }
    }
    /// Exact requested Box payload. Allocator qualification and nested storage
    /// are the caller's owning producer contribution, never inferred here.
    pub const fn storage_layout() -> Layout {
        Layout::new::<Node<K, V, C>>()
    }
    /// Borrow the original key without reconstructing it.
    pub fn key(&self) -> &K {
        &self.node.key
    }
    /// Borrow the original value without moving its physical node.
    pub fn value(&self) -> &V {
        &self.node.value
    }
}

/// A balanced map whose insertion and pruning only move preexisting owners.
/// Duplicate insertion returns the exact supplied node without replacing a value.
#[derive(Debug)]
pub struct PreparedIndex<K, V, C> {
    root: Link<K, V, C>,
    len: usize,
}
impl<K, V, C> Default for PreparedIndex<K, V, C> {
    fn default() -> Self {
        Self { root: None, len: 0 }
    }
}
impl<K, V, C> PreparedIndex<K, V, C> {
    /// Number of indexed nodes.
    pub const fn len(&self) -> usize {
        self.len
    }
    /// Whether the index is empty.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Borrowed lookup; the comparison observes the intact tree and must order
    /// the requested key relative to the supplied stored key.
    pub fn get_by(&self, mut compare: impl FnMut(&K) -> Ordering) -> Option<&V> {
        let mut current = self.root.as_ref();
        while let Some(node) = current {
            match compare(&node.node.key) {
                Ordering::Less => current = node.node.left.as_ref(),
                Ordering::Greater => current = node.node.right.as_ref(),
                Ordering::Equal => return Some(&node.node.value),
            }
        }
        None
    }
    /// Borrow every key/value in sorted order, using fixed traversal storage.
    /// No node, key or provider-owned value is cloned.
    pub fn iter(&self) -> PreparedIndexIter<'_, K, V, C> {
        PreparedIndexIter {
            pending: self.root.as_ref(),
            path: [None; MAX_DEPTH],
            depth: 0,
        }
    }
    /// Actual fixed iterator representation, separate from node allocations.
    pub const fn iteration_control_layout() -> Layout {
        Layout::new::<PreparedIndexIter<'_, K, V, C>>()
    }
    /// Visit borrowed values without allocating or changing node ownership.
    /// A false result stops the traversal; a callback panic leaves the tree intact.
    pub fn all_values(&self, mut predicate: impl FnMut(&V) -> bool) -> bool {
        fn visit<K, V, C>(node: &Link<K, V, C>, predicate: &mut impl FnMut(&V) -> bool) -> bool {
            node.as_ref().is_none_or(|node| {
                visit(&node.node.left, predicate)
                    && predicate(&node.node.value)
                    && visit(&node.node.right, predicate)
            })
        }
        visit(&self.root, &mut predicate)
    }
    /// Observe every row before structural mutation. If observation unwinds,
    /// ownership, ordering and population remain intact. A successful pass
    /// detaches stale nodes and rebuilds only by moving prepared owners in O(n).
    /// The returned owners must be destroyed outside the caller's locks.
    pub fn extract_if(
        &mut self,
        mut stale: impl FnMut(&K, &V) -> bool,
    ) -> DetachedIndexNodes<K, V, C> {
        let kept = observe(&mut self.root, &mut stale);
        if kept == self.len {
            return DetachedIndexNodes::default();
        }
        let mut kept_head = None;
        let mut retired = DetachedIndexNodes::default();
        flatten(self.root.take(), &mut kept_head, &mut retired.head);
        self.root = rebuild(&mut kept_head, kept);
        self.len = kept;
        retired
    }
    /// Maximum named traversal depth of an AVL tree addressable by usize.
    /// Node/key/value layouts and the caller's closure/result transports are
    /// separate. This constant describes the worker, not an allocation grant.
    pub const fn maximum_depth() -> usize {
        MAX_DEPTH
    }
    /// Conservative simultaneous storage for the actual named worker frames:
    /// one fixed insertion path, bounded owner/borrow recursion, and rotation/
    /// drain transports. Does not include caller closures or an ABI/process
    /// stack guarantee. Heap nodes and their nested payloads are separate.
    pub fn worker_control_layout() -> Option<Layout> {
        let owners =
            Layout::array::<RelinkFrame<K, V, C>>(MAX_DEPTH.checked_mul(2)?.checked_add(6)?)
                .ok()?;
        let observations = Layout::array::<ObserveFrame<'_, K, V, C>>(MAX_DEPTH).ok()?;
        let scalar = Layout::array::<usize>(MAX_DEPTH.checked_mul(4)?).ok()?;
        let (layout, _) = Self::insertion_path_layout().extend(owners).ok()?;
        let (layout, _) = layout.extend(observations).ok()?;
        let (layout, _) = layout.extend(scalar).ok()?;
        Some(layout.pad_to_align())
    }

    /// Requested fixed insertion path storage, with no growing Vec.
    pub const fn insertion_path_layout() -> Layout {
        Layout::new::<[bool; MAX_DEPTH]>()
    }
}
impl<K: Ord, V, C> PreparedIndex<K, V, C> {
    /// Borrow the caller's candidate until every comparison succeeds. False
    /// leaves the exact candidate intact (duplicate, missing node or capacity).
    /// A comparison panic likewise cannot destroy its provider-owned fields
    /// before the caller releases its locks. Only structural commit takes it.
    pub fn insert(&mut self, candidate: &mut Option<PreparedIndexNode<K, V, C>>) -> bool {
        self.insert_or_replace(candidate, |_| false).is_ok()
    }

    /// Insert, or replace an equal key only after the caller's predicate accepts
    /// the still-intact row. Comparisons/predicate run before taking ownership.
    /// A refused candidate stays with the caller; a replaced child-free node
    /// is returned for retirement outside the caller's lock. No value is dropped.
    pub fn insert_or_replace(
        &mut self,
        candidate: &mut Option<PreparedIndexNode<K, V, C>>,
        replace: impl FnOnce(&V) -> bool,
    ) -> Result<Option<PreparedIndexNode<K, V, C>>, ()> {
        let incoming = candidate.as_ref().ok_or(())?;
        let mut path = [false; MAX_DEPTH];
        let mut depth = 0;
        let mut current = self.root.as_ref();
        let mut equal = false;
        while let Some(node) = current {
            if depth == MAX_DEPTH {
                return Err(());
            }
            let right = match incoming.node.key.cmp(&node.node.key) {
                Ordering::Equal => {
                    if !replace(&node.node.value) {
                        return Err(());
                    }
                    equal = true;
                    break;
                }
                Ordering::Less => false,
                Ordering::Greater => true,
            };
            path[depth] = right;
            depth += 1;
            current = if right {
                node.node.right.as_ref()
            } else {
                node.node.left.as_ref()
            };
        }
        if equal {
            let (root, old) = replace_at(
                self.root.take().expect("observed root"),
                candidate.take().expect("borrowed candidate"),
                &path[..depth],
            );
            self.root = Some(root);
            Ok(Some(old))
        } else {
            let next_len = self.len.checked_add(1).ok_or(())?;
            self.root = Some(insert_at(
                self.root.take(),
                candidate.take().expect("borrowed candidate"),
                &path[..depth],
            ));
            self.len = next_len;
            Ok(None)
        }
    }
}
/// An allocation-free sorted loan of the index's existing keys and values.
/// Its fixed path is bounded by the same owning AVL invariant as insertion.
pub struct PreparedIndexIter<'a, K, V, C> {
    pending: Option<&'a PreparedIndexNode<K, V, C>>,
    path: [Option<&'a PreparedIndexNode<K, V, C>>; MAX_DEPTH],
    depth: usize,
}
impl<'a, K, V, C> Iterator for PreparedIndexIter<'a, K, V, C> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(node) = self.pending.take() {
            self.path[self.depth] = Some(node);
            self.depth += 1;
            self.pending = node.node.left.as_ref();
        }
        self.depth = self.depth.checked_sub(1)?;
        let node = self.path[self.depth].take().expect("borrowed path");
        self.pending = node.node.right.as_ref();
        Some((&node.node.key, &node.node.value))
    }
}
impl<K, V, C> Drop for PreparedIndex<K, V, C> {
    fn drop(&mut self) {
        // No recursive destruction, even if a future mutation policy changes
        // the tree shape. No provider callback is needed to detach ownership.
        let mut nodes = DetachedIndexNodes {
            head: self.root.take(),
        };
        nodes.drain();
    }
}

/// Detached nodes retain their keys, values and individual custody until the
/// caller releases them. Retirement is iterative and allocation-free.
#[derive(Debug)]
pub struct DetachedIndexNodes<K, V, C> {
    head: Link<K, V, C>,
}
impl<K, V, C> Default for DetachedIndexNodes<K, V, C> {
    fn default() -> Self {
        Self { head: None }
    }
}
impl<K, V, C> DetachedIndexNodes<K, V, C> {
    /// Whether any node was detached.
    pub fn is_empty(&self) -> bool {
        self.head.is_none()
    }
    fn drain(&mut self) {
        while let Some(mut node) = self.head.take() {
            // Rotate an arbitrary detached tree into a right-only chain before
            // dropping each child-free node; no comparisons or allocations.
            if let Some(mut left) = node.node.left.take() {
                node.node.left = left.node.right.take();
                left.node.right = Some(node);
                self.head = Some(left);
            } else {
                self.head = node.node.right.take();
                drop(node);
            }
        }
    }
}
impl<K, V, C> Drop for DetachedIndexNodes<K, V, C> {
    fn drop(&mut self) {
        // If a provider destructor panics, finish detaching the remaining
        // child-free nodes during unwind instead of recursively dropping a tail.
        struct Finish<'a, K, V, C>(&'a mut DetachedIndexNodes<K, V, C>);
        impl<K, V, C> Drop for Finish<'_, K, V, C> {
            fn drop(&mut self) {
                self.0.drain();
            }
        }
        let finish = Finish(self);
        finish.0.drain();
    }
}

fn height<K, V, C>(node: &Link<K, V, C>) -> usize {
    node.as_ref().map_or(0, |n| n.node.height)
}
fn update<K, V, C>(node: &mut PreparedIndexNode<K, V, C>) {
    node.node.height = height(&node.node.left).max(height(&node.node.right)) + 1;
}
fn left<K, V, C>(mut root: PreparedIndexNode<K, V, C>) -> PreparedIndexNode<K, V, C> {
    let mut next = root.node.right.take().expect("right-heavy AVL node");
    root.node.right = next.node.left.take();
    update(&mut root);
    next.node.left = Some(root);
    update(&mut next);
    next
}
fn right<K, V, C>(mut root: PreparedIndexNode<K, V, C>) -> PreparedIndexNode<K, V, C> {
    let mut next = root.node.left.take().expect("left-heavy AVL node");
    root.node.left = next.node.right.take();
    update(&mut root);
    next.node.right = Some(root);
    update(&mut next);
    next
}
fn balance<K, V, C>(mut root: PreparedIndexNode<K, V, C>) -> PreparedIndexNode<K, V, C> {
    update(&mut root);
    let l = height(&root.node.left);
    let r = height(&root.node.right);
    if l > r + 1 {
        let child = root.node.left.as_ref().expect("left-heavy AVL node");
        if height(&child.node.right) > height(&child.node.left) {
            root.node.left = Some(left(root.node.left.take().expect("left child")));
        }
        right(root)
    } else if r > l + 1 {
        let child = root.node.right.as_ref().expect("right-heavy AVL node");
        if height(&child.node.left) > height(&child.node.right) {
            root.node.right = Some(right(root.node.right.take().expect("right child")));
        }
        left(root)
    } else {
        root
    }
}
fn insert_at<K, V, C>(
    root: Link<K, V, C>,
    candidate: PreparedIndexNode<K, V, C>,
    path: &[bool],
) -> PreparedIndexNode<K, V, C> {
    let Some(mut node) = root else {
        debug_assert!(path.is_empty());
        return candidate;
    };
    let (&right, tail) = path.split_first().expect("observed insertion path");
    if right {
        node.node.right = Some(insert_at(node.node.right.take(), candidate, tail));
    } else {
        node.node.left = Some(insert_at(node.node.left.take(), candidate, tail));
    }
    balance(node)
}
fn replace_at<K, V, C>(
    mut root: PreparedIndexNode<K, V, C>,
    mut incoming: PreparedIndexNode<K, V, C>,
    path: &[bool],
) -> (PreparedIndexNode<K, V, C>, PreparedIndexNode<K, V, C>) {
    let Some((&right, tail)) = path.split_first() else {
        incoming.node.left = root.node.left.take();
        incoming.node.right = root.node.right.take();
        incoming.node.height = root.node.height;
        incoming.node.keep = root.node.keep;
        return (incoming, root);
    };
    let child = if right {
        &mut root.node.right
    } else {
        &mut root.node.left
    };
    let (replacement, old) = replace_at(
        child.take().expect("observed replacement path"),
        incoming,
        tail,
    );
    *child = Some(replacement);
    (root, old)
}

fn observe<K, V, C>(root: &mut Link<K, V, C>, stale: &mut impl FnMut(&K, &V) -> bool) -> usize {
    let Some(node) = root else { return 0 };
    let mut frame = ObserveFrame { node, left: 0 };
    frame.left = observe(&mut frame.node.node.left, stale);
    frame.node.node.keep = !stale(&frame.node.node.key, &frame.node.node.value);
    frame.left + usize::from(frame.node.node.keep) + observe(&mut frame.node.node.right, stale)
}
fn flatten<K, V, C>(root: Link<K, V, C>, kept: &mut Link<K, V, C>, retired: &mut Link<K, V, C>) {
    let Some(mut node) = root else { return };
    let mut frame = RelinkFrame {
        left: node.node.left.take(),
        right: node.node.right.take(),
        node: Some(node),
        len: 0,
    };
    flatten(frame.right.take(), kept, retired);
    let mut node = frame.node.take().expect("current relink node");
    let destination = if node.node.keep {
        &mut *kept
    } else {
        &mut *retired
    };
    node.node.right = destination.take();
    *destination = Some(node);
    flatten(frame.left.take(), kept, retired);
}
fn rebuild<K, V, C>(head: &mut Link<K, V, C>, len: usize) -> Link<K, V, C> {
    if len == 0 {
        return None;
    }
    let mut frame = RelinkFrame {
        node: None,
        left: rebuild(head, len / 2),
        right: None,
        len,
    };
    frame.node = head.take();
    *head = frame
        .node
        .as_mut()
        .expect("observed retained node count")
        .node
        .right
        .take();
    frame.right = rebuild(head, frame.len - frame.len / 2 - 1);
    let mut node = frame.node.take().expect("current rebuilt node");
    node.node.left = frame.left.take();
    node.node.right = frame.right.take();
    update(&mut node);
    Some(node)
}

#[cfg(test)]
mod tests;
