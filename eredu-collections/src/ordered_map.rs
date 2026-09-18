//! Safe ordered storage with explicit node allocations.
//!
//! AVL rotations move existing boxes. Insertions allocate exactly one node;
//! removals and all iterators allocate no backing. The traversal frontier is
//! fixed by the AVL height bound and the addressable number of nonzero nodes.
use alloc::boxed::Box;
use core::{
    alloc::Layout,
    borrow::Borrow,
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
    iter::FusedIterator,
    mem,
    ops::Index,
};

/// Fixed insertion failure; callback errors remain in their original type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TryInsertError<E> {
    /// The number of entries cannot be represented.
    SizeOverflow,
    /// The prospective node callback refused the allocation.
    Funding(E),
}
impl<E: fmt::Display> fmt::Display for TryInsertError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SizeOverflow => f.write_str("ordered map size overflow"),
            Self::Funding(error) => error.fmt(f),
        }
    }
}
impl<E: core::error::Error + 'static> core::error::Error for TryInsertError<E> {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Funding(error) => Some(error),
            Self::SizeOverflow => None,
        }
    }
}

type Link<K, V> = Option<Box<Node<K, V>>>;
struct Node<K, V> {
    key: K,
    value: V,
    left: Link<K, V>,
    right: Link<K, V>,
    height: u8,
    retained: bool,
}
pub struct Map<K, V> {
    root: Link<K, V>,
    len: usize,
}

fn height<K, V>(node: &Link<K, V>) -> u8 {
    node.as_ref().map_or(0, |node| node.height)
}
fn update<K, V>(node: &mut Node<K, V>) {
    node.height = 1 + height(&node.left).max(height(&node.right));
}
fn rotate_left<K, V>(mut node: Box<Node<K, V>>) -> Box<Node<K, V>> {
    let mut next = node.right.take().expect("AVL right child");
    node.right = next.left.take();
    update(&mut node);
    next.left = Some(node);
    update(&mut next);
    next
}
fn rotate_right<K, V>(mut node: Box<Node<K, V>>) -> Box<Node<K, V>> {
    let mut next = node.left.take().expect("AVL left child");
    node.left = next.right.take();
    update(&mut node);
    next.right = Some(node);
    update(&mut next);
    next
}
fn balance<K, V>(mut node: Box<Node<K, V>>) -> Box<Node<K, V>> {
    update(&mut node);
    if i16::from(height(&node.left)) - i16::from(height(&node.right)) > 1 {
        let left = node.left.as_ref().expect("AVL left branch");
        if height(&left.left) < height(&left.right) {
            node.left = Some(rotate_left(node.left.take().unwrap()));
        }
        rotate_right(node)
    } else if i16::from(height(&node.right)) - i16::from(height(&node.left)) > 1 {
        let right = node.right.as_ref().expect("AVL right branch");
        if height(&right.right) < height(&right.left) {
            node.right = Some(rotate_right(node.right.take().unwrap()));
        }
        rotate_left(node)
    } else {
        node
    }
}
fn insert_node<K: Ord, V>(root: Link<K, V>, inserted: Box<Node<K, V>>) -> Box<Node<K, V>> {
    let Some(mut node) = root else {
        return inserted;
    };
    if inserted.key < node.key {
        node.left = Some(insert_node(node.left.take(), inserted));
    } else {
        node.right = Some(insert_node(node.right.take(), inserted));
    }
    balance(node)
}
fn take_min<K, V>(mut node: Box<Node<K, V>>) -> (Link<K, V>, Box<Node<K, V>>) {
    let Some(left) = node.left.take() else {
        return (node.right.take(), node);
    };
    let (left, result) = take_min(left);
    node.left = left;
    (Some(balance(node)), result)
}
fn take_max<K, V>(mut node: Box<Node<K, V>>) -> (Link<K, V>, Box<Node<K, V>>) {
    let Some(right) = node.right.take() else {
        return (node.left.take(), node);
    };
    let (right, result) = take_max(right);
    node.right = right;
    (Some(balance(node)), result)
}
fn remove_node<K, V, Q>(root: Link<K, V>, key: &Q) -> (Link<K, V>, Option<(K, V)>)
where
    K: Borrow<Q>,
    Q: Ord + ?Sized,
{
    let Some(mut node) = root else {
        return (None, None);
    };
    match key.cmp(node.key.borrow()) {
        Ordering::Less => {
            let (left, found) = remove_node(node.left.take(), key);
            node.left = left;
            (Some(balance(node)), found)
        }
        Ordering::Greater => {
            let (right, found) = remove_node(node.right.take(), key);
            node.right = right;
            (Some(balance(node)), found)
        }
        Ordering::Equal => {
            let right = node.right.take();
            let left = node.left.take();
            let next = match right {
                None => left,
                Some(right) => {
                    let (right, mut next) = take_min(right);
                    next.left = left;
                    next.right = right;
                    Some(balance(next))
                }
            };
            (next, Some((node.key, node.value)))
        }
    }
}

impl<K, V> Map<K, V> {
    pub const fn new() -> Self {
        Self { root: None, len: 0 }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn allocation_size(&self) -> usize {
        self.len
            .checked_mul(Layout::new::<Node<K, V>>().size())
            .expect("live AVL allocation extent")
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn clear(&mut self) {
        self.root = None;
        self.len = 0;
    }
    fn node<Q: Ord + ?Sized>(&self, key: &Q) -> Option<&Node<K, V>>
    where
        K: Borrow<Q>,
    {
        let mut cursor = self.root.as_deref();
        while let Some(node) = cursor {
            match key.cmp(node.key.borrow()) {
                Ordering::Less => cursor = node.left.as_deref(),
                Ordering::Greater => cursor = node.right.as_deref(),
                Ordering::Equal => return Some(node),
            }
        }
        None
    }
    pub fn get<Q: Ord + ?Sized>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
    {
        self.node(key).map(|node| &node.value)
    }
    pub fn get_key_value<Q: Ord + ?Sized>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
    {
        self.node(key).map(|node| (&node.key, &node.value))
    }
    pub fn contains_key<Q: Ord + ?Sized>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
    {
        self.node(key).is_some()
    }
    pub fn get_mut<Q: Ord + ?Sized>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
    {
        let mut cursor = self.root.as_deref_mut();
        while let Some(node) = cursor {
            match key.cmp(node.key.borrow()) {
                Ordering::Less => cursor = node.left.as_deref_mut(),
                Ordering::Greater => cursor = node.right.as_deref_mut(),
                Ordering::Equal => return Some(&mut node.value),
            }
        }
        None
    }
    pub fn remove_entry<Q: Ord + ?Sized>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
    {
        let (root, found) = remove_node(self.root.take(), key);
        self.root = root;
        if found.is_some() {
            self.len -= 1;
        }
        found
    }
    pub fn remove<Q: Ord + ?Sized>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
    {
        self.remove_entry(key).map(|(_, value)| value)
    }
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter::new(self.root.as_deref(), self.len)
    }
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut::new(self.root.as_deref_mut(), self.len)
    }
    pub fn keys(&self) -> Keys<'_, K, V> {
        self.iter().map(|(key, _)| key)
    }
    pub fn values(&self) -> Values<'_, K, V> {
        self.iter().map(|(_, value)| value)
    }
    pub fn values_mut(&mut self) -> ValuesMut<'_, K, V> {
        self.iter_mut().map(|(_, value)| value)
    }
    pub fn into_values(self) -> IntoValues<K, V> {
        self.into_iter().map(|(_, value)| value)
    }
}
impl<K: Ord, V> Map<K, V> {
    /// Fixed controls for the reached lookup and recursive insertion path.
    ///
    /// Node backing is separate and remains the `try_insert_with` callback's
    /// exact layout. The caller also owns its callback/result enclosure. This
    /// inventory follows the actual search depth; rotations reuse existing boxes.
    pub fn insertion_control_bytes(&self, key: &K) -> Option<usize> {
        let mut depth = 1usize;
        let mut node = self.root.as_deref();
        while let Some(current) = node {
            match key.cmp(&current.key) {
                Ordering::Equal => break,
                Ordering::Less => node = current.left.as_deref(),
                Ordering::Greater => node = current.right.as_deref(),
            }
            depth = depth.checked_add(1)?;
        }
        let lookup = mem::size_of::<(&Self, &K, Option<&Node<K, V>>, usize, Ordering)>();
        let insert_frame = mem::size_of::<(Link<K, V>, Box<Node<K, V>>, Box<Node<K, V>>, bool)>();
        let balance = mem::size_of::<(Box<Node<K, V>>, &Node<K, V>, i16)>();
        let rotation = mem::size_of::<(Box<Node<K, V>>, Box<Node<K, V>>, Link<K, V>)>();
        depth
            .checked_mul(insert_frame)?
            .checked_add(lookup)?
            .checked_add(balance)?
            .checked_add(rotation.checked_mul(2)?)
    }

    /// Inserts an entry after admitting the exact node layout before allocation.
    ///
    /// Replacing an existing value does not call `reserve`. The callback owns
    /// admission policy; the map retains no account. Key/value construction and
    /// callback storage are separate responsibilities of the caller. Refusal
    /// leaves the map unchanged. Host allocator exhaustion follows `Box::new`.
    pub fn try_insert_with<E>(
        &mut self,
        key: K,
        value: V,
        reserve: impl FnOnce(Layout) -> Result<(), E>,
    ) -> Result<Option<V>, TryInsertError<E>> {
        if let Some(prior) = self.get_mut(&key) {
            return Ok(Some(mem::replace(prior, value)));
        }
        let len = self
            .len
            .checked_add(1)
            .ok_or(TryInsertError::SizeOverflow)?;
        reserve(Layout::new::<Node<K, V>>()).map_err(TryInsertError::Funding)?;
        let inserted = Box::new(Node {
            key,
            value,
            left: None,
            right: None,
            height: 1,
            retained: true,
        });
        self.root = Some(insert_node(self.root.take(), inserted));
        self.len = len;
        Ok(None)
    }
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.try_insert_with(key, value, |_| Ok::<_, core::convert::Infallible>(()))
            .expect("ordinary ordered node allocation")
    }
    pub fn append(&mut self, other: &mut Self) {
        for (key, value) in mem::replace(other, Self::new()) {
            self.insert(key, value);
        }
    }
    pub fn retain(&mut self, mut predicate: impl FnMut(&K, &mut V) -> bool) {
        // User predicates run while the entire structure remains installed.
        // If one panics, the tree and length remain valid; only already visited
        // values may have changed. The second pass has no user callbacks.
        fn mark<K, V>(root: &mut Link<K, V>, predicate: &mut impl FnMut(&K, &mut V) -> bool) {
            if let Some(node) = root {
                mark(&mut node.left, predicate);
                node.retained = predicate(&node.key, &mut node.value);
                mark(&mut node.right, predicate);
            }
        }
        fn visit<K: Ord, V>(root: Link<K, V>, len: &mut usize) -> Link<K, V> {
            let mut node = root?;
            node.left = visit(node.left.take(), len);
            let keep = node.retained;
            node.right = visit(node.right.take(), len);
            // Filtering can remove an entire subtree, so rejoin by inserting
            // existing nodes; no new storage or allocation is required.
            fn merge<K: Ord, V>(root: Link<K, V>, other: Link<K, V>) -> Link<K, V> {
                let Some(mut node) = other else {
                    return root;
                };
                let left = node.left.take();
                let right = node.right.take();
                node.height = 1;
                let root = Some(insert_node(root, node));
                merge(merge(root, left), right)
            }
            let left = node.left.take();
            let right = node.right.take();
            if keep {
                node.height = 1;
                merge(merge(Some(node), left), right)
            } else {
                *len -= 1;
                merge(left, right)
            }
        }
        mark(&mut self.root, &mut predicate);
        self.root = visit(self.root.take(), &mut self.len);
    }
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        if self.contains_key(&key) {
            Entry::Occupied(OccupiedEntry { map: self, key })
        } else {
            Entry::Vacant(VacantEntry { map: self, key })
        }
    }
}

// An AVL tree holding n nodes has height below 2*log2(n+1). Every node has
// nonzero size, hence at most usize::BITS levels doubled are addressable.
const HEIGHT: usize = 2 * usize::BITS as usize;
const FRONTIER: usize = 4 * HEIGHT + 4;
struct Deque<T> {
    items: [Option<T>; FRONTIER],
    front: usize,
    len: usize,
}
impl<T> Deque<T> {
    fn new() -> Self {
        Self {
            items: core::array::from_fn(|_| None),
            front: 0,
            len: 0,
        }
    }
    fn push_front(&mut self, value: T) {
        assert!(self.len < FRONTIER, "AVL frontier bound");
        self.front = (self.front + FRONTIER - 1) % FRONTIER;
        self.items[self.front] = Some(value);
        self.len += 1;
    }
    fn push_back(&mut self, value: T) {
        assert!(self.len < FRONTIER, "AVL frontier bound");
        let index = (self.front + self.len) % FRONTIER;
        self.items[index] = Some(value);
        self.len += 1;
    }
    fn pop_front(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        let value = self.items[self.front].take();
        self.front = (self.front + 1) % FRONTIER;
        self.len -= 1;
        value
    }
    fn pop_back(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        self.items[(self.front + self.len) % FRONTIER].take()
    }
}
// Each end walks a compact stack of node references. The remaining count
// prevents the two cursors from returning the same middle entry.
struct Cursor<'a, K, V> {
    nodes: [Option<&'a Node<K, V>>; HEIGHT],
    len: usize,
}
impl<'a, K, V> Cursor<'a, K, V> {
    fn new(mut root: Option<&'a Node<K, V>>, back: bool) -> Self {
        let mut cursor = Self {
            nodes: [None; HEIGHT],
            len: 0,
        };
        while let Some(node) = root {
            cursor.nodes[cursor.len] = Some(node);
            cursor.len += 1;
            root = if back {
                node.right.as_deref()
            } else {
                node.left.as_deref()
            };
        }
        cursor
    }
    fn take(&mut self, back: bool) -> Option<(&'a K, &'a V)> {
        self.len = self.len.checked_sub(1)?;
        let node = self.nodes[self.len].take().expect("AVL cursor node");
        let mut next = if back {
            node.left.as_deref()
        } else {
            node.right.as_deref()
        };
        while let Some(child) = next {
            self.nodes[self.len] = Some(child);
            self.len += 1;
            next = if back {
                child.right.as_deref()
            } else {
                child.left.as_deref()
            };
        }
        Some((&node.key, &node.value))
    }
}
pub struct Iter<'a, K, V> {
    front: Cursor<'a, K, V>,
    back: Cursor<'a, K, V>,
    left: usize,
}
impl<'a, K, V> Iter<'a, K, V> {
    fn new(root: Option<&'a Node<K, V>>, len: usize) -> Self {
        Self {
            front: Cursor::new(root, false),
            back: Cursor::new(root, true),
            left: len,
        }
    }
    fn take(&mut self, back: bool) -> Option<(&'a K, &'a V)> {
        if self.left == 0 {
            return None;
        }
        self.left -= 1;
        if back {
            self.back.take(true)
        } else {
            self.front.take(false)
        }
    }
}
impl<K, V> Clone for Iter<'_, K, V> {
    fn clone(&self) -> Self {
        Self {
            front: Cursor {
                nodes: self.front.nodes,
                len: self.front.len,
            },
            back: Cursor {
                nodes: self.back.nodes,
                len: self.back.len,
            },
            left: self.left,
        }
    }
}
enum MutableTask<'a, K, V> {
    Tree(&'a mut Node<K, V>),
    Entry(&'a K, &'a mut V),
}
pub struct IterMut<'a, K, V> {
    tasks: Deque<MutableTask<'a, K, V>>,
    left: usize,
}
impl<'a, K, V> IterMut<'a, K, V> {
    fn new(root: Option<&'a mut Node<K, V>>, len: usize) -> Self {
        let mut tasks = Deque::new();
        if let Some(root) = root {
            tasks.push_back(MutableTask::Tree(root));
        }
        Self { tasks, left: len }
    }
    fn take(&mut self, back: bool) -> Option<(&'a K, &'a mut V)> {
        loop {
            let task = if back {
                self.tasks.pop_back()?
            } else {
                self.tasks.pop_front()?
            };
            match task {
                MutableTask::Entry(key, value) => {
                    self.left -= 1;
                    return Some((key, value));
                }
                MutableTask::Tree(node) => {
                    if back {
                        if let Some(left) = node.left.as_deref_mut() {
                            self.tasks.push_back(MutableTask::Tree(left));
                        }
                        self.tasks
                            .push_back(MutableTask::Entry(&node.key, &mut node.value));
                        if let Some(right) = node.right.as_deref_mut() {
                            self.tasks.push_back(MutableTask::Tree(right));
                        }
                    } else {
                        if let Some(right) = node.right.as_deref_mut() {
                            self.tasks.push_front(MutableTask::Tree(right));
                        }
                        self.tasks
                            .push_front(MutableTask::Entry(&node.key, &mut node.value));
                        if let Some(left) = node.left.as_deref_mut() {
                            self.tasks.push_front(MutableTask::Tree(left));
                        }
                    }
                }
            }
        }
    }
}
macro_rules! borrowed_iter {
    ($name:ident, $value:ty) => {
        impl<'a, K, V> Iterator for $name<'a, K, V> {
            type Item = (&'a K, $value);
            fn next(&mut self) -> Option<Self::Item> {
                self.take(false)
            }
            fn size_hint(&self) -> (usize, Option<usize>) {
                (self.left, Some(self.left))
            }
        }
        impl<'a, K, V> DoubleEndedIterator for $name<'a, K, V> {
            fn next_back(&mut self) -> Option<Self::Item> {
                self.take(true)
            }
        }
        impl<K, V> ExactSizeIterator for $name<'_, K, V> {}
        impl<K, V> FusedIterator for $name<'_, K, V> {}
        impl<K, V> fmt::Debug for $name<'_, K, V> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct(stringify!($name))
                    .field("remaining", &self.left)
                    .finish()
            }
        }
    };
}
borrowed_iter!(Iter, &'a V);
borrowed_iter!(IterMut, &'a mut V);
pub type Keys<'a, K, V> = core::iter::Map<Iter<'a, K, V>, fn((&'a K, &'a V)) -> &'a K>;
pub type Values<'a, K, V> = core::iter::Map<Iter<'a, K, V>, fn((&'a K, &'a V)) -> &'a V>;
pub type ValuesMut<'a, K, V> =
    core::iter::Map<IterMut<'a, K, V>, fn((&'a K, &'a mut V)) -> &'a mut V>;
pub type IntoValues<K, V> = core::iter::Map<IntoIter<K, V>, fn((K, V)) -> V>;
pub struct IntoIter<K, V> {
    map: Map<K, V>,
}
impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);
    fn next(&mut self) -> Option<Self::Item> {
        let (root, node) = take_min(self.map.root.take()?);
        self.map.root = root;
        self.map.len -= 1;
        Some((node.key, node.value))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.map.len, Some(self.map.len))
    }
}
impl<K, V> DoubleEndedIterator for IntoIter<K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let (root, node) = take_max(self.map.root.take()?);
        self.map.root = root;
        self.map.len -= 1;
        Some((node.key, node.value))
    }
}
impl<K, V> ExactSizeIterator for IntoIter<K, V> {}
impl<K, V> FusedIterator for IntoIter<K, V> {}
impl<K, V> fmt::Debug for IntoIter<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IntoIter")
            .field("remaining", &self.map.len)
            .finish()
    }
}
impl<K, V> IntoIterator for Map<K, V> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;
    fn into_iter(self) -> Self::IntoIter {
        IntoIter { map: self }
    }
}
impl<K: Ord, V> FromIterator<(K, V)> for Map<K, V> {
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut result = Self::new();
        result.extend(iter);
        result
    }
}
impl<K: Ord, V, const N: usize> From<[(K, V); N]> for Map<K, V> {
    fn from(values: [(K, V); N]) -> Self {
        values.into_iter().collect()
    }
}
impl<K: Ord, V> Extend<(K, V)> for Map<K, V> {
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        for (key, value) in iter {
            self.insert(key, value);
        }
    }
}
impl<K: Ord + Clone, V: Clone> Clone for Map<K, V> {
    fn clone(&self) -> Self {
        self.iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect()
    }
}
impl<K: PartialEq, V: PartialEq> PartialEq for Map<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.iter().eq(other.iter())
    }
}
impl<K: Eq, V: Eq> Eq for Map<K, V> {}
impl<K: Hash, V: Hash> Hash for Map<K, V> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len.hash(state);
        for entry in self.iter() {
            entry.hash(state);
        }
    }
}
impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for Map<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}
impl<K: Borrow<Q>, V, Q: Ord + ?Sized> Index<&Q> for Map<K, V> {
    type Output = V;
    fn index(&self, key: &Q) -> &V {
        self.get(key).expect("no entry found for key")
    }
}

pub enum Entry<'a, K, V> {
    Vacant(VacantEntry<'a, K, V>),
    Occupied(OccupiedEntry<'a, K, V>),
}
pub struct VacantEntry<'a, K, V> {
    map: &'a mut Map<K, V>,
    key: K,
}
pub struct OccupiedEntry<'a, K, V> {
    map: &'a mut Map<K, V>,
    key: K,
}
impl<'a, K: Ord + Clone, V> VacantEntry<'a, K, V> {
    pub fn key(&self) -> &K {
        &self.key
    }
    pub fn insert(self, value: V) -> &'a mut V {
        let key = self.key.clone();
        self.map.insert(self.key, value);
        self.map.get_mut(&key).expect("inserted entry")
    }
}
impl<'a, K: Ord, V> OccupiedEntry<'a, K, V> {
    pub fn key(&self) -> &K {
        self.map.get_key_value(&self.key).expect("occupied entry").0
    }
    pub fn get(&self) -> &V {
        self.map.get(&self.key).expect("occupied entry")
    }
    pub fn get_mut(&mut self) -> &mut V {
        self.map.get_mut(&self.key).expect("occupied entry")
    }
    pub fn into_mut(self) -> &'a mut V {
        self.map.get_mut(&self.key).expect("occupied entry")
    }
    pub fn insert(&mut self, value: V) -> V {
        mem::replace(self.get_mut(), value)
    }
    pub fn remove(self) -> V {
        self.map.remove(&self.key).expect("occupied entry")
    }
    pub fn remove_entry(self) -> (K, V) {
        self.map.remove_entry(&self.key).expect("occupied entry")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        collections::BTreeMap,
        eprintln, format,
        string::String,
        sync::atomic::{AtomicUsize, Ordering as AtomicOrdering},
        vec::Vec,
    };
    fn invariant(node: &Link<u32, i64>, min: Option<u32>, max: Option<u32>) -> (usize, u8) {
        let Some(node) = node else {
            return (0, 0);
        };
        assert!(min.map_or(true, |min| min < node.key));
        assert!(max.map_or(true, |max| node.key < max));
        let (left, lh) = invariant(&node.left, min, Some(node.key));
        let (right, rh) = invariant(&node.right, Some(node.key), max);
        assert!(i16::from(lh).abs_diff(i16::from(rh)) <= 1);
        assert_eq!(node.height, 1 + lh.max(rh));
        (left + right + 1, node.height)
    }
    fn same(actual: &Map<u32, i64>, expected: &BTreeMap<u32, i64>) {
        assert_eq!(actual.len(), expected.len());
        assert_eq!(invariant(&actual.root, None, None).0, actual.len());
        assert!(actual.iter().eq(expected.iter()));
        let (mut actual, mut expected) = (actual.iter(), expected.iter());
        while expected.len() > 0 {
            assert_eq!(actual.next(), expected.next());
            assert_eq!(actual.next_back(), expected.next_back());
            assert_eq!(actual.len(), expected.len());
        }
        assert_eq!(actual.next(), None);
        assert_eq!(actual.next_back(), None);
    }
    #[test]
    fn shuffled_operations_match_ordered_map_and_balance() {
        let (mut actual, mut expected) = (Map::new(), BTreeMap::new());
        let mut state = 0x147a99ce_u32;
        for iteration in 0..20_000 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let key = (state >> 8) % 257;
            match state % 5 {
                0 => assert_eq!(actual.remove_entry(&key), expected.remove_entry(&key)),
                1 => assert_eq!(actual.get(&key), expected.get(&key)),
                _ => assert_eq!(
                    actual.insert(key, iteration),
                    expected.insert(key, iteration)
                ),
            }
            if iteration % 31 == 0 {
                same(&actual, &expected);
            }
        }
        same(&actual, &expected);
        let (mut actual_iter, mut expected_iter) = (actual.iter_mut(), expected.iter_mut());
        let mut i = 0;
        while expected_iter.len() > 0 {
            let a = if i % 2 == 0 {
                actual_iter.next()
            } else {
                actual_iter.next_back()
            };
            let b = if i % 2 == 0 {
                expected_iter.next()
            } else {
                expected_iter.next_back()
            };
            let ((ak, av), (bk, bv)) = (a.unwrap(), b.unwrap());
            assert_eq!(ak, bk);
            *av += i;
            *bv += i;
            i += 1;
        }
        assert!(actual_iter.next().is_none());
        drop(actual_iter);
        drop(expected_iter);
        actual.retain(|key, value| {
            *value += 3;
            key % 3 == 0
        });
        expected.retain(|key, value| {
            *value += 3;
            key % 3 == 0
        });
        same(&actual, &expected);
        let (mut a, mut b) = (actual.into_iter(), expected.into_iter());
        while b.len() > 0 {
            assert_eq!(a.next(), b.next());
            assert_eq!(a.next_back(), b.next_back());
        }
        assert_eq!(a.next(), None);
    }
    #[test]
    fn entries_append_and_shared_iterator_clones_match_ordered_map() {
        let mut actual = Map::new();
        match actual.entry(2_u32) {
            Entry::Vacant(e) => {
                assert_eq!(*e.key(), 2);
                *e.insert(3) = 4;
            }
            _ => panic!(),
        }
        match actual.entry(2) {
            Entry::Occupied(mut e) => {
                assert_eq!(*e.key(), 2);
                assert_eq!(e.insert(5), 4);
                assert_eq!(*e.get(), 5);
                *e.get_mut() += 1;
            }
            _ => panic!(),
        }
        let mut other = [(1, 7), (2, 8), (3, 9)].into_iter().collect::<Map<_, _>>();
        actual.append(&mut other);
        assert!(other.is_empty());
        assert!(
            actual
                .iter()
                .eq([(1, 7), (2, 8), (3, 9)].iter().map(|(k, v)| (k, v)))
        );
        let mut iter = actual.iter();
        assert_eq!(iter.next(), Some((&1, &7)));
        assert!(iter.clone().eq(iter));
        match actual.entry(2) {
            Entry::Occupied(e) => assert_eq!(e.remove_entry(), (2, 8)),
            _ => panic!(),
        }
        actual.clear();
        assert!(actual.is_empty());
    }
    struct Refuse {
        calls: AtomicUsize,
    }
    impl Refuse {
        fn reserve(&self, layout: Layout) -> Result<(), ()> {
            assert_eq!(layout, Layout::new::<Node<u32, i64>>());
            self.calls.fetch_add(1, AtomicOrdering::Relaxed);
            Err(())
        }
    }
    #[test]
    fn prospective_node_refusal_keeps_prior_map_and_replacement_needs_no_allocation() {
        let mut map = Map::new();
        map.insert(2, 7);
        let funding = Refuse {
            calls: AtomicUsize::new(0),
        };
        assert_eq!(
            map.try_insert_with(2, 8, |layout| funding.reserve(layout)),
            Ok(Some(7))
        );
        assert_eq!(funding.calls.load(AtomicOrdering::Relaxed), 0);
        assert_eq!(
            map.try_insert_with(3, 9, |layout| funding.reserve(layout)),
            Err(TryInsertError::Funding(()))
        );
        assert_eq!(funding.calls.load(AtomicOrdering::Relaxed), 1);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&2), Some(&8));
    }
    #[test]
    fn retain_panic_preserves_a_valid_reusable_tree() {
        let mut map = (0..40)
            .map(|key| (key, i64::from(key)))
            .collect::<Map<_, _>>();
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            map.retain(|key, value| {
                assert_ne!(*key, 17);
                *value += 1;
                key % 2 == 0
            });
        }));
        assert!(failed.is_err());
        assert_eq!(map.len(), 40);
        assert_eq!(invariant(&map.root, None, None).0, 40);
        map.retain(|key, _| key % 2 == 0);
        assert_eq!(map.len(), 20);
        assert_eq!(invariant(&map.root, None, None).0, 20);
        assert!(map.keys().copied().eq((0..40).filter(|key| key % 2 == 0)));
    }
    #[test]
    #[ignore = "standalone comparative measurement"]
    fn measure_small_and_shuffled_maps() {
        use std::{hint::black_box, time::Instant};
        for (count, rounds) in [(7_u32, 20_000_u32), (50_000, 4)] {
            let mut keys: Vec<String> = (0..count).map(|i| format!("key-{i:08}")).collect();
            let mut state = 17_u32;
            for i in (1..keys.len()).rev() {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                keys.swap(i, state as usize % (i + 1));
            }
            let start = Instant::now();
            for _ in 0..rounds {
                let mut map = Map::new();
                for (i, key) in keys.iter().enumerate() {
                    map.insert(key.clone(), i);
                }
                for key in &keys {
                    black_box(map.get(key));
                }
                black_box(map.iter().map(|(_, value)| *value).sum::<usize>());
                for key in &keys {
                    black_box(map.remove(key));
                }
            }
            let ours = start.elapsed();
            let start = Instant::now();
            for _ in 0..rounds {
                let mut map = BTreeMap::new();
                for (i, key) in keys.iter().enumerate() {
                    map.insert(key.clone(), i);
                }
                for key in &keys {
                    black_box(map.get(key));
                }
                black_box(map.iter().map(|(_, value)| *value).sum::<usize>());
                for key in &keys {
                    black_box(map.remove(key));
                }
            }
            let reference = start.elapsed();
            eprintln!(
                "entries={count} rounds={rounds} ordered_ms={} upstream_btree_ms={} ratio={:.3}",
                ours.as_secs_f64() * 1000.0,
                reference.as_secs_f64() * 1000.0,
                ours.as_secs_f64() / reference.as_secs_f64()
            );
        }
    }
}
