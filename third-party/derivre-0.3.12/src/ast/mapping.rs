//! Shared expression traversal; storage, copies and node processing may refuse.
use super::{Expr, ExprRef, ExprSet};
use crate::{ParserAllocationFunding, ParserStorageError, SourceHashMap};
pub(crate) mod workspace;
use std::hash::Hash;

// These are internal mechanisms, not public allocation witnesses. Original
// consumers must supply their concrete paid constructors for all four mutation
// boundaries, including process-owned expression growth and heap-valued clones.
pub(crate) trait Cache<K, V> {
    type Error;
    fn get(&self, key: &K) -> Option<&V>;
    fn contains_key(&self, key: &K) -> bool;
    fn insert(&mut self, key: K, value: V) -> Result<(), Self::Error>;
}
pub(crate) trait Memory<V> {
    type Error;
    fn begin(&mut self, scratch: &mut Scratch<V>, root: ExprRef) -> Result<(), Self::Error>;
    fn copy(&mut self, value: &V) -> Result<V, Self::Error>;
    fn push_node(&mut self, nodes: &mut Vec<ExprRef>, node: ExprRef) -> Result<(), Self::Error>;
    fn push_value(&mut self, values: &mut Vec<V>, value: V) -> Result<(), Self::Error>;
}
/// Caller-owned destinations remain intact on failure. No rollback or capacity
/// refund is implied by a failed node, value copy, or memo publication.
pub(crate) struct Scratch<V> {
    pub(crate) nodes: Vec<ExprRef>,
    pub(crate) values: Vec<V>,
}
impl<V> Scratch<V> {
    pub(crate) fn new() -> Self {
        Self {
            nodes: Vec::new(),
            values: Vec::new(),
        }
    }
}
/// Copying mapping values is an explicit producer operation. Inline values
/// and vectors of inline values share these checked constructors.
pub trait MappingValue: Sized {
    fn copy_with_funding(&self, funding: &ParserAllocationFunding) -> Result<Self, ParserStorageError>;
}
impl MappingValue for ExprRef {
    fn copy_with_funding(&self, _: &ParserAllocationFunding) -> Result<Self, ParserStorageError> { Ok(*self) }
}
impl<T: Copy> MappingValue for Vec<T> {
    fn copy_with_funding(&self, funding: &ParserAllocationFunding) -> Result<Self, ParserStorageError> {
        let mut copy = Vec::new();
        funding.try_extend_copy(&mut copy, self)?;
        Ok(copy)
    }
}
pub(super) struct GrowingCache<'a, K, V> {
    pub(super) values: &'a mut SourceHashMap<K, V>,
    pub(super) funding: &'a ParserAllocationFunding,
}
impl<K: Eq + Hash, V> Cache<K, V> for GrowingCache<'_, K, V> {
    type Error = crate::ParserError;
    fn get(&self, key: &K) -> Option<&V> { self.values.get(key) }
    fn contains_key(&self, key: &K) -> bool { self.values.contains_key(key) }
    fn insert(&mut self, key: K, value: V) -> Result<(), Self::Error> {
        self.funding.try_insert(self.values, key, value)?;
        Ok(())
    }
}
pub(super) struct GrowingMemory<'a>(pub(super) &'a ParserAllocationFunding);
impl<V: MappingValue> Memory<V> for GrowingMemory<'_> {
    type Error = crate::ParserError;
    fn begin(&mut self, scratch: &mut Scratch<V>, root: ExprRef) -> Result<(), Self::Error> {
        self.0.try_push(&mut scratch.nodes, root)?;
        Ok(())
    }
    fn copy(&mut self, value: &V) -> Result<V, Self::Error> { Ok(value.copy_with_funding(self.0)?) }
    fn push_node(&mut self, nodes: &mut Vec<ExprRef>, node: ExprRef) -> Result<(), Self::Error> {
        self.0.try_push(nodes, node)?;
        Ok(())
    }
    fn push_value(&mut self, values: &mut Vec<V>, value: V) -> Result<(), Self::Error> {
        self.0.try_push(values, value)?;
        Ok(())
    }
}

impl ExprSet {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_map_with_storage<K: Eq + Hash, V, E>(
        &mut self,
        root: ExprRef,
        cache: &mut impl Cache<K, V, Error = E>,
        concat_nullable_check: bool,
        mk_key: impl Fn(ExprRef) -> K,
        scratch: &mut Scratch<V>,
        memory: &mut impl Memory<V, Error = E>,
        mut process: impl FnMut(&mut ExprSet, &mut Vec<V>, ExprRef) -> Result<V, E>,
    ) -> Result<V, E> {
        // A hit needs only its actual value copy. In particular, it does not
        // allocate or mutate traversal destinations, matching ordinary mapping.
        if let Some(value) = cache.get(&mk_key(root)) {
            return memory.copy(value);
        }
        memory.begin(scratch, root)?;
        while let Some(&node) = scratch.nodes.last() {
            let key = mk_key(node);
            if cache.contains_key(&key) {
                scratch.nodes.pop();
                continue;
            }
            let expression = self.get(node);
            let concat = concat_nullable_check && matches!(expression, Expr::Concat(_, _));
            let byte_concat =
                concat_nullable_check && matches!(expression, Expr::ByteConcat(_, _, _));
            let count = scratch.nodes.len();
            scratch.values.clear();
            if !byte_concat {
                for &arg in expression.args() {
                    let stop = concat && !self.is_nullable(arg);
                    if let Some(value) = cache.get(&mk_key(arg)) {
                        let value = memory.copy(value)?;
                        memory.push_value(&mut scratch.values, value)?;
                    } else {
                        memory.push_node(&mut scratch.nodes, arg)?;
                    }
                    if stop {
                        break;
                    }
                }
            }
            if scratch.nodes.len() != count {
                continue;
            }
            scratch.nodes.pop();
            let value = process(self, &mut scratch.values, node)?;
            cache.insert(key, value)?;
        }
        memory.copy(
            cache
                .get(&mk_key(root))
                .expect("completed expression mapping"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::ExprFlags;
    use std::collections::BTreeMap;
    #[derive(Debug)]
    struct Failure {
        at: &'static str,
        partial: Vec<ExprRef>,
    }
    struct Memo {
        values: BTreeMap<ExprRef, Vec<ExprRef>>,
        inserts: usize,
        fail_at: usize,
    }
    impl Memo {
        fn new(fail_at: usize) -> Self {
            Self {
                values: BTreeMap::new(),
                inserts: 0,
                fail_at,
            }
        }
    }
    impl Cache<ExprRef, Vec<ExprRef>> for Memo {
        type Error = Failure;
        fn get(&self, key: &ExprRef) -> Option<&Vec<ExprRef>> {
            self.values.get(key)
        }
        fn contains_key(&self, key: &ExprRef) -> bool {
            self.values.contains_key(key)
        }
        fn insert(&mut self, key: ExprRef, value: Vec<ExprRef>) -> Result<(), Failure> {
            self.inserts += 1;
            if self.inserts == self.fail_at {
                return Err(Failure {
                    at: "publication",
                    partial: value,
                });
            }
            self.values.insert(key, value);
            Ok(())
        }
    }
    struct Fixed {
        begins: usize,
        copies: usize,
        fail_copy: usize,
    }
    impl Fixed {
        fn new() -> Self {
            Self {
                begins: 0,
                copies: 0,
                fail_copy: usize::MAX,
            }
        }
    }
    impl Memory<Vec<ExprRef>> for Fixed {
        type Error = Failure;
        fn begin(
            &mut self,
            scratch: &mut Scratch<Vec<ExprRef>>,
            root: ExprRef,
        ) -> Result<(), Failure> {
            self.begins += 1;
            scratch.nodes.clear();
            scratch.values.clear();
            self.push_node(&mut scratch.nodes, root)
        }
        fn copy(&mut self, value: &Vec<ExprRef>) -> Result<Vec<ExprRef>, Failure> {
            self.copies += 1;
            if self.copies == self.fail_copy {
                return Err(Failure {
                    at: "copy",
                    partial: value[..1].to_vec(),
                });
            }
            Ok(value.clone())
        }
        fn push_node(&mut self, nodes: &mut Vec<ExprRef>, node: ExprRef) -> Result<(), Failure> {
            if nodes.len() == nodes.capacity() {
                return Err(Failure {
                    at: "nodes",
                    partial: vec![node],
                });
            }
            nodes.push(node);
            Ok(())
        }
        fn push_value(
            &mut self,
            values: &mut Vec<Vec<ExprRef>>,
            value: Vec<ExprRef>,
        ) -> Result<(), Failure> {
            if values.len() == values.capacity() {
                return Err(Failure {
                    at: "values",
                    partial: value,
                });
            }
            values.push(value);
            Ok(())
        }
    }
    fn process(_: &mut ExprSet, values: &mut Vec<Vec<ExprRef>>, node: ExprRef) -> Vec<ExprRef> {
        let mut result = Vec::new();
        for value in values.iter() {
            result.extend_from_slice(value);
        }
        result.push(node);
        result
    }
    fn scratch(n: usize) -> Scratch<Vec<ExprRef>> {
        Scratch {
            nodes: Vec::with_capacity(n),
            values: Vec::with_capacity(n),
        }
    }
    #[test]
    fn fallible_mapping_keeps_nullable_skip_order_cache_hits_and_failed_publication_prefixes() {
        let mut source = ExprSet::new(256, crate::ParserAllocationFunding::unenforced()).unwrap();
        let a = source.mk(Expr::Byte(b'a')).unwrap();
        let b = source.mk(Expr::Byte(b'b')).unwrap();
        let concat = source.mk(Expr::Concat(ExprFlags::POSITIVE, [a, b])).unwrap();
        let bytes = source.mk(Expr::ByteConcat(ExprFlags::POSITIVE, b"xy", b)).unwrap();
        for (root, skip, expected) in [
            (concat, false, vec![a, b, concat]),
            (concat, true, vec![a, concat]),
            (bytes, false, vec![b, bytes]),
            (bytes, true, vec![bytes]),
        ] {
            let mut memo = Memo::new(usize::MAX);
            let mut memory = Fixed::new();
            let mut scratch = scratch(source.len());
            let result = source
                .try_map_with_storage(
                    root,
                    &mut memo,
                    skip,
                    |r| r,
                    &mut scratch,
                    &mut memory,
                    |e, v, r| Ok(process(e, v, r)),
                )
                .unwrap();
            assert_eq!(result, expected);
            let mut ordinary = crate::SourceHashMap::default();
            assert_eq!(
                source.map(root, &mut ordinary, skip, |r| r, |e, v, r| Ok(process(e, v, r))).unwrap(),
                expected
            );
            assert_eq!(memory.begins, 1);
            // The preexisting memo path bypasses traversal initialization and
            // processing; an actual value-copy failure still retains its prefix.
            let before = memo.inserts;
            let result = source
                .try_map_with_storage(
                    root,
                    &mut memo,
                    skip,
                    |r| r,
                    &mut scratch,
                    &mut memory,
                    |_, _, _| panic!("cache hit must not process"),
                )
                .unwrap();
            assert_eq!(result, expected);
            assert_eq!(memory.begins, 1);
            assert_eq!(memo.inserts, before);
            memory.fail_copy = memory.copies + 1;
            let error = source
                .try_map_with_storage(
                    root,
                    &mut memo,
                    skip,
                    |r| r,
                    &mut scratch,
                    &mut memory,
                    |_, _, _| panic!("cache hit must not process"),
                )
                .unwrap_err();
            assert_eq!(error.at, "copy");
            assert_eq!(error.partial.as_slice(), &expected[..1]);
            assert_eq!(memo.values.get(&root).unwrap(), &expected);
        }
        let mut memo = Memo::new(2);
        let mut memory = Fixed::new();
        let mut scratch = scratch(source.len());
        let error = source
            .try_map_with_storage(
                concat,
                &mut memo,
                false,
                |r| r,
                &mut scratch,
                &mut memory,
                |e, v, r| Ok(process(e, v, r)),
            )
            .unwrap_err();
        assert_eq!(error.at, "publication");
        assert_eq!(error.partial, vec![a]);
        assert_eq!(memo.inserts, 2);
        assert_eq!(memo.values.get(&b), Some(&vec![b]));
        assert!(!memo.values.contains_key(&a));
        assert!(!memo.values.contains_key(&concat));
        assert_eq!(scratch.nodes, vec![concat]);
        drop(source);
        assert_eq!(memo.values.get(&b), Some(&vec![b]));
        assert_eq!(error.partial, vec![a]);
    }
}
