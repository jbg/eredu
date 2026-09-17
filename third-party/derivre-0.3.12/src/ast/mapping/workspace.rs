//! Actual source-bound traversal destinations; node processing remains separate.
use super::{Cache, Memory, Scratch};
use crate::ast::{ExprRef, ExprSet};
use std::{
    alloc::Layout,
    collections::TryReserveError,
    fmt,
    hash::Hash,
    marker::PhantomData,
    mem::{size_of, size_of_val},
};

#[derive(Clone, Copy, Debug)]
pub(crate) struct Requirements {
    pub(crate) buffers: usize,
    pub(crate) controls: usize,
    pub(crate) total: usize,
}
#[derive(Clone, Copy, Debug)]
struct Geometry {
    nodes: usize,
    stack: usize,
    values: usize,
}
#[derive(Debug)]
enum ConstructionCause {
    Overflow,
    Source,
    Capacity,
    Allocation(TryReserveError),
}
pub(crate) struct ConstructionFailure<V> {
    cause: ConstructionCause,
    scratch: Scratch<V>,
}
impl<V> fmt::Debug for ConstructionFailure<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MappingConstructionFailure")
            .field("cause", &self.cause)
            .field("node_capacity", &self.scratch.nodes.capacity())
            .field("value_capacity", &self.scratch.values.capacity())
            .finish()
    }
}
impl<V> fmt::Display for ConstructionFailure<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            ConstructionCause::Overflow => f.write_str("mapping source geometry overflow"),
            ConstructionCause::Source => f.write_str("mapping source reference is invalid"),
            ConstructionCause::Capacity => {
                f.write_str("mapping destination differs from its exact capacity")
            }
            ConstructionCause::Allocation(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl<V: 'static> std::error::Error for ConstructionFailure<V> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            ConstructionCause::Allocation(e) => Some(e),
            _ => None,
        }
    }
}
fn fail<V>(cause: ConstructionCause) -> ConstructionFailure<V> {
    ConstructionFailure {
        cause,
        scratch: Scratch::new(),
    }
}
/// The mutable source loan prevents another arena from supplying its nodes.
pub(crate) struct Plan<'a, V> {
    source: &'a mut ExprSet,
    geometry: Geometry,
    requirements: Requirements,
    marker: PhantomData<V>,
}
impl<V> fmt::Debug for Plan<'_, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MappingWorkspacePlan")
            .field("requirements", &self.requirements)
            .finish()
    }
}
pub(crate) struct Workspace<'a, V> {
    source: &'a mut ExprSet,
    scratch: Scratch<V>,
    geometry: Geometry,
    failed: bool,
}
/// Storage only. An enclosing machine retains the exact source owner and
/// reborrows it for each call; this object grants no source/execution authority.
pub(crate) struct Parts<V> {
    scratch: Scratch<V>,
    geometry: Geometry,
    failed: bool,
}
impl<V> Parts<V> {
    pub(crate) fn prepare_reached(
        &mut self,
        source: &ExprSet,
    ) -> Result<(), crate::ast::PreparedExprError> {
        if source.len() <= self.geometry.nodes {
            return Ok(());
        }
        let (nodes, stack, values) = source.reached_workspace_geometry()?;
        source.grow_prepared_workspace(&mut self.scratch.nodes, stack)?;
        source.grow_prepared_workspace(&mut self.scratch.values, values)?;
        self.geometry.nodes = nodes;
        self.geometry.stack = self.geometry.stack.max(stack);
        self.geometry.values = self.geometry.values.max(values);
        Ok(())
    }
    pub(crate) fn bind(self, source: &mut ExprSet) -> Workspace<'_, V> {
        Workspace {
            source,
            scratch: self.scratch,
            geometry: self.geometry,
            failed: self.failed,
        }
    }
}
impl<V> Workspace<'_, V> {
    pub(crate) fn into_parts(self) -> Parts<V> {
        Parts {
            scratch: self.scratch,
            geometry: self.geometry,
            failed: self.failed,
        }
    }
}
impl<V> fmt::Debug for Workspace<'_, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MappingWorkspace")
            .field("geometry", &self.geometry)
            .field("failed", &self.failed)
            .finish()
    }
}
impl ExprSet {
    /// Quotes only actual source traversal/value slots. Copied V payloads,
    /// memo storage and processing allocations have separate producers.
    pub(crate) fn mapping_workspace_plan<V>(
        &mut self,
    ) -> Result<Plan<'_, V>, ConstructionFailure<V>> {
        let result = (|| -> Result<(Geometry, Requirements), ConstructionCause> {
            let mut geometry = Geometry {
                nodes: self.len(),
                stack: 1,
                values: 0,
            };
            for i in 1..self.len() {
                let r = ExprRef::new(u32::try_from(i).map_err(|_| ConstructionCause::Overflow)?);
                let args = self.get_args(r);
                if args
                    .iter()
                    .any(|r| !r.is_valid() || r.as_usize() >= self.len())
                {
                    return Err(ConstructionCause::Source);
                }
                geometry.stack = geometry
                    .stack
                    .checked_add(args.len())
                    .ok_or(ConstructionCause::Overflow)?;
                geometry.values = geometry.values.max(args.len());
            }
            let buffers = Layout::array::<ExprRef>(geometry.stack)
                .map_err(|_| ConstructionCause::Overflow)?
                .size()
                .checked_add(
                    Layout::array::<V>(geometry.values)
                        .map_err(|_| ConstructionCause::Overflow)?
                        .size(),
                )
                .ok_or(ConstructionCause::Overflow)?;
            let controls =
                Plan::<V>::inspection_control_bytes().ok_or(ConstructionCause::Overflow)?;
            let total = buffers
                .checked_add(controls)
                .ok_or(ConstructionCause::Overflow)?;
            Ok((
                geometry,
                Requirements {
                    buffers,
                    controls,
                    total,
                },
            ))
        })();
        match result {
            Ok((geometry, requirements)) => Ok(Plan {
                source: self,
                geometry,
                requirements,
                marker: PhantomData,
            }),
            Err(cause) => Err(fail(cause)),
        }
    }
}
impl<'a, V> Plan<'a, V> {
    /// Uses only the same owner's finite encoding/entry destinations. This mode
    /// may visit newly emitted, valid nodes of that owner without granting growth.
    pub(crate) fn prepared_bounds(mut self) -> Result<Self, ConstructionFailure<V>> {
        let (_, nodes, encoded) = self
            .source
            .prepared_extents()
            .map_err(|_| fail(ConstructionCause::Source))?;
        let args = encoded
            .checked_sub(1)
            .ok_or_else(|| fail(ConstructionCause::Source))?;
        if nodes < self.geometry.nodes || args < self.geometry.values {
            return Err(fail(ConstructionCause::Source));
        }
        let stack = nodes
            .checked_mul(args)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| fail(ConstructionCause::Overflow))?;
        let values = args
            .checked_mul(args)
            .ok_or_else(|| fail(ConstructionCause::Overflow))?;
        let buffers = Layout::array::<ExprRef>(stack)
            .map_err(|_| fail(ConstructionCause::Overflow))?
            .size()
            .checked_add(
                Layout::array::<V>(values)
                    .map_err(|_| fail(ConstructionCause::Overflow))?
                    .size(),
            )
            .ok_or_else(|| fail(ConstructionCause::Overflow))?;
        self.geometry = Geometry {
            nodes,
            stack,
            values,
        };
        self.requirements.buffers = buffers;
        self.requirements.total = buffers
            .checked_add(self.requirements.controls)
            .ok_or_else(|| fail(ConstructionCause::Overflow))?;
        Ok(self)
    }
    pub(crate) fn inspection_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<Workspace<'a, V>>(),
            size_of::<Parts<V>>(),
            size_of::<Scratch<V>>(),
            size_of::<Geometry>(),
            size_of::<Requirements>(),
            size_of::<ConstructionFailure<V>>(),
            size_of::<ConstructionCause>(),
            size_of::<Result<Self, ConstructionFailure<V>>>(),
            size_of::<Result<Workspace<'a, V>, ConstructionFailure<V>>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Result<Layout, std::alloc::LayoutError>>(),
            size_of::<std::slice::Iter<'_, ExprRef>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    pub(crate) fn requirements(&self) -> Requirements {
        self.requirements
    }
    pub(crate) fn compile(self) -> Result<Workspace<'a, V>, ConstructionFailure<V>> {
        let mut scratch = Scratch::new();
        let result = (|| -> Result<(), ConstructionCause> {
            scratch
                .nodes
                .try_reserve_exact(self.geometry.stack)
                .map_err(ConstructionCause::Allocation)?;
            if scratch.nodes.capacity() != self.geometry.stack {
                return Err(ConstructionCause::Capacity);
            }
            scratch
                .values
                .try_reserve_exact(self.geometry.values)
                .map_err(ConstructionCause::Allocation)?;
            if size_of::<V>() != 0 && scratch.values.capacity() != self.geometry.values {
                return Err(ConstructionCause::Capacity);
            }
            Ok(())
        })();
        match result {
            Ok(()) => Ok(Workspace {
                source: self.source,
                scratch,
                geometry: self.geometry,
                failed: false,
            }),
            Err(cause) => Err(ConstructionFailure { cause, scratch }),
        }
    }
}
#[derive(Debug)]
pub(crate) enum OperationError<E> {
    Failed,
    Source { node: ExprRef, source_nodes: usize },
    NodeCapacity,
    ValueCapacity,
    Copy(E),
    Process(E),
    Memo(E),
}
impl<E: fmt::Display> fmt::Display for OperationError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Failed => f.write_str("mapping workspace retains a failed prefix"),
            Self::Source { node, source_nodes } => write!(
                f,
                "mapping node {} is outside its source of {source_nodes} nodes",
                node.as_u32()
            ),
            Self::NodeCapacity => f.write_str("mapping node destinations exhausted"),
            Self::ValueCapacity => f.write_str("mapping value destinations exhausted"),
            Self::Copy(e) | Self::Process(e) | Self::Memo(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for OperationError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Copy(e) | Self::Process(e) | Self::Memo(e) => Some(e),
            _ => None,
        }
    }
}
struct Fixed<'a, F> {
    geometry: Geometry,
    copy: &'a mut F,
}
impl<V, E, F: FnMut(&V) -> Result<V, E>> Memory<V> for Fixed<'_, F> {
    type Error = OperationError<E>;
    fn begin(&mut self, scratch: &mut Scratch<V>, root: ExprRef) -> Result<(), Self::Error> {
        self.check(root)?;
        scratch.nodes.clear();
        scratch.values.clear();
        self.push_node(&mut scratch.nodes, root)
    }
    fn copy(&mut self, value: &V) -> Result<V, Self::Error> {
        (self.copy)(value).map_err(OperationError::Copy)
    }
    fn push_node(&mut self, nodes: &mut Vec<ExprRef>, node: ExprRef) -> Result<(), Self::Error> {
        self.check(node)?;
        if nodes.len() >= self.geometry.stack || nodes.len() == nodes.capacity() {
            return Err(OperationError::NodeCapacity);
        }
        nodes.push(node);
        Ok(())
    }
    fn push_value(&mut self, values: &mut Vec<V>, value: V) -> Result<(), Self::Error> {
        if values.len() >= self.geometry.values || values.len() == values.capacity() {
            return Err(OperationError::ValueCapacity);
        }
        values.push(value);
        Ok(())
    }
}
impl<F> Fixed<'_, F> {
    fn check<E>(&self, node: ExprRef) -> Result<(), OperationError<E>> {
        if !node.is_valid() || node.as_usize() >= self.geometry.nodes {
            return Err(OperationError::Source {
                node,
                source_nodes: self.geometry.nodes,
            });
        }
        Ok(())
    }
}
struct Memo<'a, C>(&'a mut C);
impl<K, V, E, C: Cache<K, V, Error = E>> Cache<K, V> for Memo<'_, C> {
    type Error = OperationError<E>;
    fn get(&self, key: &K) -> Option<&V> {
        self.0.get(key)
    }
    fn contains_key(&self, key: &K) -> bool {
        self.0.contains_key(key)
    }
    fn insert(&mut self, key: K, value: V) -> Result<(), Self::Error> {
        self.0.insert(key, value).map_err(OperationError::Memo)
    }
}
impl<V> Workspace<'_, V> {
    pub(crate) fn source(&self) -> &ExprSet {
        self.source
    }

    /// The exact captured source may append expressions inside its separately
    /// qualified processor. Prepared bounds admit valid new nodes only inside
    /// that same finite owner; plain source bounds retain their original extent.
    /// No implicit value cloning, memo growth or processing permission exists.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn map<K: Eq + Hash, E>(
        &mut self,
        root: ExprRef,
        cache: &mut impl Cache<K, V, Error = E>,
        skip: bool,
        key: impl Fn(ExprRef) -> K,
        mut copy: impl FnMut(&V) -> Result<V, E>,
        mut process: impl FnMut(&mut ExprSet, &mut Vec<V>, ExprRef) -> Result<V, E>,
    ) -> Result<V, OperationError<E>> {
        if self.failed {
            return Err(OperationError::Failed);
        }
        let mut memory = Fixed {
            geometry: self.geometry,
            copy: &mut copy,
        };
        if !self.source.is_valid(root) {
            self.failed = true;
            return Err(OperationError::Source {
                node: root,
                source_nodes: self.geometry.nodes,
            });
        }
        if let Err(error) = memory.check(root) {
            self.failed = true;
            return Err(error);
        }
        let result = self.source.try_map_with_storage(
            root,
            &mut Memo(cache),
            skip,
            key,
            &mut self.scratch,
            &mut memory,
            |e, v, r| process(e, v, r).map_err(OperationError::Process),
        );
        if result.is_err() {
            self.failed = true;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Expr, ExprFlags};
    use std::collections::BTreeMap;
    #[derive(Debug, PartialEq, Eq)]
    struct CopyFailure(ExprRef);
    #[derive(Default)]
    struct MemoStore {
        values: BTreeMap<ExprRef, ExprRef>,
        publications: usize,
    }
    impl Cache<ExprRef, ExprRef> for MemoStore {
        type Error = CopyFailure;
        fn get(&self, key: &ExprRef) -> Option<&ExprRef> {
            self.values.get(key)
        }
        fn contains_key(&self, key: &ExprRef) -> bool {
            self.values.contains_key(key)
        }
        fn insert(&mut self, key: ExprRef, value: ExprRef) -> Result<(), CopyFailure> {
            self.publications += 1;
            self.values.insert(key, value);
            Ok(())
        }
    }
    #[test]
    fn source_bound_mapping_buffers_keep_mutations_memo_prefixes_and_failed_scope_state() {
        let mut source = ExprSet::new(256);
        let a = source.mk(Expr::Byte(b'a'));
        let b = source.mk(Expr::Byte(b'b'));
        let root = source.mk(Expr::Concat(ExprFlags::POSITIVE, [a, b]));
        let initial_nodes = source.len();
        let plan = source.mapping_workspace_plan::<ExprRef>().unwrap();
        let requirements = plan.requirements();
        assert!(requirements.buffers > 0 && requirements.controls > 0);
        assert_eq!(
            requirements.total,
            requirements.buffers + requirements.controls
        );
        let mut workspace = plan.compile().unwrap();
        let capacities = (
            workspace.scratch.nodes.capacity(),
            workspace.scratch.values.capacity(),
        );
        let mut memo = MemoStore::default();
        let mut created = None;
        let result = workspace
            .map(
                root,
                &mut memo,
                false,
                |r| r,
                |value| Ok(*value),
                |expressions, _, node| {
                    if node == a {
                        created = Some(expressions.mk(Expr::Byte(b'z')));
                    }
                    Ok(node)
                },
            )
            .unwrap();
        assert_eq!(result, root);
        assert_eq!(memo.publications, 3);
        assert_eq!(
            (
                workspace.scratch.nodes.capacity(),
                workspace.scratch.values.capacity()
            ),
            capacities
        );
        let created = created.unwrap();
        assert!(created.as_usize() >= initial_nodes);
        let error = workspace
            .map(
                created,
                &mut memo,
                false,
                |r| r,
                |v| Ok(*v),
                |_, _, _| panic!("foreign scope root must not process"),
            )
            .unwrap_err();
        assert!(
            matches!(error,OperationError::Source{source_nodes,..} if source_nodes==initial_nodes)
        );
        assert!(matches!(
            workspace.map(
                root,
                &mut memo,
                false,
                |r| r,
                |v| Ok(*v),
                |_, _, _| panic!("failed scope must not resume")
            ),
            Err(OperationError::Failed)
        ));
        assert_eq!(memo.publications, 3);
        drop(workspace);
        // Failure retires traversal custody; it does not roll back actual
        // expression mutations or erase already published memo entries.
        assert!(source.is_valid(created));
        assert_eq!(memo.values.get(&root), Some(&root));

        let mut workspace = source
            .mapping_workspace_plan::<ExprRef>()
            .unwrap()
            .compile()
            .unwrap();
        let capacities = (
            workspace.scratch.nodes.capacity(),
            workspace.scratch.values.capacity(),
        );
        let mut memo = MemoStore::default();
        let error = workspace
            .map(
                root,
                &mut memo,
                false,
                |r| r,
                |v| Err(CopyFailure(*v)),
                |_, _, r| Ok(r),
            )
            .unwrap_err();
        assert!(matches!(error,OperationError::Copy(CopyFailure(node)) if node==a));
        assert_eq!(memo.publications, 2);
        assert_eq!(memo.values.get(&a), Some(&a));
        assert_eq!(memo.values.get(&b), Some(&b));
        assert!(!memo.values.contains_key(&root));
        assert_eq!(workspace.scratch.nodes, vec![root]);
        assert_eq!(
            (
                workspace.scratch.nodes.capacity(),
                workspace.scratch.values.capacity()
            ),
            capacities
        );
        assert!(workspace.failed);
        drop(workspace);
        assert!(source.is_valid(created));

        source.mk(Expr::Not(ExprFlags::POSITIVE, ExprRef::new(u32::MAX)));
        let failure = source.mapping_workspace_plan::<ExprRef>().unwrap_err();
        assert!(matches!(failure.cause, ConstructionCause::Source));
        assert_eq!(failure.scratch.nodes.capacity(), 0);
        assert_eq!(failure.scratch.values.capacity(), 0);
    }
}

impl<V> Parts<V> {
    pub(crate) fn copy_destination_controls<E>(&self) -> Option<usize> {
        crate::copy_storage::frame_bytes::<(&Self, Self), E>()
    }
    pub(crate) fn copy_destination(&self) -> Self {
        Self { scratch: Scratch::new(), geometry: self.geometry, failed: self.failed }
    }
    pub(crate) fn copy_required_bytes<E, G>(&self, values: G) -> Option<usize>
    where G: FnOnce(&Vec<V>) -> Option<usize> {
        use crate::copy_storage as copy;
        copy::frame_bytes::<(&mut Self, &Self, G), E>()?
            .checked_add(copy::fixed_required_bytes::<_, E>(&self.scratch.nodes)?)?
            .checked_add(values(&self.scratch.values)?)
    }
}
impl<V> Parts<V> {
    pub(crate) fn restore_copy<F, E, G>(
        &mut self, source: &Self, funding: &F, values: G,
    ) -> Result<(), crate::copy_storage::Error<E>>
    where F: Fn(usize) -> Result<(), E>,
        G: FnOnce(&mut Vec<V>, &Vec<V>, &F) -> Result<(), crate::copy_storage::Error<E>>,
    {
        use crate::copy_storage as copy;
        copy::frame::<(&mut Self, &Self, G), _, _>(funding)?;
        copy::fixed(&mut self.scratch.nodes, &source.scratch.nodes, funding)?;
        values(&mut self.scratch.values, &source.scratch.values, funding)?;
        self.geometry = source.geometry;
        self.failed = source.failed;
        Ok(())
    }
}
