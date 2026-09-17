//! One relevance DFS. The context owns equations; memory owns paid destinations.
use super::{RelevanceCache, SymRes};
use crate::{
    ast::{ExprRef, ExprSet},
    HashSet,
};

pub(crate) struct StackFrame {
    pub(crate) expression: ExprRef,
    pub(crate) cursor: usize,
    pub(crate) children: Vec<ExprRef>,
}
pub(crate) trait Context {
    type Error;
    fn cached(&self, node: ExprRef) -> Option<bool>;
    fn cache(&mut self, node: ExprRef, value: bool) -> Result<(), Self::Error>;
    fn positive(&self, node: ExprRef) -> bool;
    fn derivative(&mut self, node: ExprRef) -> Result<SymRes, Self::Error>;
    fn pay(&mut self, count: usize) -> Result<(), Self::Error>;
    fn prepare_weight(&mut self, node: ExprRef) -> Result<(), Self::Error>;
    fn weight(&self, node: ExprRef) -> u32;
    fn check_fuel(&self) -> Result<(), Self::Error>;
}
pub(crate) trait Memory {
    type Error;
    fn next(&mut self) -> Option<ExprRef>;
    fn visited(&self, node: ExprRef) -> bool;
    fn visit(&mut self, node: ExprRef) -> Result<(), Self::Error>;
    fn begin_children(&mut self, count: usize) -> Result<(), Self::Error>;
    fn child(&mut self, node: ExprRef) -> Result<(), Self::Error>;
    fn children(&mut self) -> &mut [ExprRef];
    fn push(&mut self, node: ExprRef) -> Result<(), Self::Error>;
    fn pop_expression(&mut self) -> Option<ExprRef>;
    fn publish_empty<C: Context<Error = Self::Error>>(
        &self,
        context: &mut C,
    ) -> Result<(), Self::Error>;
}
/// Caller memory retains all actual unfinished destinations on a typed failure.
pub(crate) fn run<C: Context, M: Memory<Error = C::Error>>(
    context: &mut C,
    memory: &mut M,
) -> Result<bool, C::Error> {
    while let Some(node) = memory.next() {
        if memory.visited(node) {
            continue;
        }
        let status = context.cached(node);
        if status == Some(false) {
            continue;
        }
        if status == Some(true) || context.positive(node) {
            while let Some(ancestor) = memory.pop_expression() {
                context.cache(ancestor, true)?;
            }
            return Ok(true);
        }
        let values = context.derivative(node)?;
        context.pay(values.len())?;
        memory.visit(node)?;
        let count = values.iter().filter(|v| !memory.visited(v.1)).count();
        memory.begin_children(count)?;
        for &(_, child) in &values {
            if !memory.visited(child) {
                memory.child(child)?;
            }
        }
        // Prepare fallible weights before entering std's infallible comparator.
        // This changes only when memo entries are populated, not sort order or
        // expression cost. Both paths use the same (weight, expression ID) key.
        for &child in memory.children().iter() {
            context.prepare_weight(child)?;
        }
        memory
            .children()
            .sort_unstable_by_key(|&child| (context.weight(child), child.as_u32()));
        memory.push(node)?;
        context.check_fuel()?;
    }
    memory.publish_empty(context)?;
    Ok(false)
}
struct OrdinaryContext<'a> {
    source: &'a mut ExprSet,
    cache: &'a mut RelevanceCache,
}
impl Context for OrdinaryContext<'_> {
    type Error = anyhow::Error;
    fn cached(&self, node: ExprRef) -> Option<bool> {
        self.cache.relevance_cache.get(&node).copied()
    }
    fn cache(&mut self, node: ExprRef, value: bool) -> anyhow::Result<()> {
        self.cache.relevance_cache.insert(node, value);
        Ok(())
    }
    fn positive(&self, node: ExprRef) -> bool {
        self.source.is_positive(node)
    }
    fn derivative(&mut self, node: ExprRef) -> anyhow::Result<SymRes> {
        Ok(self.cache.deriv(self.source, node))
    }
    fn pay(&mut self, count: usize) -> anyhow::Result<()> {
        self.source.pay(count);
        Ok(())
    }
    fn prepare_weight(&mut self, node: ExprRef) -> anyhow::Result<()> {
        self.source.get_weight(node);
        Ok(())
    }
    fn weight(&self, node: ExprRef) -> u32 {
        self.source.cached_weight(node)
    }
    fn check_fuel(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.source.cost() <= self.cache.cost_limit,
            "maximum relevance check fuel {} exceeded",
            self.cache.max_fuel
        );
        Ok(())
    }
}
struct OrdinaryMemory {
    visited: HashSet<ExprRef>,
    stack: Vec<StackFrame>,
    pending: Vec<ExprRef>,
}
impl Memory for OrdinaryMemory {
    type Error = anyhow::Error;
    fn next(&mut self) -> Option<ExprRef> {
        next(&mut self.stack)
    }
    fn visited(&self, node: ExprRef) -> bool {
        self.visited.contains(&node)
    }
    fn visit(&mut self, node: ExprRef) -> anyhow::Result<()> {
        self.visited.insert(node);
        Ok(())
    }
    fn begin_children(&mut self, count: usize) -> anyhow::Result<()> {
        self.pending = Vec::with_capacity(count);
        Ok(())
    }
    fn child(&mut self, node: ExprRef) -> anyhow::Result<()> {
        self.pending.push(node);
        Ok(())
    }
    fn children(&mut self) -> &mut [ExprRef] {
        &mut self.pending
    }
    fn push(&mut self, node: ExprRef) -> anyhow::Result<()> {
        if !self.pending.is_empty() {
            self.stack.push(StackFrame {
                expression: node,
                cursor: 0,
                children: std::mem::take(&mut self.pending),
            });
        }
        Ok(())
    }
    fn pop_expression(&mut self) -> Option<ExprRef> {
        self.stack.pop().map(|f| f.expression)
    }
    fn publish_empty<C: Context<Error = Self::Error>>(
        &self,
        context: &mut C,
    ) -> anyhow::Result<()> {
        for &node in &self.visited {
            context.cache(node, false)?;
        }
        Ok(())
    }
}
pub(crate) fn next(stack: &mut Vec<StackFrame>) -> Option<ExprRef> {
    loop {
        let frame = stack.last_mut()?;
        if frame.cursor == frame.children.len() {
            stack.pop();
            continue;
        }
        let node = frame.children[frame.cursor];
        frame.cursor += 1;
        return Some(node);
    }
}
pub(crate) fn ordinary(
    cache: &mut RelevanceCache,
    source: &mut ExprSet,
    root: ExprRef,
) -> anyhow::Result<bool> {
    let mut memory = OrdinaryMemory {
        visited: HashSet::default(),
        stack: vec![StackFrame {
            expression: root,
            cursor: 0,
            children: vec![root],
        }],
        pending: Vec::new(),
    };
    run(&mut OrdinaryContext { source, cache }, &mut memory)
}
