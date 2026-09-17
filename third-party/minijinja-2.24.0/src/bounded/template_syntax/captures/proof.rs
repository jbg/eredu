//! Test-only observation around the actual shared worker and real closed store.
//! The three visitation vectors belong to this observer, not the production workspace.
#![forbid(unsafe_code)]
use super::tests::{fixtures, syntax};
use super::*;
use crate::compiler::meta::view::{Storage, Task};

struct Observed<'o, 's: 'o> {
    scratch: Scratch<'o, 's>,
    expressions: Vec<bool>,
    statements: Vec<bool>,
    macros: Vec<bool>,
    appends: [usize; 4],
}
impl<'o, 's: 'o> Storage<'s, Packed<'o, 's>> for Observed<'o, 's> {
    type Error = Cause;
    type Nested = ();
    fn push_task(&mut self, task: Task<'s, Packed<'o, 's>>) -> Result<(), Cause> {
        self.scratch.push_task(task)?;
        self.appends[Buffer::Tasks.index()] += 1;
        Ok(())
    }
    fn pop_task(&mut self) -> Option<Task<'s, Packed<'o, 's>>> {
        let task = self.scratch.pop_task()?;
        let seen = match task {
            Task::Expr(id) | Task::Assign(id) => Some(&mut self.expressions[id.0]),
            Task::Stmt(id) => Some(&mut self.statements[id.0]),
            Task::Macro(id, _) => Some(&mut self.macros[id.0]),
            _ => None,
        };
        if let Some(seen) = seen {
            assert!(!*seen, "one reachable node was visited twice");
            *seen = true;
        }
        Some(task)
    }
    fn is_assigned(&self, name: &str) -> bool {
        self.scratch.is_assigned(name)
    }
    fn assign(&mut self, name: &'s str) -> Result<(), Cause> {
        let before = self.scratch.assigned.len();
        self.scratch.assign(name)?;
        self.appends[Buffer::Assigned.index()] += self.scratch.assigned.len() - before;
        Ok(())
    }
    fn capture(&mut self, name: &'s str) -> Result<(), Cause> {
        let before = self.scratch.captures.len();
        self.scratch.capture(name)?;
        self.appends[Buffer::Captures.index()] += self.scratch.captures.len() - before;
        Ok(())
    }
    fn push_scope(&mut self) -> Result<(), Cause> {
        self.scratch.push_scope()?;
        self.appends[Buffer::Scopes.index()] += 1;
        Ok(())
    }
    fn pop_scope(&mut self) -> Result<(), Cause> {
        self.scratch.pop_scope()
    }
    fn tracks_nested(&self) -> bool {
        self.scratch.tracks_nested()
    }
    fn nested_start(&mut self, name: &'s str) -> Result<(), Cause> {
        self.scratch.nested_start(name)
    }
    fn nested_attr(&mut self, attrs: &mut (), name: &'s str) -> Result<(), Cause> {
        self.scratch.nested_attr(attrs, name)
    }
    fn nested_finish(&mut self, attrs: (), name: &'s str) -> Result<(), Cause> {
        self.scratch.nested_finish(attrs, name)
    }
    fn nested_variable(&mut self, name: &'s str) -> Result<(), Cause> {
        self.scratch.nested_variable(name)
    }
}
fn prove(owner: &ParsedTemplate<'_>) {
    let Some(first) = owner.macros().next() else {
        return;
    };
    let plan = Plan::for_macro(first).unwrap();
    let limits = plan.requirements.counts;
    let workspace = plan.construct().unwrap();
    let mut observed = Observed {
        scratch: workspace.scratch,
        expressions: vec![false; owner.storage.expressions.nodes.len()],
        statements: vec![false; owner.storage.statements.len()],
        macros: vec![false; owner.storage.statements.len()],
        appends: [0; 4],
    };
    for handle in owner.macros() {
        observed.scratch.clear();
        observed.expressions.fill(false);
        observed.statements.fill(false);
        observed.macros.fill(false);
        observed.appends.fill(0);
        observed.push_scope().unwrap();
        shared::run(Packed(owner), &mut observed, Task::Macro(handle.id, false)).unwrap();
        assert_eq!(observed.scratch.scopes.len(), 1);
        assert!(observed.scratch.tasks.is_empty());
        for (actual, bound) in observed.appends.into_iter().zip(limits) {
            assert!(
                actual <= bound,
                "actual append total {actual} exceeds {bound}"
            );
        }
    }
}

#[test]
fn actual_capture_task_and_append_events_fit_the_source_inventory_without_duplicate_visits() {
    prove(&syntax(super::tests::SOURCE));
    for (_, source) in fixtures() {
        let plan = super::super::Plan::inspect(source, "proof", Default::default()).unwrap();
        if let Ok(owner) = plan.construct() {
            prove(&owner);
        }
    }
}
