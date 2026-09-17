//! The four actual once-reserved buffers; every append has a private bound.
#![forbid(unsafe_code)]
use super::{view::Packed, Buffer, Cause};
use crate::compiler::meta::view::{Storage, Task};

pub(super) struct Scratch<'o, 's: 'o> {
    pub(super) assigned: Vec<&'s str>,
    pub(super) scopes: Vec<usize>,
    pub(super) captures: Vec<&'s str>,
    pub(super) tasks: Vec<Task<'s, Packed<'o, 's>>>,
    limits: [usize; 4],
}
impl<'o, 's: 'o> Scratch<'o, 's> {
    pub(super) fn new(limits: [usize; 4]) -> Self {
        Self {
            assigned: Vec::new(),
            scopes: Vec::new(),
            captures: Vec::new(),
            tasks: Vec::new(),
            limits,
        }
    }
    pub(super) fn clear(&mut self) {
        self.assigned.clear();
        self.scopes.clear();
        self.captures.clear();
        self.tasks.clear();
    }
    pub(super) fn capacities(&self) -> [usize; 4] {
        [
            self.assigned.capacity(),
            self.scopes.capacity(),
            self.captures.capacity(),
            self.tasks.capacity(),
        ]
    }
    #[cfg(test)]
    pub(super) fn set_limit(&mut self, buffer: Buffer, limit: usize) {
        self.limits[buffer.index()] = limit;
    }
    fn before_push(&self, target: Buffer, len: usize, capacity: usize) -> Result<(), Cause> {
        if len >= self.limits[target.index()] || len == capacity {
            Err(Cause::Capacity(target))
        } else {
            Ok(())
        }
    }
}
impl<'o, 's: 'o> Storage<'s, Packed<'o, 's>> for Scratch<'o, 's> {
    type Error = Cause;
    type Nested = ();
    fn push_task(&mut self, value: Task<'s, Packed<'o, 's>>) -> Result<(), Cause> {
        self.before_push(Buffer::Tasks, self.tasks.len(), self.tasks.capacity())?;
        self.tasks.push(value);
        Ok(())
    }
    fn pop_task(&mut self) -> Option<Task<'s, Packed<'o, 's>>> {
        self.tasks.pop()
    }
    fn is_assigned(&self, name: &str) -> bool {
        self.assigned.contains(&name)
    }
    fn assign(&mut self, name: &'s str) -> Result<(), Cause> {
        let start = *self.scopes.last().ok_or(Cause::Geometry)?;
        if !self.assigned[start..].contains(&name) {
            self.before_push(
                Buffer::Assigned,
                self.assigned.len(),
                self.assigned.capacity(),
            )?;
            self.assigned.push(name);
        }
        Ok(())
    }
    fn capture(&mut self, name: &'s str) -> Result<(), Cause> {
        if !self.captures.contains(&name) {
            self.before_push(
                Buffer::Captures,
                self.captures.len(),
                self.captures.capacity(),
            )?;
            self.captures.push(name);
        }
        Ok(())
    }
    fn push_scope(&mut self) -> Result<(), Cause> {
        self.before_push(Buffer::Scopes, self.scopes.len(), self.scopes.capacity())?;
        self.scopes.push(self.assigned.len());
        Ok(())
    }
    fn pop_scope(&mut self) -> Result<(), Cause> {
        let start = self.scopes.pop().ok_or(Cause::Geometry)?;
        self.assigned.truncate(start);
        Ok(())
    }
    fn tracks_nested(&self) -> bool {
        false
    }
    fn nested_start(&mut self, _: &'s str) -> Result<(), Cause> {
        Err(Cause::Geometry)
    }
    fn nested_attr(&mut self, _: &mut (), _: &'s str) -> Result<(), Cause> {
        Err(Cause::Geometry)
    }
    fn nested_finish(&mut self, _: (), _: &'s str) -> Result<(), Cause> {
        Err(Cause::Geometry)
    }
    fn nested_variable(&mut self, _: &'s str) -> Result<(), Cause> {
        Err(Cause::Geometry)
    }
}
