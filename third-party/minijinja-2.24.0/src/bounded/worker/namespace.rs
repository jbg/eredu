//! Namespace identity and fields for the existing shared VM operations.
//! Live aliases occupy distinct slots. This keyword profile excludes namespace
//! objects inside fields, so every live object has a stack/local root and the
//! actual source bounds all simultaneous objects without an iteration guess.
use super::*;
use std::mem::{size_of, size_of_val};
#[derive(Clone, Copy, Debug, Default)]
pub(in crate::bounded) struct Namespace {
    active: bool,
    marked: bool,
    keywords: bool,
    len: usize,
}
pub(in crate::bounded) type NamespaceField = Local;
#[derive(Clone, Copy, Debug, Default)]
pub(in crate::bounded) struct Layout {
    pub objects: usize,
    pub fields: usize,
    pub per_object: usize,
    pub register_start: usize,
}
impl Layout {
    pub(in crate::bounded) fn inspect(source: &PreparedTemplate) -> Option<Self> {
        let mut present = false;
        let mut fields = 0usize;
        for instruction in &source.instructions {
            match instruction {
                Instruction::CallFunction(_, _) => present = true,
                Instruction::SetAttr(_) => fields = fields.checked_add(1)?,
                Instruction::BuildKwargs(count) => fields = fields.checked_add(*count)?,
                _ => {}
            }
        }
        let objects = if present {
            source
                .geometry
                .operands
                .checked_add(source.geometry.locals)?
                .checked_add(macros::closure_capacity(source)?)?
                .checked_add(1)?
        } else {
            0
        };
        Some(Self {
            objects,
            fields: objects.checked_mul(fields)?,
            per_object: fields,
            register_start: Borrowed::count(source.geometry)?,
        })
    }
}
fn local_value(local: &Local) -> Slot {
    local.value
}
fn reference(value: Slot) -> Option<usize> {
    match value {
        Slot::Namespace(index) | Slot::Kwargs(index) => Some(index),
        _ => None,
    }
}
impl<'source> Engine<'source, '_, '_> {
    fn namespace_layout(&self) -> Result<Layout, Error> {
        Layout::inspect(self.source).ok_or(Error::Overflow)
    }
    fn namespace_new(&mut self) -> Result<usize, Error> {
        for value in self.workspace.namespaces.iter_mut() {
            value.marked = false;
        }
        for slot in self.workspace.operands[..self.depth].iter().copied().chain(
            self.workspace.locals[..self.locals]
                .iter()
                .map(local_value as fn(&Local) -> Slot),
        ) {
            if let Some(index) = reference(slot) {
                let object = self
                    .workspace
                    .namespaces
                    .get_mut(index)
                    .ok_or(Error::Geometry)?;
                if !object.active {
                    return Err(Error::Geometry);
                }
                object.marked = true;
            }
        }
        for field in &*self.workspace.closure_fields {
            if let Some(index) = reference(field.value) {
                self.workspace
                    .namespaces
                    .get_mut(index)
                    .ok_or(Error::Geometry)?
                    .marked = true;
            }
        }
        let layout = self.namespace_layout()?;
        // Keyword containers may loan namespace arguments. Actual namespace
        // fields still reject nested objects, so one explicit edge pass suffices.
        for index in 0..self.workspace.namespaces.len() {
            let object = self.workspace.namespaces[index];
            if object.marked && object.keywords {
                let start = index
                    .checked_mul(layout.per_object)
                    .ok_or(Error::Overflow)?;
                for offset in start..start.checked_add(object.len).ok_or(Error::Overflow)? {
                    if let Some(target) = reference(
                        self.workspace
                            .namespace_fields
                            .get(offset)
                            .ok_or(Error::Geometry)?
                            .value,
                    ) {
                        self.workspace
                            .namespaces
                            .get_mut(target)
                            .ok_or(Error::Geometry)?
                            .marked = true;
                    }
                }
            }
        }
        let index = self
            .workspace
            .namespaces
            .iter()
            .position(|value| !value.marked)
            .ok_or(Error::Geometry)?;
        let start = index
            .checked_mul(layout.per_object)
            .ok_or(Error::Overflow)?;
        let end = start
            .checked_add(layout.per_object)
            .ok_or(Error::Overflow)?;
        self.workspace
            .namespace_fields
            .get_mut(start..end)
            .ok_or(Error::Geometry)?
            .fill(NamespaceField::default());
        for offset in start..end {
            self.borrowed.push(
                Slot::Undefined,
                layout
                    .register_start
                    .checked_add(offset)
                    .ok_or(Error::Overflow)?,
            )?;
        }
        self.workspace.namespaces[index] = Namespace {
            active: true,
            marked: true,
            keywords: false,
            len: 0,
        };
        Ok(index)
    }
    pub(super) fn keyword_range(&self, index: usize) -> Result<std::ops::Range<usize>, Error> {
        let len = self.namespace_len(index)?;
        let layout = self.namespace_layout()?;
        let start = index
            .checked_mul(layout.per_object)
            .ok_or(Error::Overflow)?;
        Ok(start..start.checked_add(len).ok_or(Error::Overflow)?)
    }
    pub(super) fn namespace_len(&self, index: usize) -> Result<usize, Error> {
        let object = self
            .workspace
            .namespaces
            .get(index)
            .ok_or(Error::Geometry)?;
        if !object.active {
            return Err(Error::Geometry);
        }
        Ok(object.len)
    }
    pub(super) fn namespace_attr(&self, index: usize, name: &str) -> Result<Slot, Error> {
        let len = self.namespace_len(index)?;
        let layout = self.namespace_layout()?;
        let start = index
            .checked_mul(layout.per_object)
            .ok_or(Error::Overflow)?;
        let end = start.checked_add(len).ok_or(Error::Overflow)?;
        for field in self
            .workspace
            .namespace_fields
            .get(start..end)
            .ok_or(Error::Geometry)?
        {
            if field.name.and_then(|range| range.text(&self.source.bytes)) == Some(name) {
                return Ok(field.value);
            }
        }
        Ok(Slot::Undefined)
    }
    pub(super) fn namespace_set(
        &mut self,
        index: usize,
        name: &str,
        value: Slot,
    ) -> Result<(), Error> {
        // Nested mutable object graphs need their own source/input population.
        // Refuse before storing an alias; never recycle an occupied namespace.
        let keywords = self
            .workspace
            .namespaces
            .get(index)
            .ok_or(Error::Geometry)?
            .keywords;
        if matches!(
            value,
            Slot::Kwargs(_) | Slot::Loop(_) | Slot::Closure(_) | Slot::Arguments { .. }
        ) || (!keywords && matches!(value, Slot::Namespace(_) | Slot::Macro(_)))
        {
            return Err(Error::Geometry);
        }
        let len = self.namespace_len(index)?;
        let layout = self.namespace_layout()?;
        let start = index
            .checked_mul(layout.per_object)
            .ok_or(Error::Overflow)?;
        let mut offset = len;
        for candidate in 0..len {
            if self
                .workspace
                .namespace_fields
                .get(start + candidate)
                .and_then(|field| field.name)
                .and_then(|range| range.text(&self.source.bytes))
                == Some(name)
            {
                offset = candidate;
                break;
            }
        }
        if offset >= layout.per_object {
            return Err(Error::Geometry);
        }
        let range = self
            .source
            .instructions
            .iter()
            .find_map(|instruction| match instruction {
                Instruction::SetAttr(range) | Instruction::Literal(range)
                    if range.text(&self.source.bytes) == Some(name) =>
                {
                    Some(*range)
                }
                _ => None,
            })
            .ok_or(Error::Geometry)?;
        let address = start.checked_add(offset).ok_or(Error::Overflow)?;
        let value = self.borrowed.push(
            value,
            layout
                .register_start
                .checked_add(address)
                .ok_or(Error::Overflow)?,
        )?;
        *self
            .workspace
            .namespace_fields
            .get_mut(address)
            .ok_or(Error::Geometry)? = NamespaceField {
            name: Some(range),
            value,
        };
        if offset == len {
            self.workspace.namespaces[index].len = len.checked_add(1).ok_or(Error::Overflow)?;
        }
        Ok(())
    }
    pub(super) fn namespace_keywords(&mut self, count: usize) -> Result<(), Error> {
        let words = count.checked_mul(2).ok_or(Error::Overflow)?;
        let start = self.depth.checked_sub(words).ok_or(Error::Geometry)?;
        let index = self.namespace_new()?;
        self.workspace.namespaces[index].keywords = true;
        // Same left-to-right key/value order as ordinary BuildKwargs. Retain
        // each input register before releasing the argument stack, including
        // duplicate-key replacement and Unicode names.
        for pair in 0..count {
            let position = start
                .checked_add(pair.checked_mul(2).ok_or(Error::Overflow)?)
                .ok_or(Error::Overflow)?;
            let key = self.workspace.operands[position];
            let value = self.workspace.operands[position + 1];
            let Slot::Text(Text::Atom(Atom::Source(range))) = key else {
                return Err(Error::Geometry);
            };
            let name = range.text(&self.source.bytes).ok_or(Error::Geometry)?;
            self.namespace_set(index, name, value)?;
        }
        while self.depth > start {
            self.pop()?;
        }
        self.push(Slot::Kwargs(index))
    }
    pub(super) fn namespace_call(
        &mut self,
        name: &'source str,
        args: Option<u16>,
    ) -> Result<(), Error> {
        if name != "namespace" || !matches!(self.lookup(name)?, Slot::Undefined) {
            return Err(Error::Geometry);
        }
        let index = match args {
            Some(0) => self.namespace_new()?,
            Some(1) => match self.pop()? {
                Slot::Kwargs(index) => index,
                _ => return Err(Error::Geometry),
            },
            _ => return Err(Error::Geometry),
        };
        self.namespace_len(index)?;
        let range = self.keyword_range(index)?;
        if self.workspace.namespace_fields[range]
            .iter()
            .any(|field| matches!(field.value, Slot::Namespace(_) | Slot::Macro(_)))
        {
            return Err(Error::Geometry);
        }
        self.workspace.namespaces[index].keywords = false;
        self.push(Slot::Namespace(index))
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    let parts = [
        size_of::<Layout>(),
        size_of::<Namespace>(),
        size_of::<NamespaceField>(),
        size_of::<[Slot; 2]>(),
        size_of::<(&str, usize, Slot)>(),
        size_of::<[usize; 6]>(),
        size_of::<Option<usize>>(),
        size_of::<Result<usize, Error>>(),
        size_of::<Result<Slot, Error>>(),
        size_of::<Result<(), Error>>(),
        size_of::<std::slice::Iter<'_, Namespace>>(),
        size_of::<std::slice::IterMut<'_, Namespace>>(),
        size_of::<std::slice::Iter<'_, NamespaceField>>(),
        size_of::<std::slice::Iter<'_, Slot>>(),
        size_of::<std::slice::Iter<'_, Local>>(),
        size_of::<std::slice::Iter<'_, Instruction>>(),
        size_of::<
            std::iter::Chain<
                std::iter::Copied<std::slice::Iter<'_, Slot>>,
                std::iter::Map<std::slice::Iter<'_, Local>, fn(&Local) -> Slot>,
            >,
        >(),
        size_of::<Result<Layout, Error>>(),
        size_of::<Option<Layout>>(),
        size_of::<Option<Range>>(),
        size_of::<Result<Option<usize>, Error>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
