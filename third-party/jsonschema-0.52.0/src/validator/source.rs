//! Cold census of the actual immutable graph; never used as execution authority.
use super::{
    workspace::{Component, Error},
    Validate,
};
use crate::{node::SchemaNode, paths::Location, Json};
use std::{
    alloc::Layout,
    mem::size_of_val,
    sync::{atomic::AtomicUsize, Arc},
};

pub(crate) struct Inspector<F: Json> {
    bytes: usize,
    seen: Vec<(usize, usize)>,
    key: fn(&F::PreparedKey) -> Option<usize>,
    number: fn(&serde_json::Number) -> Option<usize>,
}
impl<F: Json> Inspector<F> {
    pub(crate) fn new(
        key: fn(&F::PreparedKey) -> Option<usize>,
        number: fn(&serde_json::Number) -> Option<usize>,
    ) -> Self {
        Self {
            bytes: 0,
            seen: Vec::new(),
            key,
            number,
        }
    }
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }
    pub(crate) fn add(&mut self, bytes: usize) -> Result<(), Error> {
        self.bytes = self.bytes.checked_add(bytes).ok_or(Error::Overflow)?;
        Ok(())
    }
    pub(crate) fn vector<T>(&mut self, values: &Vec<T>) -> Result<(), Error> {
        self.add(
            Layout::array::<T>(values.capacity())
                .map_err(|_| Error::Overflow)?
                .size(),
        )
    }
    pub(crate) fn string(&mut self, value: &String) -> Result<(), Error> {
        self.add(value.capacity())
    }
    pub(crate) fn key(&mut self, value: &F::PreparedKey) -> Result<(), Error> {
        self.add(
            (self.key)(value).ok_or(Error::Unqualified(Component::Source(
                "prepared property key",
            )))?,
        )
    }
    pub(crate) fn arc<T: ?Sized>(&mut self, value: &Arc<T>) -> Result<bool, Error> {
        let id = Arc::as_ptr(value).cast::<()>() as usize;
        let bytes = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::for_value(value.as_ref()))
            .map_err(|_| Error::Overflow)?
            .0
            .pad_to_align()
            .size();
        if let Some((_, prior)) = self.seen.iter().find(|(prior, _)| *prior == id) {
            if *prior != bytes {
                return Err(Error::Capacity);
            }
            return Ok(false);
        }
        self.add(bytes)?;
        self.seen.push((id, bytes));
        Ok(true)
    }
    pub(crate) fn location(&mut self, value: &Location) -> Result<(), Error> {
        self.arc(&value.as_arc()).map(|_| ())
    }
    pub(crate) fn uri(
        &mut self,
        value: &Option<Arc<referencing::Uri<String>>>,
    ) -> Result<(), Error> {
        if let Some(value) = value {
            if self.arc(value)? {
                self.add(value.storage_capacity())?;
            }
        }
        Ok(())
    }
    pub(crate) fn formatted(&mut self, value: &std::sync::OnceLock<Arc<str>>) -> Result<(), Error> {
        if let Some(value) = value.get() {
            self.arc(value)?;
        }
        Ok(())
    }
    pub(crate) fn node(&mut self, value: &SchemaNode<F>) -> Result<(), Error> {
        value.original_source(self)
    }
    pub(crate) fn nodes(&mut self, values: &Vec<SchemaNode<F>>) -> Result<(), Error> {
        self.vector(values)?;
        for value in values {
            self.node(value)?;
        }
        Ok(())
    }
    pub(crate) fn boxed(
        &mut self,
        value: &crate::keywords::BoxedValidator<F>,
    ) -> Result<(), Error> {
        self.add(size_of_val(value.as_ref()))?;
        value.original_source(self)
    }
    pub(crate) fn number(&mut self, value: &serde_json::Number) -> Result<(), Error> {
        self.add(
            (self.number)(value).ok_or(Error::Unqualified(Component::Source(
                "serde number backing",
            )))?,
        )
    }
    pub(crate) fn literal(&mut self, value: &jsonschema_value::literal::Literal) -> Result<(), Error> {
        use jsonschema_value::literal::Literal;
        match value {
            Literal::Null | Literal::Bool(_) => Ok(()),
            Literal::String(value) => self.string(value),
            Literal::Number(value) => self.number(value),
            Literal::Array(values) => {
                self.vector(values)?;
                for value in values { self.literal(value)?; }
                Ok(())
            }
            Literal::Object(values) => {
                self.vector(values)?;
                for (key, value) in values { self.string(key)?; self.literal(value)?; }
                Ok(())
            }
        }
    }

    pub(crate) fn value(&mut self, value: &serde_json::Value) -> Result<(), Error> {
        use serde_json::Value;
        match value {
            Value::Null | Value::Bool(_) => Ok(()),
            Value::String(value) => self.string(value),
            Value::Number(value) => self.add((self.number)(value).ok_or(Error::Unqualified(
                Component::Source("serde number backing"),
            ))?),
            Value::Array(values) => {
                self.vector(values)?;
                for value in values {
                    self.value(value)?;
                }
                Ok(())
            }
            Value::Object(_) => Err(Error::Unqualified(Component::Source(
                "retained serde object backing",
            ))),
        }
    }
}
