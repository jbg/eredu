use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
    rc::Rc,
};

use safemlx::{error::Exception, ops::zeros_dtype, Array, Dtype, Stream};

use crate::backend::nn::nested::NestedValue;

use super::PhysicalParameters;

/// Trait for a module parameter.
pub trait PhysicalParameter {
    /// Freeze the parameter.
    fn freeze(&mut self, recursive: bool);

    /// Unfreeze the parameter.
    fn unfreeze(&mut self, recursive: bool);

    /// Get the parameter as a nested value.
    fn as_nested_value(&self) -> NestedValue<Rc<str>, &Array>;

    /// Get the parameter as a mutable nested value.
    fn as_nested_value_mut(&mut self) -> NestedValue<Rc<str>, &mut Array>;

    /// Get the parameter as a nested value if it is trainable.
    fn as_trainable_nested_value(&self) -> Option<NestedValue<Rc<str>, &Array>>;
}

/// A simple wrapper for a module parameter.
#[derive(Debug, Clone)]
pub struct PhysicalParam<T> {
    /// The value of the parameter.
    pub value: T,

    /// Whether the parameter is frozen.
    ///
    /// Access this state through the [`PhysicalParameter`] trait.
    is_frozen: bool,
}

impl<T> PhysicalParam<T> {
    /// Create a new `PhysicalParam`
    pub fn new(value: T) -> Self {
        Self {
            value,
            is_frozen: false,
        }
    }
}

impl<T> From<T> for PhysicalParam<T> {
    fn from(inner: T) -> Self {
        Self::new(inner)
    }
}

impl PhysicalParam<Array> {
    /// Create a placeholder parameter with the expected shape but no
    /// materialized tensor contents.
    ///
    /// This is useful for models that will immediately strict-load real
    /// checkpoint weights, while still needing shape metadata for validation.
    pub fn unloaded(
        shape: &[i32],
        dtype: Dtype,
        stream: impl AsRef<Stream>,
    ) -> Result<Self, Exception> {
        Ok(Self::new(zeros_dtype(shape, dtype, stream)?))
    }
}

impl PhysicalParam<Option<Array>> {
    /// Create a present optional placeholder parameter with the expected shape
    /// but no materialized tensor contents.
    pub fn unloaded_some(
        shape: &[i32],
        dtype: Dtype,
        stream: impl AsRef<Stream>,
    ) -> Result<Self, Exception> {
        Ok(Self::new(Some(
            PhysicalParam::<Array>::unloaded(shape, dtype, stream)?.value,
        )))
    }
}

impl<T> Deref for PhysicalParam<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T> DerefMut for PhysicalParam<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

impl<T> AsRef<T> for PhysicalParam<T> {
    fn as_ref(&self) -> &T {
        &self.value
    }
}

impl<T> AsMut<T> for PhysicalParam<T> {
    fn as_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

impl PhysicalParameter for PhysicalParam<Array> {
    fn freeze(&mut self, _recursive: bool) {
        self.is_frozen = true;
    }

    fn unfreeze(&mut self, _recursive: bool) {
        self.is_frozen = false;
    }

    fn as_nested_value<'a>(&self) -> NestedValue<Rc<str>, &Array> {
        NestedValue::Value(&self.value)
    }

    fn as_nested_value_mut<'a>(&mut self) -> NestedValue<Rc<str>, &mut Array> {
        NestedValue::Value(&mut self.value)
    }

    fn as_trainable_nested_value<'a>(&self) -> Option<NestedValue<Rc<str>, &Array>> {
        match self.is_frozen {
            true => None,
            false => Some(NestedValue::Value(&self.value)),
        }
    }
}

impl PhysicalParameter for PhysicalParam<Option<Array>> {
    fn freeze(&mut self, _recursive: bool) {
        self.is_frozen = true;
    }

    fn unfreeze(&mut self, _recursive: bool) {
        self.is_frozen = false;
    }

    fn as_nested_value(&self) -> NestedValue<Rc<str>, &Array> {
        match &self.value {
            Some(array) => NestedValue::Value(array),
            // An empty map entry will be ignored during flattening
            None => NestedValue::Map(HashMap::with_capacity(0)),
        }
    }

    fn as_nested_value_mut(&mut self) -> NestedValue<Rc<str>, &mut Array> {
        match &mut self.value {
            Some(array) => NestedValue::Value(array),
            // An empty map entry will be ignored during flattening
            None => NestedValue::Map(HashMap::with_capacity(0)),
        }
    }

    fn as_trainable_nested_value(&self) -> Option<NestedValue<Rc<str>, &Array>> {
        match self.is_frozen {
            true => None,
            false => self.value.as_ref().map(NestedValue::Value),
        }
    }
}

impl<M> PhysicalParameter for Option<M>
where
    M: PhysicalParameters,
{
    fn freeze(&mut self, recursive: bool) {
        if let Some(m) = self.as_mut() {
            m.freeze(recursive);
        }
    }

    fn unfreeze(&mut self, recursive: bool) {
        if let Some(m) = self.as_mut() {
            m.unfreeze(recursive);
        }
    }

    fn as_nested_value(&self) -> NestedValue<Rc<str>, &Array> {
        match self {
            Some(m) => m.as_nested_value(),
            None => NestedValue::Map(HashMap::with_capacity(0)),
        }
    }

    fn as_nested_value_mut(&mut self) -> NestedValue<Rc<str>, &mut Array> {
        match self {
            Some(m) => m.as_nested_value_mut(),
            None => NestedValue::Map(HashMap::with_capacity(0)),
        }
    }

    fn as_trainable_nested_value(&self) -> Option<NestedValue<Rc<str>, &Array>> {
        match self {
            Some(m) => m.as_trainable_nested_value(),
            None => None,
        }
    }
}

impl<T> PhysicalParameter for T
where
    T: PhysicalParameters,
{
    fn freeze(&mut self, recursive: bool) {
        self.freeze_parameters(recursive);
    }

    fn unfreeze(&mut self, recursive: bool) {
        self.unfreeze_parameters(recursive);
    }

    fn as_nested_value(&self) -> NestedValue<Rc<str>, &Array> {
        self.parameters().into()
    }

    fn as_nested_value_mut(&mut self) -> NestedValue<Rc<str>, &mut Array> {
        self.parameters_mut().into()
    }

    fn as_trainable_nested_value(&self) -> Option<NestedValue<Rc<str>, &Array>> {
        Some(self.trainable_parameters().into())
    }
}
