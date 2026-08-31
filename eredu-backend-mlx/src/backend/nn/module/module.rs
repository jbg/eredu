use std::{collections::HashMap, rc::Rc};

use safemlx::{Array, Stream};

use crate::backend::nn::nested::{NestedHashMap, NestedValue};

/// Type alias for borrowed module parameters.
pub type ModuleParamRef<'a> = NestedHashMap<Rc<str>, &'a Array>;

/// Type alias for mutably borrowed module parameters.
pub type ModuleParamMut<'a> = NestedHashMap<Rc<str>, &'a mut Array>;

/// Type alias for borrowed flattened module parameters.
pub type FlattenedModuleParamRef<'a> = HashMap<Rc<str>, &'a Array>;

/// Type alias for mutably borrowed flattened module parameters.
pub type FlattenedModuleParamMut<'a> = HashMap<Rc<str>, &'a mut Array>;

/// Trait for a neural network module.
pub trait Module<Input>: PhysicalParameters + std::fmt::Debug {
    /// Output type of the module.
    type Output;

    /// Error type for the module.
    type Error: std::error::Error;

    /// Forward pass of the module.
    fn forward(&mut self, input: Input, stream: &Stream) -> Result<Self::Output, Self::Error>;
}

/// Private traversal of physical MLX parameter slots.
///
/// Stable identities, trainability, checkpoint binding, and strict loading are
/// owned by [`eredu_nn::Parameterized`]. This trait exists only to let native
/// kernels expose their physical storage to the neutral MLX operator adapter.
pub trait PhysicalParameters {
    /// Get references to the module parameters.
    fn parameters(&self) -> ModuleParamRef<'_>;

    /// Get mutable references to the module parameters.
    fn parameters_mut(&mut self) -> ModuleParamMut<'_>;

    /// Get references to the trainable parameters. A parameter is trainable if it is NOT frozen.
    fn trainable_parameters(&self) -> ModuleParamRef<'_>;

    /// Freeze all parameters in the module.
    fn freeze_parameters(&mut self, recursive: bool);

    /// Unfreeze all parameters in the module.
    fn unfreeze_parameters(&mut self, recursive: bool);
}

impl<T> PhysicalParameters for &'_ mut T
where
    T: PhysicalParameters + ?Sized,
{
    fn parameters(&self) -> ModuleParamRef<'_> {
        (**self).parameters()
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        (**self).parameters_mut()
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        (**self).trainable_parameters()
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        (**self).freeze_parameters(recursive);
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        (**self).unfreeze_parameters(recursive);
    }
}

impl<T> PhysicalParameters for Box<T>
where
    T: PhysicalParameters + ?Sized,
{
    fn parameters(&self) -> ModuleParamRef<'_> {
        self.as_ref().parameters()
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        self.as_mut().parameters_mut()
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        self.as_ref().trainable_parameters()
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        self.as_mut().freeze_parameters(recursive);
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        self.as_mut().unfreeze_parameters(recursive);
    }
}

impl<T> PhysicalParameters for Vec<T>
where
    T: PhysicalParameters,
{
    fn parameters(&self) -> ModuleParamRef<'_> {
        let mut parameters = NestedHashMap::new();
        self.iter().enumerate().for_each(|(i, module)| {
            let value = module.parameters();
            parameters.insert(Rc::from(i.to_string()), NestedValue::Map(value.entries));
        });
        parameters
    }

    fn parameters_mut(&mut self) -> ModuleParamMut<'_> {
        let mut parameters = NestedHashMap::new();
        self.iter_mut().enumerate().for_each(|(i, module)| {
            let value = module.parameters_mut();
            parameters.insert(Rc::from(i.to_string()), NestedValue::Map(value.entries));
        });
        parameters
    }

    fn trainable_parameters(&self) -> ModuleParamRef<'_> {
        let mut parameters = NestedHashMap::new();
        self.iter().enumerate().for_each(|(i, module)| {
            let value = module.trainable_parameters();
            parameters.insert(Rc::from(i.to_string()), NestedValue::Map(value.entries));
        });
        parameters
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        self.iter_mut().for_each(|module| {
            module.freeze_parameters(recursive);
        });
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        self.iter_mut().for_each(|module| {
            module.unfreeze_parameters(recursive);
        });
    }
}
