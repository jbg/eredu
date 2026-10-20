//! Typed access to pinned modules retained by a component kernel.
use crate::{
    ComponentAdapter, ComponentDefinition, ComponentDefinitionError, EmptyParameterModule,
    StaticParameterVisitor, StaticParameterVisitorMut,
};
use eredu_nn::{NeuralBackend, Parameterized};
use std::{borrow::Borrow, collections::BTreeSet};

/// Semantic role traversal and aggregate roots for a kernel's pinned resources.
/// Equations borrow these same resources directly from the kernel.
pub trait ComponentPinnedResources<B: NeuralBackend> {
    /// Aggregate module used for ordinary binding and completion roots.
    type Modules: Parameterized<B::Tensor>;
    /// Borrows every physically retained pinned parameter.
    fn pinned_modules(&self) -> &Self::Modules;
    /// Mutably borrows every physically retained pinned parameter.
    fn pinned_modules_mut(&mut self) -> &mut Self::Modules;
    /// Visits each physically retained role once, using its canonical storage name.
    fn visit_pinned<V: StaticParameterVisitor<B>>(&self, visitor: &mut V) -> Result<(), V::Error>;
    /// Mutably visits the same roles and physical modules.
    fn visit_pinned_mut<V: StaticParameterVisitorMut<B>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error>;
}

/// Selects kernel-owned pinned resources without a second resource owner.
#[derive(Debug, Default, Clone, Copy)]
pub struct KernelParameterModules;

/// Statically dispatched pinned-resource access used by the shared adapter.
pub trait ComponentResourceAccess<B: NeuralBackend, K> {
    /// Aggregate module retained by the ordinary runtime.
    type Modules: Parameterized<B::Tensor>;
    /// Borrows the selected aggregate roots.
    fn modules<'a>(&'a self, kernel: &'a K) -> &'a Self::Modules;
    /// Mutably borrows the selected aggregate roots.
    fn modules_mut<'a>(&'a mut self, kernel: &'a mut K) -> &'a mut Self::Modules;
    /// Visits canonical pinned roles.
    fn visit<V: StaticParameterVisitor<B>>(
        &self,
        kernel: &K,
        visitor: &mut V,
    ) -> Result<(), V::Error>;
    /// Mutably visits canonical pinned roles.
    fn visit_mut<V: StaticParameterVisitorMut<B>>(
        &mut self,
        kernel: &mut K,
        visitor: &mut V,
    ) -> Result<(), V::Error>;
}
impl<B: NeuralBackend, K> ComponentResourceAccess<B, K> for EmptyParameterModule {
    type Modules = Self;
    fn modules<'a>(&'a self, _: &'a K) -> &'a Self {
        self
    }
    fn modules_mut<'a>(&'a mut self, _: &'a mut K) -> &'a mut Self {
        self
    }
    fn visit<V: StaticParameterVisitor<B>>(&self, _: &K, _: &mut V) -> Result<(), V::Error> {
        Ok(())
    }
    fn visit_mut<V: StaticParameterVisitorMut<B>>(
        &mut self,
        _: &mut K,
        _: &mut V,
    ) -> Result<(), V::Error> {
        Ok(())
    }
}
impl<B: NeuralBackend, K: ComponentPinnedResources<B>> ComponentResourceAccess<B, K>
    for KernelParameterModules
{
    type Modules = K::Modules;
    fn modules<'a>(&'a self, kernel: &'a K) -> &'a Self::Modules {
        kernel.pinned_modules()
    }
    fn modules_mut<'a>(&'a mut self, kernel: &'a mut K) -> &'a mut Self::Modules {
        kernel.pinned_modules_mut()
    }
    fn visit<V: StaticParameterVisitor<B>>(
        &self,
        kernel: &K,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        kernel.visit_pinned(visitor)
    }
    fn visit_mut<V: StaticParameterVisitorMut<B>>(
        &mut self,
        kernel: &mut K,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        kernel.visit_pinned_mut(visitor)
    }
}

struct Roles(BTreeSet<String>);
impl<B: NeuralBackend> StaticParameterVisitor<B> for Roles {
    type Error = String;
    fn visit<M: Parameterized<B::Tensor>>(&mut self, role: &str, _: &M) -> Result<(), String> {
        if role.trim().is_empty() || !self.0.insert(role.to_owned()) {
            return Err(format!("duplicate or empty pinned role {role:?}"));
        }
        Ok(())
    }
}
impl<D: Borrow<ComponentDefinition>, K> ComponentAdapter<D, K, KernelParameterModules> {
    fn pinned<B: NeuralBackend>(definition: D, kernel: K) -> Result<Self, ComponentDefinitionError>
    where
        K: ComponentPinnedResources<B>,
    {
        let mut roles = Roles(BTreeSet::new());
        kernel
            .visit_pinned(&mut roles)
            .map_err(|resource| ComponentDefinitionError::Invalid {
                component: definition.borrow().id().to_owned(),
                resource,
            })?;
        Ok(Self::from_resources(
            definition,
            kernel,
            KernelParameterModules,
            roles.0,
        ))
    }
    /// Couples a complete declaration with exactly its physically retained pinned roles.
    pub fn with_pinned_resources<B: NeuralBackend>(
        definition: D,
        kernel: K,
    ) -> Result<Self, ComponentDefinitionError>
    where
        K: ComponentPinnedResources<B>,
    {
        let adapter = Self::pinned::<B>(definition, kernel)?;
        adapter.validate_resource_roles(adapter.definition().parameters().groups())?;
        Ok(adapter)
    }
    /// Couples selected local modules with the declaration and exact partition ownership.
    pub fn with_partition_resources<B: NeuralBackend, G, A>(
        definition: D,
        kernel: K,
        partition: &crate::ArchitecturePartition<G, A>,
    ) -> Result<Self, ComponentDefinitionError>
    where
        K: ComponentPinnedResources<B>,
    {
        Self::pinned::<B>(definition, kernel)?.select_partition(partition)
    }
}
