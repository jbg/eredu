//! Typed access to pinned modules retained by a component kernel.
use crate::{
    ComponentAdapter, ComponentDefinition, ComponentDefinitionError, EmptyParameterModule,
    StaticParameterVisitor, StaticParameterVisitorMut,
};
use eredu_nn::{NeuralBackend, Parameterized};
use std::{
    borrow::Borrow,
    collections::{BTreeMap, BTreeSet},
    fmt::Debug,
};

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

/// Selects complete pinned resources from family-declared canonical groups.
/// Groups carrying the same resource identity must coexist; independent resources
/// within a role stay independently selectable. Bindings may refine a canonical
/// storage owner, but cannot change its parameter geometry or select a group twice.
/// Groups outside the requested role scope belong to sibling components.
pub fn select_complete_pinned_resources<'a, R: Ord + Debug>(
    component: &str,
    declared: impl IntoIterator<Item = (R, &'a crate::OwnedParameterGroupSpec)>,
    bindings: &[crate::OwnedParameterGroupSpec],
    roles: &[&str],
) -> Result<BTreeSet<R>, ComponentDefinitionError> {
    let declared = declared.into_iter().collect::<Vec<_>>();
    let selected = select_pinned_groups(
        component,
        declared.iter().map(|(_, group)| *group),
        bindings,
        roles,
        crate::ParameterGroupOwner::refines_storage_owner,
    )?;
    complete_pinned_resources(component, declared, &selected)
}

fn select_pinned_groups<'a>(
    component: &str,
    declared: impl Iterator<Item = &'a crate::OwnedParameterGroupSpec> + Clone,
    bindings: &[crate::OwnedParameterGroupSpec],
    roles: &[&str],
    ownership: impl Fn(&crate::ParameterGroupOwner, &crate::ParameterGroupOwner) -> bool,
) -> Result<Vec<&'a crate::OwnedParameterGroupSpec>, ComponentDefinitionError> {
    let invalid = |resource| ComponentDefinitionError::Invalid {
        component: component.into(),
        resource,
    };
    if roles.iter().any(|role| role.trim().is_empty()) {
        return Err(invalid("empty pinned role".into()));
    }
    let mut canonical = BTreeMap::new();
    for group in declared.clone() {
        for member in group.members() {
            if canonical
                .insert(member.target(), group)
                .is_some_and(|previous| previous != group)
            {
                return Err(invalid(format!(
                    "conflicting pinned parameter {}",
                    member.target()
                )));
            }
        }
    }
    let mut selected = BTreeSet::new();
    for binding in bindings.iter().filter(|binding| {
        roles
            .iter()
            .any(|role| binding.owner().accepts_static_role(role))
    }) {
        let index = declared
            .clone()
            .position(|group| {
                group.group() == binding.group() && ownership(group.owner(), binding.owner())
            })
            .ok_or_else(|| {
                invalid(format!(
                    "incompatible pinned group {}",
                    binding.group().logical_name()
                ))
            })?;
        if !selected.insert(index) {
            return Err(invalid(format!(
                "duplicate pinned group {}",
                binding.group().logical_name()
            )));
        }
    }
    Ok(declared
        .enumerate()
        .filter_map(|(index, group)| selected.contains(&index).then_some(group))
        .collect())
}

fn complete_pinned_resources<'a, R: Ord + Debug>(
    component: &str,
    declared: impl IntoIterator<Item = (R, &'a crate::OwnedParameterGroupSpec)>,
    selected: &[&crate::OwnedParameterGroupSpec],
) -> Result<BTreeSet<R>, ComponentDefinitionError> {
    let mut counts = BTreeMap::<R, (usize, usize)>::new();
    for (resource, group) in declared {
        let count = counts.entry(resource).or_default();
        count.0 += usize::from(selected.contains(&group));
        count.1 += 1;
    }
    let mut present = BTreeSet::new();
    for (resource, (selected, total)) in counts {
        if selected != 0 {
            if selected != total {
                return Err(ComponentDefinitionError::Invalid {
                    component: component.into(),
                    resource: format!("incomplete pinned resource {resource:?}"),
                });
            }
            present.insert(resource);
        }
    }
    Ok(present)
}

impl ComponentDefinition {
    /// Resolves a scoped subset of complete canonical pinned groups.
    /// Other roles and execution units belong to sibling components. Independent
    /// resources within one role need not be selected together.
    pub fn select_pinned_groups<'a>(
        &'a self,
        bindings: &[crate::OwnedParameterGroupSpec],
        roles: &[&str],
    ) -> Result<Vec<&'a crate::OwnedParameterGroupSpec>, ComponentDefinitionError> {
        select_pinned_groups(
            self.id(),
            self.parameters().groups().iter(),
            bindings,
            roles,
            |declared, selected| declared == selected,
        )
    }

    /// Resolves optional pinned modules whose canonical groups must coexist as one resource.
    ///
    /// Every supplied group in the requested roles must exactly match this definition.
    /// A role selects either all its groups or none, including shared-storage aliases.
    /// Other roles and execution-unit groups remain available for sibling components.
    pub fn select_complete_pinned_roles(
        &self,
        bindings: &[crate::OwnedParameterGroupSpec],
        roles: &[&str],
    ) -> Result<BTreeSet<String>, ComponentDefinitionError> {
        let selected = self.select_pinned_groups(bindings, roles)?;
        complete_pinned_resources(
            self.id(),
            roles.iter().flat_map(|role| {
                self.parameters()
                    .groups()
                    .iter()
                    .filter(move |binding| binding.owner().accepts_static_role(role))
                    .map(move |binding| ((*role).to_owned(), binding))
            }),
            &selected,
        )
    }
}

// Composite modules borrow the same kernel-owned aggregate used by execution.
impl<T: 'static, D: Borrow<ComponentDefinition>, K: Parameterized<T>> Parameterized<T>
    for ComponentAdapter<D, K, KernelParameterModules>
{
    fn visit_parameters<'a, V: eredu_nn::ParameterVisitor<'a, T>>(&'a self, visitor: &mut V) {
        self.kernel().visit_parameters(visitor);
    }
    fn visit_parameters_mut<'a, V: eredu_nn::ParameterVisitorMut<'a, T>>(
        &'a mut self,
        visitor: &mut V,
    ) {
        self.kernel_mut().visit_parameters_mut(visitor);
    }
    fn set_trainable(&mut self, trainable: bool) {
        self.kernel_mut().set_trainable(trainable);
    }
}
