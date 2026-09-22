//! Statically dispatched layered-architecture lifecycle and resident execution.

#![allow(clippy::too_many_arguments, clippy::type_complexity)]

pub(crate) mod invocation;
mod observation_paths;
pub(crate) use observation_paths::BorrowedHook;
mod ordered_completion;
mod resident_construction;
pub use resident_construction::ordinary_addressed_units;
mod metadata;
use invocation::{LayeredInvocation, OrdinaryLayeredInput};
pub use metadata::LayeredMetadata;
use ordered_completion::BackendLayerwiseCompletion;
pub use ordered_completion::OrderedLayerwiseCompletion;

use observation_paths::ObservationBinding;
pub use observation_paths::{
    BoundCaptureSelection, ObservationBinding as LayeredObservationBinding,
    PrefillObservationDeclaration, PrefillReadoutStage, PreparedCaptureSelection,
    PreparedCaptureSelectionError, PreparedLayeredObservationError,
    PreparedLayeredObservationPaths, PreparedObservationBindingIdentity,
    SharedLayeredObservationPaths,
};

use std::collections::BTreeMap;

use eredu_checkpoint::{recipe::DerivedWeightRecipe, store::CheckpointSource};
use eredu_nn::{NeuralBackend, Parameterized, Tensor};

use crate::{
    observe_and_intervene, ActivationObserver, ExecutionGraph, ExecutionGroupSchedule,
    ExecutionScheduleError, ExecutionUnitLayout, ExpertPass, NoAuxiliaryBoundary,
    ObservedExpertProvider, RoutedExpertProvider, RoutedObservationPoints, RuntimeState,
    StateLayout, SubmissionBackend,
};

/// Statically dispatched visitor over one immutable pinned parameter module.
pub trait StaticParameterVisitor<B: NeuralBackend> {
    /// Failure returned by the consumer.
    type Error;

    /// Visits the module bound to one architecture-declared static role.
    fn visit<M>(&mut self, role: &str, module: &M) -> Result<(), Self::Error>
    where
        M: Parameterized<B::Tensor>;
}

/// Statically dispatched visitor over one mutable pinned parameter module.
pub trait StaticParameterVisitorMut<B: NeuralBackend> {
    /// Failure returned by the consumer.
    type Error;

    /// Visits the mutable module bound to one architecture-declared static role.
    fn visit_mut<M>(&mut self, role: &str, module: &mut M) -> Result<(), Self::Error>
    where
        M: Parameterized<B::Tensor>;
}

/// Architecture-owned enumeration and binding of pinned parameter modules.
///
/// Parameter descriptions select the roles owned by a partition. This
/// contract resolves those roles to concrete neutral modules without making a
/// backend know family fields or checkpoint roots.
pub trait ArchitectureParameters<B: NeuralBackend> {
    /// Architecture-owned failure while deriving geometry or topology.
    type DefinitionError;

    /// Constructs the authoritative geometry in the supplied metadata destination.
    /// `None` selects ordinary construction; checked callers supply their actual producer.
    fn state_layout(
        &self,
        metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<StateLayout, Self::DefinitionError>;

    /// Declares cache identity for the exact rank-local layout and global offset.
    /// Identity text, validation errors and fixed controls share the destination.
    fn state_identity(
        &self,
        state: &crate::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
        metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<crate::ModelStateIdentity, Self::DefinitionError>;

    /// Borrows the retained immutable description or constructs it through the
    /// backend context's metadata destination. Owning consumers copy explicitly.
    fn parameter_description(
        &self,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<std::borrow::Cow<'_, crate::ArchitectureParameterDescription>, Self::DefinitionError>;

    /// Returns architecture-owned checkpoint rewrites for pinned parameters.
    fn static_parameter_recipes(
        &self,
        _source: &dyn CheckpointSource,
    ) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
        Ok(BTreeMap::new())
    }

    /// Visits every available pinned parameter module exactly once.
    fn visit_static_parameters<V>(&self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: StaticParameterVisitor<B>;

    /// Stable retained-value slot ceiling for this exact static topology.
    /// Unknown is not inferred from parameter or current-value traversal. This
    /// diagnostic grants no completion, replacement, source or storage authority.
    fn retained_static_value_slot_bound(&self) -> Option<usize> {
        None
    }

    /// Borrows all currently retained numerical values outside execution units.
    /// Include pinned module parameters, operator helpers and other static
    /// tensors, without evaluating, constructing or mutating them. Return true
    /// only for complete coverage; false may still contribute known values.
    ///
    /// The default traverses declared parameter modules but cannot establish
    /// that the architecture retains no other numerical owners. Architectures
    /// with complete static aggregates should traverse those explicitly.
    fn visit_retained_static_values(&self, visitor: &mut dyn FnMut(&B::Tensor)) -> bool {
        struct Values<'a, T>(&'a mut dyn FnMut(&T));
        impl<B: NeuralBackend> StaticParameterVisitor<B> for Values<'_, B::Tensor> {
            type Error = std::convert::Infallible;
            fn visit<M: Parameterized<B::Tensor>>(
                &mut self,
                _: &str,
                module: &M,
            ) -> Result<(), Self::Error> {
                module.visit_retained_values(self.0);
                Ok(())
            }
        }
        match self.visit_static_parameters(&mut Values(visitor)) {
            Ok(()) => false,
            Err(never) => match never {},
        }
    }

    /// Mutably visits every available pinned parameter module exactly once.
    fn visit_static_parameters_mut<V>(&mut self, visitor: &mut V) -> Result<(), V::Error>
    where
        V: StaticParameterVisitorMut<B>;
}

/// Backend-native activation and architecture-owned forward context.
pub struct LayeredForwardState<T, C> {
    /// Initial activation supplied to the first execution unit.
    pub hidden: T,
    /// Masks, positions, or other architecture-owned forward values.
    pub context: C,
}

/// Architecture-authored semantic kind for one transport-visible execution group.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ArchitectureGroupKind {
    /// Primary text decoding.
    Decoder,
    /// Embedded prediction after the primary decoder output.
    Prediction,
    /// Visual encoding.
    VisionEncoder,
    /// Audio encoding.
    AudioEncoder,
    /// Learned modality projection.
    Projector,
    /// Learned or structural modality merge.
    Merger,
    /// Final multimodal assembly.
    ModalityFinalization,
}

/// Backend-neutral lifecycle for the pipeline ingress phase of a layered graph.
///
/// Architectures declare semantic group kinds and request optionality while
/// concrete backends report only whether optional encoder roots have work. The
/// runtime derives all downstream activity, admits dependency-ready compatible
/// batches, and owns completion transitions. Pipeline backends therefore share
/// the same graph lifecycle instead of reconstructing it around native streams
/// and routes.
#[derive(Debug)]
pub struct LayeredPipelineSchedule<'a> {
    graph: &'a ExecutionGraph,
    schedule: ExecutionGroupSchedule<'a>,
    active: Vec<bool>,
    completed: usize,
}

impl<'a> LayeredPipelineSchedule<'a> {
    /// Creates the pipeline ingress lifecycle from canonical architecture contracts.
    ///
    /// Each contract pairs a group's semantic kind with its declared request
    /// optionality. `request_group_active` is called only for optional encoder
    /// roots. Mandatory encoders, decoder ingress, and finalization always run;
    /// structural merge activity is derived from dependency activity; and
    /// prediction is a later phase.
    pub fn try_new<E>(
        graph: &'a ExecutionGraph,
        group_contracts: impl IntoIterator<Item = (ArchitectureGroupKind, bool)>,
        request_group_active: impl FnMut(usize) -> Result<bool, E>,
    ) -> Result<Self, E>
    where
        E: From<LayeredPipelineScheduleError>,
    {
        let contracts = group_contracts.into_iter().collect::<Vec<_>>();
        let active = Self::activity_for(
            graph,
            &contracts,
            vec![false; contracts.len()],
            request_group_active,
        )?;
        Ok(Self {
            graph,
            schedule: ExecutionGroupSchedule::new(graph),
            active,
            completed: 0,
        })
    }

    pub(crate) fn try_new_with_metadata<E>(
        graph: &'a ExecutionGraph,
        contracts: &[(ArchitectureGroupKind, bool)],
        request_group_active: impl FnMut(usize) -> Result<bool, E>,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<Result<Self, E>, eredu_nn::Error>
    where
        E: From<LayeredPipelineScheduleError>,
    {
        context.charge_metadata(
            std::mem::size_of::<(
                Self,
                E,
                Result<Self, E>,
                Result<Result<Self, E>, eredu_nn::Error>,
                &[(ArchitectureGroupKind, bool)],
            )>()
            .checked_add(std::mem::size_of_val(&request_group_active))
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?,
        )?;
        let mut active = context.metadata_vec(contracts.len())?;
        active.resize(contracts.len(), false);
        let active = match Self::activity_for(graph, contracts, active, request_group_active) {
            Ok(active) => active,
            Err(cause) => return Ok(Err(cause)),
        };
        Ok(Ok(Self {
            graph,
            schedule: ExecutionGroupSchedule::new_with_metadata(graph, context)?,
            active,
            completed: 0,
        }))
    }

    fn activity_for<E>(
        graph: &ExecutionGraph,
        group_contracts: &[(ArchitectureGroupKind, bool)],
        mut active: Vec<bool>,
        mut request_group_active: impl FnMut(usize) -> Result<bool, E>,
    ) -> Result<Vec<bool>, E>
    where
        E: From<LayeredPipelineScheduleError>,
    {
        if group_contracts.len() != graph.groups().len() {
            return Err(LayeredPipelineScheduleError::GroupContractCount {
                graph: graph.groups().len(),
                declared: group_contracts.len(),
            }
            .into());
        }
        for &group in graph.execution_order() {
            let (kind, request_optional) = group_contracts[group];
            if request_optional
                && (!matches!(
                    kind,
                    ArchitectureGroupKind::VisionEncoder | ArchitectureGroupKind::AudioEncoder
                ) || !graph.groups()[group].dependencies().is_empty())
            {
                return Err(LayeredPipelineScheduleError::InvalidRequestOptionalGroup {
                    group,
                    kind,
                }
                .into());
            }
            active[group] = match kind {
                ArchitectureGroupKind::VisionEncoder | ArchitectureGroupKind::AudioEncoder => {
                    !request_optional || request_group_active(group)?
                }
                ArchitectureGroupKind::Projector | ArchitectureGroupKind::Merger => graph
                    .dependencies(group)
                    .expect("validated execution order contains a known group")
                    .iter()
                    .any(|&dependency| active[dependency]),
                ArchitectureGroupKind::ModalityFinalization | ArchitectureGroupKind::Decoder => {
                    true
                }
                ArchitectureGroupKind::Prediction => false,
            };
        }
        Ok(active)
    }

    /// Returns whether a group participates in this pipeline ingress pass.
    pub fn is_active(&self, group: usize) -> Option<bool> {
        self.active.get(group).copied()
    }

    /// Returns all group activity in canonical architecture order.
    pub fn activity(&self) -> &[bool] {
        &self.active
    }

    /// Returns dependency-ready groups in stable architecture order.
    pub fn ready_groups(&self) -> impl Iterator<Item = usize> + '_ {
        self.schedule.startable_groups()
    }

    /// Selects a deterministic maximal compatible subset of ready groups.
    pub fn compatible_batch(&self, mut compatible: impl FnMut(usize, usize) -> bool) -> Vec<usize> {
        let mut selected = Vec::new();
        for candidate in self.ready_groups() {
            if selected
                .iter()
                .copied()
                .all(|group| compatible(group, candidate))
            {
                selected.push(candidate);
            }
        }
        selected
    }

    /// Returns dependency slots in architecture declaration order.
    pub fn dependencies(&self, group: usize) -> Option<&[usize]> {
        self.graph.dependencies(group)
    }

    /// Commits architecture setup for a dependency-ready group.
    ///
    /// The returned producer slots no longer need to retain their outputs for
    /// another consumer after this group has captured its dependencies.
    pub fn started(&mut self, group: usize) -> Result<Vec<usize>, LayeredPipelineScheduleError> {
        self.schedule.started(group).map_err(Into::into)
    }

    pub(crate) fn started_without_release(
        &mut self,
        group: usize,
    ) -> Result<(), LayeredPipelineScheduleError> {
        self.schedule
            .started_with_release(group, |_| {})
            .map_err(Into::into)
    }

    /// Commits one successfully submitted group and unlocks its dependents.
    pub fn ordered(&mut self, group: usize) -> Result<(), LayeredPipelineScheduleError> {
        self.schedule.ordered(group)?;
        self.completed += 1;
        Ok(())
    }

    /// Returns whether every architecture group has completed this phase.
    pub fn is_complete(&self) -> bool {
        self.completed == self.active.len()
    }
}

/// Invalid backend-neutral pipeline lifecycle declaration or transition.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum LayeredPipelineScheduleError {
    /// The physical realization did not preserve one contract per canonical group.
    #[error(
        "execution graph contains {graph} groups but the pipeline declared {declared} group contracts"
    )]
    GroupContractCount {
        /// Number of canonical graph groups.
        graph: usize,
        /// Number of supplied lifecycle contracts.
        declared: usize,
    },
    /// Request optionality was attached to a group which cannot consume request media directly.
    #[error("execution group {group} of kind {kind:?} cannot be request-optional")]
    InvalidRequestOptionalGroup {
        /// Canonical architecture group slot.
        group: usize,
        /// Declared semantic group kind.
        kind: ArchitectureGroupKind,
    },
    /// An ordinary execution-group lifecycle invariant failed.
    #[error(transparent)]
    Transition(#[from] ExecutionScheduleError),
}

/// Pipeline ownership policy for one architecture execution group.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ArchitectureGroupPlacement {
    /// Balance the group across every pipeline owner.
    Pipeline,
    /// Place the complete group on the architecture output owner.
    OutputOwner,
}

/// Architecture-level merge destination resolved by a concrete pipeline topology.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ArchitectureMergeDestination {
    /// Use the group's terminal owner.
    LastOwner,
    /// Return the result to the first pipeline owner for dependency assembly.
    FirstPipelineOwner,
    /// Deliver the result to the architecture output owner.
    OutputOwner,
}

/// Cartesian subgroup semantics required while a group executes.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ArchitectureParallelSubgroup {
    /// Tensor sharding without routed expert exchange.
    TensorSharded,
    /// Decoder tensor and routed-expert parallelism.
    Decoder,
}

/// Backend-neutral transport and placement semantics for one execution group.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ArchitectureGroupTransport {
    /// Physical pipeline ownership policy.
    pub placement: ArchitectureGroupPlacement,
    /// Semantic compute kind.
    pub kind: ArchitectureGroupKind,
    /// Static roles owned by the group's first physical owner.
    pub first_owner_static_roles: Vec<String>,
    /// Static roles owned by the group's terminal physical owner.
    pub last_owner_static_roles: Vec<String>,
    /// Dependency merge destination.
    pub merge_destination: ArchitectureMergeDestination,
    /// Optional active Cartesian subgroup contract.
    pub parallel_subgroup: Option<ArchitectureParallelSubgroup>,
    /// Whether request media may omit this root encoder group entirely.
    pub request_optional: bool,
}

/// Borrowed role names and scalar transport policy from an actual architecture declaration.
#[derive(Clone, Copy)]
pub struct ArchitectureGroupTransportDeclaration<'a> {
    /// Physical ownership of the group.
    pub placement: ArchitectureGroupPlacement,
    /// Semantic compute role.
    pub kind: ArchitectureGroupKind,
    /// Actual first-owner static role names.
    pub first_owner_static_roles: &'a [&'a str],
    /// Actual final-owner static role names.
    pub last_owner_static_roles: &'a [&'a str],
    /// Dependency merge destination.
    pub merge_destination: ArchitectureMergeDestination,
    /// Active Cartesian subgroup.
    pub parallel_subgroup: Option<ArchitectureParallelSubgroup>,
    /// Whether request media may omit this group.
    pub request_optional: bool,
}
impl ArchitectureGroupTransportDeclaration<'_> {
    /// Constructs the ordinary owned policy from these same declarations.
    pub fn into_owned(self) -> ArchitectureGroupTransport {
        ArchitectureGroupTransport {
            placement: self.placement,
            kind: self.kind,
            first_owner_static_roles: self
                .first_owner_static_roles
                .iter()
                .map(|name| (*name).to_owned())
                .collect(),
            last_owner_static_roles: self
                .last_owner_static_roles
                .iter()
                .map(|name| (*name).to_owned())
                .collect(),
            merge_destination: self.merge_destination,
            parallel_subgroup: self.parallel_subgroup,
            request_optional: self.request_optional,
        }
    }
    /// Compares every scalar and ordered role name without constructing owned rows.
    pub fn matches(&self, expected: &ArchitectureGroupTransport) -> bool {
        self.placement == expected.placement
            && self.kind == expected.kind
            && self
                .first_owner_static_roles
                .iter()
                .copied()
                .eq(expected.first_owner_static_roles.iter().map(String::as_str))
            && self
                .last_owner_static_roles
                .iter()
                .copied()
                .eq(expected.last_owner_static_roles.iter().map(String::as_str))
            && self.merge_destination == expected.merge_destination
            && self.parallel_subgroup == expected.parallel_subgroup
            && self.request_optional == expected.request_optional
    }
}

/// Stable layered traversal boundary exposed to generic runtime drivers.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum LayeredTraversalPoint {
    /// Output of one execution unit before the next unit starts.
    Unit {
        /// Execution-group index.
        group: usize,
        /// Group-local unit index.
        index: usize,
    },
    /// Output of one completed execution group.
    Group {
        /// Execution-group index.
        group: usize,
    },
}

/// Decision returned immediately before one execution unit is acquired.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum LayeredUnitAction {
    /// Execute this unit normally.
    Execute,
    /// Omit this unit and every remaining unit in the current group.
    SkipRemainingGroup,
}

/// Statically dispatched hook shared by resident and bounded layered traversal.
///
/// The hook can observe unit/group outputs and can omit only a complete group
/// tail. Drivers are responsible for proving that an omission preserves their
/// semantics before returning [`LayeredUnitAction::SkipRemainingGroup`].
pub trait LayeredTraversalHook<B, C, E>
where
    B: NeuralBackend,
{
    /// Synchronously borrows the complete compact media cut after its owner is
    /// installed and before decoder assembly. This is not completion, capture
    /// permission, source publication, or a reason to reset an allocation ledger.
    fn retained_media_cut(
        &mut self,
        _roots: &mut dyn FnMut(&mut dyn FnMut(&B::Tensor)),
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), E> {
        Ok(())
    }

    /// Whether architecture-owned input, internal unit and readout observation is enabled.
    fn observes_activations(&self) -> bool {
        false
    }

    /// Borrows an activation at an architecture-owned boundary.
    fn observe_activation(&mut self, _path: &str, _value: &B::Tensor) -> Result<(), E> {
        Ok(())
    }

    /// Defers additional diagnostic work to the observing owner's reservation.
    fn observe_generated_activation(
        &mut self,
        path: &str,
        _prototype: &B::Tensor,
        _source: &eredu_core::capture::GeneratedCaptureSource,
        generate: &mut dyn FnMut() -> Result<B::Tensor, E>,
    ) -> Result<(), E> {
        if self.observes_activations() {
            self.observe_activation(path, &generate()?)
        } else {
            Ok(())
        }
    }
    /// Forward the actual generated program and its caller-owned root retention.
    fn observe_generated_activation_retained(
        &mut self,
        path: &str,
        prototype: &B::Tensor,
        source: &eredu_core::capture::GeneratedCaptureSource,
        factory: &mut dyn eredu_nn::RetainedGeneratedTensorFactory<B::Tensor, E>,
    ) -> Result<(), E> {
        self.observe_generated_activation(path, prototype, source, &mut || {
            factory.generate(&mut |_| Ok(()))
        })
    }

    /// Returns an admitted replacement at an architecture-owned boundary.
    fn intervene_activation(
        &mut self,
        _path: &str,
        _value: &B::Tensor,
    ) -> Result<Option<B::Tensor>, E> {
        Ok(None)
    }

    /// Chooses whether to execute the next unit.
    fn before_unit(
        &mut self,
        _group: usize,
        _index: usize,
        _remaining_units: usize,
        _value: &mut B::Tensor,
        _forward: &mut C,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<LayeredUnitAction, E> {
        Ok(LayeredUnitAction::Execute)
    }

    /// Observes and may replace the activation selected for one ready group.
    fn after_group_begin(
        &mut self,
        _group: usize,
        _value: &mut B::Tensor,
        _forward: &mut C,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), E> {
        Ok(())
    }

    /// Observes one executed unit output.
    fn after_unit(
        &mut self,
        _group: usize,
        _index: usize,
        _value: &mut B::Tensor,
        _forward: &mut C,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), E> {
        Ok(())
    }

    /// Observes one completed execution-group output.
    fn after_group(
        &mut self,
        _group: usize,
        _value: &mut B::Tensor,
        _forward: &mut C,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), E> {
        Ok(())
    }
}

/// A borrowed hook delegates to its existing owner without rebuilding paths.
impl<B, C, E, H> LayeredTraversalHook<B, C, E> for &mut H
where
    B: NeuralBackend,
    H: LayeredTraversalHook<B, C, E> + ?Sized,
{
    fn retained_media_cut(
        &mut self,
        roots: &mut dyn FnMut(&mut dyn FnMut(&B::Tensor)),
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), E> {
        (**self).retained_media_cut(roots, context)
    }
    fn observes_activations(&self) -> bool {
        (**self).observes_activations()
    }
    fn observe_activation(&mut self, path: &str, value: &B::Tensor) -> Result<(), E> {
        (**self).observe_activation(path, value)
    }
    fn observe_generated_activation(
        &mut self,
        path: &str,
        prototype: &B::Tensor,
        source: &eredu_core::capture::GeneratedCaptureSource,
        generate: &mut dyn FnMut() -> Result<B::Tensor, E>,
    ) -> Result<(), E> {
        (**self).observe_generated_activation(path, prototype, source, generate)
    }
    fn observe_generated_activation_retained(
        &mut self,
        path: &str,
        prototype: &B::Tensor,
        source: &eredu_core::capture::GeneratedCaptureSource,
        factory: &mut dyn eredu_nn::RetainedGeneratedTensorFactory<B::Tensor, E>,
    ) -> Result<(), E> {
        (**self).observe_generated_activation_retained(path, prototype, source, factory)
    }
    fn intervene_activation(
        &mut self,
        path: &str,
        value: &B::Tensor,
    ) -> Result<Option<B::Tensor>, E> {
        (**self).intervene_activation(path, value)
    }
    fn before_unit(
        &mut self,
        group: usize,
        index: usize,
        remaining_units: usize,
        value: &mut B::Tensor,
        forward: &mut C,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<LayeredUnitAction, E> {
        (**self).before_unit(group, index, remaining_units, value, forward, context)
    }
    fn after_group_begin(
        &mut self,
        group: usize,
        value: &mut B::Tensor,
        forward: &mut C,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), E> {
        (**self).after_group_begin(group, value, forward, context)
    }
    fn after_unit(
        &mut self,
        group: usize,
        index: usize,
        value: &mut B::Tensor,
        forward: &mut C,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), E> {
        (**self).after_unit(group, index, value, forward, context)
    }
    fn after_group(
        &mut self,
        group: usize,
        value: &mut B::Tensor,
        forward: &mut C,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), E> {
        (**self).after_group(group, value, forward, context)
    }
}

/// Statically combines two traversal hooks over one production forward pass.
///
/// Both hooks observe every reached boundary in left-to-right order. A unit is
/// skipped when either hook proves that the remaining group tail can be
/// omitted; errors stop delegation before any later callback is invoked.
pub struct CompositeLayeredTraversalHook<L, R> {
    left: L,
    right: R,
}

impl<L, R> CompositeLayeredTraversalHook<L, R> {
    /// Creates one ordered pair of traversal hooks.
    pub const fn new(left: L, right: R) -> Self {
        Self { left, right }
    }

    /// Returns both hooks after traversal.
    pub fn into_parts(self) -> (L, R) {
        (self.left, self.right)
    }
}

impl<B, C, E, L, R> LayeredTraversalHook<B, C, E> for CompositeLayeredTraversalHook<L, R>
where
    B: NeuralBackend,
    L: LayeredTraversalHook<B, C, E>,
    R: LayeredTraversalHook<B, C, E>,
{
    fn observes_activations(&self) -> bool {
        self.left.observes_activations() || self.right.observes_activations()
    }
    fn retained_media_cut(
        &mut self,
        roots: &mut dyn FnMut(&mut dyn FnMut(&B::Tensor)),
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), E> {
        self.left.retained_media_cut(roots, context)?;
        self.right.retained_media_cut(roots, context)
    }

    fn observe_activation(&mut self, path: &str, value: &B::Tensor) -> Result<(), E> {
        self.left.observe_activation(path, value)?;
        self.right.observe_activation(path, value)
    }
    fn observe_generated_activation(
        &mut self,
        path: &str,
        prototype: &B::Tensor,
        source: &eredu_core::capture::GeneratedCaptureSource,
        generate: &mut dyn FnMut() -> Result<B::Tensor, E>,
    ) -> Result<(), E> {
        self.left
            .observe_generated_activation(path, prototype, source, generate)?;
        self.right
            .observe_generated_activation(path, prototype, source, generate)
    }
    /// Forward the actual generated program and its caller-owned root retention.
    fn observe_generated_activation_retained(
        &mut self,
        path: &str,
        prototype: &B::Tensor,
        source: &eredu_core::capture::GeneratedCaptureSource,
        factory: &mut dyn eredu_nn::RetainedGeneratedTensorFactory<B::Tensor, E>,
    ) -> Result<(), E> {
        self.left
            .observe_generated_activation_retained(path, prototype, source, factory)?;
        self.right
            .observe_generated_activation_retained(path, prototype, source, factory)
    }

    fn intervene_activation(
        &mut self,
        path: &str,
        value: &B::Tensor,
    ) -> Result<Option<B::Tensor>, E> {
        let left = self.left.intervene_activation(path, value)?;
        let right = self
            .right
            .intervene_activation(path, left.as_ref().unwrap_or(value))?;
        Ok(right.or(left))
    }
    fn before_unit(
        &mut self,
        group: usize,
        index: usize,
        remaining_units: usize,
        value: &mut B::Tensor,
        forward: &mut C,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<LayeredUnitAction, E> {
        let left = self
            .left
            .before_unit(group, index, remaining_units, value, forward, context)?;
        let right =
            self.right
                .before_unit(group, index, remaining_units, value, forward, context)?;
        Ok(
            if left == LayeredUnitAction::SkipRemainingGroup
                || right == LayeredUnitAction::SkipRemainingGroup
            {
                LayeredUnitAction::SkipRemainingGroup
            } else {
                LayeredUnitAction::Execute
            },
        )
    }

    fn after_unit(
        &mut self,
        group: usize,
        index: usize,
        value: &mut B::Tensor,
        forward: &mut C,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), E> {
        self.left
            .after_unit(group, index, value, forward, context)?;
        self.right.after_unit(group, index, value, forward, context)
    }

    fn after_group_begin(
        &mut self,
        group: usize,
        value: &mut B::Tensor,
        forward: &mut C,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), E> {
        self.left
            .after_group_begin(group, value, forward, context)?;
        self.right.after_group_begin(group, value, forward, context)
    }

    fn after_group(
        &mut self,
        group: usize,
        value: &mut B::Tensor,
        forward: &mut C,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), E> {
        self.left.after_group(group, value, forward, context)?;
        self.right.after_group(group, value, forward, context)
    }
}

/// Observes one traversal-owned boundary, applies its intervention, then
/// observes the effective value without a second intervention opportunity.
/// Both names are borrowed so prepared traversals can reuse their retained
/// paths. Architecture-owned boundaries must not also call this helper.
pub fn observe_outer_boundary<T: Clone, E, O: ActivationObserver<T, E> + ?Sized>(
    observer: &mut O,
    path: &str,
    effective_path: &str,
    value: &T,
) -> Result<T, E> {
    let effective = observe_and_intervene(observer, path, value)?;
    observer.observe(effective_path, &effective)?;
    Ok(effective)
}

struct NoopLayeredTraversalHook;

pub(crate) struct TraversalActivationObserver<'a, H: ?Sized, B, C, E> {
    pub(crate) hook: &'a mut H,
    pub(crate) types: std::marker::PhantomData<fn() -> (B, C, E)>,
}
impl<B, C, E, H> ActivationObserver<B::Tensor, E> for TraversalActivationObserver<'_, H, B, C, E>
where
    B: NeuralBackend,
    H: LayeredTraversalHook<B, C, E> + ?Sized,
{
    fn observe(&mut self, path: &str, value: &B::Tensor) -> Result<(), E> {
        self.hook.observe_activation(path, value)
    }
    fn observe_generated(
        &mut self,
        path: &str,
        prototype: &B::Tensor,
        source: &eredu_core::capture::GeneratedCaptureSource,
        generate: &mut dyn FnMut() -> Result<B::Tensor, E>,
    ) -> Result<(), E> {
        self.hook
            .observe_generated_activation(path, prototype, source, generate)
    }
    /// Forward the actual generated program and its caller-owned root retention.
    fn observe_generated_retained(
        &mut self,
        path: &str,
        prototype: &B::Tensor,
        source: &eredu_core::capture::GeneratedCaptureSource,
        factory: &mut dyn eredu_nn::RetainedGeneratedTensorFactory<B::Tensor, E>,
    ) -> Result<(), E> {
        self.hook
            .observe_generated_activation_retained(path, prototype, source, factory)
    }

    fn intervene(&mut self, path: &str, value: &B::Tensor) -> Result<Option<B::Tensor>, E> {
        self.hook.intervene_activation(path, value)
    }
}

impl<B, C, E> LayeredTraversalHook<B, C, E> for NoopLayeredTraversalHook where B: NeuralBackend {}

struct ActivationObserverTraversalHook<'a, O: ?Sized> {
    observer: std::rc::Rc<std::cell::RefCell<&'a mut O>>,
    units: Vec<Vec<Option<String>>>,
    group_inputs: Vec<Option<String>>,
    group_outputs: Vec<Option<String>>,
}

impl<B, C, E, O> LayeredTraversalHook<B, C, E> for ActivationObserverTraversalHook<'_, O>
where
    B: NeuralBackend,
    O: ActivationObserver<B::Tensor, E> + ?Sized,
{
    fn observes_activations(&self) -> bool {
        true
    }
    fn observe_activation(&mut self, path: &str, value: &B::Tensor) -> Result<(), E> {
        self.observer.borrow_mut().observe(path, value)
    }
    fn observe_generated_activation(
        &mut self,
        path: &str,
        prototype: &B::Tensor,
        source: &eredu_core::capture::GeneratedCaptureSource,
        generate: &mut dyn FnMut() -> Result<B::Tensor, E>,
    ) -> Result<(), E> {
        self.observer
            .borrow_mut()
            .observe_generated(path, prototype, source, generate)
    }
    /// Forward the actual generated program and its caller-owned root retention.
    fn observe_generated_activation_retained(
        &mut self,
        path: &str,
        prototype: &B::Tensor,
        source: &eredu_core::capture::GeneratedCaptureSource,
        factory: &mut dyn eredu_nn::RetainedGeneratedTensorFactory<B::Tensor, E>,
    ) -> Result<(), E> {
        self.observer
            .borrow_mut()
            .observe_generated_retained(path, prototype, source, factory)
    }

    fn intervene_activation(
        &mut self,
        path: &str,
        value: &B::Tensor,
    ) -> Result<Option<B::Tensor>, E> {
        self.observer.borrow_mut().intervene(path, value)
    }
    fn before_unit(
        &mut self,
        group: usize,
        index: usize,
        _remaining_units: usize,
        value: &mut B::Tensor,
        _forward: &mut C,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<LayeredUnitAction, E> {
        let Some(unit) = &self.units[group][index] else {
            return Ok(LayeredUnitAction::Execute);
        };
        let path = eredu_core::UnitObservation::Input.path(unit);
        let mut observer = self.observer.borrow_mut();
        *value =
            observe_outer_boundary(&mut **observer, &path, &format!("{path}.effective"), value)?;
        Ok(LayeredUnitAction::Execute)
    }

    fn after_unit(
        &mut self,
        group: usize,
        index: usize,
        value: &mut B::Tensor,
        _forward: &mut C,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), E> {
        let Some(unit) = &self.units[group][index] else {
            return Ok(());
        };
        let path = eredu_core::UnitObservation::Output.path(unit);
        let mut observer = self.observer.borrow_mut();
        *value =
            observe_outer_boundary(&mut **observer, &path, &format!("{path}.effective"), value)?;
        Ok(())
    }

    fn after_group_begin(
        &mut self,
        group: usize,
        value: &mut B::Tensor,
        _forward: &mut C,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), E> {
        if let Some(path) = self.group_inputs.get(group).and_then(Option::as_deref) {
            let mut observer = self.observer.borrow_mut();
            *value = observe_and_intervene(&mut **observer, path, value)?;
        }
        Ok(())
    }

    fn after_group(
        &mut self,
        group: usize,
        value: &mut B::Tensor,
        _forward: &mut C,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), E> {
        if let Some(path) = self.group_outputs.get(group).and_then(Option::as_deref) {
            let mut observer = self.observer.borrow_mut();
            *value = observe_and_intervene(&mut **observer, path, value)?;
        }
        Ok(())
    }
}

struct AfterUnitTraversalHook<F> {
    after_unit: F,
}

struct AfterUnitContextTraversalHook<F> {
    after_unit: F,
}

impl<B, C, E, F> LayeredTraversalHook<B, C, E> for AfterUnitTraversalHook<F>
where
    B: NeuralBackend,
    F: FnMut(usize, usize, &B::Tensor, &mut C) -> Result<(), E>,
{
    fn after_unit(
        &mut self,
        group: usize,
        index: usize,
        value: &mut B::Tensor,
        forward: &mut C,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), E> {
        (self.after_unit)(group, index, value, forward)
    }
}

impl<B, C, E, F> LayeredTraversalHook<B, C, E> for AfterUnitContextTraversalHook<F>
where
    B: NeuralBackend,
    F: FnMut(usize, usize, &mut C) -> Result<(), E>,
{
    fn after_unit(
        &mut self,
        group: usize,
        index: usize,
        _value: &mut B::Tensor,
        forward: &mut C,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), E> {
        (self.after_unit)(group, index, forward)
    }
}

/// Backend-neutral lifecycle implemented once by a layered architecture.
///
/// All hot values remain concrete associated types. Resident and bounded
/// runtime policies call these same methods without erasing tensors, units, or
/// mutable layer state.
pub trait LayeredArchitecture<B, S>:
    ArchitectureParameters<B, DefinitionError = Self::Error>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
{
    /// Internal hooks emitted by the ordinary begin/unit/finish observed methods.
    /// Overriding an execution method does not implicitly declare hook coverage.
    fn observation_hooks(&self) -> crate::inspection::ObservationHookSupport {
        Default::default()
    }

    /// Non-serialized ordinary-text row semantics emitted by this actual
    /// architecture beside its hooks. This is a semantic declaration, not native
    /// support, capture admission or execution authority. Empty means unfinished
    /// row-assembly coverage, not architectural inapplicability.
    ///
    /// Declarations apply only to canonical ordinary text with its retained
    /// causal prefix, no supplied attention mask or activation interventions.
    /// Construction/rebinding occurs under loading/preparation authority.
    fn prefill_observation_declarations(
        &self,
        _metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<Vec<PrefillObservationDeclaration>, Self::Error> {
        Ok(Vec::new())
    }

    /// Decoder-row equivalence for this architecture's validated retained-media
    /// ingress. This never declares encoder axes or intervention equivalence.
    /// The session must authenticate the actual source/ingress plan before use.
    fn media_prefill_observation_declarations(
        &self,
        _metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<Vec<PrefillObservationDeclaration>, Self::Error> {
        Ok(Vec::new())
    }

    /// Borrowed prepared model input.
    type Input<'a>
    where
        Self: 'a;

    /// Inspects the decoder dimensions without native allocation or execution.
    /// `None` identifies an auxiliary invocation whose equations are not covered
    /// by an ordinary prompt/decode workspace quote (for example an embedded
    /// proposal pass). It must be separately admitted before budgeted execution.
    fn inference_input_shape(input: &Self::Input<'_>) -> Result<Option<[u64; 2]>, Self::Error>;
    /// Pinned model modules such as embeddings, final normalization, and head.
    type StaticModules: Parameterized<B::Tensor>;
    /// One ordered execution unit.
    type Unit: Parameterized<B::Tensor>;
    /// Architecture-owned state retained for one complete forward pass.
    type ForwardContext;
    /// Allocation-free iterator over transient tensors retained by a unit submission.
    type RetainedContextValues<'a>: Iterator<Item = &'a B::Tensor>
    where
        Self: 'a,
        B::Tensor: 'a;
    /// Concrete architecture or backend failure.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Declares transport and physical placement semantics for one canonical group slot.
    fn group_transport(&self, group: usize) -> ArchitectureGroupTransport;

    /// Compares actual transport policy. Overrides may borrow their original role declarations.
    fn group_transport_matches(&self, group: usize, expected: &ArchitectureGroupTransport) -> bool {
        self.group_transport(group) == *expected
    }

    /// Returns the stable identifier of the primary pipeline execution group.
    ///
    /// Pipeline composition resolves this identifier against [`Self::execution_graph`]
    /// instead of guessing the primary group from its semantic kind. Architectures may
    /// therefore declare multiple decoder-shaped groups without making composition
    /// dependent on declaration order.
    fn primary_execution_group(&self) -> &str;

    /// Returns stable identifiers for ordered embedded-prediction groups.
    ///
    /// The order is the architecture's prediction-depth order. Semantic group kinds
    /// remain lifecycle metadata and are not used as group addresses.
    fn prediction_execution_groups(&self) -> Vec<String> {
        Vec::new()
    }

    /// Requests the existing target-capture retention before equation execution.
    /// Architectures that already retain it need no setup. This is not a support
    /// claim: consumers must still require an actual `prediction_target_capture`.
    fn retain_prediction_target_capture(&mut self) {}

    /// Borrows the exact ordinary-target value consumed by an additive prediction extension.
    fn prediction_target_capture(_context: &Self::ForwardContext) -> Option<&B::Tensor> {
        None
    }

    /// Declares the exact capture placeholder submitted by a non-output pipeline rank.
    fn prediction_target_placeholder_shape(
        &self,
        _forward: &Self::ForwardContext,
    ) -> Result<Option<Vec<i32>>, Self::Error> {
        Ok(None)
    }

    /// Declares how the complete mutable-state layout is divided among realized partitions.
    fn state_partition_plan(&self, layout: &StateLayout) -> crate::ArchitectureStatePartitionPlan;

    /// Loans the validated source declaration between ordered execution groups.
    /// Owning consumers explicitly materialize it in their metadata destination.
    fn execution_graph(&self) -> Result<crate::ArchitectureExecutionGraph<'_>, Self::Error>;

    /// Returns the number of ordered execution units in one graph group.
    /// The optional destination funds all produced diagnostics and controls.
    fn group_unit_count(
        &self,
        group: usize,
        metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<usize, Self::Error>;

    /// Returns the stable architecture-owned path of one group-local execution unit.
    fn unit_path(
        &self,
        group: usize,
        index: usize,
        metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<String, Self::Error>;

    /// Whether observed unit execution owns both input/output capture and
    /// intervention, including their effective companions. The traversal omits
    /// its outer copies of those exact seams. This includes observed provider
    /// and parallel entry points and preserves in-unit state/capture timing.
    fn observes_unit_boundaries(&self, _group: usize, _index: usize) -> bool {
        false
    }

    /// Architecture-owned ingress name in the selected metadata destination.
    fn group_input_observation_path(
        &self,
        _group: usize,
        _metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<Option<String>, Self::Error> {
        Ok(None)
    }

    /// Architecture-owned completion name in the selected metadata destination.
    fn group_output_observation_path(
        &self,
        _group: usize,
        _metadata_context: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<Option<String>, Self::Error> {
        Ok(None)
    }

    /// Borrows pinned modules for parameter discovery and binding.
    fn static_modules(&self) -> &Self::StaticModules;

    /// Mutably borrows pinned modules for parameter binding.
    fn static_modules_mut(&mut self) -> &mut Self::StaticModules;

    /// Builds one unloaded execution unit using backend-native operators.
    fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Self::Unit, Self::Error>;

    /// Embeds input and prepares architecture-owned forward values.
    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>;

    /// Prepares the same forward with architecture-owned input observations.
    fn begin_forward_observed<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        _observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        O: ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        self.begin_forward(input, state, context)
    }

    /// Selects or merges the activation consumed by one ready execution group.
    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;

    /// Returns whether one ready group is needed for this forward pass.
    fn should_execute_group(&self, _group: usize, _forward: &Self::ForwardContext) -> bool {
        true
    }

    /// Maps one execution unit to its architecture-global mutable-state slot.
    ///
    /// The default matches a single flattened decoder schedule. Composite
    /// architectures can keep parameter-only groups outside their state layout
    /// and remap later groups onto their semantic decoder or predictor layers.
    fn state_ordinal(&self, _group: usize, _index: usize, ordinal: usize) -> usize {
        ordinal
    }

    /// Returns every architecture-global state slot retained by one unit.
    ///
    /// The default retains the single slot returned by [`Self::state_ordinal`].
    /// Composite units can return a contiguous range when one residency unit
    /// internally executes several stateful layers.
    fn retained_state_ordinals(
        &self,
        group: usize,
        index: usize,
        ordinal: usize,
    ) -> std::ops::Range<usize> {
        let state = self.state_ordinal(group, index, ordinal);
        state..state + 1
    }

    /// Executes one ordered unit against its concrete mutable layer state.
    fn forward_unit(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;

    /// Executes the same unit with architecture-owned internal observation hooks.
    #[allow(clippy::too_many_arguments)]
    fn forward_unit_observed<O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        _observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        self.forward_unit(group, index, unit, hidden, state, forward, context)
    }

    /// Converts a completed group's output into its dependency-facing value.
    fn complete_execution_group(
        &mut self,
        _group: usize,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &mut Self::ForwardContext,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        Ok(hidden.clone())
    }

    /// Supplies final public demand before any group or unit projection.
    /// Architectures with earlier decision readouts merge their other consumers
    /// with this demand while preserving complete mutable state.
    fn set_readout_demand(
        &self,
        _forward: &mut Self::ForwardContext,
        _demand: eredu_core::OutputDemand,
    ) {
    }

    /// Selects hidden positions before final normalization and vocabulary
    /// projection. Family geometry owns the sequence axis; mutable state and
    /// prediction captures in `forward` retain their complete sequence.
    /// Return `None` exactly for state-only execution.
    fn select_readout_positions(
        &self,
        hidden: &B::Tensor,
        forward: &Self::ForwardContext,
        demand: eredu_core::OutputDemand,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, Self::Error>;

    /// Applies final normalization and output projection.
    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;

    /// Applies the same final readout with architecture-owned internal hooks.
    fn finish_forward_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        _observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        self.finish_forward(hidden, state, forward, context)
    }

    /// Existing forward-owned metadata destination, when its constructors participate.
    /// The default reports no destination and preserves ordinary allocation behavior.
    fn forward_metadata(
        &self,
        _forward: &Self::ForwardContext,
    ) -> Option<LayeredMetadata<Self::Error>> {
        None
    }

    /// Visits the same unit retention in order. The default preserves the
    /// existing iterator; implementations with paid forward metadata can avoid
    /// an intermediate owning container by lending their actual fields.
    fn visit_retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        group: usize,
        index: usize,
        visitor: &mut dyn FnMut(&'a B::Tensor),
    ) {
        for value in self.retained_context_values(forward, group, index) {
            visitor(value);
        }
    }

    /// Borrows transient forward tensors required by one unit's submission.
    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        group: usize,
        index: usize,
    ) -> Self::RetainedContextValues<'a>;
}

/// Optional statically dispatched parallel lifecycle for a layered architecture.
///
/// The runtime owns traversal and exact unit completion while the architecture
/// owns parallel embedding, block, and output semantics. Backend-native
/// collective contexts cross this boundary unchanged.
pub trait ParallelLayeredArchitecture<B, S>: LayeredArchitecture<B, S>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
{
    /// Internal hooks emitted by the parallel begin/unit/finish observed methods.
    fn parallel_observation_hooks(&self) -> crate::inspection::ObservationHookSupport {
        Default::default()
    }

    /// Embeds input and prepares forward values for rank-local execution.
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>;

    /// Prepares parallel input with architecture-owned internal observations.
    fn begin_forward_parallel_observed<'a, O>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        _observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        O: ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        self.begin_forward_parallel(input, state, parallel, context)
    }

    /// Executes one rank-local unit and its required collectives.
    fn forward_unit_parallel(
        &mut self,
        group_index: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;

    /// Executes rank-local internal boundaries when supplied by the architecture.
    /// The default preserves ordinary parallel execution.
    #[allow(clippy::too_many_arguments)]
    fn forward_unit_parallel_observed<O>(
        &mut self,
        group_index: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        _observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        self.forward_unit_parallel(
            group_index,
            index,
            unit,
            hidden,
            state,
            forward,
            parallel,
            context,
        )
    }

    /// Selects or merges a ready group's activation under a parallel context.
    fn begin_execution_group_parallel(
        &mut self,
        group_index: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        state: &mut S,
        forward: &mut Self::ForwardContext,
        _parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.begin_execution_group(group_index, initial, dependencies, state, forward, context)
    }

    /// Converts a completed group's output under a parallel context.
    fn complete_execution_group_parallel(
        &mut self,
        group_index: usize,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        _parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.complete_execution_group(group_index, hidden, state, forward, context)
    }

    /// Applies the rank-local output projection and returns complete logits.
    fn finish_forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;

    /// Finishes parallel readout with architecture-owned internal observations.
    fn finish_forward_parallel_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        _observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        O: ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        self.finish_forward_parallel(hidden, state, forward, parallel, context)
    }
}

/// Input accepted at a rank-local layered partition boundary.
///
/// A partition either embeds borrowed token ids or consumes an owned hidden
/// tensor prepared by architecture ingress or a preceding pipeline owner.
#[derive(Debug)]
pub enum LayeredPartitionInput<'a, T, A = NoAuxiliaryBoundary> {
    /// Token ids supplied to the architecture input owner.
    Tokens(&'a T),
    /// Architecture-prepared or upstream hidden state.
    Hidden {
        /// Evolving activation received from the preceding owner.
        hidden: T,
        /// Architecture-typed context carried across the partition boundary.
        auxiliary: A,
    },
}

/// Architecture-owned result of completing one layered partition.
///
/// The runtime and concrete transports only distinguish a final output from a
/// transport boundary. Families retain ownership of auxiliary boundary values
/// and of any hidden activation required by an embedded predictor.
pub enum LayeredPartitionOutput<T, A = NoAuxiliaryBoundary> {
    /// State-only execution retains completion dependencies without scores.
    StateOnly {
        /// Full final hidden value, including any prediction dependency.
        retained: T,
    },
    /// Complete architecture output produced by the output owner.
    Final {
        /// Projected architecture output, normally vocabulary logits.
        output: T,
        /// Optional pre-projection value consumed by an embedded predictor.
        retained: Option<T>,
    },
    /// Values transported to the next pipeline owner.
    Boundary {
        /// Evolving activation.
        hidden: T,
        /// Architecture-typed auxiliary context.
        auxiliary: A,
    },
}

/// Architecture-owned preparation for rank-local partition execution.
///
/// The neutral partition driver owns validation, group sequencing, and output
/// ownership. Architectures own the semantic conversion of partition inputs,
/// entry into and completion of their selected execution group, and typed
/// partition output.
pub trait PartitionedLayeredArchitecture<B, S>: ParallelLayeredArchitecture<B, S>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
{
    /// Internal hooks emitted by partition begin/finish and ordinary or parallel
    /// unit calls. This describes the global call path, independent of local ownership.
    fn partition_observation_hooks(
        &self,
        _tensor_parallel: bool,
    ) -> crate::inspection::ObservationHookSupport {
        Default::default()
    }

    /// Architecture-owned schema for primary and auxiliary partition transport.
    type Boundary: crate::ArchitectureBoundary;

    /// Derives the complete transport schema in the supplied metadata destination.
    /// Checked callers supply their actual producer; refusal never changes destination.
    fn boundary_schema(
        &self,
        metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<Self::Boundary, Self::Error>;

    /// Prepares a replicated partition from tokens or upstream hidden state.
    fn begin_partition<'a>(
        &mut self,
        input: LayeredPartitionInput<
            'a,
            B::Tensor,
            <Self::Boundary as crate::ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &crate::StateLayout,
        first_state_ordinal: usize,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>;

    /// Prepares the tensor-parallel form of the same partition.
    #[allow(clippy::too_many_arguments)]
    fn begin_partition_parallel<'a>(
        &mut self,
        input: LayeredPartitionInput<
            'a,
            B::Tensor,
            <Self::Boundary as crate::ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &crate::StateLayout,
        first_state_ordinal: usize,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>;

    /// Prepares the same partition with architecture-owned internal observations.
    #[allow(clippy::too_many_arguments)]
    fn begin_partition_observed<'a, O>(
        &mut self,
        input: LayeredPartitionInput<
            'a,
            B::Tensor,
            <Self::Boundary as crate::ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
        mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &crate::StateLayout,
        first_state_ordinal: usize,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        _observer: &mut O,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error>
    where
        O: ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        match parallel {
            Some(parallel) => self.begin_partition_parallel(
                input,
                mask,
                state,
                expected,
                first_state_ordinal,
                parallel,
                context,
            ),
            None => {
                self.begin_partition(input, mask, state, expected, first_state_ordinal, context)
            }
        }
    }

    /// Enters the selected execution group after the partition input has been
    /// prepared. The default is the ordinary layered group entry with no graph
    /// dependencies; graph architectures may override this when their
    /// partition input is already assembled.
    #[allow(clippy::too_many_arguments)]
    fn enter_partition_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        match parallel {
            Some(parallel) => self.begin_execution_group_parallel(
                group,
                initial,
                &[],
                state,
                forward,
                parallel,
                context,
            ),
            None => self.begin_execution_group(group, initial, &[], state, forward, context),
        }
    }

    /// Completes the selected execution group before the architecture emits
    /// its final value or typed pipeline boundary.
    #[allow(clippy::too_many_arguments)]
    fn leave_partition_group(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        match parallel {
            Some(parallel) => self.complete_execution_group_parallel(
                group, hidden, state, forward, parallel, context,
            ),
            None => self.complete_execution_group(group, hidden, state, forward, context),
        }
    }

    /// Emits an architecture partition after its execution group has closed.
    #[allow(clippy::too_many_arguments)]
    fn finish_partition(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<
        LayeredPartitionOutput<
            B::Tensor,
            <Self::Boundary as crate::ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
        Self::Error,
    >;

    /// Emits the selected boundary or observed final readout under the same ownership.
    #[allow(clippy::too_many_arguments)]
    fn finish_partition_observed<O>(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        _observer: &mut O,
    ) -> Result<
        LayeredPartitionOutput<
            B::Tensor,
            <Self::Boundary as crate::ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
        Self::Error,
    >
    where
        O: ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        self.finish_partition(hidden, state, forward, owns_output, parallel, context)
    }

    /// Restores a retained prediction value from full hidden positions after
    /// selective readout. Families retaining transformed or concatenated state
    /// override this equation; the default retains the ordinary hidden value.
    fn partition_prediction_capture(
        &self,
        hidden: &B::Tensor,
        _forward: &Self::ForwardContext,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        Ok(hidden.clone())
    }

    /// Selects readout positions only on the output owner, preserving transport
    /// and prediction values at their full sequence geometry.
    fn finish_partition_with_readout<O>(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        owns_output: bool,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        demand: eredu_core::OutputDemand,
    ) -> Result<
        LayeredPartitionOutput<
            B::Tensor,
            <Self::Boundary as crate::ArchitectureBoundary>::Boundary<B::Tensor>,
        >,
        Self::Error,
    >
    where
        O: ActivationObserver<B::Tensor, Self::Error> + ?Sized,
    {
        if !owns_output {
            return self.finish_partition_observed(
                hidden, state, forward, false, parallel, context, observer,
            );
        }
        let Some(selected) = self.select_readout_positions(hidden, forward, demand, context)?
        else {
            return Ok(LayeredPartitionOutput::StateOnly {
                retained: hidden.clone(),
            });
        };
        let mut result = self.finish_partition_observed(
            &selected, state, forward, true, parallel, context, observer,
        )?;
        if let LayeredPartitionOutput::Final {
            retained: Some(retained),
            ..
        } = &mut result
        {
            if demand != eredu_core::OutputDemand::Sequence {
                *retained = self.partition_prediction_capture(hidden, forward, context)?;
            }
        }
        Ok(result)
    }
}

/// Provider-aware unit execution for architectures with routed feed-forward work.
///
/// Partition drivers retain ownership of expert residency while the neutral
/// architecture retains attention, residual, routing, and unit semantics.
pub trait RoutedLayeredArchitecture<B, S>: LayeredArchitecture<B, S>
where
    B: eredu_nn::GroupedNeuralBackend,
    S: RuntimeState<B>,
{
    /// Classifies routed storage access from architecture-visible unit input.
    ///
    /// Architectures with a nonstandard pass contract may override this
    /// method; concrete backends must not derive the semantic pass.
    fn expert_pass_for_unit(
        &self,
        _group: usize,
        _index: usize,
        hidden: &B::Tensor,
        _forward: &Self::ForwardContext,
    ) -> ExpertPass {
        let sequence = hidden
            .shape()
            .get(hidden.shape().len().saturating_sub(2))
            .copied()
            .unwrap_or(1);
        if sequence > 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        }
    }

    /// Returns the architecture-owned routing observation point for one unit.
    ///
    /// Architectures without observable routed work in the selected unit return
    /// `None`. Concrete backends must not reconstruct semantic paths or expert
    /// cardinality.
    fn routed_observation_points(
        &self,
        _group: usize,
        _index: usize,
    ) -> Result<Option<RoutedObservationPoints>, Self::Error> {
        Ok(None)
    }

    /// Whether the ordinary provider-aware unit call emits internal component hooks.
    fn routed_unit_observations(&self) -> bool {
        false
    }

    /// Whether the selected provider-aware call emits sparse routed-unit hooks.
    /// This does not imply coverage of other internal component boundaries.
    fn routed_sparse_observations(&self) -> bool {
        false
    }

    /// Executes one unit through a runtime-supplied routed-expert provider.
    #[allow(clippy::too_many_arguments)]
    fn forward_unit_with_provider<P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display;

    /// Executes one unit with architecture-owned pass classification.
    #[allow(clippy::too_many_arguments)]
    fn forward_unit_with_inferred_provider<P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        provider: &mut P,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let pass = self.expert_pass_for_unit(group, index, hidden, forward);
        self.forward_unit_with_provider(
            group, index, unit, hidden, state, forward, pass, provider, context,
        )
    }

    /// Executes one provider-backed unit while exposing its architecture-owned
    /// routed/shared/combined observation stages.
    #[allow(clippy::too_many_arguments)]
    fn forward_unit_observed_with_provider<P, O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: ActivationObserver<B::Tensor, Self::Error> + ?Sized,
        Self::Error: std::fmt::Display,
    {
        match self.routed_observation_points(group, index)? {
            Some(point) => {
                let mut observed = ObservedExpertProvider::new(provider, observer, point);
                self.forward_unit_with_provider(
                    group,
                    index,
                    unit,
                    hidden,
                    state,
                    forward,
                    pass,
                    &mut observed,
                    context,
                )
            }
            None => self.forward_unit_with_provider(
                group, index, unit, hidden, state, forward, pass, provider, context,
            ),
        }
    }

    /// Executes one observed provider-backed unit with architecture-owned pass
    /// classification.
    #[allow(clippy::too_many_arguments)]
    fn forward_unit_observed_with_inferred_provider<P, O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        provider: &mut P,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: ActivationObserver<B::Tensor, Self::Error> + ?Sized,
        Self::Error: std::fmt::Display,
    {
        let pass = self.expert_pass_for_unit(group, index, hidden, forward);
        self.forward_unit_observed_with_provider(
            group, index, unit, hidden, state, forward, pass, provider, context, observer,
        )
    }
}

/// Tensor-parallel provider-aware unit execution.
pub trait ParallelRoutedLayeredArchitecture<B, S>:
    RoutedLayeredArchitecture<B, S> + ParallelLayeredArchitecture<B, S>
where
    B: eredu_nn::GroupedNeuralBackend,
    S: RuntimeState<B>,
{
    /// Whether the tensor-parallel provider-aware unit call emits internal component hooks.
    fn parallel_routed_unit_observations(&self) -> bool {
        false
    }

    /// Sparse routed-unit coverage of the tensor-parallel provider-aware call.
    fn parallel_routed_sparse_observations(&self) -> bool {
        false
    }

    /// Executes one tensor-parallel unit through a runtime-supplied provider.
    #[allow(clippy::too_many_arguments)]
    fn forward_unit_parallel_with_provider<P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: crate::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display;

    /// Preserves provider routing observations while allowing genuine component hooks.
    #[allow(clippy::too_many_arguments)]
    fn forward_unit_parallel_observed_with_provider<P, O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: crate::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: ActivationObserver<B::Tensor, Self::Error> + ?Sized,
        Self::Error: std::fmt::Display,
    {
        match self.routed_observation_points(group, index)? {
            Some(point) => {
                let mut observed = ObservedExpertProvider::new(provider, observer, point);
                self.forward_unit_parallel_with_provider(
                    group,
                    index,
                    unit,
                    hidden,
                    state,
                    forward,
                    pass,
                    &mut observed,
                    parallel,
                    context,
                )
            }
            None => self.forward_unit_parallel_with_provider(
                group, index, unit, hidden, state, forward, pass, provider, parallel, context,
            ),
        }
    }

    /// Selects ordinary/observed provider execution without changing the provider or parallel context.
    #[allow(clippy::too_many_arguments)]
    fn forward_unit_with_provider_observation<P, O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: ExpertPass,
        provider: &mut P,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: Option<&mut O>,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: crate::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: ActivationObserver<B::Tensor, Self::Error> + ?Sized,
        Self::Error: std::fmt::Display,
    {
        match (parallel, observer) {
            (Some(parallel), Some(observer)) => self.forward_unit_parallel_observed_with_provider(
                group, index, unit, hidden, state, forward, pass, provider, parallel, context,
                observer,
            ),
            (Some(parallel), None) => self.forward_unit_parallel_with_provider(
                group, index, unit, hidden, state, forward, pass, provider, parallel, context,
            ),
            (None, Some(observer)) => self.forward_unit_observed_with_provider(
                group, index, unit, hidden, state, forward, pass, provider, context, observer,
            ),
            (None, None) => self.forward_unit_with_provider(
                group, index, unit, hidden, state, forward, pass, provider, context,
            ),
        }
    }
}

/// Fully resident runtime using the same lifecycle as bounded execution.
pub struct ResidentRuntime<A, B, S>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
{
    architecture: A,
    graph: resident_construction::ResidentGraph,
    units: Vec<Vec<A::Unit>>,
    observation_binding: ObservationBinding,
    backend: std::marker::PhantomData<fn() -> (B, S)>,
}

impl<A, B, S> ResidentRuntime<A, B, S>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
{
    /// Builds every execution unit once and keeps it resident.
    pub fn new(
        architecture: A,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Self, A::Error> {
        resident_construction::ordinary(architecture, context)
    }

    /// Runs one complete prefill or decode pass without dynamic dispatch.
    pub fn forward<'a>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, A::Error> {
        self.forward_with_context(input, state, context)
            .map(|(output, _)| output)
    }

    /// Runs one complete pass and returns its architecture-owned context.
    ///
    /// This is the resident counterpart of the bounded runtime's context
    /// result and lets callers retain target captures without storing
    /// request-local tensors on the model object.
    pub fn forward_with_context<'a>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(B::Tensor, A::ForwardContext), A::Error> {
        self.forward_with_traversal_hook(input, state, context, &mut NoopLayeredTraversalHook)
    }

    /// Runs one resident pass through a statically dispatched traversal hook.
    pub fn forward_with_traversal_hook<'a, H>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        hook: &mut H,
    ) -> Result<(B::Tensor, A::ForwardContext), A::Error>
    where
        H: LayeredTraversalHook<B, A::ForwardContext, A::Error> + ?Sized,
    {
        self.forward_with_traversal_hook_with_readout(
            input,
            state,
            context,
            hook,
            eredu_core::OutputDemand::Sequence,
        )
        .map(|(output, forward)| (output.expect("sequence readout returns scores"), forward))
    }

    /// Runs the same traversal with explicit vocabulary output demand.
    pub fn forward_with_traversal_hook_with_readout<'a, H>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        hook: &mut H,
        demand: eredu_core::OutputDemand,
    ) -> Result<(Option<B::Tensor>, A::ForwardContext), A::Error>
    where
        H: LayeredTraversalHook<B, A::ForwardContext, A::Error> + ?Sized,
    {
        self.forward_with_invocation_and_traversal_hook(
            OrdinaryLayeredInput::new(input),
            state,
            context,
            hook,
            demand,
        )
    }

    pub(crate) fn forward_with_invocation_and_traversal_hook<I, H>(
        &mut self,
        invocation: I,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        hook: &mut H,
        demand: eredu_core::OutputDemand,
    ) -> Result<(Option<B::Tensor>, A::ForwardContext), A::Error>
    where
        I: LayeredInvocation<A, B, S>,
        H: LayeredTraversalHook<B, A::ForwardContext, A::Error> + ?Sized,
    {
        self.forward_with_invocation_and_unit_executor(
            invocation,
            state,
            context,
            hook,
            demand,
            |architecture, group, index, unit, hidden, state, forward, context, hook| {
                if hook.observes_activations() {
                    architecture.forward_unit_observed(
                        group,
                        index,
                        unit,
                        hidden,
                        state,
                        forward,
                        context,
                        &mut TraversalActivationObserver {
                            hook,
                            types: std::marker::PhantomData,
                        },
                    )
                } else {
                    architecture.forward_unit(group, index, unit, hidden, state, forward, context)
                }
            },
        )
    }

    pub(crate) fn forward_with_invocation_and_unit_executor<I, H, E>(
        &mut self,
        mut invocation: I,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        hook: &mut H,
        demand: eredu_core::OutputDemand,
        mut execute: E,
    ) -> Result<(Option<B::Tensor>, A::ForwardContext), A::Error>
    where
        I: LayeredInvocation<A, B, S>,
        E: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Unit,
            &B::Tensor,
            &mut S,
            &mut A::ForwardContext,
            &<B::Tensor as Tensor>::Context,
            &mut H,
        ) -> Result<B::Tensor, A::Error>,
        H: LayeredTraversalHook<B, A::ForwardContext, A::Error> + ?Sized,
    {
        let forward = invocation.begin(&mut self.architecture, state, context, hook)?;
        let metadata = metadata::Destination(<A as LayeredArchitecture<B, S>>::forward_metadata(
            &self.architecture,
            &forward.context,
        ));
        metadata.controls::<(
            E,
            Vec<Option<B::Tensor>>,
            Vec<B::Tensor>,
            Vec<&B::Tensor>,
            metadata::Destination<A::Error>,
        )>()?;
        let mut initial = forward.hidden;
        let mut forward_context = forward.context;
        self.architecture
            .set_readout_demand(&mut forward_context, demand);
        let mut schedule = metadata.schedule(&self.graph)?;
        let mut outputs: Vec<Option<B::Tensor>> = metadata.vector(self.graph.groups().len())?;
        outputs.resize_with(self.graph.groups().len(), || None);
        for &group in self.graph.execution_order() {
            if let Some(retained) = invocation.retained_group(group) {
                schedule
                    .started_with_release(group, |dependency| outputs[dependency] = None)
                    .expect("retained group is dependency ready");
                outputs[group] = retained;
                schedule
                    .ordered(group)
                    .expect("retained group is ordered once");
                continue;
            }
            if invocation.before_group_with_hook(
                &mut self.architecture,
                group,
                &mut initial,
                &mut forward_context,
                state,
                None,
                context,
                hook,
            )? {
                self.architecture
                    .set_readout_demand(&mut forward_context, demand);
            }
            let dependency_slots = schedule
                .dependencies(group)
                .expect("validated execution order contains a known group");
            let mut dependencies = metadata.vector(dependency_slots.len())?;
            dependencies.extend(dependency_slots.iter().filter_map(|&dependency| {
                if invocation.is_inactive_dependency(dependency) {
                    None
                } else {
                    Some(
                        outputs[dependency]
                            .as_ref()
                            .expect("active topological dependency has completed")
                            .clone(),
                    )
                }
            }));
            let mut dependency_refs = metadata.vector(dependencies.len())?;
            dependency_refs.extend(dependencies.iter());
            let mut hidden = self.architecture.begin_execution_group(
                group,
                &initial,
                &dependency_refs,
                state,
                &mut forward_context,
                context,
            )?;
            hook.after_group_begin(group, &mut hidden, &mut forward_context, context)?;
            schedule
                .started_with_release(group, |dependency| outputs[dependency] = None)
                .expect("topological execution starts only ready groups");
            let active = self
                .architecture
                .should_execute_group(group, &forward_context);
            if active {
                let unit_count = self.units[group].len();
                for (index, unit) in self.units[group].iter_mut().enumerate() {
                    if hook.before_unit(
                        group,
                        index,
                        unit_count - index,
                        &mut hidden,
                        &mut forward_context,
                        context,
                    )? == LayeredUnitAction::SkipRemainingGroup
                    {
                        break;
                    }
                    hidden = execute(
                        &mut self.architecture,
                        group,
                        index,
                        unit,
                        &hidden,
                        state,
                        &mut forward_context,
                        context,
                        hook,
                    )?;
                    hook.after_unit(group, index, &mut hidden, &mut forward_context, context)?;
                }
            }
            hidden = self.architecture.complete_execution_group(
                group,
                &hidden,
                state,
                &mut forward_context,
                context,
            )?;
            hook.after_group(group, &mut hidden, &mut forward_context, context)?;
            if active {
                invocation.after_group(group, &hidden);
            } else {
                invocation.after_inactive_group(group);
            }
            outputs[group] = Some(hidden);
            schedule
                .ordered(group)
                .expect("started group can be ordered exactly once");
        }
        let hidden = outputs[self.graph.output()]
            .take()
            .expect("validated graph output completed");
        let selected_hidden = self.architecture.select_readout_positions(
            &hidden,
            &forward_context,
            demand,
            context,
        )?;
        let output = if let Some(selected_hidden) = selected_hidden {
            let output = if hook.observes_activations() {
                self.architecture.finish_forward_observed(
                    &selected_hidden,
                    state,
                    &forward_context,
                    context,
                    &mut TraversalActivationObserver {
                        hook,
                        types: std::marker::PhantomData,
                    },
                )
            } else {
                self.architecture
                    .finish_forward(&selected_hidden, state, &forward_context, context)
            }?;
            Some(output)
        } else {
            None
        };
        Ok((output, forward_context))
    }

    /// Borrows the architecture and its pinned parameter topology.
    pub const fn architecture(&self) -> &A {
        &self.architecture
    }

    /// Runs a typed prediction operation against retained target modules.
    /// Only an operation declaring stable target geometry and observation
    /// declarations preserves the binding; arbitrary mutable access still
    /// invalidates it before execution, including on error or unwind.
    pub fn apply_prediction_target_operation<O>(
        &mut self,
        operation: O,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<O::Output, A::Error>
    where
        O: crate::PredictionTargetOperation<A, B, S>,
    {
        let architecture = if operation.preserves_architecture_declarations() {
            &mut self.architecture
        } else {
            self.architecture_mut()
        };
        operation.apply(architecture, state, parallel, context)
    }

    /// Mutably borrows the architecture.
    pub fn architecture_mut(&mut self) -> &mut A {
        self.observation_binding.invalidate();
        &mut self.architecture
    }

    /// Borrows resident execution units for loading or inspection.
    pub fn units(&self) -> &[Vec<A::Unit>] {
        &self.units
    }

    /// Mutably borrows resident execution units for parameter binding.
    pub fn units_mut(&mut self) -> &mut [Vec<A::Unit>] {
        &mut self.units
    }

    /// Decomposes the runtime without cloning backend-native values.
    pub fn into_parts(self) -> (A, Vec<A::Unit>) {
        (
            self.architecture,
            self.units.into_iter().flatten().collect(),
        )
    }
}

/// Policy controlling acquisition and exact release of one execution unit.
pub trait LayerwisePolicy<B, U>
where
    B: NeuralBackend,
{
    /// Concrete lease owning one populated unit and all residency guards.
    type Lease: std::ops::DerefMut<Target = U>;
    /// Concrete acquisition or completion failure.
    type Error;

    /// Whether every bound unit is idle and can be visited without materialization.
    fn resident_parameters_available(&self) -> bool {
        false
    }

    /// Stable slot ceiling for current idle policy-owned topology. Future unit
    /// materialization is separate. Pending or unknown ownership returns None.
    fn retained_value_slot_bound(&self) -> Option<usize>
    where
        U: Parameterized<B::Tensor>,
    {
        None
    }

    /// Inspects values currently owned by the policy without acquiring units,
    /// mutating parameters or establishing native completion. Include resident
    /// modules, their numerical helpers and overrides used on future loads. Unloaded parameters
    /// have no value here; source and future materialization are separate costs.
    /// False means incomplete coverage, including when some values were visited.
    fn visit_retained_values(&self, _visitor: &mut dyn FnMut(&B::Tensor)) -> bool
    where
        U: Parameterized<B::Tensor>,
    {
        false
    }

    /// Borrows strict metadata and actual values from the idle resident owner.
    /// The higher-ranked visitor cannot retain a unit reference beyond its
    /// lexical source loan. No unit acquisition, mutation, evaluation or
    /// completion is authorized. False means incomplete coverage and callers
    /// must discard partial rows. The existing quote pays callback/source controls.
    fn visit_resident_parameter_sources<V>(
        &self,
        _visitor: &mut V,
        _context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<bool, eredu_nn::Error>
    where
        U: Parameterized<B::Tensor>,
        V: for<'source> eredu_nn::ParameterSourceVisitor<'source, B::Tensor>,
    {
        Ok(false)
    }

    /// Visits all permanently bound units, with no native work or reloading.
    /// Returns false before invoking the visitor when this mechanism is unavailable.
    fn visit_resident_units(&mut self, _visitor: &mut impl FnMut(&mut U)) -> bool {
        false
    }

    /// Borrows one unit for bounded parameter work independently of forward order.
    /// A true result means the operation ran and completion was established. On
    /// failure, native implementations retain the loan until terminal evidence.
    fn inspect_unit<E, F, V>(
        &mut self,
        _ordinal: usize,
        _address: crate::ExecutionUnitAddress,
        _build: F,
        _operation: V,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,

        _preparation: Option<&B::ParameterPreparation<'_>>,
    ) -> Result<bool, LayerwiseAcquireError<E, Self::Error>>
    where
        F: FnOnce(&<B::Tensor as eredu_nn::Tensor>::Context) -> Result<U, E>,
        V: FnOnce(&mut U) -> Result<(), Self::Error>,
    {
        Ok(false)
    }

    /// Lends the actual retained slots and future sources to a prepared publication pass.
    fn visit_parameter_publication(
        &mut self,
        _publication: &mut dyn crate::parameter_operations::ParameterPublication<B::Tensor>,
    ) -> Result<bool, Self::Error> {
        Ok(false)
    }

    /// Selects the supplied executor for every group of this forward.
    /// A policy may opt in only from its retained preparation and must validate
    /// that executor before returning true. The default preserves backend forks.
    fn uses_shared_group_executor(&self, _executor: &B::Executor) -> Result<bool, Self::Error>
    where
        B: SubmissionBackend,
    {
        Ok(false)
    }

    /// Submits one graph-boundary value using this retained execution policy.
    /// The returned owner cannot borrow this policy, executor or input; it may
    /// retain finite backend resources through subsequent group scheduling.
    /// The default preserves ordinary backend submission and ordering exactly.
    /// Initial submission precedes `begin`; later groups use the same method
    /// through the active policy. No arbitrary host-retention API is implied.
    /// Creation errors retain the policy's own concrete type; the default uses
    /// the backend completion error without allocating a conversion wrapper.
    fn submit_group(
        &mut self,
        executor: &B::Executor,
        value: &B::Tensor,
    ) -> Result<
        impl OrderedLayerwiseCompletion<B::Executor> + 'static,
        impl std::error::Error + Send + Sync + 'static,
    >
    where
        B: SubmissionBackend,
    {
        B::submit(executor, [value]).map(BackendLayerwiseCompletion::<B>::new)
    }

    /// Starts one forward after architecture input preparation.
    fn begin(
        &mut self,
        initial: &B::Tensor,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), Self::Error>;

    /// Aborts an incomplete forward and releases all policy-owned state.
    ///
    /// `active` contains the unit lease when execution stopped after
    /// acquisition but before exact completion. Implementations with no
    /// forward-scoped state may rely on the default, which simply drops it.
    fn abort(
        &mut self,
        active: Option<(usize, crate::ExecutionUnitAddress, Self::Lease)>,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) {
        drop(active);
    }

    /// Acquires one populated unit for exclusive execution.
    ///
    /// The flat ordinal addresses storage while `address` preserves the
    /// architecture execution group and group-local unit index for scheduling.
    fn acquire<E, F>(
        &mut self,
        ordinal: usize,
        address: crate::ExecutionUnitAddress,
        build: F,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Self::Lease, LayerwiseAcquireError<E, Self::Error>>
    where
        F: FnOnce(&<B::Tensor as eredu_nn::Tensor>::Context) -> Result<U, E>;

    /// Retains the unit and dependent native values through exact completion.
    fn complete<'a, StateValues, ContextValues>(
        &mut self,
        ordinal: usize,
        address: crate::ExecutionUnitAddress,
        lease: Self::Lease,
        output: &'a B::Tensor,
        state_values: StateValues,
        context_values: ContextValues,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), Self::Error>
    where
        B::Tensor: 'a,
        StateValues: Iterator<Item = &'a B::Tensor>,
        ContextValues: Iterator<Item = &'a B::Tensor>;

    /// Completes the final output and releases any remaining unit guards.
    fn finish(
        &mut self,
        output: &B::Tensor,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), Self::Error>;
}

/// Failure-safe ownership of one policy forward and its current unit lease.
pub struct LayerwisePolicyForward<'a, B, U, P>
where
    B: NeuralBackend,
    P: LayerwisePolicy<B, U>,
{
    policy: &'a mut P,
    context: &'a <B::Tensor as eredu_nn::Tensor>::Context,
    active: Option<(usize, crate::ExecutionUnitAddress, P::Lease)>,
    finished: bool,
    unit: std::marker::PhantomData<fn() -> U>,
}

impl<'a, B, U, P> LayerwisePolicyForward<'a, B, U, P>
where
    B: NeuralBackend,
    P: LayerwisePolicy<B, U>,
{
    /// Begins one failure-safe policy transaction.
    pub fn begin(
        policy: &'a mut P,
        initial: &B::Tensor,
        context: &'a <B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Self, P::Error> {
        if let Err(error) = policy.begin(initial, context) {
            policy.abort(None, context);
            return Err(error);
        }
        Ok(Self {
            policy,
            context,
            active: None,
            finished: false,
            unit: std::marker::PhantomData,
        })
    }

    /// Acquires one unit and retains its lease until completion or abort.
    pub fn acquire<E, F>(
        &mut self,
        ordinal: usize,
        address: crate::ExecutionUnitAddress,
        build: F,
    ) -> Result<&mut P::Lease, LayerwiseAcquireError<E, P::Error>>
    where
        F: FnOnce(&<B::Tensor as eredu_nn::Tensor>::Context) -> Result<U, E>,
    {
        debug_assert!(self.active.is_none());
        let lease = self.policy.acquire(ordinal, address, build, self.context)?;
        self.active = Some((ordinal, address, lease));
        Ok(&mut self
            .active
            .as_mut()
            .expect("acquired policy lease is active")
            .2)
    }

    /// Completes and returns the currently active unit lease.
    pub fn complete<'value, StateValues, ContextValues>(
        &mut self,
        output: &'value B::Tensor,
        state_values: StateValues,
        context_values: ContextValues,
    ) -> Result<(), P::Error>
    where
        B::Tensor: 'value,
        StateValues: Iterator<Item = &'value B::Tensor>,
        ContextValues: Iterator<Item = &'value B::Tensor>,
    {
        let (ordinal, address, lease) = self
            .active
            .take()
            .expect("policy completion follows one acquisition");
        self.policy.complete(
            ordinal,
            address,
            lease,
            output,
            state_values,
            context_values,
            self.context,
        )
    }

    /// Completes the transaction; dropping before this call aborts it.
    pub fn finish(&mut self, output: &B::Tensor) -> Result<(), P::Error> {
        self.policy.finish(output, self.context)?;
        self.finished = true;
        Ok(())
    }
}

impl<B, U, P> Drop for LayerwisePolicyForward<'_, B, U, P>
where
    B: NeuralBackend,
    P: LayerwisePolicy<B, U>,
{
    fn drop(&mut self) {
        if !self.finished {
            self.policy.abort(self.active.take(), self.context);
        }
    }
}

/// Failure while a layerwise policy acquires or populates one architecture unit.
#[derive(Debug)]
pub enum LayerwiseAcquireError<A, P> {
    /// The neutral architecture could not construct its unloaded unit.
    Architecture(A),
    /// The execution policy could not acquire residency or populate the unit.
    Policy(P),
}

/// Failure from architecture execution or layerwise residency policy.
#[derive(Debug, thiserror::Error)]
pub enum LayerwiseRuntimeError<A, P>
where
    A: std::fmt::Display,
    P: std::fmt::Display,
{
    /// Architecture construction or forward failure.
    #[error("layered architecture failed: {0}")]
    Architecture(#[source] A),
    /// Invalid access to architecture-declared mutable state.
    #[error(transparent)]
    State(#[from] crate::StateError),
    /// Architecture execution groups did not map to one stable residency-unit order.
    #[error(transparent)]
    Layout(#[from] crate::ExecutionUnitLayoutError),
    /// Unit acquisition or exact-completion failure.
    #[error("layerwise execution policy failed: {0}")]
    Policy(P),
    /// Backend-native graph submission or dependency ordering failed.
    #[error("layerwise backend submission failed: {0}")]
    Submission(#[source] eredu_core::BackendFailure),
}

/// Bounded-unit runtime invoking the same architecture lifecycle as resident execution.
pub struct LayerwiseRuntime<A, B, S, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as eredu_nn::Tensor>::Context>,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    P: LayerwisePolicy<B, A::Unit>,
{
    architecture: A,
    policy: P,
    executors: Option<Vec<B::OwnedExecutor>>,
    prepared_geometry: Option<crate::PreparedReplicatedTextExecutionGeometry>,
    geometry_stale: bool,
    observation_binding: ObservationBinding,
    backend: std::marker::PhantomData<fn() -> (B, S)>,
}

impl<A, B, S, P> crate::parameter_operations::LayeredParameterOwner<B, S>
    for LayerwiseRuntime<A, B, S, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as eredu_nn::Tensor>::Context>,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    P: LayerwisePolicy<B, A::Unit>,
{
    type Architecture = A;
    type Policy = P;
    fn parameter_parts(&mut self) -> Option<(&mut A, &mut P)> {
        self.observation_binding.invalidate();
        self.geometry_stale |= self.prepared_geometry.is_some();
        Some((&mut self.architecture, &mut self.policy))
    }
    fn parameter_parts_ref(&self) -> Option<(&A, &P)> {
        Some((&self.architecture, &self.policy))
    }
    fn visit_loaded_parameters(
        &mut self,
        visitor: &mut dyn eredu_nn::ParameterSlotVisitor<B::Tensor>,
    ) -> bool {
        crate::parameter_operations::visit_loaded_parameters_in_parts::<A, B, S, P>(
            &mut self.architecture,
            &mut self.policy,
            visitor,
        )
    }
    fn with_parameter_slots(
        &mut self,
        location: &crate::parameter_operations::PreparedParameterLocation,
        operation: &mut crate::parameter_operations::ParameterSlotOperation<
            '_,
            B::Tensor,
            P::Error,
        >,
        context: &<B::Tensor as Tensor>::Context,

        _preparation: Option<&B::ParameterPreparation<'_>>,
    ) -> Result<bool, LayerwiseAcquireError<A::Error, P::Error>> {
        crate::parameter_operations::with_parameter_slots_in_parts::<A, B, S, P>(
            &mut self.architecture,
            &mut self.policy,
            location,
            operation,
            context,
            _preparation,
        )
    }

    /// Lends the actual retained slots and future sources to a prepared publication pass.
    fn visit_parameter_publication(
        &mut self,
        publication: &mut dyn crate::parameter_operations::ParameterPublication<B::Tensor>,
    ) -> Result<bool, P::Error> {
        crate::parameter_operations::visit_parameter_publication_in_parts::<A, B, S, P>(
            &mut self.architecture,
            &mut self.policy,
            publication,
        )
    }
}

impl<A, B, S, P> LayerwiseRuntime<A, B, S, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as eredu_nn::Tensor>::Context>,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    P: LayerwisePolicy<B, A::Unit>,
    A::Error: std::fmt::Display,
    P::Error: std::fmt::Display,
{
    /// Creates a layerwise runtime from concrete architecture, state, and policy.
    pub const fn new(architecture: A, policy: P) -> Self {
        Self {
            architecture,
            policy,
            executors: None,
            prepared_geometry: None,
            geometry_stale: false,
            observation_binding: ObservationBinding::new(),
            backend: std::marker::PhantomData,
        }
    }

    /// Uses the existing geometry from the matching validated construction
    /// contract. No graph, declaration or layout clone is made. Mutation through
    /// an arbitrary architecture handle makes subsequent traversal refuse this
    /// retained geometry instead of falling back to ordinary reconstruction.
    pub fn new_with_prepared_geometry(
        architecture: A,
        policy: P,
        geometry: crate::PreparedReplicatedTextExecutionGeometry,
    ) -> Self {
        Self {
            architecture,
            policy,
            executors: None,
            prepared_geometry: Some(geometry),
            geometry_stale: false,
            observation_binding: ObservationBinding::new(),
            backend: std::marker::PhantomData,
        }
    }

    /// Creates a layerwise runtime while evaluating the policy before moving
    /// the architecture. This is useful when policy realization needs to
    /// borrow the architecture's canonical unit constructor first.
    pub const fn new_policy_first(policy: P, architecture: A) -> Self {
        Self::new(architecture, policy)
    }

    /// Borrows the concrete architecture instance.
    pub const fn architecture(&self) -> &A {
        &self.architecture
    }

    /// Runs a typed prediction operation against retained target modules.
    /// Only an operation declaring stable target geometry and observation
    /// declarations preserves the binding; arbitrary mutable access still
    /// invalidates it before execution, including on error or unwind.
    pub fn apply_prediction_target_operation<O>(
        &mut self,
        operation: O,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<O::Output, A::Error>
    where
        O: crate::PredictionTargetOperation<A, B, S>,
    {
        let architecture = if operation.preserves_architecture_declarations() {
            &mut self.architecture
        } else {
            self.architecture_mut()
        };
        operation.apply(architecture, state, parallel, context)
    }

    /// Mutably borrows the concrete architecture instance.
    pub fn architecture_mut(&mut self) -> &mut A {
        self.observation_binding.invalidate();
        self.geometry_stale |= self.prepared_geometry.is_some();
        &mut self.architecture
    }

    /// Borrows the concrete execution policy for cold-path diagnostics.
    pub const fn policy(&self) -> &P {
        &self.policy
    }

    /// Mutably borrows the concrete execution policy.
    pub fn policy_mut(&mut self) -> &mut P {
        &mut self.policy
    }

    /// Runs a bounded parameter operation through the architecture's unit owner.
    pub fn inspect_parameter_unit<V>(
        &mut self,
        ordinal: usize,
        address: crate::ExecutionUnitAddress,
        operation: V,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<bool, LayerwiseAcquireError<A::Error, P::Error>>
    where
        V: FnOnce(&mut A::Unit) -> Result<(), P::Error>,
    {
        let architecture = &self.architecture;
        self.policy.inspect_unit(
            ordinal,
            address,
            |context| architecture.build_unit(address.group(), address.index(), context),
            operation,
            context,
            None,
        )
    }

    /// Runs one complete prefill or decode pass with exact unit release points.
    pub fn forward<'a>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, LayerwiseRuntimeError<A::Error, P::Error>> {
        self.forward_with_traversal_hook(input, state, context, &mut NoopLayeredTraversalHook)
            .map(|(output, _)| output)
    }

    /// Runs one pass and exposes mutable architecture context after each unit.
    pub fn forward_with_context_hook<'a, H>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        hook: H,
    ) -> Result<(B::Tensor, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        H: FnMut(usize, usize, &mut A::ForwardContext) -> Result<(), A::Error>,
    {
        self.forward_with_unit_executor_and_context_hook(
            input,
            state,
            context,
            |architecture, group, index, unit, hidden, state, forward, context| {
                architecture.forward_unit(group, index, unit, hidden, state, forward, context)
            },
            hook,
        )
    }

    /// Runs one pass with a statically dispatched architecture-unit executor.
    ///
    /// Composition can use this cold API to inject routed expert execution or
    /// observation while the runtime retains graph traversal, residency, and
    /// exact completion ownership.
    pub fn forward_with_unit_executor<'a, E>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        execute: E,
    ) -> Result<B::Tensor, LayerwiseRuntimeError<A::Error, P::Error>>
    where
        E: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Unit,
            &B::Tensor,
            &mut S,
            &mut A::ForwardContext,
            &<B::Tensor as eredu_nn::Tensor>::Context,
        ) -> Result<B::Tensor, A::Error>,
    {
        self.forward_with_unit_executor_and_context_hook(
            input,
            state,
            context,
            execute,
            |_, _, _| Ok(()),
        )
        .map(|(output, _)| output)
    }

    /// Runs the production sequential traversal with stable unit-boundary observation.
    pub fn forward_with_observer<'a, Observer>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut Observer,
    ) -> Result<B::Tensor, LayerwiseRuntimeError<A::Error, P::Error>>
    where
        Observer: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.forward_with_observer_and_context(input, state, context, observer)
            .map(|(output, _)| output)
    }

    /// Runs observed sequential traversal and retains architecture forward resources for the
    /// caller's completion boundary.
    pub fn forward_with_observer_and_context<'a, Observer>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut Observer,
    ) -> Result<(B::Tensor, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        Observer: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.forward_with_observer_and_context_with_readout(
            input,
            state,
            context,
            observer,
            eredu_core::OutputDemand::Sequence,
        )
        .map(|(output, forward)| (output.expect("sequence readout returns scores"), forward))
    }

    /// Runs the same traversal with explicit vocabulary output demand.
    pub fn forward_with_observer_and_context_with_readout<'a, Observer>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut Observer,
        demand: eredu_core::OutputDemand,
    ) -> Result<(Option<B::Tensor>, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        Observer: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.forward_invocation_with_internal_observer(
            OrdinaryLayeredInput::new(input),
            state,
            context,
            |architecture, group, index, unit, hidden, state, forward, context, observer| {
                architecture.forward_unit_observed(
                    group, index, unit, hidden, state, forward, context, observer,
                )
            },
            observer,
            demand,
            true,
        )
    }

    /// Runs a custom production unit executor with stable boundary observation.
    pub fn forward_with_unit_executor_and_observer<'a, E, Observer>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        execute: E,
        observer: &mut Observer,
    ) -> Result<B::Tensor, LayerwiseRuntimeError<A::Error, P::Error>>
    where
        E: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Unit,
            &B::Tensor,
            &mut S,
            &mut A::ForwardContext,
            &<B::Tensor as eredu_nn::Tensor>::Context,
        ) -> Result<B::Tensor, A::Error>,
        Observer: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.forward_with_unit_executor_and_observer_and_context(
            input, state, context, execute, observer,
        )
        .map(|(output, _)| output)
    }

    /// Runs a custom observed executor while returning its retained forward resources.
    pub fn forward_with_unit_executor_and_observer_and_context<'a, E, Observer>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        mut execute: E,
        observer: &mut Observer,
    ) -> Result<(B::Tensor, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        E: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Unit,
            &B::Tensor,
            &mut S,
            &mut A::ForwardContext,
            &<B::Tensor as eredu_nn::Tensor>::Context,
        ) -> Result<B::Tensor, A::Error>,
        Observer: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.forward_with_internal_observer_and_context(
            input,
            state,
            context,
            |architecture, group, index, unit, hidden, state, forward, context, observer| {
                let owned = architecture.observes_unit_boundaries(group, index);
                let path = owned
                    .then(|| architecture.unit_path(group, index, None))
                    .transpose()?;
                let input = path
                    .as_ref()
                    .map(|path| {
                        observe_outer_boundary(
                            observer,
                            &format!("{path}.input"),
                            &format!("{path}.input.effective"),
                            hidden,
                        )
                    })
                    .transpose()?;
                let output = execute(
                    architecture,
                    group,
                    index,
                    unit,
                    input.as_ref().unwrap_or(hidden),
                    state,
                    forward,
                    context,
                )?;
                match path {
                    Some(path) => observe_outer_boundary(
                        observer,
                        &format!("{path}.output"),
                        &format!("{path}.output.effective"),
                        &output,
                    ),
                    None => Ok(output),
                }
            },
            observer,
        )
    }

    /// Runs a caller-selected unit through the shared observed traversal, retaining forward resources.
    pub fn forward_with_internal_observer_and_context<'a, E, Observer>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        execute: E,
        observer: &mut Observer,
    ) -> Result<(B::Tensor, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        E: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Unit,
            &B::Tensor,
            &mut S,
            &mut A::ForwardContext,
            &<B::Tensor as eredu_nn::Tensor>::Context,
            &mut Observer,
        ) -> Result<B::Tensor, A::Error>,
        Observer: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.forward_with_internal_observer_and_context_with_readout(
            input,
            state,
            context,
            execute,
            observer,
            eredu_core::OutputDemand::Sequence,
        )
        .map(|(output, forward)| (output.expect("sequence readout returns scores"), forward))
    }

    /// Runs the same traversal with explicit vocabulary output demand.
    pub fn forward_with_internal_observer_and_context_with_readout<'a, E, Observer>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        execute: E,
        observer: &mut Observer,
        demand: eredu_core::OutputDemand,
    ) -> Result<(Option<B::Tensor>, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        E: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Unit,
            &B::Tensor,
            &mut S,
            &mut A::ForwardContext,
            &<B::Tensor as eredu_nn::Tensor>::Context,
            &mut Observer,
        ) -> Result<B::Tensor, A::Error>,
        Observer: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.forward_invocation_with_internal_observer(
            OrdinaryLayeredInput::new(input),
            state,
            context,
            execute,
            observer,
            demand,
            false,
        )
    }

    pub(crate) fn forward_invocation_with_internal_observer<I, E, Observer>(
        &mut self,
        invocation: I,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        mut execute: E,
        observer: &mut Observer,
        demand: eredu_core::OutputDemand,
        retained_unit_selection: bool,
    ) -> Result<(Option<B::Tensor>, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        I: LayeredInvocation<A, B, S>,
        E: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Unit,
            &B::Tensor,
            &mut S,
            &mut A::ForwardContext,
            &<B::Tensor as eredu_nn::Tensor>::Context,
            &mut Observer,
        ) -> Result<B::Tensor, A::Error>,
        Observer: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        if self.geometry_stale {
            return Err(crate::ExecutionUnitLayoutError::StalePreparedGeometry.into());
        }
        let ordinary_graph;
        let graph = match self.prepared_geometry.as_ref() {
            Some(geometry) => geometry.graph(),
            None => {
                ordinary_graph = self
                    .architecture
                    .execution_graph()
                    .map_err(LayerwiseRuntimeError::Architecture)?
                    .into_owned();
                &ordinary_graph
            }
        };
        let mut units = Vec::with_capacity(graph.groups().len());
        let mut group_inputs = Vec::with_capacity(graph.groups().len());
        let mut group_outputs = Vec::with_capacity(graph.groups().len());
        for group in 0..graph.groups().len() {
            let count = self
                .architecture
                .group_unit_count(group, None)
                .map_err(LayerwiseRuntimeError::Architecture)?;
            units.push(
                (0..count)
                    .map(|index| {
                        if self.architecture.observes_unit_boundaries(group, index) {
                            Ok(None)
                        } else {
                            self.architecture.unit_path(group, index, None).map(Some)
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(LayerwiseRuntimeError::Architecture)?,
            );
            group_inputs.push(
                self.architecture
                    .group_input_observation_path(group, None)
                    .map_err(LayerwiseRuntimeError::Architecture)?,
            );
            group_outputs.push(
                self.architecture
                    .group_output_observation_path(group, None)
                    .map_err(LayerwiseRuntimeError::Architecture)?,
            );
        }
        let observer = std::rc::Rc::new(std::cell::RefCell::new(observer));
        let internal_observer = observer.clone();
        let mut hook = ActivationObserverTraversalHook {
            observer,
            units,
            group_inputs,
            group_outputs,
        };
        self.forward_with_unit_executor_and_invocation(
            invocation,
            state,
            context,
            |architecture, group, index, unit, hidden, state, forward, context, _hook| {
                execute(
                    architecture,
                    group,
                    index,
                    unit,
                    hidden,
                    state,
                    forward,
                    context,
                    &mut **internal_observer.borrow_mut(),
                )
            },
            &mut hook,
            false,
            retained_unit_selection,
            demand,
        )
    }

    /// Runs canonical provider-backed unit execution with unit-boundary and
    /// routed-expert observation.
    ///
    /// Observation wraps [`RoutedLayeredArchitecture::forward_unit_with_provider`]
    /// instead of replacing it. Architecture-owned validation, state lookup,
    /// shape handling, routing, and provider dispatch therefore remain shared
    /// with ordinary execution.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_provider_and_observer<'a, Provider, Observer>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        pass: ExpertPass,
        provider: &mut Provider,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut Observer,
    ) -> Result<B::Tensor, LayerwiseRuntimeError<A::Error, P::Error>>
    where
        B: eredu_nn::GroupedNeuralBackend,
        A: RoutedLayeredArchitecture<B, S>,
        A::Error: std::fmt::Display,
        Provider: RoutedExpertProvider<B>,
        Provider::Error: std::fmt::Display,
        Observer: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.forward_with_provider_and_observer_and_context(
            input, state, pass, provider, context, observer,
        )
        .map(|(output, _)| output)
    }

    /// Runs provider-backed observed execution while retaining architecture forward resources.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_provider_and_observer_and_context<'a, Provider, Observer>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        pass: ExpertPass,
        provider: &mut Provider,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut Observer,
    ) -> Result<(B::Tensor, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        B: eredu_nn::GroupedNeuralBackend,
        A: RoutedLayeredArchitecture<B, S>,
        A::Error: std::fmt::Display,
        Provider: RoutedExpertProvider<B>,
        Provider::Error: std::fmt::Display,
        Observer: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.forward_with_provider_and_observer_and_context_with_readout(
            input,
            state,
            pass,
            provider,
            context,
            observer,
            eredu_core::OutputDemand::Sequence,
        )
        .map(|(output, forward)| (output.expect("sequence readout returns scores"), forward))
    }

    /// Runs the same traversal with explicit vocabulary output demand.
    pub fn forward_with_provider_and_observer_and_context_with_readout<'a, Provider, Observer>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        pass: ExpertPass,
        provider: &mut Provider,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut Observer,
        demand: eredu_core::OutputDemand,
    ) -> Result<(Option<B::Tensor>, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        B: eredu_nn::GroupedNeuralBackend,
        A: RoutedLayeredArchitecture<B, S>,
        A::Error: std::fmt::Display,
        Provider: RoutedExpertProvider<B>,
        Provider::Error: std::fmt::Display,
        Observer: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.forward_invocation_with_internal_observer(
            OrdinaryLayeredInput::new(input),
            state,
            context,
            |architecture, group, index, unit, hidden, state, forward, context, observer| {
                architecture.forward_unit_observed_with_provider(
                    group, index, unit, hidden, state, forward, pass, provider, context, observer,
                )
            },
            observer,
            demand,
            true,
        )
    }

    /// Runs provider-backed observed execution with architecture-owned pass
    /// classification.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_inferred_provider_and_observer<'a, Provider, Observer>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        provider: &mut Provider,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut Observer,
    ) -> Result<B::Tensor, LayerwiseRuntimeError<A::Error, P::Error>>
    where
        B: eredu_nn::GroupedNeuralBackend,
        A: RoutedLayeredArchitecture<B, S>,
        A::Error: std::fmt::Display,
        Provider: RoutedExpertProvider<B>,
        Provider::Error: std::fmt::Display,
        Observer: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.forward_with_inferred_provider_and_observer_and_context(
            input, state, provider, context, observer,
        )
        .map(|(output, _)| output)
    }

    /// Runs provider-backed observed execution with architecture-owned pass
    /// classification while retaining forward resources.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_inferred_provider_and_observer_and_context<'a, Provider, Observer>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        provider: &mut Provider,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut Observer,
    ) -> Result<(B::Tensor, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        B: eredu_nn::GroupedNeuralBackend,
        A: RoutedLayeredArchitecture<B, S>,
        A::Error: std::fmt::Display,
        Provider: RoutedExpertProvider<B>,
        Provider::Error: std::fmt::Display,
        Observer: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.forward_invocation_with_internal_observer(
            OrdinaryLayeredInput::new(input),
            state,
            context,
            |architecture, group, index, unit, hidden, state, forward, context, observer| {
                architecture.forward_unit_observed_with_inferred_provider(
                    group, index, unit, hidden, state, forward, provider, context, observer,
                )
            },
            observer,
            eredu_core::OutputDemand::Sequence,
            true,
        )
        .map(|(output, forward)| (output.expect("sequence readout returns scores"), forward))
    }

    /// Runs one pass with both a custom unit executor and post-unit context hook.
    pub fn forward_with_unit_executor_and_context_hook<'a, E, H>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        execute: E,
        mut hook: H,
    ) -> Result<(B::Tensor, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        E: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Unit,
            &B::Tensor,
            &mut S,
            &mut A::ForwardContext,
            &<B::Tensor as eredu_nn::Tensor>::Context,
        ) -> Result<B::Tensor, A::Error>,
        H: FnMut(usize, usize, &mut A::ForwardContext) -> Result<(), A::Error>,
    {
        self.forward_with_unit_executor_and_activation_hook(
            input,
            state,
            context,
            execute,
            |group, index, _hidden, forward| hook(group, index, forward),
        )
    }

    /// Runs one pass with a custom unit executor and exposes each post-unit
    /// activation together with the mutable architecture context.
    ///
    /// The activation is the ordinary output of the execution unit. Target
    /// state taps and inspection therefore observe the production forward
    /// without requiring a second family-specific model path.
    pub fn forward_with_unit_executor_and_activation_hook<'a, E, H>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        execute: E,
        hook: H,
    ) -> Result<(B::Tensor, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        E: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Unit,
            &B::Tensor,
            &mut S,
            &mut A::ForwardContext,
            &<B::Tensor as eredu_nn::Tensor>::Context,
        ) -> Result<B::Tensor, A::Error>,
        H: FnMut(usize, usize, &B::Tensor, &mut A::ForwardContext) -> Result<(), A::Error>,
    {
        self.forward_with_unit_executor_and_traversal_hook(
            input,
            state,
            context,
            execute,
            &mut AfterUnitTraversalHook { after_unit: hook },
        )
    }

    /// Runs one bounded pass through a statically dispatched traversal hook.
    pub fn forward_with_traversal_hook<'a, H>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        hook: &mut H,
    ) -> Result<(B::Tensor, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        H: LayeredTraversalHook<B, A::ForwardContext, A::Error> + ?Sized,
    {
        self.forward_with_traversal_hook_with_readout(
            input,
            state,
            context,
            hook,
            eredu_core::OutputDemand::Sequence,
        )
        .map(|(output, forward)| (output.expect("sequence readout returns scores"), forward))
    }

    /// Runs the same traversal with explicit vocabulary output demand.
    pub fn forward_with_traversal_hook_with_readout<'a, H>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        hook: &mut H,
        demand: eredu_core::OutputDemand,
    ) -> Result<(Option<B::Tensor>, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        H: LayeredTraversalHook<B, A::ForwardContext, A::Error> + ?Sized,
    {
        self.forward_with_unit_executor_and_traversal_hook_impl(
            input,
            state,
            context,
            |architecture, group, index, unit, hidden, state, forward, context| {
                architecture.forward_unit(group, index, unit, hidden, state, forward, context)
            },
            hook,
            true,
            demand,
        )
    }

    /// Runs one bounded pass with custom unit execution and a shared traversal hook.
    pub fn forward_with_unit_executor_and_traversal_hook<'a, E, H>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        execute: E,
        hook: &mut H,
    ) -> Result<(B::Tensor, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        E: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Unit,
            &B::Tensor,
            &mut S,
            &mut A::ForwardContext,
            &<B::Tensor as eredu_nn::Tensor>::Context,
        ) -> Result<B::Tensor, A::Error>,
        H: LayeredTraversalHook<B, A::ForwardContext, A::Error> + ?Sized,
    {
        self.forward_with_unit_executor_and_traversal_hook_with_readout(
            input,
            state,
            context,
            execute,
            hook,
            eredu_core::OutputDemand::Sequence,
        )
        .map(|(output, forward)| (output.expect("sequence readout returns scores"), forward))
    }

    /// Runs the same traversal with explicit vocabulary output demand.
    pub fn forward_with_unit_executor_and_traversal_hook_with_readout<'a, E, H>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        execute: E,
        hook: &mut H,
        demand: eredu_core::OutputDemand,
    ) -> Result<(Option<B::Tensor>, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        E: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Unit,
            &B::Tensor,
            &mut S,
            &mut A::ForwardContext,
            &<B::Tensor as eredu_nn::Tensor>::Context,
        ) -> Result<B::Tensor, A::Error>,
        H: LayeredTraversalHook<B, A::ForwardContext, A::Error> + ?Sized,
    {
        self.forward_with_unit_executor_and_traversal_hook_impl(
            input, state, context, execute, hook, false, demand,
        )
    }

    fn forward_with_unit_executor_and_traversal_hook_impl<'a, E, H>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        mut execute: E,
        hook: &mut H,
        observe_unit_internals: bool,
        demand: eredu_core::OutputDemand,
    ) -> Result<(Option<B::Tensor>, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        E: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Unit,
            &B::Tensor,
            &mut S,
            &mut A::ForwardContext,
            &<B::Tensor as eredu_nn::Tensor>::Context,
        ) -> Result<B::Tensor, A::Error>,
        H: LayeredTraversalHook<B, A::ForwardContext, A::Error> + ?Sized,
    {
        self.forward_with_unit_executor_and_invocation(
            OrdinaryLayeredInput::new(input),
            state,
            context,
            |architecture, group, index, unit, hidden, state, forward, context, _hook| {
                execute(
                    architecture,
                    group,
                    index,
                    unit,
                    hidden,
                    state,
                    forward,
                    context,
                )
            },
            hook,
            observe_unit_internals,
            false,
            demand,
        )
    }

    pub(crate) fn forward_with_unit_executor_and_invocation<I, E, H>(
        &mut self,
        mut invocation: I,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        mut execute: E,
        hook: &mut H,
        observe_unit_internals: bool,
        preserve_observation_binding: bool,
        demand: eredu_core::OutputDemand,
    ) -> Result<(Option<B::Tensor>, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        I: LayeredInvocation<A, B, S>,
        E: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Unit,
            &B::Tensor,
            &mut S,
            &mut A::ForwardContext,
            &<B::Tensor as eredu_nn::Tensor>::Context,
            &mut H,
        ) -> Result<B::Tensor, A::Error>,
        H: LayeredTraversalHook<B, A::ForwardContext, A::Error> + ?Sized,
    {
        // Fixed architecture/provider equations preserve declarations. Caller-selected
        // unit callbacks invalidate the binding before gaining mutable access.
        if !observe_unit_internals && !preserve_observation_binding {
            self.observation_binding.invalidate();
        }

        if self.geometry_stale {
            return Err(crate::ExecutionUnitLayoutError::StalePreparedGeometry.into());
        }
        let ordinary_geometry;
        let (graph, layout) = match self.prepared_geometry.as_ref() {
            Some(geometry) => (geometry.graph(), geometry.units()),
            None => {
                let graph = self
                    .architecture
                    .execution_graph()
                    .map_err(LayerwiseRuntimeError::Architecture)?
                    .into_owned();
                let counts = (0..graph.groups().len())
                    .map(|group| {
                        self.architecture
                            .group_unit_count(group, None)
                            .map_err(LayerwiseRuntimeError::Architecture)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let layout = ExecutionUnitLayout::new(&graph, counts)?;
                ordinary_geometry = (graph, layout);
                (&ordinary_geometry.0, &ordinary_geometry.1)
            }
        };
        if !observe_unit_internals && !preserve_observation_binding {
            self.geometry_stale |= self.prepared_geometry.is_some();
        }
        let shared_executor = self
            .policy
            .uses_shared_group_executor(context)
            .map_err(LayerwiseRuntimeError::Policy)?;
        if !shared_executor && self.executors.as_ref().map(Vec::len) != Some(graph.groups().len()) {
            self.executors = Some(B::fork_executors(context, graph.groups().len()).map_err(
                |error| {
                    LayerwiseRuntimeError::Submission(eredu_core::BackendFailure::from_error(error))
                },
            )?);
        }
        let executors = self.executors.as_ref();
        let forward = invocation
            .begin(&mut self.architecture, state, context, hook)
            .map_err(LayerwiseRuntimeError::Architecture)?;
        let metadata = metadata::Destination(<A as LayeredArchitecture<B, S>>::forward_metadata(
            &self.architecture,
            &forward.context,
        ));
        if preserve_observation_binding && !observe_unit_internals {
            metadata
                .controls::<(
                    E,
                    E,
                    &mut H,
                    Result<
                        (Option<B::Tensor>, A::ForwardContext),
                        LayerwiseRuntimeError<A::Error, P::Error>,
                    >,
                )>()
                .map_err(LayerwiseRuntimeError::Architecture)?;
        }
        metadata
            .controls::<(
                Vec<Option<B::Tensor>>,
                Vec<B::Tensor>,
                Vec<&B::Tensor>,
                metadata::Destination<A::Error>,
            )>()
            .map_err(LayerwiseRuntimeError::Architecture)?;
        let initial_completion = (graph.groups().len() > 1)
            .then(|| self.policy.submit_group(context, &forward.hidden))
            .transpose()
            .map_err(|error| {
                LayerwiseRuntimeError::Submission(eredu_core::BackendFailure::from_error(error))
            })?;
        let mut policy = LayerwisePolicyForward::begin(&mut self.policy, &forward.hidden, context)
            .map_err(LayerwiseRuntimeError::Policy)?;
        let mut initial = forward.hidden;
        let mut forward_context = forward.context;
        self.architecture
            .set_readout_demand(&mut forward_context, demand);
        let mut schedule = metadata
            .schedule(&graph)
            .map_err(LayerwiseRuntimeError::Architecture)?;
        let mut outputs: Vec<Option<B::Tensor>> = metadata
            .vector(graph.groups().len())
            .map_err(LayerwiseRuntimeError::Architecture)?;
        outputs.resize_with(graph.groups().len(), || None);
        let mut completions = metadata
            .completions(&initial_completion, graph.groups().len())
            .map_err(LayerwiseRuntimeError::Architecture)?;
        for &group in graph.execution_order() {
            if let Some(retained) = invocation.retained_group(group) {
                schedule
                    .started_with_release(group, |dependency| outputs[dependency] = None)
                    .expect("retained group is dependency ready");
                outputs[group] = retained;
                schedule
                    .ordered(group)
                    .expect("retained group is ordered once");
                continue;
            }
            let executor = if shared_executor {
                context
            } else {
                std::borrow::Borrow::borrow(
                    &executors.expect("layered runtime initialized its executor cache")[group],
                )
            };
            let group_dependencies = schedule
                .dependencies(group)
                .expect("validated execution order contains a known group");
            if group_dependencies.is_empty() {
                if let Some(completion) = &initial_completion {
                    completion.order_after(executor).map_err(|error| {
                        LayerwiseRuntimeError::Submission(eredu_core::BackendFailure::from_error(
                            error,
                        ))
                    })?;
                }
            }
            for &dependency in group_dependencies {
                if let Some(completion) = completions[dependency].as_ref() {
                    completion.order_after(executor).map_err(|error| {
                        LayerwiseRuntimeError::Submission(eredu_core::BackendFailure::from_error(
                            error,
                        ))
                    })?;
                } else {
                    debug_assert!(
                        invocation.is_retained_group(dependency),
                        "only completed retained ingress omits a new completion"
                    );
                }
            }
            if invocation
                .before_group_with_hook(
                    &mut self.architecture,
                    group,
                    &mut initial,
                    &mut forward_context,
                    state,
                    None,
                    executor,
                    hook,
                )
                .map_err(LayerwiseRuntimeError::Architecture)?
            {
                self.architecture
                    .set_readout_demand(&mut forward_context, demand);
            }
            let dependency_slots = schedule
                .dependencies(group)
                .expect("validated execution order contains a known group");
            let mut dependencies = metadata
                .vector(dependency_slots.len())
                .map_err(LayerwiseRuntimeError::Architecture)?;
            dependencies.extend(dependency_slots.iter().filter_map(|&dependency| {
                if invocation.is_inactive_dependency(dependency) {
                    None
                } else {
                    Some(
                        outputs[dependency]
                            .as_ref()
                            .expect("active topological dependency has completed")
                            .clone(),
                    )
                }
            }));
            let mut dependency_refs = metadata
                .vector(dependencies.len())
                .map_err(LayerwiseRuntimeError::Architecture)?;
            dependency_refs.extend(dependencies.iter());
            let mut hidden = self
                .architecture
                .begin_execution_group(
                    group,
                    &initial,
                    &dependency_refs,
                    state,
                    &mut forward_context,
                    executor,
                )
                .map_err(LayerwiseRuntimeError::Architecture)?;
            hook.after_group_begin(group, &mut hidden, &mut forward_context, executor)
                .map_err(LayerwiseRuntimeError::Architecture)?;
            schedule
                .started_with_release(group, |dependency| outputs[dependency] = None)
                .expect("topological execution starts only ready groups");
            let active = self
                .architecture
                .should_execute_group(group, &forward_context);
            if active {
                let unit_count = layout
                    .group_range(group)
                    .expect("layout covers every graph group")
                    .len();
                for index in 0..unit_count {
                    if hook
                        .before_unit(
                            group,
                            index,
                            unit_count - index,
                            &mut hidden,
                            &mut forward_context,
                            executor,
                        )
                        .map_err(LayerwiseRuntimeError::Architecture)?
                        == LayeredUnitAction::SkipRemainingGroup
                    {
                        break;
                    }
                    let ordinal = layout
                        .ordinal(group, index)
                        .expect("group-local unit belongs to the layout");
                    let address = layout
                        .address(ordinal)
                        .expect("group-local unit has a stable policy address");
                    let lease = policy
                        .acquire(ordinal, address, |executor| {
                            self.architecture.build_unit(group, index, executor)
                        })
                        .map_err(|error| match error {
                            LayerwiseAcquireError::Architecture(error) => {
                                LayerwiseRuntimeError::Architecture(error)
                            }
                            LayerwiseAcquireError::Policy(error) => {
                                LayerwiseRuntimeError::Policy(error)
                            }
                        })?;
                    hidden = if observe_unit_internals && hook.observes_activations() {
                        self.architecture.forward_unit_observed(
                            group,
                            index,
                            lease,
                            &hidden,
                            state,
                            &mut forward_context,
                            executor,
                            &mut TraversalActivationObserver {
                                hook,
                                types: std::marker::PhantomData,
                            },
                        )
                    } else {
                        execute(
                            &mut self.architecture,
                            group,
                            index,
                            lease,
                            &hidden,
                            state,
                            &mut forward_context,
                            executor,
                            hook,
                        )
                    }
                    .map_err(LayerwiseRuntimeError::Architecture)?;
                    hook.after_unit(group, index, &mut hidden, &mut forward_context, executor)
                        .map_err(LayerwiseRuntimeError::Architecture)?;
                    metadata
                        .controls::<(Vec<&B::Tensor>, Option<usize>)>()
                        .map_err(LayerwiseRuntimeError::Architecture)?;
                    let mut state_values = metadata
                        .vector(0)
                        .map_err(LayerwiseRuntimeError::Architecture)?;
                    for state_ordinal in self
                        .architecture
                        .retained_state_ordinals(group, index, ordinal)
                    {
                        let address = address.with_index(state_ordinal);
                        if metadata.context().is_some() {
                            let mut count = Some(0usize);
                            state
                                .visit_unit_retained_values(state_ordinal, address, &mut |_| {
                                    count = count.and_then(|count| count.checked_add(1));
                                })
                                .map_err(LayerwiseRuntimeError::State)?;
                            let count = count.ok_or_else(|| {
                                LayerwiseRuntimeError::Architecture(metadata.map(
                                    eredu_nn::workspace::WorkspaceMetadataError::Overflow.into(),
                                ))
                            })?;
                            metadata
                                .reserve(&mut state_values, count)
                                .map_err(LayerwiseRuntimeError::Architecture)?;
                            let mut failure = None;
                            state
                                .visit_unit_retained_values(state_ordinal, address, &mut |value| {
                                    if failure.is_none() {
                                        failure = metadata.push(&mut state_values, value).err();
                                    }
                                })
                                .map_err(LayerwiseRuntimeError::State)?;
                            if let Some(error) = failure {
                                return Err(LayerwiseRuntimeError::Architecture(error));
                            }
                        } else {
                            state_values.extend(
                                state
                                    .retained_values(state_ordinal, address)
                                    .map_err(LayerwiseRuntimeError::State)?,
                            );
                        }
                    }
                    let architecture = &self.architecture;
                    let retained_forward = &forward_context;
                    let context_values = metadata
                        .retained(|visitor| {
                            architecture.visit_retained_context_values(
                                retained_forward,
                                group,
                                index,
                                visitor,
                            );
                        })
                        .map_err(LayerwiseRuntimeError::Architecture)?;
                    policy
                        .complete(
                            &hidden,
                            state_values.into_iter(),
                            context_values.into_iter(),
                        )
                        .map_err(LayerwiseRuntimeError::Policy)?;
                }
            }
            hidden = self
                .architecture
                .complete_execution_group(group, &hidden, state, &mut forward_context, executor)
                .map_err(LayerwiseRuntimeError::Architecture)?;
            hook.after_group(group, &mut hidden, &mut forward_context, executor)
                .map_err(LayerwiseRuntimeError::Architecture)?;
            if active {
                invocation.after_group(group, &hidden);
            } else {
                invocation.after_inactive_group(group);
            }
            outputs[group] = Some(hidden);
            if graph.groups().len() > 1 {
                completions[group] = Some(
                    policy
                        .policy
                        .submit_group(
                            executor,
                            outputs[group]
                                .as_ref()
                                .expect("group output was stored before submission"),
                        )
                        .map_err(|error| {
                            LayerwiseRuntimeError::Submission(
                                eredu_core::BackendFailure::from_error(error),
                            )
                        })?,
                );
            }
            schedule
                .ordered(group)
                .expect("started group can be ordered exactly once");
        }
        let hidden = outputs[graph.output()]
            .take()
            .expect("validated graph output completed");
        if let Some(completion) = &completions[graph.output()] {
            completion.order_after(context).map_err(|error| {
                LayerwiseRuntimeError::Submission(eredu_core::BackendFailure::from_error(error))
            })?;
        }
        let selected_hidden = self
            .architecture
            .select_readout_positions(&hidden, &forward_context, demand, context)
            .map_err(LayerwiseRuntimeError::Architecture)?;
        let output = if let Some(selected_hidden) = selected_hidden {
            let output = if hook.observes_activations() {
                self.architecture.finish_forward_observed(
                    &selected_hidden,
                    state,
                    &forward_context,
                    context,
                    &mut TraversalActivationObserver {
                        hook,
                        types: std::marker::PhantomData,
                    },
                )
            } else {
                self.architecture
                    .finish_forward(&selected_hidden, state, &forward_context, context)
            }
            .map_err(LayerwiseRuntimeError::Architecture)?;
            Some(output)
        } else {
            None
        };
        policy
            .finish(output.as_ref().unwrap_or(&hidden))
            .map_err(LayerwiseRuntimeError::Policy)?;
        // The readout and policy are complete. Consume each actual boundary
        // owner before returning or allowing the next bounded span. If one
        // fails, the iterator retains remaining owners on their safe Drop path.
        for completion in completions.into_iter().flatten() {
            completion.finish().map_err(|error| {
                LayerwiseRuntimeError::Submission(eredu_core::BackendFailure::from_error(error))
            })?;
        }
        if let Some(completion) = initial_completion {
            completion.finish().map_err(|error| {
                LayerwiseRuntimeError::Submission(eredu_core::BackendFailure::from_error(error))
            })?;
        }
        Ok((output, forward_context))
    }

    /// Runs one complete rank-local pass through the neutral parallel lifecycle.
    pub fn forward_parallel<'a>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<B::Tensor, LayerwiseRuntimeError<A::Error, P::Error>>
    where
        A: ParallelLayeredArchitecture<B, S>,
    {
        self.forward_parallel_fixed_with_readout(
            input,
            state,
            parallel,
            context,
            eredu_core::OutputDemand::Sequence,
        )
        .map(|(output, _)| output.expect("sequence readout returns scores"))
    }

    /// Runs one rank-local pass and exposes mutable context after each unit.
    pub fn forward_parallel_with_context_hook<'a, H>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        hook: H,
    ) -> Result<(B::Tensor, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        A: ParallelLayeredArchitecture<B, S>,
        H: FnMut(usize, usize, &mut A::ForwardContext) -> Result<(), A::Error>,
    {
        self.forward_parallel_with_unit_executor_and_traversal_hook(
            input,
            state,
            parallel,
            context,
            |architecture, group, index, unit, hidden, state, forward, parallel, context| {
                architecture.forward_unit_parallel(
                    group, index, unit, hidden, state, forward, parallel, context,
                )
            },
            &mut AfterUnitContextTraversalHook { after_unit: hook },
        )
    }

    /// Runs one parallel pass with a custom statically dispatched unit executor.
    pub fn forward_parallel_with_unit_executor<'a, E>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        execute: E,
    ) -> Result<B::Tensor, LayerwiseRuntimeError<A::Error, P::Error>>
    where
        A: ParallelLayeredArchitecture<B, S>,
        E: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Unit,
            &B::Tensor,
            &mut S,
            &mut A::ForwardContext,
            &B::ParallelContext,
            &<B::Tensor as eredu_nn::Tensor>::Context,
        ) -> Result<B::Tensor, A::Error>,
    {
        self.forward_parallel_with_unit_executor_and_context_hook(
            input,
            state,
            parallel,
            context,
            execute,
            |_, _, _| Ok(()),
        )
        .map(|(output, _)| output)
    }

    /// Runs the production parallel traversal with stable unit-boundary observation.
    pub fn forward_parallel_with_observer<'a, Observer>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut Observer,
    ) -> Result<B::Tensor, LayerwiseRuntimeError<A::Error, P::Error>>
    where
        A: ParallelLayeredArchitecture<B, S>,
        Observer: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.forward_parallel_with_observer_and_context(input, state, parallel, context, observer)
            .map(|(output, _)| output)
    }

    /// Observes parallel unit internals through the ordinary residency traversal,
    /// retaining its forward resources for the caller's completion boundary.
    pub fn forward_parallel_with_observer_and_context<'a, Observer>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut Observer,
    ) -> Result<(B::Tensor, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        A: ParallelLayeredArchitecture<B, S>,
        Observer: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.forward_parallel_with_observer_and_context_with_readout(
            input,
            state,
            parallel,
            context,
            observer,
            eredu_core::OutputDemand::Sequence,
        )
        .map(|(output, forward)| (output.expect("sequence readout returns scores"), forward))
    }

    /// Runs the fixed ordinary parallel equations without an observer or a
    /// caller-supplied unit/context mutation hook. Its retained geometry remains
    /// valid across spans, just as for the fixed observed entry.
    pub fn forward_parallel_fixed_with_readout<'a>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        demand: eredu_core::OutputDemand,
    ) -> Result<(Option<B::Tensor>, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        A: ParallelLayeredArchitecture<B, S>,
    {
        // `true` preserves the checked geometry and observation binding. The no-op
        // hook observes no activations, so the worker calls only the ordinary numerical
        // input/unit/readout methods and creates no observation path owner.
        self.forward_parallel_with_unit_executor_and_traversal_hook_impl(
            input,
            state,
            parallel,
            context,
            |architecture, group, index, unit, hidden, state, forward, parallel, context| {
                architecture.forward_unit_parallel(
                    group, index, unit, hidden, state, forward, parallel, context,
                )
            },
            &mut NoopLayeredTraversalHook,
            true,
            demand,
        )
    }

    /// Runs the same traversal with explicit vocabulary output demand.
    pub fn forward_parallel_with_observer_and_context_with_readout<'a, Observer>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut Observer,
        demand: eredu_core::OutputDemand,
    ) -> Result<(Option<B::Tensor>, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        A: ParallelLayeredArchitecture<B, S>,
        Observer: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.forward_parallel_invocation_with_internal_observer(
            OrdinaryLayeredInput::new(input),
            state,
            parallel,
            context,
            |architecture,
             group,
             index,
             unit,
             hidden,
             state,
             forward,
             parallel,
             context,
             observer| {
                architecture.forward_unit_parallel_observed(
                    group, index, unit, hidden, state, forward, parallel, context, observer,
                )
            },
            observer,
            demand,
            true,
        )
    }

    /// Runs a caller-selected parallel unit through shared input, group, unit and readout hooks.
    pub fn forward_parallel_with_internal_observer_and_context<'a, E, Observer>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        execute: E,
        observer: &mut Observer,
    ) -> Result<(B::Tensor, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        A: ParallelLayeredArchitecture<B, S>,
        E: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Unit,
            &B::Tensor,
            &mut S,
            &mut A::ForwardContext,
            &B::ParallelContext,
            &<B::Tensor as eredu_nn::Tensor>::Context,
            &mut Observer,
        ) -> Result<B::Tensor, A::Error>,
        Observer: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.forward_parallel_with_internal_observer_and_context_with_readout(
            input,
            state,
            parallel,
            context,
            execute,
            observer,
            eredu_core::OutputDemand::Sequence,
        )
        .map(|(output, forward)| (output.expect("sequence readout returns scores"), forward))
    }

    /// Runs the same traversal with explicit vocabulary output demand.
    pub fn forward_parallel_with_internal_observer_and_context_with_readout<'a, E, Observer>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        execute: E,
        observer: &mut Observer,
        demand: eredu_core::OutputDemand,
    ) -> Result<(Option<B::Tensor>, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        A: ParallelLayeredArchitecture<B, S>,
        E: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Unit,
            &B::Tensor,
            &mut S,
            &mut A::ForwardContext,
            &B::ParallelContext,
            &<B::Tensor as eredu_nn::Tensor>::Context,
            &mut Observer,
        ) -> Result<B::Tensor, A::Error>,
        Observer: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.forward_parallel_invocation_with_internal_observer(
            OrdinaryLayeredInput::new(input),
            state,
            parallel,
            context,
            execute,
            observer,
            demand,
            false,
        )
    }

    pub(crate) fn forward_parallel_invocation_with_internal_observer<I, E, Observer>(
        &mut self,
        invocation: I,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        mut execute: E,
        observer: &mut Observer,
        demand: eredu_core::OutputDemand,
        retained_unit_selection: bool,
    ) -> Result<(Option<B::Tensor>, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        I: LayeredInvocation<A, B, S>,
        A: ParallelLayeredArchitecture<B, S>,
        E: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Unit,
            &B::Tensor,
            &mut S,
            &mut A::ForwardContext,
            &B::ParallelContext,
            &<B::Tensor as eredu_nn::Tensor>::Context,
            &mut Observer,
        ) -> Result<B::Tensor, A::Error>,
        Observer: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        if self.geometry_stale {
            return Err(crate::ExecutionUnitLayoutError::StalePreparedGeometry.into());
        }
        let ordinary_graph;
        let graph = match self.prepared_geometry.as_ref() {
            Some(geometry) => geometry.graph(),
            None => {
                ordinary_graph = self
                    .architecture
                    .execution_graph()
                    .map_err(LayerwiseRuntimeError::Architecture)?
                    .into_owned();
                &ordinary_graph
            }
        };
        let mut units = Vec::with_capacity(graph.groups().len());
        let mut group_inputs = Vec::with_capacity(graph.groups().len());
        let mut group_outputs = Vec::with_capacity(graph.groups().len());
        for group in 0..graph.groups().len() {
            let count = self
                .architecture
                .group_unit_count(group, None)
                .map_err(LayerwiseRuntimeError::Architecture)?;
            units.push(
                (0..count)
                    .map(|index| {
                        if self.architecture.observes_unit_boundaries(group, index) {
                            Ok(None)
                        } else {
                            self.architecture.unit_path(group, index, None).map(Some)
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(LayerwiseRuntimeError::Architecture)?,
            );
            group_inputs.push(
                self.architecture
                    .group_input_observation_path(group, None)
                    .map_err(LayerwiseRuntimeError::Architecture)?,
            );
            group_outputs.push(
                self.architecture
                    .group_output_observation_path(group, None)
                    .map_err(LayerwiseRuntimeError::Architecture)?,
            );
        }
        let observer = std::rc::Rc::new(std::cell::RefCell::new(observer));
        let internal_observer = observer.clone();
        let mut hook = ActivationObserverTraversalHook {
            observer,
            units,
            group_inputs,
            group_outputs,
        };
        self.forward_parallel_with_unit_executor_and_invocation_hook(
            invocation,
            state,
            parallel,
            context,
            |architecture, group, index, unit, hidden, state, forward, parallel, context, _hook| {
                execute(
                    architecture,
                    group,
                    index,
                    unit,
                    hidden,
                    state,
                    forward,
                    parallel,
                    context,
                    &mut **internal_observer.borrow_mut(),
                )
            },
            &mut hook,
            false,
            retained_unit_selection,
            demand,
        )
    }

    /// Runs a custom parallel unit executor with stable boundary observation.
    pub fn forward_parallel_with_unit_executor_and_observer<'a, E, Observer>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        mut execute: E,
        observer: &mut Observer,
    ) -> Result<B::Tensor, LayerwiseRuntimeError<A::Error, P::Error>>
    where
        A: ParallelLayeredArchitecture<B, S>,
        E: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Unit,
            &B::Tensor,
            &mut S,
            &mut A::ForwardContext,
            &B::ParallelContext,
            &<B::Tensor as eredu_nn::Tensor>::Context,
        ) -> Result<B::Tensor, A::Error>,
        Observer: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.forward_parallel_with_unit_executor(
            input,
            state,
            parallel,
            context,
            |architecture, group, index, unit, hidden, state, forward, parallel, context| {
                let path = architecture.unit_path(group, index, None)?;
                let input = observe_outer_boundary(
                    observer,
                    &format!("{path}.input"),
                    &format!("{path}.input.effective"),
                    hidden,
                )?;
                let output = execute(
                    architecture,
                    group,
                    index,
                    unit,
                    &input,
                    state,
                    forward,
                    parallel,
                    context,
                )?;
                observe_outer_boundary(
                    observer,
                    &format!("{path}.output"),
                    &format!("{path}.output.effective"),
                    &output,
                )
            },
        )
    }

    /// Runs provider-backed parallel execution with boundary and routing observation.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_parallel_with_provider_and_observer<'a, Provider, Observer>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        pass: ExpertPass,
        provider: &mut Provider,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut Observer,
    ) -> Result<B::Tensor, LayerwiseRuntimeError<A::Error, P::Error>>
    where
        B: eredu_nn::GroupedNeuralBackend,
        A: ParallelRoutedLayeredArchitecture<B, S>,
        A::Error: std::fmt::Display,
        Provider: crate::TensorParallelRoutedExpertProvider<B>,
        Provider::Error: std::fmt::Display,
        Observer: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.forward_parallel_invocation_with_internal_observer(
            OrdinaryLayeredInput::new(input),
            state,
            parallel,
            context,
            |architecture,
             group,
             index,
             unit,
             hidden,
             state,
             forward,
             parallel,
             context,
             observer| {
                architecture.forward_unit_parallel_observed_with_provider(
                    group, index, unit, hidden, state, forward, pass, provider, parallel, context,
                    observer,
                )
            },
            observer,
            eredu_core::OutputDemand::Sequence,
            true,
        )
        .map(|(output, _)| output.expect("sequence readout returns scores"))
    }

    /// Runs one parallel pass with custom unit execution and a post-unit hook.
    pub fn forward_parallel_with_unit_executor_and_context_hook<'a, E, H>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        execute: E,
        hook: H,
    ) -> Result<(B::Tensor, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        A: ParallelLayeredArchitecture<B, S>,
        E: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Unit,
            &B::Tensor,
            &mut S,
            &mut A::ForwardContext,
            &B::ParallelContext,
            &<B::Tensor as eredu_nn::Tensor>::Context,
        ) -> Result<B::Tensor, A::Error>,
        H: FnMut(usize, usize, &mut A::ForwardContext) -> Result<(), A::Error>,
    {
        self.forward_parallel_with_unit_executor_and_traversal_hook(
            input,
            state,
            parallel,
            context,
            execute,
            &mut AfterUnitContextTraversalHook { after_unit: hook },
        )
    }

    /// Runs one parallel pass through a statically dispatched traversal hook.
    pub fn forward_parallel_with_traversal_hook<'a, H>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        hook: &mut H,
    ) -> Result<(B::Tensor, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        A: ParallelLayeredArchitecture<B, S>,
        H: LayeredTraversalHook<B, A::ForwardContext, A::Error> + ?Sized,
    {
        self.forward_parallel_with_traversal_hook_with_readout(
            input,
            state,
            parallel,
            context,
            hook,
            eredu_core::OutputDemand::Sequence,
        )
        .map(|(output, forward)| (output.expect("sequence readout returns scores"), forward))
    }

    /// Runs the same traversal with explicit vocabulary output demand.
    pub fn forward_parallel_with_traversal_hook_with_readout<'a, H>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        hook: &mut H,
        demand: eredu_core::OutputDemand,
    ) -> Result<(Option<B::Tensor>, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        A: ParallelLayeredArchitecture<B, S>,
        H: LayeredTraversalHook<B, A::ForwardContext, A::Error> + ?Sized,
    {
        self.forward_parallel_with_unit_executor_and_traversal_hook_impl(
            input,
            state,
            parallel,
            context,
            |architecture, group, index, unit, hidden, state, forward, parallel, context| {
                architecture.forward_unit_parallel(
                    group, index, unit, hidden, state, forward, parallel, context,
                )
            },
            hook,
            true,
            demand,
        )
    }

    /// Runs one parallel pass with custom unit execution and a shared traversal hook.
    pub fn forward_parallel_with_unit_executor_and_traversal_hook<'a, E, H>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        execute: E,
        hook: &mut H,
    ) -> Result<(B::Tensor, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        A: ParallelLayeredArchitecture<B, S>,
        E: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Unit,
            &B::Tensor,
            &mut S,
            &mut A::ForwardContext,
            &B::ParallelContext,
            &<B::Tensor as eredu_nn::Tensor>::Context,
        ) -> Result<B::Tensor, A::Error>,
        H: LayeredTraversalHook<B, A::ForwardContext, A::Error> + ?Sized,
    {
        self.forward_parallel_with_unit_executor_and_traversal_hook_with_readout(
            input,
            state,
            parallel,
            context,
            execute,
            hook,
            eredu_core::OutputDemand::Sequence,
        )
        .map(|(output, forward)| (output.expect("sequence readout returns scores"), forward))
    }

    /// Runs the same traversal with explicit vocabulary output demand.
    pub fn forward_parallel_with_unit_executor_and_traversal_hook_with_readout<'a, E, H>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        execute: E,
        hook: &mut H,
        demand: eredu_core::OutputDemand,
    ) -> Result<(Option<B::Tensor>, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        A: ParallelLayeredArchitecture<B, S>,
        E: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Unit,
            &B::Tensor,
            &mut S,
            &mut A::ForwardContext,
            &B::ParallelContext,
            &<B::Tensor as eredu_nn::Tensor>::Context,
        ) -> Result<B::Tensor, A::Error>,
        H: LayeredTraversalHook<B, A::ForwardContext, A::Error> + ?Sized,
    {
        self.forward_parallel_with_unit_executor_and_traversal_hook_impl(
            input, state, parallel, context, execute, hook, false, demand,
        )
    }

    fn forward_parallel_with_unit_executor_and_traversal_hook_impl<'a, E, H>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        execute: E,
        hook: &mut H,
        observe_unit_internals: bool,
        demand: eredu_core::OutputDemand,
    ) -> Result<(Option<B::Tensor>, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        A: ParallelLayeredArchitecture<B, S>,
        E: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Unit,
            &B::Tensor,
            &mut S,
            &mut A::ForwardContext,
            &B::ParallelContext,
            &<B::Tensor as eredu_nn::Tensor>::Context,
        ) -> Result<B::Tensor, A::Error>,
        H: LayeredTraversalHook<B, A::ForwardContext, A::Error> + ?Sized,
    {
        self.forward_parallel_with_unit_executor_and_invocation(
            OrdinaryLayeredInput::new(input),
            state,
            parallel,
            context,
            execute,
            hook,
            observe_unit_internals,
            demand,
        )
    }

    pub(crate) fn forward_parallel_with_unit_executor_and_invocation<I, E, H>(
        &mut self,
        invocation: I,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        mut execute: E,
        hook: &mut H,
        observe_unit_internals: bool,
        demand: eredu_core::OutputDemand,
    ) -> Result<(Option<B::Tensor>, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        I: LayeredInvocation<A, B, S>,
        A: ParallelLayeredArchitecture<B, S>,
        E: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Unit,
            &B::Tensor,
            &mut S,
            &mut A::ForwardContext,
            &B::ParallelContext,
            &<B::Tensor as eredu_nn::Tensor>::Context,
        ) -> Result<B::Tensor, A::Error>,
        H: LayeredTraversalHook<B, A::ForwardContext, A::Error> + ?Sized,
    {
        self.forward_parallel_with_unit_executor_and_invocation_hook(
            invocation,
            state,
            parallel,
            context,
            |architecture, group, index, unit, hidden, state, forward, parallel, context, _hook| {
                execute(
                    architecture,
                    group,
                    index,
                    unit,
                    hidden,
                    state,
                    forward,
                    parallel,
                    context,
                )
            },
            hook,
            observe_unit_internals,
            false,
            demand,
        )
    }

    pub(crate) fn forward_parallel_with_unit_executor_and_invocation_hook<I, E, H>(
        &mut self,
        mut invocation: I,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        mut execute: E,
        hook: &mut H,
        observe_unit_internals: bool,
        retained_unit_selection: bool,
        demand: eredu_core::OutputDemand,
    ) -> Result<(Option<B::Tensor>, A::ForwardContext), LayerwiseRuntimeError<A::Error, P::Error>>
    where
        I: LayeredInvocation<A, B, S>,
        A: ParallelLayeredArchitecture<B, S>,
        E: FnMut(
            &mut A,
            usize,
            usize,
            &mut A::Unit,
            &B::Tensor,
            &mut S,
            &mut A::ForwardContext,
            &B::ParallelContext,
            &<B::Tensor as eredu_nn::Tensor>::Context,
            &mut H,
        ) -> Result<B::Tensor, A::Error>,
        H: LayeredTraversalHook<B, A::ForwardContext, A::Error> + ?Sized,
    {
        if !observe_unit_internals && !retained_unit_selection {
            self.observation_binding.invalidate();
        }

        if self.geometry_stale {
            return Err(crate::ExecutionUnitLayoutError::StalePreparedGeometry.into());
        }
        let ordinary_geometry;
        let (graph, layout) = match self.prepared_geometry.as_ref() {
            Some(geometry) => (geometry.graph(), geometry.units()),
            None => {
                let graph = self
                    .architecture
                    .execution_graph()
                    .map_err(LayerwiseRuntimeError::Architecture)?
                    .into_owned();
                let counts = (0..graph.groups().len())
                    .map(|group| {
                        self.architecture
                            .group_unit_count(group, None)
                            .map_err(LayerwiseRuntimeError::Architecture)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let layout = ExecutionUnitLayout::new(&graph, counts)?;
                ordinary_geometry = (graph, layout);
                (&ordinary_geometry.0, &ordinary_geometry.1)
            }
        };
        if !observe_unit_internals && !retained_unit_selection {
            self.geometry_stale |= self.prepared_geometry.is_some();
        }
        let shared_executor = self
            .policy
            .uses_shared_group_executor(context)
            .map_err(LayerwiseRuntimeError::Policy)?;
        if !shared_executor && self.executors.as_ref().map(Vec::len) != Some(graph.groups().len()) {
            self.executors = Some(B::fork_executors(context, graph.groups().len()).map_err(
                |error| {
                    LayerwiseRuntimeError::Submission(eredu_core::BackendFailure::from_error(error))
                },
            )?);
        }
        let executors = self.executors.as_ref();
        let forward = invocation
            .begin_parallel(&mut self.architecture, state, parallel, context, hook)
            .map_err(LayerwiseRuntimeError::Architecture)?;
        let metadata = metadata::Destination(<A as LayeredArchitecture<B, S>>::forward_metadata(
            &self.architecture,
            &forward.context,
        ));
        if retained_unit_selection {
            metadata
                .controls::<(
                    E,
                    E,
                    &mut H,
                    bool,
                    Result<
                        (Option<B::Tensor>, A::ForwardContext),
                        LayerwiseRuntimeError<A::Error, P::Error>,
                    >,
                )>()
                .map_err(LayerwiseRuntimeError::Architecture)?;
        }
        metadata
            .controls::<(
                Vec<Option<B::Tensor>>,
                Vec<B::Tensor>,
                Vec<&B::Tensor>,
                metadata::Destination<A::Error>,
            )>()
            .map_err(LayerwiseRuntimeError::Architecture)?;
        let initial_completion = (graph.groups().len() > 1)
            .then(|| self.policy.submit_group(context, &forward.hidden))
            .transpose()
            .map_err(|error| {
                LayerwiseRuntimeError::Submission(eredu_core::BackendFailure::from_error(error))
            })?;
        let mut policy = LayerwisePolicyForward::begin(&mut self.policy, &forward.hidden, context)
            .map_err(LayerwiseRuntimeError::Policy)?;
        let mut initial = forward.hidden;
        let mut forward_context = forward.context;
        self.architecture
            .set_readout_demand(&mut forward_context, demand);
        let mut schedule = metadata
            .schedule(&graph)
            .map_err(LayerwiseRuntimeError::Architecture)?;
        let mut outputs: Vec<Option<B::Tensor>> = metadata
            .vector(graph.groups().len())
            .map_err(LayerwiseRuntimeError::Architecture)?;
        outputs.resize_with(graph.groups().len(), || None);
        let mut completions = metadata
            .completions(&initial_completion, graph.groups().len())
            .map_err(LayerwiseRuntimeError::Architecture)?;
        for &group in graph.execution_order() {
            if let Some(retained) = invocation.retained_group(group) {
                schedule
                    .started_with_release(group, |dependency| outputs[dependency] = None)
                    .expect("retained group is dependency ready");
                outputs[group] = retained;
                schedule
                    .ordered(group)
                    .expect("retained group is ordered once");
                continue;
            }
            let executor = if shared_executor {
                context
            } else {
                std::borrow::Borrow::borrow(
                    &executors.expect("layered runtime initialized its executor cache")[group],
                )
            };
            let group_dependencies = schedule
                .dependencies(group)
                .expect("validated execution order contains a known group");
            if group_dependencies.is_empty() {
                if let Some(completion) = &initial_completion {
                    completion.order_after(executor).map_err(|error| {
                        LayerwiseRuntimeError::Submission(eredu_core::BackendFailure::from_error(
                            error,
                        ))
                    })?;
                }
            }
            for &dependency in group_dependencies {
                if let Some(completion) = completions[dependency].as_ref() {
                    completion.order_after(executor).map_err(|error| {
                        LayerwiseRuntimeError::Submission(eredu_core::BackendFailure::from_error(
                            error,
                        ))
                    })?;
                } else {
                    debug_assert!(
                        invocation.is_retained_group(dependency),
                        "only completed retained ingress omits a new completion"
                    );
                }
            }
            if invocation
                .before_group_with_hook(
                    &mut self.architecture,
                    group,
                    &mut initial,
                    &mut forward_context,
                    state,
                    Some(parallel),
                    executor,
                    hook,
                )
                .map_err(LayerwiseRuntimeError::Architecture)?
            {
                self.architecture
                    .set_readout_demand(&mut forward_context, demand);
            }
            let dependency_slots = schedule
                .dependencies(group)
                .expect("validated execution order contains a known group");
            let mut dependencies = metadata
                .vector(dependency_slots.len())
                .map_err(LayerwiseRuntimeError::Architecture)?;
            dependencies.extend(dependency_slots.iter().filter_map(|&dependency| {
                if invocation.is_inactive_dependency(dependency) {
                    None
                } else {
                    Some(
                        outputs[dependency]
                            .as_ref()
                            .expect("active topological dependency has completed")
                            .clone(),
                    )
                }
            }));
            let mut dependency_refs = metadata
                .vector(dependencies.len())
                .map_err(LayerwiseRuntimeError::Architecture)?;
            dependency_refs.extend(dependencies.iter());
            let mut hidden = self
                .architecture
                .begin_execution_group_parallel(
                    group,
                    &initial,
                    &dependency_refs,
                    state,
                    &mut forward_context,
                    parallel,
                    executor,
                )
                .map_err(LayerwiseRuntimeError::Architecture)?;
            hook.after_group_begin(group, &mut hidden, &mut forward_context, executor)
                .map_err(LayerwiseRuntimeError::Architecture)?;
            schedule
                .started_with_release(group, |dependency| outputs[dependency] = None)
                .expect("topological execution starts only ready groups");
            let active = self
                .architecture
                .should_execute_group(group, &forward_context);
            if active {
                let unit_count = layout
                    .group_range(group)
                    .expect("layout covers every graph group")
                    .len();
                for index in 0..unit_count {
                    if hook
                        .before_unit(
                            group,
                            index,
                            unit_count - index,
                            &mut hidden,
                            &mut forward_context,
                            executor,
                        )
                        .map_err(LayerwiseRuntimeError::Architecture)?
                        == LayeredUnitAction::SkipRemainingGroup
                    {
                        break;
                    }
                    let ordinal = layout
                        .ordinal(group, index)
                        .expect("group-local unit belongs to the layout");
                    let address = layout
                        .address(ordinal)
                        .expect("group-local unit has a stable policy address");
                    let lease = policy
                        .acquire(ordinal, address, |executor| {
                            self.architecture.build_unit(group, index, executor)
                        })
                        .map_err(|error| match error {
                            LayerwiseAcquireError::Architecture(error) => {
                                LayerwiseRuntimeError::Architecture(error)
                            }
                            LayerwiseAcquireError::Policy(error) => {
                                LayerwiseRuntimeError::Policy(error)
                            }
                        })?;
                    hidden = if observe_unit_internals && hook.observes_activations() {
                        self.architecture.forward_unit_parallel_observed(
                            group,
                            index,
                            lease,
                            &hidden,
                            state,
                            &mut forward_context,
                            parallel,
                            executor,
                            &mut TraversalActivationObserver {
                                hook,
                                types: std::marker::PhantomData,
                            },
                        )
                    } else {
                        execute(
                            &mut self.architecture,
                            group,
                            index,
                            lease,
                            &hidden,
                            state,
                            &mut forward_context,
                            parallel,
                            executor,
                            hook,
                        )
                    }
                    .map_err(LayerwiseRuntimeError::Architecture)?;
                    hook.after_unit(group, index, &mut hidden, &mut forward_context, executor)
                        .map_err(LayerwiseRuntimeError::Architecture)?;
                    metadata
                        .controls::<(Vec<&B::Tensor>, Option<usize>)>()
                        .map_err(LayerwiseRuntimeError::Architecture)?;
                    let mut state_values = metadata
                        .vector(0)
                        .map_err(LayerwiseRuntimeError::Architecture)?;
                    for state_ordinal in self
                        .architecture
                        .retained_state_ordinals(group, index, ordinal)
                    {
                        let address = address.with_index(state_ordinal);
                        if metadata.context().is_some() {
                            let mut count = Some(0usize);
                            state
                                .visit_unit_retained_values(state_ordinal, address, &mut |_| {
                                    count = count.and_then(|count| count.checked_add(1));
                                })
                                .map_err(LayerwiseRuntimeError::State)?;
                            let count = count.ok_or_else(|| {
                                LayerwiseRuntimeError::Architecture(metadata.map(
                                    eredu_nn::workspace::WorkspaceMetadataError::Overflow.into(),
                                ))
                            })?;
                            metadata
                                .reserve(&mut state_values, count)
                                .map_err(LayerwiseRuntimeError::Architecture)?;
                            let mut failure = None;
                            state
                                .visit_unit_retained_values(state_ordinal, address, &mut |value| {
                                    if failure.is_none() {
                                        failure = metadata.push(&mut state_values, value).err();
                                    }
                                })
                                .map_err(LayerwiseRuntimeError::State)?;
                            if let Some(error) = failure {
                                return Err(LayerwiseRuntimeError::Architecture(error));
                            }
                        } else {
                            state_values.extend(
                                state
                                    .retained_values(state_ordinal, address)
                                    .map_err(LayerwiseRuntimeError::State)?,
                            );
                        }
                    }
                    let architecture = &self.architecture;
                    let retained_forward = &forward_context;
                    let context_values = metadata
                        .retained(|visitor| {
                            architecture.visit_retained_context_values(
                                retained_forward,
                                group,
                                index,
                                visitor,
                            );
                        })
                        .map_err(LayerwiseRuntimeError::Architecture)?;
                    policy
                        .complete(
                            &hidden,
                            state_values.into_iter(),
                            context_values.into_iter(),
                        )
                        .map_err(LayerwiseRuntimeError::Policy)?;
                }
            }
            hidden = self
                .architecture
                .complete_execution_group_parallel(
                    group,
                    &hidden,
                    state,
                    &mut forward_context,
                    parallel,
                    executor,
                )
                .map_err(LayerwiseRuntimeError::Architecture)?;
            hook.after_group(group, &mut hidden, &mut forward_context, executor)
                .map_err(LayerwiseRuntimeError::Architecture)?;
            if active {
                invocation.after_group(group, &hidden);
            } else {
                invocation.after_inactive_group(group);
            }
            outputs[group] = Some(hidden);
            if graph.groups().len() > 1 {
                completions[group] = Some(
                    policy
                        .policy
                        .submit_group(
                            executor,
                            outputs[group]
                                .as_ref()
                                .expect("group output was stored before submission"),
                        )
                        .map_err(|error| {
                            LayerwiseRuntimeError::Submission(
                                eredu_core::BackendFailure::from_error(error),
                            )
                        })?,
                );
            }
            schedule
                .ordered(group)
                .expect("started group can be ordered exactly once");
        }
        let hidden = outputs[graph.output()]
            .take()
            .expect("validated graph output completed");
        if let Some(completion) = &completions[graph.output()] {
            completion.order_after(context).map_err(|error| {
                LayerwiseRuntimeError::Submission(eredu_core::BackendFailure::from_error(error))
            })?;
        }
        let selected_hidden = self
            .architecture
            .select_readout_positions(&hidden, &forward_context, demand, context)
            .map_err(LayerwiseRuntimeError::Architecture)?;
        let output = if let Some(selected_hidden) = selected_hidden {
            let output = if hook.observes_activations() {
                self.architecture.finish_forward_parallel_observed(
                    &selected_hidden,
                    state,
                    &forward_context,
                    parallel,
                    context,
                    &mut TraversalActivationObserver {
                        hook,
                        types: std::marker::PhantomData,
                    },
                )
            } else {
                self.architecture.finish_forward_parallel(
                    &selected_hidden,
                    state,
                    &forward_context,
                    parallel,
                    context,
                )
            }
            .map_err(LayerwiseRuntimeError::Architecture)?;
            Some(output)
        } else {
            None
        };
        policy
            .finish(output.as_ref().unwrap_or(&hidden))
            .map_err(LayerwiseRuntimeError::Policy)?;
        // The readout and policy are complete. Consume each actual boundary
        // owner before returning or allowing the next bounded span. If one
        // fails, the iterator retains remaining owners on their safe Drop path.
        for completion in completions.into_iter().flatten() {
            completion.finish().map_err(|error| {
                LayerwiseRuntimeError::Submission(eredu_core::BackendFailure::from_error(error))
            })?;
        }
        if let Some(completion) = initial_completion {
            completion.finish().map_err(|error| {
                LayerwiseRuntimeError::Submission(eredu_core::BackendFailure::from_error(error))
            })?;
        }
        Ok((output, forward_context))
    }
}

/// Indexed owned unit used by [`ResidentUnitWindow`].
pub struct ResidentUnitLease<U> {
    index: usize,
    unit: U,
}

impl<U> std::ops::Deref for ResidentUnitLease<U> {
    type Target = U;

    fn deref(&self) -> &Self::Target {
        &self.unit
    }
}

impl<U> std::ops::DerefMut for ResidentUnitLease<U> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.unit
    }
}

/// Minimal one-at-a-time unit window useful for conformance and resident storage.
pub struct ResidentUnitWindow<U> {
    units: Vec<Option<U>>,
}

impl<U> ResidentUnitWindow<U> {
    /// Creates a window over an ordered set of already populated units.
    pub fn new(units: Vec<U>) -> Self {
        Self {
            units: units.into_iter().map(Some).collect(),
        }
    }
}

impl<B, U> LayerwisePolicy<B, U> for ResidentUnitWindow<U>
where
    B: NeuralBackend,
{
    type Lease = ResidentUnitLease<U>;
    type Error = ResidentUnitWindowError;

    fn retained_value_slot_bound(&self) -> Option<usize>
    where
        U: Parameterized<B::Tensor>,
    {
        self.units.iter().try_fold(0usize, |n, unit| {
            n.checked_add(unit.as_ref()?.retained_value_slot_bound()?)
        })
    }

    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&B::Tensor)) -> bool
    where
        U: Parameterized<B::Tensor>,
    {
        let mut complete = self.units.iter().all(Option::is_some);
        for unit in self.units.iter().flatten() {
            complete &= unit.visit_retained_values(visitor);
        }
        complete
    }

    fn begin(
        &mut self,
        _initial: &B::Tensor,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    fn abort(
        &mut self,
        active: Option<(usize, crate::ExecutionUnitAddress, Self::Lease)>,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) {
        let Some((ordinal, _, lease)) = active else {
            return;
        };
        debug_assert_eq!(lease.index, ordinal);
        if let Some(slot) = self.units.get_mut(lease.index) {
            debug_assert!(slot.is_none());
            if slot.is_none() {
                *slot = Some(lease.unit);
            }
        }
    }

    fn acquire<E, F>(
        &mut self,
        index: usize,
        _address: crate::ExecutionUnitAddress,
        _build: F,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<Self::Lease, LayerwiseAcquireError<E, Self::Error>>
    where
        F: FnOnce(&<B::Tensor as eredu_nn::Tensor>::Context) -> Result<U, E>,
    {
        let count = self.units.len();
        let unit = self
            .units
            .get_mut(index)
            .ok_or(ResidentUnitWindowError::UnknownUnit { index, count })
            .map_err(LayerwiseAcquireError::Policy)?
            .take()
            .ok_or(ResidentUnitWindowError::AlreadyAcquired { index })
            .map_err(LayerwiseAcquireError::Policy)?;
        Ok(ResidentUnitLease { index, unit })
    }

    fn complete<'a, StateValues, ContextValues>(
        &mut self,
        index: usize,
        _address: crate::ExecutionUnitAddress,
        lease: Self::Lease,
        _output: &'a B::Tensor,
        _state_values: StateValues,
        _context_values: ContextValues,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), Self::Error>
    where
        B::Tensor: 'a,
        StateValues: Iterator<Item = &'a B::Tensor>,
        ContextValues: Iterator<Item = &'a B::Tensor>,
    {
        if lease.index != index {
            return Err(ResidentUnitWindowError::MismatchedUnit {
                expected: index,
                actual: lease.index,
            });
        }
        let slot = self
            .units
            .get_mut(index)
            .expect("acquired unit index remains in the window");
        if slot.replace(lease.unit).is_some() {
            return Err(ResidentUnitWindowError::AlreadyResident { index });
        }
        Ok(())
    }

    fn finish(
        &mut self,
        _output: &B::Tensor,
        _context: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// Invalid access to an owned resident-unit window.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum ResidentUnitWindowError {
    /// The requested unit is outside the ordered window.
    #[error("unit {index} is outside the {count}-unit window")]
    UnknownUnit {
        /// Requested index.
        index: usize,
        /// Window size.
        count: usize,
    },
    /// A unit was acquired twice without an intervening completion.
    #[error("unit {index} is already acquired")]
    AlreadyAcquired {
        /// Requested index.
        index: usize,
    },
    /// Completion returned a lease for the wrong unit.
    #[error("unit completion expected {expected}, received {actual}")]
    MismatchedUnit {
        /// Expected unit.
        expected: usize,
        /// Lease unit.
        actual: usize,
    },
    /// A completion attempted to overwrite a resident unit.
    #[error("unit {index} is already resident")]
    AlreadyResident {
        /// Conflicting index.
        index: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::{ArchitectureGroupKind, LayeredPipelineSchedule, LayeredPipelineScheduleError};
    use crate::{ExecutionGraph, ExecutionGroupSpec, ExecutionScheduleError};

    fn pipeline_graph() -> ExecutionGraph {
        ExecutionGraph::new(
            vec![
                ExecutionGroupSpec::root("vision"),
                ExecutionGroupSpec::root("audio"),
                ExecutionGroupSpec::with_dependencies("projector", ["vision"]),
                ExecutionGroupSpec::with_dependencies("merge", ["projector", "audio"]),
                ExecutionGroupSpec::with_dependencies("decoder", ["merge"]),
                ExecutionGroupSpec::with_dependencies("prediction", ["decoder"]),
            ],
            "prediction",
        )
        .unwrap()
    }

    #[test]
    fn pipeline_schedule_owns_activity_propagation_and_ready_batches() {
        let graph = pipeline_graph();
        let contracts = [
            (ArchitectureGroupKind::VisionEncoder, true),
            (ArchitectureGroupKind::AudioEncoder, true),
            (ArchitectureGroupKind::Projector, false),
            (ArchitectureGroupKind::Merger, false),
            (ArchitectureGroupKind::Decoder, false),
            (ArchitectureGroupKind::Prediction, false),
        ];
        let mut queried = Vec::new();
        let mut schedule = LayeredPipelineSchedule::try_new(&graph, contracts, |group| {
            queried.push(group);
            Ok::<_, LayeredPipelineScheduleError>(group == 0)
        })
        .unwrap();

        assert_eq!(queried, [0, 1]);
        assert_eq!(schedule.activity(), [true, false, true, true, true, false]);
        assert_eq!(schedule.compatible_batch(|_, _| true), [0, 1]);
        schedule.started(0).unwrap();
        schedule.started(1).unwrap();
        schedule.ordered(0).unwrap();
        schedule.ordered(1).unwrap();
        assert_eq!(schedule.ready_groups().collect::<Vec<_>>(), [2]);
        for group in [2, 3, 4, 5] {
            schedule.started(group).unwrap();
            schedule.ordered(group).unwrap();
        }
        assert!(schedule.is_complete());
    }

    #[test]
    fn pipeline_schedule_rejects_kind_and_transition_drift() {
        let graph = pipeline_graph();
        let error = LayeredPipelineSchedule::try_new(
            &graph,
            [(ArchitectureGroupKind::Decoder, false)],
            |_| Ok::<_, LayeredPipelineScheduleError>(true),
        )
        .unwrap_err();
        assert_eq!(
            error,
            LayeredPipelineScheduleError::GroupContractCount {
                graph: 6,
                declared: 1,
            }
        );

        let contracts = [
            (ArchitectureGroupKind::VisionEncoder, true),
            (ArchitectureGroupKind::AudioEncoder, true),
            (ArchitectureGroupKind::Projector, false),
            (ArchitectureGroupKind::Merger, false),
            (ArchitectureGroupKind::Decoder, false),
            (ArchitectureGroupKind::Prediction, false),
        ];
        let mut schedule = LayeredPipelineSchedule::try_new(&graph, contracts, |_| {
            Ok::<_, LayeredPipelineScheduleError>(true)
        })
        .unwrap();
        assert_eq!(
            schedule.started(2),
            Err(LayeredPipelineScheduleError::Transition(
                ExecutionScheduleError::DependenciesPending { group: 2 }
            ))
        );
    }

    #[test]
    fn pipeline_schedule_consumes_declared_request_optionality() {
        let graph = ExecutionGraph::new(
            vec![
                ExecutionGroupSpec::root("mandatory_vision"),
                ExecutionGroupSpec::root("optional_audio"),
                ExecutionGroupSpec::with_dependencies(
                    "decoder",
                    ["mandatory_vision", "optional_audio"],
                ),
            ],
            "decoder",
        )
        .unwrap();
        let contracts = [
            (ArchitectureGroupKind::VisionEncoder, false),
            (ArchitectureGroupKind::AudioEncoder, true),
            (ArchitectureGroupKind::Decoder, false),
        ];
        let mut queried = Vec::new();
        let schedule = LayeredPipelineSchedule::try_new(&graph, contracts, |group| {
            queried.push(group);
            Ok::<_, LayeredPipelineScheduleError>(false)
        })
        .unwrap();

        assert_eq!(queried, [1]);
        assert_eq!(schedule.activity(), [true, false, true]);

        let invalid = [
            (ArchitectureGroupKind::VisionEncoder, false),
            (ArchitectureGroupKind::AudioEncoder, false),
            (ArchitectureGroupKind::Decoder, true),
        ];
        assert_eq!(
            LayeredPipelineSchedule::try_new(&graph, invalid, |_| {
                Ok::<_, LayeredPipelineScheduleError>(true)
            })
            .unwrap_err(),
            LayeredPipelineScheduleError::InvalidRequestOptionalGroup {
                group: 2,
                kind: ArchitectureGroupKind::Decoder,
            }
        );
    }
}
