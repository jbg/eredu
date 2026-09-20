//! Exact shared path payloads and runtime-bound borrowed ordinary hooks.
use super::*;
mod prefill;
use crate::host_metadata::{HostMetadataIdentity, MetadataCustody};
use eredu_core::{SharedStorageAttachmentError, SharedStorageDomain};
pub use prefill::{
    BoundCaptureSelection, PrefillObservationDeclaration, PrefillReadoutStage,
    PreparedCaptureSelection, PreparedCaptureSelectionError,
};
use std::{
    fmt,
    mem::size_of,
    sync::{Arc, OnceLock, Weak},
};

#[derive(Debug, PartialEq, Eq)]
struct UnitPaths {
    input: String,
    output: String,
    effective_input: String,
    effective_output: String,
    outer: bool,
}
#[derive(Debug, PartialEq, Eq)]
struct GroupPaths {
    id: String,
    units: Box<[UnitPaths]>,
    input: Option<String>,
    output: Option<String>,
}
struct PathsInner {
    payload: Box<[GroupPaths]>,
    prefill: Box<[PrefillObservationDeclaration]>,
    media_prefill: Box<[PrefillObservationDeclaration]>,
    #[cfg(test)]
    retired: Option<tests::PayloadRetired>,
    custody: MetadataCustody,
}

/// Immutable complete boundary paths produced by one actual layered architecture.
///
/// Construction belongs to loading/preparation before finite admission. The
/// actual owner must be registered and retain its source authority before it is
/// used by a managed request. Capacity is not allocation or execution permission.
/// Cloning shares all strings; no mutable or owning payload export is provided.
#[derive(Clone)]
pub struct SharedLayeredObservationPaths(Arc<PathsInner>);
impl SharedLayeredObservationPaths {
    fn collect<B, S, A>(architecture: &A) -> Result<Self, A::Error>
    where
        B: NeuralBackend,
        S: RuntimeState<B>,
        A: LayeredArchitecture<B, S>,
    {
        let graph = architecture.execution_graph()?;
        let mut groups = Vec::with_capacity(graph.group_count());
        for group in 0..graph.group_count() {
            let count = architecture.group_unit_count(group, None)?;
            let mut units = Vec::with_capacity(count);
            for index in 0..count {
                let unit = architecture.unit_path(group, index, None)?;
                let input = eredu_core::UnitObservation::Input.path(&unit);
                let output = eredu_core::UnitObservation::Output.path(&unit);
                units.push(UnitPaths {
                    effective_input: format!("{input}.effective"),
                    effective_output: format!("{output}.effective"),
                    input,
                    output,
                    outer: !architecture.observes_unit_boundaries(group, index),
                });
            }
            groups.push(GroupPaths {
                id: graph.group_id(group).expect("source group").to_owned(),
                units: units.into_boxed_slice(),
                input: architecture.group_input_observation_path(group, None)?,
                output: architecture.group_output_observation_path(group, None)?,
            });
        }
        Ok(Self(Arc::new(PathsInner {
            payload: groups.into_boxed_slice(),
            media_prefill: architecture
                .media_prefill_observation_declarations(None)?
                .into_boxed_slice(),
            prefill: architecture
                .prefill_observation_declarations(None)?
                .into_boxed_slice(),
            #[cfg(test)]
            retired: None,
            custody: MetadataCustody::new(),
        })))
    }

    fn validate<B, S, A>(
        &self,
        architecture: &A,
        destination: &metadata::Destination<A::Error>,
    ) -> Result<(), PreparedLayeredObservationError<A::Error>>
    where
        B: NeuralBackend,
        S: RuntimeState<B>,
        A: LayeredArchitecture<B, S>,
    {
        destination.controls::<(&Self, &A, &metadata::Destination<A::Error>, usize, usize,
            Vec<PrefillObservationDeclaration>, String, Option<String>,
            crate::ArchitectureExecutionGraph<'_>,
            std::iter::Enumerate<std::slice::Iter<'_, GroupPaths>>,
            std::iter::Enumerate<std::slice::Iter<'_, UnitPaths>>,
            Result<Vec<PrefillObservationDeclaration>, A::Error>,
            Result<String, A::Error>, Result<Option<String>, A::Error>,
            Result<(), PreparedLayeredObservationError<A::Error>>)>()
            .map_err(PreparedLayeredObservationError::Execution)?;
        if architecture
            .prefill_observation_declarations(destination.context())
            .map_err(PreparedLayeredObservationError::Execution)?
            .as_slice()
            != self.0.prefill.as_ref()
        {
            return Err(PreparedLayeredObservationError::SemanticMismatch);
        }
        if architecture
            .media_prefill_observation_declarations(destination.context())
            .map_err(PreparedLayeredObservationError::Execution)?
            .as_slice()
            != self.0.media_prefill.as_ref()
        {
            return Err(PreparedLayeredObservationError::SemanticMismatch);
        }
        let graph = architecture
            .execution_graph()
            .map_err(PreparedLayeredObservationError::Execution)?;
        if graph.group_count() != self.0.payload.len() {
            return Err(PreparedLayeredObservationError::SemanticMismatch);
        }
        for (group, retained) in self.0.payload.iter().enumerate()
        {
            let count = architecture
                .group_unit_count(group, destination.context())
                .map_err(PreparedLayeredObservationError::Execution)?;
            if graph.group_id(group) != Some(retained.id.as_str())
                || count != retained.units.len()
                || architecture
                    .group_input_observation_path(group, destination.context())
                    .map_err(PreparedLayeredObservationError::Execution)?
                    != retained.input
                || architecture
                    .group_output_observation_path(group, destination.context())
                    .map_err(PreparedLayeredObservationError::Execution)?
                    != retained.output
            {
                return Err(PreparedLayeredObservationError::SemanticMismatch);
            }
            for (index, retained) in retained.units.iter().enumerate() {
                let path = architecture
                    .unit_path(group, index, destination.context())
                    .map_err(PreparedLayeredObservationError::Execution)?;
                // Declaration temporaries belong to cold preparation. Every unit
                // is validated, including architecture-owned internal boundaries.
                if retained.outer == architecture.observes_unit_boundaries(group, index)
                    || !retained.input.strip_suffix(".input").is_some_and(|unit| unit == path)
                    || !retained.output.strip_suffix(".output").is_some_and(|unit| unit == path)
                    || retained.effective_input.strip_suffix(".effective") != Some(retained.input.as_str())
                    || retained.effective_output.strip_suffix(".effective") != Some(retained.output.as_str())
                {
                    return Err(PreparedLayeredObservationError::SemanticMismatch);
                }
            }
        }
        Ok(())
    }
    /// Payload-free identity used by the existing host metadata registry.
    pub fn identity(&self) -> &HostMetadataIdentity {
        self.0.custody.identity()
    }
    /// Exact retained inline payload, boxed entries and String capacities.
    /// Arc, custody and allocator bookkeeping are excluded. No allocation occurs.
    /// Construction temporaries are separate loading/preparation obligations.
    pub fn capacity_bytes(&self) -> Option<u64> {
        let mut bytes = extent::<Box<[GroupPaths]>>(1)?
            .checked_add(extent::<GroupPaths>(self.0.payload.len())?)?;
        for group in &self.0.payload {
            bytes = bytes
                .checked_add(group.id.capacity().try_into().ok()?)?
                .checked_add(extent::<UnitPaths>(group.units.len())?)?;
            for path in [group.input.as_ref(), group.output.as_ref()]
                .into_iter()
                .flatten()
            {
                bytes = bytes.checked_add(path.capacity().try_into().ok()?)?;
            }
            for unit in group.units.iter() {
                bytes = bytes
                    .checked_add(unit.input.capacity().try_into().ok()?)?
                    .checked_add(unit.output.capacity().try_into().ok()?)?
                    .checked_add(unit.effective_input.capacity().try_into().ok()?)?
                    .checked_add(unit.effective_output.capacity().try_into().ok()?)?;
            }
        }
        bytes = bytes
            .checked_add(extent::<Box<[PrefillObservationDeclaration]>>(1)?)?
            .checked_add(extent::<PrefillObservationDeclaration>(
                self.0.prefill.len(),
            )?)?;
        for declaration in &self.0.prefill {
            bytes = bytes.checked_add(declaration.path_capacity().try_into().ok()?)?;
        }
        bytes = bytes
            .checked_add(extent::<Box<[PrefillObservationDeclaration]>>(1)?)?
            .checked_add(extent::<PrefillObservationDeclaration>(
                self.0.media_prefill.len(),
            )?)?;
        for declaration in &self.0.media_prefill {
            bytes = bytes.checked_add(declaration.path_capacity().try_into().ok()?)?;
        }
        Some(bytes)
    }
    /// Whether both handles retain the same actual allocation owner.
    pub fn same_storage(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    /// Number of architecture groups represented by this exact owner.
    pub fn group_count(&self) -> usize {
        self.0.payload.len()
    }
    /// Number of ordered units in a represented group.
    pub fn unit_count(&self, group: usize) -> Option<usize> {
        self.0.payload.get(group).map(|g| g.units.len())
    }
    /// Full declared unit-boundary paths, including architecture-owned boundaries.
    /// None means an absent group or unit index.
    pub fn unit_paths(&self, group: usize, index: usize) -> Option<(&str, &str)> {
        let unit = self.0.payload.get(group)?.units.get(index)?;
        Some((&unit.input, &unit.output))
    }
    /// Effective read-only companions of the declared unit boundaries.
    /// Both names belong to this same prepared source and allocate nothing here.
    pub fn unit_effective_paths(&self, group: usize, index: usize) -> Option<(&str, &str)> {
        let unit = self.0.payload.get(group)?.units.get(index)?;
        Some((&unit.effective_input, &unit.effective_output))
    }
    fn outer_unit_paths(&self, group: usize, index: usize) -> Option<&UnitPaths> {
        let unit = self.0.payload.get(group)?.units.get(index)?;
        unit.outer.then_some(unit)
    }
    /// Optional complete group-input path.
    pub fn group_input(&self, group: usize) -> Option<&str> {
        self.0.payload.get(group)?.input.as_deref()
    }
    /// Optional complete group-output path.
    pub fn group_output(&self, group: usize) -> Option<&str> {
        self.0.payload.get(group)?.output.as_deref()
    }
    /// Attaches exact source custody once per domain, including earlier aliases.
    /// The provider may perform only closed accounting under the owner lock;
    /// never retain this payload in the handle or reenter it from the provider.
    /// Payloads retire before handles, and handle destructors run outside locks.
    pub(crate) fn original_attachment_ready(
        &self,
        domain: &SharedStorageDomain,
    ) -> Result<(), crate::working_memory::WorkingMemoryError> {
        self.0.custody.original_attachment_ready(domain)
    }

    pub fn try_attach<E>(
        &self,
        domain: &SharedStorageDomain,
        acquire: impl FnOnce() -> Result<Box<dyn Send + Sync>, E>,
    ) -> Result<bool, SharedStorageAttachmentError<E>> {
        self.0.custody.try_attach(domain, acquire)
    }
}
impl fmt::Debug for SharedLayeredObservationPaths {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedLayeredObservationPaths")
            .field("identity", self.identity())
            .field("groups", &self.group_count())
            .field("capacity_bytes", &self.capacity_bytes())
            .finish()
    }
}
fn extent<T>(count: usize) -> Option<u64> {
    size_of::<T>().checked_mul(count)?.try_into().ok()
}

/// Runtime-local generation of the shared immutable layered observation source.
/// Initial collection and semantic rebinding belong to authorized preparation;
/// this identity grants neither model work nor native storage permission.
#[derive(Debug, Clone)]
struct BindingOwner {
    runtime: Arc<()>,
    // The physical Arc (including any Weak aliases) retires before its payer.
    funding: Option<eredu_core::HostMetadataFunding>,
}
pub struct ObservationBinding(OnceLock<BindingOwner>);
impl ObservationBinding {
    /// Creates an empty generation without allocating an identity.
    pub const fn new() -> Self {
        Self(OnceLock::new())
    }
    /// Invalidates loans before exposing mutable architecture declarations.
    pub fn invalidate(&mut self) {
        self.0.take();
    }
    /// Collect the actual architecture declarations during initial loading.
    pub fn prepare_architecture<B, S, A>(&self, architecture: &A)
        -> Result<PreparedLayeredObservationPaths, PreparedLayeredObservationError<A::Error>>
    where B: NeuralBackend, S: RuntimeState<B>, A: LayeredArchitecture<B, S> {
        self.prepare(SharedLayeredObservationPaths::collect::<B, S, A>(architecture)
            .map_err(PreparedLayeredObservationError::Execution)?, &metadata::Destination(None))
            .map_err(PreparedLayeredObservationError::Execution)
    }

    /// Validate declarations and bind an existing source at a cold boundary.
    pub fn bind_architecture<B, S, A>(&self, architecture: &A, source: &SharedLayeredObservationPaths, metadata: Option<LayeredMetadata<A::Error>>)
        -> Result<PreparedLayeredObservationPaths, PreparedLayeredObservationError<A::Error>>
    where B: NeuralBackend, S: RuntimeState<B>, A: LayeredArchitecture<B, S> {
        let destination = metadata::Destination(metadata);
        destination.controls::<(&Self, &A, &SharedLayeredObservationPaths,
            metadata::Destination<A::Error>, Result<PreparedLayeredObservationPaths,
            PreparedLayeredObservationError<A::Error>>)>().map_err(PreparedLayeredObservationError::Execution)?;
        source.validate::<B, S, A>(architecture, &destination)?;
        self.prepare(source.clone(), &destination).map_err(PreparedLayeredObservationError::Execution)
    }

    /// Check this exact generation without rebuilding declarations or allocating.
    pub fn validate_binding(&self, paths: &PreparedLayeredObservationPaths)
        -> Result<(), PreparedLayeredObservationError<std::convert::Infallible>> {
        self.validate(paths)
    }

    fn prepare<E>(&self, source: SharedLayeredObservationPaths, destination: &metadata::Destination<E>)
        -> Result<PreparedLayeredObservationPaths, E> {
        destination.controls::<(&Self, SharedLayeredObservationPaths, &metadata::Destination<E>,
            BindingOwner, PreparedLayeredObservationPaths, Result<BindingOwner, BindingOwner>,
            Result<PreparedLayeredObservationPaths, E>)>()?;
        if self.0.get().is_none() {
            let runtime = match destination.context() {
                Some(context) => context.metadata_arc(()).map_err(|cause| destination.map(cause.into()))?,
                None => Arc::new(()),
            };
            let funding = destination.context().and_then(eredu_nn::workspace::WorkspaceContext::metadata_funding);
            // A racing producer drops its own paid shell, preserving the first identity.
            let _ = self.0.set(BindingOwner { runtime, funding });
        }
        Ok(PreparedLayeredObservationPaths { source, runtime: self.0.get().expect("initialized binding").clone() })
    }
    fn validate<E>(
        &self,
        paths: &PreparedLayeredObservationPaths,
    ) -> Result<(), PreparedLayeredObservationError<E>> {
        if !self
            .0
            .get()
            .is_some_and(|runtime| Arc::ptr_eq(&runtime.runtime, &paths.runtime.runtime))
        {
            return Err(PreparedLayeredObservationError::BindingMismatch);
        }
        Ok(())
    }
}

impl fmt::Debug for ObservationBinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LayeredObservationBinding").field("prepared", &self.0.get().is_some()).finish_non_exhaustive()
    }
}
impl Default for ObservationBinding {
    fn default() -> Self { Self::new() }
}

/// Read-only source binding issued by one actual runtime after cold preparation.
/// This is no execution grant or host funding certificate. Mutable architecture
/// exposure invalidates the binding, while aliases retain their physical source.
/// Architecture declarations must remain stable during ordinary equation calls,
/// including mutations hidden behind shared references. This binding authenticates
/// semantic paths, not arbitrary architecture state or native-work permission.
#[derive(Debug)]
pub struct PreparedLayeredObservationPaths {
    source: SharedLayeredObservationPaths,
    runtime: BindingOwner,
}
/// Non-Clone fingerprint of one already prepared runtime generation.
/// It grants no traversal, rebinding, submission or current-state authority.
/// Consumers must also validate the actual current token and session identity.
/// The Weak keeps the original allocation address unavailable for reuse.
#[derive(Debug)]
pub struct PreparedObservationBindingIdentity {
    runtime: Weak<()>,
    funding: Option<eredu_core::HostMetadataFunding>,
}
impl PreparedObservationBindingIdentity {
    /// Compare with an actual token without upgrading or allocating.
    pub fn matches(&self, current: &PreparedLayeredObservationPaths) -> bool {
        self.runtime.strong_count() != 0 && self.runtime.as_ptr() == Arc::as_ptr(&current.runtime.runtime)
    }

    /// Retained Arc control block plus fingerprint construction/move overlap.
    /// This diagnostic itself acquires no host funding.
    pub fn control_peak_bytes() -> Option<u64> {
        size_of::<Self>()
            .checked_mul(3)?
            .checked_add(2 * size_of::<usize>())?
            .try_into()
            .ok()
    }
}
impl PreparedLayeredObservationPaths {
    /// Retain only the identity of this existing generation, not its authority.
    pub fn binding_identity(&self) -> PreparedObservationBindingIdentity {
        PreparedObservationBindingIdentity {
            runtime: Arc::downgrade(&self.runtime.runtime),
            funding: self.runtime.funding.clone(),
        }
    }

    /// Borrows the exact shared source for loading publication and workspace reuse.
    pub fn source(&self) -> &SharedLayeredObservationPaths {
        &self.source
    }
    /// Conservative inline overlap for constructing/moving the ordinary borrowed
    /// traversal hook. The hook allocates no paths, tables, Rc or RefCell. This
    /// excludes the observer payload, ordinary tensor cloning/transformation,
    /// and architecture internal hook workspace.
    pub fn traversal_host_peak_bytes(&self) -> Option<u64> {
        // A partition pass lends this same token through begin/execute, while
        // the existing borrowed hook remains live. Account both loan copies
        // and the fixed token-validation result before either entry runs.
        extent::<BorrowedHook<'_, dyn ActivationObserver<(), std::convert::Infallible>>>(2)?
            .checked_add(extent::<Option<&Self>>(2)?)?
            .checked_add(extent::<Result<(), PreparedLayeredObservationError<std::convert::Infallible>>>(2)?)
    }
}

/// Rejection of a prepared source or failure during its ordinary equations.
#[derive(Debug, thiserror::Error)]
pub enum PreparedLayeredObservationError<E> {
    /// A different runtime or a mutable architecture exposure invalidated binding.
    #[error("observation paths are not bound to this current runtime")]
    BindingMismatch,
    /// Actual group/unit declarations differ from the retained source.
    #[error("observation path source differs from the actual architecture declarations")]
    SemanticMismatch,
    /// The observer needs full sequence readout but the caller selected less.
    #[error("prepared observer requires sequence readout")]
    ReadoutDemand,
    /// Original typed architecture/execution error.
    #[error(transparent)]
    Execution(E),
}

pub(crate) struct BorrowedHook<'a, O: ?Sized> {
    pub(crate) observer: &'a mut O,
    pub(crate) paths: &'a SharedLayeredObservationPaths,
}
impl<B, C, E, O> LayeredTraversalHook<B, C, E> for BorrowedHook<'_, O>
where
    B: NeuralBackend,
    O: ActivationObserver<B::Tensor, E> + ?Sized,
{
    fn observes_activations(&self) -> bool {
        true
    }
    fn observe_activation(&mut self, path: &str, value: &B::Tensor) -> Result<(), E> {
        self.observer.observe(path, value)
    }
    fn observe_generated_activation(
        &mut self,
        path: &str,
        prototype: &B::Tensor,
        source: &eredu_core::capture::GeneratedCaptureSource,
        generate: &mut dyn FnMut() -> Result<B::Tensor, E>,
    ) -> Result<(), E> {
        self.observer
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
            .observe_generated_retained(path, prototype, source, factory)
    }

    fn intervene_activation(
        &mut self,
        path: &str,
        value: &B::Tensor,
    ) -> Result<Option<B::Tensor>, E> {
        self.observer.intervene(path, value)
    }
    fn before_unit(
        &mut self,
        group: usize,
        index: usize,
        _: usize,
        value: &mut B::Tensor,
        _: &mut C,
        _: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<LayeredUnitAction, E> {
        if let Some(paths) = self.paths.outer_unit_paths(group, index) {
            *value =
                observe_outer_boundary(self.observer, &paths.input, &paths.effective_input, value)?;
        }
        Ok(LayeredUnitAction::Execute)
    }
    fn after_unit(
        &mut self,
        group: usize,
        index: usize,
        value: &mut B::Tensor,
        _: &mut C,
        _: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), E> {
        if let Some(paths) = self.paths.outer_unit_paths(group, index) {
            *value = observe_outer_boundary(
                self.observer,
                &paths.output,
                &paths.effective_output,
                value,
            )?;
        }
        Ok(())
    }
    fn after_group_begin(
        &mut self,
        group: usize,
        value: &mut B::Tensor,
        _: &mut C,
        _: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), E> {
        if let Some(path) = self.paths.group_input(group) {
            *value = observe_and_intervene(self.observer, path, value)?;
        }
        Ok(())
    }
    fn after_group(
        &mut self,
        group: usize,
        value: &mut B::Tensor,
        _: &mut C,
        _: &<B::Tensor as eredu_nn::Tensor>::Context,
    ) -> Result<(), E> {
        if let Some(path) = self.paths.group_output(group) {
            *value = observe_and_intervene(self.observer, path, value)?;
        }
        Ok(())
    }
}

impl<A, B, S> ResidentRuntime<A, B, S>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
{
    fn validate_observation_shape(&self, destination: &metadata::Destination<A::Error>) -> Result<(), PreparedLayeredObservationError<A::Error>> {
        destination.controls::<(&Self, &metadata::Destination<A::Error>, usize,
            crate::ArchitectureExecutionGraph<'_>, Result<usize, A::Error>,
            Result<(), PreparedLayeredObservationError<A::Error>>)>()
            .map_err(PreparedLayeredObservationError::Execution)?;
        let graph = self
            .architecture
            .execution_graph()
            .map_err(PreparedLayeredObservationError::Execution)?;
        if !graph.matches(&self.graph) || graph.group_count() != self.units.len() {
            return Err(PreparedLayeredObservationError::SemanticMismatch);
        }
        for (group, units) in self.units.iter().enumerate() {
            if self
                .architecture
                .group_unit_count(group, destination.context())
                .map_err(PreparedLayeredObservationError::Execution)?
                != units.len()
            {
                return Err(PreparedLayeredObservationError::SemanticMismatch);
            }
        }
        Ok(())
    }
    /// Constructs and binds full paths under the caller's original loading or
    /// preparation authority. Publish the source before finite admission.
    pub fn prepare_observation_paths(
        &self,
    ) -> Result<PreparedLayeredObservationPaths, PreparedLayeredObservationError<A::Error>> {
        self.validate_observation_shape(&metadata::Destination(None))?;
        self.observation_binding.prepare(
            SharedLayeredObservationPaths::collect::<B, S, A>(&self.architecture)
                .map_err(PreparedLayeredObservationError::Execution)?, &metadata::Destination(None),
        ).map_err(PreparedLayeredObservationError::Execution)
    }
    /// Coldly validates actual semantic declarations and binds the same retained
    /// payload to this runtime, e.g. for a metadata equation projection. Temporary
    /// declaration strings/graph belong to preparation, not a future forward.
    pub fn bind_observation_paths(
        &self,
        source: &SharedLayeredObservationPaths,
        metadata: Option<LayeredMetadata<A::Error>>,
    ) -> Result<PreparedLayeredObservationPaths, PreparedLayeredObservationError<A::Error>> {
        let destination = metadata::Destination(metadata);
        destination.controls::<(&Self, &SharedLayeredObservationPaths, metadata::Destination<A::Error>,
            Result<PreparedLayeredObservationPaths, PreparedLayeredObservationError<A::Error>>)>()
            .map_err(PreparedLayeredObservationError::Execution)?;
        self.validate_observation_shape(&destination)?;
        source.validate::<B, S, A>(&self.architecture, &destination)?;
        self.observation_binding.prepare(source.clone(), &destination).map_err(PreparedLayeredObservationError::Execution)
    }
    /// Checks only the existing private runtime token, without allocating,
    /// reconstructing declarations, evaluating tensors or granting execution.
    pub fn validate_observation_binding(
        &self,
        paths: &PreparedLayeredObservationPaths,
    ) -> Result<(), PreparedLayeredObservationError<std::convert::Infallible>> {
        self.observation_binding.validate(paths)
    }
    /// Borrows the actual resident provider through the existing prepared
    /// internal traversal, without requiring a parallel execution contract.
    pub fn forward_serial_routed_with_traversal_hook_with_readout<'a,Provider,H>(
        &mut self,input:A::Input<'a>,state:&mut S,pass:ExpertPass,provider:&mut Provider,
        context:&<B::Tensor as Tensor>::Context,hook:&mut H,demand:eredu_core::OutputDemand)
        ->Result<(Option<B::Tensor>,A::ForwardContext),A::Error>
    where B:eredu_nn::GroupedNeuralBackend,A:RoutedLayeredArchitecture<B,S>,
        Provider:crate::RoutedExpertProvider<B>,Provider::Error:std::fmt::Display,
        H:crate::LayeredTraversalHook<B,A::ForwardContext,A::Error>+?Sized,
    {
        self.forward_with_invocation_and_unit_executor(OrdinaryLayeredInput::new(input),state,context,hook,demand,
            |architecture,group,index,unit,hidden,state,forward,context,_hook|
                architecture.forward_unit_with_provider(group,index,unit,hidden,state,forward,pass,provider,context))
    }
    /// Executes the provider through the existing prepared observation binding.
    pub fn forward_serial_routed_with_prepared_paths<'a, Provider, O>(
        &mut self, input: A::Input<'a>, state: &mut S, pass: ExpertPass,
        provider: &mut Provider, context: &<B::Tensor as Tensor>::Context,
        observer: &mut O, paths: &PreparedLayeredObservationPaths,
        demand: eredu_core::OutputDemand,
    ) -> Result<(Option<B::Tensor>, A::ForwardContext), PreparedLayeredObservationError<A::Error>>
    where B: eredu_nn::GroupedNeuralBackend,
        A: RoutedLayeredArchitecture<B, S>, A::Error: std::fmt::Display,
        Provider: crate::RoutedExpertProvider<B>, Provider::Error: std::fmt::Display,
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.observation_binding.validate(paths)?;
        if observer.requires_sequence_readout() && demand != eredu_core::OutputDemand::Sequence {
            return Err(PreparedLayeredObservationError::ReadoutDemand);
        }
        self.forward_with_invocation_and_unit_executor(
            OrdinaryLayeredInput::new(input), state, context,
            &mut BorrowedHook { observer, paths: &paths.source }, demand,
            |architecture, group, index, unit, hidden, state, forward, context, hook|
                architecture.forward_unit_observed_with_provider(
                    group, index, unit, hidden, state, forward, pass, provider, context, hook.observer))
            .map_err(PreparedLayeredObservationError::Execution)
    }

    /// Runs the ordinary observed equations with borrowed precomputed paths.
    /// No path/table/shared-observer allocation occurs in the hook. Outer model
    /// logits observation and transactional completion remain enclosing duties.
    pub fn forward_with_prepared_observer_and_context_with_readout<'a, O>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut O,
        paths: &PreparedLayeredObservationPaths,
        demand: eredu_core::OutputDemand,
    ) -> Result<(Option<B::Tensor>, A::ForwardContext), PreparedLayeredObservationError<A::Error>>
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.observation_binding.validate(paths)?;
        if observer.requires_sequence_readout() && demand != eredu_core::OutputDemand::Sequence {
            return Err(PreparedLayeredObservationError::ReadoutDemand);
        }
        self.forward_with_traversal_hook_with_readout(
            input,
            state,
            context,
            &mut BorrowedHook {
                observer,
                paths: &paths.source,
            },
            demand,
        )
        .map_err(PreparedLayeredObservationError::Execution)
    }
}
impl<A, B, S, P> LayerwiseRuntime<A, B, S, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as eredu_nn::Tensor>::Context>,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    P: LayerwisePolicy<B, A::Unit>,
    A::Error: fmt::Display,
    P::Error: fmt::Display,
{
    /// Constructs actual stable paths during loading/preparation, before finite
    /// admission. This allocates source payloads; it grants no host funding.
    pub fn prepare_observation_paths(
        &self,
    ) -> Result<PreparedLayeredObservationPaths, PreparedLayeredObservationError<A::Error>> {
        if self.geometry_stale {
            return Err(PreparedLayeredObservationError::SemanticMismatch);
        }
        self.observation_binding.prepare(
            SharedLayeredObservationPaths::collect::<B, S, A>(&self.architecture)
                .map_err(PreparedLayeredObservationError::Execution)?, &metadata::Destination(None),
        ).map_err(PreparedLayeredObservationError::Execution)
    }
    /// Reuses the exact retained source after cold semantic declaration checks.
    pub fn bind_observation_paths(
        &self,
        source: &SharedLayeredObservationPaths,
        metadata: Option<LayeredMetadata<A::Error>>,
    ) -> Result<PreparedLayeredObservationPaths, PreparedLayeredObservationError<A::Error>> {
        let destination = metadata::Destination(metadata);
        destination.controls::<(&Self, &SharedLayeredObservationPaths, metadata::Destination<A::Error>,
            Result<PreparedLayeredObservationPaths, PreparedLayeredObservationError<A::Error>>)>()
            .map_err(PreparedLayeredObservationError::Execution)?;
        if self.geometry_stale {
            return Err(PreparedLayeredObservationError::SemanticMismatch);
        }
        source.validate::<B, S, A>(&self.architecture, &destination)?;
        self.observation_binding.prepare(source.clone(), &destination).map_err(PreparedLayeredObservationError::Execution)
    }
    /// Checks only the existing private runtime token, without allocating,
    /// reconstructing declarations, evaluating tensors or granting execution.
    pub fn validate_observation_binding(
        &self,
        paths: &PreparedLayeredObservationPaths,
    ) -> Result<(), PreparedLayeredObservationError<std::convert::Infallible>> {
        self.observation_binding.validate(paths)
    }
    /// Borrows the actual resident provider through the existing prepared
    /// internal traversal, without requiring a parallel execution contract.
    pub fn forward_serial_routed_with_traversal_hook_with_readout<'a,Provider,H>(
        &mut self,input:A::Input<'a>,state:&mut S,pass:ExpertPass,provider:&mut Provider,
        context:&<B::Tensor as Tensor>::Context,hook:&mut H,demand:eredu_core::OutputDemand)
        ->Result<(Option<B::Tensor>,A::ForwardContext),LayerwiseRuntimeError<A::Error,P::Error>>
    where B:eredu_nn::GroupedNeuralBackend,A:RoutedLayeredArchitecture<B,S>,
        Provider:crate::RoutedExpertProvider<B>,Provider::Error:std::fmt::Display,
        H:crate::LayeredTraversalHook<B,A::ForwardContext,A::Error>+?Sized,
    {
        self.forward_with_unit_executor_and_invocation(
            OrdinaryLayeredInput::new(input), state, context,
            |architecture, group, index, unit, hidden, state, forward, context, _hook|
                architecture.forward_unit_with_provider(
                    group, index, unit, hidden, state, forward, pass, provider, context),
            hook, false, true, demand)
    }
    /// Executes the provider through the existing prepared observation binding.
    pub fn forward_serial_routed_with_prepared_paths<'a, Provider, O>(
        &mut self, input: A::Input<'a>, state: &mut S, pass: ExpertPass,
        provider: &mut Provider, context: &<B::Tensor as Tensor>::Context,
        observer: &mut O, paths: &PreparedLayeredObservationPaths,
        demand: eredu_core::OutputDemand,
    ) -> Result<(Option<B::Tensor>, A::ForwardContext), PreparedLayeredObservationError<LayerwiseRuntimeError<A::Error, P::Error>>>
    where B: eredu_nn::GroupedNeuralBackend,
        A: RoutedLayeredArchitecture<B, S>, A::Error: std::fmt::Display,
        Provider: crate::RoutedExpertProvider<B>, Provider::Error: std::fmt::Display,
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.forward_invocation_with_prepared_internal_observer(
            OrdinaryLayeredInput::new(input), state, context,
            |architecture, group, index, unit, hidden, state, forward, context, observer|
                architecture.forward_unit_observed_with_provider(
                    group, index, unit, hidden, state, forward, pass, provider, context, observer),
            observer, paths, demand)
    }

    /// Uses the same ordinary unit acquisition, observed internals and abort
    /// behavior with a stack hook borrowing the precomputed source. Outer logits
    /// observation remains an enclosing duty and is not emitted here.
    pub fn forward_with_prepared_observer_and_context_with_readout<'a, O>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        observer: &mut O,
        paths: &PreparedLayeredObservationPaths,
        demand: eredu_core::OutputDemand,
    ) -> Result<
        (Option<B::Tensor>, A::ForwardContext),
        PreparedLayeredObservationError<LayerwiseRuntimeError<A::Error, P::Error>>,
    >
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.observation_binding.validate(paths)?;
        if observer.requires_sequence_readout() && demand != eredu_core::OutputDemand::Sequence {
            return Err(PreparedLayeredObservationError::ReadoutDemand);
        }
        self.forward_with_traversal_hook_with_readout(
            input,
            state,
            context,
            &mut BorrowedHook {
                observer,
                paths: &paths.source,
            },
            demand,
        )
        .map_err(PreparedLayeredObservationError::Execution)
    }
    /// Runs the fixed ordinary tensor-parallel equations with the same retained
    /// path source and borrowed hook as the serial prepared entry. No path tree
    /// or shared observer is allocated during forward, and ordinary layerwise
    /// acquisition, internal unit hooks, readout and abort behavior are preserved.
    pub fn forward_parallel_with_prepared_observer_and_context_with_readout<'a, O>(
        &mut self, input: A::Input<'a>, state: &mut S, parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context, observer: Option<&mut O>,
        paths: &PreparedLayeredObservationPaths, demand: eredu_core::OutputDemand,
    ) -> Result<(Option<B::Tensor>, A::ForwardContext),
        PreparedLayeredObservationError<LayerwiseRuntimeError<A::Error, P::Error>>>
    where A: ParallelLayeredArchitecture<B, S>,
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.observation_binding.validate(paths)?;
        let Some(observer) = observer else {
            // An unscheduled span preserves the same immutable path binding;
            // it executes only ordinary equations and creates no observation.
            return self.forward_parallel_with_unit_executor_and_traversal_hook_impl(
                input, state, parallel, context,
                |architecture, group, index, unit, hidden, state, forward, parallel, context|
                    architecture.forward_unit_parallel(group, index, unit, hidden, state, forward, parallel, context),
                &mut NoopLayeredTraversalHook, true, demand,
            ).map_err(PreparedLayeredObservationError::Execution);
        };
        if observer.requires_sequence_readout() && demand != eredu_core::OutputDemand::Sequence {
            return Err(PreparedLayeredObservationError::ReadoutDemand);
        }
        self.forward_parallel_with_unit_executor_and_traversal_hook_impl(
            input, state, parallel, context,
            |architecture, group, index, unit, hidden, state, forward, parallel, context|
                architecture.forward_unit_parallel(group, index, unit, hidden, state, forward, parallel, context),
            &mut BorrowedHook { observer, paths: &paths.source }, true, demand,
        ).map_err(PreparedLayeredObservationError::Execution)
    }

    /// Uses the architecture's existing routed/provider equations with the
    /// actual prepared path loan. This accepts a typed expert provider, not a
    /// caller-supplied mutable architecture or unit callback.
    pub fn forward_routed_with_prepared_paths<'a, Provider, O>(
        &mut self, input: A::Input<'a>, state: &mut S, parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context, provider: &mut Provider, pass: ExpertPass,
        observer: &mut O, demand: eredu_core::OutputDemand,
        paths: Option<&PreparedLayeredObservationPaths>,
    ) -> Result<(Option<B::Tensor>, A::ForwardContext),
        PreparedLayeredObservationError<LayerwiseRuntimeError<A::Error, P::Error>>>
    where B: eredu_nn::GroupedNeuralBackend,
        A: ParallelRoutedLayeredArchitecture<B, S>, A::Error: std::fmt::Display,
        Provider: crate::TensorParallelRoutedExpertProvider<B>, Provider::Error: std::fmt::Display,
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        match parallel {
            Some(parallel) => self.forward_parallel_with_internal_observer_and_prepared_paths(
                input, state, parallel, context,
                |architecture, group, index, unit, hidden, state, forward, parallel, context, observer|
                    architecture.forward_unit_parallel_observed_with_provider(
                        group, index, unit, hidden, state, forward, pass, provider, parallel, context, observer,
                    ),
                observer, demand, paths,
            ),
            None => self.forward_with_internal_observer_and_prepared_paths(
                input, state, context,
                |architecture, group, index, unit, hidden, state, forward, context, observer|
                    architecture.forward_unit_observed_with_provider(
                        group, index, unit, hidden, state, forward, pass, provider, context, observer,
                    ),
                observer, demand, paths,
            ),
        }
    }

    /// Runs the selected internal/provider equation with an optional prepared
    /// path loan. A present loan never enters the allocating ordinary observer.
    fn forward_with_internal_observer_and_prepared_paths<'a, E, O>(
        &mut self, input: A::Input<'a>, state: &mut S,
        context: &<B::Tensor as Tensor>::Context, execute: E, observer: &mut O,
        demand: eredu_core::OutputDemand, paths: Option<&PreparedLayeredObservationPaths>,
    ) -> Result<(Option<B::Tensor>, A::ForwardContext),
        PreparedLayeredObservationError<LayerwiseRuntimeError<A::Error, P::Error>>>
    where E: FnMut(&mut A, usize, usize, &mut A::Unit, &B::Tensor, &mut S,
        &mut A::ForwardContext, &<B::Tensor as Tensor>::Context, &mut O) -> Result<B::Tensor, A::Error>,
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        match paths {
            Some(paths) => self.forward_invocation_with_prepared_internal_observer(
                OrdinaryLayeredInput::new(input), state, context, execute, observer, paths, demand,
            ),
            // This helper is private to the fixed typed routed provider entry.
            // Its canonical equations cannot replace the architecture geometry.
            None => self.forward_invocation_with_internal_observer(
                OrdinaryLayeredInput::new(input), state, context, execute, observer, demand, true,
            ).map_err(PreparedLayeredObservationError::Execution),
        }
    }

    /// Lends the same retained hook to a selected parallel provider equation.
    /// The ordinary layerwise worker owns acquisition, dependency completion,
    /// readout and abort; the callback cannot replace that invocation driver.
    fn forward_parallel_with_internal_observer_and_prepared_paths<'a, E, O>(
        &mut self, input: A::Input<'a>, state: &mut S, parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context, mut execute: E, observer: &mut O,
        demand: eredu_core::OutputDemand, paths: Option<&PreparedLayeredObservationPaths>,
    ) -> Result<(Option<B::Tensor>, A::ForwardContext),
        PreparedLayeredObservationError<LayerwiseRuntimeError<A::Error, P::Error>>>
    where A: ParallelLayeredArchitecture<B, S>,
        E: FnMut(&mut A, usize, usize, &mut A::Unit, &B::Tensor, &mut S,
            &mut A::ForwardContext, &B::ParallelContext, &<B::Tensor as Tensor>::Context,
            &mut O) -> Result<B::Tensor, A::Error>,
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        let Some(paths) = paths else {
            return self.forward_parallel_invocation_with_internal_observer(
                OrdinaryLayeredInput::new(input), state, parallel, context,
                execute, observer, demand, true,
            ).map_err(PreparedLayeredObservationError::Execution);
        };
        self.observation_binding.validate(paths)?;
        if observer.requires_sequence_readout() && demand != eredu_core::OutputDemand::Sequence {
            return Err(PreparedLayeredObservationError::ReadoutDemand);
        }
        self.forward_parallel_with_unit_executor_and_invocation_hook(
            OrdinaryLayeredInput::new(input), state, parallel, context,
            |architecture, group, index, unit, hidden, state, forward, parallel, context, hook|
                execute(architecture, group, index, unit, hidden, state, forward, parallel, context, hook.observer),
            &mut BorrowedHook { observer, paths: &paths.source }, false, true, demand,
        ).map_err(PreparedLayeredObservationError::Execution)
    }

    /// Selected internal unit execution over an already authenticated invocation.
    /// The callback remains the selected canonical/provider equation; callers
    /// cannot use this crate-private entry to authorize arbitrary mutable access.
    pub(crate) fn forward_invocation_with_prepared_internal_observer<I, E, O>(
        &mut self,
        invocation: I,
        state: &mut S,
        context: &<B::Tensor as eredu_nn::Tensor>::Context,
        mut execute: E,
        observer: &mut O,
        paths: &PreparedLayeredObservationPaths,
        demand: eredu_core::OutputDemand,
    ) -> Result<
        (Option<B::Tensor>, A::ForwardContext),
        PreparedLayeredObservationError<LayerwiseRuntimeError<A::Error, P::Error>>,
    >
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
            &mut O,
        ) -> Result<B::Tensor, A::Error>,
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.observation_binding.validate(paths)?;
        if observer.requires_sequence_readout() && demand != eredu_core::OutputDemand::Sequence {
            return Err(PreparedLayeredObservationError::ReadoutDemand);
        }
        self.forward_with_unit_executor_and_invocation(
            invocation,
            state,
            context,
            |architecture, group, index, unit, hidden, state, forward, context, hook| {
                execute(
                    architecture,
                    group,
                    index,
                    unit,
                    hidden,
                    state,
                    forward,
                    context,
                    hook.observer,
                )
            },
            &mut BorrowedHook {
                observer,
                paths: &paths.source,
            },
            false,
            true,
            demand,
        )
        .map_err(PreparedLayeredObservationError::Execution)
    }
}

#[cfg(test)]
mod tests;
