//! Outer executable publication and fact reporting after portable typed construction.

use eredu_core::backend::TextExecutableTypes;
use eredu_core::{
    checkpoint_mechanisms::{ModelBindingResources, TextBindingContext, TextBindingResources},
    SubmissionBackend,
};

use crate::{
    CacheResidencyPolicy, LayeredArchitecture, ReplicatedTextArchitecture,
    ReplicatedTextRequirements, ReplicatedTextSession, ReplicatedTextSessionMechanisms,
    RuntimeState, SelectedReplicatedTextRealization,
};
use eredu_checkpoint::store::SharedCheckpointSource;
use eredu_core::communication::CommunicationManifest;
use eredu_core::{cache::PromptCacheModelIdentity, checkpoint::TensorDtype, SessionCapabilities};
use eredu_nn::{NeuralBackend, Tensor};
use std::{collections::BTreeSet, sync::Arc};

/// Typed binding of embedded prediction during portable executable erasure.
pub mod embedded_erasure;
pub(crate) mod erased_text;
mod prediction_ingress;
/// Portable erasure for typed replicated text and composite sessions.
pub mod session_erasure;
pub use erased_text::{
    install_text_route, ErasedExternalPredictionExecutable, ErasedTextExecutable,
    InstalledTextExecutable, TextExecutableRoute,
};

mod bound_text_session;
pub use bound_text_session::{
    BoundTextSession, CompositeSessionIngress, NoSelectedPrediction, SelectedPrediction,
    TextParameterBanks,
};

/// Typed composite partition handoff and native binding visitors.
pub mod composite_partition_binding;
/// Typed construction for admitted dense partitions.
pub mod dense_partition;
/// Loaded model reports and architecture projections.
pub mod discovery;
/// Family-neutral distributed session executors and selected collective waves.
pub mod partition_executors;

/// Shared rank-local construction for partitioned text.
pub mod partition_rank;

mod partition_facts;
pub use partition_facts::PreparedPartitionSessionFacts;

mod profiles;
pub use profiles::{
    ReplicatedTextProfileDispatcher, ReplicatedTextStateProfiles, SharedReplicatedTextVisitor,
};

mod route_inputs;
pub use route_inputs::{
    CompositeRoute, KeyValueRoute, PartitionedCompositeRoute, PartitionedDenseRoute,
    PartitionedRoutedRoute, PredictionMechanisms, ReplicatedRoute, RoutedRoute, WithoutPrediction,
};

mod portable_types;
pub use portable_types::{PortableTextTypes, TextPredictionPrimitives};

pub mod route_assembly;
pub use route_assembly::{BoundExecutionRoutes, TextRankRoutes};

mod routes;
pub use routes::{PreparedExecutionRoutes, Unavailable};

/// Input observations and shared admission checks.
pub mod input_admission;

/// Portable invocation and native representations for a retained text executable.
pub mod retained_executable;
/// Invocation and input accounting for a retained text session.
pub mod retained_session;
/// Portable text invocation and its native tensor contracts.
pub mod text_invocation;

/// Provider construction for prepared routed partitions.
pub mod routed_partition;
/// Typed routed partition handoff and native binding visitors.
pub mod routed_partition_binding;
/// Session construction for prepared routed models.
pub mod routed_session;

/// Failure before publication of a fully constructed executable.
#[derive(Debug, thiserror::Error)]
pub enum PreparedExecutionError<E> {
    /// Selected sources or architecture construction violated their contract.
    #[error("prepared execution construction failed: {0}")]
    Architecture(String),
    /// The constructed executable and retained discovery facts disagree.
    #[error(transparent)]
    Discovery(eredu_core::capture::CaptureError),
    /// A native materialization, communication, or publication mechanism failed.
    #[error("prepared execution mechanism failed: {0}")]
    Backend(#[source] E),
    /// A native mechanism failed while binding portable components.
    #[error(transparent)]
    Binding(eredu_core::BackendFailure),
    /// The selected manifest does not agree with the native rank placement.
    #[error("communication setup differs from selected model placement")]
    CommunicationPlacementMismatch,
    /// The realized resources differ from the exact selected communication manifest.
    #[error("realized model communication differs from the selected manifest")]
    CommunicationRealizationMismatch,
    /// A partitioned selection has no realized communication.
    #[error("partitioned construction has no realized communication")]
    MissingCommunication,
    /// Ordinary execution was paired with partition communication.
    #[error("ordinary construction has unexpected realized communication")]
    UnexpectedCommunication,
    /// Extension selection and its exact prepared source do not agree.
    #[error("prepared prediction selection and source role disagree")]
    PredictionSourceMismatch,
    /// The selected class has no supplied typed construction mechanism.
    #[error("selected execution has no supplied typed construction mechanism")]
    UnavailableExecution,
    /// An embedded extension was selected without a native extension materializer.
    #[error("selected prediction extension has no supplied construction mechanism")]
    UnavailablePrediction,
}

/// Architecture-resolved source of the dtype used by generic floating runtime
/// state.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FloatingStateDtypeSource {
    parameter: String,
    checkpoint_tensor: String,
    dtype: TensorDtype,
}

impl FloatingStateDtypeSource {
    /// Records the parameter and exact admitted physical tensor.
    pub fn new(parameter: String, checkpoint_tensor: String, dtype: TensorDtype) -> Self {
        Self {
            parameter,
            checkpoint_tensor,
            dtype,
        }
    }

    /// Architecture-declared checkpoint parameter whose loaded values establish
    /// the ordinary decoder activation dtype.
    pub fn parameter(&self) -> &str {
        &self.parameter
    }

    /// Physical tensor name selected from the architecture's admitted aliases.
    pub fn checkpoint_tensor(&self) -> &str {
        &self.checkpoint_tensor
    }

    /// Inspected scalar dtype of the selected physical tensor.
    pub fn dtype(&self) -> &TensorDtype {
        &self.dtype
    }
}

/// Selected realization and capabilities passed to a native prediction binder.
pub struct PredictionBinding {
    selected: crate::SelectedSpeculativeRealization,
    capability: crate::CapabilityEstimate,
}

impl PredictionBinding {
    /// Couples the admitted speculative mechanism to its joint capability estimate.
    pub fn new(
        selected: crate::SelectedSpeculativeRealization,
        capability: crate::CapabilityEstimate,
    ) -> Self {
        Self {
            selected,
            capability,
        }
    }

    /// Consumes the exact selected predictor and its joint capability estimate.
    pub fn into_parts(
        self,
    ) -> (
        crate::SelectedSpeculativeRealization,
        crate::CapabilityEstimate,
    ) {
        (self.selected, self.capability)
    }
}

/// Checked native resources for a partition-specific typed binding visitor.
pub struct PreparedPartitionResources<C> {
    communication: C,
    target: SharedCheckpointSource,
    extension_sources: BTreeSet<String>,
}

impl<C> PreparedPartitionResources<C> {
    /// Couples checked rank communication with the exact target source roles.
    pub fn new(
        communication: C,
        target: SharedCheckpointSource,
        extension_sources: BTreeSet<String>,
    ) -> Self {
        Self {
            communication,
            target,
            extension_sources,
        }
    }

    /// Consumes the role-bound source, communication and extension keys.
    pub fn into_parts(self) -> (C, SharedCheckpointSource, BTreeSet<String>) {
        (self.communication, self.target, self.extension_sources)
    }
}

/// Checked facts and native resources for a partitioned prediction binding visitor.
pub struct PreparedPartitionPredictionResources<C> {
    partition: PreparedPartitionResources<C>,
    prediction: PredictionBinding,
}

impl<C> PreparedPartitionPredictionResources<C> {
    /// Adds the selected predictor facts to the checked target partition resources.
    pub fn new(partition: PreparedPartitionResources<C>, prediction: PredictionBinding) -> Self {
        Self {
            partition,
            prediction,
        }
    }

    /// Consumes the rank resources and their paired predictor selection together.
    pub fn into_parts(self) -> (PreparedPartitionResources<C>, PredictionBinding) {
        (self.partition, self.prediction)
    }
}

/// Native binding facts projected from one exact prepared model source graph.
pub trait PreparedModelBinding {
    /// The session admission retained before any native work.
    fn session_capabilities(&self) -> SessionCapabilities;

    /// Rank-local communication selected before native realization.
    fn communication_manifest(&self) -> Option<&CommunicationManifest>;
}

/// Optional portable processor retained by the installed text session.
pub type PreparedTextProcessor<M> = Option<
    Arc<
        dyn crate::processor_binding::PreparedProcessorProgram<
            <M as TextExecutableTypes>::InputTensor,
            <M as TextExecutableTypes>::Error,
        >,
    >,
>;

/// Erased executable produced by the portable binder for the selected resources.
pub type BoundExecutableFor<B> = InstalledTextExecutable<TextTypesFor<B>>;

/// Portable text executable profile for a native binding.
pub type TextTypesFor<B> = PortableTextTypes<<B as ModelBindingResources>::TextValues>;

/// Model product assembled by the portable binder from the selected mechanism types.
pub type BoundModelFor<B> = LoadedTextExecutable<
    BoundExecutableFor<B>,
    <B as ModelBindingResources>::Communication,
    PreparedTextProcessor<TextTypesFor<B>>,
    <B as ModelBindingResources>::Target,
>;

/// Checks native rank placement against the selected manifest before native setup.
fn admitted_model_communication<E>(
    manifest: Option<&CommunicationManifest>,
    distributed: bool,
) -> Result<Option<&CommunicationManifest>, PreparedExecutionError<E>> {
    if manifest.is_some() == distributed {
        Ok(manifest)
    } else {
        Err(PreparedExecutionError::CommunicationPlacementMismatch)
    }
}

/// Agrees and realizes the exact rank manifest before allocating model resources.
pub fn bind_model_communication<B>(
    backend: &B,
    manifest: Option<&CommunicationManifest>,
    placement: &B::Placement,
) -> Result<Option<B::Communication>, PreparedExecutionError<B::BindingError>>
where
    B: ModelBindingResources,
    B::BindingError: std::error::Error + Send + Sync + 'static,
{
    let manifest = admitted_model_communication(manifest, B::has_native_rank(placement))?;
    backend.reclaim_retired_bindings();
    let prepared_communication = manifest
        .map(|manifest| {
            let transport = backend
                .open_communication_setup(placement)
                .map_err(PreparedExecutionError::Backend)?;
            crate::communication::prepare_world_communication(transport, manifest)
                .map_err(eredu_core::BackendFailure::from_error)
                .map_err(PreparedExecutionError::Binding)
        })
        .transpose()?;
    let communication = prepared_communication
        .map(|setup| backend.realize_model_communication(setup))
        .transpose()
        .map_err(PreparedExecutionError::Backend)?;
    if communication.as_ref().map(B::bound_communication_manifest) != manifest {
        return Err(PreparedExecutionError::CommunicationRealizationMismatch);
    }
    Ok(communication)
}

/// Constructs a prepared model using native resources supplied for its exact placement.
pub fn construct_prepared_model<'a, B, S, P, D>(
    backend: &'a B,
    sources: S,
    placement: B::Binding,
) -> Result<eredu_core::PreparedModel<BoundModelFor<B>>, PreparedExecutionError<B::Error>>
where
    B: eredu_core::ModelLoadingBackend
        + ModelBindingResources<Placement = B::Binding, BindingError = B::Error>
        + TextBindingContext<'a, RankPlacement: TextRankRoutes<'a, B>>
        + TextBindingResources<
            Neural: eredu_nn::BlockwiseAttentionBackend,
            CacheStorage: crate::state::HybridCacheStorage<
                Value = <<B as TextBindingResources>::Neural as eredu_nn::NeuralBackend>::Tensor,
            > + eredu_core::cache::mechanisms::PoolingCacheStorage,
        >,
    B::Error: From<eredu_core::BackendFailure>,
    P: crate::processor_binding::PreparedProcessorProgram<
            <TextTypesFor<B> as TextExecutableTypes>::InputTensor,
            B::Error,
        > + 'static,
    B::TextValues: eredu_core::backend::NativeTextValues<Error = B::Error>,
    D: PreparedExecutableDiscovery + discovery::LoadedModelDiscovery + 'static,
    S: PreparedModelBinding
        + PreparedExecutableConstruction<
            B::Communication,
            BoundExecutableFor<B>,
            B::Error,
            BoundExecutionRoutes<'a, B>,
            Processor = P,
            Discovery = D,
        >,
{
    let routes = route_assembly::bind_text_routes(backend);
    let capabilities = sources.session_capabilities();
    let communication =
        bind_model_communication(backend, sources.communication_manifest(), &placement)?;
    crate::automatic_support::initialize_allocator_defaults(backend.allocator_cache_defaults())
        .map_err(PreparedExecutionError::Binding)?;
    let target = backend
        .open_binding_target(&placement, communication.as_ref())
        .map_err(PreparedExecutionError::Backend)?;
    let model = construct_prepared_executable(backend, sources, routes, communication, target)?;
    Ok(eredu_core::PreparedModel::new(model, capabilities))
}

/// Retained semantic discovery refined with facts from the constructed executable.
pub trait PreparedExecutableDiscovery: Sized {
    /// Binds the exact partition parameter topology and active hooks.
    fn bind_executable(
        self,
        parameters: Option<Arc<crate::ArchitectureParameterDescription>>,
        hooks: Option<crate::inspection::ObservationHookSupport>,
    ) -> Result<Self, eredu_core::capture::CaptureError>;
}

/// The model-owned stage of executable construction.
pub trait PreparedExecutableConstruction<C, E, F, R>: Sized {
    /// Retained processor selected for raw-media input.
    type Processor;
    /// Discovery projection for the selected execution.
    type Discovery: PreparedExecutableDiscovery;
    /// Resolves selected source roles and returns one dispatch operation.
    fn prepare(
        self,
        routes: R,
        observation: eredu_core::ObservationMechanisms,
        capture: eredu_core::capture::CaptureCapabilities,
    ) -> Result<
        PreparedExecutablePlan<
            Self::Processor,
            Self::Discovery,
            impl FnOnce(Option<C>) -> Result<E, PreparedExecutionError<F>> + use<Self, C, E, F, R>,
        >,
        PreparedExecutionError<F>,
    >;
}

/// Exact metadata and the one-shot typed dispatch for a prepared executable.
pub struct PreparedExecutablePlan<P, D, R> {
    /// Admitted floating-state tensor encoding.
    pub floating_source: FloatingStateDtypeSource,
    /// Physical state representation required by the selected mechanism, if specified.
    pub expected_dtype: Option<crate::StateStorageDtype>,
    /// Exact mutable state residency policy.
    pub state_residency: CacheResidencyPolicy,
    /// Retained raw-media processing policy.
    pub processor: Option<P>,
    /// Discovery projected from the selected model and collector mechanisms.
    pub discovery: D,
    /// Typed construction route that consumes the native communication handle.
    pub dispatch: R,
}

/// Prepares the exact model, validates native facts, and retains the bound executable.
fn construct_prepared_executable<C, A, S, P, D, R, M>(
    backend: &A,
    construction: S,
    routes: R,
    communication: Option<C>,
    target: A::Target,
) -> Result<
    LoadedTextExecutable<InstalledTextExecutable<M>, C, PreparedTextProcessor<M>, A::Target>,
    PreparedExecutionError<A::BindingError>,
>
where
    C: Clone,
    A: ModelBindingResources<
        Communication = C,
        BindingError: std::error::Error + Send + Sync + From<eredu_core::BackendFailure> + 'static,
    >,
    M: TextExecutableTypes<Error = A::BindingError>,
    P: crate::processor_binding::PreparedProcessorProgram<M::InputTensor, M::Error> + 'static,
    S: PreparedExecutableConstruction<
        C,
        InstalledTextExecutable<M>,
        A::BindingError,
        R,
        Processor = P,
        Discovery = D,
    >,
    D: PreparedExecutableDiscovery + discovery::LoadedModelDiscovery + 'static,
{
    let plan = construction.prepare(
        routes,
        backend.observation_mechanisms(),
        backend.capture_capabilities(),
    )?;
    use crate::StateStorageDtype as Storage;
    use eredu_nn::TensorElementType as Element;
    let dtype = match backend.floating_state_element(plan.floating_source.dtype()) {
        Some(Element::F16) => Storage::F16,
        Some(Element::Bf16) => Storage::Bf16,
        Some(Element::F32) => Storage::F32,
        Some(Element::F64) => Storage::F64,
        Some(Element::Complex64) => Storage::Complex64,
        _ => {
            return Err(PreparedExecutionError::Architecture(format!(
            "checkpoint tensor {:?} with dtype {:?} has no native floating state representation",
            plan.floating_source.checkpoint_tensor(),
            plan.floating_source.dtype(),
        )))
        }
    };
    if !dtype.is_floating()
        || plan
            .expected_dtype
            .is_some_and(|expected| expected != dtype)
    {
        return Err(PreparedExecutionError::Architecture(
            "native floating-state dtype differs from the admitted storage dtype".into(),
        ));
    }
    let retained_communication = communication.clone();
    let mut executable = (plan.dispatch)(communication)?;
    let discovery = plan
        .discovery
        .bind_executable(
            executable.partition_parameter_description().cloned(),
            executable.partition_observation_hooks(),
        )
        .map_err(PreparedExecutionError::Discovery)?;
    let discovery: Arc<dyn discovery::LoadedModelDiscovery> = Arc::new(discovery);
    let processor: PreparedTextProcessor<M> = plan.processor.map(|program| Arc::new(program) as _);
    if let Some(target) = executable.external_prediction_mut() {
        target.target_placements().bind(Arc::clone(&discovery));
    }
    backend
        .finish_binding()
        .map_err(PreparedExecutionError::Backend)?;
    Ok(LoadedTextExecutable {
        executable,
        layout: crate::text_session_driver::TextSessionLayout::new(
            plan.state_residency,
            dtype.bytes(),
        ),
        communication: retained_communication,
        processor,
        discovery,
        target,
    })
}

/// Selected executable and its retained resources, ready to install or run as a draft.
pub struct LoadedTextExecutable<E, C, P, T> {
    /// Bound executable for the selected placement.
    pub executable: E,
    /// Selected mutable-state placement and precision.
    pub layout: crate::text_session_driver::TextSessionLayout,
    /// Exact checked communication handle, if distributed.
    pub communication: Option<C>,
    /// Architecture-selected portable processor program.
    pub processor: P,
    /// Retained architecture discovery, shared with consumers that need its projections.
    pub discovery: Arc<dyn discovery::LoadedModelDiscovery>,
    /// Exact native placement alongside the portable executable.
    pub target: T,
}

impl<E, C, P, T> LoadedTextExecutable<E, C, P, T> {
    /// Projects selected geometry, refining local models with live residency.
    pub fn memory_geometry(
        &self,
        residency: Option<&crate::ResidencyReport>,
    ) -> Result<crate::memory_forecast::LoadedMemoryGeometry, eredu_core::CapabilityError> {
        let mut geometry = self.discovery.generation_memory()?.clone();
        geometry.apply_resident_storage(self.communication.is_none(), residency);
        Ok(geometry)
    }
}

/// Checked architecture value passed across the architecture/backend boundary.
pub struct PreparedReplicatedTextArchitecture<A> {
    architecture: A,
    source_architecture: Option<A>,
    requirements: ReplicatedTextRequirements,
    contract: crate::PreparedReplicatedTextContract,
    capability_estimate: crate::CapabilityEstimate,
    effective_model_type: String,
}

impl<A> PreparedReplicatedTextArchitecture<A> {
    /// Couples selected module ownership with its checked construction contract.
    pub fn new(
        architecture: A,
        source_architecture: Option<A>,
        requirements: ReplicatedTextRequirements,
        contract: crate::PreparedReplicatedTextContract,
        capability_estimate: crate::CapabilityEstimate,
        effective_model_type: String,
    ) -> Self {
        Self {
            architecture,
            source_architecture,
            requirements,
            contract,
            capability_estimate,
            effective_model_type,
        }
    }

    /// Returns the exact architecture and artifact requirements.
    pub const fn requirements(&self) -> &ReplicatedTextRequirements {
        &self.requirements
    }

    /// Returns the authoritative selected realization.
    pub const fn selected(&self) -> &SelectedReplicatedTextRealization {
        self.contract.selected()
    }

    /// Returns the architecture-derived identity for persistent prompt state.
    pub const fn prompt_cache_identity(&self) -> &PromptCacheModelIdentity {
        self.contract.prompt_cache_identity()
    }

    /// Returns the architecture capability estimate presented by the session.
    pub const fn capability_estimate(&self) -> &crate::CapabilityEstimate {
        &self.capability_estimate
    }

    /// Returns the normalized model-type label presented by the session.
    pub fn effective_model_type(&self) -> &str {
        &self.effective_model_type
    }

    /// Consumes the handoff into opaque architecture module ownership.
    pub fn into_modules(self) -> PreparedReplicatedTextModules<A> {
        PreparedReplicatedTextModules {
            architecture: Some(self.architecture),
            source_architecture: self.source_architecture,
            contract: Some(self.contract),
        }
    }
}

/// Opaque ownership of selected-format and optional source-format modules.
pub struct PreparedReplicatedTextModules<A> {
    architecture: Option<A>,
    source_architecture: Option<A>,
    contract: Option<crate::PreparedReplicatedTextContract>,
}

impl<A> PreparedReplicatedTextModules<A> {
    /// Takes the selected-format architecture exactly once.
    pub fn take_architecture(&mut self) -> A {
        self.architecture
            .take()
            .expect("prepared architecture was already taken")
    }

    /// Takes the source-format architecture used by a selected transform.
    pub fn take_source_architecture(&mut self) -> Option<A> {
        self.source_architecture.take()
    }

    /// Takes the validated neutral construction contract exactly once.
    pub fn take_contract(&mut self) -> crate::PreparedReplicatedTextContract {
        self.contract
            .take()
            .expect("prepared construction contract was already taken")
    }
}

/// Native resources attached to a direct local executable.
pub trait LocalTextResources {
    /// Exact native tensor and queue resources.
    type LocalResources: text_invocation::TextInvocationResources;

    /// Supplies the resources for the admitted local placement.
    fn local_resources(&self) -> Self::LocalResources;
}

/// Selected prediction admission carried until the portable executable is bound.
pub trait LocalPredictionResources: LocalTextResources {
    /// Transfers the exact selected predictor and its joint capabilities.
    fn into_prediction_binding(self) -> PredictionBinding;
}

/// Supplies native binding and publication for an admitted ordinary text session.
pub trait ReplicatedTextArchitectureVisitor<B, S>: Sized + LocalTextResources
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    S: RuntimeState<B>,
{
    /// Installed portable executable.
    type Output;
    /// Backend binding failure.
    type Error;
    /// Native parameter and state mechanisms for the selected architecture.
    type Mechanisms<A>: ReplicatedTextSessionMechanisms<
        A,
        B,
        State = S,
        Error: std::fmt::Display,
        PolicyError: std::fmt::Display,
    >
    where
        A: ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error> + 'static,
        A::StaticModules: Clone;

    /// Borrows the queue for this construction.
    fn context(&self) -> &B::Executor;

    /// Supplies native binding inputs for the exact source.
    fn mechanisms<A>(&self, store: SharedCheckpointSource) -> Self::Mechanisms<A>
    where
        A: ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error> + 'static,
        A::StaticModules: Clone;

    /// Concrete executable for this completed route.
    type BoundReplicated<A>: From<
            BoundTextSession<
                ReplicatedTextSession<A, B, Self::Mechanisms<A>>,
                (),
                Self::LocalResources,
                NoSelectedPrediction,
            >,
        > + TextExecutableRoute<Self::Output>
    where
        A: ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error> + 'static,
        A::StaticModules: Clone;

    /// Converts a binding failure from portable construction.
    fn construction_failure(error: String) -> Self::Error;
}

/// Consumes a checked ordinary text handoff using selected portable construction.
pub trait PreparedReplicatedTextVisitor<B: SubmissionBackend, S: RuntimeState<B>> {
    /// Installed or prepared result.
    type Output;
    /// Failure preparing the selected execution.
    type Error;

    /// Binds the checked architecture and its exact checkpoint source.
    fn build<A>(
        self,
        prepared: PreparedReplicatedTextArchitecture<A>,
        source: SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error> + 'static,
        A::StaticModules: Clone;
}

impl<B, S, V> PreparedReplicatedTextVisitor<B, S> for V
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    S: RuntimeState<B>,
    V: ReplicatedTextArchitectureVisitor<B, S>,
{
    type Output = V::Output;
    type Error = V::Error;

    fn build<A>(
        self,
        prepared: PreparedReplicatedTextArchitecture<A>,
        source: SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error> + 'static,
        A::StaticModules: Clone,
    {
        construct_replicated_text::<B, S, A, Self>(self, prepared, source)
    }
}

/// Constructs and erases an ordinary session with the selected native resources.
pub fn construct_replicated_text<B, S, A, V>(
    visitor: V,
    prepared: PreparedReplicatedTextArchitecture<A>,
    store: SharedCheckpointSource,
) -> Result<V::Output, V::Error>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    S: RuntimeState<B>,
    V: ReplicatedTextArchitectureVisitor<B, S>,
    A: ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error> + 'static,
    A::StaticModules: Clone,
{
    let mechanisms = visitor.mechanisms(store);
    let (session, facts) = construct_selected_text_session(
        prepared,
        mechanisms,
        Default::default(),
        visitor.context(),
    )
    .map_err(V::construction_failure)?;
    let bound = facts.bind(
        session,
        (),
        visitor.local_resources(),
        NoSelectedPrediction,
        true,
    );
    Ok(install_text_route::<V::BoundReplicated<A>, _, _>(bound))
}

/// Retained architecture facts for final adaptation of a constructed text session.
#[derive(Debug, Clone)]
pub struct PreparedTextSessionFacts {
    capability: crate::CapabilityEstimate,
    effective_model_type: String,
    residency: crate::LayerWeightResidency,
}

impl PreparedTextSessionFacts {
    /// Retains the immutable adaptation facts.
    pub fn from_parts(
        capability: crate::CapabilityEstimate,
        effective_model_type: String,
        residency: crate::LayerWeightResidency,
    ) -> Self {
        Self {
            capability,
            effective_model_type,
            residency,
        }
    }

    /// Copies immutable adaptation facts before consuming the selected modules.
    pub fn from_prepared<A>(prepared: &PreparedReplicatedTextArchitecture<A>) -> Self {
        Self {
            capability: prepared.capability_estimate().clone(),
            effective_model_type: prepared.effective_model_type().to_owned(),
            residency: prepared.selected().residency(),
        }
    }

    /// Capability estimate retained by the selected target architecture.
    pub const fn capability(&self) -> &crate::CapabilityEstimate {
        &self.capability
    }

    /// Normalized model-type label retained by the selected target architecture.
    pub fn effective_model_type(&self) -> &str {
        &self.effective_model_type
    }

    /// Exact selected execution-unit weight residency.
    pub const fn residency(&self) -> crate::LayerWeightResidency {
        self.residency
    }

    /// Binds the typed session and its exact placement to the executable owner.
    pub fn bind<S, I, N: text_invocation::TextInvocationResources, P>(
        self,
        session: S,
        ingress: I,
        native: N,
        prediction: P,
        publishes_scores: bool,
    ) -> BoundTextSession<S, I, N, P> {
        BoundTextSession {
            session,
            ingress,
            native,
            banks: Default::default(),
            prediction,
            facts: self,
            publishes_scores,
        }
    }
}

/// Constructs an ordinary text session without exposing module/contract assembly to adapters.
#[allow(clippy::type_complexity)]
pub fn construct_selected_text_session<B, A, M>(
    prepared: PreparedReplicatedTextArchitecture<A>,
    mechanisms: M,
    weights: crate::ReplicatedWeightOptions<M::WeightStorage>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<(ReplicatedTextSession<A, B, M>, PreparedTextSessionFacts), String>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    A::Error: std::fmt::Display,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    let facts = PreparedTextSessionFacts::from_prepared(&prepared);
    let mut modules = prepared.into_modules();
    let session = crate::construct_replicated_text_session(
        modules.take_architecture(),
        modules.take_source_architecture(),
        modules.take_contract(),
        mechanisms,
        weights,
        context,
    )
    .map_err(|error| error.to_string())?;
    Ok((session, facts))
}

/// Checked composite architecture passed to a backend-generic session constructor.
pub struct PreparedCompositeTextArchitecture<A, C, R> {
    architecture: A,
    source_architecture: Option<A>,
    requirements: R,
    contract: crate::PreparedReplicatedTextContract,
    processor: crate::SelectedProcessorExecution,
    admission: C,
    capability_estimate: crate::CapabilityEstimate,
    effective_model_type: String,
}

impl<A, C, R> PreparedCompositeTextArchitecture<A, C, R> {
    /// Pairs the admitted input policy and exact selected processor with the modules.
    pub fn new(
        architecture: A,
        source_architecture: Option<A>,
        requirements: R,
        contract: crate::PreparedReplicatedTextContract,
        processor: crate::SelectedProcessorExecution,
        admission: C,
        capability_estimate: crate::CapabilityEstimate,
        effective_model_type: String,
    ) -> Self {
        Self {
            architecture,
            source_architecture,
            requirements,
            contract,
            processor,
            admission,
            capability_estimate,
            effective_model_type,
        }
    }

    /// Exact architecture, artifact, and processor requirements.
    pub const fn requirements(&self) -> &R {
        &self.requirements
    }

    /// Authoritative processor mechanism selection.
    pub const fn processor(&self) -> &crate::SelectedProcessorExecution {
        &self.processor
    }

    /// Architecture-derived identity coupled to the shared text session.
    pub const fn prompt_cache_identity(&self) -> &PromptCacheModelIdentity {
        self.contract.prompt_cache_identity()
    }

    /// Exact selected text execution realization.
    pub const fn selected(&self) -> &SelectedReplicatedTextRealization {
        self.contract.selected()
    }

    /// Architecture capability estimate presented by the shared session.
    pub const fn capability_estimate(&self) -> &crate::CapabilityEstimate {
        &self.capability_estimate
    }

    /// Normalized model-type label presented by the shared session.
    pub fn effective_model_type(&self) -> &str {
        &self.effective_model_type
    }

    /// Consumes the handoff into generic session-construction inputs.
    pub fn into_parts(
        self,
    ) -> (
        A,
        Option<A>,
        crate::PreparedReplicatedTextContract,
        crate::SelectedProcessorExecution,
        C,
    ) {
        (
            self.architecture,
            self.source_architecture,
            self.contract,
            self.processor,
            self.admission,
        )
    }
}

/// Exact ingress admission and adaptation facts retained with a constructed composite session.
pub struct PreparedCompositeSessionFacts<C> {
    text: PreparedTextSessionFacts,
    processor: crate::SelectedProcessorExecution,
    admission: C,
}

impl<C> PreparedCompositeSessionFacts<C> {
    /// Combines constructed session facts with the retained processor selection
    /// and architecture-owned ingress policy for final native adaptation.
    pub fn new(
        text: PreparedTextSessionFacts,
        processor: crate::SelectedProcessorExecution,
        admission: C,
    ) -> Self {
        Self {
            text,
            processor,
            admission,
        }
    }

    /// Binds the selected ingress together with the installed composite session.
    pub fn bind<S, N: text_invocation::TextInvocationResources>(
        self,
        session: S,
        native: N,
        publishes_scores: bool,
    ) -> BoundTextSession<S, CompositeSessionIngress<C>, N, NoSelectedPrediction> {
        self.text.bind(
            session,
            CompositeSessionIngress::new(self.admission, self.processor),
            native,
            NoSelectedPrediction,
            publishes_scores,
        )
    }
}

/// Constructs direct composite execution with its exact retained ingress admission.
#[allow(clippy::type_complexity)]
pub fn construct_selected_composite_session<B, A, Admission, R, M>(
    prepared: PreparedCompositeTextArchitecture<A, Admission, R>,
    mechanisms: M,
    weights: crate::ReplicatedWeightOptions<M::WeightStorage>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<
    (
        ReplicatedTextSession<A, B, M>,
        PreparedCompositeSessionFacts<Admission>,
    ),
    String,
>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    <A as LayeredArchitecture<B, M::State>>::Error: std::fmt::Display,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    let facts = PreparedTextSessionFacts::from_parts(
        prepared.capability_estimate().clone(),
        prepared.effective_model_type().to_owned(),
        prepared.selected().residency(),
    );
    let (architecture, source, contract, processor, admission) = prepared.into_parts();
    let session = crate::construct_replicated_text_session(
        architecture,
        source,
        contract,
        mechanisms,
        weights,
        context,
    )
    .map_err(|error| error.to_string())?;
    Ok((
        session,
        PreparedCompositeSessionFacts::new(facts, processor, admission),
    ))
}

/// Family-blind backend visitor for an exact replicated prediction target.
pub trait ReplicatedPredictionTargetVisitor<B, S, M>: Sized + LocalPredictionResources
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: RuntimeState<B>,
    M: crate::speculative::PredictionExtensionMaterializer<B>,
{
    /// Completed backend adapter.
    type Output;
    /// Backend binding failure.
    type Error: From<eredu_core::parameters::ParameterError>;
    /// Native parameter and state resources for the selected architecture.
    type Mechanisms<A>: ReplicatedTextSessionMechanisms<
        A,
        B,
        State = S,
        Error: std::fmt::Display,
        PolicyError: std::fmt::Display,
    >
    where
        A: ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
            + crate::speculative::PredictionTarget<B>
            + 'static,
        A::StaticModules: Clone;

    /// Borrows the queue for this construction.
    fn context(&self) -> &B::Executor;

    /// Supplies native state and parameter bindings for the paired target.
    fn prediction_mechanisms<A>(
        &self,
        extension: &mut <A as crate::speculative::PredictionTarget<B>>::Extension<M>,
        store: SharedCheckpointSource,
    ) -> Result<(Self::Mechanisms<A>, crate::ReplicatedWeightOptions<
        <Self::Mechanisms<A> as crate::replicated_session::ReplicatedTextPrimitives<B>>::WeightStorage,
    >), Self::Error>
    where
        A: ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
            + crate::speculative::PredictionTarget<B>
            + 'static,
        A::StaticModules: Clone;

    /// Concrete executable for this completed route.
    type BoundPrediction<A>: From<
            BoundTextSession<
                ReplicatedTextSession<A, B, Self::Mechanisms<A>>,
                (),
                Self::LocalResources,
                SelectedPrediction<<A as crate::speculative::PredictionTarget<B>>::Extension<M>>,
            >,
        > + crate::executable_binding::TextExecutableRoute<Self::Output>
    where
        A: ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
            + crate::speculative::PredictionTarget<B>
            + 'static,
        A::StaticModules: Clone;
}

/// Installs the paired prediction catalog before native executable erasure.
pub fn bind_local_replicated_prediction<B, S, M, A, V>(
    visitor: V,
    session: ReplicatedTextSession<A, B, V::Mechanisms<A>>,
    extension: <A as crate::speculative::PredictionTarget<B>>::Extension<M>,
    facts: PreparedTextSessionFacts,
) -> Result<V::Output, V::Error>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    S: RuntimeState<B>,
    M: crate::speculative::PredictionExtensionMaterializer<B>,
    A: ReplicatedTextArchitecture<B, S, Error = eredu_nn::Error>
        + crate::speculative::PredictionTarget<B>
        + 'static,
    A::StaticModules: Clone,
    V: ReplicatedPredictionTargetVisitor<B, S, M>,
{
    let resources = visitor.local_resources();
    let (selected, capability) = visitor.into_prediction_binding().into_parts();
    let executable = facts
        .bind(session, (), resources, NoSelectedPrediction, true)
        .install_prediction::<M, _>(
            SelectedPrediction {
                extension,
                selected,
            },
            capability,
        )?;
    Ok(crate::executable_binding::install_text_route::<
        V::BoundPrediction<A>,
        _,
        _,
    >(executable))
}

/// Supplies prediction-aware visitors while architecture dispatch selects a replicated state profile.
pub trait ReplicatedPredictionProfileDispatcher<B, M>: Sized
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + eredu_nn::BlockwiseAttentionBackend
        + eredu_nn::DistributedNeuralBackend
        + eredu_nn::TensorParallelGroupedNeuralBackend
        + eredu_nn::HyperNeuralBackend,
    M: crate::speculative::PredictionExtensionMaterializer<B>,
{
    /// Completed construction output.
    type Output;
    /// Mechanism binding failure.
    type Error;
    /// State representation selected for the target.
    type State: crate::LayerRuntimeState<B>;
    /// Exact prediction-aware visitor for that representation.
    type Visitor: ReplicatedPredictionTargetVisitor<
        B,
        Self::State,
        M,
        Output = Self::Output,
        Error = Self::Error,
    >;

    /// Consumes the adapter into its exact typed visitor.
    fn into_visitor(self) -> Self::Visitor;
}
