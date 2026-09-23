//! Architecture-aware construction of exact backend-neutral checkpoint sources.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use eredu_checkpoint::{
    gguf_store::open_prepared_gguf_source,
    store::{
        CheckpointSource, CompositeCheckpointSource, RestrictedCheckpointSource,
        SharedCheckpointSource, StoreError, TensorMetadata,
    },
    validation::{resolve_gguf_plan, ResolvedCheckpointPlan},
};
use eredu_core::{
    artifact::{
        ArtifactError, ArtifactFile, ArtifactIdentity, DeferredArtifactIdentity, GgufCompanionRole,
    },
    ArtifactFormat, ModelArtifact, ModelPreparationPlan,
};

use crate::{
    configuration::PredictionExtensionPlan, processor_plan::ArtifactArchitecturePlan,
    SelectedPreparation,
};

// Internal projection of the total execution selection onto source admission.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum MediaProjectorSourcePolicy {
    Forbidden,
    Allowed,
}

/// Exact resolved contracts retained with a prepared source graph.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PreparedSourceResolutions {
    primary: ResolvedCheckpointPlan,
    target: ResolvedCheckpointPlan,
    companions: BTreeMap<GgufCompanionRole, ResolvedCheckpointPlan>,
    target_companions: BTreeMap<GgufCompanionRole, ResolvedCheckpointPlan>,
}

impl PreparedSourceResolutions {
    /// Contract used to construct the primary physical source.
    pub const fn primary(&self) -> &ResolvedCheckpointPlan {
        &self.primary
    }

    /// Contract authorizing the ordinary prediction target.
    pub const fn target(&self) -> &ResolvedCheckpointPlan {
        &self.target
    }

    /// Contract used to construct one separately admitted companion source.
    pub fn companion(&self, role: &GgufCompanionRole) -> Option<&ResolvedCheckpointPlan> {
        self.companions.get(role)
    }

    /// All companion contracts in deterministic semantic-role order.
    pub fn companions(
        &self,
    ) -> impl Iterator<Item = (&GgufCompanionRole, &ResolvedCheckpointPlan)> {
        self.companions.iter()
    }

    /// Companion contract that is an explicit member of the target graph.
    pub fn target_companion(&self, role: &GgufCompanionRole) -> Option<&ResolvedCheckpointPlan> {
        self.target_companions.get(role)
    }

    /// All companion contracts consumed by the exact target graph.
    pub fn target_companions(
        &self,
    ) -> impl Iterator<Item = (&GgufCompanionRole, &ResolvedCheckpointPlan)> {
        self.target_companions.iter()
    }
}

/// One exact architecture-selected source graph prepared before native materialization.
///
/// Every logical view shares the same underlying physical source objects. Creating
/// target or extension views therefore cannot multiply reader caches or reopen an
/// admitted artifact.
pub struct PreparedModelSources {
    selected: SelectedPreparation,
    inspection: eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
    graph: PreparedModelSourceGraph,
}

/// Exact source roles released only by consuming their paired total selection.
pub struct PreparedModelSourceGraph {
    source_identity: DeferredArtifactIdentity,
    execution_identity: String,
    format: ArtifactFormat,
    architecture: ArtifactArchitecturePlan,
    prediction_extension: Option<PredictionExtensionPlan>,
    pub(crate) prediction_placement: crate::prediction_extension::PredictionPlacementSlot,
    primary: SharedCheckpointSource,
    companions: BTreeMap<GgufCompanionRole, SharedCheckpointSource>,
    complete: SharedCheckpointSource,
    target: SharedCheckpointSource,
    extension: Option<SharedCheckpointSource>,
    resolutions: PreparedSourceResolutions,
    source_metadata: BTreeMap<String, TensorMetadata>,
}

/// Retained discovery declarations; content hashing happens only on discovery demand.
#[derive(Debug, Clone)]
pub struct PreparedModelDiscovery {
    resource_selection: eredu_runtime::SelectedReplicatedTextRealization,
    generation_memory: Result<eredu_runtime::memory_forecast::LoadedMemoryGeometry, String>,
    identity: DeferredArtifactIdentity,
    execution_identity: String,
    descriptor: eredu_core::ArchitectureDescriptor,
    partition_selection: Option<crate::SelectedExecution>,
    partition_parameters: Option<Arc<eredu_runtime::ArchitectureParameterDescription>>,
    partition_parameter_index: Option<Arc<crate::parameter_partition::ParameterMemberIndex>>,
    partition_hooks: Option<eredu_runtime::inspection::ObservationHookSupport>,
    support: eredu_core::ObservationSupportReport,
    observation_context: eredu_runtime::inspection::ObservationExecutionContext,
    intervention_points: Vec<eredu_core::intervention::InterventionPoint>,
    prediction: Option<PreparedPredictionDiscovery>,
}

#[derive(Debug, Clone)]
struct PreparedPredictionDiscovery {
    resources: Result<eredu_runtime::prediction_resources::EmbeddedPredictionTopology, String>,
    placement: crate::prediction_extension::PredictionPlacementSlot,
    descriptor: eredu_core::ArchitectureDescriptor,
    intervention_points: Vec<eredu_core::intervention::InterventionPoint>,
}

impl PreparedModelDiscovery {
    /// Retained selected geometry for loaded request forecasts; never resolves source identity.
    pub fn generation_memory(
        &self,
    ) -> Result<&eredu_runtime::memory_forecast::LoadedMemoryGeometry, eredu_core::CapabilityError> {
        self.generation_memory
            .as_ref()
            .map_err(|reason| eredu_core::CapabilityError::Observation(reason.clone()))
    }
    /// Retains hook facts projected from the actual constructed executor and
    /// architecture. Backend collectors cannot infer these from parameter shapes.
    pub fn bind_partition_observation_hooks(
        mut self,
        hooks: Option<eredu_runtime::inspection::ObservationHookSupport>,
    ) -> Result<Self, eredu_core::capture::CaptureError> {
        match (&self.partition_selection, &self.partition_parameters, hooks) {
            (Some(_), Some(_), Some(_)) | (None, None, None) => {}
            _ => {
                return Err(eredu_core::capture::CaptureError::Invalid(
                    "partition hook facts differ from the constructed execution".into(),
                ))
            }
        }
        self.partition_hooks = hooks;
        Ok(self)
    }

    /// Pairs retained cold discovery with the global parameter declaration that
    /// the shared partition constructor checked before native materialization.
    /// Backends forward this declaration from the completed session; checkpoint
    /// metadata and locally visited native slots are not substitutes.
    pub fn bind_partition_parameters(
        mut self,
        parameters: Option<Arc<eredu_runtime::ArchitectureParameterDescription>>,
    ) -> Result<Self, eredu_core::capture::CaptureError> {
        use eredu_core::capture::CaptureError;
        match (&self.partition_selection, &parameters) {
            (Some(selected), Some(parameters)) => {
                let requirements = selected.text_realization().requirements();
                if parameters.graph() != requirements.execution_graph()
                    || parameters.unit_layout() != requirements.execution_units()
                {
                    return Err(CaptureError::Invalid(
                        "prepared parameter topology differs from retained execution".into(),
                    ));
                }
            }
            (None, None) => {}
            _ => {
                return Err(CaptureError::Invalid(
                    "prepared partition metadata and retained execution disagree".into(),
                ));
            }
        }
        self.partition_parameter_index = self
            .partition_selection
            .as_ref()
            .zip(parameters.as_ref())
            .map(|(selected, parameters)| {
                Arc::new(crate::parameter_partition::ParameterMemberIndex::new(
                    selected.text_realization().materialization_tasks(),
                    parameters,
                ))
            });
        self.partition_parameters = parameters;
        Ok(self)
    }

    /// Placement retained by successful prediction materialization for these exact
    /// prepared sources. Cold discovery has no constructed placement yet.
    pub fn prediction_placement(
        &self,
    ) -> Option<&Arc<crate::prediction_extension::PreparedPredictionPlacement>> {
        self.prediction.as_ref()?.placement.get()
    }

    /// Retained cold prediction invocation, sharing, state and target-feature facts.
    /// Querying loaded discovery performs no materialization or execution.
    pub fn embedded_prediction_topology(
        &self,
    ) -> Result<
        Option<&eredu_runtime::prediction_resources::EmbeddedPredictionTopology>,
        eredu_core::resources::ResourceDescriptionError,
    > {
        self.prediction
            .as_ref()
            .map(|prediction| {
                prediction.resources.as_ref().map_err(|reason| {
                    eredu_core::resources::ResourceDescriptionError::Invalid(reason.clone())
                })
            })
            .transpose()
    }

    /// Composes ordinary prepared slots, actual prediction-local state and retained
    /// target values. It neither acquires modules nor advances the session.
    pub fn describe_embedded_prediction_resources(
        &self,
        slots: &[eredu_runtime::parameter_operations::PreparedParameterSlot],
        residency: Option<&eredu_runtime::ResidencyReport>,
        query: &eredu_runtime::prediction_resources::PredictionResourceQuery,
    ) -> Result<
        Option<eredu_core::resources::ResourceDescription>,
        eredu_core::resources::ResourceDescriptionError,
    > {
        let Some(topology) = self.embedded_prediction_topology()? else {
            return Ok(None);
        };
        let placement = self.prediction_placement();
        eredu_runtime::prediction_resources::describe_prediction_resources(
            topology,
            &self.resource_selection,
            placement.map(|p| p.state()),
            placement.map_or(&[], |p| p.modules()),
            slots,
            residency,
            query,
        )
        .map(Some)
    }

    /// Exact architecture execution identity, independent of rank-local shapes.
    pub fn execution_identity(&self) -> &str {
        &self.execution_identity
    }

    /// Global selected parameter tasks retained by the actual partition
    /// construction. These are declarations, not evidence of loaded local slots.
    /// Reading them neither resolves sources nor allocates native resources.
    pub fn partition_parameter_tasks(
        &self,
    ) -> Option<impl Iterator<Item = &eredu_runtime::ReplicatedTextMaterializationTask> + Clone>
    {
        self.partition_parameters.as_ref()?;
        let text = self.partition_selection.as_ref()?.text_realization();
        Some(
            text.materialization_tasks()
                .iter()
                .chain(text.auxiliary_materialization_tasks()),
        )
    }

    /// Resolves one global parameter through the retained executable task,
    /// physical placement and pipeline ownership. It never resolves sources or
    /// accesses native values, and charges metadata before allocating it.
    pub fn parameter_partition_layout_for_rank(
        &self,
        parameter: &str,
        global_rank: usize,
        reservation: &mut impl eredu_core::capture::CaptureReservation,
    ) -> Result<
        Option<crate::parameter_partition::ParameterPartitionLayout>,
        eredu_core::parameters::ParameterError,
    > {
        let Some(selected) = &self.partition_selection else {
            return Ok(None);
        };
        let parameters = self.partition_parameters.as_ref().ok_or_else(|| {
            eredu_core::parameters::ParameterError::Invalid(
                "parameter discovery has not been bound to the constructed partition".into(),
            )
        })?;
        let target = selected.parameter_partition_layout_with_index(
            parameters,
            parameter,
            global_rank,
            self.partition_parameter_index.as_deref(),
            reservation,
        );
        match (target, self.prediction_placement()) {
            (Err(eredu_core::parameters::ParameterError::Missing(_)), Some(prediction)) => {
                if selected.parallel_topology().map(|rank| rank.topology())
                    != Some(prediction.topology().topology())
                {
                    return Err(eredu_core::parameters::ParameterError::Invalid(
                        "prediction parameter placement differs from retained target topology"
                            .into(),
                    ));
                }
                crate::partitioned_execution::prediction_parameter_layout_for_rank(
                    prediction,
                    selected
                        .text_realization()
                        .auxiliary_materialization_tasks(),
                    parameter,
                    global_rank,
                    reservation,
                )
                .map(Some)
            }
            (result, _) => result,
        }
    }

    /// Compiles bounded global producer layouts from the actual retained model.
    /// This performs no content hashing, artifact access or native work. Layouts
    /// are declarations; they do not grant capture or communication authority.
    pub fn component_partition_layouts(
        &self,
        max_ranks: usize,
    ) -> Result<
        Option<crate::component_partition::ComponentPartitionLayouts>,
        crate::component_partition::ComponentPartitionError,
    > {
        let Some(selected) = &self.partition_selection else {
            return Ok(None);
        };
        let parameters = self.partition_parameters.as_ref().ok_or_else(|| {
            eredu_core::capture::CaptureError::Invalid(
                "component discovery has not been bound to the constructed partition".into(),
            )
        })?;
        selected.component_partition_layouts(&self.descriptor, parameters, max_ranks)
    }

    /// Combines target ownership with the actual prepared prediction modules.
    /// Each prediction scope keeps its independent head and invocation geometry;
    /// this projection performs no native work and grants no capture authority.
    pub fn speculative_component_partition_layouts(
        &self,
        execution: &crate::speculative_execution::SpeculativeActivationExecution,
        max_ranks: usize,
    ) -> Result<
        Option<crate::component_partition::ComponentPartitionLayouts>,
        crate::component_partition::ComponentPartitionError,
    > {
        let Some(target) = self.component_partition_layouts(max_ranks)? else {
            return Ok(None);
        };
        let prediction = self.prediction.as_ref().ok_or_else(|| {
            eredu_core::capture::CaptureError::Unsupported(
                "prepared sources have no selected prediction catalog".into(),
            )
        })?;
        let placement = prediction.placement.get().ok_or_else(|| {
            eredu_core::capture::CaptureError::Invalid(
                "prediction placement has not been bound by materialization".into(),
            )
        })?;
        target
            .with_prediction(&prediction.descriptor, placement, execution)
            .map(Some)
    }

    /// Resolves the exact content identity for a caller using capture features.
    pub fn capture(
        &self,
    ) -> Result<eredu_core::capture::CaptureDiscovery, eredu_core::capture::CaptureError> {
        Ok(eredu_core::capture::CaptureDiscovery {
            artifact_identity: self
                .identity
                .resolve()
                .map_err(|error| eredu_core::capture::CaptureError::Invalid(error.to_string()))?
                .to_string(),
            catalog: self.descriptor.observations.clone(),
            support: self.support.clone(),
        })
    }

    /// Refines partition support using retained architecture placement and exact
    /// native collector facts. Existing selection/phase/mechanism gates still
    /// apply; a callback cannot enable a disabled instrumented execution route.
    pub fn capture_with_partition_support(
        &self,
        layouts: &crate::component_partition::ComponentPartitionLayouts,
        mut partition: impl FnMut(&eredu_core::ObservationPoint) -> eredu_core::ObservationSupportStatus,
    ) -> Result<eredu_core::capture::CaptureDiscovery, eredu_core::capture::CaptureError> {
        let mut discovery = self.capture()?;
        let mut support = eredu_runtime::inspection::observation_support_with_partition(
            &self.descriptor.observations,
            self.observation_context,
            |point| match layouts
                .capture_hook_support(&point.path, self.partition_hooks.unwrap_or_default())
            {
                eredu_core::ObservationSupportStatus::Supported => partition(point),
                status => status,
            },
        );
        support.capture = self.support.capture.clone();
        discovery.support = support;
        Ok(discovery)
    }

    /// Combines retained semantic points with current native intervention facts.
    pub fn intervention(
        &self,
        mechanisms: &eredu_core::intervention::InterventionMechanisms,
    ) -> Result<eredu_core::intervention::InterventionDiscovery, eredu_core::capture::CaptureError>
    {
        Ok(eredu_runtime::inspection::intervention_support(
            self.intervention_points.clone(),
            &self.capture()?,
            mechanisms,
        ))
    }

    /// Projects invocation-scoped support from actual typed prediction hooks.
    /// Only the declared prediction condition is discharged; media, collector,
    /// phase and partition requirements retain their own admission gates.
    pub fn speculative_activations(
        &self,
        execution: &crate::speculative_execution::SpeculativeActivationExecution,
        mechanisms: &eredu_core::intervention::InterventionMechanisms,
        session_identity: &str,
        active_overlay: Option<&str>,
    ) -> Result<
        eredu_core::speculative::SpeculativeActivationDiscovery,
        eredu_core::capture::CaptureError,
    > {
        self.speculative_activations_inner(
            execution,
            mechanisms,
            session_identity,
            active_overlay,
            None,
            |_| {
                eredu_core::ObservationSupportStatus::Unverified(
                    "partition collector is not bound".into(),
                )
            },
        )
    }

    /// Refines the internal report with exact retained producer layouts and
    /// native collector facts. Prediction hooks come from the sealed execution
    /// contract; target hooks retain their ordinary constructed-executor gate.
    #[allow(clippy::too_many_arguments)]
    pub fn speculative_activations_with_partition_support(
        &self,
        execution: &crate::speculative_execution::SpeculativeActivationExecution,
        mechanisms: &eredu_core::intervention::InterventionMechanisms,
        session_identity: &str,
        active_overlay: Option<&str>,
        layouts: &crate::component_partition::ComponentPartitionLayouts,
        partition: impl FnMut(&eredu_core::ObservationPoint) -> eredu_core::ObservationSupportStatus,
    ) -> Result<
        eredu_core::speculative::SpeculativeActivationDiscovery,
        eredu_core::capture::CaptureError,
    > {
        self.speculative_activations_inner(
            execution,
            mechanisms,
            session_identity,
            active_overlay,
            Some(layouts),
            partition,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn speculative_activations_inner(
        &self,
        execution: &crate::speculative_execution::SpeculativeActivationExecution,
        mechanisms: &eredu_core::intervention::InterventionMechanisms,
        session_identity: &str,
        active_overlay: Option<&str>,
        layouts: Option<&crate::component_partition::ComponentPartitionLayouts>,
        mut partition: impl FnMut(&eredu_core::ObservationPoint) -> eredu_core::ObservationSupportStatus,
    ) -> Result<
        eredu_core::speculative::SpeculativeActivationDiscovery,
        eredu_core::capture::CaptureError,
    > {
        use eredu_core::{capture::CaptureError, speculative::*, ObservationRequirement};
        if self.observation_context.partitioned && layouts.is_none() {
            return Err(CaptureError::Unsupported(
                "partition capture requires invocation-aware producer admission".into(),
            ));
        }
        if let Some(layouts) = layouts {
            if self
                .partition_selection
                .as_ref()
                .and_then(|selected| selected.parallel_topology())
                .map(|rank| rank.topology())
                != Some(layouts.topology())
            {
                return Err(CaptureError::Invalid(
                    "speculative layouts differ from retained execution topology".into(),
                ));
            }
        }
        let prediction = self.prediction.as_ref().ok_or_else(|| {
            CaptureError::Unsupported("prepared sources have no selected prediction catalog".into())
        })?;
        let mut captures = self.capture()?;
        captures.catalog = prediction.descriptor.observations.clone();
        let mut bindings = std::collections::BTreeMap::new();
        for point in &mut captures.catalog.points {
            let scope = crate::speculative_execution::speculative_capture_scope(
                &prediction.descriptor,
                &point.node_id,
            )?;
            execution.validate_scope(scope)?;
            if scope != SpeculativeCaptureScope::Target {
                point
                    .requirements
                    .retain(|r| *r != ObservationRequirement::PredictionExecution);
            }
            bindings.insert(point.node_id.clone(), scope);
        }
        let mut context = self.observation_context;
        context.prediction_inspection = true;
        captures.support = eredu_runtime::inspection::observation_support_with_partition(
            &captures.catalog,
            context,
            |point| {
                use eredu_core::ObservationSupportStatus as Status;
                let Some(layouts) = layouts else {
                    return Status::Unverified("partition layout is absent".into());
                };
                let hooks = match bindings.get(&point.node_id) {
                    Some(SpeculativeCaptureScope::Target) => layouts.capture_hook_support(
                        &point.path,
                        self.partition_hooks.unwrap_or_default(),
                    ),
                    Some(
                        SpeculativeCaptureScope::Prediction { .. }
                        | SpeculativeCaptureScope::PredictionContext
                        | SpeculativeCaptureScope::FusedProposal,
                    ) if layouts.observation_site(&point.path).is_some() => Status::Supported,
                    _ => Status::Unverified(
                        "selected prediction has no producer declaration for this point".into(),
                    ),
                };
                match hooks {
                    Status::Supported => partition(point),
                    status => status,
                }
            },
        );
        captures.support.capture = self.support.capture.clone();
        // Keep the semantic declaration; only this report discharges its
        // prediction requirement for the scoped execution.
        captures.catalog = prediction.descriptor.observations.clone();
        let mut interventions = eredu_runtime::inspection::intervention_support(
            prediction.intervention_points.clone(),
            &captures,
            mechanisms,
        );
        for point in &interventions.points {
            let scope = crate::speculative_execution::speculative_capture_scope(
                &prediction.descriptor,
                &point.node_id,
            )?;
            bindings.insert(point.node_id.clone(), scope);
        }
        interventions.session_identity = Some(session_identity.to_owned());
        Ok(SpeculativeActivationDiscovery {
            schema_version: SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
            execution_identity: serde_json::to_string(&(self.execution_identity(), active_overlay))
                .map_err(|error| CaptureError::Invalid(error.to_string()))?,
            captures,
            interventions,
            bindings: bindings
                .into_iter()
                .map(|(node_id, scope)| SpeculativeCaptureBinding { node_id, scope })
                .collect(),
        })
    }

    /// Whether exact content identity has been requested for this source graph.
    pub fn identity_is_resolved(&self) -> bool {
        self.identity.is_resolved()
    }
}

impl PreparedModelSources {
    /// Retains architecture-owned mutable hooks with actual selected-session facts.
    pub fn intervention_discovery(
        &self,
        capture: &eredu_core::capture::CaptureDiscovery,
        mechanisms: &eredu_core::intervention::InterventionMechanisms,
    ) -> eredu_core::intervention::InterventionDiscovery {
        eredu_runtime::inspection::intervention_support(
            self.architecture().intervention_points(),
            capture,
            mechanisms,
        )
    }

    /// Authoritative total selection inseparably paired with these exact sources.
    pub const fn selected(&self) -> &SelectedPreparation {
        &self.selected
    }

    /// Projects catalog semantics from the retained architecture and combines them
    /// with side-effect-free backend capture facts for this exact selection.
    pub fn prepare_discovery(
        &self,
        mechanisms: eredu_core::ObservationMechanisms,
        capture: eredu_core::capture::CaptureCapabilities,
    ) -> PreparedModelDiscovery {
        let descriptor = self.architecture().architecture_descriptor();
        let observation_context = eredu_runtime::inspection::ObservationExecutionContext {
            activation_inspection: self
                .selected()
                .session_capabilities()
                .activation_inspection(),
            prediction_inspection: false,
            partitioned: self.selected().execution().parallel_topology().is_some(),
            selected: true,
            mechanisms,
        };
        let mut support = eredu_runtime::inspection::observation_support(
            &descriptor.observations,
            observation_context,
        );
        support.capture = capture;
        PreparedModelDiscovery {
            resource_selection: self.selected().text_realization().clone(),
            generation_memory: crate::memory_estimation::selected_generation_memory_geometry(
                self.architecture(),
                self.selected().execution(),
            )
            .map(|mut geometry| {
                if self.prediction_extension().is_some() {
                    geometry.workspace = None;
                    geometry.execution_topology = None;
                }
                geometry
            })
            .map_err(|error| error.to_string()),
            identity: self.graph.source_identity().clone(),
            execution_identity: self.execution_identity().to_owned(),
            descriptor,
            partition_selection: self
                .selected()
                .execution()
                .parallel_topology()
                .map(|_| self.selected().execution().clone()),
            partition_parameters: None,
            partition_parameter_index: None,
            partition_hooks: None,
            support,
            observation_context,
            intervention_points: self.architecture().intervention_points(),
            prediction: self.prediction_extension().map(|_| {
                let complete = self.inspection.architecture_plan();
                PreparedPredictionDiscovery {
                    resources: self
                        .selected()
                        .embedded_prediction_topology()
                        .map_err(|error| error.to_string())
                        .and_then(|topology| {
                            topology
                                .ok_or_else(|| "selected prediction topology is missing".to_owned())
                        }),
                    placement: Arc::clone(&self.graph.prediction_placement),
                    descriptor: complete.architecture_descriptor(),
                    intervention_points: complete.intervention_points(),
                }
            }),
        }
    }

    /// Resolves exact model identity for explicitly requested capture discovery.
    pub fn capture_discovery(
        &self,
        mechanisms: eredu_core::ObservationMechanisms,
        capture: eredu_core::capture::CaptureCapabilities,
    ) -> Result<eredu_core::capture::CaptureDiscovery, eredu_core::capture::CaptureError> {
        self.prepare_discovery(mechanisms, capture).capture()
    }

    /// Consumes the authoritative pairing immediately before typed execution dispatch.
    pub(crate) fn into_parts(
        self,
    ) -> (
        SelectedPreparation,
        eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
        PreparedModelSourceGraph,
    ) {
        (self.selected, self.inspection, self.graph)
    }

    /// Exact prepared source graph paired with the selection.
    pub const fn graph(&self) -> &PreparedModelSourceGraph {
        &self.graph
    }

    /// Computes and retains the exact source identity when explicitly requested.
    pub fn source_identity(&self) -> Result<ArtifactIdentity, Arc<ArtifactError>> {
        self.graph.source_identity().resolve()
    }

    /// Architecture identity retained by the selected neutral execution requirements.
    pub fn execution_identity(&self) -> &str {
        self.graph.execution_identity()
    }

    /// Admitted physical container format.
    pub const fn format(&self) -> ArtifactFormat {
        self.graph.format()
    }

    /// Target architecture paired with these exact source roles.
    pub const fn architecture(&self) -> &ArtifactArchitecturePlan {
        self.graph.architecture()
    }

    /// Selected embedded prediction extension, when requested and admitted.
    pub const fn prediction_extension(&self) -> Option<&PredictionExtensionPlan> {
        self.graph.prediction_extension()
    }

    /// Primary artifact source, excluding separately stored companions.
    pub fn primary(&self) -> &SharedCheckpointSource {
        self.graph.primary()
    }

    /// Separately stored source for one architecture-declared semantic role.
    pub fn companion(&self, role: &GgufCompanionRole) -> Option<&SharedCheckpointSource> {
        self.graph.companion(role)
    }

    /// Separately stored companions in deterministic semantic-role order.
    pub fn companions(
        &self,
    ) -> impl Iterator<Item = (&GgufCompanionRole, &SharedCheckpointSource)> {
        self.graph.companions()
    }

    /// Complete selected source graph used while materializing auxiliary roles.
    pub fn complete(&self) -> &SharedCheckpointSource {
        self.graph.complete()
    }

    /// Explicit ordinary-target projection, which cannot expose extension-only keys.
    pub fn target(&self) -> &SharedCheckpointSource {
        self.graph.target()
    }

    /// Explicit extension-only projection, when embedded prediction was selected.
    pub fn extension(&self) -> Option<&SharedCheckpointSource> {
        self.graph.extension()
    }

    /// Exact resolved contracts used to build and project this source graph.
    pub const fn resolutions(&self) -> &PreparedSourceResolutions {
        self.graph.resolutions()
    }

    /// Metadata snapshot for every key in the complete selected source graph.
    pub const fn source_metadata(&self) -> &BTreeMap<String, TensorMetadata> {
        self.graph.source_metadata()
    }
}

impl PreparedModelSourceGraph {
    /// Deferred identity of the complete admitted physical source graph.
    pub const fn source_identity(&self) -> &DeferredArtifactIdentity {
        &self.source_identity
    }

    /// Architecture identity retained by the selected neutral execution requirements.
    pub fn execution_identity(&self) -> &str {
        &self.execution_identity
    }

    /// Admitted physical container format.
    pub const fn format(&self) -> ArtifactFormat {
        self.format
    }

    /// Target architecture paired with these exact source roles.
    pub const fn architecture(&self) -> &ArtifactArchitecturePlan {
        &self.architecture
    }

    /// Selected embedded prediction extension, when requested and admitted.
    pub const fn prediction_extension(&self) -> Option<&PredictionExtensionPlan> {
        self.prediction_extension.as_ref()
    }

    /// Primary artifact source, excluding separately stored companions.
    pub fn primary(&self) -> &SharedCheckpointSource {
        &self.primary
    }

    /// Separately stored source for one architecture-declared semantic role.
    pub fn companion(&self, role: &GgufCompanionRole) -> Option<&SharedCheckpointSource> {
        self.companions.get(role)
    }

    /// Separately stored companions in deterministic semantic-role order.
    pub fn companions(
        &self,
    ) -> impl Iterator<Item = (&GgufCompanionRole, &SharedCheckpointSource)> {
        self.companions.iter()
    }

    /// Complete selected source graph used while materializing auxiliary roles.
    pub fn complete(&self) -> &SharedCheckpointSource {
        &self.complete
    }

    /// Explicit ordinary-target projection, which cannot expose extension-only keys.
    pub fn target(&self) -> &SharedCheckpointSource {
        &self.target
    }

    /// Explicit extension-only projection, when embedded prediction was selected.
    pub fn extension(&self) -> Option<&SharedCheckpointSource> {
        self.extension.as_ref()
    }

    /// Exact resolved contracts used to build and project this source graph.
    pub const fn resolutions(&self) -> &PreparedSourceResolutions {
        &self.resolutions
    }

    /// Metadata snapshot for every key in the complete selected source graph.
    pub const fn source_metadata(&self) -> &BTreeMap<String, TensorMetadata> {
        &self.source_metadata
    }
}

/// Failure while converting one selected portable artifact into exact neutral sources.
#[derive(Debug, thiserror::Error)]
pub enum PreparedModelSourcesError {
    /// The artifact and its architecture-owned admission proof disagree.
    #[error("invalid prepared source selection: {0}")]
    InvalidSelection(String),
    /// Architecture projection or artifact identity validation failed.
    #[error(transparent)]
    Artifact(#[from] ArtifactError),
    /// An exact checkpoint source or logical view could not be constructed.
    #[error(transparent)]
    Store(#[from] StoreError),
}

/// Consumes one admitted artifact into the only architecture-aware prepared source graph.
///
/// This operation performs the SafeTensors/GGUF choice once, constructs every
/// admitted physical source once, and publishes no backend-native resource.
/// Tensor payload conversion remains lazy behind checkpoint leases.
pub fn prepare_model_sources(
    plan: ModelPreparationPlan<ArtifactArchitecturePlan>,
    selected: SelectedPreparation,
) -> Result<PreparedModelSources, PreparedModelSourcesError> {
    if !selected
        .admission_token()
        .same_admission(&plan.inspection().admission_token())
    {
        return Err(PreparedModelSourcesError::InvalidSelection(
            "selected preparation originated from a different artifact inspection".into(),
        ));
    }
    if plan.admitted_session_capabilities() != selected.session_capabilities() {
        return Err(PreparedModelSourcesError::InvalidSelection(
            "selected session facilities differ from the admitted preparation plan".into(),
        ));
    }
    let selected_admission = selected.admission();
    if plan.policy() != selected_admission.request().policy()
        || plan.route() != selected_admission.route()
    {
        return Err(PreparedModelSourcesError::InvalidSelection(
            "selected preparation policy or route differs from the admitted preparation plan"
                .into(),
        ));
    }
    let max_cached_sources = selected.text_realization().max_cached_shards();
    let media_projector = if selected.allows_media_projector() {
        MediaProjectorSourcePolicy::Allowed
    } else {
        MediaProjectorSourcePolicy::Forbidden
    };
    let selected_prediction_extension = selected.prediction_extension().cloned();
    let execution_identity = selected
        .text_realization()
        .requirements()
        .architecture_identity()
        .to_owned();
    let inspection = plan.inspection().clone();
    let complete_architecture = inspection.architecture_plan().clone();
    let projection = complete_architecture.prediction_target_projection()?;
    let (architecture, admitted_extension) = projection.map_or_else(
        || (complete_architecture.clone(), None),
        |(target, extension)| (target, Some(extension)),
    );
    let prediction_extension = match (admitted_extension, selected_prediction_extension) {
        (Some(admitted), Some(selected)) if admitted.same_admission(&selected) => Some(selected),
        (Some(_), Some(_)) => {
            return Err(PreparedModelSourcesError::InvalidSelection(
                "selected prediction extension differs from artifact admission".into(),
            ))
        }
        (None, Some(_)) => {
            return Err(PreparedModelSourcesError::InvalidSelection(
                "selected prediction extension has no artifact admission".into(),
            ))
        }
        (_, None) => None,
    };
    // Keep the source graph's construction requirements identical to cold
    // selection when the request disables the artifact's additive predictor.
    let architecture = if prediction_extension.is_some() {
        architecture
    } else {
        architecture.without_prediction_extension()
    };
    let execution_inspection = plan
        .inspection()
        .clone()
        .map_architecture_plan(|_| architecture.clone());
    let expected_execution_identity =
        match crate::replicated_text::replicated_text_execution_class(&execution_inspection)
            .map_err(|error| PreparedModelSourcesError::InvalidSelection(error.to_string()))?
        {
            crate::replicated_text::ReplicatedTextExecutionClass::Replicated(requirements) => {
                requirements.architecture_identity().to_owned()
            }
            crate::replicated_text::ReplicatedTextExecutionClass::Routed(requirements) => {
                requirements.text().architecture_identity().to_owned()
            }
            crate::replicated_text::ReplicatedTextExecutionClass::Composite(requirements) => {
                requirements.execution().architecture_identity().to_owned()
            }
        };
    if execution_identity != expected_execution_identity {
        return Err(PreparedModelSourcesError::InvalidSelection(format!(
            "selected execution identity {execution_identity:?} does not match inspected artifact identity {expected_execution_identity:?}"
        )));
    }
    let source_identity = source_graph_identity(plan.inspection(), &execution_identity)?;

    let graph = match plan.into_artifact() {
        ModelArtifact::SafeTensors {
            tensors, shards, ..
        } => prepare_safetensors_sources(
            source_identity,
            execution_identity,
            architecture,
            prediction_extension,
            tensors,
            shards,
            max_cached_sources,
        ),
        ModelArtifact::Gguf { validated, .. } => {
            if prediction_extension.is_some() {
                return Err(PreparedModelSourcesError::InvalidSelection(
                    "GGUF artifacts do not admit embedded prediction source projections".into(),
                ));
            }
            prepare_gguf_sources(
                source_identity,
                execution_identity,
                architecture,
                validated,
                max_cached_sources,
                media_projector,
            )
        }
        _ => Err(PreparedModelSourcesError::InvalidSelection(
            "unsupported artifact format for prepared model sources".into(),
        )),
    }?;
    Ok(PreparedModelSources {
        selected,
        inspection,
        graph,
    })
}

fn source_graph_identity(
    inspection: &eredu_core::ArtifactInspection<ArtifactArchitecturePlan>,
    execution_identity: &str,
) -> Result<DeferredArtifactIdentity, ArtifactError> {
    match inspection.format() {
        ArtifactFormat::SafeTensors => DeferredArtifactIdentity::safetensors(
            execution_identity,
            inspection.safetensors_shards().ok_or_else(|| {
                ArtifactError::InvalidArtifact(
                    "SafeTensors inspection omitted admitted shards".into(),
                )
            })?,
        ),
        ArtifactFormat::Gguf => {
            let validated = inspection.validated_gguf().ok_or_else(|| {
                ArtifactError::InvalidArtifact("GGUF inspection omitted its admission proof".into())
            })?;
            let mut files = Vec::new();
            files.extend(validated.checkpoint().shards().iter().map(|shard| {
                ArtifactFile::new(
                    format!("primary/split/{:05}", shard.split_no()),
                    shard.path(),
                )
            }));
            for (role, companion) in validated.companions() {
                let role = match role {
                    GgufCompanionRole::MediaProjector => "media-projector".to_owned(),
                    GgufCompanionRole::Named(name) => format!("named/{name}"),
                    _ => {
                        return Err(ArtifactError::InvalidArtifact(
                            "unsupported GGUF companion role in source identity".into(),
                        ))
                    }
                };
                files.extend(companion.checkpoint().shards().iter().map(|shard| {
                    ArtifactFile::new(
                        format!("companion/{role}/split/{:05}", shard.split_no()),
                        shard.path(),
                    )
                }));
            }
            DeferredArtifactIdentity::filesystem(execution_identity, files)
        }
        _ => Err(ArtifactError::InvalidArtifact(
            "unsupported artifact format in prepared source identity".into(),
        )),
    }
}

fn prepare_safetensors_sources(
    source_identity: DeferredArtifactIdentity,
    execution_identity: String,
    architecture: ArtifactArchitecturePlan,
    prediction_extension: Option<PredictionExtensionPlan>,
    tensors: eredu_core::checkpoint::TensorCatalog,
    shards: eredu_checkpoint::safetensors::SafetensorsShards,
    max_cached_shards: usize,
) -> Result<PreparedModelSourceGraph, PreparedModelSourcesError> {
    let target_architecture = architecture.safetensors_architecture().ok_or_else(|| {
        PreparedModelSourcesError::InvalidSelection(
            "SafeTensors artifact omitted its architecture plan".into(),
        )
    })?;
    let target_resolution = target_architecture
        .checkpoint_resolution()
        .ok_or_else(|| {
            PreparedModelSourcesError::InvalidSelection(
                "SafeTensors target omitted its admitted checkpoint resolution".into(),
            )
        })?
        .clone();
    let source_resolution = prediction_extension
        .as_ref()
        .map(PredictionExtensionPlan::complete_architecture)
        .unwrap_or(target_architecture)
        .checkpoint_resolution()
        .ok_or_else(|| {
            PreparedModelSourcesError::InvalidSelection(
                "SafeTensors source omitted its admitted checkpoint resolution".into(),
            )
        })?
        .clone();
    let primary = eredu_core::artifact::open_prepared_safetensors_artifact(
        &tensors,
        shards,
        source_resolution.clone(),
        max_cached_shards,
    )?;
    let complete = Arc::clone(&primary);
    let (target, extension) = match prediction_extension.as_ref() {
        Some(extension) => {
            let extension_keys = extension.source_keys(target_architecture)?;
            let target_keys = target_resolution.source_keys().clone();
            projected_prediction_views(&complete, target_keys, extension_keys)?
        }
        None => (Arc::clone(&complete), None),
    };
    let source_metadata = metadata_snapshot(complete.as_ref())?;
    Ok(PreparedModelSourceGraph {
        prediction_placement: Arc::default(),
        source_identity,
        execution_identity,
        format: ArtifactFormat::SafeTensors,
        architecture,
        prediction_extension,
        primary,
        companions: BTreeMap::new(),
        complete,
        target,
        extension,
        resolutions: PreparedSourceResolutions {
            primary: source_resolution,
            target: target_resolution,
            companions: BTreeMap::new(),
            target_companions: BTreeMap::new(),
        },
        source_metadata,
    })
}

fn prepare_gguf_sources(
    source_identity: DeferredArtifactIdentity,
    execution_identity: String,
    architecture: ArtifactArchitecturePlan,
    validated: eredu_core::ValidatedGguf,
    max_cached_readers: usize,
    media_projector: MediaProjectorSourcePolicy,
) -> Result<PreparedModelSourceGraph, PreparedModelSourcesError> {
    let primary_plan = architecture.gguf_plan().ok_or_else(|| {
        PreparedModelSourcesError::InvalidSelection(
            "GGUF artifact omitted its architecture plan".into(),
        )
    })?;
    let projector_plan = architecture.gguf_media_projector();
    let (checkpoint, mut admitted_companions) = validated.into_parts();
    let admitted_projector = admitted_companions.remove(&GgufCompanionRole::MediaProjector);
    if let Some(role) = admitted_companions.keys().next() {
        return Err(PreparedModelSourcesError::InvalidSelection(format!(
            "GGUF artifact retained an unsupported companion role {role:?}"
        )));
    }
    if media_projector == MediaProjectorSourcePolicy::Forbidden
        && (projector_plan.is_some() || admitted_projector.is_some())
    {
        return Err(PreparedModelSourcesError::InvalidSelection(
            "selected execution cannot consume the admitted GGUF media projector".into(),
        ));
    }
    let admitted_projector = match (projector_plan, admitted_projector) {
        (Some(plan), Some(checkpoint)) => Some((plan, checkpoint)),
        (None, None) => None,
        (Some(_), None) => {
            return Err(PreparedModelSourcesError::InvalidSelection(
                "GGUF projector plan omitted its admitted companion checkpoint".into(),
            ))
        }
        (None, Some(_)) => {
            return Err(PreparedModelSourcesError::InvalidSelection(
                "GGUF projector checkpoint omitted its typed architecture plan".into(),
            ))
        }
    };

    let primary_mapping = admitted_projector
        .as_ref()
        .map_or(primary_plan.tensor_mapping(), |(plan, _)| {
            plan.primary_tensor_mapping()
        });
    let primary_resolution =
        resolve_gguf_plan(&checkpoint, primary_plan.checkpoint()).map_err(|validation| {
            PreparedModelSourcesError::InvalidSelection(format!(
                "GGUF primary checkpoint contract no longer resolves: {validation:?}"
            ))
        })?;
    let primary: SharedCheckpointSource = Arc::new(open_prepared_gguf_source(
        checkpoint,
        primary_plan.checkpoint(),
        primary_mapping,
        max_cached_readers,
    )?);
    let mut companions = BTreeMap::new();
    let mut companion_resolutions = BTreeMap::new();
    if let Some((plan, admitted)) = admitted_projector {
        let resolution =
            resolve_gguf_plan(admitted.checkpoint(), plan.checkpoint()).map_err(|validation| {
                PreparedModelSourcesError::InvalidSelection(format!(
                    "GGUF media-projector contract no longer resolves: {validation:?}"
                ))
            })?;
        let source: SharedCheckpointSource = Arc::new(open_prepared_gguf_source(
            admitted.checkpoint().clone(),
            plan.checkpoint(),
            plan.tensor_mapping(),
            max_cached_readers,
        )?);
        companions.insert(GgufCompanionRole::MediaProjector, source);
        companion_resolutions.insert(GgufCompanionRole::MediaProjector, resolution);
    }
    let complete = if companions.is_empty() {
        Arc::clone(&primary)
    } else {
        Arc::new(CompositeCheckpointSource::new(
            std::iter::once(Arc::clone(&primary)).chain(companions.values().cloned()),
        )?)
    };
    let source_metadata = metadata_snapshot(complete.as_ref())?;
    let target_companion_resolutions = companion_resolutions.clone();
    Ok(PreparedModelSourceGraph {
        prediction_placement: Arc::default(),
        source_identity,
        execution_identity,
        format: ArtifactFormat::Gguf,
        architecture,
        prediction_extension: None,
        primary: Arc::clone(&primary),
        companions,
        complete: Arc::clone(&complete),
        target: complete,
        extension: None,
        resolutions: PreparedSourceResolutions {
            primary: primary_resolution.clone(),
            target: primary_resolution,
            companions: companion_resolutions,
            target_companions: target_companion_resolutions,
        },
        source_metadata,
    })
}

fn projected_prediction_views(
    complete: &SharedCheckpointSource,
    target_keys: BTreeSet<String>,
    extension_keys: BTreeSet<String>,
) -> Result<(SharedCheckpointSource, Option<SharedCheckpointSource>), StoreError> {
    if !target_keys.is_disjoint(&extension_keys) {
        return Err(StoreError::Internal(
            "prediction target and extension source projections overlap".into(),
        ));
    }
    let complete_keys = complete.source_keys().into_iter().collect::<BTreeSet<_>>();
    let projected = target_keys
        .union(&extension_keys)
        .cloned()
        .collect::<BTreeSet<_>>();
    if projected != complete_keys {
        return Err(StoreError::Internal(
            "prediction target and extension projections do not cover the selected source".into(),
        ));
    }
    let target: SharedCheckpointSource = Arc::new(RestrictedCheckpointSource::including(
        Arc::clone(complete),
        "prediction-target",
        target_keys,
    )?);
    let extension: SharedCheckpointSource = Arc::new(RestrictedCheckpointSource::including(
        Arc::clone(complete),
        "prediction-extension",
        extension_keys,
    )?);
    Ok((target, Some(extension)))
}

fn metadata_snapshot(
    source: &dyn CheckpointSource,
) -> Result<BTreeMap<String, TensorMetadata>, StoreError> {
    source
        .source_keys()
        .into_iter()
        .map(|key| source.source_metadata(&key).map(|metadata| (key, metadata)))
        .collect()
}

#[cfg(test)]
#[path = "prepared_execution/source_contract_tests.rs"]
mod construction_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_checkpoint::store::{
        CheckpointLease, EncodedTensorLease, MemoryWeightStore, ReadPolicy, TensorReadRequest,
        TensorSelection, WeightStoreDiagnostics,
    };
    use safetensors::tensor::Dtype;
    use std::{
        collections::{BTreeMap, HashMap, HashSet},
        fs::File,
        sync::atomic::{AtomicUsize, Ordering},
    };

    fn write_f32_gguf(
        path: &std::path::Path,
        metadata: &BTreeMap<String, eredu_gguf::MetadataValue>,
        plan: &eredu_checkpoint::schema::GgufCheckpointPlan,
    ) {
        use eredu_gguf::{GgmlType, TensorInput, Writer};

        let tensors = plan
            .common_tensors
            .iter()
            .chain(
                plan.layout_groups
                    .iter()
                    .filter(|group| group.required)
                    .filter_map(|group| group.variants.first())
                    .flat_map(|variant| variant.tensors.iter()),
            )
            .filter(|constraint| {
                constraint.requirement == eredu_checkpoint::schema::TensorRequirement::Required
            })
            .map(|constraint| {
                let dimensions = constraint
                    .shape
                    .iter()
                    .rev()
                    .map(|dimension| u64::try_from(*dimension).unwrap())
                    .collect::<Vec<_>>();
                let bytes = vec![0_u8; constraint.shape.iter().product::<usize>() * 4];
                (constraint.key.clone(), dimensions, bytes)
            })
            .collect::<Vec<_>>();
        let inputs = tensors
            .iter()
            .map(|(name, dimensions, bytes)| TensorInput {
                name,
                dimensions,
                ggml_type: GgmlType::F32,
                data: bytes,
            })
            .collect::<Vec<_>>();
        Writer::default()
            .write(File::create(path).unwrap(), metadata, &inputs)
            .unwrap();
    }

    fn gemma4_gguf_fixture() -> tempfile::TempDir {
        use eredu_gguf::{MetadataArray, MetadataValue};

        let root = tempfile::tempdir().unwrap();
        let model_metadata = BTreeMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String("gemma4".into()),
            ),
            ("gemma4.block_count".into(), MetadataValue::Uint32(2)),
            ("gemma4.embedding_length".into(), MetadataValue::Uint32(4)),
            (
                "gemma4.attention.head_count".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "gemma4.attention.key_length".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "gemma4.attention.key_length_swa".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "gemma4.attention.shared_kv_layers".into(),
                MetadataValue::Uint32(0),
            ),
            (
                "gemma4.attention.layer_norm_rms_epsilon".into(),
                MetadataValue::Float32(1e-6),
            ),
            ("gemma4.vocab_size".into(), MetadataValue::Uint32(8)),
            ("gemma4.context_length".into(), MetadataValue::Uint32(32)),
            (
                "gemma4.final_logit_softcapping".into(),
                MetadataValue::Float32(30.0),
            ),
            (
                "gemma4.feed_forward_length".into(),
                MetadataValue::Array(MetadataArray::Uint32(vec![8, 8])),
            ),
            (
                "gemma4.attention.head_count_kv".into(),
                MetadataValue::Array(MetadataArray::Uint32(vec![1, 1])),
            ),
            (
                "gemma4.attention.sliding_window_pattern".into(),
                MetadataValue::Array(MetadataArray::Bool(vec![false, false])),
            ),
            ("gemma4.image_token_id".into(), MetadataValue::Int32(2)),
        ]);
        let model_hash = model_metadata
            .clone()
            .into_iter()
            .collect::<HashMap<_, _>>();
        let catalog = HashSet::from([
            "output.weight".to_owned(),
            "blk.0.attn_k.weight".to_owned(),
            "blk.0.attn_v.weight".to_owned(),
            "blk.1.attn_k.weight".to_owned(),
            "blk.1.attn_v.weight".to_owned(),
        ]);
        let text = crate::gemma4::ModelArgs::from_gguf_metadata(&catalog, &model_hash).unwrap();
        let model_plan = crate::gemma4::gguf_plan(&text).unwrap();
        write_f32_gguf(
            &root.path().join("model.gguf"),
            &model_metadata,
            &model_plan,
        );

        let projector_metadata = BTreeMap::from([
            (
                "general.architecture".into(),
                MetadataValue::String("clip".into()),
            ),
            (
                "general.type".into(),
                MetadataValue::String("mmproj".into()),
            ),
            ("clip.has_vision_encoder".into(), MetadataValue::Bool(true)),
            ("clip.has_audio_encoder".into(), MetadataValue::Bool(false)),
            (
                "clip.vision.projector_type".into(),
                MetadataValue::String("gemma4".into()),
            ),
            (
                "clip.vision.embedding_length".into(),
                MetadataValue::Uint32(4),
            ),
            (
                "clip.vision.feed_forward_length".into(),
                MetadataValue::Uint32(8),
            ),
            ("clip.vision.block_count".into(), MetadataValue::Uint32(1)),
            (
                "clip.vision.attention.head_count".into(),
                MetadataValue::Uint32(1),
            ),
            (
                "clip.vision.attention.head_count_kv".into(),
                MetadataValue::Uint32(1),
            ),
            (
                "clip.vision.attention.key_length".into(),
                MetadataValue::Uint32(4),
            ),
            ("clip.vision.patch_size".into(), MetadataValue::Uint32(2)),
            (
                "clip.vision.pooling_kernel_size".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "clip.vision.position_embedding_size".into(),
                MetadataValue::Uint32(4),
            ),
            (
                "clip.vision.attention.layer_norm_rms_epsilon".into(),
                MetadataValue::Float32(1e-6),
            ),
        ]);
        let projector_hash = projector_metadata
            .clone()
            .into_iter()
            .collect::<HashMap<_, _>>();
        let family =
            crate::gemma4::family_from_gguf_metadata(text, &model_hash, Some(&projector_hash))
                .unwrap();
        let projector_plan = crate::gemma4::mmproj_gguf_plan(&family).unwrap();
        write_f32_gguf(
            &root.path().join("mmproj.gguf"),
            &projector_metadata,
            &projector_plan,
        );
        root
    }

    struct LeaseCountingSource {
        source: MemoryWeightStore,
        acquisitions: Arc<AtomicUsize>,
    }

    impl CheckpointSource for LeaseCountingSource {
        fn source_keys(&self) -> Vec<String> {
            self.source.source_keys()
        }

        fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            self.source.source_metadata(key)
        }

        fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
            self.acquisitions.fetch_add(1, Ordering::Relaxed);
            self.source.acquire_lease(request)
        }

        fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
            self.source.source_diagnostics()
        }
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    #[test]
    fn prediction_views_are_disjoint_exact_and_share_one_source() {
        let acquisitions = Arc::new(AtomicUsize::new(0));
        let source: SharedCheckpointSource = Arc::new(LeaseCountingSource {
            source: MemoryWeightStore::from_safetensors([
                (
                    "target.weight".into(),
                    Dtype::F32,
                    vec![1],
                    f32_bytes(&[1.0]),
                ),
                (
                    "extension.weight".into(),
                    Dtype::F32,
                    vec![1],
                    f32_bytes(&[2.0]),
                ),
            ])
            .unwrap(),
            acquisitions: Arc::clone(&acquisitions),
        });
        let (target, extension) = projected_prediction_views(
            &source,
            BTreeSet::from(["target.weight".into()]),
            BTreeSet::from(["extension.weight".into()]),
        )
        .unwrap();
        let extension = extension.unwrap();

        assert_eq!(target.source_keys(), ["target.weight"]);
        assert_eq!(extension.source_keys(), ["extension.weight"]);
        assert!(target.source_metadata("extension.weight").is_err());
        assert!(extension.source_metadata("target.weight").is_err());
        assert_eq!(
            target
                .acquire_lease(TensorReadRequest {
                    key: "target.weight".into(),
                    selection: TensorSelection::Full,
                    policy: ReadPolicy::RequireBounded,
                })
                .unwrap()
                .encoded_bytes()
                .unwrap(),
            f32_bytes(&[1.0])
        );
        assert_eq!(
            extension
                .acquire_lease(TensorReadRequest {
                    key: "extension.weight".into(),
                    selection: TensorSelection::Full,
                    policy: ReadPolicy::RequireBounded,
                })
                .unwrap()
                .encoded_bytes()
                .unwrap(),
            f32_bytes(&[2.0])
        );
        assert_eq!(acquisitions.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn prediction_views_reject_overlap_and_incomplete_partition() {
        let source: SharedCheckpointSource = Arc::new(
            MemoryWeightStore::from_safetensors([
                ("a".into(), Dtype::F32, vec![1], f32_bytes(&[1.0])),
                ("b".into(), Dtype::F32, vec![1], f32_bytes(&[2.0])),
            ])
            .unwrap(),
        );
        assert!(projected_prediction_views(
            &source,
            BTreeSet::from(["a".into()]),
            BTreeSet::from(["a".into(), "b".into()]),
        )
        .is_err());
        assert!(
            projected_prediction_views(&source, BTreeSet::from(["a".into()]), BTreeSet::new(),)
                .is_err()
        );
    }

    #[test]
    fn gguf_primary_target_and_companion_roles_are_exact_and_payload_lazy() {
        let root = gemma4_gguf_fixture();
        let inspection =
            crate::configuration::inspect_artifact(root.path().join("model.gguf")).unwrap();
        let selected = crate::select_preparation(
            &inspection,
            &eredu_runtime::NormalizedLoadRequest::default(),
            &crate::preparation_selection::tests::BoundedIndependentAdapter::default(),
        )
        .unwrap();
        let plan = eredu_core::plan_model_preparation(
            inspection,
            eredu_core::PreparationPolicy::new(None, eredu_core::ResidencyRequest::FullyResident),
            selected.session_capabilities(),
        )
        .unwrap();
        let sources = prepare_model_sources(plan, selected).unwrap();

        let role = GgufCompanionRole::MediaProjector;
        let primary = sources.primary().source_keys();
        let companion = sources.companion(&role).unwrap().source_keys();
        let complete = sources.complete().source_keys();
        assert_eq!(sources.format(), ArtifactFormat::Gguf);
        assert!(!primary.is_empty());
        assert!(!companion.is_empty());
        assert!(primary.iter().all(|key| !companion.contains(key)));
        assert_eq!(complete.len(), primary.len() + companion.len());
        assert_eq!(sources.target().source_keys(), complete);
        assert!(sources.extension().is_none());
        assert!(sources.resolutions().companion(&role).is_some());
        for source in [
            sources.primary(),
            sources.companion(&role).unwrap(),
            sources.target(),
        ] {
            let diagnostics = source.source_diagnostics().unwrap();
            assert_eq!(diagnostics.physical_reads, 0);
            assert_eq!(diagnostics.physical_read_bytes, 0);
            assert!(diagnostics.payload_shard_paths.is_empty());
        }

        let forbidden =
            crate::configuration::inspect_artifact(root.path().join("model.gguf")).unwrap();
        let (_llama_root, llama) = crate::preparation_selection::tests::inspected_llama();
        let selected = crate::select_preparation(
            &llama,
            &eredu_runtime::NormalizedLoadRequest::default(),
            &crate::preparation_selection::tests::BoundedIndependentAdapter::default(),
        )
        .unwrap();
        let forbidden = eredu_core::plan_model_preparation(
            forbidden,
            eredu_core::PreparationPolicy::new(None, eredu_core::ResidencyRequest::FullyResident),
            selected.session_capabilities(),
        )
        .unwrap();
        assert!(matches!(
            prepare_model_sources(forbidden, selected),
            Err(PreparedModelSourcesError::InvalidSelection(_))
        ));
    }
}
