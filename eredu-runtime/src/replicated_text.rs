//! Selection contracts for replicated text architectures.

use std::collections::BTreeSet;

use eredu_checkpoint::{LinearFormat, SourceTensorEncoding};
use eredu_core::{ParallelTopology, QuantizationRequest, ResidencyRequest, SessionCapabilities};
use eredu_nn::{NeuralBackend, NeuralOperatorCapabilities};

use crate::{
    ArchitectureGroupTransport, CacheResidencyPolicy, ExecutionGraph, ExecutionUnitLayout,
    LayeredArchitecture, RuntimeState, StateLayout,
};

/// Statically dispatched text-input seam for an ordinary layered decoder.
///
/// Hybrid, routed, composite, partitioned, prediction, and realtime execution
/// use separate extension contracts rather than adding requirements here.
pub trait ReplicatedTextArchitecture<B, S>: LayeredArchitecture<B, S>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
{
    /// Forms the architecture-owned borrowed input for one text pass.
    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a>;
}

/// Backend implementation route for one source-to-executable weight lowering.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum WeightLoweringKind {
    /// The admitted source encoding is retained by the executable operator.
    Direct,
    /// Payload materialization performs an admitted transformation.
    Transform,
}

/// One exact weight lowering implemented by a backend.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct WeightLoweringCapability {
    /// Admitted source encoding.
    source: SourceTensorEncoding,
    /// Backend-neutral executable format produced by the lowering.
    executable: LinearFormat,
    /// Whether the lowering is direct or transforming.
    kind: WeightLoweringKind,
}

impl WeightLoweringCapability {
    /// Creates one exact backend lowering mechanism.
    pub fn new(
        source: SourceTensorEncoding,
        executable: LinearFormat,
        kind: WeightLoweringKind,
    ) -> Self {
        Self {
            source,
            executable,
            kind,
        }
    }

    /// Returns the admitted source encoding.
    pub const fn source(&self) -> &SourceTensorEncoding {
        &self.source
    }

    /// Returns the executable format produced by this mechanism.
    pub const fn executable(&self) -> LinearFormat {
        self.executable
    }

    /// Returns whether materialization retains or transforms the source.
    pub const fn kind(&self) -> WeightLoweringKind {
        self.kind
    }
}

/// Weight-residency mechanism implemented by a backend.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum WeightResidencyMechanism {
    /// All parameters remain device resident.
    Resident,
    /// A bounded device window is staged from host storage.
    Windowed,
    /// Bounded host and device windows are populated from disk.
    DiskStreamed,
}

/// Mutable-state residency mechanism implemented by a backend.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum StateResidencyMechanism {
    /// State tensors remain on the execution device.
    Device,
    /// State blocks use bounded paged storage.
    Paged,
}

/// Architecture-valid transform target for one linear parameter.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ParameterTransformTarget {
    /// Requested load-time transform.
    request: QuantizationRequest,
    /// Executable format produced for this parameter.
    executable: LinearFormat,
}

impl ParameterTransformTarget {
    /// Creates one architecture-admitted load-time transform target.
    pub const fn new(request: QuantizationRequest, executable: LinearFormat) -> Self {
        Self {
            request,
            executable,
        }
    }

    /// Returns the caller request selecting this transform.
    pub const fn request(&self) -> QuantizationRequest {
        self.request
    }

    /// Returns the architecture-admitted executable format.
    pub const fn executable(&self) -> LinearFormat {
        self.executable
    }
}

/// Exact admitted source and executable constraints for one logical parameter.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ReplicatedTextParameterRequirement {
    /// Canonical logical parameter identity.
    name: String,
    /// Physical outputs admitted as sources for this logical parameter.
    sources: Vec<String>,
    /// Encoding of the selected physical source.
    source_encoding: SourceTensorEncoding,
    /// Architecture-selected native executable format.
    native_executable: LinearFormat,
    /// Architecture permits validated affine load-time transforms.
    affine_transforms: bool,
    /// Architecture permits the MXFP4 load-time transform.
    mxfp4_transform: bool,
}

impl ReplicatedTextParameterRequirement {
    /// Creates one exact logical-parameter requirement.
    pub fn new(
        name: impl Into<String>,
        sources: Vec<String>,
        source_encoding: SourceTensorEncoding,
        native_executable: LinearFormat,
    ) -> Result<Self, ReplicatedTextContractError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(ReplicatedTextContractError::invalid(
                "logical parameter identity is empty",
            ));
        }
        if sources.is_empty() || sources.iter().any(|source| source.trim().is_empty()) {
            return Err(ReplicatedTextContractError::invalid(format!(
                "logical parameter {name:?} has no valid physical source"
            )));
        }
        Ok(Self {
            name,
            sources,
            source_encoding,
            native_executable,
            affine_transforms: false,
            mxfp4_transform: false,
        })
    }

    /// Permits validated affine load-time transforms for this parameter.
    pub const fn with_affine_transforms(mut self) -> Self {
        self.affine_transforms = true;
        self
    }

    /// Permits the MXFP4 load-time transform for this parameter.
    pub const fn with_mxfp4_transform(mut self) -> Self {
        self.mxfp4_transform = true;
        self
    }

    /// Returns the canonical logical identity.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns exact admitted physical source identities.
    pub fn sources(&self) -> &[String] {
        &self.sources
    }

    /// Returns the admitted physical source encoding.
    pub const fn source_encoding(&self) -> &SourceTensorEncoding {
        &self.source_encoding
    }

    /// Returns the architecture-native executable format.
    pub const fn native_executable(&self) -> LinearFormat {
        self.native_executable
    }

    /// Resolves a caller transform through architecture-owned constraints.
    pub fn transform_target(
        &self,
        request: QuantizationRequest,
    ) -> Result<Option<ParameterTransformTarget>, ReplicatedTextContractError> {
        let executable = match request {
            QuantizationRequest::Affine { group_size, bits } if self.affine_transforms => {
                let group_size = i32::try_from(group_size).map_err(|_| {
                    ReplicatedTextContractError::invalid("affine group size exceeds i32")
                })?;
                Some(LinearFormat::Affine(
                    eredu_checkpoint::AffineQuantization::new(group_size, i32::from(bits))
                        .map_err(|error| ReplicatedTextContractError::invalid(error.to_string()))?,
                ))
            }
            QuantizationRequest::MxFp4 if self.mxfp4_transform => Some(LinearFormat::MxFp4),
            _ => None,
        };
        Ok(executable.map(|executable| ParameterTransformTarget::new(request, executable)))
    }
}

/// Invalid public replicated-text contract construction.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[error("invalid replicated text contract: {message}")]
pub struct ReplicatedTextContractError {
    message: String,
}

impl ReplicatedTextContractError {
    fn invalid(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// Returns the stable semantic diagnostic.
    pub fn message(&self) -> &str {
        &self.message
    }
}

/// Exact architecture and artifact requirements for replicated text execution.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ReplicatedTextRequirements {
    /// Optional neural operations required by the architecture equations.
    operators: NeuralOperatorCapabilities,
    /// Stable architecture-owned execution graph.
    execution_graph: ExecutionGraph,
    /// Exact group-major execution-unit geometry.
    execution_units: ExecutionUnitLayout,
    /// Architecture-owned transport semantics in graph-group order.
    group_transports: Vec<ArchitectureGroupTransport>,
    /// Complete architecture-owned mutable-state geometry.
    state_layout: StateLayout,
    /// Canonical logical parameter requirements.
    parameters: Vec<ReplicatedTextParameterRequirement>,
}

impl ReplicatedTextRequirements {
    /// Creates exact requirements from architecture and admitted-artifact facts only.
    pub fn new(
        operators: NeuralOperatorCapabilities,
        execution_graph: ExecutionGraph,
        execution_units: ExecutionUnitLayout,
        group_transports: Vec<ArchitectureGroupTransport>,
        state_layout: StateLayout,
        parameters: Vec<ReplicatedTextParameterRequirement>,
    ) -> Result<Self, ReplicatedTextContractError> {
        if group_transports.len() != execution_graph.groups().len() {
            return Err(ReplicatedTextContractError::invalid(format!(
                "{} group transports do not match {} execution groups",
                group_transports.len(),
                execution_graph.groups().len()
            )));
        }
        let mut names = BTreeSet::new();
        if parameters
            .iter()
            .any(|parameter| !names.insert(parameter.name()))
        {
            return Err(ReplicatedTextContractError::invalid(
                "logical parameter identities are not unique",
            ));
        }
        Ok(Self {
            operators,
            execution_graph,
            execution_units,
            group_transports,
            state_layout,
            parameters,
        })
    }

    /// Returns required optional neural-operation semantics.
    pub const fn operators(&self) -> NeuralOperatorCapabilities {
        self.operators
    }
    /// Returns the architecture-owned execution graph.
    pub const fn execution_graph(&self) -> &ExecutionGraph {
        &self.execution_graph
    }
    /// Returns group-major execution-unit geometry.
    pub const fn execution_units(&self) -> &ExecutionUnitLayout {
        &self.execution_units
    }
    /// Returns architecture-owned group transports.
    pub fn group_transports(&self) -> &[ArchitectureGroupTransport] {
        &self.group_transports
    }
    /// Returns complete mutable-state geometry.
    pub const fn state_layout(&self) -> &StateLayout {
        &self.state_layout
    }
    /// Returns canonical logical parameter requirements.
    pub fn parameters(&self) -> &[ReplicatedTextParameterRequirement] {
        &self.parameters
    }
}

/// Family- and execution-class-neutral backend mechanism report.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct BackendMechanismCapabilities {
    /// Optional neural operations implemented by the backend.
    operators: NeuralOperatorCapabilities,
    /// Exact admitted source-to-executable lowerings.
    weight_lowerings: Vec<WeightLoweringCapability>,
    /// Ordinary parameter residency mechanisms.
    weight_residencies: Vec<WeightResidencyMechanism>,
    /// Mutable-state residency mechanisms.
    state_residencies: Vec<StateResidencyMechanism>,
    /// Exact session facilities implemented by the constructed session.
    session: SessionCapabilities,
    /// Prompt-cache persistence mechanism is available.
    prompt_cache: bool,
    /// Exact completion ownership is implemented for submitted work.
    exact_completion: bool,
}

impl BackendMechanismCapabilities {
    /// Creates a fail-closed mechanism report.
    pub fn new(
        operators: NeuralOperatorCapabilities,
        weight_lowerings: Vec<WeightLoweringCapability>,
        weight_residencies: Vec<WeightResidencyMechanism>,
        state_residencies: Vec<StateResidencyMechanism>,
    ) -> Self {
        Self {
            operators,
            weight_lowerings,
            weight_residencies,
            state_residencies,
            session: SessionCapabilities::default(),
            prompt_cache: false,
            exact_completion: false,
        }
    }

    /// Adds supported session-observation and persistence mechanisms.
    pub const fn with_session(mut self, session: SessionCapabilities) -> Self {
        self.session = session;
        self
    }
    /// Declares prompt-cache persistence support.
    pub const fn with_prompt_cache(mut self, supported: bool) -> Self {
        self.prompt_cache = supported;
        self
    }
    /// Declares exact native-completion ownership support.
    pub const fn with_exact_completion(mut self, supported: bool) -> Self {
        self.exact_completion = supported;
        self
    }
    /// Returns neural-operation mechanisms.
    pub const fn operators(&self) -> NeuralOperatorCapabilities {
        self.operators
    }
    /// Returns source-to-executable weight-lowering mechanisms.
    pub fn weight_lowerings(&self) -> &[WeightLoweringCapability] {
        &self.weight_lowerings
    }
    /// Returns weight-residency mechanisms.
    pub fn weight_residencies(&self) -> &[WeightResidencyMechanism] {
        &self.weight_residencies
    }
    /// Returns mutable-state residency mechanisms.
    pub fn state_residencies(&self) -> &[StateResidencyMechanism] {
        &self.state_residencies
    }
    /// Returns session-observation and persistence mechanisms.
    pub const fn session(&self) -> SessionCapabilities {
        self.session
    }
    /// Returns whether prompt-cache persistence is supported.
    pub const fn prompt_cache(&self) -> bool {
        self.prompt_cache
    }
    /// Returns whether exact native-completion ownership is supported.
    pub const fn exact_completion(&self) -> bool {
        self.exact_completion
    }
}

/// Caller choices resolved while selecting one replicated text realization.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ReplicatedTextSelectionRequest {
    /// Requested execution topology.
    topology: Option<ParallelTopology>,
    /// Requested ordinary parameter residency.
    residency: ResidencyRequest,
    /// Requested mutable-state implementation and its exact residency policy.
    state: CacheResidencyPolicy,
    /// Optional load-time transform.
    quantization: Option<QuantizationRequest>,
    /// Requested optional session facilities.
    session: SessionCapabilities,
    /// Whether prompt-cache persistence is requested.
    prompt_cache: bool,
    /// Whether exact completion ownership is requested.
    exact_completion: bool,
}

impl ReplicatedTextSelectionRequest {
    /// Creates a replicated request with fail-closed optional facilities.
    pub fn new(residency: ResidencyRequest, state: CacheResidencyPolicy) -> Self {
        Self {
            topology: None,
            residency,
            state,
            quantization: None,
            session: SessionCapabilities::default(),
            prompt_cache: false,
            exact_completion: false,
        }
    }
    /// Sets the requested topology.
    pub const fn with_topology(mut self, topology: ParallelTopology) -> Self {
        self.topology = Some(topology);
        self
    }
    /// Sets the optional load-time transform.
    pub const fn with_quantization(mut self, quantization: QuantizationRequest) -> Self {
        self.quantization = Some(quantization);
        self
    }
    /// Sets requested session facilities.
    pub const fn with_session(mut self, session: SessionCapabilities) -> Self {
        self.session = session;
        self
    }
    /// Requests prompt-cache persistence.
    pub const fn with_prompt_cache(mut self, required: bool) -> Self {
        self.prompt_cache = required;
        self
    }
    /// Requests exact completion ownership.
    pub const fn with_exact_completion(mut self, required: bool) -> Self {
        self.exact_completion = required;
        self
    }
    /// Returns the requested topology, where `None` means replicated.
    pub const fn topology(&self) -> Option<ParallelTopology> {
        self.topology
    }
    /// Returns the requested weight residency.
    pub const fn residency(&self) -> ResidencyRequest {
        self.residency
    }
    /// Returns the requested state policy.
    pub const fn state(&self) -> &CacheResidencyPolicy {
        &self.state
    }
    /// Returns the requested transform.
    pub const fn quantization(&self) -> Option<QuantizationRequest> {
        self.quantization
    }
    /// Returns requested session facilities.
    pub const fn session(&self) -> SessionCapabilities {
        self.session
    }
    /// Returns whether prompt-cache persistence is requested.
    pub const fn prompt_cache(&self) -> bool {
        self.prompt_cache
    }
    /// Returns whether exact completion ownership is requested.
    pub const fn exact_completion(&self) -> bool {
        self.exact_completion
    }
}

/// Selected lowering for one canonical logical parameter.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SelectedParameterRealization {
    /// Canonical logical parameter identity.
    name: String,
    /// Physical outputs admitted as sources for this logical parameter.
    sources: Vec<String>,
    /// Admitted physical encoding.
    source_encoding: SourceTensorEncoding,
    /// Exact executable format used to construct the architecture module.
    executable: LinearFormat,
    /// Backend lowering selected for materialization.
    lowering: WeightLoweringKind,
}

impl SelectedParameterRealization {
    /// Returns the canonical logical identity.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Returns admitted physical source identities.
    pub fn sources(&self) -> &[String] {
        &self.sources
    }
    /// Returns the admitted source encoding.
    pub const fn source_encoding(&self) -> &SourceTensorEncoding {
        &self.source_encoding
    }
    /// Returns the selected executable format.
    pub const fn executable(&self) -> LinearFormat {
        self.executable
    }
    /// Returns the selected backend lowering kind.
    pub const fn lowering(&self) -> WeightLoweringKind {
        self.lowering
    }
}

/// Authoritative realization selected before architecture or payload construction.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SelectedReplicatedTextRealization {
    /// Selected ordinary parameter residency.
    residency: WeightResidencyMechanism,
    /// Selected mutable-state implementation and its exact residency policy.
    state: CacheResidencyPolicy,
    /// Exact per-parameter source, executable format, and lowering.
    parameters: Vec<SelectedParameterRealization>,
    /// Required observation facilities admitted by the backend.
    session: SessionCapabilities,
    /// Prompt-cache persistence is selected for this lifecycle.
    prompt_cache: bool,
    /// Exact completion ownership selected for this lifecycle.
    exact_completion: bool,
}

impl SelectedReplicatedTextRealization {
    /// Returns selected weight residency.
    pub const fn residency(&self) -> WeightResidencyMechanism {
        self.residency
    }
    /// Returns selected mutable-state policy.
    pub const fn state(&self) -> &CacheResidencyPolicy {
        &self.state
    }
    /// Returns exact per-parameter realizations.
    pub fn parameters(&self) -> &[SelectedParameterRealization] {
        &self.parameters
    }
    /// Returns selected session facilities.
    pub const fn session(&self) -> SessionCapabilities {
        self.session
    }
    /// Returns whether prompt-cache persistence was selected.
    pub const fn prompt_cache(&self) -> bool {
        self.prompt_cache
    }
    /// Returns whether exact completion ownership was selected.
    pub const fn exact_completion(&self) -> bool {
        self.exact_completion
    }
}

/// Complete fail-closed selection diagnostic.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[error("replicated text realization is unsupported: {issues}", issues = .issues.join("; "))]
pub struct ReplicatedTextSelectionError {
    issues: Vec<String>,
}

impl ReplicatedTextSelectionError {
    /// Every missing semantic or mechanism requirement in stable order.
    pub fn issues(&self) -> &[String] {
        &self.issues
    }
}

/// Deterministically selects one realization without constructing backend payloads.
pub fn select_replicated_text_realization(
    requirements: &ReplicatedTextRequirements,
    request: &ReplicatedTextSelectionRequest,
    capabilities: &BackendMechanismCapabilities,
) -> Result<SelectedReplicatedTextRealization, ReplicatedTextSelectionError> {
    let mut issues = Vec::new();
    if request
        .topology
        .is_some_and(|topology| !topology.is_replicated())
    {
        issues.push("replicated execution topology".into());
    }
    if !capabilities.operators.contains(requirements.operators) {
        issues.extend(
            capabilities
                .operators
                .missing_capability_names(requirements.operators)
                .into_iter()
                .map(|name| format!("neural operation {name}")),
        );
    }
    let residency = match request.residency {
        ResidencyRequest::FullyResident => Some(WeightResidencyMechanism::Resident),
        ResidencyRequest::LayerwiseHost => Some(WeightResidencyMechanism::Windowed),
        ResidencyRequest::DenseDiskStream => Some(WeightResidencyMechanism::DiskStreamed),
        ResidencyRequest::ExpertCache => {
            issues.push("independently addressable parameter-bank residency".into());
            None
        }
    };
    if let Some(residency) = residency {
        if !capabilities.weight_residencies.contains(&residency) {
            issues.push(format!("weight residency {residency:?}"));
        }
    }
    let state_residency = match &request.state {
        CacheResidencyPolicy::Device => StateResidencyMechanism::Device,
        CacheResidencyPolicy::Paged(_) => StateResidencyMechanism::Paged,
    };
    if !capabilities.state_residencies.contains(&state_residency) {
        issues.push(format!("state residency {state_residency:?}"));
    }
    for (required, supported, name) in [
        (
            request.session.persistent_cache,
            capabilities.session.persistent_cache,
            "persistent_cache",
        ),
        (
            request.session.output_observation,
            capabilities.session.output_observation,
            "output_observation",
        ),
        (
            request.session.activation_inspection,
            capabilities.session.activation_inspection,
            "activation_inspection",
        ),
    ] {
        if required && !supported {
            issues.push(format!("session capability {name}"));
        }
    }
    if request.prompt_cache && !capabilities.prompt_cache {
        issues.push("prompt-cache persistence".into());
    }
    if request.exact_completion && !capabilities.exact_completion {
        issues.push("exact completion ownership".into());
    }

    let mut parameters = Vec::with_capacity(requirements.parameters.len());
    let mut names = BTreeSet::new();
    for parameter in &requirements.parameters {
        if parameter.name.trim().is_empty() || !names.insert(parameter.name.as_str()) {
            issues.push(format!(
                "unique nonempty logical parameter identity {:?}",
                parameter.name
            ));
            continue;
        }
        let executable = match request.quantization {
            Some(request) => match parameter.transform_target(request) {
                Ok(target) => target.map(|target| target.executable()),
                Err(error) => {
                    issues.push(error.to_string());
                    None
                }
            },
            None => Some(parameter.native_executable),
        };
        let Some(executable) = executable else {
            issues.push(format!(
                "architecture transform {:?} for {:?}",
                request.quantization, parameter.name
            ));
            continue;
        };
        let Some(lowering) = capabilities.weight_lowerings.iter().find(|lowering| {
            lowering.source == parameter.source_encoding && lowering.executable == executable
        }) else {
            issues.push(format!(
                "weight lowering {:?} -> {:?} for {:?}",
                parameter.source_encoding, executable, parameter.name
            ));
            continue;
        };
        parameters.push(SelectedParameterRealization {
            name: parameter.name.clone(),
            sources: parameter.sources.clone(),
            source_encoding: parameter.source_encoding.clone(),
            executable,
            lowering: lowering.kind,
        });
    }
    if !issues.is_empty() {
        return Err(ReplicatedTextSelectionError { issues });
    }
    Ok(SelectedReplicatedTextRealization {
        residency: residency.expect("unsupported residency returned an issue"),
        state: request.state.clone(),
        parameters,
        session: request.session,
        prompt_cache: request.prompt_cache,
        exact_completion: request.exact_completion,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ArchitectureGroupKind, ArchitectureGroupPlacement, ArchitectureGroupTransport,
        ArchitectureMergeDestination, ExecutionGroupSpec, ExecutionUnitLayout, StateLayout,
    };
    use eredu_checkpoint::{AffineQuantization, StoredDtype};
    use eredu_core::{cache::LayerCachePolicy, AttentionPolicy, LayerSchedule};

    fn paged_state() -> CacheResidencyPolicy {
        CacheResidencyPolicy::Paged(
            crate::PagedCacheOptions::new(4, 1 << 20, 1 << 20, 1)
                .unwrap()
                .with_full_attention(true),
        )
    }

    fn requirements() -> ReplicatedTextRequirements {
        let graph =
            ExecutionGraph::new(vec![ExecutionGroupSpec::root("decoder")], "decoder").unwrap();
        let execution_units = ExecutionUnitLayout::new(&graph, [1]).unwrap();
        ReplicatedTextRequirements::new(
            NeuralOperatorCapabilities::EXP,
            graph,
            execution_units,
            vec![ArchitectureGroupTransport {
                placement: ArchitectureGroupPlacement::Pipeline,
                kind: ArchitectureGroupKind::Decoder,
                first_owner_static_roles: vec!["embedding".into()],
                last_owner_static_roles: vec!["output".into()],
                merge_destination: ArchitectureMergeDestination::LastOwner,
                parallel_subgroup: None,
                request_optional: false,
            }],
            StateLayout::new(
                LayerSchedule::new(
                    1,
                    vec![LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 8).unwrap()],
                )
                .unwrap(),
            )
            .unwrap(),
            vec![ReplicatedTextParameterRequirement::new(
                "model.layers.0.mlp.weight",
                vec!["blk.0.ffn.weight".into()],
                SourceTensorEncoding::Safetensors(StoredDtype::F16),
                LinearFormat::Dense,
            )
            .unwrap()
            .with_affine_transforms()],
        )
        .unwrap()
    }

    fn capabilities() -> BackendMechanismCapabilities {
        let source = SourceTensorEncoding::Safetensors(StoredDtype::F16);
        BackendMechanismCapabilities::new(
            NeuralOperatorCapabilities::EXP,
            vec![
                WeightLoweringCapability::new(
                    source.clone(),
                    LinearFormat::Dense,
                    WeightLoweringKind::Direct,
                ),
                WeightLoweringCapability::new(
                    source,
                    LinearFormat::Affine(AffineQuantization::new(64, 4).unwrap()),
                    WeightLoweringKind::Transform,
                ),
            ],
            vec![
                WeightResidencyMechanism::Resident,
                WeightResidencyMechanism::Windowed,
                WeightResidencyMechanism::DiskStreamed,
            ],
            vec![
                StateResidencyMechanism::Device,
                StateResidencyMechanism::Paged,
            ],
        )
        .with_session(SessionCapabilities {
            persistent_cache: true,
            output_observation: true,
            activation_inspection: true,
        })
        .with_prompt_cache(true)
        .with_exact_completion(true)
    }

    fn request(residency: ResidencyRequest) -> ReplicatedTextSelectionRequest {
        ReplicatedTextSelectionRequest::new(residency, paged_state())
            .with_session(SessionCapabilities {
                persistent_cache: true,
                output_observation: true,
                activation_inspection: true,
            })
            .with_prompt_cache(true)
            .with_exact_completion(true)
    }

    #[test]
    fn selection_is_deterministic_and_keeps_source_format_distinct() {
        let request = request(ResidencyRequest::DenseDiskStream).with_quantization(
            QuantizationRequest::Affine {
                group_size: 64,
                bits: 4,
            },
        );
        let left =
            select_replicated_text_realization(&requirements(), &request, &capabilities()).unwrap();
        let right =
            select_replicated_text_realization(&requirements(), &request, &capabilities()).unwrap();
        assert_eq!(left, right);
        assert_eq!(left.residency(), WeightResidencyMechanism::DiskStreamed);
        assert_eq!(left.state(), &paged_state());
        assert_eq!(
            left.parameters()[0].lowering(),
            WeightLoweringKind::Transform
        );
        assert_ne!(
            format!("{:?}", left.parameters()[0].source_encoding()),
            format!("{:?}", left.parameters()[0].executable())
        );
    }

    #[test]
    fn selection_reports_all_missing_mechanisms_together() {
        let capabilities = BackendMechanismCapabilities::new(
            NeuralOperatorCapabilities::NONE,
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
        let error = select_replicated_text_realization(
            &requirements(),
            &request(ResidencyRequest::LayerwiseHost),
            &capabilities,
        )
        .unwrap_err();
        assert!(error.issues().len() >= 7, "{:?}", error.issues());
        assert!(error.issues().iter().any(|issue| issue.contains("exp")));
        assert!(error
            .issues()
            .iter()
            .any(|issue| issue.contains("weight lowering")));
    }
}
