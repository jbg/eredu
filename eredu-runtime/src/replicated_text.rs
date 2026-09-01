//! Selection contracts for replicated text architectures.

use std::collections::BTreeSet;

use eredu_checkpoint::{LinearFormat, SourceTensorEncoding};
use eredu_core::{QuantizationRequest, ResidencyRequest, SessionCapabilities};
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
    pub source: SourceTensorEncoding,
    /// Backend-neutral executable format produced by the lowering.
    pub executable: LinearFormat,
    /// Whether the lowering is direct or transforming.
    pub kind: WeightLoweringKind,
}

/// Residency mechanism implemented for ordinary replicated execution units.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ReplicatedTextResidency {
    /// All parameters remain device resident.
    Resident,
    /// A bounded device window is staged from host storage.
    Windowed,
    /// Bounded host and device windows are populated from disk.
    DiskStreamed,
}

/// Residency of the selected mutable state representation.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ReplicatedTextStateResidency {
    /// State tensors remain on the execution device.
    Device,
    /// State blocks use bounded paged storage.
    Paged,
}

/// Architecture-valid transform target for one linear parameter.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ParameterTransformTarget {
    /// Requested load-time transform.
    pub request: QuantizationRequest,
    /// Executable format produced for this parameter.
    pub executable: LinearFormat,
}

/// Exact admitted source and executable constraints for one logical parameter.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ReplicatedTextParameterRequirement {
    /// Canonical logical parameter identity.
    pub name: String,
    /// Physical outputs admitted as sources for this logical parameter.
    pub sources: Vec<String>,
    /// Encoding of the selected physical source.
    pub source_encoding: SourceTensorEncoding,
    /// Architecture-selected native executable format.
    pub native_executable: LinearFormat,
    /// Architecture-valid load-time transformations.
    pub transform_targets: Vec<ParameterTransformTarget>,
}

/// Exact architecture and artifact requirements for replicated text execution.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ReplicatedTextRequirements {
    /// Optional neural operations required by the architecture equations.
    pub operators: NeuralOperatorCapabilities,
    /// Stable architecture-owned execution graph.
    pub execution_graph: ExecutionGraph,
    /// Exact group-major execution-unit geometry.
    pub execution_units: ExecutionUnitLayout,
    /// Architecture-owned transport semantics in graph-group order.
    pub group_transports: Vec<ArchitectureGroupTransport>,
    /// Complete architecture-owned mutable-state geometry.
    pub state_layout: StateLayout,
    /// Canonical logical parameter requirements.
    pub parameters: Vec<ReplicatedTextParameterRequirement>,
    /// Exact session facilities required by the caller.
    pub session: SessionCapabilities,
    /// Prompt-cache persistence is part of the admitted lifecycle.
    pub prompt_cache: bool,
    /// Submitted native work must expose exact completion ownership.
    pub exact_completion: bool,
}

/// Family-neutral backend mechanism report used for replicated text selection.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ReplicatedTextBackendCapabilities {
    /// Optional neural operations implemented by the backend.
    pub operators: NeuralOperatorCapabilities,
    /// Exact admitted source-to-executable lowerings.
    pub weight_lowerings: Vec<WeightLoweringCapability>,
    /// Ordinary parameter residency mechanisms.
    pub residencies: Vec<ReplicatedTextResidency>,
    /// Mutable-state residency mechanisms.
    pub state_residencies: Vec<ReplicatedTextStateResidency>,
    /// Exact session facilities implemented by the constructed session.
    pub session: SessionCapabilities,
    /// Prompt-cache persistence mechanism is available.
    pub prompt_cache: bool,
    /// Exact completion ownership is implemented for submitted work.
    pub exact_completion: bool,
}

/// Caller choices resolved while selecting one replicated text realization.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ReplicatedTextSelectionRequest {
    /// Requested ordinary parameter residency.
    pub residency: ResidencyRequest,
    /// Requested mutable-state implementation and its exact residency policy.
    pub state: CacheResidencyPolicy,
    /// Optional load-time transform.
    pub quantization: Option<QuantizationRequest>,
}

/// Selected lowering for one canonical logical parameter.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SelectedParameterRealization {
    /// Canonical logical parameter identity.
    pub name: String,
    /// Physical outputs admitted as sources for this logical parameter.
    pub sources: Vec<String>,
    /// Admitted physical encoding.
    pub source_encoding: SourceTensorEncoding,
    /// Exact executable format used to construct the architecture module.
    pub executable: LinearFormat,
    /// Backend lowering selected for materialization.
    pub lowering: WeightLoweringKind,
}

/// Authoritative realization selected before architecture or payload construction.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SelectedReplicatedTextRealization {
    /// Selected ordinary parameter residency.
    pub residency: ReplicatedTextResidency,
    /// Selected mutable-state implementation and its exact residency policy.
    pub state: CacheResidencyPolicy,
    /// Exact per-parameter source, executable format, and lowering.
    pub parameters: Vec<SelectedParameterRealization>,
    /// Required observation facilities admitted by the backend.
    pub session: SessionCapabilities,
    /// Prompt-cache persistence is selected for this lifecycle.
    pub prompt_cache: bool,
    /// Exact completion ownership selected for this lifecycle.
    pub exact_completion: bool,
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
    capabilities: &ReplicatedTextBackendCapabilities,
) -> Result<SelectedReplicatedTextRealization, ReplicatedTextSelectionError> {
    let mut issues = Vec::new();
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
        ResidencyRequest::FullyResident => Some(ReplicatedTextResidency::Resident),
        ResidencyRequest::LayerwiseHost => Some(ReplicatedTextResidency::Windowed),
        ResidencyRequest::DenseDiskStream => Some(ReplicatedTextResidency::DiskStreamed),
        ResidencyRequest::ExpertCache => {
            issues.push("independently addressable parameter-bank residency".into());
            None
        }
    };
    if let Some(residency) = residency {
        if !capabilities.residencies.contains(&residency) {
            issues.push(format!("weight residency {residency:?}"));
        }
    }
    let state_residency = match &request.state {
        CacheResidencyPolicy::Device => ReplicatedTextStateResidency::Device,
        CacheResidencyPolicy::Paged(_) => ReplicatedTextStateResidency::Paged,
    };
    if !capabilities.state_residencies.contains(&state_residency) {
        issues.push(format!("state residency {state_residency:?}"));
    }
    for (required, supported, name) in [
        (
            requirements.session.persistent_cache,
            capabilities.session.persistent_cache,
            "persistent_cache",
        ),
        (
            requirements.session.output_observation,
            capabilities.session.output_observation,
            "output_observation",
        ),
        (
            requirements.session.activation_inspection,
            capabilities.session.activation_inspection,
            "activation_inspection",
        ),
    ] {
        if required && !supported {
            issues.push(format!("session capability {name}"));
        }
    }
    if requirements.prompt_cache && !capabilities.prompt_cache {
        issues.push("prompt-cache persistence".into());
    }
    if requirements.exact_completion && !capabilities.exact_completion {
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
            Some(request) => parameter
                .transform_targets
                .iter()
                .find(|target| target.request == request)
                .map(|target| target.executable),
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
        session: requirements.session,
        prompt_cache: requirements.prompt_cache,
        exact_completion: requirements.exact_completion,
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
        ReplicatedTextRequirements {
            operators: NeuralOperatorCapabilities::EXP,
            execution_graph: graph,
            execution_units,
            group_transports: vec![ArchitectureGroupTransport {
                placement: ArchitectureGroupPlacement::Pipeline,
                kind: ArchitectureGroupKind::Decoder,
                first_owner_static_roles: vec!["embedding".into()],
                last_owner_static_roles: vec!["output".into()],
                merge_destination: ArchitectureMergeDestination::LastOwner,
                parallel_subgroup: None,
                request_optional: false,
            }],
            state_layout: StateLayout::new(
                LayerSchedule::new(
                    1,
                    vec![LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 8).unwrap()],
                )
                .unwrap(),
            )
            .unwrap(),
            parameters: vec![ReplicatedTextParameterRequirement {
                name: "model.layers.0.mlp.weight".into(),
                sources: vec!["blk.0.ffn.weight".into()],
                source_encoding: SourceTensorEncoding::Safetensors(StoredDtype::F16),
                native_executable: LinearFormat::Dense,
                transform_targets: vec![ParameterTransformTarget {
                    request: QuantizationRequest::Affine {
                        group_size: 64,
                        bits: 4,
                    },
                    executable: LinearFormat::Affine(AffineQuantization::new(64, 4).unwrap()),
                }],
            }],
            session: SessionCapabilities {
                persistent_cache: true,
                output_observation: true,
                activation_inspection: true,
            },
            prompt_cache: true,
            exact_completion: true,
        }
    }

    fn capabilities() -> ReplicatedTextBackendCapabilities {
        let source = SourceTensorEncoding::Safetensors(StoredDtype::F16);
        ReplicatedTextBackendCapabilities {
            operators: NeuralOperatorCapabilities::EXP,
            weight_lowerings: vec![
                WeightLoweringCapability {
                    source: source.clone(),
                    executable: LinearFormat::Dense,
                    kind: WeightLoweringKind::Direct,
                },
                WeightLoweringCapability {
                    source,
                    executable: LinearFormat::Affine(AffineQuantization::new(64, 4).unwrap()),
                    kind: WeightLoweringKind::Transform,
                },
            ],
            residencies: vec![
                ReplicatedTextResidency::Resident,
                ReplicatedTextResidency::Windowed,
                ReplicatedTextResidency::DiskStreamed,
            ],
            state_residencies: vec![
                ReplicatedTextStateResidency::Device,
                ReplicatedTextStateResidency::Paged,
            ],
            session: SessionCapabilities {
                persistent_cache: true,
                output_observation: true,
                activation_inspection: true,
            },
            prompt_cache: true,
            exact_completion: true,
        }
    }

    #[test]
    fn selection_is_deterministic_and_keeps_source_format_distinct() {
        let request = ReplicatedTextSelectionRequest {
            residency: ResidencyRequest::DenseDiskStream,
            state: paged_state(),
            quantization: Some(QuantizationRequest::Affine {
                group_size: 64,
                bits: 4,
            }),
        };
        let left =
            select_replicated_text_realization(&requirements(), &request, &capabilities()).unwrap();
        let right =
            select_replicated_text_realization(&requirements(), &request, &capabilities()).unwrap();
        assert_eq!(left, right);
        assert_eq!(left.residency, ReplicatedTextResidency::DiskStreamed);
        assert_eq!(left.state, paged_state());
        assert_eq!(left.parameters[0].lowering, WeightLoweringKind::Transform);
        assert_ne!(
            format!("{:?}", left.parameters[0].source_encoding),
            format!("{:?}", left.parameters[0].executable)
        );
    }

    #[test]
    fn selection_reports_all_missing_mechanisms_together() {
        let mut capabilities = capabilities();
        capabilities.operators = NeuralOperatorCapabilities::NONE;
        capabilities.weight_lowerings.clear();
        capabilities.residencies.clear();
        capabilities.state_residencies.clear();
        capabilities.session = SessionCapabilities::default();
        capabilities.prompt_cache = false;
        capabilities.exact_completion = false;
        let error = select_replicated_text_realization(
            &requirements(),
            &ReplicatedTextSelectionRequest {
                residency: ResidencyRequest::LayerwiseHost,
                state: paged_state(),
                quantization: None,
            },
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
