//! Embedded prediction is an additional invocation of ordinary modules.
//!
//! Logical sharing, physical allocation sharing and execution order are separate
//! contracts. Startup forecasts compose these descriptions with explicit mechanism
//! calibration; settled embedded continuation observations remain a separate contract.

use std::collections::BTreeMap;

use eredu_core::speculative::{SpeculativeCaptureBinding, SpeculativeCaptureScope};
use eredu_core::{resources::*, ArchitectureEdge, ArchitectureNode, ArchitectureParameterGroup};
use eredu_nn::ParameterMetadata;

use crate::{SpeculativeCaptureSchema, SpeculativeStrategyClass};

/// Physical invocation model, independent of the number of decoder blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PredictionExecutionMode {
    /// Each proposal depth is an independent invocation; units execute in order.
    Sequential,
    /// One proposal invocation produces several rows through ordered blocks.
    Fused,
}
impl PredictionExecutionMode {
    /// Rows consumed by ordinary prediction context preparation.
    pub const fn prefill_sequence_len(self, target_sequence: usize) -> usize {
        match self {
            Self::Sequential => target_sequence.saturating_sub(1),
            Self::Fused => target_sequence,
        }
    }
    /// Reuses the strategy admitted by ordinary speculative selection.
    pub fn from_strategy(
        class: SpeculativeStrategyClass,
    ) -> Result<Self, ResourceDescriptionError> {
        match class {
            SpeculativeStrategyClass::EmbeddedSequential => Ok(Self::Sequential),
            SpeculativeStrategyClass::EmbeddedFused => Ok(Self::Fused),
            _ => Err(invalid(
                "external drafting has no embedded invocation contract",
            )),
        }
    }
}

/// One physical module retained by the actual extension construction driver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedPredictionModule {
    /// Stable ordinal used by parameter-slot traversal and materialization.
    pub ordinal: usize,
    /// Ordinary auxiliary residency owner, if the module has parameters.
    pub residency_owner: Option<crate::AuxiliaryModuleResidency>,
    /// Owners retained across the ordered unit calls overlap those calls.
    pub shared: bool,
    /// Exact local declarations visited during module construction. Aliases stay
    /// logical until the materializer supplies a canonical backing.
    pub parameters: Vec<ParameterMetadata>,
}

/// Ordinary state geometry for one additional prediction execution.
/// Cold topology uses global ordinals; prepared placement uses local lane ordinals
/// and replaces global geometry before local allocation sizing.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PredictionStateLayer {
    /// State ordinal in the containing cold or prepared description.
    pub layer: usize,
    /// Policy consumed by state construction, including component presence/dtypes.
    pub policy: eredu_core::cache::LayerCachePolicy,
    /// The same processed-token offset used by ordinary context preparation.
    pub processed_token_offset: i32,
}

/// Additional logical invocations selected by the ordinary embedded driver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddedPredictionTopology {
    /// Ordinary construction geometry for the selected prediction modules.
    pub execution_topology: Option<crate::execution_topology::TextExecutionTopology>,
    /// Concrete mechanisms whose ordinary invocation geometry remains unavailable.
    pub missing: Vec<String>,
    /// Sequential depths and fused rows have different context preparation.
    pub mode: PredictionExecutionMode,
    /// Admitted proposal rows, independent of the physical module count.
    pub proposal_capacity: usize,
    /// Additional semantic operations, retaining ordinary mechanism geometry.
    pub nodes: Vec<ArchitectureNode>,
    /// Data dependencies, never interpreted as allocation lifetimes.
    pub edges: Vec<ArchitectureEdge>,
    /// Canonical logical identities, including shared target groups. These are
    /// uses of target parameters, not additional physical parameter allocations.
    pub parameters: Vec<ArchitectureParameterGroup>,
    /// Exact scope for every additional node, inherited from ordinary bindings.
    pub invocations: Vec<SpeculativeCaptureBinding>,
    /// Exact target feature shapes selected for the predictor.
    pub target_features: SpeculativeCaptureSchema,
    /// Global state ordinals actually invoked by prediction nodes. These are
    /// distinct from physical parameter sharing and from proposal row ordinals.
    pub state: Vec<PredictionStateLayer>,
}
impl EmbeddedPredictionTopology {
    /// Operations participating in one physical scope. A fused context builder
    /// does not accidentally acquire the fused proposal's decoder or score head.
    pub fn nodes_for_scope(
        &self,
        scope: SpeculativeCaptureScope,
    ) -> impl Iterator<Item = &ArchitectureNode> {
        self.nodes.iter().filter(move |node| {
            self.invocations
                .iter()
                .any(|binding| binding.node_id == node.id && binding.scope == scope)
        })
    }

    /// Checked logical target-feature extents. Request-sized axes are resolved by
    /// the same capture schema used to validate actual execution output.
    /// Logical values may be views: physical allocations require explicit binding.
    pub fn target_feature_bytes(
        &self,
        shapes: Vec<Vec<usize>>,
        scalar_bytes: u64,
    ) -> Result<Vec<u64>, ResourceDescriptionError> {
        if scalar_bytes == 0 {
            return Err(invalid("target feature scalar width must be positive"));
        }
        self.target_features
            .instantiate(shapes.clone())
            .map_err(|e| invalid(e.to_string()))?;
        shapes
            .into_iter()
            .map(|shape| {
                shape
                    .into_iter()
                    .try_fold(scalar_bytes, |bytes, dimension| {
                        bytes
                            .checked_mul(dimension as u64)
                            .ok_or_else(|| invalid("target feature payload overflowed"))
                    })
            })
            .collect()
    }
}

/// Adds authoritative existing conversion allocations to a resource description.
/// A target conversion used by several prediction heads remains one allocation;
/// logical parameter aliases alone never establish native backing identity.
pub fn include_resident_parameter_conversions(
    description: &mut ResourceDescription,
    report: &crate::ResidencyReport,
) -> Result<(), ResourceDescriptionError> {
    description.validate()?;
    let mut candidate = description.clone();
    include_resident_parameter_conversions_inner(&mut candidate, report)?;
    candidate.validate()?;
    *description = candidate;
    Ok(())
}
fn include_resident_parameter_conversions_inner(
    description: &mut ResourceDescription,
    report: &crate::ResidencyReport,
) -> Result<(), ResourceDescriptionError> {
    let Some(conversions) = report.device_parameter_conversions() else {
        add_missing(
            description,
            "cached parameter conversion backing identities are unavailable".into(),
        );
        return Ok(());
    };
    let mut merged: BTreeMap<_, _> = description
        .allocations
        .iter()
        .cloned()
        .map(|allocation| (allocation.identity.clone(), allocation))
        .collect();
    for conversion in conversions {
        conversion.allocation.validate()?;
        if let Some(prior) = merged.get_mut(&conversion.allocation.identity) {
            if prior.placement != conversion.allocation.placement
                || prior.size != conversion.allocation.size
            {
                return Err(invalid(
                    "one cached conversion backing has conflicting physical facts",
                ));
            }
            for usage in &conversion.allocation.uses {
                if !prior.uses.contains(usage) {
                    prior.uses.push(usage.clone());
                }
            }
        } else {
            merged.insert(
                conversion.allocation.identity.clone(),
                conversion.allocation.clone(),
            );
        }
    }
    description.allocations = merged.into_values().collect();
    description.validate()
}

/// Binds retained target features to actual producer allocations, preserving
/// views of target outputs. Missing bindings stay precise gaps rather than
/// inventing an allocation per semantic feature.
pub fn include_target_features(
    description: &mut ResourceDescription,
    topology: &EmbeddedPredictionTopology,
    shapes: Vec<Vec<usize>>,
    scalar_bytes: u64,
    backings: &BTreeMap<String, ResourceAllocation>,
) -> Result<(), ResourceDescriptionError> {
    description.validate()?;
    let mut candidate = description.clone();
    include_target_features_inner(&mut candidate, topology, shapes, scalar_bytes, backings)?;
    candidate.validate()?;
    *description = candidate;
    Ok(())
}
fn include_target_features_inner(
    description: &mut ResourceDescription,
    topology: &EmbeddedPredictionTopology,
    shapes: Vec<Vec<usize>>,
    scalar_bytes: u64,
    backings: &BTreeMap<String, ResourceAllocation>,
) -> Result<(), ResourceDescriptionError> {
    let bytes = topology.target_feature_bytes(shapes, scalar_bytes)?;
    for (feature, bytes) in topology.target_features.entries().iter().zip(bytes) {
        let Some(backing) = backings.get(feature.path().as_str()) else {
            add_missing(
                description,
                format!(
                    "target feature {}: producer backing, capacity and retention are not bound",
                    feature.path().as_str()
                ),
            );
            continue;
        };
        backing.validate()?;
        // Views need only fit inside the backing; multiple feature views can
        // share one larger allocation without pretending they are copies.
        let current = match &backing.size {
            ResourceSize::Fixed { extent } => extent,
            ResourceSize::ContextDependent { current, .. } => current,
        };
        if current
            .payload
            .upper_bytes
            .is_some_and(|upper| bytes > upper)
        {
            return Err(invalid("target feature does not fit its producer backing"));
        }
        let mut allocation = backing.clone();
        let usage = ResourceUse {
            owner: ResourceIdentity {
                scope: format!(
                    "{}:{}{}",
                    description.scope.scope.len(),
                    description.scope.scope,
                    description.scope.key
                ),
                key: format!("prediction/feature/{}", feature.path().as_str()),
            },
            role: ResourceRole::RetainedTensor,
        };
        if !allocation.uses.contains(&usage) {
            allocation.uses.push(usage);
        }
        if let Some(prior) = description
            .allocations
            .iter_mut()
            .find(|prior| prior.identity == allocation.identity)
        {
            if prior.placement != allocation.placement || prior.size != allocation.size {
                return Err(invalid(
                    "target feature backing conflicts with its producer allocation",
                ));
            }
            for usage in allocation.uses {
                if !prior.uses.contains(&usage) {
                    prior.uses.push(usage);
                }
            }
        } else {
            description.allocations.push(allocation);
        }
    }
    description.validate()
}
fn add_missing(description: &mut ResourceDescription, reason: String) {
    match &mut description.coverage {
        ResourceCoverage::Partial { reasons } => {
            if !reasons.contains(&reason) {
                reasons.push(reason);
            }
        }
        _ => {
            description.coverage = ResourceCoverage::Partial {
                reasons: vec![reason],
            }
        }
    }
}
fn invalid(reason: impl Into<String>) -> ResourceDescriptionError {
    ResourceDescriptionError::Invalid(reason.into())
}

/// Explicit hypothetical context supplied to an embedded resource description.
/// This is not an observation of installed state or a forecast of transactions.
#[derive(Debug, Clone)]
pub struct PredictionResourceQuery {
    /// Common instance, horizon and physical execution pool.
    pub prepared: crate::execution_resources::PreparedResourceQuery,
    /// Actual scalar width for floating prediction state, if selected/observed.
    pub floating_state_bytes: Option<u64>,
    /// Actual floating target-feature width, if selected/observed.
    pub feature_scalar_bytes: Option<u64>,
    /// Exact per-entry capture shapes, checked by the ordinary capture schema.
    pub feature_shapes: Vec<Vec<usize>>,
    /// Producer allocations for retained target values, keyed by capture path.
    pub feature_backings: BTreeMap<String, ResourceAllocation>,
}

/// Describes additional state and retained features alongside ordinary prepared
/// parameter slots. It never treats invocation edges as lifetime/evaluation
/// boundaries. Transaction, temporary mechanism and allocator facts remain gaps.
#[allow(clippy::too_many_arguments)]
pub fn describe_prediction_resources(
    topology: &EmbeddedPredictionTopology,
    selected: &crate::SelectedReplicatedTextRealization,
    local_state: Option<&[PredictionStateLayer]>,
    modules: &[PreparedPredictionModule],
    slots: &[crate::parameter_operations::PreparedParameterSlot],
    residency: Option<&crate::ResidencyReport>,
    query: &PredictionResourceQuery,
) -> Result<ResourceDescription, ResourceDescriptionError> {
    let mut module_ordinals = std::collections::BTreeSet::new();
    let mut declared = BTreeMap::new();
    for module in modules {
        if !module_ordinals.insert(module.ordinal) {
            return Err(invalid("prepared prediction module ordinal is duplicated"));
        }
        if module
            .residency_owner
            .as_ref()
            .is_some_and(|owner| owner.shared() != module.shared)
        {
            return Err(invalid(
                "prediction module overlap differs from its residency owner",
            ));
        }
        for parameter in &module.parameters {
            if declared
                .insert((module.ordinal, parameter.id.as_str()), parameter)
                .is_some()
            {
                return Err(invalid(
                    "prepared prediction parameter declaration is duplicated within its module",
                ));
            }
        }
    }
    let mut present = std::collections::BTreeSet::new();
    for slot in slots {
        let crate::parameter_operations::PreparedParameterLocation::Prediction { module } =
            slot.location
        else {
            continue;
        };
        let key = (module, slot.parameter.id.as_str());
        let expected = declared.get(&key).ok_or_else(|| {
            invalid(format!(
                "prediction slot {:?} is not declared by prepared module {module}",
                slot.parameter.id.as_str()
            ))
        })?;
        // Trainability is mutable state, not parameter ownership or geometry.
        let mut expected = (**expected).clone();
        expected.trainable = slot.parameter.trainable;
        if expected != slot.parameter {
            return Err(invalid(format!(
                "prediction slot {:?} differs from its prepared module declaration",
                slot.parameter.id.as_str()
            )));
        }
        if !present.insert(key) {
            return Err(invalid(
                "prediction parameter slot is duplicated within its module",
            ));
        }
    }
    let declarations = modules
        .iter()
        .flat_map(|module| module.parameters.iter().cloned())
        .collect::<Vec<_>>();
    let mut description = crate::execution_resources::describe_prepared_resources(
        selected,
        None,
        slots,
        &declarations,
        &query.prepared,
    )?;
    for (module, name) in declared.keys().filter(|key| !present.contains(*key)) {
        add_missing(&mut description, format!("prediction module {module} parameter {name:?}: exact prepared output backing is unavailable"));
    }
    let end = query
        .prepared
        .prefix_positions
        .checked_add(query.prepared.additional_positions)
        .ok_or_else(|| invalid("prediction state horizon overflowed"))?;
    if query.floating_state_bytes == Some(0) {
        return Err(invalid("state scalar width must be positive"));
    }
    if local_state.is_none() {
        add_missing(
            &mut description,
            "prediction state has no prepared rank-local geometry".into(),
        );
    }
    let local_state = local_state.unwrap_or(&[]);
    for layer in local_state {
        if layer.processed_token_offset > 0 {
            return Err(invalid(
                "prediction state offset advances beyond target context",
            ));
        }
        let offset = u64::from(layer.processed_token_offset.unsigned_abs());
        let first = query.prepared.prefix_positions.saturating_sub(offset);
        let last = end.saturating_sub(offset);
        for component in layer.policy.components() {
            let key = format!(
                "prediction/state/{}/{}",
                layer.layer,
                component.role().stable_name()
            );
            let width = match component.dtype() {
                eredu_core::cache::StateTensorDtype::Floating => query.floating_state_bytes,
                _ => Some(4),
            };
            let Some(width) = width else {
                add_missing(
                    &mut description,
                    format!("{key}: floating state dtype is not selected"),
                );
                continue;
            };
            let mut current_elements = component
                .element_bounds(query.prepared.batch_size, first)
                .map_err(|e| invalid(e.to_string()))?;
            let mut peak_elements = component
                .element_peak_bounds(query.prepared.batch_size, first, last)
                .map_err(|e| invalid(e.to_string()))?;
            // A sliding attention equation does not specify physical truncation.
            // Its legal payload spans window storage through full-prefix retention.
            let sliding = if !matches!(
                component.role(),
                eredu_core::cache::StateComponentRole::Fixed(_)
            ) {
                layer
                    .policy
                    .attention()
                    .and_then(|attention| attention.window())
            } else {
                None
            };
            if let Some(window) = sliding {
                let window = u64::from(window.get());
                current_elements.minimum = component
                    .element_bounds(query.prepared.batch_size, first.min(window))
                    .map_err(|e| invalid(e.to_string()))?
                    .minimum;
                peak_elements.minimum = component
                    .element_peak_bounds(
                        query.prepared.batch_size,
                        first.min(window),
                        last.min(window),
                    )
                    .map_err(|e| invalid(e.to_string()))?
                    .minimum;
            }
            let extent = |elements: eredu_core::cache::StateElementBounds| {
                let lower = elements
                    .minimum
                    .checked_mul(width)
                    .ok_or_else(|| invalid("prediction state payload overflowed"))?;
                let upper = elements
                    .maximum
                    .checked_mul(width)
                    .ok_or_else(|| invalid("prediction state payload overflowed"))?;
                let payload = ResourceByteBounds {
                    lower_bytes: lower,
                    upper_bytes: Some(upper),
                    kind: if lower == upper {
                        eredu_core::ObservationKind::Exact
                    } else {
                        eredu_core::ObservationKind::Estimated
                    },
                    detail: if sliding.is_some() {
                        "ordinary sliding prediction state permits window payload through full-prefix retention"
                    } else {
                        "ordinary prepared prediction state component geometry"
                    }.into(),
                };
                Ok::<_, ResourceDescriptionError>(ResourceExtent {
                    capacity: ResourceByteBounds::unknown(
                        lower,
                        "native prediction state allocation capacity is not described",
                    ),
                    payload,
                })
            };
            let identity = ResourceIdentity {
                scope: format!(
                    "{}:{}{}",
                    query.prepared.scope.scope.len(),
                    query.prepared.scope.scope,
                    query.prepared.scope.key
                ),
                key,
            };
            description.allocations.push(ResourceAllocation {
                identity: identity.clone(),
                uses: vec![ResourceUse {
                    owner: identity,
                    role: ResourceRole::MutableState,
                }],
                placement: query.prepared.device_pool.clone(),
                size: ResourceSize::ContextDependent {
                    current: extent(current_elements)?,
                    horizon_peak: extent(peak_elements)?,
                },
            });
        }
    }
    if let Some(width) = query.feature_scalar_bytes {
        include_target_features(
            &mut description,
            topology,
            query.feature_shapes.clone(),
            width,
            &query.feature_backings,
        )?;
    } else {
        add_missing(
            &mut description,
            "retained target feature scalar representation is not selected".into(),
        );
    }
    if let Some(residency) = residency {
        include_resident_parameter_conversions(&mut description, residency)?;
    } else {
        add_missing(
            &mut description,
            "cached parameter conversions have no residency observation".into(),
        );
    }
    for invocation in &topology.invocations {
        add_missing(&mut description, format!("prediction invocation {} ({:?}): selected mechanism scratch, evaluation retention and output backings are not bound", invocation.node_id, invocation.scope));
    }
    description.validate()?;
    Ok(description)
}
