//! Startup composition for the ordinary embedded invocation contracts.
use super::*;
use crate::prediction_resources::{
    EmbeddedPredictionTopology, PredictionExecutionMode, PredictionStateLayer,
};

/// Serializable embedded startup geometry and explicit mechanism calibration.
/// Parameters are deliberately absent: they belong to the target residency owner.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddedPredictionMemoryPlan {
    /// Ordinary sequential-depth or fused-row invocation contract.
    pub mode: PredictionExecutionMode,
    /// Exact selected prediction state components and context offsets.
    pub state: Vec<PredictionStateLayer>,
    /// Ordinary cache allocation granularity used for the capacity envelope.
    pub allocation_granularity: u64,
    /// Additional predictor execution, calibrated like the ordinary target.
    pub execution: ExecutionMemoryPlan,
    /// Captured target-feature payload per target position, before transaction copies.
    pub target_feature_bytes_per_position: MemoryBytes,
    /// Coverage that cannot be inferred from the admitted ordinary contracts.
    pub missing: Vec<String>,
}

impl EmbeddedPredictionMemoryPlan {
    /// Derives a pure startup plan from retained selection. No resource is created,
    /// observed, reserved, or evaluated. Shared target weights remain in residency.
    pub fn from_topology(
        topology: &EmbeddedPredictionTopology,
        target: &GenerationMemoryRequest,
        calibration: &ForecastCalibration,
    ) -> Result<Self, CapabilityError> {
        let execution = target
            .domains
            .iter()
            .flat_map(|d| &d.executions)
            .next()
            .ok_or_else(|| invalid("embedded prediction has no target execution pool"))?;
        let mut prediction = execution.clone();
        let layers = if topology.state.is_empty() {
            eredu_core::LayerSchedule::empty()
        } else {
            eredu_core::LayerSchedule::new(
                topology.state.len(),
                topology
                    .state
                    .iter()
                    .map(|layer| layer.policy.clone())
                    .collect(),
            )
            .map_err(|error| invalid(error.to_string()))?
        };
        prediction.state_layout = eredu_core::StateMemoryLayout::new(
            layers,
            topology
                .state
                .iter()
                .map(|layer| layer.processed_token_offset)
                .collect(),
            topology
                .execution_topology
                .as_ref()
                .map_or(execution.state_layout.hidden_size, |topology| {
                    topology.hidden_size
                }),
            execution.state_layout.allocation_granularity,
            execution.state_layout.completeness,
        )?;
        prediction.workspace = None;
        prediction.execution_topology = topology.execution_topology.clone();
        prediction.logits = LogitsWorkspace::EveryPosition;
        let mut request = target.clone();
        for domain in &mut request.domains {
            domain.executions = if domain.executions.is_empty() {
                vec![]
            } else {
                vec![prediction.clone()]
            };
        }
        calibration.apply(&mut request)?;
        prediction = request
            .domains
            .into_iter()
            .flat_map(|d| d.executions)
            .next()
            .unwrap();
        let promoted = topology
            .execution_topology
            .as_ref()
            .is_some_and(|t| t.selected_parameter_promotion_bytes.is_some())
            || target.domains.iter().flat_map(|d| &d.executions).any(|e| {
                e.execution_topology
                    .as_ref()
                    .is_some_and(|t| t.selected_parameter_promotion_bytes.is_some())
            });
        let mut features = MemoryBytes::exact(0);
        for entry in topology.target_features.entries() {
            let shape = entry.shape();
            // The ordinary target-capture contract is [batch, sequence, ...].
            // Keep unfamiliar capture layouts explicit rather than silently
            // treating a request-dependent axis as a single row.
            if shape.len() < 3
                || shape[0] != 1
                || !entry.bounded_dimensions().contains(&1)
                || entry.bounded_dimensions().iter().any(|&axis| axis > 1)
            {
                features = features.add(&MemoryBytes::unknown(format!(
                    "target feature {} has no single-lane sequence-axis contract",
                    entry.path().as_str()
                )))?;
                continue;
            }
            let bytes = shape[2..]
                .iter()
                .try_fold(u64::from(target.scalar_bytes.get()), |n, &dim| {
                    mul(n, dim as u64)
                })?;
            let upper_width = if promoted {
                u64::from(target.scalar_bytes.get()).max(4)
            } else {
                u64::from(target.scalar_bytes.get())
            };
            let upper = shape[2..]
                .iter()
                .try_fold(upper_width, |n, &dim| mul(n, dim as u64))?;
            features = features.add(&MemoryBytes::estimated(
                bytes,
                upper,
                "ordinary retained feature geometry and selected arithmetic promotion",
            ))?;
        }
        let mut missing = topology.missing.clone();
        if target
            .domains
            .iter()
            .filter(|d| !d.executions.is_empty())
            .count()
            != 1
            || target.domains.iter().any(|d| d.executions.len() > 1)
        {
            missing.push(
                "prediction rank-local placement across multiple executions is unavailable".into(),
            );
        }
        Ok(Self {
            mode: topology.mode,
            state: topology.state.clone(),
            allocation_granularity: execution.state_layout.allocation_granularity,
            execution: prediction,
            target_feature_bytes_per_position: features,
            missing,
        })
    }

    pub(super) fn state_growth(
        &self,
        target: &GenerationMemoryRequest,
    ) -> Result<u64, CapabilityError> {
        if self.state.is_empty() {
            return Ok(0);
        }
        let layout = eredu_core::StateMemoryLayout::new(
            eredu_core::LayerSchedule::new(
                self.state.len(),
                self.state
                    .iter()
                    .map(|layer| layer.policy.clone())
                    .collect(),
            )
            .map_err(|error| invalid(error.to_string()))?,
            self.state
                .iter()
                .map(|layer| layer.processed_token_offset)
                .collect(),
            self.execution.state_layout.hidden_size,
            self.allocation_granularity,
            self.execution.state_layout.completeness,
        )?;
        let state = eredu_core::estimate_runtime_state(
            &layout,
            target.input,
            0,
            target.batch_size,
            target.scalar_bytes,
        )?;
        mul(state.bytes_per_position_per_batch, target.batch_size)
    }

    pub(crate) fn validate_continuation(
        &self,
        target: &GenerationMemoryRequest,
        native: &mut EmbeddedContinuationMemoryPlan,
    ) -> Result<(), CapabilityError> {
        if native.layer_positions.len() != self.state.len() {
            return Err(invalid(
                "installed prediction frontiers do not match the selected state members",
            ));
        }
        native.current_state.validate()?;
        native.peak_state.validate()?;
        native.retained_features.validate()?;
        if native
            .layer_positions
            .iter()
            .any(|&position| position > target.input.model_positions)
        {
            return Err(invalid(
                "settled prediction state advances beyond the target frontier",
            ));
        }
        // A sliced capture can pin its full prefix backing. Preserve that upper
        // envelope even when the observed logical view is just one row.
        let prefix = interval(
            &self.target_feature_bytes_per_position,
            target.input.model_positions,
            "retained target capture may pin a complete prefix backing",
        )?;
        native.retained_features = native.retained_features.maximum(&prefix);
        let mut current = 0;
        let mut endpoint = 0;
        for (layer, &positions) in self.state.iter().zip(&native.layer_positions) {
            let layout = eredu_core::StateMemoryLayout::new(
                eredu_core::LayerSchedule::new(1, vec![layer.policy.clone()])
                    .map_err(|e| invalid(e.to_string()))?,
                vec![0],
                self.execution.state_layout.hidden_size,
                self.allocation_granularity,
                self.execution.state_layout.completeness,
            )?;
            let payload = |positions| {
                eredu_core::estimate_runtime_state_payload_lower_bound(
                    &layout,
                    eredu_core::InputTokenCount::text(positions),
                    target.batch_size,
                    target.scalar_bytes,
                )
            };
            current = add(current, payload(positions)?)?;
            endpoint = add(
                endpoint,
                payload(add(positions, native.additional_input_tokens)?)?,
            )?;
        }
        native.current_state.lower_bytes = native.current_state.lower_bytes.max(current);
        native.peak_state.lower_bytes = native
            .peak_state
            .lower_bytes
            .max(endpoint)
            .max(native.current_state.lower_bytes);
        native.current_state.kind = ObservationKind::Estimated;
        native.peak_state.kind = ObservationKind::Estimated;
        native.current_state.validate()?;
        native.peak_state.validate()?;
        if native
            .current_state
            .upper_bytes
            .zip(native.peak_state.upper_bytes)
            .is_some_and(|(current, peak)| current > peak)
        {
            return Err(invalid(
                "prediction horizon capacity does not include installed state",
            ));
        }
        Ok(())
    }

    pub(crate) fn costs(
        &self,
        target: &GenerationMemoryRequest,
        positions: u64,
        query: u64,
        prefill: bool,
    ) -> Result<(MemoryBytes, MemoryBytes, MemoryBytes), CapabilityError> {
        self.costs_with_state(target, positions, query, prefill, None)
    }

    pub(super) fn costs_with_state(
        &self,
        target: &GenerationMemoryRequest,
        positions: u64,
        query: u64,
        prefill: bool,
        installed: Option<&MemoryBytes>,
    ) -> Result<(MemoryBytes, MemoryBytes, MemoryBytes), CapabilityError> {
        if self.allocation_granularity == 0 {
            return Err(invalid(
                "prediction cache allocation granularity must be positive",
            ));
        }
        self.target_feature_bytes_per_position.validate()?;
        let mut state = MemoryBytes::exact(0);
        for layer in &self.state {
            if layer.processed_token_offset > 0 {
                return Err(invalid(
                    "prediction state offset advances beyond target context",
                ));
            }
            layer
                .policy
                .validate()
                .map_err(|e| invalid(e.to_string()))?;
            let frontier =
                positions.saturating_sub(u64::from(layer.processed_token_offset.unsigned_abs()));
            let capacity = mul(
                frontier.div_ceil(self.allocation_granularity),
                self.allocation_granularity,
            )?;
            for component in layer.policy.components() {
                let bytes = match component.dtype() {
                    eredu_core::cache::StateTensorDtype::Floating => {
                        u64::from(target.scalar_bytes.get())
                    }
                    _ => 4,
                };
                let lower_frontier = if !matches!(
                    component.role(),
                    eredu_core::cache::StateComponentRole::Fixed(_)
                ) {
                    layer
                        .policy
                        .attention()
                        .and_then(|a| a.window())
                        .map_or(frontier, |w| frontier.min(u64::from(w.get())))
                } else {
                    frontier
                };
                let actual = component
                    .element_bounds(target.batch_size, lower_frontier)
                    .map_err(|e| invalid(e.to_string()))?;
                // Include interior peaks (pooling/remainder state) and possible
                // allocation slack, not merely the final logical endpoint.
                let peak = component
                    .element_peak_bounds(target.batch_size, 0, capacity)
                    .map_err(|e| invalid(e.to_string()))?;
                let promoted = self
                    .execution
                    .execution_topology
                    .as_ref()
                    .is_some_and(|t| t.selected_parameter_promotion_bytes.is_some())
                    || target.domains.iter().flat_map(|d| &d.executions).any(|e| {
                        e.execution_topology
                            .as_ref()
                            .is_some_and(|t| t.selected_parameter_promotion_bytes.is_some())
                    });
                let upper_bytes = if promoted
                    && component.dtype() == eredu_core::cache::StateTensorDtype::Floating
                {
                    bytes.max(4)
                } else {
                    bytes
                };
                state = state.add(&MemoryBytes::estimated(
                    mul(actual.minimum, bytes)?, mul(peak.maximum, upper_bytes)?,
                    "prediction state component envelope including context offset, interior peaks and cache granularity",
                ))?;
            }
        }
        if let Some(installed) = installed {
            state = installed.clone();
        }
        let rows = if prefill {
            self.mode
                .prefill_sequence_len(usize::try_from(query).map_err(|_| {
                    CapabilityError::ArithmeticOverflow {
                        operation: "prediction prefill row count",
                    }
                })?) as u64
        } else {
            query
        };
        let mut workspace = if rows == 0 {
            MemoryBytes::exact(0)
        } else if self.execution.execution_topology.is_some() {
            crate::workspace_resources::workspace(
                &self.execution,
                target,
                positions,
                rows,
                state.upper_bytes.unwrap_or(state.lower_bytes),
            )?
        } else {
            MemoryBytes::unknown("embedded prediction module invocation topology unavailable")
        };
        for reason in &self.missing {
            workspace = workspace.add(&MemoryBytes::unknown(format!(
                "embedded prediction: {reason}"
            )))?;
        }
        let features = interval(&self.target_feature_bytes_per_position, positions,
            "retained target-feature logical payload may be a view of target workspace; no distinct lower allocation is inferred")?;
        Ok((state, workspace, features))
    }
}

fn invalid(detail: impl Into<String>) -> CapabilityError {
    CapabilityError::InvalidConfiguration {
        field: "embedded prediction forecast",
        detail: detail.into(),
    }
}

/// Reuses observed cached conversions through their exact native retaining bindings.
/// Their physical backing is already charged once in target residency. This only
/// removes future conversion payload for matching selected materialization tasks;
/// neither an aggregate byte count nor an unbound semantic alias receives credit.
/// Only complete matching binding payloads receive credit: smaller allocations
/// do not establish subrange coverage. Reapplying the observation is idempotent.
pub fn apply_embedded_parameter_conversion_credit(
    target: &mut GenerationMemoryRequest,
    prediction: &mut EmbeddedPredictionMemoryPlan,
    conversions: Option<&[crate::ResidentParameterConversion]>,
) -> Result<Vec<String>, CapabilityError> {
    let Some(conversions) = conversions else {
        return Ok(vec!["Embedded conversion ownership observation unavailable; selected future-conversion allowances remain conservative.".into()]);
    };
    let resident = crate::residency::conversion_payload_bytes(conversions)
        .map_err(|error| invalid(error.to_string()))?;
    let available = target
        .domains
        .iter()
        .try_fold(0, |n, d| add(n, d.resident_parameters.lower_bytes))?;
    if resident > available {
        return Err(invalid(
            "observed conversion backing exceeds declared parameter residency",
        ));
    }
    let mut candidate_target = target.clone();
    let mut candidate_prediction = prediction.clone();
    let mut removed = 0;
    for topology in candidate_target
        .domains
        .iter_mut()
        .flat_map(|d| &mut d.executions)
        .chain(std::iter::once(&mut candidate_prediction.execution))
        .filter_map(|e| e.execution_topology.as_mut())
    {
        let Some(total) = topology.selected_parameter_promotion_bytes.as_mut() else {
            continue;
        };
        let attributed = topology
            .selected_parameter_promotion_payloads
            .values()
            .try_fold(0, |n, &bytes| add(n, bytes))?;
        if attributed > *total {
            return Err(invalid(
                "selected conversion attribution exceeds aggregate payload",
            ));
        }
        let mut credited = 0;
        for conversion in conversions {
            let eredu_core::resources::ResourceSize::Fixed { extent } = &conversion.allocation.size
            else {
                unreachable!("validated current conversion");
            };
            let names = conversion
                .bindings
                .iter()
                .filter_map(|binding| binding.logical_target.as_deref())
                .collect::<std::collections::BTreeSet<_>>();
            for name in names {
                if let Some(remaining) =
                    topology.selected_parameter_promotion_payloads.get_mut(name)
                {
                    // A binding identifies its complete tensor, not a subrange.
                    // Partial observations or multiple smaller allocations do
                    // not prove that its future conversion is already resident.
                    if *remaining == extent.payload.lower_bytes {
                        credited = add(credited, *remaining)?;
                        *remaining = 0;
                    }
                }
            }
        }
        if credited > *total {
            return Err(invalid(
                "attributed conversion credit exceeds selected conversion payload",
            ));
        }
        *total -= credited;
        removed = add(removed, credited)?;
    }
    *target = candidate_target;
    *prediction = candidate_prediction;
    Ok(vec![format!("Observed {resident} bytes of distinct resident conversion backing; authoritative retaining bindings replace {removed} bytes of selected target/prediction future-conversion allowances. Shared aliases retain one parameter residency charge; partial or unattributed binding payloads receive no future-work credit.")])
}
