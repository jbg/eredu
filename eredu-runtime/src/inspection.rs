//! Backend-neutral activation, target-state, and routed-expert observation contracts.

use std::collections::{BTreeMap, BTreeSet};

/// Combines explicit architecture intervention declarations with exact loaded
/// observation support and backend arithmetic facts. Read-only catalog values
/// never acquire intervention capabilities through this projection.
pub fn intervention_support(
    mut points: Vec<eredu_core::intervention::InterventionPoint>,
    capture: &eredu_core::capture::CaptureDiscovery,
    mechanisms: &eredu_core::intervention::InterventionMechanisms,
) -> eredu_core::intervention::InterventionDiscovery {
    use eredu_core::{intervention::*, ObservationSupportStatus as S};
    for point in &mut points {
        point
            .operations
            .retain(|kind| mechanisms.operations.contains(kind));
        point
            .dtypes
            .retain(|dtype| mechanisms.dtypes.contains(dtype));
        point
            .score_stages
            .retain(|stage| mechanisms.score_stages.contains(stage));
        let path = if point.routing.is_some() {
            eredu_core::RoutingObservationField::SelectedExperts.path(&point.path)
        } else {
            point.path.clone()
        };
        if let Some(support) = capture.support.points.iter().find(|p| p.path == path) {
            point.prefill = support.prefill.clone();
            point.decode = support.decode.clone();
        }
        if point.axes.is_empty()
            || point.operations.is_empty()
            || (point.routing.is_none() && point.dtypes.is_empty())
        {
            point.prefill = S::Unsupported(
                "required intervention geometry or native mechanism is not declared".into(),
            );
            point.decode = point.prefill.clone();
        }
    }
    InterventionDiscovery {
        schema_version: INTERVENTION_SCHEMA_VERSION,
        artifact_identity: capture.artifact_identity.clone(),
        session_identity: None,
        points,
    }
}

/// Selected, model-independent conditions used to report capture support.
#[derive(Debug, Clone, Copy)]
pub struct ObservationExecutionContext {
    /// Exact admitted session enables the instrumented execution route.
    pub activation_inspection: bool,
    /// Rank ownership and provider-specific routing need more precise discovery.
    pub partitioned: bool,
    /// An execution configuration was successfully selected.
    pub selected: bool,
    /// Side-effect-free native collector facts.
    pub mechanisms: eredu_core::ObservationMechanisms,
}

/// Resolves support without modifying the logical graph or inventing observations.
pub fn observation_support(
    catalog: &eredu_core::ObservationCatalog,
    context: ObservationExecutionContext,
) -> eredu_core::ObservationSupportReport {
    eredu_core::ObservationSupportReport {
        schema_version: eredu_core::DISCOVERY_SCHEMA_VERSION,
        capture: Default::default(),
        points: catalog
            .points
            .iter()
            .map(|point| eredu_core::ObservationSupport {
                path: point.path.clone(),
                prefill: point_support(point, point.prefill, context),
                decode: point_support(point, point.decode, context),
                floating_to_f32: context.mechanisms.floating_to_f32,
            })
            .collect(),
    }
}

fn point_support(
    point: &eredu_core::ObservationPoint,
    phase_available: bool,
    context: ObservationExecutionContext,
) -> eredu_core::ObservationSupportStatus {
    use eredu_core::{ObservationRequirement as R, ObservationSupportStatus as S};
    if !phase_available {
        return S::Unsupported("The architecture does not emit this point in this phase".into());
    }
    if !context.selected {
        return S::Unverified("No admitted execution configuration".into());
    }
    if !context.activation_inspection {
        return S::Unsupported("Selected session does not enable activation inspection".into());
    }
    if !context.mechanisms.activation_tensors {
        return S::Unsupported("Backend has not declared tensor capture support".into());
    }
    if point.requirements.contains(&R::RoutingEvents) && !context.mechanisms.routing_tensors {
        return S::Unsupported("Backend does not collect normalized routing events".into());
    }
    if context.partitioned {
        return S::Unverified(
            "Partition-local ownership and routing observation coverage are not yet described"
                .into(),
        );
    }
    if point.requirements.contains(&R::MediaInput) {
        return S::Conditional("Requires the corresponding media input during prefill".into());
    }
    if point.requirements.contains(&R::PredictionExecution) {
        return S::Conditional("Requires the corresponding prediction execution group".into());
    }
    S::Supported
}

/// One ordinary block output selected for a target/draft consumer.
pub struct TargetStateTap<'a, T> {
    /// Architecture block ordinal.
    pub layer: usize,
    /// Backend-native block output.
    pub value: &'a T,
}

/// Ordered target-state capture without backend storage or family policy.
#[derive(Debug, Clone)]
pub struct TargetStateCapture<T> {
    requested: Vec<usize>,
    captured: BTreeMap<usize, T>,
}

impl<T> TargetStateCapture<T> {
    /// Creates a capture plan with exact, ordered layer identities.
    pub fn new(
        requested: impl IntoIterator<Item = usize>,
    ) -> Result<Self, TargetStateCaptureError> {
        let requested = requested.into_iter().collect::<Vec<_>>();
        if requested.is_empty() {
            return Err(TargetStateCaptureError::Empty);
        }
        let mut unique = BTreeSet::new();
        if let Some(duplicate) = requested.iter().find(|layer| !unique.insert(**layer)) {
            return Err(TargetStateCaptureError::DuplicateRequest(*duplicate));
        }
        Ok(Self {
            requested,
            captured: BTreeMap::new(),
        })
    }

    /// Returns whether this plan requests one block output.
    pub fn wants(&self, layer: usize) -> bool {
        self.requested.contains(&layer)
    }

    /// Captures one requested block output exactly once.
    pub fn capture(&mut self, tap: TargetStateTap<'_, T>) -> Result<(), TargetStateCaptureError>
    where
        T: Clone,
    {
        if !self.wants(tap.layer) {
            return Err(TargetStateCaptureError::Unrequested(tap.layer));
        }
        if self.captured.insert(tap.layer, tap.value.clone()).is_some() {
            return Err(TargetStateCaptureError::DuplicateCapture(tap.layer));
        }
        Ok(())
    }

    /// Returns captured values in declared request order, rejecting omissions.
    pub fn into_ordered(mut self) -> Result<Vec<T>, TargetStateCaptureError> {
        let mut ordered = Vec::with_capacity(self.requested.len());
        for layer in self.requested {
            ordered.push(
                self.captured
                    .remove(&layer)
                    .ok_or(TargetStateCaptureError::Missing(layer))?,
            );
        }
        Ok(ordered)
    }
}

/// Invalid target-state capture lifecycle.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum TargetStateCaptureError {
    /// At least one state tap must be requested.
    #[error("target-state capture requires at least one layer")]
    Empty,
    /// One layer was requested more than once.
    #[error("target-state layer {0} was requested more than once")]
    DuplicateRequest(usize),
    /// A block emitted a state that was not requested.
    #[error("target-state layer {0} was not requested")]
    Unrequested(usize),
    /// A requested state was captured more than once.
    #[error("target-state layer {0} was captured more than once")]
    DuplicateCapture(usize),
    /// A requested block never emitted its output.
    #[error("target-state layer {0} was not captured")]
    Missing(usize),
}

/// Normalized routed-expert data emitted by architecture implementations.
pub struct RoutingObservation<'a, T> {
    /// Stable path-like name of the routed block.
    pub path: &'a str,
    /// Selected expert IDs shaped `[..., top_k]`.
    pub selected_experts: &'a T,
    /// Selected scores before optional top-k renormalization.
    pub selected_scores: &'a T,
    /// Final route weights applied to expert outputs.
    pub coefficients: &'a T,
    /// Combined routed expert contribution.
    pub routed_output: &'a T,
    /// Rank-local contribution before expert-parallel reduction.
    pub local_routed_output: Option<&'a T>,
    /// Globally reduced expert-parallel contribution.
    pub reduced_routed_output: Option<&'a T>,
    /// Shared-expert contribution when the architecture has one.
    pub shared_output: Option<&'a T>,
    /// Combined routed and shared contribution when reported separately.
    pub combined_output: Option<&'a T>,
    /// Total number of routed experts.
    pub expert_count: i32,
}

impl<T> RoutingObservation<'_, T> {
    /// Enumerates only present event fields using the same typed paths as discovery.
    pub fn for_each_tensor(&self, mut visit: impl FnMut(String, &T)) {
        use eredu_core::RoutingObservationField as Field;
        for (field, value) in [
            (Field::SelectedExperts, Some(self.selected_experts)),
            (Field::SelectedScores, Some(self.selected_scores)),
            (Field::Coefficients, Some(self.coefficients)),
            (Field::RoutedOutput, Some(self.routed_output)),
            (Field::LocalRoutedOutput, self.local_routed_output),
            (Field::ReducedRoutedOutput, self.reduced_routed_output),
            (Field::SharedOutput, self.shared_output),
            (Field::CombinedOutput, self.combined_output),
        ] {
            if let Some(value) = value {
                visit(field.path(self.path), value);
            }
        }
    }
}

/// Statically dispatched activation observation and intervention contract.
pub trait ActivationObserver<T, E> {
    /// Obtains a validated control before the selector dispatches any experts.
    /// The default ordinary path allocates and materializes nothing.
    fn routing_control(
        &mut self,
        _path: &str,
        _token_rows: u64,
    ) -> Result<Option<eredu_nn::routing_intervention::GroupSelectionControl>, E> {
        Ok(None)
    }

    /// Receives original/effective decisions before they reach the expert provider.
    fn routing_applied(
        &mut self,
        _path: &str,
        _original: Option<RoutingDecision<'_, T>>,
        _effective: RoutingDecision<'_, T>,
    ) -> Result<(), E> {
        Ok(())
    }

    /// Records a failed control under the enclosing forward pass's failure owner.
    fn routing_failed(&mut self, _path: &str, _message: &str) {}

    /// Completes an ordinary forward pass under its existing failure owner.
    /// Scheduled-but-missing interventions must fail before prediction commitment.
    fn finish(&mut self) -> Result<(), E> {
        Ok(())
    }

    /// Observes a named backend-native tensor.
    fn observe(&mut self, path: &str, value: &T) -> Result<(), E>;

    /// Optionally replaces an activation before it is consumed or returned.
    fn intervene(&mut self, _path: &str, _value: &T) -> Result<Option<T>, E> {
        Ok(None)
    }

    /// Observes normalized routed-expert decisions and contributions.
    fn observe_routing(&mut self, _routing: RoutingObservation<'_, T>) -> Result<(), E> {
        Ok(())
    }
}

/// Borrowed pre-dispatch decision, independent of expert outputs or shared experts.
pub struct RoutingDecision<'a, T> {
    /// Exact global expert IDs shaped `[token_rows, top_k]`.
    pub ids: &'a T,
    /// Final coefficients used for dispatch.
    pub coefficients: &'a T,
}

impl<'a, T> From<&'a eredu_nn::GroupSelection<T>> for RoutingDecision<'a, T> {
    fn from(routes: &'a eredu_nn::GroupSelection<T>) -> Self {
        Self {
            ids: routes.group_indices(),
            coefficients: routes.coefficients(),
        }
    }
}

/// Observes an activation and applies an optional replacement without
/// materializing its backend-native value.
pub fn observe_and_intervene<T, E, O>(observer: &mut O, path: &str, value: &T) -> Result<T, E>
where
    T: Clone,
    O: ActivationObserver<T, E> + ?Sized,
{
    observer.observe(path, value)?;
    Ok(observer
        .intervene(path, value)?
        .unwrap_or_else(|| value.clone()))
}

/// Observes final model logits and applies an optional replacement.
///
/// Family and topology adapters must return this value, rather than merely
/// reporting [`eredu_core::MODEL_LOGITS_OBSERVATION_PATH`], so final-output
/// intervention has the same semantics as every other activation point.
pub fn observe_model_logits<T, E, O>(observer: &mut O, logits: &T) -> Result<T, E>
where
    T: Clone,
    O: ActivationObserver<T, E> + ?Sized,
{
    observe_and_intervene(observer, eredu_core::MODEL_LOGITS_OBSERVATION_PATH, logits)
}

/// Zero-sized observer used by the ordinary unobserved inference path.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopObserver;

impl<T, E> ActivationObserver<T, E> for NoopObserver {
    fn observe(&mut self, _path: &str, _value: &T) -> Result<(), E> {
        Ok(())
    }
}

impl<T, E, F> ActivationObserver<T, E> for F
where
    F: FnMut(&str, &T) -> Result<(), E>,
{
    fn observe(&mut self, path: &str, value: &T) -> Result<(), E> {
        self(path, value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_observer_is_static_and_passthrough() {
        fn observe<O: ActivationObserver<i32, ()>>(observer: &mut O) {
            observer.observe("layer.output", &7).unwrap();
            assert_eq!(observer.intervene("layer.output", &7).unwrap(), None);
        }
        observe(&mut NoopObserver);
    }

    #[test]
    fn observed_activation_can_be_replaced_without_an_erased_hot_path() {
        struct ReplacingObserver {
            observed: Vec<String>,
        }

        impl ActivationObserver<i32, ()> for ReplacingObserver {
            fn observe(&mut self, path: &str, _value: &i32) -> Result<(), ()> {
                self.observed.push(path.into());
                Ok(())
            }

            fn intervene(&mut self, path: &str, value: &i32) -> Result<Option<i32>, ()> {
                Ok((path == "model.layers.0.output").then_some(value + 4))
            }
        }

        let mut observer = ReplacingObserver {
            observed: Vec::new(),
        };
        let output = observe_and_intervene(&mut observer, "model.layers.0.output", &3).unwrap();
        assert_eq!(output, 7);
        assert_eq!(observer.observed, ["model.layers.0.output"]);
    }

    #[test]
    fn final_logits_observation_returns_the_intervention() {
        struct ReplacingLogits;

        impl ActivationObserver<i32, ()> for ReplacingLogits {
            fn observe(&mut self, path: &str, value: &i32) -> Result<(), ()> {
                assert_eq!(path, eredu_core::MODEL_LOGITS_OBSERVATION_PATH);
                assert_eq!(*value, 3);
                Ok(())
            }

            fn intervene(&mut self, path: &str, value: &i32) -> Result<Option<i32>, ()> {
                assert_eq!(path, eredu_core::MODEL_LOGITS_OBSERVATION_PATH);
                Ok(Some(value + 4))
            }
        }

        assert_eq!(observe_model_logits(&mut ReplacingLogits, &3), Ok(7));
    }

    #[test]
    fn target_states_are_captured_once_in_request_order() {
        let mut capture = TargetStateCapture::new([5, 1, 3]).unwrap();
        assert!(capture.wants(1));
        assert!(!capture.wants(2));
        capture
            .capture(TargetStateTap {
                layer: 1,
                value: &10,
            })
            .unwrap();
        capture
            .capture(TargetStateTap {
                layer: 5,
                value: &50,
            })
            .unwrap();
        capture
            .capture(TargetStateTap {
                layer: 3,
                value: &30,
            })
            .unwrap();
        assert_eq!(capture.into_ordered().unwrap(), [50, 10, 30]);
    }

    #[test]
    fn target_state_capture_rejects_duplicates_omissions_and_unrequested_layers() {
        assert_eq!(
            TargetStateCapture::<i32>::new([2, 2]).unwrap_err(),
            TargetStateCaptureError::DuplicateRequest(2)
        );
        let mut capture = TargetStateCapture::new([2]).unwrap();
        assert_eq!(
            capture
                .capture(TargetStateTap {
                    layer: 3,
                    value: &7,
                })
                .unwrap_err(),
            TargetStateCaptureError::Unrequested(3)
        );
        assert_eq!(
            capture.into_ordered().unwrap_err(),
            TargetStateCaptureError::Missing(2)
        );
    }
}
