//! Backend-neutral activation, target-state, and routed-expert observation contracts.

use std::collections::{BTreeMap, BTreeSet};

mod error_bridge;
pub use error_bridge::ObserverErrorBridge;
mod speculative;
pub use speculative::{with_speculative_activation, SpeculativeActivationObserver};

/// Execution site responsible for an architecture-declared internal observation.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ObservationHookSite {
    /// Embedding or other prepared input transformation.
    Input,
    /// An internal boundary within one execution unit.
    Unit,
    /// Sparse expert scalars emitted inside the selected routed provider.
    RoutedUnits,
    /// Final normalization or projection before output publication.
    Readout,
    /// The shared executor's authoritative final-output seam.
    Publication,
}

/// Hook coverage of an actual architecture call path and its selected executor.
/// Coordinates and native collector support do not establish this coverage.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct ObservationHookSupport {
    input: bool,
    unit: bool,
    routed_units: bool,
    readout: bool,
    publication: bool,
}

impl ObservationHookSupport {
    /// Declares internal sites emitted by the corresponding architecture methods.
    /// The enclosing executor declares publication separately.
    pub const fn internal(input: bool, unit: bool, readout: bool) -> Self {
        Self {
            input,
            unit,
            routed_units: false,
            readout,
            publication: false,
        }
    }

    /// Refines unit coverage when a specialized executor supplies the unit call.
    pub const fn with_units(mut self, supported: bool) -> Self {
        self.unit = supported;
        self
    }

    /// Declares sparse provider hooks independently of other internal boundaries.
    pub const fn with_routed_units(mut self, supported: bool) -> Self {
        self.routed_units = supported;
        self
    }

    /// Declares the enclosing executor's authoritative output seam.
    pub const fn with_publication(mut self, supported: bool) -> Self {
        self.publication = supported;
        self
    }

    /// Whether this actual call path emits the requested site.
    pub const fn supports(self, site: ObservationHookSite) -> bool {
        match site {
            ObservationHookSite::Input => self.input,
            ObservationHookSite::Unit => self.unit,
            ObservationHookSite::RoutedUnits => self.routed_units,
            ObservationHookSite::Readout => self.readout,
            ObservationHookSite::Publication => self.publication,
        }
    }
}

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
            || (point.routed_units.is_some() && !mechanisms.routed_units)
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
    /// Selected call path supplies activation hooks inside prediction invocations.
    /// Target hooks and a declared prediction topology alone do not establish this.
    pub prediction_inspection: bool,
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
    observation_support_with_partition(catalog, context, |_| {
        eredu_core::ObservationSupportStatus::Unverified(
            "Partition-local ownership and routing observation coverage are not yet described"
                .into(),
        )
    })
}

/// Combines retained invocation placement and collector facts with the ordinary
/// selected-session gates. The callback is consulted only for partitioned
/// points whose architecture phase and native activation mechanism are available.
/// It must describe the actual selected ownership and submit no native work.
pub fn observation_support_with_partition(
    catalog: &eredu_core::ObservationCatalog,
    context: ObservationExecutionContext,
    mut partition: impl FnMut(&eredu_core::ObservationPoint) -> eredu_core::ObservationSupportStatus,
) -> eredu_core::ObservationSupportReport {
    eredu_core::ObservationSupportReport {
        schema_version: eredu_core::DISCOVERY_SCHEMA_VERSION,
        capture: Default::default(),
        points: catalog
            .points
            .iter()
            .map(|point| eredu_core::ObservationSupport {
                path: point.path.clone(),
                prefill: point_support(point, point.prefill, context, &mut partition),
                decode: point_support(point, point.decode, context, &mut partition),
                floating_to_f32: context.mechanisms.floating_to_f32,
            })
            .collect(),
    }
}

fn point_support(
    point: &eredu_core::ObservationPoint,
    phase_available: bool,
    context: ObservationExecutionContext,
    partition: &mut impl FnMut(&eredu_core::ObservationPoint) -> eredu_core::ObservationSupportStatus,
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
    if point.requirements.contains(&R::PredictionExecution) && !context.prediction_inspection {
        return S::Unsupported(
            "Selected call path does not supply prediction activation hooks".into(),
        );
    }
    if matches!(
        point.value_type,
        eredu_core::ObservationValueType::RoutedUnits { .. }
    ) && !context.mechanisms.routed_unit_tensors
    {
        return S::Unsupported("Backend does not collect bounded routed-unit values".into());
    }
    if point.requirements.contains(&R::RoutingEvents) && !context.mechanisms.routing_tensors {
        return S::Unsupported("Backend does not collect normalized routing events".into());
    }
    if context.partitioned {
        match partition(point) {
            S::Supported => {}
            status => return status,
        }
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
    /// Whether this observer needs the shared session's transactional lifecycle.
    /// The value must stay fixed throughout one forward. Partitioned sessions
    /// agree participation before invoking callbacks that may communicate.
    fn transactional(&self) -> bool {
        false
    }

    /// Performs local admission before model/state work. This callback must not
    /// submit collectives: peers may be rejecting their own local admission.
    fn prepare_transaction(
        &mut self,
        _epoch: eredu_core::DistributedCommitEpoch,
        _pass: crate::ExpertPass,
    ) -> Result<(), E> {
        Ok(())
    }

    /// Coordinates already prepared observer authority at the common pre-forward
    /// boundary. Every rank participates, including inactive pipeline stages.
    fn coordinate_transaction(
        &mut self,
        _epoch: eredu_core::DistributedCommitEpoch,
    ) -> Result<(), E> {
        Ok(())
    }

    /// Stages delivery after exact output/state completion and before state
    /// commitment. It may communicate at this common all-rank boundary. Results
    /// must remain provisional until `finish_transaction` reports commitment.
    fn complete_transaction(
        &mut self,
        _epoch: eredu_core::DistributedCommitEpoch,
    ) -> Result<(), E> {
        Ok(())
    }

    /// Publishes staged records after commitment or discards them after failure.
    /// This infallible notification must perform no native work or communication.
    fn finish_transaction(&mut self, _epoch: eredu_core::DistributedCommitEpoch, _committed: bool) {
    }

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

    /// Supplies admitted sparse-unit work for one routed invocation. Providers
    /// carry this observer through their native units, compaction and exchange.
    /// An absent observer must not add unit tensor work to ordinary execution.
    fn routed_unit_observer(
        &mut self,
        _path: &str,
    ) -> Result<Option<&mut dyn crate::RoutedUnitObserver<T>>, E> {
        Ok(None)
    }

    /// Completes an ordinary forward pass under its existing failure owner.
    /// Scheduled-but-missing interventions must fail before prediction commitment.
    fn finish(&mut self) -> Result<(), E> {
        Ok(())
    }

    /// Observes a named backend-native tensor.
    fn observe(&mut self, path: &str, value: &T) -> Result<(), E>;

    /// Offers an ordinary replica's source at an authoritative observation seam.
    /// The default performs no work: this rank neither publishes an observation
    /// nor applies an intervention. A partition collector can complete prepaid
    /// dependencies here while the selected owner supplies the actual record.
    fn observe_replica(&mut self, _path: &str, _value: &T) -> Result<(), E> {
        Ok(())
    }

    /// Offers a diagnostic value that requires additional native work. The
    /// prototype supplies its exact geometry without creating that work. Bounded
    /// observers reserve creation and capture costs before invoking the factory;
    /// absent or skipped points must not invoke it. The default observes all values.
    fn observe_generated(
        &mut self,
        path: &str,
        _prototype: &T,
        _source: &eredu_core::capture::GeneratedCaptureSource,
        generate: &mut dyn FnMut() -> Result<T, E>,
    ) -> Result<(), E> {
        self.observe(path, &generate()?)
    }

    /// Optionally replaces an activation before it is consumed or returned.
    fn intervene(&mut self, _path: &str, _value: &T) -> Result<Option<T>, E> {
        Ok(None)
    }

    /// Observes normalized routed-expert decisions and contributions.
    fn observe_routing(&mut self, _routing: RoutingObservation<'_, T>) -> Result<(), E> {
        Ok(())
    }
}

/// Sized adapter for a borrowed observer, including a trait object.
pub struct BorrowedActivationObserver<'a, O: ?Sized>(pub &'a mut O);

impl<T, E, O: ActivationObserver<T, E> + ?Sized> ActivationObserver<T, E>
    for BorrowedActivationObserver<'_, O>
{
    fn routed_unit_observer(
        &mut self,
        path: &str,
    ) -> Result<Option<&mut dyn crate::RoutedUnitObserver<T>>, E> {
        self.0.routed_unit_observer(path)
    }
    fn transactional(&self) -> bool {
        self.0.transactional()
    }
    fn prepare_transaction(
        &mut self,
        epoch: eredu_core::DistributedCommitEpoch,
        pass: crate::ExpertPass,
    ) -> Result<(), E> {
        self.0.prepare_transaction(epoch, pass)
    }
    fn coordinate_transaction(
        &mut self,
        epoch: eredu_core::DistributedCommitEpoch,
    ) -> Result<(), E> {
        self.0.coordinate_transaction(epoch)
    }
    fn complete_transaction(&mut self, epoch: eredu_core::DistributedCommitEpoch) -> Result<(), E> {
        self.0.complete_transaction(epoch)
    }
    fn finish_transaction(&mut self, epoch: eredu_core::DistributedCommitEpoch, committed: bool) {
        self.0.finish_transaction(epoch, committed)
    }
    fn observe(&mut self, path: &str, value: &T) -> Result<(), E> {
        self.0.observe(path, value)
    }
    fn observe_replica(&mut self, path: &str, value: &T) -> Result<(), E> {
        self.0.observe_replica(path, value)
    }
    fn observe_generated(
        &mut self,
        path: &str,
        prototype: &T,
        source: &eredu_core::capture::GeneratedCaptureSource,
        generate: &mut dyn FnMut() -> Result<T, E>,
    ) -> Result<(), E> {
        self.0.observe_generated(path, prototype, source, generate)
    }
    fn intervene(&mut self, path: &str, value: &T) -> Result<Option<T>, E> {
        self.0.intervene(path, value)
    }
    fn routing_control(
        &mut self,
        path: &str,
        rows: u64,
    ) -> Result<Option<eredu_nn::routing_intervention::GroupSelectionControl>, E> {
        self.0.routing_control(path, rows)
    }
    fn routing_applied(
        &mut self,
        path: &str,
        original: Option<RoutingDecision<'_, T>>,
        effective: RoutingDecision<'_, T>,
    ) -> Result<(), E> {
        self.0.routing_applied(path, original, effective)
    }
    fn routing_failed(&mut self, path: &str, message: &str) {
        self.0.routing_failed(path, message)
    }
    fn observe_routing(&mut self, routing: RoutingObservation<'_, T>) -> Result<(), E> {
        self.0.observe_routing(routing)
    }
    fn finish(&mut self) -> Result<(), E> {
        self.0.finish()
    }
}

/// Keeps provisional host evidence from escaping an unsuccessful forward,
/// including early returns and unwinding. Native rollback stays with the session.
pub(crate) struct ObservationTransactionGuard<'a, T, E, O: ActivationObserver<T, E> + ?Sized> {
    pub(crate) observer: &'a mut O,
    epoch: eredu_core::DistributedCommitEpoch,
    active: bool,
    marker: std::marker::PhantomData<fn(T) -> E>,
}

impl<'a, T, E, O: ActivationObserver<T, E> + ?Sized> ObservationTransactionGuard<'a, T, E, O> {
    pub(crate) fn new(observer: &'a mut O, epoch: eredu_core::DistributedCommitEpoch) -> Self {
        let active = observer.transactional();
        Self {
            observer,
            epoch,
            active,
            marker: std::marker::PhantomData,
        }
    }
    pub(crate) fn finish(mut self, committed: bool) {
        if self.active {
            self.active = false;
            self.observer.finish_transaction(self.epoch, committed);
        }
    }
}

impl<T, E, O: ActivationObserver<T, E> + ?Sized> Drop for ObservationTransactionGuard<'_, T, E, O> {
    fn drop(&mut self) {
        if self.active {
            self.observer.finish_transaction(self.epoch, false);
        }
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
    fn observe_generated(
        &mut self,
        _: &str,
        _: &T,
        _: &eredu_core::capture::GeneratedCaptureSource,
        _: &mut dyn FnMut() -> Result<T, E>,
    ) -> Result<(), E> {
        Ok(())
    }
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
    fn prediction_declarations_require_hooks_in_the_selected_call_path() {
        use eredu_core::*;
        let catalog = ObservationCatalog {
            schema_version: DISCOVERY_SCHEMA_VERSION,
            completeness: DescriptionCompleteness::Complete,
            points: vec![ObservationPoint {
                path: "prediction.values".into(),
                node_id: "prediction".into(),
                meaning: "Prediction activation".into(),
                value_type: ObservationValueType::Tensor,
                dtype: ObservationDtype::Floating,
                axes: None,
                prefill: true,
                decode: true,
                requirements: vec![
                    ObservationRequirement::ActivationHooks,
                    ObservationRequirement::PredictionExecution,
                ],
                position: ObservationPosition::BeforeIntervention,
                retained_bytes: None,
                host_bytes: None,
            }],
        };
        let mut context = ObservationExecutionContext {
            activation_inspection: true,
            prediction_inspection: false,
            selected: true,
            partitioned: true,
            mechanisms: ObservationMechanisms {
                activation_tensors: true,
                routing_tensors: true,
                routed_unit_tensors: true,
                floating_to_f32: true,
            },
        };
        let denied = observation_support_with_partition(&catalog, context, |_| {
            panic!("no prediction hooks")
        });
        assert!(matches!(
            denied.points[0].prefill,
            ObservationSupportStatus::Unsupported(_)
        ));
        assert!(matches!(
            denied.points[0].decode,
            ObservationSupportStatus::Unsupported(_)
        ));
        context.prediction_inspection = true;
        let missing_transport = observation_support_with_partition(&catalog, context, |_| {
            ObservationSupportStatus::Unverified("Prediction transport not selected".into())
        });
        assert!(matches!(
            missing_transport.points[0].prefill,
            ObservationSupportStatus::Unverified(_)
        ));
        let supported = observation_support_with_partition(&catalog, context, |_| {
            ObservationSupportStatus::Supported
        });
        assert!(matches!(
            supported.points[0].prefill,
            ObservationSupportStatus::Conditional(_)
        ));
        assert!(matches!(
            supported.points[0].decode,
            ObservationSupportStatus::Conditional(_)
        ));
    }

    #[test]
    fn partition_placement_cannot_override_phase_selection_or_collector_gates() {
        use eredu_core::*;
        let point = ObservationPoint {
            path: "component".into(),
            node_id: "node".into(),
            meaning: "values".into(),
            value_type: ObservationValueType::Tensor,
            dtype: ObservationDtype::Floating,
            axes: None,
            prefill: true,
            decode: false,
            requirements: vec![ObservationRequirement::ActivationHooks],
            position: ObservationPosition::BeforeIntervention,
            retained_bytes: None,
            host_bytes: None,
        };
        let catalog = ObservationCatalog {
            schema_version: 1,
            completeness: DescriptionCompleteness::Complete,
            points: vec![point],
        };
        let context = ObservationExecutionContext {
            activation_inspection: true,
            prediction_inspection: false,
            partitioned: true,
            selected: true,
            mechanisms: ObservationMechanisms {
                activation_tensors: true,
                routing_tensors: false,
                routed_unit_tensors: false,
                floating_to_f32: true,
            },
        };
        let mut calls = 0;
        let supported = observation_support_with_partition(&catalog, context, |_| {
            calls += 1;
            ObservationSupportStatus::Supported
        });
        assert_eq!(calls, 1);
        assert_eq!(
            supported.points[0].prefill,
            ObservationSupportStatus::Supported
        );
        assert!(matches!(
            supported.points[0].decode,
            ObservationSupportStatus::Unsupported(_)
        ));
        for context in [
            ObservationExecutionContext {
                selected: false,
                ..context
            },
            ObservationExecutionContext {
                activation_inspection: false,
                ..context
            },
            ObservationExecutionContext {
                mechanisms: ObservationMechanisms {
                    activation_tensors: false,
                    ..context.mechanisms
                },
                ..context
            },
        ] {
            let result = observation_support_with_partition(&catalog, context, |_| {
                panic!("disabled collector")
            });
            assert_ne!(
                result.points[0].prefill,
                ObservationSupportStatus::Supported
            );
        }
        let result = observation_support_with_partition(&catalog, context, |_| {
            ObservationSupportStatus::Unverified("selected hook transport unavailable".into())
        });
        assert!(matches!(
            result.points[0].prefill,
            ObservationSupportStatus::Unverified(_)
        ));
    }

    #[test]
    fn noop_observer_is_static_and_passthrough() {
        fn observe<O: ActivationObserver<i32, ()>>(observer: &mut O) {
            observer.observe("layer.output", &7).unwrap();
            assert_eq!(observer.intervene("layer.output", &7).unwrap(), None);
        }
        observe(&mut NoopObserver);
    }

    #[test]
    fn sparse_partition_support_requires_its_native_collector_and_exact_hook() {
        use eredu_core::*;
        let sparse = ObservationHookSupport::default().with_routed_units(true);
        assert!(sparse.supports(ObservationHookSite::RoutedUnits));
        assert!(!sparse.supports(ObservationHookSite::Unit));
        assert!(!ObservationHookSupport::internal(true, true, true)
            .supports(ObservationHookSite::RoutedUnits));
        let catalog = ObservationCatalog {
            schema_version: 1,
            completeness: DescriptionCompleteness::Complete,
            points: vec![ObservationPoint {
                path: "bank.units".into(),
                node_id: "bank".into(),
                meaning: "expert scalars".into(),
                value_type: ObservationValueType::RoutedUnits {
                    routing: "bank".into(),
                    geometry: capture::RoutedUnitGeometry {
                        experts: 5,
                        units_per_expert: 6,
                        routes_per_token: 2,
                    },
                },
                dtype: ObservationDtype::Floating,
                axes: None,
                prefill: true,
                decode: true,
                requirements: vec![ObservationRequirement::ActivationHooks],
                position: ObservationPosition::BeforeIntervention,
                retained_bytes: None,
                host_bytes: None,
            }],
        };
        let mut context = ObservationExecutionContext {
            activation_inspection: true,
            prediction_inspection: false,
            partitioned: true,
            selected: true,
            mechanisms: ObservationMechanisms {
                activation_tensors: true,
                routing_tensors: true,
                routed_unit_tensors: false,
                floating_to_f32: true,
            },
        };
        let unsupported = observation_support_with_partition(&catalog, context, |_| {
            panic!("no sparse collector")
        });
        assert!(matches!(
            unsupported.points[0].prefill,
            ObservationSupportStatus::Unsupported(_)
        ));
        context.mechanisms.routed_unit_tensors = true;
        for expected in [
            ObservationSupportStatus::Supported,
            ObservationSupportStatus::Unverified(
                "selected invocation group lacks a native hook".into(),
            ),
        ] {
            let actual =
                observation_support_with_partition(&catalog, context, |_| expected.clone());
            assert_eq!(actual.points[0].prefill, expected);
            assert_eq!(actual.points[0].decode, expected);
        }
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
        // Ordinary observers receive no duplicate observation or intervention
        // from a nonpublishing rank's differently valued source.
        BorrowedActivationObserver(&mut ReplacingLogits)
            .observe_replica(eredu_core::MODEL_LOGITS_OBSERVATION_PATH, &99)
            .unwrap();
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
