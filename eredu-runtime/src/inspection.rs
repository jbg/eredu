//! Backend-neutral activation, target-state, and routed-expert observation contracts.

use std::collections::{BTreeMap, BTreeSet};

mod prefill_retention;
pub use prefill_retention::{
    PrefillChunkRetentionContext, PrefillOpeningExecution, PrefillOpeningState,
    PreparedPrefillChunkRetention, SettledPrefillChunkRetention,
};
pub(crate) use prefill_retention::{RuntimeOpeningExecution, RuntimeOpeningState};

mod error_bridge;
pub use error_bridge::ObserverErrorBridge;
mod intervention_projection;
pub use intervention_projection::{
    FundedInterventionDiscovery, InterventionDiscoveryPreparationError,
    intervention_discovery_preparation_bytes, prepare_intervention_discovery,
};
pub use intervention_projection::{
    static_intervention_validation_control_bytes,
    validate_activation_intervention_declarations_with_phases,
    validate_prepared_intervention_declarations, validate_static_intervention_declarations,
    validate_static_intervention_declarations_with_phases,
};
mod speculative;
pub use speculative::{SpeculativeActivationObserver, with_speculative_activation};

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
    use eredu_core::intervention::*;
    for point in &mut points {
        let support = intervention_projection::support_for(point, &capture.support.points);
        intervention_projection::apply(point, support, mechanisms.borrowed());
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
    observation_support_with_partition_source(
        catalog,
        context,
        eredu_core::capture::CaptureSourceConstruction::new(None),
        |point, _| Ok(partition(point)),
    )
    .expect("ordinary support source construction")
}

/// Same selected support worker with prospective source destinations. The
/// callback must construct its own status under the supplied policy; the caller
/// retains that policy's actual account in the enclosing source and all errors.
pub fn observation_support_with_partition_source(
    catalog: &eredu_core::ObservationCatalog,
    context: ObservationExecutionContext,
    construction: eredu_core::capture::CaptureSourceConstruction<'_>,
    mut partition: impl FnMut(
        &eredu_core::ObservationPoint,
        eredu_core::capture::CaptureSourceConstruction<'_>,
    ) -> Result<
        eredu_core::ObservationSupportStatus,
        eredu_core::capture::CaptureError,
    >,
) -> Result<eredu_core::ObservationSupportReport, eredu_core::capture::CaptureError> {
    use eredu_core::{ObservationSupport, ObservationSupportReport};
    construction.controls(
        observation_phase_validation_control_bytes()
            .and_then(|n| {
                n.checked_add(std::mem::size_of::<(
                    ObservationSupportReport,
                    ObservationSupport,
                    &eredu_core::ObservationCatalog,
                    ObservationExecutionContext,
                )>())
            })
            .and_then(|n| n.checked_add(std::mem::size_of_val(&partition)))
            .ok_or(eredu_core::capture::CaptureError::Overflow)?,
    )?;
    let mut points = construction.vector(catalog.points.len())?;
    for point in &catalog.points {
        points.push(ObservationSupport {
            path: construction.text(&point.path)?,
            prefill: point_support(point, point.prefill, context, construction, &mut partition)?,
            decode: point_support(point, point.decode, context, construction, &mut partition)?,
            floating_to_f32: context.mechanisms.floating_to_f32,
        });
    }
    Ok(ObservationSupportReport {
        schema_version: eredu_core::DISCOVERY_SCHEMA_VERSION,
        capture: Default::default(),
        points,
    })
}

fn point_support(
    point: &eredu_core::ObservationPoint,
    phase_available: bool,
    context: ObservationExecutionContext,
    construction: eredu_core::capture::CaptureSourceConstruction<'_>,
    partition: &mut impl FnMut(
        &eredu_core::ObservationPoint,
        eredu_core::capture::CaptureSourceConstruction<'_>,
    ) -> Result<
        eredu_core::ObservationSupportStatus,
        eredu_core::capture::CaptureError,
    >,
) -> Result<eredu_core::ObservationSupportStatus, eredu_core::capture::CaptureError> {
    if let Some(status) = point_support_before_partition(point, phase_available, context, false) {
        return status.owned(construction);
    }
    if context.partitioned {
        match partition(point, construction)? {
            eredu_core::ObservationSupportStatus::Supported => {}
            status => return Ok(status),
        }
    }
    point_support_after_partition(point, false).owned(construction)
}

#[derive(Clone, Copy)]
enum BorrowedPointSupport {
    Supported,
    Conditional(&'static str),
    Unsupported(&'static str),
    Unverified(&'static str),
}
impl BorrowedPointSupport {
    fn owned(
        self,
        construction: eredu_core::capture::CaptureSourceConstruction<'_>,
    ) -> Result<eredu_core::ObservationSupportStatus, eredu_core::capture::CaptureError> {
        use eredu_core::ObservationSupportStatus as S;
        Ok(match self {
            Self::Supported => S::Supported,
            Self::Conditional(reason) => S::Conditional(construction.text(reason)?),
            Self::Unsupported(reason) => S::Unsupported(construction.text(reason)?),
            Self::Unverified(reason) => S::Unverified(construction.text(reason)?),
        })
    }
    fn admissible(self) -> bool {
        matches!(self, Self::Supported | Self::Conditional(_))
    }
}
fn point_support_before_partition(
    point: &eredu_core::ObservationPoint,
    phase_available: bool,
    context: ObservationExecutionContext,
    prediction_requirement_discharged: bool,
) -> Option<BorrowedPointSupport> {
    use BorrowedPointSupport as S;
    use eredu_core::ObservationRequirement as R;
    let status = if !phase_available {
        S::Unsupported("The architecture does not emit this point in this phase")
    } else if !context.selected {
        S::Unverified("No admitted execution configuration")
    } else if !context.activation_inspection {
        S::Unsupported("Selected session does not enable activation inspection")
    } else if !context.mechanisms.activation_tensors {
        S::Unsupported("Backend has not declared tensor capture support")
    } else if !prediction_requirement_discharged
        && point.requirements.contains(&R::PredictionExecution)
        && !context.prediction_inspection
    {
        S::Unsupported("Selected call path does not supply prediction activation hooks")
    } else if matches!(
        point.value_type,
        eredu_core::ObservationValueType::RoutedUnits { .. }
    ) && !context.mechanisms.routed_unit_tensors
    {
        S::Unsupported("Backend does not collect bounded routed-unit values")
    } else if point.requirements.contains(&R::RoutingEvents) && !context.mechanisms.routing_tensors
    {
        S::Unsupported("Backend does not collect normalized routing events")
    } else {
        return None;
    };
    Some(status)
}
fn point_support_after_partition(
    point: &eredu_core::ObservationPoint,
    prediction_requirement_discharged: bool,
) -> BorrowedPointSupport {
    use eredu_core::ObservationRequirement as R;
    if point.requirements.contains(&R::MediaInput) {
        BorrowedPointSupport::Conditional("Requires the corresponding media input during prefill")
    } else if !prediction_requirement_discharged
        && point.requirements.contains(&R::PredictionExecution)
    {
        BorrowedPointSupport::Conditional("Requires the corresponding prediction execution group")
    } else {
        BorrowedPointSupport::Supported
    }
}

/// Borrowed selected-hook predicate using the same phase/collector requirements
/// as ordinary discovery. Only the architecture owner may discharge the declared
/// prediction requirement with its actual selected invocation hooks. Partitioned
/// execution still needs its separate producer projection and is refused here.
/// This neither allocates diagnostics nor establishes native capture authority.
pub fn observation_phase_is_admissible(
    point: &eredu_core::ObservationPoint,
    phase: eredu_core::capture::CapturePhase,
    context: ObservationExecutionContext,
    prediction_requirement_discharged: bool,
) -> bool {
    let available = match phase {
        eredu_core::capture::CapturePhase::Prefill => point.prefill,
        eredu_core::capture::CapturePhase::Decode => point.decode,
    };
    if let Some(status) =
        point_support_before_partition(point, available, context, prediction_requirement_discharged)
    {
        return status.admissible();
    }
    !context.partitioned
        && point_support_after_partition(point, prediction_requirement_discharged).admissible()
}
/// Exact nonallocating status comparison using the same selected ordinary hook
/// projection. The architecture supplies its real prediction requirement proof.
pub fn observation_phase_matches(
    point: &eredu_core::ObservationPoint,
    phase: eredu_core::capture::CapturePhase,
    context: ObservationExecutionContext,
    prediction_requirement_discharged: bool,
    expected: &eredu_core::ObservationSupportStatus,
) -> bool {
    if context.partitioned {
        return false;
    }
    let available = match phase {
        eredu_core::capture::CapturePhase::Prefill => point.prefill,
        eredu_core::capture::CapturePhase::Decode => point.decode,
    };
    let actual = point_support_before_partition(
        point,
        available,
        context,
        prediction_requirement_discharged,
    )
    .unwrap_or_else(|| point_support_after_partition(point, prediction_requirement_discharged));
    match (actual, expected) {
        (BorrowedPointSupport::Supported, eredu_core::ObservationSupportStatus::Supported) => true,
        (
            BorrowedPointSupport::Conditional(a),
            eredu_core::ObservationSupportStatus::Conditional(b),
        )
        | (
            BorrowedPointSupport::Unsupported(a),
            eredu_core::ObservationSupportStatus::Unsupported(b),
        )
        | (
            BorrowedPointSupport::Unverified(a),
            eredu_core::ObservationSupportStatus::Unverified(b),
        ) => a == b,
        _ => false,
    }
}
/// Fixed controls for the borrowed predicate, including the actual support
/// projection and source requirement iterator. No observation payload is copied.
pub fn observation_phase_validation_control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let frames = [
        size_of::<(
            &eredu_core::ObservationPoint,
            eredu_core::capture::CapturePhase,
            ObservationExecutionContext,
            bool,
        )>(),
        size_of::<(
            &eredu_core::ObservationPoint,
            eredu_core::capture::CapturePhase,
            ObservationExecutionContext,
            bool,
            &eredu_core::ObservationSupportStatus,
        )>(),
        size_of::<(&eredu_core::ObservationPoint, bool)>(),
        size_of::<(
            &eredu_core::ObservationPoint,
            bool,
            ObservationExecutionContext,
            bool,
        )>(),
        size_of::<(&eredu_core::ObservationPoint, bool)>(),
        size_of::<Option<BorrowedPointSupport>>(),
        size_of::<BorrowedPointSupport>(),
        size_of::<bool>(),
        size_of::<std::slice::Iter<'static, eredu_core::ObservationRequirement>>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
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

/// Explicit use of an ordinary selector decision by an observation callback.
/// This read-only declaration performs no work and grants no resource authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingUnmodifiedInterest {
    /// The callback is unused; ordinary selection adds no notification work.
    None,
    /// Only source shape/dtype metadata may be inspected. The callback must not
    /// retain tensor aliases, evaluate values, or create dependent native work.
    Metadata,
    /// The callback may inspect values or retain aliases. Native adapters must
    /// preserve their existing source custody and charge actual attachment work.
    Values,
}

/// Statically dispatched activation observation and intervention contract.
pub trait ActivationObserver<T, E> {
    /// Whether activation paths, generated evidence, or interventions are consumed.
    /// Returning false permits the same equations to omit activation-only hooks;
    /// lifecycle, cancellation, transport, and retained-media callbacks still run.
    fn observes_activations(&self) -> bool {
        true
    }
    /// Observes the actual compact media roots at the encoder/decoder cut.
    /// This borrowed notification grants no source, copy or execution authority.
    fn retained_media_cut(&mut self, _visit: &mut dyn FnMut(&mut dyn FnMut(&T))) -> Result<(), E> {
        Ok(())
    }

    /// Requires the executor's already prepared, borrowed traversal paths.
    /// This requirement stays fixed through one forward. It grants no funding,
    /// admission or completion authority. Missing/stale bindings must reject
    /// before state work; allocating semantic rebinding belongs to preparation.
    fn requires_prepared_traversal(&self) -> bool {
        false
    }

    /// Whether this observer's tensor operations preserve their semantics when
    /// called on explicit prefill spans. Arbitrary whole-tensor callbacks must
    /// opt in deliberately; sequence readout demand alone does not prove this.
    fn supports_prefill_spans(&self) -> bool {
        false
    }

    /// Whether the observer accepts complete architecture context tensors at a
    /// declared target frontier during split prefill. Such values may have
    /// policy-specific context axes; they are not prompt-row fragments.
    fn supports_prefill_context(&self) -> bool {
        false
    }

    /// Announces complete retained context at this installed target frontier.
    /// No prompt-row window is implied. The observer must use the tensor's
    /// declared axes; this grants no allocation or completion authority.
    fn begin_prefill_context(&mut self, _frontier: u64) -> Result<(), E> {
        Ok(())
    }

    /// Whether readout observations/interventions require complete sequence
    /// geometry. Unknown observers preserve their original full-row semantics;
    /// no-op or explicitly final-row collectors can opt out.
    fn requires_sequence_readout(&self) -> bool {
        true
    }

    /// Ordinary prepared-media decoder capture binding. This grants no original
    /// authority; the selected session validates actual source/path/geometry.
    fn ordinary_prefill_capture(&self) -> Option<&crate::capture::OrdinaryPrefillCapture> {
        None
    }

    /// Original accepted physical capture contract, when supplied by its owner.
    /// A truthful sequence requirement alone does not authenticate admission.
    fn admitted_prefill_capture(
        &self,
    ) -> Option<&crate::working_memory::AdmittedPrefillCapture<'_>> {
        None
    }

    /// Original saved-token opening, authenticated separately from prompt-row
    /// assembly. It permits only the source's sealed single-token continuation.
    fn admitted_capture_continuation(
        &self,
    ) -> Option<&crate::working_memory::AdmittedCaptureContinuation<'_>> {
        None
    }

    /// Source-only internal capture declaration for this exact outer invocation.
    /// The native adapter must quote its actual source/shape/scope and substitute
    /// the local funded observer before callbacks. This is not native authority.
    fn original_speculative_capture(
        &self,
    ) -> Option<crate::capture::OriginalSpeculativeCaptureInvocation<'_>> {
        None
    }

    /// Read-only cold source at a drained boundary. Prospective descriptions
    /// grant no invocation, native role, capture destination or completion.
    fn original_speculative_capture_preview(
        &self,
    ) -> Option<crate::capture::OriginalSpeculativeCapturePreview<'_>> {
        None
    }

    /// Receives a model phase's already paid shared frame after exact retirement.
    /// Unknown observers refuse; this handoff performs no capture/native work.
    fn retain_original_speculative_capture(
        &mut self,
        _capture: eredu_core::speculative::SpeculativeActivationCapture,
    ) -> Result<(), crate::capture::CaptureProtocolError> {
        Err(crate::capture::CaptureProtocolError::Invocation)
    }

    /// Whether this observer needs the shared session's transactional lifecycle.
    /// The value must stay fixed throughout one forward. Partitioned sessions
    /// agree participation before invoking callbacks that may communicate.
    fn transactional(&self) -> bool {
        false
    }

    /// Announces the next semantic chunk of one scheduled prefill before its
    /// input is prepared. This does not replace the chunk's ordinary transaction
    /// callbacks or grant execution, allocation, or completion authority.
    /// Errors must enter the existing all-rank input preparation agreement.
    fn begin_prefill_chunk(&mut self, _chunk: &crate::prefill::PrefillChunk) -> Result<(), E> {
        Ok(())
    }

    /// Ends one scheduled prefill after final score indexing and reservation
    /// settlement. `committed` is false on cancellation, error, or unwind; earlier
    /// chunks may nevertheless have committed their state. This notification is
    /// infallible and must perform no native work or communication. It does not
    /// establish native completion or authorize a retained-record drain.
    fn finish_prefill(&mut self, _committed: bool) {}

    /// Requests lexical access to the actual state before input preparation.
    /// The original enclosing request must price traversal/inventory work. This
    /// flag grants no allocation, completion or remaining-capacity authority.
    fn requires_prefill_opening_state(&self) -> bool {
        false
    }

    /// Same original retention boundary with the current selected state source.
    /// The shared driver lends it under its existing preparation guard, before
    /// source.prepare_chunk. Errors enter the unchanged input agreement. The
    /// default preserves existing retention; reading values is explicitly opt-in.
    fn prepare_prefill_chunk_with_opening(
        &mut self,
        context: &PrefillChunkRetentionContext<'_>,
        _opening: &PrefillOpeningState<'_, T>,
    ) -> Result<Option<PreparedPrefillChunkRetention>, E> {
        self.prepare_prefill_chunk_retention(context)
    }

    /// Register exact original-bank retention after chunk annotation and before
    /// input preparation, under the existing reservation guard. None preserves
    /// ordinary full-span retention; it proves no reusable chunk coverage.
    fn prepare_prefill_chunk_retention(
        &mut self,
        _context: &PrefillChunkRetentionContext<'_>,
    ) -> Result<Option<PreparedPrefillChunkRetention>, E> {
        Ok(None)
    }

    /// Consume canonical successful chunk evidence at post-chunk cancellation
    /// readiness. This callback cannot communicate or create new native work.
    /// It may reject retirement; the runtime settles the existing guard and
    /// agrees the rejection before advancing. A ticket alone proves no memory
    /// credit, native-record destruction, or whole-work certification.
    fn retire_prefill_chunk_retention(
        &mut self,
        _settled: SettledPrefillChunkRetention,
    ) -> Result<(), E> {
        Ok(())
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

    /// Borrows the ordinary decision after selection and before expert dispatch.
    /// This does not announce an applied control; the default performs no work.
    /// Declares ordinary-decision interest without allocation or native work.
    fn routing_unmodified_interest(&self, _path: &str) -> RoutingUnmodifiedInterest {
        RoutingUnmodifiedInterest::None
    }

    /// Borrows the actual ordinary decision when the declared interest requests it.
    fn routing_unmodified(
        &mut self,
        _path: &str,
        _effective: RoutingDecision<'_, T>,
    ) -> Result<(), E> {
        Ok(())
    }

    /// Receives original/effective decisions for a requested routing control.
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
    /// Forward the actual generated program and its caller-owned root retention.
    fn observe_generated_retained(
        &mut self,
        path: &str,
        prototype: &T,
        source: &eredu_core::capture::GeneratedCaptureSource,
        factory: &mut dyn eredu_nn::RetainedGeneratedTensorFactory<T, E>,
    ) -> Result<(), E> {
        self.observe_generated(path, prototype, source, &mut || {
            factory.generate(&mut |_| Ok(()))
        })
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
    fn observes_activations(&self) -> bool {
        self.0.observes_activations()
    }
    fn retained_media_cut(&mut self, visit: &mut dyn FnMut(&mut dyn FnMut(&T))) -> Result<(), E> {
        self.0.retained_media_cut(visit)
    }
    fn requires_prepared_traversal(&self) -> bool {
        self.0.requires_prepared_traversal()
    }
    fn supports_prefill_spans(&self) -> bool {
        self.0.supports_prefill_spans()
    }
    fn supports_prefill_context(&self) -> bool {
        self.0.supports_prefill_context()
    }
    fn begin_prefill_context(&mut self, frontier: u64) -> Result<(), E> {
        self.0.begin_prefill_context(frontier)
    }
    fn requires_sequence_readout(&self) -> bool {
        self.0.requires_sequence_readout()
    }
    fn admitted_prefill_capture(
        &self,
    ) -> Option<&crate::working_memory::AdmittedPrefillCapture<'_>> {
        self.0.admitted_prefill_capture()
    }
    fn ordinary_prefill_capture(&self) -> Option<&crate::capture::OrdinaryPrefillCapture> {
        self.0.ordinary_prefill_capture()
    }
    fn admitted_capture_continuation(
        &self,
    ) -> Option<&crate::working_memory::AdmittedCaptureContinuation<'_>> {
        self.0.admitted_capture_continuation()
    }

    fn original_speculative_capture(
        &self,
    ) -> Option<crate::capture::OriginalSpeculativeCaptureInvocation<'_>> {
        self.0.original_speculative_capture()
    }
    fn original_speculative_capture_preview(
        &self,
    ) -> Option<crate::capture::OriginalSpeculativeCapturePreview<'_>> {
        self.0.original_speculative_capture_preview()
    }
    fn retain_original_speculative_capture(
        &mut self,
        capture: eredu_core::speculative::SpeculativeActivationCapture,
    ) -> Result<(), crate::capture::CaptureProtocolError> {
        self.0.retain_original_speculative_capture(capture)
    }

    fn routed_unit_observer(
        &mut self,
        path: &str,
    ) -> Result<Option<&mut dyn crate::RoutedUnitObserver<T>>, E> {
        self.0.routed_unit_observer(path)
    }
    fn transactional(&self) -> bool {
        self.0.transactional()
    }
    fn begin_prefill_chunk(&mut self, chunk: &crate::prefill::PrefillChunk) -> Result<(), E> {
        self.0.begin_prefill_chunk(chunk)
    }
    fn finish_prefill(&mut self, committed: bool) {
        self.0.finish_prefill(committed);
    }
    fn requires_prefill_opening_state(&self) -> bool {
        self.0.requires_prefill_opening_state()
    }
    fn prepare_prefill_chunk_with_opening(
        &mut self,
        context: &PrefillChunkRetentionContext<'_>,
        opening: &PrefillOpeningState<'_, T>,
    ) -> Result<Option<PreparedPrefillChunkRetention>, E> {
        self.0.prepare_prefill_chunk_with_opening(context, opening)
    }
    fn prepare_prefill_chunk_retention(
        &mut self,
        context: &PrefillChunkRetentionContext<'_>,
    ) -> Result<Option<PreparedPrefillChunkRetention>, E> {
        self.0.prepare_prefill_chunk_retention(context)
    }
    fn retire_prefill_chunk_retention(
        &mut self,
        settled: SettledPrefillChunkRetention,
    ) -> Result<(), E> {
        self.0.retire_prefill_chunk_retention(settled)
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
    /// Forward the actual generated program and its caller-owned root retention.
    fn observe_generated_retained(
        &mut self,
        path: &str,
        prototype: &T,
        source: &eredu_core::capture::GeneratedCaptureSource,
        factory: &mut dyn eredu_nn::RetainedGeneratedTensorFactory<T, E>,
    ) -> Result<(), E> {
        self.0
            .observe_generated_retained(path, prototype, source, factory)
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
    /// Declares ordinary-decision interest without allocation or native work.
    fn routing_unmodified_interest(&self, path: &str) -> RoutingUnmodifiedInterest {
        self.0.routing_unmodified_interest(path)
    }

    fn routing_unmodified(
        &mut self,
        path: &str,
        effective: RoutingDecision<'_, T>,
    ) -> Result<(), E> {
        self.0.routing_unmodified(path, effective)
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

/// Keeps a logical prefill observation provisional through all chunk commits,
/// cancellation checks, final score indexing and reservation settlement. This
/// inline borrowed guard allocates nothing and establishes no native completion.
pub(crate) struct PrefillObservationGuard<'a, T, E, O: ActivationObserver<T, E> + ?Sized> {
    pub(crate) observer: &'a mut O,
    active: bool,
    marker: std::marker::PhantomData<fn(T) -> E>,
}

impl<'a, T, E, O: ActivationObserver<T, E> + ?Sized> PrefillObservationGuard<'a, T, E, O> {
    pub(crate) fn new(observer: &'a mut O) -> Self {
        Self {
            observer,
            active: true,
            marker: std::marker::PhantomData,
        }
    }
    pub(crate) fn finish(mut self, committed: bool) {
        self.active = false;
        self.observer.finish_prefill(committed);
    }
}

impl<T, E, O: ActivationObserver<T, E> + ?Sized> Drop for PrefillObservationGuard<'_, T, E, O> {
    fn drop(&mut self) {
        if self.active {
            self.observer.finish_prefill(false);
        }
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
    fn observes_activations(&self) -> bool {
        false
    }
    fn supports_prefill_context(&self) -> bool {
        true
    }
    fn supports_prefill_spans(&self) -> bool {
        true
    }
    fn requires_sequence_readout(&self) -> bool {
        false
    }
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
        assert!(
            !ObservationHookSupport::internal(true, true, true)
                .supports(ObservationHookSite::RoutedUnits)
        );
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

#[cfg(test)]
#[path = "inspection/prefill_tests.rs"]
mod prefill_tests;
