//! Shared lifecycle integration for globally admitted partition observations.
use super::*;
use eredu_core::{Completion, DistributedCommitEpoch};

/// Architecture-owned producers and all ranks that execute the actual hook.
/// Replicas may be hook members without producing duplicate capture fragments.
pub struct PartitionCapturePlacement {
    /// Distinct global coverage from actual local tensors.
    pub producers: Vec<PartitionCaptureProducer>,
    /// Sorted unique world ranks that encounter this hook, including replicas.
    pub hook_members: Vec<usize>,
    /// Exact ordinary source geometry on each hook member, in the same order.
    /// Includes nonexporting replicas and shards with empty local selections.
    pub source_shapes: Vec<Vec<u64>>,
}

/// One executing routed provider, including idle owners and nonexporting replicas.
#[derive(Debug, Clone)]
pub struct PartitionRoutedCaptureSource {
    /// Exact world rank of the invocation member.
    pub rank: usize,
    /// Retained expert, scalar-column and original source-peer coordinates.
    pub ownership: RoutedUnitCaptureOwnership,
    /// Last dimension of the actual provider input, before internal chunking.
    pub input_width: u64,
}

/// Architecture-owned sparse placement. Source members may outnumber producers.
pub struct PartitionRoutedCapturePlacement {
    /// Canonical invocation path requested by the prepared provider.
    pub routing: String,
    /// Whether this selection observes effective rather than original unit values.
    pub effective: bool,
    /// Distinct authoritative producers of the global selected routes and units.
    pub producers: Vec<PartitionRoutedCaptureProducer>,
    /// Sorted unique executing world ranks, including idle owners and replicas.
    pub sources: Vec<PartitionRoutedCaptureSource>,
}

/// Cold semantic projection supplied by the retained architecture selection.
/// Runtime owns admission and native work; implementations declare only where
/// this exact invocation executes and how its local scalar coordinates map.
pub trait PartitionCaptureLayout {
    /// Projects an explicitly shaped invocation through retained architecture
    /// ownership. Implementations must preserve the original admitted plan.
    fn capture_placement_at(
        &self,
        plan: &AdmittedCapturePlan,
        index: usize,
        phase: CapturePhase,
        prediction: u64,
        invocation: Option<CaptureInvocationShape>,
        limits: PartitionCaptureReceiptLimits,
    ) -> Result<PartitionCapturePlacement, CaptureError> {
        if invocation.is_some() {
            return Err(CaptureError::Unsupported(
                "partition layout has no explicit invocation projection".into(),
            ));
        }
        self.capture_placement(plan, index, phase, prediction, limits)
    }
    /// Sparse placement under an independently admitted physical invocation.
    fn routed_capture_placement_at(
        &self,
        plan: &AdmittedCapturePlan,
        index: usize,
        phase: CapturePhase,
        prediction: u64,
        invocation: Option<CaptureInvocationShape>,
        limits: PartitionCaptureReceiptLimits,
    ) -> Result<PartitionRoutedCapturePlacement, CaptureError> {
        if invocation.is_some() {
            return Err(CaptureError::Unsupported(
                "partition layout has no explicit routed invocation projection".into(),
            ));
        }
        self.routed_capture_placement(plan, index, phase, prediction, limits)
    }
    /// Whether this invocation publishes disjoint coordinates or complete
    /// selected additive terms. This fact must come from retained architecture
    /// semantics, never be inferred from duplicate producer geometry.
    fn capture_combination(
        &self,
        _path: &str,
    ) -> Result<PartitionCaptureCombination, CaptureError> {
        Ok(PartitionCaptureCombination::Disjoint)
    }
    /// Resolves one scheduled selection without tensors, devices or collectives.
    fn capture_placement(
        &self,
        plan: &AdmittedCapturePlan,
        index: usize,
        phase: CapturePhase,
        prediction: u64,
        limits: PartitionCaptureReceiptLimits,
    ) -> Result<PartitionCapturePlacement, CaptureError>;
    /// Resolves sparse provider placement without source tensors or native work.
    fn routed_capture_placement(
        &self,
        _plan: &AdmittedCapturePlan,
        _index: usize,
        _phase: CapturePhase,
        _prediction: u64,
        _limits: PartitionCaptureReceiptLimits,
    ) -> Result<PartitionRoutedCapturePlacement, CaptureError> {
        Err(CaptureError::Unsupported(
            "partitioned routed placement is not implemented".into(),
        ))
    }
}

/// Observer failures retain both typed capture errors and native transport causes.
#[derive(Debug, thiserror::Error)]
pub enum PartitionCaptureObserverError<E: std::error::Error + 'static> {
    /// Producer transform, factory admission, or ordinary capture policy failed.
    #[error(transparent)]
    Capture(#[from] CaptureExecutionError<E>),
    /// Coordination, exact completion, or global receipt delivery failed.
    #[error(transparent)]
    Exchange(#[from] PartitionCaptureExchangeError),
    /// Another member of this invocation rejected its producer work.
    #[error("a partition capture hook member rejected its source work")]
    HookRejected,
}

impl<E: std::error::Error + 'static> From<CaptureError> for PartitionCaptureObserverError<E> {
    fn from(error: CaptureError) -> Self {
        Self::Capture(CaptureExecutionError::Admission(error))
    }
}

struct Selection<'a, T: PartitionCaptureHookTransport> {
    work: SessionPartitionCapture<'a, T>,
    hook: Option<SessionPartitionHook<'a, T>>,
    source: Option<SessionPartitionSource<'a, T>>,
    routed: Option<RoutedSelection>,
}

struct RoutedSelection {
    routing: String,
    effective: bool,
    ownership: Option<RoutedUnitCaptureOwnership>,
}

mod routed;

// Static adapters let capture-only backends retain the ordinary observer. Native
// composition explicitly supplies the stronger activation mechanism and layout.
struct InterventionCallbacks<B: CaptureBackend, T: PartitionCaptureHookTransport, L> {
    begin_routed: fn(
        &mut CaptureSession,
        &mut SessionPartitionIntervention<'_, T>,
        &mut B,
        &crate::RoutedUnitInvocation<'_, B::Tensor>,
        bool,
    ) -> Result<(), PartitionCaptureObserverError<B::Error>>,
    apply_routed: fn(
        &mut CaptureSession,
        &mut SessionPartitionIntervention<'_, T>,
        &mut B,
        &PartitionRoutedUnitCaptureSource<'_, B::Tensor>,
    ) -> Result<Option<B::Tensor>, PartitionCaptureObserverError<B::Error>>,
    finish_routed: fn(
        &mut CaptureSession,
        &mut SessionPartitionIntervention<'_, T>,
        bool,
    ) -> Result<(), PartitionCaptureObserverError<B::Error>>,
    prepare:
        for<'t> fn(
            &mut CaptureSession,
            &'t T,
            &L,
            &B,
            usize,
            PartitionCaptureReceiptLimits,
        )
            -> Result<SessionPartitionIntervention<'t, T>, PartitionCaptureExchangeError>,
    apply: fn(
        &mut CaptureSession,
        &mut SessionPartitionIntervention<'_, T>,
        &mut B,
        &B::Tensor,
    ) -> Result<Option<B::Tensor>, PartitionCaptureObserverError<B::Error>>,
}

/// Uses the ordinary shared forward's preparation, coordination, exact completion
/// and final commit callbacks. It performs no world collective from a local hook.
/// Native composition provides retained transport, architecture layout, estimates
/// and error conversion. Construction submits and reserves nothing.
pub struct PartitionCaptureObserver<
    'a,
    B: CaptureBackend,
    T: PartitionCaptureHookTransport,
    L,
    N,
    F,
> {
    session: &'a mut CaptureSession,
    backend: B,
    transport: &'a T,
    layout: &'a L,
    estimate: N,
    map_error: F,
    prediction: Option<u64>,
    limits: PartitionCaptureReceiptLimits,
    selections: Vec<Selection<'a, T>>,
    interventions: Vec<SessionPartitionIntervention<'a, T>>,
    intervention_callbacks: Option<InterventionCallbacks<B, T, L>>,
    coordination: Option<SessionPartitionCoordination<'a, T>>,
    identity: Option<PartitionCaptureIdentity>,
    size_receipts: bool,
    routed_path: Option<String>,
    routed_active: bool,
    routed_error: Option<&'a dyn Fn(eredu_nn::Error) -> eredu_nn::Error>,
}

impl<'a, B: CaptureBackend, T: PartitionCaptureHookTransport, L, N, F>
    PartitionCaptureObserver<'a, B, T, L, N, F>
{
    /// Borrows one unstarted forward of an already partition-bound capture run.
    pub fn for_step(
        session: &'a mut CaptureSession,
        backend: B,
        transport: &'a T,
        layout: &'a L,
        prediction: u64,
        limits: PartitionCaptureReceiptLimits,
        estimate: N,
        map_error: F,
    ) -> Self {
        Self {
            session,
            backend,
            transport,
            layout,
            estimate,
            map_error,
            prediction: Some(prediction),
            limits,
            selections: Vec::new(),
            interventions: Vec::new(),
            intervention_callbacks: None,
            coordination: None,
            identity: None,
            size_receipts: false,
            routed_path: None,
            routed_active: false,
            routed_error: None,
        }
    }

    /// Records routed-provider failures before crossing the neural error domain.
    /// The handler must preserve the original cause and performs no native work.
    pub fn with_routed_error_handler(
        mut self,
        handler: &'a dyn Fn(eredu_nn::Error) -> eredu_nn::Error,
    ) -> Self {
        self.routed_error = Some(handler);
        self
    }

    fn routed_result<R>(&self, result: Result<R, eredu_nn::Error>) -> Result<R, eredu_nn::Error> {
        result.map_err(|error| match self.routed_error {
            Some(handler) => handler(error),
            None => error,
        })
    }

    /// Enables globally projected activation operations through this same owner
    /// and protocol, using the native activation mechanism and retained layout.
    pub fn with_interventions(mut self) -> Self
    where
        B: eredu_core::intervention::InterventionBackend,
        L: crate::intervention::PartitionActivationLayout + PartitionCaptureLayout,
        T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        self.intervention_callbacks = Some(InterventionCallbacks {
            begin_routed: CaptureSession::begin_partition_routed_intervention::<B, T>,
            apply_routed: CaptureSession::apply_partition_routed_intervention::<B, T>,
            finish_routed: CaptureSession::finish_partition_routed_intervention::<B::Error, T>,
            prepare: CaptureSession::prepare_partition_intervention::<B, T, L>,
            apply: CaptureSession::apply_partition_intervention::<B, T>,
        });
        self
    }

    /// Selects loaded-session binding and derives each receipt byte allowance
    /// from its actual fragments and native encoded-size estimates. The supplied
    /// limit remains a ceiling; small selections do not reserve that whole ceiling.
    /// Binding and sizing happen in shared preparation, before any model work.
    pub fn with_session_identity(mut self, identity: PartitionCaptureIdentity) -> Self {
        self.identity = Some(identity);
        self.size_receipts = true;
        self
    }
}

impl<B, T, L, N, F, E> crate::ActivationObserver<B::Tensor, E>
    for PartitionCaptureObserver<'_, B, T, L, N, F>
where
    B: CaptureBackend,
    B::Error: Send + Sync,
    T: PartitionCaptureHookTransport,
    L: PartitionCaptureLayout,
    T::Error: Send + Sync + 'static,
    <T::Completion as Completion>::Error: Send + Sync + 'static,
    N: FnMut(
        &[u64],
        &CaptureSelection,
        &ResolvedCaptureSlice,
    ) -> Result<PartitionCaptureNativeEstimate, CaptureError>,
    F: Fn(PartitionCaptureObserverError<B::Error>) -> E,
{
    fn routed_unit_observer(
        &mut self,
        path: &str,
    ) -> Result<Option<&mut dyn crate::RoutedUnitObserver<B::Tensor>>, E> {
        if !self.selections.iter().any(|selection| {
            selection
                .routed
                .as_ref()
                .is_some_and(|routed| routed.routing == path)
        }) && !self
            .interventions
            .iter()
            .any(|work| work.routed_path() == Some(path))
        {
            return Ok(None);
        }
        if self.routed_active && self.routed_path.as_deref() != Some(path) {
            return Err((self.map_error)(
                CaptureExecutionError::Admission(CaptureError::Invalid(
                    "routed capture invocation changed before completion".into(),
                ))
                .into(),
            ));
        }
        if self.routed_path.as_deref() != Some(path) {
            self.routed_path = Some(path.into());
        }
        Ok(Some(self))
    }
    fn transactional(&self) -> bool {
        true
    }

    fn prepare_transaction(
        &mut self,
        epoch: DistributedCommitEpoch,
        pass: crate::ExpertPass,
    ) -> Result<(), E> {
        let prepare = (|| -> Result<(), PartitionCaptureObserverError<B::Error>> {
            if let Some(identity) = self.identity.take() {
                self.session
                    .ensure_partition_capture(identity)
                    .map_err(CaptureExecutionError::Admission)?;
            }
            let prediction = self.prediction.take().ok_or_else(|| {
                CaptureExecutionError::Admission(CaptureError::Invalid(
                    "partition observer already prepared this forward".into(),
                ))
            })?;
            if self.session.plan.invocation_bounds().is_some() {
                if self.session.prediction != prediction || self.session.invocation.is_none() {
                    return Err(CaptureError::Invalid(
                        "partition observer requires the admitted active invocation".into(),
                    )
                    .into());
                }
                self.session
                    .prepare_transaction(epoch, pass)
                    .map_err(CaptureExecutionError::Admission)?;
            } else {
                self.session
                    .prepare_step_transaction(epoch, pass, prediction)
                    .map_err(CaptureExecutionError::Admission)?;
            }
            let phase = self.session.phase;
            let coordination = self
                .session
                .reserve_partition_coordination(self.transport)?;
            for index in 0..self.session.plan().plan().selections.len() {
                if self
                    .session
                    .records
                    .as_ref()
                    .expect("prepared capture step")[index]
                    .outcome
                    != CaptureOutcome::Missing
                {
                    continue;
                }
                let before = self.session.ledger.step();
                let prepared =
                    (|| -> Result<Selection<'_, T>, PartitionCaptureObserverError<B::Error>> {
                        let plan = self.session.plan.clone();
                        let selection = &plan.plan().selections[index];
                        if matches!(selection.transform, CaptureTransform::RoutedUnits) {
                            return self.prepare_routed_selection(index, phase, prediction);
                        }
                        let placement = self
                            .layout
                            .capture_placement_at(
                                self.session.plan(),
                                index,
                                phase,
                                prediction,
                                self.session.invocation,
                                self.limits,
                            )
                            .map_err(CaptureExecutionError::Admission)?;
                        let combination = self.layout.capture_combination(&selection.path)?;
                        let native_selection = super::sum::reserve_native_selection(
                            &plan,
                            index,
                            combination,
                            self.transport.participant_count(),
                            &mut self.session.ledger,
                        )?;
                        let mut limits = self.limits;
                        if self.size_receipts {
                            let point = &self.session.plan().points()[index];
                            let bound = placement
                                .producers
                                .iter()
                                .try_fold(0u64, |largest, producer| {
                                    // Bounded context strings and the outer producer envelope.
                                    let bytes = producer.projection.fragments().iter().try_fold(
                                        16_384u64,
                                        |bytes, fragment| {
                                            let native = (self.estimate)(
                                                producer.projection.local_shape(),
                                                &native_selection,
                                                fragment.local(),
                                            )?;
                                            add(
                                                bytes,
                                                add(
                                                    fragment_metadata_usage(
                                                        selection,
                                                        point,
                                                        producer.projection.global_shape().len(),
                                                    )?
                                                    .encoded_bytes,
                                                    native.capture.encoded_bytes,
                                                )?,
                                            )
                                        },
                                    )?;
                                    Ok::<_, CaptureError>(largest.max(bytes))
                                })
                                .map_err(CaptureExecutionError::Admission)?;
                            if bound > limits.max_record_bytes {
                                return Err(CaptureExecutionError::Admission(
                                    CaptureError::Limit {
                                        budget: CaptureBudget::Encoded,
                                        cumulative: false,
                                    },
                                )
                                .into());
                            }
                            limits.max_record_bytes = bound;
                        }
                        let mut work = self.session.prepare_partition_capture_combined(
                            self.transport,
                            index,
                            placement.producers,
                            combination,
                            limits,
                            &mut self.estimate,
                        )?;
                        let hook = self
                            .session
                            .prepare_partition_hook(&mut work, placement.hook_members.clone())?;
                        let source = self.session.prepare_partition_source(
                            &mut work,
                            &self.backend,
                            placement.hook_members,
                            placement.source_shapes,
                        )?;
                        Ok(Selection {
                            work,
                            hook: Some(hook),
                            source,
                            routed: None,
                        })
                    })();
                match prepared {
                    Ok(selection) => self.selections.push(selection),
                    Err(error) => {
                        let limit = match &error {
                            PartitionCaptureObserverError::Capture(
                                CaptureExecutionError::Admission(CaptureError::Limit {
                                    budget,
                                    cumulative,
                                }),
                            )
                            | PartitionCaptureObserverError::Exchange(
                                PartitionCaptureExchangeError::Capture(CaptureError::Limit {
                                    budget,
                                    cumulative,
                                }),
                            ) => Some((*budget, *cumulative)),
                            _ => None,
                        };
                        if let Some((budget, cumulative)) = limit.filter(|_| {
                            self.session.plan().plan().limits.on_limit == CaptureLimitPolicy::Skip
                        }) {
                            self.session
                                .skip_partition_selection(index, budget, cumulative, before)
                                .map_err(CaptureExecutionError::Admission)?;
                        } else {
                            return Err(error);
                        }
                    }
                }
            }
            if let Some(plan) = self.session.intervention_plan() {
                let count = plan.plan().operations.len();
                let callbacks = self.intervention_callbacks.as_ref().ok_or_else(|| {
                    CaptureExecutionError::Admission(CaptureError::Unsupported(
                        "partition observer has no activation mechanism".into(),
                    ))
                })?;
                for index in 0..count {
                    if self
                        .session
                        .interventions
                        .as_ref()
                        .expect("retained admission")
                        .records
                        .as_ref()
                        .expect("prepared intervention step")[index]
                        .outcome
                        != eredu_core::intervention::InterventionOutcome::Missing
                    {
                        continue;
                    }
                    self.interventions.push((callbacks.prepare)(
                        self.session,
                        self.transport,
                        self.layout,
                        &self.backend,
                        index,
                        self.limits,
                    )?);
                }
            }
            self.coordination = Some(self.session.seal_partition_coordination(coordination)?);
            Ok(())
        })();
        prepare.map_err(&self.map_error)
    }

    fn coordinate_transaction(&mut self, _epoch: DistributedCommitEpoch) -> Result<(), E> {
        let started = std::time::Instant::now();
        let result = (|| {
            let coordination = self.coordination.take().ok_or_else(|| {
                (self.map_error)(
                    PartitionCaptureExchangeError::Capture(CaptureError::Invalid(
                        "partition observer was not prepared".into(),
                    ))
                    .into(),
                )
            })?;
            self.session
                .coordinate_partition_capture(coordination)
                .map_err(|error| (self.map_error)(error.into()))
        })();
        self.session.capture_seconds += started.elapsed().as_secs_f64();
        result
    }

    fn observe(&mut self, path: &str, value: &B::Tensor) -> Result<(), E> {
        let started = std::time::Instant::now();
        let result = (|| {
            for selection in &mut self.selections {
                if self.session.plan().plan().selections[selection.work.index].path != path {
                    continue;
                }
                let hook = selection.hook.take().ok_or_else(|| {
                    (self.map_error)(
                        CaptureExecutionError::Admission(CaptureError::Invalid(
                            "partition hook was already observed".into(),
                        ))
                        .into(),
                    )
                })?;
                if !hook.participates() {
                    return Err((self.map_error)(
                        CaptureExecutionError::Admission(CaptureError::Invalid(
                            "partition observation reached a nonmember invocation".into(),
                        ))
                        .into(),
                    ));
                }
                let local = (|| {
                    if let Some(source) = selection.source.take() {
                        if let Err(error) = self.session.ready_partition_source(
                            &mut selection.work,
                            source,
                            &mut self.backend,
                            value,
                        ) {
                            self.session.failed_partition_source(&mut selection.work);
                            return Err((self.map_error)(error));
                        }
                    }
                    self.session
                        .observe_partition(&mut selection.work, &mut self.backend, value)
                        .map_err(|error| (self.map_error)(error.into()))
                })();
                let agreed = self
                    .session
                    .agree_partition_hook(&mut selection.work, hook, local.is_ok())
                    .map_err(|error| (self.map_error)(error.into()));
                local?;
                if !agreed? {
                    return Err((self.map_error)(
                        PartitionCaptureObserverError::HookRejected,
                    ));
                }
            }
            Ok(())
        })();
        self.session.capture_seconds += started.elapsed().as_secs_f64();
        result
    }

    fn intervene(&mut self, path: &str, value: &B::Tensor) -> Result<Option<B::Tensor>, E> {
        let started = std::time::Instant::now();
        let result = (|| {
            let mut effective = None;
            for work in &mut self.interventions {
                let plan = self
                    .session
                    .intervention_plan()
                    .expect("prepared admission");
                if work.routed_path().is_some() || plan.plan().operations[work.index].target != path
                {
                    continue;
                }
                let input = effective.as_ref().unwrap_or(value);
                let output =
                    (self
                        .intervention_callbacks
                        .as_ref()
                        .expect("prepared mechanism")
                        .apply)(self.session, work, &mut self.backend, input)
                    .map_err(&self.map_error)?;
                if output.is_some() {
                    effective = output;
                }
            }
            Ok(effective)
        })();
        self.session.capture_seconds += started.elapsed().as_secs_f64();
        result
    }

    fn observe_replica(&mut self, path: &str, value: &B::Tensor) -> Result<(), E> {
        self.observe(path, value)
    }

    fn observe_generated(
        &mut self,
        path: &str,
        prototype: &B::Tensor,
        source: &eredu_core::capture::GeneratedCaptureSource,
        generate: &mut dyn FnMut() -> Result<B::Tensor, E>,
    ) -> Result<(), E> {
        let started = std::time::Instant::now();
        let result = (|| {
            for selection in &mut self.selections {
                if self.session.plan().plan().selections[selection.work.index].path != path {
                    continue;
                }
                let hook = selection.hook.take().ok_or_else(|| {
                    (self.map_error)(
                        CaptureExecutionError::Admission(CaptureError::Invalid(
                            "partition hook was already observed".into(),
                        ))
                        .into(),
                    )
                })?;
                if !hook.participates() {
                    return Err((self.map_error)(
                        CaptureExecutionError::Admission(CaptureError::Invalid(
                            "partition observation reached a nonmember invocation".into(),
                        ))
                        .into(),
                    ));
                }
                let local = (|| {
                    if let Some(preparation) = selection.source.take() {
                        if let Err(error) = self.session.ready_partition_source(
                            &mut selection.work,
                            preparation,
                            &mut self.backend,
                            prototype,
                        ) {
                            self.session.failed_partition_source(&mut selection.work);
                            return Err((self.map_error)(error));
                        }
                    }
                    self.session.observe_generated_partition(
                        &mut selection.work,
                        &mut self.backend,
                        prototype,
                        source,
                        generate,
                        &|error| (self.map_error)(error.into()),
                    )
                })();
                let agreed = self
                    .session
                    .agree_partition_hook(&mut selection.work, hook, local.is_ok())
                    .map_err(|error| (self.map_error)(error.into()));
                local?;
                if !agreed? {
                    return Err((self.map_error)(
                        PartitionCaptureObserverError::HookRejected,
                    ));
                }
            }
            Ok(())
        })();
        self.session.capture_seconds += started.elapsed().as_secs_f64();
        result
    }

    fn complete_transaction(&mut self, epoch: DistributedCommitEpoch) -> Result<(), E> {
        let started = std::time::Instant::now();
        let result = (|| {
            for selection in self.selections.drain(..) {
                self.session
                    .complete_partition_capture(selection.work)
                    .map_err(|error| (self.map_error)(error.into()))?;
            }
            for work in self.interventions.drain(..) {
                self.session
                    .complete_partition_intervention(work)
                    .map_err(|error| (self.map_error)(error.into()))?;
            }
            self.session
                .complete_transaction(epoch)
                .map_err(|error| (self.map_error)(CaptureExecutionError::Admission(error).into()))
        })();
        self.session.capture_seconds += started.elapsed().as_secs_f64();
        result
    }

    fn finish_transaction(&mut self, epoch: DistributedCommitEpoch, committed: bool) {
        self.selections.clear();
        self.interventions.clear();
        self.coordination = None;
        self.routed_path = None;
        self.routed_active = false;
        self.session.finish_transaction(epoch, committed);
    }
}
