//! Prepaid one-array graph-boundary submissions for the accepted request.
//! The selected graph and actual MLX policy finish determine these populations.
use super::*;
use crate::backend::{
    nn::shared::{
        MlxSubmissionCompletion, NeuralSubmissionShape, OriginalNeuralSubmissionCompletion,
        PreparedNeuralSubmission as NativePreparedNeuralSubmission, SubmissionPreparationCause, SubmissionPreparationError,
    },
    submission_recovery::observed::Observer,
};
use eredu_core::Completion;
use eredu_runtime::{OrderedLayerwiseCompletion, SubmissionBackend};
use safemlx::OriginalScopeObserver;

use eredu_runtime::working_memory::OriginalOperationMetadataCustody;
pub(super) type PreparedNeuralSubmission<C = OriginalOperationMetadataCustody, P = OriginalScopeObserver> = NativePreparedNeuralSubmission<C, P>;
type PreparedSlot = PreparedNeuralSubmission;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct NeuralPopulation {
    pub(super) submissions: usize,
    pub(super) per_forward: usize,
    pub(super) shape: NeuralSubmissionShape,
}
impl NeuralPopulation {
    pub(super) fn from_layout(
        layout: &ExecutionUnitLayout,
        geometry: eredu_core::InferenceGeometry,
    ) -> Result<Self, Error> {
        Self::from_layout_with_final(layout, geometry, true)
    }
    pub(super) fn from_layout_with_final(
        layout: &ExecutionUnitLayout,
        geometry: eredu_core::InferenceGeometry,
        final_submission: bool,
    ) -> Result<Self, Error> {
        let prefill = PrefillControlPlan::new(geometry, true).map_err(memory)?;
        let forwards = prefill
            .span_count()
            .checked_add(geometry.max_output_tokens.saturating_sub(1))
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(overflow)?;
        Self::from_forwards(layout, forwards, final_submission)
    }
    /// One non-text invocation of the same selected policy. Group traversal
    /// and the concrete policy's final-output submission remain separate.
    pub(super) fn single_forward(
        layout: &ExecutionUnitLayout,
        final_submission: bool,
    ) -> Result<Self, Error> {
        Self::from_forwards(layout, 1, final_submission)
    }
    fn from_forwards(
        layout: &ExecutionUnitLayout,
        forwards: usize,
        final_submission: bool,
    ) -> Result<Self, Error> {
        let groups = layout.submission_geometry();
        // Only the concrete MLX policy owns this final-output submission. The
        // neutral graph report deliberately does not include it.
        let per_forward = groups
            .group_submissions_per_forward()
            .checked_add(usize::from(final_submission))
            .ok_or_else(overflow)?;
        Ok(Self {
            submissions: per_forward.checked_mul(forwards).ok_or_else(overflow)?,
            per_forward,
            // Initial/group/final calls all submit one existing tensor. Every
            // once-only slot receives the maximum actual graph fan-out because
            // retained groups can change which sequential slot reaches a call.
            shape: NeuralSubmissionShape::new(1, groups.max_consumers_per_submission())
                .ok_or_else(overflow)?,
        })
    }
    pub(super) fn rust_control_bytes(self) -> Option<u64> {
        measured_layout(
            self,
            &factory::<OriginalOperationMetadataCustody, OriginalScopeObserver>(self.shape, None),
        )?.checked_add(u64::try_from(size_of::<OperationControls>().checked_mul(self.submissions)?).ok()?)
    }
    pub(super) fn native_requirements(self) -> Option<NeuralNativeRequirements> {
        let submissions = self.submissions;
        Some(NeuralNativeRequirements {
            submissions,
            root_handles: submissions.checked_mul(self.shape.arrays())?,
            consumer_waits: submissions.checked_mul(self.shape.consumers())?,
            recovery_observers: submissions.checked_mul(self.shape.consumers().checked_add(1)?)?,
        })
    }
}

/// Native contributions that the accepted role's fit proof must close. These
/// are unfinished storage/producer mechanisms, not model-family limitations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NeuralFitContribution {
    /// One output may reach an arbitrary existing dependency DAG. Graph owns
    /// persistent nodes; Record owns traversal containers. A root count alone
    /// does not prove the complete graph and traversal fit the same role.
    ReachableEvaluationDagInRoleGraph,
    /// Actual outstanding Eval/Wait records and selected StreamReceipt/Event
    /// controls must fit the same retained role's Record arena.
    EvaluationAndWaitReceiptsInRoleRecord,
    /// Stream/Event/backend queue or source-owner controls outside those arenas
    /// require their real owning layouts; arena totals cannot price them.
    OutsideArenaStreamEventAndQueueOwners,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PendingNeuralFit {
    pub(crate) populations: NeuralNativeRequirements,
    pub(crate) roots_per_submission: usize,
    pub(crate) waits_per_submission: usize,
    /// Fixed recipe for the finite wait bank. Preserve per-wait shape for later
    /// role assignment; these in-arena requests are never extra Rust-bank bytes.
    pub(crate) wait_records: Option<safemlx::OperationWaitRecordLayout>,
}
impl PendingNeuralFit {
    pub(crate) fn missing(self) -> [NeuralFitContribution; 3] {
        [
            NeuralFitContribution::ReachableEvaluationDagInRoleGraph,
            NeuralFitContribution::EvaluationAndWaitReceiptsInRoleRecord,
            NeuralFitContribution::OutsideArenaStreamEventAndQueueOwners,
        ]
    }
}
/// The accepting variant requires the actual selected equation recipe augmented
/// with each same-role group boundary; geometry/population alone is insufficient.
/// Source-copy materialization remains an independent enclosing Graph producer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NeuralProducerFit {
    Pending(PendingNeuralFit),
    Recipe(PendingNeuralFit),
}
impl NeuralProducerFit {
    pub(super) fn pending(population: NeuralPopulation) -> Option<Self> {
        Some(Self::Pending(PendingNeuralFit {
            populations: population.native_requirements()?,
            roots_per_submission: population.shape.arrays(),
            waits_per_submission: population.shape.consumers(),
            wait_records: population.native_requirements()?.wait_record_layout(),
        }))
    }
    pub(crate) fn requirement(self) -> PendingNeuralFit {
        match self {
            Self::Pending(requirement) | Self::Recipe(requirement) => requirement,
        }
    }
    pub(super) fn additional_control_bytes(self) -> Option<u64> {
        match self {
            // Typed pending is deliberately consulted by production admission:
            // closing only residency storage cannot activate a neural quote.
            Self::Pending(_) => None,
            Self::Recipe(requirement) => {
                let fixed = requirement.populations.fixed_control_bytes()?;
                let submits = safemlx::OperationEvent::nested_submission_control_bytes()?
                    .checked_mul(requirement.populations.submissions)?;
                let waits = requirement.wait_records?;
                let wait_controls = waits
                    .named_control_bytes()
                    .checked_mul(requirement.populations.consumer_waits)?;
                fixed.checked_add(u64::try_from(submits.checked_add(wait_controls)?).ok()?)
            }
        }
    }
}

impl<U: 'static> OriginalOperationPlan<'_, U> {
    /// Called while quoting the exact selected policy, before arena selection.
    pub(crate) fn bind_neural_recipe(
        &self,
        recipe: &mut crate::backend::nn::workspace::ResidentNativeRecipe,
    ) -> Result<(), Error> {
        let geometry=self.geometry.ok_or_else(unknown)?;
        if self
            .manager
            .original_foreground_disk_descriptors()
            .is_some()
            && self.foreground_disk_descriptors().is_none()
        {
            return Err(unknown());
        }
        recipe.bind_neural_boundaries(
            geometry,
            self.neural.per_forward,
            self.neural.shape.consumers(),
        )?;
        let residency = self.residency.ok_or_else(unknown)?;
        recipe.bind_host_transfer_population(
            geometry,
            residency.forwards,
            residency.transfers,
            residency.observations,
            residency.unprepared.transfer.output_arrays,
            storage::TRANSFERS_PER_NONEMPTY_WINDOW,
            self.window_sources
                .as_deref()
                .ok_or_else(unknown)?
                .as_slice(),
        )?;
        if let Some(source) = self.foreground_disk_descriptors() {
            recipe.bind_foreground_disk_copies(
                source,
                self.foreground_disk_forward_population()
                    .ok_or_else(unknown)?,
                self.foreground_disk_population().ok_or_else(unknown)?,
            )?;
        }
        Ok(())
    }
    /// The quote and later installation both recompute this policy's population
    /// and match the same retained immutable recipe before allocating its bank.
    pub(crate) fn with_neural_recipe(
        mut self,
        recipe: Option<&crate::backend::nn::workspace::ResidentNativeRecipe>,
    ) -> Result<Self, Error> {
        if let Some(recipe) = recipe {
            let geometry=self.geometry.ok_or_else(unknown)?;
            if self
                .manager
                .original_foreground_disk_descriptors()
                .is_some()
                && self.foreground_disk_descriptors().is_none()
            {
                return Err(unknown());
            }
            if !recipe.matches_neural_boundaries(
                geometry,
                self.neural.submissions,
                self.neural.shape.consumers(),
            ) {
                return Err(identity());
            }
            let residency = self.residency.ok_or_else(unknown)?;
            if !recipe.matches_host_transfer_population(
                residency.forwards,
                residency.transfers,
                residency.observations,
                residency.unprepared.transfer.output_arrays,
                storage::TRANSFERS_PER_NONEMPTY_WINDOW,
                self.window_sources
                    .as_deref()
                    .ok_or_else(unknown)?
                    .as_slice(),
            ) || (self.retained_sources.is_some() && !recipe.has_host_copy_recipe())
            {
                return Err(identity());
            }
            match self.foreground_disk_descriptors() {
                Some(source) => {
                    if !recipe.matches_foreground_disk_copies(
                        source,
                        self.foreground_disk_forward_population()
                            .ok_or_else(unknown)?,
                        self.foreground_disk_population().ok_or_else(unknown)?,
                    ) {
                        return Err(identity());
                    }
                }
                None if recipe.has_foreground_disk_copy_recipe() => return Err(identity()),
                None => {}
            }
            self.neural_fit = NeuralProducerFit::Recipe(self.neural_fit.requirement());
        }
        Ok(self)
    }
}

/// Exact entry-point populations, not inferred Record/Graph traversal sizes.
/// Enclosing registered roles already own their Record/Graph arena quotas.
/// Producer-fit composition must validate the one-output evaluation frontier,
/// Record-backed traversal and each dependency wait against those same arenas,
/// without a heap fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct NeuralNativeRequirements {
    pub(crate) submissions: usize,
    pub(crate) root_handles: usize,
    pub(crate) consumer_waits: usize,
    pub(crate) recovery_observers: usize,
}
impl NeuralNativeRequirements {
    fn wait_record_layout(self) -> Option<safemlx::OperationWaitRecordLayout> {
        safemlx::OperationEvent::wait_record_layout(self.consumer_waits)
    }

    /// Measured fixed event/observer wrappers. Each actual C array clone handle
    /// and its final slot are now included in PreparedNeuralSubmission storage.
    /// OperationEvent includes C controls already resident in the enclosing
    /// Graph arena. This report is therefore not added wholesale to Rust bank
    /// bytes: fit composition separates its arena and outside-arena members.
    /// It is not an additional arena charge or a producer-fit certificate.
    pub(crate) fn fixed_control_bytes(self) -> Option<u64> {
        let events = self
            .submissions
            .checked_mul(safemlx::OperationEvent::control_bytes()?)?;
        let observers = self
            .recovery_observers
            .checked_mul(OriginalScopeObserver::control_bytes()?)?;
        u64::try_from(events.checked_add(observers)?).ok()
    }
}

pub(super) fn factory<C: Clone + 'static, P: Observer>(
    shape: NeuralSubmissionShape,
    controls: Option<C>,
) -> impl FnMut(usize) -> Result<PreparedNeuralSubmission<C, P>, SubmissionPreparationError<C, P>> {
    move |_| {
        PreparedNeuralSubmission::<C, P>::try_new(
            shape,
            controls
                .as_ref()
                .expect("accepted original neural custody")
                .clone(),
        )
    }
}
fn measured_layout<F>(population: NeuralPopulation, _factory: &F) -> Option<u64>
where
    F: FnMut(
        usize,
    ) -> Result<
        PreparedNeuralSubmission,
        SubmissionPreparationError<OriginalOperationMetadataCustody, OriginalScopeObserver>,
    >,
{
    let bank = PreparedOperationBank::<PreparedNeuralSubmission>::layout::<
        F,
        SubmissionPreparationError<OriginalOperationMetadataCustody, OriginalScopeObserver>,
    >(
        population.submissions,
        PreparedSlot::control_bytes(population.shape)?,
    )?;
    let controls = [
        size_of::<NeuralPopulation>(),
        size_of::<NeuralNativeRequirements>(),
        size_of::<NeuralProducerFit>(),
        size_of::<PendingNeuralFit>(),
        size_of::<[NeuralFitContribution; 3]>(),
        size_of::<OrderedNeuralCompletion>(),
        size_of::<Result<OrderedNeuralCompletion, Error>>(),
        // Shared traversal forwards the source through these actual selected
        // runtime error enums; their Submission variants own BackendFailure.
        size_of::<eredu_runtime::LayerwiseRuntimeError<eredu_nn::Error, Error>>(),
        size_of::<eredu_runtime::ReplicatedTextSessionError<eredu_nn::Error, Error, Error>>(),
        size_of::<Result<bool, Error>>(),
        size_of::<Result<(), Error>>(),
        size_of::<(PreparedNeuralSubmission, OriginalScopeObserver)>(),
        size_of::<Result<(PreparedNeuralSubmission, OriginalScopeObserver), Error>>(),
        size_of::<crate::backend::nn::shared::OriginalSubmissionFailure<OriginalOperationMetadataCustody>>(),
        size_of::<
            Result<
                OriginalNeuralSubmissionCompletion<OriginalOperationMetadataCustody>,
                crate::backend::nn::shared::OriginalSubmissionFailure<OriginalOperationMetadataCustody>,
            >,
        >(),
        size_of::<[&'static MlxTensor; 1]>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)?;
    bank.total_control_bytes
        .checked_add(u64::try_from(controls).ok()?)?
        .checked_add(clone_error_control_bytes()?)?
        .checked_add(
            u64::try_from(
                eredu_core::BackendFailure::source_retention_peak_bytes::<NeuralBoundaryFailure>()?
                    .checked_add(eredu_core::BackendFailure::source_retention_peak_bytes::<
                        Error,
                    >()?)?
                    .checked_add(eredu_core::BackendFailure::source_retention_peak_bytes::<
                        safemlx::error::Exception,
                    >()?)?,
            )
            .ok()?,
        )
}

pub(super) fn prepare(
    population: NeuralPopulation,
    controls: &OriginalTextControlGuard,
) -> Result<PreparedOperationBank<PreparedNeuralSubmission>, Error> {
    prepare_with_custody(population, controls.metadata_custody().into())
}
/// One actual slot factory/error worker for text and independent roles. Its
/// caller has already authenticated and consumed the corresponding finite bank.
pub(super) fn prepare_with_custody(
    population: NeuralPopulation,
    controls: OriginalOperationMetadataCustody,
) -> Result<PreparedOperationBank<PreparedNeuralSubmission>, Error> {
    PreparedOperationBank::try_new(
        population.submissions,
        factory::<OriginalOperationMetadataCustody, OriginalScopeObserver>(
            population.shape,
            Some(controls.clone()),
        ),
    )
    .map_err(|error| preparation_error(error, &controls))
}
// The closed source removes its Box before dropping this value. The actual
// Exception (including owned text/source children) then drops before custody.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct NeuralClonePreparationFailure {
    #[source]
    cause: safemlx::error::Exception,
    _controls: OriginalOperationMetadataCustody,
}
fn clone_error(cause: safemlx::error::Exception, controls: &OriginalOperationMetadataCustody) -> Error {
    Error::with_original_control_source(
        eredu_core::BackendFailure::from_error(NeuralClonePreparationFailure {
            cause,
            _controls: controls.clone(),
        }),
        false,
    )
}
pub(super) fn clone_error_control_bytes() -> Option<u64> {
    // Nested native error text/source allocations and allocator infrastructure
    // remain pending; this is the concrete closed wrapper/retirement layout.
    u64::try_from(eredu_core::BackendFailure::source_retention_peak_bytes::<
        NeuralClonePreparationFailure,
    >()?.checked_add(eredu_core::BackendFailure::source_retention_peak_bytes::<
        PreparationFailure<OriginalOperationMetadataCustody>,
    >()?)?
        .checked_add(size_of::<OriginalOperationMetadataCustody>())?
        .checked_add(size_of::<eredu_runtime::working_memory::OriginalTextMetadataCustody>())?
        .checked_add(size_of::<eredu_runtime::working_memory::OriginalSpeculativeBudgetCustody>())?
        .checked_add(size_of::<Result<PreparedOperationBank<PreparedNeuralSubmission>, Error>>())?)
    .ok()
}

fn preparation_error<F>(
    error: BankPreparationError<
        PreparedNeuralSubmission,
        F,
        SubmissionPreparationError<OriginalOperationMetadataCustody, OriginalScopeObserver>,
    >,
    controls: &OriginalOperationMetadataCustody,
) -> Error {
    let (cause, prefix, factory) = error.into_parts();
    let error = match cause {
        BankPreparationCause::Overflow => overflow(),
        BankPreparationCause::Reserve(cause) => reserve_error(cause, controls),
        BankPreparationCause::Slot { cause, .. } => {
            let SubmissionPreparationError {
                cause,
                pending,
                controls,
            } = cause;
            let error = match cause {
                SubmissionPreparationCause::Overflow => overflow(),
                SubmissionPreparationCause::Reserve { cause, .. } => {
                    reserve_error(cause, &controls)
                }
                SubmissionPreparationCause::NativeClone { cause, .. } => {
                    clone_error(cause, &controls)
                }
            };
            // Keep the real cause/guard owner before releasing this never-started
            // partial slot. Its deferred payload retains its own independent Q.
            drop(pending);
            drop(controls);
            error
        }
    };
    drop(prefix);
    drop(factory);
    error
}

// The neutral scheduler's closed error erasure retains this exact account.
// Failed completion owners stay on their existing safe retirement path.
#[derive(thiserror::Error)]
#[error("{cause}")]
pub(super) struct NeuralBoundaryFailure<C = OriginalTextControlGuard> {
    #[source]
    cause: Error,
    _controls: C,
}
impl<C> std::fmt::Debug for NeuralBoundaryFailure<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NeuralBoundaryFailure")
            .field("cause", &self.cause)
            .finish_non_exhaustive()
    }
}
pub(super) fn boundary_error<C: Clone + Send + Sync + 'static>(
    cause: Error,
    controls: &C,
) -> Error {
    Error::with_original_control_source(
        eredu_core::BackendFailure::from_error(NeuralBoundaryFailure {
            cause,
            _controls: controls.clone(),
        }),
        false,
    )
}
pub(super) fn submit_prepared(
    prepared: PreparedSlot,
    observer: OriginalScopeObserver,
    value: &MlxTensor,
    stream: &Stream,
    controls: &OperationControls,
) -> Result<OrderedNeuralCompletion, Error> {
    prepared
        .submit_nested(value, observer, stream)
        .map(|completion| match controls {
            OperationControls::Text(controls) => OrderedNeuralCompletion::Original { completion, controls: controls.clone() },
            OperationControls::Speculative(controls) => OrderedNeuralCompletion::Speculative { completion, controls: controls.clone() },
            OperationControls::Realtime(controls) => OrderedNeuralCompletion::Realtime { completion, controls: controls.clone() },
        })
        .map_err(|failure| Error::from(failure.into_cause()))
}

/// Private policy completion, not the unrestricted public SubmissionBackend
/// retention API. Both variants remain their original concrete owners.
pub(in crate::backend::runtime::execution::generic) enum OrderedNeuralCompletion {
    Ordinary(MlxSubmissionCompletion),
    Original {
        completion: OriginalNeuralSubmissionCompletion<OriginalOperationMetadataCustody>,
        controls: OriginalTextControlGuard,
    },
    Speculative {
        completion: OriginalNeuralSubmissionCompletion<OriginalOperationMetadataCustody>,
        controls: SpeculativeOperationRole,
    },
    Realtime {
        completion: OriginalNeuralSubmissionCompletion<OriginalOperationMetadataCustody>,
        controls: eredu_runtime::working_memory::OriginalRealtimeBudgetCustody,
    },
}
impl Completion for OrderedNeuralCompletion {
    type Error = Error;
    fn is_complete(&self) -> Result<bool, Error> {
        match self {
            Self::Ordinary(value) => value.is_complete().map_err(Error::from),
            Self::Original {
                completion,
                controls,
            } => completion
                .is_complete()
                .map_err(|e| boundary_error(e.into(), controls)),
            Self::Speculative {
                completion,
                controls,
            } => completion
                .is_complete()
                .map_err(|e| boundary_error(e.into(), controls)),
            Self::Realtime {completion,controls}=>completion.is_complete()
                .map_err(|e|boundary_error(e.into(),controls)),
        }
    }
    fn wait(&self) -> Result<(), Error> {
        match self {
            Self::Ordinary(value) => value.wait().map_err(Error::from),
            Self::Original {
                completion,
                controls,
            } => completion
                .wait()
                .map_err(|e| boundary_error(e.into(), controls)),
            Self::Speculative {
                completion,
                controls,
            } => completion
                .wait()
                .map_err(|e| boundary_error(e.into(), controls)),
            Self::Realtime {completion,controls}=>completion.wait()
                .map_err(|e|boundary_error(e.into(),controls)),
        }
    }
    fn resources_releasable(&self) -> bool {
        match self {
            Self::Ordinary(value) => value.resources_releasable(),
            Self::Original { completion, .. } => completion.resources_releasable(),
            Self::Speculative { completion, .. } => completion.resources_releasable(),
            Self::Realtime { completion, .. } => completion.resources_releasable(),
        }
    }
}
impl OrderedLayerwiseCompletion<Stream> for OrderedNeuralCompletion {
    fn order_after(&self, stream: &Stream) -> Result<(), Error> {
        match self {
            Self::Ordinary(value) => {
                MlxNeuralBackend::order_after(value, stream).map_err(Error::from)
            }
            Self::Original {
                completion,
                controls,
            } => completion
                .order_after(stream)
                .map_err(|e| boundary_error(e.into(), controls)),
            Self::Speculative {
                completion,
                controls,
            } => completion
                .order_after(stream)
                .map_err(|e| boundary_error(e.into(), controls)),
            Self::Realtime {completion,controls}=>completion.order_after(stream)
                .map_err(|e|boundary_error(e.into(),controls)),
        }
    }
    fn finish(self) -> Result<(), Error> {
        match self {
            Self::Ordinary(value) => value.wait().map_err(Error::from),
            Self::Original {
                completion,
                controls,
            } => completion
                .finish()
                .map_err(|e| boundary_error(e, &controls)),
            Self::Speculative {
                completion,
                controls,
            } => completion
                .finish()
                .map_err(|e| boundary_error(e, &controls)),
            Self::Realtime {completion,controls}=>completion.finish()
                .map_err(|e|boundary_error(e,&controls)),
        }
    }
}
impl<U: 'static, P> MlxLayerwisePolicy<U, P> {
    pub(in crate::backend::runtime::execution::generic) fn uses_shared_neural_executor(
        &self,
        stream: &Stream,
    ) -> Result<bool, Error> {
        let Some(original) = self.original_operation_projection() else {
            return Ok(false);
        };
        let controls = original.controls.clone();
        let result = (|| {
            let access = original.access()?;
            access.bank.registry.authenticate()?;
            if !access.bank.selected_stream.matches_source(stream) {
                return Err(identity());
            }
            Ok(true)
        })();
        result.map_err(|cause| boundary_error(cause, &controls))
    }
    pub(crate) fn selected_shared_neural_executor(
        selected: Result<&Self, &MlxResidentPolicy<U>>,
        stream: &Stream,
    ) -> Result<bool, Error> {
        match selected {
            Ok(policy) => policy.uses_shared_neural_executor(stream),
            Err(policy) => policy.original_neural.uses_shared_executor(stream),
        }
    }

    /// The composition wrapper borrows its actual selected policy. Either
    /// mechanism consumes only its installed request bank; without an active
    /// original installation it preserves ordinary submission.
    pub(crate) fn submit_selected_neural(
        selected: Result<&Self, &MlxResidentPolicy<U>>,
        stream: &Stream,
        value: &MlxTensor,
    ) -> Result<impl OrderedLayerwiseCompletion<Stream> + 'static, Error> {
        match selected {
            Ok(policy) => policy.submit_neural(stream, value),
            Err(policy) => policy.original_neural.submit(stream, value),
        }
    }

    pub(in crate::backend::runtime::execution::generic) fn submit_neural(
        &self,
        stream: &Stream,
        value: &MlxTensor,
    ) -> Result<OrderedNeuralCompletion, Error> {
        match self.original_operation_projection() {
            Some(original) => {
                // No storage mode is inferred from TLS. A stale/fenced/mismatched
                // installed request errors before checkout and never falls back.
                let controls = original.controls.clone();
                let result = (|| {
                    let access = original.access()?;
                    if !access.bank.selected_stream.matches_source(stream) {
                        return Err(identity());
                    }
                    let (prepared, observer) = access.checkout_neural()?;
                    submit_prepared(prepared, observer, value, stream, &controls)
                })();
                result.map_err(|cause| boundary_error(cause, &controls))
            }
            None => MlxNeuralBackend::submit(stream, [value])
                .map(OrderedNeuralCompletion::Ordinary)
                .map_err(Error::from),
        }
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
impl OriginalOperationPlan<'_, ()> {
    pub(crate) fn exercise_neural_clone_preparation_failure(
        controls: &OriginalTextControlGuard,
        cause: safemlx::error::Exception,
    ) -> Error {
        // Exercise the actual bank-error conversion with one completed cold
        // slot and one owned pending slot. This injects the transport variant;
        // it does not simulate or claim native allocator exhaustion.
        let shape = NeuralSubmissionShape::new(1, 0).unwrap();
        let mut cause = Some(cause);
        let controls: OriginalOperationMetadataCustody = controls.metadata_custody().into();
        let error = PreparedOperationBank::try_new(2, |ordinal| {
            let pending = PreparedSlot::try_new(shape, controls.clone()).unwrap();
            if ordinal == 0 {
                Ok(pending)
            } else {
                Err(SubmissionPreparationError {
                    cause: SubmissionPreparationCause::NativeClone {
                        index: 0,
                        cause: cause.take().unwrap(),
                    },
                    pending: Some(pending),
                    controls: controls.clone(),
                })
            }
        })
        .unwrap_err();
        assert_eq!(error.prepared_len(), 1);
        preparation_error(error, &controls)
    }
}
