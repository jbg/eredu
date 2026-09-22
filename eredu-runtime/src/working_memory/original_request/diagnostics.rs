//! Closed post-comparison construction. No producer callback can add a buffer.
use super::*;
use eredu_core::{
    AdmissionPolicyDecision, AdmissionStateRequirements, ExecutionWorkspaceEstimate,
    ExecutionWorkspaceRequirements, RuntimeStateEstimate, SelectedStateBacking,
    SelectedStateRequirements, StateMemoryAssumptions, StateWindowPlan, WorkspaceBound,
};
use std::{alloc::Layout, collections::TryReserveError, fmt, mem::size_of, num::NonZeroU8};

/// Scalar and borrowed descriptive projection of an already complete producer.
/// Construction of this view does not create a complete producer or permission.
#[derive(Clone, Copy)]
pub(super) struct StateReport<'a> {
    pub report_recipe: Option<report_workspace::ReportRecipe>,
    pub geometry: InferenceGeometry,
    pub fixed_state_bytes: u64,
    pub bytes_per_position_per_batch: u64,
    pub context_state_bytes: u64,
    pub selected_state_bytes: u64,
    pub multimodal_embedding_bytes: u64,
    pub media_execution_workspace_bytes: u64,
    pub components: [u64; 6],
    pub dtype_bytes: NonZeroU8,
    pub allocation_granularity: u64,
    pub sliding_windows: ReportWindows<'a>,
    // Selected backing followed by the existing six ordered workspace terms.
    pub descriptions: [&'a str; 7],
}
// Closed construction variants: the selected schedule proves its own ordering;
// the existing complete recurrent fixture supplies its actual explicit windows.
// Neither variant certifies selected backing or execution completeness.
#[derive(Clone, Copy)]
pub(super) enum ReportWindows<'a> {
    Explicit(&'a [u64]),
    State(StateWindowPlan<'a>),
}
impl ReportWindows<'_> {
    fn control_bytes(self) -> Result<usize, WorkingMemoryError> {
        let iterator = match self {
            Self::Explicit(values) => std::mem::size_of_val(&values.iter()),
            Self::State(plan) => std::mem::size_of_val(&plan.iter()),
        };
        [
            iterator,
            size_of::<Option<u64>>(),
            size_of::<Result<(), eredu_core::StateWindowDestinationError>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or(WorkingMemoryError::Overflow)
    }
    fn len(self) -> usize {
        match self {
            Self::Explicit(values) => values.len(),
            Self::State(plan) => plan.len(),
        }
    }
    fn validate(self) -> Result<(), WorkingMemoryError> {
        if let Self::Explicit(values) = self {
            if values.windows(2).any(|v| v[0] >= v[1]) || values.first() == Some(&0) {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
        }
        Ok(())
    }
    fn fill(self, destination: &mut [u64]) -> Result<(), WorkingMemoryError> {
        self.validate()?;
        if destination.len() != self.len() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        match self {
            Self::Explicit(values) => destination.copy_from_slice(values),
            Self::State(plan) => plan
                .fill(destination)
                .map_err(|_| WorkingMemoryError::IdentityMismatch)?,
        }
        Ok(())
    }
}
impl StateReport<'_> {
    pub(super) fn requirements(&self) -> Result<AdmissionStateRequirements, WorkingMemoryError> {
        let g = self.geometry;
        g.validate_fixed()
            .map_err(|_| WorkingMemoryError::IdentityMismatch)?;
        let requested_positions = g
            .cached_positions
            .checked_add(g.input_positions)
            .and_then(|n| n.checked_add(g.max_output_tokens))
            .ok_or(WorkingMemoryError::Overflow)?;
        let logical = self
            .fixed_state_bytes
            .checked_add(self.context_state_bytes)
            .ok_or(WorkingMemoryError::Overflow)?;
        let requested_state_bytes = logical
            .max(self.selected_state_bytes)
            .checked_add(self.multimodal_embedding_bytes)
            .and_then(|n| n.checked_add(self.media_execution_workspace_bytes))
            .ok_or(WorkingMemoryError::Overflow)?;
        self.sliding_windows.validate()?;
        Ok(AdmissionStateRequirements {
            physical_domains: true,
            requested_state_bytes,
            batch_size: g.batch_size,
            requested_positions,
            persistent_state_completeness: EstimationCompleteness::Complete,
            completeness: EstimationCompleteness::Complete,
            selected_state_backing: Some(SelectedStateRequirements {
                geometry: g,
                bytes: Some(self.selected_state_bytes),
            }),
            execution_workspace: Some(ExecutionWorkspaceRequirements {
                geometry: g,
                components: self.components.map(Some),
            }),
        })
    }
    fn heap_bytes(&self) -> Result<usize, WorkingMemoryError> {
        self.descriptions.iter().try_fold(
            Layout::array::<u64>(self.sliding_windows.len())
                .map_err(|_| WorkingMemoryError::Overflow)?
                .size(),
            |sum, value| {
                // Layout rejects the same impossible isize-sized allocation
                // before Q; no destination exists yet.
                let bytes = Layout::array::<u8>(value.len())
                    .map_err(|_| WorkingMemoryError::Overflow)?
                    .size();
                sum.checked_add(bytes).ok_or(WorkingMemoryError::Overflow)
            },
        )
    }
    /// Concrete common constructor/control floor. A producer adds its actual
    /// numerical, output, source-wrapper and backend controls independently.
    pub(super) fn control_bytes<S: Send + Sync + 'static>(
        &self,
    ) -> Result<u64, WorkingMemoryError> {
        control_mutex::require_known_layout()?;
        fn arc<T>() -> Result<usize, WorkingMemoryError> {
            Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
                .extend(Layout::new::<T>())
                .map(|(layout, _)| layout.pad_to_align().size())
                .map_err(|_| WorkingMemoryError::Overflow)
        }
        let terms = [
            self.heap_bytes()?,
            self.report_recipe
                .map(report_workspace::ReportRecipe::bytes)
                .transpose()?
                .unwrap_or(0),
            self.sliding_windows.control_bytes()?,
            size_of::<StateReport<'_>>(),
            size_of::<Failure>(),
            size_of::<CandidateFailure<Failure>>(),
            size_of::<Result<(Sources, WorkingMemoryReservation), Failure>>(),
            size_of::<eredu_core::RuntimeStateFacts<'_>>(),
            size_of::<eredu_core::StateWindowDestinationError>(),
            size_of::<Result<eredu_core::RuntimeStateFacts<'_>, AdmissionPolicyError>>(),
            size_of::<funding::AccountNode>(),
            arc::<Reservation>()?,
            arc::<WorkingMemoryReservation>()?,
            arc::<text_preparation::TextPreparationAuthority>()?,
            text_preparation::synchronization_control_bytes()
                .ok_or(WorkingMemoryError::UnknownBound)?,
            size_of::<funding::PendingAccount>(),
            size_of::<funding::AccountTicket>(),
            size_of::<Construction<S>>(),
            size_of::<ConstructionFailure<S>>(),
            size_of::<
                Result<
                    (
                        S,
                        WorkingMemoryReservation,
                        Option<report_workspace::Storage>,
                    ),
                    ConstructionFailure<S>,
                >,
            >(),
            size_of::<Reservation>(),
            size_of::<ReservationOwner>(),
            size_of::<WorkingMemoryReservation>(),
            size_of::<RequestOwner>(),
            size_of::<InferenceRequest>(),
            size_of::<InferenceTextPreparation>(),
            size_of::<InferenceTextStep>(),
            size_of::<InferenceTextStepReceipt>(),
            size_of::<Result<InferenceTextPreparation, WorkingMemoryError>>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            eredu_core::BackendFailure::source_retention_peak_bytes::<ConstructionFailure<S>>()
                .ok_or(WorkingMemoryError::Overflow)?,
        ];
        terms
            .into_iter()
            .try_fold(0usize, usize::checked_add)
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(WorkingMemoryError::Overflow)
    }
}

#[derive(Debug, Default)]
struct Storage {
    report: Option<report_workspace::Storage>,
    descriptions: [String; 7],
    sliding_windows: Vec<u64>,
}
#[derive(Debug)]
enum Cause {
    Report(report_workspace::ReportFailure),
    Accounting(WorkingMemoryError),
    Reserve {
        destination: usize,
        cause: TryReserveError,
    },
}
impl fmt::Display for Cause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Accounting(error) => error.fmt(f),
            Self::Report(error) => fmt::Display::fmt(error, f),
            Self::Reserve { destination, cause } => {
                write!(f, "original request diagnostic {destination}: {cause}")
            }
        }
    }
}
impl std::error::Error for Cause {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match self {
            Self::Accounting(error) => error,
            Self::Report(error) => error,
            Self::Reserve { cause, .. } => cause,
        })
    }
}
// Field order is custody: real buffers, source alias, then same original account.
struct Construction<S> {
    storage: Storage,
    source: S,
    ticket: funding::AccountTicket,
}
pub(super) struct ConstructionFailure<S> {
    cause: Cause,
    construction: Construction<S>,
}
impl<S> fmt::Debug for ConstructionFailure<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OriginalRequestConstructionFailure")
            .field("cause", &self.cause)
            .field("account", &self.construction.ticket.id())
            .finish_non_exhaustive()
    }
}
impl<S> fmt::Display for ConstructionFailure<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl<S: 'static> std::error::Error for ConstructionFailure<S> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
impl Storage {
    fn fill(&mut self, report: StateReport<'_>, fail: Option<usize>) -> Result<(), Cause> {
        for (i, (destination, source)) in self
            .descriptions
            .iter_mut()
            .zip(report.descriptions)
            .enumerate()
        {
            let requested = if fail == Some(i) {
                usize::MAX
            } else {
                source.len()
            };
            destination
                .try_reserve_exact(requested)
                .map_err(|cause| Cause::Reserve {
                    destination: i,
                    cause,
                })?;
            destination.push_str(source);
        }
        let requested = if fail == Some(7) {
            usize::MAX
        } else {
            report.sliding_windows.len()
        };
        self.sliding_windows
            .try_reserve_exact(requested)
            .map_err(|cause| Cause::Reserve {
                destination: 7,
                cause,
            })?;
        self.sliding_windows.resize(report.sliding_windows.len(), 0);
        report
            .sliding_windows
            .fill(&mut self.sliding_windows)
            .map_err(Cause::Accounting)?;
        Ok(())
    }
    fn into_state(
        self,
        report: StateReport<'_>,
        requirements: AdmissionStateRequirements,
    ) -> RuntimeStateEstimate {
        let [
            selected,
            activations,
            attention,
            vocabulary,
            state_update,
            materialization,
            retained,
        ] = self.descriptions;
        let [a, b, c, d, e, f] = report.components;
        RuntimeStateEstimate {
            physical_domains: None,
            fixed_state_bytes: report.fixed_state_bytes,
            bytes_per_position_per_batch: report.bytes_per_position_per_batch,
            context_state_bytes: report.context_state_bytes,
            selected_state_backing: Some(SelectedStateBacking {
                geometry: report.geometry,
                bound: WorkspaceBound::bounded(report.selected_state_bytes, selected),
            }),
            multimodal_embedding_bytes: report.multimodal_embedding_bytes,
            media_execution_workspace_bytes: report.media_execution_workspace_bytes,
            requested_state_bytes: requirements.requested_state_bytes,
            execution_workspace: Some(ExecutionWorkspaceEstimate {
                physical_domains: None,
                geometry: report.geometry,
                activations: WorkspaceBound::bounded(a, activations),
                attention: WorkspaceBound::bounded(b, attention),
                vocabulary: WorkspaceBound::bounded(c, vocabulary),
                state_update: WorkspaceBound::bounded(d, state_update),
                materialization: WorkspaceBound::bounded(e, materialization),
                retained: WorkspaceBound::bounded(f, retained),
            }),
            persistent_state_completeness: requirements.persistent_state_completeness,
            assumptions: StateMemoryAssumptions {
                floating_state_dtype_bytes: report.dtype_bytes,
                batch_size: requirements.batch_size,
                requested_positions: requirements.requested_positions,
                sliding_window_bounds: self.sliding_windows,
                allocation_granularity: report.allocation_granularity,
            },
            completeness: requirements.completeness,
        }
    }
}

/// Called only with the accepted decision and its same published account.
/// Source validation and complete recipe population precede this operation.
pub(super) fn construct<S>(
    ticket: funding::AccountTicket,
    source: S,
    execution: &InferenceExecutionIdentity,
    report: StateReport<'_>,
    requirements: AdmissionStateRequirements,
    decision: AdmissionPolicyDecision,
    capacity: Option<eredu_core::MemoryLimits>,
) -> Result<
    (
        S,
        WorkingMemoryReservation,
        Option<report_workspace::Storage>,
    ),
    ConstructionFailure<S>,
> {
    #[cfg(test)]
    let fail = FAIL_DESTINATION.with(|value| value.take());
    #[cfg(not(test))]
    let fail = None;
    construct_inner(
        ticket,
        source,
        execution,
        report,
        requirements,
        decision,
        capacity,
        fail,
    )
}
fn construct_inner<S>(
    ticket: funding::AccountTicket,
    source: S,
    execution: &InferenceExecutionIdentity,
    report: StateReport<'_>,
    requirements: AdmissionStateRequirements,
    decision: AdmissionPolicyDecision,
    capacity: Option<eredu_core::MemoryLimits>,
    fail: Option<usize>,
) -> Result<
    (
        S,
        WorkingMemoryReservation,
        Option<report_workspace::Storage>,
    ),
    ConstructionFailure<S>,
> {
    let mut construction = Construction {
        storage: Storage::default(),
        source,
        ticket,
    };
    if let Err(error) = construction.ticket.status() {
        return Err(ConstructionFailure {
            cause: Cause::Accounting(error),
            construction,
        });
    }
    if let Err(cause) = construction.storage.fill(report, fail) {
        return Err(ConstructionFailure {
            cause,
            construction,
        });
    }
    if let Some(recipe) = report.report_recipe {
        let storage = construction
            .storage
            .report
            .insert(report_workspace::Storage::default());
        if let Err(error) = storage.construct(recipe, fail) {
            return Err(ConstructionFailure {
                cause: Cause::Report(error),
                construction,
            });
        }
    }
    #[cfg(test)]
    after_fill_for_test(construction.ticket.pool());
    // Publication success is not a permanent lock-health certificate. Retain
    // the complete filled storage and its exact ticket if settlement is poisoned.
    if let Err(error) = construction.ticket.status() {
        return Err(ConstructionFailure {
            cause: Cause::Accounting(error),
            construction,
        });
    }
    let Construction {
        mut storage,
        source,
        ticket,
    } = construction;
    let report_storage = storage.report.take();
    let admission = crate::working_memory::memory_fixture::attribute_host_admission(
        ticket.pool(),
        Admission {
            memory_limits: Default::default(),
            additional_headroom: Default::default(),
            requested_positions: decision.requested_positions,
            state: storage.into_state(report, requirements),
            incremental_required_bytes: decision.incremental_required_bytes,
        },
    );
    let value = Reservation {
        account_id: ticket.id(),
        pool: ticket.pool().clone(),
        execution: execution.clone(),
        admission,
        geometry: report.geometry,
        requirements: crate::working_memory::memory_fixture::host_requirements(
            ticket.pool(),
            decision
                .incremental_required_bytes
                .expect("host fixture requirement"),
        ),
        capacity,
        funding: None,
        borrowed_storage: None,
        span_workspace: None,
        start: ControlMutex::new(text_preparation::RequestStart::Fresh),
        planning_metadata: None,
    };
    Ok((source, ticket.into_reservation(value), report_storage))
}

#[cfg(test)]
thread_local! {
    static FAIL_DESTINATION: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}
#[cfg(test)]
pub(super) fn fail_destination_for_test(at: usize) {
    assert!(at < 11);
    FAIL_DESTINATION.with(|value| {
        assert!(value.replace(Some(at)).is_none());
    });
}
#[cfg(test)]
impl<S> ConstructionFailure<S> {
    pub(super) fn destination(&self) -> Option<usize> {
        match self.cause {
            Cause::Reserve { destination, .. } => Some(destination),
            Cause::Report(report_workspace::ReportFailure::Reserve { destination, .. }) => {
                Some(destination)
            }
            _ => None,
        }
    }
    pub(super) fn prefix_bytes(&self) -> usize {
        self.construction
            .storage
            .descriptions
            .iter()
            .map(String::capacity)
            .sum::<usize>()
            + self.construction.storage.sliding_windows.capacity() * size_of::<u64>()
            + self
                .construction
                .storage
                .report
                .as_ref()
                .map_or(0, report_workspace::Storage::retained_bytes)
            + match &self.cause {
                Cause::Report(report_workspace::ReportFailure::Workspace(error)) => {
                    error.retained_heap_bytes()
                }
                _ => 0,
            }
    }
}

#[cfg(test)]
#[derive(Clone, Copy)]
pub(super) enum AfterFill {
    Poison,
    Unwind,
}
#[cfg(test)]
thread_local! { static AFTER_FILL: std::cell::Cell<Option<AfterFill>> = const { std::cell::Cell::new(None) }; }
#[cfg(test)]
pub(super) fn set_after_fill_for_test(action: AfterFill) {
    AFTER_FILL.with(|value| assert!(value.replace(Some(action)).is_none()));
}
#[cfg(test)]
fn after_fill_for_test(pool: &MemoryLedger) {
    match AFTER_FILL.with(|value| value.take()) {
        Some(AfterFill::Unwind) => panic!("diagnostic filled before reservation publication"),
        Some(AfterFill::Poison) => {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _usage = pool.0.usage.lock().unwrap();
                panic!("diagnostic settlement poison");
            }));
            assert!(result.is_err());
        }
        None => {}
    }
}
