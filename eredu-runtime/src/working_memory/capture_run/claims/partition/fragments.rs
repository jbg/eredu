//! Finite original-account host destinations for actual receipt fragments.
use super::*;
use crate::capture::partition::{
    PartitionCaptureFragmentAllowanceError, PartitionCaptureNativeEstimate,
    PartitionCaptureReceiptPlan, PreparedPartitionFragmentAllowance, PreparedPartitionFragmentLoan,
};
use eredu_core::capture::PartitionCaptureCombination;

#[derive(Debug)]
pub(in crate::working_memory) enum FragmentHostPlan<'a> {
    Routed(CapturePartitionRoutedHostPlan<'a>),
    Tensor(CaptureTensorHostPlan<'a>),
    Summary(CaptureSummaryHostPlan<'a>),
    Histogram(CaptureHistogramHostPlan<'a>),
}
impl<'a> FragmentHostPlan<'a> {
    fn prepare(
        receipt: &'a PartitionCaptureReceiptPlan,
        producer: usize,
        fragment: usize,
    ) -> Result<Self, CaptureRunHostError> {
        if receipt.routed_producer(producer).is_some() {
            return CapturePartitionRoutedHostPlan::prepare(receipt, producer, fragment)
                .map(Self::Routed);
        }
        use crate::capture::partition::{
            PartitionInvocationCaptureGeometry, PartitionInvocationCaptureKind,
        };
        let source = PartitionInvocationCaptureGeometry::from_receipt(receipt, producer, fragment)
            .map_err(|_| CaptureRunHostError::ReceiptMismatch)?;
        Ok(match source.into_kind() {
            PartitionInvocationCaptureKind::Tensor(geometry) => {
                Self::Tensor(CaptureTensorHostPlan::prepare(geometry)?)
            }
            PartitionInvocationCaptureKind::Summary(geometry) => {
                Self::Summary(CaptureSummaryHostPlan::prepare(geometry)?)
            }
            PartitionInvocationCaptureKind::Histogram(geometry) => {
                Self::Histogram(CaptureHistogramHostPlan::prepare(geometry)?)
            }
        })
    }

    pub(in crate::working_memory) fn initialization_peak_bytes(
        &self,
    ) -> Result<u64, WorkingMemoryError> {
        let payload = match self {
            Self::Routed(p) => p.initialization_peak_bytes(),
            Self::Tensor(p) => p.initialization_peak_bytes(),
            Self::Summary(p) => p.initialization_peak_bytes(),
            Self::Histogram(p) => p.initialization_peak_bytes(),
        };
        payload
            .checked_add(CaptureTensorCustody::shared_host_control_bytes()?)
            .ok_or(WorkingMemoryError::Overflow)
    }
    fn destination<'c>(
        self,
        identity: ReceiptIdentity,
    ) -> Result<PartitionFragmentDestination<'a, 'c>, CaptureRunHostError> {
        Ok(match self {
            Self::Routed(_) => return Err(CaptureRunHostError::ReceiptMismatch),
            Self::Tensor(plan) => PartitionFragmentDestination::Tensor(CaptureTensorClaim {
                plan,
                identity,
                exclusive: PhantomData,
            }),
            Self::Summary(plan) => PartitionFragmentDestination::Summary(
                CaptureSummaryClaim::from_partition_plan(plan, identity),
            ),
            Self::Histogram(plan) => PartitionFragmentDestination::Histogram(
                CaptureHistogramClaim::from_partition_plan(plan, identity),
            ),
        })
    }
}

/// Exact host program for all local/received fragments of one retained receipt.
/// The ordinary frame separately owns the final assembled payload. Native C,
/// metadata/source controls and transport remain separately admitted.
#[derive(Debug)]
pub struct PartitionFragmentHostPlan<'a> {
    receipt: &'a PartitionCaptureReceiptPlan,
    slots: usize,
    merge_units: usize,
    table: u64,
    fragments: u64,
    assembly: Option<(usize, usize, u64)>,
    prefill: Option<eredu_core::InferenceGeometry>,
}
impl<'a> PartitionFragmentHostPlan<'a> {
    /// Price the actual projected destinations without allocating a table or
    /// native object. Empty producers acknowledge without a fabricated payload.
    pub fn prepare(receipt: &'a PartitionCaptureReceiptPlan) -> Result<Self, CaptureRunHostError> {
        if receipt.identity().len() != 64 {
            return Err(CaptureRunHostError::ReceiptMismatch);
        }
        let mut slots = 0usize;
        let mut fragments = 0u64;
        for (rank, projection) in receipt.producers() {
            for fragment in 0..projection.fragments().len() {
                let plan = FragmentHostPlan::prepare(receipt, rank, fragment)?;
                fragments = fragments
                    .checked_add(plan.initialization_peak_bytes()?)
                    .ok_or(WorkingMemoryError::Overflow)?;
                slots = slots.checked_add(1).ok_or(WorkingMemoryError::Overflow)?;
            }
        }
        let assembly = Self::assembly_source(receipt)?;
        let merge_units = if receipt
            .producers()
            .any(|(rank, _)| receipt.routed_producer(rank).is_some())
        {
            usize::try_from(
                receipt
                    .producers()
                    .next()
                    .ok_or(CaptureRunHostError::ReceiptMismatch)?
                    .1
                    .global_slice()
                    .shape[2],
            )
            .map_err(|_| WorkingMemoryError::Overflow)?
        } else {
            0
        };
        let controls = Self::inspection_control_bytes()
            .and_then(|n| n.checked_add(receipt.fragment_host_comparison_control_bytes()?))
            .ok_or(WorkingMemoryError::Overflow)?;
        let table = slots
            .checked_mul(size_of::<Slot>())
            .filter(|n| *n <= isize::MAX as usize)
            .and_then(|n| n.checked_add(merge_units))
            .and_then(|n| n.checked_add(controls))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(WorkingMemoryError::Overflow)?;
        let table = table
            .checked_add(CaptureTensorCustody::shared_host_control_bytes()?)
            .ok_or(WorkingMemoryError::Overflow)?;
        table
            .checked_add(fragments)
            .and_then(|n| n.checked_add(assembly.map_or(0, |(_, _, bytes)| bytes)))
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(Self {
            receipt,
            slots,
            merge_units,
            table,
            fragments,
            assembly,
            prefill: None,
        })
    }
    /// Actual table, named construction/error frames and all surviving fragment P.
    pub fn initialization_peak_bytes(&self) -> u64 {
        self.table + self.fragments + self.assembly_peak_bytes()
    }
    /// A real additional F32 materialization only for additive nonlinear transforms.
    pub fn assembly_peak_bytes(&self) -> u64 {
        self.assembly.map_or(0, |(_, _, bytes)| bytes)
    }
    fn assembly_source(
        receipt: &PartitionCaptureReceiptPlan,
    ) -> Result<Option<(usize, usize, u64)>, CaptureRunHostError> {
        let source = receipt.shared_plan_source();
        if receipt.combination() != PartitionCaptureCombination::SumF64ToF32
            || !matches!(
                source.admission().plan().selections[receipt.context().selection_index].transform,
                CaptureTransform::Summary | CaptureTransform::Histogram { .. }
            )
        {
            return Ok(None);
        }
        for (producer, projection) in receipt.producers() {
            if !projection.fragments().is_empty() {
                let plan = FragmentHostPlan::prepare(receipt, producer, 0)?;
                if !matches!(plan, FragmentHostPlan::Tensor(_)) {
                    return Err(CaptureRunHostError::ReceiptMismatch);
                }
                return Ok(Some((producer, 0, plan.initialization_peak_bytes()?)));
            }
        }
        Ok(None)
    }
    /// Sum of the existing tensor/scalar/bin destination programs.
    pub const fn fragment_peak_bytes(&self) -> u64 {
        self.fragments
    }
    /// Number of real destination constructors, including all receiving ranks.
    pub const fn fragment_count(&self) -> usize {
        self.slots
    }
    pub(in crate::working_memory) const fn table_peak_bytes(&self) -> u64 {
        self.table
    }
    /// Fixed cold/constructor/loan/record frames for the surrounding paid source.
    /// This is not authority to allocate a buffer from a caller-supplied count.
    pub fn inspection_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>() * 2,
            size_of::<FragmentHostPlan<'_>>() * 2,
            size_of::<Result<FragmentHostPlan<'_>, CaptureRunHostError>>(),
            size_of::<PreparedPartitionFragmentDestinations>() * 2,
            size_of::<Slot>() * 2,
            size_of::<Vec<Slot>>(),
            size_of::<PartitionFragmentDestination<'_, '_>>() * 2,
            size_of::<NativePartitionFragmentDestination<'_, '_>>() * 2,
            size_of::<PartitionFragmentValue>() * 2,
            size_of::<Option<PartitionFragmentValue>>(),
            size_of::<CaptureTensorCustody>() * 3,
            size_of::<ReceiptIdentity>() * 2,
            size_of::<PartitionFragmentDestinationError>(),
            size_of::<FragmentCause>(),
            size_of::<
                Result<PreparedPartitionFragmentDestinations, PartitionFragmentDestinationError>,
            >(),
            size_of::<
                Result<
                    NativePartitionFragmentDestination<'_, '_>,
                    PartitionFragmentDestinationError,
                >,
            >(),
            size_of::<Result<(), PartitionFragmentDestinationError>>(),
            size_of::<Option<&mut Slot>>(),
            size_of::<Option<&Slot>>(),
            size_of::<std::slice::Iter<'_, Slot>>(),
            size_of::<std::slice::IterMut<'_, Slot>>(),
            size_of::<(
                &WorkingMemoryFundingRun,
                &WorkingMemoryReservation,
                Self,
                PreparedPartitionFragmentAllowance,
            )>(),
            size_of::<(
                &mut PreparedPartitionFragmentDestinations,
                &PartitionCaptureReceiptPlan,
                usize,
                usize,
                &TensorDtype,
                CaptureUsage,
            )>(),
            size_of::<(usize, usize, usize, u64, bool)>(),
            size_of::<[u8; 64]>(),
            crate::capture::partition::PartitionInvocationCaptureGeometry::control_bytes()?,
            assembly::control_bytes()?,
            size_of::<Vec<bool>>(),
            size_of::<Result<Vec<bool>, WorkingMemoryError>>(),
            usize::try_from(
                crate::working_memory::qualified_storage::vector_control_bytes::<bool>().ok()?,
            )
            .ok()?,
            routed::control_bytes()?,
            super::invocation_assembly_control_bytes()?,
            host_owner::control_bytes()?,
            super::super::prefill::assembly_target_control_bytes()?,
            usize::try_from(
                crate::working_memory::qualified_storage::vector_control_bytes::<Slot>().ok()?,
            )
            .ok()?,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}

/// The same typed destination accepted by ordinary native capture workers.
/// This is host custody only; its independent native loan must also be consumed.
#[derive(Debug)]
pub enum PartitionFragmentDestination<'a, 'c> {
    /// Raw selected term; additive nonlinear captures deliberately use this arm.
    Tensor(CaptureTensorClaim<'a, 'c>),
    /// Disjoint finite statistics from exactly this local fragment.
    Summary(CaptureSummaryClaim<'a, 'c>),
    /// Disjoint fixed-edge bins from exactly this local fragment.
    Histogram(CaptureHistogramClaim<'a, 'c>),
    /// Final global sparse rows; local fragments use their separate typed bank.
    Routed(CaptureRoutedClaim<'a, 'c>),
}
/// One spent local destination paired with its actual per-fragment native credits.
#[must_use = "consume through the existing typed native capture worker"]
#[derive(Debug)]
pub struct NativePartitionFragmentDestination<'a, 'c> {
    destination: PartitionFragmentDestination<'a, 'c>,
    loan: PreparedPartitionFragmentLoan,
}
impl<'a, 'c> NativePartitionFragmentDestination<'a, 'c> {
    /// Move the existing host claim and source-bound quota to the shared worker.
    pub fn into_parts(
        self,
    ) -> (
        PartitionFragmentDestination<'a, 'c>,
        PreparedPartitionFragmentLoan,
    ) {
        (self.destination, self.loan)
    }
}
/// Completed fragment values retain their precise original host account. They
/// cannot be attached directly to the ordinary whole-frame claim.
#[derive(Debug)]
pub enum PartitionFragmentValue {
    /// Completed raw F32 values.
    Tensor(ClaimedCaptureTensor),
    /// Completed finite statistics.
    Summary(ClaimedCaptureSummary),
    /// Completed bin counts.
    Histogram(ClaimedCaptureHistogram),
    /// Complete global sparse rows released only through partition delivery.
    Routed(ClaimedAssembledRoutedUnits),
}
impl PartitionFragmentValue {
    fn identity(&self) -> &ReceiptIdentity {
        match self {
            Self::Tensor(v) => &v.identity,
            Self::Summary(v) => v.evidence_identity(),
            Self::Histogram(v) => v.partition_identity(),
            Self::Routed(v) => v.identity(),
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Available,
    Taken,
    Failed,
    Complete,
}
#[derive(Debug)]
struct Slot {
    producer: usize,
    fragment: usize,
    state: State,
    value: Option<PartitionFragmentValue>,
    prefill: Option<prefill::Target>,
    routed: Option<PreparedPartitionRoutedCapture>,
    routed_value: Option<ClaimedPartitionRoutedUnits>,
    routed_invocations: u64,
    routed_pending: bool,
    assembly_cursor: usize,
    custody: CaptureTensorCustody,
}
#[derive(Debug, thiserror::Error)]
enum FragmentCause {
    #[error("partition fragment destination differs from its retained receipt")]
    Source,
    #[error(transparent)]
    Host(#[from] CaptureRunHostError),
    #[error(transparent)]
    Memory(#[from] WorkingMemoryError),
    #[error(transparent)]
    Native(#[from] PartitionCaptureFragmentAllowanceError),
    #[error(transparent)]
    Summary(#[from] CaptureSummaryFailure),
    #[error(transparent)]
    Histogram(#[from] CaptureHistogramFailure),
    #[error(transparent)]
    Routed(#[from] PartitionRoutedCaptureFailure),
    #[error("completed sparse fragment: {0}")]
    RoutedRecord(#[source] CaptureRunHostError, ClaimedPartitionRoutedUnits),
}
/// Failure retains any rejected completed payload before its paid source/host
/// owner. Dropping an error never restores a native quota or destination slot.
#[derive(Debug, thiserror::Error)]
#[error("partition fragment destination: {cause}")]
pub struct PartitionFragmentDestinationError {
    #[source]
    cause: FragmentCause,
    _value: Option<PartitionFragmentValue>,
    _source: SharedCapturePlan,
    _custody: Option<CaptureTensorCustody>,
}
/// Finite receipt-bound local and receive destinations. Each fragment has a
/// distinct private host identity under the same original account, so equal
/// shapes cannot exchange producer/fragment identities.
#[derive(Debug)]
pub struct PreparedPartitionFragmentDestinations {
    rows: Vec<Slot>,
    routed_merge: Vec<bool>,
    identity: [u8; 64],
    allowance: PreparedPartitionFragmentAllowance,
    source: SharedCapturePlan,
    scratch: Option<(usize, usize, CaptureTensorCustody)>,
    prefill: Option<eredu_core::InferenceGeometry>,
    custody: CaptureTensorCustody,
}
impl WorkingMemoryFundingRun {
    /// Protect exactly the quoted table and per-fragment destinations from this
    /// original reservation before their allocation. No native scope is created.
    /// The caller must include this P alongside the existing final-frame P.
    pub fn prepare_partition_fragments(
        &self,
        reservation: &WorkingMemoryReservation,
        plan: PartitionFragmentHostPlan<'_>,
        allowance: PreparedPartitionFragmentAllowance,
    ) -> Result<PreparedPartitionFragmentDestinations, PartitionFragmentDestinationError> {
        let source = allowance.source().clone();
        let fail = |cause| PartitionFragmentDestinationError {
            cause,
            _value: None,
            _source: source.clone(),
            _custody: None,
        };
        if !allowance.matches(plan.receipt) || allowance.local_rank() >= plan.receipt.world_size() {
            return Err(fail(FragmentCause::Source));
        }
        for (producer, projection) in plan.receipt.producers() {
            for fragment in 0..projection.fragments().len() {
                if allowance.fragment_source(producer, fragment).is_none() {
                    return Err(fail(FragmentCause::Source));
                }
            }
        }
        let storage = self.prepare_fragment_host_storage(reservation, &plan)?;
        Ok(storage.bind(plan.receipt, allowance))
    }
    fn prepare_fragment_host_storage(
        &self,
        reservation: &WorkingMemoryReservation,
        plan: &PartitionFragmentHostPlan<'_>,
    ) -> Result<FragmentHostStorage, PartitionFragmentDestinationError> {
        construct_fragment_host_storage(plan, FragmentCustodySource::Text(self, reservation))
    }
}

enum FragmentCustodySource<'a> {
    Text(&'a WorkingMemoryFundingRun, &'a WorkingMemoryReservation),
    Model(&'a crate::working_memory::OriginalSpeculativeBudgetCustody),
}
impl FragmentCustodySource<'_> {
    fn table(
        &self,
        plan: &PartitionFragmentHostPlan<'_>,
    ) -> Result<CaptureTensorCustody, WorkingMemoryError> {
        match self {
            Self::Text(run, reservation) => run.hold_partition_fragment_table(reservation, plan),
            Self::Model(custody) => {
                let held = CaptureTensorCustody::Model((*custody).clone());
                held.validate()?;
                Ok(held)
            }
        }
    }
    fn fragment(
        &self,
        plan: &FragmentHostPlan<'_>,
    ) -> Result<CaptureTensorCustody, WorkingMemoryError> {
        match self {
            Self::Text(run, reservation) => run.hold_partition_fragment(reservation, plan),
            Self::Model(custody) => {
                let held = CaptureTensorCustody::Model((*custody).clone());
                held.validate()?;
                Ok(held)
            }
        }
    }
}

fn construct_fragment_host_storage(
    plan: &PartitionFragmentHostPlan<'_>,
    payer: FragmentCustodySource<'_>,
) -> Result<FragmentHostStorage, PartitionFragmentDestinationError> {
    let source = plan.receipt.shared_plan_source().clone();
    let fail = |cause| PartitionFragmentDestinationError {
        cause,
        _value: None,
        _source: source.clone(),
        _custody: None,
    };
    let custody = payer.table(plan).map_err(|e| fail(e.into()))?;
    custody.validate().map_err(|e| fail(e.into()))?;
    let rows = crate::working_memory::qualified_storage::vector(plan.slots, true).map_err(|e| {
        PartitionFragmentDestinationError {
            cause: e.into(),
            _value: None,
            _source: source.clone(),
            _custody: Some(custody.share_scheduled()),
        }
    })?;
    let mut routed_merge = crate::working_memory::qualified_storage::vector(plan.merge_units, true)
        .map_err(|e| PartitionFragmentDestinationError {
            cause: e.into(),
            _value: None,
            _source: source.clone(),
            _custody: Some(custody.share_scheduled()),
        })?;
    routed_merge.resize(plan.merge_units, false);
    let mut storage = FragmentHostStorage {
        rows,
        routed_merge,
        scratch: None,
        prefill: plan.prefill,
        custody,
    };
    for (producer, projection) in plan.receipt.producers() {
        for fragment in 0..projection.fragments().len() {
            let child = FragmentHostPlan::prepare(plan.receipt, producer, fragment)
                .map_err(|e| storage.error(&source, e.into(), None))?;
            let custody = payer
                .fragment(&child)
                .map_err(|e| storage.error(&source, e.into(), None))?;
            storage.rows.push(Slot {
                producer,
                fragment,
                state: State::Available,
                value: None,
                prefill: None,
                routed: None,
                routed_value: None,
                routed_invocations: 0,
                routed_pending: false,
                assembly_cursor: 0,
                custody,
            });
        }
    }
    if let Some((producer, fragment, bytes)) = plan.assembly {
        let child = FragmentHostPlan::prepare(plan.receipt, producer, fragment)
            .map_err(|e| storage.error(&source, e.into(), None))?;
        if child
            .initialization_peak_bytes()
            .map_err(|e| storage.error(&source, e.into(), None))?
            != bytes
        {
            return Err(storage.error(&source, FragmentCause::Source, None));
        }
        let custody = payer
            .fragment(&child)
            .map_err(|e| storage.error(&source, e.into(), None))?;
        storage.scratch = Some((producer, fragment, custody));
    }
    Ok(storage)
}
impl PreparedPartitionFragmentDestinations {
    fn error(
        &self,
        cause: FragmentCause,
        value: Option<PartitionFragmentValue>,
    ) -> PartitionFragmentDestinationError {
        PartitionFragmentDestinationError {
            cause,
            _value: value,
            _source: self.source.clone(),
            _custody: Some(self.custody.share_scheduled()),
        }
    }
    fn matches(&self, receipt: &PartitionCaptureReceiptPlan) -> bool {
        self.allowance.matches(receipt) && receipt.identity().as_bytes() == self.identity
    }
    fn destination<'a, 'c>(
        &'c mut self,
        receipt: &'a PartitionCaptureReceiptPlan,
        producer: usize,
        fragment: usize,
    ) -> Result<PartitionFragmentDestination<'a, 'c>, PartitionFragmentDestinationError> {
        self.custody
            .validate()
            .map_err(|e| self.error(e.into(), None))?;
        if !self.matches(receipt) {
            return Err(self.error(FragmentCause::Source, None));
        }
        let slot = self.rows.iter_mut().find(|r| {
            r.producer == producer && r.fragment == fragment && r.state == State::Available
        });
        let Some(slot) = slot else {
            return Err(self.error(FragmentCause::Source, None));
        };
        slot.state = State::Taken;
        let identity = ReceiptIdentity {
            phase: receipt.context().phase,
            prediction: receipt.context().prediction,
            index: receipt.context().selection_index,
            custody: slot.custody.share_scheduled(),
        };
        let plan = FragmentHostPlan::prepare(receipt, producer, fragment)
            .map_err(|e| self.error(e.into(), None))?;
        plan.destination(identity)
            .map_err(|e| self.error(e.into(), None))
    }
    /// Consume the matching local native allowance before issuing a destination.
    /// A mismatched actual quote leaves that native attempt permanently spent.
    pub fn take_local<'a, 'c>(
        &'c mut self,
        receipt: &'a PartitionCaptureReceiptPlan,
        fragment: usize,
        dtype: &TensorDtype,
        actual: PartitionCaptureNativeEstimate,
    ) -> Result<NativePartitionFragmentDestination<'a, 'c>, PartitionFragmentDestinationError> {
        if !self.matches(receipt) || self.prefill.is_some() {
            return Err(self.error(FragmentCause::Source, None));
        }
        let producer = self.allowance.local_rank();
        let loan = self
            .allowance
            .take_local_fragment(receipt, fragment, dtype, actual)
            .map_err(|e| self.error(e.into(), None))?;
        let destination = self.destination(receipt, producer, fragment)?;
        Ok(NativePartitionFragmentDestination { destination, loan })
    }
    // Only the shared bounded receipt decoder can issue a remote destination.
    pub(in crate::working_memory::capture_run) fn take_received<'a, 'c>(
        &'c mut self,
        receipt: &'a PartitionCaptureReceiptPlan,
        producer: usize,
        fragment: usize,
    ) -> Result<PartitionFragmentDestination<'a, 'c>, PartitionFragmentDestinationError> {
        if producer == self.allowance.local_rank() {
            return Err(self.error(FragmentCause::Source, None));
        }
        self.destination(receipt, producer, fragment)
    }
    /// Retain a completed typed value under the exact source/ordinal that issued
    /// it. A failed insertion keeps the value in the error and poisons the slot.
    pub fn record(
        &mut self,
        receipt: &PartitionCaptureReceiptPlan,
        producer: usize,
        fragment: usize,
        dtype: &TensorDtype,
        charged: CaptureUsage,
        value: PartitionFragmentValue,
    ) -> Result<(), PartitionFragmentDestinationError> {
        if !self.matches(receipt) {
            return Err(self.error(FragmentCause::Source, Some(value)));
        }
        if let Err(e) = self.custody.validate() {
            return Err(self.error(e.into(), Some(value)));
        }
        let expected = self.allowance.fragment_source(producer, fragment);
        let slot = self
            .rows
            .iter_mut()
            .find(|r| r.producer == producer && r.fragment == fragment && r.state == State::Taken);
        let Some(slot) = slot else {
            return Err(self.error(FragmentCause::Source, Some(value)));
        };
        slot.state = State::Failed;
        let identity = value.identity();
        if !expected
            .is_some_and(|(source, estimate)| source == dtype && estimate.capture == charged)
            || !slot.custody.same_schedule(&identity.custody)
            || identity.phase != receipt.context().phase
            || identity.prediction != receipt.context().prediction
            || identity.index != receipt.context().selection_index
        {
            return Err(self.error(FragmentCause::Source, Some(value)));
        }
        slot.value = Some(value);
        slot.state = State::Complete;
        Ok(())
    }
    /// Completed source values, including genuine empty-producer absence.
    pub fn complete(&self) -> bool {
        self.rows.iter().all(|r| r.state == State::Complete)
    }
    /// Borrow one immutable value; no native completion or all-rank vote implied.
    pub fn value(&self, producer: usize, fragment: usize) -> Option<&PartitionFragmentValue> {
        self.rows
            .iter()
            .find(|r| {
                r.producer == producer && r.fragment == fragment && r.state == State::Complete
            })?
            .value
            .as_ref()
    }
}

mod receive;
pub(crate) use receive::PartitionFragmentReceiveError;
pub(super) use receive::RoutedReceiver;

mod encode;
pub(crate) use encode::PartitionFragmentEncodingError;

mod assembly;
pub(crate) use assembly::PartitionFragmentAssemblyError;

mod delivery;
pub(crate) use delivery::{
    PartitionCaptureRankSource, PartitionFragmentDelivered, PartitionFragmentDeliveryError,
    PreparedPartitionFragmentDelivery,
};

mod prefill;

pub(crate) use delivery::{
    PartitionCaptureHookContinuation, PartitionCaptureHookReturnError, PartitionLocalCaptureHook,
};

mod host_owner;
use host_owner::FragmentHostStorage;
pub use host_owner::{
    OwnedPartitionFragmentHostPlan, PartitionFragmentHostBindingError,
    PartitionFragmentHostPreparationError, PreparedPartitionFragmentHostFunding,
};

mod routed;
