//! Original-account fixed sparse destinations, with ordinary receipt semantics.
use super::*;
mod partition;
pub use partition::*;
mod invocation;
pub use invocation::{CaptureRoutedBatchWriter, CaptureRoutedBatchTransfer, CaptureRoutedModelTransfer};
pub(in crate::working_memory) use invocation::RoutedInvocationTarget;
mod prefill;
pub use prefill::*;
use eredu_core::{TensorObservation, TensorObservationData};

/// Allocation-free cause from the fixed sparse row producer.
#[derive(Debug, Clone, thiserror::Error)]
pub enum CaptureRoutedHostError {
    /// The existing original account is no longer healthy.
    #[error(transparent)]
    Memory(#[from] WorkingMemoryError),
    /// Sparse coordinates, row extent or complete chunk coverage are invalid.
    #[error(transparent)]
    Geometry(#[from] RoutedUnitValidationError),
    /// Shared sparse column assembly rejected identity, overlap or extent.
    #[error(transparent)]
    Assembly(#[from] RoutedUnitAssemblyError),
}

/// Exact sparse row/value banks and their original source selection.
#[derive(Debug)]
pub struct CaptureRoutedHostPlan<'a> {
    geometry: CaptureRoutedUnitsGeometry<'a>,
    peak: u64,
    source_rows: usize,
}
impl<'a> CaptureRoutedHostPlan<'a> {
    /// Price all destinations before constructing any row or scalar buffer.
    pub fn prepare(geometry: CaptureRoutedUnitsGeometry<'a>) -> Result<Self, WorkingMemoryError> {
        let source_rows = geometry.source_shape()[0];
        Self::prepare_source_rows(geometry, source_rows)
    }
    fn prepare_source_rows(geometry: CaptureRoutedUnitsGeometry<'a>, source_rows: usize)
        -> Result<Self, WorkingMemoryError> {
        let rows = geometry.rows();
        let units = geometry.shape()[2];
        let row_values = units
            .checked_mul(size_of::<f32>())
            .filter(|n| *n <= isize::MAX as usize)
            .ok_or(WorkingMemoryError::Overflow)?;
        let per_row = size_of::<Option<RowBuffer>>()
            .checked_add(size_of::<RoutedUnitCaptureRow>())
            .and_then(|n| n.checked_add(size_of::<RoutedUnitRowIdentity>()))
            .and_then(|n| n.checked_add(size_of::<usize>()))
            .and_then(|n| n.checked_add(row_values))
            .ok_or(WorkingMemoryError::Overflow)?;
        let payload = rows
            .checked_mul(per_row)
            .and_then(|n| {
                n.checked_add(source_rows.checked_mul(size_of::<[u64; 2]>())?)
            })
            .and_then(|n| n.checked_add(12 * size_of::<u64>()))
            .filter(|n| *n <= isize::MAX as usize)
            .ok_or(WorkingMemoryError::Overflow)?;
        let frames = [
            size_of::<Self>(),
            size_of::<CaptureRoutedClaim<'a, 'a>>(),
            size_of::<RoutedInvocationTarget>(),
            size_of::<CaptureRoutedBatchWriter<'a, 'a>>(),
            size_of::<CaptureRoutedBatchTransfer<'a, 'a, 'a, u8>>(),
            size_of::<CaptureRoutedModelTransfer<'a, 'a, 'a>>(),
            size_of::<Result<CaptureRoutedModelTransfer<'a, 'a, 'a>, WorkingMemoryError>>(),
            size_of::<ScheduledCaptureRoutedUnits<'a, 'a>>(),
            size_of::<ScheduledCaptureRoutedTransfer<'a, 'a, 'a, u8>>(),
            size_of::<ClaimedCaptureRoutedUnits>(),size_of::<ClaimedAssembledRoutedUnits>(),
            size_of::<Result<ClaimedAssembledRoutedUnits,CaptureRoutedFailure>>(),
            size_of::<(&mut ScheduledCaptureRoutedUnits<'a,'a>,&RoutedUnitCaptureRow,&mut [bool])>(),
            size_of::<(&mut ScheduledCaptureRoutedUnits<'a,'a>,&RoutedUnitCaptureRow,&ResolvedCaptureSlice,&mut [bool])>(),
            size_of::<(&mut ScheduledCaptureStep<'a>,ClaimedAssembledRoutedUnits,TensorDtype,CaptureUsage)>(),
            size_of::<(&mut PreparedCaptureStep<'a>,usize,TensorDtype,RoutedUnitCapture,&ResolvedCaptureSlice,&mut [RoutedUnitRowIdentity],CaptureUsage,bool)>(),
            size_of::<CaptureRoutedFailure>(),
            size_of::<RoutedDestination>(),
            size_of::<OwnedCaptureRoutedUnits>(),
            size_of::<CaptureRoutedPrefillPlan<'_>>(),
            size_of::<CaptureRoutedPrefillFragment<'_, '_>>(),
            size_of::<CaptureRoutedPrefillWriter<'_, '_, '_, '_>>(),
            size_of::<Option<CaptureRoutedPrefillPlan<'_>>>(),
            size_of::<std::ops::RangeInclusive<u64>>() + 6 * size_of::<u64>() + size_of::<&CaptureRoutedPrefillFragment<'_, '_>>(),
            size_of::<RowBuffer>(),
            size_of::<TensorObservation>(),
            size_of::<RoutedUnitCaptureRow>(),
            size_of::<Result<TensorObservation, eredu_core::ObservationError>>(),
        ];
        let peak = frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
            .and_then(|n| n.checked_mul(3))
            .and_then(|n| n.checked_add(CaptureRoutedUnitsGeometry::preparation_control_bytes()?))
            .and_then(|n| n.checked_add(payload))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(Self { geometry, peak, source_rows })
    }
    /// Borrow immutable sparse source geometry without native or allocation authority.
    pub fn geometry(&self) -> &CaptureRoutedUnitsGeometry<'a> {
        &self.geometry
    }
    /// All fixed payloads, scratch and moved constructor/result controls.
    pub fn initialization_peak_bytes(&self) -> u64 {
        self.peak
    }
}
#[derive(Debug)]
struct RowBuffer {
    shape: Vec<usize>,
    values: Vec<f32>,
}
#[derive(Debug, Clone, Copy)]
struct RowCoordinates {
    peer: Option<u64>,
    token: u64,
    slot: u64,
    expert: u64,
    coefficient: f32,
}
#[derive(Debug)]
struct RoutedDestination {
    value: RoutedUnitCapture,
    unused: Vec<Option<RowBuffer>>,
    active: Option<(RowCoordinates, RowBuffer)>,
    scratch: Vec<RoutedUnitRowIdentity>,
    slice: ResolvedCaptureSlice,
}
/// One original H claim; dropping or failing it cannot refund its issuance.
#[derive(Debug)]
pub struct CaptureRoutedClaim<'a, 'c> {
    plan: CaptureRoutedHostPlan<'a>,
    identity: claims::ReceiptIdentity,
    exclusive: PhantomData<&'c mut ()>,
}
impl<'a, 'c> CaptureRoutedClaim<'a, 'c> {
    pub(in crate::working_memory) fn from_partition_plan(plan:CaptureRoutedHostPlan<'a>,identity:claims::ReceiptIdentity)->Self {
        Self{plan,identity,exclusive:PhantomData}
    }
    pub(in crate::working_memory) fn partition_identity(&self)->&claims::ReceiptIdentity{&self.identity}

    /// Exact source and selected coordinates.
    pub fn geometry(&self) -> &CaptureRoutedUnitsGeometry<'a> {
        self.plan.geometry()
    }
    /// Authenticate the scheduled native owner, without issuing a native allocation.
    pub fn validate_native_scope(
        &self,
        native: &WorkingMemoryFundingScope,
    ) -> Result<(), WorkingMemoryError> {
        self.identity.custody.validate_scheduled_native(native)
    }
    /// Allocate only the already priced destination after original custody validates.
    pub fn prepare(self) -> Result<ScheduledCaptureRoutedUnits<'a, 'c>, CaptureRunHostError> {
        let owner = OwnedCaptureRoutedUnits::allocate(&self.plan, self.identity)?;
        Ok(ScheduledCaptureRoutedUnits { owner, exclusive: PhantomData })
    }
    /// Attach exact existing source backing before lending the paid destination.
    pub fn prepare_with_source<'s, K: Clone + Ord + Send + Sync + 'static>(
        self,
        native: &'s mut WorkingMemoryFundingScope,
        source: WorkingMemoryStorage<K>,
    ) -> Result<ScheduledCaptureRoutedTransfer<'a, 'c, 's, K>, CaptureRunHostError> {
        let rollback = self
            .identity
            .custody
            .bind_scheduled_source(native, &source)?;
        let builder = self.prepare()?;
        let native = rollback.commit();
        Ok(ScheduledCaptureRoutedTransfer {
            builder,
            source,
            native,
        })
    }
}
/// Sparse scalar writer. Every buffer has its final capacity before the first row.
#[derive(Debug)]
pub struct ScheduledCaptureRoutedUnits<'a, 'c> {
    owner: OwnedCaptureRoutedUnits,
    exclusive: PhantomData<(&'a AdmittedCapturePlan, &'c mut ())>,
}
/// A frame may retain this partial destination without borrowing its own plan.
/// Exact source extent and receipt identity move with the preallocated payload.
#[derive(Debug)]
pub(in crate::working_memory) struct OwnedCaptureRoutedUnits {
    data: RoutedDestination,
    failure: Option<CaptureRoutedHostError>,
    source_tokens: u64,
    identity: claims::ReceiptIdentity,
}
impl OwnedCaptureRoutedUnits {
    pub(super) fn allocate(plan: &CaptureRoutedHostPlan<'_>, identity: claims::ReceiptIdentity)
        -> Result<Self, CaptureRunHostError> {
        identity.custody.validate()?;
        let g = plan.geometry();
        let mut unused = Vec::with_capacity(g.rows());
        for _ in 0..g.rows() {
            unused.push(Some(RowBuffer {
                shape: vec![g.shape()[2]],
                values: Vec::with_capacity(g.shape()[2]),
            }));
        }
        let data = RoutedDestination {
            value: RoutedUnitCapture {
                geometry: g.bank(),
                source_token_ranges: Vec::with_capacity(plan.source_rows),
                rows: Vec::with_capacity(g.rows()),
            },
            unused,
            active: None,
            scratch: vec![RoutedUnitRowIdentity::default(); g.rows()],
            slice: ResolvedCaptureSlice {
                starts: g.starts().to_vec(),
                ends: g.ends().to_vec(),
                strides: g.strides().to_vec(),
                shape: g.shape().iter().map(|n| *n as u64).collect(),
            },
        };
        Ok(Self { data, failure: None,
            source_tokens: plan.source_rows as u64, identity })
    }
    fn check(&self) -> Result<(), CaptureRoutedHostError> {
        if let Some(error) = &self.failure {
            return Err(error.clone());
        }
        self.identity.custody.validate().map_err(Into::into)
    }
    fn remember(
        &mut self,
        result: Result<(), CaptureRoutedHostError>,
    ) -> Result<(), CaptureRoutedHostError> {
        if let Err(error) = &result {
            if self.failure.is_none() {
                self.failure = Some(error.clone());
            }
        }
        result
    }
    /// Start one original route; the final ordinary validator checks its identity.
    pub fn begin_row(
        &mut self,
        peer: Option<u64>,
        token: u64,
        slot: u64,
        expert: u64,
        coefficient: f32,
    ) -> Result<(), CaptureRoutedHostError> {
        let result = (|| {
            self.check()?;
            if self.data.active.is_some() {
                return Err(RoutedUnitValidationError::Row.into());
            }
            let buffer = self
                .data
                .unused
                .get_mut(self.data.value.rows.len())
                .and_then(Option::take)
                .ok_or(RoutedUnitValidationError::Row)?;
            self.data.active = Some((
                RowCoordinates {
                    peer,
                    token,
                    slot,
                    expert,
                    coefficient,
                },
                buffer,
            ));
            Ok(())
        })();
        self.remember(result)
    }
    /// Copy one selected scalar into an existing slot; non-finite values are preserved.
    pub fn push_f32(&mut self, value: f32) -> Result<(), CaptureRoutedHostError> {
        let result = (|| {
            self.check()?;
            let (_, row) = self
                .data
                .active
                .as_mut()
                .ok_or(RoutedUnitValidationError::Row)?;
            if row.values.len() >= row.shape[0] {
                return Err(RoutedUnitValidationError::Selection.into());
            }
            row.values.push(value);
            Ok(())
        })();
        self.remember(result)
    }
    /// Move one completely filled scalar row into the preallocated receipt bank.
    pub fn finish_row(&mut self) -> Result<(), CaptureRoutedHostError> {
        let result = (|| {
            self.check()?;
            let (_, row) = self
                .data
                .active
                .as_ref()
                .ok_or(RoutedUnitValidationError::Row)?;
            if row.values.len() != row.shape[0] {
                return Err(RoutedUnitValidationError::Selection.into());
            }
            let (coordinates, row) = self.data.active.take().expect("checked active row");
            let values = TensorObservation::new(row.shape, TensorObservationData::F32(row.values))
                .expect("fixed one-dimensional row extent");
            self.data.value.rows.push(RoutedUnitCaptureRow {
                source_peer: coordinates.peer,
                token: coordinates.token,
                slot: coordinates.slot,
                expert: coordinates.expert,
                coefficient: coordinates.coefficient,
                unit_start: self.data.slice.starts[2],
                unit_stride: self.data.slice.strides[2],
                values,
            });
            Ok(())
        })();
        self.remember(result)
    }
    /// Record a real nonempty source chunk, including unselected token rows.
    pub fn source_chunk(&mut self, start: u64, end: u64) -> Result<(), CaptureRoutedHostError> {
        let result = (|| {
            self.check()?;
            let ranges = &mut self.data.value.source_token_ranges;
            // A batched chunk visits batch 0's span, then batch 1's span.
            // Arrival order therefore need not be logical token order. Reject
            // overlaps now; the shared final validator sorts and checks coverage.
            if start >= end
                || end > self.source_tokens
                || ranges.len() >= self.source_tokens as usize
                || ranges.iter().any(|range| start < range[1] && range[0] < end)
            {
                return Err(RoutedUnitValidationError::Chunks.into());
            }
            // Every accepted range consumes at least one original source token.
            ranges.push([start, end]);
            Ok(())
        })();
        self.remember(result)
    }
    fn fail(self, error: CaptureRoutedHostError) -> CaptureRoutedFailure {
        CaptureRoutedFailure {
            data: self.data,
            error,
            custody: self.identity.custody,
        }
    }
    /// Validate exact ordinary coverage and duplicate identities using the paid scratch.
    pub(in crate::working_memory) fn finish(mut self) -> Result<ClaimedCaptureRoutedUnits, CaptureRoutedFailure> {
        let result = (|| {
            self.check()?;
            if self.data.active.is_some() {
                return Err(RoutedUnitValidationError::Incomplete.into());
            }
            self.data.value.finish_ordinary_with_scratch(
                &self.data.slice,
                self.source_tokens,
                &mut self.data.scratch,
            )?;
            Ok(())
        })();
        if let Err(error) = result {
            return Err(self.fail(error));
        }
        Ok(ClaimedCaptureRoutedUnits {
            value: self.data.value,
            scratch: self.data.scratch,
            slice: self.data.slice,
            identity: self.identity,
        })
    }
}
/// Complete original-account distributed rows. Construction is restricted to
/// the shared final assembler; contribution provenance remains a separate owner.
#[derive(Debug)]
pub struct ClaimedAssembledRoutedUnits {value:ClaimedCaptureRoutedUnits}
impl ClaimedAssembledRoutedUnits {
    pub(in crate::working_memory) fn identity(&self)->&claims::ReceiptIdentity{&self.value.identity}
}
impl ScheduledCaptureRoutedUnits<'_, '_> {
    pub(in crate::working_memory) fn begin_assembled_row(&mut self,row:&RoutedUnitCaptureRow,seen:&mut [bool])
        ->Result<(),CaptureRoutedHostError> {
        self.owner.begin_row(row.source_peer,row.token,row.slot,row.expert,row.coefficient)?;
        let (_,buffer)=self.owner.data.active.as_mut().ok_or(RoutedUnitValidationError::Row)?;
        if seen.len()!=buffer.shape[0]||buffer.values.capacity()<seen.len(){return Err(RoutedUnitValidationError::Scratch.into());}
        seen.fill(false);buffer.values.resize(seen.len(),0.0);Ok(())
    }
    pub(in crate::working_memory) fn merge_assembled_row(&mut self,row:&RoutedUnitCaptureRow,destination:&ResolvedCaptureSlice,seen:&mut [bool])
        ->Result<u64,CaptureRoutedHostError> {
        self.owner.check()?;
        let (coordinates,buffer)=self.owner.data.active.as_mut().ok_or(RoutedUnitValidationError::Row)?;
        if (coordinates.peer,coordinates.token,coordinates.slot)!=(row.source_peer,row.token,row.slot){return Err(RoutedUnitValidationError::Row.into());}
        row.merge_partition_values(coordinates.expert,coordinates.coefficient,destination,&mut buffer.values,seen).map_err(Into::into)
    }
    pub(in crate::working_memory) fn finish_partition(mut self)->Result<ClaimedAssembledRoutedUnits,CaptureRoutedFailure> {
        let result=(||{self.owner.check()?;
            if self.owner.data.active.is_some(){return Err(CaptureRoutedHostError::Geometry(RoutedUnitValidationError::Incomplete));}
            self.owner.data.value.finish_partition_with_scratch(&self.owner.data.slice,&mut self.owner.data.scratch)?;Ok(())})();
        if let Err(cause)=result{return Err(self.owner.fail(cause));}
        Ok(ClaimedAssembledRoutedUnits{value:ClaimedCaptureRoutedUnits{value:self.owner.data.value,
            scratch:self.owner.data.scratch,slice:self.owner.data.slice,identity:self.owner.identity}})
    }
}
impl ScheduledCaptureRoutedUnits<'_, '_> {
    /// Begin one selected original route in the fixed destination.
    pub fn begin_row(&mut self, peer: Option<u64>, token: u64, slot: u64,
        expert: u64, coefficient: f32) -> Result<(), CaptureRoutedHostError> {
        self.owner.begin_row(peer, token, slot, expert, coefficient)
    }
    /// Append to the current preallocated scalar row.
    pub fn push_f32(&mut self, value: f32) -> Result<(), CaptureRoutedHostError> {
        self.owner.push_f32(value)
    }
    /// Commit a completely filled row without reallocating its payload.
    pub fn finish_row(&mut self) -> Result<(), CaptureRoutedHostError> {
        self.owner.finish_row()
    }
    /// Record one actual source interval; logical order is validated at sealing.
    pub fn source_chunk(&mut self, start: u64, end: u64) -> Result<(), CaptureRoutedHostError> {
        self.owner.source_chunk(start, end)
    }
    fn fail(self, error: CaptureRoutedHostError) -> CaptureRoutedFailure {
        self.owner.fail(error)
    }
    /// Seal through the same ordinary sparse receipt validator.
    pub fn finish(self) -> Result<ClaimedCaptureRoutedUnits, CaptureRoutedFailure> {
        self.owner.finish()
    }
}
/// Complete sparse payload with its original coordinate and budget owner retained.
#[derive(Debug)]
pub struct ClaimedCaptureRoutedUnits {
    value: RoutedUnitCapture,
    scratch: Vec<RoutedUnitRowIdentity>,
    slice: ResolvedCaptureSlice,
    identity: claims::ReceiptIdentity,
}
impl ClaimedCaptureRoutedUnits {
    /// Read-only sparse values; no mutable payload or raw owning export is available.
    pub fn observation(&self) -> &RoutedUnitCapture {
        &self.value
    }
}
/// Failed or incomplete destinations retain every initialized buffer before custody.
#[derive(Debug)]
pub struct CaptureRoutedFailure {
    data: RoutedDestination,
    error: CaptureRoutedHostError,
    custody: CaptureTensorCustody,
}
impl CaptureRoutedFailure {
    /// Original typed failure; the spent claim cannot be reconstructed.
    pub fn error(&self) -> &CaptureRoutedHostError {
        &self.error
    }
}
impl fmt::Display for CaptureRoutedFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.error, f)
    }
}
impl std::error::Error for CaptureRoutedFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}
/// Exact source-bound destination and exclusive scheduled native scope.
pub struct ScheduledCaptureRoutedTransfer<'a, 'c, 's, K: Ord + Send + 'static> {
    builder: ScheduledCaptureRoutedUnits<'a, 'c>,
    source: WorkingMemoryStorage<K>,
    native: &'s mut WorkingMemoryFundingScope,
}
impl<K: Ord + Send + 'static> ScheduledCaptureRoutedTransfer<'_, '_, '_, K> {
    /// Revalidate the original account and source around native settlement.
    pub fn validate(&self) -> Result<(), WorkingMemoryError> {
        self.builder
            .owner
            .identity
            .custody
            .validate_transfer(self.native, &self.source)
    }
    /// Start a route copied by the enclosing settled native worker.
    pub fn begin_row(
        &mut self,
        peer: Option<u64>,
        token: u64,
        slot: u64,
        expert: u64,
        coefficient: f32,
    ) -> Result<(), CaptureRoutedHostError> {
        self.validate()?;
        self.builder
            .begin_row(peer, token, slot, expert, coefficient)
    }
    /// Copy one scalar after validating its exact source/account.
    pub fn push_f32(&mut self, value: f32) -> Result<(), CaptureRoutedHostError> {
        self.validate()?;
        self.builder.push_f32(value)
    }
    /// Complete an already filled row without further allocation.
    pub fn finish_row(&mut self) -> Result<(), CaptureRoutedHostError> {
        self.validate()?;
        self.builder.finish_row()
    }
    /// Record original source coverage without refunding earlier chunks.
    pub fn source_chunk(&mut self, start: u64, end: u64) -> Result<(), CaptureRoutedHostError> {
        self.validate()?;
        self.builder.source_chunk(start, end)
    }
    /// Publish the host receipt only; native completion remains caller-owned.
    pub fn finish(self) -> Result<ClaimedCaptureRoutedUnits, CaptureRoutedFailure> {
        if let Err(error) = self.validate() {
            return Err(self.builder.fail(error.into()));
        }
        self.builder.finish()
    }
}
impl<K: Ord + Send + 'static> fmt::Debug for ScheduledCaptureRoutedTransfer<'_, '_, '_, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScheduledCaptureRoutedTransfer")
            .field("builder", &self.builder)
            .finish_non_exhaustive()
    }
}
impl<'a> ScheduledCaptureStep<'a> {
    /// Spend this selection's original host destination exactly once.
    pub fn take_routed_units(
        &mut self,
        index: usize,
    ) -> Result<CaptureRoutedClaim<'a, '_>, CaptureRunHostError> {
        self.claim.custody.validate()?;
        if self.frame.prefill.is_some() {
            return Err(CapturePrefillHostError::Identity.into());
        }
        if self
            .claim
            .row
            .get(index.checked_add(1).ok_or(WorkingMemoryError::Overflow)?)
            != Some(&ClaimState::Available)
            || !self
                .frame
                .records()
                .get(index)
                .is_some_and(|r| matches!(r.outcome, CaptureOutcome::Missing))
        {
            return Err(CaptureRunHostError::ClaimUnavailable { index });
        }
        let geometry = match self.claim.window {
            Some(window) => CaptureRoutedUnitsGeometry::prepare_window(
                self.claim.source.admission(),
                index,
                self.claim.phase,
                self.claim.prediction,
                self.claim
                    .invocation
                    .ok_or(CaptureRunHostError::ReceiptMismatch)?,
                window,
            ),
            None => CaptureRoutedUnitsGeometry::prepare(
                self.claim.source.admission(),
                index,
                self.claim.phase,
                self.claim.prediction,
                self.claim.invocation,
            ),
        }
        .map_err(CaptureStepError::from)?;
        let plan = CaptureRoutedHostPlan::prepare(geometry)?;
        self.claim.row[index + 1] = ClaimState::Spent;
        Ok(CaptureRoutedClaim {
            plan,
            identity: claims::ReceiptIdentity {
                phase: self.claim.phase,
                prediction: self.claim.prediction,
                index,
                custody: self.claim.custody.share_scheduled(),
            },
            exclusive: PhantomData,
        })
    }
    pub(in crate::working_memory) fn record_assembled_routed_units(&mut self,receipt:ClaimedAssembledRoutedUnits,dtype:TensorDtype,usage:CaptureUsage)
        ->Result<(),CaptureRunHostError> {
        self.claim.custody.validate()?;let mut receipt=receipt.value;
        if !self.claim.custody.same_schedule(&receipt.identity.custody)||self.claim.phase!=receipt.identity.phase
            ||self.claim.prediction!=receipt.identity.prediction{return Err(CaptureRunHostError::ReceiptMismatch);}
        self.frame.record_partition_routed_units(receipt.identity.index,dtype,receipt.value,&receipt.slice,&mut receipt.scratch,usage)?;Ok(())
    }
    /// Move only a complete receipt belonging to this exact source and prediction.
    pub fn record_routed_units(
        &mut self,
        mut receipt: ClaimedCaptureRoutedUnits,
        dtype: TensorDtype,
        usage: CaptureUsage,
    ) -> Result<(), CaptureRunHostError> {
        self.claim.custody.validate()?;
        if !self.claim.custody.same_schedule(&receipt.identity.custody)
            || self.claim.phase != receipt.identity.phase
            || self.claim.prediction != receipt.identity.prediction
        {
            return Err(CaptureRunHostError::ReceiptMismatch);
        }
        self.frame.record_routed_units(
            receipt.identity.index,
            dtype,
            receipt.value,
            &receipt.slice,
            &mut receipt.scratch,
            usage,
        )?;
        Ok(())
    }
}
