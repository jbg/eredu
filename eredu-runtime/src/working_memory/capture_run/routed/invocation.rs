//! Short provider loans over the ordinary frame's existing sparse destination.
use super::*;

#[derive(Debug)]
pub(in crate::working_memory) enum RoutedInvocationTarget {
    Active {
        owner: OwnedCaptureRoutedUnits,
        dtype: TensorDtype,
    },
    Failed(CaptureRoutedFailure),
}

impl<'a> ScheduledCaptureStep<'a> {
    /// Whether this ordinary invocation has already spent its single sparse claim.
    pub fn routed_invocation_started(&self, index: usize) -> bool {
        self.frame.routed_target(index).is_some()
    }

    /// Install one paid destination after the caller reserves logical usage.
    /// No native source or completion permission follows from this host claim.
    pub fn begin_routed_invocation(
        &mut self,
        index: usize,
        dtype: TensorDtype,
        usage: CaptureUsage,
    ) -> Result<(), CaptureRunHostError> {
        if self.routed_invocation_started(index)
            || self
                .claim
                .row
                .get(index.checked_add(1).ok_or(WorkingMemoryError::Overflow)?)
                != Some(&ClaimState::Available)
            || !self
                .frame
                .records()
                .get(index)
                .is_some_and(|record| matches!(record.outcome, CaptureOutcome::Missing))
        {
            return Err(CaptureRunHostError::ClaimUnavailable { index });
        }
        self.frame
            .charge_prefill_target(index, dtype.clone(), usage)?;
        let owner = self.take_routed_units(index)?.prepare()?.owner;
        *self.frame.routed_target_mut(index)? =
            Some(RoutedInvocationTarget::Active { owner, dtype });
        Ok(())
    }

    /// Borrow the exact original geometry and previously funded destination.
    /// Dropping a writer before completion makes subsequent loans fail.
    pub fn take_routed_batch(
        &mut self,
        index: usize,
    ) -> Result<CaptureRoutedBatchWriter<'_, 'a>, CaptureRunHostError> {
        self.claim.custody.validate()?;
        if self.frame.prefill.is_some() {
            return Err(CapturePrefillHostError::Identity.into());
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
        let Some(RoutedInvocationTarget::Active { owner, .. }) =
            self.frame.routed_target_mut(index)?.as_mut()
        else {
            return Err(CaptureRunHostError::ClaimUnavailable { index });
        };
        owner.check()?;
        Ok(CaptureRoutedBatchWriter {
            owner,
            geometry,
            finished: false,
        })
    }

    /// Seal all real provider batches through ordinary sparse receipt validation.
    pub fn finish_routed_invocation(&mut self, index: usize) -> Result<(), CaptureRunHostError> {
        self.claim.custody.validate()?;
        let target = self
            .frame
            .routed_target_mut(index)?
            .take()
            .ok_or(CaptureRunHostError::ClaimUnavailable { index })?;
        let RoutedInvocationTarget::Active { owner, dtype } = target else {
            *self.frame.routed_target_mut(index)? = Some(target);
            return Err(CaptureRunHostError::ClaimUnavailable { index });
        };
        match owner.finish() {
            Ok(receipt) => self.record_routed_units(receipt, dtype, CaptureUsage::default()),
            Err(failure) => {
                let cause = failure.error().clone();
                *self.frame.routed_target_mut(index)? =
                    Some(RoutedInvocationTarget::Failed(failure));
                Err(cause.into())
            }
        }
    }
}

/// Exclusive short loan. Payload and original custody remain in the frame.
#[derive(Debug)]
pub struct CaptureRoutedBatchWriter<'t, 'a> {
    owner: &'t mut OwnedCaptureRoutedUnits,
    geometry: CaptureRoutedUnitsGeometry<'a>,
    finished: bool,
}
impl CaptureRoutedBatchWriter<'_, '_> {
    /// Exact ordinary source and selected coordinates.
    pub fn geometry(&self) -> &CaptureRoutedUnitsGeometry<'_> {
        &self.geometry
    }
    /// Whether a real source row belongs in the selected sparse destination.
    pub fn selects(&self, token: u64, slot: u64) -> bool {
        [token, slot].into_iter().enumerate().all(|(axis, value)| {
            value >= self.geometry.starts()[axis]
                && value < self.geometry.ends()[axis]
                && (value - self.geometry.starts()[axis]) % self.geometry.strides()[axis] == 0
        })
    }
    /// Match the original run/account before native completion or scalar writing.
    pub fn validate_native_scope(
        &self,
        native: &WorkingMemoryFundingScope,
    ) -> Result<(), WorkingMemoryError> {
        self.owner
            .identity
            .custody
            .validate_scheduled_native(native)
    }
    /// Authenticate the same actual model occurrence as dense model capture.
    /// This is not a scheduled account conversion or native source permission.
    pub fn validate_model_custody(
        &self,
        expected: &crate::working_memory::OriginalSpeculativeBudgetCustody,
    ) -> Result<(), WorkingMemoryError> {
        self.owner.identity.custody.validate_model(expected)
    }

    /// Start one selected original row.
    pub fn begin_row(
        &mut self,
        token: u64,
        slot: u64,
        expert: u64,
        coefficient: f32,
    ) -> Result<(), CaptureRoutedHostError> {
        if !self.selects(token, slot) {
            return self
                .owner
                .remember(Err(RoutedUnitValidationError::Selection.into()));
        }
        self.owner.begin_row(None, token, slot, expert, coefficient)
    }
    /// Write into the already allocated row.
    pub fn push_f32(&mut self, value: f32) -> Result<(), CaptureRoutedHostError> {
        self.owner.push_f32(value)
    }
    /// Finish exactly one complete selected row.
    pub fn finish_row(&mut self) -> Result<(), CaptureRoutedHostError> {
        self.owner.finish_row()
    }
    /// Record actual source coverage, including batches with no selected rows.
    pub fn source_chunk(&mut self, start: u64, end: u64) -> Result<(), CaptureRoutedHostError> {
        self.owner.source_chunk(start, end)
    }
    /// Finish this provider loan. Whole-invocation coverage is checked at sealing.
    pub fn finish(mut self) -> Result<(), CaptureRoutedHostError> {
        self.owner.check()?;
        if self.owner.data.active.is_some() {
            return self
                .owner
                .remember(Err(RoutedUnitValidationError::Incomplete.into()));
        }
        self.finished = true;
        Ok(())
    }
}
impl Drop for CaptureRoutedBatchWriter<'_, '_> {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self
                .owner
                .remember(Err(RoutedUnitValidationError::Incomplete.into()));
        }
    }
}
impl<'t, 'a> CaptureRoutedBatchWriter<'t, 'a> {
    /// Lend the fixed destination under its already-paid model account. Actual
    /// numerical roots/completion stay with that caller's original equation.
    pub fn prepare_model<'s>(
        self,
        model: &'s crate::working_memory::OriginalSpeculativeBudgetCustody,
    ) -> Result<CaptureRoutedModelTransfer<'t, 'a, 's>, WorkingMemoryError> {
        self.validate_model_custody(model)?;
        Ok(CaptureRoutedModelTransfer {
            writer: self,
            model,
        })
    }

    /// Bind independently admitted backing before lending this fixed destination.
    pub fn prepare_with_source<'s, K: Clone + Ord + Send + Sync + 'static>(
        self,
        native: &'s mut WorkingMemoryFundingScope,
        source: WorkingMemoryStorage<K>,
    ) -> Result<CaptureRoutedBatchTransfer<'t, 'a, 's, K>, CaptureRunHostError> {
        let rollback = self
            .owner
            .identity
            .custody
            .bind_scheduled_source(native, &source)?;
        let native = rollback.commit();
        Ok(CaptureRoutedBatchTransfer {
            writer: self,
            source,
            native,
        })
    }
}

/// Native source loan and host writer retain the same original schedule.
pub struct CaptureRoutedBatchTransfer<'t, 'a, 's, K: Ord + Send + 'static> {
    writer: CaptureRoutedBatchWriter<'t, 'a>,
    source: WorkingMemoryStorage<K>,
    native: &'s mut WorkingMemoryFundingScope,
}
impl<K: Ord + Send + 'static> CaptureRoutedBatchTransfer<'_, '_, '_, K> {
    /// Exact original source geometry.
    pub fn geometry(&self) -> &CaptureRoutedUnitsGeometry<'_> {
        self.writer.geometry()
    }
    /// Selected token/slot membership.
    pub fn selects(&self, token: u64, slot: u64) -> bool {
        self.writer.selects(token, slot)
    }
    /// Authenticate before and after settlement.
    pub fn validate(&self) -> Result<(), WorkingMemoryError> {
        self.writer
            .owner
            .identity
            .custody
            .validate_transfer(self.native, &self.source)
    }
    /// Start a selected row after checking native source custody.
    pub fn begin_row(
        &mut self,
        token: u64,
        slot: u64,
        expert: u64,
        coefficient: f32,
    ) -> Result<(), CaptureRoutedHostError> {
        self.validate()?;
        self.writer.begin_row(token, slot, expert, coefficient)
    }
    /// Copy one scalar into its paid slot.
    pub fn push_f32(&mut self, value: f32) -> Result<(), CaptureRoutedHostError> {
        self.validate()?;
        self.writer.push_f32(value)
    }
    /// Complete the selected row.
    pub fn finish_row(&mut self) -> Result<(), CaptureRoutedHostError> {
        self.validate()?;
        self.writer.finish_row()
    }
    /// Track one actual provider span.
    pub fn source_chunk(&mut self, start: u64, end: u64) -> Result<(), CaptureRoutedHostError> {
        self.validate()?;
        self.writer.source_chunk(start, end)
    }
    /// Release this short loan, retaining the frame's incomplete or complete target.
    pub fn finish(self) -> Result<(), CaptureRoutedHostError> {
        self.validate()?;
        self.writer.finish()
    }
}
impl<K: Ord + Send + 'static> fmt::Debug for CaptureRoutedBatchTransfer<'_, '_, '_, K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CaptureRoutedBatchTransfer")
            .field("writer", &self.writer)
            .finish_non_exhaustive()
    }
}

/// Short original model loan over the same fixed sparse destination. This owns
/// no reservation, source pin, native scope or independent logical ledger.
#[derive(Debug)]
pub struct CaptureRoutedModelTransfer<'t, 'a, 's> {
    writer: CaptureRoutedBatchWriter<'t, 'a>,
    model: &'s crate::working_memory::OriginalSpeculativeBudgetCustody,
}
impl CaptureRoutedModelTransfer<'_, '_, '_> {
    /// Exact original source geometry.
    pub fn geometry(&self) -> &CaptureRoutedUnitsGeometry<'_> {
        self.writer.geometry()
    }
    /// Selected token/slot membership.
    pub fn selects(&self, token: u64, slot: u64) -> bool {
        self.writer.selects(token, slot)
    }
    /// Revalidate the original model account before every scalar write.
    pub fn validate(&self) -> Result<(), WorkingMemoryError> {
        self.writer.validate_model_custody(self.model)
    }
    /// Start one selected source row.
    pub fn begin_row(
        &mut self,
        token: u64,
        slot: u64,
        expert: u64,
        coefficient: f32,
    ) -> Result<(), CaptureRoutedHostError> {
        self.validate()?;
        self.writer.begin_row(token, slot, expert, coefficient)
    }
    /// Copy into the existing paid row.
    pub fn push_f32(&mut self, value: f32) -> Result<(), CaptureRoutedHostError> {
        self.validate()?;
        self.writer.push_f32(value)
    }
    /// Finish exactly one complete selected row.
    pub fn finish_row(&mut self) -> Result<(), CaptureRoutedHostError> {
        self.validate()?;
        self.writer.finish_row()
    }
    /// Keep the actual source span, including an empty selected intersection.
    pub fn source_chunk(&mut self, start: u64, end: u64) -> Result<(), CaptureRoutedHostError> {
        self.validate()?;
        self.writer.source_chunk(start, end)
    }
    /// End the short loan; the original frame still owns final coverage checks.
    pub fn finish(self) -> Result<(), CaptureRoutedHostError> {
        self.validate()?;
        self.writer.finish()
    }
}
