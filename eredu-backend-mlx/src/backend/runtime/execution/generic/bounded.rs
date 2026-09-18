//! Runtime acquisition for bounded layerwise execution policies.

use super::*;
use eredu_core::Completion;
use eredu_runtime::{OrderedLayerwiseCompletion, SubmissionBackend};

#[derive(Clone, Copy)]
pub(super) enum AcquisitionRecovery {
    SingleAttempt,
    AfterRetirement,
}

impl<U, P> LayerwisePolicy<MlxNeuralBackend, U> for MlxLayerwisePolicy<U, P>
where
    U: Parameterized<MlxTensor> + 'static,
    P: MlxUnitPopulator<U>,
{
    type Lease = MlxUnitLease<U>;
    type Error = Error;

    fn uses_shared_group_executor(&self, stream: &Stream) -> Result<bool, Error> {
        self.uses_shared_neural_executor(stream)
    }

    fn submit_group(
        &mut self,
        stream: &Stream,
        value: &MlxTensor,
    ) -> Result<
        impl OrderedLayerwiseCompletion<Stream> + 'static,
        impl std::error::Error + Send + Sync + 'static,
    > {
        self.submit_neural(stream, value)
    }

    fn retained_value_slot_bound(&self) -> Option<usize> {
        if !self.pending_is_empty().unwrap_or(false)
            || self.dense.as_ref().is_some_and(|dense| {
                dense.forward.is_some() || dense.windows.iter().any(Option::is_some)
            })
        {
            return None;
        }
        self.populator.retained_value_slot_bound()
    }

    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&MlxTensor)) -> bool {
        // Pending units can still be owned by native recovery. Admission must
        // reach the ordinary completed boundary before inventorying this policy.
        if !self.pending_is_empty().unwrap_or(false)
            || self.dense.as_ref().is_some_and(|dense| {
                dense.forward.is_some() || dense.windows.iter().any(Option::is_some)
            })
        {
            return false;
        }
        self.populator.visit_retained_values(visitor)
    }

    fn publish_parameter_replacements(
        &mut self,
        values: &std::collections::BTreeMap<String, MlxTensor>,
        active: bool,
    ) -> Result<bool, Error> {
        if !self.pending_is_empty().unwrap_or(false)
            || self
                .dense
                .as_ref()
                .is_some_and(|dense| dense.forward.is_some())
        {
            return Err(Error::Parallel(
                "parameter publication requires an idle policy".into(),
            ));
        }
        Ok(self
            .populator
            .publish_parameter_replacements(values, active))
    }

    fn inspect_unit<E, F, V>(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        build: F,
        operation: V,
        stream: &Stream,
    ) -> Result<bool, LayerwiseAcquireError<E, Self::Error>>
    where
        F: FnOnce(&Stream) -> Result<U, E>,
        V: FnOnce(&mut U) -> Result<(), Self::Error>,
    {
        crate::backend::submission_recovery::reap();
        crate::backend::ordinary_retirement::reclaim_all();
        if self.layout.address(ordinal) != Some(address) {
            return Err(LayerwiseAcquireError::Policy(Error::Parallel(
                "parameter unit address differs from the selected layout".into(),
            )));
        }
        // Managed direct dispatch is exclusively the quoted forward traversal;
        // ad-hoc parameter inspection has no admitted constructor/window span.
        if self.residency.admitted_disk_route_active() {
            return Err(LayerwiseAcquireError::Policy(Error::Other(Box::new(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            ))));
        }
        self.drain().map_err(LayerwiseAcquireError::Policy)?;
        if self.dense.as_ref().is_some_and(|dense| {
            dense.forward.is_some() || dense.windows.iter().any(Option::is_some)
        }) {
            return Err(LayerwiseAcquireError::Policy(Error::Parallel(
                "parameter inspection requires an idle streaming policy".into(),
            )));
        }
        self.trim_device_window(ordinal, address)
            .map_err(LayerwiseAcquireError::Policy)?;
        let unit = MlxModule::new(build(stream).map_err(LayerwiseAcquireError::Architecture)?);
        let transfer = self
            .residency
            .acquire_many_with_transfer(&[(self.unit_ids[ordinal].clone(), 1)], MemoryTier::Device)
            .map_err(|error| LayerwiseAcquireError::Policy(error.into()))?;
        transfer
            .order_after(stream)
            .map_err(|error| LayerwiseAcquireError::Policy(error.into()))?;
        let mut lease = MlxUnitLease::new(
            unit,
            MlxUnitTransfer::Ordinary {
                _transfer: transfer,
            },
        )
        .map_err(LayerwiseAcquireError::Policy)?;
        let (unit, transfer) = lease.population_parts();
        self.populator
            .populate(unit, transfer)
            .map_err(LayerwiseAcquireError::Policy)?;
        operation(&mut lease).map_err(LayerwiseAcquireError::Policy)?;
        // The operation evaluates every result it exports. Seal an event and the
        // enclosing scope as well, so errors and side submissions retain the unit.
        let marker = safemlx::ops::zeros_dtype(&[1], safemlx::Dtype::Float32, stream)
            .map_err(|error| LayerwiseAcquireError::Policy(error.into()))?;
        let event = async_eval_with_operation_event([&marker])
            .map_err(|error| LayerwiseAcquireError::Policy(error.into()))?;
        lease
            .submitted(event)
            .map_err(LayerwiseAcquireError::Policy)?;
        lease.finish().map_err(LayerwiseAcquireError::Policy)?;
        // A transaction can inspect another unit before the outer session scope
        // ends. Release completed transfer leases at this explicit host boundary
        // so the next unit can evict this one within the selected window.
        crate::backend::ordinary_retirement::reclaim_all();
        Ok(true)
    }

    fn begin(&mut self, initial: &MlxTensor, _stream: &Stream) -> Result<(), Self::Error> {
        let original = self.original_operation_projection().is_some();
        let background = self
            .original_operation_projection()
            .map(|view| view.access()?.has_background())
            .transpose()?
            .unwrap_or(false);
        if self.residency.admitted_disk_route_active()
            && (!self.dense_window_matches_layout()
                || !self.dense.as_ref().is_some_and(|dense| {
                    dense.controller.is_foreground()
                        || (background && dense.controller.is_source_prepared())
                }))
        {
            return Err(Error::Other(Box::new(
                eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
            )));
        }
        crate::backend::submission_recovery::reap();
        crate::backend::ordinary_retirement::reclaim();
        let Some(dense) = &mut self.dense else {
            return Ok(());
        };
        if dense.aborted {
            // Manager/telemetry cleanup can acquire locks. Defer it from the
            // error path to this explicit ordinary operation; policy teardown
            // instead stages the entire dense owner for unlocked reclamation.
            for window in &mut dense.windows {
                window.take();
            }
            for group in &mut dense.groups {
                group.take();
            }
            dense.forward.take();
            if dense.controller.is_source_prepared() {
                dense.windows = Vec::new();
                dense.groups = Vec::new();
            }
            dense.aborted = false;
        }
        if dense.windows.iter().any(Option::is_some)
            || dense.forward.is_some()
            || dense.groups.iter().any(Option::is_some)
        {
            return Err(Error::Parallel(
                "dense MLX policy began a forward while another remained active".into(),
            ));
        }
        if dense.controller.is_source_prepared() {
            if original {
                if dense.windows.capacity() != 0 || dense.groups.capacity() != 0 {
                    return Err(Error::PrefillControl(
                        eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch,
                    ));
                }
            } else {
                dense
                    .windows
                    .resize_with(self.layout.group_count(), || None);
                dense.groups.resize_with(self.layout.group_count(), || None);
            }
        }
        let prefill = initial.as_array().dim(1) > 1;
        let forward = dense.controller.forward_guard(prefill, &self.residency)?;
        dense.prefill = prefill;
        dense.forward = Some(forward);
        Ok(())
    }

    fn abort(
        &mut self,
        active: Option<(usize, ExecutionUnitAddress, Self::Lease)>,
        _stream: &Stream,
    ) {
        if let Some(original) = self.original_operation_projection() {
            if let Ok(access) = original.access() {
                access.fence();
            }
        }
        drop(active);
        // Each unit's pre-submission owner retires independently. Neither an
        // error nor unwinding is evidence that its consumer graph has stopped.
        if let Some(original) = self.original_operation_projection() {
            if let Ok(access) = original.access() {
                let _ = access.discard_pending();
            }
        } else {
            self.pending.clear();
        }
        if let Some(dense) = &mut self.dense {
            dense.aborted = true;
        }
    }

    fn acquire<E, BF>(
        &mut self,
        index: usize,
        address: ExecutionUnitAddress,
        build: BF,
        stream: &Stream,
    ) -> Result<Self::Lease, LayerwiseAcquireError<E, Self::Error>>
    where
        BF: FnOnce(&Stream) -> Result<U, E>,
    {
        let original = self
            .original_operation_projection()
            .map(|view| view.access())
            .transpose()
            .map_err(LayerwiseAcquireError::Policy)?;
        let prepared = original
            .as_ref()
            .map(|view| view.checkout_unit())
            .transpose()
            .map_err(LayerwiseAcquireError::Policy)?;
        if original.is_none() {
            crate::backend::submission_recovery::reap();
            crate::backend::ordinary_retirement::reclaim();
        }
        if self.layout.address(index) != Some(address) {
            return Err(LayerwiseAcquireError::Policy(Error::Parallel(format!(
                "execution unit {index} does not match group {} unit {}",
                address.group(),
                address.index()
            ))));
        }
        let has_background = original
            .as_ref()
            .map(|view| view.has_background())
            .transpose()
            .map_err(LayerwiseAcquireError::Policy)?
            .unwrap_or(false);
        if original.is_some()
            && self.dense.as_ref().is_some_and(|dense| {
                !dense.controller.is_source_prepared()
                    || (!dense.controller.is_foreground() && !has_background)
                    || self
                        .residency
                        .original_foreground_disk_descriptors()
                        .is_none()
            })
        {
            return Err(LayerwiseAcquireError::Policy(Error::PrefillControl(
                eredu_runtime::working_memory::WorkingMemoryError::UnknownBound,
            )));
        }
        let mut residency_attempt = original
            .as_ref()
            .map(|view| view.checkout_residency(index))
            .transpose()
            .map_err(LayerwiseAcquireError::Policy)?;
        let admitted = self
            .prepare_admitted_disk_window(index, address)
            .map_err(LayerwiseAcquireError::Policy)?;
        let recovery = if original.is_some() || admitted {
            AcquisitionRecovery::SingleAttempt
        } else {
            AcquisitionRecovery::AfterRetirement
        };
        let unloaded = build(stream).map_err(LayerwiseAcquireError::Architecture)?;
        self.reap_completed()
            .map_err(LayerwiseAcquireError::Policy)?;
        let finish = |this: &mut Self,
                      transfer: MlxUnitTransfer,
                      attempt: &mut Option<OriginalResidencyAttempt>|
         -> Result<Self::Lease, Error> {
            if let MlxUnitTransfer::Ordinary {
                _transfer: transfer,
            } = &transfer
            {
                match &original {
                    Some(source) => source.with_residency(
                        attempt.as_mut().expect("source-funded indexed attempt"),
                        |slots, current| {
                            transfer
                                .order_after_original(stream, slots.observations, current)
                                .map_err(Error::from)
                        },
                    )?,
                    None => transfer.order_after(stream)?,
                }
            }
            if original.is_some() {
                if let Some(dense) = &this.dense {
                    dense.controller.observe_group(
                        &this.residency,
                        this.layout
                            .group_id(address.group())
                            .expect("validated group")
                            .as_str(),
                        dense.prefill,
                    )?;
                }
            }
            let unit = MlxModule::new(unloaded);
            let mut lease = match prepared {
                Some((prepared, observer)) => {
                    MlxUnitLease::from_prepared(prepared, unit, transfer, observer)
                }
                None => MlxUnitLease::new(unit, transfer)?,
            };
            let (unit, transfer) = lease.population_parts();
            match &original {
                Some(source) => {
                    this.populator
                        .populate_original(unit, transfer, source.binding_row_limit()?)?
                }
                None => this.populator.populate(unit, transfer)?,
            }
            Ok(lease)
        };
        let result = (|| -> Result<Self::Lease, Error> {
            // The next window may retire the preceding unit only after its
            // actual consumer and transfer complete, on either source path.
            self.drain_one()?;
            if self.dense.is_some() && original.is_none() {
                let transfer = self.acquire_dense_transfer(index, address, recovery, stream)?;
                return finish(
                    self,
                    MlxUnitTransfer::Dense {
                        _transfer: transfer,
                    },
                    &mut residency_attempt,
                );
            }

            // The ordinary dense scheduler replaces its named protection
            // before invoking this same manager trim. Original indexed windows
            // have no ordinary scheduler guard to replace here.
            self.trim_device_window(index, address)?;
            let window = self
                .layout
                .window_range(
                    index,
                    std::num::NonZeroUsize::new(self.window_depth).expect("validated window depth"),
                )
                .expect("validated unit address has a window");
            let requests = match &original {
                Some(source) => std::borrow::Cow::Borrowed(source.requests(window)?),
                None => std::borrow::Cow::Owned(
                    self.unit_ids[window]
                        .iter()
                        .cloned()
                        .map(|id| (id, 1))
                        .collect::<Vec<_>>(),
                ),
            };
            if has_background {
                let source = original
                    .as_ref()
                    .expect("background source is authenticated");
                // Keep promotion, ordering and population inside this attempt:
                // failure must fence the real background read coordinator.
                return source.with_background_window(index, |reads, window| {
                    let transfer = source.with_host_and_device(
                        index,
                        residency_attempt
                            .as_mut()
                            .expect("source-funded indexed attempt"),
                        |host, device, current| {
                            window
                                .promote(&self.residency, reads, &requests, host, device, current)
                                .map_err(Error::from)
                        },
                    )?;
                    finish(
                        self,
                        MlxUnitTransfer::Ordinary {
                            _transfer: transfer,
                        },
                        &mut residency_attempt,
                    )
                });
            }
            // Funded attempts are finite. Only ordinary execution can retry,
            // and only after retiring an actual pending completion owner.
            let transfer = self.recover_acquisition(recovery, |this| match &original {
                Some(source) => source.with_residency(
                    residency_attempt
                        .as_mut()
                        .expect("source-funded indexed attempt"),
                    |slots, current| {
                        this.residency
                            .acquire_many_with_original_transfer(
                                &requests,
                                MemoryTier::Device,
                                slots,
                                current,
                            )
                            .map_err(Error::from)
                    },
                ),
                None => this
                    .residency
                    .acquire_many_with_transfer(&requests, MemoryTier::Device)
                    .map_err(Error::from),
            })?;
            finish(
                self,
                MlxUnitTransfer::Ordinary {
                    _transfer: transfer,
                },
                &mut residency_attempt,
            )
        })();
        result.map_err(LayerwiseAcquireError::Policy)
    }

    fn complete<'a, StateValues, ContextValues>(
        &mut self,
        _index: usize,
        _address: ExecutionUnitAddress,
        mut lease: Self::Lease,
        output: &'a MlxTensor,
        state_values: StateValues,
        context_values: ContextValues,
        _stream: &Stream,
    ) -> Result<(), Self::Error>
    where
        MlxTensor: 'a,
        StateValues: Iterator<Item = &'a MlxTensor>,
        ContextValues: Iterator<Item = &'a MlxTensor>,
    {
        let arrays = std::iter::once(output)
            .chain(state_values)
            .chain(context_values)
            .map(MlxTensor::as_array);
        let original = if let Some(observer) = lease.original_observer() {
            let access = self
                .original_operation_projection()
                .ok_or(Error::PrefillScopeUnavailable)?
                .access()?;
            access.validate_observer(observer)?;
            Some(access)
        } else {
            None
        };
        let event = match lease.original_observer() {
            Some(observer) => {
                safemlx::transforms::async_eval_with_original_operation_event(arrays, observer)?
            }
            None => async_eval_with_operation_event(arrays)?,
        };
        lease.submitted(event)?;
        match original {
            Some(original) => original.push_pending(lease)?,
            None => self.pending.push_back(lease),
        }
        Ok(())
    }

    fn finish(&mut self, output: &MlxTensor, stream: &Stream) -> Result<(), Self::Error> {
        let original = self.original_operation_projection().is_some();
        // Use the same prepaid bank as graph boundaries. Terminal original
        // payload destruction and exact-owner drain precede pending-unit drain.
        self.submit_neural(stream, output)?.finish()?;
        self.drain()?;
        if let Some(view) = self.original_operation_projection() {
            if let Some(report) = view.access()?.finish_background_forward()? {
                self.dense
                    .as_ref()
                    .ok_or(Error::PrefillScopeUnavailable)?
                    .controller
                    .record_background(report)?;
            }
        }
        if self.dense.is_none() && (self.sample_mlx_memory || self.sample_process_memory) {
            self.residency
                .sample_memory(self.sample_mlx_memory, self.sample_process_memory)?;
        }
        if let Some(dense) = &mut self.dense {
            let result = (|| -> Result<(), Error> {
                for window in &mut dense.windows {
                    window.take();
                }
                for group in &mut dense.groups {
                    if let Some(group) = group.take() {
                        group.complete()?;
                    }
                }
                if original && dense.controller.is_source_prepared() {
                    for group in 0..self.layout.group_count() {
                        if self
                            .layout
                            .group_range(group)
                            .expect("validated group")
                            .is_empty()
                        {
                            continue;
                        }
                        dense.controller.record_group_execution(
                            self.layout
                                .group_id(group)
                                .expect("validated group")
                                .as_str(),
                        )?;
                    }
                }
                if let Some(forward) = dense.forward.take() {
                    forward.complete()?;
                }
                Ok(())
            })();
            if dense.controller.is_source_prepared() {
                // Ordinary sessions on the same source use the same driver,
                // but their mutable buffers cannot become original source credit.
                dense.windows = Vec::new();
                dense.groups = Vec::new();
            }
            result?;
        }
        Ok(())
    }
}

impl<U, P> MlxLayerwisePolicy<U, P>
where
    U: Parameterized<MlxTensor> + 'static,
    P: MlxUnitPopulator<U>,
{
    fn recover_acquisition<T>(
        &mut self,
        recovery: AcquisitionRecovery,
        mut acquire: impl FnMut(&mut Self) -> Result<T, Error>,
    ) -> Result<T, Error> {
        loop {
            match acquire(self) {
                Ok(value) => return Ok(value),
                Err(_)
                    if matches!(recovery, AcquisitionRecovery::AfterRetirement)
                        && !self.pending_is_empty()? =>
                {
                    self.drain_one()?
                }
                Err(cause) => return Err(cause),
            }
        }
    }

    fn acquire_dense_transfer(
        &mut self,
        index: usize,
        address: ExecutionUnitAddress,
        recovery: AcquisitionRecovery,
        stream: &Stream,
    ) -> Result<DensePreparedTransfer, Error> {
        let group = address.group();
        let group_id = self
            .layout
            .group_id(group)
            .expect("validated unit address names its execution group")
            .as_str();
        let range = self
            .layout
            .group_range(group)
            .expect("validated unit address has a group range");
        let dense = self.dense.as_mut().expect("dense policy is active");
        // acquire has settled the preceding consumer. A different group's
        // window can now retire only if every scheduled transfer was consumed.
        // Keep unconsumed windows intact on rejection; they are still owners.
        if dense.windows.iter().enumerate().any(|(other, window)| {
            other != group && window.as_ref().is_some_and(|window| !window.is_exhausted())
        }) {
            return Err(Error::Parallel(
                "dense execution changed groups before consuming its window".into(),
            ));
        }
        for other in 0..dense.windows.len() {
            if other != group {
                dense.windows[other].take();
                if let Some(completed) = dense.groups[other].take() {
                    completed.complete()?;
                }
            }
        }
        // All other groups are now idle. Their cached copies do not widen the
        // selected device window; exact persistent alias owners remain exempt.
        self.residency
            .trim_device_units(&self.unit_ids, &self.unit_ids[range.clone()])?;
        if dense.groups[group].is_none() {
            dense.groups[group] = Some(dense.controller.group_guard(&self.residency, group_id));
        }
        if dense.windows[group].is_none() {
            dense.windows[group] = Some(dense.controller.transfer_window(
                &self.residency,
                group_id,
                &self.unit_ids[range.clone()],
                0..range.len(),
                dense.prefill,
            )?);
        }
        self.recover_acquisition(recovery, |this| {
            this.dense
                .as_mut()
                .and_then(|dense| dense.windows[group].as_mut())
                .expect("dense forward begins before acquisition")
                .refill()
        })?;
        let transfer = self
            .dense
            .as_mut()
            .and_then(|dense| dense.windows[group].as_mut())
            .expect("dense forward begins before acquisition")
            .next(stream)?;
        if transfer.index() != address.index() {
            return Err(Error::Parallel(format!(
                "dense transfer returned unit {}, expected {index}",
                range.start + transfer.index(),
            )));
        }
        Ok(transfer)
    }
}
