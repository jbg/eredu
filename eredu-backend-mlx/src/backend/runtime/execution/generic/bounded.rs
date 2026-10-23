//! Runtime acquisition for bounded layerwise execution policies.

use super::*;
use eredu_core::Completion;
use eredu_runtime::SubmissionBackend;

impl<U, P> LayerwisePolicy<MlxNeuralBackend, U> for MlxLayerwisePolicy<U, P>
where
    U: Parameterized<MlxTensor> + 'static,
    P: MlxUnitPopulator<U>,
{
    type Lease = MlxUnitLease<U>;
    type Error = Error;

    fn publish_parameter_replacements(
        &mut self,
        values: &std::collections::BTreeMap<String, MlxTensor>,
        active: bool,
    ) -> Result<bool, Error> {
        if !self.pending.is_empty()
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
        self.drain().map_err(LayerwiseAcquireError::Policy)?;
        if self.dense.as_ref().is_some_and(|dense| {
            dense.forward.is_some() || dense.windows.iter().any(Option::is_some)
        }) {
            return Err(LayerwiseAcquireError::Policy(Error::Parallel(
                "parameter inspection requires an idle streaming policy".into(),
            )));
        }
        let window = self
            .identity_addresses
            .storage_window(ordinal, self.window_depth)
            .expect("validated inspection address");
        self.trim_device_window(&window)
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
        let event = async_eval_with_event([&marker])
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
        drop(active);
        // Each unit's pre-submission owner retires independently. Neither an
        // error nor unwinding is evidence that its consumer graph has stopped.
        self.pending.clear();
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
        let mapping = self.identity_addresses.clone();
        let invocation = mapping
            .resolve(index, address)
            .map_err(|error| LayerwiseAcquireError::Policy(error.into()))?;
        self.acquire_mapped(&mapping, invocation, build, stream)
    }

    fn complete<'a, StateValues, ContextValues>(
        &mut self,
        index: usize,
        address: ExecutionUnitAddress,
        lease: Self::Lease,
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
        let mapping = self.identity_addresses.clone();
        let invocation = mapping.resolve(index, address)?;
        self.complete_mapped(
            &mapping,
            invocation,
            lease,
            output,
            state_values,
            context_values,
        )
    }

    fn finish(&mut self, output: &MlxTensor, stream: &Stream) -> Result<(), Self::Error> {
        MlxNeuralBackend::submit(stream, [output])?.wait()?;
        self.drain()?;
        if self.dense.is_none() && (self.sample_mlx_memory || self.sample_process_memory) {
            self.residency
                .sample_memory(self.sample_mlx_memory, self.sample_process_memory)?;
        }
        if let Some(dense) = &mut self.dense {
            for window in &mut dense.windows {
                window.take();
            }
            for group in &mut dense.groups {
                if let Some(group) = group.take() {
                    group.complete()?;
                }
            }
            if let Some(forward) = dense.forward.take() {
                forward.complete()?;
            }
        }
        Ok(())
    }
}

impl<U, P> MlxLayerwisePolicy<U, P>
where
    U: Parameterized<MlxTensor> + 'static,
    P: MlxUnitPopulator<U>,
{
    pub(crate) fn acquire_mapped<E, BF>(
        &mut self,
        mapping: &eredu_runtime::ExecutionUnitAddressMap,
        invocation: eredu_runtime::ExecutionUnitLeaseAddress,
        build: BF,
        stream: &Stream,
    ) -> Result<MlxUnitLease<U>, LayerwiseAcquireError<E, Error>>
    where
        BF: FnOnce(&Stream) -> Result<U, E>,
    {
        let storage = invocation.storage_location();
        let index = storage.ordinal();
        let address = storage.address();
        crate::backend::submission_recovery::reap();
        crate::backend::ordinary_retirement::reclaim();
        let mut unloaded = Some(build(stream).map_err(LayerwiseAcquireError::Architecture)?);
        self.reap_completed()
            .map_err(LayerwiseAcquireError::Policy)?;
        if self.dense.is_some() {
            let group = address.group();
            let group_id = self
                .layout
                .group_id(group)
                .expect("validated unit address names its execution group")
                .as_str();
            let dense = self.dense.as_mut().expect("dense policy is active");
            if dense.groups[group].is_none() {
                dense.groups[group] = Some(dense.controller.group_guard(&self.residency, group_id));
            }
            if dense.windows[group].is_none() {
                let sequence = mapping
                    .group_storage_ordinals(group)
                    .expect("validated invocation group");
                dense.windows[group] = Some(
                    dense
                        .controller
                        .transfer_window(
                            &self.residency,
                            group_id,
                            &self.unit_ids,
                            sequence,
                            dense.prefill,
                        )
                        .map_err(LayerwiseAcquireError::Policy)?,
                );
            }
            loop {
                let refill = self
                    .dense
                    .as_mut()
                    .and_then(|dense| dense.windows[group].as_mut())
                    .expect("dense forward begins before acquisition")
                    .refill();
                match refill {
                    Ok(()) => break,
                    Err(_) if !self.pending.is_empty() => {
                        self.drain_one().map_err(LayerwiseAcquireError::Policy)?
                    }
                    Err(error) => return Err(LayerwiseAcquireError::Policy(error)),
                }
            }
            let transfer = self
                .dense
                .as_mut()
                .and_then(|dense| dense.windows[group].as_mut())
                .expect("dense forward begins before acquisition")
                .next(stream)
                .map_err(LayerwiseAcquireError::Policy)?;
            let group_start = mapping
                .group_range(group)
                .expect("validated invocation group")
                .start;
            if transfer.step().storage_index() != index
                || transfer.step().ordinal() != invocation.invocation_ordinal() - group_start
            {
                return Err(LayerwiseAcquireError::Policy(Error::Parallel(format!(
                    "dense transfer returned unit {}, expected {index}",
                    transfer.step().storage_index()
                ))));
            }
            let unit = MlxModule::new(unloaded.take().expect("unloaded unit is consumed once"));
            let mut lease = MlxUnitLease::new(
                unit,
                MlxUnitTransfer::Dense {
                    _transfer: transfer,
                },
            )
            .map_err(LayerwiseAcquireError::Policy)?;
            let (unit, transfer) = lease.population_parts();
            self.populator
                .populate(unit, transfer)
                .map_err(LayerwiseAcquireError::Policy)?;
            return Ok(lease);
        }
        // An ordinary lookahead transfer owns every lease in its window until
        // the preceding unit's exact completion.  Release that completed
        // window before requesting the overlapping next window; otherwise an
        // overlapping in-flight unit would wait for the very transfer guard
        // retained by `pending` on this thread.
        self.drain_one().map_err(LayerwiseAcquireError::Policy)?;
        let window = mapping
            .storage_window(invocation.invocation_ordinal(), self.window_depth)
            .expect("validated invocation");
        self.trim_device_window(&window)
            .map_err(LayerwiseAcquireError::Policy)?;
        let requests = window
            .iter()
            .map(|storage| (self.unit_ids[storage.ordinal()].clone(), 1))
            .collect::<Vec<_>>();
        let transfer = loop {
            match self
                .residency
                .acquire_many_with_transfer(&requests, MemoryTier::Device)
            {
                Ok(transfer) => break transfer,
                Err(_) if !self.pending.is_empty() => {
                    self.drain_one().map_err(LayerwiseAcquireError::Policy)?
                }
                Err(error) => return Err(LayerwiseAcquireError::Policy(error.into())),
            }
        };
        transfer
            .order_after(stream)
            .map_err(Error::from)
            .map_err(LayerwiseAcquireError::Policy)?;
        let unit = MlxModule::new(unloaded.take().expect("unloaded unit is consumed once"));
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
        Ok(lease)
    }

    pub(crate) fn complete_mapped<'a, StateValues, ContextValues>(
        &mut self,
        mapping: &eredu_runtime::ExecutionUnitAddressMap,
        invocation: eredu_runtime::ExecutionUnitLeaseAddress,
        mut lease: MlxUnitLease<U>,
        output: &'a MlxTensor,
        state_values: StateValues,
        context_values: ContextValues,
    ) -> Result<(), Error>
    where
        MlxTensor: 'a,
        StateValues: Iterator<Item = &'a MlxTensor>,
        ContextValues: Iterator<Item = &'a MlxTensor>,
    {
        let address = invocation.storage_location().address();
        let event = async_eval_with_event(
            std::iter::once(output)
                .chain(state_values)
                .chain(context_values)
                .map(MlxTensor::as_array),
        )?;
        lease.submitted(event)?;
        self.pending.push_back(lease);
        if self.dense.is_some() && mapping.is_group_end(invocation.invocation_ordinal()) {
            // Device budgets cover one group window, not the sum of successive
            // groups. Establish every consumer's exact completion before
            // releasing this group's transfers and host/device protections.
            self.drain()?;
            let dense = self.dense.as_mut().expect("active dense group");
            dense.windows[address.group()].take();
            if let Some(group) = dense.groups[address.group()].take() {
                group.complete()?;
            }
        }
        Ok(())
    }
}
