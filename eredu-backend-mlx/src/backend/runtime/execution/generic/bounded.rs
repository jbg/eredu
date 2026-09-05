//! Runtime acquisition for bounded layerwise execution policies.

use super::*;

impl<U, P> LayerwisePolicy<MlxNeuralBackend, U> for MlxLayerwisePolicy<U, P>
where
    U: Parameterized<MlxTensor>,
    P: MlxUnitPopulator<U>,
{
    type Lease = MlxUnitLease<U>;
    type Error = Error;

    fn begin(&mut self, initial: &MlxTensor, _stream: &Stream) -> Result<(), Self::Error> {
        let Some(dense) = &mut self.dense else {
            return Ok(());
        };
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
        while let Some((event, lease)) = self.pending.pop_front() {
            let _ = event.synchronize();
            drop(lease);
        }
        if let Some(dense) = &mut self.dense {
            for window in &mut dense.windows {
                window.take();
            }
            for group in &mut dense.groups {
                group.take();
            }
            dense.forward.take();
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
        let mut unloaded = Some(build(stream).map_err(LayerwiseAcquireError::Architecture)?);
        if self.layout.address(index) != Some(address) {
            return Err(LayerwiseAcquireError::Policy(Error::Parallel(format!(
                "execution unit {index} does not match group {} unit {}",
                address.group(),
                address.index()
            ))));
        }
        self.reap_completed()
            .map_err(LayerwiseAcquireError::Policy)?;
        if self.dense.is_some() {
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
            if dense.groups[group].is_none() {
                dense.groups[group] = Some(dense.controller.group_guard(&self.residency, group_id));
            }
            if dense.windows[group].is_none() {
                dense.windows[group] = Some(
                    dense
                        .controller
                        .transfer_window(
                            &self.residency,
                            group_id,
                            &self.unit_ids,
                            range,
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
            if transfer.index() != index {
                return Err(LayerwiseAcquireError::Policy(Error::Parallel(format!(
                    "dense transfer returned unit {}, expected {index}",
                    transfer.index()
                ))));
            }
            let mut unit = MlxModule::new(unloaded.take().expect("unloaded unit is consumed once"));
            self.populator
                .populate(&mut unit, transfer.lease())
                .map_err(LayerwiseAcquireError::Policy)?;
            return Ok(MlxUnitLease {
                unit,
                _transfer: MlxUnitTransfer::Dense {
                    _transfer: transfer,
                },
            });
        }
        // An ordinary lookahead transfer owns every lease in its window until
        // the preceding unit's exact completion.  Release that completed
        // window before requesting the overlapping next window; otherwise an
        // overlapping in-flight unit would wait for the very transfer guard
        // retained by `pending` on this thread.
        self.drain_one().map_err(LayerwiseAcquireError::Policy)?;
        self.trim_device_window(index, address)
            .map_err(LayerwiseAcquireError::Policy)?;
        let group_range = self
            .layout
            .group_range(address.group())
            .expect("validated unit address has a group range");
        let end = index.saturating_add(self.window_depth).min(group_range.end);
        let requests = self.unit_ids[index..end]
            .iter()
            .cloned()
            .map(|id| (id, 1))
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
        let mut unit = MlxModule::new(unloaded.take().expect("unloaded unit is consumed once"));
        self.populator
            .populate(&mut unit, &transfer.leases()[0])
            .map_err(LayerwiseAcquireError::Policy)?;
        Ok(MlxUnitLease {
            unit,
            _transfer: MlxUnitTransfer::Ordinary {
                _transfer: transfer,
            },
        })
    }

    fn complete<'a, StateValues, ContextValues>(
        &mut self,
        _index: usize,
        _address: ExecutionUnitAddress,
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
        let event = async_eval_with_event(
            std::iter::once(output)
                .chain(state_values)
                .chain(context_values)
                .map(MlxTensor::as_array),
        )?;
        self.pending.push_back((event, lease));
        Ok(())
    }

    fn finish(&mut self, output: &MlxTensor, _stream: &Stream) -> Result<(), Self::Error> {
        async_eval_with_event([output.as_array()])?.synchronize()?;
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

impl<U, P> Drop for MlxLayerwisePolicy<U, P> {
    fn drop(&mut self) {
        for (event, _) in &self.pending {
            let _ = event.synchronize();
        }
    }
}
