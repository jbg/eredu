//! Fully resident execution policy and unit acquisition.

use super::*;

/// Permanently populated MLX units used by fully resident execution.
pub struct MlxResidentPolicy<U> {
    pub(super) units: Vec<Option<MlxModule<U>>>,
    pub(super) residency: ResidencyManager,
    pub(super) store: SharedCheckpointSource,
    pub(super) unit_ids: Vec<OffloadUnitId>,
    pub(super) layout: ExecutionUnitLayout,
    pub(super) window_depth: usize,
    pub(super) _transfer: ResidentTransfer,
}

/// Exclusive borrow-by-ownership of one permanently resident unit.
pub struct MlxResidentUnit<U> {
    index: usize,
    unit: MlxModule<U>,
}

impl<U> std::ops::Deref for MlxResidentUnit<U> {
    type Target = U;

    fn deref(&self) -> &Self::Target {
        &self.unit.inner
    }
}

impl<U> std::ops::DerefMut for MlxResidentUnit<U> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.unit.inner
    }
}

impl<U> MlxResidentPolicy<U> {
    /// Returns the checkpoint source backing this resident policy.
    pub fn checkpoint_store(&self) -> &dyn eredu_checkpoint::store::CheckpointSource {
        self.store.as_ref()
    }

    /// Clones the shared checkpoint source backing this resident policy.
    pub fn checkpoint_store_arc(&self) -> SharedCheckpointSource {
        Arc::clone(&self.store)
    }

    pub(crate) fn trim_parameter_conversions(
        &self,
    ) -> Result<
        Vec<eredu_core::residency::ParameterConversionRetentionTrimReport>,
        eredu_core::residency::ParameterConversionTrimError,
    > {
        self.residency.trim_parameter_conversions()
    }

    /// Returns current weight-residency accounting.
    pub fn residency_report(&self) -> Result<eredu_runtime::ResidencyReport, Error> {
        self.residency.report().map_err(Into::into)
    }

    /// Returns the number of pinned static-weight leases.
    pub fn static_lease_count(&self) -> usize {
        1
    }

    fn execution_group(&self, group: usize) -> Result<ResidentLayerGroup, Error> {
        let range = self
            .layout
            .group_range(group)
            .ok_or_else(|| Error::Parallel(format!("unknown execution group {group}")))?;
        let id = self
            .layout
            .group_id(group)
            .expect("validated layout names every execution group")
            .as_str();
        ResidentLayerGroup::new(
            id,
            self.unit_ids[range.clone()].iter().cloned(),
            self.window_depth.min(range.len()),
        )
        .map_err(|error| Error::Parallel(error.to_string()))
    }

    /// Returns residency reports for every semantic execution group.
    pub fn execution_group_reports(&self) -> Result<Vec<ResidentLayerGroupReport>, Error> {
        (0..self.layout.group_count())
            .map(|group| {
                self.execution_group(group)?
                    .report(&self.residency)
                    .map_err(Into::into)
            })
            .collect()
    }
}

impl<U: Parameterized<MlxTensor>> LayerwisePolicy<MlxNeuralBackend, U> for MlxResidentPolicy<U> {
    type Lease = MlxResidentUnit<U>;
    type Error = Error;

    fn publish_parameter_replacements(
        &mut self,
        values: &std::collections::BTreeMap<String, MlxTensor>,
        _active: bool,
    ) -> Result<bool, Error> {
        if !self.resident_parameters_available() {
            return Ok(false);
        }
        self.visit_resident_units(&mut |unit| {
            unit.visit_parameters_mut(&mut ParameterPublisher(values))
        });
        Ok(true)
    }

    fn resident_parameters_available(&self) -> bool {
        self.units.iter().all(Option::is_some)
    }

    fn visit_resident_units(&mut self, visitor: &mut impl FnMut(&mut U)) -> bool {
        if !self.resident_parameters_available() {
            return false;
        }
        for unit in &mut self.units {
            visitor(&mut unit.as_mut().expect("idle resident unit").inner);
        }
        true
    }

    fn begin(&mut self, _initial: &MlxTensor, _stream: &Stream) -> Result<(), Self::Error> {
        Ok(())
    }

    fn inspect_unit<E, F, V>(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        _build: F,
        operation: V,
        _stream: &Stream,
    ) -> Result<bool, LayerwiseAcquireError<E, Self::Error>>
    where
        F: FnOnce(&Stream) -> Result<U, E>,
        V: FnOnce(&mut U) -> Result<(), Self::Error>,
    {
        if self.layout.address(ordinal) != Some(address) {
            return Err(LayerwiseAcquireError::Policy(Error::Parallel(
                "parameter unit address differs from the selected layout".into(),
            )));
        }
        let unit = self
            .units
            .get_mut(ordinal)
            .and_then(Option::as_mut)
            .ok_or_else(|| {
                LayerwiseAcquireError::Policy(Error::Parallel("parameter unit is not idle".into()))
            })?;
        operation(&mut unit.inner).map_err(LayerwiseAcquireError::Policy)?;
        // Resident ownership remains in the outer session's completion scope.
        Ok(true)
    }

    fn abort(
        &mut self,
        active: Option<(usize, ExecutionUnitAddress, Self::Lease)>,
        _stream: &Stream,
    ) {
        let Some((ordinal, _, lease)) = active else {
            return;
        };
        debug_assert_eq!(lease.index, ordinal);
        if let Some(slot) = self.units.get_mut(lease.index) {
            debug_assert!(slot.is_none());
            if slot.is_none() {
                *slot = Some(lease.unit);
            }
        }
    }

    fn acquire<E, BF>(
        &mut self,
        index: usize,
        _address: eredu_runtime::ExecutionUnitAddress,
        _build: BF,
        _stream: &Stream,
    ) -> Result<Self::Lease, LayerwiseAcquireError<E, Self::Error>>
    where
        BF: FnOnce(&Stream) -> Result<U, E>,
    {
        let count = self.units.len();
        let unit = self
            .units
            .get_mut(index)
            .ok_or_else(|| {
                Error::Parallel(format!("resident unit {index} is outside {count} units"))
            })
            .map_err(LayerwiseAcquireError::Policy)?
            .take()
            .ok_or_else(|| Error::Parallel(format!("resident unit {index} is already acquired")))
            .map_err(LayerwiseAcquireError::Policy)?;
        Ok(MlxResidentUnit { index, unit })
    }

    fn complete<'a, StateValues, ContextValues>(
        &mut self,
        index: usize,
        _address: eredu_runtime::ExecutionUnitAddress,
        lease: Self::Lease,
        _output: &'a MlxTensor,
        _state_values: StateValues,
        _context_values: ContextValues,
        _stream: &Stream,
    ) -> Result<(), Self::Error>
    where
        MlxTensor: 'a,
        StateValues: Iterator<Item = &'a MlxTensor>,
        ContextValues: Iterator<Item = &'a MlxTensor>,
    {
        if lease.index != index {
            return Err(Error::Parallel(format!(
                "resident completion returned unit {} to slot {index}",
                lease.index
            )));
        }
        let slot = &mut self.units[index];
        if slot.replace(lease.unit).is_some() {
            return Err(Error::Parallel(format!(
                "resident unit slot {index} was unexpectedly occupied"
            )));
        }
        Ok(())
    }

    fn finish(&mut self, _output: &MlxTensor, _stream: &Stream) -> Result<(), Self::Error> {
        Ok(())
    }
}
