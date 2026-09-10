//! Device-side index discovery, gather, and scatter movement.

use super::*;

/// MLX integer discovery and device-side indexed movement mechanism.
#[derive(Debug, Default, Clone, Copy)]
pub struct MlxIndexedMovement;

impl IndexedMovement<MlxNeuralBackend> for MlxIndexedMovement {
    type Error = Error;

    fn index_demands(
        &mut self,
        indices: &MlxTensor,
        upper_bound: usize,
        stream: &Stream,
    ) -> Result<Vec<(usize, u64)>, Self::Error> {
        if !matches!(
            indices.as_array().dtype(),
            Dtype::Int32 | Dtype::Uint32 | Dtype::Int64 | Dtype::Uint64
        ) {
            return Err(AddressableParameterBankError::InvalidSelectionDtype {
                actual: indices.as_array().dtype(),
            }
            .into());
        }
        let upper = i32::try_from(upper_bound).map_err(|_| {
            Error::ArchitectureModel("indexed movement upper bound exceeds MLX i32 indexing".into())
        })?;
        if upper == 0 {
            return Err(Error::ArchitectureModel(
                "indexed movement upper bound must be nonzero".into(),
            ));
        }
        let flat = indices.as_array().reshape(&[-1], stream)?;
        let below = flat.lt(Array::from_int(upper), stream)?;
        let valid = if matches!(flat.dtype(), Dtype::Uint32 | Dtype::Uint64) {
            below
        } else {
            flat.ge(Array::from_int(0), stream)?
                .logical_and(below, stream)?
        };
        let invalid =
            crate::backend::compaction::count_nonzero(&valid.logical_not(stream)?, stream)?;
        let flat_i32 = if flat.dtype() == Dtype::Int32 {
            flat
        } else {
            flat.as_dtype(Dtype::Int32, stream)?
        };
        let safe = r#where(
            &valid,
            flat_i32,
            Array::zeros::<i32>(&[indices.as_array().size() as i32], stream)?,
            stream,
        )?;
        let ones = Array::ones::<i32>(&[safe.size() as i32], stream)?;
        let histogram = segment_sum(&ones, &safe, upper, 0, stream)?;
        eval([&histogram, &invalid])?;
        let invalid_count = invalid.evaluated()?.as_slice::<i32>()[0];
        if invalid_count != 0 {
            return Err(AddressableParameterBankError::InvalidSelectionSet {
                namespace: 0,
                invalid_count: invalid_count as usize,
                global_span: upper_bound,
            }
            .into());
        }
        Ok(histogram
            .evaluated()?
            .as_slice::<i32>()
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, count)| *count != 0)
            .map(|(index, count)| (index, count as u64))
            .collect())
    }

    fn remap_indices(
        &mut self,
        indices: &MlxTensor,
        mapping: &[(usize, usize)],
        stream: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        let span = mapping
            .iter()
            .map(|(source, _)| source.saturating_add(1))
            .max()
            .ok_or_else(|| Error::ArchitectureModel("indexed remapping is empty".into()))?;
        let mut lookup = vec![-1i32; span];
        for &(source, destination) in mapping {
            let destination = i32::try_from(destination).map_err(|_| {
                Error::ArchitectureModel("compact index exceeds MLX i32 indexing".into())
            })?;
            if source >= span || lookup[source] != -1 {
                return Err(Error::ArchitectureModel(
                    "indexed remapping contains a duplicate source".into(),
                ));
            }
            lookup[source] = destination;
        }
        let lookup = Array::from_slice(&lookup, &[span as i32]).copy(stream)?;
        let normalized = if indices.as_array().dtype() == Dtype::Int32 {
            indices.as_array().clone()
        } else {
            indices.as_array().as_dtype(Dtype::Int32, stream)?
        };
        Ok(MlxTensor::from_array(lookup.take(&normalized, stream)?))
    }

    fn select_rows(
        &mut self,
        value: &MlxTensor,
        start: usize,
        end: usize,
        stream: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        let start = i32::try_from(start)
            .map_err(|_| Error::ArchitectureModel("row start exceeds MLX indexing".into()))?;
        let end = i32::try_from(end)
            .map_err(|_| Error::ArchitectureModel("row end exceeds MLX indexing".into()))?;
        Ok(MlxTensor::from_array(
            value.as_array().try_index_device(start..end, stream)?,
        ))
    }

    fn concatenate_rows(
        &mut self,
        values: &[MlxTensor],
        stream: &Stream,
    ) -> Result<MlxTensor, Self::Error> {
        if values.is_empty() {
            return Err(Error::ArchitectureModel(
                "row concatenation requires at least one partition".into(),
            ));
        }
        let values = values.iter().map(MlxTensor::as_array).collect::<Vec<_>>();
        Ok(MlxTensor::from_array(concatenate_axis(&values, 0, stream)?))
    }
}

impl AddressableGroupedBank<MlxNeuralBackend> for AddressableParameterBank {
    type Acquisition = AcquiredParameterGroups;
    type Report = ParameterBankResidencyReport;
    type Error = Error;

    fn member_bytes(&self, key: ParameterBankKey) -> Option<u64> {
        self.catalog.get(&key).copied()
    }

    fn acquire(
        &mut self,
        request: ParameterBankAcquisition<'_>,
        stream: &Stream,
    ) -> Result<Self::Acquisition, Self::Error> {
        let entries = request
            .entries()
            .iter()
            .map(|(key, count)| (*key, *count))
            .collect::<Vec<_>>();
        let pass = match request.access() {
            ParameterBankAccess::Bulk => BankAccessClass::Bulk,
            ParameterBankAccess::Incremental => BankAccessClass::Incremental,
            _ => {
                return Err(Error::ArchitectureModel(
                    "unsupported addressable storage access class".into(),
                ))
            }
        };
        self.acquire_entry_demand(&entries, pass, stream)
            .map_err(Into::into)
    }

    fn gated_product_groups(
        &mut self,
        acquisition: &Self::Acquisition,
        spec: &eredu_nn::GroupedGatedProductSpec,
        stream: &Stream,
    ) -> Result<<MlxNeuralBackend as GroupedNeuralBackend>::GatedProductGroups, Self::Error> {
        let started = Instant::now();
        let mut groups = MlxNeuralBackend::grouped_gated_product(spec.clone(), stream)?;
        let bindings = groups
            .local_parameter_names()
            .into_iter()
            .map(|name| {
                acquisition
                    .compact_binding(&name, stream)
                    .map(|value| (name, value))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        groups.bind_local_parameters(bindings)?;
        self.record_compact_bank(
            acquisition.identities()[0].bank(),
            acquisition.pass(),
            acquisition.scratch_bytes(),
            started.elapsed(),
        )?;
        Ok(groups)
    }

    /// Constructs one compact selected-linear bank from acquired bindings.
    fn linear_groups(
        &mut self,
        acquisition: &Self::Acquisition,
        spec: &eredu_nn::GroupedLinearSpec,
        stream: &Stream,
    ) -> Result<<MlxNeuralBackend as GroupedNeuralBackend>::LinearGroups, Self::Error> {
        let started = Instant::now();
        let mut groups = MlxNeuralBackend::grouped_linear_bank(spec.clone(), stream)?;
        let bindings = groups
            .local_parameter_names()
            .into_iter()
            .map(|name| {
                acquisition
                    .compact_binding(&name, stream)
                    .map(|value| (name, value))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        groups.bind_local_parameters(bindings)?;
        self.record_compact_bank(
            acquisition.identities()[0].bank(),
            acquisition.pass(),
            acquisition.scratch_bytes(),
            started.elapsed(),
        )?;
        Ok(groups)
    }

    fn relu2_groups(
        &mut self,
        acquisition: &Self::Acquisition,
        spec: &eredu_nn::GroupedRelu2Spec,
        stream: &Stream,
    ) -> Result<<MlxNeuralBackend as GroupedNeuralBackend>::Relu2Groups, Self::Error> {
        let started = Instant::now();
        let mut groups = MlxNeuralBackend::grouped_relu2(spec.clone(), stream)?;
        let bindings = groups
            .local_parameter_names()
            .into_iter()
            .map(|name| {
                acquisition
                    .compact_binding(&name, stream)
                    .map(|value| (name, value))
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        groups.bind_local_parameters(bindings)?;
        self.record_compact_bank(
            acquisition.identities()[0].bank(),
            acquisition.pass(),
            acquisition.scratch_bytes(),
            started.elapsed(),
        )?;
        Ok(groups)
    }

    fn complete(
        &mut self,
        mut acquisition: Self::Acquisition,
        output: &MlxTensor,
        _: &Stream,
    ) -> Result<(), Self::Error> {
        eval([output.as_array()])?;
        acquisition.transfer.synchronize()?;
        Ok(())
    }

    fn report(&self) -> Result<Self::Report, Self::Error> {
        AddressableParameterBank::report(self).map_err(Into::into)
    }
}

impl AddressableGroupedBank<MlxNeuralBackend> for SharedAddressableParameterBank {
    type Acquisition = AcquiredParameterGroups;
    type Report = ParameterBankResidencyReport;
    type Error = Error;

    fn member_bytes(&self, key: ParameterBankKey) -> Option<u64> {
        if self.scope.is_some_and(|bank| bank != key.bank()) {
            return None;
        }
        self.inner.lock().ok()?.member_bytes(key)
    }

    fn acquire(
        &mut self,
        request: ParameterBankAcquisition<'_>,
        stream: &Stream,
    ) -> Result<Self::Acquisition, Self::Error> {
        if request
            .entries()
            .iter()
            .any(|(key, _)| self.scope.is_some_and(|bank| bank != key.bank()))
        {
            return Err(Error::ArchitectureModel(
                "acquisition exceeds selected bank scope".into(),
            ));
        }
        self.inner
            .lock()
            .map_err(|_| {
                Error::ArchitectureModel("addressable parameter bank lock was poisoned".into())
            })?
            .acquire(request, stream)
    }

    fn gated_product_groups(
        &mut self,
        acquisition: &Self::Acquisition,
        spec: &eredu_nn::GroupedGatedProductSpec,
        stream: &Stream,
    ) -> Result<<MlxNeuralBackend as GroupedNeuralBackend>::GatedProductGroups, Self::Error> {
        self.inner
            .lock()
            .map_err(|_| {
                Error::ArchitectureModel("addressable parameter bank lock was poisoned".into())
            })?
            .gated_product_groups(acquisition, spec, stream)
    }

    /// Constructs one compact selected-linear bank from acquired bindings.
    fn linear_groups(
        &mut self,
        acquisition: &Self::Acquisition,
        spec: &eredu_nn::GroupedLinearSpec,
        stream: &Stream,
    ) -> Result<<MlxNeuralBackend as GroupedNeuralBackend>::LinearGroups, Self::Error> {
        self.inner
            .lock()
            .map_err(|_| {
                Error::ArchitectureModel("addressable parameter bank lock was poisoned".into())
            })?
            .linear_groups(acquisition, spec, stream)
    }

    fn relu2_groups(
        &mut self,
        acquisition: &Self::Acquisition,
        spec: &eredu_nn::GroupedRelu2Spec,
        stream: &Stream,
    ) -> Result<<MlxNeuralBackend as GroupedNeuralBackend>::Relu2Groups, Self::Error> {
        self.inner
            .lock()
            .map_err(|_| {
                Error::ArchitectureModel("addressable parameter bank lock was poisoned".into())
            })?
            .relu2_groups(acquisition, spec, stream)
    }

    fn complete(
        &mut self,
        acquisition: Self::Acquisition,
        output: &MlxTensor,
        stream: &Stream,
    ) -> Result<(), Self::Error> {
        self.inner
            .lock()
            .map_err(|_| {
                Error::ArchitectureModel("addressable parameter bank lock was poisoned".into())
            })?
            .complete(acquisition, output, stream)
    }

    fn report(&self) -> Result<Self::Report, Self::Error> {
        SharedAddressableParameterBank::report(self)
    }
}
