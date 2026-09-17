//! Device-side index discovery, gather, and scatter movement.

use super::*;
use eredu_nn::Tensor;

/// MLX integer discovery and device-side indexed movement mechanism.
#[derive(Default, Clone)]
pub struct MlxIndexedMovement {
    binding: Option<indexed_source::IndexedBankBinding>,
    original: Option<indexed_source::OriginalIndexedChunkSource>,
    invocation: Option<indexed_source::OriginalIndexedResidencyInvocation>,
}
impl std::fmt::Debug for MlxIndexedMovement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlxIndexedMovement").field("bound", &self.binding.is_some())
            .field("original", &self.original.is_some()).finish()
    }
}
#[path = "movement/indexed_source.rs"]
mod indexed_source;
pub(crate) use indexed_source::{IndexedChunkLayout, OriginalIndexedChunkSource, IndexedResidencyPlan, IndexedConstructorPartitions, OriginalIndexedResidencyInvocation, OriginalIndexedResidencyFactory, IndexedBankSource, IndexedBindingLayout, IndexedBindingStorage, IndexedBindingIdentity, IndexedRequestSource, IndexedRequestInstallation};
impl MlxIndexedMovement {
    /// The model binder retains this exact movement source, including its
    /// per-scope request channel. It never reconstructs a source from the pool.
    pub(crate) fn indexed_bank_source(&self)->Option<&IndexedBankSource>{self.binding.as_ref()}
    /// Retains the actual scoped cache and constructor-selected chunk policy.
    pub(crate) fn for_bank(bank: SharedAddressableParameterBank,
        options: eredu_runtime::ParameterBankLoadOptions) -> Self {
        Self { binding: Some(IndexedBankSource::new(bank,options)), original: None, invocation: None }
    }
}

#[path = "movement/invocation_callback.rs"]
mod invocation_callback;

impl IndexedMovement<MlxNeuralBackend> for MlxIndexedMovement {
    type Error = Error;

    fn with_invocation_source(
        callback:&mut dyn eredu_runtime::expert::IndexedInvocationCallback<MlxTensor,Self>,
        request:eredu_runtime::expert::IndexedInvocationRequest<'_,MlxTensor>,
        source:Option<eredu_nn::PreparedIndexedInvocationLoan<'_>>,stream:&Stream)
        ->Result<Result<eredu_nn::TensorParallelGroupedOutput<MlxTensor>,()>,
            eredu_runtime::expert::IndexedDemandLoanError<Self::Error>> {
        invocation_callback::run(callback,request,source,stream)
    }

    fn with_demand_loan<R,E,F>(&mut self,demands:&eredu_runtime::expert::IndexedDemandSource,run:F)
        ->Result<Result<R,E>,eredu_runtime::expert::IndexedDemandLoanError<Self::Error>>
    where F:FnOnce(Option<eredu_runtime::expert::PreparedIndexedDemandLoan<'_>>)->Result<R,E> {
        use eredu_runtime::expert::IndexedDemandLoanError as Failure;
        if let Some(source)=&self.original {return source.with_demand_loan(demands,run).map_err(Failure::Backend);}
        indexed_source::require_ordinary().map_err(Failure::Backend)?;
        if demands.funding().is_some(){return Err(Failure::MissingProducer);}
        Ok(run(None))
    }
    fn copy_route_value(&mut self,value:&MlxTensor,demands:&eredu_runtime::expert::IndexedDemandSource,
        stream:&Stream)->Result<MlxTensor,eredu_runtime::expert::IndexedDemandLoanError<Self::Error>> {
        use eredu_runtime::expert::IndexedDemandLoanError as Failure;
        if let Some(source)=&self.original {return source.copy_route_value(value,demands,stream).map_err(Failure::Backend);}
        indexed_source::require_ordinary().map_err(Failure::Backend)?;
        if demands.funding().is_some(){return Err(Failure::MissingProducer);}
        Ok(value.clone())
    }

    fn index_demand_source_for_chunk(&mut self, indices: &MlxTensor,
        census: eredu_runtime::expert::AddressableChunkCensus, stream: &Stream)
        -> Result<eredu_runtime::expert::IndexedDemandSource, Self::Error> {
        if let Some(invocation)=self.invocation.clone() {
            invocation.begin_chunk(self,indices,census,stream)?;
        }
        if let Some(source) = &self.original {
            return source.discover(self.binding.as_ref(), indices, census, stream);
        }
        indexed_source::require_ordinary()?;
        self.index_demands(indices, census.members(), stream)
            .map(eredu_runtime::expert::IndexedDemandSource::ordinary)
    }

    fn remap_demand_indices(&mut self, indices: &MlxTensor, mapping: &[(usize, usize)],
        source: &eredu_runtime::expert::IndexedDemandSource, stream: &Stream)
        -> Result<MlxTensor, Self::Error> {
        if let Some(original) = &self.original {
            return original.remap(self.binding.as_ref(), indices, mapping, source, stream);
        }
        indexed_source::require_ordinary()?;
        self.remap_indices(indices, mapping, stream)
    }

    fn index_demands(
        &mut self,
        indices: &MlxTensor,
        upper_bound: usize,
        stream: &Stream,
    ) -> Result<Vec<(usize, u64)>, Self::Error> {
        indexed_source::require_ordinary()?;
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
        let below = flat.lt(Array::try_from_int(upper)?, stream)?;
        let valid = if matches!(flat.dtype(), Dtype::Uint32 | Dtype::Uint64) {
            below
        } else {
            flat.ge(Array::try_from_int(0)?, stream)?
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
        indexed_source::require_ordinary()?;
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
        let lookup = Array::try_from_slice(&lookup, &[span as i32])?.copy(stream)?;
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
        if let Some(invocation)=&self.invocation {
            invocation.reserve_parent_adapter(self,stream,
                crate::tensor::narrow::control_bytes(value.shape().len()),
                std::mem::size_of::<(&MlxTensor,usize,usize,i32,i32,&Stream)>())?;
        } else { indexed_source::require_ordinary()?; }
        // The same semantic interval and Slice source used by the cold parent.
        value.narrow_axis(0,start,end,stream).map_err(Error::Neural)
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
        if let Some(invocation)=&self.invocation {
            invocation.reserve_parent_adapter(self,stream,
                safemlx::ops::concatenate_axis_control_bytes(),
                std::mem::size_of::<(&[MlxTensor],&Stream)>())?;
        } else { indexed_source::require_ordinary()?; }
        // ArrayValue accepts the borrowed tensor slice directly. No extra
        // pointer directory or cloned native handles are constructed here.
        MlxTensor::concatenate(values,0,stream).map_err(Error::Neural)
    }
}

impl AddressableGroupedBank<MlxNeuralBackend> for AddressableParameterBank {
    type Acquisition = AcquiredParameterGroups;
    type Report = ParameterBankResidencyReport;
    type Error = Error;

    fn member_bytes(&self, key: ParameterBankKey) -> Option<u64> {
        self.effective_member_bytes.get(&key).copied()
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
        indexed_source::require_ordinary()?;
        let started = Instant::now();
        let mut groups = MlxNeuralBackend::grouped_gated_product(spec.clone(), stream)?;
        let bindings = groups
            .local_parameter_names()
            .into_iter()
            .map(|name| {
                self.compact_parameter_binding(acquisition, &name, stream)
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
        indexed_source::require_ordinary()?;
        let started = Instant::now();
        let mut groups = MlxNeuralBackend::grouped_linear_bank(spec.clone(), stream)?;
        let bindings = groups
            .local_parameter_names()
            .into_iter()
            .map(|name| {
                self.compact_parameter_binding(acquisition, &name, stream)
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
        indexed_source::require_ordinary()?;
        let started = Instant::now();
        let mut groups = MlxNeuralBackend::grouped_relu2(spec.clone(), stream)?;
        let bindings = groups
            .local_parameter_names()
            .into_iter()
            .map(|name| {
                self.compact_parameter_binding(acquisition, &name, stream)
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
        stream: &Stream,
    ) -> Result<(), Self::Error> {
        if acquisition.original.is_some() {
            return Err(Error::PrefillControl(eredu_runtime::working_memory::WorkingMemoryError::IdentityMismatch));
        }
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

    fn acquire_from_demand(&mut self,request:ParameterBankAcquisition<'_>,
        demands:&eredu_runtime::expert::IndexedDemandSource,
        loan:Option<eredu_runtime::expert::PreparedIndexedDemandLoan<'_>>,stream:&Stream)
        ->Result<Self::Acquisition,eredu_runtime::expert::IndexedDemandLoanError<Self::Error>> {
        use eredu_runtime::expert::IndexedDemandLoanError as Failure;
        let Some(loan)=loan else {
            indexed_source::require_ordinary().map_err(Failure::Backend)?;
            if demands.funding().is_some(){return Err(Failure::MissingProducer);}
            return self.acquire(request,stream).map_err(Failure::Backend);
        };
        let source=loan.source::<OriginalIndexedChunkSource>().ok_or(Failure::MissingProducer)?;
        source.validate_acquisition_bank(self).map_err(Failure::Backend)?;
        source.acquire_from_demand(request,demands,loan.funding(),stream).map_err(Failure::Backend)
    }

    fn gated_product_groups(
        &mut self,
        acquisition: &Self::Acquisition,
        spec: &eredu_nn::GroupedGatedProductSpec,
        stream: &Stream,
    ) -> Result<<MlxNeuralBackend as GroupedNeuralBackend>::GatedProductGroups, Self::Error> {
        if let Some(source)=&acquisition.original {
            source.validate_acquisition_bank(self)?;
            return source.gated_product_groups(acquisition,spec,stream);
        }
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
        if let Some(source)=&acquisition.original {
            source.validate_acquisition_bank(self)?;
            return source.linear_groups(acquisition,spec,stream);
        }
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
        if let Some(source)=&acquisition.original {
            source.validate_acquisition_bank(self)?;
            return source.relu2_groups(acquisition,spec,stream);
        }
        self.inner
            .lock()
            .map_err(|_| {
                Error::ArchitectureModel("addressable parameter bank lock was poisoned".into())
            })?
            .relu2_groups(acquisition, spec, stream)
    }

    fn complete(
        &mut self,
        mut acquisition: Self::Acquisition,
        output: &MlxTensor,
        stream: &Stream,
    ) -> Result<(), Self::Error> {
        if let Some(source) = acquisition.original.take() {
            source.validate_acquisition_bank(self)?;
            return source.complete_acquisition(acquisition, output, stream);
        }
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
