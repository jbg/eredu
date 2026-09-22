//! Ordinary compact construction uses the same physical row binder as native scopes.
use super::super::super::source::AddressableSourceInspection;
use super::*;
use crate::backend::nn::shared::PreparedCompactBindings;
use eredu_nn::workspace::ParameterMetadataAllocation;
use eredu_nn::{GroupedGatedProductSpec, GroupedLinearSpec, GroupedRelu2Spec, Tensor};

trait CompactConsumer {
    type Output;
    fn metadata_bytes(&self) -> Result<Option<usize>, Error>;
    fn construct(
        self,
        funding: &HostMetadataFunding,
        bindings: PreparedCompactBindings<'_>,
    ) -> Result<Self::Output, Error>;
}
struct GatedConsumer<'a>(&'a GroupedGatedProductSpec);
struct LinearConsumer<'a>(&'a GroupedLinearSpec);
struct Relu2Consumer<'a>(&'a GroupedRelu2Spec);
macro_rules! consumer {
    ($consumer:ident, $spec:ty, $output:ident, $clone_bytes:ident, $clone:ident, $bytes:ident, $construct:ident) => {
        impl CompactConsumer for $consumer<'_> {
            type Output = <MlxNeuralBackend as GroupedNeuralBackend>::$output;
            fn metadata_bytes(&self) -> Result<Option<usize>, Error> {
                Ok(HostMetadataFunding::$clone_bytes(self.0)
                    .and_then(|bytes| bytes.checked_add(MlxNeuralBackend::$bytes(self.0)?)))
            }
            fn construct(
                self,
                funding: &HostMetadataFunding,
                bindings: PreparedCompactBindings<'_>,
            ) -> Result<Self::Output, Error> {
                MlxNeuralBackend::$construct(
                    funding.$clone(self.0).map_err(Error::Neural)?,
                    bindings,
                )
                .map_err(Error::Neural)
            }
        }
    };
}
consumer!(
    LinearConsumer,
    GroupedLinearSpec,
    LinearGroups,
    grouped_linear_clone_bytes,
    clone_grouped_linear,
    grouped_linear_construction_bytes,
    grouped_linear_from_bindings
);
consumer!(
    Relu2Consumer,
    GroupedRelu2Spec,
    Relu2Groups,
    grouped_relu2_clone_bytes,
    clone_grouped_relu2,
    grouped_relu2_construction_bytes,
    grouped_relu2_from_bindings
);
impl CompactConsumer for GatedConsumer<'_> {
    type Output = <MlxNeuralBackend as GroupedNeuralBackend>::GatedProductGroups;
    fn metadata_bytes(&self) -> Result<Option<usize>, Error> {
        let constructor = MlxNeuralBackend::grouped_gated_product_construction_bytes(self.0)
            .map_err(Error::Neural)?;
        Ok(
            HostMetadataFunding::grouped_gated_product_clone_bytes(self.0)
                .and_then(|bytes| bytes.checked_add(constructor?)),
        )
    }
    fn construct(
        self,
        funding: &HostMetadataFunding,
        bindings: PreparedCompactBindings<'_>,
    ) -> Result<Self::Output, Error> {
        MlxNeuralBackend::grouped_gated_product_from_bindings(
            funding
                .clone_grouped_gated_product(self.0)
                .map_err(Error::Neural)?,
            bindings,
        )
        .map_err(Error::Neural)
    }
}

struct CompactRowsInspection<'a, 'names> {
    owner: &'a OrdinaryBankHostSource,
    acquisition: &'a AcquiredParameterGroups,
    names: &'names [&'names str],
    stream: &'a Stream,
    bindings: &'a mut PreparedCompactBindings<'names>,
    rows: &'a mut Vec<Array>,
}
impl AddressableSourceInspection for CompactRowsInspection<'_, '_> {
    type Value = ();
    type Error = Error;
    fn inspect(self, source: AddressableBankSourceLoan<'_>) -> Result<(), Error> {
        if source.parameter_revision() != self.owner.revision
            || !source.same_source(self.owner.binding.storage())
        {
            return Err(self.owner.failure(Cause::Identity));
        }
        super::super::super::parameters::fill_compact_rows(
            &source,
            self.acquisition,
            self.names,
            self.stream,
            &self.owner.funding,
            self.bindings,
            self.rows,
        )
        .map_err(|cause| self.owner.failure(Cause::Compact(cause)))
    }
}

fn names_control_bytes<const N: usize>() -> Option<usize> {
    let frames = [
        size_of::<[Option<&str>; N]>(),
        size_of::<[&str; N]>(),
        size_of::<Result<([&str; N], usize), Error>>(),
        size_of::<(usize, usize, &str)>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}

impl OrdinaryBankHostSource {
    /// Host storage of this exact compact constructor. Numerical allocations,
    /// discovery, acquisition and transfer have their own source quotations.
    pub(crate) fn gated_product_metadata_bytes(
        &self,
        spec: &GroupedGatedProductSpec,
    ) -> Result<usize, Error> {
        self.compact_metadata_bytes(
            spec.group_count(),
            MlxNeuralBackend::gated_compact_parameter_names(spec).map_err(Error::Neural)?,
            GatedConsumer(spec),
        )
    }
    pub(crate) fn linear_metadata_bytes(&self, spec: &GroupedLinearSpec) -> Result<usize, Error> {
        self.compact_metadata_bytes(
            spec.group_count(),
            MlxNeuralBackend::linear_compact_parameter_names(spec),
            LinearConsumer(spec),
        )
    }
    pub(crate) fn relu2_metadata_bytes(&self, spec: &GroupedRelu2Spec) -> Result<usize, Error> {
        self.compact_metadata_bytes(
            spec.group_count(),
            MlxNeuralBackend::relu2_compact_parameter_names(spec),
            Relu2Consumer(spec),
        )
    }
    fn compact_metadata_bytes<const N: usize, C: CompactConsumer>(
        &self,
        groups: i32,
        fields: [Option<&str>; N],
        consumer: C,
    ) -> Result<usize, Error> {
        let count = usize::try_from(groups)
            .ok()
            .filter(|count| *count != 0 && *count <= self.census.maximum_members())
            .ok_or_else(|| self.failure(Cause::Identity))?;
        let (names, names_count) = self.names(fields)?;
        let names = &names[..names_count];
        let components = [
            names_control_bytes::<N>(),
            compact_transport_control_bytes::<C::Output, C>(count, names_count),
            vector_bytes::<Array>(count),
            PreparedCompactBindings::layout_bytes(names_count),
            SharedAddressableParameterBank::workspace_source_control_bytes::<
                CompactRowsInspection<'_, '_>,
            >(),
            super::super::super::parameters::compact_rows_control_bytes(),
            consumer.metadata_bytes()?,
        ];
        let fixed = components
            .into_iter()
            .try_fold(0usize, |bytes, part| bytes.checked_add(part?))
            .ok_or_else(|| self.failure(Cause::Overflow))?;
        self.binding
            .with_workspace_source(&self.funding, |source| {
                if source.parameter_revision() != self.revision
                    || !source.same_source(self.binding.storage())
                {
                    return Err(self.failure(Cause::Identity));
                }
                // Every candidate member must supply exactly these physical fields.
                // The numerical visitor selects any `count` members from this source.
                for (key, _) in source.unit_members(self.census.bank(), self.census.unit()) {
                    let mut found = 0usize;
                    for member in source.members().filter(|member| member.key == key) {
                        if names.binary_search(&member.binding.as_str()).is_err() {
                            return Err(self.failure(Cause::Identity));
                        }
                        found = found
                            .checked_add(1)
                            .ok_or_else(|| self.failure(Cause::Overflow))?;
                    }
                    if found != names_count {
                        return Err(self.failure(Cause::Identity));
                    }
                }
                let mut bytes = fixed;
                for &name in names {
                    let mut replacements = 0usize;
                    let mut largest = 0usize;
                    for member in source.members().filter(|member| {
                        member.key.bank() == self.census.bank()
                            && member.key.unit() == self.census.unit()
                            && member.binding == name
                    }) {
                        if let Some((replacement, _)) = source.replacement(member) {
                            replacements = replacements
                                .checked_add(1)
                                .ok_or_else(|| self.failure(Cause::Overflow))?;
                            largest = largest.max(
                                crate::tensor::narrow::control_bytes(replacement.shape().len())
                                    .ok_or_else(|| self.failure(Cause::Overflow))?,
                            );
                        }
                    }
                    // A finite candidate-union allowance retains the largest actual
                    // narrow control for each possibly selected replacement field.
                    bytes = largest
                        .checked_mul(replacements.min(count))
                        .and_then(|n| bytes.checked_add(n))
                        .ok_or_else(|| self.failure(Cause::Overflow))?;
                }
                Ok(bytes)
            })
            .map_err(|cause| self.failure(Cause::Source(cause)))?
    }
    fn names<'a, const N: usize>(
        &self,
        values: [Option<&'a str>; N],
    ) -> Result<([&'a str; N], usize), Error> {
        let bytes = names_control_bytes::<N>().ok_or_else(|| self.failure(Cause::Overflow))?;
        self.funding
            .reserve_metadata(bytes)
            .map_err(|cause| self.failure(Cause::Funding(cause)))?;
        let mut names = [""; N];
        let mut count = 0;
        for name in values.into_iter().flatten() {
            if name.is_empty() {
                return Err(self.failure(Cause::Identity));
            }
            names[count] = name;
            count += 1;
        }
        names[..count].sort_unstable();
        if count == 0 || names[..count].windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(self.failure(Cause::Identity));
        }
        Ok((names, count))
    }

    fn validate_groups(
        &self,
        acquisition: &AcquiredParameterGroups,
        groups: i32,
    ) -> Result<(), Error> {
        if usize::try_from(groups).ok() != Some(acquisition.identities.len()) {
            return Err(self.failure(Cause::Identity));
        }
        Ok(())
    }

    pub(crate) fn gated_product_groups(
        &self,
        acquisition: &AcquiredParameterGroups,
        spec: &GroupedGatedProductSpec,
        stream: &Stream,
    ) -> Result<<MlxNeuralBackend as GroupedNeuralBackend>::GatedProductGroups, Error> {
        self.validate_groups(acquisition, spec.group_count())?;
        let (names, count) = self
            .names(MlxNeuralBackend::gated_compact_parameter_names(spec).map_err(Error::Neural)?)?;
        self.with_compact_bindings(acquisition, &names[..count], stream, GatedConsumer(spec))
    }
    pub(crate) fn linear_groups(
        &self,
        acquisition: &AcquiredParameterGroups,
        spec: &GroupedLinearSpec,
        stream: &Stream,
    ) -> Result<<MlxNeuralBackend as GroupedNeuralBackend>::LinearGroups, Error> {
        self.validate_groups(acquisition, spec.group_count())?;
        let (names, count) = self.names(MlxNeuralBackend::linear_compact_parameter_names(spec))?;
        self.with_compact_bindings(acquisition, &names[..count], stream, LinearConsumer(spec))
    }
    pub(crate) fn relu2_groups(
        &self,
        acquisition: &AcquiredParameterGroups,
        spec: &GroupedRelu2Spec,
        stream: &Stream,
    ) -> Result<<MlxNeuralBackend as GroupedNeuralBackend>::Relu2Groups, Error> {
        self.validate_groups(acquisition, spec.group_count())?;
        let (names, count) = self.names(MlxNeuralBackend::relu2_compact_parameter_names(spec))?;
        self.with_compact_bindings(acquisition, &names[..count], stream, Relu2Consumer(spec))
    }

    fn with_compact_bindings<'names, C: CompactConsumer>(
        &self,
        acquisition: &AcquiredParameterGroups,
        names: &'names [&'names str],
        stream: &Stream,
        consumer: C,
    ) -> Result<C::Output, Error> {
        if acquisition.original.is_some()
            || !acquisition.ordinary.as_ref().is_some_and(|source| {
                source.binding.same_binding(&self.binding)
                    && source.revision == self.revision
                    && source.census == self.census
                    && source.funding.same_account(&self.funding)
            })
        {
            return Err(self.failure(Cause::Identity));
        }
        let count = acquisition.identities.len();
        let bytes = compact_transport_control_bytes::<C::Output, C>(count, names.len())
            .ok_or_else(|| self.failure(Cause::Overflow))?;
        self.funding
            .reserve_metadata(bytes)
            .map_err(|cause| self.failure(Cause::Funding(cause)))?;
        let started = Instant::now();
        let mut bindings =
            PreparedCompactBindings::new(names.len(), &self.funding).map_err(Error::Neural)?;
        let mut rows = vector(count, Some(&self.funding))?;
        self.binding
            .storage()
            .with_workspace_inspection(
                &self.funding,
                CompactRowsInspection {
                    owner: self,
                    acquisition,
                    names,
                    stream,
                    bindings: &mut bindings,
                    rows: &mut rows,
                },
            )
            .map_err(|cause| self.failure(Cause::Source(cause)))??;
        drop(rows);
        let result = consumer
            .construct(&self.funding, bindings)
            .map_err(|cause| self.failure(Cause::Consumer(cause)))?;
        let bank = self
            .binding
            .storage()
            .inner
            .lock()
            .map_err(|_| self.failure(Cause::Identity))?;
        bank.record_compact_bank(
            self.census.bank(),
            acquisition.pass(),
            acquisition.scratch_bytes(),
            started.elapsed(),
        )?;
        Ok(result)
    }
}

/// Actual compact row-transport frames and native handle constructor controls.
/// Row vectors, source loans, spec clones and the typed module constructor are
/// separate producers and must be composed by the enclosing source program.
pub(crate) fn compact_transport_control_bytes<T, F>(count: usize, names: usize) -> Option<usize> {
    let operands = count.checked_mul(names)?;
    let clone = safemlx::PreparedArrayClone::control_bytes()
        .and_then(|bytes| bytes.checked_add(Array::inspection_clone_handle_bytes()))?;
    let frames = [
        size_of::<T>(),
        size_of::<F>(),
        size_of::<Result<T, Error>>(),
        size_of::<Result<PreparedCompactBindings<'_>, Error>>(),
        size_of::<(
            &OrdinaryBankHostSource,
            &AcquiredParameterGroups,
            &[&str],
            &Stream,
        )>(),
        size_of::<(Instant, Duration)>(),
        size_of::<Vec<Array>>(),
        size_of::<std::sync::MutexGuard<'_, AddressableParameterBank>>(),
        size_of::<std::sync::MutexGuard<'_, ParameterBankStatisticsTable>>(),
        size_of::<(
            &ParameterBankStatisticsTable,
            &mut ParameterBankStatistics,
            usize,
            u64,
        )>(),
        size_of::<Result<(), AddressableParameterBankError>>(),
        clone.checked_mul(operands)?,
        safemlx::ops::concatenate_axis_control_bytes()
            .and_then(|bytes| bytes.checked_mul(names))?,
        Array::descriptor_comparison_control_bytes()
            .and_then(|bytes| bytes.checked_mul(operands))?,
        eredu_nn::Error::retained_source_construction_bytes::<Failure>()?,
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
