//! Portable ownership of an installed row range.

use std::{borrow::Borrow, collections::BTreeMap};

use eredu_core::{
    checkpoint_mechanisms::CheckpointParameterBackend,
    residency::{
        parameter_binding::{preflight_bindings, WeightBinding},
        ParameterLoan, ResidencyTransferOrdering,
    },
    BackendFailure,
};
use eredu_nn::{DistributedNeuralBackend, NeuralBackend, ParameterId, Tensor};

use crate::{
    dense::DenseResidencyMechanisms,
    expert::{AddressableParameterBank, BankTransfer, RowLookupPoolReport},
    residency::{ResidencyAccess, ResidencyDriver},
    ResidencyReport, RowResidencyRange,
};

use super::{
    BoundedRowLookup, ParameterBank, ParameterBankAccess, ParameterBankKey, PreparedRowLookup,
    RowLookupBank, RowLookupDescriptor, RowLookupError, RowLookupMechanismSupport,
    RowLookupProvider, RowLookupProviders, RowLookupSpec, SelectedRowLookupRequirements,
    SelectedRowLookups, TensorParallelRowLookup,
};

pub use eredu_core::preparation::rows::RowDecode;

/// Installed portable row residency and its native decoder.
pub struct BoundRowBank<N: DenseResidencyMechanisms, D> {
    residency: ResidencyDriver<N>,
    range: RowResidencyRange,
    descriptor: RowLookupDescriptor,
    decode_limit: u64,
    decoder: D,
}

/// Validates installed source identity, acquisition limits and workspace before scalar reads.
pub fn preflight_row_range<N: DenseResidencyMechanisms>(
    residency: &ResidencyDriver<N>,
    prepared: &PreparedRowLookup,
    decode_limit: u64,
    support: &impl RowLookupMechanismSupport,
) -> Result<RowResidencyRange, RowLookupError> {
    let parameter = &prepared.spec().parameter;
    let range = residency
        .row_range(prepared.range().range().prefix())
        .ok_or_else(|| RowLookupError::Missing(parameter.clone()))?;
    if !range.same_binding(prepared.range()) {
        return Err(RowLookupError::Specification(parameter.clone()));
    }
    let workspace = support
        .workspace(prepared.descriptor())?
        .ok_or_else(|| RowLookupError::Specification(parameter.clone()))?;
    let required = prepared.maximum_acquisition_rows() * range.range().bytes();
    if let Some(limit) = residency
        .offload_config()
        .map_err(RowLookupError::bank)?
        .device_budget_bytes()
    {
        if required > limit {
            return Err(RowLookupError::Budget {
                resource: "row acquisition residency",
                required,
                limit,
            });
        }
    }
    if workspace.decode_bytes > decode_limit {
        return Err(RowLookupError::Budget {
            resource: "native row decode scratch",
            required: workspace.decode_bytes,
            limit: decode_limit,
        });
    }
    Ok(range.clone())
}

impl<N: DenseResidencyMechanisms, D> BoundRowBank<N, D> {
    /// Installs the preflighted range, its selected limits, and the native decoder.
    pub fn from_preflight(
        residency: ResidencyDriver<N>,
        descriptor: RowLookupDescriptor,
        range: RowResidencyRange,
        decode_limit: u64,
        decoder: D,
    ) -> Self {
        Self {
            residency,
            range,
            descriptor,
            decode_limit,
            decoder,
        }
    }
}

impl<B, N, D> ParameterBank<B> for BoundRowBank<N, D>
where
    B: NeuralBackend,
    N: DenseResidencyMechanisms,
    D: RowDecode<B, N>,
    N::Transfer: ResidencyTransferOrdering<<B::Tensor as Tensor>::Context, Error = N::Error>,
{
    type Residency = ResidencyDriver<N>;
    type Acquisition = BankTransfer<D::Loan>;
    type Report = ResidencyReport;
    type Error = D::Error;

    fn member_bytes(&self, key: ParameterBankKey) -> Option<u64> {
        self.descriptor.member_bytes(key)
    }

    fn begin_consumption(
        &self,
        acquisition: BankTransfer<N::Transfer>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Acquisition, D::Error> {
        acquisition.begin_loan(|transfer| D::begin_loan(transfer, context))
    }

    fn complete(
        &mut self,
        acquisition: Self::Acquisition,
        output: &B::Tensor,
    ) -> Result<(), D::Error> {
        acquisition.transfer.finish(output)
    }

    fn report(&self) -> Result<ResidencyReport, D::Error> {
        self.residency.report().map_err(Into::into)
    }
}

impl<B, N, D> RowLookupBank<B> for BoundRowBank<N, D>
where
    B: NeuralBackend,
    N: DenseResidencyMechanisms,
    D: RowDecode<B, N>,
    N::Transfer: ResidencyTransferOrdering<<B::Tensor as Tensor>::Context, Error = N::Error>,
{
    fn descriptor(&self) -> &RowLookupDescriptor {
        &self.descriptor
    }

    fn residency(&self) -> &ResidencyDriver<N> {
        &self.residency
    }

    fn decode_scratch_bytes(&self, rows: u64) -> Result<u64, D::Error> {
        self.decoder.workspace(
            rows,
            self.range.range().bytes(),
            self.descriptor.spec().decode_geometry(),
        )
    }

    fn decode_scratch_limit(&self) -> u64 {
        self.decode_limit
    }

    fn rows(
        &mut self,
        acquisition: Self::Acquisition,
        spec: &RowLookupSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, D::Error> {
        if spec != self.descriptor.spec() {
            return Err(BackendFailure::from_error(RowLookupError::Specification(
                spec.parameter.clone(),
            ))
            .into());
        }
        let mut loan = acquisition.transfer;
        let rows = self.decoder.decode(
            &mut loan,
            self.range.binding(),
            spec.decode_geometry(),
            context,
        )?;
        loan.finish(&rows)?;
        Ok(rows)
    }
}

pub use eredu_core::preparation::rows::RowDecodeBinding;

type LocalRowProviders<N, D> = RowLookupProviders<BoundedRowLookup<BoundRowBank<N, D>>>;
type ParallelRowProviders<B, N, D, G> =
    RowLookupProviders<TensorParallelRowLookup<B, BoundedRowLookup<BoundRowBank<N, D>>, G>>;

enum RowProviders<B: DistributedNeuralBackend, N: DenseResidencyMechanisms, D, G> {
    Local(LocalRowProviders<N, D>),
    Tensor(ParallelRowProviders<B, N, D, G>),
}

struct RowPool<N: DenseResidencyMechanisms> {
    id: u64,
    residency: ResidencyDriver<N>,
}

/// Selected row lookup providers and the one residency pool shared with grouped banks.
pub struct BoundRowLookups<B: DistributedNeuralBackend, N: DenseResidencyMechanisms, D, G> {
    providers: RowProviders<B, N, D, G>,
    pool: Option<RowPool<N>>,
    requirements: SelectedRowLookupRequirements,
}

impl<B, N, D, G> BoundRowLookups<B, N, D, G>
where
    B: DistributedNeuralBackend + CheckpointParameterBackend,
    N: DenseResidencyMechanisms + Clone,
    D: RowDecode<B, N>,
    G: Clone + Borrow<B::ParallelContext>,
    N::Transfer: ResidencyTransferOrdering<<B::Tensor as Tensor>::Context, Error = N::Error>,
{
    /// Binds the exact local row sources after all range and scalar preflights pass.
    pub fn bind_local(
        pool: &AddressableParameterBank<N, B::Tensor>,
        selected: &SelectedRowLookups,
        binding: &impl RowDecodeBinding<B, N, D>,
    ) -> Result<Self, D::Error> {
        let (pool, banks) = Self::bind_banks(pool, selected, binding)?;
        Ok(Self {
            providers: RowProviders::Local(
                selected
                    .prepared()
                    .bind(banks)
                    .map_err(BackendFailure::from_error)?,
            ),
            pool: Some(pool),
            requirements: selected.requirements(),
        })
    }

    /// Binds selected scopes. Only the elected owner can acquire table rows or materialize scalars.
    pub fn bind_tensor_parallel(
        pool: Option<&AddressableParameterBank<N, B::Tensor>>,
        selected: &SelectedRowLookups,
        parallel: G,
        status_parallel: G,
        rank: usize,
        owner: usize,
        binding: &impl RowDecodeBinding<B, N, D>,
    ) -> Result<Self, D::Error> {
        let partitions = B::parallel_size(parallel.borrow());
        let status_partitions = B::parallel_size(status_parallel.borrow());
        if status_partitions == 0
            || status_partitions > i32::MAX as usize
            || B::parallel_rank(status_parallel.borrow()) >= status_partitions
            || rank != B::parallel_rank(parallel.borrow())
            || rank >= partitions
            || owner >= partitions
            || pool.is_some() != (rank == owner)
        {
            return Err(BackendFailure::from_error(RowLookupError::Geometry).into());
        }
        let (pool, banks) = if let Some(pool) = pool {
            let (pool, banks) = Self::bind_banks(pool, selected, binding)?;
            (Some(pool), banks)
        } else {
            selected
                .validate_binding(selected.options().offload(), &binding.support())
                .map_err(BackendFailure::from_error)?;
            (None, BTreeMap::new())
        };
        let providers = selected
            .prepared()
            .bind_tensor_parallel::<B, _, _>(banks, parallel, status_parallel, rank, owner)
            .map_err(BackendFailure::from_error)?;
        Ok(Self {
            providers: RowProviders::Tensor(providers),
            pool,
            requirements: selected.requirements(),
        })
    }

    fn bind_banks(
        pool: &AddressableParameterBank<N, B::Tensor>,
        selected: &SelectedRowLookups,
        binding: &impl RowDecodeBinding<B, N, D>,
    ) -> Result<(RowPool<N>, BTreeMap<ParameterId, BoundRowBank<N, D>>), D::Error> {
        let residency = pool.residency_manager().clone();
        let support = binding.support();
        selected
            .validate_binding(residency.offload_config()?, &support)
            .map_err(BackendFailure::from_error)?;
        let entries = selected.prepared().entries();
        let mut preflighted = Vec::with_capacity(entries.len());
        for (id, entry) in entries {
            let decode_limit = selected
                .workspace(id)
                .expect("selected exact row set")
                .decode_bytes;
            let range = preflight_row_range(&residency, entry, decode_limit, &support)
                .map_err(BackendFailure::from_error)?;
            let scalar = entry
                .scale()
                .map(|scale| {
                    let metadata = scale
                        .recipe
                        .infer(scale.source.as_ref())
                        .map_err(RowLookupError::bank)?;
                    let binding = WeightBinding::from_recipe(
                        scale.parameter.as_str(),
                        scale.recipe.clone(),
                        metadata.byte_len,
                    )
                    .map_err(RowLookupError::bank)?;
                    preflight_bindings::<B>(scale.source.as_ref(), std::slice::from_ref(&binding))
                        .map_err(RowLookupError::bank)?;
                    Ok::<_, RowLookupError>((scale.source.as_ref(), binding, metadata.dtype))
                })
                .transpose()
                .map_err(BackendFailure::from_error)?;
            preflighted.push((id, entry, range, decode_limit, scalar));
        }

        let mut banks = BTreeMap::new();
        for (id, entry, range, decode_limit, scalar) in preflighted {
            let scale = scalar
                .map(|(source, exact, dtype)| {
                    let mut values = crate::materialize_parameter_owners(
                        source,
                        [&exact],
                        &mut binding.parameter_reads(),
                    )
                    .map_err(BackendFailure::from_error)?;
                    values
                        .remove(exact.name())
                        .map(|value| (dtype, value))
                        .ok_or_else(|| {
                            BackendFailure::from_error(std::io::Error::other(
                                "exact scalar read returned no value",
                            ))
                        })
                })
                .transpose()?;
            let decoder = binding.decoder(scale)?;
            banks.insert(
                id.clone(),
                BoundRowBank::from_preflight(
                    residency.clone(),
                    entry.descriptor().clone(),
                    range,
                    decode_limit,
                    decoder,
                ),
            );
        }
        Ok((
            RowPool {
                id: pool.pool_id(),
                residency,
            },
            banks,
        ))
    }

    /// Aggregate pool residency on its owner; tensor peers have no pool.
    pub fn report(&self) -> Result<Option<ResidencyReport>, D::Error> {
        self.pool
            .as_ref()
            .map(|pool| pool.residency.report().map_err(Into::into))
            .transpose()
    }

    /// Selected row facts and the physical pool identity, only on its owner.
    pub fn pool_report(&self) -> Result<Option<RowLookupPoolReport>, D::Error> {
        self.pool
            .as_ref()
            .map(|pool| {
                Ok(RowLookupPoolReport::new(
                    pool.id,
                    pool.residency.report()?,
                    self.requirements,
                ))
            })
            .transpose()
    }

    /// Retained scalar and per-invocation workspace declarations.
    pub const fn requirements(&self) -> SelectedRowLookupRequirements {
        self.requirements
    }
}
impl<B, N, D, G> RowLookupProvider<B> for BoundRowLookups<B, N, D, G>
where
    B: DistributedNeuralBackend,
    N: DenseResidencyMechanisms,
    D: RowDecode<B, N>,
    G: Clone + Borrow<B::ParallelContext>,
    N::Transfer: ResidencyTransferOrdering<<B::Tensor as Tensor>::Context, Error = N::Error>,
{
    fn has_row_parameter(&self, parameter: &ParameterId) -> bool {
        match &self.providers {
            RowProviders::Local(providers) => providers.has_row_parameter(parameter),
            RowProviders::Tensor(providers) => providers.has_row_parameter(parameter),
        }
    }
    fn lookup_rows(
        &mut self,
        spec: &RowLookupSpec,
        rows: &[u64],
        access: ParameterBankAccess,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, RowLookupError> {
        match &mut self.providers {
            RowProviders::Local(providers) => providers.lookup_rows(spec, rows, access, context),
            RowProviders::Tensor(providers) => providers.lookup_rows(spec, rows, access, context),
        }
    }
}

/// Installs local selected row providers from an existing portable bank.
pub trait BindLocalRows<Bank, Binding>: Sized {
    /// Failure in row ownership or native scalar materialization.
    type Error;
    /// Binds only the exact already selected rows.
    fn bind_local_rows(
        bank: &Bank,
        rows: &SelectedRowLookups,
        binding: &Binding,
    ) -> Result<Self, Self::Error>;
}
impl<B, N, D, G, Binding>
    BindLocalRows<crate::expert::SharedAddressableParameterBank<N, B::Tensor>, Binding>
    for BoundRowLookups<B, N, D, G>
where
    B: DistributedNeuralBackend + CheckpointParameterBackend,
    N: DenseResidencyMechanisms + Clone,
    D: RowDecode<B, N, Error: From<eredu_core::parameters::ParameterError>>,
    Binding: RowDecodeBinding<B, N, D>,
    G: Clone + Borrow<B::ParallelContext>,
    N::Transfer: ResidencyTransferOrdering<<B::Tensor as Tensor>::Context, Error = N::Error>,
{
    type Error = D::Error;
    fn bind_local_rows(
        bank: &crate::expert::SharedAddressableParameterBank<N, B::Tensor>,
        rows: &SelectedRowLookups,
        binding: &Binding,
    ) -> Result<Self, Self::Error> {
        let pool = bank.lock_pool()?;
        Self::bind_local(&pool, rows, binding)
    }
}

impl<B, N, D, G, Binding>
    crate::executable_binding::partition_rank::BindPartitionRowStorage<
        crate::expert::SharedAddressableParameterBank<N, B::Tensor>,
        Binding,
        G,
    > for BoundRowLookups<B, N, D, G>
where
    B: DistributedNeuralBackend + CheckpointParameterBackend,
    N: DenseResidencyMechanisms + Clone,
    D: RowDecode<B, N, Error: From<eredu_core::parameters::ParameterError>>,
    Binding: RowDecodeBinding<B, N, D>,
    G: Clone + Borrow<B::ParallelContext>,
    N::Transfer: ResidencyTransferOrdering<<B::Tensor as Tensor>::Context, Error = N::Error>,
{
    type Error = D::Error;

    fn bind_partition_row_storage(
        bank: Option<&crate::expert::SharedAddressableParameterBank<N, B::Tensor>>,
        rows: &SelectedRowLookups,
        placement: crate::executable_binding::partition_rank::PartitionRowPlacement<'_, G>,
        binding: &Binding,
    ) -> Result<Self, Self::Error> {
        let pool = bank.map(|bank| bank.lock_pool()).transpose()?;
        let pool = pool.as_deref();
        match placement.parallel {
            Some(parallel) => Self::bind_tensor_parallel(
                pool,
                rows,
                parallel.clone(),
                placement.status.unwrap_or(parallel).clone(),
                placement.rank,
                placement.owner,
                binding,
            ),
            None if placement.rank == 0 && placement.owner == 0 && placement.status.is_none() => {
                Self::bind_local(
                    pool.ok_or_else(|| BackendFailure::from_error(RowLookupError::Geometry))?,
                    rows,
                    binding,
                )
            }
            None => Err(BackendFailure::from_error(RowLookupError::Geometry).into()),
        }
    }
}
