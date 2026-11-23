//! Transactions on the installed portable addressable bank.
use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Deref,
    time::Instant,
};

use eredu_core::residency::{CompactGroupBindings, NativeBankMechanisms, ParameterLoan};
use eredu_core::{residency::ResidencyTransferOrdering, BackendFailure};
use eredu_nn::{GroupedNeuralBackend, Tensor};

use super::{
    AddressableParameterBank, BankAcquisitionResources, BankTransfer, GroupedBankAcquisitionGuard,
    ParameterBankResidencyReport, SharedAddressableParameterBank,
};
use crate::{
    dense::DenseResidencyMechanisms, parameter_operations::BankParameterReadSource,
    residency::ResidencyDriver, AddressableGroupedBank, ParameterBank, ParameterBankKey,
};

type Context<B> = <<B as eredu_nn::NeuralBackend>::Tensor as Tensor>::Context;

struct AcquisitionGuard<B, N: DenseResidencyMechanisms, G> {
    pool: G,
    scope: Option<usize>,
    types: std::marker::PhantomData<fn() -> (B, N)>,
}

impl<B, N, G> GroupedBankAcquisitionGuard for AcquisitionGuard<B, N, G>
where
    B: GroupedNeuralBackend,
    N: DenseResidencyMechanisms + NativeBankMechanisms<Backend = B>,
    G: Deref<Target = AddressableParameterBank<N, B::Tensor>>,
{
    type Residency = ResidencyDriver<N>;

    fn resources(&self) -> BankAcquisitionResources<'_, Self::Residency> {
        self.pool.acquisition_resources(self.scope)
    }
    fn host_capacity_exhausted(error: &N::Error) -> bool {
        N::host_capacity_exhausted(error)
    }
}

impl<B, N> BankParameterReadSource<B::Tensor, Context<B>, N::BankError>
    for SharedAddressableParameterBank<N, B::Tensor>
where
    B: GroupedNeuralBackend,
    N: DenseResidencyMechanisms + NativeBankMechanisms<Backend = B>,
    N::Transfer: ResidencyTransferOrdering<Context<B>, Error = N::Error>,
{
    type Residency = ResidencyDriver<N>;
    type Loan = N::Loan;

    fn read_resources(
        &self,
        selected: &BTreeSet<String>,
    ) -> Result<(Self::Residency, BTreeMap<String, B::Tensor>), N::BankError> {
        let bank = self.lock_pool()?;
        let replacements = selected
            .iter()
            .filter_map(|id| {
                bank.replacement(id)
                    .map(|value| (id.clone(), value.clone()))
            })
            .collect();
        Ok((bank.residency_manager().clone(), replacements))
    }
    fn begin_loan(transfer: N::Transfer, context: &Context<B>) -> Result<Self::Loan, N::BankError> {
        N::begin_loan(transfer, context)
    }
    fn assemble_members(
        copies: Vec<B::Tensor>,
        context: &Context<B>,
    ) -> Result<B::Tensor, N::BankError> {
        N::assemble_members(copies, context)
    }
}

fn bind_groups<B, N, G>(
    bank: &AddressableParameterBank<N, B::Tensor>,
    acquisition: &mut BankTransfer<N::Loan>,
    mut groups: G,
    context: &Context<B>,
    started: Instant,
) -> Result<G, N::BankError>
where
    B: GroupedNeuralBackend,
    N: DenseResidencyMechanisms + NativeBankMechanisms<Backend = B>,
    G: CompactGroupBindings<B::Tensor>,
{
    let keys = &acquisition.identities;
    let bank_id = keys
        .first()
        .ok_or_else(|| {
            eredu_nn::Error::backend("compact grouped binding requires a selected member")
        })?
        .bank();
    let mut bindings = BTreeMap::new();
    for name in groups.binding_names() {
        let values = keys
            .iter()
            .enumerate()
            .map(|(index, key)| {
                let value = match bank.replacement_row(*key, &name) {
                    Some((row, value)) => N::replacement(value, row, context)?,
                    None => N::member(&acquisition.transfer, index, &name)?,
                };
                acquisition.transfer.retain(&value);
                Ok::<_, N::BankError>(value)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let compact = N::concatenate(&values, context)?;
        acquisition.transfer.retain(&compact);
        bindings.insert(name, compact);
    }
    groups.bind_compact(bindings)?;
    bank.record_compact_bank(
        bank_id,
        acquisition.access,
        acquisition.scratch_bytes,
        started.elapsed(),
    )
    .map_err(BackendFailure::from_error)?;
    Ok(groups)
}

impl<B, N> ParameterBank<B> for AddressableParameterBank<N, B::Tensor>
where
    B: GroupedNeuralBackend,
    N: DenseResidencyMechanisms + NativeBankMechanisms<Backend = B>,
    N::Transfer: ResidencyTransferOrdering<Context<B>, Error = N::Error>,
{
    type Residency = ResidencyDriver<N>;
    type Acquisition = BankTransfer<N::Loan>;
    type Report = ParameterBankResidencyReport;
    type Error = N::BankError;

    fn member_bytes(&self, key: ParameterBankKey) -> Option<u64> {
        self.effective_member_bytes().get(&key).copied()
    }
    fn begin_consumption(
        &self,
        acquisition: BankTransfer<N::Transfer>,
        context: &Context<B>,
    ) -> Result<Self::Acquisition, Self::Error> {
        acquisition.begin_loan(|transfer| N::begin_loan(transfer, context))
    }

    fn complete(
        &mut self,
        acquisition: Self::Acquisition,
        output: &B::Tensor,
    ) -> Result<(), Self::Error> {
        acquisition.transfer.finish(output)
    }
    fn report(&self) -> Result<Self::Report, Self::Error> {
        self.report()
            .map_err(BackendFailure::from_error)
            .map_err(Into::into)
    }
}

impl<B, N> AddressableGroupedBank<B> for AddressableParameterBank<N, B::Tensor>
where
    B: GroupedNeuralBackend,
    N: DenseResidencyMechanisms + NativeBankMechanisms<Backend = B>,
    N::Transfer: ResidencyTransferOrdering<Context<B>, Error = N::Error>,
    B::GatedProductGroups: CompactGroupBindings<B::Tensor>,
    B::LinearGroups: CompactGroupBindings<B::Tensor>,
    B::Relu2Groups: CompactGroupBindings<B::Tensor>,
{
    fn acquisition_guard(
        &self,
    ) -> Result<impl GroupedBankAcquisitionGuard<Residency = Self::Residency> + '_, Self::Error>
    {
        Ok(AcquisitionGuard::<B, N, _> {
            pool: self,
            scope: None,
            types: std::marker::PhantomData,
        })
    }

    fn gated_product_groups(
        &mut self,
        acquisition: &mut Self::Acquisition,
        spec: &eredu_nn::GroupedGatedProductSpec,
        context: &Context<B>,
    ) -> Result<B::GatedProductGroups, Self::Error> {
        let started = Instant::now();

        let groups = B::grouped_gated_product(spec.clone(), context)?;
        bind_groups::<B, N, _>(self, acquisition, groups, context, started)
    }

    fn linear_groups(
        &mut self,
        acquisition: &mut Self::Acquisition,
        spec: &eredu_nn::GroupedLinearSpec,
        context: &Context<B>,
    ) -> Result<B::LinearGroups, Self::Error> {
        let started = Instant::now();

        let groups = B::grouped_linear_bank(spec.clone(), context)?;
        bind_groups::<B, N, _>(self, acquisition, groups, context, started)
    }

    fn relu2_groups(
        &mut self,
        acquisition: &mut Self::Acquisition,
        spec: &eredu_nn::GroupedRelu2Spec,
        context: &Context<B>,
    ) -> Result<B::Relu2Groups, Self::Error> {
        let started = Instant::now();

        let groups = B::grouped_relu2(spec.clone(), context)?;
        bind_groups::<B, N, _>(self, acquisition, groups, context, started)
    }
}

impl<B, N> ParameterBank<B> for SharedAddressableParameterBank<N, B::Tensor>
where
    B: GroupedNeuralBackend,
    N: DenseResidencyMechanisms + NativeBankMechanisms<Backend = B>,
    N::Transfer: ResidencyTransferOrdering<Context<B>, Error = N::Error>,
{
    type Residency = ResidencyDriver<N>;
    type Acquisition = BankTransfer<N::Loan>;
    type Report = ParameterBankResidencyReport;
    type Error = N::BankError;

    fn member_bytes(&self, key: ParameterBankKey) -> Option<u64> {
        if !self.owns_parameter_bank(key.bank()) {
            return None;
        }
        self.lock_pool()
            .ok()?
            .effective_member_bytes()
            .get(&key)
            .copied()
    }
    fn begin_consumption(
        &self,
        acquisition: BankTransfer<N::Transfer>,
        context: &Context<B>,
    ) -> Result<Self::Acquisition, Self::Error> {
        acquisition.begin_loan(|transfer| N::begin_loan(transfer, context))
    }

    fn complete(
        &mut self,
        acquisition: Self::Acquisition,
        output: &B::Tensor,
    ) -> Result<(), Self::Error> {
        acquisition.transfer.finish(output)
    }
    fn report(&self) -> Result<Self::Report, Self::Error> {
        self.report()
            .map_err(BackendFailure::from_error)
            .map_err(Into::into)
    }
}

impl<B, N> AddressableGroupedBank<B> for SharedAddressableParameterBank<N, B::Tensor>
where
    B: GroupedNeuralBackend,
    N: DenseResidencyMechanisms + NativeBankMechanisms<Backend = B>,
    N::Transfer: ResidencyTransferOrdering<Context<B>, Error = N::Error>,
    B::GatedProductGroups: CompactGroupBindings<B::Tensor>,
    B::LinearGroups: CompactGroupBindings<B::Tensor>,
    B::Relu2Groups: CompactGroupBindings<B::Tensor>,
{
    fn acquisition_guard(
        &self,
    ) -> Result<impl GroupedBankAcquisitionGuard<Residency = Self::Residency> + '_, Self::Error>
    {
        Ok(AcquisitionGuard::<B, N, _> {
            pool: self.lock_pool()?,
            scope: self.scope(),
            types: std::marker::PhantomData,
        })
    }

    fn gated_product_groups(
        &mut self,
        acquisition: &mut Self::Acquisition,
        spec: &eredu_nn::GroupedGatedProductSpec,
        context: &Context<B>,
    ) -> Result<B::GatedProductGroups, Self::Error> {
        let started = Instant::now();
        let pool = self.lock_pool()?;
        let groups = B::grouped_gated_product(spec.clone(), context)?;
        bind_groups::<B, N, _>(&pool, acquisition, groups, context, started)
    }

    fn linear_groups(
        &mut self,
        acquisition: &mut Self::Acquisition,
        spec: &eredu_nn::GroupedLinearSpec,
        context: &Context<B>,
    ) -> Result<B::LinearGroups, Self::Error> {
        let started = Instant::now();
        let pool = self.lock_pool()?;
        let groups = B::grouped_linear_bank(spec.clone(), context)?;
        bind_groups::<B, N, _>(&pool, acquisition, groups, context, started)
    }

    fn relu2_groups(
        &mut self,
        acquisition: &mut Self::Acquisition,
        spec: &eredu_nn::GroupedRelu2Spec,
        context: &Context<B>,
    ) -> Result<B::Relu2Groups, Self::Error> {
        let started = Instant::now();
        let pool = self.lock_pool()?;
        let groups = B::grouped_relu2(spec.clone(), context)?;
        bind_groups::<B, N, _>(&pool, acquisition, groups, context, started)
    }
}
