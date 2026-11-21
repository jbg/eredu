//! Portable constructors for the admitted resident and bounded policies.
use std::ops::DerefMut;

use eredu_core::BackendFailure;
use eredu_nn::{NeuralBackend, Parameterized, Tensor};

use crate::dense::DenseResidencyMechanisms;

use super::{
    BoundUnitStorage, BoundedDenseRun, BoundedUnitBinding, NativeResidentUnits, ResidentUnitPolicy,
    SelectedLayerwisePolicy, SelectedUnitResources,
};

/// Installs fully resident execution from the prepared native transfers.
pub trait BindResidentPolicy<B: NeuralBackend, U, S>: Sized {
    /// Adopts the admitted storage and installs the physical units in order.
    fn bind_resident(
        storage: S,
        units: Vec<U>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, BackendFailure>;
}

/// Installs bounded execution from prepared native storage.
pub trait BindBoundedPolicy<U, S>: Sized {
    /// Adopts storage and constructs a portable window driver.
    fn bind_bounded(storage: S) -> Self;
}

impl<B, U, N, O> BindResidentPolicy<B, U, BoundUnitStorage<N, O>>
    for ResidentUnitPolicy<
        B,
        N,
        O::Module<U>,
        super::storage::ResidentUnitRetention<N, O>,
        O::Error,
    >
where
    B: NeuralBackend,
    U: Parameterized<B::Tensor> + 'static,
    N: DenseResidencyMechanisms + Clone + Send + Sync + 'static,

    O: NativeResidentUnits<B, N::Transfer> + super::NativeBoundedUnits<Tensor = B::Tensor>,
{
    fn bind_resident(
        storage: BoundUnitStorage<N, O>,
        units: Vec<U>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, BackendFailure> {
        storage
            .into_resident_units::<B, U>(units, context)
            .map_err(BackendFailure::from_error)
    }
}

impl<B, U, R, N, L, D, I> BindResidentPolicy<B, U, N> for SelectedLayerwisePolicy<R, N, L, D, I>
where
    B: NeuralBackend,
    R: SelectedUnitResources + BindResidentPolicy<B, U, N>,
    N: SelectedUnitResources<Error = R::Error>,
{
    fn bind_resident(
        storage: N,
        units: Vec<U>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, BackendFailure> {
        R::bind_resident(storage, units, context).map(Self::resident)
    }
}

impl<U: 'static, R, N, L, D, I> BindBoundedPolicy<U, N> for SelectedLayerwisePolicy<R, N, L, D, I>
where
    R: SelectedUnitResources,
    N: SelectedUnitResources<Error = R::Error> + BoundedUnitBinding<Lease<U> = L, DenseOwner = D>,
    D: DerefMut<Target = BoundedDenseRun<N::Residency>>,
{
    fn bind_bounded(storage: N) -> Self {
        Self::bounded::<U>(storage)
    }
}
