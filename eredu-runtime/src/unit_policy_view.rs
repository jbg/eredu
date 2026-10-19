//! Typed component views into a single completion-owned parameter policy.

use crate::{ExecutionUnitAddress, ExecutionUnitLayout, LayerwiseAcquireError, LayerwisePolicy};
use eredu_nn::{NeuralBackend, Tensor};
use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

/// Architecture-owned embedding of one true parameter unit in a typed unit sum.
/// A policy must return the variant built for the requested canonical ordinal.
pub trait UnitProjection<Whole, Part> {
    /// Wraps a newly constructed component unit without constructing other units.
    fn wrap(unit: Part) -> Whole;
    /// Borrows the component from the acquired matching variant.
    fn get(unit: &Whole) -> &Part;
    /// Mutably borrows the component from the acquired matching variant.
    fn get_mut(unit: &mut Whole) -> &mut Part;
}

/// Invalid component addressing or a failure from the shared parameter policy.
#[derive(Debug, thiserror::Error)]
pub enum UnitPolicyViewError<E> {
    /// The component does not fit in the complete parameter layout.
    #[error("component parameter range is outside the shared layout")]
    InvalidRange,
    /// The mapped range names different parameter owners or unit indices.
    #[error("component parameter owners differ from the shared layout")]
    IncompatibleLayout,
    /// An invocation uses an address outside its canonical component layout.
    #[error("invalid component parameter address at ordinal {ordinal}")]
    InvalidAddress {
        /// Component-local ordinal supplied by the invocation.
        ordinal: usize,
    },
    /// Completion does not refer to the unit actually acquired.
    #[error("component parameter completion does not match the acquired unit")]
    MismatchedLease,
    /// The shared policy failed.
    #[error(transparent)]
    Policy(E),
}

/// A component-local policy backed by the same complete policy and its budgets.
/// The view neither owns a second cache nor reserves another memory allowance.
pub struct UnitPolicyView<'a, B, Whole, Part, P, Projection> {
    policy: &'a mut P,
    complete: &'a ExecutionUnitLayout,
    component: &'a ExecutionUnitLayout,
    offset: usize,
    types: PhantomData<fn() -> (B, Whole, Part, Projection)>,
}
impl<'a, B, Whole, Part, P, Projection> UnitPolicyView<'a, B, Whole, Part, P, Projection> {
    /// Maps a contiguous component range onto the authoritative complete layout.
    pub fn new(
        policy: &'a mut P,
        complete: &'a ExecutionUnitLayout,
        component: &'a ExecutionUnitLayout,
        offset: usize,
    ) -> Result<Self, UnitPolicyViewError<std::convert::Infallible>> {
        if offset
            .checked_add(component.len())
            .is_none_or(|end| end > complete.len())
        {
            return Err(UnitPolicyViewError::InvalidRange);
        }
        for local in 0..component.len() {
            let part = component
                .address(local)
                .expect("component layout covers its range");
            let whole = complete
                .address(offset + local)
                .expect("validated component range");
            if part.index() != whole.index()
                || component.group_id(part.group()) != complete.group_id(whole.group())
            {
                return Err(UnitPolicyViewError::IncompatibleLayout);
            }
        }
        Ok(Self {
            policy,
            complete,
            component,
            offset,
            types: PhantomData,
        })
    }
    fn address<E>(
        &self,
        ordinal: usize,
        address: ExecutionUnitAddress,
    ) -> Result<(usize, ExecutionUnitAddress), UnitPolicyViewError<E>> {
        if self.component.address(ordinal) != Some(address) {
            return Err(UnitPolicyViewError::InvalidAddress { ordinal });
        }
        let index = self.offset + ordinal;
        Ok((
            index,
            self.complete
                .address(index)
                .expect("validated component range"),
        ))
    }
}

/// A projected borrow that retains the original complete-policy lease and address.
pub struct ProjectedUnitLease<L, Whole, Part, Projection> {
    lease: L,
    ordinal: usize,
    address: ExecutionUnitAddress,
    types: PhantomData<fn() -> (Whole, Part, Projection)>,
}
impl<L: Deref<Target = Whole>, Whole, Part, Projection: UnitProjection<Whole, Part>> Deref
    for ProjectedUnitLease<L, Whole, Part, Projection>
{
    type Target = Part;
    fn deref(&self) -> &Part {
        Projection::get(&self.lease)
    }
}
impl<L: DerefMut<Target = Whole>, Whole, Part, Projection: UnitProjection<Whole, Part>> DerefMut
    for ProjectedUnitLease<L, Whole, Part, Projection>
{
    fn deref_mut(&mut self) -> &mut Part {
        Projection::get_mut(&mut self.lease)
    }
}
impl<B, Whole, Part, P, Projection> LayerwisePolicy<B, Part>
    for UnitPolicyView<'_, B, Whole, Part, P, Projection>
where
    B: NeuralBackend,
    P: LayerwisePolicy<B, Whole>,
    Projection: UnitProjection<Whole, Part>,
{
    type Lease = ProjectedUnitLease<P::Lease, Whole, Part, Projection>;
    type Error = UnitPolicyViewError<P::Error>;
    fn permits_deferred_serial_groups(&self) -> bool {
        self.policy.permits_deferred_serial_groups()
    }
    fn begin(
        &mut self,
        initial: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), Self::Error> {
        self.policy
            .begin(initial, context)
            .map_err(UnitPolicyViewError::Policy)
    }
    fn acquire<E, F>(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        build: F,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Lease, LayerwiseAcquireError<E, Self::Error>>
    where
        F: FnOnce(&<B::Tensor as Tensor>::Context) -> Result<Part, E>,
    {
        let (ordinal, address) = self
            .address(ordinal, address)
            .map_err(LayerwiseAcquireError::Policy)?;
        let lease = self
            .policy
            .acquire(
                ordinal,
                address,
                |ctx| build(ctx).map(Projection::wrap),
                context,
            )
            .map_err(|error| match error {
                LayerwiseAcquireError::Architecture(error) => {
                    LayerwiseAcquireError::Architecture(error)
                }
                LayerwiseAcquireError::Policy(error) => {
                    LayerwiseAcquireError::Policy(UnitPolicyViewError::Policy(error))
                }
            })?;
        Ok(ProjectedUnitLease {
            lease,
            ordinal,
            address,
            types: PhantomData,
        })
    }
    fn complete<'value, S, C>(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        lease: Self::Lease,
        output: &'value B::Tensor,
        state: S,
        inputs: C,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), Self::Error>
    where
        B::Tensor: 'value,
        S: Iterator<Item = &'value B::Tensor>,
        C: Iterator<Item = &'value B::Tensor>,
    {
        let requested = self.address(ordinal, address);
        if !matches!(requested, Ok((index, address)) if index == lease.ordinal && address == lease.address)
        {
            self.policy
                .abort(Some((lease.ordinal, lease.address, lease.lease)), context);
            return Err(requested
                .err()
                .unwrap_or(UnitPolicyViewError::MismatchedLease));
        }
        self.policy
            .complete(
                lease.ordinal,
                lease.address,
                lease.lease,
                output,
                state,
                inputs,
                context,
            )
            .map_err(UnitPolicyViewError::Policy)
    }
    fn abort(
        &mut self,
        active: Option<(usize, ExecutionUnitAddress, Self::Lease)>,
        context: &<B::Tensor as Tensor>::Context,
    ) {
        self.policy.abort(
            active.map(|(_, _, lease)| (lease.ordinal, lease.address, lease.lease)),
            context,
        );
    }
    fn finish(
        &mut self,
        output: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), Self::Error> {
        self.policy
            .finish(output, context)
            .map_err(UnitPolicyViewError::Policy)
    }
}
