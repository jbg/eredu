//! Fallible entrances into the canonical limb workers.
use super::{shift, subtraction, BigUint};
use crate::{
    allocation::{Allocation, AllocationError, Allocator},
    big_digit::BigDigits,
};
use alloc::borrow::Cow;
use core::{cmp, mem};
use num_traits::Zero;
pub(super) fn into_owned(
    value: Cow<'_, BigUint>,
    allocations: Allocator<'_>,
) -> Result<BigUint, AllocationError> {
    match value {
        Cow::Owned(value) => Ok(value),
        Cow::Borrowed(value) => value.clone_with_allocations(allocations.0),
    }
}
pub(super) trait IntoCow<'a> {
    fn into_cow(self) -> Cow<'a, BigUint>;
}
impl<'a> IntoCow<'a> for BigUint {
    fn into_cow(self) -> Cow<'a, BigUint> {
        Cow::Owned(self)
    }
}
impl<'a> IntoCow<'a> for &'a BigUint {
    fn into_cow(self) -> Cow<'a, BigUint> {
        Cow::Borrowed(self)
    }
}
impl BigUint {
    /// Copy the same integer after admitting its actual limb destination.
    pub fn clone_with_allocations(&self, source: &dyn Allocation) -> Result<Self, AllocationError> {
        Allocator(source).controls::<(&Self, Self, Result<Self, AllocationError>)>()?;
        Ok(Self {
            data: BigDigits::from_slice_with_allocations(&self.data, Allocator(source))?,
        })
    }
    /// Shift through the original limb worker with prospective storage.
    pub fn shl_with_allocations(
        self,
        bits: usize,
        source: &dyn Allocation,
    ) -> Result<Self, AllocationError> {
        shift::biguint_shl_with_allocations(Cow::Owned(self), bits, Allocator(source))
    }
    /// Shift through the original limb worker with prospective storage.
    pub fn shr_with_allocations(
        self,
        bits: usize,
        source: &dyn Allocation,
    ) -> Result<Self, AllocationError> {
        shift::biguint_shr_with_allocations(Cow::Owned(self), bits, Allocator(source))
    }
    /// Subtract through the original limb worker, including reached compaction.
    pub fn sub_assign_with_allocations(
        &mut self,
        other: &Self,
        source: &dyn Allocation,
    ) -> Result<(), AllocationError> {
        subtraction::sub2(&mut self.data, &other.data);
        self.data.normalize_with_allocations(Allocator(source))
    }
    pub fn gcd_with_allocations(
        &self,
        other: &Self,
        source: &dyn Allocation,
    ) -> Result<Self, AllocationError> {
        let allocations = Allocator(source);
        allocations.controls::<(
            Self,
            Self,
            u64,
            u64,
            Allocator<'_>,
            Result<Self, AllocationError>,
        )>()?;
        #[inline]
        fn twos(x: &BigUint) -> u64 {
            x.trailing_zeros().unwrap_or(0)
        }

        // Stein's algorithm
        if self.is_zero() {
            return other.clone_with_allocations(source);
        }
        if other.is_zero() {
            return self.clone_with_allocations(source);
        }
        let mut m = self.clone_with_allocations(source)?;
        let mut n = other.clone_with_allocations(source)?;

        // find common factors of 2
        let shift = cmp::min(twos(&n), twos(&m));

        // divide m and n by 2 until odd
        // m inside loop
        let shift_n = twos(&n);
        n = shift::biguint_shr_with_allocations(Cow::Owned(n), shift_n, allocations)?;

        while !m.is_zero() {
            let shift_m = twos(&m);
            m = shift::biguint_shr_with_allocations(Cow::Owned(m), shift_m, allocations)?;
            if n > m {
                mem::swap(&mut n, &mut m)
            }
            m.sub_assign_with_allocations(&n, source)?;
        }

        shift::biguint_shl_with_allocations(Cow::Owned(n), shift, allocations)
    }
    /// Divide through the original selected limb algorithm with prospective storage.
    pub fn div_with_allocations(
        self,
        other: Self,
        source: &dyn Allocation,
    ) -> Result<Self, AllocationError> {
        super::division::div_rem_with_allocations(
            Cow::Owned(self),
            Cow::Owned(other),
            Allocator(source),
        )
        .map(|(q, _)| q)
    }
    /// Multiply with the original size-selected algorithm and prospective storage.
    pub fn mul_with_allocations(
        self,
        other: Self,
        source: &dyn Allocation,
    ) -> Result<Self, AllocationError> {
        super::multiplication::multiply_with_allocations(
            Cow::Owned(self),
            Cow::Owned(other),
            Allocator(source),
        )
    }
    /// Add through the original digit worker, admitting reached extension/carry.
    pub fn add_with_allocations(
        mut self,
        other: &Self,
        source: &dyn Allocation,
    ) -> Result<Self, AllocationError> {
        let allocations = Allocator(source);
        allocations.controls::<(
            Self,
            Self,
            u64,
            u64,
            Allocator<'_>,
            Result<Self, AllocationError>,
        )>()?;
        if self.data.len() < other.data.len() {
            self.data
                .resize_with_allocations(other.data.len(), 0, allocations)?;
        }
        let carry = super::addition::__add2(&mut self.data, &other.data);
        if carry != 0 {
            self.data.push_with_allocations(carry, allocations)?;
        }
        Ok(self)
    }
    /// Multiply borrowed integers using the same original ownership choices.
    pub fn mul_ref_with_allocations(
        &self,
        other: &Self,
        source: &dyn Allocation,
    ) -> Result<Self, AllocationError> {
        super::multiplication::multiply_with_allocations(
            Cow::Borrowed(self),
            Cow::Borrowed(other),
            Allocator(source),
        )
    }
    /// Add borrowed integers using the original larger-operand copy rule.
    pub fn add_ref_with_allocations(
        &self,
        other: &Self,
        source: &dyn Allocation,
    ) -> Result<Self, AllocationError> {
        if self.data.len() >= other.data.len() {
            self.clone_with_allocations(source)?
                .add_with_allocations(other, source)
        } else {
            other
                .clone_with_allocations(source)?
                .add_with_allocations(self, source)
        }
    }
}
