use super::BigUint;
use crate::allocation::{AllocationError, Allocator, Unenforced};

use crate::big_digit::{self, BigDigits};

use alloc::borrow::Cow;
use core::mem;
use core::ops::{Shl, ShlAssign, Shr, ShrAssign};
use num_traits::{PrimInt, Zero};

#[inline]
pub(super) fn biguint_shl<T: PrimInt>(n: Cow<'_, BigUint>, shift: T) -> BigUint {
    biguint_shl_with_allocations(n, shift, Allocator(&Unenforced))
        .expect("ordinary integer allocation")
}
pub(super) fn biguint_shl_with_allocations<T: PrimInt>(
    n: Cow<'_, BigUint>,
    shift: T,
    allocations: Allocator<'_>,
) -> Result<BigUint, AllocationError> {
    allocations.controls::<(
        Cow<'_, BigUint>,
        usize,
        u8,
        BigUint,
        crate::big_digit::BigDigit,
        Allocator<'_>,
        Result<BigUint, AllocationError>,
    )>()?;
    if shift < T::zero() {
        panic!("attempt to shift left with negative");
    }
    if n.is_zero() {
        return super::allocated::into_owned(n, allocations);
    }
    let bits = T::from(big_digit::BITS).unwrap();
    let digits = (shift / bits).to_usize().expect("capacity overflow");
    let shift = (shift % bits).to_u8().unwrap();
    biguint_shl2(n, digits, shift, allocations)
}

fn biguint_shl2(
    n: Cow<'_, BigUint>,
    digits: usize,
    shift: u8,
    allocations: Allocator<'_>,
) -> Result<BigUint, AllocationError> {
    allocations.controls::<(
        Cow<'_, BigUint>,
        usize,
        u8,
        BigUint,
        crate::big_digit::BigDigit,
        Allocator<'_>,
        Result<BigUint, AllocationError>,
    )>()?;
    let mut data = match digits {
        0 => super::allocated::into_owned(n, allocations)?.data,
        _ => {
            let len = digits
                .checked_add(n.data.len())
                .and_then(|n| n.checked_add(1))
                .ok_or(AllocationError::SizeOverflow)?;
            let mut data = allocations.vector(len)?;
            data.resize(digits, 0);
            data.extend(n.data.iter());
            BigDigits::from_vec(data)
        }
    };

    if shift > 0 {
        let mut carry = 0;
        let carry_shift = big_digit::BITS - shift;
        for elem in data[digits..].iter_mut() {
            let new_carry = *elem >> carry_shift;
            *elem = (*elem << shift) | carry;
            carry = new_carry;
        }
        if carry != 0 {
            data.push_with_allocations(carry, allocations)?;
        }
    }

    Ok(BigUint { data })
}

#[inline]
fn biguint_shr<T: PrimInt>(n: Cow<'_, BigUint>, shift: T) -> BigUint {
    biguint_shr_with_allocations(n, shift, Allocator(&Unenforced))
        .expect("ordinary integer allocation")
}
pub(super) fn biguint_shr_with_allocations<T: PrimInt>(
    n: Cow<'_, BigUint>,
    shift: T,
    allocations: Allocator<'_>,
) -> Result<BigUint, AllocationError> {
    allocations.controls::<(
        Cow<'_, BigUint>,
        usize,
        u8,
        BigUint,
        crate::big_digit::BigDigit,
        Allocator<'_>,
        Result<BigUint, AllocationError>,
    )>()?;
    if shift < T::zero() {
        panic!("attempt to shift right with negative");
    }
    if n.is_zero() {
        return super::allocated::into_owned(n, allocations);
    }
    let bits = T::from(big_digit::BITS).unwrap();
    let digits = (shift / bits).to_usize().unwrap_or(usize::MAX);
    let shift = (shift % bits).to_u8().unwrap();
    biguint_shr2(n, digits, shift, allocations)
}

fn biguint_shr2(
    n: Cow<'_, BigUint>,
    digits: usize,
    shift: u8,
    allocations: Allocator<'_>,
) -> Result<BigUint, AllocationError> {
    allocations.controls::<(
        Cow<'_, BigUint>,
        usize,
        u8,
        BigUint,
        crate::big_digit::BigDigit,
        Allocator<'_>,
        Result<BigUint, AllocationError>,
    )>()?;
    if digits >= n.data.len() {
        return Ok(match n {
            Cow::Borrowed(_) => BigUint::ZERO,
            Cow::Owned(mut n) => {
                n.set_zero();
                n
            }
        });
    }
    let mut data = match n {
        Cow::Borrowed(n) => BigDigits::from_slice_with_allocations(&n.data[digits..], allocations)?,
        Cow::Owned(mut n) => {
            n.data.drain_front(digits);
            n.data.shrink_with_allocations(allocations)?;
            n.data
        }
    };

    if shift > 0 {
        let mut borrow = 0;
        let borrow_shift = big_digit::BITS - shift;
        for elem in data.iter_mut().rev() {
            let new_borrow = *elem << borrow_shift;
            *elem = (*elem >> shift) | borrow;
            borrow = new_borrow;
        }
        // Assuming we were normal before, only one
        // most-significant digit might be off now.
        if !data.is_normal() {
            data.pop();
        }
    }

    Ok(BigUint { data })
}

macro_rules! impl_shift {
    (@ref $Shx:ident :: $shx:ident, $ShxAssign:ident :: $shx_assign:ident, $rhs:ty) => {
        impl $Shx<&$rhs> for BigUint {
            type Output = BigUint;

            #[inline]
            fn $shx(self, rhs: &$rhs) -> BigUint {
                $Shx::$shx(self, *rhs)
            }
        }
        impl $Shx<&$rhs> for &BigUint {
            type Output = BigUint;

            #[inline]
            fn $shx(self, rhs: &$rhs) -> BigUint {
                $Shx::$shx(self, *rhs)
            }
        }
        impl $ShxAssign<&$rhs> for BigUint {
            #[inline]
            fn $shx_assign(&mut self, rhs: &$rhs) {
                $ShxAssign::$shx_assign(self, *rhs);
            }
        }
    };
    ($($rhs:ty),+) => {$(
        impl Shl<$rhs> for BigUint {
            type Output = BigUint;

            #[inline]
            fn shl(self, rhs: $rhs) -> BigUint {
                biguint_shl(Cow::Owned(self), rhs)
            }
        }
        impl Shl<$rhs> for &BigUint {
            type Output = BigUint;

            #[inline]
            fn shl(self, rhs: $rhs) -> BigUint {
                biguint_shl(Cow::Borrowed(self), rhs)
            }
        }
        impl ShlAssign<$rhs> for BigUint {
            #[inline]
            fn shl_assign(&mut self, rhs: $rhs) {
                let n = mem::replace(self, Self::ZERO);
                *self = n << rhs;
            }
        }
        impl_shift! { @ref Shl::shl, ShlAssign::shl_assign, $rhs }

        impl Shr<$rhs> for BigUint {
            type Output = BigUint;

            #[inline]
            fn shr(self, rhs: $rhs) -> BigUint {
                biguint_shr(Cow::Owned(self), rhs)
            }
        }
        impl Shr<$rhs> for &BigUint {
            type Output = BigUint;

            #[inline]
            fn shr(self, rhs: $rhs) -> BigUint {
                biguint_shr(Cow::Borrowed(self), rhs)
            }
        }
        impl ShrAssign<$rhs> for BigUint {
            #[inline]
            fn shr_assign(&mut self, rhs: $rhs) {
                let n = mem::replace(self, Self::ZERO);
                *self = n >> rhs;
            }
        }
        impl_shift! { @ref Shr::shr, ShrAssign::shr_assign, $rhs }
    )*};
}

impl_shift! { u8, u16, u32, u64, u128, usize }
impl_shift! { i8, i16, i32, i64, i128, isize }
