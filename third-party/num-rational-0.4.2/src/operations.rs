//! Explicit fallible integer producers for the original rational workers.
use core::convert::Infallible;
use num_integer::Integer;
/// Complete integer operation contract used by reduction and division.
pub trait IntegerOperations<T> {
    /// The original producer failure.
    type Error;
    /// Admit concrete control storage before entering an original worker.
    fn reserve_controls(&self, bytes: usize) -> Result<(), Self::Error>;
    /// Construct zero.
    fn zero(&self) -> Result<T, Self::Error>;
    /// Construct one.
    fn one(&self) -> Result<T, Self::Error>;
    /// Copy the actual integer representation.
    fn clone(&self, value: &T) -> Result<T, Self::Error>;
    /// Compute the original greatest common divisor.
    fn gcd(&self, left: &T, right: &T) -> Result<T, Self::Error>;
    /// Divide using the original integer mechanism.
    fn div(&self, left: T, right: T) -> Result<T, Self::Error>;
    /// Multiply using the original integer mechanism.
    fn mul(&self, left: T, right: T) -> Result<T, Self::Error>;
    /// Subtract using the original integer mechanism.
    fn sub(&self, left: T, right: T) -> Result<T, Self::Error>;
}
/// Explicit ordinary operations for the shared rational worker.
pub struct Ordinary;
impl<T: Clone + Integer> IntegerOperations<T> for Ordinary {
    type Error = Infallible;
    fn reserve_controls(&self, _: usize) -> Result<(), Self::Error> {
        Ok(())
    }
    fn zero(&self) -> Result<T, Self::Error> {
        Ok(T::zero())
    }
    fn one(&self) -> Result<T, Self::Error> {
        Ok(T::one())
    }
    fn clone(&self, value: &T) -> Result<T, Self::Error> {
        Ok(value.clone())
    }
    fn gcd(&self, left: &T, right: &T) -> Result<T, Self::Error> {
        Ok(left.gcd(right))
    }
    fn div(&self, left: T, right: T) -> Result<T, Self::Error> {
        Ok(left / right)
    }
    fn mul(&self, left: T, right: T) -> Result<T, Self::Error> {
        Ok(left * right)
    }
    fn sub(&self, left: T, right: T) -> Result<T, Self::Error> {
        Ok(left - right)
    }
}
/// Borrowed source for the original unsigned big-integer producers.
#[cfg(feature = "num-bigint")]
pub struct BigUintOperations<'a>(pub &'a dyn num_bigint::allocation::Allocation);
#[cfg(feature = "num-bigint")]
impl IntegerOperations<num_bigint::BigUint> for BigUintOperations<'_> {
    type Error = num_bigint::allocation::AllocationError;
    fn reserve_controls(&self, bytes: usize) -> Result<(), Self::Error> {
        self.0.reserve(bytes)
    }
    fn zero(&self) -> Result<num_bigint::BigUint, Self::Error> {
        Ok(num_bigint::BigUint::ZERO)
    }
    fn one(&self) -> Result<num_bigint::BigUint, Self::Error> {
        Ok(num_bigint::BigUint::ONE)
    }
    fn clone(&self, value: &num_bigint::BigUint) -> Result<num_bigint::BigUint, Self::Error> {
        value.clone_with_allocations(self.0)
    }
    fn gcd(
        &self,
        left: &num_bigint::BigUint,
        right: &num_bigint::BigUint,
    ) -> Result<num_bigint::BigUint, Self::Error> {
        left.gcd_with_allocations(right, self.0)
    }
    fn div(
        &self,
        left: num_bigint::BigUint,
        right: num_bigint::BigUint,
    ) -> Result<num_bigint::BigUint, Self::Error> {
        left.div_with_allocations(right, self.0)
    }
    fn mul(
        &self,
        left: num_bigint::BigUint,
        right: num_bigint::BigUint,
    ) -> Result<num_bigint::BigUint, Self::Error> {
        left.mul_with_allocations(right, self.0)
    }
    fn sub(
        &self,
        mut left: num_bigint::BigUint,
        right: num_bigint::BigUint,
    ) -> Result<num_bigint::BigUint, Self::Error> {
        left.sub_assign_with_allocations(&right, self.0)?;
        Ok(left)
    }
}
