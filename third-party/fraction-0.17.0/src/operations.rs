//! Explicit storage producers for the original fraction conversion worker.
use crate::{CheckedAdd, CheckedMul, FromPrimitive, Integer};
pub use num::rational::operations::{IntegerOperations, Ordinary};
use std::fmt::{self, Write};
/// Integer source operations reached by the original decimal/float conversion.
pub trait FractionOperations<T>: IntegerOperations<T> {
    /// Convert the same finite binary value to an integer magnitude.
    fn from_f64(&self, value: f64) -> Result<Option<T>, Self::Error>
    where
        T: FromPrimitive;
    /// Parse the same radix source, preserving invalid-input semantics.
    fn parse(&self, source: &str) -> Result<Option<T>, Self::Error>;
    /// Checked addition preserving the selected integer's overflow behavior.
    fn checked_add(&self, left: &T, right: &T) -> Result<Option<T>, Self::Error>;
    /// Checked multiplication preserving the selected integer's overflow behavior.
    fn checked_mul(&self, left: &T, right: &T) -> Result<Option<T>, Self::Error>;
    /// Grow the concrete formatting destination before the reached write.
    fn grow_string(&self, output: &mut String, required: usize) -> Result<(), Self::Error>;
}
impl<T: Clone + Integer + CheckedAdd + CheckedMul> FractionOperations<T> for Ordinary {
    fn from_f64(&self, value: f64) -> Result<Option<T>, Self::Error>
    where
        T: FromPrimitive,
    {
        Ok(T::from_f64(value))
    }
    fn parse(&self, source: &str) -> Result<Option<T>, Self::Error> {
        Ok(T::from_str_radix(source, 10).ok())
    }
    fn checked_add(&self, left: &T, right: &T) -> Result<Option<T>, Self::Error> {
        Ok(left.checked_add(right))
    }
    fn checked_mul(&self, left: &T, right: &T) -> Result<Option<T>, Self::Error> {
        Ok(left.checked_mul(right))
    }
    fn grow_string(&self, output: &mut String, required: usize) -> Result<(), Self::Error> {
        if required > output.capacity() {
            let capacity = output
                .capacity()
                .checked_mul(2)
                .map(|n| n.max(required))
                .expect("integer text capacity");
            output
                .try_reserve_exact(capacity - output.len())
                .expect("ordinary integer text allocation");
        }
        Ok(())
    }
}
/// Format through the original Display implementation, with reached writes paid.
pub(crate) fn format_signed<T, O: FractionOperations<T>>(
    value: impl fmt::Display,
    operations: &O,
) -> Result<String, O::Error> {
    struct Writer<'a, T, O: FractionOperations<T>> {
        text: String,
        operations: &'a O,
        error: Option<O::Error>,
        marker: std::marker::PhantomData<T>,
    }
    impl<T, O: FractionOperations<T>> Write for Writer<'_, T, O> {
        fn write_str(&mut self, text: &str) -> fmt::Result {
            if self.error.is_some() {
                return Err(fmt::Error);
            }
            // The input is a primitive float; the formatter cannot emit an unrepresentable string.
            let required = self
                .text
                .len()
                .checked_add(text.len())
                .expect("float text length");
            match self.operations.grow_string(&mut self.text, required) {
                Ok(()) => {
                    self.text.push_str(text);
                    Ok(())
                }
                Err(error) => {
                    self.error = Some(error);
                    Err(fmt::Error)
                }
            }
        }
    }
    operations.reserve_controls(std::mem::size_of::<(
        Writer<'_, T, O>,
        fmt::Arguments<'_>,
        fmt::Result,
    )>())?;
    let mut writer = Writer {
        text: String::new(),
        operations,
        error: None,
        marker: std::marker::PhantomData,
    };
    let result = write!(writer, "{:+}", value);
    if let Some(error) = writer.error {
        return Err(error);
    }
    result.expect("primitive float formatting");
    Ok(writer.text)
}
#[cfg(feature = "with-bigint")]
pub use num::bigint::allocation::{Allocation, AllocationError, Unenforced};
#[cfg(feature = "with-bigint")]
pub use num::rational::operations::BigUintOperations;
#[cfg(feature = "with-bigint")]
impl FractionOperations<num::bigint::BigUint> for BigUintOperations<'_> {
    fn from_f64(&self, value: f64) -> Result<Option<num::bigint::BigUint>, Self::Error> {
        num::bigint::BigUint::from_f64_with_allocations(value, self.0)
    }
    fn parse(&self, source: &str) -> Result<Option<num::bigint::BigUint>, Self::Error> {
        num::bigint::BigUint::from_str_radix_with_allocations(source, 10, self.0).map(Result::ok)
    }
    fn checked_add(
        &self,
        left: &num::bigint::BigUint,
        right: &num::bigint::BigUint,
    ) -> Result<Option<num::bigint::BigUint>, Self::Error> {
        left.add_ref_with_allocations(right, self.0).map(Some)
    }
    fn checked_mul(
        &self,
        left: &num::bigint::BigUint,
        right: &num::bigint::BigUint,
    ) -> Result<Option<num::bigint::BigUint>, Self::Error> {
        left.mul_ref_with_allocations(right, self.0).map(Some)
    }
    fn grow_string(&self, output: &mut String, required: usize) -> Result<(), Self::Error> {
        if required <= output.capacity() {
            return Ok(());
        }
        let capacity = output
            .capacity()
            .checked_mul(2)
            .map(|n| n.max(required))
            .ok_or(AllocationError::SizeOverflow)?;
        self.0.reserve(capacity)?;
        output
            .try_reserve_exact(capacity - output.len())
            .map_err(|_| AllocationError::HostAllocation)?;
        if output.capacity() != capacity {
            return Err(AllocationError::HostAllocation);
        }
        Ok(())
    }
}
