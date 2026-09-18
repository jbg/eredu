//! Allocation admission for syntax construction.
//!
//! A caller supplies a borrowed policy and retains its funding owner until the
//! resulting syntax and errors have retired. The policy is called before each
//! reached allocation, including replacement vector storage. Ordinary parsing
//! uses [`Unenforced`] through the same construction workers.

use alloc::{boxed::Box, string::String, vec::Vec};
use core::{fmt, mem::size_of};

/// A fixed, allocation-free refusal from syntax storage construction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub enum AllocationError {
    /// The caller refused the prospective allocation.
    Refused,
    /// The prospective allocation size cannot be represented.
    SizeOverflow,
    /// The host allocator refused a vector or string allocation.
    HostAllocation,
}

impl fmt::Display for AllocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Refused => "syntax allocation refused",
            Self::SizeOverflow => "syntax allocation size overflow",
            Self::HostAllocation => "syntax host allocation failed",
        })
    }
}

/// Admission for the byte capacity of one prospective allocation.
pub trait Allocation {
    /// Reserve before storage is allocated or replaced.
    fn reserve(&self, bytes: usize) -> Result<(), AllocationError>;
}

/// Ordinary allocation policy for the shared syntax construction workers.
#[derive(Clone, Copy, Debug)]
pub struct Unenforced;

impl Allocation for Unenforced {
    #[inline]
    fn reserve(&self, _: usize) -> Result<(), AllocationError> {
        Ok(())
    }
}

/// Borrowed policy plus the dependency's concrete storage producers.
#[derive(Clone, Copy)]
pub struct Allocator<'a> {
    policy: &'a dyn Allocation,
}

impl fmt::Debug for Allocator<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Allocator")
    }
}

impl<'a> Allocator<'a> {
    /// Borrow an allocation policy.
    pub fn new(policy: &'a dyn Allocation) -> Self {
        Self { policy }
    }

    /// Use ordinary allocation through the same workers.
    pub const fn unenforced() -> Self {
        Self {
            policy: &Unenforced,
        }
    }

    /// Fund the exact requested byte capacity before allocation.
    pub fn reserve(self, bytes: usize) -> Result<(), AllocationError> {
        if bytes == 0 {
            return Ok(());
        }
        self.policy.reserve(bytes)
    }

    /// Fund a box before invoking its producer.
    pub fn boxed<T>(self, value: T) -> Result<Box<T>, AllocationError> {
        self.reserve(size_of::<T>())?;
        Ok(Box::new(value))
    }

    /// Ensure capacity using an explicit geometric growth policy.
    ///
    /// Replacements are charged in full: old storage can coexist with the new
    /// allocation while its elements are moved. The minimum capacity is one,
    /// so total capacity allocated by a growing vector is less than four times
    /// its greatest reached length. This same policy is used by tree teardown.
    pub fn grow<T>(
        self,
        values: &mut Vec<T>,
        additional: usize,
    ) -> Result<(), AllocationError> {
        let needed = values
            .len()
            .checked_add(additional)
            .ok_or(AllocationError::SizeOverflow)?;
        if needed <= values.capacity() {
            return Ok(());
        }
        let capacity = values
            .capacity()
            .checked_mul(2)
            .ok_or(AllocationError::SizeOverflow)?
            .max(needed);
        let bytes = capacity
            .checked_mul(size_of::<T>())
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or(AllocationError::SizeOverflow)?;
        self.reserve(bytes)?;
        values
            .try_reserve_exact(capacity - values.len())
            .map_err(|_| AllocationError::HostAllocation)
    }

    /// Append an element after funding any replacement storage.
    #[inline]
    pub fn push<T>(
        self,
        values: &mut Vec<T>,
        value: T,
    ) -> Result<(), AllocationError> {
        if values.len() == values.capacity() {
            self.grow(values, 1)?;
        }
        values.push(value);
        Ok(())
    }

    /// Copy a slice into newly funded exact-capacity storage.
    pub fn copy_slice<T: Copy>(
        self,
        value: &[T],
    ) -> Result<Vec<T>, AllocationError> {
        let mut result = Vec::new();
        self.grow(&mut result, value.len())?;
        result.extend_from_slice(value);
        Ok(result)
    }

    /// Extend a vector after funding replacement capacity.
    pub fn extend_copy<T: Copy>(
        self,
        values: &mut Vec<T>,
        other: &[T],
    ) -> Result<(), AllocationError> {
        self.grow(values, other.len())?;
        values.extend_from_slice(other);
        Ok(())
    }

    /// Convert a vector to a boxed slice, funding a possible shrinking move.
    pub fn boxed_slice<T>(
        self,
        value: Vec<T>,
    ) -> Result<Box<[T]>, AllocationError> {
        if value.capacity() != value.len() {
            self.reserve(
                value
                    .len()
                    .checked_mul(size_of::<T>())
                    .ok_or(AllocationError::SizeOverflow)?,
            )?;
        }
        Ok(value.into_boxed_slice())
    }

    /// Convert a string to boxed text, funding a possible shrinking move.
    pub fn boxed_str(
        self,
        value: String,
    ) -> Result<Box<str>, AllocationError> {
        if value.capacity() != value.len() {
            self.reserve(value.len())?;
        }
        Ok(value.into_boxed_str())
    }

    /// Move a vector suffix into a newly funded exact-capacity allocation.
    pub fn split_off<T>(
        self,
        value: &mut Vec<T>,
        at: usize,
    ) -> Result<Vec<T>, AllocationError> {
        let count = value
            .len()
            .checked_sub(at)
            .ok_or(AllocationError::SizeOverflow)?;
        self.reserve(
            count
                .checked_mul(size_of::<T>())
                .ok_or(AllocationError::SizeOverflow)?,
        )?;
        Ok(value.split_off(at))
    }

    /// Copy a string into newly funded storage.
    pub fn copy_str(self, value: &str) -> Result<String, AllocationError> {
        self.reserve(value.len())?;
        let mut output = String::new();
        output
            .try_reserve_exact(value.len())
            .map_err(|_| AllocationError::HostAllocation)?;
        output.push_str(value);
        Ok(output)
    }

    /// Append text after funding an explicit geometric replacement.
    pub fn push_str(
        self,
        value: &mut String,
        text: &str,
    ) -> Result<(), AllocationError> {
        let needed = value
            .len()
            .checked_add(text.len())
            .ok_or(AllocationError::SizeOverflow)?;
        if needed > value.capacity() {
            let capacity = value
                .capacity()
                .checked_mul(2)
                .ok_or(AllocationError::SizeOverflow)?
                .max(needed);
            self.reserve(capacity)?;
            value
                .try_reserve_exact(capacity - value.len())
                .map_err(|_| AllocationError::HostAllocation)?;
        }
        value.push_str(text);
        Ok(())
    }

    /// Append a scalar without a temporary heap allocation.
    pub fn push_char(
        self,
        value: &mut String,
        c: char,
    ) -> Result<(), AllocationError> {
        self.push_str(value, c.encode_utf8(&mut [0; 4]))
    }
}

/// Push into prepaid iterative teardown storage. With capacities 1,2,4,...,
/// fewer than four slots are allocated per pushed node across all replacements.
/// Construction reserves this credit before a node can need iterative drop.
pub(crate) fn retirement_push<T>(values: &mut Vec<T>, value: T) {
    if values.len() == values.capacity() {
        let capacity = values.capacity().checked_mul(2).unwrap().max(1);
        values.reserve_exact(capacity - values.len());
    }
    values.push(value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::Cell;

    struct Funding {
        calls: Cell<usize>,
        bytes: Cell<usize>,
        refuse_at: usize,
    }

    impl Allocation for Funding {
        fn reserve(&self, bytes: usize) -> Result<(), AllocationError> {
            let call = self.calls.get();
            self.calls.set(call + 1);
            if call >= self.refuse_at {
                return Err(AllocationError::Refused);
            }
            self.bytes.set(self.bytes.get().checked_add(bytes).unwrap());
            Ok(())
        }
    }

    #[test]
    fn ast_refuses_every_reached_growth_without_losing_syntax_behavior() {
        for pattern in [
            r"(?P<name>ab|c(?:de)+){2,4}",
            "(?x) # comment text\n [a-z&&[^x]~~[0-9]] ",
            r"\p{Script_Extensions=Greek}\u{1F600}[[:alpha:]]",
            r"(?P<name>x)(?P<name>y)",
            "((a|b)[xyz]",
        ] {
            let expected = crate::ast::parse::Parser::new().parse(pattern);
            let funding = Funding {
                calls: Cell::new(0),
                bytes: Cell::new(0),
                refuse_at: usize::MAX,
            };
            let mut parser = crate::ast::parse::Parser::new();
            let actual = parser.parse_with_allocations(pattern, &funding);
            assert_eq!(expected, actual, "{pattern}");
            let calls = funding.calls.get();
            assert!(calls > 0 && funding.bytes.get() > 0);
            drop(actual);
            drop(parser);
            for refuse_at in 0..calls {
                let funding = Funding {
                    calls: Cell::new(0),
                    bytes: Cell::new(0),
                    refuse_at,
                };
                let mut parser = crate::ast::parse::Parser::new();
                let error = parser
                    .parse_with_allocations(pattern, &funding)
                    .unwrap_err();
                assert_eq!(
                    error.kind(),
                    &crate::ast::ErrorKind::Allocation(
                        AllocationError::Refused
                    )
                );
                assert!(error.pattern().is_empty());
                assert_eq!(
                    funding.calls.get(),
                    refuse_at + 1,
                    "calls after refusal: {pattern}"
                );
                // The original error formatter allocated its own notation
                // vectors. A fixed-capacity writer exercises the streaming
                // formatter, including the empty-pattern refusal case.
                struct Writer(usize);
                impl core::fmt::Write for Writer {
                    fn write_str(&mut self, text: &str) -> core::fmt::Result {
                        self.0 += text.len();
                        Ok(())
                    }
                }
                let mut writer = Writer(0);
                core::fmt::write(&mut writer, format_args!("{error}"))
                    .unwrap();
                assert!(writer.0 > 0);
                drop(error);
                drop(parser);
            }
        }
    }

    #[test]
    fn full_parser_refuses_each_ast_hir_and_unicode_growth() {
        for pattern in [
            r"(?i)(?:abc|a[0-9]+|(?P<word>\p{Greek}+))",
            r"(?-u:\x00[\x01-\x7F])",
            r"[[a-z]--[d-f]]|[[0-9]~~[5-8]]",
            r"(?:cat[0-9]|cat[a-z])",
            r"\p{unknown_property}",
        ] {
            let expected = crate::Parser::new().parse(pattern);
            let funding = Funding {
                calls: Cell::new(0),
                bytes: Cell::new(0),
                refuse_at: usize::MAX,
            };
            let mut parser = crate::Parser::new();
            let actual = parser.parse_with_allocations(pattern, &funding);
            assert_eq!(expected, actual, "{pattern}");
            let calls = funding.calls.get();
            assert!(calls > 0 && funding.bytes.get() > 0);
            drop(actual);
            drop(parser);
            for refuse_at in 0..calls {
                let funding = Funding {
                    calls: Cell::new(0),
                    bytes: Cell::new(0),
                    refuse_at,
                };
                let mut parser = crate::Parser::new();
                let error = parser
                    .parse_with_allocations(pattern, &funding)
                    .unwrap_err();
                match error {
                    crate::Error::Parse(ref error) => assert_eq!(
                        error.kind(),
                        &crate::ast::ErrorKind::Allocation(
                            AllocationError::Refused
                        )
                    ),
                    crate::Error::Translate(ref error) => assert_eq!(
                        error.kind(),
                        &crate::hir::ErrorKind::Allocation(
                            AllocationError::Refused
                        )
                    ),
                }
                assert_eq!(
                    funding.calls.get(),
                    refuse_at + 1,
                    "calls after refusal: {pattern}"
                );
                drop(error);
                drop(parser);
            }
        }
    }

    #[test]
    fn geometric_replacement_is_funded_before_mutation() {
        let funding = Funding {
            calls: Cell::new(0),
            bytes: Cell::new(0),
            refuse_at: 1,
        };
        let allocation = Allocator::new(&funding);
        let mut values = Vec::new();
        allocation.push(&mut values, 17u64).unwrap();
        let capacity = values.capacity();
        assert_eq!(
            allocation.push(&mut values, 23),
            Err(AllocationError::Refused)
        );
        assert_eq!(values, [17]);
        assert_eq!(values.capacity(), capacity);
        assert_eq!(funding.bytes.get(), size_of::<u64>());
    }
}
