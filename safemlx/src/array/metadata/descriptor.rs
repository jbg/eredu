//! Fixed descriptor facts tied to the actual source and no-hooks runtime loan.
use super::{Array, ArrayAllocationInfo, Dtype};
use crate::utils::runtime_lock;
use std::{fmt, mem::MaybeUninit};

/// A fixed descriptor query or exact shape fill was refused before destination writes.
/// These are inline checked causes; no native exception is formatted or replaced.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ArrayDescriptorError {
    /// Another thread owns the native runtime; no native query was performed.
    #[error("native runtime is busy during descriptor inspection")]
    RuntimeBusy,
    /// A required native argument was absent.
    #[error("invalid native descriptor argument")]
    InvalidArgument,
    /// The source has no native descriptor.
    #[error("native array descriptor is empty")]
    EmptyDescriptor,
    /// The source's logical byte count cannot be represented.
    #[error("native descriptor logical byte count overflows")]
    LogicalBytesOverflow,
    /// The descriptor contains an invalid element representation.
    #[error("native descriptor datatype is invalid")]
    InvalidDtype,
    /// The retained source identity or observed native state changed since count.
    #[error("native descriptor source or state changed")]
    SourceChanged,
    /// The caller's destination does not have exactly the counted rank.
    #[error("descriptor shape destination length differs from rank")]
    DestinationLength,
    /// The native destination was null despite a nonzero rank.
    #[error("descriptor shape destination is null")]
    NullDestination,
    /// The linked fixed protocol returned an unknown status.
    #[error("unknown native descriptor status {0}")]
    UnknownStatus(u32),
}
impl ArrayDescriptorError {
    fn from_status(status: u32) -> Self {
        match status {
            1 => Self::InvalidArgument,
            2 => Self::EmptyDescriptor,
            3 => Self::LogicalBytesOverflow,
            4 => Self::InvalidDtype,
            5 => Self::SourceChanged,
            6 => Self::DestinationLength,
            7 => Self::NullDestination,
            value => Self::UnknownStatus(value),
        }
    }
}

/// Immutable scalar observation. It owns no descriptor, allocation or authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArrayDescriptorFacts {
    rank: usize,
    dtype: Dtype,
    logical_bytes: usize,
    allocation: Option<ArrayAllocationInfo>,
}
impl ArrayDescriptorFacts {
    /// Number of dimensions; the exact caller destination length.
    pub const fn rank(self) -> usize {
        self.rank
    }
    /// Element representation of the same descriptor.
    pub const fn dtype(self) -> Dtype {
        self.dtype
    }
    /// Logical tensor bytes, independent of backing capacity.
    pub const fn logical_bytes(self) -> usize {
        self.logical_bytes
    }
    /// Certified completed full backing, or unknown without evaluation or polling.
    pub const fn allocation(self) -> Option<ArrayAllocationInfo> {
        self.allocation
    }
}

/// Descriptor data borrowed from both its source and an existing runtime guard.
///
/// Construct this through [`crate::RuntimeCallGuard::descriptor`]. The read/fill
/// body and this borrowed loan's destruction neither acquire nor release runtime
/// ownership. The private runtime lock now uses only try-acquire routes, so its
/// final unlock cannot enter parking bookkeeping. Current-thread TLS and actual
/// dependency features still require their layout/build evidence; this loan
/// supplies no original budget grant.
///
/// The source cannot retire before the loan's final use:
/// ```compile_fail,E0505
/// use safemlx::{Array, RuntimeCallDeadline};
/// let mut runtime = RuntimeCallDeadline::new(std::time::Duration::from_secs(5)).unwrap().enter().unwrap();
/// let source = Array::from_slice(&[2_i32, 3], &[2]);
/// let loan = runtime.descriptor(&source).unwrap();
/// drop(source);
/// assert_eq!(loan.shape(), [2]);
/// ```
/// The actual outer guard also cannot retire before the loan:
/// ```compile_fail,E0505
/// use safemlx::{Array, RuntimeCallDeadline};
/// let source = Array::from_slice(&[2_i32, 3], &[2]);
/// let mut runtime = RuntimeCallDeadline::new(std::time::Duration::from_secs(5)).unwrap().enter().unwrap();
/// let loan = runtime.descriptor(&source).unwrap();
/// drop(runtime);
/// assert_eq!(loan.shape(), [2]);
/// ```
/// A borrowed runtime owner is not transferable to a foreign thread:
/// ```compile_fail,E0277
/// use safemlx::{Array, RuntimeCallDeadline};
/// let source = Array::from_slice(&[5_i32, 7], &[2]);
/// let mut runtime = RuntimeCallDeadline::new(std::time::Duration::from_secs(5)).unwrap().enter().unwrap();
/// let loan = runtime.descriptor(&source).unwrap();
/// std::thread::scope(|threads| { threads.spawn(move || loan.facts()); });
/// ```
/// A shared reference to the Sync guard cannot construct a foreign-thread loan:
/// ```compile_fail,E0596
/// use safemlx::{Array, RuntimeCallDeadline};
/// let source = Array::from_slice(&[5_i32, 7], &[2]);
/// let runtime = RuntimeCallDeadline::new(std::time::Duration::from_secs(5)).unwrap().enter().unwrap();
/// std::thread::scope(|threads| {
///     let guard = &runtime;
///     let source = &source;
///     threads.spawn(move || guard.descriptor(source).unwrap().facts());
/// });
/// ```
/// An ordinary caller can provide the exact destination under its existing guard:
/// ```
/// use safemlx::{Array, RuntimeCallDeadline};
/// let source = Array::from_slice(&[2_i32, 3], &[2]);
/// let mut runtime = RuntimeCallDeadline::new(std::time::Duration::from_secs(5)).unwrap().enter().unwrap();
/// let loan = runtime.descriptor(&source).unwrap();
/// let mut shape = vec![0; loan.facts().rank()];
/// loan.fill_shape(&mut shape).unwrap();
/// assert_eq!(shape, [2]);
/// drop(loan);
/// drop(runtime);
/// drop(source);
/// ```
pub struct ArrayDescriptorLoan<'loan> {
    source: &'loan Array,
    snapshot: DescriptorSnapshot,
    _guard: &'loan runtime_lock::RuntimeLockGuard,
}

/// Ordinary owning adapter around a source descriptor and a runtime try-lock.
///
/// The native body runs no hooks or evaluation. Acquisition and final unlock use
/// the existing private mutex with no parked waiter path. Source/guard storage,
/// calling-thread TLS and dependency feature evidence remain separate required
/// facts; this adapter supplies no complete original allocation bound. An
/// existing outer owner can instead lend [`ArrayDescriptorLoan`].
pub struct OwnedArrayDescriptorLoan<'source> {
    source: &'source Array,
    snapshot: DescriptorSnapshot,
    // Last field: the descriptor/source borrows retire before final unlock.
    _guard: runtime_lock::RuntimeLockGuard,
}

struct DescriptorSnapshot {
    raw: safemlx_sys::mlx_array_descriptor,
    facts: ArrayDescriptorFacts,
}
impl DescriptorSnapshot {
    fn read(
        source: &Array,
        _guard: &runtime_lock::RuntimeLockGuard,
    ) -> Result<Self, ArrayDescriptorError> {
        let mut raw = MaybeUninit::<safemlx_sys::mlx_array_descriptor>::uninit();
        // SAFETY: source retains the exact C handle and the actual guard already
        // serializes native entry. Only success initializes the complete output.
        let status =
            unsafe { safemlx_sys::mlx_array_descriptor_read(raw.as_mut_ptr(), source.as_ptr()) };
        if status != 0 {
            return Err(ArrayDescriptorError::from_status(status));
        }
        // SAFETY: status zero initializes every field of the plain witness.
        let raw = unsafe { raw.assume_init() };
        let dtype = Dtype::try_from(raw.dtype).map_err(|_| ArrayDescriptorError::InvalidDtype)?;
        let facts = ArrayDescriptorFacts {
            rank: raw.rank,
            dtype,
            logical_bytes: raw.logical_bytes,
            allocation: raw.known.then(|| {
                ArrayAllocationInfo::from_native(
                    raw.identity,
                    raw.allocation_bytes,
                    raw.host_transfer,
                )
            }),
        };
        Ok(Self { raw, facts })
    }
    fn shape(&self) -> &[i32] {
        if self.raw.rank == 0 {
            return &[];
        }
        // SAFETY: this private snapshot is stored only in the two wrappers that
        // retain the actual source and a live runtime guard. Shape never mutates.
        unsafe { std::slice::from_raw_parts(self.raw.shape, self.raw.rank) }
    }
    fn fill(&self, source: &Array, destination: &mut [i32]) -> Result<(), ArrayDescriptorError> {
        // SAFETY: wrapper retains the original source/guard. The mutable caller
        // destination is disjoint and its actual count is checked before writes.
        let status = unsafe {
            safemlx_sys::mlx_array_descriptor_fill_shape(
                source.as_ptr(),
                &self.raw,
                destination.as_mut_ptr(),
                destination.len(),
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(ArrayDescriptorError::from_status(status))
        }
    }
}

macro_rules! descriptor_accessors {
    ($name:ident) => {
        impl fmt::Debug for $name<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct(stringify!($name))
                    .field("facts", &self.snapshot.facts)
                    .finish_non_exhaustive()
            }
        }
        impl $name<'_> {
            /// Fixed scalar facts observed from this retained source.
            pub const fn facts(&self) -> ArrayDescriptorFacts {
                self.snapshot.facts
            }
            /// Compares the actual shared array descriptor under this runtime
            /// loan. Both array owners remain borrowed throughout comparison;
            /// matching shape, dtype or backing alone cannot establish identity.
            pub fn same_descriptor(&self, other: &Array) -> Result<bool, ArrayDescriptorError> {
                let other = DescriptorSnapshot::read(other, &self._guard)?;
                Ok(self.snapshot.raw.descriptor == other.raw.descriptor)
            }
            /// Borrows immutable source shape; creates no destination allocation.
            pub fn shape(&self) -> &[i32] {
                self.snapshot.shape()
            }
            /// Observes realized row-major contiguity under this exact source and
            /// runtime loan. Unfinished or unrecognized backing remains unknown;
            /// this never evaluates, polls, allocates or creates an owner.
            pub fn row_contiguous(&self) -> Option<bool> {
                self.snapshot.facts.allocation()?;
                crate::array::host_read::checked_layout(
                    self.snapshot.shape(),
                    self.source.signed_strides(),
                    self.source.item_size(),
                )
                .ok()
                .map(|layout| layout.contiguous)
            }
            /// Borrows realized signed element strides under this retained source
            /// and runtime loan. Unknown backing stays unknown; no evaluation,
            /// polling, allocation or owner construction occurs.
            pub fn completed_strides(&self) -> Option<&[i64]> {
                self.snapshot.facts.allocation()?;
                Some(self.source.signed_strides())
            }
            /// Rechecks source/current-state facts before writing exactly rank
            /// entries. Every failure preserves the complete destination; a
            /// zero-length fill still validates its actual source.
            pub fn fill_shape(&self, destination: &mut [i32]) -> Result<(), ArrayDescriptorError> {
                self.snapshot.fill(self.source, destination)
            }
        }
    };
}
descriptor_accessors!(ArrayDescriptorLoan);
descriptor_accessors!(OwnedArrayDescriptorLoan);

impl<'loan> ArrayDescriptorLoan<'loan> {
    pub(crate) fn under_guard(
        source: &'loan Array,
        guard: &'loan runtime_lock::RuntimeLockGuard,
    ) -> Result<Self, ArrayDescriptorError> {
        let snapshot = DescriptorSnapshot::read(source, guard)?;
        Ok(Self {
            source,
            snapshot,
            _guard: guard,
        })
    }
}

impl Array {
    /// Ordinary try-lock adapter for this exact retained native descriptor.
    ///
    /// Returns immediately on foreign runtime contention. Read/fill do not poll,
    /// evaluate, reclaim, run housekeeping, or construct native errors. Unknown
    /// backing stays unknown. Final release cannot enter this private mutex's
    /// parking machinery. Runtime layout/TLS and dependency feature verification
    /// remain required; this is not a complete original bound.
    pub fn try_descriptor(&self) -> Result<OwnedArrayDescriptorLoan<'_>, ArrayDescriptorError> {
        let guard =
            runtime_lock::try_enter_for_recovery().ok_or(ArrayDescriptorError::RuntimeBusy)?;
        let snapshot = DescriptorSnapshot::read(self, &guard)?;
        Ok(OwnedArrayDescriptorLoan {
            source: self,
            snapshot,
            _guard: guard,
        })
    }

    /// Fixed descriptor controls plus the second-source inspection performed
    /// by [`ArrayDescriptorLoan::same_descriptor`].
    pub fn descriptor_comparison_control_bytes() -> Option<usize> {
        Self::descriptor_control_bytes()?.checked_add(std::mem::size_of::<DescriptorSnapshot>())?
            .checked_add(std::mem::size_of::<Result<DescriptorSnapshot, ArrayDescriptorError>>())?
            .checked_add(std::mem::size_of::<Result<bool, ArrayDescriptorError>>())?
            .checked_add(std::mem::size_of::<&Array>())
    }

    /// Actual fixed kernel/loan/adapter representations, without source shape or
    /// caller destination backing. Static runtime and calling-thread TLS facts
    /// are separate in [`crate::utils::runtime_lock_layout`]; this sum is no
    /// whole-request grant or replacement for actual dependency feature evidence.
    pub fn descriptor_control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        // SAFETY: linked sizeof facts only, with no runtime/source/callback.
        let native = unsafe { safemlx_sys::mlx_array_descriptor_control_bytes() };
        let controls = [
            size_of::<ArrayDescriptorLoan<'static>>(),
            size_of::<OwnedArrayDescriptorLoan<'static>>(),
            size_of::<DescriptorSnapshot>(),
            size_of::<ArrayDescriptorFacts>(),
            size_of::<ArrayDescriptorError>(),
            size_of::<Dtype>(),
            size_of::<Result<Dtype, num_enum::TryFromPrimitiveError<Dtype>>>(),
            size_of::<Result<Dtype, ArrayDescriptorError>>(),
            size_of::<Option<ArrayAllocationInfo>>(),
            size_of::<Result<DescriptorSnapshot, ArrayDescriptorError>>(),
            size_of::<Result<ArrayDescriptorLoan<'static>, ArrayDescriptorError>>(),
            size_of::<Result<OwnedArrayDescriptorLoan<'static>, ArrayDescriptorError>>(),
            size_of::<Result<(), ArrayDescriptorError>>(),
            size_of::<MaybeUninit<safemlx_sys::mlx_array_descriptor>>(),
            size_of::<runtime_lock::RuntimeLockGuard>(),
            size_of::<Option<runtime_lock::RuntimeLockGuard>>(),
            size_of::<crate::utils::RuntimeLockLayout>(),
            size_of::<&Array>(),
            size_of::<&mut crate::RuntimeCallGuard>(),
            size_of::<&runtime_lock::RuntimeLockGuard>(),
            size_of::<&mut [i32]>(),
            size_of::<&[i64]>(),
            size_of::<Option<&[i64]>>(),
            size_of::<crate::array::host_read::LayoutSpan>(),
            size_of::<Result<crate::array::host_read::LayoutSpan, crate::error::AsSliceError>>(),
            size_of::<Option<bool>>(),
            size_of::<u32>(),
        ];
        controls.into_iter().try_fold(
            native.checked_add(size_of_val(&controls))?,
            usize::checked_add,
        )
    }
}

#[cfg(test)]
mod tests;
