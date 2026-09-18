//! Prospective destinations shared by the original partition source workers.
use eredu_core::{HostMetadataFunding, HostMetadataFundingError};
use std::{
    alloc::Layout,
    fmt,
    mem::{size_of, size_of_val},
};
#[derive(Debug, thiserror::Error)]
pub(crate) enum Cause {
    #[error("{0}")]
    Semantic(String),
    #[error("{0}")]
    Funding(#[from] HostMetadataFundingError),
    #[error("partition source allocation: {0}")]
    Allocation(#[from] std::collections::TryReserveError),
    #[error("partition source capacity differs from its request")]
    Capacity,
}
impl Cause {
    pub(crate) fn ordinary(self) -> String {
        match self {
            Self::Semantic(value) => value,
            error => panic!("ordinary partition source allocation: {error}"),
        }
    }
}
#[derive(Clone, Copy)]
pub(crate) struct Allocation<'a>(pub(crate) Option<&'a HostMetadataFunding>);
impl Allocation<'_> {
    pub(crate) fn controls_of<T>(self, _: &T) -> Result<(), Cause> {
        self.controls::<T>()
    }
    pub(crate) fn reserve(self, bytes: usize) -> Result<(), Cause> {
        if let Some(funding) = self.0 {
            funding.reserve_metadata(bytes)?;
        }
        Ok(())
    }
    pub(crate) fn controls<T>(self) -> Result<(), Cause> {
        let parts = [
            size_of::<T>(),
            size_of::<Self>(),
            size_of::<Result<T, Cause>>(),
            HostMetadataFunding::reservation_control_bytes(),
        ];
        self.reserve(
            parts
                .into_iter()
                .try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or(HostMetadataFundingError::Overflow)?,
        )
    }
    pub(crate) fn vector<T>(self, count: usize) -> Result<Vec<T>, Cause> {
        self.controls::<(Vec<T>, usize, Layout, std::collections::TryReserveError)>()?;
        self.reserve(
            Layout::array::<T>(count)
                .map_err(|_| HostMetadataFundingError::Overflow)?
                .size(),
        )?;
        let mut values = Vec::new();
        values.try_reserve_exact(count)?;
        if size_of::<T>() != 0 && values.capacity() != count {
            return Err(Cause::Capacity);
        }
        Ok(values)
    }
    pub(crate) fn copy<T: Copy>(self, source: &[T]) -> Result<Vec<T>, Cause> {
        self.controls::<(&[T], Vec<T>)>()?;
        let mut values = self.vector(source.len())?;
        values.extend_from_slice(source);
        Ok(values)
    }
    pub(crate) fn grow<T>(self, values: &mut Vec<T>, additional: usize) -> Result<(), Cause> {
        self.controls::<(
            &mut Vec<T>,
            usize,
            Layout,
            std::collections::TryReserveError,
        )>()?;
        let required = values
            .len()
            .checked_add(additional)
            .ok_or(HostMetadataFundingError::Overflow)?;
        if required <= values.capacity() {
            return Ok(());
        }
        let minimum = if size_of::<T>() == 1 {
            8
        } else if size_of::<T>() <= 1024 {
            4
        } else {
            1
        };
        let target = required
            .max(
                values
                    .capacity()
                    .checked_mul(2)
                    .ok_or(HostMetadataFundingError::Overflow)?,
            )
            .max(minimum);
        self.reserve(
            Layout::array::<T>(target)
                .map_err(|_| HostMetadataFundingError::Overflow)?
                .size(),
        )?;
        values.try_reserve_exact(target - values.len())?;
        if size_of::<T>() != 0 && values.capacity() != target {
            return Err(Cause::Capacity);
        }
        Ok(())
    }
    pub(crate) fn push<T>(self, target: &mut Vec<T>, value: T) -> Result<(), Cause> {
        self.grow(target, 1)?;
        target.push(value);
        Ok(())
    }
    pub(crate) fn partition_error(self, cause: eredu_runtime::ArchitecturePartitionError) -> Cause {
        use eredu_runtime::ArchitecturePartitionError as E;
        match cause {
            E::MetadataFunding(cause) => Cause::Funding(cause),
            E::MetadataAllocation(cause) => Cause::Allocation(cause),
            E::MetadataCapacity => Cause::Capacity,
            cause => self.error(format_args!("{cause}")),
        }
    }
    pub(crate) fn text(self, source: &str) -> Result<String, Cause> {
        self.controls::<(&str, String, Result<String, std::string::FromUtf8Error>)>()?;
        Ok(String::from_utf8(self.copy(source.as_bytes())?).expect("copied UTF-8"))
    }
    pub(crate) fn error(self, args: fmt::Arguments<'_>) -> Cause {
        match self.format(args) {
            Ok(text) => Cause::Semantic(text),
            Err(error) => error,
        }
    }
    pub(crate) fn format(self, args: fmt::Arguments<'_>) -> Result<String, Cause> {
        use fmt::Write;
        struct Count(usize);
        impl fmt::Write for Count {
            fn write_str(&mut self, s: &str) -> fmt::Result {
                self.0 = self.0.checked_add(s.len()).ok_or(fmt::Error)?;
                Ok(())
            }
        }
        self.controls::<(Count, fmt::Arguments<'_>, fmt::Result, String)>()?;
        let mut count = Count(0);
        count
            .write_fmt(args)
            .map_err(|_| HostMetadataFundingError::Overflow)?;
        let mut value = String::from_utf8(self.vector(count.0)?).expect("empty UTF-8");
        value
            .write_fmt(args)
            .expect("fixed parameter diagnostic formatter");
        assert_eq!(value.len(), count.0);
        Ok(value)
    }
    pub(crate) fn additional(
        self,
        tensor: eredu_runtime::LocalTensorLayout,
        placement: eredu_runtime::TensorPlacement,
    ) -> Result<eredu_runtime::LocalTensorLayout, Cause> {
        Ok(match self.0 {
            Some(funding) => tensor
                .try_with_additional_placement(placement, funding)
                .map_err(|cause| match cause {
                    eredu_runtime::ParallelLayoutStorageError::Funding(cause) => {
                        Cause::Funding(cause)
                    }
                    eredu_runtime::ParallelLayoutStorageError::Allocation(cause) => {
                        Cause::Allocation(cause)
                    }
                    eredu_runtime::ParallelLayoutStorageError::Capacity => Cause::Capacity,
                })?,
            None => tensor.with_additional_placement(placement),
        })
    }
    pub(crate) fn insert(
        self,
        layout: &mut eredu_runtime::LocalModelLayout,
        target: String,
        tensor: eredu_runtime::LocalTensorLayout,
    ) -> Result<(), Cause> {
        self.controls::<(
            &mut eredu_runtime::LocalModelLayout,
            String,
            eredu_runtime::LocalTensorLayout,
        )>()?;
        match self.0 {
            Some(funding) => layout.try_insert_with_funding(target, tensor, funding)?,
            None => layout.insert(target, tensor),
        };
        Ok(())
    }
}
