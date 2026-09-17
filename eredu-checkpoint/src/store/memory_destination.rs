//! Final in-memory lease storage over the original selection worker.
use super::*;
use std::{alloc::Layout, collections::TryReserveError};

mod worker;
use worker::select;

// Ordinary owns exactly its prior index Vec. The fixed modes retain a genuine
// selection loan, preserving order/duplicates without manufacturing an index Vec.
enum Indices<'a> {
    Owned(Vec<usize>),
    Range(Range<usize>),
    Borrowed(&'a [usize]),
}
impl Indices<'_> {
    fn iter(&self) -> IndexIter<'_> {
        match self {
            Self::Owned(indices) => IndexIter::Borrowed(indices.iter().copied()),
            Self::Range(range) => IndexIter::Range(range.clone()),
            Self::Borrowed(indices) => IndexIter::Borrowed(indices.iter().copied()),
        }
    }
}
enum IndexIter<'a> {
    Range(Range<usize>),
    Borrowed(std::iter::Copied<std::slice::Iter<'a, usize>>),
}
impl Iterator for IndexIter<'_> {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        match self {
            Self::Range(range) => range.next(),
            Self::Borrowed(indices) => indices.next(),
        }
    }
}
trait Storage<'a> {
    type Buffer;
    fn range(&mut self, range: Range<usize>) -> Indices<'a>;
    fn indices(&mut self, indices: &'a Vec<usize>) -> Indices<'a>;
    fn output(&mut self, capacity: usize) -> Result<Self::Buffer, StoreError>;
    fn append(output: &mut Self::Buffer, bytes: &[u8]) -> Result<(), StoreError>;
    fn len(output: &Self::Buffer) -> usize;
}
struct Ordinary;
impl<'a> Storage<'a> for Ordinary {
    type Buffer = Vec<u8>;
    fn range(&mut self, range: Range<usize>) -> Indices<'a> {
        Indices::Owned(range.collect())
    }
    fn indices(&mut self, indices: &'a Vec<usize>) -> Indices<'a> {
        Indices::Owned(indices.clone())
    }
    fn output(&mut self, capacity: usize) -> Result<Self::Buffer, StoreError> {
        Ok(Vec::with_capacity(capacity))
    }
    fn append(output: &mut Self::Buffer, bytes: &[u8]) -> Result<(), StoreError> {
        output.extend_from_slice(bytes);
        Ok(())
    }
    fn len(output: &Self::Buffer) -> usize {
        output.len()
    }
}
pub(super) fn ordinary_selection(
    key: &str,
    dtype: Dtype,
    shape: &[usize],
    data: &[u8],
    selection: &TensorSelection,
    output_shape: &[usize],
    policy: ReadPolicy,
) -> Result<(Range<usize>, Option<Vec<u8>>), StoreError> {
    select(
        key,
        dtype,
        shape,
        data,
        selection,
        output_shape,
        policy,
        Ordinary,
    )
}
struct Count;
impl<'a> Storage<'a> for Count {
    type Buffer = usize;
    fn range(&mut self, range: Range<usize>) -> Indices<'a> {
        Indices::Range(range)
    }
    fn indices(&mut self, indices: &'a Vec<usize>) -> Indices<'a> {
        Indices::Borrowed(indices)
    }
    fn output(&mut self, _: usize) -> Result<usize, StoreError> {
        Ok(0)
    }
    fn append(output: &mut usize, bytes: &[u8]) -> Result<(), StoreError> {
        *output = output
            .checked_add(bytes.len())
            .ok_or_else(|| StoreError::Overflow {
                context: "prepared memory selection length".into(),
            })?;
        Ok(())
    }
    fn len(output: &usize) -> usize {
        *output
    }
}
struct Fixed<'d> {
    output: Option<&'d mut [u8]>,
}
struct Filled<'d> {
    output: &'d mut [u8],
    written: usize,
}
impl<'a, 'd> Storage<'a> for Fixed<'d> {
    type Buffer = Filled<'d>;
    fn range(&mut self, range: Range<usize>) -> Indices<'a> {
        Indices::Range(range)
    }
    fn indices(&mut self, indices: &'a Vec<usize>) -> Indices<'a> {
        Indices::Borrowed(indices)
    }
    fn output(&mut self, capacity: usize) -> Result<Self::Buffer, StoreError> {
        let output = self.output.take().expect("one gathered output");
        if output.len() != capacity {
            return Err(StoreError::Internal(
                "prepared memory selection destination length changed".into(),
            ));
        }
        Ok(Filled { output, written: 0 })
    }
    fn append(output: &mut Self::Buffer, bytes: &[u8]) -> Result<(), StoreError> {
        let end = output
            .written
            .checked_add(bytes.len())
            .ok_or_else(|| StoreError::Overflow {
                context: "prepared memory selection length".into(),
            })?;
        output
            .output
            .get_mut(output.written..end)
            .ok_or_else(|| {
                StoreError::Internal("prepared memory selection destination exhausted".into())
            })?
            .copy_from_slice(bytes);
        output.written = end;
        Ok(())
    }
    fn len(output: &Self::Buffer) -> usize {
        output.written
    }
}

#[derive(Debug)]
pub(super) struct Prepared {
    selected_bytes: Option<Arc<[u8]>>,
    gathered_length: Option<usize>,
    staging: Vec<u8>,
    output_shape: Vec<usize>,
    selection: TensorSelection,
    policy: ReadPolicy,
    span: Range<usize>,
    // Retained last: failed copy/scratch/final byte owners retire first.
    tensor: Arc<MemoryTensor>,
}
impl Prepared {
    pub(crate) fn retained_payload_capacity(&self) -> Option<(usize, usize)> {
        use super::acquisition::storage::{selection, vector};
        Some((
            self.staging
                .capacity()
                .checked_add(self.selected_bytes.as_ref().map_or(0, |bytes| bytes.len()))?
                .checked_add(vector::<usize>(self.output_shape.capacity())?)?
                .checked_add(selection(&self.selection)?)?,
            usize::from(self.selected_bytes.is_some()),
        ))
    }

    pub(super) fn prepare(
        store: &MemoryWeightStore,
        request: &TensorReadRequest,
    ) -> Result<Self, Failure> {
        Self::prepare_with_payload(store, request, true)
    }
    pub(super) fn prepare_with_payload(
        store: &MemoryWeightStore,
        request: &TensorReadRequest,
        allocate_payload: bool,
    ) -> Result<Self, Failure> {
        let tensor = store
            .tensors
            .get(&request.key)
            .cloned()
            .ok_or_else(|| Failure {
                cause: Cause::Store(StoreError::UnknownTensor {
                    key: request.key.clone(),
                }),
                pending: None,
            })?;
        let mut prepared = Self {
            selected_bytes: None,
            gathered_length: None,
            staging: Vec::new(),
            output_shape: Vec::new(),
            selection: request.selection.clone(),
            policy: request.policy,
            span: 0..0,
            tensor,
        };
        match prepared.prepare_storage(&request.key, allocate_payload) {
            Ok(()) => Ok(prepared),
            Err(cause) => Err(Failure {
                cause,
                pending: Some(prepared),
            }),
        }
    }
    fn prepare_storage(&mut self, key: &str, allocate_payload: bool) -> Result<(), Cause> {
        self.output_shape =
            validate_selection(key, &self.tensor.metadata.logical_shape, &self.selection)?;
        let (span, gathered) = select(
            key,
            self.tensor.dtype,
            &self.tensor.metadata.logical_shape,
            &self.tensor.bytes,
            &self.selection,
            &self.output_shape,
            self.policy,
            Count,
        )?;
        self.span = span;
        self.gathered_length = gathered;
        if allocate_payload {
            self.ensure_bytes()?;
        }
        Ok(())
    }
    fn ensure_bytes(&mut self) -> Result<(), Cause> {
        if self.selected_bytes.is_some() {
            return Ok(());
        }
        if let Some(length) = self.gathered_length {
            Layout::array::<u8>(length).map_err(|_| StoreError::Overflow {
                context: "prepared memory selection length".into(),
            })?;
            self.staging
                .try_reserve_exact(length)
                .map_err(Cause::Reserve)?;
            self.staging.resize(length, 0);
            // The final Arc slice and this Vec overlap during conversion. Both
            // are distinct allocations at eager preparation or deferred acquisition.
            self.selected_bytes = Some(Arc::from(std::mem::take(&mut self.staging)));
        }
        Ok(())
    }
    pub(super) fn matches(&self, store: &MemoryWeightStore, request: &TensorReadRequest) -> bool {
        store
            .tensors
            .get(&request.key)
            .is_some_and(|tensor| Arc::ptr_eq(tensor, &self.tensor))
            && self.selection == request.selection
            && self.policy == request.policy
    }
    pub(super) fn acquire(mut self, key: &str) -> Result<CheckpointLease, Failure> {
        let result = self
            .ensure_bytes()
            .and_then(|()| self.fill(key).map_err(Cause::Store));
        let proof = match result {
            Ok(proof) => proof,
            Err(error) => {
                return Err(Failure {
                    cause: error,
                    pending: Some(self),
                });
            }
        };
        Ok(CheckpointLease::Memory(MemoryLease {
            tensor: self.tensor,
            selection: self.selection,
            output_shape: self.output_shape,
            proof,
            span: self.span,
            selected_bytes: self.selected_bytes,
        }))
    }
    fn fill(&mut self, key: &str) -> Result<BoundedReadProof, StoreError> {
        let bytes = match self.selected_bytes.as_mut() {
            Some(bytes) => Arc::get_mut(bytes).expect("unpublished final memory destination"),
            None => &mut [],
        };
        let (span, gathered) = select(
            key,
            self.tensor.dtype,
            &self.tensor.metadata.logical_shape,
            &self.tensor.bytes,
            &self.selection,
            &self.output_shape,
            self.policy,
            Fixed {
                output: Some(bytes),
            },
        )?;
        let gathered_length = gathered.as_ref().map(|output| output.written);
        drop(gathered);
        if span != self.span
            || gathered_length != self.selected_bytes.as_ref().map(|bytes| bytes.len())
        {
            return Err(StoreError::Internal(
                "prepared memory selection geometry changed".into(),
            ));
        }
        let length = self
            .selected_bytes
            .as_ref()
            .map_or(span.len(), |bytes| bytes.len());
        let full_selection = matches!(self.selection, TensorSelection::Full);
        Ok(BoundedReadProof {
            physically_bounded: matches!(self.policy, ReadPolicy::RequireBounded) || full_selection,
            offset_bytes: u64::try_from(span.start).map_err(|_| StoreError::Overflow {
                context: "in-memory selection byte offset".into(),
            })?,
            length_bytes: u64::try_from(length).map_err(|_| StoreError::Overflow {
                context: "in-memory selection byte length".into(),
            })?,
            physical_reads: 0,
            physical_read_bytes: 0,
        })
    }
}
#[derive(Debug)]
enum Cause {
    Store(StoreError),
    Reserve(TryReserveError),
}
impl From<StoreError> for Cause {
    fn from(error: StoreError) -> Self {
        Self::Store(error)
    }
}
pub(super) struct Failure {
    cause: Cause,
    pending: Option<Prepared>,
}
impl std::fmt::Debug for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryDestinationFailure")
            .field("cause", &self.cause)
            .field("retains_destination", &self.pending.is_some())
            .finish()
    }
}
impl Failure {
    pub(super) fn store_error(&self) -> Option<&StoreError> {
        match &self.cause {
            Cause::Store(error) => Some(error),
            _ => None,
        }
    }
}
impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.cause {
            Cause::Store(error) => error.fmt(f),
            Cause::Reserve(error) => error.fmt(f),
        }
    }
}
impl std::error::Error for Failure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Store(error) => Some(error),
            Cause::Reserve(error) => Some(error),
        }
    }
}
#[cfg(test)]
mod tests;
