//! Shared payload worker, independent of header storage policy.
use super::read_destination::ConversionDestination;
use super::*;
use crate::TensorDescriptorView;
pub(super) struct Payload<'a, R> {
    pub(super) inner: &'a mut R,
    pub(super) endian: Endian,
    pub(super) limits: &'a Limits,
}
impl<R: Read + Seek> Payload<'_, R> {
    pub(super) fn read_raw_with_storage<'a>(
        &mut self,
        tensor: TensorDescriptorView<'_>,
        storage: RawStorage<'a>,
    ) -> std::result::Result<RawBuffer<'a>, ReadDestinationError> {
        check_limit(
            "tensor allocation",
            tensor.byte_len,
            self.limits.max_allocation_bytes,
        )?;
        storage.validate(tensor.byte_len)?;
        self.inner
            .seek(SeekFrom::Start(tensor.data_offset))
            .map_err(|source| Error::Io {
                offset: tensor.data_offset,
                source,
            })?;
        let len =
            usize::try_from(tensor.byte_len).map_err(|_| Error::Overflow("tensor allocation"))?;
        let mut data = storage.full(len);
        self.inner
            .read_exact(data.bytes_mut())
            .map_err(|source| Error::Io {
                offset: tensor.data_offset,
                source,
            })?;
        Ok(data)
    }
    /// Execute a validated metadata-only physical selection plan.
    pub(crate) fn read_tensor_plan_with_storage<C: ConversionDestination>(
        &mut self,
        plan: plan_storage::AxisView<'_>,
        storage: RawStorage<'_>,
        conversion: C,
    ) -> std::result::Result<C::Output, ReadDestinationError> {
        check_limit(
            "tensor allocation",
            plan.encoded_byte_len(),
            self.limits.max_allocation_bytes,
        )?;
        let selected_len = usize::try_from(plan.encoded_byte_len())
            .map_err(|_| Error::Overflow("selected tensor allocation"))?;
        storage.validate(plan.encoded_byte_len())?;
        let mut raw = storage.selected(selected_len);
        for span in plan.encoded_spans() {
            let span_len = usize::try_from(span.byte_len())
                .map_err(|_| Error::Overflow("selection span allocation"))?;
            self.inner
                .seek(SeekFrom::Start(span.offset()))
                .map_err(|source| Error::Io {
                    offset: span.offset(),
                    source,
                })?;
            let start = raw.len();
            let end = start
                .checked_add(span_len)
                .ok_or(Error::Overflow("selected tensor allocation"))?;
            raw.resize(end)?;
            self.inner
                .read_exact(&mut raw.bytes_mut()[start..end])
                .map_err(|source| Error::Io {
                    offset: span.offset(),
                    source,
                })?;
        }
        read_destination::convert_destination(
            plan.selected_descriptor(),
            raw.bytes(),
            self.endian,
            conversion,
        )
    }

    /// Execute a validated block-aligned contiguous-span plan.
    pub(crate) fn read_dense_tensor_span_with_storage<C: ConversionDestination>(
        &mut self,
        plan: plan_storage::SpanView<'_>,
        storage: RawStorage<'_>,
        conversion: C,
    ) -> std::result::Result<C::Output, ReadDestinationError> {
        check_limit(
            "tensor allocation",
            plan.encoded_byte_len(),
            self.limits.max_allocation_bytes,
        )?;
        let span = plan.encoded_span();
        storage.validate(plan.encoded_byte_len())?;
        self.inner
            .seek(SeekFrom::Start(span.offset()))
            .map_err(|source| Error::Io {
                offset: span.offset(),
                source,
            })?;
        let len = usize::try_from(span.byte_len())
            .map_err(|_| Error::Overflow("dense tensor span allocation"))?;
        let mut raw = storage.full(len);
        self.inner
            .read_exact(raw.bytes_mut())
            .map_err(|source| Error::Io {
                offset: span.offset(),
                source,
            })?;
        read_destination::convert_destination(
            plan.selected_descriptor(),
            raw.bytes(),
            self.endian,
            conversion,
        )
    }
    pub(super) fn read_tensor_with_storage<C: ConversionDestination>(
        &mut self,
        tensor: TensorDescriptorView<'_>,
        storage: RawStorage<'_>,
        conversion: C,
    ) -> std::result::Result<C::Output, ReadDestinationError> {
        let raw = self.read_raw_with_storage(tensor, storage)?;
        read_destination::convert_destination(tensor, raw.bytes(), self.endian, conversion)
    }
}
