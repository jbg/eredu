//! Five routed sources share settlement and copying across chunk and invocation owners.
use super::*;
use crate::backend::array_copy::{
    CompletedPartitionRoutedCaptureSource, CompletedRoutedCaptureSource,
    PartitionRoutedCaptureLayout, RoutedCaptureTransfer,
};
use crate::backend::runtime::residency::storage::StorageIdentity;
use eredu_core::capture::RoutedUnitCaptureSource;
use eredu_runtime::working_memory::{CaptureRoutedBatchWriter, CaptureRoutedPrefillWriter};

mod copy;

enum RoutedWriter<'t, 'f, 'p, 'a> {
    Prefill(CaptureRoutedPrefillWriter<'t, 'f, 'p, 'a>),
    Invocation(CaptureRoutedBatchWriter<'t, 'a>),
    Partition {
        writer: eredu_runtime::working_memory::CapturePartitionRoutedWriter<'t, 'a>,
        origins: Option<eredu_core::capture::RoutedUnitOrigins<'a>>,
        units: &'a eredu_core::component::ComponentCoordinateMap,
        prefill: bool,
    },
}
impl RoutedWriter<'_, '_, '_, '_> {
    fn bank_tokens(&self) -> (eredu_core::capture::RoutedUnitGeometry, u64) {
        match self {
            Self::Prefill(w) => (
                w.fragment().plan().geometry().bank(),
                w.fragment().source_tokens(),
            ),
            Self::Invocation(w) => (w.geometry().bank(), w.geometry().source_shape()[0] as u64),
            Self::Partition { writer, .. } => (
                writer.source_layout().geometry,
                writer.invocation_source_tokens(),
            ),
        }
    }
    fn validate_native_scope(
        &self,
        scope: &WorkingMemoryFundingScope,
    ) -> Result<(), WorkingMemoryError> {
        match self {
            Self::Prefill(w) => w.validate_native_scope(scope),
            Self::Invocation(w) => w.validate_native_scope(scope),
            Self::Partition { writer, .. } => writer.validate_native_scope(scope),
        }
    }
}

impl FundedWork {
    pub(in crate::composition::mlx::session::model_session::text_funding) fn capture_prefill_routed(
        &self,
        source: &RoutedUnitCaptureSource<'_, Array>,
        writer: CaptureRoutedPrefillWriter<'_, '_, '_, '_>,
        stream: &Stream,
        completion: CaptureCompletion<'_>,
    ) -> Result<(), Error> {
        self.capture_routed(source, RoutedWriter::Prefill(writer), stream, completion)
    }
    pub(in crate::composition::mlx::session::model_session::text_funding) fn capture_routed_batch(
        &self,
        source: &RoutedUnitCaptureSource<'_, Array>,
        writer: CaptureRoutedBatchWriter<'_, '_>,
        stream: &Stream,
        completion: CaptureCompletion<'_>,
    ) -> Result<(), Error> {
        self.capture_routed(source, RoutedWriter::Invocation(writer), stream, completion)
    }
    pub(in crate::composition::mlx::session::model_session::text_funding) fn capture_partition_routed(
        &self,
        source: &eredu_core::capture::PartitionRoutedUnitCaptureSource<'_, Array>,
        writer: eredu_runtime::working_memory::CapturePartitionRoutedWriter<'_, '_>,
        stream: &Stream,
        completion: CaptureCompletion<'_>,
        prefill: bool,
    ) -> Result<(), Error> {
        self.capture_routed(
            &source.source,
            RoutedWriter::Partition {
                writer,
                origins: source.origins,
                units: source.unit_coordinates,
                prefill,
            },
            stream,
            completion,
        )
    }
    fn capture_routed(
        &self,
        source: &RoutedUnitCaptureSource<'_, Array>,
        writer: RoutedWriter<'_, '_, '_, '_>,
        stream: &Stream,
        completion: CaptureCompletion<'_>,
    ) -> Result<(), Error> {
        let prefill = matches!(
            &writer,
            RoutedWriter::Prefill(_) | RoutedWriter::Partition { prefill: true, .. }
        );
        if prefill {
            let mut slot = self.capture.try_borrow_mut().map_err(error)?;
            let capture: &mut CaptureCarrier = slot
                .as_mut()
                .ok_or_else(|| error(CaptureCarrierError::Identity))?;
            capture.require_capacity(5)?;
            if capture.publications.borrow().capacity() - capture.publications.borrow().len() < 5 {
                return Err(error(WorkingMemoryError::UnknownBound));
            }
            let attempt = CaptureAttempt::new(&capture.progress)?;
            let segment = capture
                .segment
                .as_mut()
                .ok_or_else(|| error(CaptureCarrierError::Identity))?;
            self.capture_routed_sources(
                source,
                writer,
                stream,
                completion,
                &capture.roots,
                &capture.publications,
                Some(segment),
            )?;
            attempt.complete();
            Ok(())
        } else {
            self.capture_routed_sources(
                source,
                writer,
                stream,
                completion,
                &self.roots,
                &self.publications,
                None,
            )
        }
    }

    fn capture_routed_sources(
        &self,
        source: &RoutedUnitCaptureSource<'_, Array>,
        writer: RoutedWriter<'_, '_, '_, '_>,
        stream: &Stream,
        completion: CaptureCompletion<'_>,
        roots: &RefCell<Vec<Array>>,
        publications: &RefCell<Vec<RetainedStoragePublication>>,
        mut segment: Option<&mut CaptureSourceSegment>,
    ) -> Result<(), Error> {
        self.validate_capture_completion(completion)?;
        let (bank, tokens) = writer.bank_tokens();
        if let RoutedWriter::Partition {
            writer,
            origins,
            units,
            ..
        } = &writer
        {
            let layout = writer.source_layout();
            let shapes = [
                source.values.shape(),
                source.token_indices.shape(),
                source.selection_indices.shape(),
                source.coefficients.shape(),
                source.source_groups.shape(),
            ];
            let dtypes = [
                source.values.dtype(),
                source.token_indices.dtype(),
                source.selection_indices.dtype(),
                source.coefficients.dtype(),
                source.source_groups.dtype(),
            ];
            CompletedPartitionRoutedCaptureSource::validate_source_layouts(
                shapes,
                Some(dtypes),
                source.token_offset,
                PartitionRoutedCaptureLayout {
                    geometry: layout.geometry,
                    source_tokens: layout.source_tokens,
                    ownership: layout.ownership,
                    origins: *origins,
                    units,
                },
            )
            .map_err(error)?;
            if u64::try_from(shapes[4][0]).ok() != Some(writer.native_rows()) {
                return Err(error(WorkingMemoryError::IdentityMismatch));
            }
        } else {
            CompletedRoutedCaptureSource::validate_borrowed(source, bank, tokens).map_err(error)?;
        }
        PreparedCaptureTensor::validate_stream(stream).map_err(error)?;
        let _activity = publication_scope::Activity::begin(&self.publishing)?;
        completion.reserve_roots(roots, 5).map_err(error)?;
        let retain_publications =
            segment.is_some() || matches!(completion, CaptureCompletion::Ordinary);
        if matches!(completion, CaptureCompletion::Ordinary) && segment.is_none() {
            publications
                .try_borrow_mut()
                .map_err(error)?
                .try_reserve_exact(5)
                .map_err(error)?;
        }
        if retain_publications {
            let values = publications.try_borrow().map_err(error)?;
            if values.capacity().saturating_sub(values.len()) < 5 {
                return Err(error(WorkingMemoryError::UnknownBound));
            }
        }
        let mut scope_owner = publication_scope::OwnedScope::take(&self.scope)?;
        let scope = scope_owner.get_mut();
        if let Some(segment) = segment.as_deref_mut() {
            segment.validate_native_scope(scope).map_err(error)?;
            if let RoutedWriter::Prefill(writer) = &writer {
                segment
                    .validate_routed_prefill_fragment(writer.fragment())
                    .map_err(error)?;
            }
        }
        writer.validate_native_scope(scope).map_err(error)?;
        let inputs = [
            source.values,
            source.token_indices,
            source.selection_indices,
            source.coefficients,
            source.source_groups,
        ];
        self.published.set(false);
        let completed = completion
            .settle_retained(inputs, stream, roots)
            .map_err(error)?;
        writer.validate_native_scope(scope).map_err(error)?;
        let mut allocations = std::array::from_fn::<_, 5, _>(|_| None);
        // Original invocation publications attach custody to backing owners.
        // Keep these lexical aliases through pinning/copy; Work retains roots
        // through ordinary completion and failure recovery.
        let mut lexical_publications: [Option<RetainedStoragePublication>; 5] =
            std::array::from_fn(|_| None);
        for (index, value) in inputs.into_iter().enumerate() {
            let publication = self.publish_capture_source(value, scope, completion)?;
            if retain_publications {
                publications.borrow_mut().push(publication);
            } else {
                lexical_publications[index] = Some(publication);
            }
            let allocation = value.try_allocation_info().map_err(error)?.ok_or_else(|| {
                error(crate::backend::array_copy::CaptureTensorNativeError::UnsettledSource)
            })?;
            allocations[index] = Some((
                StorageIdentity::Native(allocation.identity()),
                u64::try_from(allocation.bytes()).map_err(error)?,
            ));
        }
        let prepared =
            crate::backend::runtime::residency::storage::prepare_funded_storage_publication(
                scope,
                allocations.len(),
            )
            .map_err(error)?;
        let source_pin = prepared
            .pin_registered_storage(
                allocations
                    .into_iter()
                    .flatten()
                    .filter(|(_, bytes)| *bytes != 0),
            )
            .map_err(error)?;
        copy::dispatch(
            source, completed, bank, tokens, writer, scope, segment, source_pin,
        )?;
        Ok(())
    }
}

pub(super) fn control_bytes() -> Option<usize> {
    [
        size_of::<RoutedWriter<'_, '_, '_, '_>>(),
        size_of::<[Option<RetainedStoragePublication>; 5]>(),
        size_of::<Option<&mut CaptureSourceSegment>>(),
        size_of::<(
            &RefCell<Vec<Array>>,
            &RefCell<Vec<RetainedStoragePublication>>,
        )>(),
        size_of::<std::cell::RefMut<'static, Option<CaptureCarrierOwner>>>(),
        size_of::<std::cell::Ref<'static, Vec<RetainedStoragePublication>>>(),
        size_of::<bool>(),
        size_of::<RoutedUnitCaptureSource<'_, Array>>(),
        size_of::<[&Array; 5]>(),
        size_of::<[safemlx::EvaluatedArray<'_>; 5]>(),
        size_of::<[Option<(StorageIdentity, u64)>; 5]>(),
        size_of::<Result<Option<safemlx::ArrayAllocationInfo>, safemlx::ArrayMetadataError>>(),
        size_of::<eredu_runtime::working_memory::WorkingMemoryStorage<StorageIdentity>>(),
        size_of::<Result<(), Error>>(),
        size_of::<RetainedStoragePublication>(),
        size_of::<(usize, u64, u64)>(),
        size_of::<PartitionRoutedCaptureLayout<'_>>() * 2,
        size_of::<eredu_core::capture::PartitionRoutedUnitCaptureLayout<'_>>(),
        size_of::<[&[i32]; 5]>(),
        size_of::<[safemlx::Dtype; 5]>(),
    ]
    .into_iter()
    .try_fold(
        CompletedPartitionRoutedCaptureSource::control_bytes()?
            .checked_add(CaptureCompletion::retained_settlement_control_bytes::<5>()?)?,
        usize::checked_add,
    )
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::{FundingFixture as _, StorageFixture as _};
