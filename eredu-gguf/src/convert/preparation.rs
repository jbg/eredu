//! The original reached reserve/shape schedule, shared by both storage owners.
use super::*;

pub(super) trait Preparation {
    type Error: From<ConversionDestinationError>;
    fn reserve(&mut self, index: usize, count: usize) -> std::result::Result<(), Self::Error>;
    fn shapes(&mut self) -> (Slot<'_, u64>, Slot<'_, u64>);
}

pub(super) fn prepare<P: Preparation>(
    descriptor: TensorDescriptorView<'_>,
    endian: Endian,
    force_affine: bool,
    storage: &mut P,
) -> std::result::Result<(), P::Error> {
    let kind = plan::prepared_kind(descriptor, endian, force_affine)?;
    let rank = descriptor.dimensions.len();
    storage.reserve(0, rank)?;
    match kind {
        ConversionKind::Dense(_) | ConversionKind::IQuant => {
            let (first, _) = storage.shapes();
            let _shape = shape(descriptor, Some(first))?;
            drop(_shape);
            let bytes = plan::byte_request(descriptor)?;
            storage.reserve(2, bytes)?;
        }
        ConversionKind::MxFp4 => {
            storage.reserve(1, rank)?;
            let (first, second) = storage.shapes();
            let shapes = mxfp4_shapes_with_storage(descriptor, Some(first), Some(second))?;
            drop(shapes);
            let (words, blocks) = plan::mxfp4_requests(descriptor)?;
            storage.reserve(3, words)?;
            storage.reserve(6, blocks)?;
        }
        ConversionKind::Affine { bits, group_size } => {
            storage.reserve(1, rank)?;
            let (first, second) = storage.shapes();
            let (weights, scales) = affine_shapes_with_storage(
                descriptor,
                bits,
                group_size,
                Some(first),
                Some(second),
            )?;
            let (groups, words) =
                affine_counts(&weights, &scales).map_err(ConversionDestinationError::from)?;
            drop((weights, scales));
            let (words, groups) =
                plan::affine_requests(descriptor, bits, group_size, words, groups)?;
            storage.reserve(3, words)?;
            storage.reserve(4, groups)?;
            storage.reserve(5, groups)?;
        }
    }
    Ok(())
}
