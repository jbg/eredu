//! Metadata-only geometry and the existing prepared converter's requests.
use super::*;
use crate::LogicalDtype;
use std::alloc::Layout;

/// One ordered physical output of the existing conversion dispatch.
/// Shape is declared geometry, not evidence that malformed bytes will convert.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversionOutputPlan {
    shape: Vec<u64>,
    dtype: LogicalDtype,
}
impl ConversionOutputPlan {
    /// Row-major shape used by the existing conversion branch.
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }
    /// Actual scalar representation, including packed U8/U32 outputs.
    pub fn dtype(&self) -> LogicalDtype {
        self.dtype
    }
    /// Checked declared element count; not an initialized length or Vec capacity.
    pub fn shape_elements(&self) -> Option<u64> {
        self.shape.iter().try_fold(1u64, |n, &d| n.checked_mul(d))
    }
}

/// Cold requested storage and geometry, with no tensor payload allocations.
///
/// This owns descriptor/shape metadata only. Requests are not actual capacities,
/// allocator charges, conversion peaks, successful conversion or admission.
#[derive(Debug, Clone)]
pub struct ConversionPlan {
    descriptor: TensorDescriptor,
    endian: Endian,
    outputs: Vec<ConversionOutputPlan>,
    limits: [usize; 7],
    requested_slots: u8,
}
impl ConversionPlan {
    /// Describe the same dtype/endian dispatch used by ordinary conversion.
    pub fn new(
        descriptor: &TensorDescriptor,
        endian: Endian,
    ) -> std::result::Result<Self, ConversionDestinationError> {
        Self::build(descriptor, endian, false)
    }
    /// Describe the separate explicit affine API, without overriding source dispatch.
    pub fn affine(
        descriptor: &TensorDescriptor,
        endian: Endian,
    ) -> std::result::Result<Self, ConversionDestinationError> {
        Self::build(descriptor, endian, true)
    }
    fn build(descriptor: &TensorDescriptor, endian: Endian, affine: bool) -> CResult<Self> {
        let kind = prepared_kind(descriptor.view(), endian, affine)?;
        let rank = descriptor.dimensions.len();
        let mut limits = [0; 7];
        limits[0] = rank;
        let mut requested_slots = 1;
        let outputs = match kind {
            ConversionKind::Dense(dtype) => {
                limits[2] = byte_request(descriptor.view())?;
                requested_slots |= 1 << 2;
                vec![ConversionOutputPlan {
                    shape: shape(descriptor.view(), None)?.into_vec(),
                    dtype: dtype.into(),
                }]
            }
            ConversionKind::IQuant => {
                limits[2] = byte_request(descriptor.view())?;
                requested_slots |= 1 << 2;
                vec![ConversionOutputPlan {
                    shape: iquant_packed_shape(
                        &shape(descriptor.view(), None)?.into_vec(),
                        descriptor.ggml_type,
                    )?,
                    dtype: LogicalDtype::U8,
                }]
            }
            ConversionKind::MxFp4 => {
                limits[1] = rank;
                let (weights, scales) = mxfp4_shapes(descriptor)?;
                let (words, groups) = mxfp4_requests(descriptor.view())?;
                limits[3] = words;
                limits[6] = groups;
                requested_slots |= (1 << 1) | (1 << 3) | (1 << 6);
                vec![
                    ConversionOutputPlan {
                        shape: weights,
                        dtype: LogicalDtype::U32,
                    },
                    ConversionOutputPlan {
                        shape: scales,
                        dtype: LogicalDtype::U8,
                    },
                ]
            }
            ConversionKind::Affine { bits, group_size } => {
                limits[1] = rank;
                let (weights, scales) = affine_shapes(descriptor, bits, group_size)?;
                let (groups, words) = affine_counts(&weights, &scales)?;
                let (words, groups) =
                    affine_requests(descriptor.view(), bits, group_size, words, groups)?;
                limits[3] = words;
                limits[4] = groups;
                limits[5] = groups;
                requested_slots |= (1 << 1) | (1 << 3) | (1 << 4) | (1 << 5);
                vec![
                    ConversionOutputPlan {
                        shape: weights,
                        dtype: LogicalDtype::U32,
                    },
                    ConversionOutputPlan {
                        shape: scales.clone(),
                        dtype: LogicalDtype::F16,
                    },
                    ConversionOutputPlan {
                        shape: scales,
                        dtype: LogicalDtype::F16,
                    },
                ]
            }
        };
        requested_layouts(descriptor.view(), limits).ok_or(ConversionDestinationError::Layout)?;
        Ok(Self {
            descriptor: descriptor.clone(),
            endian,
            outputs,
            limits,
            requested_slots,
        })
    }
    /// Exact selected descriptor whose geometry and byte length were inspected.
    pub fn descriptor(&self) -> &TensorDescriptor {
        &self.descriptor
    }
    /// Retained byte order determining native-block versus expanded dispatch.
    pub fn endian(&self) -> Endian {
        self.endian
    }
    /// Physical outputs in the existing converter's producer order.
    pub fn outputs(&self) -> &[ConversionOutputPlan] {
        &self.outputs
    }
    /// Shape1, shape2, bytes, U32 words, U16 scales/biases and E8M0 requests.
    /// These bound prepared writes; they do not bound allocator-returned capacity.
    pub fn requested_elements(&self) -> [usize; 7] {
        self.limits
    }
    /// Checked requested G2 layouts, including its owner and descriptor binding.
    pub fn requested_layouts(&self) -> ConversionLayouts {
        requested_layouts(self.descriptor.view(), self.limits).expect("validated plan layouts")
    }
    /// G2 supplied-buffer requests, excluding its moved selected descriptor.
    /// The original plan dispatch records reached slots, including zero counts.
    pub fn supplied_storage_bound(&self) -> Option<crate::StorageRequestBound> {
        let mut out = crate::StorageRequestBound::default();
        for (i, count) in self.limits.iter().copied().enumerate() {
            if self.requested_slots & (1 << i) == 0 {
                continue;
            }
            match i {
                0 | 1 => out.add::<u64>(count)?,
                2 | 6 => out.add::<u8>(count)?,
                3 => out.add::<u32>(count)?,
                4 | 5 => out.add::<u16>(count)?,
                _ => unreachable!(),
            }
        }
        Some(out)
    }
    /// Actual metadata representation and buffer capacities retained by this plan.
    /// Excludes allocator charge and all tensor payload/native storage.
    pub fn metadata_bytes(&self) -> Option<usize> {
        let mut bytes = Layout::new::<Self>()
            .size()
            .checked_add(self.descriptor.name.capacity())?
            .checked_add(
                Layout::array::<u64>(self.descriptor.dimensions.capacity())
                    .ok()?
                    .size(),
            )?
            .checked_add(
                Layout::array::<ConversionOutputPlan>(self.outputs.capacity())
                    .ok()?
                    .size(),
            )?;
        for output in &self.outputs {
            bytes =
                bytes.checked_add(Layout::array::<u64>(output.shape.capacity()).ok()?.size())?;
        }
        Some(bytes)
    }
}

pub(super) fn prepared_kind(
    d: TensorDescriptorView<'_>,
    endian: Endian,
    affine: bool,
) -> CResult<ConversionKind> {
    if affine {
        let (bits, group_size) = affine_config(d.ggml_type)
            .ok_or_else(|| Error::tensor(d.name, "tensor encoding has no affine conversion"))?;
        Ok(ConversionKind::Affine { bits, group_size })
    } else {
        Ok(conversion_kind(d.ggml_type, endian)?)
    }
}
pub(super) fn byte_request(d: TensorDescriptorView<'_>) -> CResult<usize> {
    usize::try_from(d.byte_len).map_err(|_| ConversionDestinationError::Layout)
}
pub(super) fn mxfp4_requests(d: TensorDescriptorView<'_>) -> CResult<(usize, usize)> {
    let blocks =
        usize::try_from(d.byte_len / 17).map_err(|_| ConversionDestinationError::Layout)?;
    let words = blocks
        .checked_mul(4)
        .ok_or(ConversionDestinationError::Layout)?;
    Ok((words, blocks))
}
pub(super) fn affine_requests(
    d: TensorDescriptorView<'_>,
    bits: u8,
    group_size: u32,
    words: u64,
    groups: u64,
) -> CResult<(usize, usize)> {
    let (block_values, block_bytes) = d.ggml_type.block_and_bytes()?;
    let blocks = d.byte_len / block_bytes;
    let emitted_words = blocks
        .checked_mul(block_values)
        .and_then(|n| n.checked_mul(u64::from(bits)))
        .map(|n| n / 32)
        .ok_or(ConversionDestinationError::Layout)?;
    let emitted_groups = blocks
        .checked_mul(block_values / u64::from(group_size))
        .ok_or(ConversionDestinationError::Layout)?;
    let words = usize::try_from(words.max(emitted_words))
        .map_err(|_| ConversionDestinationError::Layout)?;
    let groups = usize::try_from(groups.max(emitted_groups))
        .map_err(|_| ConversionDestinationError::Layout)?;
    Ok((words, groups))
}
pub(super) fn requested_layouts(
    d: TensorDescriptorView<'_>,
    n: [usize; 7],
) -> Option<ConversionLayouts> {
    Some(ConversionLayouts {
        vectors: [
            Layout::array::<u64>(n[0]).ok()?,
            Layout::array::<u64>(n[1]).ok()?,
            Layout::array::<u8>(n[2]).ok()?,
            Layout::array::<u32>(n[3]).ok()?,
            Layout::array::<u16>(n[4]).ok()?,
            Layout::array::<u16>(n[5]).ok()?,
            Layout::array::<u8>(n[6]).ok()?,
        ],
        descriptor_name: Layout::array::<u8>(d.name.len()).ok()?,
        descriptor_dimensions: Layout::array::<u64>(d.dimensions.len()).ok()?,
        owner: Layout::new::<PreparedConversion>(),
    })
}

#[cfg(test)]
mod tests;
