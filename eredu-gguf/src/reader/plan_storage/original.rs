//! Unchanged pre-G3 struct/impl bodies, defining independent local plan types.
use super::super::super::*;

#[derive(Debug)]
pub struct TensorSelectionPlan {
    selection: TensorSelection,
    gguf_dimension: usize,
    alignment: SelectionAlignment,
    selected_descriptor: TensorDescriptor,
    source_data_offset: u64,
    repetition_stride: u64,
    repetitions: u64,
    relative_spans: Vec<RelativeEncodedSpan>,
    encoded_byte_len: u64,
}
impl TensorSelectionPlan {
    /// Build and validate a physical selection plan without reading payloads.
    pub fn new(tensor: &TensorDescriptor, selection: TensorSelection) -> Result<Self> {
        let rank = tensor.dimensions.len();
        let logical_axis = selection.axis();
        if logical_axis >= rank {
            return Err(Error::tensor(
                &tensor.name,
                format!("selection axis {logical_axis} is outside rank {rank}"),
            ));
        }
        if tensor.byte_len == 0 || tensor.dimensions.contains(&0) {
            return Err(Error::tensor(
                &tensor.name,
                "cannot select from an empty tensor",
            ));
        }

        let gguf_dimension = rank - 1 - logical_axis;
        let dimension_u64 = tensor.dimensions[gguf_dimension];
        let dimension = usize::try_from(dimension_u64)
            .map_err(|_| Error::Overflow("selected tensor dimension"))?;
        let (block_values, block_bytes) = tensor.ggml_type.block_and_bytes()?;
        if !tensor.dimensions[0].is_multiple_of(block_values) {
            return Err(Error::tensor(
                &tensor.name,
                format!(
                    "fastest dimension {} is not divisible by GGUF block length {block_values}",
                    tensor.dimensions[0]
                ),
            ));
        }
        let source_byte_len = tensor
            .element_count()?
            .checked_div(block_values)
            .and_then(|blocks| blocks.checked_mul(block_bytes))
            .ok_or(Error::Overflow("tensor descriptor byte length"))?;
        if source_byte_len != tensor.byte_len {
            return Err(Error::tensor(
                &tensor.name,
                format!(
                    "descriptor declares {} encoded bytes but its shape and type require {source_byte_len}",
                    tensor.byte_len
                ),
            ));
        }
        let selected_axis_multiple = if gguf_dimension == 0 { block_values } else { 1 };
        let selected_axis_multiple_usize = usize::try_from(selected_axis_multiple)
            .map_err(|_| Error::Overflow("selection alignment"))?;

        let (selected_values, encoded_ranges) = match &selection {
            TensorSelection::Range { start, end, .. } => {
                if start >= end || *end > dimension {
                    return Err(Error::tensor(
                        &tensor.name,
                        format!(
                            "selection range {start}..{end} exceeds row-major axis {logical_axis} dimension {dimension}"
                        ),
                    ));
                }
                if start % selected_axis_multiple_usize != 0
                    || end % selected_axis_multiple_usize != 0
                {
                    return Err(Error::tensor(
                        &tensor.name,
                        format!(
                            "selection range {start}..{end} on row-major axis {logical_axis} must align to {selected_axis_multiple}-value GGUF blocks"
                        ),
                    ));
                }
                let encoded_start = start / selected_axis_multiple_usize;
                let encoded_end = end / selected_axis_multiple_usize;
                (
                    end - start,
                    vec![(encoded_start, encoded_end - encoded_start)],
                )
            }
            TensorSelection::Indices { indices, .. } => {
                if indices.is_empty() || indices.iter().any(|index| *index >= dimension) {
                    return Err(Error::tensor(
                        &tensor.name,
                        format!(
                            "selection indices {indices:?} exceed row-major axis {logical_axis} dimension {dimension}"
                        ),
                    ));
                }
                let encoded_indices = if selected_axis_multiple_usize == 1 {
                    indices.clone()
                } else {
                    if !indices.len().is_multiple_of(selected_axis_multiple_usize) {
                        return Err(Error::tensor(
                            &tensor.name,
                            format!(
                                "selection indices on row-major axis {logical_axis} must contain complete {selected_axis_multiple}-value GGUF blocks"
                            ),
                        ));
                    }
                    let mut blocks =
                        Vec::with_capacity(indices.len() / selected_axis_multiple_usize);
                    for chunk in indices.chunks_exact(selected_axis_multiple_usize) {
                        let start = chunk[0];
                        if start % selected_axis_multiple_usize != 0
                            || chunk
                                .iter()
                                .copied()
                                .ne(start..start + selected_axis_multiple_usize)
                        {
                            return Err(Error::tensor(
                                &tensor.name,
                                format!(
                                    "selection indices on row-major axis {logical_axis} must preserve every complete aligned {selected_axis_multiple}-value GGUF block"
                                ),
                            ));
                        }
                        blocks.push(start / selected_axis_multiple_usize);
                    }
                    blocks
                };
                (indices.len(), coalesce_indices(&encoded_indices))
            }
        };

        let mut encoded_dimensions = tensor.dimensions.clone();
        encoded_dimensions[0] /= block_values;
        let inner_units =
            encoded_dimensions[..gguf_dimension]
                .iter()
                .try_fold(1u64, |product, dimension| {
                    product
                        .checked_mul(*dimension)
                        .ok_or(Error::Overflow("selection inner stride"))
                })?;
        let inner_bytes = inner_units
            .checked_mul(block_bytes)
            .ok_or(Error::Overflow("selection inner byte stride"))?;
        let repetition_stride = encoded_dimensions[gguf_dimension]
            .checked_mul(inner_bytes)
            .ok_or(Error::Overflow("selection repetition stride"))?;
        let mut repetitions = encoded_dimensions[gguf_dimension + 1..].iter().try_fold(
            1u64,
            |product, dimension| {
                product
                    .checked_mul(*dimension)
                    .ok_or(Error::Overflow("selection repetition count"))
            },
        )?;
        let mut relative_spans = encoded_ranges
            .into_iter()
            .map(|(start, count)| {
                let start =
                    u64::try_from(start).map_err(|_| Error::Overflow("selection span offset"))?;
                let count =
                    u64::try_from(count).map_err(|_| Error::Overflow("selection span length"))?;
                Ok(RelativeEncodedSpan {
                    offset: start
                        .checked_mul(inner_bytes)
                        .ok_or(Error::Overflow("selection span offset"))?,
                    byte_len: count
                        .checked_mul(inner_bytes)
                        .ok_or(Error::Overflow("selection span length"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let full_canonical_selection = relative_spans.len() == 1
            && relative_spans[0].offset == 0
            && relative_spans[0].byte_len == repetition_stride;
        if full_canonical_selection {
            repetitions = 1;
            relative_spans[0].byte_len = tensor.byte_len;
        }
        let encoded_bytes_per_repetition =
            relative_spans.iter().try_fold(0u64, |total, span| {
                total
                    .checked_add(span.byte_len)
                    .ok_or(Error::Overflow("selected tensor byte length"))
            })?;
        let encoded_byte_len = encoded_bytes_per_repetition
            .checked_mul(repetitions)
            .ok_or(Error::Overflow("selected tensor byte length"))?;

        let mut selected_descriptor = tensor.clone();
        selected_descriptor.dimensions[gguf_dimension] = u64::try_from(selected_values)
            .map_err(|_| Error::Overflow("selected tensor dimension"))?;
        selected_descriptor.byte_len = encoded_byte_len;
        let expected_byte_len = selected_descriptor
            .element_count()?
            .checked_div(block_values)
            .and_then(|blocks| blocks.checked_mul(block_bytes))
            .ok_or(Error::Overflow("selected tensor descriptor byte length"))?;
        if expected_byte_len != encoded_byte_len {
            return Err(Error::tensor(
                &tensor.name,
                format!(
                    "selection plan produced {encoded_byte_len} encoded bytes but its rewritten descriptor requires {expected_byte_len}"
                ),
            ));
        }
        let maximum_relative_end = relative_spans.iter().try_fold(0u64, |maximum, span| {
            let end = span
                .offset
                .checked_add(span.byte_len)
                .ok_or(Error::Overflow("selection span end"))?;
            Ok::<_, Error>(maximum.max(end))
        })?;
        if maximum_relative_end > repetition_stride && !full_canonical_selection {
            return Err(Error::tensor(
                &tensor.name,
                "selection span exceeds its encoded repetition stride",
            ));
        }
        let final_span_end = tensor
            .data_offset
            .checked_add(
                repetition_stride
                    .checked_mul(repetitions.saturating_sub(1))
                    .ok_or(Error::Overflow("selection final span offset"))?,
            )
            .and_then(|offset| offset.checked_add(maximum_relative_end))
            .ok_or(Error::Overflow("selection final span end"))?;
        let tensor_end = tensor
            .data_offset
            .checked_add(tensor.byte_len)
            .ok_or(Error::Overflow("tensor end offset"))?;
        if final_span_end > tensor_end {
            return Err(Error::tensor(
                &tensor.name,
                "selection plan exceeds the encoded tensor payload",
            ));
        }

        Ok(Self {
            selection,
            gguf_dimension,
            alignment: SelectionAlignment {
                block_values,
                block_bytes,
                selected_axis_multiple,
            },
            selected_descriptor,
            source_data_offset: tensor.data_offset,
            repetition_stride,
            repetitions,
            relative_spans,
            encoded_byte_len,
        })
    }

    /// Original logical selection represented by this plan.
    pub const fn selection(&self) -> &TensorSelection {
        &self.selection
    }

    /// Selected logical row-major axis.
    pub fn logical_axis(&self) -> usize {
        self.selection.axis()
    }

    /// Corresponding fastest-first GGUF dimension.
    pub const fn gguf_dimension(&self) -> usize {
        self.gguf_dimension
    }

    /// Native block geometry and selected-axis alignment.
    pub const fn alignment(&self) -> SelectionAlignment {
        self.alignment
    }

    /// Descriptor used to convert the compacted selected payload.
    pub const fn selected_descriptor(&self) -> &TensorDescriptor {
        &self.selected_descriptor
    }

    /// Exact number of encoded bytes read by the plan.
    pub const fn encoded_byte_len(&self) -> u64 {
        self.encoded_byte_len
    }

    /// Exact absolute encoded file spans in output order.
    pub fn encoded_spans(&self) -> impl Iterator<Item = EncodedSpan> + '_ {
        (0..self.repetitions).flat_map(move |repetition| {
            self.relative_spans.iter().map(move |span| EncodedSpan {
                offset: self.source_data_offset + repetition * self.repetition_stride + span.offset,
                byte_len: span.byte_len,
            })
        })
    }
}

#[derive(Debug)]
pub struct DenseTensorSpanPlan {
    selection: DenseTensorSpan,
    selected_descriptor: TensorDescriptor,
    encoded_span: EncodedSpan,
}
impl DenseTensorSpanPlan {
    /// Validates a block-aligned contiguous span without reading its payload.
    ///
    /// `selection` is expressed in the descriptor's physical scalar units.
    /// Native quantized spans must begin and end on complete GGML blocks.
    pub fn new(tensor: &TensorDescriptor, selection: DenseTensorSpan) -> Result<Self> {
        let tensor_elements = tensor.element_count()?;
        let (block_values, block_bytes) = tensor.ggml_type.block_and_bytes()?;
        if !tensor_elements.is_multiple_of(block_values) {
            return Err(Error::tensor(
                &tensor.name,
                format!(
                    "tensor element count {tensor_elements} is not divisible by {:?} block length {block_values}",
                    tensor.ggml_type
                ),
            ));
        }
        let expected_source_bytes = tensor_elements
            .checked_div(block_values)
            .and_then(|blocks| blocks.checked_mul(block_bytes))
            .ok_or(Error::Overflow("dense tensor byte length"))?;
        if expected_source_bytes != tensor.byte_len {
            return Err(Error::tensor(
                &tensor.name,
                format!(
                    "descriptor declares {} encoded bytes but its dense shape and type require {expected_source_bytes}",
                    tensor.byte_len
                ),
            ));
        }
        let selected_elements = selection.element_count()?;
        let selected_end = selection
            .offset_elements
            .checked_add(selected_elements)
            .ok_or(Error::Overflow("dense tensor span end"))?;
        if selected_end > tensor_elements {
            return Err(Error::tensor(
                &tensor.name,
                format!(
                    "contiguous scalar span {}..{selected_end} exceeds tensor element count {tensor_elements}",
                    selection.offset_elements
                ),
            ));
        }
        if !selection.offset_elements.is_multiple_of(block_values)
            || !selected_elements.is_multiple_of(block_values)
        {
            return Err(Error::tensor(
                &tensor.name,
                format!(
                    "contiguous span {}..{selected_end} must align to {:?} block length {block_values}",
                    selection.offset_elements, tensor.ggml_type
                ),
            ));
        }
        let fastest = selection.shape.last().copied().ok_or_else(|| {
            Error::tensor(&tensor.name, "contiguous span has no fastest dimension")
        })?;
        if !fastest.is_multiple_of(block_values) {
            return Err(Error::tensor(
                &tensor.name,
                format!(
                    "contiguous span fastest dimension {fastest} must align to {:?} block length {block_values}",
                    tensor.ggml_type
                ),
            ));
        }
        let byte_offset = selection
            .offset_elements
            .checked_div(block_values)
            .and_then(|blocks| blocks.checked_mul(block_bytes))
            .ok_or(Error::Overflow("dense tensor span byte offset"))?;
        let byte_len = selected_elements
            .checked_div(block_values)
            .and_then(|blocks| blocks.checked_mul(block_bytes))
            .ok_or(Error::Overflow("dense tensor span byte length"))?;
        let offset = tensor
            .data_offset
            .checked_add(byte_offset)
            .ok_or(Error::Overflow("dense tensor span file offset"))?;
        let end = offset
            .checked_add(byte_len)
            .ok_or(Error::Overflow("dense tensor span file end"))?;
        let tensor_end = tensor
            .data_offset
            .checked_add(tensor.byte_len)
            .ok_or(Error::Overflow("tensor end offset"))?;
        if end > tensor_end {
            return Err(Error::tensor(
                &tensor.name,
                "contiguous scalar span exceeds the encoded tensor payload",
            ));
        }

        let mut selected_descriptor = tensor.clone();
        selected_descriptor.dimensions = selection.shape.iter().rev().copied().collect();
        selected_descriptor.relative_offset = selected_descriptor
            .relative_offset
            .checked_add(byte_offset)
            .ok_or(Error::Overflow("dense tensor span relative offset"))?;
        selected_descriptor.data_offset = offset;
        selected_descriptor.byte_len = byte_len;
        Ok(Self {
            selection,
            selected_descriptor,
            encoded_span: EncodedSpan { offset, byte_len },
        })
    }

    /// Original logical span represented by this plan.
    pub const fn selection(&self) -> &DenseTensorSpan {
        &self.selection
    }

    /// Descriptor used to convert the compact selected payload.
    pub const fn selected_descriptor(&self) -> &TensorDescriptor {
        &self.selected_descriptor
    }

    /// Exact physical file range read by the plan.
    pub const fn encoded_span(&self) -> EncodedSpan {
        self.encoded_span
    }

    /// Exact number of encoded bytes read by the plan.
    pub const fn encoded_byte_len(&self) -> u64 {
        self.encoded_span.byte_len
    }
}

fn coalesce_indices(indices: &[usize]) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    for &index in indices {
        match ranges.last_mut() {
            Some((start, count)) if *start + *count == index => *count += 1,
            _ => ranges.push((index, 1)),
        }
    }
    ranges
}
