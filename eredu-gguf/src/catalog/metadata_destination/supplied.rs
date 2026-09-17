//! Final supplied metadata, including the catalog's actual ordered names.
use super::*;
use crate::reader::plan_storage::{axis_plan_view, span_plan_view, DescriptorSlot};
use crate::{StorageFamily, StorageProvider, StoredBuffer, StoredDescriptor, SuppliedStorageError};

pub(super) fn axis_requests(selection: &TensorSelection) -> (usize, usize) {
    match selection {
        TensorSelection::Range { .. } => (0, 1),
        TensorSelection::Indices { indices, .. } => (indices.len(), indices.len()),
    }
}

/// The current exhaustive GGUF conversion dispatch emits at most three outputs.
/// These are actual catalog names, not names reconstructed by a native backend.
#[derive(Debug)]
pub struct StoredOutputNames<F: StorageFamily> {
    names: [StoredBuffer<F, u8>; 3],
    count: usize,
}
impl<F: StorageFamily> StoredOutputNames<F> {
    /// Number of actual physical outputs in catalog order.
    pub fn len(&self) -> usize {
        self.count
    }
    /// Whether this catalog group has no outputs.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    /// Borrow a name from its intact allocation owner.
    pub fn get(&self, index: usize) -> Option<&str> {
        (index < self.count).then(|| {
            std::str::from_utf8(self.names[index].as_slice())
                .expect("name copied from catalog UTF-8")
        })
    }
    /// Borrow all names in the unchanged source order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &str> {
        (0..self.count).map(|i| self.get(i).expect("bounded actual name"))
    }
}
#[derive(Debug)]
enum BoundAxis<F: StorageFamily> {
    Range {
        axis: usize,
        start: usize,
        end: usize,
    },
    Indices {
        axis: usize,
        indices: StoredBuffer<F, usize>,
    },
}
#[derive(Debug)]
enum StoredSelection<F: StorageFamily> {
    Full,
    Axis {
        bound: BoundAxis<F>,
        indices: StoredBuffer<F, usize>,
        ranges: StoredBuffer<F, (usize, usize)>,
        dimensions: StoredBuffer<F, u64>,
        spans: StoredBuffer<F, crate::reader::RelativeEncodedSpan>,
        selected: StoredDescriptor<F>,
    },
    Span {
        offset: u64,
        shape: StoredBuffer<F, u64>,
        selected: StoredDescriptor<F>,
    },
}
/// All final result metadata and plan scratch retain their supplied owners.
#[derive(Debug)]
pub struct StoredTensorMetadata<F: StorageFamily> {
    base: StoredDescriptor<F>,
    final_descriptor: StoredDescriptor<F>,
    names: StoredOutputNames<F>,
    selection: StoredSelection<F>,
    location: TensorLocation,
    endian: Endian,
    used: bool,
    completed: bool,
    pair_attempted: bool,
}
/// Metadata preparation failure, retaining every successful allocation prefix.
#[derive(Debug)]
pub struct StoredMetadataFailure<F: StorageFamily> {
    cause: SuppliedStorageError<F::Error>,
    destination: StoredTensorMetadata<F>,
}
impl<F: StorageFamily> StoredMetadataFailure<F> {
    /// Actual provider/metadata cause and the intact partial destination.
    pub fn into_parts(self) -> (SuppliedStorageError<F::Error>, StoredTensorMetadata<F>) {
        (self.cause, self.destination)
    }
    /// Typed source-preserving cause.
    pub fn cause(&self) -> &SuppliedStorageError<F::Error> {
        &self.cause
    }
    /// Actual retained metadata prefix.
    pub fn destination(&self) -> &StoredTensorMetadata<F> {
        &self.destination
    }
}
impl<F: StorageFamily> std::fmt::Display for StoredMetadataFailure<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.cause.fmt(f)
    }
}
impl<F: StorageFamily> std::error::Error for StoredMetadataFailure<F> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
impl<F: StorageFamily> StoredTensorMetadata<F> {
    /// Prepare the actual materializer's catalog at the existing G3 boundary.
    /// No reader opens and no physical validation plan is used to size storage.
    pub fn prepare<P: StorageProvider<Family = F>>(
        source: TensorMetadataSource<'_>,
        selection: MetadataSelection<'_>,
        provider: &mut P,
    ) -> std::result::Result<Self, StoredMetadataFailure<F>> {
        let mut out = Self {
            base: StoredDescriptor::default(),
            final_descriptor: StoredDescriptor::default(),
            names: StoredOutputNames {
                names: std::array::from_fn(|_| StoredBuffer::default()),
                count: 0,
            },
            selection: StoredSelection::Full,
            location: source.location,
            endian: source.endian,
            used: false,
            completed: false,
            pair_attempted: false,
        };
        let result = (|| {
            // Preserve the existing all-layout validation before the first reserve.
            source
                .layouts(selection)
                .ok_or(MetadataDestinationError::Layout)?;
            let tensor = &source.tensor.descriptor;
            let rank = tensor.dimensions.len();
            out.base.prepare(provider, tensor.view(), rank)?;
            crate::supplied_storage::metadata_capacity(
                "output names",
                source.tensor.outputs.len(),
                3,
            )?;
            out.names.count = source.tensor.outputs.len();
            for (target, output) in out.names.names.iter_mut().zip(&source.tensor.outputs) {
                target.prepare(provider, output.name.len(), 0)?;
                target.copy_metadata(output.name.as_bytes(), "output name")?;
            }
            match selection {
                MetadataSelection::Full => {}
                MetadataSelection::Axis(selection) => {
                    let bound = match selection {
                        TensorSelection::Range { axis, start, end } => BoundAxis::Range {
                            axis: *axis,
                            start: *start,
                            end: *end,
                        },
                        TensorSelection::Indices { axis, .. } => BoundAxis::Indices {
                            axis: *axis,
                            indices: StoredBuffer::default(),
                        },
                    };
                    out.selection = StoredSelection::Axis {
                        bound,
                        indices: StoredBuffer::default(),
                        ranges: StoredBuffer::default(),
                        dimensions: StoredBuffer::default(),
                        spans: StoredBuffer::default(),
                        selected: StoredDescriptor::default(),
                    };
                    let StoredSelection::Axis {
                        bound,
                        indices,
                        ranges,
                        dimensions,
                        spans,
                        selected,
                    } = &mut out.selection
                    else {
                        unreachable!()
                    };
                    if let (
                        BoundAxis::Indices {
                            indices: target, ..
                        },
                        TensorSelection::Indices {
                            indices: values, ..
                        },
                    ) = (bound, selection)
                    {
                        target.prepare(provider, values.len(), 0)?;
                        target.copy_metadata(values, "retained selection")?;
                    }
                    let (count, range_count) = axis_requests(selection);
                    indices.prepare(provider, count, 0)?;
                    ranges.prepare(provider, range_count, (0, 0))?;
                    dimensions.prepare(provider, rank, 0)?;
                    spans.prepare(
                        provider,
                        range_count,
                        crate::reader::RelativeEncodedSpan::empty(),
                    )?;
                    selected.prepare(provider, tensor.view(), rank)?;
                    out.final_descriptor
                        .reserve_only(provider, &tensor.name, rank)?;
                }
                MetadataSelection::Span(selection) => {
                    out.selection = StoredSelection::Span {
                        offset: selection.offset_elements(),
                        shape: StoredBuffer::default(),
                        selected: StoredDescriptor::default(),
                    };
                    let StoredSelection::Span {
                        shape, selected, ..
                    } = &mut out.selection
                    else {
                        unreachable!()
                    };
                    shape.prepare(provider, selection.shape().len(), 0)?;
                    shape.copy_metadata(selection.shape(), "retained span")?;
                    selected.prepare(provider, tensor.view(), rank.max(selection.shape().len()))?;
                    out.final_descriptor.reserve_only(
                        provider,
                        &tensor.name,
                        selection.shape().len(),
                    )?;
                }
            }
            Ok::<_, SuppliedStorageError<F::Error>>(())
        })();
        match result {
            Ok(()) => Ok(out),
            Err(cause) => Err(StoredMetadataFailure {
                cause,
                destination: out,
            }),
        }
    }
    pub(in crate::catalog) fn policy(&mut self) -> MetadataPolicy<'_> {
        let selection = match &mut self.selection {
            StoredSelection::Full => SelectionLoan::Full,
            StoredSelection::Axis {
                bound,
                indices,
                ranges,
                dimensions,
                spans,
                selected,
            } => {
                let bound = match bound {
                    BoundAxis::Range { axis, start, end } => AxisBinding::Range {
                        axis: *axis,
                        start: *start,
                        end: *end,
                    },
                    BoundAxis::Indices { axis, indices } => AxisBinding::Indices {
                        axis: *axis,
                        indices: indices.as_slice(),
                    },
                };
                SelectionLoan::Axis {
                    bound,
                    storage: AxisStorage::fixed(
                        indices.loan(),
                        ranges.loan(),
                        dimensions.loan(),
                        spans.loan(),
                        selected.loan(),
                    ),
                }
            }
            StoredSelection::Span {
                offset,
                shape,
                selected,
            } => SelectionLoan::Span {
                offset: *offset,
                shape: shape.as_slice(),
                selected: selected.loan(),
            },
        };
        MetadataPolicy {
            base: Some(DescriptorSlot::Fixed(self.base.loan())),
            final_descriptor: Some(DescriptorSlot::Fixed(self.final_descriptor.loan())),
            names: None,
            selection: None,
            completed: None,
            binding: Some((self.location, self.endian, &mut self.used)),
            supplied_names: Some(std::array::from_fn(|i| self.names.get(i))),
            supplied_count: self.names.len(),
            supplied_selection: Some(selection),
        }
    }
    pub(crate) fn mark_completed(&mut self) {
        self.completed = true;
    }
    fn finish(self) -> (StoredDescriptor<F>, StoredOutputNames<F>) {
        let descriptor = match self.selection {
            StoredSelection::Full => self.base,
            _ => self.final_descriptor,
        };
        (descriptor, self.names)
    }
}

pub(super) enum AxisBinding<'a> {
    Range {
        axis: usize,
        start: usize,
        end: usize,
    },
    Indices {
        axis: usize,
        indices: &'a [usize],
    },
}
impl AxisBinding<'_> {
    fn matches(&self, selection: &TensorSelection) -> bool {
        match (self, selection) {
            (
                Self::Range {
                    axis: a,
                    start: b,
                    end: c,
                },
                TensorSelection::Range { axis, start, end },
            ) => a == axis && b == start && c == end,
            (
                Self::Indices {
                    axis: a,
                    indices: b,
                },
                TensorSelection::Indices { axis, indices },
            ) => a == axis && *b == indices.as_slice(),
            _ => false,
        }
    }
}
pub(super) enum SelectionLoan<'a> {
    Full,
    Axis {
        bound: AxisBinding<'a>,
        storage: AxisStorage<'a>,
    },
    Span {
        offset: u64,
        shape: &'a [u64],
        selected: crate::supplied_storage::FixedDescriptor<'a>,
    },
}
impl<'a> SelectionLoan<'a> {
    pub(super) fn axis(
        self,
        tensor: crate::TensorDescriptorView<'_>,
        selection: &TensorSelection,
    ) -> std::result::Result<AxisMetadataPlan<'a>, MetadataDestinationError> {
        match self {
            Self::Axis { bound, storage } if bound.matches(selection) => Ok(
                AxisMetadataPlan::Prepared(axis_plan_view(tensor, selection, storage)?),
            ),
            _ => Err(MetadataDestinationError::Binding),
        }
    }
    pub(super) fn span(
        self,
        tensor: crate::TensorDescriptorView<'_>,
        selection: &DenseTensorSpan,
    ) -> std::result::Result<SpanMetadataPlan<'a>, MetadataDestinationError> {
        match self {
            Self::Span {
                offset,
                shape,
                selected,
            } if offset == selection.offset_elements() && shape == selection.shape() => {
                Ok(SpanMetadataPlan::Prepared(span_plan_view(
                    tensor,
                    selection,
                    Some(DescriptorSlot::Fixed(selected)),
                )?))
            }
            _ => Err(MetadataDestinationError::Binding),
        }
    }
}

/// A converted group with all final metadata and payload allocation owners intact.
#[derive(Debug)]
pub struct StoredCheckpointTensor<F: StorageFamily> {
    /// Actual retained catalog shard coordinate.
    pub shard_index: usize,
    /// Actual retained catalog physical tensor coordinate.
    pub tensor_index: usize,
    /// Selected descriptor, with supplied name and dimensions.
    pub descriptor: StoredDescriptor<F>,
    /// Actual companion names, in original catalog order.
    pub output_names: StoredOutputNames<F>,
    /// Completed shared conversion output, with emitted lengths.
    pub converted: crate::StoredConvertedTensor<F>,
}
/// One private pair of fresh conversion and metadata destinations. The shared
/// materializer completes both members together; completed members cannot be
/// extracted and combined into a different successful result.
#[derive(Debug)]
pub struct StoredTensorPair<F: StorageFamily> {
    // Output metadata retires before conversion, including every failed prefix.
    metadata: StoredTensorMetadata<F>,
    conversion: crate::StoredConversion<F>,
}
impl<F: StorageFamily> StoredTensorPair<F> {
    /// Pair only unused destinations. Refusal preserves both supplied owners,
    /// even if equal descriptors describe different completed selections.
    pub fn try_new(
        metadata: StoredTensorMetadata<F>,
        conversion: crate::StoredConversion<F>,
    ) -> std::result::Result<Self, (StoredTensorMetadata<F>, crate::StoredConversion<F>)> {
        if metadata.used || metadata.completed || metadata.pair_attempted || conversion.started() {
            return Err((metadata, conversion));
        }
        Ok(Self {
            metadata,
            conversion,
        })
    }
    /// Retain or inspect failed owners independently. Used members cannot be
    /// paired again, and there is no separate public metadata/result join.
    pub fn into_parts(self) -> (StoredTensorMetadata<F>, crate::StoredConversion<F>) {
        (self.metadata, self.conversion)
    }
    /// Whether completed conversion remains under a later failed metadata owner.
    pub fn conversion_completed(&self) -> bool {
        self.conversion.completed()
    }
    pub(in crate::catalog) fn loans(
        &mut self,
    ) -> std::result::Result<
        (
            &mut StoredTensorMetadata<F>,
            &mut crate::StoredConversion<F>,
        ),
        MetadataDestinationError,
    > {
        if self.metadata.pair_attempted || self.conversion.started() {
            return Err(MetadataDestinationError::Used);
        }
        // Sticky before lookup/open/read, including an error before conversion.
        // Both extracted members independently refuse any later fresh pairing.
        self.metadata.pair_attempted = true;
        self.conversion.mark_pair_attempted();
        Ok((&mut self.metadata, &mut self.conversion))
    }
    /// Move the exact jointly completed owners. A failed/incomplete pair is
    /// returned intact; no external conversion can be substituted at this point.
    pub fn into_result(self) -> std::result::Result<StoredCheckpointTensor<F>, Self> {
        if !self.metadata.completed || !self.conversion.completed() {
            return Err(self);
        }
        let location = self.metadata.location;
        let (descriptor, output_names) = self.metadata.finish();
        Ok(StoredCheckpointTensor {
            shard_index: location.shard_index,
            tensor_index: location.tensor_index,
            descriptor,
            output_names,
            converted: self.conversion.finish(),
        })
    }
}

#[cfg(test)]
mod tests;
