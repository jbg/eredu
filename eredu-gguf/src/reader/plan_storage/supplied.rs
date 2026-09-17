//! Demand-driven supplied slots at the original selected-descriptor boundaries.
use super::*;
use crate::{
    MetadataSelection, StorageFamily, StorageProvider, StoredBuffer, StoredDescriptor,
    SuppliedStorageError,
};
use std::{cell::RefCell, fmt};

/// Owns the selected descriptor and every reached physical-plan allocation.
/// Source custody is supplied by the surrounding checkpoint lease owner.
#[derive(Debug)]
pub struct StoredPhysicalDescriptor<F: StorageFamily> {
    descriptor: StoredDescriptor<F>,
    indices: StoredBuffer<F, usize>,
    ranges: StoredBuffer<F, (usize, usize)>,
    dimensions: StoredBuffer<F, u64>,
    spans: StoredBuffer<F, RelativeEncodedSpan>,
}
/// The original semantic/provider cause and the entire selected-plan prefix.
#[derive(Debug)]
pub struct StoredPhysicalFailure<F: StorageFamily> {
    cause: SuppliedStorageError<F::Error>,
    destination: StoredPhysicalDescriptor<F>,
}
impl<F: StorageFamily> StoredPhysicalFailure<F> {
    /// Move cause and prefix without releasing the actual source owner's data.
    pub fn into_parts(self) -> (SuppliedStorageError<F::Error>, StoredPhysicalDescriptor<F>) {
        (self.cause, self.destination)
    }
    /// Actual cause, retaining the provider's typed source.
    pub fn cause(&self) -> &SuppliedStorageError<F::Error> {
        &self.cause
    }
    /// Actual partial descriptor/scratch owners.
    pub fn destination(&self) -> &StoredPhysicalDescriptor<F> {
        &self.destination
    }
}
impl<F: StorageFamily> fmt::Display for StoredPhysicalFailure<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl<F: StorageFamily> std::error::Error for StoredPhysicalFailure<F> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
struct PhysicalRequests {
    rank: usize,
    descriptor_rank: usize,
    axis: Option<(usize, usize)>,
}
impl PhysicalRequests {
    fn new(source: &TensorDescriptor, selection: MetadataSelection<'_>) -> Self {
        let rank = source.dimensions.len();
        let descriptor_rank = match selection {
            MetadataSelection::Span(span) => rank.max(span.shape().len()),
            _ => rank,
        };
        let axis = match selection {
            MetadataSelection::Axis(TensorSelection::Range { .. }) => Some((0, 1)),
            MetadataSelection::Axis(TensorSelection::Indices { indices, .. }) => {
                Some((indices.len(), indices.len()))
            }
            _ => None,
        };
        Self {
            rank,
            descriptor_rank,
            axis,
        }
    }
    fn bound(&self, name: usize) -> Option<crate::StorageRequestBound> {
        let mut out = crate::StorageRequestBound::default();
        out.descriptor(name, self.descriptor_rank)?;
        if let Some((indices, ranges)) = self.axis {
            out.add::<usize>(indices)?;
            out.add::<(usize, usize)>(ranges)?;
            out.add::<u64>(self.rank)?;
            out.add::<RelativeEncodedSpan>(ranges)?;
        }
        Some(out)
    }
}
/// Source-derived ceiling for every actual lazy physical descriptor/scratch slot.
/// No planner, provider or allocation runs; early errors may reach fewer slots.
pub(crate) fn physical_storage_bound(
    source: &TensorDescriptor,
    selection: MetadataSelection<'_>,
) -> Option<crate::StorageRequestBound> {
    PhysicalRequests::new(source, selection).bound(source.name.len())
}
impl<F: StorageFamily> StoredPhysicalDescriptor<F> {
    /// Execute the original full/axis/span descriptor preparation using supplied
    /// slots only at reached allocation points. No allocating sizing plan runs.
    pub fn prepare<P: StorageProvider<Family = F>>(
        source: &TensorDescriptor,
        selection: MetadataSelection<'_>,
        provider: &mut P,
    ) -> std::result::Result<Self, StoredPhysicalFailure<F>> {
        let mut out = Self {
            descriptor: StoredDescriptor::default(),
            indices: StoredBuffer::default(),
            ranges: StoredBuffer::default(),
            dimensions: StoredBuffer::default(),
            spans: StoredBuffer::default(),
        };
        let result = {
            let demand = Demand {
                provider: RefCell::new(provider),
                cause: RefCell::new(None),
            };
            let requests = PhysicalRequests::new(source, selection);
            let rank = requests.rank;
            let descriptor_rank = requests.descriptor_rank;
            let mut descriptor = LazyDescriptor {
                owner: &mut out.descriptor,
                demand: &demand,
                rank: descriptor_rank,
            };
            let result = match selection {
                MetadataSelection::Full => descriptor.copy_from(source.view()),
                MetadataSelection::Axis(selection) => {
                    let (count, range_count) = requests.axis.expect("axis requests");
                    let mut indices = LazyBuffer {
                        owner: &mut out.indices,
                        demand: &demand,
                        limit: count,
                        initial: 0,
                    };
                    let mut ranges = LazyBuffer {
                        owner: &mut out.ranges,
                        demand: &demand,
                        limit: range_count,
                        initial: (0, 0),
                    };
                    let mut dimensions = LazyBuffer {
                        owner: &mut out.dimensions,
                        demand: &demand,
                        limit: rank,
                        initial: 0,
                    };
                    let mut spans = LazyBuffer {
                        owner: &mut out.spans,
                        demand: &demand,
                        limit: range_count,
                        initial: RelativeEncodedSpan {
                            offset: 0,
                            byte_len: 0,
                        },
                    };
                    axis_plan_view(
                        source.view(),
                        selection,
                        AxisStorage::lazy(
                            &mut indices,
                            &mut ranges,
                            &mut dimensions,
                            &mut spans,
                            &mut descriptor,
                        ),
                    )
                    .map(drop)
                }
                MetadataSelection::Span(selection) => span_plan_view(
                    source.view(),
                    selection,
                    Some(DescriptorSlot::Lazy(&mut descriptor)),
                )
                .map(drop),
            };
            drop(descriptor);
            match demand.cause.into_inner() {
                Some(cause) => Err(cause),
                None => result.map_err(SuppliedStorageError::Metadata),
            }
        };
        match result {
            Ok(()) => Ok(out),
            Err(cause) => Err(StoredPhysicalFailure {
                cause,
                destination: out,
            }),
        }
    }
    /// Borrow the actual selected fields after the shared planner has succeeded.
    pub fn descriptor(&self) -> TensorDescriptorView<'_> {
        self.descriptor.view()
    }
    /// Move only the selected descriptor forward; completed scratch retires now.
    /// Its bank is cumulative and receives no allocation/attempt refund.
    pub fn into_descriptor(self) -> StoredDescriptor<F> {
        self.descriptor
    }
}

struct Demand<'a, P: StorageProvider> {
    provider: RefCell<&'a mut P>,
    cause: RefCell<Option<SuppliedStorageError<<P::Family as StorageFamily>::Error>>>,
}
impl<P: StorageProvider> Demand<'_, P> {
    fn allocate<T: Copy + fmt::Debug + 'static>(
        &self,
        owner: &mut StoredBuffer<P::Family, T>,
        count: usize,
        initial: T,
    ) -> PlanResult<()> {
        let result = owner.prepare(&mut **self.provider.borrow_mut(), count, initial);
        match result {
            Ok(()) => Ok(()),
            Err(cause) => {
                let old = self.cause.replace(Some(cause));
                debug_assert!(old.is_none());
                Err(MetadataDestinationError::StorageRefused)
            }
        }
    }
    fn descriptor(
        &self,
        owner: &mut StoredDescriptor<P::Family>,
        source: TensorDescriptorView<'_>,
        rank: usize,
    ) -> PlanResult<()> {
        let result = owner.prepare(&mut **self.provider.borrow_mut(), source, rank);
        match result {
            Ok(()) => Ok(()),
            Err(cause) => {
                let old = self.cause.replace(Some(cause));
                debug_assert!(old.is_none());
                Err(MetadataDestinationError::StorageRefused)
            }
        }
    }
}
struct LazyBuffer<'a, 'd, P: StorageProvider, T: Copy + fmt::Debug + 'static> {
    owner: &'a mut StoredBuffer<P::Family, T>,
    demand: &'a Demand<'d, P>,
    limit: usize,
    initial: T,
}
impl<P: StorageProvider, T: Copy + fmt::Debug + 'static> LazyBuffer<'_, '_, P, T> {
    fn allocate(&mut self) -> PlanResult<()> {
        if !self.owner.prepared() {
            self.demand.allocate(self.owner, self.limit, self.initial)?;
        }
        Ok(())
    }
}
impl<P: StorageProvider, T: Copy + fmt::Debug + 'static> PlanBuffer<T>
    for LazyBuffer<'_, '_, P, T>
{
    fn prepare(&mut self, requested: Option<usize>, field: &'static str) -> PlanResult<()> {
        if let Some(required) = requested {
            capacity(field, required, self.limit)?;
            self.allocate()?;
        }
        self.owner.loan().clear();
        Ok(())
    }
    fn push(&mut self, value: T, field: &'static str) -> PlanResult<()> {
        self.allocate()?;
        let loan = self.owner.loan();
        let len = loan
            .used
            .checked_add(1)
            .ok_or(MetadataDestinationError::Layout)?;
        capacity(field, len, loan.limit.min(loan.data.len()))?;
        loan.data[*loan.used] = value;
        *loan.used = len;
        Ok(())
    }
    fn as_slice(&self) -> &[T] {
        self.owner.as_slice()
    }
    fn as_mut_slice(&mut self) -> &mut [T] {
        self.owner.emitted_mut()
    }
}
struct LazyDescriptor<'a, 'd, P: StorageProvider> {
    owner: &'a mut StoredDescriptor<P::Family>,
    demand: &'a Demand<'d, P>,
    rank: usize,
}
impl<P: StorageProvider> DescriptorBuffer for LazyDescriptor<'_, '_, P> {
    fn copy_from(&mut self, source: TensorDescriptorView<'_>) -> PlanResult<()> {
        self.demand.descriptor(self.owner, source, self.rank)
    }
    fn view(&self) -> TensorDescriptorView<'_> {
        self.owner.view()
    }
    fn dimensions_mut(&mut self) -> &mut [u64] {
        self.owner.dimensions.emitted_mut()
    }
    fn set_byte_len(&mut self, value: u64) {
        self.owner.byte_len = value;
    }
    fn set_offsets(&mut self, relative: u64, data: u64) {
        self.owner.relative_offset = relative;
        self.owner.data_offset = data;
    }
    fn replace_dimensions(&mut self, shape: &[u64]) -> PlanResult<()> {
        self.owner
            .dimensions
            .replace_reversed(shape, "span dimensions")
    }
}
