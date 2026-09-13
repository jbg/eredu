//! Provider-owned coordinates for sparse grouped-unit evidence.
use super::RoutedExpertRequest;
use eredu_nn::{Error, GroupedUnitBatch, GroupedUnitObserver, Tensor};
#[cfg(test)]
mod tests;

/// Actual units together with their provider's original route identities.
/// The native batch may use a compact bank and a smaller token chunk. Original
/// route IDs and the retained group map identify checkpoint-global experts
/// without materializing an expanded activation tensor.
pub struct RoutedUnitBatch<'a, T> {
    /// Actual selected values and native sorted-route coordinates.
    pub units: GroupedUnitBatch<'a, T>,
    /// Pre-compaction route IDs, `[provider_tokens, top_k]`.
    pub source_groups: &'a T,
    /// Maps source group IDs to checkpoint-global expert IDs. Absent means the
    /// source IDs already have global meaning.
    pub global_groups: Option<&'a [usize]>,
    /// Provider chunk start in `source_groups`; add the native batch token offset
    /// and each native token index to recover the source token row.
    pub provider_token_offset: usize,
    /// Original peer/token/slot identity when the provider consumes exchanged rows.
    pub origins: Option<RoutedUnitOrigins<'a>>,
    /// Retained within-expert scalar coordinates when this invocation is
    /// partitioned. Absence alone is not a claim of global distributed coverage.
    pub unit_coordinates: Option<&'a eredu_core::component::ComponentCoordinateMap>,
}

impl<T> RoutedUnitBatch<'_, T> {
    /// Borrows the existing native inputs for bounded capture without retaining
    /// arrays, inspecting shapes or copying provider metadata.
    pub fn capture_source(
        &self,
    ) -> Result<
        eredu_core::capture::RoutedUnitCaptureSource<'_, T>,
        eredu_core::capture::CaptureError,
    > {
        let token_offset = self
            .source_token(0)
            .and_then(|offset| u64::try_from(offset).ok())
            .ok_or(eredu_core::capture::CaptureError::Overflow)?;
        Ok(eredu_core::capture::RoutedUnitCaptureSource {
            values: self.units.values,
            token_indices: self.units.token_indices,
            selection_indices: self.units.selection_indices,
            coefficients: self.units.coefficients,
            source_groups: self.source_groups,
            token_offset,
            global_groups: self.global_groups,
        })
    }

    /// Borrows actual prepared placement and source origins for a partition
    /// collector. Retained expected ownership is supplied separately by admission.
    pub fn partition_capture_source(
        &self,
    ) -> Result<
        eredu_core::capture::PartitionRoutedUnitCaptureSource<'_, T>,
        eredu_core::capture::CaptureError,
    > {
        Ok(eredu_core::capture::PartitionRoutedUnitCaptureSource {
            source: self.capture_source()?,
            origins: self.origins.map(|origins| origins.capture_coordinates()),
            unit_coordinates: self.unit_coordinates.ok_or_else(|| {
                eredu_core::capture::CaptureError::Invalid(
                    "partitioned routed source has no prepared unit map".into(),
                )
            })?,
        })
    }

    /// Reborrows the same route data through a backend's transparent tensor view.
    /// This does not allocate, clone, retain, evaluate or change any coordinate.
    pub fn map_tensors<U>(&self, map: impl Fn(&T) -> &U) -> RoutedUnitBatch<'_, U> {
        RoutedUnitBatch {
            units: GroupedUnitBatch {
                values: map(self.units.values),
                group_indices: map(self.units.group_indices),
                selection_indices: map(self.units.selection_indices),
                token_indices: map(self.units.token_indices),
                coefficients: map(self.units.coefficients),
                token_offset: self.units.token_offset,
                total_token_count: self.units.total_token_count,
                group_count: self.units.group_count,
            },
            source_groups: map(self.source_groups),
            global_groups: self.global_groups,
            provider_token_offset: self.provider_token_offset,
            origins: self.origins,
            unit_coordinates: self.unit_coordinates,
        }
    }
    /// Resolves a source-route group ID without interpreting a compact native ID.
    pub fn global_group(&self, source_group: usize) -> Option<usize> {
        match self.global_groups {
            Some(groups) => groups.get(source_group).copied(),
            None => Some(source_group),
        }
    }

    /// Checked source token row for a native chunk-relative token index. The
    /// caller validates the result against the actual source tensor geometry.
    pub fn source_token(&self, native_token: usize) -> Option<usize> {
        self.provider_token_offset
            .checked_add(self.units.token_offset)?
            .checked_add(native_token)
    }
}

pub use eredu_core::capture::RoutedUnitOrigin;

/// Validated borrowed origin tags in peer-major exchange receive order.
#[derive(Debug, Clone, Copy)]
pub struct RoutedUnitOrigins<'a> {
    coordinates: eredu_core::capture::RoutedUnitOrigins<'a>,
}
impl<'a> RoutedUnitOrigins<'a> {
    /// Checks exact metadata coverage without creating additional host vectors.
    pub fn new(
        peer_counts: &'a [usize],
        route_positions: &'a [usize],
        routes_per_token: usize,
    ) -> Result<Self, Error> {
        Ok(Self {
            coordinates: eredu_core::capture::RoutedUnitOrigins::new(
                peer_counts,
                route_positions,
                routes_per_token,
            )
            .map_err(Error::backend_source)?,
        })
    }
    /// Lends the same checked host tags to a neutral bounded collector.
    pub const fn capture_coordinates(&self) -> eredu_core::capture::RoutedUnitOrigins<'a> {
        self.coordinates
    }
    /// Resolves one received row through the preserved source route tag.
    pub fn resolve(&self, row: usize) -> Option<RoutedUnitOrigin> {
        self.coordinates.resolve(row)
    }
}

impl<T: Tensor> RoutedUnitBatch<'_, T> {
    /// Resolves a native chunk-relative token and slot, including provider chunks
    /// and expert exchange. Tokens remain distinct from prediction ordinals.
    pub fn route_origin(
        &self,
        native_token: usize,
        native_slot: usize,
    ) -> Option<RoutedUnitOrigin> {
        let shape = self.source_groups.shape();
        let (&width, leading) = shape.split_last()?;
        let width = usize::try_from(width).ok()?;
        let rows = leading
            .iter()
            .try_fold(1usize, |n, &m| n.checked_mul(usize::try_from(m).ok()?))?;
        let token = self.source_token(native_token)?;
        if native_slot >= width || token >= rows {
            return None;
        }
        match self.origins {
            Some(origins) if width == 1 => origins.resolve(token),
            Some(_) => None,
            None => Some(RoutedUnitOrigin {
                source_peer: None,
                token,
                slot: native_slot,
            }),
        }
    }
}

/// Actual borrowed input to one local provider invocation. Exchanged input has
/// one native row per received route, including zero rows for an idle owner.
/// This metadata supplies neither submission nor distributed delivery authority.
pub struct RoutedUnitInvocation<'a, T> {
    /// Actual provider input, before any internal residency or native chunking.
    pub input: &'a T,
    /// Original source coordinates when this local input has been exchanged.
    pub origins: Option<RoutedUnitOrigins<'a>>,
    /// Actual prepared scalar-column order, when already bound by a bank adapter.
    /// An outer EP scope may precede that adapter; chunks still carry the map.
    pub unit_coordinates: Option<&'a eredu_core::component::ComponentCoordinateMap>,
}

/// Sparse routed observation/intervention supplied by an admitted execution owner.
/// Implementations reserve retention, host materialization and replacements before
/// doing that work. A local provider has not established distributed delivery.
pub trait RoutedUnitObserver<T> {
    /// Validates and prepares one actual local invocation before provider work.
    /// Implementations may use already prepaid source preparation and group votes.
    fn begin_invocation(&mut self, _invocation: &RoutedUnitInvocation<'_, T>) -> Result<(), Error> {
        Ok(())
    }
    /// Runs after provider work, including failed preparation and idle owners,
    /// before its caller can enter a reverse exchange or tensor reduction.
    /// `success` describes local provider execution, not native completion or
    /// committed delivery. Implementations establish their required failure vote.
    fn finish_invocation(&mut self, _success: bool) -> Result<(), Error> {
        Ok(())
    }
    /// Forwarding adapters preserve this flag so nested residency/bank wrappers
    /// cannot report the same local invocation twice. It grants no work authority.
    fn invocation_active(&self) -> bool {
        false
    }
    /// Borrows original units and exact source route identity.
    fn observe(&mut self, batch: &RoutedUnitBatch<'_, T>) -> Result<(), Error>;
    /// Replaces actual units before the down projection, with unchanged dtype/shape.
    fn intervene(&mut self, _batch: &RoutedUnitBatch<'_, T>) -> Result<Option<T>, Error> {
        Ok(None)
    }
    /// Borrows effective units before optional projection input quantization.
    fn observe_effective(&mut self, _batch: &RoutedUnitBatch<'_, T>) -> Result<(), Error> {
        Ok(())
    }
}

/// Borrows the provider's existing route tensors/maps for one native invocation.
/// No extra native tensor, host index vector, or capture is created by this adapter.
pub struct ProviderUnitObserver<'a, T: Tensor> {
    observer: &'a mut dyn RoutedUnitObserver<T>,
    source_groups: &'a T,
    global_groups: Option<&'a [usize]>,
    provider_token_offset: usize,
}
impl<'a, T: Tensor> ProviderUnitObserver<'a, T> {
    /// Establishes original route identity before a provider compacts its bank.
    pub fn new(
        observer: &'a mut dyn RoutedUnitObserver<T>,
        source_groups: &'a T,
        global_groups: Option<&'a [usize]>,
        provider_token_offset: usize,
    ) -> Self {
        Self {
            observer,
            source_groups,
            global_groups,
            provider_token_offset,
        }
    }
}

impl<T: Tensor> GroupedUnitObserver<T> for ProviderUnitObserver<'_, T> {
    fn observe(&mut self, batch: &GroupedUnitBatch<'_, T>) -> Result<(), Error> {
        self.observer.observe(&RoutedUnitBatch {
            units: batch.with_values(batch.values),
            source_groups: self.source_groups,
            global_groups: self.global_groups,
            provider_token_offset: self.provider_token_offset,
            origins: None,
            unit_coordinates: None,
        })
    }
    fn intervene(&mut self, batch: &GroupedUnitBatch<'_, T>) -> Result<Option<T>, Error> {
        self.observer.intervene(&RoutedUnitBatch {
            units: batch.with_values(batch.values),
            source_groups: self.source_groups,
            global_groups: self.global_groups,
            provider_token_offset: self.provider_token_offset,
            origins: None,
            unit_coordinates: None,
        })
    }
    fn observe_effective(&mut self, batch: &GroupedUnitBatch<'_, T>) -> Result<(), Error> {
        self.observer.observe_effective(&RoutedUnitBatch {
            units: batch.with_values(batch.values),
            source_groups: self.source_groups,
            global_groups: self.global_groups,
            provider_token_offset: self.provider_token_offset,
            origins: None,
            unit_coordinates: None,
        })
    }
}

struct ExchangedUnitObserver<'a, T: Tensor> {
    inner: &'a mut dyn RoutedUnitObserver<T>,
    origins: RoutedUnitOrigins<'a>,
}
impl<T: Tensor> RoutedUnitObserver<T> for ExchangedUnitObserver<'_, T> {
    fn begin_invocation(&mut self, invocation: &RoutedUnitInvocation<'_, T>) -> Result<(), Error> {
        if invocation.origins.is_some() {
            return Err(Error::backend(
                "routed invocation origins were already exchanged",
            ));
        }
        self.inner.begin_invocation(&RoutedUnitInvocation {
            input: invocation.input,
            origins: Some(self.origins),
            unit_coordinates: invocation.unit_coordinates,
        })
    }
    fn finish_invocation(&mut self, success: bool) -> Result<(), Error> {
        self.inner.finish_invocation(success)
    }
    fn invocation_active(&self) -> bool {
        self.inner.invocation_active()
    }
    fn observe(&mut self, batch: &RoutedUnitBatch<'_, T>) -> Result<(), Error> {
        self.inner.observe(&exchanged_batch(batch, self.origins)?)
    }
    fn intervene(&mut self, batch: &RoutedUnitBatch<'_, T>) -> Result<Option<T>, Error> {
        self.inner.intervene(&exchanged_batch(batch, self.origins)?)
    }
    fn observe_effective(&mut self, batch: &RoutedUnitBatch<'_, T>) -> Result<(), Error> {
        self.inner
            .observe_effective(&exchanged_batch(batch, self.origins)?)
    }
}
fn exchanged_batch<'a, T>(
    batch: &'a RoutedUnitBatch<'_, T>,
    origins: RoutedUnitOrigins<'a>,
) -> Result<RoutedUnitBatch<'a, T>, Error> {
    if batch.origins.is_some() {
        return Err(Error::backend("routed-unit origins were already exchanged"));
    }
    Ok(RoutedUnitBatch {
        units: batch.units.with_values(batch.units.values),
        source_groups: batch.source_groups,
        global_groups: batch.global_groups,
        provider_token_offset: batch.provider_token_offset,
        origins: Some(origins),
        unit_coordinates: batch.unit_coordinates,
    })
}

struct PartitionUnitObserver<'a, T: Tensor> {
    inner: &'a mut dyn RoutedUnitObserver<T>,
    coordinates: &'a eredu_core::component::ComponentCoordinateMap,
}
fn partition_batch<'a, T: Tensor>(
    batch: &'a RoutedUnitBatch<'_, T>,
    coordinates: &'a eredu_core::component::ComponentCoordinateMap,
) -> Result<RoutedUnitBatch<'a, T>, Error> {
    if batch.unit_coordinates.is_some()
        || batch.units.values.shape().len() != 2
        || batch
            .units
            .values
            .shape()
            .last()
            .and_then(|n| usize::try_from(*n).ok())
            != Some(coordinates.local_count())
    {
        return Err(Error::backend(
            "routed unit coordinates do not match this native partition",
        ));
    }
    let mut local = batch.map_tensors(|value| value);
    local.unit_coordinates = Some(coordinates);
    Ok(local)
}
impl<T: Tensor> RoutedUnitObserver<T> for PartitionUnitObserver<'_, T> {
    fn begin_invocation(&mut self, invocation: &RoutedUnitInvocation<'_, T>) -> Result<(), Error> {
        if invocation.unit_coordinates.is_some() {
            return Err(Error::backend(
                "routed invocation scalar coordinates were already bound",
            ));
        }
        self.inner.begin_invocation(&RoutedUnitInvocation {
            input: invocation.input,
            origins: invocation.origins,
            unit_coordinates: Some(self.coordinates),
        })
    }
    fn finish_invocation(&mut self, success: bool) -> Result<(), Error> {
        self.inner.finish_invocation(success)
    }
    fn invocation_active(&self) -> bool {
        self.inner.invocation_active()
    }
    fn observe(&mut self, batch: &RoutedUnitBatch<'_, T>) -> Result<(), Error> {
        self.inner
            .observe(&partition_batch(batch, self.coordinates)?)
    }
    fn intervene(&mut self, batch: &RoutedUnitBatch<'_, T>) -> Result<Option<T>, Error> {
        self.inner
            .intervene(&partition_batch(batch, self.coordinates)?)
    }
    fn observe_effective(&mut self, batch: &RoutedUnitBatch<'_, T>) -> Result<(), Error> {
        self.inner
            .observe_effective(&partition_batch(batch, self.coordinates)?)
    }
}

struct InvocationUnitObserver<'a, T> {
    inner: &'a mut dyn RoutedUnitObserver<T>,
}
impl<T> RoutedUnitObserver<T> for InvocationUnitObserver<'_, T> {
    fn invocation_active(&self) -> bool {
        true
    }
    fn begin_invocation(&mut self, _: &RoutedUnitInvocation<'_, T>) -> Result<(), Error> {
        Err(Error::backend("local routed invocation is already active"))
    }
    fn finish_invocation(&mut self, _: bool) -> Result<(), Error> {
        Err(Error::backend(
            "nested provider cannot finish its enclosing invocation",
        ))
    }
    fn observe(&mut self, batch: &RoutedUnitBatch<'_, T>) -> Result<(), Error> {
        self.inner.observe(batch)
    }
    fn intervene(&mut self, batch: &RoutedUnitBatch<'_, T>) -> Result<Option<T>, Error> {
        self.inner.intervene(batch)
    }
    fn observe_effective(&mut self, batch: &RoutedUnitBatch<'_, T>) -> Result<(), Error> {
        self.inner.observe_effective(batch)
    }
}

/// Encloses one actual local provider invocation, including its idle path and all
/// internal chunks. The final callback runs even if preparation or execution
/// fails, and finishes before the caller can enter another model collective.
/// Nested bank/residency adapters borrow the scope without repeating callbacks.
/// Disabled observation calls the provider directly and inspects no tensors.
pub fn with_routed_unit_invocation<T, R, E>(
    observer: Option<&mut dyn RoutedUnitObserver<T>>,
    invocation: RoutedUnitInvocation<'_, T>,
    execute: impl FnOnce(Option<&mut dyn RoutedUnitObserver<T>>) -> Result<R, E>,
    map_error: impl Fn(Error) -> E,
) -> Result<R, E> {
    let Some(observer) = observer else {
        return execute(None);
    };
    if observer.invocation_active() {
        return execute(Some(observer));
    }
    let result = match observer.begin_invocation(&invocation) {
        Ok(()) => execute(Some(&mut InvocationUnitObserver { inner: observer })),
        Err(error) => Err(map_error(error)),
    };
    let finished = observer
        .finish_invocation(result.is_ok())
        .map_err(map_error);
    // Preserve the first local cause when both execution and the vote fail.
    result.and_then(|output| finished.map(|()| output))
}

/// Attaches architecture-retained scalar coordinates through provider chunking
/// and exchange. Disabled observation creates no value or coordinate tensor.
/// The map supplies no distributed admission, peer or completion authority.
pub fn with_partition_unit_observer<T: Tensor, R>(
    observer: &mut Option<&mut dyn RoutedUnitObserver<T>>,
    coordinates: &eredu_core::component::ComponentCoordinateMap,
    execute: impl FnOnce(Option<&mut dyn RoutedUnitObserver<T>>) -> R,
) -> R {
    match observer {
        Some(inner) => execute(Some(&mut PartitionUnitObserver {
            inner: &mut **inner,
            coordinates,
        })),
        None => execute(None),
    }
}

/// Attaches already exchanged origin tags while preserving the provider's source
/// groups, compact-bank map and actual mutable unit values.
pub fn with_exchanged_unit_observer<T: Tensor, R>(
    observer: &mut Option<&mut dyn RoutedUnitObserver<T>>,
    origins: RoutedUnitOrigins<'_>,
    execute: impl FnOnce(Option<&mut dyn RoutedUnitObserver<T>>) -> R,
) -> R {
    match observer {
        Some(inner) => execute(Some(&mut ExchangedUnitObserver {
            inner: &mut **inner,
            origins,
        })),
        None => execute(None),
    }
}

/// Borrows an optional observer for one provider chunk without extending its
/// lifetime, creating a source tensor, or allocating a coordinate map.
pub fn with_provider_unit_observer<T: Tensor, R>(
    observer: &mut Option<&mut dyn RoutedUnitObserver<T>>,
    source_groups: &T,
    global_groups: Option<&[usize]>,
    provider_token_offset: usize,
    execute: impl FnOnce(Option<&mut dyn GroupedUnitObserver<T>>) -> R,
) -> R {
    match observer {
        Some(observer) => execute(Some(&mut ProviderUnitObserver::new(
            &mut **observer,
            source_groups,
            global_groups,
            provider_token_offset,
        ))),
        None => execute(None),
    }
}

/// Prepared resident banks execute the whole local input without an outer route
/// exchange. Their scope must finish before the caller enters a tensor reduction.
pub fn with_resident_unit_coordinates<T: Tensor, R, E>(
    coordinates: Option<&(eredu_core::component::ComponentCoordinateMap, bool)>,
    request: RoutedExpertRequest<'_, '_, T>,
    execute: impl FnOnce(RoutedExpertRequest<'_, '_, T>) -> Result<R, E>,
) -> Result<R, Error>
where
    E: std::error::Error + Send + Sync + 'static,
{
    let Some((coordinates, partitioned)) = coordinates else {
        return execute(request).map_err(Error::backend_source);
    };
    let RoutedExpertRequest {
        bank,
        layer,
        input,
        routes,
        pass,
        unit_observer,
    } = request;
    with_routed_unit_invocation(
        unit_observer,
        RoutedUnitInvocation {
            input,
            origins: None,
            unit_coordinates: Some(coordinates),
        },
        |mut observer| {
            if !partitioned {
                return execute(RoutedExpertRequest {
                    bank,
                    layer,
                    input,
                    routes,
                    pass,
                    unit_observer: observer,
                })
                .map_err(Error::backend_source);
            }
            with_partition_unit_observer(&mut observer, coordinates, |observer| {
                execute(RoutedExpertRequest {
                    bank,
                    layer,
                    input,
                    routes,
                    pass,
                    unit_observer: observer,
                })
                .map_err(Error::backend_source)
            })
        },
        |error| error,
    )
}
