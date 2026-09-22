//! Paid destinations for the ordinary complete, contiguous and sparse receipt constructor.
use super::*;
use eredu_core::capture::{
    CaptureContiguousProjectionPlan, CaptureCoordinateProjectionPlan, CaptureHistogramGeometry,
    CaptureRoutedUnitsGeometry, CaptureSummaryGeometry, CaptureTensorGeometry, CaptureTransform,
};
use eredu_nn::workspace::HostMetadataFundingError;
use std::mem::{size_of, size_of_val};

#[derive(Debug, thiserror::Error)]
pub(in crate::capture::partition) enum Cause {
    #[error("{0}")]
    Coordinates(#[from] eredu_core::component::ComponentCoordinateConstructionError),

    #[error("complete producer source: {0}")]
    Source(&'static str),
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
    #[error(transparent)]
    Destination(#[from] eredu_nn::Error),
    #[error(transparent)]
    Receipt(#[from] PartitionCaptureMergeError),
    #[error(transparent)]
    Capture(#[from] CaptureError),
}

/// A receipt construction refusal retains the exact shared admission and the
/// account that paid all partial destinations. Neither work nor quota is refunded.
#[derive(Debug, thiserror::Error)]
#[error("partition capture receipt construction: {cause}")]
pub struct PartitionCaptureReceiptConstructionError {
    #[source]
    cause: Cause,
    _source: SharedCapturePlan,
    _metadata: HostMetadataFunding,
}

/// Private destinations consumed only by the same validating constructor used
/// by ordinary receipts. The account retires after every destination.
pub(super) struct PreparedReceiptStorage {
    pub(super) slice: Option<ResolvedCaptureSlice>,
    pub(super) terminal_rows: Option<usize>,
    pub(super) global_shape: Vec<u64>,
    pub(super) identity: String,
    pub(super) metadata: HostMetadataFunding,
    pub(super) construction_metadata: Option<usize>,
}

/// Retained contiguous coordinates for one expected producer. The caller
/// supplies architecture-selected placement; this is no native execution grant.
#[derive(Debug, Clone)]
pub struct PartitionCaptureContiguousProducer {
    /// Actual world rank that must acknowledge this source, including empty overlap.
    pub rank: usize,
    /// Local coordinates along the common physical axis in the global tensor.
    pub coordinates: std::ops::Range<u64>,
}
/// Original component coordinates for one canonical dense exporter. This
/// source preserves permutations and empty ownership without an enclosing span.
#[derive(Debug, Clone, Copy)]
pub struct PartitionCaptureCoordinateProducer<'a> {
    /// Actual world rank, including an explicit empty exporter.
    pub rank: usize,
    /// Borrowed immutable physical-to-global component map.
    pub coordinates: &'a eredu_core::component::ComponentCoordinateMap,
}
/// Borrowed architecture-selected sparse ownership for one actual producer.
/// No projection, allocation or native work is authorized by these source facts.
#[derive(Debug, Clone, Copy)]
pub struct PartitionCaptureRoutedProducerSource<'a> {
    /// Expected world rank, including explicitly acknowledged empty ownership.
    pub rank: usize,
    /// Exact original expert/unit maps and logical input provenance.
    pub ownership: &'a RoutedUnitCaptureOwnership,
}
#[derive(Debug)]
enum ProducerProjection<'a> {
    Contiguous(CaptureContiguousProjectionPlan<'a>),
    Coordinates(CaptureCoordinateProjectionPlan<'a>),
    Routed(CaptureCoordinateProjectionPlan<'a>),
}
impl ProducerProjection<'_> {
    fn fragments(&self) -> usize {
        match self {
            Self::Contiguous(p) => p.fragments(),
            Self::Coordinates(p) => p.fragments(),
            Self::Routed(p) => p.fragments(),
        }
    }
    fn requested_bytes(&self) -> Option<usize> {
        match self {
            Self::Contiguous(p) => Some(p.requested_bytes()),
            Self::Coordinates(p) => Some(p.requested_bytes()),
            // The ordinary sparse declaration validator reconstructs the exact
            // projection independently. Both real allocations are paid here.
            Self::Routed(p) => p.requested_bytes().checked_mul(2),
        }
    }
    fn construct(self) -> CaptureSlicePartition {
        match self {
            Self::Contiguous(p) => p.construct(),
            Self::Coordinates(p) => p.construct(),
            Self::Routed(p) => p.construct(),
        }
    }
}
#[derive(Clone, Copy)]
enum ProducerSources<'a> {
    Complete(usize),
    Coordinates {
        axis: usize,
        rows: &'a [PartitionCaptureCoordinateProducer<'a>],
        combination: PartitionCaptureCombination,
    },
    Routed(&'a [PartitionCaptureRoutedProducerSource<'a>]),
    Contiguous {
        axis: usize,
        rows: &'a [PartitionCaptureContiguousProducer],
        combination: PartitionCaptureCombination,
    },
}
impl ProducerSources<'_> {
    fn len(self) -> usize {
        match self {
            Self::Complete(_) => 1,
            Self::Contiguous { rows, .. } => rows.len(),
            Self::Coordinates { rows, .. } => rows.len(),
            Self::Routed(rows) => rows.len(),
        }
    }
    fn axis(self) -> usize {
        match self {
            Self::Complete(_) => 0,
            Self::Contiguous { axis, .. } | Self::Coordinates { axis, .. } => axis,
            Self::Routed(_) => 2,
        }
    }
    fn combination(self) -> PartitionCaptureCombination {
        match self {
            Self::Complete(_) => PartitionCaptureCombination::Disjoint,
            Self::Contiguous { combination, .. } | Self::Coordinates { combination, .. } => {
                combination
            }
            Self::Routed(_) => PartitionCaptureCombination::Disjoint,
        }
    }
    fn row(self, index: usize, global: &[u64]) -> (usize, std::ops::Range<u64>) {
        match self {
            Self::Complete(rank) => (rank, 0..global[0]),
            Self::Contiguous { rows, .. } => (rows[index].rank, rows[index].coordinates.clone()),
            Self::Routed(rows) => (rows[index].rank, 0..0),
            Self::Coordinates { rows, .. } => (rows[index].rank, 0..0),
        }
    }
}

impl PartitionCaptureReceiptPlan {
    /// Construct one complete-global producer using the original shared capture
    /// admission, paid host destinations, and the ordinary receipt validator and
    /// identity writer. Placement comes from the enclosing architecture selection;
    /// this method grants no native transform, transport, or completion authority.
    ///
    /// The supplied logical ledger remains the ordinary capture quota owner. The
    /// separate metadata account pays this closed constructor's actual storage.
    pub fn new_complete_shared_funded(
        source: &SharedCapturePlan,
        context: &PartitionCaptureContext,
        producer: usize,
        world_size: usize,
        limits: PartitionCaptureReceiptLimits,
        metadata: &HostMetadataFunding,
        ledger: &mut dyn CaptureReservation,
    ) -> Result<Self, PartitionCaptureReceiptConstructionError> {
        Self::construct_shared_funded(
            source,
            context,
            ProducerSources::Complete(producer),
            world_size,
            limits,
            metadata,
            ledger,
            false,
            None,
        )
        .map(|(receipt, _)| receipt)
    }
    pub(crate) fn new_complete_shared_funded_global(
        source: &SharedCapturePlan,
        context: &PartitionCaptureContext,
        producer: usize,
        world_size: usize,
        limits: PartitionCaptureReceiptLimits,
        metadata: &HostMetadataFunding,
        ledger: &mut dyn CaptureReservation,
    ) -> Result<(Self, CaptureUsage), PartitionCaptureReceiptConstructionError> {
        Self::construct_shared_funded(
            source,
            context,
            ProducerSources::Complete(producer),
            world_size,
            limits,
            metadata,
            ledger,
            true,
            None,
        )
    }
    pub(crate) fn new_complete_shared_funded_terminal_global(
        source: &SharedCapturePlan,
        context: &PartitionCaptureContext,
        producer: usize,
        world_size: usize,
        limits: PartitionCaptureReceiptLimits,
        metadata: &HostMetadataFunding,
        ledger: &mut dyn CaptureReservation,
        terminal_rows: Option<usize>,
    ) -> Result<(Self, CaptureUsage), PartitionCaptureReceiptConstructionError> {
        Self::construct_shared_funded(
            source,
            context,
            ProducerSources::Complete(producer),
            world_size,
            limits,
            metadata,
            ledger,
            true,
            terminal_rows,
        )
    }
    /// Build prepaid contiguous disjoint shards or complete additive terms
    /// through the ordinary ownership validator and canonical identity worker.
    /// Sum keeps its existing raw-term/F64-compensated assembly contract; this
    /// constructor cannot turn shard-local probabilities into global evidence.
    pub fn new_contiguous_shared_funded(
        source: &SharedCapturePlan,
        context: &PartitionCaptureContext,
        axis: usize,
        producers: &[PartitionCaptureContiguousProducer],
        combination: PartitionCaptureCombination,
        world_size: usize,
        limits: PartitionCaptureReceiptLimits,
        metadata: &HostMetadataFunding,
        ledger: &mut dyn CaptureReservation,
    ) -> Result<Self, PartitionCaptureReceiptConstructionError> {
        Self::construct_shared_funded(
            source,
            context,
            ProducerSources::Contiguous {
                axis,
                rows: producers,
                combination,
            },
            world_size,
            limits,
            metadata,
            ledger,
            false,
            None,
        )
        .map(|(receipt, _)| receipt)
    }

    /// Construct exact positive-run dense projections using the same ordinary
    /// receipt validation, identity, ledger and canonical producer ordering.
    /// Owning geometry storage is paid from this original metadata account.
    pub fn new_coordinates_shared_funded(
        source: &SharedCapturePlan,
        context: &PartitionCaptureContext,
        axis: usize,
        producers: &[PartitionCaptureCoordinateProducer<'_>],
        combination: PartitionCaptureCombination,
        world_size: usize,
        limits: PartitionCaptureReceiptLimits,
        metadata: &HostMetadataFunding,
        ledger: &mut dyn CaptureReservation,
    ) -> Result<Self, PartitionCaptureReceiptConstructionError> {
        Self::construct_shared_funded(
            source,
            context,
            ProducerSources::Coordinates {
                axis,
                rows: producers,
                combination,
            },
            world_size,
            limits,
            metadata,
            ledger,
            false,
            None,
        )
        .map(|(receipt, _)| receipt)
    }

    /// Construct sparse receipts from borrowed exact ownership through the same
    /// ordinary declaration, overlap, coverage, schedule and identity workers.
    /// Coordinate copies and both actual projection constructions are paid before
    /// allocation. This does not authorize sparse Host or native execution.
    pub fn new_routed_shared_funded(
        source: &SharedCapturePlan,
        context: &PartitionCaptureContext,
        producers: &[PartitionCaptureRoutedProducerSource<'_>],
        world_size: usize,
        limits: PartitionCaptureReceiptLimits,
        metadata: &HostMetadataFunding,
        ledger: &mut dyn CaptureReservation,
    ) -> Result<Self, PartitionCaptureReceiptConstructionError> {
        Self::construct_shared_funded(
            source,
            context,
            ProducerSources::Routed(producers),
            world_size,
            limits,
            metadata,
            ledger,
            false,
            None,
        )
        .map(|(receipt, _)| receipt)
    }

    fn construct_shared_funded(
        source: &SharedCapturePlan,
        context: &PartitionCaptureContext,
        producers_source: ProducerSources<'_>,
        world_size: usize,
        limits: PartitionCaptureReceiptLimits,
        metadata: &HostMetadataFunding,
        ledger: &mut dyn CaptureReservation,
        global_preparation_required: bool,
        terminal_rows: Option<usize>,
    ) -> Result<(Self, CaptureUsage), PartitionCaptureReceiptConstructionError> {
        let error = |cause| PartitionCaptureReceiptConstructionError {
            cause,
            _source: source.clone(),
            _metadata: metadata.clone(),
        };
        let controls = constructor_control_bytes()
            .ok_or_else(|| error(Cause::Source("constructor controls overflow")))?;
        let controls = if matches!(producers_source, ProducerSources::Routed(_)) {
            controls
                .checked_add(
                    routed_constructor_control_bytes()
                        .ok_or_else(|| error(Cause::Source("sparse controls overflow")))?,
                )
                .ok_or_else(|| error(Cause::Source("sparse controls overflow")))?
        } else {
            controls
        };
        metadata
            .reserve_metadata(controls)
            .map_err(|cause| error(cause.into()))?;
        // Check the borrowed exact source before cloning any context payload.
        // Explicit axes are resolved through this same original admission, not
        // reconstructed from equal element counts or an ordinary request shape.
        // Sparse ownership and terminal vocabulary keep their separate sources.
        if context.capture_plan_identity != source.admission().identity()
            || world_size == 0
            || producers_source.len() == 0
            || producers_source.len() > limits.max_producers
            || limits.max_producers == 0
            || limits.max_record_bytes == 0
            || [
                &context.artifact_identity,
                &context.execution_identity,
                &context.run_identity,
                &context.capture_plan_identity,
            ]
            .into_iter()
            .chain(context.overlay_identity.iter())
            .any(|identity| identity.is_empty() || identity.len() > 256)
        {
            return Err(error(Cause::Source(
                "receipt identity, context or participant differs",
            )));
        }
        let summary = source
            .admission()
            .plan()
            .selections
            .get(context.selection_index)
            .is_some_and(|selection| matches!(selection.transform, CaptureTransform::Summary));
        let histogram = source
            .admission()
            .plan()
            .selections
            .get(context.selection_index)
            .is_some_and(|selection| {
                matches!(selection.transform, CaptureTransform::Histogram { .. })
            });
        if context.invocation.is_some()
            && (terminal_rows.is_some()
                || source
                    .admission()
                    .plan()
                    .selections
                    .get(context.selection_index)
                    .is_some_and(|selection| {
                        super::super::vocabulary::is_vocabulary(&selection.transform)
                    }))
        {
            return Err(error(Cause::Source(
                "explicit vocabulary invocation requires its typed terminal source",
            )));
        }
        let vocabulary_geometry = super::super::CompleteVocabularyGeometry::prepare(
            source.admission(),
            context.selection_index,
            context.phase,
            context.prediction,
            terminal_rows,
        )
        .map_err(|_| error(Cause::Source("terminal source geometry differs")))?;
        let routed_geometry = if matches!(producers_source, ProducerSources::Routed(_)) {
            Some(
                CaptureRoutedUnitsGeometry::prepare_receipt(source.admission(), context)
                    .map_err(|_| error(Cause::Source("selection has no exact sparse geometry")))?,
            )
        } else {
            None
        };
        let tensor_geometry =
            if summary || histogram || vocabulary_geometry.is_some() || routed_geometry.is_some() {
                None
            } else {
                Some(
                    CaptureTensorGeometry::prepare_receipt(source.admission(), context).map_err(
                        |_| {
                            error(Cause::Source(
                                "selection has no complete floating tensor geometry",
                            ))
                        },
                    )?,
                )
            };
        let summary_geometry = if summary {
            Some(
                CaptureSummaryGeometry::prepare_receipt(source.admission(), context).map_err(
                    |_| error(Cause::Source("selection has no complete summary geometry")),
                )?,
            )
        } else {
            None
        };
        let histogram_geometry = if histogram {
            Some(
                CaptureHistogramGeometry::prepare_receipt(source.admission(), context).map_err(
                    |_| {
                        error(Cause::Source(
                            "selection has no complete histogram geometry",
                        ))
                    },
                )?,
            )
        } else {
            None
        };
        let terminal_ends = vocabulary_geometry
            .as_ref()
            .map(|value| value.shape().map(|n| n as u64))
            .unwrap_or([0; 3]);
        let (source_shape, source_starts, source_ends, source_strides) = match (
            &tensor_geometry,
            &summary_geometry,
            &histogram_geometry,
            &vocabulary_geometry,
            &routed_geometry,
        ) {
            (Some(geometry), None, None, None, None) => (
                geometry.source_shape(),
                geometry.starts(),
                geometry.ends(),
                geometry.strides(),
            ),
            (None, Some(geometry), None, None, None) => (
                geometry.source_shape(),
                geometry.starts(),
                geometry.ends(),
                geometry.strides(),
            ),
            (None, None, Some(geometry), None, None) => (
                geometry.source_shape(),
                geometry.starts(),
                geometry.ends(),
                geometry.strides(),
            ),
            (None, None, None, Some(geometry), None) => (
                &geometry.shape()[..],
                &[0, 0, 0][..],
                &terminal_ends[..],
                &[1, 1, 1][..],
            ),
            (None, None, None, None, Some(geometry)) => (
                geometry.source_shape(),
                geometry.starts(),
                geometry.ends(),
                geometry.strides(),
            ),
            _ => return Err(error(Cause::Source("selection payload geometry differs"))),
        };
        let rank = source_shape.len();
        if rank == 0 {
            return Err(error(Cause::Source(
                "complete producer needs a physical axis",
            )));
        }
        let mut global = [0u64; 32];
        for (destination, extent) in global.iter_mut().zip(source_shape) {
            *destination = u64::try_from(*extent)
                .map_err(|_| error(Cause::Source("source extent overflow")))?;
        }
        // Preview's output is flat, but its producer projection still describes
        // all selected source coordinates. Preserve the normalized source slice.
        // This same source census describes a later reconstruction from the
        // retained prototype. Current destinations still debit before allocation.
        let mut construction_metadata = constructor_control_bytes()
            .and_then(|n| {
                n.checked_add(
                    eredu_nn::workspace::WorkspaceContext::metadata_vec_bytes::<u64>(rank)?
                        .checked_mul(5)?,
                )
            })
            .and_then(|n| {
                n.checked_add(eredu_nn::workspace::WorkspaceContext::metadata_vec_bytes::<
                    (usize, ProducerProjection<'_>),
                >(producers_source.len())?)
            })
            .and_then(|n| {
                n.checked_add(eredu_nn::workspace::WorkspaceContext::metadata_vec_bytes::<
                    PartitionCaptureProducer,
                >(producers_source.len())?)
            })
            .and_then(|n| n.checked_add(64))
            .ok_or_else(|| error(Cause::Source("constructor population overflow")))?;
        if let ProducerSources::Routed(rows) = producers_source {
            construction_metadata = construction_metadata
                .checked_add(
                    routed_constructor_control_bytes()
                        .ok_or_else(|| error(Cause::Source("routed controls overflow")))?,
                )
                .and_then(|n| {
                    n.checked_add(eredu_nn::workspace::WorkspaceContext::metadata_vec_bytes::<
                        (usize, RoutedUnitCaptureOwnership),
                    >(rows.len())?)
                })
                .ok_or_else(|| error(Cause::Source("routed population overflow")))?;
            for row in rows {
                construction_metadata = construction_metadata
                    .checked_add(
                        row.ownership
                            .coordinates
                            .experts()
                            .clone_metadata_bytes()
                            .and_then(|n| {
                                n.checked_add(
                                    row.ownership.coordinates.units().clone_metadata_bytes()?,
                                )
                            })
                            .ok_or_else(|| {
                                error(Cause::Source("routed clone population overflow"))
                            })?,
                    )
                    .ok_or_else(|| error(Cause::Source("routed population overflow")))?;
            }
        }
        let mut starts = metadata
            .metadata_vec(rank)
            .map_err(|cause| error(cause.into()))?;
        let mut ends = metadata
            .metadata_vec(rank)
            .map_err(|cause| error(cause.into()))?;
        let mut strides = metadata
            .metadata_vec(rank)
            .map_err(|cause| error(cause.into()))?;
        let mut shape = metadata
            .metadata_vec(rank)
            .map_err(|cause| error(cause.into()))?;
        starts.extend_from_slice(source_starts);
        ends.extend_from_slice(source_ends);
        strides.extend_from_slice(source_strides);
        for ((start, end), stride) in starts.iter().zip(&ends).zip(&strides) {
            shape.push((end - start).div_ceil(*stride));
        }
        let slice = ResolvedCaptureSlice {
            starts,
            ends,
            strides,
            shape,
        };
        let mut projection_sources = metadata
            .metadata_vec(producers_source.len())
            .map_err(|cause| error(cause.into()))?;
        let mut fragments = 0usize;
        for index in 0..producers_source.len() {
            let (producer_rank, coordinates) = producers_source.row(index, &global[..rank]);
            if producer_rank >= world_size {
                return Err(error(Cause::Source(
                    "producer rank differs from selected world",
                )));
            }
            let projection = match producers_source {
                ProducerSources::Coordinates { axis, rows, .. } => ProducerProjection::Coordinates(
                    CaptureCoordinateProjectionPlan::prepare(
                        &global[..rank],
                        &slice,
                        axis,
                        rows[index].coordinates,
                        limits.max_fragments,
                    )
                    .map_err(|_| {
                        error(Cause::Source(
                            "retained component projection exceeds its bound",
                        ))
                    })?,
                ),
                ProducerSources::Routed(rows) => ProducerProjection::Routed(
                    CaptureCoordinateProjectionPlan::prepare(
                        &global[..rank],
                        &slice,
                        2,
                        rows[index].ownership.coordinates.units(),
                        limits.max_fragments,
                    )
                    .map_err(|_| {
                        error(Cause::Source(
                            "retained sparse projection exceeds its bound",
                        ))
                    })?,
                ),
                _ => ProducerProjection::Contiguous(
                    CaptureContiguousProjectionPlan::prepare(
                        &global[..rank],
                        &slice,
                        producers_source.axis(),
                        coordinates,
                        limits.max_fragments,
                    )
                    .map_err(|_| {
                        error(Cause::Source(
                            "retained contiguous projection exceeds its bound",
                        ))
                    })?,
                ),
            };
            fragments = fragments
                .checked_add(projection.fragments())
                .ok_or_else(|| error(Cause::Source("fragment population overflow")))?;
            if fragments > limits.max_fragments {
                return Err(error(Cause::Source(
                    "total fragments exceed the receipt bound",
                )));
            }
            let projection_bytes = projection
                .requested_bytes()
                .ok_or_else(|| error(Cause::Source("projection population overflow")))?;
            construction_metadata = construction_metadata
                .checked_add(projection_bytes)
                .ok_or_else(|| error(Cause::Source("constructor population overflow")))?;
            metadata
                .reserve_metadata(projection_bytes)
                .map_err(|cause| error(cause.into()))?;
            projection_sources.push((producer_rank, projection));
        }
        let local_preparation = Self::preparation_population_usage(
            producers_source.len() as u64,
            rank,
            fragments as u64,
        )
        .and_then(|usage| {
            usage.checked_add(CaptureUsage {
                host_bytes: 2048,
                ..Default::default()
            })
        })
        .map_err(|cause| error(cause.into()))?;
        let global_preparation = if global_preparation_required {
            local_preparation
                .checked_mul(world_size as u64)
                .map_err(|cause| error(cause.into()))?
        } else {
            CaptureUsage::default()
        };
        let mut preparation = if global_preparation_required {
            Some(
                ledger
                    .reserve_quota(global_preparation)
                    .and_then(|mut quota| quota.reserve_quota(local_preparation))
                    .map_err(|cause| error(cause.into()))?,
            )
        } else {
            None
        };
        let ledger: &mut dyn CaptureReservation = match preparation.as_mut() {
            Some(quota) => quota,
            None => ledger,
        };
        let ownership = match producers_source {
            ProducerSources::Routed(rows) => {
                let mut owned = metadata
                    .metadata_vec(rows.len())
                    .map_err(|cause| error(cause.into()))?;
                // The ordinary routed constructor reserves this same logical
                // ownership population in addition to projection preparation.
                let ownership_usage =
                    Self::routed_ownership_source_usage(rows.iter().map(|row| row.ownership))
                        .map_err(|cause| error(cause.into()))?;
                reserve(ledger, ownership_usage).map_err(|cause| error(cause.into()))?;
                for row in rows {
                    let copy = copy_routed_ownership(row.ownership, metadata)
                        .map_err(|cause| error(cause))?;
                    owned.push((row.rank, copy));
                }
                RoutedOwnership::prepared(owned).map_err(|cause| error(cause.into()))?
            }
            _ => BTreeMap::new().into(),
        };
        let mut producers = metadata
            .metadata_vec(producers_source.len())
            .map_err(|cause| error(cause.into()))?;
        for (rank, projection) in projection_sources {
            producers.push(PartitionCaptureProducer {
                rank,
                projection: projection.construct(),
            });
        }
        let mut global_shape = metadata
            .metadata_vec(rank)
            .map_err(|cause| error(cause.into()))?;
        global_shape.extend_from_slice(&global[..rank]);
        let context = copy_context(context, metadata).map_err(|cause| error(cause.into()))?;
        // The shared SHA-256 writer emits exactly 64 ASCII hexadecimal bytes.
        // No caller capacity or growth policy is accepted as allocation authority.
        metadata
            .reserve_metadata(64)
            .map_err(|cause| error(cause.into()))?;
        let storage = PreparedReceiptStorage {
            slice: Some(slice),
            terminal_rows,
            global_shape,
            identity: String::with_capacity(64),
            metadata: metadata.clone(),
            construction_metadata: Some(construction_metadata),
        };
        Self::new_with_ownership(
            source.clone(),
            context,
            producers,
            ownership,
            producers_source.combination(),
            world_size,
            limits,
            ledger,
            Some(storage),
        )
        .map(|receipt| (receipt, global_preparation))
        .map_err(|cause| error(cause.into()))
    }
}

pub(in crate::capture::partition) fn copy_routed_ownership(
    source: &RoutedUnitCaptureOwnership,
    metadata: &HostMetadataFunding,
) -> Result<RoutedUnitCaptureOwnership, Cause> {
    fn copy(
        source: &eredu_core::component::ComponentCoordinateMap,
        metadata: &HostMetadataFunding,
    ) -> Result<eredu_core::component::ComponentCoordinateMap, Cause> {
        source.try_clone_with_funding(metadata).map_err(Into::into)
    }
    Ok(RoutedUnitCaptureOwnership {
        coordinates: eredu_core::component::RoutedComponentCoordinateMap::new(
            copy(source.coordinates.experts(), metadata)?,
            copy(source.coordinates.units(), metadata)?,
        ),
        source_peer: source.source_peer,
        source_peers: source.source_peers,
    })
}

/// Runtime-owned quotation of the shared paid context-copy worker.
/// The core context remains a neutral identity description.
pub trait PartitionCaptureContextMetadata {
    /// Backing and transports for the actual source labels.
    fn copy_metadata_bytes(&self) -> Option<usize>;
    /// Same worker with a source-declared maximum future run-label width.
    fn copy_metadata_bytes_with_run_length(&self, run_length: usize) -> Option<usize>;
}
impl PartitionCaptureContextMetadata for PartitionCaptureContext {
    /// Prospective backing and transports of the shared paid label copy.
    fn copy_metadata_bytes(&self) -> Option<usize> {
        self.copy_metadata_bytes_with_run_length(self.run_identity.len())
    }
    /// Same label copy with the source-declared maximum future run-name width.
    /// This is descriptive only and cannot bind a run or receipt identity.
    fn copy_metadata_bytes_with_run_length(&self, run_length: usize) -> Option<usize> {
        if run_length < self.run_identity.len() {
            return None;
        }
        let mut bytes = 0usize;
        for length in [
            self.artifact_identity.len(),
            self.execution_identity.len(),
            run_length,
            self.capture_plan_identity.len(),
        ]
        .into_iter()
        .chain(self.overlay_identity.as_ref().map(String::len))
        {
            bytes = bytes.checked_add(
                eredu_nn::workspace::WorkspaceContext::metadata_string_bytes(length)?,
            )?;
        }
        Some(bytes)
    }
}
impl PartitionCaptureReceiptPlan {
    /// The original paid constructor's retained geometry population.
    /// Ordinary receipts provide no such fact.
    pub fn reconstruction_metadata_bytes(&self, run_length: usize) -> Option<usize> {
        self.construction_metadata?.checked_add(
            self.context
                .copy_metadata_bytes_with_run_length(run_length)?,
        )
    }
}

pub(in crate::capture::partition) fn copy_context(
    source: &PartitionCaptureContext,
    metadata: &HostMetadataFunding,
) -> Result<PartitionCaptureContext, eredu_nn::Error> {
    Ok(PartitionCaptureContext {
        artifact_identity: metadata
            .metadata_string(format_args!("{}", source.artifact_identity))?,
        execution_identity: metadata
            .metadata_string(format_args!("{}", source.execution_identity))?,
        run_identity: metadata.metadata_string(format_args!("{}", source.run_identity))?,
        overlay_identity: source
            .overlay_identity
            .as_ref()
            .map(|value| metadata.metadata_string(format_args!("{value}")))
            .transpose()?,
        capture_plan_identity: metadata
            .metadata_string(format_args!("{}", source.capture_plan_identity))?,
        selection_index: source.selection_index,
        phase: source.phase,
        prediction: source.prediction,
        forward_epoch: source.forward_epoch,
        invocation: source.invocation,
        invocation_window: source.invocation_window,
    })
}

fn constructor_control_bytes() -> Option<usize> {
    // Named source, constructor, validator, digest and return populations coexist
    // through the shared call. Vector/string backing is paid separately above.
    let parts = [
        size_of::<PartitionCaptureReceiptPlan>() * 2,
        size_of::<PreparedReceiptStorage>() * 2,
        size_of::<Option<PreparedReceiptStorage>>(),
        size_of::<Option<super::super::CompleteVocabularyGeometry<'_>>>() * 2,
        super::super::CompleteVocabularyGeometry::control_bytes()?.checked_mul(2)?,
        size_of::<
            Result<
                Option<super::super::CompleteVocabularyGeometry<'_>>,
                eredu_core::capture::CaptureTensorGeometryError,
            >,
        >(),
        size_of::<(Option<usize>, [u64; 3], [u64; 3], [u64; 3])>(),
        size_of::<Option<CaptureTensorGeometry<'_>>>() * 2,
        size_of::<Option<CaptureSummaryGeometry<'_>>>() * 2,
        size_of::<Option<CaptureHistogramGeometry<'_>>>() * 2,
        CaptureHistogramGeometry::preparation_control_bytes()?,
        size_of::<(&[usize], &[u64], &[u64], &[u64], bool)>(),
        size_of::<[u64; 32]>(),
        size_of::<ResolvedCaptureSlice>() * 2,
        size_of::<[Vec<u64>; 5]>(),
        size_of::<Vec<PartitionCaptureProducer>>(),
        size_of::<PartitionCaptureProducer>() * 2,
        size_of::<PartitionCaptureContext>() * 2,
        size_of::<Option<eredu_core::capture::CaptureInvocationShape>>(),
        size_of::<PartitionCaptureReceiptLimits>(),
        size_of::<(SharedCapturePlan, HostMetadataFunding)>(),
        size_of::<BTreeMap<usize, CaptureSlicePartition>>(),
        size_of::<BTreeMap<usize, RoutedUnitCaptureOwnership>>(),
        size_of::<Vec<PartitionCaptureProducer>>() * 2,
        size_of::<ProducerSources<'_>>() * 2,
        size_of::<PartitionCaptureCoordinateProducer<'_>>() * 2,
        size_of::<(
            &SharedCapturePlan,
            &PartitionCaptureContext,
            usize,
            &[PartitionCaptureCoordinateProducer<'_>],
            PartitionCaptureCombination,
            usize,
            PartitionCaptureReceiptLimits,
            &HostMetadataFunding,
            &mut dyn CaptureReservation,
        )>(),
        size_of::<(
            &SharedCapturePlan,
            &PartitionCaptureContext,
            usize,
            &[PartitionCaptureContiguousProducer],
            PartitionCaptureCombination,
            usize,
            PartitionCaptureReceiptLimits,
            &HostMetadataFunding,
            &mut dyn CaptureReservation,
        )>(),
        size_of::<(usize, usize)>(),
        size_of::<std::ops::Range<usize>>() * 2,
        size_of::<(usize, usize, std::ops::Range<u64>)>(),
        size_of::<(usize, ProducerProjection<'_>)>() * 2,
        size_of::<Vec<(usize, ProducerProjection<'_>)>>(),
        size_of::<std::vec::IntoIter<(usize, ProducerProjection<'_>)>>(),
        size_of::<
            Result<
                CaptureContiguousProjectionPlan<'_>,
                eredu_core::capture::CaptureContiguousProjectionError,
            >,
        >(),
        size_of::<std::slice::Iter<'_, PartitionCaptureContiguousProducer>>(),
        size_of::<Result<usize, usize>>(),
        size_of::<Option<String>>(),
        size_of::<Option<HostMetadataFunding>>(),
        size_of::<CaptureUsage>() * 6,
        size_of::<Option<CaptureQuota>>(),
        size_of::<
            Result<
                (PartitionCaptureReceiptPlan, CaptureUsage),
                PartitionCaptureReceiptConstructionError,
            >,
        >(),
        size_of::<sha2::Sha256>(),
        size_of::<sha2::digest::Output<sha2::Sha256>>(),
        size_of::<std::fmt::Arguments<'_>>(),
        size_of::<String>() * 2,
        size_of::<Cause>(),
        size_of::<PartitionCaptureReceiptConstructionError>(),
        size_of::<Result<PartitionCaptureReceiptPlan, PartitionCaptureReceiptConstructionError>>(),
        size_of::<Result<PartitionCaptureReceiptPlan, PartitionCaptureMergeError>>(),
        size_of::<Result<PartitionCaptureContext, eredu_nn::Error>>(),
        size_of::<(
            &SharedCapturePlan,
            &PartitionCaptureContext,
            &HostMetadataFunding,
            &mut dyn CaptureReservation,
            usize,
            usize,
        )>(),
        size_of::<(&CaptureSlicePartition, &ResolvedCaptureSlice, usize, u64)>(),
        size_of::<std::slice::Iter<'_, PartitionCaptureProducer>>() * 2,
        size_of::<std::slice::Iter<'_, CaptureFragmentGeometry>>() * 2,
        size_of::<std::collections::btree_map::Iter<'_, usize, CaptureSlicePartition>>(),
        size_of::<std::collections::btree_map::Iter<'_, usize, RoutedUnitCaptureOwnership>>(),
        size_of::<std::slice::Iter<'_, PartitionCaptureProducer>>(),
        size_of::<
            std::iter::Zip<
                std::iter::Zip<std::slice::Iter<'_, u64>, std::slice::Iter<'_, u64>>,
                std::slice::Iter<'_, u64>,
            >,
        >(),
        size_of::<<sha2::digest::Output<sha2::Sha256> as IntoIterator>::IntoIter>(),
        size_of::<[&String; 4]>() * 2,
        size_of::<[usize; 5]>(),
        size_of::<[u64; 3]>(),
        size_of::<[u64; 5]>(),
        size_of::<std::array::IntoIter<u64, 3>>(),
        size_of::<std::array::IntoIter<u64, 5>>(),
        size_of::<Option<eredu_core::capture::PartitionCaptureInvocationWindow>>(),
        size_of::<(usize, u64, u64, bool, [u8; 8])>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}

impl PartitionCaptureReceiptConstructionError {
    pub(crate) fn limit_skip(&self) -> Option<CaptureSkipReason> {
        let cause = match &self.cause {
            Cause::Capture(cause) | Cause::Receipt(PartitionCaptureMergeError::Capture(cause)) => {
                cause.cause()
            }
            _ => return None,
        };
        match cause {
            CaptureError::Limit { budget, cumulative } => Some(CaptureSkipReason::Limit {
                budget: *budget,
                cumulative: *cumulative,
            }),
            _ => None,
        }
    }
}

fn routed_constructor_control_bytes() -> Option<usize> {
    let parts = [
        size_of::<Option<CaptureRoutedUnitsGeometry<'_>>>() * 2,
        CaptureRoutedUnitsGeometry::preparation_control_bytes()?,
        size_of::<PartitionCaptureRoutedProducerSource<'_>>() * 2,
        RoutedOwnership::control_bytes()?,
        size_of::<Vec<(usize, RoutedUnitCaptureOwnership)>>() * 2,
        size_of::<RoutedUnitCaptureOwnership>() * 3,
        size_of::<eredu_core::component::RoutedComponentCoordinateMap>(),
        size_of::<(&RoutedUnitCaptureOwnership, &HostMetadataFunding)>(),
        size_of::<Result<RoutedUnitCaptureOwnership, Cause>>(),
        size_of::<(
            &eredu_core::component::ComponentCoordinateMap,
            &HostMetadataFunding,
        )>(),
        size_of::<Result<eredu_core::component::ComponentCoordinateMap, Cause>>(),
        size_of::<std::slice::Iter<'_, PartitionCaptureRoutedProducerSource<'_>>>(),
        size_of::<std::slice::Windows<'_, (usize, RoutedUnitCaptureOwnership)>>(),
        size_of::<Result<RoutedOwnership, CaptureError>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}

pub(in crate::capture::partition) fn component_adapter_error(
    source: &SharedCapturePlan,
    metadata: &HostMetadataFunding,
    cause: eredu_nn::Error,
) -> PartitionCaptureReceiptConstructionError {
    PartitionCaptureReceiptConstructionError {
        cause: Cause::Destination(cause),
        _source: source.clone(),
        _metadata: metadata.clone(),
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
