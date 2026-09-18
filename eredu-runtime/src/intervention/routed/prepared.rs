//! Original source custody around the ordinary sparse lowering worker.
use super::*;
use crate::working_memory::OriginalInterventionSource;
use eredu_nn::workspace::{
    WorkspaceContext, HostMetadataFunding, HostMetadataFundingError,
};
use std::mem::{size_of, size_of_val};
use worker::{Allocation, RoutedInterventionLoweringError as Geometry, RowKey};

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error(transparent)]
    Geometry(#[from] Geometry),
    #[error(transparent)]
    Selection(#[from] InterventionGeometryError),
    #[error(transparent)]
    Axes(#[from] CaptureAxisError),
    #[error(transparent)]
    Window(#[from] CaptureWindowSourceError),
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
    #[error(transparent)]
    Metadata(#[from] eredu_nn::Error),
}
/// A refused or partially constructed recipe retains its actual immutable source
/// and cumulative metadata account. It cannot be extracted as native authority.
#[derive(Debug, thiserror::Error)]
#[error("prepared sparse intervention failed: {cause}")]
pub struct PreparedRoutedInterventionError {
    #[source]
    cause: Cause,
    _declaration: OriginalInterventionSource,
    _funding: HostMetadataFunding,
}
/// Fixed coordinate destination for one actual serial provider chunk. Source
/// authentication, progress, receipt spending and native completion stay with the
/// existing observer. This owner grants none of them.
#[derive(Debug)]
pub struct PreparedRoutedInterventionRows {
    selected: ResolvedCaptureSlice,
    rows: Vec<RoutedUnitLocation>,
    geometry: RoutedUnitGeometry,
    source_tokens: u64,
    source_range: [u64; 2],
    logical_origin: u64,
    expected_rows: usize,
    failed: Option<Geometry>,
    operation: usize,
    phase: CapturePhase,
    prediction: u64,
    declaration: OriginalInterventionSource,
    funding: HostMetadataFunding,
}
/// Completed native-row-order recipe with its exact one-dimensional selected
/// action. Both the immutable declaration and paid destination survive aliases
/// borrowed by the existing native selection/update worker.
#[derive(Debug)]
pub struct PreparedRoutedIntervention {
    lowered: LoweredRoutedIntervention,
    local: ResolvedCaptureSlice,
    source_tokens:u64,
    source_range:[u64;2],
    operation: usize,
    phase: CapturePhase,
    prediction: u64,
    declaration: OriginalInterventionSource,
    _funding: HostMetadataFunding,
}
struct Paid<'a>(&'a HostMetadataFunding);
impl Allocation for Paid<'_> {
    type Error = Cause;
    fn vector<T>(&mut self, capacity: usize) -> Result<Vec<T>, Cause> {
        Ok(self.0.metadata_vec(capacity)?)
    }
}
impl PreparedRoutedInterventionRows {
    /// Prepare before reading native coordinates. `source_tokens`, `source_range`
    /// and `actual_rows` describe this actual source, not admission maxima. Window
    /// coordinates use the existing logical-geometry resolver and offset rule.
    pub fn prepare(
        source: &OriginalInterventionSource,
        operation: usize,
        phase: CapturePhase,
        prediction: u64,
        invocation: Option<CaptureInvocationShape>,
        window: Option<CaptureInvocationWindow>,
        source_tokens: u64,
        source_range: [u64; 2],
        actual_rows: usize,
        funding: HostMetadataFunding,
    ) -> Result<Self, PreparedRoutedInterventionError> {
        let result: Result<_, Cause> = (|| {
            funding.reserve_metadata(Self::control_bytes().ok_or(Geometry::Overflow)?)?;
            let plan = source.plan().admission();
            let point = plan.points().get(operation).ok_or(Geometry::Coordinates)?;
            let action = &plan
                .plan()
                .operations
                .get(operation)
                .ok_or(Geometry::Coordinates)?
                .action;
            let geometry = point
                .routed_units
                .as_ref()
                .ok_or(Geometry::Coordinates)?
                .geometry;
            let components = geometry
                .experts
                .checked_mul(geometry.units_per_expert)
                .ok_or(Geometry::Overflow)?;
            if source_tokens == 0
                || geometry.experts == 0
                || geometry.units_per_expert == 0
                || geometry.routes_per_token == 0
                || source_range[0] >= source_range[1]
                || source_range[1] > source_tokens
                || u64::try_from(actual_rows).map_err(|_| Geometry::Overflow)?
                    != (source_range[1] - source_range[0])
                        .checked_mul(geometry.routes_per_token)
                        .ok_or(Geometry::Overflow)?
            {
                return Err(Geometry::Coordinates.into());
            }
            let (logical, logical_origin, logical_tokens) = if let Some(window) = window {
                let physical = invocation.ok_or(Geometry::Coordinates)?;
                if physical
                    .batch
                    .checked_mul(physical.sequence)
                    .ok_or(Geometry::Overflow)?
                    != source_tokens
                {
                    return Err(Geometry::Coordinates.into());
                }
                let logical = window.validate_fixed(physical)?;
                (
                    Some(logical),
                    window.start,
                    logical
                        .batch
                        .checked_mul(logical.sequence)
                        .ok_or(Geometry::Overflow)?,
                )
            } else {
                (invocation, 0, source_tokens)
            };
            let mut selected = slice_destination(&funding, 2)?;
            let shape = [logical_tokens, components];
            let dtype = action.dtype().ok_or(Geometry::Coordinates)?;
            match logical {
                Some(invocation) => plan.resolve_prepared_invocation_at(
                    operation,
                    phase,
                    prediction,
                    invocation,
                    &shape,
                    dtype,
                    &mut selected,
                )?,
                None => plan.resolve_prepared_at(
                    operation,
                    phase,
                    prediction,
                    &shape,
                    dtype,
                    &mut selected,
                )?,
            }
            worker::validate_slice(geometry, &selected)?;
            let rows = funding.metadata_vec(actual_rows)?;
            Ok((selected, rows, geometry, logical_origin))
        })();
        match result {
            Ok((selected, rows, geometry, logical_origin)) => Ok(Self {
                selected,
                rows,
                geometry,
                source_tokens,
                source_range,
                logical_origin,
                expected_rows: actual_rows,
                failed: None,
                operation,
                phase,
                prediction,
                declaration: source.clone(),
                funding,
            }),
            Err(cause) => Err(PreparedRoutedInterventionError {
                cause,
                _declaration: source.clone(),
                _funding: funding,
            }),
        }
    }
    /// Borrowed exact bank declaration, for validating the same five native loans.
    pub fn geometry(&self) -> RoutedUnitGeometry {
        self.geometry
    }
    /// Physical full invocation token extent, before the optional logical window.
    pub fn source_tokens(&self) -> u64 {
        self.source_tokens
    }
    /// Physical chunk range; row coordinates have not yet been window shifted.
    pub fn source_range(&self) -> [u64; 2] {
        self.source_range
    }
    /// Exact native row count accepted at preparation.
    pub fn expected_rows(&self) -> usize {
        self.expected_rows
    }
    /// Append one scalar native coordinate without allocation or source cloning.
    /// Any failure poisons this destination; the caller keeps custody until drop.
    pub fn push_row(&mut self, mut row: RoutedUnitLocation) -> Result<(), Geometry> {
        if let Some(cause) = self.failed {
            return Err(cause);
        }
        let result = (|| {
            if self.rows.len() == self.expected_rows
                || row.source_peer.is_some()
                || row.token < self.source_range[0]
                || row.token >= self.source_range[1]
            {
                return Err(Geometry::Coordinates);
            }
            row.token = row
                .token
                .checked_add(self.logical_origin)
                .ok_or(Geometry::Overflow)?;
            self.rows.push(row);
            Ok(())
        })();
        if let Err(cause) = result {
            self.failed = Some(cause);
        }
        result
    }
    /// Poison the destination after the enclosing native reader fails. Its exact
    /// native cause remains with that reader's ordinary error transport.
    pub fn reject_source(&mut self) {
        self.failed.get_or_insert(Geometry::Coordinates);
    }
    /// Lower only after all actual rows have arrived. Every scratch/result vector
    /// is reserved through the same cumulative account before construction.
    pub fn finish(self) -> Result<PreparedRoutedIntervention, PreparedRoutedInterventionError> {
        let Self {
            selected,
            rows,
            geometry,
            expected_rows,
            failed,
            operation,
            phase,
            prediction,
            declaration,
            funding,
            source_tokens,
            source_range,
            ..
        } = self;
        let result: Result<_, Cause> = (|| {
            if let Some(cause) = failed {
                return Err(cause.into());
            }
            if rows.len() != expected_rows {
                return Err(Geometry::Coordinates.into());
            }
            let action = &declaration
                .plan()
                .admission()
                .plan()
                .operations
                .get(operation)
                .ok_or(Geometry::Coordinates)?
                .action;
            let lowered = worker::lower(
                geometry,
                &rows,
                &selected,
                action,
                None,
                &mut Paid(&funding),
            )?;
            let mut local = slice_destination(&funding, 1)?;
            let n = u64::try_from(lowered.indices.len()).map_err(|_| Geometry::Overflow)?;
            local.starts[0] = 0;
            local.ends[0] = n;
            local.strides[0] = 1;
            local.shape[0] = n;
            Ok((lowered, local))
        })();
        drop(rows);
        drop(selected);
        match result {
            Ok((lowered, local)) => Ok(PreparedRoutedIntervention {
                lowered,
                local,
                source_tokens,source_range,
                operation,
                phase,
                prediction,
                declaration,
                _funding: funding,
            }),
            Err(cause) => Err(PreparedRoutedInterventionError {
                cause,
                _declaration: declaration,
                _funding: funding,
            }),
        }
    }
    /// Upper bound from the actual chunk rows and immutable action only. This
    /// includes exact Vec construction/error controls, not native execution cost.
    pub fn required_bytes(
        source: &OriginalInterventionSource,
        operation: usize,
        actual_rows: usize,
    ) -> Option<usize> {
        let plan = source.plan().admission();
        let action = &plan.plan().operations.get(operation)?.action;
        let geometry = plan
            .points()
            .get(operation)?
            .routed_units
            .as_ref()?
            .geometry;
        let values = actual_rows.checked_mul(usize::try_from(geometry.units_per_expert).ok()?)?;
        let mut bytes = Self::control_bytes()?;
        for n in [
            WorkspaceContext::metadata_vec_bytes::<RoutedUnitLocation>(actual_rows)?,
            WorkspaceContext::metadata_vec_bytes::<RowKey>(actual_rows)?,
            WorkspaceContext::metadata_vec_bytes::<u64>(values)?,
            WorkspaceContext::metadata_vec_bytes::<usize>(values)?,
        ] {
            bytes = bytes.checked_add(n)?;
        }
        for rank in [2, 1] {
            for _ in 0..4 {
                bytes = bytes.checked_add(WorkspaceContext::metadata_vec_bytes::<u64>(rank)?)?;
            }
        }
        if let InterventionAction::MaskComponents { indices, .. } = action {
            bytes =
                bytes.checked_add(WorkspaceContext::metadata_vec_bytes::<u32>(indices.len())?)?;
        }
        if let InterventionAction::Replace { tensor } | InterventionAction::Add { tensor } = action
        {
            bytes = bytes.checked_add(WorkspaceContext::metadata_vec_bytes::<u64>(1)?)?;
            bytes = bytes.checked_add(match &tensor.values {
                InterventionValues::Float32(_) => {
                    WorkspaceContext::metadata_vec_bytes::<f32>(values)?
                }
                InterventionValues::Float16(_) | InterventionValues::Bfloat16(_) => {
                    WorkspaceContext::metadata_vec_bytes::<u16>(values)?
                }
            })?;
        }
        Some(bytes)
    }
    /// Enclosing owner, source resolution and shared lowering call frames. Vector
    /// backing/refusal control is separately queried by required_bytes.
    pub fn control_bytes() -> Option<usize> {
        let parts = [
            worker::selection_control_bytes()?,
            size_of::<(&HostMetadataFunding, usize)>(),
            size_of::<Result<ResolvedCaptureSlice, Cause>>(),
            size_of::<(&OriginalInterventionSource, usize, usize)>(),
            size_of::<std::ops::Range<usize>>(),
            size_of::<[usize; 4]>(),
            size_of::<Self>(),
            size_of::<PreparedRoutedIntervention>(),
            size_of::<PreparedRoutedInterventionError>(),
            size_of::<Cause>(),
            size_of::<OriginalInterventionSource>(),
            size_of::<HostMetadataFunding>(),
            size_of::<Result<Self, PreparedRoutedInterventionError>>(),
            size_of::<Result<PreparedRoutedIntervention, PreparedRoutedInterventionError>>(),
            size_of::<ResolvedCaptureSlice>(),
            size_of::<LoweredRoutedIntervention>(),
            size_of::<InterventionAction>(),
            size_of::<InterventionTensor>(),
            size_of::<InterventionValues>(),
            size_of::<Option<(Vec<u32>, bool)>>(),
            size_of::<Vec<RowKey>>(),
            size_of::<Vec<u64>>(),
            size_of::<Vec<usize>>(),
            size_of::<Paid<'_>>(),
            size_of::<(
                &OriginalInterventionSource,
                usize,
                CapturePhase,
                u64,
                Option<CaptureInvocationShape>,
                Option<CaptureInvocationWindow>,
                u64,
                [u64; 2],
                usize,
                HostMetadataFunding,
            )>(),
            size_of::<(
                RoutedUnitGeometry,
                &[RoutedUnitLocation],
                &ResolvedCaptureSlice,
                &InterventionAction,
                Option<&eredu_core::component::RoutedComponentCoordinateMap>,
                &mut Paid<'_>,
            )>(),
            size_of::<Result<LoweredRoutedIntervention, Cause>>(),
            size_of::<Result<(LoweredRoutedIntervention, ResolvedCaptureSlice), Cause>>(),
            size_of::<
                Result<
                    (
                        ResolvedCaptureSlice,
                        Vec<RoutedUnitLocation>,
                        RoutedUnitGeometry,
                        u64,
                    ),
                    Cause,
                >,
            >(),
            size_of::<Result<(), InterventionGeometryError>>(),
            size_of::<Result<(), HostMetadataFundingError>>(),
            size_of::<std::slice::Windows<'_, RowKey>>(),
            size_of::<std::iter::Enumerate<std::slice::Iter<'_, RoutedUnitLocation>>>(),
            size_of::<std::ops::Range<u64>>(),
            size_of::<[u64; 12]>(),
            size_of::<[usize; 8]>(),
            size_of::<(&mut Self, RoutedUnitLocation)>(),
            size_of::<Result<(), Geometry>>(),
            gather_controls::<f32>(),
            gather_controls::<u16>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}
fn gather_controls<T>() -> usize {
    size_of::<(&InterventionTensor, u64, &[usize], &mut Paid<'_>)>()
        + size_of::<(&[T], &[usize], &mut Paid<'_>)>()
        + size_of::<Vec<T>>()
        + size_of::<Result<Vec<T>, Cause>>()
        + size_of::<Result<InterventionTensor, Cause>>()
        + size_of::<std::slice::Iter<'static, usize>>()
        + size_of::<T>()
}
fn slice_destination(
    funding: &HostMetadataFunding,
    rank: usize,
) -> Result<ResolvedCaptureSlice, Cause> {
    let vector = || -> Result<Vec<u64>, Cause> {
        let mut values = funding.metadata_vec(rank)?;
        values.resize(rank, 0);
        Ok(values)
    };
    Ok(ResolvedCaptureSlice {
        starts: vector()?, ends: vector()?, strides: vector()?, shape: vector()?,
    })
}
impl PreparedRoutedIntervention {
    /// Immutable original declaration; an observer must compare its own source.
    pub fn source(&self) -> &OriginalInterventionSource {
        &self.declaration
    }
    /// Exact operation, independent phase and prediction retained at preparation.
    pub fn coordinates(&self) -> (usize, CapturePhase, u64) {
        (self.operation, self.phase, self.prediction)
    }
    /// Actual physical source extent and chunk consumed by the scalar reader.
    pub fn source_chunk(&self)->(u64,[u64;2]){(self.source_tokens,self.source_range)}
    /// Native flattened indices in original native row order.
    pub fn indices(&self) -> &[u64] {
        &self.lowered.indices
    }
    /// Borrowed gathered action; None means no participating selected value.
    pub fn action(&self) -> Option<&InterventionAction> {
        self.lowered.action.as_ref()
    }
    /// Exact one-dimensional replacement region for the existing action worker.
    pub fn local_slice(&self) -> &ResolvedCaptureSlice {
        &self.local
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
