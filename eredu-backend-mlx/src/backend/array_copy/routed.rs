//! Completed original route coordinates copied directly into the fixed sparse target.
use super::CaptureTensorNativeError as Error;
mod intervention;
mod model;
pub(crate) use model::{execute as execute_speculative_routed_capture, control_bytes as speculative_routed_capture_control_bytes};
pub(crate) use intervention::RoutedInterventionReadError;
use eredu_core::capture::{RoutedUnitCaptureSource, RoutedUnitGeometry};
use eredu_runtime::working_memory::{
    CaptureRoutedPrefillWriter, CaptureRoutedPrefillTransfer, CaptureRunHostError,
    CaptureRoutedBatchTransfer, CaptureRoutedModelTransfer,
};
use safemlx::{Array, Dtype, EvaluatedArray};

/// Exact five completed source loans. No tensor, index vector or row DTO is born.
/// This object supplies neither completion nor source registration authority.
pub(crate) struct CompletedRoutedCaptureSource<'a> {
    inputs: [EvaluatedArray<'a>; 5],
    token_offset: u64,
    global_groups: Option<&'a [usize]>,
    bank: RoutedUnitGeometry,
    rows: usize,
    chunk_tokens: u64,
    source_tokens: u64,
}
#[derive(Debug, PartialEq)]
struct Route {
    token: u64,
    slot: u64,
    expert: u64,
    coefficient: f32,
}
impl<'a> CompletedRoutedCaptureSource<'a> {
    /// Completed loans must borrow the same five actual source descriptors.
    /// The accepted owner settles/registers all five before this constructor.
    pub(crate) fn new(
        source: &RoutedUnitCaptureSource<'a, Array>,
        inputs: [EvaluatedArray<'a>; 5],
        bank: RoutedUnitGeometry,
        source_tokens: u64,
    ) -> Result<Self, Error> {
        let geometry = Self::validate_borrowed(source, bank, source_tokens)?;
        Self::bind(source, inputs, bank, source_tokens, geometry)
    }
    fn bind(source: &RoutedUnitCaptureSource<'a, Array>, inputs: [EvaluatedArray<'a>; 5],
        bank: RoutedUnitGeometry, source_tokens: u64, (rows, chunk_tokens): (usize, u64),
    ) -> Result<Self, Error> {
        let borrowed = [
            source.values,
            source.token_indices,
            source.selection_indices,
            source.coefficients,
            source.source_groups,
        ];
        for (actual, expected) in inputs.iter().zip(borrowed) {
            if !std::ptr::eq(actual.as_array(), expected) {
                return Err(Error::SourceChanged);
            }
        }
        Ok(Self {
            inputs,
            token_offset: source.token_offset,
            global_groups: source.global_groups,
            bank,
            rows,
            chunk_tokens,
            source_tokens,
        })
    }
    /// Read-only admission check over the actual five native descriptors.
    /// Does not evaluate, retain a handle, allocate a snapshot or read values.
    pub(crate) fn validate_borrowed(
        source: &RoutedUnitCaptureSource<'_, Array>,
        bank: RoutedUnitGeometry,
        source_tokens: u64,
    ) -> Result<(usize, u64), Error> {
        Self::validate_layouts(
            [source.values.shape(), source.token_indices.shape(), source.selection_indices.shape(),
             source.coefficients.shape(), source.source_groups.shape()],
            [source.values.dtype(), source.token_indices.dtype(), source.selection_indices.dtype(),
             source.coefficients.dtype(), source.source_groups.dtype()],
            source.token_offset, bank, source_tokens)
    }
    /// Shared read-only native layout contract used by cold and completed sources.
    pub(crate) fn validate_layouts(
        shapes: [&[i32]; 5], dtypes: [Dtype; 5], token_offset: u64,
        bank: RoutedUnitGeometry, source_tokens: u64,
    ) -> Result<(usize, u64), Error> {
        let geometry = Self::validate_geometry(shapes, token_offset, bank, source_tokens)?;
        Self::validate_dtypes(dtypes)?;
        Ok(geometry)
    }
    fn validate_dtypes(dtypes: [Dtype; 5]) -> Result<(), Error> {
        for index in [0, 3] {
            if !matches!(
                dtypes[index],
                Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16
            ) {
                return Err(Error::UnsupportedDtype(dtypes[index]));
            }
        }
        for index in [1, 2, 4] {
            if !matches!(
                dtypes[index],
                Dtype::Uint32 | Dtype::Int32
            ) {
                return Err(Error::UnsupportedDtype(dtypes[index]));
            }
        }
        Ok(())
    }
    /// Source-independent shape/coordinate contract. Preliminary workspace
    /// geometry cannot certify a native dtype or any execution authority.
    pub(crate) fn validate_geometry(
        shapes: [&[i32]; 5], token_offset: u64,
        bank: RoutedUnitGeometry, source_tokens: u64,
    ) -> Result<(usize, u64), Error> {
        if bank.units_per_expert == 0 { return Err(Error::ShapeMismatch); }
        Self::validate_physical_geometry(shapes, token_offset, bank, source_tokens)
    }
    // An explicitly retained empty local unit map is valid for a partition.
    // Global bank validation and serial callers still require positive units.
    fn validate_physical_geometry(
        shapes: [&[i32]; 5], token_offset: u64,
        bank: RoutedUnitGeometry, source_tokens: u64,
    ) -> Result<(usize, u64), Error> {
        let values = shapes[0];
        let coefficients = shapes[3];
        let groups = shapes[4];
        if bank.experts == 0
            || bank.routes_per_token == 0
            || values.len() != 2
            || coefficients.len() != 2
            || groups.len() != 2
            || values[0] < 0
            || coefficients[0] < 0
            || groups[0] < 0
            || values[1] as u64 != bank.units_per_expert
            || coefficients[1] as u64 != bank.routes_per_token
            || groups[1] as u64 != bank.routes_per_token
            || groups[0] as u64 != source_tokens
            || shapes[1] != [values[0]]
            || shapes[2] != [values[0]]
            || (values[0] as u64)
                > (coefficients[0] as u64)
                    .checked_mul(bank.routes_per_token)
                    .ok_or(Error::GeometryOverflow)?
            || token_offset
                .checked_add(coefficients[0] as u64)
                .is_none_or(|end| end > source_tokens)
        {
            return Err(Error::ShapeMismatch);
        }
        Ok((values[0] as usize, coefficients[0] as u64))
    }
    fn integer(&self, input: usize, index: usize) -> Result<u64, Error> {
        let value = &self.inputs[input];
        match value.as_array().dtype() {
            Dtype::Uint32 => value.try_get::<u32>(index)?.map(u64::from),
            Dtype::Int32 => value
                .try_get::<i32>(index)?
                .and_then(|value| value.try_into().ok()),
            _ => None,
        }
        .ok_or(Error::ShapeMismatch)
    }
    fn floating(&self, input: usize, index: usize) -> Result<f32, Error> {
        let value = &self.inputs[input];
        match value.as_array().dtype() {
            Dtype::Float32 => value.try_get::<f32>(index)?,
            Dtype::Float16 => value.try_get::<half::f16>(index)?.map(half::f16::to_f32),
            Dtype::Bfloat16 => value.try_get::<half::bf16>(index)?.map(half::bf16::to_f32),
            _ => None,
        }
        .ok_or(Error::ShapeMismatch)
    }
    fn route(&self, index: usize) -> Result<Route, Error> {
        let token = self.integer(1, index)?;
        let selection = self.integer(2, index)?;
        if token >= self.chunk_tokens || selection / self.bank.routes_per_token != token {
            return Err(Error::ShapeMismatch);
        }
        let physical = self
            .token_offset
            .checked_add(token)
            .ok_or(Error::GeometryOverflow)?;
        let slot = selection % self.bank.routes_per_token;
        let group_index = physical
            .checked_mul(self.bank.routes_per_token)
            .and_then(|start| start.checked_add(slot))
            .ok_or(Error::GeometryOverflow)?;
        let group = self.integer(
            4,
            usize::try_from(group_index).map_err(|_| Error::GeometryOverflow)?,
        )?;
        let expert = match self.global_groups {
            Some(map) => *map
                .get(usize::try_from(group).map_err(|_| Error::GeometryOverflow)?)
                .ok_or(Error::ShapeMismatch)? as u64,
            None => group,
        };
        let coefficient = self.floating(
            3,
            usize::try_from(selection).map_err(|_| Error::GeometryOverflow)?,
        )?;
        if expert >= self.bank.experts || !coefficient.is_finite() {
            return Err(Error::ShapeMismatch);
        }
        Ok(Route {
            token: physical,
            slot,
            expert,
            coefficient,
        })
    }
    pub(crate) fn control_bytes() -> Option<usize> {
        let frames = [
            std::mem::size_of::<Self>(),
            std::mem::size_of::<Result<Self, Error>>(),
            std::mem::size_of::<Route>(),
            std::mem::size_of::<Result<Route, Error>>(),
            std::mem::size_of::<[&Array; 5]>(),
            std::mem::size_of::<[&[i32]; 5]>(),
            std::mem::size_of::<[Dtype; 5]>(),
            std::mem::size_of::<(usize, u64)>(),
            std::mem::size_of::<(&RoutedUnitCaptureSource<'_, Array>, [EvaluatedArray<'_>; 5],
                RoutedUnitGeometry, u64, (usize, u64))>(),
            std::mem::size_of::<([&[i32]; 5], u64, RoutedUnitGeometry, u64)>(),
            std::mem::size_of::<[usize; 3]>(),
            EvaluatedArray::lookup_control_bytes::<u32>()?,
            EvaluatedArray::lookup_control_bytes::<i32>()?,
            EvaluatedArray::lookup_control_bytes::<f32>()?,
            EvaluatedArray::lookup_control_bytes::<half::f16>()?,
            EvaluatedArray::lookup_control_bytes::<half::bf16>()?,
            std::mem::size_of::<RoutedCaptureTransfer<'_, '_, '_, '_, '_>>(),
            std::mem::size_of::<CaptureRoutedPrefillWriter<'_, '_, '_, '_>>(),
            std::mem::size_of::<CaptureRoutedPrefillTransfer<'_, '_, '_, '_, '_, crate::backend::runtime::residency::storage::StorageIdentity>>(),
            std::mem::size_of::<(usize, u64, u64, u64)>(),
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
    /// Consume a short exclusive claim from the same native account. Every
    /// scalar goes directly into its already funded fixed destination; a failed
    /// source or partial copy leaves the payload and account in the frame.
    pub(crate) fn copy_routed(
        self,
        mut writer: RoutedCaptureTransfer<'_, '_, '_, '_, '_>,
    ) -> Result<(), Error> {
        writer.validate()?;
        if self.source_tokens != writer.source_tokens()
            || self.bank != writer.geometry().bank()
        {
            return Err(Error::ClaimMismatch);
        }
        let geometry = writer.geometry();
        let unit_start = geometry.starts()[2];
        let unit_stride = geometry.strides()[2];
        let units = geometry.shape()[2];
        for index in 0..self.rows {
            let route = self.route(index)?;
            if !writer.selects(route.token, route.slot) {
                continue;
            }
            writer.begin_row(route.token, route.slot, route.expert, route.coefficient)
                .map_err(CaptureRunHostError::from)?;
            let base = (index as u64).checked_mul(self.bank.units_per_expert)
                .ok_or(Error::GeometryOverflow)?;
            for unit in 0..units {
                let offset = (unit as u64).checked_mul(unit_stride)
                    .and_then(|offset| unit_start.checked_add(offset))
                    .and_then(|offset| base.checked_add(offset))
                    .ok_or(Error::GeometryOverflow)?;
                let value = self.floating(
                    0, usize::try_from(offset).map_err(|_| Error::GeometryOverflow)?)?;
                writer.push_f32(value).map_err(CaptureRunHostError::from)?;
            }
            writer.finish_row().map_err(CaptureRunHostError::from)?;
        }
        writer.source_chunk(self.token_offset, self.token_offset + self.chunk_tokens)
            .map_err(CaptureRunHostError::from)?;
        writer.finish().map_err(CaptureRunHostError::from)?;
        Ok(())
    }

}

type Identity = crate::backend::runtime::residency::storage::StorageIdentity;

/// One static scalar producer serves chunked and ordinary frame destinations.
pub(crate) enum RoutedCaptureTransfer<'t, 'f, 'p, 'a, 's> {
    Prefill(CaptureRoutedPrefillTransfer<'t, 'f, 'p, 'a, 's, Identity>),
    Invocation(CaptureRoutedBatchTransfer<'t, 'a, 's, Identity>),
    Model(CaptureRoutedModelTransfer<'t, 'a, 's>),
}
impl RoutedCaptureTransfer<'_, '_, '_, '_, '_> {
    fn geometry(&self) -> &eredu_core::capture::CaptureRoutedUnitsGeometry<'_> {
        match self {
            Self::Prefill(writer) => writer.fragment().plan().geometry(),
            Self::Invocation(writer) => writer.geometry(),
            Self::Model(writer) => writer.geometry(),
        }
    }
    fn source_tokens(&self) -> u64 {
        match self {
            Self::Prefill(writer) => writer.fragment().source_tokens(),
            Self::Invocation(writer) => writer.geometry().source_shape()[0] as u64,
            Self::Model(writer) => writer.geometry().source_shape()[0] as u64,
        }
    }
    fn selects(&self, token: u64, slot: u64) -> bool {
        match self {
            Self::Prefill(writer) => writer.fragment().selects(token, slot),
            Self::Invocation(writer) => writer.selects(token, slot),
            Self::Model(writer) => writer.selects(token, slot),
        }
    }
    fn validate(&self) -> Result<(), CaptureRunHostError> {
        match self { Self::Prefill(w) => w.validate(), Self::Invocation(w) => w.validate().map_err(Into::into), Self::Model(w) => w.validate().map_err(Into::into) }
    }
    fn begin_row(&mut self, token: u64, slot: u64, expert: u64, coefficient: f32)
        -> Result<(), CaptureRunHostError> {
        match self {
            Self::Prefill(w) => w.begin_row(token, slot, expert, coefficient),
            Self::Invocation(w) => w.begin_row(token, slot, expert, coefficient).map_err(Into::into), Self::Model(w) => w.begin_row(token, slot, expert, coefficient).map_err(Into::into),
        }
    }
    fn push_f32(&mut self, value: f32)
        -> Result<(), CaptureRunHostError> {
        match self { Self::Prefill(w) => w.push_f32(value), Self::Invocation(w) => w.push_f32(value).map_err(Into::into), Self::Model(w) => w.push_f32(value).map_err(Into::into) }
    }
    fn finish_row(&mut self) -> Result<(), CaptureRunHostError> {
        match self { Self::Prefill(w) => w.finish_row(), Self::Invocation(w) => w.finish_row().map_err(Into::into), Self::Model(w) => w.finish_row().map_err(Into::into) }
    }
    fn source_chunk(&mut self, start: u64, end: u64)
        -> Result<(), CaptureRunHostError> {
        match self {
            Self::Prefill(w) => w.source_chunk(start, end),
            Self::Invocation(w) => w.source_chunk(start, end).map_err(Into::into), Self::Model(w) => w.source_chunk(start, end).map_err(Into::into),
        }
    }
    fn finish(self) -> Result<(), CaptureRunHostError> {
        match self { Self::Prefill(w) => w.finish(), Self::Invocation(w) => w.finish().map_err(Into::into), Self::Model(w) => w.finish().map_err(Into::into) }
    }
}

mod partition;
pub(crate) use partition::{CompletedPartitionRoutedCaptureSource, PartitionRoutedCaptureLayout};

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn completed_sparse_sources_resolve_original_routes_and_strided_half_values() {
        let stream = safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let values = Array::from_slice(
            &(1..=12)
                .map(|v| half::f16::from_f32(v as f32 * 0.25))
                .collect::<Vec<_>>(),
            &[4, 3],
        );
        let values = values
            .as_strided(&[4, 3][..], &[3, -1][..], 2, &stream)
            .unwrap();
        let tokens = Array::from_slice(&[1_u32, 0, 1, 0], &[4]);
        let slots = Array::from_slice(&[3_i32, 0, 2, 1], &[4]);
        let coefficients = Array::from_slice(&[0.125_f32, 0.875, 0.25, 0.75], &[2, 2]);
        let groups = Array::from_slice(&[0_u32, 1, 2, 0, 1, 2], &[3, 2]);
        let source = RoutedUnitCaptureSource {
            values: &values,
            token_indices: &tokens,
            selection_indices: &slots,
            coefficients: &coefficients,
            source_groups: &groups,
            token_offset: 1,
            global_groups: Some(&[4, 1, 3]),
        };
        let bank = RoutedUnitGeometry {
            experts: 5,
            units_per_expert: 3,
            routes_per_token: 2,
        };
        let completed = || {
            [
                values.evaluated().unwrap(),
                tokens.evaluated().unwrap(),
                slots.evaluated().unwrap(),
                coefficients.evaluated().unwrap(),
                groups.evaluated().unwrap(),
            ]
        };
        let reader = CompletedRoutedCaptureSource::new(&source, completed(), bank, 3).unwrap();
        assert_eq!(
            reader.route(0).unwrap(),
            Route {
                token: 2,
                slot: 1,
                expert: 3,
                coefficient: 0.75
            }
        );
        assert_eq!(
            reader.route(1).unwrap(),
            Route {
                token: 1,
                slot: 0,
                expert: 3,
                coefficient: 0.125
            }
        );
        assert_eq!(
            (0..12)
                .map(|index| reader.floating(0, index).unwrap())
                .collect::<Vec<_>>(),
            vec![0.75, 0.5, 0.25, 1.5, 1.25, 1.0, 2.25, 2.0, 1.75, 3.0, 2.75, 2.5]
        );
        assert!(reader.route(4).is_err());
        assert!(CompletedRoutedCaptureSource::new(&source, completed(), bank, 4).is_err());
        let foreign = Array::from_slice(&[1_u32, 0, 1, 0], &[4]);
        let mut substituted = completed();
        substituted[1] = foreign.evaluated().unwrap();
        assert!(matches!(
            CompletedRoutedCaptureSource::new(&source, substituted, bank, 3),
            Err(Error::SourceChanged)
        ));
    }
}
