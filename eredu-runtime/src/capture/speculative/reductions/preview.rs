//! One prepaid logical prefix; physical values keep their existing records.
use super::*;
use crate::capture::prefix::PrefixDestination;
use eredu_core::{
    ObservationDtype, TensorObservation, TensorObservationData, checkpoint::TensorDtype,
};

#[derive(Debug, thiserror::Error)]
pub(in crate::capture) enum PreviewError {
    #[error("{0}")]
    Invalid(&'static str),
    #[error("Preview arithmetic overflow")]
    Overflow,
    #[error(transparent)]
    Window(#[from] CaptureWindowError),
    #[error(transparent)]
    Prefix(#[from] crate::capture::prefix::PrefixError),
}
impl From<PreviewError> for CaptureError {
    fn from(value: PreviewError) -> Self {
        match value {
            PreviewError::Invalid(message) => Self::Invalid(message.into()),
            PreviewError::Overflow => Self::Overflow,
            PreviewError::Window(error) => error.into(),
            PreviewError::Prefix(error) => error.into(),
        }
    }
}
fn invalid(message: &'static str) -> PreviewError {
    PreviewError::Invalid(message)
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    F32,
    I64,
    U64,
    Bool,
}
impl Kind {
    fn source(dtype: &TensorDtype) -> Result<Self, PreviewError> {
        Ok(match dtype {
            TensorDtype::F16 | TensorDtype::Bf16 | TensorDtype::F32 | TensorDtype::F64 => Self::F32,
            TensorDtype::I8 | TensorDtype::I16 | TensorDtype::I32 | TensorDtype::I64 => Self::I64,
            TensorDtype::U8 | TensorDtype::U16 | TensorDtype::U32 | TensorDtype::U64 => Self::U64,
            TensorDtype::Bool => Self::Bool,
            _ => return Err(invalid("Preview requires actual scalar source precision")),
        })
    }
    fn matches(self, values: &TensorObservationData) -> bool {
        matches!(
            (self, values),
            (Self::F32, TensorObservationData::F32(_))
                | (Self::I64, TensorObservationData::I64(_))
                | (Self::U64, TensorObservationData::U64(_))
                | (Self::Bool, TensorObservationData::Bool(_))
        )
    }
    fn allocate(self, count: usize) -> TensorObservationData {
        self.allocate_with_metadata(count, Metadata::ordinary())
            .expect("ordinary typed Preview destination")
    }
    fn allocation_controls() -> Option<usize> {
        std::mem::size_of::<(Self, usize, Metadata<'_>, TensorObservationData)>().checked_add(
            std::mem::size_of::<Result<TensorObservationData, ConstructionError>>(),
        )
    }
    fn allocate_with_metadata(
        self,
        count: usize,
        metadata: Metadata<'_>,
    ) -> Result<TensorObservationData, ConstructionError> {
        metadata.controls(Self::allocation_controls().ok_or(ConstructionError::Overflow)?)?;
        Ok(match self {
            Self::F32 => TensorObservationData::F32(metadata.zeros(count, 0.0f32)?),
            Self::I64 => TensorObservationData::I64(metadata.zeros(count, 0i64)?),
            Self::U64 => TensorObservationData::U64(metadata.zeros(count, 0u64)?),
            Self::Bool => TensorObservationData::Bool(metadata.zeros(count, false)?),
        })
    }
}

#[derive(Clone, Copy)]
pub(in crate::capture) struct Plan {
    maximum: u64,
    output: usize,
    declared: ObservationDtype,
    width: usize,
}
impl Plan {
    pub(in crate::capture) fn prepare(
        point: &eredu_core::ObservationPoint,
        maximum: u64,
        geometry: &Geometry,
    ) -> Result<Self, CaptureError> {
        Self::prepare_fixed(point, maximum, geometry).map_err(Into::into)
    }
    pub(in crate::capture) fn prepare_fixed(
        point: &eredu_core::ObservationPoint,
        maximum: u64,
        geometry: &Geometry,
    ) -> Result<Self, PreviewError> {
        let width = match point.dtype {
            ObservationDtype::Floating => std::mem::size_of::<f32>(),
            ObservationDtype::Integer => std::mem::size_of::<u64>(),
            ObservationDtype::Boolean => std::mem::size_of::<bool>(),
            ObservationDtype::Unknown => {
                return Err(invalid(
                    "logical Preview requires a declared scalar category",
                ));
            }
        };
        let output = usize::try_from(maximum.min(geometry.elements()))
            .map_err(|_| PreviewError::Overflow)?;
        output
            .checked_mul(width)
            .filter(|bytes| *bytes <= isize::MAX as usize)
            .ok_or(PreviewError::Overflow)?;
        Ok(Self {
            maximum,
            output,
            declared: point.dtype,
            width,
        })
    }
    pub(in crate::capture) fn usage(self) -> Result<CaptureUsage, CaptureError> {
        Ok(CaptureUsage {
            host_bytes: add(
                std::mem::size_of::<usize>() as u64,
                mul(self.output as u64, self.width as u64)?,
            )?,
            // Existing raw tensor encoding bound; global Truncated fields are
            // covered by record metadata and checked at terminal serialization.
            encoded_bytes: add(128, mul(self.output as u64, 32)?)?,
            ..Default::default()
        })
    }
    pub(in crate::capture) fn allocate(self) -> (Preview, Option<CapturePayload>) {
        self.allocate_with_metadata(Metadata::ordinary())
            .expect("ordinary Preview destination")
    }
    pub(in crate::capture) fn allocate_with_metadata(
        self,
        metadata: Metadata<'_>,
    ) -> Result<(Preview, Option<CapturePayload>), ConstructionError> {
        use std::mem::{size_of, size_of_val};
        let controls = [
            size_of::<Self>(),
            size_of::<Preview>(),
            size_of::<Metadata<'_>>(),
            size_of::<Option<Kind>>(),
            size_of::<Option<Vec<usize>>>(),
            size_of::<Option<CapturePayload>>(),
            size_of::<TensorObservation>(),
            size_of::<Result<(Preview, Option<CapturePayload>), ConstructionError>>(),
        ];
        metadata.controls(
            controls
                .into_iter()
                .try_fold(size_of_val(&controls), usize::checked_add)
                .ok_or(ConstructionError::Overflow)?,
        )?;
        let kind = match self.declared {
            ObservationDtype::Floating => Some(Kind::F32),
            ObservationDtype::Boolean => Some(Kind::Bool),
            ObservationDtype::Integer => None,
            ObservationDtype::Unknown => unreachable!("validated category"),
        };
        let mut shape = Some(metadata.copy(&[self.output])?);
        let payload = match kind {
            Some(kind) => Some(CapturePayload::Tensor(
                TensorObservation::new(
                    shape.take().expect("single prefix shape"),
                    kind.allocate_with_metadata(self.output, metadata)?,
                )
                .expect("fixed prefix shape"),
            )),
            None => {
                // The source selects signedness only at the first actual row.
                // Both integer representations have the same 8-byte layout;
                // this private plan prepays its one eventual typed destination.
                metadata
                    .controls(Kind::allocation_controls().ok_or(ConstructionError::Overflow)?)?;
                metadata.deferred_vec::<u64>(self.output)?;
                None
            }
        };
        Ok((
            Preview {
                plan: self,
                shape,
                copied: 0,
                pending_copied: 0,
            },
            payload,
        ))
    }
}

pub(in crate::capture) struct Preview {
    plan: Plan,
    // Integer signedness is known only at the first actual source. The shape
    // and exact 8*N numeric allowance already exist; allocate one typed Vec.
    shape: Option<Vec<usize>>,
    copied: u64,
    pending_copied: u64,
}
fn mapping(
    geometry: &Geometry,
    start: u64,
    end: u64,
) -> Result<([u64; 32], [u64; 32]), PreviewError> {
    let mut local = *geometry.selected();
    local[geometry.axis()] = geometry.local_rows_fixed(start, end)?;
    let mut starts = [0; 32];
    starts[geometry.axis()] = start
        .saturating_sub(geometry.start())
        .div_ceil(geometry.stride())
        .min(geometry.selected()[geometry.axis()]);
    Ok((local, starts))
}
impl Preview {
    fn kind(&self, dtype: Option<&TensorDtype>) -> Result<Kind, PreviewError> {
        let kind =
            Kind::source(dtype.ok_or_else(|| invalid("Preview source precision is absent"))?)?;
        if !matches!(
            (self.plan.declared, kind),
            (ObservationDtype::Floating, Kind::F32)
                | (ObservationDtype::Integer, Kind::I64 | Kind::U64)
                | (ObservationDtype::Boolean, Kind::Bool)
        ) {
            return Err(invalid("Preview source category differs from admission"));
        }
        Ok(kind)
    }
    pub(in crate::capture) fn validate(
        &mut self,
        geometry: &Geometry,
        physical: &CaptureRecord,
        current: &CaptureRecord,
        start: u64,
        end: u64,
        count: u64,
    ) -> Result<(), CaptureError> {
        self.validate_fixed(geometry, physical, current, start, end, count)
            .map_err(Into::into)
    }
    pub(in crate::capture) fn validate_fixed(
        &mut self,
        geometry: &Geometry,
        physical: &CaptureRecord,
        current: &CaptureRecord,
        start: u64,
        end: u64,
        count: u64,
    ) -> Result<(), PreviewError> {
        self.validate_value_fixed(
            geometry,
            physical.source_dtype.as_ref(),
            &physical.outcome,
            physical.payload.as_ref(),
            current.payload.as_ref(),
            start,
            end,
            count,
        )
    }
    pub(in crate::capture) fn validate_value(
        &mut self,
        geometry: &Geometry,
        dtype: Option<&TensorDtype>,
        outcome: &CaptureOutcome,
        physical: Option<&CapturePayload>,
        current: Option<&CapturePayload>,
        start: u64,
        end: u64,
        count: u64,
    ) -> Result<(), CaptureError> {
        self.validate_value_fixed(
            geometry, dtype, outcome, physical, current, start, end, count,
        )
        .map_err(Into::into)
    }
    pub(in crate::capture) fn validate_value_fixed(
        &mut self,
        geometry: &Geometry,
        dtype: Option<&TensorDtype>,
        outcome: &CaptureOutcome,
        physical: Option<&CapturePayload>,
        current: Option<&CapturePayload>,
        start: u64,
        end: u64,
        count: u64,
    ) -> Result<(), PreviewError> {
        let kind = self.kind(dtype)?; // required even for zero/no-overlap values
        if let Some(payload) = current {
            let value = payload
                .as_tensor()
                .ok_or_else(|| invalid("logical Preview payload changed"))?;
            if value.shape() != [self.plan.output]
                || value.data().len() != self.plan.output
                || !kind.matches(value.data())
            {
                return Err(invalid("logical Preview precision changed"));
            }
        }
        let absent = count == 0
            && matches!(
                outcome,
                CaptureOutcome::Skipped {
                    reason: CaptureSkipReason::NotInvoked
                }
            );
        let length = if absent {
            if physical.is_some() {
                return Err(invalid("no-overlap Preview retained unexpected values"));
            }
            0
        } else {
            if *outcome
                != crate::capture::completed_capture_outcome(
                    &CaptureTransform::Preview {
                        max_elements: self.plan.maximum,
                    },
                    count,
                )
            {
                return Err(invalid("physical Preview outcome changed"));
            }
            let value = physical
                .and_then(CapturePayload::as_tensor)
                .ok_or_else(|| invalid("physical Preview tensor is absent"))?;
            let length = usize::try_from(count.min(self.plan.maximum))
                .map_err(|_| PreviewError::Overflow)?;
            if value.shape() != [length]
                || value.data().len() != length
                || !kind.matches(value.data())
            {
                return Err(invalid("physical Preview shape or precision changed"));
            }
            length
        };
        let (local, starts) = mapping(geometry, start, end)?;
        let strides = [1; 32];
        let map = PrefixDestination::new_fixed(
            &geometry.selected()[..geometry.rank()],
            &local[..geometry.rank()],
            &starts[..geometry.rank()],
            &strides[..geometry.rank()],
        )?;
        let mut copied = 0;
        for ordinal in 0..length {
            let output = map
                .ordinal(ordinal as u64)
                .ok_or_else(|| invalid("physical Preview exceeds selected fragment"))?;
            if output < self.plan.output as u64 {
                copied += 1;
            }
        }
        self.pending_copied = self
            .copied
            .checked_add(copied)
            .ok_or(PreviewError::Overflow)?;
        if self.pending_copied > self.plan.output as u64
            || (end == geometry.source()[geometry.axis()]
                && self.pending_copied != self.plan.output as u64)
        {
            return Err(invalid("logical Preview prefix coverage is incomplete"));
        }
        Ok(())
    }
    pub(in crate::capture) fn update(
        &mut self,
        geometry: &Geometry,
        physical: &CaptureRecord,
        current: &mut CaptureRecord,
        start: u64,
        end: u64,
    ) {
        self.update_value(
            geometry,
            physical.source_dtype.as_ref(),
            physical.payload.as_ref(),
            &mut current.payload,
            start,
            end,
        );
    }
    pub(in crate::capture) fn update_value(
        &mut self,
        geometry: &Geometry,
        dtype: Option<&TensorDtype>,
        physical: Option<&CapturePayload>,
        current: &mut Option<CapturePayload>,
        start: u64,
        end: u64,
    ) {
        let kind = self
            .kind(dtype)
            .expect("all source rows validated before update");
        let (shape, mut values) = match current.take() {
            Some(CapturePayload::Tensor(tensor)) => tensor.into_parts(),
            None => (
                self.shape.take().expect("one prepaid integer shape"),
                kind.allocate(self.plan.output),
            ),
            _ => unreachable!("private logical Preview tensor"),
        };
        if let Some(input) = physical.and_then(CapturePayload::as_tensor) {
            let (local, starts) = mapping(geometry, start, end).expect("validated geometry");
            let strides = [1; 32];
            let map = PrefixDestination::new_fixed(
                &geometry.selected()[..geometry.rank()],
                &local[..geometry.rank()],
                &starts[..geometry.rank()],
                &strides[..geometry.rank()],
            )
            .expect("validated mapping");
            for ordinal in 0..input.data().len() {
                let output = map
                    .ordinal(ordinal as u64)
                    .expect("validated physical prefix");
                if output >= self.plan.output as u64 {
                    continue;
                }
                let output = output as usize;
                match (&mut values, input.data()) {
                    (TensorObservationData::F32(out), TensorObservationData::F32(input)) => {
                        out[output] = input[ordinal]
                    }
                    (TensorObservationData::I64(out), TensorObservationData::I64(input)) => {
                        out[output] = input[ordinal]
                    }
                    (TensorObservationData::U64(out), TensorObservationData::U64(input)) => {
                        out[output] = input[ordinal]
                    }
                    (TensorObservationData::Bool(out), TensorObservationData::Bool(input)) => {
                        out[output] = input[ordinal]
                    }
                    _ => unreachable!("validated actual precision"),
                }
            }
        }
        *current = Some(CapturePayload::Tensor(
            TensorObservation::new(shape, values).expect("unchanged fixed prefix"),
        ));
    }
    pub(in crate::capture) fn commit(&mut self) {
        self.copied = self.pending_copied;
    }
}

#[cfg(test)]
mod tests;
