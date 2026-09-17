//! Output captures share macro text accumulation without introducing a lexical scope.
use super::*;
impl Engine<'_, '_, '_> {
    pub(super) fn begin_output_capture(
        &mut self,
        mode: crate::output::CaptureMode,
    ) -> Result<(), Error> {
        if !matches!(mode, crate::output::CaptureMode::Capture) {
            return Err(Error::Geometry);
        }
        let range = self.value_range(3)?;
        let parent = match self.capture {
            Some(index) => Slot::Scalar(Scalar::U64(
                u64::try_from(index).map_err(|_| Error::Overflow)?,
            )),
            None => Slot::Undefined,
        };
        self.retain_value(range.start, parent)?;
        self.retain_value(
            range.start + 1,
            Slot::Scalar(Scalar::U64(
                u64::try_from(self.calls).map_err(|_| Error::Overflow)?,
            )),
        )?;
        self.retain_value(range.start + 2, Slot::Undefined)?;
        self.counts.values = range.start.checked_add(3).ok_or(Error::Overflow)?;
        self.capture = Some(range.start);
        Ok(())
    }
    fn output_capture_calls(&self, index: usize) -> Result<usize, Error> {
        match self.workspace.values.get(index + 1) {
            Some(Slot::Scalar(Scalar::U64(calls))) => {
                usize::try_from(*calls).map_err(|_| Error::Overflow)
            }
            _ => Err(Error::Geometry),
        }
    }
    pub(super) fn emit_into_capture(&mut self, value: Slot) -> Result<bool, Error> {
        let Some(index) = self.capture else {
            return Ok(false);
        };
        let calls = self.output_capture_calls(index)?;
        if calls > self.calls {
            return Err(Error::Geometry);
        }
        if calls != self.calls {
            return Ok(false);
        }
        let previous = match self.workspace.values[index + 2] {
            Slot::Undefined => None,
            Slot::Text(text) => Some(text),
            _ => return Err(Error::Geometry),
        };
        let output = self.append_captured_output(previous, value)?;
        self.retain_value(index + 2, output.map(Slot::Text).unwrap_or(Slot::Undefined))?;
        Ok(true)
    }
    pub(super) fn end_output_capture(&mut self) -> Result<Slot, Error> {
        let index = self.capture.ok_or(Error::Geometry)?;
        if self.output_capture_calls(index)? != self.calls {
            return Err(Error::Geometry);
        }
        let parent = match self.workspace.values[index] {
            Slot::Undefined => None,
            Slot::Scalar(Scalar::U64(parent)) => {
                Some(usize::try_from(parent).map_err(|_| Error::Overflow)?)
            }
            _ => return Err(Error::Geometry),
        };
        if parent.is_some_and(|parent| parent >= index) {
            return Err(Error::Geometry);
        }
        let output = match self.workspace.values[index + 2] {
            Slot::Undefined => {
                Slot::Text(measured::Measured::default().concat(self.counts.concat, 0))
            }
            Slot::Text(text) => Slot::Text(text),
            _ => return Err(Error::Geometry),
        };
        self.capture = parent;
        Ok(output)
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let frames = [
        size_of::<crate::output::CaptureMode>(),
        size_of::<Range>(),
        size_of::<[Slot; 5]>(),
        size_of::<[Option<Text>; 2]>(),
        size_of::<[Option<usize>; 3]>(),
        size_of::<[usize; 4]>(),
        size_of::<(&mut Engine<'_, '_, '_>, Option<Text>, Slot)>(),
        size_of::<Result<Option<Text>, Error>>(),
        size_of::<Result<bool, Error>>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
}
