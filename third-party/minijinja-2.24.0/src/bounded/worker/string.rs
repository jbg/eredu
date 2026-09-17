//! The ordinary string filter's text identity and default scalar formatting.
use super::{generated::Output, measured::Measured, *};

impl Engine<'_, '_, '_> {
    pub(super) fn apply_float(&mut self,arguments:Option<u16>)->Result<(),Error>{
        if arguments!=Some(1){return Err(Error::Geometry);}
        let value=self.pop()?;
        let value=match value {
            Slot::Undefined=>0.0,
            value @ Slot::Text(_)=>{
                let atom=self.text_atom(value)?;
                let text=atom_text(self.source,self.context,self.borrowed,self.workspace.context_text.as_deref(),atom)?;
                primitive::scalar::parse_float(text).map_err(|_|Error::Geometry)?
            }
            other=>primitive::scalar::filter_float(scalar(other).ok_or(Error::Geometry)?),
        };
        self.push(Slot::Scalar(Scalar::F64(value)))
    }

    pub(super) fn apply_string(&mut self, arguments: Option<u16>) -> Result<(), Error> {
        if arguments != Some(1) {
            return Err(Error::Geometry);
        }
        let value = self.pop()?;
        let value = self.string_value(value)?;
        self.push(value)
    }
    pub(super) fn string_value(&mut self, value: Slot) -> Result<Slot, Error> {
        if matches!(value, Slot::Text(_)) {
            // The ordinary filter keeps an existing string. The same paid
            // register transfer preserves a borrowed atom or generated result.
            return Ok(value);
        }
        let scalar = if matches!(value, Slot::Undefined) {
            None
        } else {
            Some(scalar(value).ok_or(Error::Geometry)?)
        };
        let (start, text_start) = self.generated_start()?;
        let mut output = Output {
            bytes: self.workspace.context_text.as_deref_mut(),
            start: text_start,
            measured: Measured::default(),
            error: None,
        };
        // Source admission fixes lenient undefined handling and plain text
        // autoescape. Undefined formats empty; scalars use the same numeric
        // spelling and formatting worker as ordinary default Value display.
        if let Some(scalar) = scalar {
            primitive::text::write_scalar(&mut output, scalar, false)
                .map_err(|_| output.error.unwrap_or(Error::Geometry))?;
        }
        let measured = output.finish()?;
        self.finish_generated(start, text_start, measured)?;
        self.pop()
    }
}

pub(super) fn control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let controls = [
        size_of::<Option<u16>>(),
        size_of::<[f64;3]>(),
        size_of::<&str>(),
        size_of::<std::num::ParseFloatError>(),
        size_of::<Result<f64,std::num::ParseFloatError>>(),
        size_of::<[Slot; 4]>(),
        size_of::<Result<Slot, Error>>(),
        size_of::<(&mut Engine<'_, '_, '_>, Slot)>(),
        size_of::<Option<Scalar>>(),
        size_of::<(usize, usize)>(),
        size_of::<Output<'_>>(),
        size_of::<Measured>(),
        size_of::<Result<(), Error>>(),
    ];
    controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)?
        .checked_add(super::text_segments::control_bytes()?)?
        .checked_add(primitive::text::scalar_control_bytes::<Output<'_>>()?)
}
