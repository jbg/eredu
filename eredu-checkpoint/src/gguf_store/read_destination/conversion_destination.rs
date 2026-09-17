//! Conversion vectors bound by the genuine consumed GGUF lease.
use super::*;
use eredu_gguf::{ConversionDestinationError, ConversionLayouts, PreparedConversion};

/// Owns final conversion vectors before raw storage and its genuine source lease.
/// Reader/cache state and result metadata still use their existing ordinary owners.
#[derive(Debug)]
pub struct PreparedGgufConversion<S: GgufRawStorage = Vec<u8>> {
    conversion: PreparedConversion,
    read: PreparedGgufRead<S>,
}

/// Retains partial conversion outputs, raw bytes and source after failure.
#[derive(Debug)]
pub struct PreparedGgufConversionFailure<S: GgufRawStorage = Vec<u8>> {
    cause: Cause,
    conversion: Option<PreparedConversion>,
    read: PreparedGgufRead<S>,
}
impl<S: GgufRawStorage> PreparedGgufConversionFailure<S> {
    /// Original checkpoint cause, if processing failed.
    pub fn store_error(&self) -> Option<&StoreError> {
        match &self.cause {
            Cause::Store(e) => Some(e),
            _ => None,
        }
    }
    /// Typed conversion/storage cause, including original conversion errors during preparation.
    pub fn conversion_error(&self) -> Option<&ConversionDestinationError> {
        match &self.cause {
            Cause::Conversion(e) => Some(e),
            _ => None,
        }
    }
    /// Successful preparation prefix or partial conversion destination.
    pub fn conversion(&self) -> Option<&PreparedConversion> {
        self.conversion.as_ref()
    }
    /// Initialized raw storage; failed contents do not report bytes successfully read.
    pub fn raw_bytes(&self) -> &[u8] {
        self.read.raw.as_ref()
    }
    /// Exact source lease retained through conversion and raw-buffer retirement.
    pub fn lease(&self) -> &GgufLease {
        &self.read.lease
    }
}
impl<S: GgufRawStorage> std::fmt::Display for PreparedGgufConversionFailure<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.cause {
            Cause::Store(e) => e.fmt(f),
            Cause::Conversion(e) => e.fmt(f),
            Cause::Metadata(e) => e.fmt(f),
            Cause::Reserve(e) => e.fmt(f),
            Cause::Layout => f.write_str("GGUF raw destination layout is not representable"),
            Cause::Length { expected, actual } => write!(
                f,
                "GGUF raw destination has {actual} bytes; expected {expected}"
            ),
        }
    }
}
impl<S: GgufRawStorage> std::error::Error for PreparedGgufConversionFailure<S> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Store(e) => Some(e),
            Cause::Conversion(e) => Some(e),
            Cause::Metadata(e) => Some(e),
            Cause::Reserve(e) => Some(e),
            _ => None,
        }
    }
}
impl<S: GgufRawStorage> PreparedGgufRead<S> {
    /// Promotes this exact raw/source owner with final vectors for its actual physical selection.
    /// Descriptor and selection-plan clones are cold owned storage, not scalar fit facts.
    pub fn prepare_conversion(
        self,
    ) -> Result<PreparedGgufConversion<S>, PreparedGgufConversionFailure<S>> {
        let selected = crate::gguf_store::conversion_plan::selected_descriptor(
            &self.lease.entry,
            self.lease.identity.selection.as_ref(),
        );
        let (descriptor, endian) = match selected {
            Ok(pair) => pair,
            Err(error) => {
                return Err(PreparedGgufConversionFailure {
                    cause: Cause::Store(error),
                    conversion: None,
                    read: self,
                });
            }
        };
        match PreparedConversion::prepare(descriptor, endian) {
            Ok(conversion) => Ok(PreparedGgufConversion {
                conversion,
                read: self,
            }),
            Err(failure) => {
                let (cause, conversion) = failure.into_parts();
                Err(PreparedGgufConversionFailure {
                    cause: Cause::Conversion(cause),
                    conversion: Some(conversion),
                    read: self,
                })
            }
        }
    }
}
impl<S: GgufRawStorage> PreparedGgufConversion<S> {
    /// Actual requested layouts of conversion vectors, binding and owner.
    pub fn conversion_layouts(&self) -> Option<ConversionLayouts> {
        self.conversion.layouts()
    }
    /// Actual vector capacities, distinct from requested layouts.
    pub fn conversion_capacities(&self) -> [usize; 7] {
        self.conversion.capacities()
    }
    /// Raw destination requested layout.
    pub fn raw_layout(&self) -> Layout {
        self.read.raw_layout()
    }
    /// Actual raw allocation capacity.
    pub fn raw_capacity(&self) -> usize {
        self.read.raw_capacity()
    }
    /// Source lease that fixes physical selection, encoding and byte order.
    pub fn lease(&self) -> &GgufLease {
        self.read.lease()
    }
    /// Runs the original cache/read/conversion driver with both prepared destinations.
    /// The intact outer owner keeps conversion buffers before raw/source on unwind.
    pub fn materialize(
        mut self,
    ) -> Result<ConvertedCheckpointTensor, PreparedGgufConversionFailure<S>> {
        match materialize_with_conversion(
            &self.read.lease,
            Some(self.read.raw.as_mut()),
            Some(&mut self.conversion),
        ) {
            Ok(output) => Ok(output),
            Err(cause) => Err(PreparedGgufConversionFailure {
                cause,
                conversion: Some(self.conversion),
                read: self.read,
            }),
        }
    }
}

mod metadata_destination;
pub use metadata_destination::{PreparedGgufTensor, PreparedGgufTensorFailure};
