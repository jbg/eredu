use super::PrettyPrinter;
use std::{
    collections::TryReserveError,
    fmt,
    mem::{size_of, size_of_val},
};

pub(crate) struct PrinterCopyPlan<'a> {
    source: &'a PrettyPrinter,
    bytes: usize,
}
#[derive(Debug)]
pub(crate) struct PrinterCopyFailure {
    cause: Option<TryReserveError>,
    mapping: Vec<u8>,
}
impl fmt::Display for PrinterCopyFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("printer mapping copy destination is unavailable")
    }
}
impl std::error::Error for PrinterCopyFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.cause.as_ref().map(|c| c as _)
    }
}
impl PrettyPrinter {
    pub(crate) fn source_copy_plan(&self) -> Option<PrinterCopyPlan<'_>> {
        if self.alphabet_mapping.len() > isize::MAX as usize {
            return None;
        }
        Some(PrinterCopyPlan {
            source: self,
            bytes: self.alphabet_mapping.len(),
        })
    }
}
impl PrinterCopyPlan<'_> {
    pub(crate) fn buffer_bytes(&self) -> usize {
        self.bytes
    }
    pub(crate) fn control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<PrettyPrinter>(),
            size_of::<PrinterCopyFailure>(),
            size_of::<Vec<u8>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Result<PrettyPrinter, PrinterCopyFailure>>(),
            size_of::<Option<Self>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    pub(crate) fn compile(self) -> Result<PrettyPrinter, PrinterCopyFailure> {
        let mut mapping = Vec::new();
        if let Err(cause) = mapping.try_reserve_exact(self.bytes) {
            return Err(PrinterCopyFailure {
                cause: Some(cause),
                mapping,
            });
        }
        if mapping.capacity() != self.bytes {
            return Err(PrinterCopyFailure {
                cause: None,
                mapping,
            });
        }
        mapping.extend_from_slice(&self.source.alphabet_mapping);
        Ok(PrettyPrinter {
            alphabet_mapping: mapping,
            alphabet_size: self.source.alphabet_size,
            has_mapping: self.source.has_mapping,
        })
    }
}
