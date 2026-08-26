use crate::{
    iquant_tables::{
        IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KSIGNS_IQ2XS,
        KVALUES_IQ4NL,
    },
    GgmlType,
};

#[derive(Debug, Clone, Copy)]
enum CodebookValues {
    I8(&'static [i8]),
    U32(&'static [u32]),
    U64(&'static [u64]),
}

/// Canonical nonlinear values used by a GGML IQ tensor encoding.
///
/// This is the backend-neutral contract for implementations that execute
/// packed IQ blocks directly. The typed accessors expose exactly one value
/// width for each supported encoding. [`Self::signs`] is present only when the
/// encoding uses the shared IQ sign table.
#[derive(Debug, Clone, Copy)]
pub struct IQuantCodebook {
    values: CodebookValues,
    signs: Option<&'static [u8]>,
}

impl IQuantCodebook {
    /// Returns the canonical codebook for an IQ encoding.
    ///
    /// Non-IQ and unsupported GGML encodings return `None`.
    pub const fn for_type(ty: GgmlType) -> Option<Self> {
        let (values, signs) = match ty {
            GgmlType::IQ2XXS => (
                CodebookValues::U64(&IQ2XXS_GRID),
                Some(KSIGNS_IQ2XS.as_slice()),
            ),
            GgmlType::IQ2XS => (
                CodebookValues::U64(&IQ2XS_GRID),
                Some(KSIGNS_IQ2XS.as_slice()),
            ),
            GgmlType::IQ2S => (CodebookValues::U64(&IQ2S_GRID), None),
            GgmlType::IQ3XXS => (
                CodebookValues::U32(&IQ3XXS_GRID),
                Some(KSIGNS_IQ2XS.as_slice()),
            ),
            GgmlType::IQ3S => (CodebookValues::U32(&IQ3S_GRID), None),
            GgmlType::IQ1S | GgmlType::IQ1M => (CodebookValues::U64(&IQ1S_GRID), None),
            GgmlType::IQ4NL | GgmlType::IQ4XS => (CodebookValues::I8(&KVALUES_IQ4NL), None),
            _ => return None,
        };
        Some(Self { values, signs })
    }

    /// Returns signed 8-bit codebook values when used by the encoding.
    pub const fn i8_values(self) -> Option<&'static [i8]> {
        match self.values {
            CodebookValues::I8(values) => Some(values),
            _ => None,
        }
    }

    /// Returns packed 32-bit codebook values when used by the encoding.
    pub const fn u32_values(self) -> Option<&'static [u32]> {
        match self.values {
            CodebookValues::U32(values) => Some(values),
            _ => None,
        }
    }

    /// Returns packed 64-bit codebook values when used by the encoding.
    pub const fn u64_values(self) -> Option<&'static [u64]> {
        match self.values {
            CodebookValues::U64(values) => Some(values),
            _ => None,
        }
    }

    /// Returns the shared sign table when used by the encoding.
    pub const fn signs(self) -> Option<&'static [u8]> {
        self.signs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codebooks_are_selected_by_encoding() {
        let iq2 = IQuantCodebook::for_type(GgmlType::IQ2XS).unwrap();
        assert_eq!(iq2.u64_values().unwrap().len(), 512);
        assert_eq!(iq2.signs().unwrap().len(), 128);
        assert!(iq2.i8_values().is_none());

        let iq4 = IQuantCodebook::for_type(GgmlType::IQ4NL).unwrap();
        assert_eq!(iq4.i8_values().unwrap().len(), 16);
        assert!(iq4.signs().is_none());

        assert!(IQuantCodebook::for_type(GgmlType::Q4K).is_none());
    }
}
