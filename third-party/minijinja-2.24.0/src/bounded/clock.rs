//! Immutable environmental facts held across measurement and final rendering.
//! The caller owns clock acquisition. This worker neither reads a clock nor
//! initializes timezone state, and writes directly to the caller's destination.
use chrono::{
    format::{DelayedFormat, Fixed, Item, Numeric, StrftimeItems},
    NaiveDateTime,
};
use std::fmt::{self, Write};

/// One local timestamp, shared by every attempt and both prompt variants.
#[derive(Clone, Copy, Debug)]
pub struct Snapshot {
    local: NaiveDateTime,
}
impl Snapshot {
    /// Capture already-observed local calendar facts without heap storage.
    pub fn from_local(local: NaiveDateTime) -> Self {
        Self { local }
    }

    /// Local calendar/time formats use Chrono's original parsing and numerical
    /// writer. Offset-bearing formats require offset facts and are not accepted
    /// by this local-only source.
    pub fn supports(format: &str) -> bool {
        StrftimeItems::new(format).all(|item| {
            !matches!(
                item,
                Item::Error
                    | Item::Numeric(Numeric::Timestamp, _)
                    | Item::Fixed(
                        Fixed::TimezoneName
                            | Fixed::TimezoneOffsetColon
                            | Fixed::TimezoneOffsetDoubleColon
                            | Fixed::TimezoneOffsetTripleColon
                            | Fixed::TimezoneOffsetColonZ
                            | Fixed::TimezoneOffset
                            | Fixed::TimezoneOffsetZ
                            | Fixed::RFC2822
                            | Fixed::RFC3339
                            | Fixed::Internal(_)
                    )
            )
        })
    }

    /// No intermediate String or owned format-item list.
    pub fn write(self, output: &mut (impl Write + ?Sized), format: &str) -> fmt::Result {
        if !Self::supports(format) {
            return Err(fmt::Error);
        }
        self.local
            .format_with_items(StrftimeItems::new(format))
            .write_to(output)
    }

    pub(super) fn control_bytes<W>() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let frames = [
            size_of::<Self>(),
            size_of::<NaiveDateTime>(),
            size_of::<StrftimeItems<'_>>(),
            size_of::<DelayedFormat<StrftimeItems<'_>>>(),
            size_of::<Option<Item<'_>>>(),
            size_of::<Item<'_>>(),
            size_of::<W>(),
            size_of::<fmt::Arguments<'_>>(),
            size_of::<(&str, usize, i64, u32)>(),
            size_of::<fmt::Result>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
}
