//! Immutable wall-clock inputs for coherent chat render variants.
use chrono::{DateTime, FixedOffset, Local, NaiveDateTime};

/// A single local calendar observation. Explicit naive fixtures have no UTC
/// offset; formats requiring that missing information return an error.
#[derive(Debug, Clone, Copy)]
pub struct ChatClockSnapshot {
    local: NaiveDateTime,
    offset: Option<FixedOffset>,
}
impl ChatClockSnapshot {
    /// Uses caller-provided local calendar fields without assuming a time zone.
    pub fn from_local(local: NaiveDateTime) -> Self {
        Self {
            local,
            offset: None,
        }
    }
    pub(crate) fn now() -> Self {
        let now = Local::now();
        Self {
            local: now.naive_local(),
            offset: Some(*now.offset()),
        }
    }
    pub(crate) fn format(self, format: &str) -> Result<String, minijinja::Error> {
        use std::fmt::Write;
        let error = || {
            minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                "invalid or unavailable strftime_now format",
            )
        };
        let items = chrono::format::StrftimeItems::new(format);
        if items
            .clone()
            .any(|item| item == chrono::format::Item::Error)
        {
            return Err(error());
        }
        if self.offset.is_none()
            && items.clone().any(|item| {
                matches!(
                    item,
                    chrono::format::Item::Numeric(chrono::format::Numeric::Timestamp, _)
                )
            })
        {
            return Err(error());
        }
        let mut output = String::new();
        match self.offset {
            Some(offset) => {
                let date =
                    DateTime::<FixedOffset>::from_naive_utc_and_offset(self.local - offset, offset);
                write!(output, "{}", date.format_with_items(items))
            }
            None => write!(output, "{}", self.local.format_with_items(items)),
        }
        .map_err(|_| error())?;
        Ok(output)
    }
}
