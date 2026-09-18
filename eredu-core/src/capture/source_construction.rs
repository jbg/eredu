//! Prospective destinations for the original descriptive capture-source worker.
use super::{admission::allocation::Allocation, *};
use crate::{DescriptionCompleteness, HostMetadataFunding};
use std::{fmt, mem::size_of};

/// Borrowed construction policy for a capture catalog and its support report.
///
/// This issues no capture, native or execution authority. Every funded caller
/// must retain the supplied account in its enclosing output and escaping error;
/// these intermediate destinations must retire before that enclosure. Ordinary
/// discovery explicitly selects `None` and executes the same field workers.
#[derive(Clone, Copy)]
pub struct CaptureSourceConstruction<'a>(Allocation<'a>);
impl<'a> CaptureSourceConstruction<'a> {
    /// Select the actual source account, or ordinary unaccounted construction.
    pub fn new(funding: Option<&'a HostMetadataFunding>) -> Self {
        Self(Allocation(funding))
    }
    /// Borrow the selected account for nested original source workers.
    pub fn funding(self) -> Option<&'a HostMetadataFunding> {
        self.0.0
    }
    /// Admit the caller's concrete fixed constructor/error transports.
    pub fn controls(self, bytes: usize) -> Result<(), CaptureError> {
        self.0.reserve(bytes)
    }
    /// Construct exact backing for a declaration's known destination population.
    pub fn vector<T>(self, count: usize) -> Result<Vec<T>, CaptureError> {
        self.0.vector(count)
    }
    /// Copy source text only after admitting its exact destination.
    pub fn text(self, text: &str) -> Result<String, CaptureError> {
        self.0.text(text)
    }
    /// Format a closed source diagnostic or identity through the counted writer.
    pub fn format(self, value: fmt::Arguments<'_>) -> Result<String, CaptureError> {
        self.0.format(value)
    }
    /// Copy a sequence of source conditions without an intermediate clone.
    pub fn texts<S: AsRef<str>>(self, source: &[S]) -> Result<Vec<String>, CaptureError> {
        self.controls(size_of::<(
            &Self,
            &[S],
            std::slice::Iter<'_, S>,
            Option<&S>,
            Result<String, CaptureError>,
        )>())?;
        let mut result = self.vector(source.len())?;
        for value in source {
            result.push(self.text(value.as_ref())?);
        }
        Ok(result)
    }
    /// Preserve every catalog field through the original point-copy worker.
    pub fn catalog(self, source: &ObservationCatalog) -> Result<ObservationCatalog, CaptureError> {
        self.controls(size_of::<(
            &ObservationCatalog,
            ObservationCatalog,
            DescriptionCompleteness,
            std::slice::Iter<'_, ObservationPoint>,
            Option<&ObservationPoint>,
            Result<ObservationPoint, CaptureError>,
        )>())?;
        let mut points = self.vector(source.points.len())?;
        for point in &source.points {
            points.push(self.0.point(point)?);
        }
        let completeness = match &source.completeness {
            DescriptionCompleteness::Complete => DescriptionCompleteness::Complete,
            DescriptionCompleteness::Partial(reasons) => {
                DescriptionCompleteness::Partial(self.texts(reasons)?)
            }
            DescriptionCompleteness::Unsupported(reasons) => {
                DescriptionCompleteness::Unsupported(self.texts(reasons)?)
            }
        };
        Ok(ObservationCatalog {
            schema_version: source.schema_version,
            points,
            completeness,
        })
    }
    /// Copy the actual native capability declaration, including its conditions.
    pub fn capabilities(
        self,
        source: &CaptureCapabilities,
    ) -> Result<CaptureCapabilities, CaptureError> {
        self.controls(size_of::<(
            &CaptureCapabilities,
            CaptureCapabilities,
            std::slice::Iter<'_, CaptureTransformKind>,
            usize,
        )>())?;
        let mut transformations = self.vector(source.transformations.len())?;
        transformations.extend_from_slice(&source.transformations);
        Ok(CaptureCapabilities {
            transformations,
            max_histogram_bins: source.max_histogram_bins,
            physical_native_limit: source.physical_native_limit,
            conditions: self.texts(&source.conditions)?,
        })
    }
    /// Copy an existing fixed support decision and its exact diagnostic text.
    pub fn status(
        self,
        source: &ObservationSupportStatus,
    ) -> Result<ObservationSupportStatus, CaptureError> {
        use ObservationSupportStatus as S;
        self.controls(size_of::<(&S, Result<S, CaptureError>)>())?;
        Ok(match source {
            S::Supported => S::Supported,
            S::Conditional(s) => S::Conditional(self.text(s)?),
            S::Unsupported(s) => S::Unsupported(self.text(s)?),
            S::Unverified(s) => S::Unverified(self.text(s)?),
        })
    }
    /// Preserve an existing report without semantic readmission.
    pub fn support(
        self,
        source: &ObservationSupportReport,
    ) -> Result<ObservationSupportReport, CaptureError> {
        self.controls(size_of::<(
            &ObservationSupportReport,
            ObservationSupportReport,
            crate::ObservationSupport,
            std::slice::Iter<'_, crate::ObservationSupport>,
            Option<&crate::ObservationSupport>,
        )>())?;
        let mut points = self.vector(source.points.len())?;
        for point in &source.points {
            points.push(crate::ObservationSupport {
                path: self.text(&point.path)?,
                prefill: self.status(&point.prefill)?,
                decode: self.status(&point.decode)?,
                floating_to_f32: point.floating_to_f32,
            });
        }
        Ok(ObservationSupportReport {
            schema_version: source.schema_version,
            points,
            capture: self.capabilities(&source.capture)?,
        })
    }
}

#[cfg(test)]
mod tests;
