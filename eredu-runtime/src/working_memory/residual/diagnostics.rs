//! One paid diagnostic report shared by aliases of the same numerical quote.
use crate::working_memory::{WorkspaceReportError, WorkspaceReportMetadata};
use eredu_core::{HostMetadataFunding, RuntimeStateEstimate};
use eredu_nn::{Error, workspace::WorkspaceMetadataError};
use std::{
    alloc::Layout,
    ops::Deref,
    sync::{Arc, atomic::AtomicUsize},
};

#[derive(Debug)]
pub(super) struct QuoteDiagnostics(Option<Arc<Report>>);

#[derive(Debug)]
struct Report {
    state: RuntimeStateEstimate,
    // The strings and windows, followed by the Arc shell, must retire before
    // the account that admitted them. No raw Arc or Weak leaves this module.
    funding: Option<HostMetadataFunding>,
}

impl QuoteDiagnostics {
    pub(super) fn new(
        state: RuntimeStateEstimate,
        metadata: WorkspaceReportMetadata<'_>,
    ) -> Result<Self, WorkspaceReportError> {
        metadata.admit::<(
            Self,
            Report,
            Option<Report>,
            Arc<Report>,
            Layout,
            usize,
            Option<HostMetadataFunding>,
        )>()?;
        let shell = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<Report>())
            .map_err(|_| Error::from(WorkspaceMetadataError::Overflow))?
            .0
            .pad_to_align()
            .size();
        metadata.charge(shell)?;
        Ok(Self(Some(Arc::new(Report {
            state,
            funding: metadata.funding(),
        }))))
    }

    // Sealing adds retention diagnostics to this candidate only. Unique cold
    // candidates reuse their report; aliases require an actual paid copy.
    pub(super) fn make_mut(&mut self) -> Result<&mut RuntimeStateEstimate, WorkspaceReportError> {
        let owner = self.0.as_ref().expect("live quote diagnostics");
        let metadata = owner
            .funding
            .as_ref()
            .map(WorkspaceReportMetadata::with_funding)
            .unwrap_or_else(WorkspaceReportMetadata::ordinary);
        metadata.admit::<(
            &mut Self,
            &Arc<Report>,
            Option<HostMetadataFunding>,
            Result<&mut RuntimeStateEstimate, super::ResidualQuoteError>,
        )>()?;
        if Arc::strong_count(owner) != 1 {
            let copy = Self::new(metadata.clone_state(&owner.state)?, metadata)?;
            *self = copy;
        }
        Ok(
            &mut Arc::get_mut(self.0.as_mut().expect("live quote diagnostics"))
                .expect("unique quote diagnostics without weak owners")
                .state,
        )
    }
}
impl Clone for QuoteDiagnostics {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl Deref for QuoteDiagnostics {
    type Target = RuntimeStateEstimate;
    fn deref(&self) -> &Self::Target {
        &self.0.as_ref().expect("live quote diagnostics").state
    }
}
impl Drop for QuoteDiagnostics {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            // into_inner releases the allocation before Report drops funding.
            drop(Arc::into_inner(owner));
        }
    }
}

impl super::IncrementalInferenceQuote {
    pub(super) fn state_mut(
        &mut self,
    ) -> Result<&mut RuntimeStateEstimate, super::ResidualQuoteError> {
        let funding = self
            .state
            .0
            .as_ref()
            .expect("live quote diagnostics")
            .funding
            .clone();
        self.state
            .make_mut()
            .map_err(|error| match funding.as_ref() {
                Some(funding) => super::ResidualQuoteError::Storage(
                    crate::working_memory::reservation_metadata::neural_error(
                        WorkspaceReportMetadata::with_funding(funding).error(error),
                        funding,
                    ),
                ),
                None => super::ResidualQuoteError::from(error.into_capability()),
            })
    }
}
