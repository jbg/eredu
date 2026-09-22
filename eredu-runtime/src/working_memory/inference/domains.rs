//! Physical requirements from the ordinary completed-span traversal.
use super::*;
use eredu_core::DomainMemoryRequirements;

/// Per-domain peaks of safely separated spans. Each span already resolves
/// actual backing placement and overlapping lifetimes before this comparison.
#[derive(Clone, Debug)]
pub struct InferenceDomainWorkspaceReport {
    pub(super) transient: DomainMemoryRequirements,
    pub(super) retained: DomainMemoryRequirements,
    pub(super) residual: Option<DomainMemoryRequirements>,
}
impl InferenceDomainWorkspaceReport {
    /// New transients, displaced state, scratch and operation-owned host work.
    pub fn transient(&self) -> &DomainMemoryRequirements {
        &self.transient
    }
    /// Complete retained state backing peak.
    pub fn retained(&self) -> &DomainMemoryRequirements {
        &self.retained
    }
    /// Exact original source-exclusion result, preserving the borrowed identities.
    pub fn residual(&self) -> Option<&DomainMemoryRequirements> {
        self.residual.as_ref()
    }
}
impl InferenceWorkspaceReport {
    /// Marks the architecture's complete media traversal: original source roots,
    /// retained embeddings, encoder intermediates and decoder state are included
    /// in the same identity-aware state/transient walk. This grants no authority.
    pub fn with_traced_media(mut self) -> Self {
        self.traced_media = true;
        self
    }
    /// Physical attribution is complete only when every actual span provides it.
    pub fn physical_domains(&self) -> Option<&InferenceDomainWorkspaceReport> {
        self.physical_domains_complete
            .then_some(self.physical_domains.as_ref())
            .flatten()
    }
    pub(super) fn refine_domain_state(
        &self,
        state: &mut RuntimeStateEstimate,
        metadata: super::super::WorkspaceReportMetadata<'_>,
    ) -> Result<(), super::super::WorkspaceReportError> {
        let Some(domains) = self.physical_domains() else {
            return Ok(());
        };
        if let Some(existing) = &mut state.physical_domains {
            if existing.geometry != self.geometry {
                return Err(eredu_core::AdmissionPolicyError::InvalidConfiguration {
                    field: "physical_state",
                    detail: "physical state and equation geometry differ",
                }
                .into());
            }
            existing.decoder_state = metadata.clone_domain_requirements(&domains.retained)?;
            if self.traced_media {
                existing.media_embeddings =
                    metadata.empty_domain_requirements(&domains.retained)?;
                existing.media_workspace = metadata.empty_domain_requirements(&domains.retained)?;
            }
        } else if self.traced_media
            || (state.multimodal_embedding_bytes == 0 && state.media_execution_workspace_bytes == 0)
        {
            // Actual traced media occupies the same canonical retained/transient
            // union; adding the logical media size again would duplicate it.
            // Otherwise architectural absence establishes the two zero terms.
            state.physical_domains = Some(eredu_core::DomainRuntimeStateEstimate {
                geometry: self.geometry,
                decoder_state: metadata.clone_domain_requirements(&domains.retained)?,
                media_embeddings: metadata.empty_domain_requirements(&domains.retained)?,
                media_workspace: metadata.empty_domain_requirements(&domains.retained)?,
            });
        }
        Ok(())
    }
}
impl<F, E, R> Inspection<'_, F>
where
    F: FnMut(&InferenceWorkspaceSpan) -> Result<R, E>,
    R: std::borrow::Borrow<WorkspaceTraceReport>,
{
    pub(super) fn record_domains(
        &mut self,
        trace: &WorkspaceTraceReport,
    ) -> Result<(), InferenceWorkspaceError<E>> {
        let Some(physical) = &trace.physical_domains else {
            self.report.physical_domains_complete = false;
            self.report.physical_domains = None;
            return Ok(());
        };
        let (Some(transient), Some(retained)) =
            (&physical.state_transient, &physical.retained_state)
        else {
            self.report.physical_domains_complete = false;
            self.report.physical_domains = None;
            return Ok(());
        };
        if !self.report.physical_domains_complete {
            return Ok(());
        }
        let metadata = self.metadata.report();
        let next = if let Some(previous) = &self.report.physical_domains {
            InferenceDomainWorkspaceReport {
                transient: metadata
                    .combine_domain_requirements(&previous.transient, transient, false)
                    .map_err(|e| metadata.error(e))?,
                retained: metadata
                    .combine_domain_requirements(&previous.retained, retained, false)
                    .map_err(|e| metadata.error(e))?,
                residual: previous
                    .residual
                    .as_ref()
                    .zip(physical.residual.as_ref())
                    .map(|(a, b)| metadata.combine_domain_requirements(a, b, false))
                    .transpose()
                    .map_err(|e| metadata.error(e))?,
            }
        } else {
            InferenceDomainWorkspaceReport {
                transient: metadata
                    .clone_domain_requirements(transient)
                    .map_err(|e| metadata.error(e))?,
                retained: metadata
                    .clone_domain_requirements(retained)
                    .map_err(|e| metadata.error(e))?,
                residual: physical
                    .residual
                    .as_ref()
                    .map(|v| metadata.clone_domain_requirements(v))
                    .transpose()
                    .map_err(|e| metadata.error(e))?,
            }
        };
        self.report.physical_domains = Some(next);
        Ok(())
    }
}
