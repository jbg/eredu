//! Borrowed validation from the actual retained selected declaration owner.
use super::*;
impl PreparedModelDiscovery {
    /// Revalidate an already admitted static plan without cloning discovery DTOs
    /// or reading artifact payloads. A caller must first have resolved the source
    /// through the normal discovery/admission path; this never hashes on demand.
    pub fn validate_static_intervention(
        &self,
        plan: &eredu_core::intervention::AdmittedInterventionPlan,
        facts: eredu_core::intervention::InterventionMechanismFacts<'_>,
        session: &str,
    ) -> Result<(), eredu_core::intervention::InterventionSourceError> {
        use eredu_core::intervention::InterventionSourceError as E;
        // Partition support has a separate loaded placement projection. The
        // existing single-source static path cannot impersonate that projection.
        if self.partition_selection.is_some() || !self.identity.is_resolved() {
            return Err(E::Identity);
        }
        let identity = self.identity.resolve().map_err(|_| E::Identity)?;
        plan.validate_source_identity(identity, session)?;
        eredu_runtime::inspection::validate_static_intervention_declarations(
            plan,
            &self.intervention_points,
            &self.support.points,
            facts,
        )
    }
    /// Validate a loaded partition's actual ordinary support projection. The
    /// caller retains the source that produced both layouts and discovery;
    /// this method neither rebuilds that projection nor grants an edit callback.
    pub fn validate_partitioned_static_intervention(
        &self,plan:&eredu_core::intervention::AdmittedInterventionPlan,
        facts:eredu_core::intervention::InterventionMechanismFacts<'_>,session:&str,
        layouts:&crate::component_partition::ComponentPartitionLayouts,
        discovery:&eredu_core::capture::CaptureDiscovery,execution:&str,
    )->Result<(),eredu_core::intervention::InterventionSourceError> {
        use eredu_core::intervention::InterventionSourceError as E;
        if !self.identity.is_resolved() || execution!=self.execution_identity()
            || self.partition_selection.as_ref().and_then(|source|source.parallel_topology())
                .map(|rank|rank.topology())!=Some(layouts.topology())
            || discovery.artifact_identity!=plan.artifact_identity() {
            return Err(E::Identity);
        }
        let identity=self.identity.resolve().map_err(|_|E::Identity)?;
        plan.validate_source_identity(identity,session)?;
        // Ordinary discovery first filters these same mechanism fields, then
        // replaces phase support with this exact loaded partition report.
        eredu_runtime::inspection::validate_static_intervention_declarations(
            plan,&self.intervention_points,&discovery.support.points,facts,
        )
    }
    /// Actual borrowed partition/source validation frames in addition to the
    /// shared field comparison. No discovery or coordinate map is copied.
    pub fn partitioned_intervention_validation_control_bytes()->Option<usize> {
        use std::mem::{size_of,size_of_val};
        let frames=[Self::static_intervention_validation_control_bytes()?,
            size_of::<(&Self,&eredu_core::intervention::AdmittedInterventionPlan,
                eredu_core::intervention::InterventionMechanismFacts<'_>,&str,
                &crate::component_partition::ComponentPartitionLayouts,
                &eredu_core::capture::CaptureDiscovery,&str)>(),
            size_of::<Option<eredu_core::ParallelRankTopology>>(),
            size_of::<Result<(),eredu_core::intervention::InterventionSourceError>>()];
        frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add)
    }
    /// The fixed identity/projection controls required before borrowed validation.
    pub fn static_intervention_validation_control_bytes() -> Option<usize> {
        use std::mem::size_of;
        eredu_core::intervention::AdmittedInterventionPlan::discovery_validation_control_bytes()?
            .checked_add(
                eredu_runtime::inspection::static_intervention_validation_control_bytes()?,
            )?
            .checked_add(size_of::<
                Result<eredu_core::artifact::ArtifactIdentity, std::sync::Arc<eredu_core::artifact::ArtifactError>>,
            >())?
            .checked_add(size_of::<(
                &Self,
                &eredu_core::intervention::AdmittedInterventionPlan,
                &str,
            )>())
    }
}
