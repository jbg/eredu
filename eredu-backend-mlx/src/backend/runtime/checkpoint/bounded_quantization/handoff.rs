//! Completed conversion with its actual source and retained transform plan.
use super::*;
use eredu_checkpoint::store::RetainedCheckpointSource;
use eredu_runtime::working_memory::SharedNativeInitializationCustody;

/// A move-only handoff of one completed overlay. Borrowers share its actual
/// source identity; validation does not construct or convert another store.
/// This retains existing custody and grants no new allocation authority.
#[derive(Debug)]
pub(crate) struct ConvertedQuantization {
    source: RetainedCheckpointSource,
    plan: BoundedQuantizationPlan,
    store: RetainedCheckpointSource,
    report: WeightMaterializationReport,
    // The retained plan and the source root can retire independently.
    _metadata: Option<Arc<SharedNativeInitializationCustody>>,
}

impl ConvertedQuantization {
    pub(super) fn new(
        source: RetainedCheckpointSource,
        plan: BoundedQuantizationPlan,
        store: QuantizedCheckpoint,
        metadata: Option<Arc<SharedNativeInitializationCustody>>,
    ) -> Self {
        let (store, report) = store.into_parts();
        let store = match &metadata {
            Some(custody) => RetainedCheckpointSource::from_materialized_with_custody(
                store, Arc::clone(custody),
            ),
            None => RetainedCheckpointSource::from_materialized(store),
        };
        Self {
            source,
            plan,
            store,
            report,
            _metadata: metadata,
        }
    }

    /// The exact completed overlay, for residency preparation before adoption.
    pub(crate) fn store(&self) -> &RetainedCheckpointSource {
        &self.store
    }

    pub(super) fn validate(
        &self,
        source: &RetainedCheckpointSource,
        plan: &BoundedQuantizationPlan,
    ) -> Result<(), Error> {
        if !self.source.same_source(source) {
            return Err(Error::Quantization(
                "prepared quantization belongs to a different checkpoint source".into(),
            ));
        }
        if &self.plan != plan {
            return Err(Error::Quantization(
                "prepared quantization does not match the selected transform plan".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn into_parts(self) -> (RetainedCheckpointSource, WeightMaterializationReport) {
        (self.store, self.report)
    }
}
