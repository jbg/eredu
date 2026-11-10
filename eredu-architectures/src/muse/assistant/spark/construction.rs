//! Exact physical encodings and target-bound construction for both Spark assistants.

use std::collections::HashMap;

use eredu_checkpoint::WeightQuantization;

use super::{SparkAssistantConfig, SparkAssistantError};
use crate::muse::assistant::{
    body::DFlashBodyDefinition, candidate_selector::CandidateSelectorDefinition,
    dspark::DsparkHeadDefinition,
};

/// Admitted assistant equations and exact per-weight physical encodings.
/// Unlisted weights retain the released BF16 storage contract.
#[derive(Debug, Clone)]
pub struct SparkAssistantLoadConfig {
    released: SparkAssistantConfig,
    formats: HashMap<String, WeightQuantization>,
}

impl SparkAssistantLoadConfig {
    /// Retains the released dense source contract.
    pub fn new(released: SparkAssistantConfig) -> Self {
        Self {
            released,
            formats: HashMap::new(),
        }
    }

    /// Borrows the artifact's equations and identity independently of physical encoding.
    pub const fn released(&self) -> &SparkAssistantConfig {
        &self.released
    }

    pub(in crate::muse::assistant) fn retained_formats(
        &self,
    ) -> &HashMap<String, WeightQuantization> {
        &self.formats
    }
    /// Receives a format map already resolved against the retained parameter declaration.
    pub(in crate::muse::assistant) fn select_retained_formats(
        &self,
        formats: HashMap<String, WeightQuantization>,
    ) -> Self {
        Self {
            released: self.released.clone(),
            formats,
        }
    }
    pub(in crate::muse::assistant) fn dflash2_scopes(
        &self,
    ) -> Result<(DFlashBodyDefinition, CandidateSelectorDefinition), SparkAssistantError> {
        self.released.dflash2_scopes(&self.formats)
    }

    pub(in crate::muse::assistant) fn dspark_scopes(
        &self,
    ) -> Result<
        (
            crate::muse::assistant::body::DsparkBodyParameters,
            DsparkHeadDefinition,
        ),
        SparkAssistantError,
    > {
        self.released.dspark_scopes(&self.formats)
    }
}
