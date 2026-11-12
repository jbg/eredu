//! Cold selection of the two co-resident VL weight policies.

use std::num::NonZeroUsize;

use eredu_checkpoint::recipe::RecipeDtype;
use eredu_core::ParallelRankTopology;
use eredu_nn::Error;
use eredu_runtime::{
    execution_unit_parameter_window_bytes, physical_binding_bytes, ExecutionUnitAddressMap,
    LayerWeightResidency, WeightBinding,
};

use crate::lfm3::{
    LanguagePipelineCommunication, PreparedLanguagePartition, SelectedLanguagePartition,
};

use super::{
    PreparedVisionExecution, PreparedVisionLanguage, PreparedVlDecoder, VisionLanguageDeclaration,
};

/// Prepared vision and decoder resources before residency selection.
pub struct PreparedVlWeights {
    vision: PreparedVisionExecution,
    decoder: PreparedVlDecoder,
}

/// Independent policies for vision units and decoder parameters.
#[derive(Debug, Clone, Copy, Default)]
pub struct VlWeightPolicies {
    /// Vision tower and projector policy.
    pub vision: LayerWeightResidency,
    /// Language decoder policy.
    pub decoder: LayerWeightResidency,
}

/// Logical device parameter capacity for the selected VL policy.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct VlDeviceWeightBytes {
    /// Vision weight window.
    pub vision: u64,
    /// Decoder weight window and pinned parameters.
    pub decoder: u64,
    /// Sum of the two independent policies' device requirements.
    pub combined: u64,
}

/// Both prepared components paired with the exact policies selected for them.
pub struct SelectedVlWeights {
    prepared: PreparedVlWeights,
    policies: VlWeightPolicies,
    device_bytes: VlDeviceWeightBytes,
}

/// Cold VL rank geometry and communication, independent of checkpoint readers.
#[derive(Debug, Clone)]
pub struct SelectedVlRankPlacement {
    decoder: SelectedLanguagePartition,
    communication: LanguagePipelineCommunication,
}

impl SelectedVlRankPlacement {
    pub(super) fn select(
        language: std::sync::Arc<crate::lfm3::Declaration>,
        rank: ParallelRankTopology,
        prefix: NonZeroUsize,
    ) -> Result<Self, Error> {
        if rank.topology().world_size() == 1 {
            return Err(Error::backend(
                "VL rank placement requires tensor or pipeline parallelism",
            ));
        }
        let decoder = SelectedLanguagePartition::select(language, rank, Some(prefix))
            .map_err(Error::backend_source)?;
        let communication = decoder
            .pipeline_communication()
            .map_err(Error::backend_source)?;
        Ok(Self {
            decoder,
            communication,
        })
    }

    /// The exact rank selected for the decoder and its vision replica.
    pub fn rank(&self) -> ParallelRankTopology {
        self.decoder.rank()
    }

    /// Manifest supplied to native communication realization.
    pub fn communication_manifest(&self) -> &eredu_runtime::CommunicationManifest {
        &self.communication.manifest
    }

    pub(crate) fn with_completion_policy(
        mut self,
        policy: eredu_runtime::CommunicationCompletionPolicy,
    ) -> Self {
        self.communication.manifest = self.communication.manifest.with_completion_policy(policy);
        self
    }

    pub(crate) fn prefix(&self) -> NonZeroUsize {
        self.decoder.vocabulary_prefix().expect("VL prefix")
    }

    pub(crate) fn session_group(&self) -> Option<eredu_core::CollectiveGroupId> {
        self.communication
            .plan
            .publication()
            .map(|publication| publication.group)
    }

    pub(crate) fn state_partition(&self) -> Option<&eredu_runtime::PartitionState> {
        self.decoder.partition().state()
    }

    /// The decoder state geometry for this rank.
    pub fn decoder_state(&self) -> &eredu_runtime::StateLayout {
        self.decoder
            .partition()
            .local_geometry()
            .complete_state_layout()
    }
}

impl VisionLanguageDeclaration {
    /// Selects rank-local decoder ownership and communication without opening sources.
    pub fn select_rank_placement(
        &self,
        rank: ParallelRankTopology,
        prefix: NonZeroUsize,
    ) -> Result<SelectedVlRankPlacement, Error> {
        SelectedVlRankPlacement::select(self.language.clone(), rank, prefix)
    }
}

/// Rank-local resources before the vision and decoder residency policies are selected.
pub struct PreparedVlRankWeights {
    vision: PreparedVisionExecution,
    decoder: PreparedLanguagePartition,
    communication: LanguagePipelineCommunication,
}

/// Prepared rank-local resources with their admitted residency policies.
pub struct SelectedVlRankWeights {
    prepared: PreparedVlRankWeights,
    policies: VlWeightPolicies,
    device_bytes: VlDeviceWeightBytes,
}

impl PreparedVisionLanguage {
    /// Prepares both components from the retained source graph.
    pub fn prepare_weights(&self, dtype: RecipeDtype) -> Result<PreparedVlWeights, Error> {
        let (vision, decoder) = self.prepare_weight_components(dtype)?;
        Ok(PreparedVlWeights { vision, decoder })
    }
}

impl PreparedVlWeights {
    pub(super) fn from_component(
        artifact: &PreparedVisionLanguage,
        component: eredu_runtime::PreparedComponent<
            std::sync::Arc<eredu_runtime::ComponentDefinition>,
        >,
    ) -> Result<Self, Error> {
        let (vision, decoder) = artifact.project_weight_components(component)?;
        Ok(Self { vision, decoder })
    }

    /// Sizes logical device parameter windows without opening tensor payloads.
    pub fn device_weight_bytes(
        &self,
        policies: VlWeightPolicies,
    ) -> Result<VlDeviceWeightBytes, Error> {
        let decoder = self.decoder.parameters();
        let pinned = decoder
            .pinned_bindings()
            .values()
            .flatten()
            .cloned()
            .collect::<Vec<_>>();
        device_bytes(
            &self.vision,
            decoder.invocations(),
            decoder.unit_bindings(),
            &pinned,
            policies,
        )
    }

    /// Derives exact rank-local decoder parameters while retaining a full vision replica.
    pub fn select_rank(
        self,
        rank: ParallelRankTopology,
        pipeline_vocabulary_prefix: NonZeroUsize,
    ) -> Result<PreparedVlRankWeights, Error> {
        let selected = self
            .decoder
            .select_rank_placement(rank, pipeline_vocabulary_prefix)?;
        self.bind_rank(selected)
    }

    /// Binds the exact placement and manifest selected from artifact metadata.
    pub fn bind_rank(
        self,
        placement: SelectedVlRankPlacement,
    ) -> Result<PreparedVlRankWeights, Error> {
        let decoder = self.decoder.bind_rank_placement(placement.decoder)?;
        Ok(PreparedVlRankWeights {
            vision: self.vision,
            decoder,
            communication: placement.communication,
        })
    }

    /// Retains the selected policies and rejects insufficient device parameter budgets.
    pub fn select(self, policies: VlWeightPolicies) -> Result<SelectedVlWeights, Error> {
        let device_bytes = self.device_weight_bytes(policies)?;
        validate_budgets(policies, device_bytes)?;
        Ok(SelectedVlWeights {
            prepared: self,
            policies,
            device_bytes,
        })
    }
}

impl SelectedVlWeights {
    /// Logical device parameter capacity retained by selection.
    pub fn device_weight_bytes(&self) -> VlDeviceWeightBytes {
        self.device_bytes
    }

    /// Transfers both components with their admitted weight policies.
    pub fn into_parts(
        self,
    ) -> (
        (PreparedVisionExecution, LayerWeightResidency),
        (PreparedVlDecoder, LayerWeightResidency),
    ) {
        (
            (self.prepared.vision, self.policies.vision),
            (self.prepared.decoder, self.policies.decoder),
        )
    }
}

impl PreparedVlRankWeights {
    /// Sizes the local decoder and the full vision replica under the requested policies.
    pub fn device_weight_bytes(
        &self,
        policies: VlWeightPolicies,
    ) -> Result<VlDeviceWeightBytes, Error> {
        let (units, pinned) = self.decoder.bindings();
        device_bytes(
            &self.vision,
            self.decoder.partition().invocations(),
            units,
            pinned,
            policies,
        )
    }

    /// Partitions one load-wide weight budget across the decoder and vision replica.
    pub(super) fn select_shared(
        self,
        policy: LayerWeightResidency,
    ) -> Result<SelectedVlRankWeights, Error> {
        if policy.is_fully_resident() {
            return self.select(VlWeightPolicies::default());
        }
        let window = self.device_weight_bytes(VlWeightPolicies {
            vision: policy,
            decoder: policy,
        })?;
        let budgets = policy.offload().map_err(Error::backend_source)?;
        if let Some(budget) = budgets.device_budget_bytes() {
            if window.combined > budget {
                return Err(Error::backend(format!(
                    "LFM3 VL rank requires {} device parameter bytes with a budget of {budget}",
                    window.combined,
                )));
            }
        }
        let (vision_host, decoder_host) = match budgets.host_budget_bytes() {
            None => (None, None),
            Some(0) => (Some(0), Some(0)),
            Some(budget) => {
                let full = self.device_weight_bytes(VlWeightPolicies::default())?;
                let vision = ((u128::from(budget) * u128::from(full.vision))
                    / u128::from(full.combined)) as u64;
                (Some(vision), Some(budget - vision))
            }
        };
        self.select(VlWeightPolicies {
            vision: policy
                .with_component_budgets(window.vision, vision_host)
                .map_err(Error::backend_source)?,
            decoder: policy
                .with_component_budgets(window.decoder, decoder_host)
                .map_err(Error::backend_source)?,
        })
    }

    /// Admits rank-local residency budgets and retains the exact policies for binding.
    pub fn select(self, policies: VlWeightPolicies) -> Result<SelectedVlRankWeights, Error> {
        let device_bytes = self.device_weight_bytes(policies)?;
        validate_budgets(policies, device_bytes)?;
        Ok(SelectedVlRankWeights {
            prepared: self,
            policies,
            device_bytes,
        })
    }
}

impl SelectedVlRankWeights {
    /// Device parameter capacity for this rank, including the replicated vision tower.
    pub fn device_weight_bytes(&self) -> VlDeviceWeightBytes {
        self.device_bytes
    }

    /// Retained decoder weight residency for the selected rank.
    pub fn decoder_residency(&self) -> LayerWeightResidency {
        self.policies.decoder
    }

    /// Communication manifest for the selected rank.
    pub fn communication_manifest(&self) -> &eredu_runtime::CommunicationManifest {
        &self.prepared.communication.manifest
    }

    /// Transfers the retained placement, policies and communication to a native binder.
    pub fn into_parts(
        self,
    ) -> (
        (PreparedVisionExecution, LayerWeightResidency),
        (PreparedLanguagePartition, LayerWeightResidency),
        LanguagePipelineCommunication,
    ) {
        (
            (self.prepared.vision, self.policies.vision),
            (self.prepared.decoder, self.policies.decoder),
            self.prepared.communication,
        )
    }
}

fn device_bytes(
    vision: &PreparedVisionExecution,
    decoder_invocations: &ExecutionUnitAddressMap,
    decoder_bindings: &[Vec<WeightBinding>],
    pinned: &[WeightBinding],
    policies: VlWeightPolicies,
) -> Result<VlDeviceWeightBytes, Error> {
    let vision = window(
        vision.invocations(),
        vision.unit_bindings(),
        policies.vision,
    )?;
    let decoder = add(
        window(decoder_invocations, decoder_bindings, policies.decoder)?,
        physical_binding_bytes(pinned).map_err(Error::backend_source)?,
    )?;
    Ok(VlDeviceWeightBytes {
        vision,
        decoder,
        combined: add(vision, decoder)?,
    })
}

fn validate_budgets(policies: VlWeightPolicies, bytes: VlDeviceWeightBytes) -> Result<(), Error> {
    for (component, policy, required) in [
        ("vision", policies.vision, bytes.vision),
        ("decoder", policies.decoder, bytes.decoder),
    ] {
        let limits = policy.offload().map_err(Error::backend_source)?;
        if let Some(budget) = limits.device_budget_bytes() {
            if required > budget {
                return Err(Error::backend(format!(
                    "LFM3 VL {component} requires {required} device parameter bytes with a budget of {budget}"
                )));
            }
        }
    }
    Ok(())
}

fn window(
    invocations: &ExecutionUnitAddressMap,
    bindings: &[Vec<WeightBinding>],
    policy: LayerWeightResidency,
) -> Result<u64, Error> {
    let bytes = bindings
        .iter()
        .map(|unit| physical_binding_bytes(unit))
        .collect::<Result<Vec<_>, _>>()
        .map_err(Error::backend_source)?;
    execution_unit_parameter_window_bytes(invocations, &bytes, policy)
        .map_err(Error::backend_source)
}

fn add(a: u64, b: u64) -> Result<u64, Error> {
    a.checked_add(b)
        .ok_or_else(|| Error::backend("LFM3 VL device parameter bytes overflowed"))
}
