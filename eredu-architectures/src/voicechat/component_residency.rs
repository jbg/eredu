//! Cold parameter admission for the complete coupled component owner.
//!
//! The selected Nemotron target and all independent speech provider allowances
//! coexist. These bounds describe logical parameter tiers and materialization
//! scratch, excluding activations, recurrent state, and checkpoint source caches.

use eredu_checkpoint::recipe::{AtomicRecipeSet, RecipeError};
use eredu_runtime::{bindings_from_recipe_set, LayerWeightResidency, RecipeBindingError};

use super::{
    backbone::{BackbonePreparationError, PreparedBackboneParameters},
    provider_construction::PreparedSpeechProviders,
    provider_residency::{
        SpeechProviderResources, SpeechResidencyError, SpeechResidencyPolicy, SpeechResidencyReport,
    },
};

/// Exact target payloads and conservative loading-unit scratch, before placement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackboneResources {
    /// Physical normalized parameter outputs, including embedding and agent head.
    pub parameter_bytes: u64,
    /// Pinned embedding, final normalization, and agent-head payloads.
    pub static_parameter_bytes: u64,
    /// Total payload of the independently acquired target blocks.
    pub layer_parameter_bytes: u64,
    /// Target block payloads in exact traversal order, before selecting a window.
    pub layer_unit_bytes: Vec<u64>,
    /// Largest independently acquired target block payload.
    pub largest_layer_bytes: u64,
    /// Scratch required while loading the pinned static bundle.
    pub static_materialization_peak_bytes: u64,
    /// Largest block's loading scratch, including conversions.
    pub layer_materialization_peak_bytes: u64,
    /// Maximum scratch across sequential static and target-block loading.
    pub materialization_peak_bytes: u64,
}

impl BackboneResources {
    /// Sizes retained recipes without allocating native slots or reading payloads.
    pub fn from_prepared(
        prepared: &PreparedBackboneParameters,
    ) -> Result<Self, VoiceChatResidencyError> {
        // A tied Nemotron readout calls the embedding projection directly; it
        // declares no second output slot. An extra head is an invalid retained
        // catalog, rather than an alias that may silently be charged or ignored.
        if prepared.args().tie_word_embeddings
            && prepared
                .parameters()
                .keys()
                .any(|key| key.starts_with("lm_head."))
        {
            return Err(VoiceChatResidencyError::UnexpectedTiedHead);
        }
        let mut recipes = vec![prepared.static_recipes()];
        for index in 0..prepared.args().num_hidden_layers as usize {
            recipes.push(prepared.layer_recipes(index)?);
        }
        let selected = recipes.iter().map(|unit| unit.len()).sum::<usize>();
        if selected != prepared.parameters().len() {
            return Err(VoiceChatResidencyError::BackboneCatalog);
        }
        let units = recipes
            .into_iter()
            .map(|recipes| {
                let recipes = AtomicRecipeSet::new(prepared.source().as_ref(), recipes)?;
                Ok(bindings_from_recipe_set(
                    prepared.source().as_ref(),
                    recipes,
                )?)
            })
            .collect::<Result<Vec<_>, VoiceChatResidencyError>>()?;
        let statics = SpeechProviderResources::from_bindings(
            prepared.source(),
            &units[..1],
            prepared.parameters(),
        )?;
        let layers = SpeechProviderResources::from_bindings(
            prepared.source(),
            &units[1..],
            prepared.parameters(),
        )?;
        Ok(Self {
            parameter_bytes: add(statics.parameter_bytes, layers.parameter_bytes)?,
            static_parameter_bytes: statics.parameter_bytes,
            layer_parameter_bytes: layers.parameter_bytes,
            layer_unit_bytes: units[1..]
                .iter()
                .map(|unit| {
                    unit.iter()
                        .filter(|binding| !binding.is_alias())
                        .try_fold(0, |bytes, binding| add(bytes, binding.expected_bytes()))
                })
                .collect::<Result<_, _>>()?,
            largest_layer_bytes: layers.largest_unit_bytes,
            static_materialization_peak_bytes: statics.materialization_peak_bytes,
            layer_materialization_peak_bytes: layers.materialization_peak_bytes,
            materialization_peak_bytes: statics
                .materialization_peak_bytes
                .max(layers.materialization_peak_bytes),
        })
    }
}

/// Enclosing finite bounds for the selected backbone and speech pools.
#[derive(Debug, Clone)]
pub struct VoiceChatResidencyPolicy {
    backbone: LayerWeightResidency,
    speech: SpeechResidencyPolicy,
    device_limit_bytes: u64,
    host_limit_bytes: u64,
}

/// Combined parameter admission; device and host are logical allocation tiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoiceChatResidencyReport {
    /// Individually checked speech parameter pools, including the selected RNNT pool.
    pub speech: SpeechResidencyReport,
    /// All Nemotron weights, independent of the selected residency.
    pub backbone_parameter_bytes: u64,
    /// Pinned Nemotron embedding, final normalization, and agent head.
    pub backbone_static_parameter_bytes: u64,
    /// Selected device allowance, including pinned statics exactly once.
    pub backbone_device_parameter_bytes: u64,
    /// Selected target-block host allowance; pinned statics are excluded.
    pub backbone_host_parameter_bytes: u64,
    /// Backbone loading scratch, including admitted disk background slots.
    pub backbone_materialization_bytes: u64,
    /// Selected backbone and independent speech device allowances.
    pub device_parameter_bytes: u64,
    /// Selected backbone and independent speech host-pool allowances.
    pub host_parameter_bytes: u64,
    /// Speech loading reservations plus backbone loading scratch.
    pub host_materialization_bytes: u64,
    /// Host parameter allowances plus materialization reservations.
    pub host_required_bytes: u64,
}

/// Cold exact-source accounting or aggregate parameter admission failed.
#[derive(Debug, thiserror::Error)]
pub enum VoiceChatResidencyError {
    /// A speech provider has invalid or insufficient parameter bounds.
    #[error(transparent)]
    Speech(#[from] SpeechResidencyError),
    /// The retained backbone cannot supply the selected target units.
    #[error(transparent)]
    Backbone(#[from] BackbonePreparationError),
    /// Exact recipe metadata cannot be evaluated.
    #[error(transparent)]
    Recipe(#[from] RecipeError),
    /// Exact physical bindings cannot be declared.
    #[error(transparent)]
    Binding(#[from] RecipeBindingError),
    /// Tied readout has only its embedding owner and no separate head slot.
    #[error("VoiceChat tied backbone retains an unexpected separate output head")]
    UnexpectedTiedHead,
    /// Retained parameters lie outside the selected static and target units.
    #[error("VoiceChat backbone parameter catalog differs from its target schedule")]
    BackboneCatalog,
    /// A bounded backbone needs finite device and host allowances.
    #[error("VoiceChat backbone has no finite {tier} parameter limit")]
    BackboneUnbounded {
        /// Logical allocation tier.
        tier: &'static str,
    },
    /// Device capacity cannot retain statics and the selected block window.
    #[error(
        "VoiceChat backbone statics and selected block window require {required} device bytes, limit {limit}"
    )]
    BackboneUnitCapacity {
        /// Pinned statics plus the largest selected consecutive block window.
        required: u64,
        /// Supplied backbone device capacity.
        limit: u64,
    },
    /// Concurrent component allowances exceed the complete owner bound.
    #[error("VoiceChat components require {required} {tier} bytes, limit {limit}")]
    Capacity {
        /// Logical allocation tier.
        tier: &'static str,
        /// Sum of simultaneously admitted component allowances.
        required: u64,
        /// Enclosing finite bound.
        limit: u64,
    },
    /// Checked aggregate byte arithmetic overflowed.
    #[error("VoiceChat component residency byte arithmetic overflow")]
    ArithmeticOverflow,
}

impl VoiceChatResidencyPolicy {
    /// Selects backbone and speech policies under whole-owner finite bounds.
    pub fn new(
        backbone: LayerWeightResidency,
        speech: SpeechResidencyPolicy,
        device_limit_bytes: u64,
        host_limit_bytes: u64,
    ) -> Self {
        Self {
            backbone,
            speech,
            device_limit_bytes,
            host_limit_bytes,
        }
    }

    /// Exact target policy, whose device allowance includes pinned statics.
    pub fn backbone(&self) -> LayerWeightResidency {
        self.backbone
    }

    /// Exact provider selection passed unchanged to speech construction.
    pub fn speech(&self) -> &SpeechResidencyPolicy {
        &self.speech
    }

    /// Admits every component before any native binding callback runs.
    pub fn validate(
        &self,
        backbone: &PreparedBackboneParameters,
        speech: &PreparedSpeechProviders,
    ) -> Result<VoiceChatResidencyReport, VoiceChatResidencyError> {
        let speech = speech.validate_residency(&self.speech)?;
        self.combine(&BackboneResources::from_prepared(backbone)?, speech)
    }

    fn combine(
        &self,
        backbone: &BackboneResources,
        speech: SpeechResidencyReport,
    ) -> Result<VoiceChatResidencyReport, VoiceChatResidencyError> {
        let (device, host) = if self.backbone.is_fully_resident() {
            (backbone.parameter_bytes, 0)
        } else {
            let offload = self
                .backbone
                .offload()
                .map_err(SpeechResidencyError::from)?;
            let device = offload
                .device_budget_bytes()
                .ok_or(VoiceChatResidencyError::BackboneUnbounded { tier: "device" })?;
            let host = offload
                .host_budget_bytes()
                .ok_or(VoiceChatResidencyError::BackboneUnbounded { tier: "host" })?;
            let depth = self
                .backbone
                .device_depth(backbone.layer_unit_bytes.len())
                .min(backbone.layer_unit_bytes.len());
            let window_bytes = if depth == 0 {
                0
            } else {
                backbone
                    .layer_unit_bytes
                    .windows(depth)
                    .map(|window| window.iter().try_fold(0, |sum, &bytes| add(sum, bytes)))
                    .try_fold(0, |largest: u64, bytes| {
                        Ok::<_, VoiceChatResidencyError>(largest.max(bytes?))
                    })?
            };
            let required = add(backbone.static_parameter_bytes, window_bytes)?;
            if device < required {
                return Err(VoiceChatResidencyError::BackboneUnitCapacity {
                    required,
                    limit: device,
                });
            }
            (device, host)
        };
        let loading_slots = match self.backbone.dense() {
            Some(options) => u64::try_from(options.background_queue_capacity())
                .ok()
                .and_then(|n| n.checked_add(1))
                .ok_or(VoiceChatResidencyError::ArithmeticOverflow)?,
            None => 1,
        };
        // Construction completes pinned loading before creating the block
        // provider. Subsequent block prefetches can overlap each other, but
        // never pinned materialization.
        let scratch = backbone.static_materialization_peak_bytes.max(
            backbone
                .layer_materialization_peak_bytes
                .checked_mul(loading_slots)
                .ok_or(VoiceChatResidencyError::ArithmeticOverflow)?,
        );
        let host_parameter_bytes = add(speech.host_parameter_bytes, host)?;
        let host_materialization_bytes = add(speech.host_materialization_bytes, scratch)?;
        let report = VoiceChatResidencyReport {
            speech,
            backbone_parameter_bytes: backbone.parameter_bytes,
            backbone_static_parameter_bytes: backbone.static_parameter_bytes,
            backbone_device_parameter_bytes: device,
            backbone_host_parameter_bytes: host,
            backbone_materialization_bytes: scratch,
            device_parameter_bytes: add(speech.device_parameter_bytes, device)?,
            host_parameter_bytes,
            host_materialization_bytes,
            host_required_bytes: add(host_parameter_bytes, host_materialization_bytes)?,
        };
        for (tier, required, limit) in [
            (
                "device",
                add(
                    report.device_parameter_bytes,
                    report.speech.execution_cache_device_bytes,
                )?,
                self.device_limit_bytes,
            ),
            ("host", report.host_required_bytes, self.host_limit_bytes),
        ] {
            if required > limit {
                return Err(VoiceChatResidencyError::Capacity {
                    tier,
                    required,
                    limit,
                });
            }
        }
        Ok(report)
    }
}

fn add(a: u64, b: u64) -> Result<u64, VoiceChatResidencyError> {
    a.checked_add(b)
        .ok_or(VoiceChatResidencyError::ArithmeticOverflow)
}
