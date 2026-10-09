//! Cold admission for independent speech-component residency pools.
//!
//! Provider windows coexist: their allowances are added, never reused as a
//! whole-stack allowance. These parameter and materialization bounds exclude
//! backbone weights, recurrent state, activation workspaces, and source caches.

use std::collections::BTreeMap;

use eredu_checkpoint::{recipe::RecipeError, store::SharedCheckpointSource};
use eredu_runtime::{LayerWeightResidency, WeightBinding, WeightResidencyPolicyError};

use super::artifact::ParameterAdmission;

/// Independently materialized speech execution providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SpeechProvider {
    /// Recurrent transcription embedding, prediction layers and joint projections.
    Rnnt,
    /// Streaming acoustic encoder and projection.
    Perception,
    /// Function-channel vocabulary projection.
    FunctionHead,
    /// Bidirectional character transformer.
    CharacterEncoder,
    /// Causal TTS transformer, shared by both guidance state lanes.
    TtsTransformer,
    /// Gaussian mixture prediction stages.
    Mog,
    /// TTS residual refinement tables and projections.
    Rvq,
    /// Character embedding, ingress, speaker conditioning, and fusion.
    Conditioning,
    /// Finite speaker waveform encoder.
    CodecEncoder,
    /// Streaming waveform decoder.
    CodecDecoder,
    /// Codec residual quantizer shared by encoder and decoder.
    CodecQuantizer,
}

impl SpeechProvider {
    /// Complete independent provider set for coupled speech construction.
    pub const ALL: [Self; 11] = [
        Self::Rnnt,
        Self::Perception,
        Self::FunctionHead,
        Self::CharacterEncoder,
        Self::TtsTransformer,
        Self::Mog,
        Self::Rvq,
        Self::Conditioning,
        Self::CodecEncoder,
        Self::CodecDecoder,
        Self::CodecQuantizer,
    ];
}

/// Exact converted payload sizing and a conservative materialization bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeechProviderResources {
    /// Sum of physical output bindings; aliases add no payload.
    pub parameter_bytes: u64,
    /// Largest independently acquired unit's physical output payload.
    pub largest_unit_bytes: u64,
    /// Encoded inputs, converter workspace, and recipe temporaries plus outputs.
    /// This conservative bound adds parameter peaks within the largest loading
    /// unit, including any shared-source duplication, and is separate from
    /// residency capacity.
    pub materialization_peak_bytes: u64,
}

impl SpeechProviderResources {
    /// Reads exact prepared recipe metadata without reading checkpoint payloads.
    /// Original admission supplies physical converter overhead; selected recipes
    /// supply execution dtype and recipe temporaries after F32 normalization.
    pub fn from_bindings(
        source: &SharedCheckpointSource,
        units: &[Vec<WeightBinding>],
        parameters: &BTreeMap<String, ParameterAdmission>,
    ) -> Result<Self, SpeechResidencyError> {
        let mut result = Self {
            parameter_bytes: 0,
            largest_unit_bytes: 0,
            materialization_peak_bytes: 0,
        };
        for unit in units {
            let mut unit_bytes = 0;
            let mut unit_peak_bytes = 0;
            for binding in unit.iter().filter(|binding| !binding.is_alias()) {
                let name = binding.logical_target().unwrap_or_else(|| binding.name());
                let admission = parameters
                    .get(name)
                    .ok_or_else(|| SpeechResidencyError::MissingParameter(name.into()))?;
                let recipe = binding.source_recipe();
                if recipe.infer(source.as_ref())?.byte_len() != binding.expected_bytes() {
                    return Err(SpeechResidencyError::InvalidParameter(name.into()));
                }
                unit_bytes = add(unit_bytes, binding.expected_bytes())?;
                let peak = add(
                    add(
                        recipe.peak_materialization_bytes(source.as_ref())?,
                        admission.conversion_workspace_bytes,
                    )?,
                    admission.stored_bytes,
                )?;
                unit_peak_bytes = add(unit_peak_bytes, peak)?;
            }
            result.parameter_bytes = add(result.parameter_bytes, unit_bytes)?;
            result.largest_unit_bytes = result.largest_unit_bytes.max(unit_bytes);
            result.materialization_peak_bytes =
                result.materialization_peak_bytes.max(unit_peak_bytes);
        }
        Ok(result)
    }
}

/// Finite aggregate allowances for independent speech parameter pools.
#[derive(Debug, Clone)]
pub struct SpeechResidencyPolicy {
    providers: BTreeMap<SpeechProvider, LayerWeightResidency>,
    device_limit_bytes: u64,
    host_limit_bytes: u64,
}

/// Checked parameter allowances, including separately reserved loading scratch.
/// Device and host are logical tiers; this is not a process/unified-memory estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeechResidencyReport {
    /// Sum of bounded device caps and resident provider outputs.
    pub device_parameter_bytes: u64,
    /// Bounded derived tensors retained independently of parameter pools.
    pub execution_cache_device_bytes: u64,
    /// Sum of independently retained host-pool caps.
    pub host_parameter_bytes: u64,
    /// Conservative concurrent materialization reservation across all providers.
    pub host_materialization_bytes: u64,
    /// Host parameter allowances plus materialization reservation.
    pub host_required_bytes: u64,
}

/// Missing bounds, invalid geometry, or insufficient aggregate capacity.
#[derive(Debug, thiserror::Error)]
pub enum SpeechResidencyError {
    /// The complete provider set must be selected and accounted before loading.
    #[error("VoiceChat speech provider {0:?} has no residency policy or resource descriptor")]
    MissingProvider(SpeechProvider),
    /// Independent unbounded pools cannot establish an aggregate bound.
    #[error("VoiceChat speech provider {provider:?} has no finite {tier} parameter limit")]
    Unbounded {
        /// Unbounded provider.
        provider: SpeechProvider,
        /// Host or device tier.
        tier: &'static str,
    },
    /// A pool cannot accommodate even its largest independent output unit.
    #[error(
        "VoiceChat speech provider {provider:?} requires at least {required} device bytes, limit {limit}"
    )]
    UnitCapacity {
        /// Rejected provider.
        provider: SpeechProvider,
        /// Minimum logical unit payload.
        required: u64,
        /// Supplied device capacity.
        limit: u64,
    },
    /// The sum of independent allowances exceeds the enclosing bound.
    #[error("VoiceChat speech {tier} allowance requires {required} bytes, limit {limit}")]
    AggregateCapacity {
        /// Host or device tier.
        tier: &'static str,
        /// Sum of simultaneously retained allowances.
        required: u64,
        /// Enclosing bound.
        limit: u64,
    },
    /// An exact normalized binding lacks source admission accounting.
    #[error("VoiceChat speech resource accounting lacks parameter {0}")]
    MissingParameter(String),
    /// Exact recipe output disagrees with the binding payload.
    #[error("VoiceChat speech resource accounting has inconsistent parameter {0}")]
    InvalidParameter(String),
    /// Checked byte or concurrent materialization arithmetic overflowed.
    #[error("VoiceChat speech residency byte arithmetic overflow")]
    ArithmeticOverflow,
    /// Invalid underlying shared residency controls.
    #[error(transparent)]
    Policy(#[from] WeightResidencyPolicyError),
    /// Exact recipe sizing failed.
    #[error(transparent)]
    Recipe(#[from] RecipeError),
}

impl SpeechResidencyPolicy {
    /// Selects all independent providers under finite aggregate tier limits.
    pub fn new(
        providers: BTreeMap<SpeechProvider, LayerWeightResidency>,
        device_limit_bytes: u64,
        host_limit_bytes: u64,
    ) -> Result<Self, SpeechResidencyError> {
        for provider in SpeechProvider::ALL {
            let policy = providers
                .get(&provider)
                .ok_or(SpeechResidencyError::MissingProvider(provider))?;
            if !policy.is_fully_resident() {
                let offload = policy.offload()?;
                for (tier, bound) in [
                    ("device", offload.device_budget_bytes()),
                    ("host", offload.host_budget_bytes()),
                ] {
                    if bound.is_none() {
                        return Err(SpeechResidencyError::Unbounded { provider, tier });
                    }
                }
            }
        }
        Ok(Self {
            providers,
            device_limit_bytes,
            host_limit_bytes,
        })
    }

    /// Exact shared policy passed unchanged to this provider's materializer.
    pub fn provider(&self, provider: SpeechProvider) -> LayerWeightResidency {
        self.providers[&provider]
    }

    /// Validates every concurrent pool before a native binder is invoked.
    /// Native host-transfer alignment remains checked by the shared materializer
    /// against each finite host cap. This does not certify activation/cache fit.
    pub fn validate(
        &self,
        providers: &BTreeMap<SpeechProvider, SpeechProviderResources>,
    ) -> Result<SpeechResidencyReport, SpeechResidencyError> {
        self.validate_with_execution_cache(providers, 0)
    }

    /// Admits architecture-sized retained derived tensors alongside provider pools.
    pub fn validate_with_execution_cache(
        &self,
        providers: &BTreeMap<SpeechProvider, SpeechProviderResources>,
        execution_cache_device_bytes: u64,
    ) -> Result<SpeechResidencyReport, SpeechResidencyError> {
        let mut report = SpeechResidencyReport {
            device_parameter_bytes: 0,
            execution_cache_device_bytes,
            host_parameter_bytes: 0,
            host_materialization_bytes: 0,
            host_required_bytes: 0,
        };
        for provider in SpeechProvider::ALL {
            let resources = providers
                .get(&provider)
                .ok_or(SpeechResidencyError::MissingProvider(provider))?;
            let policy = self.provider(provider);
            let (device, host) = if policy.is_fully_resident() {
                (resources.parameter_bytes, 0)
            } else {
                let offload = policy.offload()?;
                let device = offload
                    .device_budget_bytes()
                    .expect("validated finite policy");
                if device < resources.largest_unit_bytes {
                    return Err(SpeechResidencyError::UnitCapacity {
                        provider,
                        required: resources.largest_unit_bytes,
                        limit: device,
                    });
                }
                (
                    device,
                    offload
                        .host_budget_bytes()
                        .expect("validated finite policy"),
                )
            };
            report.device_parameter_bytes = add(report.device_parameter_bytes, device)?;
            report.host_parameter_bytes = add(report.host_parameter_bytes, host)?;
            // Independent disk providers may prefetch while another is executing.
            // Reserve foreground plus all admitted background slots conservatively.
            let parallel_materializations = match policy.dense() {
                Some(options) => u64::try_from(options.background_queue_capacity())
                    .ok()
                    .and_then(|n| n.checked_add(1))
                    .ok_or(SpeechResidencyError::ArithmeticOverflow)?,
                None => 1,
            };
            let scratch = resources
                .materialization_peak_bytes
                .checked_mul(parallel_materializations)
                .ok_or(SpeechResidencyError::ArithmeticOverflow)?;
            report.host_materialization_bytes = add(report.host_materialization_bytes, scratch)?;
        }
        report.host_required_bytes = add(
            report.host_parameter_bytes,
            report.host_materialization_bytes,
        )?;
        for (tier, required, limit) in [
            (
                "device",
                add(
                    report.device_parameter_bytes,
                    report.execution_cache_device_bytes,
                )?,
                self.device_limit_bytes,
            ),
            ("host", report.host_required_bytes, self.host_limit_bytes),
        ] {
            if required > limit {
                return Err(SpeechResidencyError::AggregateCapacity {
                    tier,
                    required,
                    limit,
                });
            }
        }
        Ok(report)
    }
}

fn add(a: u64, b: u64) -> Result<u64, SpeechResidencyError> {
    a.checked_add(b)
        .ok_or(SpeechResidencyError::ArithmeticOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::residency::OffloadConfig;
    use eredu_runtime::{DenseDiskStreamLoadOptions, LayerwiseLoadOptions};

    fn descriptors() -> BTreeMap<SpeechProvider, SpeechProviderResources> {
        SpeechProvider::ALL
            .into_iter()
            .map(|provider| {
                (
                    provider,
                    if provider == SpeechProvider::Rnnt {
                        rnnt()
                    } else {
                        SpeechProviderResources {
                            parameter_bytes: 100,
                            largest_unit_bytes: 20,
                            materialization_peak_bytes: 150,
                        }
                    },
                )
            })
            .collect()
    }

    fn rnnt() -> SpeechProviderResources {
        SpeechProviderResources {
            parameter_bytes: 7,
            largest_unit_bytes: 7,
            materialization_peak_bytes: 11,
        }
    }

    fn policy(placement: LayerWeightResidency, device: u64, host: u64) -> SpeechResidencyPolicy {
        SpeechResidencyPolicy::new(
            SpeechProvider::ALL
                .into_iter()
                .map(|id| (id, placement))
                .collect(),
            device,
            host,
        )
        .unwrap()
    }

    #[test]
    fn retained_execution_cache_cannot_consume_parameter_capacity_twice() {
        let resources = descriptors();
        assert!(matches!(
            policy(LayerWeightResidency::FullyResident, 1007, 1511)
                .validate_with_execution_cache(&resources, 8),
            Err(SpeechResidencyError::AggregateCapacity {
                tier: "device",
                required: 1015,
                limit: 1007
            })
        ));
        let report = policy(LayerWeightResidency::FullyResident, 1015, 1511)
            .validate_with_execution_cache(&resources, 8)
            .unwrap();
        assert_eq!(report.device_parameter_bytes, 1007);
        assert_eq!(report.execution_cache_device_bytes, 8);
    }

    #[test]
    fn independent_pools_include_bounded_rnnt_and_loading_scratch() {
        let resources = descriptors();
        let resident = policy(LayerWeightResidency::FullyResident, 1007, 1511)
            .validate(&resources)
            .unwrap();
        assert_eq!(
            (
                resident.device_parameter_bytes,
                resident.host_required_bytes
            ),
            (1007, 1511)
        );
        let host = LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
            OffloadConfig::new(Some(40), Some(120), 2).unwrap(),
        ));
        let report = policy(host, 440, 2831).validate(&resources).unwrap();
        assert_eq!(
            (
                report.device_parameter_bytes,
                report.host_parameter_bytes,
                report.host_materialization_bytes
            ),
            (440, 1320, 1511)
        );
        assert!(matches!(
            policy(host, 439, 2831).validate(&resources),
            Err(SpeechResidencyError::AggregateCapacity {
                tier: "device",
                required: 440,
                limit: 439
            })
        ));
        let disk = LayerWeightResidency::DenseDiskStream(
            DenseDiskStreamLoadOptions::new(40, 50, 1, 2).unwrap(),
        );
        let report = policy(disk, 440, 5083).validate(&resources).unwrap();
        assert_eq!(
            (
                report.host_parameter_bytes,
                report.host_materialization_bytes
            ),
            (550, 4533)
        );
        assert!(matches!(
            policy(disk, 440, 5082).validate(&resources),
            Err(SpeechResidencyError::AggregateCapacity {
                tier: "host",
                required: 5083,
                limit: 5082
            })
        ));
    }

    #[test]
    fn unbounded_and_overflowing_allowances_are_rejected() {
        let unbounded = SpeechProvider::ALL
            .into_iter()
            .map(|id| (id, LayerWeightResidency::LayerwiseHost(Default::default())))
            .collect();
        assert!(matches!(
            SpeechResidencyPolicy::new(unbounded, u64::MAX, u64::MAX),
            Err(SpeechResidencyError::Unbounded { .. })
        ));
        let mut resources = descriptors();
        resources
            .get_mut(&SpeechProvider::Perception)
            .unwrap()
            .parameter_bytes = u64::MAX;
        assert!(matches!(
            policy(LayerWeightResidency::FullyResident, u64::MAX, u64::MAX).validate(&resources),
            Err(SpeechResidencyError::ArithmeticOverflow)
        ));
    }
}
