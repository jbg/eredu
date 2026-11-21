//! Capability and resource orchestration for an installed text executable.

use super::{TextExecutable, TextPrefillInput, TextSessionDriver, TextSessionMechanisms};
use crate::{
    executable_binding::retained_session::PreparedInputPartPlan,
    executable_binding::ErasedTextExecutable, CapabilityEstimate, ResidencyReport,
};
use eredu_core::{
    backend::TextTransactionMechanisms,
    capability::NativeMemoryFacts,
    input::PreparedInputInspector,
    residency::{
        MemoryTier, ParameterConversionRetentionReport, ParameterConversionRetentionTrimReport,
        ParameterConversionTrimError,
    },
    BackendFailure, CapabilityError, InputMetadataKey, InputModality, InputTokenCount,
    ModelCapabilities, ModelCapabilitySession, ModelRuntime, ObservationKind, Observed,
    PreparedInputPart, PreparedInputPayload, RuntimeStateEstimate, StaticMemoryReport,
    TextGenerationBackend,
};

/// Physical byte counts for a separately owned parameter bank.
pub struct BankMemoryFacts {
    /// Bytes of owned parameters.
    pub owned_bytes: u64,
    /// Current host-resident bytes.
    pub host_resident_bytes: u64,
    /// Current device-resident bytes.
    pub device_resident_bytes: u64,
}

/// Observes native facts on the prepared executable and prompt.
pub trait TextCapabilityMechanisms<B: TextGenerationBackend>:
    TextSessionMechanisms<B> + super::operations::TextResidencyTelemetry<Self::Resources>
{
    /// Retained model capabilities and state geometry.
    fn capability(model: &Self::Model) -> &CapabilityEstimate {
        model.as_ref().capability_estimate()
    }

    /// Prepared prompt plan and exact logical bytes of declared media values.
    fn input_parts(
        model: &Self::Model,
        input: &Self::Prefill,
    ) -> Result<Vec<(PreparedInputPartPlan, Option<u64>)>, CapabilityError> {
        let input = input.borrowed();
        let plans = model.as_ref().prepared_input_plans(input)?;
        if plans.len() != input.parts.len() {
            return Err(CapabilityError::Observation(
                "prepared input accounting plan count differs from input parts".into(),
            ));
        }
        let inspector =
            <Self::TextValues as eredu_core::backend::NativeTextValues>::input_inspector();
        plans
            .into_iter()
            .zip(input.parts)
            .map(|(plan, part)| {
                let bytes = if matches!(plan, PreparedInputPartPlan::Media { .. }) {
                    prepared_media_bytes(part, &inspector)?
                } else {
                    None
                };
                Ok((plan, bytes))
            })
            .collect()
    }

    /// Ordered decoder IDs, including any declared media placeholders.
    fn input_ids(
        model: &Self::Model,
        input: &Self::Prefill,
        backend: &B,
    ) -> Result<Vec<u32>, B::Error> {
        model
            .as_ref()
            .prepared_input_token_ids(input.borrowed(), Self::context(backend))
    }

    /// Releases optional conversion claims on the exact installed executable.
    fn trim_conversions(
        model: &mut Self::Model,
    ) -> Result<Vec<ParameterConversionRetentionTrimReport>, ParameterConversionTrimError> {
        model.as_mut().trim_parameter_conversions()
    }

    /// Observes retained conversion claims without changing them.
    fn conversion_retention(
        model: &Self::Model,
    ) -> Result<Observed<Vec<ParameterConversionRetentionReport>>, BackendFailure> {
        model.as_ref().parameter_conversion_retention()
    }
}

impl<B, N> TextCapabilityMechanisms<B> for N
where
    B: TextGenerationBackend,
    N: TextSessionMechanisms<B, Prefill = crate::text_session_driver::TextPromptFor<B>>
        + super::operations::TextResidencyTelemetry<N::Resources>,
{
}

fn prepared_media_bytes<T>(
    part: &PreparedInputPart<T>,
    inspector: &impl PreparedInputInspector<T>,
) -> Result<Option<u64>, CapabilityError> {
    let PreparedInputPayload::Tensor(payload) = part.payload() else {
        return Ok(None);
    };
    let mut arrays = std::iter::once(payload).chain(
        [
            InputMetadataKey::PatchGrid,
            InputMetadataKey::PatchPositions,
            InputMetadataKey::AudioMask,
        ]
        .into_iter()
        .filter_map(|key| part.metadata_value(key)),
    );
    arrays
        .try_fold(0u64, |total, array| {
            let identity = inspector
                .identity(array)
                .map_err(|error| CapabilityError::Observation(error.to_string()))?;
            let bytes =
                crate::input_transfer::InputTransferResources::tensor_payload_bytes(&identity)
                    .map_err(|error| CapabilityError::Observation(error.to_string()))?;
            checked_add(total, bytes, "prepared media input bytes")
        })
        .map(Some)
}

impl<B, N> ModelCapabilitySession<B> for TextSessionDriver<N::Resources, N>
where
    B: TextGenerationBackend,
    Self: eredu_core::backend::TextOperationDriver<
        B,
        Prompt = crate::text_session_driver::TextPromptFor<B>,
    >,
    N: TextCapabilityMechanisms<B, Prefill = crate::text_session_driver::TextPromptFor<B>>
        + TextTransactionMechanisms,
{
    fn model_capabilities(
        runtime: &ModelRuntime<B, Self>,
    ) -> Result<ModelCapabilities, CapabilityError> {
        Ok(N::capability(&runtime.session().resources().model)
            .clone()
            .into_parts()
            .0)
    }

    fn count_prepared_input(
        runtime: &ModelRuntime<B, Self>,
        input: &crate::text_session_driver::TextPromptFor<B>,
    ) -> Result<InputTokenCount, CapabilityError> {
        let mut text = 0;
        let mut media = 0;
        let mut workspace = 0;
        let mut kind = ObservationKind::Exact;
        let model = &runtime.session().resources().model;
        for (plan, bytes) in N::input_parts(model, input)? {
            match plan {
                PreparedInputPartPlan::Text { positions }
                | PreparedInputPartPlan::Projected {
                    modality: InputModality::Text,
                    positions,
                } => {
                    text = checked_add(text, positions, "prepared text-token total")?;
                }
                PreparedInputPartPlan::Projected { positions, .. } => {
                    media = checked_add(media, positions, "prepared media-position total")?;
                }
                PreparedInputPartPlan::Media { shape } => {
                    let bytes = bytes.ok_or_else(|| CapabilityError::UnsupportedInput {
                        architecture: N::capability(model)
                            .clone()
                            .into_parts()
                            .0
                            .effective_model_type,
                        reason: "architecture media plan requires a tensor payload".into(),
                    })?;
                    let native = shape.execution_workspace_scalars.checked_mul(4).ok_or(
                        CapabilityError::ArithmeticOverflow {
                            operation: "prepared media execution workspace bytes",
                        },
                    )?;
                    media = checked_add(
                        media,
                        shape.decoder_positions,
                        "prepared media-position total",
                    )?;
                    workspace = checked_add(
                        workspace,
                        checked_add(bytes, native, "prepared media total workspace bytes")?,
                        "prepared media-workspace total",
                    )?;
                    kind = ObservationKind::Conservative;
                }
            }
        }
        Ok(InputTokenCount::prepared(
            text,
            media,
            checked_add(text, media, "prepared model-position total")?,
            workspace,
            kind,
        ))
    }

    fn prepared_input_token_ids(
        runtime: &ModelRuntime<B, Self>,
        input: &crate::text_session_driver::TextPromptFor<B>,
    ) -> Result<Vec<u32>, BackendFailure> {
        let session = runtime.session();
        session
            .ensure_no_submission_in_flight()
            .map_err(BackendFailure::from_error)?;
        N::input_ids(&session.resources().model, input, runtime.backend())
            .map_err(BackendFailure::from_error)
    }

    fn estimate_runtime_state(
        runtime: &ModelRuntime<B, Self>,
        input: InputTokenCount,
        max_output_tokens: u64,
        batch_size: u64,
    ) -> Result<RuntimeStateEstimate, CapabilityError> {
        let session = runtime.session();
        eredu_core::estimate_runtime_state(
            N::capability(&session.resources().model).state_layout(),
            input,
            max_output_tokens,
            batch_size,
            session.effective_floating_state_dtype_bytes(),
        )
    }

    fn static_memory(
        runtime: &ModelRuntime<B, Self>,
    ) -> Result<StaticMemoryReport, CapabilityError> {
        let session = runtime.session();
        let observation_error = |error: B::Error| CapabilityError::Observation(error.to_string());
        let residency = session.residency_report().map_err(observation_error)?;
        let banks = session
            .parameter_bank_report()
            .map_err(observation_error)?
            .map(|report| bank_memory_facts(Some(report)))
            .unwrap_or_default();
        let mut components = session
            .component_residency_reports()
            .map_err(observation_error)?
            .into_iter();
        let residency = residency.or_else(|| components.next());
        static_memory_from_residency_and_components(
            residency,
            components,
            banks,
            N::native_memory(),
        )
    }

    fn trim_parameter_conversions(
        runtime: &mut ModelRuntime<B, Self>,
    ) -> Result<Vec<ParameterConversionRetentionTrimReport>, ParameterConversionTrimError> {
        let session = runtime.session_mut();
        session
            .native()
            .check_native_health()
            .map_err(BackendFailure::from_error)?;
        session
            .submission_idle()
            .map_err(|_| ParameterConversionTrimError::NotQuiescent)?;
        session
            .with_model_transaction(|model| Ok(N::trim_conversions(model)))
            .map_err(BackendFailure::from_error)?
    }

    fn parameter_conversion_retention(
        runtime: &ModelRuntime<B, Self>,
    ) -> Result<Observed<Vec<ParameterConversionRetentionReport>>, BackendFailure> {
        N::conversion_retention(&runtime.session().resources().model)
    }
}

fn checked_add(left: u64, right: u64, operation: &'static str) -> Result<u64, CapabilityError> {
    left.checked_add(right)
        .ok_or(CapabilityError::ArithmeticOverflow { operation })
}

/// Projects the physical owners of each admitted parameter bank.
pub fn bank_memory_facts(
    banks: Option<crate::expert::ParameterBanksResidencyReport>,
) -> Vec<BankMemoryFacts> {
    let Some(report) = banks else {
        return Vec::new();
    };
    report
        .banks()
        .values()
        .map(|bank| BankMemoryFacts {
            owned_bytes: bank.owned_bytes(),
            host_resident_bytes: bank.host_resident_bytes(),
            device_resident_bytes: bank.device_resident_bytes(),
        })
        .collect()
}

/// Combines selected parameter residency with native memory counters.
pub fn static_memory_from_residency_and_components(
    residency: Option<ResidencyReport>,
    components: impl Iterator<Item = ResidencyReport>,
    banks: impl IntoIterator<Item = BankMemoryFacts>,
    native: NativeMemoryFacts,
) -> Result<StaticMemoryReport, CapabilityError> {
    let retention = residency.as_ref().map_or_else(
        eredu_core::residency::unreported_parameter_conversion_retention,
        |report| report.offload().parameter_conversion_retention().clone(),
    );
    let mut conversions = residency
        .as_ref()
        .map(|report| report.device_parameter_conversion_bytes());
    let (mut logical, mut host, mut device, mut disk, mut cached_shards) =
        if let Some(report) = residency {
            let planned = report.offload().planned_bytes();
            let resident = report.offload().resident_bytes();
            let logical = checked_add(
                checked_add(
                    planned.get(MemoryTier::Host),
                    planned.get(MemoryTier::Device),
                    "planned host plus device parameters",
                )?,
                planned.get(MemoryTier::Disk),
                "complete planned parameter bytes",
            )?;
            (
                Observed::Available {
                    value: logical,
                    kind: ObservationKind::Exact,
                    source: "validated bounded-residency plan".into(),
                },
                Observed::Available {
                    value: resident.get(MemoryTier::Host),
                    kind: ObservationKind::Exact,
                    source: "bounded-residency manager".into(),
                },
                Observed::Available {
                    value: resident.get(MemoryTier::Device),
                    kind: ObservationKind::Exact,
                    source: "bounded-residency manager".into(),
                },
                Observed::Available {
                    value: planned.get(MemoryTier::Disk),
                    kind: ObservationKind::Exact,
                    source: "bounded-residency plan".into(),
                },
                Observed::Available {
                    value: report.weight_store().currently_cached_shards as u64,
                    kind: ObservationKind::Observational,
                    source: "checkpoint shard cache".into(),
                },
            )
        } else {
            (
                Observed::Unavailable {
                    reason: "loaded model exposes neither resident parameters nor a residency plan"
                        .into(),
                },
                Observed::Unavailable {
                    reason: "host residency unavailable".into(),
                },
                Observed::Unavailable {
                    reason: "device residency unavailable".into(),
                },
                Observed::Unavailable {
                    reason: "disk residency unavailable".into(),
                },
                Observed::Unavailable {
                    reason: "checkpoint shard-cache information unavailable".into(),
                },
            )
        };
    for component in components {
        let planned = component.offload().planned_bytes();
        let resident = component.offload().resident_bytes();
        for tier in [MemoryTier::Device, MemoryTier::Host, MemoryTier::Disk] {
            add_parameter_observation(&mut logical, planned.get(tier))?;
        }
        add_parameter_observation(&mut host, resident.get(MemoryTier::Host))?;
        add_parameter_observation(&mut device, resident.get(MemoryTier::Device))?;
        add_parameter_observation(&mut disk, planned.get(MemoryTier::Disk))?;
        if let Some(bytes) = conversions.as_mut() {
            *bytes = checked_add(
                *bytes,
                component.device_parameter_conversion_bytes(),
                "component conversion bytes",
            )?;
        }
        cached_shards = Observed::unavailable("component checkpoint readers can share storage");
    }
    // Independently addressable parameters are excluded from ordinary block
    // residency. Count each bank's owned entries once, even when banks share
    // the same physical pool and its aggregate telemetry.
    for bank in banks {
        add_parameter_observation(&mut logical, bank.owned_bytes)?;
        add_parameter_observation(&mut host, bank.host_resident_bytes)?;
        add_parameter_observation(&mut device, bank.device_resident_bytes)?;
        add_parameter_observation(&mut disk, bank.owned_bytes)?;
    }
    if let Some(bytes) = conversions {
        add_parameter_observation(&mut device, bytes)?;
    }
    Ok(StaticMemoryReport {
        parameter_conversion_retention: retention,
        logical_parameter_bytes: logical,
        current_host_resident_bytes: host,
        current_device_resident_bytes: device,
        current_device_parameter_conversion_bytes: conversions.map_or_else(
            || Observed::unavailable("parameter conversion ownership unavailable"),
            |bytes| Observed::exact(bytes, "resident parameter conversion owners"),
        ),
        planned_disk_backed_bytes: disk,
        backend_active_allocation_bytes: native.active_allocation_bytes,
        backend_allocator_cache_bytes: native.allocator_cache_bytes,
        physical_semantics: native.physical_semantics,
        currently_cached_shards: cached_shards,
    })
}

fn add_parameter_observation(
    observation: &mut Observed<u64>,
    extra: u64,
) -> Result<(), CapabilityError> {
    if let Observed::Available { value, source, .. } = observation {
        *value = checked_add(*value, extra, "resident parameter storage bytes")?;
        if extra != 0 {
            *source = "ordinary residency plus independently owned native parameter storage".into();
        }
    }
    Ok(())
}
