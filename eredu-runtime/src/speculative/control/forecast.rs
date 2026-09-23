//! Pure observations of the installed canonical lane; never enter scheduler work.
use super::*;
use crate::memory_estimation::*;
use crate::memory_forecast::*;

fn unsupported(reason: impl Into<String>) -> GenerationForecastError {
    GenerationForecastError::UnsupportedContinuation(reason.into())
}
fn allowance(bytes: Option<u64>, detail: &str) -> MemoryBytes {
    match bytes {
        Some(n) => MemoryBytes::estimated(0, n, detail),
        None => MemoryBytes::unknown(detail),
    }
}

impl<'a, E, S, C, P> Session<'a, E, S, C, P>
where
    E: SpeculativeExecutor + 'a,
    S: SpeculativeSampling<Logits = E::Logits, Error = E::Error, Context<'a> = E::Context<'a>> + 'a,
    C: SpeculativeConstraint,
    P: SpeculativePublisher<C>,
{
    pub(super) fn forecast_inner(
        &self,
        additional_tokens: u64,
        options: &GenerationForecastOptions,
    ) -> Result<SpeculativeContinuationForecast, GenerationForecastError> {
        self.healthy().map_err(|e| unsupported(e.to_string()))?;
        let request = self
            .request()
            .ok_or_else(|| unsupported("prefill has not committed"))?;
        request
            .validate_control_edit()
            .map_err(|e| unsupported(e.to_string()))?;
        let (target_profile, mechanism) = self.forecast_profiles.as_ref().ok_or_else(|| {
            unsupported("selected speculative mechanism has no continuation memory profile")
        })?;
        let scheduler = self.scheduler.requests.options();
        let width = request.forecast_max_draft_tokens();
        if width > mechanism.proposal_capacity {
            return Err(unsupported(
                "configured proposals exceed the selected mechanism capacity",
            ));
        }
        let extra = speculative_continuation_positions(additional_tokens, width, scheduler)?;
        let native = request
            .continuation_memory_observation(self.scheduler.executor, extra)
            .map_err(|error| match error {
                SpeculativeControlError::Backend(error) => GenerationForecastError::Backend(error),
                error => unsupported(error.to_string()),
            })?
            .ok_or_else(|| unsupported("executor cannot observe settled speculative state"))?;
        let model = |mut profile: LoadedMemoryProfile,
                     native: eredu_core::speculative::SpeculativeModelMemoryObservation|
         -> Result<_, GenerationForecastError> {
            let mut parameters = native.parameters;
            // A narrow resident owner can omit independently managed parameter
            // banks. Do not present that partial observation as the whole model.
            if parameters
                .logical_parameter_bytes
                .value()
                .zip(profile.parameters.logical_parameter_bytes.value())
                .is_some_and(|(observed, selected)| observed < selected)
            {
                parameters.current_device_resident_bytes = eredu_core::Observed::unavailable("settled shared parameter observation omits independently managed parameter owners");
                parameters.current_host_resident_bytes = eredu_core::Observed::unavailable("settled shared parameter observation omits independently managed parameter owners");
            }
            if mechanism.embedded.is_some() {
                parameters.current_device_parameter_conversion_bytes = eredu_core::Observed::unavailable(
                    "embedded conversions receive binding-specific credit after shared residency composition",
                );
            }
            profile.parameters = parameters;
            profile.available = native.available;
            profile.allocator_cache_limit = native.allocator_cache_limit;
            // The selection geometry is valid for this live owner; ordinary startup
            // profile callers may have deliberately removed workspace on advanced state.
            let (mut request, _) = loaded_generation_request(
                profile,
                eredu_core::InputTokenCount::text(1),
                0,
                eredu_core::PrefillChunkPolicy::Unchunked,
                ForecastExecutionContract {
                    full_pass_reason: None,
                    logits: LogitsWorkspace::EveryPosition,
                },
                options,
            )?;
            request.input = eredu_core::InputTokenCount::text(native.current_positions);
            request.max_output_tokens = Some(extra);
            request.forecast_output_tokens = extra;
            for pool in &mut request.domains {
                pool.retained_input = MemoryBytes::exact(0);
            }
            let plan = ContinuationMemoryPlan {
                current_positions: native.current_positions,
                additional_input_tokens: extra,
                current_state: allowance(native.current_state_bytes, "installed native state and backing-capacity allowance; not measured distinct residency"),
                peak_state: allowance(native.peak_state_bytes, "native continuation state/capacity envelope including speculative overshoot and interior peaks"),
            }.with_logical_state_bounds(&request)?;
            Ok((request, plan))
        };
        let (mut target, target_state) = model(target_profile.clone(), native.target)?;
        let (draft, draft_state, mut embedded, embedded_state) = match (
            &mechanism.draft,
            &mechanism.embedded,
            native.draft,
            native.embedded,
        ) {
            (Some(profile), None, Some(observation), None) => {
                let (draft, state) = model(profile.clone(), observation)?;
                (Some(draft), Some(state), None, None)
            }
            (None, Some(topology), None, Some(observation)) => {
                let prediction = EmbeddedPredictionMemoryPlan::from_topology(
                    topology,
                    &target,
                    &options.calibration,
                )?;
                let (_, _, prefix_features) =
                    prediction.costs(&target, target_state.current_positions, 0, false)?;
                let state = EmbeddedContinuationMemoryPlan {
                    layer_positions: observation.prediction.layer_positions,
                    additional_input_tokens: extra,
                    current_state: allowance(observation.prediction.current_state_bytes, "installed prediction state and native backing-capacity allowance"),
                    peak_state: allowance(observation.prediction.peak_state_bytes, "prediction native horizon capacity including speculative overshoot and interior peaks"),
                    retained_features: allowance(observation.retained_feature_bytes, "retained target capture descriptor/copy allowance").add(&prefix_features)?,
                };
                (None, None, Some(prediction), Some(state))
            }
            _ => {
                return Err(unsupported(
                    "settled observations do not match the selected speculative mechanism",
                ))
            }
        };
        let conversion_assumptions = if let Some(prediction) = &mut embedded {
            let conversions = target
                .domains
                .iter()
                .all(|domain| domain.resident_parameters.upper_bytes.is_some())
                .then_some(native.parameter_conversions.as_deref())
                .flatten();
            apply_embedded_parameter_conversion_credit(&mut target, prediction, conversions)?
        } else {
            vec![]
        };
        // At canonical boundaries no proposal distributions or pending completion
        // remain. RNG/sampler/semantic and history storage are observed explicitly.
        // Retain several complete host copies through speculative fork/replay.
        let host = if extra == 0 {
            request.continuation_host_bytes()
        } else {
            request.continuation_host_peak_bytes(extra)
        }
        .and_then(|n| {
            n.checked_mul(
                if scheduler.lookahead_blocks != 0 && scheduler.max_optimistic_branches != 0 {
                    9
                } else {
                    6
                },
            )
        })
        .and_then(|n| n.checked_add(self.trace.limits().total_bytes))
        .and_then(|n| n.checked_add(std::mem::size_of::<Self>() as u64));
        let host = if self.forecast_instrumented {
            None
        } else {
            host
        };
        let seed = allowance(
            native.seed_bytes,
            "currently retained assistant seed snapshot",
        );
        let seed = embedded_state
            .as_ref()
            .map_or(seed.clone(), |state| seed.maximum(&state.current_state));
        let continuation = SpeculativeContinuationMemoryPlan {
            additional_tokens,
            target: target_state,
            draft: draft_state,
            embedded: embedded_state,
            seed,
            host_retention: allowance(host, "sampler/RNG, canonical and forked semantic/history state plus one compact JSON trace allowance; charged conservatively in every pool because native sampling placement is unavailable; unknown custom growth or instrumentation retention stays unknown"),
            retained_snapshots: allowance(Some(self.snapshot_usage().retained_bytes), "live snapshot/branch reservations; conservative per-pool allowance, not resident credit"),
        };
        let speculative = SpeculativeMemoryPlan {
            embedded,
            draft,
            auxiliary_bytes_per_position: mechanism.auxiliary_bytes_per_position.clone(),
            sampling_bytes_per_vocabulary_entry: mechanism
                .sampling_bytes_per_vocabulary_entry
                .clone(),
            max_draft_tokens: width,
            scheduler,
            shared_allocator: mechanism.shared_allocator,
        };
        let mut continuation = continuation;
        if let (Some(prediction), Some(state)) = (&speculative.embedded, &mut continuation.embedded)
        {
            prediction.validate_continuation(&target, state)?;
        }
        let mut estimate =
            estimate_speculative_continuation_memory(&target, &speculative, &continuation)?;
        estimate.assumptions.extend(conversion_assumptions);
        Ok(SpeculativeContinuationForecast {
            estimate,
            request: target,
            speculative,
            continuation,
        })
    }
}
