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
        let policies = request.forecast_decoding_policies(self.scheduler.requests.options());
        let mut scheduler = request.forecast_scheduler_options(self.scheduler.requests.options());
        let mut width = request.forecast_max_draft_tokens();
        let mut draft_adapter = false;
        for (mode, options) in &policies {
            if let eredu_core::decoding::DecodingMode::SelfSpeculation(speculation) = mode {
                width = width.max(speculation.request().max_draft_tokens.get() as u64);
                draft_adapter |= speculation.request().draft_adapter;
            }
            scheduler.lookahead_blocks = scheduler.lookahead_blocks.max(options.lookahead_blocks);
            scheduler.max_optimistic_branches = scheduler
                .max_optimistic_branches
                .max(options.max_optimistic_branches);
            scheduler.max_in_flight_verifications = scheduler
                .max_in_flight_verifications
                .max(options.max_in_flight_verifications);
        }
        if width > mechanism.proposal_capacity {
            return Err(unsupported(
                "configured proposals exceed the selected mechanism capacity",
            ));
        }
        let refinement_active = policies
            .iter()
            .any(|(mode, _)| matches!(mode, eredu_core::decoding::DecodingMode::Diffusion { .. }));
        let mut refinement_plan = if refinement_active {
            Some(
                mechanism
                    .refinement
                    .as_ref()
                    .ok_or_else(|| unsupported("retained refinement geometry is unavailable"))?
                    .clone(),
            )
        } else {
            None
        };
        if let Some(refinement) = &mut refinement_plan {
            for (mode, _) in &policies {
                if let eredu_core::decoding::DecodingMode::Diffusion {
                    policy: Some(policy),
                } = mode
                {
                    refinement.stability_history = refinement
                        .stability_history
                        .max(policy.stability_history_length());
                }
            }
        }
        let refinement = refinement_plan.as_ref();
        let mut extra = speculative_continuation_positions(additional_tokens, width, scheduler)?
            .max(request.forecast_mode_settlement_positions());
        if additional_tokens > 0 {
            if let Some(refinement) = refinement {
                extra = extra.max(
                    additional_tokens
                        .checked_add(u64::from(refinement.canvas_positions.get()))
                        .ok_or(eredu_core::CapabilityError::ArithmeticOverflow {
                            operation: "refinement continuation canvas",
                        })?,
                );
            }
        }
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
            // Startup profiles retain immutable geometry, never live conversion
            // credit: trim/invalidation can replace equal-sized allocations.
            profile.parameter_conversions = native.parameter_conversions;
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
                    input_retention: crate::memory_forecast::PrefillInputRetention::TokenIds,
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
                query_tokens: 1,
                current_state: allowance(native.current_state_bytes, "installed native state and backing-capacity allowance; not measured distinct residency"),
                peak_state: allowance(native.peak_state_bytes, "native continuation state/capacity envelope including speculative overshoot and interior peaks"),
            }.with_logical_state_bounds(&request)?;
            Ok((request, plan))
        };
        let (mut target, target_state) = model(target_profile.clone(), native.target)?;
        let mut conditioned_plan = None;
        let (draft, draft_state, mut embedded, embedded_state) = if let Some(profile) =
            &mechanism.conditioned
        {
            let observation = native.draft.ok_or_else(|| {
                unsupported("installed conditioned context observations are unavailable")
            })?;
            if native.embedded.is_some()
                || observation.current_positions != target_state.current_positions
            {
                return Err(unsupported(
                    "conditioned context observations do not cover the installed target frontier",
                ));
            }
            let current_positions = observation.current_positions;
            let current_state = allowance(
                observation.current_state_bytes,
                "installed projected conditioned context and descriptor allowance",
            );
            let peak_state = allowance(observation.peak_state_bytes,
                "future native conditioned buffer capacity is unavailable; logical context growth is accounted separately");
            let refreshed = ConditionedAssistantMemoryProfile {
                topology: profile.topology.clone(),
                parameters: eredu_core::speculative::SpeculativeParameterMemoryObservation {
                    parameters: observation.parameters,
                    parameter_conversions: observation.parameter_conversions,
                    available: observation.available,
                    allocator_cache_limit: observation.allocator_cache_limit,
                },
                host_execution: profile.host_execution,
                residency: profile.residency.clone(),
                parameter_bounds: profile.parameter_bounds.clone(),
                parameter_loading: profile.parameter_loading.clone(),
                native_workspace: profile.native_workspace.clone(),
            };
            let (draft, selected) = refreshed.into_request(
                eredu_core::InputTokenCount::text(current_positions),
                extra,
                options,
            )?;
            let state = ContinuationMemoryPlan {
                current_positions,
                additional_input_tokens: extra,
                query_tokens: selected.topology.body.block_positions,
                current_state,
                peak_state,
            };
            conditioned_plan = Some(selected);
            (Some(draft), Some(state), None, None)
        } else {
            match (
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
                (None, None, None, None) => (None, None, None, None),
                _ => {
                    return Err(unsupported(
                        "settled observations do not match the selected speculative mechanism",
                    ))
                }
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
        )
        .add(&allowance(
            request
                .retained_next_scores_estimate(self.scheduler.executor)
                .map(|estimate| estimate.retained_bytes),
            "settled next-target score row retained across the mode boundary",
        ))?
        .add(&allowance(
            request
                .retained_refinement_estimate(self.scheduler.executor)
                .map(|estimate| estimate.retained_bytes),
            "tentative position rows and complete refinement commitment cursor",
        ))?;
        let seed = embedded_state
            .as_ref()
            .map_or(seed.clone(), |state| seed.maximum(&state.current_state));
        let continuation = SpeculativeContinuationMemoryPlan {
            additional_tokens,
            settlement_input_tokens: request.forecast_mode_settlement_positions(),
            target: target_state,
            draft: draft_state,
            embedded: embedded_state,
            seed,
            host_retention: allowance(host, "sampler/RNG, canonical and forked semantic/history state plus one compact JSON trace allowance; charged conservatively in every pool because native sampling placement is unavailable; unknown custom growth or instrumentation retention stays unknown"),
            retained_snapshots: allowance(Some(self.snapshot_usage().retained_bytes), "live snapshot/branch reservations; conservative per-pool allowance, not resident credit"),
        };
        let mut shared_backbone = mechanism.shared_backbone.clone();
        if !draft_adapter {
            if let Some(shared) = &mut shared_backbone {
                shared.adapter = None;
            }
        }
        let speculative = SpeculativeMemoryPlan {
            conditioned: conditioned_plan,
            shared_backbone,
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
        if let Some(refinement) = refinement {
            let mut refinement_request = target.clone();
            refinement_request.forecast_output_tokens = additional_tokens;
            refinement_request.max_output_tokens = Some(additional_tokens);
            if request.has_tentative_refinement()
                && request.forecast_mode_settlement_positions() > 0
            {
                refinement_request.forecast_output_tokens =
                    refinement_request.forecast_output_tokens.max(1);
                refinement_request.max_output_tokens = Some(additional_tokens.max(1));
            }

            for pool in &mut refinement_request.domains {
                pool.retained_input = pool
                    .retained_input
                    .add(&continuation.seed)?
                    .add(&continuation.host_retention)?
                    .add(&continuation.retained_snapshots)?;
            }
            let refined = estimate_refinement_continuation_memory(
                &refinement_request,
                refinement,
                &continuation.target,
            )?;
            crate::memory_forecast::merge_decoding_phase_estimate(
                &mut estimate,
                refined,
                &refinement_request,
            )?;
        }
        estimate.assumptions.extend(conversion_assumptions);
        if policies.len() > 1 {
            estimate.assumptions.push("The active and admitted pending modes share parameter residency and canonical state. Proposal width, lookahead and draft adapter workspace cover both policies through the settled handoff.".into());
        }
        Ok(SpeculativeContinuationForecast {
            decoding: Some(request.decoding_state()),
            estimate,
            request: target,
            speculative,
            continuation,
        })
    }
}
