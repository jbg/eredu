//! Separate parameter owners with windowed target context and fixed proposal blocks.
use super::*;

/// Decoder/context geometry declared by the architecture, independent of storage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConditionedPredictionBody {
    /// Residual and encoded-context width.
    pub hidden: u64,
    /// Ordered target publications concatenated for context projection.
    pub captures: u64,
    /// Independently projected context K/V banks.
    pub layers: u64,
    /// Query heads.
    pub query_heads: u64,
    /// Channels per attention head.
    pub head_width: u64,
    /// Total channels in each key or value bank.
    pub key_value_width: u64,
    /// Gated feed-forward width.
    pub intermediate: u64,
    /// Compacted committed context positions.
    pub context_window: u64,
    /// Full trained invocation, even when fewer proposals are requested.
    pub block_positions: u64,
    /// Dynamic convolution confined to the proposal block, when selected.
    pub convolution: Option<ConditionedBlockConvolution>,
}

/// Two dynamically filtered wrappers per decoder layer, without persistent carry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConditionedBlockConvolution {
    /// Causal taps per wrapper.
    pub taps: u64,
    /// Channels sharing each dynamic coefficient.
    pub group_size: u64,
}

/// Scoring equations whose allocation geometry differs from ordinary target logits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ConditionedPredictionScoring {
    /// Every proposal state is projected by the existing target head.
    TargetProjection,
    /// Dense draft scores and a predecessor transition, scattered by an I64 map.
    MappedVocabulary {
        /// Complete draft and target vocabulary, proven equal at preparation.
        vocabulary: u64,
        /// Predecessor/successor transition width.
        rank: u64,
    },
    /// Target top-k candidates, predecessor/successor gathers and dense scatter.
    CandidateSelector {
        /// Complete target vocabulary.
        vocabulary: u64,
        /// Conditional codebook width.
        rank: u64,
        /// Selected candidates per proposal position.
        candidates: u64,
    },
}

/// Exact logical context/body/scoring geometry, without native allocation claims.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConditionedPredictionTopology {
    /// Windowed context and fixed-block invocation.
    pub body: ConditionedPredictionBody,
    /// Complete selected scoring equation.
    pub scoring: ConditionedPredictionScoring,
}

/// Native parameter and mechanism observations for a separately owned assistant.
pub struct ConditionedAssistantMemoryProfile {
    /// Architecture-owned equations retained through construction.
    pub topology: ConditionedPredictionTopology,
    /// Authoritative parameter/allocator observations; conversion payload is included once.
    pub parameters: eredu_core::speculative::SpeculativeParameterMemoryObservation,
    /// Whether native execution consumes the host pool.
    pub host_execution: bool,
    /// Selected immutable parameter policy.
    pub residency: crate::LayerWeightResidency,
    /// Complete managed-pool bounds from retained exact units and native host
    /// capacity facts. Missing bounds never imply two whole-checkpoint copies.
    pub parameter_bounds: Option<ParameterResidencyMemoryBounds>,
    /// Retained source and outstanding transfer payloads, independent of managed
    /// cache occupancy. Missing coverage is unknown, including for resident owners.
    pub parameter_loading: Option<ParameterLoadingMemoryProfile>,
    /// Unmodeled native operator/dequantization scratch and unreused parameter
    /// promotions, including possible new retained conversions; never inferred as zero.
    pub native_workspace: MemoryBytes,
}

/// Serialized invocation plan; its parameters belong to the paired draft request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConditionedAssistantMemoryPlan {
    /// Exact selected equations.
    pub topology: ConditionedPredictionTopology,
    /// Physical pool where context, proposal and scoring operations execute.
    pub execution_domain: MemoryDomain,
    /// Mechanism-specific scratch beyond the declared logical tensor envelope.
    pub native_workspace: MemoryBytes,
}

impl ConditionedPredictionTopology {
    /// Rejects malformed geometry rather than wrapping counts or dropping components.
    pub fn validate(&self) -> Result<(), CapabilityError> {
        let b = &self.body;
        if [
            b.hidden,
            b.captures,
            b.layers,
            b.query_heads,
            b.head_width,
            b.key_value_width,
            b.intermediate,
            b.context_window,
            b.block_positions,
        ]
        .contains(&0)
        {
            return Err(invalid("conditioned assistant dimensions must be positive"));
        }
        if let Some(c) = &b.convolution {
            if c.taps == 0 || c.group_size == 0 || b.hidden % c.group_size != 0 {
                return Err(invalid("invalid conditioned block convolution"));
            }
        }
        match self.scoring {
            ConditionedPredictionScoring::TargetProjection => {}
            ConditionedPredictionScoring::MappedVocabulary { vocabulary, rank }
                if vocabulary > 0 && rank > 0 => {}
            ConditionedPredictionScoring::CandidateSelector {
                vocabulary,
                rank,
                candidates,
            } if rank > 0 && candidates > 0 && candidates <= vocabulary => {}
            _ => return Err(invalid("invalid conditioned scoring geometry")),
        }
        Ok(())
    }
}

fn invalid(detail: impl Into<String>) -> CapabilityError {
    CapabilityError::InvalidConfiguration {
        field: "conditioned assistant memory",
        detail: detail.into(),
    }
}

impl ConditionedAssistantMemoryProfile {
    /// Composes immutable parameter pools without fabricating an autoregressive state layout.
    pub fn into_request(
        self,
        input: eredu_core::InputTokenCount,
        output: u64,
        options: &GenerationForecastOptions,
    ) -> Result<(GenerationMemoryRequest, ConditionedAssistantMemoryPlan), GenerationForecastError>
    {
        self.topology.validate()?;
        self.native_workspace.validate()?;
        if let Some(loading) = &self.parameter_loading {
            loading.validate()?;
        }
        let mut parameters = self.parameters.parameters;
        if let Some(bounds) = &self.parameter_bounds {
            if bounds.residency() != self.residency {
                return Err(invalid(
                    "parameter pool bounds belong to a different residency policy",
                )
                .into());
            }
            if parameters
                .current_host_resident_bytes
                .value()
                .is_some_and(|n| *n > bounds.host_capacity_bytes())
            {
                return Err(
                    invalid("observed host parameters exceed their managed pool bound").into(),
                );
            }
            if let (Some(device), Some(conversions)) = (
                parameters.current_device_resident_bytes.value(),
                parameters.current_device_parameter_conversion_bytes.value(),
            ) {
                if *device > add(bounds.device_payload_bytes(), *conversions)? {
                    return Err(invalid(
                        "observed device parameters exceed their managed pool bound",
                    )
                    .into());
                }
            }
        }
        let unified = parameters.physical_semantics == eredu_core::PhysicalMemorySemantics::Unified;
        if self.host_execution {
            parameters.physical_semantics = eredu_core::PhysicalMemorySemantics::Unified;
        }
        let separate =
            parameters.physical_semantics == eredu_core::PhysicalMemorySemantics::SeparateTiers;
        let execution_domain = if self.host_execution && !unified {
            MemoryDomain::Host
        } else if separate {
            MemoryDomain::Device("default".into())
        } else {
            MemoryDomain::Unified
        };
        let overhead = options.backend_overhead.clone().unwrap_or_else(|| {
            match (
                self.parameters.allocator_cache_limit.value(),
                parameters.backend_allocator_cache_bytes.value(),
            ) {
                (Some(&limit), Some(&retained)) => {
                    options.calibration.allocator_overhead(limit, retained)
                }
                _ => MemoryBytes::unknown("assistant allocator policy or retention unavailable"),
            }
        });
        let mut domains = Vec::new();
        for (mut domain, mut resident) in static_parameter_placement(&parameters, None)? {
            let parameter_domain = domain.clone();
            if self.host_execution && !unified {
                domain = MemoryDomain::Host;
            }
            let baseline = resident.lower_bytes;
            if !self.residency.is_fully_resident() {
                resident.upper_bytes = match &self.parameter_bounds {
                    Some(bounds) => {
                        let host = bounds.host_capacity_bytes();
                        let device = parameters
                            .current_device_parameter_conversion_bytes
                            .value()
                            .map(|n| add(bounds.device_payload_bytes(), *n))
                            .transpose()?;
                        match parameter_domain {
                            MemoryDomain::Host => Some(host),
                            MemoryDomain::Device(_) => device,
                            MemoryDomain::Unified => device.map(|n| add(host, n)).transpose()?,
                        }
                    }
                    None => None,
                };
                resident.detail = if self.parameter_bounds.is_some() {
                    "selected managed host capacity and device payload limits; current parameter conversions included once; detached transfers and scratch excluded"
                } else {
                    "assistant managed parameter pool bounds unavailable"
                }.into();
                // A smaller bound is stale or belongs to a different owner;
                // increasing it silently would hide an invalid observation.
                resident.validate()?;
            }
            // Unpublished reservations have no allocation attribution, even when
            // this parameter-only request has no ordinary decoder invocation.
            if parameters
                .parameter_conversion_retention
                .value()
                .is_none_or(|groups| {
                    groups.iter().any(|g| {
                        g.usage
                            .value()
                            .is_none_or(|u| u.reserved_payload_bytes != 0)
                    })
                })
            {
                resident.upper_bytes = None;
            }
            let executes = domain == execution_domain;
            let mut budget = if executes {
                options.budget.clone()
            } else {
                options.host_budget.clone()
            };
            if budget.available_bytes.is_none() && (!separate || !executes) {
                budget.available_bytes = self
                    .parameters
                    .available
                    .available_memory_bytes
                    .value()
                    .copied();
            }
            domains.push(DomainMemoryPlan {
                domain,
                resident_parameters: resident,
                already_resident_bytes: baseline,
                retained_input: MemoryBytes::exact(0),
                staging: match &self.parameter_loading {
                    Some(loading) => loading.staging(
                        !matches!(parameter_domain, MemoryDomain::Device(_)),
                        executes,
                    )?,
                    None => MemoryBytes::unknown(
                        "assistant retained source and materialization/transfer payload coverage unavailable",
                    ),
                },
                backend_overhead: if executes {
                    overhead.clone()
                } else {
                    MemoryBytes::exact(0)
                },
                loading_peak: MemoryBytes::exact(0),
                executions: vec![],
                budget,
            });
        }
        Ok((
            GenerationMemoryRequest {
                parameter_conversion_retention: parameters
                    .parameter_conversion_retention
                    .value()
                    .map(|reports| {
                        ConversionRetentionMemoryPlan::observe(
                            reports,
                            self.parameters.parameter_conversions,
                            &[],
                        )
                    })
                    .transpose()?,
                input,
                max_output_tokens: Some(output),
                forecast_output_tokens: output,
                batch_size: 1,
                prefill_chunk_tokens: input.model_positions.max(1),
                scalar_bytes: NonZeroU8::new(4).unwrap(),
                domains,
            },
            ConditionedAssistantMemoryPlan {
                topology: self.topology,
                execution_domain,
                native_workspace: self.native_workspace,
            },
        ))
    }
}

impl ConditionedAssistantMemoryPlan {
    /// Pending target suffix views can retain complete publication backing.
    /// The same envelope also bounds a transferred copy on the assistant pool.
    pub(super) fn captured_features(&self, positions: u64) -> Result<MemoryBytes, CapabilityError> {
        let b = &self.topology.body;
        Ok(MemoryBytes::estimated(
            0,
            mul(mul(mul(positions, b.hidden)?, b.captures)?, 4)?,
            "captured target publications, including full-prefix backing of suffix views",
        ))
    }

    pub(super) fn validate(
        &self,
        draft: Option<&GenerationMemoryRequest>,
        width: u64,
    ) -> Result<(), CapabilityError> {
        self.topology.validate()?;
        self.native_workspace.validate()?;
        let Some(draft) = draft else {
            return Err(invalid(
                "conditioned invocation requires its separate parameter owner",
            ));
        };
        if draft.domains.iter().any(|d| !d.executions.is_empty())
            || !draft
                .domains
                .iter()
                .any(|d| d.domain == self.execution_domain)
            || width > self.topology.body.block_positions
        {
            return Err(invalid("parameter-only owner, execution pool or proposal capacity differs from conditioned invocation"));
        }
        Ok(())
    }

    /// Logical FP32 tensor envelope. Views may alias; all lower bounds remain zero.
    /// This is separate from native scratch and allocator/driver calibration.
    pub(super) fn costs(
        &self,
        positions: u64,
        context_update_rows: u64,
        target_vocabulary: Option<u64>,
    ) -> Result<(MemoryBytes, MemoryBytes, MemoryBytes), CapabilityError> {
        let b = &self.topology.body;
        let retained = positions.min(b.context_window);
        let kv = mul(mul(2, b.layers)?, b.key_value_width)?;
        let state = mul(mul(retained, add(b.hidden, kv)?)?, 4)?;
        let block = b.block_positions;
        let query_width = mul(b.query_heads, b.head_width)?;
        let context = mul(context_update_rows, add(mul(4, b.hidden)?, kv)?)?;
        let vectors = mul(
            mul(block, b.layers)?,
            add(
                add(mul(12, b.hidden)?, mul(4, query_width)?)?,
                add(mul(6, b.key_value_width)?, mul(4, b.intermediate)?)?,
            )?,
        )?;
        let attention = mul(
            mul(mul(b.layers, b.query_heads)?, block)?,
            mul(3, add(retained, block)?)?,
        )?;
        let conv = match &b.convolution {
            None => 0,
            Some(c) => mul(
                mul(mul(2, b.layers)?, block)?,
                add(
                    mul(add(c.taps, 4)?, b.hidden)?,
                    mul(2, mul(b.hidden / c.group_size, c.taps)?)?,
                )?,
            )?,
        };
        let mut scoring = match self.topology.scoring {
            ConditionedPredictionScoring::TargetProjection => match target_vocabulary {
                Some(v) => MemoryBytes::estimated(
                    0,
                    mul(mul(block, v)?, 4)?,
                    "borrowed target projection score block",
                ),
                None => MemoryBytes::unknown("borrowed target vocabulary geometry unavailable"),
            },
            ConditionedPredictionScoring::MappedVocabulary { vocabulary, rank } => {
                // Raw block, transition/transform vectors, F32 target scatter,
                // confidence concatenation, I64 map and I32 draft IDs.
                let floats = add(
                    mul(add(block, 8)?, vocabulary)?,
                    add(b.hidden, mul(2, rank)?)?,
                )?;
                MemoryBytes::estimated(
                    0,
                    add(mul(floats, 4)?, mul(vocabulary, 12)?)?,
                    "dense mapped scores, transition, confidence and signed vocabulary map",
                )
            }
            ConditionedPredictionScoring::CandidateSelector {
                vocabulary,
                rank,
                candidates,
            } => {
                let scores = mul(add(mul(3, block)?, 2)?, vocabulary)?;
                let selected = add(mul(mul(block, candidates)?, add(rank, 4)?)?, mul(3, rank)?)?;
                MemoryBytes::estimated(0, mul(add(scores, selected)?, 4)?, "target raw projection, top-k indices/unary scores, selected codebooks, conditional rank and dense scatter")
            }
        };
        scoring = scoring.add(&MemoryBytes::estimated(0,
            mul(add(add(context, vectors)?, add(attention, conv)?)?, 4)?,
            "conditioned context projection, full-block dense activations, attention scores/masks and block-local convolution; simultaneous logical FP32 envelope"))?;
        Ok((
            MemoryBytes::estimated(
                0,
                state,
                "compacted encoded context and per-layer K/V; no proposal-block convolution carry",
            ),
            scoring.add(&self.native_workspace)?,
            self.captured_features(positions)?,
        ))
    }
}
