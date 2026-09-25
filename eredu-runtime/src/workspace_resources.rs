//! Calibrated, family-independent workspace planning over ordinary module topology.
//!
//! This is a planning envelope, not an exact native allocation inventory. Phase-4
//! logical geometry determines the reusable mechanism extents; explicitly labeled
//! calibration supplies scratch and lazy-retention envelopes. Opaque native facts
//! are never silently reclassified as exact observations.

mod legacy;

use std::collections::{BTreeMap, BTreeSet};

use eredu_core::{resources::*, CapabilityError, ObservationKind, Observed};
use eredu_nn::{
    mechanism_memory::{LogicalValueKind, MechanismInvocation},
    AttentionArithmetic, TensorElementType,
};

use crate::{execution_topology::*, memory_estimation::*, resource_lifetimes::*};

fn invalid(detail: impl Into<String>) -> CapabilityError {
    CapabilityError::InvalidConfiguration {
        field: "execution_topology",
        detail: detail.into(),
    }
}
fn add(a: u64, b: u64) -> Result<u64, CapabilityError> {
    a.checked_add(b).ok_or(CapabilityError::ArithmeticOverflow {
        operation: "mechanism workspace sum",
    })
}
fn mul(a: u64, b: u64) -> Result<u64, CapabilityError> {
    a.checked_mul(b).ok_or(CapabilityError::ArithmeticOverflow {
        operation: "mechanism workspace product",
    })
}
fn product(values: &[u64]) -> Result<u64, CapabilityError> {
    values.iter().try_fold(1, |a, b| mul(a, *b))
}
fn id(key: impl Into<String>) -> ResourceIdentity {
    ResourceIdentity {
        scope: "target-workspace".into(),
        key: key.into(),
    }
}
fn element(bytes: u8) -> Result<TensorElementType, CapabilityError> {
    match bytes {
        1 => Ok(TensorElementType::U8),
        2 => Ok(TensorElementType::Bf16),
        4 => Ok(TensorElementType::F32),
        8 => Ok(TensorElementType::F64),
        _ => Err(invalid("unsupported activation scalar width")),
    }
}
fn output_bytes(invocation: &MechanismInvocation) -> Result<u64, CapabilityError> {
    invocation
        .logical_values()
        .map_err(|e| invalid(e.to_string()))?
        .iter()
        .filter(|value| value.kind == LogicalValueKind::Output)
        .try_fold(0, |total, value| {
            add(
                total,
                value.logical_bytes().map_err(|e| invalid(e.to_string()))?,
            )
        })
}
fn projection(p: &ProjectionTopology, rows: u64, scalar: TensorElementType) -> MechanismInvocation {
    MechanismInvocation::Projection {
        rows,
        input: p.input,
        output: p.output,
        format: p.format,
        element: scalar,
        weight_element: None,
        bias: p.bias,
    }
}

struct Schedule {
    plan: ResourceLifetimePlan,
    next: u64,
}
impl Schedule {
    fn new(missing: Vec<String>) -> Self {
        Self {
            plan: ResourceLifetimePlan {
                coverage: if missing.is_empty() {
                    ResourceCoverage::Complete
                } else {
                    ResourceCoverage::Partial { reasons: missing }
                },
                events: vec![],
            },
            next: 0,
        }
    }
    // Global conversion/cache/calibration allowances overlap every phase of the
    // lazy schedule, including peaks preceding a synchronous attention boundary.
    fn global_allocation(
        &mut self,
        name: &str,
        bytes: MemoryBytes,
        hold: ResourceLifetime,
    ) -> Result<(), CapabilityError> {
        self.allocation(name, bytes, hold)?;
        let event = self.plan.events.pop().expect("allocation appends an event");
        self.plan.events.insert(0, event);
        Ok(())
    }
    fn allocation(
        &mut self,
        name: &str,
        bytes: MemoryBytes,
        hold: ResourceLifetime,
    ) -> Result<(), CapabilityError> {
        bytes.validate()?;
        let identity = id(format!("{}:{name}", self.next));
        self.next += 1;
        let bounds = ResourceByteBounds {
            lower_bytes: bytes.lower_bytes,
            upper_bytes: bytes.upper_bytes,
            kind: ObservationKind::Estimated,
            detail: bytes.detail,
        };
        let extent = ResourceExtent {
            payload: bounds.clone(),
            capacity: bounds,
        };
        self.plan.events.push(ResourceLifetimeEvent::Acquire(
            ResourceLifetimeDescription {
                resources: ResourceDescription {
                    schema_version: RESOURCE_DESCRIPTION_SCHEMA_VERSION,
                    scope: identity.clone(),
                    context: ResourceContext::default(),
                    horizon: ResourceContext::default(),
                    coverage: ResourceCoverage::Complete,
                    allocations: vec![ResourceAllocation {
                        identity: identity.clone(),
                        uses: vec![ResourceUse {
                            owner: identity.clone(),
                            role: ResourceRole::Workspace,
                        }],
                        placement: Observed::Available {
                            value: id("execution-pool"),
                            kind: ObservationKind::Exact,
                            source: "one execution's already selected physical domain".into(),
                        },
                        size: ResourceSize::Fixed { extent },
                    }],
                },
                lifetimes: BTreeMap::from([(identity, vec![hold])]),
                // A layer's internal buffers do not all coexist. Only an individual
                // output is a defensible minimum; graph retention is an upper envelope.
                live_at_acquire: false,
            },
        ));
        Ok(())
    }
}

/// Describe a selected text invocation's calibrated workspace lifetimes.
///
/// Ordinary topology is authoritative. Older aggregate-only reports are lowered
/// through a compatibility envelope preserving their historical bounds without
/// inferring missing module identities, formats or invocation order.
///
/// Parameters and installed state are owned by separate resource producers. Cache
/// replacement and uncached conversions are additional allocations here. No native
/// tensor, stream, state transition or observation budget is touched.
///
/// The returned identities are local to this one execution and synthetic pool.
/// Compose it independently; callers combining multiple returned plans must first
/// rebind their identities and placements to authoritative execution namespaces.
pub fn describe_text_workspace(
    execution: &ExecutionMemoryPlan,
    request: &GenerationMemoryRequest,
    positions: u64,
    query: u64,
    persistent: u64,
) -> Result<ResourceLifetimePlan, CapabilityError> {
    // A settled continuation start observes installed state only. It has no
    // invocation, so it allocates neither logits nor cache-update workspace.
    if query == 0 {
        return Ok(ResourceLifetimePlan {
            coverage: ResourceCoverage::Complete,
            events: vec![],
        });
    }
    if execution.execution_topology.is_none() {
        return legacy::describe(execution, request, positions, query, persistent);
    }
    let topology = execution
        .execution_topology
        .as_ref()
        .ok_or_else(|| invalid("module topology unavailable"))?;
    if topology.hidden_size == 0
        || topology.vocabulary_size == 0
        || topology.output_invocations == 0
        || topology.layers.is_empty()
        || request.batch_size == 0
        || positions < query
    {
        return Err(invalid(
            "positive module dimensions and a key frontier covering the queries are required",
        ));
    }
    if topology.output.input != topology.hidden_size
        || topology.output.output != topology.vocabulary_size
    {
        return Err(invalid(
            "output projection disagrees with selected residual/vocabulary geometry",
        ));
    }
    validate_topology(topology)?;
    let copies = execution.workspace_overlap.upper_live_copies;
    if copies == Some(0) {
        return Err(invalid("workspace live copies must be positive"));
    }
    let layers = topology.layers.len() as u64;
    let scalar = element(request.scalar_bytes.get())?;
    let scalar_bytes = u64::from(request.scalar_bytes.get());
    let widened = topology.selected_parameter_promotion_bytes.is_some()
        || topology.layers.iter().any(|layer| {
            matches!(
                layer.mixer,
                TokenMixerTopology::GatedConvolution { .. }
                    | TokenMixerTopology::Attention {
                        input_scores: true,
                        ..
                    }
                    | TokenMixerTopology::Attention { softcap: true, .. }
            )
        });
    let upper_scalar = if widened {
        scalar_bytes.max(4)
    } else {
        scalar_bytes
    };
    let rows = mul(request.batch_size, query)?;
    let mut schedule = Schedule::new(topology.missing.clone());
    let global = ResourceLifetime::Evaluation(id("request-evaluation"));
    let embedding = product(&[rows, topology.hidden_size, scalar_bytes])?;
    let embedding_upper = product(&[rows, topology.hidden_size, upper_scalar])?;
    schedule.allocation(
        "embedding-output",
        MemoryBytes::estimated(
            embedding,
            embedding_upper,
            "selected token embedding output before the first residual block",
        ),
        global.clone(),
    )?;
    schedule.allocation(
        "embedding-transform",
        MemoryBytes::estimated(
            0,
            embedding_upper,
            "one optional embedding scaling or representation-transform output",
        ),
        global.clone(),
    )?;
    let mut max_layer = 0u64;
    let mut attention_scratch = MemoryBytes::exact(0);
    let mut transient_pending = BTreeSet::new();
    for (index, layer) in topology.layers.iter().enumerate() {
        let hold = ResourceLifetime::Evaluation(id(format!("layer-{index}")));
        let transient = ResourceLifetime::Evaluation(id(format!("transient-layer-{index}")));
        if let Some(copies) = copies {
            if index as u64 >= copies {
                schedule
                    .plan
                    .events
                    .push(ResourceLifetimeEvent::Evaluate(id(format!(
                        "layer-{}",
                        index as u64 - copies
                    ))));
            }
        }
        if let Some(copies) = copies {
            if index as u64 >= copies && transient_pending.remove(&(index - copies as usize)) {
                schedule
                    .plan
                    .events
                    .push(ResourceLifetimeEvent::Evaluate(id(format!(
                        "transient-layer-{}",
                        index as u64 - copies
                    ))));
            }
        }
        let layer_start = schedule.plan.events.len();
        let charge = |schedule: &mut Schedule,
                      name: &str,
                      lower: u64,
                      upper: u64|
         -> Result<(), CapabilityError> {
            schedule.allocation(name, MemoryBytes::estimated(lower, upper, "reusable module outputs and calibrated native intermediates; lazy graph retention envelope"), hold.clone())
        };
        // Norm and residual outputs come from the construction's actual invocation
        // count, not a family-specific aggregate hidden-width formula.
        let hidden_outputs = add(layer.normalization_count, 2)?;
        charge(
            &mut schedule,
            "normalization-and-residual",
            0,
            product(&[rows, topology.hidden_size, hidden_outputs, upper_scalar])?,
        )?;
        for projection_spec in &layer.input_projections {
            charge(
                &mut schedule,
                "input-fusion-projection",
                output_bytes(&projection(projection_spec, rows, scalar))?,
                product(&[
                    rows,
                    add(projection_spec.input, projection_spec.output)?,
                    upper_scalar,
                ])?,
            )?;
        }
        let projections = match &layer.mixer {
            TokenMixerTopology::Attention { projections, .. }
            | TokenMixerTopology::GatedConvolution { projections, .. } => projections.as_slice(),
            TokenMixerTopology::Unknown { .. } => &[],
        };
        for p in projections {
            let payload = output_bytes(&projection(p, rows, scalar))?;
            charge(
                &mut schedule,
                "mixer-projection",
                payload,
                product(&[rows, p.output, upper_scalar])?,
            )?;
        }
        match &layer.mixer {
            TokenMixerTopology::Unknown { reason } => {
                schedule.allocation("token-mixer", MemoryBytes::unknown(reason), hold.clone())?;
            }
            TokenMixerTopology::Attention {
                query_heads,
                kv_heads,
                key_width,
                value_width,
                input_scores,
                softcap,
                sinks,
                query_key_normalization,
                rotary,
                output_gate,
                ..
            } => {
                if *output_gate {
                    charge(
                        &mut schedule,
                        "attention-output-gate",
                        0,
                        product(&[rows, *query_heads, *value_width, upper_scalar, 2])?,
                    )?;
                }
                let invocation = MechanismInvocation::Attention {
                    batch: request.batch_size,
                    query_heads: *query_heads,
                    kv_heads: *kv_heads,
                    queries: query,
                    keys: positions,
                    key_width: *key_width,
                    value_width: *value_width,
                    element: scalar,
                    arithmetic: if *input_scores {
                        AttentionArithmetic::InputScores
                    } else {
                        AttentionArithmetic::Fused
                    },
                    softcap: *softcap,
                    sinks: *sinks,
                };
                let payload = output_bytes(&invocation)?;
                charge(
                    &mut schedule,
                    "attention-output",
                    payload,
                    product(&[rows, *query_heads, *value_width, upper_scalar])?,
                )?;
                let qk_width = mul(add(*query_heads, *kv_heads)?, *key_width)?;
                if *query_key_normalization {
                    charge(
                        &mut schedule,
                        "query-key-normalization",
                        0,
                        product(&[rows, qk_width, upper_scalar])?,
                    )?;
                }
                if *rotary {
                    charge(
                        &mut schedule,
                        "rotary-outputs",
                        0,
                        product(&[rows, qk_width, upper_scalar])?,
                    )?;
                }
                if *input_scores || *softcap {
                    let bytes = explicit_attention(
                        execution.input_score_attention_mechanism,
                        ExplicitAttentionGeometry {
                            batch: request.batch_size,
                            heads: *query_heads,
                            expanded_width: mul(*query_heads, (*key_width).max(*value_width))?,
                            queries: query,
                            keys: positions,
                            arithmetic: if *input_scores {
                                AttentionArithmetic::InputScores
                            } else {
                                AttentionArithmetic::Fused
                            },
                        },
                        if topology.selected_parameter_promotion_bytes.is_some() || !*input_scores {
                            upper_scalar
                        } else {
                            scalar_bytes
                        },
                    )?;
                    let completes = *input_scores
                        && input_score_evaluates_dependencies(
                            execution.input_score_attention_mechanism,
                            query,
                            positions,
                        )?;
                    if completes {
                        // Finished tile outputs survive concatenation. Keep their
                        // full allowance with other owner-held module outputs;
                        // only invocation-local score/layout storage is released.
                        let full = execution
                            .input_score_attention_mechanism
                            .and_then(|facts| facts.full_key_tiles)
                            .unwrap();
                        let retained = product(&[
                            rows,
                            *query_heads,
                            (*key_width).max(*value_width),
                            if topology.selected_parameter_promotion_bytes.is_some() {
                                upper_scalar
                            } else {
                                scalar_bytes
                            },
                            full.retained_output_copies,
                        ])?;
                        schedule.allocation(
                            "completed-attention-outputs",
                            MemoryBytes::estimated(
                                0,
                                retained,
                                "completed tile outputs and concatenation remain owner-held",
                            ),
                            hold.clone(),
                        )?;
                        let invocation = id(format!("attention-evaluation-{index}"));
                        schedule.allocation(
                            "explicit-attention",
                            MemoryBytes {
                                upper_bytes: bytes.upper_bytes.map(|upper| upper - retained),
                                ..bytes
                            },
                            ResourceLifetime::Evaluation(invocation.clone()),
                        )?;
                        // Acquire before completion includes the first batch's
                        // overlap with all preceding unevaluated dependencies.
                        schedule
                            .plan
                            .events
                            .push(ResourceLifetimeEvent::Evaluate(invocation));
                        for prior in std::mem::take(&mut transient_pending) {
                            schedule
                                .plan
                                .events
                                .push(ResourceLifetimeEvent::Evaluate(id(format!(
                                    "transient-layer-{prior}"
                                ))));
                        }
                    } else {
                        schedule.allocation("explicit-attention", bytes, hold.clone())?;
                    }
                } else {
                    let scratch = attention_scratch_bytes(
                        &execution.attention,
                        request.batch_size,
                        *query_heads,
                        query,
                        positions,
                    )?;
                    attention_scratch = attention_scratch.maximum(&scratch);
                }
            }
            TokenMixerTopology::GatedConvolution {
                channels, kernel, ..
            } => {
                let invocation = MechanismInvocation::Convolution {
                    batch: request.batch_size,
                    tokens: query,
                    channels: *channels,
                    kernel: *kernel,
                    element: scalar,
                };
                let payload = output_bytes(&invocation)?;
                charge(
                    &mut schedule,
                    "convolution-output",
                    payload,
                    product(&[rows, *channels, upper_scalar])?,
                )?;
                // Gated inputs and padded/contiguous backing may be retained by
                // history views. Keep those owners; only unfolded convolution
                // scratch and the contiguous kernel use the transient frontier.
                let intermediates = convolution_intermediates(
                    request.batch_size,
                    query,
                    *channels,
                    *kernel,
                    upper_scalar,
                )?;
                let scratch = product(&[
                    request.batch_size,
                    *channels,
                    add(mul(query, *kernel)?, *kernel)?,
                    upper_scalar,
                ])?;
                charge(
                    &mut schedule,
                    "convolution-retained-inputs",
                    0,
                    intermediates - scratch,
                )?;
                schedule.allocation(
                    "convolution-scratch",
                    MemoryBytes::estimated(
                        0,
                        scratch,
                        "unfolded convolution and contiguous kernel scratch",
                    ),
                    transient.clone(),
                )?;
                transient_pending.insert(index);
            }
        }
        match &layer.feed_forward {
            FeedForwardTopology::Gated {
                intermediate_size,
                projections,
            } => {
                transient_pending.insert(index);
                for (projection_index, p) in projections.iter().enumerate() {
                    schedule.allocation(
                        "feed-forward-projection",
                        MemoryBytes::estimated(
                            output_bytes(&projection(p, rows, scalar))?,
                            product(&[rows, p.output, upper_scalar])?,
                            "gated feed-forward projection; final output remains owner-held",
                        ),
                        if projection_index + 1 == projections.len() {
                            hold.clone()
                        } else {
                            transient.clone()
                        },
                    )?;
                }
                schedule.allocation(
                    "gated-product",
                    MemoryBytes::estimated(
                        0,
                        product(&[rows, *intermediate_size, upper_scalar])?,
                        "gated feed-forward intermediate consumed before a later input evaluation",
                    ),
                    transient.clone(),
                )?;
            }
            FeedForwardTopology::Routed {
                experts,
                selected,
                intermediate_size,
                router,
                projections,
            } => {
                charge(
                    &mut schedule,
                    "router-projection",
                    output_bytes(&projection(router, rows, scalar))?,
                    product(&[rows, *experts, upper_scalar])?,
                )?;
                // Router scores/probabilities, selection indices, sorted routes
                // and weighted reduction are a mechanism-wide planning envelope.
                charge(
                    &mut schedule,
                    "expert-routing",
                    0,
                    add(
                        product(&[rows, *experts, 8])?,
                        product(&[rows, *selected, 16])?,
                    )?,
                )?;
                for p in projections {
                    let invocation = MechanismInvocation::ExpertDispatch {
                        rows,
                        experts: *experts,
                        selected: *selected,
                        input: p.input,
                        output: p.output,
                        format: p.format,
                        element: scalar,
                    };
                    let output = output_bytes(&invocation)?;
                    charge(
                        &mut schedule,
                        "expert-projection",
                        output,
                        product(&[
                            rows,
                            *selected,
                            add(p.input, mul(p.output, 2)?)?,
                            upper_scalar,
                        ])?,
                    )?;
                }
                charge(
                    &mut schedule,
                    "expert-gated-product",
                    0,
                    product(&[rows, *selected, *intermediate_size, upper_scalar, 2])?,
                )?;
            }
            FeedForwardTopology::Unknown { reason } => {
                schedule.allocation("feed-forward", MemoryBytes::unknown(reason), hold.clone())?
            }
        }
        let mut layer_upper = 0;
        for event in &schedule.plan.events[layer_start..] {
            if let ResourceLifetimeEvent::Acquire(description) = event {
                for allocation in &description.resources.allocations {
                    if let ResourceSize::Fixed { extent } = &allocation.size {
                        layer_upper = add(layer_upper, extent.capacity.upper_bytes.unwrap_or(0))?;
                    }
                }
            }
        }
        max_layer = max_layer.max(layer_upper);
    }
    // Extra equivalent layer sets are explicit calibration, not invented copies
    // of parameter/state storage. Missing graph retention remains unbounded.
    if copies.is_none() || copies.is_some_and(|copies| copies > layers) {
        schedule.global_allocation(
            "additional-lazy-overlap",
            MemoryBytes {
                lower_bytes: 0,
                upper_bytes: copies
                    .map(|copies| mul(max_layer, copies - layers))
                    .transpose()?,
                kind: ObservationKind::Estimated,
                detail: execution.workspace_overlap.detail.clone(),
            },
            global.clone(),
        )?;
    }
    schedule.global_allocation("attention-scratch", attention_scratch, global.clone())?;
    let logits_rows = mul(
        mul(request.batch_size, topology.output_invocations)?,
        if execution.logits == LogitsWorkspace::EveryPosition {
            query
        } else {
            1
        },
    )?;
    schedule.allocation(
        "final-normalization",
        MemoryBytes::estimated(
            product(&[logits_rows, topology.hidden_size, scalar_bytes])?,
            product(&[logits_rows, topology.hidden_size, upper_scalar])?,
            "final normalization follows the selected final-row or every-row output contract",
        ),
        global.clone(),
    )?;
    let logits = output_bytes(&projection(&topology.output, logits_rows, scalar))?;
    schedule.allocation(
        "logits-projection",
        MemoryBytes::estimated(
            logits,
            product(&[logits_rows, topology.output.output, upper_scalar])?,
            "selected vocabulary projection rows and possible arithmetic promotion",
        ),
        global.clone(),
    )?;
    if topology.output_softcap {
        schedule.allocation(
            "output-softcap",
            MemoryBytes::estimated(
                0,
                product(&[logits_rows, topology.vocabulary_size, upper_scalar, 3])?,
                "output softcap divide, tanh and multiply invocation envelope",
            ),
            global.clone(),
        )?;
    }
    schedule.allocation("sampling", MemoryBytes::estimated(0, product(&[logits_rows, topology.vocabulary_size, 4])?, "legacy portable float32 probability allowance; native sampling scratch is covered by backend overhead calibration"), global.clone())?;
    let (credit, projection_scratch, projection_details) =
        crate::projection_memory::refine(topology, rows, logits_rows)?;
    if !projection_details.is_empty() {
        schedule.global_allocation(
            "covered-projection-workspace",
            MemoryBytes {
                lower_bytes: 0,
                upper_bytes: copies
                    .map(|copies| {
                        add(
                            projection_scratch,
                            mul(projection_scratch, copies.saturating_sub(layers))?
                                .div_ceil(layers),
                        )
                    })
                    .transpose()?,
                kind: ObservationKind::Estimated,
                detail: format!(
                    "{}; summed native payload plus existing concurrent-layer workspace calibration",
                    projection_details.join("; ")
                ),
            },
            global.clone(),
        )?;
    }
    if let Some(parameters) = topology.selected_parameter_promotion_bytes {
        let parameters = parameters
            .checked_sub(credit)
            .ok_or_else(|| invalid("projection promotion credit exceeds selected task payload"))?;
        schedule.global_allocation(
            "uncached-parameter-conversions",
            parameter_conversions(parameters, layers, copies, persistent)?,
            global.clone(),
        )?;
    }
    schedule.global_allocation(
        "cache-update",
        cache_replacement(&execution.cache_update, persistent),
        global,
    )?;
    Ok(schedule.plan)
}

pub(crate) fn workspace(
    execution: &ExecutionMemoryPlan,
    request: &GenerationMemoryRequest,
    positions: u64,
    query: u64,
    persistent: u64,
) -> Result<MemoryBytes, CapabilityError> {
    let plan = describe_text_workspace(execution, request, positions, query, persistent)?;
    let report = compose_resource_peaks(&plan).map_err(|e| invalid(e.to_string()))?;
    if report.pools.is_empty() && report.missing.is_empty() {
        return Ok(MemoryBytes::exact(0));
    }
    let mut missing = report.missing.clone();
    for event in &plan.events {
        if let ResourceLifetimeEvent::Acquire(description) = event {
            for allocation in &description.resources.allocations {
                if let ResourceSize::Fixed { extent } = &allocation.size {
                    if extent.capacity.upper_bytes.is_none() {
                        missing.push(extent.capacity.detail.clone());
                    }
                }
            }
        }
    }
    missing.sort();
    missing.dedup();
    let peak = report
        .pools
        .first()
        .ok_or_else(|| invalid("workspace resource plan has no execution pool"))?;
    Ok(MemoryBytes {
        lower_bytes: peak.peak.capacity.lower_bytes,
        upper_bytes: peak.peak.capacity.upper_bytes,
        kind: ObservationKind::Estimated,
        detail: if missing.is_empty() {
            "generic mechanism calibration composed with explicit lazy-evaluation lifetimes".into()
        } else {
            missing.join("; ")
        },
    })
}

/// Shared calibration for materialized or fused score arithmetic. Selection
/// determines whether the score matrix is a real payload or only an upper bound.
fn attention_scratch_bytes(
    selected: &AttentionWorkspace,
    batch: u64,
    heads: u64,
    queries: u64,
    keys: u64,
) -> Result<MemoryBytes, CapabilityError> {
    Ok(match selected {
        AttentionWorkspace::Materialized | AttentionWorkspace::ScoreMatrixUpperBound => {
            let upper = product(&[batch, heads, queries, keys, 8])?;
            MemoryBytes::estimated(
                if matches!(selected, AttentionWorkspace::Materialized) {
                    upper
                } else {
                    0
                },
                upper,
                "calibrated float32 score/probability upper envelope for selected attention",
            )
        }
        AttentionWorkspace::Fused { scratch } => scratch.clone(),
        AttentionWorkspace::Unknown => {
            MemoryBytes::unknown("attention kernel workspace unavailable")
        }
    })
}

/// Gates, padded input and its contiguous copy, unfolded native convolution and
/// contiguous kernel. Input projection and convolution output are separate.
fn convolution_intermediates(
    batch: u64,
    queries: u64,
    channels: u64,
    kernel: u64,
    scalar: u64,
) -> Result<u64, CapabilityError> {
    if kernel == 0 {
        return Err(invalid("convolution kernel must be positive"));
    }
    let padded = add(queries, kernel - 1)?;
    let rows = add(
        add(mul(queries, 2)?, mul(padded, 2)?)?,
        add(mul(queries, kernel)?, kernel)?,
    )?;
    product(&[batch, channels, rows, scalar])
}

fn parameter_conversions(
    parameters: u64,
    layers: u64,
    copies: Option<u64>,
    persistent: u64,
) -> Result<MemoryBytes, CapabilityError> {
    let layers = layers.max(1);
    let upper = copies
        .map(|copies| {
            add(
                add(
                    parameters,
                    mul(parameters, copies.saturating_sub(layers))?.div_ceil(layers),
                )?,
                mul(persistent, 2)?,
            )
        })
        .transpose()?;
    Ok(MemoryBytes {
        lower_bytes: 0,
        upper_bytes: upper,
        kind: ObservationKind::Estimated,
        detail: "selected-task F32 conversion payload plus promoted state/replacement calibration"
            .into(),
    })
}

fn cache_replacement(selected: &CacheUpdateWorkspace, persistent: u64) -> MemoryBytes {
    match selected {
        CacheUpdateWorkspace::InPlace => MemoryBytes::exact(0),
        CacheUpdateWorkspace::CopyState => MemoryBytes::estimated(
            0,
            persistent,
            "one additional state allocation may overlap the installed cache",
        ),
        CacheUpdateWorkspace::Unknown => MemoryBytes::unknown("cache update overlap unavailable"),
    }
}

/// Only dimensions consumed by the native calibration; aggregate legacy records
/// need not invent KV heads, parameter encodings or other module identities.
struct ExplicitAttentionGeometry {
    batch: u64,
    heads: u64,
    expanded_width: u64,
    queries: u64,
    keys: u64,
    arithmetic: AttentionArithmetic,
}

/// Selection predicate for a real synchronous upstream evaluation, not merely
/// bounded scratch. Small invocations stay lazy; blockwise key paths lack this release contract.
fn input_score_evaluates_dependencies(
    facts: Option<InputScoreAttentionMechanism>,
    queries: u64,
    keys: u64,
) -> Result<bool, CapabilityError> {
    let Some(facts) = facts else { return Ok(false) };
    let Some(full) = facts.full_key_tiles else {
        return Ok(false);
    };
    if !full.evaluates_input_dependencies
        || keys > full.max_key_positions
        || mul(queries, keys)? <= facts.score_tile_elements
    {
        return Ok(false);
    }
    let tile = (facts.score_tile_elements / keys.max(1)).clamp(1, facts.max_query_rows);
    Ok(queries.div_ceil(tile) > full.max_live_query_tiles)
}

fn explicit_attention(
    facts: Option<InputScoreAttentionMechanism>,
    geometry: ExplicitAttentionGeometry,
    scalar: u64,
) -> Result<MemoryBytes, CapabilityError> {
    let ExplicitAttentionGeometry {
        batch,
        heads,
        expanded_width,
        queries: query,
        keys: positions,
        arithmetic,
    } = geometry;
    let Some(facts) = facts else {
        return Ok(MemoryBytes::unknown(
            "input-score attention native tile/retention calibration unavailable",
        ));
    };
    if [
        facts.score_tile_elements,
        facts.max_query_rows,
        facts.key_value_copies,
        facts.score_bytes,
    ]
    .contains(&0)
        || facts.full_key_tiles.is_some_and(|full| {
            [
                full.max_key_positions,
                full.shared_key_value_copies,
                full.max_live_query_tiles,
                full.retained_output_copies,
            ]
            .contains(&0)
        })
    {
        return Err(invalid(
            "input-score native tile/retention limits must be positive",
        ));
    }
    // Native softcap with Fused arithmetic promotes scores to F32 and uses
    // a full matrix. Only InputScores selects bounded query/key tiling.
    let tile = if arithmetic != AttentionArithmetic::InputScores
        || mul(query, positions)? <= facts.score_tile_elements
    {
        query.max(1)
    } else {
        (facts.score_tile_elements / positions.max(1)).clamp(1, facts.max_query_rows)
    };
    let tiles = query.div_ceil(tile);
    let (copies, live_rows, output_copies, score_positions) = match facts
        .full_key_tiles
        .filter(|full| positions <= full.max_key_positions)
    {
        Some(full) => (
            full.shared_key_value_copies,
            if tiles <= full.max_live_query_tiles {
                query
            } else {
                mul(tile, full.max_live_query_tiles)?
            },
            full.retained_output_copies,
            add(positions, 1)?,
        ),
        None => (mul(facts.key_value_copies, tiles)?, query, 0, positions),
    };
    let expanded = product(&[batch, expanded_width, positions, scalar, copies])?;
    let scores = product(&[batch, heads, live_rows, score_positions, facts.score_bytes])?;
    let outputs = product(&[batch, expanded_width, query, scalar, output_copies])?;
    Ok(MemoryBytes::estimated(0, add(add(expanded, scores)?, outputs)?, "input-score mechanism calibration: shared K/V layouts, bounded live tile graphs, completed outputs"))
}

fn validate_topology(topology: &TextExecutionTopology) -> Result<(), CapabilityError> {
    let check_projections = |projections: &[ProjectionTopology],
                             input: u64,
                             output: u64|
     -> Result<(), CapabilityError> {
        if projections.is_empty()
            || projections
                .iter()
                .any(|p| p.input == 0 || p.output == 0 || p.parameter.trim().is_empty())
            || projections.first().is_some_and(|p| p.input != input)
            || projections.last().is_some_and(|p| p.output != output)
        {
            return Err(invalid(
                "projection invocation sequence disagrees with module input/output geometry",
            ));
        }
        Ok(())
    };
    for layer in &topology.layers {
        for p in &layer.input_projections {
            if p.input == 0 || p.output != topology.hidden_size || p.parameter.trim().is_empty() {
                return Err(invalid(
                    "input fusion projection disagrees with residual stream geometry",
                ));
            }
        }
        match &layer.mixer {
            TokenMixerTopology::Unknown { reason } if reason.trim().is_empty() => {
                return Err(invalid("missing token mixer mechanism requires a reason"));
            }
            TokenMixerTopology::Unknown { .. } => {}
            TokenMixerTopology::Attention {
                query_heads,
                kv_heads,
                key_width,
                value_width,
                projections,
                ..
            } => {
                if *query_heads == 0
                    || *kv_heads == 0
                    || *key_width == 0
                    || *value_width == 0
                    || !query_heads.is_multiple_of(*kv_heads)
                {
                    return Err(invalid(
                        "attention heads and dimensions must be positive and compatible",
                    ));
                }
                check_projections(projections, topology.hidden_size, topology.hidden_size)?;
                let output_width = mul(*query_heads, *value_width)?;
                if projections.last().is_some_and(|p| p.input != output_width) {
                    return Err(invalid(
                        "attention output projection disagrees with value heads",
                    ));
                }
            }
            TokenMixerTopology::GatedConvolution {
                channels,
                kernel,
                projections,
            } => {
                if *channels == 0 || *kernel == 0 {
                    return Err(invalid("convolution channels and kernel must be positive"));
                }
                check_projections(projections, topology.hidden_size, topology.hidden_size)?;
            }
        }
        match &layer.feed_forward {
            FeedForwardTopology::Gated {
                intermediate_size,
                projections,
            } => {
                if *intermediate_size == 0 {
                    return Err(invalid("gated intermediate width must be positive"));
                }
                check_projections(projections, topology.hidden_size, topology.hidden_size)?;
                if projections
                    .last()
                    .is_some_and(|p| p.input != *intermediate_size)
                {
                    return Err(invalid(
                        "gated output projection disagrees with intermediate width",
                    ));
                }
            }
            FeedForwardTopology::Routed {
                experts,
                selected,
                intermediate_size,
                router,
                projections,
            } => {
                if *experts == 0
                    || *selected == 0
                    || selected > experts
                    || *intermediate_size == 0
                    || router.input != topology.hidden_size
                    || router.output != *experts
                {
                    return Err(invalid("expert selection and router geometry disagree"));
                }
                check_projections(projections, topology.hidden_size, topology.hidden_size)?;
            }
            FeedForwardTopology::Unknown { reason } if reason.trim().is_empty() => {
                return Err(invalid("missing mechanism coverage requires a reason"))
            }
            FeedForwardTopology::Unknown { .. } => {}
        }
    }
    Ok(())
}
