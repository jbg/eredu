//! Compatibility lowering for reports written before ordinary module topology.
//!
//! These aggregate records cannot identify projection encodings, exact invocation
//! sequences, or parameter aliases. Do not invent an ordinary topology from them.
//! Preserve their historical linear/logit and layer-overlap envelope, reuse the
//! same calibrated mechanisms, and feed one aggregate allocation to the common
//! lifetime composer. Production preparation does not construct these records.

use super::*;

fn legacy_invalid(field: &'static str, detail: &str) -> CapabilityError {
    CapabilityError::InvalidConfiguration {
        field,
        detail: detail.into(),
    }
}

pub(super) fn describe(
    execution: &ExecutionMemoryPlan,
    request: &GenerationMemoryRequest,
    positions: u64,
    query: u64,
    persistent: u64,
) -> Result<ResourceLifetimePlan, CapabilityError> {
    let bytes = aggregate_envelope(execution, request, positions, query, persistent)?;
    let mut schedule = Schedule::new(vec![]);
    schedule.allocation(
        "legacy-aggregate-envelope",
        bytes,
        ResourceLifetime::Evaluation(id("request-evaluation")),
    )?;
    Ok(schedule.plan)
}

fn aggregate_envelope(
    execution: &ExecutionMemoryPlan,
    request: &GenerationMemoryRequest,
    positions: u64,
    query: u64,
    persistent: u64,
) -> Result<MemoryBytes, CapabilityError> {
    if execution.workspace_overlap.upper_live_copies == Some(0) {
        return Err(legacy_invalid(
            "workspace_overlap",
            "upper live copies must be positive",
        ));
    }
    let Some(g) = &execution.workspace else {
        return Ok(MemoryBytes::unknown(
            "decoder workspace geometry unavailable",
        ));
    };
    let width = add(
        add(mul(4, g.hidden_size)?, mul(3, g.intermediate_size)?)?,
        add(g.query_width, mul(2, g.key_value_width)?)?,
    )?;
    let linear = product(&[
        request.batch_size,
        query,
        width,
        request.scalar_bytes.get().into(),
    ])?;
    let logits_positions = match execution.logits {
        LogitsWorkspace::EveryPosition => query,
        LogitsWorkspace::FinalPosition => 1,
    };
    // Sampling commonly promotes logits to float32; retain both projection output
    // and float32 probabilities. The explicit model is conservative, not exhaustive.
    let logits = product(&[
        request.batch_size,
        logits_positions,
        g.vocabulary_size,
        add(u64::from(request.scalar_bytes.get()), 4)?,
    ])?;
    // Mixed-width execution may project F32 logits even when persistent state
    // has a narrower nominal dtype. Widen only that projection payload: the
    // existing four-byte sampling/probability copy is still charged once.
    let logits_upper = if g.mixed_precision_parameter_bytes.is_some() {
        product(&[
            request.batch_size,
            logits_positions,
            g.vocabulary_size,
            add(u64::from(request.scalar_bytes.get()).max(4), 4)?,
        ])?
    } else {
        logits
    };
    let linear_upper = execution
        .workspace_overlap
        .upper_live_copies
        .map(|copies| {
            let bytes = if g.input_score_attention.is_some()
                || g.gated_convolution.is_some()
                || g.mixed_precision_parameter_bytes.is_some()
            {
                product(&[
                    request.batch_size,
                    query,
                    width,
                    u64::from(request.scalar_bytes.get()).max(4),
                ])?
            } else {
                linear
            };
            mul(bytes, copies)
        })
        .transpose()?;
    let convolution = if let Some(conv) = &g.gated_convolution {
        let layers = execution.state_layout.layer_layout().len() as u64;
        if conv.channels == 0 || conv.kernel_size == 0 || conv.layers == 0 || conv.layers > layers {
            return Err(legacy_invalid(
                "gated_convolution",
                "positive channels/kernel and a layer count within the selected layout are required",
            ));
        }
        // The old envelope included the three-way input projection and
        // convolution output here; ordinary topology charges those separately.
        let one = add(
            product(&[
                request.batch_size,
                conv.channels,
                query,
                4,
                u64::from(request.scalar_bytes.get()).max(4),
            ])?,
            convolution_intermediates(
                request.batch_size,
                query,
                conv.channels,
                conv.kernel_size,
                u64::from(request.scalar_bytes.get()).max(4),
            )?,
        )?;
        execution
            .workspace_overlap
            .upper_live_copies
            .map(|copies| {
                let live = copies.min(add(conv.layers, copies.saturating_sub(layers))?);
                mul(one, live)
            })
            .transpose()?
    } else {
        None
    };
    let linear_upper = match (&g.gated_convolution, linear_upper, convolution) {
        (Some(_), Some(linear), Some(conv)) => Some(add(linear, conv)?),
        (Some(_), _, _) => None,
        (None, linear, _) => linear,
    };
    let mut bytes = MemoryBytes {
        lower_bytes: add(linear, logits)?,
        upper_bytes: linear_upper
            .map(|upper| add(upper, logits_upper))
            .transpose()?,
        kind: ObservationKind::Estimated,
        detail: execution.workspace_overlap.detail.clone(),
    };
    let attention = if g.query_heads == 0 {
        MemoryBytes::exact(0)
    } else {
        attention_scratch_bytes(
            &execution.attention,
            request.batch_size,
            g.query_heads,
            query,
            positions,
        )?
    };
    bytes = bytes.add(&attention)?;
    if let Some(explicit) = &g.input_score_attention {
        let layers = execution.state_layout.layer_layout().len() as u64;
        if explicit.layers == 0
            || explicit.layers > layers
            || g.query_heads == 0
            || g.query_width % g.query_heads != 0
        {
            return Err(legacy_invalid(
                "input_score_attention",
                "invalid explicit attention layer/head geometry",
            ));
        }
        let extra = match (
            explicit.mechanism,
            execution.workspace_overlap.upper_live_copies,
        ) {
            (Some(facts), Some(copies)) => {
                let live = copies.min(add(explicit.layers, copies.saturating_sub(layers))?);
                let scalar_bytes = if g.mixed_precision_parameter_bytes.is_some() {
                    u64::from(request.scalar_bytes.get()).max(4)
                } else {
                    u64::from(request.scalar_bytes.get())
                };
                let mut extra = explicit_attention(
                    Some(facts),
                    ExplicitAttentionGeometry {
                        batch: request.batch_size,
                        heads: g.query_heads,
                        expanded_width: g.query_width,
                        queries: query,
                        keys: positions,
                        arithmetic: AttentionArithmetic::InputScores,
                    },
                    scalar_bytes,
                )?;
                extra.upper_bytes = extra
                    .upper_bytes
                    .map(|bytes| mul(bytes, live))
                    .transpose()?;
                extra
            }
            _ => MemoryBytes::unknown(
                "input-score attention native workspace facts or overlap unavailable",
            ),
        };
        bytes = bytes.add(&extra)?;
    }

    if let Some(parameters) = g.mixed_precision_parameter_bytes {
        bytes = bytes.add(&parameter_conversions(
            parameters,
            execution.state_layout.layer_layout().len() as u64,
            execution.workspace_overlap.upper_live_copies,
            persistent,
        )?)?;
    }

    let mut replacement = cache_replacement(&execution.cache_update, persistent);
    // Old reports asserted replacement bytes at the lower end. Preserve that
    // historical contract without changing ordinary topology's possible overlap.
    if execution.cache_update == CacheUpdateWorkspace::CopyState {
        replacement.lower_bytes = persistent;
    }
    bytes.add(&replacement)
}
