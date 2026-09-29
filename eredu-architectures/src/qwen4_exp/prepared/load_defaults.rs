//! Practical ordinary-load defaults; explicit execution controls remain authoritative.
use super::{Config, TargetLoadError};
use eredu_runtime::{
    AppendStreamLimits, AppendStreamLoadPolicy, BoundedExecutionPolicy,
    BoundedExecutionPolicyError, CacheResidencyPolicy, DraftingLoadRequest, InvocationLimits,
    NormalizedLoadRequest, ParameterBankLoadOptions, RowLookupLimits, RowLookupLoadPolicy,
    TiledSelectionLimits,
};

/// Resolves ordinary loading without requiring callers to calculate family workspaces.
/// These allowances are ceilings, not eager allocations or physical peak predictions.
pub(crate) fn normalize_load_request(
    request: &NormalizedLoadRequest,
    config: &Config,
) -> Result<NormalizedLoadRequest, TargetLoadError> {
    let mut request = request.clone();
    if request.bounded_execution().is_none() {
        let policy = default_policy(&request, config)?;
        request = request.with_bounded_execution(policy);
    }
    // Ordinary inference does not opt into speculation merely because an artifact
    // contains prediction weights. Explicit Embedded remains a separate request.
    if request.drafting() == DraftingLoadRequest::ArchitectureDefault {
        request = request.with_drafting(DraftingLoadRequest::Disabled);
    }
    Ok(request)
}

fn product(values: &[u64]) -> Result<u64, TargetLoadError> {
    values.iter().try_fold(1u64, |result, value| {
        result.checked_mul(*value).ok_or_else(overflow)
    })
}

fn sum(values: &[u64]) -> Result<u64, TargetLoadError> {
    values.iter().try_fold(0u64, |result, value| {
        result.checked_add(*value).ok_or_else(overflow)
    })
}

fn overflow() -> TargetLoadError {
    BoundedExecutionPolicyError::Invalid("managed execution geometry").into()
}

fn default_policy(
    request: &NormalizedLoadRequest,
    config: &Config,
) -> Result<BoundedExecutionPolicy, TargetLoadError> {
    let (batch, chunk_tokens) = request
        .parallel_execution()
        .map(|parallel| parallel.invocation_limits())
        .unwrap_or((1, config.max_positions.min(512)));
    let invocation = InvocationLimits::new(batch, chunk_tokens, config.max_positions)?;
    let attention = &config.attention;
    let entries = (config.max_positions / attention.ratio).max(1) as usize;
    let tile = entries.min(256);
    let page = match request.state_residency() {
        CacheResidencyPolicy::Paged(options) => options.block_size_tokens() as usize,
        CacheResidencyPolicy::Device => entries.min(256),
    };
    let selection = super::super::qsa::QsaSelectionSpec {
        heads: attention.index_heads,
        dimensions: attention.index_head_dim,
        ratio: attention.ratio,
        token_budget: attention.budget,
        tile_blocks: tile as i32,
        workspace_bytes: 0,
    }
    .required_workspace()
    .map_err(|_| overflow())?;
    // Broad headroom for projected values, selected positions and partial blocks.
    // The existing equation admission still checks its exact logical minimum.
    let projected_width = product(&[
        attention.index_heads as u64 + 1,
        attention.index_head_dim as u64,
    ])?;
    let invocation_workspace = sum(&[
        selection,
        product(&[
            invocation.invocation_tokens() as u64,
            sum(&[
                projected_width,
                attention.budget as u64,
                attention.ratio as u64,
            ])?,
            64,
        ])?,
        product(&[attention.ratio as u64, attention.index_head_dim as u64, 64])?,
        1 << 20,
    ])?;

    let order = config
        .ngram
        .order
        .checked_sub(1)
        .filter(|n| *n > 0)
        .ok_or_else(overflow)?;
    let heads = product(&[order as u64, config.ngram.heads as u64])?;
    let requests = product(&[invocation.invocation_tokens() as u64, heads])?;
    let width = (config.ngram.embedding_dim as u64)
        .checked_div(heads)
        .filter(|n| *n > 0)
        .ok_or_else(overflow)?;
    let acquisition_rows = requests.min(256);
    // Acquisitions cover ordinary floating rows and published GGUF encodings.
    let acquisition = product(&[acquisition_rows, width, 8])?;
    let host = product(&[requests, 128])?;
    let output = product(&[requests, width, 12])?;
    let scratch = sum(&[
        host,
        output,
        product(&[acquisition_rows, width, 64])?,
        1 << 20,
    ])?;
    let bank = match request.weight_residency().parameter_bank_cache() {
        Some(explicit) => explicit,
        None => ParameterBankLoadOptions::new(
            eredu_core::residency::OffloadConfig::new(
                Some((64 << 20).max(acquisition)),
                Some(0),
                1,
            )
            .map_err(|_| overflow())?,
            scratch,
            scratch,
        )
        .map_err(|_| overflow())?,
    };
    let rows = RowLookupLoadPolicy::new(
        RowLookupLimits {
            requests: usize::try_from(requests).map_err(|_| overflow())?,
            rows_per_acquisition: acquisition_rows as usize,
            acquisition_bytes: acquisition,
            host_bytes: host,
            output_bytes: output,
        },
        bank,
        product(&[config.ngram.layers.len().max(1) as u64, 16])?,
    )?;
    // Both streams share allowances sized for the wider record. Full history is
    // retained; sparse selection only reduces the work of attention.
    let record = product(&[attention.index_head_dim.max(attention.ratio) as u64, 4])?;
    let append = AppendStreamLoadPolicy::new(
        AppendStreamLimits {
            entries,
            page_entries: page,
            read_entries: tile,
        },
        product(&[entries as u64, record])?,
        product(&[sum(&[page as u64, tile as u64])?, record, 2])?,
        // A coarse catalog allowance avoids making native handle size an input
        // to architecture policy. No catalog is allocated at this ceiling.
        product(&[entries.div_ceil(page) as u64, 4096])?,
    )?;
    Ok(BoundedExecutionPolicy::new(
        invocation,
        TiledSelectionLimits::new(tile as i32, selection, invocation_workspace)?,
        rows,
        append,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn released() -> Config {
        Config::from_json(&serde_json::from_str(include_str!("../config/released.json")).unwrap())
            .unwrap()
    }

    #[test]
    fn released_defaults_cover_context_rows_and_preserve_prediction_metadata() {
        let config = released();
        let request = normalize_load_request(&NormalizedLoadRequest::default(), &config).unwrap();
        let policy = request.bounded_execution().unwrap();
        assert_eq!(policy.invocation().chunk_tokens(), 512);
        assert_eq!(policy.invocation().history_tokens(), config.max_positions);
        assert_eq!(policy.rows().limits().requests, 512 * 16);
        assert_eq!(policy.append().limits().entries, 262144 / 4);
        assert_eq!(
            policy.rows().bank().offload().device_budget_bytes(),
            Some(64 << 20)
        );
        assert_eq!(policy.rows().bank().offload().host_budget_bytes(), Some(0));
        assert_eq!(request.drafting(), DraftingLoadRequest::Disabled);
        assert!(config.prediction.is_some());
    }

    #[test]
    fn parallel_defaults_preserve_declared_invocation_geometry() {
        let config = released();
        let topology = eredu_core::ParallelTopology::new(2, 1, 1, 1).unwrap();
        let request = NormalizedLoadRequest::default()
            .with_parallel_execution(
                eredu_runtime::ParallelLoadRequest::new(
                    eredu_core::ParallelRankTopology::new(topology, 0).unwrap(),
                    eredu_runtime::PipelineWireContract::new(
                        eredu_runtime::PipelineActivationDtype::Float32,
                    ),
                    2,
                    64,
                    eredu_runtime::CommunicationCompletionPolicy::new(
                        std::time::Duration::from_secs(5),
                        eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
                    )
                    .unwrap(),
                )
                .unwrap(),
            )
            .unwrap();
        let normalized = normalize_load_request(&request, &config).unwrap();
        normalized.validate_model_preparation().unwrap();
        let policy = normalized.bounded_execution().unwrap();
        assert_eq!(policy.invocation().batch(), 2);
        assert_eq!(policy.invocation().chunk_tokens(), 64);
        assert_eq!(policy.invocation().history_tokens(), config.max_positions);
        assert_eq!(policy.rows().limits().requests, 2 * 64 * 16);
        assert_eq!(
            normalized.parallel_execution(),
            request.parallel_execution()
        );
    }

    #[test]
    fn explicit_policy_and_embedded_intent_are_preserved() {
        let config = released();
        let explicit = normalize_load_request(&NormalizedLoadRequest::default(), &config)
            .unwrap()
            .with_drafting(DraftingLoadRequest::embedded(3).unwrap());
        assert_eq!(
            normalize_load_request(&explicit, &config).unwrap(),
            explicit
        );
    }
}
