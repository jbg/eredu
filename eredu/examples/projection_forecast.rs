//! Forecast mechanism facts against the released-checkpoint allocation envelope.
use anyhow::{ensure, Context};
use eredu::api::*;
use eredu_backend_mlx::{allocator_memory, MlxBackendFactory};
use eredu_core::{
    residency::ParameterConversionRetentionPolicy, ExecutionPlan, GenerationConfigOverrides,
    InputTokenCount, TextGenerationConfig,
};
use eredu_runtime::memory_estimation::estimate_generation_memory;
use serde_json::json;
use std::path::PathBuf;

fn binding_count(request: &eredu_runtime::memory_estimation::GenerationMemoryRequest) -> usize {
    request
        .domains
        .iter()
        .flat_map(|d| &d.executions)
        .filter_map(|e| e.execution_topology.as_ref())
        .map(|t| t.projection_storage.len())
        .sum()
}

fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 4,
        "usage: projection_forecast MODEL OUTPUT BYTES|unlimited POSITIONS"
    );
    let path = PathBuf::from(&args[0]);
    let positions: usize = args[3].parse()?;
    let policy = if args[2] == "unlimited" {
        ParameterConversionRetentionPolicy::Unlimited
    } else {
        ParameterConversionRetentionPolicy::Bounded {
            max_bytes: args[2].parse()?,
        }
    };
    set_local_allocator_cache_limit(0)?;
    let factory = MlxBackendFactory::default();
    let plan = ExecutionPlan::fully_resident(local_device_plan(LocalDevice::Accelerator(0))?)
        .with_parameter_conversion_retention(Some(policy));
    let inspection = eredu_backend_mlx::native::inspect_model_preparation(
        &path,
        eredu_backend_mlx::native::MlxInspectionOptions::new(factory.load_request_for_plan(&plan)?),
    )?;
    let mut cold_options = GenerationMemoryOptions::for_local_device(
        InputTokenCount::text(positions as u64),
        plan.device(),
    )?;
    cold_options.max_output_tokens = Some(8);
    cold_options.prefill_chunk_tokens = positions as u64;
    let cold = forecast_inspected_generation(&inspection, &cold_options, &Default::default())?;
    for d in &cold.request.domains {
        for e in &d.executions {
            if let Some(t) = &e.execution_topology {
                ensure!(
                    t.projection_storage.is_empty(),
                    "cold forecast claims native binding coverage"
                );
            }
        }
    }
    let (mut model, _) = LoadedModel::load_execution_plan(&factory, &path, &plan)?.into_parts();
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            temperature: Some(0.0),
            max_new_tokens: Some(8),
            ..Default::default()
        },
        ..Default::default()
    };
    model.synchronize()?;
    let before = allocator_memory()?.active_bytes();
    let retention_before = model.parameter_conversion_retention()?;
    let loaded = model.forecast_token_ids(&vec![1; positions], settings, &Default::default())?;
    ensure!(
        allocator_memory()?.active_bytes() == before,
        "forecast allocated native storage"
    );
    ensure!(
        model.parameter_conversion_retention()? == retention_before,
        "forecast changed conversion admission"
    );
    let coverage: usize = loaded
        .request
        .domains
        .iter()
        .flat_map(|d| &d.executions)
        .filter_map(|e| e.execution_topology.as_ref())
        .map(|t| t.projection_storage.len())
        .sum();
    ensure!(coverage > 0, "no native binding coverage");
    let mut unrefined = loaded.request.clone();
    for d in &mut unrefined.domains {
        for e in &mut d.executions {
            if let Some(t) = &mut e.execution_topology {
                t.projection_storage.clear();
            }
        }
    }
    let unrefined_estimate = estimate_generation_memory(&unrefined)?;
    let upper = loaded.estimate.domains[0]
        .additional_generation_peak
        .upper_bytes
        .context("missing loaded bound")?;
    let old_upper = unrefined_estimate.domains[0]
        .additional_generation_peak
        .upper_bytes
        .context("missing unrefined bound")?;
    ensure!(
        upper < old_upper,
        "covered invocations did not reduce forecast: scalar={}",
        loaded.request.scalar_bytes
    );
    // A row just beyond the admitted GEMM coverage must retain the prefill allowance.
    let outside = model.forecast_token_ids(&vec![1; 2001], settings, &Default::default())?;
    let mut without = outside.request.clone();
    for d in &mut without.domains {
        for e in &mut d.executions {
            if let Some(t) = &mut e.execution_topology {
                t.projection_storage.clear();
            }
        }
    }
    let outside_without = estimate_generation_memory(&without)?;
    ensure!(
        outside.estimate.domains[0].generation_peak.upper_bytes
            == outside_without.domains[0].generation_peak.upper_bytes,
        "unsupported prefill received conversion credit"
    );
    reset_local_allocator_peak()?;
    let config = model.resolve_generation_config(settings.overrides)?;
    let mut tokens =
        model.generate_tokens(vec![1; positions], TextGenerationConfig::new(config))?;
    let first = tokens.next().context("no first token")??.token_id()?;
    tokens.synchronize()?;
    let peak = allocator_memory()?.peak_bytes();
    ensure!(
        peak.saturating_sub(before) <= upper,
        "loaded forecast below observed prefill growth"
    );
    let continuation = tokens.forecast_remaining_generation(4, &Default::default())?;
    let mut ids = vec![first];
    for token in tokens {
        ids.push(token?.token_id()?);
    }
    ensure!(ids.len() == 8, "incomplete cached generation");
    // Controlled continuation consumes the same facts without advancing tokens.
    model.reset()?;
    let chat = model.prepare_chat(eredu::runtime::chat::ChatTemplateRequest {
        messages: vec![json!({"role":"user","content":"hello"})],
        add_generation_prompt: true,
        ..Default::default()
    })?;
    let prepared = model.prepare_observed_token_ids(
        &chat,
        vec![1; positions],
        settings,
        eredu_core::capture::CapturePlan::none(),
        TraceLimits {
            per_record_bytes: 16384,
            total_bytes: 1 << 20,
        },
    )?;
    let mut run = model.start_controlled_text(prepared, &[], Default::default(), |_| {
        std::ops::ControlFlow::Continue(())
    })?;
    run.step(|_| std::ops::ControlFlow::Continue(()))?;
    let control_tokens = run.token_ids().to_vec();
    let controlled = run.forecast_remaining_generation(4, &Default::default())?;
    ensure!(
        run.token_ids() == control_tokens && binding_count(&controlled.request) > 0,
        "controlled forecast lost binding facts or advanced"
    );
    run.run(|_| std::ops::ControlFlow::Continue(()))?;
    ensure!(run.token_ids() == ids, "controlled tokens changed");
    drop(run);
    drop(model);
    let speculative_plan = plan.with_drafting(eredu_core::DraftingPlan::External {
        model: path.display().to_string(),
        placement: eredu_core::DraftPlacementPlan::Target,
        max_draft_tokens: 2,
        lookahead: false,
        adaptive_lookahead: false,
    });
    let mut loaded_spec = LoadedModel::load_execution_plan(&factory, &path, &speculative_plan)?;
    let spec_options = loaded_spec
        .speculative_generation_options()?
        .context("missing drafting options")?;
    let (target, draft) = loaded_spec.parts_mut();
    let draft = draft.as_speculative_draft().context("missing draft")?;
    let speculative = target.forecast_speculative_token_ids(
        &vec![1; positions],
        settings,
        &draft,
        spec_options,
        &Default::default(),
    )?;
    ensure!(
        binding_count(&speculative.request) > 0
            && binding_count(
                speculative
                    .speculative
                    .as_ref()
                    .unwrap()
                    .draft
                    .as_ref()
                    .unwrap()
            ) > 0,
        "target/draft forecast lost binding facts"
    );
    let mut speculative_continuation = None;
    let spec_output = target.with_controlled_text_speculative(
        PreparedChatSpeculativeGenerationRequest {
            input: PreparedChatInput::token_ids(&chat, vec![1; positions]),
            drafting: draft,
            settings,
            options: spec_options,
            caller_stop_sequences: &[],
            cancellation: Default::default(),
            on_event: |_| {},
        },
        Default::default(),
        |session| {
            session.step()?;
            let tokens = session.token_ids().to_vec();
            let forecast = session
                .forecast_remaining_generation(4, &Default::default())
                .unwrap();
            assert_eq!(session.token_ids(), tokens);
            assert!(binding_count(&forecast.request) > 0);
            speculative_continuation = Some(forecast);
            while session.step()?.is_some() {}
            Ok(())
        },
    )?;
    ensure!(spec_output.token_ids() == ids, "speculative tokens changed");
    let mut resources = Vec::new();
    for d in &loaded.request.domains {
        for e in &d.executions {
            let plan = eredu_runtime::workspace_resources::describe_text_workspace(
                e,
                &loaded.request,
                positions as u64,
                positions as u64,
                0,
            )?;
            for event in plan.events {
                if let eredu_runtime::resource_lifetimes::ResourceLifetimeEvent::Acquire(
                    description,
                ) = event
                {
                    resources.extend(description.resources.allocations.into_iter().filter(|a| {
                        a.identity.key.ends_with("covered-projection-workspace")
                            || a.identity.key.ends_with("uncached-parameter-conversions")
                    }));
                }
            }
        }
    }
    std::fs::write(
        &args[1],
        serde_json::to_vec_pretty(
            &json!({"positions":positions,"retention":args[2],"tokens":ids,
        "cold":cold,"loaded":loaded,"unrefined_estimate":unrefined_estimate,"continuation":continuation,"controlled":controlled,"speculative":speculative,"speculative_continuation":speculative_continuation,
        "observed_prefill_active_growth":peak.saturating_sub(before),"native_binding_count":coverage,
        "unsupported_2001_preserves_prefill_bound":true,"forecast_allocates_native_storage":false,"workspace_resources":resources}),
        )?,
    )?;
    Ok(())
}
