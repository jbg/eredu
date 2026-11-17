//! Complete selected-execution discovery and capture using explicit native resources.
use eredu_backend_mlx::{native, MlxLoadRequest};
use eredu_core::{
    capture::*, ControlledTextGeneration, GenerationConfigOverrides, ModelRuntime,
    ObservationSupportStatus, TextGenerationBackend, TextGenerationConfig, TokenFilter,
    TokenFilterController,
};
use safemlx::{Device, DeviceType};

struct Unconstrained;

impl TokenFilterController for Unconstrained {
    type Error = std::convert::Infallible;

    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        Ok(TokenFilter::All)
    }

    fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
        Ok(())
    }

    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }
}

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: capture_discovery <artifact> [token-id ...]"))?;
    let tokens = std::env::args()
        .skip(2)
        .map(|v| v.parse::<u32>())
        .collect::<Result<Vec<_>, _>>()?;
    anyhow::ensure!(
        !tokens.is_empty(),
        "provide one or more valid token IDs for this artifact"
    );
    let options = MlxLoadRequest::default();
    // This step only reads headers/configuration. It creates no device or tensors.
    let report = native::inspect_model(&path, native::MlxInspectionOptions::new(options.clone()))?;
    let architecture = report.architecture_descriptor.as_ref().ok_or_else(|| {
        anyhow::anyhow!("artifact has no admitted architecture: {:?}", report.issues)
    })?;
    for node in &architecture.nodes {
        println!("{}: {:?}", node.id, node.kind);
    }
    for edge in &architecture.edges {
        println!("{} -> {} ({:?})", edge.from, edge.to, edge.kind);
    }
    let support = report
        .observation_support
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("no selected execution support report"))?;
    for point in &support.points {
        println!(
            "{}: prefill {:?}; decode {:?}",
            point.path, point.prefill, point.decode
        );
    }
    let point = support
        .points
        .iter()
        .find(|p| {
            p.prefill == ObservationSupportStatus::Supported
                && p.decode == ObservationSupportStatus::Supported
        })
        .ok_or_else(|| anyhow::anyhow!("no observation supported in both phases"))?;

    let execution = native::ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let backend = native::backend(stream, stream);
    let prepared = eredu_core::load_model(&backend, &path, options)?;
    let mut runtime = ModelRuntime::from_prepared(backend, prepared)?;
    let declarations =
        <eredu_backend_mlx::backend::MlxBackend as TextGenerationBackend>::capture_declarations(
            &runtime,
        )?;
    let limits = CaptureUsage {
        captures: 1,
        retained_bytes: 64 << 20,
        host_bytes: 16 << 20,
        encoded_bytes: 64 << 20,
    };
    let plan = CapturePlan {
        schema_version: CAPTURE_SCHEMA_VERSION,
        provenance: CaptureProvenance::Session,
        selections: vec![CaptureSelection {
            id: "activation".into(),
            path: point.path.clone(),
            schedule: CaptureSchedule {
                prefill: true,
                decode: true,
                end_prediction: Some(2),
                ..Default::default()
            },
            slices: vec![],
            transform: CaptureTransform::FullTensor,
        }],
        limits: CaptureLimits {
            per_step: limits,
            cumulative: limits.checked_mul(2)?,
            physical_native_bytes: None,
            on_limit: CaptureLimitPolicy::Fail,
        },
    }
    .admit(
        &declarations.catalog,
        &declarations.support,
        &declarations.support.capture,
        CaptureRequestShape {
            batch: 1,
            prompt_tokens: tokens.len() as u64,
            max_predictions: 2,
        },
    )?;
    let config = eredu_core::resolve_generation_config(
        None,
        GenerationConfigOverrides {
            max_new_tokens: Some(2),
            ..Default::default()
        },
    )?;
    let mut generation = ControlledTextGeneration::new(
        &mut runtime,
        tokens,
        TextGenerationConfig::new(config),
        Unconstrained,
    )?;
    generation.enable_capture(plan)?;
    for phase in [CapturePhase::Prefill, CapturePhase::Decode] {
        generation
            .next()
            .ok_or_else(|| anyhow::anyhow!("missing {phase:?} prediction"))??;
        let step = generation
            .take_captured_step()?
            .ok_or_else(|| anyhow::anyhow!("missing {phase:?} capture"))?;
        anyhow::ensure!(step.phase == phase, "capture has the wrong phase");
        for record in step.records {
            println!("{phase:?} {}: {:?}", record.path, record.payload);
        }
    }
    Ok(())
}
