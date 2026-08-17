//! Minimal two-or-more-process microbatched pipeline generation probe.

use safemlx::{
    distributed::{self, Backend},
    DeviceType, Stream,
};
use safemlx_lm::{
    architectures::distributed::pipeline::{
        load_pipeline_model, PipelineInferencePhase, PipelineInferenceScheduler,
        PipelineMicrobatchInput, PipelineStep,
    },
    runtime::generation::sampler::DefaultSampler,
    DeviceAssignment, MlxBackend, ParallelTopology, RequestId, RequestStatus, SchedulerLimits,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = std::env::args()
        .nth(1)
        .ok_or("usage: pipeline_generate MODEL_DIR")?;
    let group = distributed::init(true, Backend::Ring)?;
    let local_index = std::env::var("LOCAL_RANK")
        .ok()
        .and_then(|rank| rank.parse().ok())
        .unwrap_or(0);
    let topology = ParallelTopology::from_group(
        &group,
        1,
        group.size(),
        1,
        DeviceAssignment::new(DeviceType::Gpu, local_index),
    )?;
    let stream = Stream::new_with_device(&topology.device.device()?);
    let weights_stream = Stream::new_with_device(&topology.device.device()?);
    let execution = MlxBackend::new(&stream).distributed(topology, &group)?;
    let mut model = load_pipeline_model(&model_dir, topology, &stream, &weights_stream)?;
    let is_first = model.stage_info().is_first;
    let requests = [RequestId::new(1), RequestId::new(2)];
    let prompts = [
        safemlx::Array::from_slice(&[1u32, 2, 3], &[1, 3]),
        safemlx::Array::from_slice(&[4u32, 5], &[1, 2]),
    ];
    let mut scheduler = PipelineInferenceScheduler::new(
        &model,
        SchedulerLimits::new(requests.len(), requests.len())?,
    )?;
    for (request, prompt) in requests.into_iter().zip(&prompts) {
        scheduler.register_request(&model, request)?;
        let input = PipelineMicrobatchInput::new(
            request,
            PipelineInferencePhase::Prefill,
            PipelineStep::new(1, prompt.shape()[1])?,
        );
        scheduler.enqueue(if is_first {
            input.with_tokens(prompt.clone())
        } else {
            input
        })?;
    }
    let mut sampler = DefaultSampler;
    let mut output = Vec::with_capacity(requests.len());
    while output.len() < requests.len() {
        output.extend(scheduler.run_queued(&mut model, &execution)?);
        std::thread::yield_now();
    }
    for generation_step in 0..8 {
        let mut next = Vec::new();
        for completed in &output {
            let request = completed.work().request();
            let synchronized = model.sample_and_synchronize(
                completed.logits(),
                completed.step(),
                &mut sampler,
                0.0,
                None,
                false,
                &execution,
            )?;
            if model.stage_info().is_last {
                eprintln!(
                    "request {} step {generation_step}: {:?}",
                    request.value(),
                    synchronized.token.evaluated()?.as_slice::<u32>()
                );
            }
            if synchronized.finished {
                scheduler.finish_request(request)?;
            } else {
                next.push((request, synchronized.token));
            }
        }
        if next.is_empty() {
            break;
        }
        let expected = next.len();
        for (request, token) in next {
            let input = PipelineMicrobatchInput::new(
                request,
                PipelineInferencePhase::Decode,
                PipelineStep::new(1, 1)?,
            );
            scheduler.enqueue(if is_first {
                input.with_tokens(token)
            } else {
                input
            })?;
        }
        output.clear();
        while output.len() < expected {
            output.extend(scheduler.run_queued(&mut model, &execution)?);
            std::thread::yield_now();
        }
    }
    for request in requests {
        if scheduler.request_status(request) == Some(RequestStatus::Active) {
            scheduler.finish_request(request)?;
        }
    }
    Ok(())
}
