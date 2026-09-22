fn fixture_weights_stream(compute: &Stream) -> Stream {
    if compute.get_device().unwrap().get_type().unwrap() == DeviceType::Cpu {
        compute.clone()
    } else {
        Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
    }
}

fn prepared_component_backend(compute: &Stream) -> MlxBackend<'static> {
    let kind = compute.get_device().unwrap().get_type().unwrap();
    let pool = crate::backend::managed_memory::ledger();
    let streams =
        crate::backend::managed_memory::gpu_stream::PreparedExecutionStreams::for_device_factory(
            &pool, kind,
        )
        .unwrap()
        .expect("the numerical oracle uses the selected prepared stream mechanism");
    let accelerator = match kind {
        DeviceType::Cpu => None,
        DeviceType::Gpu => Some(if cfg!(feature = "cuda") {
            crate::backend::MlxAcceleratorFamily::Cuda
        } else {
            crate::backend::MlxAcceleratorFamily::Metal
        }),
    };
    let identity = crate::backend::MlxDeviceIdentity::from_realized_device(
        &streams.execution().get_device().unwrap(),
        accelerator,
    )
    .unwrap();
    MlxBackend::for_prepared_execution_plan(streams, identity)
}

include!("unit_and_worker/activation_contract.rs");
include!("unit_and_worker/worker_protocol.rs");
include!("unit_and_worker/speculative_delivery.rs");
include!("unit_and_worker/parameter_overlays.rs");
include!("unit_and_worker/component_capture.rs");
include!("unit_and_worker/admitted_expert.rs");
include!("unit_and_worker/provider_failures.rs");
include!("unit_and_worker/prediction_and_adapters.rs");
include!("unit_and_worker/prediction_components.rs");
include!("unit_and_worker/v4_components.rs");

include!("unit_and_worker/prepared_workspace.rs");

include!("unit_and_worker/prediction_storage.rs");

include!("unit_and_worker/idle_model_storage.rs");

include!("unit_and_worker/media_prefill.rs");
