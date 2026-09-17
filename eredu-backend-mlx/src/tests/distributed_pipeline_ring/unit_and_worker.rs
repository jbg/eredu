fn fixture_weights_stream(compute: &Stream) -> Stream {
    if compute.get_device().unwrap().get_type().unwrap() == DeviceType::Cpu {
        compute.clone()
    } else {
        Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
    }
}

include!("unit_and_worker/activation_contract.rs");
include!("unit_and_worker/worker_protocol.rs");
include!("unit_and_worker/speculative_delivery.rs");
include!("unit_and_worker/parameter_overlays.rs");
include!("unit_and_worker/component_capture.rs");
include!("unit_and_worker/provider_failures.rs");
include!("unit_and_worker/prediction_and_adapters.rs");
include!("unit_and_worker/prediction_components.rs");
include!("unit_and_worker/v4_components.rs");

include!("unit_and_worker/prepared_workspace.rs");

include!("unit_and_worker/prediction_storage.rs");

include!("unit_and_worker/idle_model_storage.rs");

include!("unit_and_worker/media_prefill.rs");
