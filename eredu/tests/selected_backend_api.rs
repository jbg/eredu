use eredu::{
    api::{LocalBackendError, LocalDevice, LocalLoadOptions, LocalRealtimeBackendFactory},
    RealtimeBackend, RealtimeModelLoadingBackend,
};

fn assert_selected_realtime_backend<B>(backend: &B)
where
    B: RealtimeBackend<Error = LocalBackendError>
        + RealtimeModelLoadingBackend<LoadOptions = LocalLoadOptions>,
{
    assert_eq!(backend.name(), "mlx");
}

#[test]
fn facade_creates_an_opaque_selected_realtime_backend() {
    let backend = LocalRealtimeBackendFactory::new(LocalDevice::Cpu)
        .create()
        .unwrap();
    assert_selected_realtime_backend(&backend);
}
