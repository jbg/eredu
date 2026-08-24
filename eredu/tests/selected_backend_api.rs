use eredu::{
    api::{LocalBackendError, LocalLoadOptions, LocalRealtimeBackend},
    RealtimeBackend, RealtimeModelLoadingBackend,
};

fn assert_selected_realtime_backend<B>()
where
    B: RealtimeBackend<Error = LocalBackendError>
        + RealtimeModelLoadingBackend<LoadOptions = LocalLoadOptions>,
{
}

#[test]
fn facade_exposes_the_selected_realtime_backend() {
    assert_selected_realtime_backend::<LocalRealtimeBackend>();
}
