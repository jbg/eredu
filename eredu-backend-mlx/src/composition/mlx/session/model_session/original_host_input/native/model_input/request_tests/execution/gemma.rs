//! Nonzero raw image/audio sources through the existing public core drivers.
use super::*;
use eredu_core::{InputExtent, InputMetadataKey, InputModality, InputPayloadKind};
use eredu_runtime::input::host::{HostInputPart, HostTensorValues, HostTensorView, PreparedHostInputPlan};

fn source(pool: &WorkingMemoryPool) -> OriginalPreparedHostInput {
    let patches = std::array::from_fn::<_, 192, _>(|i| (i as f32 - 93.0) / 193.0);
    let audio = std::array::from_fn::<_, 512, _>(|i| (i as f32 - 251.0) / 513.0);
    let image_metadata = [
        (InputMetadataKey::PatchGrid, HostTensorView { shape: &[1,3], values: HostTensorValues::I32(&[1,2,2]) }),
        (InputMetadataKey::PatchPositions, HostTensorView { shape: &[1,4,2], values: HostTensorValues::I32(&[0,0,0,1,1,0,1,1]) }),
    ];
    let audio_metadata = [(InputMetadataKey::AudioMask, HostTensorView {
        shape: &[1,4], values: HostTensorValues::Bool(&[true,true,true,true]),
    })];
    let parts = [
        HostInputPart { modality: InputModality::Text, kind: InputPayloadKind::TokenIds,
            payload: HostTensorView { shape: &[1,2], values: HostTensorValues::U32(&[1,2]) }, metadata: &[], extents: &[] },
        HostInputPart { modality: InputModality::Image, kind: InputPayloadKind::Tensor,
            payload: HostTensorView { shape: &[1,4,48], values: HostTensorValues::F32(&patches) },
            metadata: &image_metadata, extents: &[InputExtent::PatchGrid { time:1,height:2,width:2 }] },
        HostInputPart { modality: InputModality::Audio, kind: InputPayloadKind::Tensor,
            payload: HostTensorView { shape: &[1,4,128], values: HostTensorValues::F32(&audio) },
            metadata: &audio_metadata, extents: &[InputExtent::AudioValidFrames(4)] },
        HostInputPart { modality: InputModality::Text, kind: InputPayloadKind::TokenIds,
            payload: HostTensorView { shape: &[1,1], values: HostTensorValues::U32(&[3]) }, metadata: &[], extents: &[] },
    ];
    pool.compile_prepared_host_input(PreparedHostInputPlan::prepare(&parts).unwrap()).unwrap()
}

fn verify(routes: &[usize]) {
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    assert!(safemlx::metal::is_available().unwrap());
    let root = tempfile::tempdir().unwrap();
    crate::tests::distributed_pipeline_ring::write_gemma4_original_media_fixture(root.path());
    let expected = run_original(&pool, root.path(), 0, 0, 5, 32, source);
    for &route in routes {
        same(&run_original(&pool, root.path(), 0, route, 5, 32, source), &expected);
    }
}

#[test]
fn gemma_original_raw_image_audio_uneven_prefill_and_decode_match_all_drivers() {
    verify(&[1,2]);
}

#[test]
fn gemma_original_raw_image_audio_pending_and_committed_restore_fork_preserve_source() {
    verify(&[3,4]);
}
