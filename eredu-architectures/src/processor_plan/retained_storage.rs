//! Exhaustive storage declarations for actual retained processor policy fields.
//! Runtime request products deliberately do not implement this contract.
use super::*;

pub(crate) trait InlineProcessorMetadata {
    fn validate_inline(&self) {}
}

// This private set contains scalar values, not references, owned containers,
// source handles or native objects. Composite implementations below enumerate
// every field so a newly introduced owner cannot inherit an empty census.
macro_rules! scalar {
    ($($ty:ty),* $(,)?) => { $(impl InlineProcessorMetadata for $ty {})* };
}
scalar!(bool, u8, u32, u64, usize, f32, f64);
impl<T: InlineProcessorMetadata> InlineProcessorMetadata for Option<T> {}
impl<T: InlineProcessorMetadata, const N: usize> InlineProcessorMetadata for [T; N] {}
impl<A: InlineProcessorMetadata, B: InlineProcessorMetadata> InlineProcessorMetadata for (A, B) {}
fn field<T: InlineProcessorMetadata>(_: &T) {}
macro_rules! fields {
    ($ty:ty { $($name:ident),* $(,)? }) => {
        impl InlineProcessorMetadata for $ty {
            fn validate_inline(&self) {
                let Self { $($name),* } = self;
                $(field($name);)*
            }
        }
    };
}
fields!(MediaFraming { start_token_id, end_token_id });
fields!(QwenProcessorSize { shortest_edge, longest_edge });
fields!(QwenVisualSource {
    size, patch_size, temporal_patch_size, merge_size, do_resize, do_rescale,
    rescale_factor, do_normalize, resample, image_mean, image_std, fps,
    min_frames, max_frames, do_sample_frames,
});
fields!(QwenProcessorPlan { image, video, framing });
fields!(Gemma4VisualPolicy { patch_size, pooling_kernel_size, max_soft_tokens });
fields!(Gemma4ProcessorPlan { image, video, image_framing, audio_framing, has_audio });
fields!(InklingProcessorPlan { image_bos_token_id, audio_bos_token_id, dmel_bins, dmel_min, dmel_max });
fields!(MuseVisualSource {
    do_resize, do_rescale, rescale_factor, do_normalize, image_mean, image_std,
    patch_size, temporal_patch_size, merge_size, max_image_tokens,
    max_video_frame_tokens, num_frames, fps, do_sample_frames, resample,
});
fields!(MuseProcessorPlan { image, video, image_only });
