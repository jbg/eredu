# eredu-media

Backend-neutral deterministic host image, audio, and video transformations for
Eredu.

The crate has no default features. Enable `image` for RGB resize and
normalization, `audio` for log-mel extraction and WAV processing, or both. Portable media request
values remain in `eredu-core`; concrete backends only lower the processed
buffers to native tensors.

Architectures retain semantic processor policy, including request geometry,
filters, normalization, and audio feature configuration. This crate validates
host inputs and returns owned processed buffers with explicit layout and shape;
it never selects an accelerator, inference stream, or native tensor implementation.

The optional `system-audio` feature supplies cross-platform CPAL microphone and
speaker devices, bounded lock-free callback queues, scalar format/channel
conversion and streaming band-limited resampling. Resampling and application
work run outside audio callbacks. Applications own pacing, input closure,
playback completion and interruption. Host audio streams are independent of
backend execution resources.
