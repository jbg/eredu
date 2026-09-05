# eredu-media

Backend-neutral deterministic host image, audio, and video transformations for
Eredu.

The crate has no default features. Enable `image` for RGB resize and
normalization, `audio` for log-mel extraction, or both. Portable media request
values remain in `eredu-core`; concrete backends only lower the processed
buffers to native tensors.

Architectures retain semantic processor policy, including request geometry,
filters, normalization, and audio feature configuration. This crate validates
host inputs and returns owned processed buffers with explicit layout and shape;
it never selects a device, stream, or native tensor implementation.
