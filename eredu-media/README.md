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

`echo-cancellation` provides mono WebRTC AEC3 in 10 ms frames, with an independent
render reference and live capture during double-talk. It is included in
`system-audio`. `audio::system::duplex` connects ordinary CPAL streams to a bounded,
timestamped reference taken after playback clipping and scalar conversion.
Reference resampling and cancellation run on the capture reader, not in device
callbacks. Final capture padding is excluded from delivered PCM; stopping or
dropping capture disables reference collection. Reference continuity loss is an
error rather than silently using stale audio. Noise suppression and automatic
gain control are disabled.

The deterministic acoustic fixtures cover delayed multipath echo, independent
near-end components during double-talk, timestamped 48-to-16 kHz conversion,
bounded-reference continuity and capture shutdown. Echo-only attenuation is
22.5 dB in the multipath fixture and 18.2 dB through timestamped resampling;
these are synthetic measurements, not physical microphone/speaker validation.
The timestamped fixture's release run on M3 Ultra takes 1.57 ms per 80 ms of PCM
for reference delivery, capture/reference resampling and cancellation, including
fixture capture construction. This is host processing cost, not an end-to-end
conversation timing measurement.

```sh
cargo test -p eredu-media --features system-audio audio::echo::tests -- --nocapture
cargo test -p eredu-media --features system-audio audio::system::echo::tests -- --nocapture
```

The bundled WebRTC build requires a C/C++ toolchain, Meson, Ninja and Rust's
`llvm-tools-preview` component. Its Meson build downloads a pinned Abseil source
when no suitable system dependency is present. On macOS, install build tools with
`brew install meson ninja pkg-config` and
`rustup component add llvm-tools-preview`.
