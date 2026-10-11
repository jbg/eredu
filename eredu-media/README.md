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

`audio-processing` provides mono WebRTC capture processing in 10 ms frames,
with optional AEC3 and noise suppression in the same processor. Noise suppression
supports off, low, moderate, high and very-high strengths, independently of echo
cancellation. AEC3 uses an independent render reference and retains live capture
during double-talk. The feature is included in
`system-audio`. `audio::system::duplex` connects ordinary CPAL streams to a bounded,
timestamped reference taken after playback clipping and scalar conversion.
Reference resampling and cancellation run on the capture reader, not in device
callbacks. Final capture padding is excluded from delivered PCM; stopping or
dropping capture disables reference collection. Reference continuity loss is an
error rather than silently using stale audio. Noise suppression defaults to off;
automatic gain control is disabled. `SystemAudioInput::set_capture_enabled`
marks delivery permission at capture time, so queued closed-mic samples remain
muted when admission opens. Real microphone PCM still reaches the filters.

`audio::admission::SpeechAdmission` provides backend-independent level admission
without a native library. Sustained RMS levels open the gate, with configurable
hysteresis, minimum duration, pre-roll and hangover. It replaces rejected windows
with silence. A fixed lookahead equal to pre-roll plus minimum duration, rounded
to 10 ms windows, preserves leading audio without replaying samples; EOF flush
returns the exact remaining source count. This is an energy gate, not a speech
classifier or intended-speaker detector. Pre-roll and hangover include surrounding
audio by design.

The deterministic acoustic fixtures cover delayed multipath echo, independent
near-end components during double-talk, timestamped 48-to-16 kHz conversion,
bounded-reference continuity and capture shutdown. Echo-only attenuation is
22.5 dB in the multipath fixture and 18.2 dB through timestamped resampling;
these are synthetic measurements, not physical microphone/speaker validation.
The high-strength stationary-noise fixture measures 18.0 dB attenuation and
retains the 0.120-amplitude near-end tone. It does not establish background-speech
rejection or perceptual speech quality.
The combined AEC3/moderate-suppression multipath fixture measures 35.2 dB
echo attenuation. Both near-end tone components retain more than half their
amplitude, with output energy 0.411 times the independent near-end energy;
suppression can therefore materially attenuate desired input too.
The timestamped fixture's release run on M3 Ultra takes 1.57 ms per 80 ms of PCM
for reference delivery, capture/reference resampling and cancellation, including
fixture capture construction. This is host processing cost, not an end-to-end
conversation timing measurement.

```sh
cargo test -p eredu-media --features system-audio audio::processing::tests -- --nocapture
cargo test -p eredu-media --features system-audio audio::system::processing::tests -- --nocapture
```

The bundled WebRTC build requires a C/C++ toolchain, Meson, Ninja and Rust's
`llvm-tools-preview` component. Its Meson build downloads a pinned Abseil source
when no suitable system dependency is present. On macOS, install build tools with
`brew install meson ninja pkg-config` and
`rustup component add llvm-tools-preview`.
