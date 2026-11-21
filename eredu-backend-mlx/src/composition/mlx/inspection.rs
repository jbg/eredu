//! Cold model-inspection mechanism facts supplied by MLX.

/// Returns authoritative cold preparation capabilities for this MLX build.
/// This provider performs no hardware discovery and creates no native device,
/// stream or tensor. Facade composition uses it with the shared architecture
/// inspection driver; it does not establish availability on a target machine.
pub fn inspection_mechanisms() -> impl eredu_core::preparation::NativePreparation {
    super::loading::MlxPreparationMechanisms
}

/// Optional host ingress paths compiled into this MLX build.
pub const fn inspection_media_features() -> eredu_core::MediaFeatureAvailability {
    eredu_core::MediaFeatureAvailability {
        image: cfg!(feature = "image"),
        audio: cfg!(feature = "audio"),
    }
}
