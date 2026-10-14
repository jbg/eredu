//! Shared released configuration and independent numerical reference fixtures.

/// Published 3B draft adapter configuration, with ancestry pinned in family evidence.
pub const DRAFT_ADAPTER_CONFIG_JSON: &str =
    include_str!("../fixtures/nemotron_labs_diffusion/3B-adapter-config.json");

/// Nonzero Pixtral processor and tower reference from the published implementation.
pub const PIXTRAL_REFERENCE_JSON: &str =
    include_str!("../fixtures/nemotron_labs_diffusion/pixtral-numeric.json");
