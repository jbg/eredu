//! Packaged K2 Horizon conformance inputs shared by backend and facade tests.
//! Source identities and publisher provenance accompany the embedded fixtures.

/// Independent NumPy numerical expectations and synthetic model parameters.
pub const NUMERICAL_REFERENCE_JSON: &str = include_str!("k2_horizon/numerical.json");
/// Publisher-template prompts and tokenization expectations.
pub const TEXT_REFERENCE_JSON: &str = include_str!("k2_horizon/text.json");
/// Publisher-generated tool-call outputs used for streaming parser conformance.
pub const TOOL_GENERATION_JSON: &str = include_str!("k2_horizon/tool-generation.json");
/// Released templates paired with their reference-case names.
pub const CHAT_TEMPLATES: [(&str, &str); 4] = [
    (
        "dense-safetensors",
        include_str!("k2_horizon/dense-safetensors.jinja"),
    ),
    ("dense-gguf", include_str!("k2_horizon/dense-gguf.jinja")),
    (
        "mova-safetensors",
        include_str!("k2_horizon/mova-safetensors.jinja"),
    ),
    ("mova-gguf", include_str!("k2_horizon/mova-gguf.jinja")),
];
