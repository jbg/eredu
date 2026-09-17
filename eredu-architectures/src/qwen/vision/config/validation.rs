//! Same vision predicates and ordered diagnostics with caller-owned destinations.
use super::{VisionAttentionPolicy, VisionConfig, VisionConfigError, VisionMode};
use std::fmt;

pub(super) trait Destination {
    type Error;
    fn controls<T>(&self) -> Result<(), Self::Error>;
    fn vector(&self, count: usize) -> Result<Vec<u32>, Self::Error>;
    fn text(&self, text: fmt::Arguments<'_>) -> Result<String, Self::Error>;
    fn error(&self, text: fmt::Arguments<'_>) -> Self::Error;
}
pub(super) struct Ordinary;
impl Destination for Ordinary {
    type Error = VisionConfigError;
    fn controls<T>(&self) -> Result<(), Self::Error> {
        Ok(())
    }
    fn vector(&self, count: usize) -> Result<Vec<u32>, Self::Error> {
        Ok(Vec::with_capacity(count))
    }
    fn text(&self, text: fmt::Arguments<'_>) -> Result<String, Self::Error> {
        Ok(text.to_string())
    }
    fn error(&self, text: fmt::Arguments<'_>) -> Self::Error {
        VisionConfigError::Invalid(text.to_string())
    }
}
pub(super) struct Checked<'a>(pub(super) &'a eredu_nn::workspace::WorkspaceContext);
impl Destination for Checked<'_> {
    type Error = eredu_nn::Error;
    fn controls<T>(&self) -> Result<(), Self::Error> {
        let controls = [
            std::mem::size_of::<T>(),
            std::mem::size_of::<Result<T, Self::Error>>(),
            std::mem::size_of::<Self>(),
        ];
        let bytes = controls
            .into_iter()
            .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
        self.0.charge_metadata(bytes)?;
        Ok(())
    }
    fn vector(&self, count: usize) -> Result<Vec<u32>, Self::Error> {
        self.0.metadata_vec(count)
    }
    fn text(&self, text: fmt::Arguments<'_>) -> Result<String, Self::Error> {
        self.0.metadata_string(text)
    }
    fn error(&self, text: fmt::Arguments<'_>) -> Self::Error {
        self.0.metadata_error(text)
    }
}

pub(super) fn validate<D: Destination>(
    config: &VisionConfig,
    destination: &D,
) -> Result<(), D::Error> {
    destination.controls::<(
        Vec<u32>,
        String,
        [(&str, i32); 9],
        Result<(), eredu_checkpoint::EncodingValidationError>,
    )>()?;
    for (name, value) in [
        ("hidden_size", config.hidden_size),
        ("intermediate_size", config.intermediate_size),
        ("num_heads", config.num_heads),
        ("num_position_embeddings", config.num_position_embeddings),
        ("in_channels", config.in_channels),
        ("patch_size", config.patch_size),
        ("spatial_merge_size", config.spatial_merge_size),
        ("temporal_patch_size", config.temporal_patch_size),
        ("out_hidden_size", config.out_hidden_size),
    ] {
        if value <= 0 {
            return Err(
                destination.error(format_args!("vision {name} must be positive, got {value}"))
            );
        }
    }
    if config.hidden_size % config.num_heads != 0 {
        return Err(destination.error(format_args!(
            "vision hidden_size {} is not divisible by num_heads {}",
            config.hidden_size, config.num_heads
        )));
    }
    if !matches!(
        config.hidden_act.as_str(),
        "silu" | "gelu" | "gelu_pytorch_tanh"
    ) {
        return Err(destination.error(format_args!(
            "unsupported vision activation {:?}",
            config.hidden_act
        )));
    }
    let mut mergers = destination.vector(config.deepstack_layer_count())?;
    for merger in config
        .layer_schedule
        .iter()
        .filter_map(|policy| policy.deepstack_merger)
    {
        mergers.push(merger);
    }
    mergers.sort_unstable();
    if mergers
        .iter()
        .enumerate()
        .any(|(index, merger)| *merger != index as u32)
    {
        return Err(destination.error(format_args!(
            "DeepStack merger banks must be unique and contiguous from zero, got {mergers:?}"
        )));
    }
    if config.mode == VisionMode::DeepStack
        && config
            .layer_schedule
            .iter()
            .any(|policy| policy.attention != VisionAttentionPolicy::Full)
    {
        return Err(destination.error(format_args!(
            "DeepStack vision schedules require full attention in every block"
        )));
    }
    if config
        .layer_schedule
        .iter()
        .any(|policy| policy.attention == VisionAttentionPolicy::Windowed)
        && config.window_size <= 0
    {
        return Err(destination.error(format_args!(
            "windowed vision schedules require a positive window_size"
        )));
    }
    for (name, format) in &config.linear_formats {
        if name.trim().is_empty() {
            return Err(destination.error(format_args!(
                "vision linear-format identity must not be empty"
            )));
        }
        let relative = name.strip_prefix("model.visual.").unwrap_or(name);
        if config
            .linear_formats
            .get(relative)
            .is_some_and(|alias| alias != format)
            || {
                let full_name = destination.text(format_args!("model.visual.{relative}"))?;
                config
                    .linear_formats
                    .get(&full_name)
                    .is_some_and(|alias| alias != format)
            }
        {
            return Err(
                destination.error(format_args!("conflicting vision linear formats for {name}"))
            );
        }
        format
            .validate_fixed()
            .map_err(|error| destination.error(format_args!("{error}")))?;
    }
    Ok(())
}

pub(super) fn validate_for<D: Destination>(
    config: &VisionConfig,
    expected: VisionMode,
    destination: &D,
) -> Result<(), D::Error> {
    destination.controls::<(VisionMode, Result<(), D::Error>)>()?;
    if config.mode != expected {
        return Err(destination.error(format_args!(
            "vision mode {:?} does not match required {:?} semantics",
            config.mode, expected,
        )));
    }
    validate(config, destination)
}
