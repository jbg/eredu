//! Validated import of existing ordinary attention and fixed-state metadata.

use super::*;
use eredu_core::cache::StateTensorDtype;
use eredu_nn::workspace::WorkspaceDtype;

const NEGATIVE_FRONTIER: &str = "negative existing state frontier";
const INVALID_ROLE: &str = "unknown or duplicated existing fixed-state role";
const OMITTED_ROLE: &str = "existing fixed-state inventory omitted a declared role";
const UNEXPECTED_ATTENTION: &str = "existing attention storage on a layer without attention";
const ATTENTION_GEOMETRY: &str = "existing attention storage differs from selected geometry";
const FIXED_GEOMETRY: &str = "existing fixed-state shape or dtype differs from its declaration";

impl WorkspaceConcatStateFactory {
    /// The same fixed diagnostics emitted by projection. The native adapter
    /// wraps direct StateError once; already-counted workspace errors pass through.
    pub fn projection_error_control_bytes(policy: &LayerCachePolicy) -> Option<usize> {
        let reasons = [
            NEGATIVE_FRONTIER,
            INVALID_ROLE,
            OMITTED_ROLE,
            UNEXPECTED_ATTENTION,
            ATTENTION_GEOMETRY,
            FIXED_GEOMETRY,
            eredu_core::cache::StateTensorDimensionError.diagnostic(),
        ];
        let length = reasons.into_iter().map(str::len).max()?;
        Some(invalid_layer_control_bytes(length)?.max(construction_error_control_bytes(policy)?))
    }

    /// Imports a complete layer inventory projected from the retained native
    /// state. No prefix replay, tensor allocation or equation execution occurs.
    /// Native composition must first verify the exact-concatenation mechanism;
    /// geometry and symbolic fixed-state policy remain architecture-owned.
    pub fn project_layer(
        &self,
        layer: usize,
        policy: &LayerCachePolicy,
        position: i32,
        keys: Option<WorkspaceTensor>,
        values: Option<WorkspaceTensor>,
        fixed: impl IntoIterator<Item = (StateTensorRole, Option<WorkspaceTensor>)>,
    ) -> Result<WorkspaceConcatLayerState, StateError> {
        self.project_layer_impl(layer, policy, position, keys, values, fixed, false)
    }

    pub(in crate::working_memory) fn project_pooling_local(
        &self,
        layer: usize,
        policy: &LayerCachePolicy,
        position: i32,
        keys: Option<WorkspaceTensor>,
        values: Option<WorkspaceTensor>,
    ) -> Result<WorkspaceConcatLayerState, StateError> {
        self.project_layer_impl(layer, policy, position, keys, values, [], true)
    }

    fn project_layer_impl(
        &self,
        layer: usize,
        policy: &LayerCachePolicy,
        position: i32,
        keys: Option<WorkspaceTensor>,
        values: Option<WorkspaceTensor>,
        fixed: impl IntoIterator<Item = (StateTensorRole, Option<WorkspaceTensor>)>,
        pooling_local: bool,
    ) -> Result<WorkspaceConcatLayerState, StateError> {
        let invalid = |reason: &str| {
            StateError::workspace_invalid_layer(&self.context, layer, format_args!("{reason}"))
        };
        if position < 0 {
            return Err(invalid(NEGATIVE_FRONTIER));
        }
        let mut result = if pooling_local {
            self.create_pooling_local(layer, policy)?
        } else {
            self.create_layer(layer, policy)?
        };
        import_fixed(&mut result.fixed, fixed, &self.context, &invalid)?;
        self.context
            .validate_values(
                keys.iter()
                    .chain(&values)
                    .chain(result.fixed.values().flatten()),
            )
            .map_err(StateError::WorkspaceConstruction)?;
        match result.attention {
            None if keys.is_some() || values.is_some() => {
                return Err(invalid(UNEXPECTED_ATTENTION));
            }
            Some(geometry) => {
                let count = geometry
                    .window
                    .map_or(position, |window| position.min(window - 1));
                let valid = |value: &WorkspaceTensor, width| {
                    value.shape() == [self.batch, geometry.heads, count, width]
                        && value.layout().dtype() == WorkspaceDtype::Float32
                };
                if (count == 0 && (keys.is_some() || values.is_some()))
                    || (count > 0
                        && !keys
                            .as_ref()
                            .is_some_and(|value| valid(value, geometry.width)))
                    || (geometry.key_only && values.is_some())
                    || (!geometry.key_only
                        && count > 0
                        && !values
                            .as_ref()
                            .is_some_and(|value| valid(value, geometry.value_width)))
                {
                    return Err(invalid(ATTENTION_GEOMETRY));
                }
            }
            None => {}
        }
        if !pooling_local {
            validate_fixed(&result.fixed, policy, self.batch, position, &invalid)?;
        }
        result.position = position;
        result.keys = keys;
        result.values = values;
        Ok(result)
    }
}

// These helpers describe actual formatting/typed-source constructors, not a
// diagnostic reserve. Only scalar fmt arguments are accepted by the callers.
pub(super) fn invalid_layer_control_bytes(length: usize) -> Option<usize> {
    WorkspaceContext::metadata_string_bytes(length)?
        .checked_add(WorkspaceContext::metadata_source_bytes::<StateError>()?)
}
fn formatted_length(args: std::fmt::Arguments<'_>) -> Option<usize> {
    struct Counter(usize);
    impl std::fmt::Write for Counter {
        fn write_str(&mut self, value: &str) -> std::fmt::Result {
            self.0 = self.0.checked_add(value.len()).ok_or(std::fmt::Error)?;
            Ok(())
        }
    }
    let mut counter = Counter(0);
    std::fmt::write(&mut counter, args).ok()?;
    Some(counter.0)
}
pub(super) fn construction_error_control_bytes(policy: &LayerCachePolicy) -> Option<usize> {
    let reasons = [
        HEAD_COUNT,
        HEAD_WIDTH,
        VALUE_WIDTH,
        COMPRESSED_MECHANISM,
        POOLING_LOCAL,
    ];
    let mut length = reasons.into_iter().map(str::len).max()?;
    if let Some(Err(cause)) = policy.attention().map(|value| value.sliding_window_i32()) {
        length = length.max(formatted_length(format_args!("{cause}"))?);
    }
    invalid_layer_control_bytes(length)
}
pub(super) const BATCH_EXTENT: &str = "workspace state batch exceeds tensor extent";
pub(super) const HEAD_COUNT: &str = "head count exceeds tensor extent";
pub(super) const HEAD_WIDTH: &str = "head width exceeds tensor extent";
pub(super) const VALUE_WIDTH: &str = "value width exceeds tensor extent";
pub(super) const COMPRESSED_MECHANISM: &str =
    "compressed state requires its selected compressed-cache metadata mechanism";
pub(super) const POOLING_LOCAL: &str = "pooling local state requires one sliding key head";

// One fixed-role ingestion and shape validator for concat and paged layers.
// Keep ingestion before attention validation and declaration checks afterwards
// in the ordinary adapter so its original first-error order is unchanged.
pub(super) fn import_fixed(
    slots: &mut FixedSlots,
    fixed: impl IntoIterator<Item=(StateTensorRole, Option<WorkspaceTensor>)>,
    context: &WorkspaceContext,
    invalid: &impl Fn(&str)->StateError,
) -> Result<(), StateError> {
        let mut seen = context
            .metadata_vec(slots.len())
            .map_err(StateError::WorkspaceConstruction)?;
        seen.resize(slots.len(), false);
        let mut supplied = 0usize;
        for (role, value) in fixed {
            let Some(index) = slots.index(&role) else {
                return Err(invalid(INVALID_ROLE));
            };
            if std::mem::replace(&mut seen[index], true) {
                return Err(invalid(INVALID_ROLE));
            }
            *slots
                .values_mut(&context)
                .map_err(StateError::WorkspaceConstruction)?
                .get_mut(index)
                .expect("validated role index") = value;
            supplied += 1;
        }
        if supplied != slots.len() {
            return Err(invalid(OMITTED_ROLE));
        }
    Ok(())
}
pub(super) fn validate_fixed(
    slots: &FixedSlots, policy: &LayerCachePolicy, batch: i32, position: i32,
    invalid: &impl Fn(&str)->StateError,
) -> Result<(), StateError> {
        for declaration in policy.fixed_state() {
            let value = slots
                .get(&declaration.role)
                .expect("complete role inventory");
            match value {
                Some(value) => {
                    // Resolve every dimension before reporting a shape mismatch,
                    // preserving the allocating resolver's first-error order.
                    let mut count = 0usize;
                    let mut same_shape = true;
                    for dimension in
                        declaration.resolved_dimensions(batch as usize, position as usize)
                    {
                        let dimension = dimension.map_err(|cause| invalid(cause.diagnostic()))?;
                        same_shape &= value.shape().get(count).copied() == Some(dimension);
                        count += 1;
                    }
                    same_shape &= count == value.shape().len();
                    let dtype = match declaration.dtype {
                        StateTensorDtype::Floating | StateTensorDtype::Float32 => {
                            WorkspaceDtype::Float32
                        }
                        StateTensorDtype::Int32 => WorkspaceDtype::Int32,
                        StateTensorDtype::Uint32 => WorkspaceDtype::Uint32,
                    };
                    if !same_shape || value.layout().dtype() != dtype {
                        return Err(invalid(FIXED_GEOMETRY));
                    }
                }
                None => {}
            }
        }
    Ok(())
}
