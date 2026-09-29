//! Geometry of independently advancing, lane-local immutable records.
use super::*;

/// Architecture-owned declaration for one named stream in each batch lane.
/// Padding may reduce the number of records. The exact frontier is persisted
/// separately; the prefix-dependent maximum is used for admission and validation.
#[derive(Debug, Clone, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct AppendStreamPolicy {
    slot: u32,
    width: NonZeroU32,
    dtype: StateTensorDtype,
    record_tokens: NonZeroU32,
}
impl AppendStreamPolicy {
    /// Declares at most one complete record per `record_tokens` processed tokens.
    pub fn new(
        slot: u32,
        width: i32,
        dtype: StateTensorDtype,
        record_tokens: i32,
    ) -> Result<Self, CachePolicyError> {
        let result = Self {
            slot,
            width: positive_u32(width, "append record width")?,
            dtype,
            record_tokens: positive_u32(record_tokens, "append record token ratio")?,
        };
        result.validate()?;
        Ok(result)
    }
    /// Stable identity within the owning layer.
    pub const fn slot(&self) -> u32 {
        self.slot
    }
    /// Scalars in one record.
    pub const fn width(&self) -> i32 {
        self.width.get() as i32
    }
    /// Accepted scalar representation.
    pub const fn dtype(&self) -> StateTensorDtype {
        self.dtype
    }
    /// Minimum processed tokens required per complete record.
    pub const fn record_tokens(&self) -> u32 {
        self.record_tokens.get()
    }
    /// Maximum lane-local frontier admitted by a processed prefix.
    pub fn maximum_records(&self, prefix: usize) -> usize {
        prefix / self.record_tokens.get() as usize
    }
    /// Validates deserialized geometry before backend state construction.
    pub fn validate(&self) -> Result<(), CachePolicyError> {
        if self.width.get() > i32::MAX as u32 || self.record_tokens.get() > i32::MAX as u32 {
            return Err(CachePolicyError::Invalid(
                "append stream geometry exceeds i32".into(),
            ));
        }
        Ok(())
    }
    /// Symbolic upper geometry used by cold selection, discovery and memory forecasts.
    pub fn component(&self) -> StateComponentPolicy {
        StateComponentPolicy {
            role: StateComponentRole::AppendStream { slot: self.slot },
            shape: vec![
                StateTensorDimension::Batch,
                StateTensorDimension::PrefixTokensDiv(self.record_tokens),
                StateTensorDimension::Fixed(self.width),
            ],
            dtype: self.dtype,
            residency: StateResidencyClass::SealablePaged,
            presence: StateTensorPresence::Optional,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn combined_components_have_bounded_lane_geometry_and_unique_slots() {
        let a = AppendStreamPolicy::new(3, 128, StateTensorDtype::Floating, 4).unwrap();
        let b = AppendStreamPolicy::new(7, 4, StateTensorDtype::Int32, 4).unwrap();
        let policy = LayerCachePolicy::key_value_with_state(
            AttentionPolicy::Full,
            2,
            256,
            vec![],
            vec![a.clone(), b],
        )
        .unwrap();
        let components = policy.components();
        assert_eq!(components.len(), 4);
        assert_eq!(components[2].role().stable_name(), "state.append.3");
        let bounds = components[2].element_bounds(2, 11).unwrap();
        assert_eq!(bounds.minimum, 0);
        assert_eq!(bounds.maximum, 2 * 2 * 128);
        assert!(LayerCachePolicy::key_value_with_state(
            AttentionPolicy::Full,
            2,
            256,
            vec![],
            vec![a.clone(), a]
        )
        .is_err());
        assert!(AppendStreamPolicy::new(0, 0, StateTensorDtype::Int32, 4).is_err());
        assert!(AppendStreamPolicy::new(0, 4, StateTensorDtype::Int32, 0).is_err());
    }
}
