//! Bounded addressable-bank transformation plans and consuming binding rewrites.
use super::{
    AddressableBankEntry, AddressableBankEntryError, AddressableBankMemberError,
    AddressableBindingTransform,
};
use crate::{OffloadUnit, ParameterBankKey, ResidencyDeclarationError, WeightBinding};
use eredu_checkpoint::{
    quantization::{BoundedQuantizationPlan, BoundedQuantizationTarget},
    recipe::RecipeError,
    store::{CheckpointSource, StoreError, TensorSelection},
};
use std::collections::{BTreeMap, BTreeSet};

/// One conversion batch sharing the exact selected encoding and companion
/// precision. The selected members remain independently leased storage units.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddressableBankTransformGroup {
    transform: AddressableBindingTransform,
    selected: BTreeSet<(ParameterBankKey, String)>,
}

impl AddressableBankTransformGroup {
    /// Exact encoding and precision retained for this conversion batch.
    pub const fn transform(&self) -> &AddressableBindingTransform {
        &self.transform
    }
    /// Independently identified member-local bindings in canonical key order.
    pub const fn selected(&self) -> &BTreeSet<(ParameterBankKey, String)> {
        &self.selected
    }
    /// Consumes the conversion selection without rebuilding its resource mapping.
    pub fn into_parts(
        self,
    ) -> (
        AddressableBindingTransform,
        BTreeSet<(ParameterBankKey, String)>,
    ) {
        (self.transform, self.selected)
    }
}

/// Ordered conversion batches and format reporting from the same retained
/// selection. Groups follow first use in canonical bank-key and binding order;
/// equal encodings with different companion precision remain separate groups.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddressableBankTransformPlan {
    groups: Vec<AddressableBankTransformGroup>,
    formats: Vec<eredu_checkpoint::WeightQuantization>,
}

impl AddressableBankTransformPlan {
    /// Projects selected transformations without selecting mechanisms or granting
    /// format support. Task admission and bounded plan validation remain separate.
    pub fn new(
        transformations: BTreeMap<(ParameterBankKey, String), AddressableBindingTransform>,
    ) -> Self {
        let mut groups = Vec::<AddressableBankTransformGroup>::new();
        let mut formats = Vec::new();
        for (binding, transform) in transformations {
            if !formats.contains(&transform.quantization()) {
                formats.push(transform.quantization());
            }
            if let Some(group) = groups.iter_mut().find(|group| group.transform == transform) {
                group.selected.insert(binding);
            } else {
                groups.push(AddressableBankTransformGroup {
                    transform,
                    selected: std::iter::once(binding).collect(),
                });
            }
        }
        Self { groups, formats }
    }
    /// Ordered encoding/precision groups used by materialization.
    pub fn groups(&self) -> &[AddressableBankTransformGroup] {
        &self.groups
    }
    /// Distinct encodings in first-use order, independent of precision grouping.
    pub fn formats(&self) -> &[eredu_checkpoint::WeightQuantization] {
        &self.formats
    }
    /// Consumes the exact conversion batches.
    pub fn into_groups(self) -> Vec<AddressableBankTransformGroup> {
        self.groups
    }
}

/// Failure preparing or binding a selected bank transformation.
#[derive(Debug, thiserror::Error)]
pub enum AddressableBankQuantizationError {
    /// No selected floating binding has local transform companions.
    #[error("entry catalog contains no floating projection bindings to quantize")]
    EmptySelection,
    /// Final independently leased entry declaration failed validation.
    #[error(transparent)]
    Entry(#[from] AddressableBankEntryError),
    /// Source recipe inference failed.
    #[error(transparent)]
    Recipe(#[from] RecipeError),
    /// Quantization plan admission or final source metadata failed.
    #[error(transparent)]
    Store(#[from] StoreError),
    /// A rebuilt binding or independently leased unit is invalid.
    #[error(transparent)]
    Declaration(#[from] ResidencyDeclarationError),
    /// Selected packed geometry is invalid.
    #[error(transparent)]
    Geometry(#[from] AddressableBankMemberError),
    /// The bounded prepared catalog extent overflowed.
    #[error("packed entry catalog size overflowed")]
    CatalogByteOverflow,
    /// Final binding extents overflowed for one independently leased member.
    #[error("quantized bank member {key:?} byte total overflowed")]
    MemberByteOverflow {
        /// Original global member identity.
        key: ParameterBankKey,
    },
}

#[derive(Debug)]
struct QuantizedBinding {
    target: BoundedQuantizationTarget,
    scale: String,
    affine_bias: Option<String>,
}

/// Consuming rewrite from original independent units to final packed source keys.
/// Unselected bindings retain their exact declarations and source recipes.
#[derive(Debug)]
pub struct AddressableBankQuantizationBindings {
    units: Vec<(ParameterBankKey, OffloadUnit)>,
    replacements: BTreeMap<(ParameterBankKey, String), QuantizedBinding>,
}

/// Prepares exact bounded targets without reading payloads or performing native
/// conversion. Synthetic output names are namespaced by global storage identity;
/// checkpoint names and family precision policy are already selected inputs.
/// Returns the checkpoint plan and a move-only rewrite for its resulting source.
pub fn prepare_addressable_bank_quantization(
    source: &dyn CheckpointSource,
    entries: impl IntoIterator<Item = AddressableBankEntry>,
    transform: AddressableBindingTransform,
    selected: &BTreeSet<(ParameterBankKey, String)>,
    max_working_set_bytes: u64,
) -> Result<
    (BoundedQuantizationPlan, AddressableBankQuantizationBindings),
    AddressableBankQuantizationError,
> {
    let entries = entries.into_iter();
    let mut units = Vec::with_capacity(entries.size_hint().0);
    let mut targets = Vec::new();
    let mut replacements = BTreeMap::new();
    let mut packed_catalog_bytes = 0u64;
    for entry in entries {
        let (identity, unit, _) = entry.into_parts();
        for binding in unit.bindings() {
            let Some(companions) = binding.quantization_companions() else {
                continue;
            };
            if !selected.contains(&(identity, binding.name().to_owned())) {
                continue;
            }
            let recipe = binding.source_recipe();
            let metadata = recipe.infer(source)?;
            let prefix = format!(
                "__eredu.entry.bank.{:05}.unit.{:05}.global.{:05}.{}",
                identity.bank(),
                identity.unit(),
                identity.member(),
                binding.name()
            );
            let target = BoundedQuantizationTarget::from_recipe(
                format!("{prefix}.weight"),
                format!("{prefix}.scales"),
                companions.affine_bias().map(|_| format!("{prefix}.biases")),
                recipe,
            )?
            .with_affine_companion_dtype(transform.companion_dtype().clone())?;
            packed_catalog_bytes = packed_catalog_bytes
                .checked_add(super::addressable_projection_bytes(
                    binding.name(),
                    metadata.shape(),
                    transform.quantization(),
                    transform.companion_dtype(),
                )?)
                .ok_or(AddressableBankQuantizationError::CatalogByteOverflow)?;
            replacements.insert(
                (identity, binding.name().to_owned()),
                QuantizedBinding {
                    target: target.clone(),
                    scale: companions.scale().to_owned(),
                    affine_bias: companions.affine_bias().map(str::to_owned),
                },
            );
            targets.push(target);
        }
        units.push((identity, unit));
    }
    if targets.is_empty() {
        return Err(AddressableBankQuantizationError::EmptySelection);
    }
    let plan = BoundedQuantizationPlan::new(
        transform.quantization(),
        max_working_set_bytes.min(packed_catalog_bytes),
        targets,
    )?;
    Ok((
        plan,
        AddressableBankQuantizationBindings {
            units,
            replacements,
        },
    ))
}

impl AddressableBankQuantizationBindings {
    /// Consumes this rewrite against the materializer's final source metadata.
    /// This allocates no tensor and preserves independent units and binding order.
    pub fn bind(
        self,
        source: &dyn CheckpointSource,
    ) -> Result<Vec<AddressableBankEntry>, AddressableBankQuantizationError> {
        let mut rebuilt = Vec::with_capacity(self.units.len());
        for (identity, unit) in self.units {
            let mut bindings = Vec::new();
            for binding in unit.bindings() {
                let Some(replacement) = self
                    .replacements
                    .get(&(identity, binding.name().to_owned()))
                else {
                    bindings.push(binding.clone());
                    continue;
                };
                bindings.push(packed_binding(
                    binding.name(),
                    replacement.target.weight_name(),
                    source,
                )?);
                bindings.push(packed_binding(
                    &replacement.scale,
                    replacement.target.scales_name(),
                    source,
                )?);
                if let Some(bias) = &replacement.affine_bias {
                    bindings.push(packed_binding(
                        bias,
                        replacement
                            .target
                            .biases_name()
                            .expect("affine target declares bias identity"),
                        source,
                    )?);
                }
            }
            let bytes = bindings.iter().try_fold(0u64, |total, binding| {
                total
                    .checked_add(binding.expected_bytes())
                    .ok_or(AddressableBankQuantizationError::MemberByteOverflow { key: identity })
            })?;
            rebuilt.push(AddressableBankEntry::new(
                identity,
                OffloadUnit::new(identity.unit_id(), bindings)?,
                bytes,
            )?);
        }
        Ok(rebuilt)
    }
}

fn packed_binding(
    local_name: &str,
    checkpoint_key: &str,
    source: &dyn CheckpointSource,
) -> Result<WeightBinding, AddressableBankQuantizationError> {
    let metadata = source.source_metadata(checkpoint_key)?;
    Ok(WeightBinding::new(
        local_name,
        checkpoint_key,
        TensorSelection::Full,
        metadata.encoded_byte_len,
    )?)
}
