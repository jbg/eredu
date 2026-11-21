//! Effective loaded parameter catalog assembled from retained portable declarations.

use super::*;
use eredu_checkpoint::{recipe::RecipeDtype, LinearFormat};
use eredu_nn::{LinearCompanionRole, ParameterMetadata, ParameterSlotVisitor, Tensor};
use std::marker::PhantomData;

#[derive(Clone)]
pub(super) struct EffectiveLayout {
    pub(super) physical: eredu_core::parameters::native::EffectiveParameterLayout,
    bank_loan: CaptureUsage,
}

impl EffectiveLayout {
    fn new(format: LinearFormat, shape: Vec<u64>) -> Self {
        Self {
            physical: eredu_core::parameters::native::EffectiveParameterLayout::new(format, shape),
            bank_loan: CaptureUsage::default(),
        }
    }

    /// Bounds host metadata for selecting this slot and its dependencies.
    pub fn selection_metadata_bytes(&self, id: &str) -> Result<u64, ParameterError> {
        let mut bytes = id.len() as u64;
        for companion in [
            LinearCompanionRole::Scale,
            LinearCompanionRole::AffineBias,
            LinearCompanionRole::GlobalScale,
        ]
        .into_iter()
        .filter_map(|role| self.physical.companion(role))
        {
            bytes = add(bytes, companion.len() as u64)?;
        }
        Ok(add(1024, mul(bytes, 8)?)?)
    }
    pub(super) fn price_loan(&self, native: CaptureUsage) -> Result<CaptureUsage, ParameterError> {
        Ok(native.checked_add(self.bank_loan)?)
    }
}

fn dtype(value: &RecipeDtype) -> Option<InterventionDtype> {
    match value {
        RecipeDtype::F32 => Some(InterventionDtype::Float32),
        RecipeDtype::F16 => Some(InterventionDtype::Float16),
        RecipeDtype::BF16 => Some(InterventionDtype::Bfloat16),
        _ => None,
    }
}
struct Catalog {
    layouts: BTreeMap<String, EffectiveLayout>,
    companions: Vec<(String, Option<String>, LinearCompanionRole)>,
    physical: BTreeMap<String, (RecipeDtype, Vec<u64>)>,
    slots: Vec<LoadedParameter>,
}
struct Visitor<'a, B, N>(&'a mut Catalog, PhantomData<fn() -> (B, N)>);

impl<B, N> ParameterSlotVisitor<N::ParameterValue> for Visitor<'_, B, N>
where
    B: TextGenerationBackend,
    N: ParameterMechanisms<B>,
{
    fn visit_slot(&mut self, metadata: ParameterMetadata, value: &mut N::ParameterValue) {
        self.0
            .record(metadata, N::parameter_dtype(value), shape(value));
    }
}

fn shape<T: Tensor>(value: &T) -> Vec<u64> {
    value.shape().iter().map(|n| *n as u64).collect()
}

impl Catalog {
    fn bind_bank_loan_costs(
        &mut self,
        prepared: &[PreparedParameterSlot],
    ) -> Result<(), ParameterError> {
        let mut slots = BTreeMap::<_, Vec<_>>::new();
        for slot in prepared {
            slots
                .entry(slot.parameter.id.as_str())
                .or_default()
                .push(slot);
        }
        for (id, layout) in &mut self.layouts {
            let mut ids = BTreeSet::from([id.clone()]);
            layout.physical.extend_dependencies(&mut ids);
            for slot in ids.iter().filter_map(|id| slots.get(id.as_str())).flatten() {
                layout.bank_loan = layout.bank_loan.checked_add(slot.bank_loan_usage()?)?;
            }
        }
        Ok(())
    }
    fn record(
        &mut self,
        metadata: ParameterMetadata,
        actual_dtype: RecipeDtype,
        actual_shape: Vec<u64>,
    ) {
        let id = metadata.id.as_str().to_string();
        if let Some(role) = metadata.linear_companion {
            self.companions.push((
                id.clone(),
                metadata.linear_companion_of.map(|owner| owner.to_string()),
                role,
            ));
        }
        let layout = self.layouts.entry(id.clone()).or_insert_with(|| {
            EffectiveLayout::new(eredu_checkpoint::LinearFormat::Dense, actual_shape.clone())
        });
        layout.physical.row_layout = metadata.linear_row_layout;
        self.slots.push(LoadedParameter {
            shared_id: metadata
                .alias_of
                .as_ref()
                .map_or_else(|| id.clone(), |alias| alias.as_str().to_string()),
            id: id.clone(),
            shape: layout.physical.shape.clone(),
            dtype: dtype(&actual_dtype),
            supported: false,
            access: None,
            condition: String::new(),
            input_transform: ProjectionInputTransform::Unspecified,
        });
        self.physical.insert(id, (actual_dtype, actual_shape));
    }
    fn finish<B: TextGenerationBackend, N: ParameterMechanisms<B>>(
        &mut self,
        originals: &ParameterTensors<B, N>,
        prepared: &[PreparedParameterSlot],
    ) -> Result<(), ParameterError> {
        // Owners and companions can arrive in either traversal order.
        for (id, owner, role) in &self.companions {
            if let Some(layout) = owner.as_ref().and_then(|owner| self.layouts.get_mut(owner)) {
                layout.physical.bind_companion(*role, id.clone());
            }
        }
        self.bind_bank_loan_costs(prepared)?;
        let companions: BTreeSet<_> = self
            .companions
            .iter()
            .map(|(id, _, _)| id.as_str())
            .collect();
        for slot in &mut self.slots {
            let layout = &self.layouts[&slot.id];
            let (physical_dtype, shape) = &self.physical[&slot.id];
            slot.supported = !companions.contains(slot.id.as_str())
                && N::effective_parameter_supported(&layout.physical, shape, physical_dtype);
            if slot.supported {
                slot.input_transform = if originals.contains_key(&slot.id) {
                    ProjectionInputTransform::Identity
                } else {
                    N::parameter_input_transform(&layout.physical, physical_dtype)
                };
                slot.dtype = Some(dtype(physical_dtype).unwrap_or(InterventionDtype::Float32));
                slot.condition = if layout.physical.format == eredu_checkpoint::LinearFormat::Dense
                {
                    "Loaded dense values after materialization; edits preserve the native floating dtype".into()
                } else {
                    format!(
                        "Effective {:?} values; queries decode with F32 arithmetic; editing retains the original packed storage and publishes a complete F32 copy of each affected parameter only, with no requantization",
                        layout.physical.format
                    )
                };
            } else {
                slot.condition = "Effective access requires a floating parameter or a supported packed matrix with its declared companions; companion slots are not independently editable".into();
            }
        }
        // Module implementations may traverse hash-backed parameter maps.
        // Public discovery remains deterministic across idle inspections.
        self.slots
            .sort_unstable_by(|left, right| left.id.cmp(&right.id));
        let mut ids = BTreeSet::new();
        if self.slots.iter().any(|slot| !ids.insert(&slot.id)) {
            return Err(ParameterError::Invalid(
                "duplicate loaded parameter identity".into(),
            ));
        }
        Ok(())
    }
}
pub(super) fn local_parameter_slots<B: TextGenerationBackend, N: ParameterMechanisms<B>>(
    model: &mut Model<B, N>,
    originals: &ParameterTensors<B, N>,
) -> Result<(Vec<LoadedParameter>, BTreeMap<String, EffectiveLayout>), ParameterError> {
    let layouts = model
        .as_ref()
        .parameter_materialization_tasks()
        .iter()
        .flat_map(|task| {
            std::iter::once(task.name().to_string())
                .chain(task.aliases().iter().cloned())
                .map(move |name| {
                    (
                        name,
                        EffectiveLayout::new(
                            task.executable(),
                            task.logical_shape().iter().map(|n| *n as u64).collect(),
                        ),
                    )
                })
        })
        .collect();
    let mut catalog = Catalog {
        layouts,
        companions: Vec::new(),
        physical: BTreeMap::new(),
        slots: vec![],
    };
    if !model
        .as_mut()
        .visit_loaded_parameters(&mut Visitor::<B, N>(&mut catalog, PhantomData))
    {
        let prepared = model.as_ref().prepared_parameter_slots();
        if prepared.is_empty() {
            return Err(ParameterError::Unsupported(
                "selected execution has no prepared parameter access mechanism".into(),
            ));
        }
        for slot in prepared {
            let physical_dtype = N::prepared_parameter_dtype(&slot.materialized.dtype)?;
            catalog.record(
                slot.parameter.clone(),
                physical_dtype,
                slot.materialized.shape.iter().map(|n| *n as u64).collect(),
            );
        }
    }
    catalog.finish::<B, N>(originals, model.as_ref().prepared_parameter_slots())?;
    Ok((catalog.slots, catalog.layouts))
}

pub(super) fn partition_parameter_slots<B: TextGenerationBackend, N: ParameterMechanisms<B>>(
    originals: &ParameterTensors<B, N>,
    slots: &[PreparedParameterSlot],
    placements: BTreeMap<&str, ParameterSlotLayout>,
    state_reset: bool,
) -> Result<(Vec<LocalParameterFacts>, BTreeMap<String, EffectiveLayout>), ParameterError> {
    let mut catalog = Catalog {
        layouts: BTreeMap::new(),
        companions: Vec::new(),
        physical: BTreeMap::new(),
        slots: Vec::new(),
    };
    let mut global_shapes = BTreeMap::new();
    for (id, layout) in placements {
        catalog.layouts.insert(
            id.into(),
            EffectiveLayout::new(layout.format, layout.local_shape),
        );
        global_shapes.insert(id, layout.global_shape);
    }
    for slot in slots {
        let id = slot.parameter.id.as_str();
        let (physical_dtype, shape) = if let Some(original) = originals.get(id) {
            let actual = N::parameter_dtype(original);
            let physical = if dtype(&actual).is_some() {
                actual
            } else {
                RecipeDtype::F32
            };
            (physical, catalog.layouts[id].physical.shape.clone())
        } else {
            (
                N::prepared_parameter_dtype(&slot.materialized.dtype)?,
                slot.materialized.shape.iter().map(|n| *n as u64).collect(),
            )
        };
        catalog.record(slot.parameter.clone(), physical_dtype, shape);
    }
    catalog.finish::<B, N>(originals, slots)?;
    let facts = catalog
        .slots
        .into_iter()
        .map(|mut parameter| {
            let local_shape = parameter.shape.clone();
            parameter.shape = global_shapes
                .remove(parameter.id.as_str())
                .expect("prepared global shape");
            parameter.access = Some(ParameterAccess {
                query: parameter.supported,
                projection: parameter.supported,
                replacement: parameter.supported && state_reset,
            });
            if parameter.supported && !state_reset {
                parameter.condition.push_str(
                    "; distributed replacement requires a bounded fresh-state exchange mechanism",
                );
            }
            parameter.supported &= state_reset;
            LocalParameterFacts {
                parameter,
                local_shape,
            }
        })
        .collect();

    Ok((facts, catalog.layouts))
}
