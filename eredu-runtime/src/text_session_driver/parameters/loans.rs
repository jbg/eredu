//! Exact parameter owner selection and bounded loans for the portable driver.

use super::*;
use crate::parameter_operations::PreparedParameterLocation;
use eredu_nn::{ParameterMetadata, ParameterSlotVisitor};

struct SelectedValues<'a, T> {
    ids: &'a BTreeSet<String>,
    values: BTreeMap<String, T>,
}

impl<T: Clone> ParameterSlotVisitor<T> for SelectedValues<'_, T> {
    fn visit_slot(&mut self, metadata: ParameterMetadata, value: &mut T) {
        if self.ids.contains(metadata.id.as_str()) {
            self.values.insert(metadata.id.to_string(), value.clone());
        }
    }
}

pub(super) fn with_parameter_values<B, N, T>(
    model: &mut Model<B, N>,
    layout: &EffectiveLayout,
    parameter: &str,
    context: &Context<B, N>,
    operation: impl FnOnce(&BTreeMap<String, N::ParameterValue>) -> Result<T, B::Error>,
) -> Result<T, B::Error>
where
    B: TextGenerationBackend,
    N: ParameterMechanisms<B>,
{
    let invalid = |message: String| N::parameter_access_error(ParameterError::Invalid(message));
    let mut ids = BTreeSet::from([parameter.to_owned()]);
    layout.physical.extend_dependencies(&mut ids);
    let prepared = model.as_ref().prepared_parameter_slots();
    let mut owner = None;
    for id in &ids {
        let slot = prepared
            .iter()
            .find(|slot| slot.parameter.id.as_str() == id)
            .ok_or_else(|| invalid(format!("selected parameter {id} has no prepared owner")))?;
        if let Some(previous) = &owner {
            let both_static = matches!(
                (previous, &slot.location),
                (
                    PreparedParameterLocation::Static { .. },
                    PreparedParameterLocation::Static { .. }
                )
            );
            if !both_static && previous != &slot.location {
                return Err(invalid(
                    "one parameter operation spans unrelated residency units".into(),
                ));
            }
        }
        owner = Some(slot.location.clone());
    }
    let owner = owner.expect("the primary parameter is always selected");
    let mut operation = Some(operation);
    let mut output = None;
    let mut visit_values =
        |visit: &mut crate::parameter_operations::ParameterSlotTraversal<'_, N::ParameterValue>| {
            let mut selected = SelectedValues {
                ids: &ids,
                values: BTreeMap::new(),
            };
            visit(&mut selected);
            if selected.values.len() != ids.len() {
                return Err(invalid("prepared parameter topology changed".into()));
            }
            let operation = operation
                .take()
                .ok_or_else(|| invalid("parameter loan ran twice".into()))?;
            output = Some(operation(&selected.values)?);
            Ok(())
        };
    if !model
        .as_mut()
        .with_parameter_slots(&owner, &ids, &mut visit_values, context)?
    {
        return Err(invalid("selected parameter loan is unavailable".into()));
    }
    output.ok_or_else(|| invalid("parameter loan did not run".into()))
}

pub(super) fn prepare_replacements<'a, B, N, I>(
    model: &mut Model<B, N>,
    layouts: &BTreeMap<String, EffectiveLayout>,
    targets: impl Iterator<Item = (&'a str, I)>,
    context: &Context<B, N>,
) -> Result<Result<(ParameterTensors<B, N>, ParameterTensors<B, N>), ParameterError>, B::Error>
where
    B: TextGenerationBackend,
    N: ParameterMechanisms<B>,
    I: Iterator<Item = (&'a ParameterRegion, &'a ParameterUpdate)>,
{
    let mut original = BTreeMap::new();
    let mut replacement = BTreeMap::new();
    for (id, edits) in targets {
        let layout = &layouts[id];
        let result = with_parameter_values::<B, N, _>(model, layout, id, context, |selected| {
            N::prepare_replacement(selected, &layout.physical, id, edits, context)
        })?;
        let (before, after) = match result {
            Ok(pair) => pair,
            Err(error) => return Ok(Err(error)),
        };
        original.insert(id.into(), before);
        replacement.insert(id.into(), after);
    }
    Ok(Ok((original, replacement)))
}
