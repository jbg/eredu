use super::*;
use eredu_nn::{GroupedLinearActivation, GroupedLinearOperator, GroupedLinearSpec};

pub(super) fn addressable_bank(
    plan: &ExpertRealizationPlan<GroupedLinearSpec>,
    catalog: &ExpertResidencyCatalog,
    selected: &[eredu_runtime::AddressableBankMember],
    capacity: usize,
    context: &NumericContext,
) -> Result<NumericGroupedBankMechanism, String> {
    let mut full_banks = BTreeMap::new();
    let mut linear = BTreeMap::new();
    let mut bytes = BTreeMap::new();
    for unit in catalog.units() {
        let address = (
            unit.owner_group().as_str().to_owned(),
            unit.identity().unit(),
        );
        if !full_banks.contains_key(&address) {
            let spec = plan
                .unit_spec(&address.0, address.1)
                .ok_or_else(|| format!("linear unit {address:?} has no plan"))?;
            full_banks.insert(
                address.clone(),
                NumericLinearGroups::new(spec.clone(), context)
                    .map_err(|error| error.to_string())?,
            );
        }
        let full = &full_banks[&address];
        let spec = full
            .spec
            .clone()
            .with_group_count(1)
            .map_err(|error| error.to_string())?;
        linear.insert(
            unit.identity(),
            full.selected(&spec, &[unit.identity().member()])
                .map_err(|error| error.to_string())?,
        );
        bytes.insert(
            unit.identity(),
            selected
                .iter()
                .find(|member| member.key() == unit.identity())
                .map(eredu_runtime::AddressableBankMember::selected_bytes)
                .or(unit.byte_len())
                .ok_or_else(|| "linear unit has no admitted byte geometry".to_owned())?,
        );
    }
    Ok(NumericGroupedBankMechanism {
        banks: BTreeMap::new(),
        linear,
        bytes,
        report: NumericBankReport::default(),
        resident: Vec::new(),
        capacity,
    })
}

include!("../support/numeric/grouped_linear.rs");
