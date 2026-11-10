//! Released expert layouts referencing the construction-owned packed projections.

use eredu_checkpoint::expert::{GatedProductExpertLayoutNames, IndependentGatedProductExpertNames};
use eredu_nn::PackedGatedProductParameters;

fn names(
    source: &PackedGatedProductParameters<'_>,
    separate: [String; 3],
    independent: Vec<IndependentGatedProductExpertNames>,
) -> GatedProductExpertLayoutNames {
    let gate_up = source.gate_up().parameter.id.as_str();
    let down = source.down().parameter.id.as_str();
    let [gate, up, separate_down] = separate;
    GatedProductExpertLayoutNames {
        target_gate_up: gate_up.into(),
        target_down: down.into(),
        packed_gate_up: gate_up.into(),
        packed_down: down.into(),
        separate_gate: gate,
        separate_up: up,
        separate_down,
        independent,
    }
}
pub(super) fn v3_names(
    source: &PackedGatedProductParameters<'_>,
    root: &str,
) -> GatedProductExpertLayoutNames {
    names(
        source,
        [
            format!("{root}.gate_proj"),
            format!("{root}.up_proj"),
            format!("{root}.down_proj"),
        ],
        (0..source.group_count())
            .map(|expert| IndependentGatedProductExpertNames {
                gate: format!("{root}.{expert}.gate_proj.weight"),
                up: format!("{root}.{expert}.up_proj.weight"),
                down: format!("{root}.{expert}.down_proj.weight"),
            })
            .collect(),
    )
}
pub(super) fn v4_names(
    source: &PackedGatedProductParameters<'_>,
    root: &str,
) -> GatedProductExpertLayoutNames {
    names(
        source,
        [
            format!("{root}.expert_banks.w1"),
            format!("{root}.expert_banks.w3"),
            format!("{root}.expert_banks.w2"),
        ],
        (0..source.group_count())
            .map(|expert| IndependentGatedProductExpertNames {
                gate: format!("{root}.experts.{expert}.w1.weight"),
                up: format!("{root}.experts.{expert}.w3.weight"),
                down: format!("{root}.experts.{expert}.w2.weight"),
            })
            .collect(),
    )
}
