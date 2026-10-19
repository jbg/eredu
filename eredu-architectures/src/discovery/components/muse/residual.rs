//! Channelwise learned or stored residual coefficients in decoder order.
use super::*;

pub(super) fn attach(g: &mut Builder, c: &DecoderConfig, layer: usize) {
    if matches!(c.residual_equation, crate::muse::ResidualEquation::Add) {
        return;
    }
    for (branch, field) in [
        ("attention", "post_attention_residual_gate"),
        ("feed_forward", "post_feedforward_residual_gate"),
    ] {
        g.attach_parameters(
            &format!("decoder.{layer}.{branch}"),
            &format!("model.layers.{layer}.{field}"),
        );
    }
}

pub(super) fn cycles(c: &DecoderConfig) -> Vec<ComponentResidualCycle> {
    let mut cycles = Vec::new();
    for layer in 0..c.num_hidden_layers as usize {
        let path = format!("model.layers.{layer}");
        for (branch, field, input) in [
            (
                "attention",
                "post_attention_residual_gate",
                format!("{path}.input.effective"),
            ),
            (
                "feed_forward",
                "post_feedforward_residual_gate",
                format!("{path}.attention.residual.effective"),
            ),
        ] {
            let parameter = |suffix| {
                let name = format!("{path}.{field}.{suffix}");
                ComponentTransformParameter {
                    parameter: name.clone(),
                    shared_parameter: name,
                    parameter_group: format!("parameters:{path}.{field}"),
                }
            };
            let combination = match c.residual_equation {
                crate::muse::ResidualEquation::Add => continue,
                crate::muse::ResidualEquation::Learned {
                    temperature,
                    minimum_variance,
                } => ComponentResidualCombination::LearnedVariancePreserving {
                    gate: parameter("gate"),
                    temperature: ComponentScalar::new(temperature),
                    minimum_variance: ComponentScalar::new(minimum_variance),
                },
                crate::muse::ResidualEquation::StoredCoefficients => {
                    ComponentResidualCombination::StoredCoefficients {
                        alpha: parameter("alpha"),
                        beta: parameter("beta"),
                    }
                }
            };
            cycles.push(ComponentResidualCycle {
                id: format!("decoder.{layer}.{branch}.residual_combination"),
                node_id: format!("decoder.{layer}.{branch}"),
                layer_index: layer,
                input,
                write: format!("{path}.{branch}.output.effective"),
                output: format!("{path}.{branch}.residual"),
                effective_output: format!("{path}.{branch}.residual.effective"),
                combination,
            });
        }
    }
    cycles
}
