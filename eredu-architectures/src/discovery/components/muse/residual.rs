//! Channelwise learned or stored residual coefficients in decoder order.
use super::*;

fn parameter(slot: &eredu_nn::ParameterSpec) -> ComponentTransformParameter {
    let name = slot.id.as_str().to_owned();
    declared_transform_parameter(
        slot,
        format!(
            "parameters:{}",
            name.rsplit_once('.').expect("declared residual slot").0
        ),
    )
}
pub(super) fn attach(g: &mut Builder, c: &DecoderConfig, layer: usize) {
    use eredu_nn::DeclareLogicalParameters;
    let scalar = c.scalar(layer).expect("validated Muse scalar declarations");
    for (branch, spec) in ["attention", "feed_forward"]
        .into_iter()
        .zip(&scalar.residuals)
    {
        for declaration in spec.logical_parameters().expect("validated residual slots") {
            let prefix = declaration
                .parameter
                .id
                .as_str()
                .rsplit_once('.')
                .expect("declared residual slot")
                .0;
            g.attach_parameters(&format!("decoder.{layer}.{branch}"), prefix);
        }
    }
}
pub(super) fn cycles(c: &DecoderConfig) -> Vec<ComponentResidualCycle> {
    use crate::muse::equations::specs::ResidualSpec;
    let mut cycles = Vec::new();
    for layer in 0..c.num_hidden_layers as usize {
        let path = format!("model.layers.{layer}");
        let scalar = c.scalar(layer).expect("validated Muse scalar declarations");
        for ((branch, input), spec) in [
            ("attention", format!("{path}.input.effective")),
            (
                "feed_forward",
                format!("{path}.attention.residual.effective"),
            ),
        ]
        .into_iter()
        .zip(&scalar.residuals)
        {
            let combination = match spec {
                ResidualSpec::Add => continue,
                ResidualSpec::Learned(spec) => {
                    ComponentResidualCombination::LearnedVariancePreserving {
                        gate: parameter(&spec.gate),
                        temperature: ComponentScalar::new(spec.temperature),
                        minimum_variance: ComponentScalar::new(spec.minimum_variance),
                    }
                }
                ResidualSpec::Stored(spec) => ComponentResidualCombination::StoredCoefficients {
                    alpha: parameter(&spec.alpha),
                    beta: parameter(&spec.beta),
                },
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
