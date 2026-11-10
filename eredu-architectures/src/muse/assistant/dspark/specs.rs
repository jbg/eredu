//! Construction-owned scoring geometry and independent acquisition boundaries.

use eredu_nn::{
    DeclareLogicalParameters, EmbeddingSpec, Error, LinearSpec, LogicalParameterDeclaration,
    LogicalParameterRole, ParameterSpec, TensorElementType,
};

use super::DsparkHeadGeometry;

#[derive(Debug, Clone)]
pub(super) struct HeadUnitConstruction {
    pub group: &'static str,
    pub module: HeadConstruction,
}

#[derive(Debug, Clone)]
pub(super) enum HeadConstruction {
    Projection(LinearSpec),
    Embedding(EmbeddingSpec),
    Confidence(LinearSpec),
    Mapping {
        parameter: ParameterSpec,
        shape: [i32; 1],
    },
}

impl DeclareLogicalParameters for HeadConstruction {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        match self {
            Self::Projection(spec) | Self::Confidence(spec) => spec.logical_parameters(),
            Self::Embedding(spec) => spec.logical_parameters(),
            Self::Mapping { parameter, shape } => Ok(vec![LogicalParameterDeclaration::new(
                parameter,
                shape,
                LogicalParameterRole::TokenMapping,
            )?]),
        }
    }
}

pub(super) fn units(
    geometry: DsparkHeadGeometry,
    formats: &std::collections::HashMap<String, eredu_checkpoint::WeightQuantization>,
) -> Result<[HeadUnitConstruction; 5], Error> {
    let format = |name: &str| {
        crate::linear_format::standard_linear_format(
            name,
            formats
                .get(name)
                .copied()
                .map(Into::into)
                .unwrap_or(eredu_checkpoint::LinearFormat::Dense),
        )
    };
    let projection = |name: &str, input, output, bias, compute| -> Result<LinearSpec, Error> {
        Ok(LinearSpec {
            input,
            output,
            weight: ParameterSpec::trainable(name).map_err(Error::backend)?,
            bias,
            format: format(name)?.with_compute_element(compute)?,
        })
    };
    let mut mapping = ParameterSpec::trainable("d2t").map_err(Error::backend)?;
    mapping.trainable = false;
    Ok([
        HeadUnitConstruction {
            group: "assistant.vocabulary",
            module: HeadConstruction::Projection(projection(
                "lm_head.weight",
                geometry.hidden,
                geometry.vocabulary,
                None,
                None,
            )?),
        },
        HeadUnitConstruction {
            group: "assistant.predecessor_table",
            module: HeadConstruction::Embedding(EmbeddingSpec {
                vocabulary: geometry.vocabulary,
                dimensions: geometry.rank,
                weight: ParameterSpec::trainable("markov_head.markov_w1.weight")
                    .map_err(Error::backend)?,
                format: format("markov_head.markov_w1.weight")?,
            }),
        },
        HeadUnitConstruction {
            group: "assistant.successor_table",
            module: HeadConstruction::Projection(projection(
                "markov_head.markov_w2.weight",
                geometry.rank,
                geometry.vocabulary,
                None,
                None,
            )?),
        },
        HeadUnitConstruction {
            group: "assistant.confidence",
            module: HeadConstruction::Confidence(projection(
                "confidence_head.proj.weight",
                geometry.hidden + geometry.rank,
                1,
                Some(
                    ParameterSpec::trainable("confidence_head.proj.bias")
                        .map_err(Error::backend)?,
                ),
                Some(TensorElementType::F32),
            )?),
        },
        HeadUnitConstruction {
            group: "assistant.token_mapping",
            module: HeadConstruction::Mapping {
                parameter: mapping,
                shape: [geometry.vocabulary],
            },
        },
    ])
}
