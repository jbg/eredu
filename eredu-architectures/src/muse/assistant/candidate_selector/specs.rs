//! Selector parameter slots and independently leased construction units.

use eredu_nn::{
    DeclareLogicalParameters, EmbeddingSpec, Error, LinearSpec, LogicalParameterDeclaration,
    ParameterSpec,
};

use super::CandidateSelectorGeometry;

#[derive(Debug, Clone)]
pub(super) struct UnitConstruction {
    pub group: &'static str,
    pub module: Construction,
}

#[derive(Debug, Clone)]
pub(super) enum Construction {
    Table(EmbeddingSpec),
    Projection(LinearSpec),
}

impl DeclareLogicalParameters for Construction {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        match self {
            Self::Table(spec) => spec.logical_parameters(),
            Self::Projection(spec) => spec.logical_parameters(),
        }
    }
}

pub(super) fn units(
    geometry: CandidateSelectorGeometry,
    formats: &std::collections::HashMap<String, eredu_checkpoint::WeightQuantization>,
) -> Result<[UnitConstruction; 3], Error> {
    let format = |name: &str| {
        crate::linear_format::matrix_linear_format(
            name,
            name.strip_suffix(".weight").unwrap_or(name),
            formats
                .get(name)
                .copied()
                .map(Into::into)
                .unwrap_or(eredu_checkpoint::LinearFormat::Dense),
        )
    };
    let table = |name: &str| -> Result<Construction, Error> {
        Ok(Construction::Table(EmbeddingSpec {
            vocabulary: geometry.vocabulary,
            dimensions: geometry.rank,
            weight: ParameterSpec::trainable(name).map_err(Error::backend)?,
            format: format(name)?,
        }))
    };
    Ok([
        UnitConstruction {
            group: "assistant.predecessor_table",
            module: table("candidate_selector.predecessor_codebook")?,
        },
        UnitConstruction {
            group: "assistant.successor_table",
            module: table("candidate_selector.successor_codebook")?,
        },
        UnitConstruction {
            group: "assistant.conditional_projection",
            module: Construction::Projection(LinearSpec {
                input: geometry.hidden,
                output: geometry.rank,
                weight: ParameterSpec::trainable("candidate_selector.hidden_projection.weight")
                    .map_err(Error::backend)?,
                bias: None,
                format: format("candidate_selector.hidden_projection.weight")?,
            }),
        },
    ])
}
