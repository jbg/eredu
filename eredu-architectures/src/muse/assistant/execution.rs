//! One architecture-owned parameter topology for Muse assistant body and heads.

use super::{
    body::{
        AssistantBodyDefinition, AssistantBodyGeometry, DFlashBodyDefinition, DFlashBodyUnit,
        DsparkBodyDefinition, DsparkBodyUnit,
    },
    candidate_selector::{CandidateSelectorDefinition, CandidateSelectorUnit},
    dspark::{DsparkHeadDefinition, DsparkHeadUnit},
};
use eredu_nn::{DistributedNeuralBackend, Error, GroupedNeuralBackend, Parameterized, Tensor};
use eredu_runtime::{ExecutionGraph, ExecutionUnitLayout, UnitPolicyView, UnitProjection};

#[derive(Debug, Clone)]
pub(super) enum Equation {
    Glimmer(DFlashBodyDefinition),
    DSpark {
        body: DsparkBodyDefinition,
        heads: DsparkHeadDefinition,
    },
    DFlash2 {
        body: DFlashBodyDefinition,
        selector: CandidateSelectorDefinition,
    },
}
#[path = "execution_parallel.rs"]
mod parallel;

/// Selected assistant neural placement, separate from parameter residency.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum MuseAssistantPlacement {
    /// Complete ordinary local execution.
    Replicated,
    /// Body projections are partitioned; scoring parameters are complete replicas.
    TensorParallel(eredu_core::ParallelRankTopology),
}

/// Complete validated assistant construction, independent of parameter residency.
#[derive(Debug, Clone)]
pub struct MuseAssistantDefinition {
    pub(super) equation: Equation,
    layout: ExecutionUnitLayout,
    pub(super) placement: MuseAssistantPlacement,
    pub(super) compute_precisions: Vec<eredu_core::checkpoint::TensorDtype>,
}
/// One independently populated body or head unit, never a complete assistant.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct MuseAssistantUnit<B: GroupedNeuralBackend + DistributedNeuralBackend> {
    module: Module<B>,
}
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
enum Module<B: GroupedNeuralBackend + DistributedNeuralBackend> {
    DFlashBody(DFlashBodyUnit<B>),
    DsparkBody(DsparkBodyUnit<B>),
    DsparkHead(DsparkHeadUnit<B>),
    Selector(CandidateSelectorUnit<B>),
}
macro_rules! projection {
    ($name:ident, $variant:ident, $part:ident) => {
        pub(super) struct $name;
        impl<B: GroupedNeuralBackend + DistributedNeuralBackend>
            UnitProjection<MuseAssistantUnit<B>, $part<B>> for $name
        {
            fn wrap(unit: $part<B>) -> MuseAssistantUnit<B> {
                MuseAssistantUnit {
                    module: Module::$variant(unit),
                }
            }
            fn get(unit: &MuseAssistantUnit<B>) -> &$part<B> {
                match &unit.module {
                    Module::$variant(unit) => unit,
                    _ => panic!("assistant policy returned a different unit variant"),
                }
            }
            fn get_mut(unit: &mut MuseAssistantUnit<B>) -> &mut $part<B> {
                match &mut unit.module {
                    Module::$variant(unit) => unit,
                    _ => panic!("assistant policy returned a different unit variant"),
                }
            }
        }
    };
}
projection!(DFlashBodyProjection, DFlashBody, DFlashBodyUnit);
projection!(DsparkBodyProjection, DsparkBody, DsparkBodyUnit);
projection!(DsparkHeadProjection, DsparkHead, DsparkHeadUnit);
projection!(SelectorProjection, Selector, CandidateSelectorUnit);

impl MuseAssistantDefinition {
    /// Maximum target-owned static calls for the selected proposal capacity.
    /// The anchor occupies an embedding position but produces no readout row.
    pub(crate) fn target_static_invocations(
        &self,
        proposals: std::num::NonZeroUsize,
    ) -> Result<
        Vec<(
            crate::composite_execution::ExternalPredictionTargetOperationKind,
            Vec<i32>,
        )>,
        Error,
    > {
        use crate::composite_execution::ExternalPredictionTargetOperationKind as Kind;
        let geometry = self.body_geometry();
        let proposals = i32::try_from(proposals.get()).map_err(Error::backend_source)?;
        let positions = match geometry.proposal {
            super::ProposalLayout::SampleFromAnchor => proposals,
            super::ProposalLayout::AnchorThenMasks => proposals
                .checked_add(1)
                .ok_or_else(|| Error::backend("assistant proposal embedding extent overflows"))?,
        };
        if positions > geometry.block {
            return Err(Error::backend(
                "selected proposal capacity exceeds assistant block geometry",
            ));
        }
        let (embedding, projection) = match &self.equation {
            Equation::Glimmer(_) => (Kind::TokenEmbeddings, Some(Kind::ProjectLogits)),
            Equation::DSpark { .. } => (Kind::RawTokenEmbeddings, None),
            Equation::DFlash2 { .. } => (
                Kind::RawTokenEmbeddings,
                Some(Kind::RawVocabularyProjection),
            ),
        };
        let mut invocations = vec![(embedding, vec![1, positions])];
        if let Some(projection) = projection {
            invocations.push((projection, vec![1, proposals, geometry.hidden]));
        }
        Ok(invocations)
    }

    /// Pure context, fixed-block and scoring geometry from the selected equations.
    pub fn memory_topology(
        &self,
    ) -> Result<eredu_runtime::memory_forecast::ConditionedPredictionTopology, Error> {
        use eredu_runtime::memory_forecast::{
            ConditionedPredictionScoring as Scoring, ConditionedPredictionTopology,
        };
        let (body, scoring) = match &self.equation {
            Equation::Glimmer(body) => (body.memory_body(), Scoring::TargetProjection),
            Equation::DSpark { body, heads } => {
                let h = heads.geometry();
                (
                    body.memory_body(),
                    Scoring::MappedVocabulary {
                        vocabulary: h.vocabulary as u64,
                        rank: h.rank as u64,
                    },
                )
            }
            Equation::DFlash2 { body, selector } => {
                let h = selector.geometry();
                (
                    body.memory_body(),
                    Scoring::CandidateSelector {
                        vocabulary: h.vocabulary as u64,
                        rank: h.rank as u64,
                        candidates: h.top_k as u64,
                    },
                )
            }
        };
        let topology = ConditionedPredictionTopology { body, scoring };
        topology.validate().map_err(Error::backend)?;
        Ok(topology)
    }

    fn new(equation: Equation) -> Result<Self, Error> {
        let (body, head) = match &equation {
            Equation::Glimmer(body) => (body.geometry(), None),
            Equation::DSpark { body, heads } => {
                if body.geometry().hidden_size() != heads.geometry().hidden {
                    return Err(Error::backend("DSpark body and head widths differ"));
                }
                (body.geometry(), Some(heads.unit_layout()))
            }
            Equation::DFlash2 { body, selector } => {
                if body.geometry().hidden_size() != selector.geometry().hidden {
                    return Err(Error::backend("DFlash2 body and selector widths differ"));
                }
                (body.geometry(), Some(selector.unit_layout()))
            }
        };
        let layouts = std::iter::once(body.unit_layout()).chain(head);
        let mut ids = Vec::new();
        let mut counts = Vec::new();
        for layout in layouts {
            for group in 0..layout.group_count() {
                ids.push(
                    layout
                        .group_id(group)
                        .expect("canonical component group")
                        .as_str()
                        .to_owned(),
                );
                counts.push(
                    layout
                        .group_range(group)
                        .expect("canonical component range")
                        .len(),
                );
            }
        }
        let graph = ExecutionGraph::chain(ids).map_err(Error::backend)?;
        let layout = ExecutionUnitLayout::new(&graph, counts).map_err(Error::backend)?;
        Ok(Self {
            equation,
            layout,
            placement: MuseAssistantPlacement::Replicated,
            compute_precisions: vec![
                eredu_core::checkpoint::TensorDtype::F16,
                eredu_core::checkpoint::TensorDtype::Bf16,
                eredu_core::checkpoint::TensorDtype::F32,
            ],
        })
    }
    /// Direct Glimmer body without assistant-owned vocabulary parameters.
    pub fn glimmer(body: DFlashBodyDefinition) -> Result<Self, Error> {
        Self::new(Equation::Glimmer(body))
    }
    /// Causal Spark body with its own vocabulary, Markov and confidence heads.
    pub fn dspark(body: DsparkBodyDefinition, heads: DsparkHeadDefinition) -> Result<Self, Error> {
        Self::new(Equation::DSpark { body, heads })
    }
    /// Convolved Spark body with its own conditional selector.
    pub fn dflash2(
        body: DFlashBodyDefinition,
        selector: CandidateSelectorDefinition,
    ) -> Result<Self, Error> {
        Self::new(Equation::DFlash2 { body, selector })
    }
    /// Retained neural placement used to validate the proposal collective context.
    pub const fn placement(&self) -> MuseAssistantPlacement {
        self.placement
    }
    /// Complete canonical acquisition order across body and auxiliary parameters.
    pub fn unit_layout(&self) -> &ExecutionUnitLayout {
        &self.layout
    }
    /// Shared context and proposal geometry for this assistant.
    pub fn body_geometry(&self) -> &AssistantBodyGeometry {
        match &self.equation {
            Equation::Glimmer(body) | Equation::DFlash2 { body, .. } => body.geometry(),
            Equation::DSpark { body, .. } => body.geometry(),
        }
    }
    /// Constructs exactly one unit from retained geometry and selected encodings.
    pub fn build_unit<B: GroupedNeuralBackend + DistributedNeuralBackend>(
        &self,
        ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<MuseAssistantUnit<B>, Error> {
        let body_count = self.body_geometry().unit_layout().len();
        if ordinal >= self.layout.len() {
            return Err(Error::backend(
                "invalid Muse assistant parameter unit ordinal",
            ));
        }
        match &self.equation {
            Equation::Glimmer(body) => Ok(DFlashBodyProjection::wrap(
                AssistantBodyDefinition::<B>::build_unit(body, ordinal, context)?,
            )),
            Equation::DSpark { body, heads } => {
                if ordinal < body_count {
                    Ok(DsparkBodyProjection::wrap(
                        AssistantBodyDefinition::<B>::build_unit(body, ordinal, context)?,
                    ))
                } else {
                    Ok(DsparkHeadProjection::wrap(
                        heads.build_unit::<B>(ordinal - body_count, context)?,
                    ))
                }
            }
            Equation::DFlash2 { body, selector } => {
                if ordinal < body_count {
                    Ok(DFlashBodyProjection::wrap(
                        AssistantBodyDefinition::<B>::build_unit(body, ordinal, context)?,
                    ))
                } else {
                    Ok(SelectorProjection::wrap(
                        selector.build_unit::<B>(ordinal - body_count, context)?,
                    ))
                }
            }
        }
    }
    /// Constructs a resident bank through the same individually addressable units.
    pub fn build_units<B: GroupedNeuralBackend + DistributedNeuralBackend>(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Vec<MuseAssistantUnit<B>>, Error> {
        (0..self.layout.len())
            .map(|i| self.build_unit::<B>(i, context))
            .collect()
    }
    pub(super) fn view<'a, B, Part, P, Projection>(
        &'a self,
        policy: &'a mut P,
        component: &'a ExecutionUnitLayout,
        offset: usize,
    ) -> Result<UnitPolicyView<'a, B, MuseAssistantUnit<B>, Part, P, Projection>, Error>
    where
        B: GroupedNeuralBackend + DistributedNeuralBackend,
    {
        UnitPolicyView::new(policy, &self.layout, component, offset).map_err(Error::backend_source)
    }
}

pub(super) fn wrap_dflash_body<B: GroupedNeuralBackend + DistributedNeuralBackend>(
    units: Vec<DFlashBodyUnit<B>>,
) -> Vec<MuseAssistantUnit<B>> {
    units.into_iter().map(DFlashBodyProjection::wrap).collect()
}
pub(super) fn wrap_dspark_body<B: GroupedNeuralBackend + DistributedNeuralBackend>(
    units: Vec<DsparkBodyUnit<B>>,
) -> Vec<MuseAssistantUnit<B>> {
    units.into_iter().map(DsparkBodyProjection::wrap).collect()
}
pub(super) fn wrap_dspark_heads<B: GroupedNeuralBackend + DistributedNeuralBackend>(
    units: Vec<DsparkHeadUnit<B>>,
) -> Vec<MuseAssistantUnit<B>> {
    units.into_iter().map(DsparkHeadProjection::wrap).collect()
}
pub(super) fn wrap_selector<B: GroupedNeuralBackend + DistributedNeuralBackend>(
    units: Vec<CandidateSelectorUnit<B>>,
) -> Vec<MuseAssistantUnit<B>> {
    units.into_iter().map(SelectorProjection::wrap).collect()
}
