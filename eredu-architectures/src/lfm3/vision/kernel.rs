//! Shared component execution of one vision ingress or one decoder pass.

use std::sync::Arc;

use eredu_nn::{
    AttentionCache, DistributedNeuralBackend, Error, GroupedNeuralBackend, Parameterized, Tensor,
};
use eredu_runtime::{
    ComponentAdapter, ComponentDefinition, ComponentEquations, ComponentKernel,
    ComponentPinnedResources, ExecutionUnitAddress, KernelParameterModules, LayerRuntimeState,
    LayeredForwardState, RuntimeStateComponents, TextComponentKernel,
};

use super::{Declaration, VisionAdvance, VisionExecution, VisionInput, VisionModule, VisionPrompt};
use crate::{
    decoder::{LayeredInput, StaticModules},
    lfm3::{Block, CompleteLfm3Kernel, ForwardContext, Lfm3Input},
};

/// One admitted invocation over the combined VL component graph.
pub enum VlInput<'a, T: Tensor> {
    /// Project one media part while leaving decoder state untouched.
    Media(VisionInput<'a, T>),
    /// Decode tokens without running the vision group.
    Text(LayeredInput<'a, T>),
    /// Decode a prefix with retained projected media rows.
    Projected(&'a VisionPrompt<'a, T>),
}

/// One independently leased resource in the VL component.
#[derive(Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum VlUnit<B: GroupedNeuralBackend + DistributedNeuralBackend> {
    /// Patchifier, vision block, or projector.
    Vision(VisionModule<B>),
    /// Ordinary LFM3 decoder block.
    Decoder(Block<B>),
}

/// Transient values retained while the selected group runs.
pub struct VlForward<T> {
    phase: Phase<T>,
}

enum Phase<T> {
    Vision(VisionExecution<T>),
    Projected,
    Decoder(ForwardContext<T>),
}

/// Portable VL equations over shared component execution and parameter leasing.
pub struct VlKernel<B: GroupedNeuralBackend + DistributedNeuralBackend> {
    vision: Declaration,
    decoder: CompleteLfm3Kernel<B>,
    fingerprint: String,
}

/// Complete replicated VL component adapter.
pub type VlLayeredModel<B> =
    ComponentAdapter<Arc<ComponentDefinition>, VlKernel<B>, KernelParameterModules>;

impl<B: GroupedNeuralBackend + DistributedNeuralBackend> VlKernel<B> {
    pub(super) fn from_declaration(
        definition: Arc<ComponentDefinition>,
        vision: Declaration,
        language: Arc<crate::lfm3::Declaration>,
        fingerprint: String,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<VlLayeredModel<B>, Error> {
        let (_, decoder, _) = CompleteLfm3Kernel::from_selected_declaration(
            language.clone(),
            language.definition().clone(),
            None,
            context,
        )?
        .into_parts();
        ComponentAdapter::with_pinned_resources::<B>(
            definition,
            Self {
                vision,
                decoder,
                fingerprint,
            },
        )
        .map_err(Error::from)
    }
}

impl<B: GroupedNeuralBackend + DistributedNeuralBackend> ComponentKernel<B> for VlKernel<B> {
    type Input<'a> = VlInput<'a, B::Tensor>;
    type Unit = VlUnit<B>;
    type ForwardContext = VlForward<B::Tensor>;
    type RetainedContextValues<'a> =
        std::iter::Flatten<std::array::IntoIter<Option<&'a B::Tensor>, 2>>;
    type Error = Error;

    fn state_identity(
        &self,
        state: &eredu_runtime::PartitionState,
        topology: eredu_core::cache::PromptCacheTopology,
    ) -> Result<eredu_runtime::ModelStateIdentity, Error> {
        eredu_runtime::ModelStateIdentity::new(
            "lfm3",
            "lfm3_vl",
            &self.fingerprint,
            self.decoder.args().num_hidden_layers as usize,
            state.global_layer_offset(),
            0,
            topology,
        )
        .map_err(Error::backend_source)
    }

    fn output_score_kind(
        forward: &Self::ForwardContext,
    ) -> Option<eredu_core::refinement::PredictionScoreKind> {
        matches!(forward.phase, Phase::Decoder(_))
            .then_some(eredu_core::refinement::PredictionScoreKind::CausalConditional)
    }

    fn build_unit(
        &self,
        address: ExecutionUnitAddress,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Error> {
        match address.group() {
            0 => {
                let unit =
                    self.vision.units().nth(address.index()).ok_or_else(|| {
                        Error::backend("VL vision unit is outside its declaration")
                    })?;
                VisionModule::new(&self.vision, unit, context).map(VlUnit::Vision)
            }
            1 => self
                .decoder
                .construct_unit(address.index(), context)
                .map(VlUnit::Decoder),
            _ => Err(Error::backend(
                "VL execution group is outside its declaration",
            )),
        }
    }

    fn executes_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        matches!(
            (group, &forward.phase),
            (0, Phase::Vision(_)) | (1, Phase::Decoder(_))
        )
    }

    fn retained_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _address: ExecutionUnitAddress,
    ) -> Self::RetainedContextValues<'a> {
        let values = match &forward.phase {
            Phase::Vision(vision) => vision
                .retained_positions()
                .map_or([None, None], |roots| roots.map(Some)),
            Phase::Projected => [None, None],
            Phase::Decoder(decoder) => [decoder.mask.as_ref(), None],
        };
        values.into_iter().flatten()
    }
}

impl<B: GroupedNeuralBackend + DistributedNeuralBackend> ComponentPinnedResources<B>
    for VlKernel<B>
{
    type Modules = StaticModules<B>;

    fn pinned_modules(&self) -> &Self::Modules {
        self.decoder.pinned_modules()
    }

    fn pinned_modules_mut(&mut self) -> &mut Self::Modules {
        self.decoder.pinned_modules_mut()
    }

    fn visit_pinned<V: eredu_runtime::StaticParameterVisitor<B>>(
        &self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        self.decoder.visit_pinned(visitor)
    }

    fn visit_pinned_mut<V: eredu_runtime::StaticParameterVisitorMut<B>>(
        &mut self,
        visitor: &mut V,
    ) -> Result<(), V::Error> {
        self.decoder.visit_pinned_mut(visitor)
    }
}

impl<B, S> ComponentEquations<B, S> for VlKernel<B>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    fn begin<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Error> {
        let input = match input {
            VlInput::Media(input) => {
                let hidden = input.pixels.clone();
                return Ok(LayeredForwardState {
                    hidden,
                    context: VlForward {
                        phase: Phase::Vision(VisionExecution::new(&self.vision, input)),
                    },
                });
            }
            VlInput::Text(input) => Lfm3Input::Text(input),
            VlInput::Projected(input) => Lfm3Input::Vision(input),
        };
        let forward = self.decoder.begin(input, state, context)?;
        Ok(LayeredForwardState {
            hidden: forward.hidden,
            context: VlForward {
                phase: Phase::Decoder(forward.context),
            },
        })
    }

    fn begin_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        _state: &mut S,
        forward: &mut Self::ForwardContext,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match (group, &forward.phase, dependencies) {
            (0, _, []) | (1, Phase::Decoder(_), [_]) => Ok(initial.clone()),
            (1, Phase::Projected, [projected]) => Ok((*projected).clone()),
            _ => Err(Error::backend("VL group input differs from its invocation")),
        }
    }

    fn execute(
        &mut self,
        address: ExecutionUnitAddress,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match (address.group(), unit, &mut forward.phase) {
            (0, VlUnit::Vision(unit), Phase::Vision(vision)) => {
                match vision.advance_with_activation(unit, hidden, context)? {
                    VisionAdvance::Continue(next) => {
                        let hidden = next.input_or_activation().clone();
                        *vision = next;
                        Ok(hidden)
                    }
                    VisionAdvance::Projected(rows) => {
                        forward.phase = Phase::Projected;
                        Ok(rows)
                    }
                }
            }
            (1, VlUnit::Decoder(unit), Phase::Decoder(decoder)) => self
                .decoder
                .execute(address, unit, hidden, state, decoder, context),
            _ => Err(Error::backend(
                "VL unit differs from the active execution group",
            )),
        }
    }

    fn finish(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match &forward.phase {
            Phase::Decoder(decoder) => self.decoder.finish(hidden, state, decoder, context),
            Phase::Projected => Ok(hidden.clone()),
            Phase::Vision(_) => Err(Error::backend(
                "VL vision invocation has not projected its rows",
            )),
        }
    }

    fn projects_final_text_position() -> bool {
        true
    }

    fn finish_text(
        &mut self,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match &forward.phase {
            Phase::Decoder(decoder) => self.decoder.finish_text(hidden, state, decoder, context),
            _ => self.finish(hidden, state, forward, context),
        }
    }
}

impl<B: GroupedNeuralBackend + DistributedNeuralBackend> TextComponentKernel<B> for VlKernel<B> {
    fn supports_chunked_prefill() -> bool {
        true
    }

    fn text_input<'a>(tokens: &'a B::Tensor, mask: Option<&'a B::Tensor>) -> Self::Input<'a> {
        VlInput::Text(LayeredInput { tokens, mask })
    }
}
